import time

import numpy as np
import torch
import torchvision
from numpy import sqrt
from torch import nn

from system import EdwardAnderson
from script.args import args
import model

from utils import (
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
    make_grid_ising,
)

def save_Jij(folder, J_i, J_j):
    np.save(folder + "/J_i.npy", J_i.cpu().numpy())
    np.save(folder + "/J_j.npy", J_j.cpu().numpy())

def load_Jij(folder):
    J_i = torch.tensor(np.load(folder + "/J_i.npy"))
    J_j = torch.tensor(np.load(folder + "/J_j.npy"))
    return J_i, J_j

def generate_sample(net, batch_size=None):
    sample_start_time = time.time()
    with torch.no_grad():
        sample, x_hat = net.sample(batch_size)
    assert not sample.requires_grad
    assert not x_hat.requires_grad
    sample_time = time.time() - sample_start_time
    return sample, x_hat, sample_time


def update_network(net, J_i, J_j, optimizer=None, scheduler=None, sample=None, **kwargs):
    beta, ham, lattice, boundary = (
        kwargs.get("beta"),
        kwargs.get("ham"),
        kwargs.get("lattice"),
        kwargs.get("boundary"),
    )
    train_start_time = time.time()
    log_prob = net.log_prob(sample)
    with torch.no_grad():
        energy = EdwardAnderson.energy(sample, boundary, J_i, J_j)
        loss = log_prob + beta * energy
    assert not energy.requires_grad
    assert not loss.requires_grad
    loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
    loss_reinforce.backward()

    if args.clip_grad:
        nn.utils.clip_grad_norm_(list(net.parameters()), args.clip_grad)

    optimizer.step()

    if args.lr_schedule:
        scheduler.step(loss.mean())
    train_time = time.time() - train_start_time
    return log_prob, energy, loss, train_time


def main():
    start_time = time.time()

    init_out_dir()
    if args.clear_checkpoint:
        clear_checkpoint()
    last_step = get_last_checkpoint_step()
    if last_step >= 0:
        my_log("\nCheckpoint found: {}\n".format(last_step))
    else:
        clear_log()
    print_args()
    modelinterface = model.ModelInterface(**vars(args))
    modelinterface.model.to(args.device)
    my_log("{}\n".format(modelinterface.model))

    params = list(modelinterface.model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log("Total number of trainable parameters: {}".format(nparams))
    named_params = list(modelinterface.model.named_parameters())

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr)
    elif args.optimizer == "sgdm":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == "adam0.5":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    else:
        raise ValueError("Unknown optimizer: {}".format(args.optimizer))

    scheduler = None
    if args.lr_schedule:
        # 0.92**80 ~ 1e-3
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.92, patience=100, threshold=1e-4, min_lr=1e-6
        )

    if last_step >= 0:
        state = torch.load("{}_save/{}.state".format(args.out_filename, last_step))
        ignore_param(state["net"], modelinterface.model)
        modelinterface.load_state_dict(state["net"])
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if args.lr_schedule and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        J_i, J_j = load_Jij("{}_save/".format(args.out_filename))
    else:
        J_i, J_j = EdwardAnderson.generate_Jij(args.L, args.glass_seed, args.boundary)
        save_Jij("{}_save/".format(args.out_filename), J_i, J_j)

    J_i = J_i.to(args.device)
    J_j = J_j.to(args.device)
    init_time = time.time() - start_time
    my_log("init_time = {:.3f}".format(init_time))

    my_log("Training...")

    sample_time = 0
    train_time = 0
    start_time = time.time()
    for step in range(last_step + 1, args.max_step + 1):
        optimizer.zero_grad()

        sample, x_hat, sample_time_i = generate_sample(modelinterface.model, args.batch_size)
        sample_time += sample_time_i
        beta = args.beta * (1 - args.beta_anneal**step + args.beta_anneal**args.max_step)
        log_prob, energy, loss, train_time_i = update_network(
            modelinterface.model,
            J_i,
            J_j,
            optimizer,
            scheduler,
            sample,
            beta=beta,
            ham=args.ham,
            lattice=args.lattice,
            boundary=args.boundary,
        )
        train_time += train_time_i
        if args.print_step and step % args.print_step == 0:
            free_energy_mean = loss.mean() / args.beta / args.L**2
            free_energy_std = loss.std() / args.beta / args.L**2
            entropy_mean = -log_prob.mean() / args.L**2
            energy_mean = energy.mean() / args.L**2
            mag = sample.mean(dim=0)
            mag_mean = mag.mean()
            mag_sqr_mean = (mag**2).mean()
            if step > 0:
                sample_time /= args.print_step
                train_time /= args.print_step
            used_time = time.time() - start_time
            my_log(
                "step = {}, F = {:.8g}, F_std = {:.8g}, S = {:.8g}, E = {:.8g}, M = {:.8g}, Q = {:.8g}, lr = {:.3g}, beta = {:.8g}, sample_time = {:.5f}, train_time = {:.5f}, used_time = {:.5f}".format(
                    step,
                    free_energy_mean.item(),
                    free_energy_std.item(),
                    entropy_mean.item(),
                    energy_mean.item(),
                    mag_mean.item(),
                    mag_sqr_mean.item(),
                    optimizer.param_groups[0]["lr"],
                    beta,
                    sample_time,
                    train_time,
                    used_time,
                )
            )
            sample_time = 0
            train_time = 0

            if args.save_sample:
                state = {
                    "sample": sample,
                    "x_hat": x_hat,
                    "log_prob": log_prob,
                    "energy": energy,
                    "loss": loss,
                }
                torch.save(state, "{}_save/{}.sample".format(args.out_filename, step))

        if args.out_filename and args.save_step and step % args.save_step == 0:
            state = {
                "net": modelinterface.model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.lr_schedule:
                state["scheduler"] = scheduler.state_dict()
            torch.save(state, "{}_save/{}.state".format(args.out_filename, step))

        if args.out_filename and args.visual_step and step % args.visual_step == 0 and step != 0:
            sample_tensor = sample
            grid = make_grid_ising(sample_tensor, dimension=2)
            torchvision.utils.save_image(
                grid,
                "{}_img/{}.png".format(args.out_filename, step),
            )

            if args.print_sample:
                x_hat_np = x_hat.view(x_hat.shape[0], -1).cpu().numpy()
                x_hat_std = np.std(x_hat_np, axis=0).reshape([args.L] * 2)

                x_hat_cov = np.cov(x_hat_np.T)
                x_hat_cov_diag = np.diag(x_hat_cov)
                x_hat_corr = x_hat_cov / (
                    sqrt(x_hat_cov_diag[:, None] * x_hat_cov_diag[None, :]) + args.epsilon
                )
                x_hat_corr = np.tril(x_hat_corr, -1)
                x_hat_corr = np.max(np.abs(x_hat_corr), axis=1)
                x_hat_corr = x_hat_corr.reshape([args.L] * 2)

                energy_np = energy.cpu().numpy()
                print(energy_np.shape)
                energy_count = np.stack(np.unique(energy_np, return_counts=True)).T

                my_log(
                    "\nsample\n{}\nx_hat\n{}\nlog_prob\n{}\nenergy\n{}\nloss\n{}\nx_hat_std\n{}\nx_hat_corr\n{}\nenergy_count\n{}\n".format(
                        sample[: args.print_sample],
                        x_hat[: args.print_sample],
                        log_prob[: args.print_sample],
                        energy[: args.print_sample],
                        loss[: args.print_sample],
                        x_hat_std,
                        x_hat_corr,
                        energy_count,
                    )
                )

            if args.print_grad:
                my_log("grad max_abs min_abs mean std")
                for name, param in named_params:
                    if param.grad is not None:
                        grad = param.grad
                        grad_abs = torch.abs(grad)
                        my_log(
                            "{} {:.3g} {:.3g} {:.3g} {:.3g}".format(
                                name,
                                torch.max(grad_abs).item(),
                                torch.min(grad_abs).item(),
                                torch.mean(grad).item(),
                                torch.std(grad).item(),
                            )
                        )
                    else:
                        my_log("{} None".format(name))
                my_log("")
