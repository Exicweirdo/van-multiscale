import time
import numpy as np
import torch

from script.sample_ising2d import partial_sample_ising2d, sample_ising2d
from script.args import args
from system import ising2D
from utils import ignore_param, ensure_dir, init_mc_filename, my_log
from chunked_data import ChunkedDataWriter
import model


def getk(**kwargs):
    if kwargs["k_type"] == "global":
        return 0
    elif kwargs["k_type"] == "uniform":
        return np.random.randint(0, kwargs["L"] ** 2)


def get_log_q_func(modelinterface):
    def log_q_func(sample):
        return modelinterface.model.log_prob(sample)

    return log_q_func


def welford_update(curr, step, mean, var_sum):
    diff = curr - mean
    mean_new = mean + diff / step
    var_sum_new = var_sum + diff * (curr - mean_new)
    return mean_new, var_sum_new


@torch.compile
@torch.no_grad()
def update(
    autoregressive_model, log_q_func, energy_func, sample_old, log_q_old, energy_old, **kwargs
):
    batch_size = kwargs.get("batch_size")
    beta = kwargs.get("beta")
    L = kwargs.get("L")
    # sample from autorregressive model and calculate acceptance
    start_index = getk(**kwargs)
    if kwargs["k_type"] == "global":
        sample, x_hat, sample_time = sample_ising2d(autoregressive_model, batch_size=batch_size)
    elif kwargs["k_type"] == "uniform":
        sample, x_hat, sample_time = partial_sample_ising2d(
            autoregressive_model, batch_size=batch_size, sample_init=sample_old, start=start_index
        )
    log_q = log_q_func(sample)
    energy = energy_func(sample)
    acceptance = torch.exp(-beta * (energy - energy_old) + log_q_old - log_q)
    uniform = torch.rand_like(acceptance)

    acc = uniform < acceptance
    sample_out = torch.where(acc.view(-1, 1, 1), sample, sample_old)
    energy = torch.where(acc, energy, energy_old)
    # randomly translate the sample

    index = torch.randint(0, L, (2,), device=sample.device)
    sample_out = torch.roll(sample_out, shifts=(index[0], index[1]), dims=(1, 2))
    # randomly flip or rotate the sample
    reflex_x, reflex_y, reflex_xy, reflex_time = torch.randint(0, 2, (4,)) > 0
    sample_out = sample_out.flip(1) if reflex_x else sample_out
    sample_out = sample_out.flip(2) if reflex_y else sample_out
    sample_out = sample_out.transpose(1, 2) if reflex_xy else sample_out
    sample_out = -sample_out if reflex_time else sample_out

    log_q = log_q_func(sample_out)
    accept_count = acc.sum()
    return sample_out, log_q, energy, accept_count, start_index


def main():
    init_mc_filename()

    modelinterface = model.ModelInterface(**vars(args))
    # print(modelinterface.model)
    modelinterface.model.to(args.device)

    state = torch.load("{}".format(args.checkpoint_file), weights_only=True)
    ignore_param(state["net"], modelinterface.model)
    modelinterface.load_state_dict(state["net"])
    log_q_func = get_log_q_func(modelinterface)
    energy_func = lambda x: ising2D.energy(x, args.ham, args.lattice, args.boundary)

    if args.z2 and args.k_type == "uniform":
        raise ValueError("Cannot use z2 and uniform k_type together")
    # initial sample from autorregressive model
    sample_init = torch.zeros(args.batch_size, args.L, args.L, device=args.device)
    sample_out, x_hat, _ = partial_sample_ising2d(
        modelinterface.model, args.batch_size, sample_init
    )
    log_q = log_q_func(sample_out)
    energy = energy_func(sample_out)

    accept_count = 0
    energy_mean = 0
    energy_var_sum = 0

    writer_proto = [
        # Uncomment to save all the sampled spins
        # ('spins', bool, (args.batch_size, args.L, args.L)),
        ("log_q", np.float32, (args.batch_size,)),
        ("energy", np.int32, (args.batch_size,)),
        ("plaquette", np.float32, (args.batch_size,)),
        ("mag", np.float32, (args.batch_size,)),
        ("accept_count", np.int32, None),
        ("k", np.int32, None),
    ]

    data_filename = args.out_filename + ".hdf5"
    ensure_dir(data_filename)

    start_time = time.time()
    with ChunkedDataWriter(data_filename, writer_proto, args.save_step) as writer:
        my_log("Sampling...")
        for step in range(args.max_step):
            sample_out, log_q, energy, accept_count_local, start_index = update(
                modelinterface.model,
                log_q_func,
                energy_func,
                sample_out,
                log_q,
                energy,
                **vars(args),
            )

            accept_count += accept_count_local.item()
            mag = sample_out.mean(dim=(1, 2))
            plaquette = ising2D.plaqutte(sample_out)
            #convert sample_out to boolean
            sample_out_bool = sample_out > 0
            
            if step >= args.warmup_step and step % args.record_step == 0:
                writer.write(
                    # sample_out_bool.cpu(), 
                    log_q.cpu(), 
                    energy.cpu(), 
                    plaquette.cpu(), 
                    mag.cpu(), 
                    accept_count_local, 
                    start_index
                )
            
            energy_mean, energy_var_sum = welford_update(
                energy.mean().item() / args.L**2, step + 1, energy_mean, energy_var_sum
            )

            if args.print_step and step % args.print_step == 0:
                accept_rate = accept_count / ((step + 1) * args.batch_size)
                energy_std = np.sqrt(energy_var_sum / (step + 1))
                my_log(
                    ", ".join(
                        [
                            f"step = {step}",
                            f"P = {accept_rate:.8g}",
                            f"E = {energy_mean:.8g}",
                            f"E_std = {energy_std:.8g}",
                            f"time = {time.time() - start_time:.3f}",
                            f"k = {start_index}",
                        ]
                    )
                )
