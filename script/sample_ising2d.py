import time
import torch

from system import ising2D

@torch.compiler.disable
def sample_ising2d(net, batch_size=None):
    sample_start_time = time.time()
    with torch.no_grad():
        sample, x_hat = net.sample(batch_size)
    sample_time = time.time() - sample_start_time
    return sample, x_hat, sample_time

@torch.compiler.disable
def partial_sample_ising2d(net, batch_size=None, sample_init=None, start=0):
    sample_start_time = time.time()
    with torch.no_grad():
        sample, x_hat = net.partial_sample(batch_size, sample_init, start)
    sample_time = time.time() - sample_start_time
    return sample, x_hat, sample_time

def calc_properties(probability_model, sample, beta_sample, **kwargs):
    ham = kwargs.get("ham", "fm")
    lattice = kwargs.get("lattice", "sqr")
    boundary = kwargs.get("boundary", "periodic")
    L = kwargs.get("L", 16)
    with torch.no_grad():
        log_prob = probability_model.log_prob(sample)
        energy = ising2D.energy(sample, ham, lattice, boundary)
        loss = log_prob + beta_sample * energy
    assert not energy.requires_grad
    assert not loss.requires_grad

    free_energy_mean = loss.mean() / beta_sample / L**2
    free_energy_std = loss.std() / beta_sample / L**2
    entropy_mean = -log_prob.mean() / L**2
    energy_mean = energy.mean() / L**2
    capacity = energy.var() * beta_sample**2 / L**2
    mag = sample.mean(dim=0)
    mag_mean = mag.mean()
    mag_abs_mean = torch.abs(sample.mean(dim=(1, 2))).mean()
    return {
        "free_energy": free_energy_mean,
        "free_energy_std": free_energy_std,
        "entropy": entropy_mean,
        "energy": energy_mean,
        "mag": mag_mean,
        "mag_abs": mag_abs_mean,
        "capacity": capacity,
    }

