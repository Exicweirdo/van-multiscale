# 2D classical Ising model
import numpy as np
import torch

def energy(sample, ham, lattice, boundary):
    term = sample[:, 1:, :] * sample[:, :-1, :]
    term = term.sum(dim=(1, 2))
    output = term
    term = sample[:, :, 1:] * sample[:, :, :-1]
    term = term.sum(dim=(1, 2))
    output += term
    if lattice == 'tri':
        term = sample[:, 1:, 1:] * sample[:, :-1, :-1]
        term = term.sum(dim=(1, 2))
        output += term

    if boundary == 'periodic':
        term = sample[:, 0, :] * sample[:, -1, :]
        term = term.sum(dim=1)
        output += term
        term = sample[:, :, 0] * sample[:, :, -1]
        term = term.sum(dim=1)
        output += term
        if lattice == 'tri':
            term = sample[:, 0, 1:] * sample[:, -1, :-1]
            term = term.sum(dim=1)
            output += term
            term = sample[:, 1:, 0] * sample[:, :-1, -1]
            term = term.sum(dim=1)
            output += term
            term = sample[:, 0, 0] * sample[:, -1, -1]
            #term = term.sum(dim=1)
            output += term

    if ham == 'fm':
        output *= -1

    return output

def energy_cross(sample, ham):
    submat1 = sample[:, ::2, ::2]
    submat2 = sample[:, 1::2, 1::2]
    out = (submat1 * submat2).sum(dim=(1, 2))
    out += (torch.roll(submat1, shifts=-1, dims=1) * submat2).sum(dim=(1, 2))
    out += (torch.roll(submat1, shifts=-1, dims=2) * submat2).sum(dim=(1, 2))
    out += (torch.roll(submat1, shifts=(-1, -1), dims=(1, 2)) * submat2).sum(dim=(1, 2))
    if ham == 'fm':
        out *= -1
    return out

def plaqutte(sample):
    """
    Calculate the plaquette correlation for a square lattice.
    """
    out = sample[:, :-1, :-1] * sample[:, 1:, :-1] * sample[:, :-1, 1:] * sample[:, 1:, 1:]
    out = out.sum(dim=(1, 2))
    return out

def rhombic(sample):
    """
    Calculate the rhombic correlation for a triangular lattice.
    """
    submat1 = sample[:, ::2, ::2]
    submat2 = sample[:, 1::2, 1::2]
    out = (submat1 * submat2 * torch.roll(submat1, shifts=-1, dims=1) * torch.roll(submat2, shifts=1, dims=2)).sum(dim=(1, 2))
    return out