import torch

def generate_Jij(L: int, seed: int, boundary):
    if boundary == "periodic":
        Li = L
    else:
        Li = L - 1
    torch.manual_seed(seed)
    # J_i(i+1) L*L i.i.d. standard normal
    J_i = torch.randn(L, Li)
    J_j = torch.randn(Li, L)
    return J_i, J_j

def energy(sample, boundary, J_i, J_j):
    if boundary == "periodic":
        term1 = J_i * sample * sample.roll(1, dims=1)
        term1 = term1.sum(dim=(1, 2))
        term2 = J_j * sample * sample.roll(1, dims=2)
        term2 = term2.sum(dim=(1, 2))
    else:
        term1 = J_i * sample[:, :, 1:] * sample[:, :, :-1]
        term1 = term1.sum(dim=(1, 2))
        term2 = J_j * sample[:, 1:, :] * sample[:, :-1, :]
        term2 = term2.sum(dim=(1, 2))
        
    return term1 + term2