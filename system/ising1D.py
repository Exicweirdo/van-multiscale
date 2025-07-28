# 1D classical Ising model


def energy(sample, ham, boundary):
    term = sample[:, 1:] * sample[:, :-1]
    output = term.sum(dim=1)
    if boundary == "periodic":
        term = sample[:, 0] * sample[:, -1]
        output += term

    if ham == "fm":
        output *= -1

    return output
