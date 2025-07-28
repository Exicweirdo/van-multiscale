import numpy as np
import pandas as pd
import os
import sys
import re
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.optimize import fsolve

def Ising_F_anal(beta: float):
    def f(x):
        return np.log(0.5 * (1 + np.sqrt(1 - (2 * np.sinh(2 * beta) /
                      np.cosh(2 * beta) ** 2) ** 2 * np.sin(x)**2)))
    integral, err = scipy.integrate.quad(f, 0, np.pi / 2)
    integral = 1 / np.pi * integral
    # print("The error of the integral is ", err * 1 / np.pi / beta)
    return - 1 / beta * (np.log(2 * np.cosh(2 * beta)) + integral)

def M_func(M, beta):
    return M - np.tanh(4 * beta * M)

def M_mean(beta):
    return fsolve(M_func, max(0, (4*beta - 1) * 3)**0.5, args=(beta))[0]

def F_mean_field(beta):
    M_ = M_mean(beta)
    return - 1 / beta * (np.log(2 * np.cosh(4 * beta * M_))) + 2 * M_**2

def F_periodic(beta, n):
    logZ = -np.log(2) / n**2 + 1 / 2 * np.log(2 * np.sinh(2 * beta))
    rlist = np.arange(1, n + 1)
    logZ += np.log(
        np.prod(2 * np.cosh(n / 2 * gamma_r(beta, rlist * 2, n))) +
        np.prod(2 * np.cosh(n / 2 * gamma_r(beta, rlist * 2 - 1, n))) +
        np.prod(2 * np.sinh(n / 2 * gamma_r(beta, rlist * 2, n))) +
        np.prod(2 * np.sinh(n / 2 * gamma_r(beta, rlist * 2 - 1, n)))
    ) / n**2
    return -logZ / beta

def plot_F(F_dict: dict, **kwargs):
    fig, ax = plt.subplots()
    for modelconfig in F_dict.keys():
        print(modelconfig)
        betas = np.array(list(F_dict[modelconfig].keys()))
        betas.sort()
        Fs = np.array([F_dict[modelconfig][beta][0] for beta in betas])
        F_stds = np.array([F_dict[modelconfig][beta][1] for beta in betas])
        ax.errorbar(betas, Fs, yerr=F_stds, label=modelconfig, **kwargs)
    ax.legend()
    dif_fig, dif_ax = plt.subplots()
    for modelconfig in F_dict.keys():
        betas = np.array(list(F_dict[modelconfig].keys()))
        betas.sort()
        Fs_dif = np.array([F_dict[modelconfig][beta][0] for beta in betas]) - \
            np.array([Ising_F_1d(beta, 128) for beta in betas])
        
        #Fs_dif = np.abs(Fs_dif)
        F_stds = np.array([F_dict[modelconfig][beta][1] for beta in betas])
        dif_ax.plot(
            betas,
            Fs_dif,
            label=modelconfig,
            **kwargs)
    dif_ax.legend()
    dif_ax.set_yscale("log")
    return fig, ax, dif_fig, dif_ax


def gamma_r(beta, r, n):
    coshgam = np.cosh(beta * 2) / np.tanh(beta * 2) - np.cos(r * np.pi / n)
    return np.arccosh(coshgam)

def Ising_F_1d(beta, N):
    F = -1 / beta * (np.log(2*np.cosh(beta)) + 1/ N * np.log(1+np.tanh(beta)**N))
    return F  

def get_F_dict(root_dir: str):
    F_dict = {}
    for systemname in os.listdir(root_dir):
        beta = float(re.search(r".*beta([\d\.]+)", systemname).group(1))
        Length = int(re.search(r".*L([\d\.]+)", systemname).group(1))

        print(beta)
        for modelconfig in os.listdir(
                os.path.join(root_dir, systemname)):
            if not F_dict.get(modelconfig):
                F_dict[modelconfig] = {}
            logfile = os.path.join(
                root_dir,
                systemname,
                modelconfig,
                "out.log")
            # read the last line that contain "F = {} ..."
            with open(logfile, "r") as f:
                for line in f.readlines():
                    if not line.startswith("step ="):
                        continue
                    lastline = line
                F = float(re.search(r".*F = ([\d\.-]+)", lastline).group(1))
                F_std = float(
                    re.search(
                        r".*F_std = ([\d\.\w-]+),",
                        lastline).group(1))
                F_dict[modelconfig][beta] = (F, F_std)
    return F_dict


if __name__ == "__main__":
    F_dict_row = get_F_dict("./out")

    
    fig_F, ax_F, dif_Fig, dif_ax = plot_F(F_dict_row, marker="o", alpha=0.5)
    betas = np.linspace(0.4, 1.2, 100)
    F_anals = np.array([Ising_F_1d(beta, 128) for beta in betas])
    ax_F.plot(betas, F_anals, label="exact")
    ax_F.legend()
    fig_F.savefig("F.png")
    dif_Fig.savefig("F_dif.png")
    
    