import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from script.sample_ising2d import calc_properties, sample_ising2d
from script.args import args
import model

from utils import init_out_filename, ignore_param, my_log, print_args

torch.set_float32_matmul_precision('high')
def main():
    parallel_experiment = 10
    Cvs = []
    Energies = []
    Free_energies = []
    Magnetizations = []
    Magnetizations_abs = []
    betas = np.arange(0.2, 1.1, 0.1)
    for beta in betas:
        args.beta = beta
        init_out_filename()
        last_step = args.max_step
        if last_step >= 0:
            pass
            #print("\nCheckpoint found: {}\n".format(last_step))
        else:
            raise ValueError("No checkpoint found")
        #print_args()

        modelinterface = model.ModelInterface(**vars(args))
        #print(modelinterface.model)
        modelinterface.model.to(args.device)
        
        state = torch.load("{}_save/{}.state".format(args.out_filename, last_step), weights_only=True)
        ignore_param(state["net"], modelinterface.model)
        modelinterface.load_state_dict(state["net"])
        
        exp_set = {}
        start_time = time.time()
        for i in range(parallel_experiment):
            sample, x_hat, sample_time = sample_ising2d(modelinterface.model, args.batch_size)
            properties_exp = calc_properties(modelinterface.model, sample, args.beta, **vars(args))
            for key, value in properties_exp.items():
                if not exp_set.get(key):
                    exp_set[key] = []
                exp_set[key].append(value.cpu().numpy())
        properties = {}
        for key, value in exp_set.items():
            properties[key] = (np.mean(value), np.std(value))
        Cvs.append(properties["capacity"])
        Energies.append(properties["energy"])
        Free_energies.append(properties["free_energy"])
        Magnetizations.append(properties["mag"])
        Magnetizations_abs.append(properties["mag_abs"])
        print("beta = {}, time = {}".format(beta, time.time() - start_time))
        
    fig, axs = plt.subplots(2, 2)
    Cv_mean, Cv_std = zip(*Cvs)
    axs[0, 0].errorbar(betas, Cv_mean, yerr=Cv_std)
    axs[0, 0].set_title('Specific Heat Capacity')
    Energy_mean, Energy_std = zip(*Energies)
    axs[0, 1].errorbar(betas, Energy_mean, yerr=Energy_std)
    axs[0, 1].set_title('Energy')
    Free_energy_mean, Free_energy_std = zip(*Free_energies)
    axs[1, 0].errorbar(betas, Free_energy_mean, yerr=Free_energy_std)
    axs[1, 0].set_title('Free Energy')
    Magnetization_mean, Magnetization_std = zip(*Magnetizations_abs)
    axs[1, 1].errorbar(betas, Magnetization_mean, yerr=Magnetization_std)
    axs[1, 1].set_title('Magnetization')
    fig.tight_layout()
    fig.savefig('properties.png')
if __name__ == "__main__":
    main()