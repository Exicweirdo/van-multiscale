import script.train_ising1d, script.train_ising2d
import torch
torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    # script.main_ising1d.main()
    script.train_ising2d.main()
