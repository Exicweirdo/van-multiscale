import script.mc_EA
import torch
torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    script.mc_EA.main()