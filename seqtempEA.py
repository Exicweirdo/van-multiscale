import script.ST_EA
import torch
torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    # script.main_ising1d.main()
    script.ST_EA.main()