import os

import torch
from torch import device


def choose_device() -> device:
    if torch.cuda.is_available():
        cuda_id = "cuda:" + os.getenv("CUDA", default="0")
        device = torch.device(cuda_id)
        print(f"Running on CUDA device {cuda_id}")

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Running on MPS.")

    else:
        device = torch.device("cpu")
        print("Running on CPU.")
        print("If you're on a Mac, check that you have MacOS 12.3+, an MPS-enabled chip, and current Pytorch.")

    return device
