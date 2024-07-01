import torch
from torch import device


def choose_device() -> device:
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

        device = torch.device("cpu")

    else:
        device = torch.device("mps")
        print("Running on MPS.")

    return device
