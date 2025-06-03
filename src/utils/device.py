import torch


def setup_device() -> torch.device:
    """
    Setup device. If Cuda is available then CUDA, CPU otherwise.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
