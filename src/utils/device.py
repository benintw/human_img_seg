import torch


def get_device(device_name: str | None = None) -> torch.device:
    """Get the appropriate device for training/

    Args:
        None

    Returns:
        torch.device: The Pytorch device object ("mps", "cuda", "cpu")
    """

    if device_name is not None:
        return torch.device(device_name)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
