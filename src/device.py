import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a training/inference device: CUDA > MPS > CPU, with graceful fallback.

    Never assumes CUDA is present (Mac has none) or MPS is present (Colab has none).
    """
    if prefer != "auto":
        return torch.device(prefer)

    if torch.cuda.is_available():
        print(f"[device] using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        print("[device] using MPS")
        return torch.device("mps")

    print("[device] using CPU")
    return torch.device("cpu")


def cpu_device() -> torch.device:
    """Edge-inference latency measurements always run on CPU, regardless of training device."""
    return torch.device("cpu")
