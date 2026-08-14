import time

import torch
import torch.nn as nn

from src.device import cpu_device


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module, quantized: bool = False) -> float:
    if quantized:
        qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        target = qmodel
    else:
        target = model
    n_bytes = sum(p.numel() * p.element_size() for p in target.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in target.buffers())
    return n_bytes / (1024 ** 2)


def flops(model: nn.Module, input_shape: tuple[int, int, int, int] = (1, 3, 224, 224)) -> float:
    from thop import profile

    dummy = torch.randn(*input_shape)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return macs * 2  # FLOPs ~= 2x MACs


def cpu_latency_ms(model: nn.Module, input_shape: tuple[int, int, int, int] = (1, 3, 224, 224), n_runs: int = 100) -> float:
    """Batch=1 CPU inference latency, forcing single-thread to simulate an edge device."""
    torch.set_num_threads(1)
    device = cpu_device()
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(dummy)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000
