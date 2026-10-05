"""Explicit CUDA-first execution; no silent fallback after training starts."""
from __future__ import annotations
import re
import torch


def select_device(requested="auto"):
    if not isinstance(requested, str) or not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", requested):
        raise ValueError("Устройство: auto, cpu, cuda или cuda:N.")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA недоступна. Проверьте драйвер и сборку PyTorch или выберите CPU.")
        index = int(requested.split(":")[1]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            raise ValueError("Выбранное устройство CUDA отсутствует.")
        requested = f"cuda:{index}"
    return torch.device(requested)


def device_identity(device):
    device = torch.device(device)
    result = {"device": str(device), "torch_cuda": torch.version.cuda}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        result.update(name=p.name, capability=list(torch.cuda.get_device_capability(device)),
                      total_memory=p.total_memory, count=torch.cuda.device_count())
    return result


def inventory():
    available = torch.cuda.is_available()
    return {"default": "cuda:0" if available else "cpu", "cuda_available": available,
            "torch": str(torch.__version__), "devices": [device_identity("cpu")] +
            ([device_identity(f"cuda:{i}") for i in range(torch.cuda.device_count())] if available else [])}
