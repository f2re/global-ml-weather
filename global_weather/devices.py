"""Explicit CUDA-first execution; no silent fallback after training starts."""

from __future__ import annotations
import os
import re
import torch
from torch.utils import deterministic


_CUBLAS_CONFIGS = (":4096:8", ":16:8")
# Set before callers start CUDA work, including during test collection. Keep
# operator settings; an initialized context cannot be repaired retroactively.
if not torch.cuda.is_initialized():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_CONFIGS[0])
_CUBLAS_CONFIG_BEFORE_INIT = os.environ.get("CUBLAS_WORKSPACE_CONFIG")


def configure_training_numerics(device: str | torch.device) -> None:
    """Require deterministic algorithms on a fixed numerical environment."""
    global _CUBLAS_CONFIG_BEFORE_INIT
    device = torch.device(device)
    if device.type == "cuda":
        initialized = torch.cuda.is_initialized()
        config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if config is None and not initialized:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_CONFIGS[0]
            config = _CUBLAS_CONFIGS[0]
        if config not in _CUBLAS_CONFIGS:
            raise ValueError(
                "Для CUDA требуется CUBLAS_WORKSPACE_CONFIG=:4096:8 или :16:8 "
                "до инициализации CUDA. Перезапустите процесс с этой настройкой."
            )
        if initialized and config != _CUBLAS_CONFIG_BEFORE_INIT:
            raise ValueError(
                "CUBLAS_WORKSPACE_CONFIG изменён после запуска CUDA. "
                "Перезапустите процесс с постоянной настройкой."
            )
        if not initialized:
            _CUBLAS_CONFIG_BEFORE_INIT = config
    torch.use_deterministic_algorithms(True, warn_only=False)
    deterministic.fill_uninitialized_memory = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def training_runtime_identity(device: str | torch.device) -> dict[str, object]:
    """Record actual process settings, rather than just intended policy flags."""
    device = torch.device(device)
    return {
        **device_identity(device),
        "numerics": {
            "policy": "strict-deterministic-1",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "fill_uninitialized_memory": deterministic.fill_uninitialized_memory,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if device.type == "cuda"
                else None
            ),
        },
    }


def select_device(requested="auto"):
    if not isinstance(requested, str) or not re.fullmatch(
        r"auto|cpu|cuda(?::[0-9]+)?", requested
    ):
        raise ValueError("Устройство: auto, cpu, cuda или cuda:N.")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA недоступна. Проверьте драйвер и сборку PyTorch или выберите CPU."
            )
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
        result.update(
            name=p.name,
            capability=list(torch.cuda.get_device_capability(device)),
            total_memory=p.total_memory,
            count=torch.cuda.device_count(),
        )
    return result


def inventory():
    available = torch.cuda.is_available()
    return {
        "default": "cuda:0" if available else "cpu",
        "cuda_available": available,
        "torch": str(torch.__version__),
        "devices": [device_identity("cpu")]
        + (
            [device_identity(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            if available
            else []
        ),
    }
