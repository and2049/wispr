from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class RuntimeSelection:
    device: str
    compute_type: str
    batch_size: int
    note: str | None = None


def resolve_runtime(
    *,
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int | None = None,
) -> RuntimeSelection:
    requested_device = device.lower()
    cuda, note = cuda_available()
    if requested_device == "auto":
        resolved_device = "cuda" if cuda else "cpu"
    elif requested_device == "cuda" and not cuda:
        reason = note or "torch reports CUDA is unavailable"
        raise RuntimeError(f"--device cuda was requested, but CUDA is unavailable: {reason}.")
    else:
        resolved_device = requested_device

    return RuntimeSelection(
        device=resolved_device,
        compute_type=resolve_compute_type(compute_type, resolved_device),
        batch_size=batch_size or default_batch_size(resolved_device),
        note=note if requested_device == "auto" and resolved_device == "cpu" else None,
    )


def resolve_compute_type(value: str, device: str) -> str:
    if value != "auto":
        return value
    return "float16" if device == "cuda" else "int8"


def default_batch_size(device: str) -> int:
    return 16 if device == "cuda" else 4


def cuda_available() -> tuple[bool, str | None]:
    try:
        torch: Any = import_module("torch")
    except ImportError:
        return False, "torch is not installed; falling back to CPU"
    try:
        return bool(torch.cuda.is_available()), None
    except Exception as error:
        return False, f"could not inspect torch CUDA state: {error}"
