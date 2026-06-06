"""Hardware detection + model-size selection for on-device realtime.

pick_device probes CUDA (injected for tests) and returns (device, compute_type).
realtime_model_size chooses the latency-tuned default per device; batch always
uses large-v3 (decided in the spec) and does not go through this function.
"""
from __future__ import annotations

from typing import Callable


def _cuda_available() -> bool:  # pragma: no cover - native
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def pick_device(cuda_available: Callable[[], bool] = _cuda_available):
    if cuda_available():
        return ("cuda", "float16")
    return ("cpu", "int8")


def realtime_model_size(device: str, override: str | None = None) -> str:
    if override:
        return override
    return "large-v3" if device == "cuda" else "small"
