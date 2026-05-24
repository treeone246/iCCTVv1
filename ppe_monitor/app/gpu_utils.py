"""Lazy GPU capability detection helpers with CPU-force override."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional


_TRUE_VALUES = {"1", "true", "yes", "on"}


def force_cpu_enabled() -> bool:
    """Return True when GPU paths should be disabled explicitly."""
    raw = os.getenv("PPE_MONITOR_FORCE_CPU", "")
    return raw.strip().lower() in _TRUE_VALUES


@lru_cache(maxsize=1)
def cupy_available() -> bool:
    """Check CuPy availability lazily."""
    if force_cpu_enabled():
        return False
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def get_cupy_module() -> Optional[Any]:
    """Return imported CuPy module when available, else None."""
    if not cupy_available():
        return None
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None
    return cp


@lru_cache(maxsize=1)
def cv2_cuda_available() -> bool:
    """Check OpenCV CUDA runtime availability lazily."""
    if force_cpu_enabled():
        return False
    try:
        import cv2
    except Exception:
        return False
    if not hasattr(cv2, "cuda"):
        return False
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount()) > 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def nvjpeg_available() -> bool:
    """Return True when a Python nvJPEG module is importable."""
    if force_cpu_enabled():
        return False
    # Keep this check intentionally conservative; many hosts do not expose
    # a stable nvjpeg Python module.
    try:
        import nvjpeg  # type: ignore # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def torchvision_nms_available() -> bool:
    """Return True when torchvision NMS API is importable."""
    if force_cpu_enabled():
        return False
    try:
        import torch  # noqa: F401
        from torchvision.ops import nms  # noqa: F401
    except Exception:
        return False
    return True


def summarize_gpu_stack() -> Dict[str, Any]:
    """Summarize GPU helper detection results for diagnostics."""
    return {
        "force_cpu": force_cpu_enabled(),
        "cupy_available": cupy_available(),
        "cv2_cuda_available": cv2_cuda_available(),
        "nvjpeg_available": nvjpeg_available(),
        "torchvision_nms_available": torchvision_nms_available(),
    }


def clear_detection_caches() -> None:
    """Clear cached detection state (useful for tests)."""
    cupy_available.cache_clear()
    get_cupy_module.cache_clear()
    cv2_cuda_available.cache_clear()
    nvjpeg_available.cache_clear()
    torchvision_nms_available.cache_clear()
