"""JPEG encoding helpers with optional nvJPEG backend and safe CPU fallback."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from .gpu_utils import nvjpeg_available


@lru_cache(maxsize=1)
def _get_nvjpeg_encoder() -> object | None:
    if not nvjpeg_available():
        return None
    try:
        import nvjpeg  # type: ignore
    except Exception:
        return None
    if hasattr(nvjpeg, "NvJpeg"):
        try:
            return nvjpeg.NvJpeg()  # type: ignore[attr-defined]
        except Exception:
            return None
    return nvjpeg


def _try_nvjpeg(image: np.ndarray, quality: int) -> Optional[bytes]:
    encoder = _get_nvjpeg_encoder()
    if encoder is None:
        return None

    # Try common API variants conservatively.
    for attr in ("encode", "encode_image", "imencode"):
        fn = getattr(encoder, attr, None)
        if fn is None:
            continue
        try:
            out = fn(image, quality=quality)  # type: ignore[misc]
        except TypeError:
            try:
                out = fn(image, quality)  # type: ignore[misc]
            except Exception:
                continue
        except Exception:
            continue

        if out is None:
            continue
        if isinstance(out, (bytes, bytearray)):
            return bytes(out)
        if isinstance(out, memoryview):
            return out.tobytes()
        if hasattr(out, "tobytes"):
            try:
                return out.tobytes()
            except Exception:
                continue
    return None


def encode_jpeg_bytes(image: np.ndarray, quality: int = 85, backend: str = "auto") -> Optional[bytes]:
    """Encode image to JPEG bytes.

    backend:
    - auto: try nvjpeg, fallback to cv2.imencode
    - nvjpeg: force nvjpeg attempt, fallback to cv2 on failure
    - cpu: use cv2.imencode only
    """
    if image is None or image.size == 0:
        return None

    mode = str(backend or "auto").lower()
    if mode in {"auto", "nvjpeg"}:
        encoded = _try_nvjpeg(image, int(quality))
        if encoded is not None:
            return encoded

    ok, arr = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return bytes(arr)
