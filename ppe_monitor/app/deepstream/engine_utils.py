"""TensorRT engine helpers for DeepStream backend."""

from __future__ import annotations

from pathlib import Path


class EngineValidationError(RuntimeError):
    """Raised when expected TensorRT engine is missing or invalid."""


def validate_engine_exists(engine_path: Path) -> Path:
    path = Path(engine_path).resolve()
    if not path.exists():
        raise EngineValidationError(
            f"TensorRT engine not found: {path.as_posix()}\n"
            "Please export the engine first:\n"
            "yolo export model=models/best2.pt format=engine device=0 half=True imgsz=640"
        )
    if not path.is_file():
        raise EngineValidationError(f"TensorRT engine path is not a file: {path.as_posix()}")
    if path.suffix.lower() not in {".engine", ".plan"}:
        raise EngineValidationError(
            f"Unexpected engine extension for {path.name}. Expected .engine or .plan."
        )
    return path
