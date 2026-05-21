"""Runtime backend config helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


VALID_BACKENDS = {"python", "deepstream"}


class RuntimeBackendConfigError(ValueError):
    """Raised when runtime backend configuration is invalid."""


@dataclass
class RuntimeBackendSettings:
    backend: str = "python"


def get_runtime_backend(config: Mapping[str, Any]) -> str:
    runtime_cfg = dict(config.get("runtime", {}) or {})
    backend = str(runtime_cfg.get("backend", "python")).strip().lower() or "python"
    if backend not in VALID_BACKENDS:
        raise RuntimeBackendConfigError(
            f"Invalid runtime.backend '{backend}'. Valid options: {', '.join(sorted(VALID_BACKENDS))}"
        )
    return backend


def resolve_project_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
