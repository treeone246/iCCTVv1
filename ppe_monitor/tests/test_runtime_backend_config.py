"""Tests for runtime backend config parsing."""

import pytest

from app.runtime_backend import RuntimeBackendConfigError, get_runtime_backend


def test_runtime_backend_defaults_to_python() -> None:
    assert get_runtime_backend({}) == "python"


def test_runtime_backend_accepts_deepstream() -> None:
    assert get_runtime_backend({"runtime": {"backend": "deepstream"}}) == "deepstream"


def test_runtime_backend_rejects_unknown_value() -> None:
    with pytest.raises(RuntimeBackendConfigError):
        get_runtime_backend({"runtime": {"backend": "invalid_backend"}})
