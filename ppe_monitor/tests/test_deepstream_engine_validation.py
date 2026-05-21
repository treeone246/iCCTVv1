"""Tests for DeepStream TensorRT engine validation."""

from pathlib import Path

import pytest

from app.deepstream.engine_utils import EngineValidationError, validate_engine_exists


def test_validate_engine_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.engine"
    with pytest.raises(EngineValidationError) as exc:
        validate_engine_exists(missing)
    assert "Please export the engine first" in str(exc.value)


def test_validate_engine_wrong_extension_raises(tmp_path: Path) -> None:
    wrong = tmp_path / "model.onnx"
    wrong.write_text("dummy", encoding="utf-8")
    with pytest.raises(EngineValidationError):
        validate_engine_exists(wrong)


def test_validate_engine_ok_for_engine(tmp_path: Path) -> None:
    engine = tmp_path / "best2.engine"
    engine.write_bytes(b"dummy")
    resolved = validate_engine_exists(engine)
    assert resolved.exists()
