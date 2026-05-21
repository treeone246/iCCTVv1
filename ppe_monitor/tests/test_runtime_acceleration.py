"""Tests for runtime acceleration summary helper."""

from app.runtime_acceleration import summarize_runtime_acceleration


def test_cuda_provider_marks_gpu_enabled() -> None:
    summary = summarize_runtime_acceleration(
        {
            "pose": {
                "artifact_format": "onnx",
                "active_provider": "CUDAExecutionProvider",
            },
            "ppe": {
                "artifact_format": "onnx",
                "active_provider": "CPUExecutionProvider",
            },
        }
    )
    assert summary["cuda_enabled"] is True
    assert summary["gpu_acceleration_enabled"] is True
    assert summary["models"]["pose"]["gpu_accelerated"] is True
    assert summary["models"]["ppe"]["gpu_accelerated"] is False


def test_engine_artifact_marks_gpu_even_without_cuda_provider() -> None:
    summary = summarize_runtime_acceleration(
        {
            "verifier": {
                "artifact_format": "engine",
                "active_provider": "n/a",
            }
        }
    )
    assert summary["cuda_enabled"] is False
    assert summary["gpu_acceleration_enabled"] is True
    assert summary["all_models_gpu_accelerated"] is True
    assert summary["models"]["verifier"]["gpu_accelerated"] is True


def test_mock_model_not_counted_as_gpu_enabled() -> None:
    summary = summarize_runtime_acceleration(
        {
            "pose": {"mode": "mock"},
        }
    )
    assert summary["checked_models"] == 0
    assert summary["cuda_enabled"] is False
    assert summary["gpu_acceleration_enabled"] is False
    assert summary["all_models_gpu_accelerated"] is False
