"""GPU utility fallback behavior tests."""

import importlib


def test_force_cpu_disables_gpu_helpers(monkeypatch) -> None:
    monkeypatch.setenv("PPE_MONITOR_FORCE_CPU", "1")
    import app.gpu_utils as gpu_utils

    importlib.reload(gpu_utils)

    assert gpu_utils.force_cpu_enabled() is True
    assert gpu_utils.cupy_available() is False
    assert gpu_utils.cv2_cuda_available() is False
    assert gpu_utils.nvjpeg_available() is False


def test_gpu_helpers_return_booleans_without_force_cpu(monkeypatch) -> None:
    monkeypatch.delenv("PPE_MONITOR_FORCE_CPU", raising=False)
    import app.gpu_utils as gpu_utils

    importlib.reload(gpu_utils)
    gpu_utils.clear_detection_caches()

    assert isinstance(gpu_utils.force_cpu_enabled(), bool)
    assert isinstance(gpu_utils.cupy_available(), bool)
    assert isinstance(gpu_utils.cv2_cuda_available(), bool)
    assert isinstance(gpu_utils.nvjpeg_available(), bool)

    summary = gpu_utils.summarize_gpu_stack()
    assert isinstance(summary, dict)
    assert "force_cpu" in summary
