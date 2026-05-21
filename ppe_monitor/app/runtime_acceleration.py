"""Runtime acceleration summary helpers for dashboard/API."""

from __future__ import annotations

from typing import Any, Dict, Mapping


GPU_ARTIFACT_FORMATS = {"engine", "plan", "trt", "tensorrt"}


def summarize_runtime_acceleration(provider_info: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize CUDA/GPU acceleration status from startup provider diagnostics."""
    models: Dict[str, Any] = {}
    cuda_enabled = False
    gpu_acceleration_enabled = False
    all_models_gpu_accelerated = True
    checked_models = 0

    for model_key, raw in dict(provider_info or {}).items():
        info = dict(raw or {})
        mode = str(info.get("mode", "")).lower()
        if mode == "mock":
            models[model_key] = {
                "mode": "mock",
                "artifact_format": str(info.get("artifact_format", "")),
                "active_provider": str(info.get("active_provider", "")),
                "gpu_accelerated": False,
                "cuda_provider": False,
                "reason": "mock_model",
            }
            continue

        artifact = str(info.get("artifact_format", "")).lower()
        active_provider = str(info.get("active_provider", ""))
        cuda_provider = active_provider == "CUDAExecutionProvider"
        gpu_by_artifact = artifact in GPU_ARTIFACT_FORMATS
        gpu_accelerated = bool(cuda_provider or gpu_by_artifact)

        checked_models += 1
        cuda_enabled = cuda_enabled or cuda_provider
        gpu_acceleration_enabled = gpu_acceleration_enabled or gpu_accelerated
        if not gpu_accelerated:
            all_models_gpu_accelerated = False

        if cuda_provider:
            reason = "onnx_cuda_provider"
        elif gpu_by_artifact:
            reason = f"artifact_{artifact}"
        elif active_provider == "CPUExecutionProvider":
            reason = "onnx_cpu_provider"
        else:
            reason = "unknown_or_cpu_path"

        models[model_key] = {
            "mode": mode or "model",
            "artifact_format": artifact,
            "active_provider": active_provider,
            "gpu_accelerated": gpu_accelerated,
            "cuda_provider": cuda_provider,
            "reason": reason,
        }

    if checked_models == 0:
        all_models_gpu_accelerated = False

    return {
        "cuda_enabled": cuda_enabled,
        "gpu_acceleration_enabled": gpu_acceleration_enabled,
        "all_models_gpu_accelerated": all_models_gpu_accelerated,
        "checked_models": checked_models,
        "models": models,
    }
