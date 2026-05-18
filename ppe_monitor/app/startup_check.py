"""Startup model loading, ONNX provider diagnostics, and mock substitution."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from .pose_tracker import MockPoseTracker, PoseTrackerBase, YOLOPoseTracker
from .ppe_detector import MockPPEDetector, PPEDetectorBase, YOLOPPEDetector
from .verifier import (
    HybridVerifier,
    MockVerifier,
    OllamaVLMClient,
    VerifierBase,
    YOLOEVerifier,
)

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - runtime dependency may differ by host
    ort = None


@dataclass
class RuntimeComponents:
    """Loaded runtime components used by pipeline and app."""

    pose_tracker: PoseTrackerBase
    ppe_detector: PPEDetectorBase
    verifier: VerifierBase
    provider_info: Dict[str, dict]


def log_event(event_type: str, **fields: object) -> None:
    payload = {"event_type": event_type, **fields}
    print(json.dumps(payload, default=str))


def _resolve_model_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _provider_preference(device: str) -> List[str]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _expected_provider(device: str) -> str:
    if device == "cpu":
        return "CPUExecutionProvider"
    return "CUDAExecutionProvider"


def _sanitize_shape(raw_shape: List[object], imgsz: int) -> List[int]:
    safe: List[int] = []
    for idx, dim in enumerate(raw_shape):
        if isinstance(dim, int) and dim > 0:
            safe.append(dim)
            continue
        if idx == 0:
            safe.append(1)
        elif idx == 1:
            safe.append(3)
        elif idx in (2, 3):
            safe.append(imgsz)
        else:
            safe.append(1)
    return safe


def inspect_onnx_model(path: Path, providers: List[str], imgsz: int) -> dict:
    if ort is None:
        return {
            "available": False,
            "warning": "onnxruntime is not importable; provider diagnostics skipped.",
        }

    available = ort.get_available_providers()
    filtered = [provider for provider in providers if provider in available]
    if not filtered:
        filtered = ["CPUExecutionProvider"]

    session = ort.InferenceSession(path.as_posix(), providers=filtered)
    inputs = [{"name": inp.name, "shape": list(inp.shape)} for inp in session.get_inputs()]
    outputs = [{"name": out.name, "shape": list(out.shape)} for out in session.get_outputs()]
    active = session.get_providers()

    warmup_ms: Optional[float] = None
    if inputs:
        inp = session.get_inputs()[0]
        dummy_shape = _sanitize_shape(list(inp.shape), imgsz)
        dummy = np.random.rand(*dummy_shape).astype(np.float32)
        start = time.perf_counter()
        session.run(None, {inp.name: dummy})
        warmup_ms = (time.perf_counter() - start) * 1000.0

    return {
        "available": True,
        "requested_providers": filtered,
        "active_provider": active[0] if active else "unknown",
        "inputs": inputs,
        "outputs": outputs,
        "warmup_ms": warmup_ms,
    }


def inspect_model_artifact(path: Path, providers: List[str], imgsz: int) -> dict:
    """Inspect model metadata when possible.

    ONNX files are inspected via onnxruntime. Non-ONNX artifacts (e.g. TensorRT
    .engine) are reported with minimal metadata and skipped from ORT probing.
    """
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        info = inspect_onnx_model(path, providers, imgsz)
        info["artifact_format"] = "onnx"
        return info

    return {
        "available": True,
        "artifact_format": suffix.lstrip(".") or "unknown",
        "active_provider": "n/a",
        "inputs": None,
        "outputs": None,
        "warmup_ms": None,
        "note": "Skipping ONNX Runtime inspection for non-ONNX artifact.",
    }


def load_runtime_components(config: dict, project_root: Path) -> RuntimeComponents:
    """Load or mock models, and emit startup diagnostics."""

    allow_mock = bool(config["models"]["allow_mock_models"])
    infer_cfg = config["inference"]
    model_cfg = config["models"]

    device = str(infer_cfg.get("device", "auto")).lower()
    provider_pref = _provider_preference(device)
    expected_provider = _expected_provider(device)
    imgsz = int(infer_cfg.get("imgsz", 640))

    specs = [
        ("pose", model_cfg["pose"], "pose"),
        ("ppe", model_cfg["ppe"], "detect"),
        ("verifier", model_cfg["verifier"], "detect"),
    ]

    provider_info: Dict[str, dict] = {}
    loaded_models: Dict[str, Optional[YOLO]] = {"pose": None, "ppe": None, "verifier": None}

    for key, configured_path, task in specs:
        model_path = _resolve_model_path(project_root, configured_path)
        model_exists = model_path.exists()

        if not model_exists:
            if allow_mock:
                log_event(
                    "model_missing_mock_enabled",
                    model_key=key,
                    path=model_path.as_posix(),
                    allow_mock_models=True,
                )
                provider_info[key] = {"mode": "mock", "path": model_path.as_posix()}
                continue
            raise FileNotFoundError(
                f"Required model missing and allow_mock_models is false: {model_path.as_posix()}"
            )

        file_size = model_path.stat().st_size
        diagnostics = inspect_model_artifact(model_path, provider_pref, imgsz)
        provider_info[key] = diagnostics
        provider_info[key]["path"] = model_path.as_posix()
        provider_info[key]["size_bytes"] = file_size
        provider_info[key]["task"] = task

        log_event(
            "model_startup_check",
            model_key=key,
            path=model_path.as_posix(),
            size_bytes=file_size,
            artifact_format=diagnostics.get("artifact_format"),
            inputs=diagnostics.get("inputs"),
            outputs=diagnostics.get("outputs"),
            active_provider=diagnostics.get("active_provider"),
            warmup_ms=diagnostics.get("warmup_ms"),
        )

        log_event(
            "model_backend_selected",
            model_key=key,
            artifact_format=diagnostics.get("artifact_format"),
            backend_hint=(
                "onnxruntime" if diagnostics.get("artifact_format") == "onnx" else "tensorrt/ultralytics"
            ),
            path=model_path.as_posix(),
        )

        active_provider = diagnostics.get("active_provider", "")
        if (
            diagnostics.get("artifact_format") == "onnx"
            and expected_provider == "CUDAExecutionProvider"
            and active_provider != "CUDAExecutionProvider"
        ):
            log_event(
                "provider_warning",
                model_key=key,
                expected_provider=expected_provider,
                active_provider=active_provider,
                message="CUDA requested/auto but CPU provider is active.",
            )

        loaded_models[key] = YOLO(model_path.as_posix(), task=task)

    pose_tracker: PoseTrackerBase
    ppe_detector: PPEDetectorBase
    verifier: VerifierBase

    if loaded_models["pose"] is None:
        pose_tracker = MockPoseTracker()
    else:
        pose_tracker = YOLOPoseTracker(
            model=loaded_models["pose"],
            conf_threshold=float(infer_cfg["conf_threshold_pose"]),
            imgsz=imgsz,
        )

    if loaded_models["ppe"] is None:
        ppe_detector = MockPPEDetector()
    else:
        label_aliases = config.get("ppe_label_aliases", {})
        ppe_detector = YOLOPPEDetector(
            model=loaded_models["ppe"],
            conf_threshold=float(infer_cfg["conf_threshold_ppe"]),
            imgsz=imgsz,
            label_aliases=label_aliases,
        )

    if loaded_models["verifier"] is None:
        verifier = MockVerifier()
    else:
        yoloe_verifier = YOLOEVerifier(
            model=loaded_models["verifier"],
            conf_threshold=float(infer_cfg["conf_threshold_verifier"]),
            imgsz=imgsz,
        )
        verifier_cfg = config.get("verifier", {})
        backend = str(verifier_cfg.get("backend", "yoloe")).lower()
        if backend == "ollama_hybrid":
            ollama_cfg = verifier_cfg.get("ollama", {})
            ollama_client = OllamaVLMClient(
                host=str(ollama_cfg.get("host", "http://127.0.0.1:11434")),
                model=str(ollama_cfg.get("model", "qwen2.5vl:3b")),
                timeout_seconds=float(ollama_cfg.get("timeout_seconds", 8.0)),
                temperature=float(ollama_cfg.get("temperature", 0.0)),
            )
            conflict_cfg = verifier_cfg.get("conflict_resolver", {})
            verifier = HybridVerifier(
                yoloe=yoloe_verifier,
                ollama=ollama_client,
                labels=dict(verifier_cfg.get("vlm_labels", {})),
                ambiguity_margin=float(conflict_cfg.get("ambiguity_margin", 0.12)),
                low_conf_threshold=float(conflict_cfg.get("low_conf_threshold", 0.40)),
                enable_vlm=bool(verifier_cfg.get("enable_vlm", True)),
            )
            log_event(
                "verifier_backend_selected",
                backend="ollama_hybrid",
                model=str(ollama_cfg.get("model", "qwen2.5vl:3b")),
                host=str(ollama_cfg.get("host", "http://127.0.0.1:11434")),
            )
        else:
            verifier = yoloe_verifier
            log_event("verifier_backend_selected", backend="yoloe")

    return RuntimeComponents(
        pose_tracker=pose_tracker,
        ppe_detector=ppe_detector,
        verifier=verifier,
        provider_info=provider_info,
    )
