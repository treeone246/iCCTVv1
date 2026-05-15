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
from .verifier import MockVerifier, VerifierBase, YOLOEVerifier

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
        diagnostics = inspect_onnx_model(model_path, provider_pref, imgsz)
        provider_info[key] = diagnostics
        provider_info[key]["path"] = model_path.as_posix()
        provider_info[key]["size_bytes"] = file_size
        provider_info[key]["task"] = task

        log_event(
            "model_startup_check",
            model_key=key,
            path=model_path.as_posix(),
            size_bytes=file_size,
            inputs=diagnostics.get("inputs"),
            outputs=diagnostics.get("outputs"),
            active_provider=diagnostics.get("active_provider"),
            warmup_ms=diagnostics.get("warmup_ms"),
        )

        active_provider = diagnostics.get("active_provider", "")
        if expected_provider == "CUDAExecutionProvider" and active_provider != "CUDAExecutionProvider":
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
        ppe_detector = YOLOPPEDetector(
            model=loaded_models["ppe"],
            conf_threshold=float(infer_cfg["conf_threshold_ppe"]),
            imgsz=imgsz,
        )

    if loaded_models["verifier"] is None:
        verifier = MockVerifier()
    else:
        verifier = YOLOEVerifier(
            model=loaded_models["verifier"],
            conf_threshold=float(infer_cfg["conf_threshold_verifier"]),
            imgsz=imgsz,
        )

    return RuntimeComponents(
        pose_tracker=pose_tracker,
        ppe_detector=ppe_detector,
        verifier=verifier,
        provider_info=provider_info,
    )
