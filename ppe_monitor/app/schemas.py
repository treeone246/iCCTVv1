"""Shared enums and payload schemas for PPE monitoring pipeline outputs."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class Classification(str, Enum):
    """Per-item frame classification state."""

    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    VIOLATION_TENTATIVE = "VIOLATION_TENTATIVE"
    INDETERMINATE = "INDETERMINATE"


class AlertStatus(str, Enum):
    """Alert lifecycle status."""

    ACTIVE = "ACTIVE"
    CLEARED = "CLEARED"


class OverallStatus(str, Enum):
    """Rollup status for person card/bbox rendering."""

    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    INDETERMINATE = "INDETERMINATE"


class VerifierVerdict(str, Enum):
    """Verifier binary result."""

    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    INDETERMINATE = "INDETERMINATE"


class VerifierResult(BaseModel):
    """Verifier output for one person-item check."""

    verdict: VerifierVerdict
    score: float = 0.0
    source: str = "verifier"


class KeypointPayload(BaseModel):
    """Single keypoint represented in pixel coordinates."""

    name: str
    x: float
    y: float
    conf: float


class BBoxPayload(BaseModel):
    """Detection/tracking box payload."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    label: str
    person_id: Optional[int] = None
    source: Optional[str] = None


class PersonPayload(BaseModel):
    """Per-person dashboard payload."""

    person_id: int
    bbox: List[float]
    keypoints: Dict[str, KeypointPayload]
    per_item_state: Dict[str, Classification]
    per_item_reason: Dict[str, str] = {}
    overall_status: OverallStatus
    helmet_color: str = "unknown"
    helmet_color_confidence: float = 0.0
    person_status: str = "Unknown role"


class AlertPayload(BaseModel):
    """Alert card payload."""

    alert_id: str
    person_id: int
    item: str
    status: AlertStatus
    reason: str
    timestamp: float
    evidence_available: bool
    evidence_jpeg_base64: Optional[str] = None
    person_status: str = "Unknown role"
    helmet_color: str = "unknown"
    helmet_color_confidence: float = 0.0
    person_crop_jpeg_base64: Optional[str] = None
    item_crop_jpeg_base64: Optional[str] = None


class MetricsPayload(BaseModel):
    """Top-line pipeline counters."""

    fps: float
    verifier_calls_last_sec: int
    tracked_count: int
    dropped_frames: int
    active_violations: int
    compliance_rate: float
    ppe_primary_raw: int = 0
    verifier_aux_raw: int = 0
    ppe_merged: int = 0
    pose_infer_calls: int = 0
    ppe_infer_calls: int = 0
    verifier_aux_infer_calls: int = 0
    verifier_crop_infer_calls: int = 0
    verifier_ollama_calls: int = 0
    ppe_model: str = ""
    ppe_task: str = ""
    ppe_fusion_mode: str = "nms"
    adaptive_scheduler_enabled: bool = False
    adaptive_detect_frames: int = 0
    adaptive_reuse_frames: int = 0
    async_verifier_enabled: bool = False
    async_verifier_enqueued: int = 0
    async_verifier_completed: int = 0
    async_verifier_dropped: int = 0
    compute_monitor_enabled: bool = False
    pose_infer_per_sec: float = 0.0
    ppe_infer_per_sec: float = 0.0
    verifier_aux_infer_per_sec: float = 0.0
    verifier_crop_infer_per_sec: float = 0.0
    verifier_ollama_calls_per_sec: float = 0.0
    pose_estimated_gflops_per_sec: float = 0.0
    ppe_estimated_gflops_per_sec: float = 0.0
    verifier_aux_estimated_gflops_per_sec: float = 0.0
    verifier_crop_estimated_gflops_per_sec: float = 0.0
    verifier_ollama_estimated_gflops_per_sec: float = 0.0
    estimated_gflops_per_sec: float = 0.0
    estimated_tflops_per_sec: float = 0.0
    estimated_tops_per_sec: float = 0.0
    estimated_flops_per_sec: float = 0.0
    estimated_compute_utilization_pct: float = 0.0
    memory_monitor_enabled: bool = False
    process_rss_mb: float = 0.0
    process_vms_mb: float = 0.0
    system_memory_used_mb: float = 0.0
    system_memory_total_mb: float = 0.0
    system_memory_utilization_pct: float = 0.0
    backend: str = "python"


class FramePayload(BaseModel):
    """Frame-level websocket metadata payload."""

    model_config = ConfigDict(use_enum_values=True)

    frame_id: int
    timestamp: float
    persons: List[PersonPayload]
    ppe_detections: List[BBoxPayload]
    active_alerts: List[AlertPayload]
    metrics: MetricsPayload
