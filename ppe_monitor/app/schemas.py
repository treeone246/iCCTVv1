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


class MetricsPayload(BaseModel):
    """Top-line pipeline counters."""

    fps: float
    verifier_calls_last_sec: int
    tracked_count: int
    dropped_frames: int
    active_violations: int
    compliance_rate: float


class FramePayload(BaseModel):
    """Frame-level websocket metadata payload."""

    model_config = ConfigDict(use_enum_values=True)

    frame_id: int
    timestamp: float
    persons: List[PersonPayload]
    ppe_detections: List[BBoxPayload]
    active_alerts: List[AlertPayload]
    metrics: MetricsPayload
