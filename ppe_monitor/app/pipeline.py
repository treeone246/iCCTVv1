"""Per-frame orchestration for tracking, association, verification, and alerts."""

import base64
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover - runtime fallback when OpenCV is unavailable
    cv2 = None

from .association import (
    AssociationEngine,
    bbox_center,
    iou as box_iou,
    torso_bbox,
)
from .cache import VerifierCache
from .compute_monitor import ComputeProfile, estimate_compute_usage
from .event_stream import EventStreamWriter
from .face_gate import SCRFDFaceGate
from .gpu_utils import torchvision_nms_available
from .jpeg_utils import encode_jpeg_bytes
from .pose_tracker import PoseTrackerBase, TrackedPerson
from .ppe_detector import PPEDetection, PPEDetectorBase, build_alias_index, canonicalize_label
from .schemas import (
    AlertPayload,
    BBoxPayload,
    Classification,
    FramePayload,
    KeypointPayload,
    MetricsPayload,
    OverallStatus,
    PersonPayload,
    VerifierResult,
    VerifierVerdict,
)
from .state_machine import PersonComplianceState
from .runtime_monitor import read_memory_snapshot
from .verifier import VerifierBase, VerifierContext


@dataclass
class ItemDecision:
    """Final per-item decision and explanatory reason."""

    classification: Classification
    reason: str
    positive_conf: float = 0.0
    negative_conf: float = 0.0


@dataclass
class StableStateTracker:
    """Sticky per-item state to reduce dashboard flicker/spam."""

    stable: Classification = Classification.INDETERMINATE
    compliant_streak: int = 0
    violation_streak: int = 0
    indeterminate_streak: int = 0
    score: float = 0.0
    positive_memory: float = 0.0
    negative_memory: float = 0.0


@dataclass
class VerifierTask:
    """Queued async verifier task payload."""

    person_id: int
    item: str
    item_crop: np.ndarray
    context: VerifierContext


@dataclass
class VerifierOutcome:
    """Completed async verifier result."""

    person_id: int
    item: str
    result: VerifierResult


def log_event(event_type: str, frame_id: int, person_id: int, **fields: object) -> None:
    payload = {"event_type": event_type, "frame_id": frame_id, "person_id": person_id, **fields}
    print(json.dumps(payload, default=str))


HELMET_COLOR_TO_STATUS: Dict[str, str] = {
    "white": "Manager/Supervisor/Engineer",
    "yellow": "Laborer/Operator/Floorhand",
    "blue": "Technician/Electrician/Mechanical Supervisor",
    "red": "Safety Officer/Firefighter",
    "green": "Medical/Paramedic/Environmental Supervisor",
}


class MonitoringPipeline:
    """Frame processor for PPE compliance decisions and websocket payload assembly."""

    def __init__(
        self,
        pose_tracker: PoseTrackerBase,
        ppe_detector: PPEDetectorBase,
        verifier: VerifierBase,
        config: dict,
    ) -> None:
        self.pose_tracker = pose_tracker
        self.ppe_detector = ppe_detector
        self.verifier = verifier
        self.config = config
        self.required_ppe: List[str] = list(config["required_ppe"])
        self.last_item_reason: Dict[Tuple[int, str], str] = {}
        self.last_item_conf: Dict[Tuple[int, str], Tuple[float, float]] = {}

        self.association = AssociationEngine(config)
        self.cache = VerifierCache()
        sm_cfg = config["state_machine"]
        self.state_machine = PersonComplianceState(
            window_size=int(sm_cfg.get("window_size", sm_cfg.get("vote_window_frames", 30))),
            violation_threshold=int(sm_cfg.get("violation_threshold", sm_cfg.get("candidate_violation_frames", 20))),
            clear_threshold=int(sm_cfg.get("clear_threshold", sm_cfg.get("compliant_clear_frames", 1))),
            confirm_seconds=float(sm_cfg.get("confirm_seconds", sm_cfg.get("confirm_stable_seconds", 2.0))),
            cooldown_seconds=float(sm_cfg.get("cooldown_seconds", sm_cfg.get("alert_cooldown_seconds", 60.0))),
        )

        cache_cfg = config["verifier_cache"]
        self.ttl_compliant = float(cache_cfg["ttl_compliant_seconds"])
        self.ttl_violation = float(cache_cfg["ttl_violation_seconds"])
        self.ttl_indeterminate = float(cache_cfg.get("ttl_indeterminate_seconds", 1.5))

        assoc_cfg = config["association"]
        roi_cfg = config.get("verifier_roi", {})
        self.verifier_roi_conf_floor = float(
            roi_cfg.get("keypoint_conf_floor", config["inference"]["keypoint_conf_floor"])
        )
        self.verifier_roi_min_side = int(roi_cfg.get("min_side_px", 96))
        self.verifier_roi_padding = dict(
            roi_cfg.get(
                "padding_scale",
                {
                    "helmet": 2.0,
                    "goggles": 1.8,
                    "gloves": 1.6,
                    "boots": 1.6,
                    "coverall": 1.25,
                },
            )
        )
        self.item_keypoints: Dict[str, List[str]] = {
            "helmet": list(assoc_cfg["helmet_keypoints"]),
            "goggles": list(assoc_cfg["goggles_keypoints"]),
            "gloves": list(assoc_cfg["gloves_keypoints"]),
            "boots": list(assoc_cfg["boots_keypoints"]),
            "coverall": list(assoc_cfg["coverall_keypoints"]),
        }
        self.alias_to_canonical = build_alias_index(config.get("ppe_label_aliases", {}))

        ens_cfg = config.get("detector_ensemble", {})
        self.ensemble_enabled = bool(ens_cfg.get("enabled", True))
        self.ensemble_yoloe_conf = float(ens_cfg.get("yoloe_conf_threshold", config["inference"]["conf_threshold_verifier"]))
        self.ensemble_iou_nms = float(ens_cfg.get("iou_nms_threshold", 0.5))
        self.ensemble_allow_from_verifier = set(str(x) for x in ens_cfg.get("allowed_items", self.required_ppe))
        self.ppe_fusion_mode = str(ens_cfg.get("fusion_mode", "nms")).lower()
        self.ensemble_nms_backend = str(ens_cfg.get("nms_backend", "auto")).lower()
        self._torchvision_nms_ready = (
            self.ensemble_nms_backend in {"auto", "torchvision"} and torchvision_nms_available()
        )

        verifier_cfg = config.get("verifier", {})
        conflict_cfg = verifier_cfg.get("conflict_resolver", {})
        self.conflict_min_iou = float(conflict_cfg.get("min_iou", 0.05))
        self.conflict_ambiguity_margin = float(conflict_cfg.get("ambiguity_margin", 0.12))
        self.conflict_low_conf = float(conflict_cfg.get("low_conf_threshold", 0.40))
        low_conf_esc_cfg = verifier_cfg.get("low_conf_escalation", {}) or {}
        self.low_conf_escalation_enabled = bool(low_conf_esc_cfg.get("enabled", True))
        self.low_conf_escalation_threshold = float(low_conf_esc_cfg.get("threshold", 0.60))
        self.low_conf_escalation_items = set(str(x) for x in low_conf_esc_cfg.get("items", ["gloves"]))
        self.low_conf_escalation_thresholds = {
            str(k): float(v)
            for k, v in dict(low_conf_esc_cfg.get("per_item_thresholds", {})).items()
        }
        self.strict_spatial_items = set(str(x) for x in verifier_cfg.get("strict_spatial_items", ["helmet"]))
        # Enforce spatial binding for noisy PPE classes to avoid false "compliant" rescues.
        self.strict_spatial_items.update({"gloves", "boots", "coverall"})
        self.strict_spatial_require_bound = bool(
            verifier_cfg.get("strict_spatial_require_bound", True)
        )
        self.periodic_verifier_enabled = bool(verifier_cfg.get("periodic_recheck_enabled", True))
        self.periodic_verifier_seconds = float(verifier_cfg.get("periodic_recheck_seconds", 60.0))
        self.periodic_verifier_items = set(
            str(x) for x in verifier_cfg.get("periodic_items", self.required_ppe)
        )
        self._last_periodic_verifier_ts: Dict[Tuple[int, str], float] = {}
        self.vlm_label_polarity: Dict[str, Dict[str, List[str]]] = {
            "helmet": {"positive": ["helmet"], "negative": []},
            "goggles": {"positive": ["goggles"], "negative": []},
            "gloves": {"positive": ["gloves"], "negative": ["no_gloves"]},
            "boots": {"positive": ["boots"], "negative": ["no_boots"]},
            "coverall": {"positive": ["coverall"], "negative": []},
        }
        config_polarity = verifier_cfg.get("label_polarity", {})
        for item, spec in config_polarity.items():
            if item not in self.vlm_label_polarity:
                continue
            pos = [str(x) for x in spec.get("positive", self.vlm_label_polarity[item]["positive"])]
            neg = [str(x) for x in spec.get("negative", self.vlm_label_polarity[item]["negative"])]
            self.vlm_label_polarity[item] = {"positive": pos, "negative": neg}
        self.per_item_conf_thresholds = {
            str(k): float(v)
            for k, v in dict(config.get("inference", {}).get("per_item_conf_thresholds", {})).items()
        }
        # For visually ambiguous items, low confidence "worn" detections are treated
        # as not-worn candidates to reduce false compliant locks.
        self.low_conf_worn_floor = float(verifier_cfg.get("worn_low_conf_floor", 0.20))
        self.low_conf_worn_items = set(
            str(x) for x in verifier_cfg.get("worn_low_conf_items", ["goggles", "gloves", "boots"])
        )

        dash_cfg = config["dashboard"]
        self.jpeg_quality = int(dash_cfg["jpeg_quality"])
        self.frame_jpeg_backend = str(dash_cfg.get("jpeg_backend", "auto")).lower()
        self.metrics_window_seconds = float(dash_cfg["metrics_window_minutes"]) * 60.0

        self.dropped_frames = 0
        self.verifier_calls: Deque[float] = deque()
        self.pose_calls: Deque[float] = deque()
        self.ppe_calls: Deque[float] = deque()
        self.verifier_aux_calls: Deque[float] = deque()
        self.verifier_ollama_calls: Deque[float] = deque()
        self.classification_history: Deque[Tuple[float, Classification]] = deque()
        self._last_frame_perf: float | None = None
        self._fps = 0.0
        self._stable_states: Dict[Tuple[int, str], StableStateTracker] = {}
        self._frames_processed = 0
        self._pose_infer_calls = 0
        self._ppe_infer_calls = 0
        self._verifier_aux_infer_calls = 0
        self._verifier_crop_infer_calls = 0
        self._verifier_ollama_calls = 0
        self._last_detector_counts = {"ppe_primary_raw": 0, "verifier_aux_raw": 0, "ppe_merged": 0}
        self._last_backend = "python"
        self._ppe_model_path = str(config.get("models", {}).get("ppe", ""))
        self._ppe_task = "detect"
        self._adaptive_detect_frames = 0
        self._adaptive_reuse_frames = 0

        scheduler_cfg = config.get("adaptive_scheduler", {}) or {}
        self.adaptive_scheduler_enabled = bool(scheduler_cfg.get("enabled", False))
        self.ppe_interval_frames = max(1, int(scheduler_cfg.get("ppe_interval_frames", 1)))
        self.ppe_max_staleness_frames = max(
            self.ppe_interval_frames,
            int(scheduler_cfg.get("max_detection_staleness_frames", self.ppe_interval_frames * 2)),
        )
        self.ppe_force_on_new_track = bool(scheduler_cfg.get("force_on_new_track", True))
        self._last_ppe_detections: List[PPEDetection] = []
        self._last_ppe_frame_id = -1_000_000_000
        self._last_track_ids: Set[int] = set()
        tracking_cfg = config.get("tracking", {}) or {}
        default_camera_id = str(
            tracking_cfg.get(
                "camera_id",
                config.get("event_stream", {}).get(
                    "camera_id",
                    config.get("performance_logging", {}).get("camera_id", "cam_01"),
                ),
            )
        )
        self.display_id_camera = self._sanitize_camera_tag(default_camera_id)
        self.display_id_enabled = bool(tracking_cfg.get("display_id_enabled", True))
        self.max_personnel_ids = max(1, int(tracking_cfg.get("max_personnel", 8)))
        self.display_id_timeout_sec = float(
            tracking_cfg.get(
                "display_id_timeout_sec",
                config.get("compliance_memory", {}).get("track_timeout_sec", 5.0),
            )
        )
        self._display_slot_by_track: Dict[int, int] = {}
        self._display_last_seen_ts: Dict[int, float] = {}

        stability_cfg = config.get("status_stability", {})
        self.stable_compliant_enter = int(stability_cfg.get("compliant_enter_frames", 2))
        self.stable_compliant_clear_violation = int(stability_cfg.get("compliant_clear_violation_frames", 3))
        self.stable_violation_enter = int(stability_cfg.get("violation_enter_frames", 4))
        self.stable_indeterminate_enter = int(stability_cfg.get("indeterminate_enter_frames", 8))
        self.stable_per_item = dict(stability_cfg.get("per_item", {}))

        mem_cfg = config.get("compliance_memory", {})
        self.mem_assume_compliant = bool(mem_cfg.get("assume_compliant", True))
        self.mem_prior_score = float(mem_cfg.get("prior_compliant_score", 5.0))
        self.mem_decay = float(mem_cfg.get("decay", 0.96))
        self.mem_positive_gain = float(mem_cfg.get("positive_gain", 1.0))
        self.mem_negative_gain = float(mem_cfg.get("negative_gain", 1.2))
        self.mem_compliant_bonus = float(mem_cfg.get("compliant_bonus", 1.0))
        self.mem_violation_penalty = float(mem_cfg.get("violation_penalty", 1.1))
        self.mem_indeterminate_decay = float(mem_cfg.get("indeterminate_decay", 0.985))
        self.mem_ok_threshold = float(mem_cfg.get("ok_threshold", 4.0))
        self.mem_bad_threshold = float(mem_cfg.get("bad_threshold", -4.0))
        self.mem_dominance_ratio = float(mem_cfg.get("dominance_ratio", 1.15))
        self.mem_dominance_min_negative = float(mem_cfg.get("dominance_min_negative", 2.5))
        self.mem_max_abs_score = float(mem_cfg.get("max_abs_score", 20.0))
        self.mem_strict_items = set(str(x) for x in mem_cfg.get("strict_items", ["coverall", "helmet"]))
        # Sticky compliant prior: for noisy items, hold COMPLIANT unless we have
        # sufficiently strong negative evidence.
        self.sticky_compliant_enabled = bool(mem_cfg.get("sticky_compliant_enabled", True))
        self.sticky_items = set(
            str(x) for x in mem_cfg.get("sticky_items", ["goggles", "gloves", "boots"])
        )
        self.sticky_positive_memory_min = float(mem_cfg.get("sticky_positive_memory_min", 2.5))
        self.sticky_negative_conf_max = float(mem_cfg.get("sticky_negative_conf_max", 0.45))
        self.event_writer = EventStreamWriter(config)
        self.face_gate = SCRFDFaceGate(config)
        helmet_color_cfg = config.get("helmet_color", {}) or {}
        self.helmet_color_enabled = bool(helmet_color_cfg.get("enabled", True))
        self.helmet_color_min_ratio = float(helmet_color_cfg.get("min_ratio", 0.08))
        self.helmet_color_min_valid_pixels = int(helmet_color_cfg.get("min_valid_pixels", 120))
        self.helmet_color_min_value = int(helmet_color_cfg.get("min_value", 40))
        self.helmet_color_min_saturation = int(helmet_color_cfg.get("min_saturation", 60))

        async_cfg = config.get("async_verifier", {}) or {}
        self.async_verifier_enabled = bool(async_cfg.get("enabled", False))
        self.async_verifier_max_queue = max(32, int(async_cfg.get("queue_max", 512)))
        self.async_verifier_drop_if_full = bool(async_cfg.get("drop_if_full", True))
        self._async_in_q: "queue.Queue[Optional[VerifierTask]]" = queue.Queue(maxsize=self.async_verifier_max_queue)
        self._async_out_q: "queue.Queue[VerifierOutcome]" = queue.Queue(maxsize=self.async_verifier_max_queue)
        self._async_pending: Set[Tuple[int, str]] = set()
        self._async_lock = threading.Lock()
        self._async_thread: Optional[threading.Thread] = None
        self._async_stop = threading.Event()
        self._async_enqueued = 0
        self._async_dropped = 0
        self._async_completed = 0
        self._verifier_infer_lock = threading.Lock()
        if self.async_verifier_enabled:
            self._async_thread = threading.Thread(
                target=self._run_async_verifier_worker,
                name="async-verifier-worker",
                daemon=True,
            )
            self._async_thread.start()

        cm_cfg = config.get("compute_monitor", {}) or {}
        self.compute_monitor_enabled = bool(cm_cfg.get("enabled", True))
        self.compute_profile = ComputeProfile(
            pose_gflops_per_infer=float(cm_cfg.get("pose_gflops_per_infer", 8.5)),
            ppe_gflops_per_infer=float(cm_cfg.get("ppe_gflops_per_infer", 20.0)),
            verifier_aux_gflops_per_infer=float(cm_cfg.get("verifier_aux_gflops_per_infer", 20.0)),
            verifier_crop_gflops_per_infer=float(cm_cfg.get("verifier_crop_gflops_per_infer", 20.0)),
            verifier_ollama_gflops_per_infer=float(cm_cfg.get("verifier_ollama_gflops_per_infer", 0.0)),
            device_peak_gflops=float(cm_cfg.get("device_peak_gflops", 0.0)),
        )
        mm_cfg = config.get("memory_monitor", {}) or {}
        self.memory_monitor_enabled = bool(mm_cfg.get("enabled", True))

    def increment_dropped_frames(self, count: int) -> None:
        if count > 0:
            self.dropped_frames += int(count)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        *,
        tracked_people_override: List[TrackedPerson] | None = None,
        ppe_detections_override: List[PPEDetection] | None = None,
        backend: str = "python",
        include_stream_jpeg: bool = True,
    ) -> tuple[FramePayload, bytes]:
        self._drain_async_verifier_results()
        self._frames_processed += 1
        if tracked_people_override is None:
            self._pose_infer_calls += 1
            self.pose_calls.append(time.time())
        frame_jpeg_cache: bytes | None = None

        def _get_frame_jpeg() -> bytes:
            nonlocal frame_jpeg_cache
            if frame_jpeg_cache is None:
                frame_jpeg_cache = self._encode_frame(frame)
            return frame_jpeg_cache

        tracked_people = tracked_people_override if tracked_people_override is not None else self.pose_tracker.track(frame)
        self._prune_stability_state(tracked_people)
        now_ts = time.time()
        display_id_map = self._assign_display_ids(tracked_people, now_ts=now_ts)
        if ppe_detections_override is not None:
            self._ppe_infer_calls += 1
            self.ppe_calls.append(time.time())
            ppe_detections = ppe_detections_override
            self._last_detector_counts = {
                "ppe_primary_raw": len(ppe_detections_override),
                "verifier_aux_raw": 0,
                "ppe_merged": len(ppe_detections_override),
            }
        else:
            ppe_detections = self._select_ppe_detections(
                frame=frame,
                frame_id=frame_id,
                tracked_people=tracked_people,
            )
        self._last_backend = str(backend or "python")

        person_payloads: List[PersonPayload] = []
        person_profile_map: Dict[int, Dict[str, Any]] = {}
        for person in tracked_people:
            display_id = display_id_map.get(
                int(person.person_id),
                f"ID_{int(person.person_id)}-{self.display_id_camera}",
            )
            per_item_state: Dict[str, Classification] = {}
            per_item_state_raw: Dict[str, Classification] = {}
            per_item_reason: Dict[str, str] = {}
            per_item_obs: Dict[str, dict] = {}
            for item in self.required_ppe:
                decision = self._classify_person_item_with_reason(person, ppe_detections, item, frame)
                item_state_raw = decision.classification
                per_item_state_raw[item] = item_state_raw
                item_state = self._stabilize_display_state(
                    person.person_id,
                    item,
                    item_state_raw,
                    positive_conf=decision.positive_conf,
                    negative_conf=decision.negative_conf,
                )
                per_item_state[item] = item_state
                if item_state != item_state_raw:
                    per_item_reason[item] = f"stabilized_hold_{item_state_raw.value.lower()}"
                else:
                    per_item_reason[item] = decision.reason
                stage = self.state_machine.get_violation_stage(person.person_id, item)
                if stage in {"VIOLATION_CANDIDATE", "VIOLATION_CONFIRMED"}:
                    per_item_reason[item] = stage.lower()
                self.last_item_reason[(person.person_id, item)] = decision.reason
                self.last_item_conf[(person.person_id, item)] = (
                    float(decision.positive_conf),
                    float(decision.negative_conf),
                )
                per_item_obs[item] = {
                    "status_raw": item_state_raw.value,
                    "status_stable": item_state.value,
                    "positive_conf": decision.positive_conf,
                    "negative_conf": decision.negative_conf,
                    "reason": decision.reason,
                    "sm_stage": stage,
                    "alert_status": self.state_machine.get_item_state(person.person_id, item).alert_status.value,
                }

                if item_state_raw in (Classification.COMPLIANT, Classification.VIOLATION):
                    self.classification_history.append((time.time(), item_state_raw))

                change = self.state_machine.update(
                    person_id=person.person_id,
                    item=item,
                    classification=item_state_raw,
                    frame_jpeg=None,
                    frame_jpeg_provider=_get_frame_jpeg,
                )
                if change is not None:
                    log_event(
                        change.event_type,
                        frame_id=frame_id,
                        person_id=person.person_id,
                        item=item,
                        alert_status=change.alert_status.value,
                    )

            helmet_is_worn_bound = self._is_item_spatially_bound(
                person=person,
                ppe_detections=ppe_detections,
                item="helmet",
                frame_shape=frame.shape,
            )
            helmet_color, person_status, helmet_color_conf = ("unknown", "Unknown role", 0.0)
            if helmet_is_worn_bound:
                helmet_color, person_status, helmet_color_conf = self._infer_person_helmet_profile(
                    person=person,
                    ppe_detections=ppe_detections,
                    frame=frame,
                )
            person_profile_map[int(person.person_id)] = {
                "display_id": display_id,
                "helmet_color": helmet_color,
                "person_status": person_status,
                "helmet_color_confidence": helmet_color_conf,
            }

            overall = self._overall_status(per_item_state)
            self.event_writer.emit_person_observation(
                frame_id=frame_id,
                track_id=person.person_id,
                bbox=person.bbox,
                per_item=per_item_obs,
                overall_status=overall.value,
                tracking_confidence=getattr(person, "track_conf", None),
            )
            person_payloads.append(
                PersonPayload(
                    person_id=person.person_id,
                    display_id=display_id,
                    bbox=[float(v) for v in person.bbox],
                    keypoints={
                        name: KeypointPayload(
                            name=name,
                            x=float(xy[0]),
                            y=float(xy[1]),
                            conf=float(person.keypoint_confidences.get(name, 0.0)),
                        )
                        for name, xy in person.keypoints.items()
                    },
                    per_item_state=per_item_state,
                    per_item_reason=per_item_reason,
                    overall_status=overall,
                    helmet_color=helmet_color,
                    helmet_color_confidence=helmet_color_conf,
                    person_status=person_status,
                )
            )
        self.event_writer.prune(p.person_id for p in tracked_people)

        active_alerts = self._build_active_alerts(
            frame=frame,
            tracked_people=tracked_people,
            ppe_detections=ppe_detections,
            person_profile_map=person_profile_map,
        )
        metrics = self._build_metrics(len(tracked_people))

        detections_payload = [
            BBoxPayload(
                x1=float(det.bbox[0]),
                y1=float(det.bbox[1]),
                x2=float(det.bbox[2]),
                y2=float(det.bbox[3]),
                conf=float(det.conf),
                label=det.label,
                source=det.source,
            )
            for det in ppe_detections
        ]

        log_event(
            "frame_processed",
            frame_id=frame_id,
            person_id=-1,
            tracked_count=len(tracked_people),
            verifier_calls_last_sec=metrics.verifier_calls_last_sec,
            dropped_frames=metrics.dropped_frames,
        )

        payload = FramePayload(
            frame_id=frame_id,
            timestamp=time.time(),
            persons=person_payloads,
            ppe_detections=detections_payload,
            active_alerts=active_alerts,
            metrics=metrics,
        )
        stream_jpeg = _get_frame_jpeg() if include_stream_jpeg else b""
        return payload, stream_jpeg

    def _stabilize_display_state(
        self,
        person_id: int,
        item: str,
        raw_state: Classification,
        positive_conf: float = 0.0,
        negative_conf: float = 0.0,
    ) -> Classification:
        key = (person_id, item)
        tracker = self._stable_states.get(key)
        if tracker is None:
            if item in self.mem_strict_items:
                initial_stable = raw_state
                initial_score = 0.0
            else:
                initial_stable = Classification.COMPLIANT if self.mem_assume_compliant else raw_state
                initial_score = self.mem_prior_score if self.mem_assume_compliant else 0.0
            tracker = StableStateTracker(stable=initial_stable, score=initial_score)
            self._stable_states[key] = tracker
            # If we received a strong negative first frame, allow immediate evaluation below.

        state_for_tracker = self._apply_sticky_compliant_prior(
            item=item,
            raw_state=raw_state,
            tracker=tracker,
            positive_conf=positive_conf,
            negative_conf=negative_conf,
        )

        if state_for_tracker == Classification.COMPLIANT:
            tracker.compliant_streak += 1
            tracker.violation_streak = 0
            tracker.indeterminate_streak = 0
        elif state_for_tracker == Classification.VIOLATION:
            tracker.violation_streak += 1
            tracker.compliant_streak = 0
            tracker.indeterminate_streak = 0
        else:
            tracker.indeterminate_streak += 1
            tracker.compliant_streak = 0
            tracker.violation_streak = 0

        # Temporal evidence memory with compliant prior bias.
        tracker.positive_memory = tracker.positive_memory * self.mem_decay + max(0.0, positive_conf) * self.mem_positive_gain
        tracker.negative_memory = tracker.negative_memory * self.mem_decay + max(0.0, negative_conf) * self.mem_negative_gain

        if state_for_tracker == Classification.COMPLIANT:
            tracker.score += self.mem_compliant_bonus
        elif state_for_tracker == Classification.VIOLATION:
            tracker.score -= self.mem_violation_penalty
        else:
            tracker.score *= self.mem_indeterminate_decay

        # Confidence evidence nudges score every frame.
        tracker.score += (max(0.0, positive_conf) * self.mem_positive_gain) - (
            max(0.0, negative_conf) * self.mem_negative_gain
        )
        tracker.score = max(-self.mem_max_abs_score, min(self.mem_max_abs_score, tracker.score))

        stable = tracker.stable
        violation_enter = self._stability_threshold(item, "violation_enter_frames", self.stable_violation_enter)
        compliant_enter = self._stability_threshold(item, "compliant_enter_frames", self.stable_compliant_enter)
        compliant_clear = self._stability_threshold(
            item,
            "compliant_clear_violation_frames",
            self.stable_compliant_clear_violation,
        )
        indeterminate_enter = self._stability_threshold(
            item,
            "indeterminate_enter_frames",
            self.stable_indeterminate_enter,
        )

        negative_dominates = (
            tracker.negative_memory >= self.mem_dominance_min_negative
            and tracker.negative_memory > tracker.positive_memory * self.mem_dominance_ratio
        )
        positive_dominates = tracker.positive_memory > tracker.negative_memory * self.mem_dominance_ratio

        # Dominance gates have highest priority: if negatives outrun positives, flip to violation.
        if negative_dominates:
            if tracker.violation_streak >= max(1, violation_enter - 1):
                stable = Classification.VIOLATION
        elif positive_dominates and tracker.compliant_streak >= compliant_enter:
            stable = Classification.COMPLIANT
        elif tracker.score >= self.mem_ok_threshold and tracker.compliant_streak >= compliant_enter:
            stable = Classification.COMPLIANT
        elif tracker.score <= self.mem_bad_threshold and tracker.violation_streak >= violation_enter:
            stable = Classification.VIOLATION

        # Secondary hysteresis fallback.
        if stable == Classification.COMPLIANT:
            if tracker.violation_streak >= violation_enter:
                stable = Classification.VIOLATION
            elif tracker.indeterminate_streak >= indeterminate_enter:
                stable = Classification.INDETERMINATE
        elif stable == Classification.VIOLATION:
            if tracker.compliant_streak >= compliant_clear:
                stable = Classification.COMPLIANT
            elif tracker.indeterminate_streak >= indeterminate_enter:
                stable = Classification.INDETERMINATE
        else:
            if tracker.compliant_streak >= compliant_enter:
                stable = Classification.COMPLIANT
            elif tracker.violation_streak >= violation_enter:
                stable = Classification.VIOLATION

        tracker.stable = stable
        return stable

    def _apply_sticky_compliant_prior(
        self,
        *,
        item: str,
        raw_state: Classification,
        tracker: StableStateTracker,
        positive_conf: float,
        negative_conf: float,
    ) -> Classification:
        """Avoid flipping stable compliant items to violation on weak evidence."""
        if not self.sticky_compliant_enabled:
            return raw_state
        if item not in self.sticky_items:
            return raw_state
        if raw_state != Classification.VIOLATION:
            return raw_state
        if tracker.stable != Classification.COMPLIANT:
            return raw_state
        if tracker.positive_memory < self.sticky_positive_memory_min:
            return raw_state
        if float(negative_conf) > self.sticky_negative_conf_max:
            return raw_state
        if float(negative_conf) > float(positive_conf):
            return raw_state
        return Classification.INDETERMINATE

    def _stability_threshold(self, item: str, key: str, default: int) -> int:
        item_cfg = self.stable_per_item.get(item, {})
        try:
            return int(item_cfg.get(key, default))
        except Exception:
            return default

    def _prune_stability_state(self, tracked_people: List[TrackedPerson]) -> None:
        active_ids = {p.person_id for p in tracked_people}
        stale_keys = [k for k in self._stable_states.keys() if k[0] not in active_ids]
        for key in stale_keys:
            self._stable_states.pop(key, None)
            self.last_item_conf.pop(key, None)
            self.last_item_reason.pop(key, None)
        stale_periodic = [k for k in self._last_periodic_verifier_ts.keys() if k[0] not in active_ids]
        for key in stale_periodic:
            self._last_periodic_verifier_ts.pop(key, None)
        if self.async_verifier_enabled:
            with self._async_lock:
                stale_pending = [k for k in self._async_pending if k[0] not in active_ids]
                for key in stale_pending:
                    self._async_pending.discard(key)

    def _select_ppe_detections(
        self,
        *,
        frame: np.ndarray,
        frame_id: int,
        tracked_people: List[TrackedPerson],
    ) -> List[PPEDetection]:
        if not self.adaptive_scheduler_enabled:
            detections = self._detect_ppe(frame)
            self._adaptive_detect_frames += 1
            self._last_track_ids = {p.person_id for p in tracked_people}
            self._last_ppe_frame_id = int(frame_id)
            self._last_ppe_detections = detections
            return detections

        track_ids = {p.person_id for p in tracked_people}
        since_last = int(frame_id) - int(self._last_ppe_frame_id)
        new_track_seen = self.ppe_force_on_new_track and any(pid not in self._last_track_ids for pid in track_ids)
        no_previous = self._last_ppe_frame_id < 0
        stale = since_last >= self.ppe_max_staleness_frames
        interval_reached = since_last >= self.ppe_interval_frames

        should_detect = no_previous or interval_reached or stale or new_track_seen
        if should_detect:
            detections = self._detect_ppe(frame)
            self._adaptive_detect_frames += 1
            self._last_ppe_frame_id = int(frame_id)
            self._last_ppe_detections = detections
            self._last_track_ids = track_ids
            return detections

        self._adaptive_reuse_frames += 1
        self._last_track_ids = track_ids
        return list(self._last_ppe_detections)

    def _classify_person_item(
        self,
        person: TrackedPerson,
        ppe_detections: List[PPEDetection],
        item: str,
        frame: np.ndarray,
    ) -> Classification:
        return self._classify_person_item_with_reason(person, ppe_detections, item, frame).classification

    def _classify_person_item_with_reason(
        self,
        person: TrackedPerson,
        ppe_detections: List[PPEDetection],
        item: str,
        frame: np.ndarray,
    ) -> ItemDecision:
        detection_dicts = [
            {"label": det.label, "bbox": det.bbox, "conf": det.conf}
            for det in ppe_detections
        ]

        base_classification, bind = self.association.classify_item(
            item=item,
            keypoints=person.keypoints,
            keypoint_confidences=person.keypoint_confidences,
            ppe_detections=detection_dicts,
            frame_shape=frame.shape,
        )
        base_reason = None
        if item == self.face_gate.item and self.face_gate.enabled:
            face_obs = self.face_gate.observe(frame, person.person_id, person.bbox)
            if self.face_gate.should_log(frame_id=self._frames_processed):
                log_event(
                    "face_gate_observation",
                    frame_id=self._frames_processed,
                    person_id=person.person_id,
                    item=item,
                    mode="shadow" if self.face_gate.shadow_mode else "enforce",
                    enforce_enabled=self.face_gate.enforce,
                    available=face_obs.available,
                    face_visible=face_obs.face_visible,
                    face_conf=round(face_obs.best_confidence, 4),
                    face_count=face_obs.face_count,
                    reason=face_obs.reason,
                    bbox=face_obs.best_bbox,
                )

            if self.face_gate.enforce and face_obs.available and not face_obs.face_visible:
                base_classification = Classification.INDETERMINATE
                bind = None
                base_reason = "goggles_face_gate_indeterminate"
            elif self.face_gate.shadow_mode and face_obs.available and not face_obs.face_visible:
                base_reason = "goggles_face_gate_shadow_would_indeterminate"

        person_crop = self._crop_to_bbox(frame, person.bbox)
        item_crop = self._crop_for_item(frame, person, item)
        vctx, ambiguous = self._build_verifier_context(person, item, ppe_detections, person_crop, item_crop)
        if (
            item in self.low_conf_worn_items
            and base_classification == Classification.COMPLIANT
            and float(vctx.positive_conf) < float(self.low_conf_worn_floor)
        ):
            base_classification = Classification.VIOLATION_TENTATIVE
            base_reason = "worn_detection_below_conf_floor"
        low_conf_escalated = self._should_force_verifier_on_low_conf(
            item=item,
            base_classification=base_classification,
            bind=bind,
            positive_conf=vctx.positive_conf,
        )
        # Strict spatial items (e.g. helmet) should not be rescued by verifier
        # when they are not keypoint-bound.
        if (
            self.strict_spatial_require_bound
            and item in self.strict_spatial_items
            and base_classification == Classification.VIOLATION_TENTATIVE
        ):
            return ItemDecision(
                Classification.VIOLATION,
                "strict_spatial_not_bound",
                positive_conf=vctx.positive_conf,
                negative_conf=vctx.negative_conf,
            )
        periodic_due = self._is_periodic_verifier_due(
            person_id=person.person_id,
            item=item,
            now_ts=time.time(),
        )
        needs_verifier = (
            base_classification == Classification.VIOLATION_TENTATIVE
            or ambiguous
            or low_conf_escalated
            or periodic_due
        )

        if not needs_verifier:
            if base_classification == Classification.COMPLIANT:
                return ItemDecision(
                    base_classification,
                    "detected_and_spatially_bound",
                    positive_conf=vctx.positive_conf,
                    negative_conf=vctx.negative_conf,
                )
            if base_classification == Classification.INDETERMINATE:
                return ItemDecision(
                    base_classification,
                    base_reason or "keypoint_not_visible_or_out_of_frame",
                    positive_conf=vctx.positive_conf,
                    negative_conf=vctx.negative_conf,
                )
            if bind is not None and bind.held:
                return ItemDecision(
                    base_classification,
                    "detected_but_held_not_worn",
                    positive_conf=vctx.positive_conf,
                    negative_conf=vctx.negative_conf,
                )
            return ItemDecision(
                base_classification,
                "direct_violation",
                positive_conf=vctx.positive_conf,
                negative_conf=vctx.negative_conf,
            )

        cached = self.cache.get(person.person_id, item)
        if cached is not None and not periodic_due:
            final_cls, reason = self._classification_from_verdict(cached.verdict, source="cache")
            return ItemDecision(
                final_cls,
                reason,
                positive_conf=vctx.positive_conf,
                negative_conf=vctx.negative_conf,
            )

        if self.async_verifier_enabled:
            pending_reason = self._queue_async_verifier_task(
                person_id=person.person_id,
                item=item,
                item_crop=item_crop,
                context=vctx,
            )
            return ItemDecision(
                Classification.INDETERMINATE,
                pending_reason,
                positive_conf=vctx.positive_conf,
                negative_conf=vctx.negative_conf,
            )

        verify_result = self._run_sync_verifier(
            person_id=person.person_id,
            item_crop=item_crop,
            item=item,
            context=vctx,
        )
        final_cls, reason = self._classification_from_verdict(
            verify_result.verdict,
            source=verify_result.source,
        )
        return ItemDecision(
            final_cls,
            reason,
            positive_conf=vctx.positive_conf,
            negative_conf=vctx.negative_conf,
        )

    def _should_force_verifier_on_low_conf(
        self,
        *,
        item: str,
        base_classification: Classification,
        bind,
        positive_conf: float,
    ) -> bool:
        """Escalate low-confidence compliant detections to verifier for selected items."""
        if not self.low_conf_escalation_enabled:
            return False
        if item not in self.low_conf_escalation_items:
            return False
        if base_classification != Classification.COMPLIANT:
            return False
        if bind is None or not getattr(bind, "bound", False):
            return False
        threshold = self.low_conf_escalation_thresholds.get(item, self.low_conf_escalation_threshold)
        return float(positive_conf) < float(threshold)

    def _run_sync_verifier(
        self,
        *,
        person_id: int,
        item_crop: np.ndarray,
        item: str,
        context: VerifierContext,
    ) -> VerifierResult:
        with self._verifier_infer_lock:
            verify_result = self.verifier.verify(item_crop, item, context=context)
        self.verifier_calls.append(time.time())
        self._verifier_crop_infer_calls += 1
        if str(getattr(verify_result, "source", "")).lower() == "ollama":
            self.verifier_ollama_calls.append(time.time())
            self._verifier_ollama_calls += 1
        if verify_result.verdict == VerifierVerdict.COMPLIANT:
            ttl = self.ttl_compliant
        elif verify_result.verdict == VerifierVerdict.VIOLATION:
            ttl = self.ttl_violation
        else:
            ttl = self.ttl_indeterminate
        self.cache.put(person_id, item, verify_result, ttl)
        self._last_periodic_verifier_ts[(int(person_id), str(item))] = time.time()
        return verify_result

    def _queue_async_verifier_task(
        self,
        *,
        person_id: int,
        item: str,
        item_crop: np.ndarray,
        context: VerifierContext,
    ) -> str:
        key = (int(person_id), str(item))
        with self._async_lock:
            if key in self._async_pending:
                return "async_verifier_pending"
            self._async_pending.add(key)

        task = VerifierTask(
            person_id=int(person_id),
            item=str(item),
            item_crop=item_crop.copy(),
            context=VerifierContext(
                person_crop=context.person_crop.copy() if context.person_crop is not None else None,
                item_crop=context.item_crop.copy() if context.item_crop is not None else None,
                positive_conf=float(context.positive_conf),
                negative_conf=float(context.negative_conf),
                expected_item=str(context.expected_item),
            ),
        )
        try:
            self._async_in_q.put_nowait(task)
            self._async_enqueued += 1
            return "async_verifier_queued"
        except queue.Full:
            with self._async_lock:
                self._async_pending.discard(key)
            self._async_dropped += 1
            if not self.async_verifier_drop_if_full:
                return "async_verifier_queue_full_nonblocking"
            return "async_verifier_queue_full"

    def _run_async_verifier_worker(self) -> None:
        while not self._async_stop.is_set():
            try:
                task = self._async_in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                return
            try:
                with self._verifier_infer_lock:
                    result = self.verifier.verify(task.item_crop, task.item, context=task.context)
            except Exception:
                result = VerifierResult(
                    verdict=VerifierVerdict.INDETERMINATE,
                    score=0.0,
                    source="async_error",
                )
            try:
                self._async_out_q.put_nowait(
                    VerifierOutcome(
                        person_id=task.person_id,
                        item=task.item,
                        result=result,
                    )
                )
                self._async_completed += 1
            except queue.Full:
                self._async_dropped += 1
                with self._async_lock:
                    self._async_pending.discard((task.person_id, task.item))

    def _drain_async_verifier_results(self) -> None:
        if not self.async_verifier_enabled:
            return
        while True:
            try:
                outcome = self._async_out_q.get_nowait()
            except queue.Empty:
                break

            verify_result = outcome.result
            self.verifier_calls.append(time.time())
            self._verifier_crop_infer_calls += 1
            if str(getattr(verify_result, "source", "")).lower() == "ollama":
                self.verifier_ollama_calls.append(time.time())
                self._verifier_ollama_calls += 1
            if verify_result.verdict == VerifierVerdict.COMPLIANT:
                ttl = self.ttl_compliant
            elif verify_result.verdict == VerifierVerdict.VIOLATION:
                ttl = self.ttl_violation
            else:
                ttl = self.ttl_indeterminate
            self.cache.put(outcome.person_id, outcome.item, verify_result, ttl)
            self._last_periodic_verifier_ts[(int(outcome.person_id), str(outcome.item))] = time.time()
            with self._async_lock:
                self._async_pending.discard((outcome.person_id, outcome.item))

    def _is_periodic_verifier_due(
        self,
        *,
        person_id: int,
        item: str,
        now_ts: float,
    ) -> bool:
        """Return True when verifier should refresh this person/item decision."""
        if not self.periodic_verifier_enabled:
            return False
        if item not in self.periodic_verifier_items:
            return False
        interval = max(1.0, float(self.periodic_verifier_seconds))
        last_ts = self._last_periodic_verifier_ts.get((int(person_id), str(item)))
        if last_ts is None:
            return True
        return (float(now_ts) - float(last_ts)) >= interval

    def close(self) -> None:
        self._async_stop.set()
        if self.async_verifier_enabled:
            try:
                self._async_in_q.put_nowait(None)
            except queue.Full:
                pass
        if self._async_thread is not None:
            self._async_thread.join(timeout=1.5)

    def _build_active_alerts(
        self,
        *,
        frame: np.ndarray,
        tracked_people: List[TrackedPerson],
        ppe_detections: List[PPEDetection],
        person_profile_map: Dict[int, Dict[str, Any]],
    ) -> List[AlertPayload]:
        alerts: List[AlertPayload] = []
        person_by_id = {int(person.person_id): person for person in tracked_people}
        for person_id, item, item_state in self.state_machine.iter_active_alerts():
            encoded_evidence = (
                base64.b64encode(item_state.evidence_jpeg).decode("ascii")
                if item_state.evidence_jpeg is not None
                else None
            )
            person = person_by_id.get(int(person_id))
            profile = person_profile_map.get(int(person_id), {})
            display_id = str(
                profile.get(
                    "display_id",
                    f"ID_{int(person_id)}-{self.display_id_camera}",
                )
            )
            helmet_color = str(profile.get("helmet_color", "unknown"))
            helmet_color_confidence = float(profile.get("helmet_color_confidence", 0.0))
            person_status = str(profile.get("person_status", self._status_for_helmet_color(helmet_color)))
            positive_conf, negative_conf = self.last_item_conf.get((person_id, item), (0.0, 0.0))
            person_crop_b64: Optional[str] = None
            item_crop_b64: Optional[str] = None
            if person is not None:
                person_crop_b64 = self._encode_crop_base64(self._crop_to_bbox(frame, person.bbox))
                item_crop_b64 = self._encode_crop_base64(self._crop_for_item(frame, person, item))
                if (
                    item == "helmet"
                    and helmet_color == "unknown"
                    and self._is_item_spatially_bound(
                        person=person,
                        ppe_detections=ppe_detections,
                        item="helmet",
                        frame_shape=frame.shape,
                    )
                ):
                    det_color, det_status, det_conf = self._infer_person_helmet_profile(
                        person=person,
                        ppe_detections=ppe_detections,
                        frame=frame,
                    )
                    helmet_color = det_color
                    helmet_color_confidence = det_conf
                    person_status = det_status
            alert_id = f"{person_id}:{item}:{int(item_state.last_transition_ts * 1000)}"
            reason = self.last_item_reason.get((person_id, item), f"missing_or_incorrect_{item}")
            alerts.append(
                AlertPayload(
                    alert_id=alert_id,
                    person_id=person_id,
                    display_id=display_id,
                    item=item,
                    status=item_state.alert_status,
                    reason=reason,
                    timestamp=item_state.last_transition_ts,
                    evidence_available=(
                        item_state.evidence_jpeg is not None
                        or person_crop_b64 is not None
                        or item_crop_b64 is not None
                    ),
                    evidence_jpeg_base64=encoded_evidence or item_crop_b64,
                    person_status=person_status,
                    helmet_color=helmet_color,
                    helmet_color_confidence=helmet_color_confidence,
                    person_crop_jpeg_base64=person_crop_b64,
                    item_crop_jpeg_base64=item_crop_b64,
                    positive_conf=float(positive_conf),
                    negative_conf=float(negative_conf),
                    acknowledged=False,
                )
            )
        return alerts

    def _build_metrics(self, tracked_count: int) -> MetricsPayload:
        now = time.time()
        while self.verifier_calls and self.verifier_calls[0] < now - 1.0:
            self.verifier_calls.popleft()
        while self.pose_calls and self.pose_calls[0] < now - 1.0:
            self.pose_calls.popleft()
        while self.ppe_calls and self.ppe_calls[0] < now - 1.0:
            self.ppe_calls.popleft()
        while self.verifier_aux_calls and self.verifier_aux_calls[0] < now - 1.0:
            self.verifier_aux_calls.popleft()
        while self.verifier_ollama_calls and self.verifier_ollama_calls[0] < now - 1.0:
            self.verifier_ollama_calls.popleft()

        while self.classification_history and self.classification_history[0][0] < now - self.metrics_window_seconds:
            self.classification_history.popleft()

        compliant = sum(1 for _, cls in self.classification_history if cls == Classification.COMPLIANT)
        total = sum(
            1
            for _, cls in self.classification_history
            if cls in (Classification.COMPLIANT, Classification.VIOLATION)
        )
        compliance_rate = 100.0 if total == 0 else (compliant * 100.0 / total)

        perf_now = time.perf_counter()
        if self._last_frame_perf is not None:
            instant_fps = 1.0 / max(1e-6, perf_now - self._last_frame_perf)
            self._fps = instant_fps if self._fps == 0.0 else (0.8 * self._fps + 0.2 * instant_fps)
        self._last_frame_perf = perf_now

        pose_rate = float(len(self.pose_calls))
        ppe_rate = float(len(self.ppe_calls))
        verifier_aux_rate = float(len(self.verifier_aux_calls))
        verifier_crop_rate = float(len(self.verifier_calls))
        verifier_ollama_rate = float(len(self.verifier_ollama_calls))
        compute = estimate_compute_usage(
            enabled=self.compute_monitor_enabled,
            profile=self.compute_profile,
            pose_infer_per_sec=pose_rate,
            ppe_infer_per_sec=ppe_rate,
            verifier_aux_infer_per_sec=verifier_aux_rate,
            verifier_crop_infer_per_sec=verifier_crop_rate,
            verifier_ollama_calls_per_sec=verifier_ollama_rate,
        )
        memory = read_memory_snapshot(enabled=self.memory_monitor_enabled)

        return MetricsPayload(
            fps=round(self._fps, 2),
            verifier_calls_last_sec=len(self.verifier_calls),
            tracked_count=tracked_count,
            dropped_frames=self.dropped_frames,
            active_violations=self.state_machine.active_alerts_count(),
            compliance_rate=round(compliance_rate, 2),
            ppe_primary_raw=int(self._last_detector_counts.get("ppe_primary_raw", 0)),
            verifier_aux_raw=int(self._last_detector_counts.get("verifier_aux_raw", 0)),
            ppe_merged=int(self._last_detector_counts.get("ppe_merged", 0)),
            ppe_infer_calls=self._ppe_infer_calls,
            verifier_aux_infer_calls=self._verifier_aux_infer_calls,
            pose_infer_calls=self._pose_infer_calls,
            verifier_crop_infer_calls=self._verifier_crop_infer_calls,
            verifier_ollama_calls=self._verifier_ollama_calls,
            ppe_model=self._ppe_model_path,
            ppe_task=self._ppe_task,
            ppe_fusion_mode=self.ppe_fusion_mode,
            adaptive_scheduler_enabled=self.adaptive_scheduler_enabled,
            adaptive_detect_frames=self._adaptive_detect_frames,
            adaptive_reuse_frames=self._adaptive_reuse_frames,
            async_verifier_enabled=self.async_verifier_enabled,
            async_verifier_enqueued=self._async_enqueued,
            async_verifier_completed=self._async_completed,
            async_verifier_dropped=self._async_dropped,
            compute_monitor_enabled=compute.enabled,
            pose_infer_per_sec=compute.pose_infer_per_sec,
            ppe_infer_per_sec=compute.ppe_infer_per_sec,
            verifier_aux_infer_per_sec=compute.verifier_aux_infer_per_sec,
            verifier_crop_infer_per_sec=compute.verifier_crop_infer_per_sec,
            verifier_ollama_calls_per_sec=compute.verifier_ollama_calls_per_sec,
            pose_estimated_gflops_per_sec=compute.pose_estimated_gflops_per_sec,
            ppe_estimated_gflops_per_sec=compute.ppe_estimated_gflops_per_sec,
            verifier_aux_estimated_gflops_per_sec=compute.verifier_aux_estimated_gflops_per_sec,
            verifier_crop_estimated_gflops_per_sec=compute.verifier_crop_estimated_gflops_per_sec,
            verifier_ollama_estimated_gflops_per_sec=compute.verifier_ollama_estimated_gflops_per_sec,
            estimated_gflops_per_sec=compute.estimated_gflops_per_sec,
            estimated_tflops_per_sec=compute.estimated_tflops_per_sec,
            estimated_tops_per_sec=compute.estimated_tops_per_sec,
            estimated_flops_per_sec=compute.estimated_flops_per_sec,
            estimated_compute_utilization_pct=compute.estimated_compute_utilization_pct,
            memory_monitor_enabled=memory.enabled,
            process_rss_mb=memory.process_rss_mb,
            process_vms_mb=memory.process_vms_mb,
            system_memory_used_mb=memory.system_memory_used_mb,
            system_memory_total_mb=memory.system_memory_total_mb,
            system_memory_utilization_pct=memory.system_memory_utilization_pct,
            backend=str(getattr(self, "_last_backend", "python")),
        )

    def _overall_status(self, per_item_state: Dict[str, Classification]) -> OverallStatus:
        if any(v == Classification.VIOLATION for v in per_item_state.values()):
            return OverallStatus.VIOLATION
        if any(v == Classification.INDETERMINATE for v in per_item_state.values()):
            return OverallStatus.INDETERMINATE
        return OverallStatus.COMPLIANT

    def _status_for_helmet_color(self, helmet_color: str) -> str:
        return HELMET_COLOR_TO_STATUS.get(str(helmet_color).lower(), "Unknown role")

    def _sanitize_camera_tag(self, value: str) -> str:
        raw = str(value or "cam_01").strip() or "cam_01"
        cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in raw)
        return cleaned or "cam_01"

    def _assign_display_ids(self, tracked_people: List[TrackedPerson], now_ts: float) -> Dict[int, str]:
        if not self.display_id_enabled:
            return {
                int(person.person_id): f"ID_{int(person.person_id)}-{self.display_id_camera}"
                for person in tracked_people
            }
        active_ids = {int(person.person_id) for person in tracked_people}
        stale = [
            track_id
            for track_id, seen_ts in self._display_last_seen_ts.items()
            if track_id not in active_ids and (now_ts - float(seen_ts)) > self.display_id_timeout_sec
        ]
        for track_id in stale:
            self._display_last_seen_ts.pop(track_id, None)
            self._display_slot_by_track.pop(track_id, None)

        used_slots = {int(self._display_slot_by_track[tid]) for tid in active_ids if tid in self._display_slot_by_track}
        slot_pool = [idx for idx in range(1, self.max_personnel_ids + 1) if idx not in used_slots]
        out: Dict[int, str] = {}
        for person in tracked_people:
            track_id = int(person.person_id)
            slot = self._display_slot_by_track.get(track_id)
            if slot is None:
                if slot_pool:
                    slot = int(slot_pool.pop(0))
                else:
                    slot = int((abs(track_id) % self.max_personnel_ids) + 1)
                self._display_slot_by_track[track_id] = slot
            self._display_last_seen_ts[track_id] = now_ts
            out[track_id] = f"ID_{int(slot)}-{self.display_id_camera}"
        return out

    def _is_item_spatially_bound(
        self,
        *,
        person: TrackedPerson,
        ppe_detections: List[PPEDetection],
        item: str,
        frame_shape: tuple[int, int, int],
    ) -> bool:
        detection_dicts = [
            {"label": det.label, "bbox": det.bbox, "conf": det.conf}
            for det in ppe_detections
        ]
        classification, bind = self.association.classify_item(
            item=item,
            keypoints=person.keypoints,
            keypoint_confidences=person.keypoint_confidences,
            ppe_detections=detection_dicts,
            frame_shape=frame_shape,
        )
        return bool(
            classification == Classification.COMPLIANT
            and bind is not None
            and getattr(bind, "bound", False)
            and not getattr(bind, "held", False)
        )

    def _infer_person_helmet_profile(
        self,
        *,
        person: TrackedPerson,
        ppe_detections: List[PPEDetection],
        frame: np.ndarray,
    ) -> tuple[str, str, float]:
        if not self.helmet_color_enabled or cv2 is None:
            return "unknown", "Unknown role", 0.0

        helmet_roi = self._extract_helmet_roi(
            person=person,
            ppe_detections=ppe_detections,
            frame=frame,
        )
        if helmet_roi is None or helmet_roi.size == 0:
            return "unknown", "Unknown role", 0.0

        color, confidence = self._detect_helmet_color_from_roi(helmet_roi)
        return color, self._status_for_helmet_color(color), confidence

    def _extract_helmet_roi(
        self,
        *,
        person: TrackedPerson,
        ppe_detections: List[PPEDetection],
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        best_bbox: Optional[tuple[float, float, float, float]] = None
        best_conf = -1.0
        for det in ppe_detections:
            if det.label != "helmet":
                continue
            if not self._detection_matches_item_region(person, "helmet", det.bbox):
                continue
            if float(det.conf) > best_conf:
                best_conf = float(det.conf)
                best_bbox = det.bbox

        if best_bbox is not None:
            crop = self._crop_to_bbox(frame, best_bbox)
            if crop is not None and crop.size > 0:
                return crop

        point_names = self.item_keypoints.get("helmet", ["nose", "left_eye", "right_eye"])
        points: List[tuple[float, float]] = []
        for name in point_names:
            conf = float(person.keypoint_confidences.get(name, 0.0))
            pt = person.keypoints.get(name)
            if pt is None or conf < self.verifier_roi_conf_floor:
                continue
            points.append((float(pt[0]), float(pt[1])))
        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        roi_w = max(1.0, max_x - min_x)
        roi_h = max(1.0, max_y - min_y)
        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5
        pad_scale = float(self.verifier_roi_padding.get("helmet", 2.0))
        target_w = max(float(self.verifier_roi_min_side), roi_w * pad_scale)
        target_h = max(float(self.verifier_roi_min_side), roi_h * pad_scale)

        candidate = (
            cx - target_w * 0.5,
            cy - target_h * 0.6,
            cx + target_w * 0.5,
            cy + target_h * 0.4,
        )
        crop = self._crop_to_bbox(frame, candidate)
        if crop is None or crop.size == 0:
            return None
        return crop

    def _detect_helmet_color_from_roi(self, helmet_roi: np.ndarray) -> tuple[str, float]:
        if cv2 is None or helmet_roi is None or helmet_roi.size == 0:
            return "unknown", 0.0
        if helmet_roi.ndim != 3 or helmet_roi.shape[2] < 3:
            return "unknown", 0.0

        hsv = cv2.cvtColor(helmet_roi, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        valid = v >= self.helmet_color_min_value
        valid_count = int(np.count_nonzero(valid))
        if valid_count < self.helmet_color_min_valid_pixels:
            return "unknown", 0.0

        min_sat = max(0, int(self.helmet_color_min_saturation))
        color_masks: Dict[str, np.ndarray] = {
            "white": (s <= 38) & (v >= 135),
            "yellow": (h >= 18) & (h <= 38) & (s >= min_sat) & (v >= 65),
            "blue": (h >= 90) & (h <= 130) & (s >= min_sat) & (v >= 55),
            "green": (h >= 40) & (h <= 90) & (s >= min_sat) & (v >= 55),
            "red": (((h <= 10) | (h >= 170)) & (s >= min_sat) & (v >= 65)),
        }

        best_color = "unknown"
        best_ratio = 0.0
        for color, mask in color_masks.items():
            ratio = float(np.count_nonzero(mask & valid)) / float(valid_count)
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color

        if best_ratio < self.helmet_color_min_ratio:
            return "unknown", best_ratio
        return best_color, best_ratio

    def _encode_crop_base64(self, crop: Optional[np.ndarray]) -> Optional[str]:
        if crop is None or crop.size == 0:
            return None
        encoded = encode_jpeg_bytes(
            crop,
            quality=self.jpeg_quality,
            backend=self.frame_jpeg_backend,
        )
        if encoded is None:
            return None
        return base64.b64encode(encoded).decode("ascii")

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        encoded = encode_jpeg_bytes(
            frame,
            quality=self.jpeg_quality,
            backend=self.frame_jpeg_backend,
        )
        if encoded is None:
            return b""
        return encoded

    def _crop_to_bbox(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(bbox[0])))
        y1 = max(0, min(h - 1, int(bbox[1])))
        x2 = max(0, min(w, int(bbox[2])))
        y2 = max(0, min(h, int(bbox[3])))
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]

    def _crop_for_item(self, frame: np.ndarray, person: TrackedPerson, item: str) -> np.ndarray:
        point_names = self.item_keypoints.get(item, [])
        points: List[tuple[float, float]] = []
        for name in point_names:
            conf = float(person.keypoint_confidences.get(name, 0.0))
            point = person.keypoints.get(name)
            if point is None or conf < self.verifier_roi_conf_floor:
                continue
            points.append((float(point[0]), float(point[1])))

        if not points:
            return self._crop_to_bbox(frame, person.bbox)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        roi_w = max(1.0, max_x - min_x)
        roi_h = max(1.0, max_y - min_y)
        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5

        pad_scale = float(self.verifier_roi_padding.get(item, 1.6))
        target_w = max(float(self.verifier_roi_min_side), roi_w * pad_scale)
        target_h = max(float(self.verifier_roi_min_side), roi_h * pad_scale)

        candidate = (
            cx - target_w * 0.5,
            cy - target_h * 0.5,
            cx + target_w * 0.5,
            cy + target_h * 0.5,
        )

        # Keep ROI inside person bounds for context while suppressing clutter.
        person_box = person.bbox
        constrained = (
            max(candidate[0], person_box[0]),
            max(candidate[1], person_box[1]),
            min(candidate[2], person_box[2]),
            min(candidate[3], person_box[3]),
        )

        if constrained[2] <= constrained[0] or constrained[3] <= constrained[1]:
            return self._crop_to_bbox(frame, person.bbox)
        return self._crop_to_bbox(frame, constrained)

    def _classification_from_verdict(self, verdict: VerifierVerdict, source: str) -> tuple[Classification, str]:
        if verdict == VerifierVerdict.COMPLIANT:
            return Classification.COMPLIANT, f"verifier_{source}_compliant"
        if verdict == VerifierVerdict.VIOLATION:
            return Classification.VIOLATION, f"verifier_{source}_violation"
        return Classification.INDETERMINATE, f"verifier_{source}_indeterminate"

    def _build_verifier_context(
        self,
        person: TrackedPerson,
        item: str,
        detections: List[PPEDetection],
        person_crop: np.ndarray,
        item_crop: np.ndarray,
    ) -> tuple[VerifierContext, bool]:
        polarity = self.vlm_label_polarity.get(item, {"positive": [item], "negative": []})
        positive_labels = set(polarity.get("positive", [item]))
        negative_labels = set(polarity.get("negative", []))

        person_box = person.bbox
        positive_conf = 0.0
        negative_conf = 0.0
        any_overlap = False

        for det in detections:
            if box_iou(det.bbox, person_box) < self.conflict_min_iou:
                continue
            if not self._detection_matches_item_region(person, item, det.bbox):
                continue
            if det.label in positive_labels:
                positive_conf = max(positive_conf, float(det.conf))
                any_overlap = True
            elif det.label in negative_labels:
                negative_conf = max(negative_conf, float(det.conf))
                any_overlap = True

        ambiguous = False
        if any_overlap and negative_labels:
            if negative_conf > 0.0:
                if abs(positive_conf - negative_conf) <= self.conflict_ambiguity_margin:
                    ambiguous = True
                if positive_conf < self.conflict_low_conf:
                    ambiguous = True
                if negative_conf > positive_conf:
                    ambiguous = True

        return (
            VerifierContext(
                person_crop=person_crop,
                item_crop=item_crop,
                positive_conf=positive_conf,
                negative_conf=negative_conf,
                expected_item=item,
            ),
            ambiguous,
        )

    def _detection_matches_item_region(
        self,
        person: TrackedPerson,
        item: str,
        det_bbox: tuple[float, float, float, float],
    ) -> bool:
        rule = self.association.rules.get(item)
        if rule is None:
            return False

        if item == "coverall":
            torso = torso_bbox(
                person.keypoints,
                person.keypoint_confidences,
                self.association.keypoint_conf_floor,
            )
            if torso is None:
                return False
            return box_iou(det_bbox, torso) >= max(0.02, rule.iou_threshold * 0.5)

        center = bbox_center(det_bbox)
        points: List[tuple[float, float]] = []
        for name in rule.expected_keypoints:
            conf = float(person.keypoint_confidences.get(name, 0.0))
            point = person.keypoints.get(name)
            if point is None or conf < rule.keypoint_conf_floor:
                continue
            points.append((float(point[0]), float(point[1])))
        if not points:
            return False

        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return False
        dx = pts[:, 0] - float(center[0])
        dy = pts[:, 1] - float(center[1])
        min_dist = float(np.min(np.hypot(dx, dy)))
        limit = max(24.0, float(rule.distance_threshold_px) * 1.35)
        return min_dist <= limit

    def _detect_ppe(self, frame: np.ndarray) -> List[PPEDetection]:
        """Run primary detector and optional YOLOE auxiliary model."""
        self._ppe_infer_calls += 1
        self.ppe_calls.append(time.time())
        primary = self.ppe_detector.detect(frame)
        if not self.ensemble_enabled:
            self._last_detector_counts = {
                "ppe_primary_raw": len(primary),
                "verifier_aux_raw": 0,
                "ppe_merged": len(primary),
            }
            return primary

        yoloe = self._detect_with_verifier_model(frame)
        combined = primary + yoloe
        if not combined:
            self._last_detector_counts = {
                "ppe_primary_raw": len(primary),
                "verifier_aux_raw": len(yoloe),
                "ppe_merged": 0,
            }
            return combined
        if self.ppe_fusion_mode == "parallel":
            merged = combined
        else:
            merged = self._nms_merge_detections(combined, iou_threshold=self.ensemble_iou_nms)
        self._last_detector_counts = {
            "ppe_primary_raw": len(primary),
            "verifier_aux_raw": len(yoloe),
            "ppe_merged": len(merged),
        }
        return merged

    def _detect_with_verifier_model(self, frame: np.ndarray) -> List[PPEDetection]:
        """Use verifier model as an auxiliary full-frame detector for recall boost."""
        if not hasattr(self.verifier, "model"):
            return []
        self._verifier_aux_infer_calls += 1
        self.verifier_aux_calls.append(time.time())
        model = getattr(self.verifier, "model")
        imgsz = getattr(self.verifier, "imgsz", self.config["inference"]["imgsz"])

        with self._verifier_infer_lock:
            results = model.predict(
                source=frame,
                conf=self.ensemble_yoloe_conf,
                imgsz=imgsz,
                verbose=False,
            )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return []

        names = result.names if isinstance(result.names, dict) else {}
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
        cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None

        out: List[PPEDetection] = []
        for idx, box in enumerate(xyxy):
            class_id = int(cls[idx]) if cls is not None else -1
            raw_label = str(names.get(class_id, class_id))
            label = canonicalize_label(raw_label, self.alias_to_canonical)
            if label not in self.ensemble_allow_from_verifier:
                continue
            score = float(conf[idx]) if conf is not None else 0.0
            item_threshold = float(self.per_item_conf_thresholds.get(label, self.ensemble_yoloe_conf))
            if score < item_threshold:
                continue
            out.append(
                PPEDetection(
                    label=label,
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    conf=score,
                    source="yoloe_aux",
                )
            )
        return out

    def _nms_merge_detections(self, detections: List[PPEDetection], iou_threshold: float) -> List[PPEDetection]:
        """Class-aware NMS merge using torchvision when available."""
        by_label: Dict[str, List[PPEDetection]] = {}
        for det in detections:
            by_label.setdefault(det.label, []).append(det)

        kept: List[PPEDetection] = []
        for label, group in by_label.items():
            if len(group) <= 1:
                kept.extend(group)
                continue

            boxes = np.asarray([det.bbox for det in group], dtype=np.float32)
            scores = np.asarray([float(det.conf) for det in group], dtype=np.float32)
            keep_idx = self._nms_keep_indices(
                boxes=boxes,
                scores=scores,
                iou_threshold=float(iou_threshold),
            )
            for idx in keep_idx:
                kept.append(group[int(idx)])
        return kept

    def _nms_keep_indices(
        self,
        *,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int64)
        if self._torchvision_nms_ready:
            try:
                import torch
                from torchvision.ops import nms as tv_nms

                keep = tv_nms(
                    torch.as_tensor(boxes, dtype=torch.float32),
                    torch.as_tensor(scores, dtype=torch.float32),
                    float(iou_threshold),
                )
                return keep.cpu().numpy().astype(np.int64, copy=False)
            except Exception:
                if self.ensemble_nms_backend == "torchvision":
                    return np.empty((0,), dtype=np.int64)
        return _nms_numpy_indices(boxes=boxes, scores=scores, iou_threshold=float(iou_threshold))


def _nms_numpy_indices(*, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Vectorized NMS index selection on CPU (fallback path)."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        denom = areas[i] + areas[rest] - inter
        iou = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0.0)
        order = rest[iou < iou_threshold]

    return np.asarray(keep, dtype=np.int64)
