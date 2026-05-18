"""Per-frame orchestration for tracking, association, verification, and alerts."""

import base64
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

from .association import (
    AssociationEngine,
    bbox_center,
    distance as point_distance,
    iou as box_iou,
    torso_bbox,
)
from .cache import VerifierCache
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
    VerifierVerdict,
)
from .state_machine import PersonComplianceState
from .verifier import VerifierBase, VerifierContext


@dataclass
class ItemDecision:
    """Final per-item decision and explanatory reason."""

    classification: Classification
    reason: str


@dataclass
class StableStateTracker:
    """Sticky per-item state to reduce dashboard flicker/spam."""

    stable: Classification = Classification.INDETERMINATE
    compliant_streak: int = 0
    violation_streak: int = 0
    indeterminate_streak: int = 0


def log_event(event_type: str, frame_id: int, person_id: int, **fields: object) -> None:
    payload = {"event_type": event_type, "frame_id": frame_id, "person_id": person_id, **fields}
    print(json.dumps(payload, default=str))


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

        self.association = AssociationEngine(config)
        self.cache = VerifierCache()
        sm_cfg = config["state_machine"]
        self.state_machine = PersonComplianceState(
            window_size=int(sm_cfg["window_size"]),
            violation_threshold=int(sm_cfg["violation_threshold"]),
            clear_threshold=int(sm_cfg["clear_threshold"]),
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

        verifier_cfg = config.get("verifier", {})
        conflict_cfg = verifier_cfg.get("conflict_resolver", {})
        self.conflict_min_iou = float(conflict_cfg.get("min_iou", 0.05))
        self.conflict_ambiguity_margin = float(conflict_cfg.get("ambiguity_margin", 0.12))
        self.conflict_low_conf = float(conflict_cfg.get("low_conf_threshold", 0.40))
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

        dash_cfg = config["dashboard"]
        self.jpeg_quality = int(dash_cfg["jpeg_quality"])
        self.metrics_window_seconds = float(dash_cfg["metrics_window_minutes"]) * 60.0

        self.dropped_frames = 0
        self.verifier_calls: Deque[float] = deque()
        self.classification_history: Deque[Tuple[float, Classification]] = deque()
        self._last_frame_perf: float | None = None
        self._fps = 0.0
        self._stable_states: Dict[Tuple[int, str], StableStateTracker] = {}
        self._frames_processed = 0
        self._ppe_infer_calls = 0
        self._verifier_aux_infer_calls = 0
        self._last_detector_counts = {"ppe_primary_raw": 0, "verifier_aux_raw": 0, "ppe_merged": 0}
        self._ppe_model_path = str(config.get("models", {}).get("ppe", ""))
        self._ppe_task = "detect"

        stability_cfg = config.get("status_stability", {})
        self.stable_compliant_enter = int(stability_cfg.get("compliant_enter_frames", 2))
        self.stable_compliant_clear_violation = int(stability_cfg.get("compliant_clear_violation_frames", 3))
        self.stable_violation_enter = int(stability_cfg.get("violation_enter_frames", 4))
        self.stable_indeterminate_enter = int(stability_cfg.get("indeterminate_enter_frames", 8))
        self.stable_per_item = dict(stability_cfg.get("per_item", {}))

    def increment_dropped_frames(self, count: int) -> None:
        if count > 0:
            self.dropped_frames += int(count)

    def process_frame(self, frame: np.ndarray, frame_id: int) -> tuple[FramePayload, bytes]:
        self._frames_processed += 1
        frame_jpeg = self._encode_frame(frame)
        tracked_people = self.pose_tracker.track(frame)
        self._prune_stability_state(tracked_people)
        ppe_detections = self._detect_ppe(frame)

        person_payloads: List[PersonPayload] = []
        for person in tracked_people:
            per_item_state: Dict[str, Classification] = {}
            per_item_state_raw: Dict[str, Classification] = {}
            per_item_reason: Dict[str, str] = {}
            for item in self.required_ppe:
                decision = self._classify_person_item_with_reason(person, ppe_detections, item, frame)
                item_state_raw = decision.classification
                per_item_state_raw[item] = item_state_raw
                item_state = self._stabilize_display_state(person.person_id, item, item_state_raw)
                per_item_state[item] = item_state
                if item_state != item_state_raw:
                    per_item_reason[item] = f"stabilized_hold_{item_state_raw.value.lower()}"
                else:
                    per_item_reason[item] = decision.reason
                self.last_item_reason[(person.person_id, item)] = decision.reason

                if item_state_raw in (Classification.COMPLIANT, Classification.VIOLATION):
                    self.classification_history.append((time.time(), item_state_raw))

                change = self.state_machine.update(
                    person_id=person.person_id,
                    item=item,
                    classification=item_state_raw,
                    frame_jpeg=frame_jpeg,
                )
                if change is not None:
                    log_event(
                        change.event_type,
                        frame_id=frame_id,
                        person_id=person.person_id,
                        item=item,
                        alert_status=change.alert_status.value,
                    )

            overall = self._overall_status(per_item_state)
            person_payloads.append(
                PersonPayload(
                    person_id=person.person_id,
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
                )
            )

        active_alerts = self._build_active_alerts()
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
        return payload, frame_jpeg

    def _stabilize_display_state(
        self,
        person_id: int,
        item: str,
        raw_state: Classification,
    ) -> Classification:
        key = (person_id, item)
        tracker = self._stable_states.get(key)
        if tracker is None:
            tracker = StableStateTracker(stable=raw_state)
            self._stable_states[key] = tracker
            return raw_state

        if raw_state == Classification.COMPLIANT:
            tracker.compliant_streak += 1
            tracker.violation_streak = 0
            tracker.indeterminate_streak = 0
        elif raw_state == Classification.VIOLATION:
            tracker.violation_streak += 1
            tracker.compliant_streak = 0
            tracker.indeterminate_streak = 0
        else:
            tracker.indeterminate_streak += 1
            tracker.compliant_streak = 0
            tracker.violation_streak = 0

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

        person_crop = self._crop_to_bbox(frame, person.bbox)
        item_crop = self._crop_for_item(frame, person, item)
        vctx, ambiguous = self._build_verifier_context(person, item, ppe_detections, person_crop, item_crop)
        needs_verifier = base_classification == Classification.VIOLATION_TENTATIVE or ambiguous

        if not needs_verifier:
            if base_classification == Classification.COMPLIANT:
                return ItemDecision(base_classification, "detected_and_spatially_bound")
            if base_classification == Classification.INDETERMINATE:
                return ItemDecision(base_classification, "keypoint_not_visible_or_out_of_frame")
            if bind is not None and bind.held:
                return ItemDecision(base_classification, "detected_but_held_not_worn")
            return ItemDecision(base_classification, "direct_violation")

        cached = self.cache.get(person.person_id, item)
        if cached is not None:
            final_cls, reason = self._classification_from_verdict(cached.verdict, source="cache")
            return ItemDecision(final_cls, reason)

        verify_result = self.verifier.verify(item_crop, item, context=vctx)
        self.verifier_calls.append(time.time())
        if verify_result.verdict == VerifierVerdict.COMPLIANT:
            ttl = self.ttl_compliant
        elif verify_result.verdict == VerifierVerdict.VIOLATION:
            ttl = self.ttl_violation
        else:
            ttl = self.ttl_indeterminate
        self.cache.put(person.person_id, item, verify_result, ttl)
        final_cls, reason = self._classification_from_verdict(
            verify_result.verdict,
            source=verify_result.source,
        )
        return ItemDecision(final_cls, reason)

    def _build_active_alerts(self) -> List[AlertPayload]:
        alerts: List[AlertPayload] = []
        for person_id, item, item_state in self.state_machine.iter_active_alerts():
            encoded_evidence = (
                base64.b64encode(item_state.evidence_jpeg).decode("ascii")
                if item_state.evidence_jpeg is not None
                else None
            )
            alert_id = f"{person_id}:{item}:{int(item_state.last_transition_ts * 1000)}"
            reason = self.last_item_reason.get((person_id, item), f"missing_or_incorrect_{item}")
            alerts.append(
                AlertPayload(
                    alert_id=alert_id,
                    person_id=person_id,
                    item=item,
                    status=item_state.alert_status,
                    reason=reason,
                    timestamp=item_state.last_transition_ts,
                    evidence_available=item_state.evidence_jpeg is not None,
                    evidence_jpeg_base64=encoded_evidence,
                )
            )
        return alerts

    def _build_metrics(self, tracked_count: int) -> MetricsPayload:
        now = time.time()
        while self.verifier_calls and self.verifier_calls[0] < now - 1.0:
            self.verifier_calls.popleft()

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
            ppe_model=self._ppe_model_path,
            ppe_task=self._ppe_task,
        )

    def _overall_status(self, per_item_state: Dict[str, Classification]) -> OverallStatus:
        if any(v == Classification.VIOLATION for v in per_item_state.values()):
            return OverallStatus.VIOLATION
        if any(v == Classification.INDETERMINATE for v in per_item_state.values()):
            return OverallStatus.INDETERMINATE
        return OverallStatus.COMPLIANT

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return b""
        return bytes(encoded)

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

        min_dist = min(point_distance(center, p) for p in points)
        limit = max(24.0, float(rule.distance_threshold_px) * 1.35)
        return min_dist <= limit

    def _detect_ppe(self, frame: np.ndarray) -> List[PPEDetection]:
        """Run primary detector and optional YOLOE ensemble detector."""
        self._ppe_infer_calls += 1
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
        model = getattr(self.verifier, "model")
        imgsz = getattr(self.verifier, "imgsz", self.config["inference"]["imgsz"])

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
            out.append(
                PPEDetection(
                    label=label,
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    conf=float(conf[idx]) if conf is not None else 0.0,
                    source="yoloe_aux",
                )
            )
        return out

    def _nms_merge_detections(self, detections: List[PPEDetection], iou_threshold: float) -> List[PPEDetection]:
        """Simple class-aware NMS merge across detector sources."""
        by_label: Dict[str, List[PPEDetection]] = {}
        for det in detections:
            by_label.setdefault(det.label, []).append(det)

        kept: List[PPEDetection] = []
        for label, group in by_label.items():
            group_sorted = sorted(group, key=lambda d: d.conf, reverse=True)
            while group_sorted:
                best = group_sorted.pop(0)
                kept.append(best)
                remain: List[PPEDetection] = []
                for det in group_sorted:
                    if box_iou(best.bbox, det.bbox) < iou_threshold:
                        remain.append(det)
                group_sorted = remain
        return kept
