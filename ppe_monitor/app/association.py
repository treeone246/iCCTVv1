"""Spatial PPE-to-limb association rules based on pose keypoints."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schemas import Classification


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
KeypointMap = Dict[str, Point]
ConfidenceMap = Dict[str, float]


@dataclass
class BindResult:
    """Association outcome for one PPE bbox against one PPE rule."""

    bound: bool
    held: bool
    confidence: float
    reason: str = ""


@dataclass
class PPERule:
    """Rule describing how one PPE item should bind to keypoints."""

    item: str
    expected_keypoints: List[str]
    keypoint_conf_floor: float
    distance_threshold_px: float = 0.0
    iou_threshold: float = 0.0
    held_distance_ratio: float = 1.2
    held_items: Tuple[str, ...] = ("helmet", "goggles", "gloves")

    def bind(
        self,
        ppe_bbox: BBox,
        keypoints: KeypointMap,
        keypoint_confidences: ConfidenceMap,
    ) -> BindResult:
        expected_points = [
            keypoints[k]
            for k in self.expected_keypoints
            if k in keypoints and keypoint_confidences.get(k, 0.0) >= self.keypoint_conf_floor
        ]
        if not expected_points:
            return BindResult(bound=False, held=False, confidence=0.0, reason="expected_keypoint_not_visible")

        center = bbox_center(ppe_bbox)
        expected_np = np.asarray(expected_points, dtype=np.float32)
        dx = expected_np[:, 0] - float(center[0])
        dy = expected_np[:, 1] - float(center[1])
        distances = np.hypot(dx, dy)
        min_expected_distance = float(np.min(distances))

        bound = False
        confidence = 0.0
        if self.item == "coverall":
            torso_box = torso_bbox(keypoints, keypoint_confidences, self.keypoint_conf_floor)
            if torso_box is None:
                return BindResult(bound=False, held=False, confidence=0.0, reason="torso_not_visible")
            overlap = iou(ppe_bbox, torso_box)
            bound = overlap >= self.iou_threshold
            confidence = overlap
        else:
            bound = min_expected_distance <= self.distance_threshold_px
            if self.distance_threshold_px > 0:
                confidence = max(0.0, 1.0 - (min_expected_distance / self.distance_threshold_px))
            if bound and self.item in {"gloves", "boots"}:
                nearest_idx = int(np.argmin(distances))
                nearest_point = expected_points[nearest_idx]
                x1, y1, x2, y2 = ppe_bbox
                side = max(1.0, min(float(x2 - x1), float(y2 - y1)))
                pad = max(6.0, side * 0.18)
                if not point_in_or_near_bbox(nearest_point, ppe_bbox, pad):
                    return BindResult(
                        bound=False,
                        held=False,
                        confidence=0.0,
                        reason="nearest_limb_keypoint_outside_bbox",
                    )

        held = False
        if self.item in self.held_items:
            wrist_points = [
                keypoints[w]
                for w in ("left_wrist", "right_wrist")
                if w in keypoints and keypoint_confidences.get(w, 0.0) >= self.keypoint_conf_floor
            ]
            if wrist_points:
                wrists_np = np.asarray(wrist_points, dtype=np.float32)
                wdx = wrists_np[:, 0] - float(center[0])
                wdy = wrists_np[:, 1] - float(center[1])
                min_wrist_distance = float(np.min(np.hypot(wdx, wdy)))
                if min_wrist_distance * self.held_distance_ratio < min_expected_distance:
                    held = True

        return BindResult(bound=bound, held=held, confidence=confidence)


class AssociationEngine:
    """Evaluates PPE detections against per-item keypoint association rules."""

    def __init__(self, config: dict) -> None:
        assoc_cfg = config["association"]
        self.keypoint_conf_floor = float(assoc_cfg["keypoint_conf_floor"])
        held_items = tuple(str(item) for item in assoc_cfg.get("held_items", []))
        self.goggles_face_gate = bool(assoc_cfg.get("goggles_face_gate_enabled", True))
        self.goggles_face_min_points = int(assoc_cfg.get("goggles_face_min_points", 2))
        self.goggles_face_points = ("nose", "left_eye", "right_eye")
        self.bilateral_items = set(
            str(item) for item in assoc_cfg.get("bilateral_items", ["gloves", "boots"])
        )
        self.bilateral_bbox_pad_px = float(assoc_cfg.get("bilateral_bbox_pad_px", 12.0))
        self.bilateral_distance_ratio = float(assoc_cfg.get("bilateral_distance_ratio", 1.15))

        self.rules: Dict[str, PPERule] = {
            "helmet": PPERule(
                item="helmet",
                expected_keypoints=list(assoc_cfg["helmet_keypoints"]),
                keypoint_conf_floor=self.keypoint_conf_floor,
                distance_threshold_px=float(assoc_cfg["helmet_distance_px"]),
                held_distance_ratio=float(assoc_cfg["held_distance_ratio"]),
                held_items=held_items,
            ),
            "goggles": PPERule(
                item="goggles",
                expected_keypoints=list(assoc_cfg["goggles_keypoints"]),
                keypoint_conf_floor=self.keypoint_conf_floor,
                distance_threshold_px=float(assoc_cfg["goggles_distance_px"]),
                held_distance_ratio=float(assoc_cfg["held_distance_ratio"]),
                held_items=held_items,
            ),
            "gloves": PPERule(
                item="gloves",
                expected_keypoints=list(assoc_cfg["gloves_keypoints"]),
                keypoint_conf_floor=self.keypoint_conf_floor,
                distance_threshold_px=float(assoc_cfg["gloves_distance_px"]),
                held_distance_ratio=float(assoc_cfg["held_distance_ratio"]),
                held_items=held_items,
            ),
            "boots": PPERule(
                item="boots",
                expected_keypoints=list(assoc_cfg["boots_keypoints"]),
                keypoint_conf_floor=self.keypoint_conf_floor,
                distance_threshold_px=float(assoc_cfg["boots_distance_px"]),
                held_distance_ratio=float(assoc_cfg["held_distance_ratio"]),
                held_items=held_items,
            ),
            "coverall": PPERule(
                item="coverall",
                expected_keypoints=list(assoc_cfg["coverall_keypoints"]),
                keypoint_conf_floor=self.keypoint_conf_floor,
                iou_threshold=float(assoc_cfg["coverall_iou_threshold"]),
                held_distance_ratio=float(assoc_cfg["held_distance_ratio"]),
                held_items=held_items,
            ),
        }

    def classify_item(
        self,
        item: str,
        keypoints: KeypointMap,
        keypoint_confidences: ConfidenceMap,
        ppe_detections: List[dict],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[Classification, Optional[BindResult]]:
        rule = self.rules[item]
        if not self._required_points_observable(rule, keypoints, keypoint_confidences, frame_shape):
            return Classification.INDETERMINATE, None
        visible_points = self._visible_points_for_rule(
            rule,
            keypoints,
            keypoint_confidences,
            frame_shape,
        )
        if not visible_points:
            return Classification.INDETERMINATE, None

        item_detections = [det for det in ppe_detections if det.get("label") == item]
        if not item_detections:
            return Classification.VIOLATION_TENTATIVE, None

        best_result: Optional[BindResult] = None
        covered_points: set[str] = set()
        for det in item_detections:
            result = rule.bind(det["bbox"], keypoints, keypoint_confidences)
            if result.held:
                return Classification.VIOLATION, result
            if item in self.bilateral_items:
                det_center = bbox_center(det["bbox"])
                for kp_name, kp in visible_points.items():
                    if kp_name in covered_points:
                        continue
                    if not point_in_or_near_bbox(kp, det["bbox"], self.bilateral_bbox_pad_px):
                        continue
                    limit = max(8.0, float(rule.distance_threshold_px) * self.bilateral_distance_ratio)
                    if distance(det_center, kp) <= limit:
                        covered_points.add(kp_name)
            if result.bound and (best_result is None or result.confidence > best_result.confidence):
                best_result = result

        if item in self.bilateral_items and len(visible_points) >= 2:
            if len(covered_points) < len(visible_points):
                return Classification.VIOLATION_TENTATIVE, None

        if best_result is not None:
            return Classification.COMPLIANT, best_result
        return Classification.VIOLATION_TENTATIVE, None

    def _visible_points_for_rule(
        self,
        rule: PPERule,
        keypoints: KeypointMap,
        keypoint_confidences: ConfidenceMap,
        frame_shape: Tuple[int, int, int],
    ) -> Dict[str, Point]:
        h, w = frame_shape[:2]
        visible: Dict[str, Point] = {}
        for kp_name in rule.expected_keypoints:
            conf = keypoint_confidences.get(kp_name, 0.0)
            point = keypoints.get(kp_name)
            if point is None or conf < self.keypoint_conf_floor:
                continue
            x, y = point
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            visible[kp_name] = point
        return visible

    def _required_points_observable(
        self,
        rule: PPERule,
        keypoints: KeypointMap,
        keypoint_confidences: ConfidenceMap,
        frame_shape: Tuple[int, int, int],
    ) -> bool:
        if rule.item == "goggles" and self.goggles_face_gate:
            return self._goggles_face_observable(keypoints, keypoint_confidences, frame_shape)

        h, w = frame_shape[:2]
        visible_points = 0

        for kp_name in rule.expected_keypoints:
            conf = keypoint_confidences.get(kp_name, 0.0)
            point = keypoints.get(kp_name)
            if point is None or conf < self.keypoint_conf_floor:
                continue
            x, y = point
            if x < 0 or y < 0 or x >= w or y >= h:
                return False
            visible_points += 1

        return visible_points > 0

    def _goggles_face_observable(
        self,
        keypoints: KeypointMap,
        keypoint_confidences: ConfidenceMap,
        frame_shape: Tuple[int, int, int],
    ) -> bool:
        h, w = frame_shape[:2]
        visible: Dict[str, bool] = {}
        visible_count = 0
        for kp_name in self.goggles_face_points:
            conf = keypoint_confidences.get(kp_name, 0.0)
            point = keypoints.get(kp_name)
            if point is None or conf < self.keypoint_conf_floor:
                visible[kp_name] = False
                continue
            x, y = point
            if x < 0 or y < 0 or x >= w or y >= h:
                return False
            visible[kp_name] = True
            visible_count += 1

        if visible_count < max(1, self.goggles_face_min_points):
            return False

        # Face is considered evaluable for goggles if:
        # - both eyes are visible, or
        # - nose plus at least one eye are visible.
        both_eyes = bool(visible.get("left_eye")) and bool(visible.get("right_eye"))
        nose_plus_one_eye = bool(visible.get("nose")) and (
            bool(visible.get("left_eye")) or bool(visible.get("right_eye"))
        )
        return both_eyes or nose_plus_one_eye


def bbox_center(box: BBox) -> Point:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def point_in_or_near_bbox(point: Point, bbox: BBox, pad: float = 0.0) -> bool:
    x, y = point
    return (
        x >= (bbox[0] - pad)
        and x <= (bbox[2] + pad)
        and y >= (bbox[1] - pad)
        and y <= (bbox[3] + pad)
    )


def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    a_area = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    b_area = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    denom = a_area + b_area - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def torso_bbox(
    keypoints: KeypointMap,
    keypoint_confidences: ConfidenceMap,
    conf_floor: float,
) -> Optional[BBox]:
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    points: List[Point] = []
    for name in required:
        if keypoint_confidences.get(name, 0.0) < conf_floor or name not in keypoints:
            return None
        points.append(keypoints[name])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
