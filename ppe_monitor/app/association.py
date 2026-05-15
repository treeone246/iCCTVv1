"""Spatial PPE-to-limb association rules based on pose keypoints."""

from dataclasses import dataclass
from math import hypot
from typing import Dict, List, Optional, Tuple

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
        min_expected_distance = min(distance(center, pt) for pt in expected_points)

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

        held = False
        if self.item in self.held_items:
            wrist_points = [
                keypoints[w]
                for w in ("left_wrist", "right_wrist")
                if w in keypoints and keypoint_confidences.get(w, 0.0) >= self.keypoint_conf_floor
            ]
            if wrist_points:
                min_wrist_distance = min(distance(center, wrist) for wrist in wrist_points)
                if min_wrist_distance * self.held_distance_ratio < min_expected_distance:
                    held = True

        return BindResult(bound=bound, held=held, confidence=confidence)


class AssociationEngine:
    """Evaluates PPE detections against per-item keypoint association rules."""

    def __init__(self, config: dict) -> None:
        assoc_cfg = config["association"]
        self.keypoint_conf_floor = float(assoc_cfg["keypoint_conf_floor"])
        held_items = tuple(str(item) for item in assoc_cfg.get("held_items", []))

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

        item_detections = [det for det in ppe_detections if det.get("label") == item]
        if not item_detections:
            return Classification.VIOLATION_TENTATIVE, None

        best_result: Optional[BindResult] = None
        for det in item_detections:
            result = rule.bind(det["bbox"], keypoints, keypoint_confidences)
            if result.held:
                return Classification.VIOLATION, result
            if result.bound and (best_result is None or result.confidence > best_result.confidence):
                best_result = result

        if best_result is not None:
            return Classification.COMPLIANT, best_result
        return Classification.VIOLATION_TENTATIVE, None

    def _required_points_observable(
        self,
        rule: PPERule,
        keypoints: KeypointMap,
        keypoint_confidences: ConfidenceMap,
        frame_shape: Tuple[int, int, int],
    ) -> bool:
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


def bbox_center(box: BBox) -> Point:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


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
