"""Pose-tracking model wrapper using Ultralytics `.track()` and ByteTrack."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass
class TrackedPerson:
    """Tracked person record for one frame."""

    person_id: int
    bbox: Tuple[float, float, float, float]
    keypoints: Dict[str, Tuple[float, float]]
    keypoint_confidences: Dict[str, float]


class PoseTrackerBase:
    """Base pose-tracker interface."""

    def track(self, frame: np.ndarray) -> List[TrackedPerson]:
        raise NotImplementedError


class YOLOPoseTracker(PoseTrackerBase):
    """Ultralytics pose tracker backed by ONNX model."""

    def __init__(self, model: YOLO, conf_threshold: float, imgsz: int) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def track(self, frame: np.ndarray) -> List[TrackedPerson]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.keypoints is None or result.boxes.xyxy is None:
            return []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None

        kp_xy = result.keypoints.xy.cpu().numpy() if result.keypoints.xy is not None else None
        kp_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
        if kp_xy is None or kp_conf is None:
            return []

        tracked: List[TrackedPerson] = []
        for idx, bbox in enumerate(boxes_xyxy):
            person_id = int(ids[idx]) if ids is not None else idx
            keypoints: Dict[str, Tuple[float, float]] = {}
            confs: Dict[str, float] = {}

            for k_idx, k_name in enumerate(COCO_KEYPOINTS):
                if k_idx >= kp_xy.shape[1]:
                    continue
                x, y = kp_xy[idx][k_idx]
                keypoints[k_name] = (float(x), float(y))
                confs[k_name] = float(kp_conf[idx][k_idx])

            tracked.append(
                TrackedPerson(
                    person_id=person_id,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    keypoints=keypoints,
                    keypoint_confidences=confs,
                )
            )
        return tracked


class MockPoseTracker(PoseTrackerBase):
    """Fallback mock pose tracker used when model file is missing."""

    def track(self, frame: np.ndarray) -> List[TrackedPerson]:
        h, w = frame.shape[:2]
        cx = w * 0.5
        cy = h * 0.5
        bbox = (w * 0.3, h * 0.2, w * 0.7, h * 0.9)
        points = {
            "nose": (cx, cy - h * 0.20),
            "left_eye": (cx - 10, cy - h * 0.22),
            "right_eye": (cx + 10, cy - h * 0.22),
            "left_shoulder": (cx - 25, cy - h * 0.1),
            "right_shoulder": (cx + 25, cy - h * 0.1),
            "left_wrist": (cx - 40, cy + h * 0.03),
            "right_wrist": (cx + 40, cy + h * 0.03),
            "left_hip": (cx - 20, cy + h * 0.10),
            "right_hip": (cx + 20, cy + h * 0.10),
            "left_ankle": (cx - 15, cy + h * 0.30),
            "right_ankle": (cx + 15, cy + h * 0.30),
        }
        confs = {name: 0.95 for name in points}
        return [
            TrackedPerson(
                person_id=1,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                keypoints={k: (float(v[0]), float(v[1])) for k, v in points.items()},
                keypoint_confidences=confs,
            )
        ]
