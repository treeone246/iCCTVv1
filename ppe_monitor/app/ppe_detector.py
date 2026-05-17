"""PPE detector wrapper around Ultralytics detection model."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class PPEDetection:
    """One PPE detection result."""

    label: str
    bbox: Tuple[float, float, float, float]
    conf: float


class PPEDetectorBase:
    """Base detector interface."""

    def detect(self, frame: np.ndarray) -> List[PPEDetection]:
        raise NotImplementedError


class YOLOPPEDetector(PPEDetectorBase):
    """Ultralytics detector for fixed PPE classes."""

    def __init__(
        self,
        model: YOLO,
        conf_threshold: float,
        imgsz: int,
        label_aliases: Dict[str, List[str]] | None = None,
    ) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.alias_to_canonical = build_alias_index(label_aliases or {})

    def detect(self, frame: np.ndarray) -> List[PPEDetection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
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

        detections: List[PPEDetection] = []
        for idx, box in enumerate(xyxy):
            class_id = int(cls[idx]) if cls is not None else -1
            raw_label = str(names.get(class_id, class_id))
            label = canonicalize_label(raw_label, self.alias_to_canonical)
            detections.append(
                PPEDetection(
                    label=label,
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    conf=float(conf[idx]) if conf is not None else 0.0,
                )
            )
        return detections


class MockPPEDetector(PPEDetectorBase):
    """Fallback mock PPE detector that returns no detections."""

    def detect(self, frame: np.ndarray) -> List[PPEDetection]:
        return []


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def build_alias_index(mapping: Dict[str, List[str]]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for canonical, aliases in mapping.items():
        c = normalize_label(canonical)
        index[c] = c
        for alias in aliases:
            index[normalize_label(str(alias))] = c
    return index


def canonicalize_label(raw_label: str, alias_index: Dict[str, str]) -> str:
    normalized = normalize_label(raw_label)
    return alias_index.get(normalized, normalized)
