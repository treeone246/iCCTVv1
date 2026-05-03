"""Helpers for converting model outputs into the bridge JSON schema."""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping, Optional, Sequence


def _to_float(value: Any) -> float:
    """Convert numeric-like values (including numpy scalars) to float."""
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def bbox_to_schema(x1: Any, y1: Any, x2: Any, y2: Any) -> dict[str, float]:
    """
    Convert bbox coordinates into schema fields without changing their meaning.
    Coordinates are preserved as numeric pixel values.
    """
    return {
        "x1": _to_float(x1),
        "y1": _to_float(y1),
        "x2": _to_float(x2),
        "y2": _to_float(y2),
    }


def adapt_parsed_detections(
    detections: Sequence[Mapping[str, Any]],
    frame_id: str,
    timestamp: Optional[float] = None,
) -> dict[str, Any]:
    """
    Adapt already-parsed detections.

    Each detection is expected to contain:
    label, score, x1, y1, x2, y2
    """
    ts = time.time() if timestamp is None else float(timestamp)
    formatted: list[dict[str, Any]] = []

    for det in detections:
        bbox = bbox_to_schema(det["x1"], det["y1"], det["x2"], det["y2"])
        formatted.append(
            {
                "label": str(det["label"]),
                "score": _to_float(det["score"]),
                **bbox,
            }
        )

    return {
        "timestamp": ts,
        "frame_id": str(frame_id),
        "detections": formatted,
    }


def adapt_yolo_detections(
    yolo_boxes: Iterable[Any],
    frame_id: str,
    class_names: Optional[Mapping[int, str] | Sequence[str]] = None,
    timestamp: Optional[float] = None,
) -> dict[str, Any]:
    """
    Example adapter for YOLO-style outputs.

    Supports common patterns:
    - dict: {"xyxy":[x1,y1,x2,y2], "conf":0.9, "cls":0}
    - object: .xyxy, .conf, .cls (Ultralytics-like)
    - tuple/list: [x1, y1, x2, y2, conf, cls]
    """
    parsed: list[dict[str, Any]] = []

    for box in yolo_boxes:
        x1: Any
        y1: Any
        x2: Any
        y2: Any
        conf: Any
        cls_id: int

        if isinstance(box, Mapping):
            xyxy = box.get("xyxy")
            if xyxy is None:
                xyxy = [box["x1"], box["y1"], box["x2"], box["y2"]]
            x1, y1, x2, y2 = xyxy
            conf = box.get("conf", box.get("confidence", 0.0))
            cls_id = int(_to_float(box.get("cls", box.get("class_id", -1))))
        elif isinstance(box, (tuple, list)) and len(box) >= 6:
            x1, y1, x2, y2, conf, cls_raw = box[:6]
            cls_id = int(_to_float(cls_raw))
        else:
            # Ultralytics-style object handling.
            xyxy = getattr(box, "xyxy", None)
            if xyxy is None:
                raise ValueError("YOLO box object missing 'xyxy'")

            if hasattr(xyxy, "__len__") and len(xyxy) > 0 and hasattr(xyxy[0], "__len__"):
                xyxy = xyxy[0]

            x1, y1, x2, y2 = xyxy
            conf = getattr(box, "conf", 0.0)
            cls_val = getattr(box, "cls", -1)

            if hasattr(conf, "__len__"):
                conf = conf[0]
            if hasattr(cls_val, "__len__"):
                cls_val = cls_val[0]
            cls_id = int(_to_float(cls_val))

        label = str(cls_id)
        if class_names is not None and cls_id >= 0:
            if isinstance(class_names, Mapping):
                label = str(class_names.get(cls_id, label))
            elif cls_id < len(class_names):
                label = str(class_names[cls_id])

        parsed.append(
            {
                "label": label,
                "score": _to_float(conf),
                "x1": _to_float(x1),
                "y1": _to_float(y1),
                "x2": _to_float(x2),
                "y2": _to_float(y2),
            }
        )

    return adapt_parsed_detections(parsed, frame_id=frame_id, timestamp=timestamp)
