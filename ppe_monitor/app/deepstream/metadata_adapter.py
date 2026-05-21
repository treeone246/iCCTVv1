"""DeepStream metadata adapter with pyds-free dataclasses for testability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ..pose_tracker import TrackedPerson
from ..ppe_detector import PPEDetection, canonicalize_label


@dataclass
class DsObjectMeta:
    """Mockable mirror of pyds NvDsObjectMeta fields used by this app."""

    class_id: int
    class_label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # left, top, width, height
    object_id: int
    source_id: int


@dataclass
class DsFrameMeta:
    frame_num: int
    source_id: int
    pts_ns: int
    objects: List[DsObjectMeta]


@dataclass
class AdaptedDeepStreamFrame:
    frame_id: int
    source_id: int
    camera_id: str
    timestamp_s: float
    persons: List[TrackedPerson]
    ppe_detections: List[PPEDetection]
    object_count: int
    class_counts: Dict[str, int]


def adapt_frame(
    ds_frame: DsFrameMeta,
    *,
    camera_id: str,
    label_map: Mapping[int, str],
    person_classes: Iterable[str],
    ppe_classes: Iterable[str],
    alias_to_canonical: Mapping[str, str] | None = None,
) -> AdaptedDeepStreamFrame:
    """Convert one DeepStream frame metadata snapshot into app-native objects."""
    person_set = {str(x).strip().lower() for x in person_classes}
    ppe_set = {str(x).strip().lower() for x in ppe_classes}
    alias = dict(alias_to_canonical or {})

    persons: List[TrackedPerson] = []
    detections: List[PPEDetection] = []
    class_counts: Dict[str, int] = {}

    for obj in ds_frame.objects:
        raw_label = str(label_map.get(int(obj.class_id), obj.class_label or obj.class_id))
        label = canonicalize_label(raw_label, alias) if alias else raw_label.strip().lower().replace(" ", "_")
        class_counts[label] = class_counts.get(label, 0) + 1
        bbox_xyxy = _xywh_to_xyxy(obj.bbox)

        if label in person_set:
            persons.append(
                TrackedPerson(
                    person_id=int(obj.object_id),
                    bbox=bbox_xyxy,
                    keypoints={},
                    keypoint_confidences={},
                )
            )
            continue

        if label in ppe_set:
            detections.append(
                PPEDetection(
                    label=label,
                    bbox=bbox_xyxy,
                    conf=float(obj.confidence),
                    source="deepstream_primary",
                )
            )

    timestamp_s = float(ds_frame.pts_ns) / 1_000_000_000.0 if ds_frame.pts_ns > 0 else _utc_now_s()
    return AdaptedDeepStreamFrame(
        frame_id=int(ds_frame.frame_num),
        source_id=int(ds_frame.source_id),
        camera_id=str(camera_id),
        timestamp_s=timestamp_s,
        persons=persons,
        ppe_detections=detections,
        object_count=len(ds_frame.objects),
        class_counts=class_counts,
    )


def _xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    left, top, width, height = [float(v) for v in bbox]
    return (left, top, left + width, top + height)


def _utc_now_s() -> float:
    return datetime.now(tz=timezone.utc).timestamp()
