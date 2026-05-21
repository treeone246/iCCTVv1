"""Compatibility helpers for DeepStream metadata to existing payload assumptions."""

from __future__ import annotations

from typing import Iterable, Tuple

from ..pose_tracker import TrackedPerson


UNAVAILABLE_KEYPOINTS: dict[str, Tuple[float, float]] = {}
UNAVAILABLE_CONFIDENCES: dict[str, float] = {}


def fill_unavailable_keypoints(persons: Iterable[TrackedPerson]) -> None:
    """Ensure person records always contain keypoint fields expected downstream."""
    for person in persons:
        if not person.keypoints:
            person.keypoints = dict(UNAVAILABLE_KEYPOINTS)
        if not person.keypoint_confidences:
            person.keypoint_confidences = dict(UNAVAILABLE_CONFIDENCES)
