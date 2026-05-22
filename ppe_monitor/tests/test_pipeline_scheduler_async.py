"""Unit tests for adaptive scheduler and async verifier queue helpers."""

import queue
import threading
from types import SimpleNamespace

import numpy as np

from app.pipeline import MonitoringPipeline, VerifierTask
from app.verifier import VerifierContext


def test_adaptive_scheduler_reuses_between_intervals() -> None:
    p = object.__new__(MonitoringPipeline)
    p.adaptive_scheduler_enabled = True
    p.ppe_interval_frames = 2
    p.ppe_max_staleness_frames = 4
    p.ppe_force_on_new_track = True
    p._last_ppe_detections = []
    p._last_ppe_frame_id = -1_000_000
    p._last_track_ids = set()
    p._adaptive_detect_frames = 0
    p._adaptive_reuse_frames = 0

    calls = {"count": 0}

    def _fake_detect(_frame):
        calls["count"] += 1
        return [SimpleNamespace(label="helmet", conf=0.9, bbox=(0.0, 0.0, 1.0, 1.0), source="ppe_primary")]

    p._detect_ppe = _fake_detect  # type: ignore[attr-defined]

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    tracks = [SimpleNamespace(person_id=1)]

    out0 = p._select_ppe_detections(frame=frame, frame_id=0, tracked_people=tracks)
    assert calls["count"] == 1
    assert len(out0) == 1

    out1 = p._select_ppe_detections(frame=frame, frame_id=1, tracked_people=tracks)
    assert calls["count"] == 1
    assert len(out1) == 1
    assert p._adaptive_reuse_frames == 1

    _ = p._select_ppe_detections(frame=frame, frame_id=2, tracked_people=tracks)
    assert calls["count"] == 2
    assert p._adaptive_detect_frames == 2


def test_async_verifier_queue_deduplicates_pending_key() -> None:
    p = object.__new__(MonitoringPipeline)
    p._async_lock = threading.Lock()
    p._async_pending = set()
    p._async_in_q = queue.Queue(maxsize=8)
    p._async_enqueued = 0
    p._async_dropped = 0
    p.async_verifier_drop_if_full = True

    crop = np.zeros((16, 16, 3), dtype=np.uint8)
    ctx = VerifierContext(expected_item="gloves")

    r1 = p._queue_async_verifier_task(
        person_id=7,
        item="gloves",
        item_crop=crop,
        context=ctx,
    )
    r2 = p._queue_async_verifier_task(
        person_id=7,
        item="gloves",
        item_crop=crop,
        context=ctx,
    )

    assert r1 == "async_verifier_queued"
    assert r2 == "async_verifier_pending"
    assert p._async_enqueued == 1
    assert len(p._async_pending) == 1


def test_async_verifier_queue_full_returns_queue_full_reason() -> None:
    p = object.__new__(MonitoringPipeline)
    p._async_lock = threading.Lock()
    p._async_pending = set()
    p._async_in_q = queue.Queue(maxsize=1)
    p._async_enqueued = 0
    p._async_dropped = 0
    p.async_verifier_drop_if_full = True

    # Pre-fill queue so next enqueue fails.
    dummy = VerifierTask(
        person_id=1,
        item="helmet",
        item_crop=np.zeros((8, 8, 3), dtype=np.uint8),
        context=VerifierContext(expected_item="helmet"),
    )
    p._async_in_q.put_nowait(dummy)

    reason = p._queue_async_verifier_task(
        person_id=9,
        item="goggles",
        item_crop=np.zeros((8, 8, 3), dtype=np.uint8),
        context=VerifierContext(expected_item="goggles"),
    )
    assert reason == "async_verifier_queue_full"
    assert p._async_dropped == 1
