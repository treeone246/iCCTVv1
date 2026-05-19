"""Tests for asynchronous per-person event stream writer safety."""

import queue

from app.event_stream import EventStreamWriter


def test_event_stream_emit_does_not_raise_when_queue_full() -> None:
    writer = EventStreamWriter({"event_stream": {"enabled": False}})
    writer.enabled = True
    writer._q = queue.Queue(maxsize=1)  # type: ignore[attr-defined]
    writer._q.put_nowait("already-full")  # type: ignore[attr-defined]

    writer.emit_person_observation(
        frame_id=12,
        track_id=5,
        bbox=(1.0, 2.0, 3.0, 4.0),
        per_item={
            "helmet": {
                "status_raw": "VIOLATION",
                "status_stable": "VIOLATION",
                "positive_conf": 0.1,
                "negative_conf": 0.9,
                "reason": "direct_violation",
                "sm_stage": "VIOLATION_CANDIDATE",
                "alert_status": "CLEARED",
            }
        },
        overall_status="VIOLATION",
        tracking_confidence=None,
    )

    assert writer.dropped == 1
