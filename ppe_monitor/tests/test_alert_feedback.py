"""Tests for alert feedback persistence and ack analytics."""

from pathlib import Path

from app.alert_feedback import AlertFeedbackStore


def test_acknowledge_updates_stats(tmp_path: Path) -> None:
    store = AlertFeedbackStore(path=tmp_path / "feedback.jsonl", enabled=True, max_recent_records=1000)
    store.observe_active_alerts(
        alerts=[{"alert_id": "a1", "person_id": 1, "item": "helmet", "display_id": "ID_1-cam"}],
        camera_id="cam",
        observed_ts=100.0,
    )
    record = store.acknowledge(
        alert_id="a1",
        person_id=1,
        display_id="ID_1-cam",
        item="helmet",
        camera_id="cam",
        acknowledged=True,
        positive_conf=0.73,
        negative_conf=0.12,
    )
    assert record["acknowledged"] is True
    assert store.is_acknowledged("a1") is True
    stats = store.stats()
    assert stats["total_feedback"] == 1
    assert stats["acknowledged"] == 1
    assert stats["by_item"]["helmet"]["acknowledged"] == 1


def test_unacknowledged_alert_auto_closed_when_cleared(tmp_path: Path) -> None:
    store = AlertFeedbackStore(path=tmp_path / "feedback.jsonl", enabled=True, max_recent_records=1000)
    store.observe_active_alerts(
        alerts=[{"alert_id": "a2", "person_id": 2, "item": "gloves", "display_id": "ID_2-cam"}],
        camera_id="cam",
        observed_ts=10.0,
    )
    store.observe_active_alerts(alerts=[], camera_id="cam", observed_ts=15.0)
    stats = store.stats()
    assert stats["total_feedback"] == 1
    assert stats["acknowledged"] == 0
    assert stats["not_acknowledged"] == 1
    assert stats["by_item"]["gloves"]["not_acknowledged"] == 1

