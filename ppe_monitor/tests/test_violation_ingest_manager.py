"""Tests for violation ingest manager modes."""

from __future__ import annotations

from app.violation_ingest import ViolationIngestManager


class _DummyPgLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def ingest_alerts(self, *, alerts, camera_id: str) -> None:  # type: ignore[no-untyped-def]
        self.calls.append((str(camera_id), len(list(alerts))))

    def status(self) -> dict:
        return {"enabled": True}


def _base_alert() -> dict:
    return {
        "alert_id": "x-1",
        "person_id": 1,
        "display_id": "ID_1-cam_01",
        "item": "helmet",
        "status": "ACTIVE",
        "reason": "helmet_missing",
        "negative_conf": 0.9,
        "timestamp": 1_700_000_000.0,
    }


def test_kafka_flink_mode_disables_local_sink() -> None:
    cfg = {
        "violation_ingest": {
            "enabled": True,
            "mode": "kafka_flink",
            "filter": {"enabled": True, "min_negative_confidence_default": 0.0},
            "queue": {"enabled": True},
            "kafka": {"enabled": False, "also_write_local": False},
        }
    }
    pg = _DummyPgLogger()
    manager = ViolationIngestManager(config=cfg, pg_logger=pg)
    try:
        manager.ingest_alerts(alerts=[_base_alert()], camera_id="cam_01")
        assert manager.local_sink_enabled is False
        assert pg.calls == []
    finally:
        manager.close()


def test_direct_mode_writes_local_sink_without_queue() -> None:
    cfg = {
        "violation_ingest": {
            "enabled": True,
            "mode": "direct",
            "filter": {"enabled": True, "min_negative_confidence_default": 0.0},
            "queue": {"enabled": False},
            "kafka": {"enabled": False, "also_write_local": True},
        }
    }
    pg = _DummyPgLogger()
    manager = ViolationIngestManager(config=cfg, pg_logger=pg)
    try:
        manager.ingest_alerts(alerts=[_base_alert()], camera_id="cam_01")
        assert manager.local_sink_enabled is True
        assert pg.calls == [("cam_01", 1)]
    finally:
        manager.close()
