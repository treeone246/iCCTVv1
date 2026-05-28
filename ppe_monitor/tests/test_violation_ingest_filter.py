"""Tests for violation ingest anti-spam filtering."""

from __future__ import annotations

import time

from app.schemas import AlertStatus
from app.violation_ingest import ViolationAlertFilter


def _base_config() -> dict:
    return {
        "verifier": {
            "label_polarity": {
                "helmet": {"positive": ["helmet"], "negative": []},
                "goggles": {"positive": ["goggles"], "negative": []},
                "gloves": {"positive": ["gloves"], "negative": ["no_gloves"]},
                "boots": {"positive": ["boots"], "negative": ["no_boots"]},
            }
        },
        "violation_ingest": {
            "enabled": True,
            "filter": {
                "enabled": True,
                "only_active": True,
                "repeat_suppression_seconds": 5.0,
                "remember_ttl_seconds": 120.0,
                "max_events_per_minute_per_key": 5,
                "min_negative_confidence_default": 0.2,
                "min_negative_confidence_per_item": {
                    "helmet": 0.35,
                    "gloves": 0.2,
                    "boots": 0.2,
                },
                "enforce_negative_conf_items": [],
                "bypass_low_conf_items": ["goggles", "gloves"],
                "allow_reason_change_bypass": True,
            },
        }
    }


def test_repeat_alert_is_suppressed() -> None:
    filt = ViolationAlertFilter(config=_base_config())
    alert = {
        "alert_id": "a1",
        "person_id": 1,
        "display_id": "ID_1-cam_01",
        "item": "helmet",
        "status": "ACTIVE",
        "reason": "helmet_not_worn",
        "negative_conf": 0.92,
    }
    first = filt.filter(alerts=[alert], camera_id="cam_01")
    second = filt.filter(alerts=[alert], camera_id="cam_01")
    assert len(first) == 1
    assert len(second) == 0


def test_low_confidence_alert_is_suppressed() -> None:
    filt = ViolationAlertFilter(config=_base_config())
    low_conf_boots = {
        "alert_id": "a2",
        "person_id": 1,
        "display_id": "ID_1-cam_01",
        "item": "boots",
        "status": "ACTIVE",
        "reason": "low_conf_boots",
        "negative_conf": 0.05,
    }
    rows = filt.filter(alerts=[low_conf_boots], camera_id="cam_01")
    assert rows == []


def test_reason_change_can_bypass_repeat_suppression() -> None:
    filt = ViolationAlertFilter(config=_base_config())
    alert_a = {
        "alert_id": "a3",
        "person_id": 3,
        "display_id": "ID_3-cam_01",
        "item": "helmet",
        "status": "ACTIVE",
        "reason": "helmet_missing",
        "negative_conf": 0.91,
    }
    alert_b = dict(alert_a)
    alert_b["reason"] = "helmet_incorrect_position"
    first = filt.filter(alerts=[alert_a], camera_id="cam_01")
    # Within repeat window, but reason changed.
    second = filt.filter(alerts=[alert_b], camera_id="cam_01")
    assert len(first) == 1
    assert len(second) == 1
    assert second[0]["reason"] == "helmet_incorrect_position"


def test_ttl_allows_new_after_expiry() -> None:
    cfg = _base_config()
    cfg["violation_ingest"]["filter"]["repeat_suppression_seconds"] = 0.2
    filt = ViolationAlertFilter(config=cfg)
    alert = {
        "alert_id": "a4",
        "person_id": 9,
        "display_id": "ID_9-cam_01",
        "item": "helmet",
        "status": "ACTIVE",
        "reason": "helmet_missing",
        "negative_conf": 0.9,
    }
    assert len(filt.filter(alerts=[alert], camera_id="cam_01")) == 1
    assert len(filt.filter(alerts=[alert], camera_id="cam_01")) == 0
    time.sleep(0.25)
    assert len(filt.filter(alerts=[alert], camera_id="cam_01")) == 1


def test_enum_status_active_is_accepted() -> None:
    filt = ViolationAlertFilter(config=_base_config())
    alert = {
        "alert_id": "a5",
        "person_id": 4,
        "display_id": "ID_4-cam_01",
        "item": "helmet",
        "status": AlertStatus.ACTIVE,
        "reason": "helmet_missing",
        "negative_conf": 0.91,
    }
    rows = filt.filter(alerts=[alert], camera_id="cam_01")
    assert len(rows) == 1


def test_goggles_low_negative_conf_not_suppressed() -> None:
    filt = ViolationAlertFilter(config=_base_config())
    alert = {
        "alert_id": "a6",
        "person_id": 6,
        "display_id": "ID_6-cam_01",
        "item": "goggles",
        "status": "ACTIVE",
        "reason": "direct_violation",
        "negative_conf": 0.0,
    }
    rows = filt.filter(alerts=[alert], camera_id="cam_01")
    assert len(rows) == 1


def test_gloves_low_negative_conf_not_suppressed_with_bypass() -> None:
    filt = ViolationAlertFilter(config=_base_config())
    alert = {
        "alert_id": "a7",
        "person_id": 7,
        "display_id": "ID_7-cam_01",
        "item": "gloves",
        "status": "ACTIVE",
        "reason": "verifier_ollama_violation",
        "negative_conf": 0.01,
    }
    rows = filt.filter(alerts=[alert], camera_id="cam_01")
    assert len(rows) == 1
