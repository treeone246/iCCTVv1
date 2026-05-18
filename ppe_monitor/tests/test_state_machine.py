"""Unit tests for vote/timer-based anti-spam state transitions."""

from app.schemas import AlertStatus, Classification
from app.state_machine import PersonComplianceState


def test_candidate_then_confirmed_after_stable_seconds() -> None:
    sm = PersonComplianceState(
        window_size=30,
        violation_threshold=20,
        clear_threshold=1,
        confirm_seconds=2.0,
        cooldown_seconds=60.0,
    )

    raised = None
    ts = 0.0
    for _ in range(45):
        evt = sm.update(
            person_id=11,
            item="helmet",
            classification=Classification.VIOLATION,
            frame_jpeg=b"evidence",
            event_ts=ts,
        )
        if evt is not None and evt.event_type == "ALERT_RAISED":
            raised = evt
        ts += 0.1

    state = sm.get_item_state(person_id=11, item="helmet")
    assert state.violation_stage == "VIOLATION_CONFIRMED"
    assert state.alert_status == AlertStatus.ACTIVE
    assert raised is not None


def test_compliant_refresh_resets_timers_and_clears() -> None:
    sm = PersonComplianceState(
        window_size=30,
        violation_threshold=20,
        clear_threshold=1,
        confirm_seconds=2.0,
        cooldown_seconds=60.0,
    )
    ts = 0.0
    for _ in range(45):
        sm.update(
            person_id=4,
            item="gloves",
            classification=Classification.VIOLATION,
            frame_jpeg=b"frame",
            event_ts=ts,
        )
        ts += 0.1

    clear_change = sm.update(
        person_id=4,
        item="gloves",
        classification=Classification.COMPLIANT,
        frame_jpeg=b"frame",
        event_ts=ts,
    )
    assert clear_change is not None
    assert clear_change.event_type == "ALERT_CLEARED"

    current = sm.get_item_state(person_id=4, item="gloves")
    assert current.alert_status == AlertStatus.CLEARED
    assert current.violation_stage == "CLEAR"
    assert current.candidate_since_ts is None
    assert current.confirmed_since_ts is None


def test_cooldown_blocks_repeat_alert_until_elapsed() -> None:
    sm = PersonComplianceState(
        window_size=30,
        violation_threshold=20,
        clear_threshold=1,
        confirm_seconds=2.0,
        cooldown_seconds=60.0,
    )

    ts = 0.0
    first_raise = None
    for _ in range(45):
        evt = sm.update(
            person_id=3,
            item="boots",
            classification=Classification.VIOLATION,
            frame_jpeg=b"frame",
            event_ts=ts,
        )
        if evt is not None and evt.event_type == "ALERT_RAISED":
            first_raise = evt
        ts += 0.1
    assert first_raise is not None

    # Still violating but within cooldown => no second raised event.
    no_repeat = sm.update(
        person_id=3,
        item="boots",
        classification=Classification.VIOLATION,
        frame_jpeg=b"frame",
        event_ts=30.0,
    )
    assert no_repeat is None

    # After cooldown elapsed, repeated raised event is allowed.
    repeat = sm.update(
        person_id=3,
        item="boots",
        classification=Classification.VIOLATION,
        frame_jpeg=b"frame",
        event_ts=65.0,
    )
    assert repeat is not None
    assert repeat.event_type == "ALERT_RAISED"
