"""Unit tests for per-person alert hysteresis state transitions."""

from app.schemas import AlertStatus, Classification
from app.state_machine import PersonComplianceState


def test_raise_alert_on_five_of_eight_violations() -> None:
    sm = PersonComplianceState(window_size=8, violation_threshold=5, clear_threshold=3)
    sequence = [
        Classification.VIOLATION,
        Classification.COMPLIANT,
        Classification.VIOLATION,
        Classification.VIOLATION,
        Classification.COMPLIANT,
        Classification.VIOLATION,
        Classification.VIOLATION,
        Classification.COMPLIANT,
    ]

    change = None
    for cls in sequence:
        change = sm.update(person_id=11, item="helmet", classification=cls, frame_jpeg=b"evidence")

    assert change is not None
    assert change.event_type == "ALERT_RAISED"
    assert change.alert_status == AlertStatus.ACTIVE


def test_clear_alert_after_three_consecutive_compliant() -> None:
    sm = PersonComplianceState(window_size=8, violation_threshold=5, clear_threshold=3)

    for cls in [
        Classification.VIOLATION,
        Classification.VIOLATION,
        Classification.VIOLATION,
        Classification.VIOLATION,
        Classification.VIOLATION,
        Classification.COMPLIANT,
        Classification.COMPLIANT,
        Classification.COMPLIANT,
        Classification.COMPLIANT,
        Classification.COMPLIANT,
        Classification.COMPLIANT,
    ]:
        sm.update(person_id=4, item="gloves", classification=cls, frame_jpeg=b"frame")

    current = sm.get_item_state(person_id=4, item="gloves")
    assert current.alert_status == AlertStatus.CLEARED


def test_indeterminate_does_not_shift_counts() -> None:
    sm = PersonComplianceState(window_size=8, violation_threshold=5, clear_threshold=3)

    for cls in [
        Classification.VIOLATION,
        Classification.VIOLATION,
        Classification.INDETERMINATE,
        Classification.VIOLATION,
        Classification.VIOLATION,
        Classification.INDETERMINATE,
        Classification.VIOLATION,
        Classification.COMPLIANT,
        Classification.COMPLIANT,
        Classification.COMPLIANT,
    ]:
        sm.update(person_id=3, item="boots", classification=cls, frame_jpeg=b"frame")

    state = sm.get_item_state(person_id=3, item="boots")
    # Two indeterminate frames should not evict violations from the rolling window.
    assert sum(1 for c in state.recent if c == Classification.VIOLATION) >= 5
