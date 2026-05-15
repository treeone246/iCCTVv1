"""Per-person per-item compliance state with alert hysteresis rules."""

from collections import deque
from dataclasses import dataclass, field
from time import time
from typing import Deque, Dict, Optional

from .schemas import AlertStatus, Classification


@dataclass
class ItemState:
    """Rolling classification state for one PPE item."""

    recent: Deque[Classification]
    alert_status: AlertStatus = AlertStatus.CLEARED
    last_transition_ts: float = 0.0
    evidence_jpeg: Optional[bytes] = None
    compliant_streak: int = 0


@dataclass
class StateChange:
    """A state transition event emitted by the state machine."""

    event_type: str
    person_id: int
    item: str
    alert_status: AlertStatus
    timestamp: float
    evidence_jpeg: Optional[bytes] = None


class PersonComplianceState:
    """Tracks and updates compliance alert state per person and PPE item."""

    def __init__(
        self,
        window_size: int = 8,
        violation_threshold: int = 5,
        clear_threshold: int = 3,
    ) -> None:
        self.window_size = window_size
        self.violation_threshold = violation_threshold
        self.clear_threshold = clear_threshold
        self._state: Dict[int, Dict[str, ItemState]] = {}

    def _ensure_item_state(self, person_id: int, item: str) -> ItemState:
        person_items = self._state.setdefault(person_id, {})
        if item not in person_items:
            person_items[item] = ItemState(recent=deque(maxlen=self.window_size))
        return person_items[item]

    def get_item_state(self, person_id: int, item: str) -> ItemState:
        return self._ensure_item_state(person_id, item)

    def active_alerts_count(self) -> int:
        count = 0
        for items in self._state.values():
            for item_state in items.values():
                if item_state.alert_status == AlertStatus.ACTIVE:
                    count += 1
        return count

    def iter_active_alerts(self) -> list[tuple[int, str, ItemState]]:
        active: list[tuple[int, str, ItemState]] = []
        for person_id, items in self._state.items():
            for item, item_state in items.items():
                if item_state.alert_status == AlertStatus.ACTIVE:
                    active.append((person_id, item, item_state))
        return active

    def update(
        self,
        person_id: int,
        item: str,
        classification: Classification,
        frame_jpeg: Optional[bytes],
        event_ts: Optional[float] = None,
    ) -> Optional[StateChange]:
        state = self._ensure_item_state(person_id, item)
        ts = event_ts if event_ts is not None else time()

        if classification != Classification.INDETERMINATE:
            state.recent.append(classification)

        if classification == Classification.COMPLIANT:
            state.compliant_streak += 1
        elif classification in (Classification.VIOLATION, Classification.VIOLATION_TENTATIVE):
            state.compliant_streak = 0

        if (
            state.alert_status != AlertStatus.ACTIVE
            and len(state.recent) >= self.window_size
            and sum(1 for c in state.recent if c == Classification.VIOLATION) >= self.violation_threshold
        ):
            state.alert_status = AlertStatus.ACTIVE
            state.last_transition_ts = ts
            state.evidence_jpeg = frame_jpeg
            return StateChange(
                event_type="ALERT_RAISED",
                person_id=person_id,
                item=item,
                alert_status=state.alert_status,
                timestamp=ts,
                evidence_jpeg=frame_jpeg,
            )

        if (
            state.alert_status == AlertStatus.ACTIVE
            and state.compliant_streak >= self.clear_threshold
        ):
            state.alert_status = AlertStatus.CLEARED
            state.last_transition_ts = ts
            return StateChange(
                event_type="ALERT_CLEARED",
                person_id=person_id,
                item=item,
                alert_status=state.alert_status,
                timestamp=ts,
                evidence_jpeg=None,
            )

        return None
