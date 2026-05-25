"""Per-person per-item compliance state with alert hysteresis rules."""

from collections import deque
from dataclasses import dataclass, field
from time import time
from typing import Callable, Deque, Dict, Optional

from .schemas import AlertStatus, Classification


@dataclass
class ItemState:
    """Rolling classification state for one PPE item."""

    recent: Deque[Classification]
    alert_status: AlertStatus = AlertStatus.CLEARED
    last_transition_ts: float = 0.0
    evidence_jpeg: Optional[bytes] = None
    compliant_streak: int = 0
    candidate_since_ts: Optional[float] = None
    confirmed_since_ts: Optional[float] = None
    last_alert_sent_ts: Optional[float] = None
    violation_stage: str = "CLEAR"


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
        window_size: int = 30,
        violation_threshold: int = 20,
        clear_threshold: int = 1,
        confirm_seconds: float = 2.0,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.window_size = max(1, int(window_size))
        # Guard against invalid configs (for example threshold > window)
        # that would otherwise prevent alerts from ever activating.
        self.violation_threshold = max(1, min(int(violation_threshold), self.window_size))
        self.clear_threshold = clear_threshold
        self.confirm_seconds = confirm_seconds
        self.cooldown_seconds = cooldown_seconds
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

    def get_violation_stage(self, person_id: int, item: str) -> str:
        return self._ensure_item_state(person_id, item).violation_stage

    def update(
        self,
        person_id: int,
        item: str,
        classification: Classification,
        frame_jpeg: Optional[bytes],
        frame_jpeg_provider: Optional[Callable[[], Optional[bytes]]] = None,
        event_ts: Optional[float] = None,
    ) -> Optional[StateChange]:
        state = self._ensure_item_state(person_id, item)
        ts = event_ts if event_ts is not None else time()
        violation_like = {Classification.VIOLATION, Classification.VIOLATION_TENTATIVE}
        resolved_frame_jpeg = False
        cached_frame_jpeg: Optional[bytes] = None

        def _resolve_frame_jpeg() -> Optional[bytes]:
            nonlocal resolved_frame_jpeg, cached_frame_jpeg
            if resolved_frame_jpeg:
                return cached_frame_jpeg
            if frame_jpeg is not None:
                cached_frame_jpeg = frame_jpeg
                resolved_frame_jpeg = True
                return cached_frame_jpeg
            if frame_jpeg_provider is None:
                resolved_frame_jpeg = True
                return None
            cached_frame_jpeg = frame_jpeg_provider()
            resolved_frame_jpeg = True
            return cached_frame_jpeg

        state.recent.append(classification)

        if classification == Classification.COMPLIANT:
            state.compliant_streak += 1
        elif classification in violation_like:
            state.compliant_streak = 0
        else:
            state.compliant_streak = 0

        # Compliance immediately refreshes/reset timers and clears active alerts.
        if classification == Classification.COMPLIANT:
            state.candidate_since_ts = None
            state.confirmed_since_ts = None
            state.violation_stage = "CLEAR"
            if state.alert_status == AlertStatus.ACTIVE and state.compliant_streak >= self.clear_threshold:
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

        violation_votes = sum(1 for c in state.recent if c in violation_like)
        if violation_votes >= self.violation_threshold:
            if state.candidate_since_ts is None:
                state.candidate_since_ts = ts
            state.violation_stage = "VIOLATION_CANDIDATE"

            if (ts - state.candidate_since_ts) >= self.confirm_seconds:
                state.confirmed_since_ts = state.confirmed_since_ts or ts
                state.violation_stage = "VIOLATION_CONFIRMED"
                if state.alert_status != AlertStatus.ACTIVE:
                    state.alert_status = AlertStatus.ACTIVE
                    state.last_transition_ts = ts
                    state.evidence_jpeg = _resolve_frame_jpeg()

                can_send = (
                    state.last_alert_sent_ts is None
                    or (ts - state.last_alert_sent_ts) >= self.cooldown_seconds
                )
                if can_send:
                    evidence = _resolve_frame_jpeg()
                    state.last_alert_sent_ts = ts
                    state.last_transition_ts = ts
                    state.evidence_jpeg = evidence
                    return StateChange(
                        event_type="ALERT_RAISED",
                        person_id=person_id,
                        item=item,
                        alert_status=state.alert_status,
                        timestamp=ts,
                        evidence_jpeg=evidence,
                    )
        else:
            # Not enough recent violation votes: keep clear unless still confirmed.
            if state.alert_status != AlertStatus.ACTIVE:
                state.violation_stage = "CLEAR"
                state.candidate_since_ts = None
                state.confirmed_since_ts = None

        return None
