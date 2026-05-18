"""Simple PPE compliance memory and anti-spam state machine.

This module intentionally stays simple and explainable:
- vote window per PPE item
- per-track temporal state transitions
- cooldown for duplicate alerts

No long-term identity across sessions/cameras is implemented.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from time import time
from typing import Deque, Dict, Optional, Tuple


class ItemStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    OK = "OK"
    VIOLATION = "VIOLATION"
    UNCLEAR = "UNCLEAR"


class PersonState(str, Enum):
    UNKNOWN = "UNKNOWN"
    COMPLIANT_CANDIDATE = "COMPLIANT_CANDIDATE"
    COMPLIANT_CONFIRMED = "COMPLIANT_CONFIRMED"
    VIOLATION_CANDIDATE = "VIOLATION_CANDIDATE"
    VIOLATION_CONFIRMED = "VIOLATION_CONFIRMED"


@dataclass
class PPEMemoryConfig:
    vote_window_frames: int = 30
    min_frames_for_decision: int = 15
    ok_ratio_threshold: float = 0.75
    violation_ratio_threshold: float = 0.25
    violation_confirm_sec: float = 2.0
    compliant_confirm_sec: float = 1.0
    alert_cooldown_sec: float = 60.0
    track_timeout_sec: float = 5.0


@dataclass
class PPEItemMemory:
    votes: Deque[Optional[bool]]

    @classmethod
    def create(cls, vote_window_frames: int) -> "PPEItemMemory":
        return cls(votes=deque(maxlen=vote_window_frames))

    def update(self, detected: Optional[bool]) -> None:
        self.votes.append(detected)

    def valid_votes(self) -> int:
        return sum(1 for v in self.votes if v is not None)

    def ok_ratio(self) -> float:
        valid = [v for v in self.votes if v is not None]
        if not valid:
            return 0.0
        ok = sum(1 for v in valid if v)
        return ok / float(len(valid))

    def decide(self, config: PPEMemoryConfig) -> ItemStatus:
        valid = self.valid_votes()
        if valid < config.min_frames_for_decision:
            return ItemStatus.UNKNOWN
        ratio = self.ok_ratio()
        if ratio >= config.ok_ratio_threshold:
            return ItemStatus.OK
        if ratio <= config.violation_ratio_threshold:
            return ItemStatus.VIOLATION
        return ItemStatus.UNCLEAR


@dataclass
class PersonPPEMemory:
    track_id: int
    camera_id: str
    config: PPEMemoryConfig
    first_seen: float = field(default_factory=time)
    last_seen: float = field(default_factory=time)
    latest_bbox: Optional[Tuple[float, float, float, float]] = None
    state: PersonState = PersonState.UNKNOWN
    last_stable_state: PersonState = PersonState.UNKNOWN
    candidate_since: Optional[float] = None
    last_alert_time: Optional[float] = None
    helmet: PPEItemMemory = field(init=False)
    coverall: PPEItemMemory = field(init=False)
    gloves: PPEItemMemory = field(init=False)
    safety_glasses: PPEItemMemory = field(init=False)
    boots: PPEItemMemory = field(init=False)

    def __post_init__(self) -> None:
        self.helmet = PPEItemMemory.create(self.config.vote_window_frames)
        self.coverall = PPEItemMemory.create(self.config.vote_window_frames)
        self.gloves = PPEItemMemory.create(self.config.vote_window_frames)
        self.safety_glasses = PPEItemMemory.create(self.config.vote_window_frames)
        self.boots = PPEItemMemory.create(self.config.vote_window_frames)

    def update(self, observations: Dict[str, Optional[bool]], bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        now = time()
        self.last_seen = now
        if bbox is not None:
            self.latest_bbox = tuple(float(v) for v in bbox)

        self.helmet.update(observations.get("helmet"))
        self.coverall.update(observations.get("coverall"))
        self.gloves.update(observations.get("gloves"))
        self.safety_glasses.update(observations.get("safety_glasses"))
        self.boots.update(observations.get("boots"))

        self._update_state(now)

    def item_statuses(self) -> Dict[str, ItemStatus]:
        return {
            "helmet": self.helmet.decide(self.config),
            "coverall": self.coverall.decide(self.config),
            "gloves": self.gloves.decide(self.config),
            "safety_glasses": self.safety_glasses.decide(self.config),
            "boots": self.boots.decide(self.config),
        }

    def _update_state(self, now: float) -> None:
        statuses = self.item_statuses()
        values = list(statuses.values())
        all_ok = all(v == ItemStatus.OK for v in values)
        any_violation = any(v == ItemStatus.VIOLATION for v in values)

        if all_ok:
            if self.state not in (PersonState.COMPLIANT_CANDIDATE, PersonState.COMPLIANT_CONFIRMED):
                self.state = PersonState.COMPLIANT_CANDIDATE
                self.candidate_since = now
            elif self.state == PersonState.COMPLIANT_CANDIDATE and self.candidate_since is not None and (now - self.candidate_since) >= self.config.compliant_confirm_sec:
                self.state = PersonState.COMPLIANT_CONFIRMED
                self.last_stable_state = PersonState.COMPLIANT_CONFIRMED
            return

        if any_violation:
            if self.state not in (PersonState.VIOLATION_CANDIDATE, PersonState.VIOLATION_CONFIRMED):
                self.state = PersonState.VIOLATION_CANDIDATE
                self.candidate_since = now
            elif self.state == PersonState.VIOLATION_CANDIDATE and self.candidate_since is not None and (now - self.candidate_since) >= self.config.violation_confirm_sec:
                self.state = PersonState.VIOLATION_CONFIRMED
                self.last_stable_state = PersonState.VIOLATION_CONFIRMED
            return

        # Mixed/unclear evidence: hold stable state instead of flickering.
        if self.last_stable_state in (PersonState.COMPLIANT_CONFIRMED, PersonState.VIOLATION_CONFIRMED):
            self.state = self.last_stable_state
        else:
            self.state = PersonState.UNKNOWN
            self.candidate_since = None

    def should_emit_alert(self) -> bool:
        now = time()
        if self.state != PersonState.VIOLATION_CONFIRMED:
            return False
        if self.last_alert_time is None or (now - self.last_alert_time) >= self.config.alert_cooldown_sec:
            self.last_alert_time = now
            return True
        return False

    def to_dict(self) -> Dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "state": self.state.value,
            "last_stable_state": self.last_stable_state.value,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "latest_bbox": list(self.latest_bbox) if self.latest_bbox is not None else None,
            "item_statuses": {k: v.value for k, v in self.item_statuses().items()},
        }


class PPEMemoryManager:
    def __init__(self, config: Optional[PPEMemoryConfig] = None) -> None:
        self.config = config or PPEMemoryConfig()
        self._mem: Dict[Tuple[str, int], PersonPPEMemory] = {}

    def get_or_create(self, camera_id: str, track_id: int) -> PersonPPEMemory:
        key = (str(camera_id), int(track_id))
        existing = self._mem.get(key)
        if existing is not None:
            return existing
        created = PersonPPEMemory(track_id=int(track_id), camera_id=str(camera_id), config=self.config)
        self._mem[key] = created
        return created

    def update_track(
        self,
        camera_id: str,
        track_id: int,
        observations: Dict[str, Optional[bool]],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> PersonPPEMemory:
        memory = self.get_or_create(camera_id, track_id)
        memory.update(observations=observations, bbox=bbox)
        return memory

    def cleanup(self) -> None:
        now = time()
        stale = [
            key
            for key, memory in self._mem.items()
            if (now - memory.last_seen) > self.config.track_timeout_sec
        ]
        for key in stale:
            self._mem.pop(key, None)

    def active_states(self) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for (camera_id, track_id), memory in self._mem.items():
            out[f"{camera_id}:{track_id}"] = memory.to_dict()
        return out


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def box_inside_person(det_box: Tuple[float, float, float, float], person_box: Tuple[float, float, float, float]) -> bool:
    cx, cy = box_center(det_box)
    return person_box[0] <= cx <= person_box[2] and person_box[1] <= cy <= person_box[3]


def build_ppe_observations_for_person(
    person_box: Tuple[float, float, float, float],
    detections: list,
    names: Optional[Dict[int, str]] = None,
) -> Dict[str, Optional[bool]]:
    """Build first-version PPE observations for one person box.

    True  -> class detected inside person bbox
    False -> class not detected
    None  -> person box is invalid/small, cannot evaluate
    """
    _ = names  # placeholder for future compatibility with raw class-id pipelines

    x1, y1, x2, y2 = person_box
    if x2 <= x1 or y2 <= y1 or ((x2 - x1) * (y2 - y1)) < 400.0:
        return {
            "helmet": None,
            "coverall": None,
            "gloves": None,
            "safety_glasses": None,
            "boots": None,
        }

    found = {
        "helmet": False,
        "coverall": False,
        "gloves": False,
        "safety_glasses": False,
        "boots": False,
    }

    for det in detections:
        try:
            label = str(det.label).strip().lower()
            det_box = (float(det.x1), float(det.y1), float(det.x2), float(det.y2))
        except Exception:
            continue
        if not box_inside_person(det_box, person_box):
            continue

        if label in {"helmet", "hard hat", "hardhat"}:
            found["helmet"] = True
        elif label in {"coverall", "industrial coverall jumpsuit", "overalls"}:
            found["coverall"] = True
        elif label in {"gloves", "work glove safety glove", "glove"}:
            found["gloves"] = True
        elif label in {
            "goggles",
            "safety_glasses",
            "safety glasses or goggles protective eyewear",
            "safety glasses",
            "glasses",
        }:
            found["safety_glasses"] = True
        elif label in {"boots", "safety boot work boot", "safety_boots", "boot"}:
            found["boots"] = True

    return found


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
