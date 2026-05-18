"""Compliance event writer for confirmed PPE violations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import time

from .ppe_memory import PersonPPEMemory, iso_utc


@dataclass
class ComplianceEventWriter:
    path: str = "outputs/compliance_events.jsonl"

    def write_alert(self, memory: PersonPPEMemory) -> None:
        payload = {
            "timestamp": iso_utc(time()),
            "event_type": "PPE_VIOLATION_CONFIRMED",
            "camera_id": memory.camera_id,
            "track_id": memory.track_id,
            "state": memory.state.value,
            "item_statuses": {k: v.value for k, v in memory.item_statuses().items()},
            "bbox": list(memory.latest_bbox) if memory.latest_bbox is not None else None,
        }
        out = Path(self.path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
