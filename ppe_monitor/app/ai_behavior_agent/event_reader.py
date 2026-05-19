"""Utilities to read recent Phase-1 detection event stream records."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_recent_events(events_jsonl: str | Path, max_recent_events: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read last N valid `ppe_observation` events from JSONL stream.

    Malformed JSON lines are ignored by design to keep the agent resilient.
    """

    path = Path(events_jsonl)
    if max_recent_events <= 0:
        return [], _window_from_events([])
    if not path.exists() or not path.is_file():
        return [], _window_from_events([])

    recent_lines: deque[str] = deque(maxlen=int(max_recent_events))
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                recent_lines.append(line)
    except OSError:
        return [], _window_from_events([])

    events: List[Dict[str, Any]] = []
    for line in recent_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if str(record.get("event_type", "")) != "ppe_observation":
            continue
        events.append(record)

    return events, _window_from_events(events)


def _window_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {"start": None, "end": None, "event_count": 0}
    first = events[0]
    last = events[-1]
    return {
        "start": first.get("timestamp"),
        "end": last.get("timestamp"),
        "event_count": len(events),
    }
