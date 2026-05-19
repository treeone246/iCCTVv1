"""Tests for behavior-agent event reader tail-window parsing."""

import json
from pathlib import Path

from app.ai_behavior_agent.event_reader import read_recent_events


def _write_jsonl(path: Path, rows: list[dict | str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, str):
                f.write(row + "\n")
            else:
                f.write(json.dumps(row) + "\n")


def test_read_recent_events_returns_last_n_valid_records(tmp_path: Path) -> None:
    path = tmp_path / "detection_events.jsonl"
    rows: list[dict | str] = [
        {"event_type": "ppe_observation", "timestamp": "2026-05-19T00:00:00Z", "track_id": 1},
        {"event_type": "frame_processed", "timestamp": "2026-05-19T00:00:01Z", "track_id": 1},
        "this is malformed json",
        {"event_type": "ppe_observation", "timestamp": "2026-05-19T00:00:02Z", "track_id": 2},
        {"event_type": "ppe_observation", "timestamp": "2026-05-19T00:00:03Z", "track_id": 3},
        {"event_type": "ppe_observation", "timestamp": "2026-05-19T00:00:04Z", "track_id": 4},
    ]
    _write_jsonl(path, rows)

    events, window = read_recent_events(path, max_recent_events=3)
    assert [e["track_id"] for e in events] == [2, 3, 4]
    assert window["start"] == "2026-05-19T00:00:02Z"
    assert window["end"] == "2026-05-19T00:00:04Z"
    assert window["event_count"] == 3
