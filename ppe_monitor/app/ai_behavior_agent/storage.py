"""File storage helpers for behavior-agent outputs and memory artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping


class BehaviorAgentStorage:
    """Persist latest/history insight JSON and optional training JSONL."""

    def __init__(self, *, output_dir: str | Path, save_training_records: bool) -> None:
        self.output_dir = Path(output_dir)
        self.latest_path = self.output_dir / "latest_behavior_insight.json"
        self.history_dir = self.output_dir / "history"
        self.training_records_path = self.output_dir / "training_records.jsonl"
        self.save_training_records = bool(save_training_records)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save_insight(self, insight: Mapping[str, Any]) -> Dict[str, str]:
        payload = dict(insight)
        payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        self._write_json(self.latest_path, payload)

        stamp = _timestamp_for_filename(str(payload.get("generated_at", "")))
        history_path = self.history_dir / f"behavior_agent_{stamp}.json"
        self._write_json(history_path, payload)

        if self.save_training_records:
            self._append_training_records(payload)

        return {
            "latest_path": str(self.latest_path),
            "history_path": str(history_path),
        }

    def list_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        files = sorted(self.history_dir.glob("behavior_agent_*.json"), reverse=True)
        out: List[Dict[str, Any]] = []
        for path in files[: max(1, int(limit))]:
            doc = safe_read_json(path, default={})
            if isinstance(doc, dict):
                doc["_path"] = str(path)
                out.append(doc)
        return out

    def _append_training_records(self, insight: Mapping[str, Any]) -> None:
        suggestions = insight.get("training_data_suggestions", [])
        if not isinstance(suggestions, list) or not suggestions:
            return
        row = {
            "generated_at": insight.get("generated_at"),
            "model": insight.get("model"),
            "time_window": insight.get("time_window"),
            "training_data_suggestions": suggestions,
        }
        try:
            with self.training_records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            return

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def safe_read_json(path: str | Path, default: Any) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    try:
        with target.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def safe_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)


def _timestamp_for_filename(iso_ts: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%S%fZ")
