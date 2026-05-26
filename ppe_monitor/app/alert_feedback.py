"""Persistent alert acknowledgment and calibration feedback store."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class AlertFeedbackStore:
    """Tracks acknowledged/unacknowledged alerts for calibration analytics."""

    def __init__(self, *, path: Path, enabled: bool = True, max_recent_records: int = 10000) -> None:
        self.enabled = bool(enabled)
        self.path = Path(path)
        self.max_recent_records = max(100, int(max_recent_records))
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._latest_by_alert_id: dict[str, dict[str, Any]] = {}
        self._active_alerts: dict[str, dict[str, Any]] = {}
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        self._records.append(record)
                        alert_id = str(record.get("alert_id", "")).strip()
                        if alert_id:
                            self._latest_by_alert_id[alert_id] = record
            if len(self._records) > self.max_recent_records:
                self._records = self._records[-self.max_recent_records :]
        except OSError:
            return

    def _append_record(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        if len(self._records) > self.max_recent_records:
            self._records = self._records[-self.max_recent_records :]
        alert_id = str(record.get("alert_id", "")).strip()
        if alert_id:
            self._latest_by_alert_id[alert_id] = record
        if not self.enabled:
            return
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except OSError:
            return

    def observe_active_alerts(
        self,
        *,
        alerts: Iterable[dict[str, Any]],
        camera_id: str,
        observed_ts: Optional[float] = None,
    ) -> None:
        """Track active lifecycle and auto-log unacknowledged clear events."""
        if not self.enabled:
            return
        ts = float(observed_ts if observed_ts is not None else time.time())
        cam = str(camera_id or "cam_01")
        current_ids: set[str] = set()
        with self._lock:
            for alert in alerts:
                alert_id = str(alert.get("alert_id", "")).strip()
                if not alert_id:
                    continue
                current_ids.add(alert_id)
                entry = self._active_alerts.get(alert_id)
                if entry is None:
                    self._active_alerts[alert_id] = {
                        "alert_id": alert_id,
                        "person_id": int(alert.get("person_id", -1)),
                        "display_id": str(alert.get("display_id", "")),
                        "item": str(alert.get("item", "")),
                        "camera_id": cam,
                        "first_seen_ts": ts,
                        "last_seen_ts": ts,
                    }
                else:
                    entry["last_seen_ts"] = ts
            cleared_ids = [aid for aid in list(self._active_alerts.keys()) if aid not in current_ids]
            for alert_id in cleared_ids:
                entry = self._active_alerts.pop(alert_id, None)
                if entry is None:
                    continue
                latest = self._latest_by_alert_id.get(alert_id, {})
                if bool(latest.get("acknowledged", False)):
                    continue
                first_seen = float(entry.get("first_seen_ts", ts))
                duration = max(0.0, ts - first_seen)
                record = {
                    "event_type": "alert_feedback_auto_closed",
                    "alert_id": alert_id,
                    "person_id": int(entry.get("person_id", -1)),
                    "display_id": str(entry.get("display_id", "")),
                    "item": str(entry.get("item", "")),
                    "camera_id": str(entry.get("camera_id", cam)),
                    "acknowledged": False,
                    "reason": "auto_closed_without_ack",
                    "observed_duration_sec": round(duration, 3),
                    "timestamp": ts,
                }
                self._append_record(record)

    def acknowledge(
        self,
        *,
        alert_id: str,
        person_id: int,
        item: str,
        camera_id: str,
        display_id: str = "",
        note: str = "",
        acknowledged: bool = True,
        positive_conf: float = 0.0,
        negative_conf: float = 0.0,
    ) -> dict[str, Any]:
        ts = time.time()
        with self._lock:
            active = self._active_alerts.get(str(alert_id), {})
            first_seen = float(active.get("first_seen_ts", ts))
            record = {
                "event_type": "alert_feedback",
                "alert_id": str(alert_id),
                "person_id": int(person_id),
                "display_id": str(display_id),
                "item": str(item),
                "camera_id": str(camera_id),
                "acknowledged": bool(acknowledged),
                "reason": str(note or ""),
                "positive_conf": float(positive_conf),
                "negative_conf": float(negative_conf),
                "time_to_decision_sec": round(max(0.0, ts - first_seen), 3),
                "timestamp": ts,
            }
            self._append_record(record)
            return record

    def is_acknowledged(self, alert_id: str) -> bool:
        with self._lock:
            latest = self._latest_by_alert_id.get(str(alert_id), {})
            return bool(latest.get("acknowledged", False))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        total = len(records)
        ack_count = sum(1 for rec in records if bool(rec.get("acknowledged", False)))
        unack_count = max(0, total - ack_count)
        time_to_decision_ack = [
            float(rec.get("time_to_decision_sec", 0.0))
            for rec in records
            if bool(rec.get("acknowledged", False))
        ]
        time_to_decision_unack = [
            float(rec.get("time_to_decision_sec", rec.get("observed_duration_sec", 0.0)))
            for rec in records
            if not bool(rec.get("acknowledged", False))
        ]

        per_item: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "acknowledged": 0, "not_acknowledged": 0}
        )
        for rec in records:
            item = str(rec.get("item", "unknown")).lower()
            per_item[item]["total"] += 1
            if bool(rec.get("acknowledged", False)):
                per_item[item]["acknowledged"] += 1
            else:
                per_item[item]["not_acknowledged"] += 1

        by_item_out: dict[str, dict[str, Any]] = {}
        for item, row in per_item.items():
            total_item = int(row["total"])
            ack_item = int(row["acknowledged"])
            by_item_out[item] = {
                "total": total_item,
                "acknowledged": ack_item,
                "not_acknowledged": int(row["not_acknowledged"]),
                "ack_rate_pct": round((ack_item * 100.0 / total_item), 2) if total_item > 0 else 0.0,
            }

        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "total_feedback": total,
            "acknowledged": ack_count,
            "not_acknowledged": unack_count,
            "ack_rate_pct": round((ack_count * 100.0 / total), 2) if total > 0 else 0.0,
            "mean_time_to_ack_sec": round(
                sum(time_to_decision_ack) / len(time_to_decision_ack), 3
            )
            if time_to_decision_ack
            else 0.0,
            "mean_time_to_not_ack_sec": round(
                sum(time_to_decision_unack) / len(time_to_decision_unack), 3
            )
            if time_to_decision_unack
            else 0.0,
            "by_item": by_item_out,
        }

