"""Background per-person observation stream for the behavior agent.

Analytics sink only. Must never block or crash the real-time pipeline:
events are queued from process_frame and written by a daemon thread.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

SCHEMA_VERSION = "1.0"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass
class _TrackEmitState:
    last_emit_ts: float = 0.0
    last_raw: Dict[str, str] = field(default_factory=dict)
    last_stable: Dict[str, str] = field(default_factory=dict)
    last_sm: Dict[str, str] = field(default_factory=dict)


class EventStreamWriter:
    """Asynchronous JSONL writer for richer per-person PPE observations."""

    def __init__(self, config: dict) -> None:
        cfg = config.get("event_stream", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.path = Path(cfg.get("path", "outputs/detection_events.jsonl"))
        self.camera_id = str(cfg.get("camera_id", "cam_01"))
        self.heartbeat_seconds = float(cfg.get("heartbeat_seconds", 1.0))
        self.on_raw = bool(cfg.get("emit_on_raw_change", True))
        self.on_stable = bool(cfg.get("emit_on_stable_change", True))
        self.on_sm = bool(cfg.get("emit_on_sm_transition", True))

        # TODO: add retention/rotation (for example date-stamped files) to bound disk growth.
        self._q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=int(cfg.get("queue_max", 2000)))
        self._track: Dict[int, _TrackEmitState] = {}
        self._seq = 0
        self.dropped = 0
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._drain, name="event-stream", daemon=True)
            self._thread.start()

    def emit_person_observation(
        self,
        *,
        frame_id: int,
        track_id: int,
        bbox,
        per_item: Dict[str, dict],
        overall_status: str,
        tracking_confidence: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return

        st = self._track.setdefault(track_id, _TrackEmitState())
        now = time.time()
        raw = {k: v["status_raw"] for k, v in per_item.items()}
        stable = {k: v["status_stable"] for k, v in per_item.items()}
        sm = {k: v["sm_stage"] for k, v in per_item.items()}

        should = (
            (self.on_raw and raw != st.last_raw)
            or (self.on_stable and stable != st.last_stable)
            or (self.on_sm and sm != st.last_sm)
            or (now - st.last_emit_ts) >= self.heartbeat_seconds
        )
        if not should:
            return

        st.last_emit_ts, st.last_raw, st.last_stable, st.last_sm = now, raw, stable, sm
        self._seq += 1
        record = {
            "schema_version": SCHEMA_VERSION,
            "event_id": f"{int(now * 1000)}-{self._seq}",
            "event_type": "ppe_observation",
            "timestamp": _iso(now),
            "camera_id": self.camera_id,
            "frame_id": int(frame_id),
            "track_id": int(track_id),
            "person_memory_id": None,
            "bbox": [float(v) for v in bbox] if bbox is not None else None,
            "ppe": {
                item: {
                    "status": d["status_raw"],
                    "status_stable": d["status_stable"],
                    "positive_conf": round(float(d["positive_conf"]), 4),
                    "negative_conf": round(float(d["negative_conf"]), 4),
                    "reason": d["reason"],
                    "sm_stage": d["sm_stage"],
                    "alert_status": d["alert_status"],
                }
                for item, d in per_item.items()
            },
            "overall_status": overall_status,
            "tracker": {
                "tracking_confidence": tracking_confidence,
                "reid_similarity": None,
                "occlusion": None,
            },
            "zone": None,
            "risk_context": None,
        }
        try:
            self._q.put_nowait(json.dumps(record, default=str))
        except queue.Full:
            self.dropped += 1

    def prune(self, active_track_ids: Iterable[int]) -> None:
        active = set(active_track_ids)
        for track_id in [tid for tid in self._track if tid not in active]:
            self._track.pop(track_id, None)

    def _drain(self) -> None:
        while True:
            line = self._q.get()
            if line is None:
                return
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def close(self) -> None:
        if self._thread is not None:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                self.dropped += 1
