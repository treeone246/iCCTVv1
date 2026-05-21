"""Periodic runtime performance logger for offline analysis."""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .jetson_exporter_bridge import JetsonSnapshot

SCHEMA_VERSION = "1.0"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class PerformanceLogWriter:
    """Asynchronous JSONL writer for periodic compute/performance snapshots."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = dict(config.get("performance_logging", {}) or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.path = Path(str(cfg.get("path", "outputs/performance_logs.jsonl")))
        self.interval_seconds = max(0.1, float(cfg.get("interval_seconds", 1.0)))
        self.queue_max = max(100, int(cfg.get("queue_max", 5000)))
        self.include_jetson = bool(cfg.get("include_jetson", True))
        self.include_raw_metrics = bool(cfg.get("include_raw_metrics", True))
        self.camera_id = str(cfg.get("camera_id", "cam_01"))

        self._q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=self.queue_max)
        self._thread: Optional[threading.Thread] = None
        self._last_emit_ts = 0.0
        self._seq = 0
        self.dropped = 0

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._drain, name="performance-logger", daemon=True)
            self._thread.start()

    def emit(
        self,
        *,
        frame_id: int,
        metrics: Mapping[str, Any],
        jetson: Optional[JetsonSnapshot] = None,
        source: str = "pipeline",
        timestamp: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        now = float(timestamp) if timestamp is not None else time.time()
        if (now - self._last_emit_ts) < self.interval_seconds:
            return
        self._last_emit_ts = now
        self._seq += 1

        record = {
            "schema_version": SCHEMA_VERSION,
            "event_type": "performance_snapshot",
            "event_id": f"{int(now * 1000)}-{self._seq}",
            "timestamp": _iso(now),
            "timestamp_epoch": now,
            "frame_id": int(frame_id),
            "source": str(source),
            "camera_id": self.camera_id,
            "summary": {
                "fps": float(metrics.get("fps", 0.0)),
                "tracked_count": int(metrics.get("tracked_count", 0)),
                "active_violations": int(metrics.get("active_violations", 0)),
                "dropped_frames": int(metrics.get("dropped_frames", 0)),
                "compliance_rate_pct": float(metrics.get("compliance_rate", 0.0)),
            },
            "compute": {
                "flops_per_sec": float(metrics.get("estimated_flops_per_sec", 0.0)),
                "gflops_per_sec": float(metrics.get("estimated_gflops_per_sec", 0.0)),
                "tflops_per_sec": float(metrics.get("estimated_tflops_per_sec", 0.0)),
                "tops_per_sec": float(metrics.get("estimated_tops_per_sec", 0.0)),
                "utilization_pct": float(metrics.get("estimated_compute_utilization_pct", 0.0)),
            },
            "model_rates": {
                "pose_infer_per_sec": float(metrics.get("pose_infer_per_sec", 0.0)),
                "ppe_infer_per_sec": float(metrics.get("ppe_infer_per_sec", 0.0)),
                "verifier_aux_infer_per_sec": float(metrics.get("verifier_aux_infer_per_sec", 0.0)),
                "verifier_crop_infer_per_sec": float(metrics.get("verifier_crop_infer_per_sec", 0.0)),
                "verifier_ollama_calls_per_sec": float(metrics.get("verifier_ollama_calls_per_sec", 0.0)),
            },
            "memory": {
                "process_rss_mb": float(metrics.get("process_rss_mb", 0.0)),
                "process_vms_mb": float(metrics.get("process_vms_mb", 0.0)),
                "system_memory_used_mb": float(metrics.get("system_memory_used_mb", 0.0)),
                "system_memory_total_mb": float(metrics.get("system_memory_total_mb", 0.0)),
                "system_memory_utilization_pct": float(metrics.get("system_memory_utilization_pct", 0.0)),
            },
        }
        if self.include_jetson:
            record["jetson"] = {
                "enabled": bool(jetson.enabled) if jetson is not None else False,
                "available": bool(jetson.available) if jetson is not None else False,
                "source_url": str(jetson.source_url) if jetson is not None else "",
                "error": str(jetson.error) if jetson is not None else "",
                "cpu_utilization_pct": float(jetson.cpu_utilization_pct) if jetson is not None else 0.0,
                "gpu_utilization_pct": float(jetson.gpu_utilization_pct) if jetson is not None else 0.0,
                "memory_utilization_pct": float(jetson.memory_utilization_pct) if jetson is not None else 0.0,
                "memory_used_mb": float(jetson.memory_used_mb) if jetson is not None else 0.0,
                "memory_total_mb": float(jetson.memory_total_mb) if jetson is not None else 0.0,
                "temperature_c": float(jetson.temperature_c) if jetson is not None else 0.0,
                "power_w": float(jetson.power_w) if jetson is not None else 0.0,
                "fan_pwm_pct": float(jetson.fan_pwm_pct) if jetson is not None else 0.0,
            }
        if self.include_raw_metrics:
            record["metrics"] = dict(metrics)

        try:
            self._q.put_nowait(json.dumps(record, default=str))
        except queue.Full:
            self.dropped += 1

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
            self._thread.join(timeout=2.0)
