"""Filtered, queue-based violation ingest path with optional Kafka fan-out."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, List

from .violation_kafka import ViolationKafkaProducer


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_status_upper(value: Any, default: str = "ACTIVE") -> str:
    if value is None:
        return str(default).strip().upper()
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value).strip().upper()
    raw = str(value).strip()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw.upper()


class ViolationAlertFilter:
    """Suppress noisy repeated alerts before they reach durable storage."""

    def __init__(self, *, config: dict) -> None:
        ingest_cfg = config.get("violation_ingest", {}) or {}
        filter_cfg = ingest_cfg.get("filter", {}) or {}
        self.enabled = bool(ingest_cfg.get("enabled", True)) and bool(filter_cfg.get("enabled", True))
        self.only_active = bool(filter_cfg.get("only_active", True))
        self.repeat_suppression_seconds = max(0.0, float(filter_cfg.get("repeat_suppression_seconds", 60.0)))
        self.remember_ttl_seconds = max(60.0, float(filter_cfg.get("remember_ttl_seconds", 600.0)))
        self.max_events_per_minute_per_key = max(1, int(filter_cfg.get("max_events_per_minute_per_key", 2)))
        self.default_min_negative_conf = float(filter_cfg.get("min_negative_confidence_default", 0.0))
        per_item_cfg = filter_cfg.get("min_negative_confidence_per_item", {}) or {}
        self.min_negative_conf_per_item = {str(k).lower(): float(v) for k, v in per_item_cfg.items()}
        self.allow_reason_change_bypass = bool(filter_cfg.get("allow_reason_change_bypass", True))
        self._last_emit_by_key: dict[str, float] = {}
        self._last_reason_by_key: dict[str, str] = {}
        self._minute_windows: dict[str, Deque[float]] = {}
        self._last_gc_ts = 0.0
        self._stats = defaultdict(int)

    def _key(self, *, camera_id: str, alert: dict[str, Any]) -> str:
        item = str(alert.get("item", "unknown")).strip().lower()
        status = _to_status_upper(alert.get("status", "ACTIVE"))
        display_id = str(alert.get("display_id", "")).strip()
        person_id = int(alert.get("person_id", -1))
        who = display_id if display_id else f"person_{person_id}"
        return f"{camera_id}:{who}:{item}:{status}"

    def _min_neg_conf_for_item(self, item: str) -> float:
        return float(self.min_negative_conf_per_item.get(item.lower(), self.default_min_negative_conf))

    def _garbage_collect(self, now_ts: float) -> None:
        if now_ts - self._last_gc_ts < 10.0:
            return
        self._last_gc_ts = now_ts
        cutoff = now_ts - self.remember_ttl_seconds
        dead_keys = [key for key, ts in self._last_emit_by_key.items() if ts < cutoff]
        for key in dead_keys:
            self._last_emit_by_key.pop(key, None)
            self._last_reason_by_key.pop(key, None)
            self._minute_windows.pop(key, None)

    def filter(self, *, alerts: Iterable[dict[str, Any]], camera_id: str) -> list[dict[str, Any]]:
        if not self.enabled:
            out: list[dict[str, Any]] = []
            for alert in alerts:
                copied = dict(alert)
                copied["ingest_camera_id"] = str(camera_id)
                copied["ingest_ts"] = float(time.time())
                out.append(copied)
            return out

        now_ts = float(time.time())
        self._garbage_collect(now_ts)
        accepted: list[dict[str, Any]] = []

        for raw in alerts:
            self._stats["seen"] += 1
            alert = dict(raw)
            status = _to_status_upper(alert.get("status", "ACTIVE"))
            if self.only_active and status != "ACTIVE":
                self._stats["suppressed_non_active"] += 1
                continue
            item = str(alert.get("item", "unknown")).strip().lower()
            raw_negative_conf = alert.get("negative_conf")
            has_negative_conf = raw_negative_conf is not None
            negative_conf = _to_float(raw_negative_conf, 0.0)
            min_neg_conf = self._min_neg_conf_for_item(item)
            # If negative confidence is absent from upstream payload, do not suppress only on confidence gate.
            if has_negative_conf and negative_conf < min_neg_conf:
                self._stats["suppressed_low_conf"] += 1
                continue

            key = self._key(camera_id=camera_id, alert=alert)
            reason = str(alert.get("reason", "")).strip().lower()
            last_reason = self._last_reason_by_key.get(key, "")
            last_emit = self._last_emit_by_key.get(key)
            if (
                last_emit is not None
                and self.repeat_suppression_seconds > 0.0
                and (now_ts - last_emit) < self.repeat_suppression_seconds
            ):
                if not (self.allow_reason_change_bypass and reason and reason != last_reason):
                    self._stats["suppressed_repeat"] += 1
                    continue

            window = self._minute_windows.setdefault(key, deque())
            minute_cutoff = now_ts - 60.0
            while window and window[0] < minute_cutoff:
                window.popleft()
            if len(window) >= self.max_events_per_minute_per_key:
                self._stats["suppressed_rate_limit"] += 1
                continue

            window.append(now_ts)
            self._last_emit_by_key[key] = now_ts
            self._last_reason_by_key[key] = reason
            alert["ingest_camera_id"] = str(camera_id)
            alert["ingest_ts"] = now_ts
            accepted.append(alert)
            self._stats["accepted"] += 1
        return accepted

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "only_active": bool(self.only_active),
            "repeat_suppression_seconds": float(self.repeat_suppression_seconds),
            "max_events_per_minute_per_key": int(self.max_events_per_minute_per_key),
            "default_min_negative_conf": float(self.default_min_negative_conf),
            "min_negative_conf_per_item": dict(self.min_negative_conf_per_item),
            "seen_keys": int(len(self._last_emit_by_key)),
            "stats": {k: int(v) for k, v in self._stats.items()},
        }


class ViolationIngestManager:
    """Queue + filter + optional Kafka ingress path for violation alerts."""

    def __init__(self, *, config: dict, pg_logger) -> None:
        ingest_cfg = config.get("violation_ingest", {}) or {}
        queue_cfg = ingest_cfg.get("queue", {}) or {}
        self.enabled = bool(ingest_cfg.get("enabled", True))
        self.mode = str(ingest_cfg.get("mode", "direct")).strip().lower()
        self.pg_logger = pg_logger
        self.filter = ViolationAlertFilter(config=config)
        self.kafka = ViolationKafkaProducer(config=config)
        # direct: local filter/queue -> PostgreSQL
        # kafka: publish to Kafka and optionally local PostgreSQL
        # kafka_flink: publish to Kafka only; Flink job handles sinking/curation
        self.local_sink_enabled = (
            self.mode == "direct"
            or (self.mode == "kafka" and self.kafka.also_write_local)
        )
        self.queue_enabled = bool(queue_cfg.get("enabled", True))
        self.batch_size = max(1, int(queue_cfg.get("batch_size", 64)))
        self.flush_interval_seconds = max(0.05, float(queue_cfg.get("flush_interval_seconds", 1.0)))
        self.drop_when_full = bool(queue_cfg.get("drop_when_full", True))
        self.max_size = max(128, int(queue_cfg.get("max_size", 5000)))
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.max_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = defaultdict(int)
        self._last_error: str | None = None
        if self.enabled and self.queue_enabled and self.local_sink_enabled:
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="violation-ingest-worker",
                daemon=True,
            )
            self._thread.start()

    def _emit_to_postgres(self, *, rows: list[dict[str, Any]]) -> None:
        if self.pg_logger is None:
            return
        groups: Dict[str, List[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            cam = str(row.get("camera_id", "cam_01"))
            alert = row.get("alert", {})
            if isinstance(alert, dict):
                groups[cam].append(alert)
        for cam, alerts in groups.items():
            if not alerts:
                continue
            self.pg_logger.ingest_alerts(alerts=alerts, camera_id=cam)

    def _worker_loop(self) -> None:
        batch: list[dict[str, Any]] = []
        next_flush = time.perf_counter() + self.flush_interval_seconds
        while not self._stop.is_set():
            timeout = max(0.01, next_flush - time.perf_counter())
            try:
                item = self._queue.get(timeout=timeout)
                batch.append(item)
            except queue.Empty:
                pass
            except Exception as exc:
                self._last_error = f"queue_error:{type(exc).__name__}"
            now_perf = time.perf_counter()
            if batch and (len(batch) >= self.batch_size or now_perf >= next_flush):
                try:
                    self._emit_to_postgres(rows=batch)
                    self._stats["flushed_rows"] += len(batch)
                    self._stats["flushed_batches"] += 1
                except Exception as exc:
                    self._last_error = f"flush_error:{type(exc).__name__}"
                batch = []
                next_flush = now_perf + self.flush_interval_seconds
        if batch:
            try:
                self._emit_to_postgres(rows=batch)
                self._stats["flushed_rows"] += len(batch)
                self._stats["flushed_batches"] += 1
            except Exception:
                pass

    def _enqueue(self, *, alert: dict[str, Any], camera_id: str) -> bool:
        record = {"camera_id": str(camera_id), "alert": alert}
        try:
            self._queue.put_nowait(record)
            self._stats["queued"] += 1
            return True
        except queue.Full:
            self._stats["queue_full"] += 1
            if self.drop_when_full:
                self._stats["dropped"] += 1
                return False
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(record)
                self._stats["dropped_oldest"] += 1
                return True
            except Exception as exc:
                self._last_error = f"queue_recover_error:{type(exc).__name__}"
                self._stats["dropped"] += 1
                return False

    def ingest_alerts(self, *, alerts: Iterable[dict[str, Any]], camera_id: str) -> None:
        if not self.enabled:
            return
        filtered = self.filter.filter(alerts=alerts, camera_id=str(camera_id))
        if not filtered:
            return
        self._stats["filtered_pass"] += len(filtered)
        kafka_mode_enabled = self.mode in {"kafka", "kafka_flink"}
        if kafka_mode_enabled and self.kafka.enabled:
            for alert in filtered:
                ok = self.kafka.publish_alert(alert=alert, camera_id=str(camera_id))
                if ok:
                    self._stats["kafka_published"] += 1
                else:
                    self._stats["kafka_failed"] += 1
        if not self.local_sink_enabled:
            return
        if self.queue_enabled and self._thread is not None:
            for alert in filtered:
                self._enqueue(alert=alert, camera_id=str(camera_id))
            return
        try:
            self.pg_logger.ingest_alerts(alerts=filtered, camera_id=str(camera_id))
            self._stats["direct_writes"] += len(filtered)
        except Exception as exc:
            self._last_error = f"direct_write_error:{type(exc).__name__}"

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.kafka.close()

    def status(self) -> dict[str, Any]:
        pg_status = self.pg_logger.status() if self.pg_logger is not None else {"enabled": False}
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "local_sink_enabled": bool(self.local_sink_enabled),
            "queue_enabled": bool(self.queue_enabled),
            "queue_size": int(self._queue.qsize()) if self.queue_enabled else 0,
            "max_queue_size": int(self.max_size),
            "batch_size": int(self.batch_size),
            "flush_interval_seconds": float(self.flush_interval_seconds),
            "drop_when_full": bool(self.drop_when_full),
            "last_error": self._last_error,
            "stats": {k: int(v) for k, v in self._stats.items()},
            "filter": self.filter.status(),
            "kafka": self.kafka.status(),
            "postgres": pg_status,
        }

    def status_json(self) -> str:
        return json.dumps(self.status(), ensure_ascii=True)
