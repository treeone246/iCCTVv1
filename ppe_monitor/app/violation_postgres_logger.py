"""PostgreSQL sink for PPE violation alerts with local ROI artifact storage."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Iterable, Optional

try:
    import psycopg  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    psycopg = None


class ViolationPostgresLogger:
    """Persist newly raised violations into PostgreSQL and local ROI files."""

    def __init__(self, *, config: dict, project_root: Path) -> None:
        pg_cfg = config.get("postgres_logging", {}) or {}
        self.enabled = bool(pg_cfg.get("enabled", False))
        self.project_root = Path(project_root).resolve()
        self.device_id = str(pg_cfg.get("device_id", "jetson-device"))
        self.default_camera_id = str(pg_cfg.get("camera_id", "cam_01"))
        self.local_roi_dir = self._resolve_path(str(pg_cfg.get("local_roi_dir", "outputs/violation_roi")))
        self.local_roi_dir.mkdir(parents=True, exist_ok=True)
        self.max_seen_alerts = max(1000, int(pg_cfg.get("max_seen_alerts", 50000)))
        self._seen: set[str] = set()
        self._seen_queue: Deque[str] = deque()
        self._lock = threading.Lock()
        self._host = str(pg_cfg.get("host", "127.0.0.1"))
        self._port = int(pg_cfg.get("port", 5432))
        self._database = str(pg_cfg.get("database", "ppe_monitor"))
        self._user = str(pg_cfg.get("user", "ppe_app"))
        self._password = str(pg_cfg.get("password", os.getenv("PPE_MONITOR_PG_PASSWORD", os.getenv("PGPASSWORD", ""))))
        self._dsn = str(
            pg_cfg.get(
                "dsn",
                (
                    f"host={self._host} "
                    f"port={self._port} "
                    f"dbname={self._database} "
                    f"user={self._user} "
                    f"password={self._password}"
                ),
            )
        ).strip()
        self._conn = None
        self._init_error: str | None = None
        self._last_error: str | None = None
        self._last_insert_ts: float = 0.0
        self._inserted_count: int = 0
        self._duplicate_count: int = 0
        self._ingest_batches: int = 0
        self._connect_fail_count: int = 0
        if self.enabled and psycopg is None:
            self._init_error = "psycopg_not_installed"
            self.enabled = False

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _connect(self):
        if not self.enabled or psycopg is None:
            return None
        if self._conn is not None:
            return self._conn
        try:
            self._conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=3)
            self._ensure_schema(self._conn)
            self._init_error = None
            self._last_error = None
        except Exception as exc:
            self._connect_fail_count += 1
            self._last_error = f"db_connect_error:{type(exc).__name__}:{str(exc)}"
            raise
        return self._conn

    def _ensure_schema(self, conn) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS violation_logs (
          violation_id UUID PRIMARY KEY,
          event_time TIMESTAMPTZ NOT NULL,
          device_id TEXT NOT NULL,
          camera_id TEXT NOT NULL,
          person_display_id TEXT,
          ppe_item TEXT NOT NULL,
          status TEXT NOT NULL,
          confidence NUMERIC,
          local_roi_path TEXT,
          roi_exists BOOLEAN DEFAULT TRUE,
          reason TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_violation_time ON violation_logs(event_time DESC);
        CREATE INDEX IF NOT EXISTS idx_violation_camera_time ON violation_logs(camera_id, event_time DESC);
        """
        with conn.cursor() as cur:
            cur.execute(sql)

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = None

    def _mark_seen(self, alert_id: str) -> bool:
        with self._lock:
            if alert_id in self._seen:
                return False
            self._seen.add(alert_id)
            self._seen_queue.append(alert_id)
            while len(self._seen_queue) > self.max_seen_alerts:
                oldest = self._seen_queue.popleft()
                self._seen.discard(oldest)
            return True

    def _safe_decode_b64(self, value: str | None) -> bytes | None:
        if not value:
            return None
        try:
            return base64.b64decode(value.encode("ascii"), validate=True)
        except Exception:
            return None

    def _write_roi_file(
        self,
        *,
        violation_uuid: uuid.UUID,
        ts: float,
        kind: str,
        payload_b64: str | None,
    ) -> str | None:
        raw = self._safe_decode_b64(payload_b64)
        if raw is None:
            return None
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        date_dir = self.local_roi_dir / dt.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{violation_uuid}_{kind}.jpg"
        full_path = date_dir / filename
        try:
            full_path.write_bytes(raw)
        except OSError:
            return None
        try:
            return str(full_path.relative_to(self.project_root))
        except ValueError:
            return str(full_path)

    def ingest_alerts(
        self,
        *,
        alerts: Iterable[dict[str, Any]],
        camera_id: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        self._ingest_batches += 1
        cam = str(camera_id or self.default_camera_id)
        conn = None
        try:
            conn = self._connect()
        except Exception as exc:
            self._init_error = f"db_connect_error:{type(exc).__name__}"
            self._last_error = self._init_error
            print(
                json.dumps(
                    {
                        "event_type": "postgres_violation_connect_error",
                        "error": self._init_error,
                    }
                )
            )
            self.close()
            return
        if conn is None:
            return

        for alert in alerts:
            try:
                alert_id = str(alert.get("alert_id", "")).strip()
                if not alert_id:
                    continue
                if not self._mark_seen(alert_id):
                    self._duplicate_count += 1
                    continue
                person_id = int(alert.get("person_id", -1))
                person_display_id = str(alert.get("display_id", ""))
                ppe_item = str(alert.get("item", "unknown"))
                status = str(alert.get("status", "ACTIVE"))
                reason = str(alert.get("reason", ""))
                confidence = float(alert.get("negative_conf", 0.0) or 0.0)
                ts = float(alert.get("timestamp", time.time()) or time.time())
                event_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                violation_uuid = uuid.uuid5(uuid.NAMESPACE_URL, alert_id)
                item_path = self._write_roi_file(
                    violation_uuid=violation_uuid,
                    ts=ts,
                    kind="item",
                    payload_b64=(
                        str(alert.get("item_crop_jpeg_base64"))
                        if alert.get("item_crop_jpeg_base64")
                        else str(alert.get("evidence_jpeg_base64"))
                        if alert.get("evidence_jpeg_base64")
                        else None
                    ),
                )
                if not item_path:
                    # still persist record even if image is unavailable.
                    item_path = str(alert.get("local_roi_path", "")) or None
                person_path = self._write_roi_file(
                    violation_uuid=violation_uuid,
                    ts=ts,
                    kind="person",
                    payload_b64=(
                        str(alert.get("person_crop_jpeg_base64"))
                        if alert.get("person_crop_jpeg_base64")
                        else None
                    ),
                )
                if person_path and not item_path:
                    item_path = person_path
                roi_exists = bool(item_path)
                sql = """
                INSERT INTO violation_logs (
                    violation_id,
                    event_time,
                    device_id,
                    camera_id,
                    person_display_id,
                    ppe_item,
                    status,
                    confidence,
                    local_roi_path,
                    roi_exists,
                    reason
                )
                VALUES (
                    %(violation_id)s,
                    %(event_time)s,
                    %(device_id)s,
                    %(camera_id)s,
                    %(person_display_id)s,
                    %(ppe_item)s,
                    %(status)s,
                    %(confidence)s,
                    %(local_roi_path)s,
                    %(roi_exists)s,
                    %(reason)s
                )
                ON CONFLICT (violation_id) DO NOTHING
                """
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        {
                            "violation_id": str(violation_uuid),
                            "event_time": event_time,
                            "device_id": self.device_id,
                            "camera_id": cam,
                            "person_display_id": person_display_id,
                            "ppe_item": ppe_item,
                            "status": status,
                            "confidence": confidence,
                            "local_roi_path": item_path,
                            "roi_exists": roi_exists,
                            "reason": reason,
                        },
                    )
                self._inserted_count += 1
                self._last_insert_ts = time.time()
            except Exception as exc:
                # Fail-soft: do not break realtime stream on DB/file issues.
                self._last_error = f"insert_error:{type(exc).__name__}"
                print(
                    json.dumps(
                        {
                            "event_type": "postgres_violation_log_error",
                            "error": self._last_error,
                        }
                    )
                )
                continue

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "init_error": self._init_error,
            "last_error": self._last_error,
            "inserted_count": int(self._inserted_count),
            "duplicate_count": int(self._duplicate_count),
            "ingest_batches": int(self._ingest_batches),
            "connect_fail_count": int(self._connect_fail_count),
            "last_insert_ts": float(self._last_insert_ts),
            "local_roi_dir": str(self.local_roi_dir),
            "device_id": str(self.device_id),
            "camera_id": str(self.default_camera_id),
            "db_host": self._host,
            "db_port": self._port,
            "db_name": self._database,
            "db_user": self._user,
            "db_password_set": bool(self._password),
        }
