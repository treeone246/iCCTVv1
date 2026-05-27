"""Consume PPE violation alerts from Kafka, filter, then persist to PostgreSQL."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from app.violation_ingest import ViolationAlertFilter
from app.violation_kafka import ViolationKafkaConsumer
from app.violation_postgres_logger import ViolationPostgresLogger


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Kafka -> filter -> PostgreSQL violation ingest worker")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--log-interval", type=float, default=5.0, help="Periodic stats log interval seconds")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    config = _load_config(config_path)

    consumer = ViolationKafkaConsumer(config=config)
    pg_logger = ViolationPostgresLogger(config=config, project_root=project_root)
    alert_filter = ViolationAlertFilter(config=config)

    if not consumer.enabled:
        print(
            json.dumps(
                {
                    "event_type": "kafka_consumer_not_enabled",
                    "consumer_status": consumer.status(),
                }
            )
        )
        return 2

    print(
        json.dumps(
            {
                "event_type": "kafka_consumer_started",
                "consumer_status": consumer.status(),
                "postgres_status": pg_logger.status(),
            }
        )
    )

    received = 0
    written = 0
    suppressed = 0
    last_log = time.time()
    try:
        while True:
            doc = consumer.poll(timeout=1.0)
            if not isinstance(doc, dict):
                now = time.time()
                if now - last_log >= float(args.log_interval):
                    print(
                        json.dumps(
                            {
                                "event_type": "kafka_consumer_heartbeat",
                                "received": received,
                                "written": written,
                                "suppressed": suppressed,
                                "consumer_status": consumer.status(),
                                "postgres_status": pg_logger.status(),
                                "filter_status": alert_filter.status(),
                            }
                        )
                    )
                    last_log = now
                continue

            received += 1
            alert = doc.get("alert")
            camera_id = str(doc.get("camera_id", "cam_01"))
            if not isinstance(alert, dict):
                suppressed += 1
                continue
            passed = alert_filter.filter(alerts=[alert], camera_id=camera_id)
            if not passed:
                suppressed += 1
                continue
            pg_logger.ingest_alerts(alerts=passed, camera_id=camera_id)
            written += len(passed)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        pg_logger.close()
        print(
            json.dumps(
                {
                    "event_type": "kafka_consumer_stopped",
                    "received": received,
                    "written": written,
                    "suppressed": suppressed,
                    "consumer_status": consumer.status(),
                    "postgres_status": pg_logger.status(),
                    "filter_status": alert_filter.status(),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
