"""Optional Kafka transport for violation events."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

try:
    from confluent_kafka import Consumer, KafkaError, Producer  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    Consumer = None
    KafkaError = None
    Producer = None


class ViolationKafkaProducer:
    """Thin wrapper that publishes alert payloads to Kafka."""

    def __init__(self, *, config: dict) -> None:
        ingest_cfg = config.get("violation_ingest", {}) or {}
        kafka_cfg = ingest_cfg.get("kafka", {}) or {}
        self.enabled = bool(ingest_cfg.get("enabled", False)) and bool(kafka_cfg.get("enabled", False))
        self.topic = str(kafka_cfg.get("topic", "ppe.violations.raw"))
        self.also_write_local = bool(kafka_cfg.get("also_write_local", True))
        self._last_error: str | None = None
        self._published_count = 0
        self._failed_count = 0
        self._producer = None
        self._init_error: str | None = None
        if not self.enabled:
            return
        if Producer is None:
            self._init_error = "confluent_kafka_not_installed"
            self.enabled = False
            return
        bootstrap = str(kafka_cfg.get("bootstrap_servers", "127.0.0.1:9092")).strip()
        client_id = str(kafka_cfg.get("client_id", "ppe-monitor-edge")).strip()
        linger_ms = int(kafka_cfg.get("linger_ms", 10))
        delivery_timeout_ms = int(kafka_cfg.get("delivery_timeout_ms", 5000))
        acks = str(kafka_cfg.get("acks", "1")).strip()
        compression_type = str(kafka_cfg.get("compression_type", "lz4")).strip()
        try:
            self._producer = Producer(
                {
                    "bootstrap.servers": bootstrap,
                    "client.id": client_id,
                    "linger.ms": linger_ms,
                    "delivery.timeout.ms": delivery_timeout_ms,
                    "acks": acks,
                    "compression.type": compression_type,
                }
            )
        except Exception as exc:
            self._init_error = f"producer_init_error:{type(exc).__name__}"
            self._last_error = self._init_error
            self.enabled = False

    def _on_delivery(self, err, msg) -> None:
        if err is not None:
            self._failed_count += 1
            self._last_error = f"produce_error:{err}"
            return
        self._published_count += 1

    def publish_alert(self, *, alert: dict[str, Any], camera_id: str) -> bool:
        if not self.enabled or self._producer is None:
            return False
        key = f"{camera_id}:{alert.get('display_id', '')}:{alert.get('item', '')}"
        payload = json.dumps(
            {
                "event_type": "ppe_violation_alert",
                "camera_id": str(camera_id),
                "published_ts": float(time.time()),
                "alert": alert,
            },
            ensure_ascii=True,
        )
        try:
            self._producer.produce(
                self.topic,
                key=key.encode("utf-8"),
                value=payload.encode("utf-8"),
                callback=self._on_delivery,
            )
            self._producer.poll(0)
            return True
        except Exception as exc:
            self._failed_count += 1
            self._last_error = f"produce_error:{type(exc).__name__}"
            return False

    def flush(self, timeout: float = 2.0) -> None:
        if self._producer is None:
            return
        try:
            self._producer.flush(timeout=timeout)
        except Exception:
            return

    def close(self) -> None:
        self.flush()
        self._producer = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "topic": self.topic,
            "also_write_local": bool(self.also_write_local),
            "init_error": self._init_error,
            "last_error": self._last_error,
            "published_count": int(self._published_count),
            "failed_count": int(self._failed_count),
        }


class ViolationKafkaConsumer:
    """Simple consumer that reads violation alerts from Kafka."""

    def __init__(self, *, config: dict) -> None:
        ingest_cfg = config.get("violation_ingest", {}) or {}
        kafka_cfg = ingest_cfg.get("kafka", {}) or {}
        self.enabled = bool(ingest_cfg.get("enabled", False)) and bool(kafka_cfg.get("enabled", False))
        self.topic = str(kafka_cfg.get("topic", "ppe.violations.raw"))
        self._last_error: str | None = None
        self._consumed_count = 0
        self._consumer = None
        self._init_error: str | None = None
        if not self.enabled:
            return
        if Consumer is None:
            self._init_error = "confluent_kafka_not_installed"
            self.enabled = False
            return
        bootstrap = str(kafka_cfg.get("bootstrap_servers", "127.0.0.1:9092")).strip()
        group_id = str(kafka_cfg.get("group_id", "ppe-violation-consumer")).strip()
        client_id = str(kafka_cfg.get("consumer_client_id", "ppe-monitor-consumer")).strip()
        auto_offset_reset = str(kafka_cfg.get("auto_offset_reset", "latest")).strip()
        try:
            self._consumer = Consumer(
                {
                    "bootstrap.servers": bootstrap,
                    "group.id": group_id,
                    "client.id": client_id,
                    "auto.offset.reset": auto_offset_reset,
                    "enable.auto.commit": True,
                }
            )
            self._consumer.subscribe([self.topic])
        except Exception as exc:
            self._init_error = f"consumer_init_error:{type(exc).__name__}"
            self._last_error = self._init_error
            self.enabled = False

    def poll(self, timeout: float = 1.0) -> Optional[dict[str, Any]]:
        if not self.enabled or self._consumer is None:
            return None
        msg = self._consumer.poll(timeout=timeout)
        if msg is None:
            return None
        if msg.error():
            if KafkaError is not None and msg.error().code() == KafkaError._PARTITION_EOF:
                return None
            self._last_error = f"consumer_error:{msg.error()}"
            return None
        try:
            raw = msg.value()
            if raw is None:
                return None
            doc = json.loads(raw.decode("utf-8"))
            if not isinstance(doc, dict):
                return None
            self._consumed_count += 1
            return doc
        except Exception as exc:
            self._last_error = f"decode_error:{type(exc).__name__}"
            return None

    def close(self) -> None:
        if self._consumer is None:
            return
        try:
            self._consumer.close()
        except Exception:
            pass
        self._consumer = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "topic": self.topic,
            "init_error": self._init_error,
            "last_error": self._last_error,
            "consumed_count": int(self._consumed_count),
        }
