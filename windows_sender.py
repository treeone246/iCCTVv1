"""Windows-side ZeroMQ sender for forwarding detection JSON to WSL."""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import zmq

import config
from model_adapter import adapt_parsed_detections


LOGGER = logging.getLogger("windows_sender")


class ZmqDetectionSender:
    """Simple PUSH sender that publishes detection payloads as JSON strings."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(self.endpoint)
        LOGGER.info("Connected PUSH socket to %s", self.endpoint)

    def send_payload(self, payload: dict[str, Any]) -> None:
        """Validate JSON serializability and send payload."""
        try:
            message = json.dumps(payload, ensure_ascii=False)
            self.socket.send_string(message)
            LOGGER.info(
                "Sent payload frame_id=%s detections=%d",
                payload.get("frame_id"),
                len(payload.get("detections", [])),
            )
        except (TypeError, ValueError) as exc:
            LOGGER.exception("Payload is not valid JSON-serializable data: %s", exc)
            raise
        except zmq.ZMQError as exc:
            LOGGER.exception("ZeroMQ send failed: %s", exc)
            raise

    def close(self) -> None:
        """Clean shutdown for socket/context."""
        self.socket.close(linger=0)
        self.context.term()
        LOGGER.info("ZeroMQ sender closed")


def build_fake_payload(frame_id: str) -> dict[str, Any]:
    """One-shot fake detection payload for connectivity testing."""
    fake_detections = [
        {
            "label": "person",
            "score": 0.95,
            "x1": 100,
            "y1": 120,
            "x2": 260,
            "y2": 500,
        }
    ]
    return adapt_parsed_detections(fake_detections, frame_id=frame_id)


def run_once(sender: ZmqDetectionSender, frame_id: str) -> None:
    payload = build_fake_payload(frame_id=frame_id)
    sender.send_payload(payload)


def run_loop(sender: ZmqDetectionSender, frame_id: str, interval_sec: float) -> None:
    frame_idx = 0
    LOGGER.info("Starting continuous send loop (interval=%.3fs)", interval_sec)
    while True:
        payload = build_fake_payload(frame_id=frame_id)
        payload["frame_id"] = f"{frame_id}_{frame_idx}"
        payload["timestamp"] = time.time()
        sender.send_payload(payload)
        frame_idx += 1
        time.sleep(interval_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows ZeroMQ detection sender")
    parser.add_argument(
        "--host",
        default=config.ZMQ_HOST,
        help="Receiver host/IP (default from config.py)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.ZMQ_PORT,
        help="Receiver port (default from config.py)",
    )
    parser.add_argument(
        "--frame-id",
        default=config.DEFAULT_FRAME_ID,
        help="Default frame_id field in outgoing payloads",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=config.SEND_INTERVAL_SEC,
        help="Loop send interval seconds (used with --loop)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Enable continuous loop sending",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    endpoint = config.zmq_endpoint(host=args.host, port=args.port)

    # If localhost forwarding is unavailable, replace host with your WSL IP.
    LOGGER.info("Using endpoint: %s", endpoint)
    LOGGER.info("Tip: if localhost fails, try --host <wsl_ip>")

    sender = ZmqDetectionSender(endpoint=endpoint)
    try:
        if args.loop:
            run_loop(sender, frame_id=args.frame_id, interval_sec=args.interval)
        else:
            run_once(sender, frame_id=args.frame_id)
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level guard for CLI app
        LOGGER.exception("Sender failed: %s", exc)
        return 1
    finally:
        sender.close()


if __name__ == "__main__":
    raise SystemExit(main())
