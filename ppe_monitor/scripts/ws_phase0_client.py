#!/usr/bin/env python3
"""WebSocket load generator and metrics collector for Phase-0 tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume /ws/stream and collect summary metrics.")
    parser.add_argument("--url", type=str, required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8000/ws/stream")
    parser.add_argument("--duration", type=float, default=120.0, help="Duration in seconds to consume stream.")
    parser.add_argument("--output", type=str, required=True, help="Path to output summary JSON.")
    parser.add_argument(
        "--metrics-jsonl",
        type=str,
        default="",
        help="Optional path to write per-message metrics snapshots (JSONL).",
    )
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--read-timeout", type=float, default=5.0)
    parser.add_argument(
        "--no-jpeg",
        action="store_true",
        help="Request JSON-only stream by adding `?jpeg=0` to websocket URL.",
    )
    return parser.parse_args()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _with_query_param(url: str, key: str, value: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query[key] = value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


async def _consume(args: argparse.Namespace) -> dict[str, Any]:
    start_wall = time.time()
    start_perf = time.perf_counter()

    fps_values: list[float] = []
    tracked_values: list[float] = []
    json_messages = 0
    bytes_messages = 0
    bytes_total = 0
    connect_attempts = 0
    last_metrics: dict[str, Any] = {}
    last_payload_ts: float | None = None

    metrics_fp = None
    if args.metrics_jsonl:
        Path(args.metrics_jsonl).parent.mkdir(parents=True, exist_ok=True)
        metrics_fp = Path(args.metrics_jsonl).open("w", encoding="utf-8")

    connect_deadline = time.perf_counter() + max(1.0, float(args.connect_timeout))
    ws_url = _with_query_param(args.url, "jpeg", "0") if args.no_jpeg else args.url
    ws = None
    while ws is None and time.perf_counter() < connect_deadline:
        connect_attempts += 1
        try:
            ws = await websockets.connect(ws_url, max_size=None)
        except Exception:
            await asyncio.sleep(0.5)

    if ws is None:
        if metrics_fp is not None:
            metrics_fp.close()
        raise RuntimeError(f"Failed to connect to {args.url} within {args.connect_timeout}s")

    try:
        end_time = start_perf + max(1.0, float(args.duration))
        while time.perf_counter() < end_time:
            timeout_left = min(float(args.read_timeout), max(0.05, end_time - time.perf_counter()))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout_left)
            except asyncio.TimeoutError:
                continue

            if isinstance(msg, (bytes, bytearray)):
                bytes_messages += 1
                bytes_total += len(msg)
                continue

            json_messages += 1
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                continue

            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict):
                last_metrics = metrics
                fps = _as_float(metrics.get("fps"), default=0.0)
                tracked = _as_float(metrics.get("tracked_count"), default=0.0)
                fps_values.append(fps)
                tracked_values.append(tracked)
                last_payload_ts = _as_float(payload.get("timestamp"), default=0.0)
                if metrics_fp is not None:
                    metrics_fp.write(
                        json.dumps(
                            {
                                "ts": last_payload_ts,
                                "fps": fps,
                                "tracked_count": tracked,
                                "verifier_ollama_calls": metrics.get("verifier_ollama_calls"),
                                "verifier_crop_infer_calls": metrics.get("verifier_crop_infer_calls"),
                                "backend": metrics.get("backend"),
                            }
                        )
                        + "\n"
                    )
            elif payload.get("event_type") == "stream_rejected":
                reason = payload.get("reason", "stream_rejected")
                raise RuntimeError(f"Server rejected stream: {reason}")
    finally:
        if metrics_fp is not None:
            metrics_fp.close()
        await ws.close()

    elapsed = max(1e-6, time.perf_counter() - start_perf)
    summary = {
        "started_at_epoch_s": start_wall,
        "duration_s": elapsed,
        "connect_attempts": connect_attempts,
        "json_messages": json_messages,
        "jpeg_messages": bytes_messages,
        "jpeg_total_bytes": bytes_total,
        "json_rate_hz": json_messages / elapsed,
        "jpeg_rate_hz": bytes_messages / elapsed,
        "mean_fps_from_payload": statistics.mean(fps_values) if fps_values else 0.0,
        "p95_fps_from_payload": (
            sorted(fps_values)[max(0, int(round(len(fps_values) * 0.95)) - 1)] if fps_values else 0.0
        ),
        "mean_tracked_count": statistics.mean(tracked_values) if tracked_values else 0.0,
        "last_payload_timestamp": last_payload_ts,
        "last_metrics": last_metrics,
    }
    return summary


async def _main_async() -> int:
    args = parse_args()
    rc = 0
    try:
        summary = await _consume(args)
    except Exception as exc:
        rc = 1
        summary = {
            "error": str(exc),
            "url": args.url,
            "duration_s": float(args.duration),
            "connect_timeout_s": float(args.connect_timeout),
            "read_timeout_s": float(args.read_timeout),
            "requested_no_jpeg": bool(args.no_jpeg),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return rc


def main() -> None:
    raise SystemExit(asyncio.run(_main_async()))


if __name__ == "__main__":
    main()
