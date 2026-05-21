#!/usr/bin/env python3
"""Lightweight Jetson Prometheus exporter using jtop (jetson-stats).

Exports normalized metrics expected by ppe_monitor bridge:
- jetson_cpu_utilization
- jetson_gpu_utilization
- jetson_memory_utilization
- jetson_memory_used_mb
- jetson_memory_total_mb
- jetson_temperature_c
- jetson_power_w
- jetson_fan_pwm_pct
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Iterable, Tuple

from prometheus_client import Gauge, start_http_server

try:
    from jtop import JtopException, jtop
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import jtop. Install with: pip install -U jetson-stats\n"
        f"Import error: {exc}"
    )


CPU = Gauge("jetson_cpu_utilization", "Jetson CPU utilization percent")
GPU = Gauge("jetson_gpu_utilization", "Jetson GPU utilization percent")
MEM_PCT = Gauge("jetson_memory_utilization", "Jetson memory utilization percent")
MEM_USED = Gauge("jetson_memory_used_mb", "Jetson memory used MB")
MEM_TOTAL = Gauge("jetson_memory_total_mb", "Jetson memory total MB")
TEMP = Gauge("jetson_temperature_c", "Jetson temperature C")
POWER = Gauge("jetson_power_w", "Jetson board power W")
FAN = Gauge("jetson_fan_pwm_pct", "Jetson fan PWM percent")
UP = Gauge("jetson_exporter_up", "Exporter up state")


def flatten_values(obj: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_values(v, key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_values(v, key))
    else:
        val = _to_float(obj)
        if val is not None:
            out[prefix] = val
    return out


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().lower().replace("%", "")
        if v.endswith("mb"):
            v = v[:-2].strip()
        if v.endswith("w"):
            v = v[:-1].strip()
        try:
            return float(v)
        except Exception:
            return None
    return None


def pick_avg(flat: Dict[str, float], keywords: Iterable[str], default: float = 0.0) -> float:
    keys = [k for k in flat.keys() if all(token in k.lower() for token in keywords)]
    if not keys:
        return default
    vals = [flat[k] for k in keys]
    return sum(vals) / max(1, len(vals))


def pick_first(flat: Dict[str, float], keyword_groups: Iterable[Tuple[str, ...]], default: float = 0.0) -> float:
    lowered = {k.lower(): v for k, v in flat.items()}
    for group in keyword_groups:
        for k, v in lowered.items():
            if all(token in k for token in group):
                return v
    return default


def derive_metrics(stats: Dict[str, Any]) -> Dict[str, float]:
    flat = flatten_values(stats)

    cpu = pick_avg(flat, ("cpu",), default=0.0)
    gpu = pick_first(
        flat,
        (
            ("gpu",),
            ("gr3d",),
        ),
        default=0.0,
    )
    mem_used = pick_first(
        flat,
        (
            ("ram", "used"),
            ("memory", "used"),
            ("mem", "used"),
        ),
        default=0.0,
    )
    mem_total = pick_first(
        flat,
        (
            ("ram", "total"),
            ("memory", "total"),
            ("mem", "total"),
        ),
        default=0.0,
    )
    mem_pct = pick_first(
        flat,
        (
            ("ram", "percent"),
            ("memory", "percent"),
            ("ram", "util"),
            ("memory", "util"),
        ),
        default=0.0,
    )
    if mem_pct <= 0.0 and mem_total > 0:
        mem_pct = (mem_used / mem_total) * 100.0

    temp = pick_first(
        flat,
        (
            ("temp", "gpu"),
            ("temp", "cpu"),
            ("temperature",),
            ("temp",),
        ),
        default=0.0,
    )
    power = pick_first(
        flat,
        (
            ("power", "tot"),
            ("power", "cur"),
            ("power",),
        ),
        default=0.0,
    )
    fan = pick_first(
        flat,
        (
            ("fan", "pwm"),
            ("fan", "speed"),
            ("fan",),
        ),
        default=0.0,
    )

    return {
        "cpu": max(0.0, min(100.0, cpu)),
        "gpu": max(0.0, min(100.0, gpu)),
        "mem_pct": max(0.0, min(100.0, mem_pct)),
        "mem_used": max(0.0, mem_used),
        "mem_total": max(0.0, mem_total),
        "temp": max(0.0, temp),
        "power": max(0.0, power),
        "fan": max(0.0, min(100.0, fan)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson jtop Prometheus exporter")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_http_server(port=int(args.port), addr=str(args.host))
    print(f"jetson_jtop_exporter listening on http://{args.host}:{args.port}/metrics")

    try:
        with jtop() as jetson:
            while jetson.ok():
                stats = dict(getattr(jetson, "stats", {}) or {})
                values = derive_metrics(stats)
                CPU.set(values["cpu"])
                GPU.set(values["gpu"])
                MEM_PCT.set(values["mem_pct"])
                MEM_USED.set(values["mem_used"])
                MEM_TOTAL.set(values["mem_total"])
                TEMP.set(values["temp"])
                POWER.set(values["power"])
                FAN.set(values["fan"])
                UP.set(1.0)
                time.sleep(max(0.2, float(args.interval)))
    except (KeyboardInterrupt, JtopException):
        pass
    finally:
        UP.set(0.0)


if __name__ == "__main__":
    main()
