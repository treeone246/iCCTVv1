"""Bridge for Jetson jtop Prometheus exporter metrics.

This module reads metrics exposed by a Jetson exporter (for example jtop-based)
and normalizes selected values for:
- app API payloads (`/api/jetson/stats`)
- unified Prometheus export through this app (`/metrics`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Iterable, List, Mapping, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class JetsonExporterConfig:
    enabled: bool = False
    mode: str = "external"
    url: str = "http://127.0.0.1:9100/metrics"
    timeout_seconds: float = 1.5
    refresh_seconds: float = 1.0
    fallback_to_local_jtop: bool = True
    metric_map: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class JetsonSnapshot:
    enabled: bool
    available: bool
    error: str = ""
    source_url: str = ""
    cpu_utilization_pct: float = 0.0
    gpu_utilization_pct: float = 0.0
    memory_utilization_pct: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    fan_pwm_pct: float = 0.0


DEFAULT_METRIC_MAP: Dict[str, List[str]] = {
    "cpu_utilization_pct": [
        "jetson_cpu_utilization",
        "jetson_cpu_utilization_pct",
        "jetson_cpu_usage_percent",
    ],
    "gpu_utilization_pct": [
        "jetson_gpu_utilization",
        "jetson_gpu_utilization_pct",
        "jetson_gpu_usage_percent",
    ],
    "memory_utilization_pct": [
        "jetson_memory_utilization",
        "jetson_memory_utilization_pct",
        "jetson_ram_utilization",
    ],
    "memory_used_mb": [
        "jetson_memory_used_mb",
        "jetson_ram_used_mb",
        "jetson_mem_used_mb",
    ],
    "memory_total_mb": [
        "jetson_memory_total_mb",
        "jetson_ram_total_mb",
        "jetson_mem_total_mb",
    ],
    "temperature_c": [
        "jetson_temperature_c",
        "jetson_temp_c",
        "jetson_thermal_c",
    ],
    "power_w": [
        "jetson_power_w",
        "jetson_power_consumption_w",
    ],
    "fan_pwm_pct": [
        "jetson_fan_pwm",
        "jetson_fan_pwm_pct",
        "jetson_fan_speed_pct",
    ],
}


class JetsonExporterBridge:
    """Fetches and normalizes metrics from Jetson Prometheus exporter."""

    def __init__(self, config: JetsonExporterConfig) -> None:
        self.config = config
        self.metric_map = dict(DEFAULT_METRIC_MAP)
        self.metric_map.update(config.metric_map or {})
        self._last_snapshot: Optional[JetsonSnapshot] = None
        self._last_read_ts: float = 0.0

    @classmethod
    def from_app_config(cls, config: Mapping[str, object]) -> "JetsonExporterBridge":
        jetson_cfg = dict(config.get("jetson_exporter", {}) or {})
        metric_map = jetson_cfg.get("metric_map", {})
        return cls(
            JetsonExporterConfig(
                enabled=bool(jetson_cfg.get("enabled", False)),
                mode=str(jetson_cfg.get("mode", "external")),
                url=str(jetson_cfg.get("url", "http://127.0.0.1:9100/metrics")),
                timeout_seconds=float(jetson_cfg.get("timeout_seconds", 1.5)),
                refresh_seconds=float(jetson_cfg.get("refresh_seconds", 1.0)),
                fallback_to_local_jtop=bool(jetson_cfg.get("fallback_to_local_jtop", True)),
                metric_map=dict(metric_map) if isinstance(metric_map, dict) else {},
            )
        )

    def read_snapshot(self) -> JetsonSnapshot:
        if not self.config.enabled:
            return JetsonSnapshot(enabled=False, available=False, source_url=self.config.url)
        now = time.monotonic()
        if (
            self._last_snapshot is not None
            and (now - self._last_read_ts) < max(0.1, float(self.config.refresh_seconds))
        ):
            return self._last_snapshot

        mode = str(self.config.mode).strip().lower()
        if mode == "local_jtop":
            snap = self._read_from_local_jtop()
            self._cache_snapshot(now, snap)
            return snap

        try:
            raw = self._fetch_text(self.config.url, self.config.timeout_seconds)
        except Exception as exc:
            if self.config.fallback_to_local_jtop:
                snap = self._read_from_local_jtop()
                if snap.available:
                    self._cache_snapshot(now, snap)
                    return snap
            snap = JetsonSnapshot(
                enabled=True,
                available=False,
                error=str(exc),
                source_url=self.config.url,
            )
            self._cache_snapshot(now, snap)
            return snap

        series = parse_prometheus_text(raw)
        snap = JetsonSnapshot(
            enabled=True,
            available=True,
            source_url=self.config.url,
            cpu_utilization_pct=self._pick(series, self.metric_map["cpu_utilization_pct"]),
            gpu_utilization_pct=self._pick(series, self.metric_map["gpu_utilization_pct"]),
            memory_utilization_pct=self._pick(series, self.metric_map["memory_utilization_pct"]),
            memory_used_mb=self._pick(series, self.metric_map["memory_used_mb"]),
            memory_total_mb=self._pick(series, self.metric_map["memory_total_mb"]),
            temperature_c=self._pick(series, self.metric_map["temperature_c"]),
            power_w=self._pick(series, self.metric_map["power_w"]),
            fan_pwm_pct=self._pick(series, self.metric_map["fan_pwm_pct"]),
        )
        self._cache_snapshot(now, snap)
        return snap

    def _cache_snapshot(self, ts: float, snap: JetsonSnapshot) -> None:
        self._last_snapshot = snap
        self._last_read_ts = ts

    def _fetch_text(self, url: str, timeout_seconds: float) -> str:
        req = Request(url=url, method="GET")
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except URLError as exc:
            raise RuntimeError(f"jetson_exporter_unreachable: {exc}") from exc

    def _pick(self, series: Mapping[str, List[float]], metric_names: Iterable[str]) -> float:
        for name in metric_names:
            values = series.get(name)
            if values:
                if len(values) == 1:
                    return round(float(values[0]), 4)
                return round(sum(float(v) for v in values) / len(values), 4)
        return 0.0

    def _read_from_local_jtop(self) -> JetsonSnapshot:
        try:
            from jtop import JtopException, jtop
        except Exception as exc:
            return JetsonSnapshot(
                enabled=True,
                available=False,
                error=f"local_jtop_unavailable: {exc}",
                source_url="local_jtop",
            )

        try:
            with jtop() as jetson:
                if not jetson.ok():
                    return JetsonSnapshot(
                        enabled=True,
                        available=False,
                        error="local_jtop_not_ready",
                        source_url="local_jtop",
                    )
                stats = dict(getattr(jetson, "stats", {}) or {})
        except Exception as exc:
            return JetsonSnapshot(
                enabled=True,
                available=False,
                error=f"local_jtop_error: {exc}",
                source_url="local_jtop",
            )

        values = _derive_metrics(stats)
        return JetsonSnapshot(
            enabled=True,
            available=True,
            source_url="local_jtop",
            cpu_utilization_pct=values["cpu"],
            gpu_utilization_pct=values["gpu"],
            memory_utilization_pct=values["mem_pct"],
            memory_used_mb=values["mem_used"],
            memory_total_mb=values["mem_total"],
            temperature_c=values["temp"],
            power_w=values["power"],
            fan_pwm_pct=values["fan"],
        )


def parse_prometheus_text(text: str) -> Dict[str, List[float]]:
    """Parse Prometheus exposition text into metric->values list."""
    out: Dict[str, List[float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        metric_key: Optional[str] = None
        value_token: Optional[str] = None
        if "{" in line and "}" in line:
            left = line.split("{", 1)[0].strip()
            right = line.split("}", 1)[1].strip()
            if right:
                parts = right.split()
                if parts:
                    metric_key = left
                    value_token = parts[0]
        else:
            parts = line.split()
            if len(parts) >= 2:
                metric_key = parts[0]
                value_token = parts[1]

        if not metric_key or value_token is None:
            continue
        try:
            value = float(value_token)
        except ValueError:
            continue
        out.setdefault(metric_key, []).append(value)
    return out


def _derive_metrics(stats: Dict[str, object]) -> Dict[str, float]:
    flat = _flatten_values(stats)
    cpu = _pick_avg(flat, ("cpu",), default=0.0)
    gpu = _pick_first(flat, (("gpu",), ("gr3d",)), default=0.0)
    mem_used = _pick_first(
        flat,
        (("ram", "used"), ("memory", "used"), ("mem", "used")),
        default=0.0,
    )
    mem_total = _pick_first(
        flat,
        (("ram", "total"), ("memory", "total"), ("mem", "total")),
        default=0.0,
    )
    mem_pct = _pick_first(
        flat,
        (("ram", "percent"), ("memory", "percent"), ("ram", "util"), ("memory", "util")),
        default=0.0,
    )
    if mem_pct <= 0.0 and mem_total > 0.0:
        mem_pct = (mem_used / mem_total) * 100.0
    temp = _pick_first(
        flat,
        (("temp", "gpu"), ("temp", "cpu"), ("temperature",), ("temp",)),
        default=0.0,
    )
    power = _pick_first(flat, (("power", "tot"), ("power", "cur"), ("power",)), default=0.0)
    fan = _pick_first(flat, (("fan", "pwm"), ("fan", "speed"), ("fan",)), default=0.0)
    return {
        "cpu": round(max(0.0, min(100.0, cpu)), 4),
        "gpu": round(max(0.0, min(100.0, gpu)), 4),
        "mem_pct": round(max(0.0, min(100.0, mem_pct)), 4),
        "mem_used": round(max(0.0, mem_used), 4),
        "mem_total": round(max(0.0, mem_total), 4),
        "temp": round(max(0.0, temp), 4),
        "power": round(max(0.0, power), 4),
        "fan": round(max(0.0, min(100.0, fan)), 4),
    }


def _flatten_values(obj: object, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_values(v, key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}.{i}" if prefix else str(i)
            out.update(_flatten_values(v, key))
    else:
        val = _to_float(obj)
        if val is not None:
            out[prefix] = val
    return out


def _to_float(value: object) -> float | None:
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


def _pick_first(
    flat: Dict[str, float],
    keyword_groups: Iterable[tuple[str, ...]],
    default: float = 0.0,
) -> float:
    lowered = {k.lower(): v for k, v in flat.items()}
    for group in keyword_groups:
        for k, v in lowered.items():
            if all(token in k for token in group):
                return v
    return default


def _pick_avg(flat: Dict[str, float], keywords: Iterable[str], default: float = 0.0) -> float:
    keys = [k for k in flat.keys() if all(token in k.lower() for token in keywords)]
    if not keys:
        return default
    vals = [flat[k] for k in keys]
    return sum(vals) / max(1, len(vals))
