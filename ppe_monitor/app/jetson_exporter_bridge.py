"""Bridge for Jetson jtop Prometheus exporter metrics.

This module reads metrics exposed by a Jetson exporter (for example jtop-based)
and normalizes selected values for:
- app API payloads (`/api/jetson/stats`)
- unified Prometheus export through this app (`/metrics`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class JetsonExporterConfig:
    enabled: bool = False
    url: str = "http://127.0.0.1:9100/metrics"
    timeout_seconds: float = 1.5
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

    @classmethod
    def from_app_config(cls, config: Mapping[str, object]) -> "JetsonExporterBridge":
        jetson_cfg = dict(config.get("jetson_exporter", {}) or {})
        metric_map = jetson_cfg.get("metric_map", {})
        return cls(
            JetsonExporterConfig(
                enabled=bool(jetson_cfg.get("enabled", False)),
                url=str(jetson_cfg.get("url", "http://127.0.0.1:9100/metrics")),
                timeout_seconds=float(jetson_cfg.get("timeout_seconds", 1.5)),
                metric_map=dict(metric_map) if isinstance(metric_map, dict) else {},
            )
        )

    def read_snapshot(self) -> JetsonSnapshot:
        if not self.config.enabled:
            return JetsonSnapshot(enabled=False, available=False, source_url=self.config.url)
        try:
            raw = self._fetch_text(self.config.url, self.config.timeout_seconds)
        except Exception as exc:
            return JetsonSnapshot(
                enabled=True,
                available=False,
                error=str(exc),
                source_url=self.config.url,
            )

        series = parse_prometheus_text(raw)
        return JetsonSnapshot(
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
