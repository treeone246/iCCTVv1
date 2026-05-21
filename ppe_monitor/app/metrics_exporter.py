"""Prometheus metrics exporter for PPE monitor runtime telemetry."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .jetson_exporter_bridge import JetsonSnapshot

try:
    from prometheus_client import CollectorRegistry, Gauge, CONTENT_TYPE_LATEST, generate_latest
except Exception:  # pragma: no cover - dependency may be missing in some envs
    CollectorRegistry = None
    Gauge = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    def generate_latest(_registry: Any = None) -> bytes:
        return b"# prometheus_client not installed\n"


class PrometheusMetricsExporter:
    """Collects pipeline metrics and renders Prometheus exposition text."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.available = self.enabled and CollectorRegistry is not None and Gauge is not None
        self._registry = CollectorRegistry() if self.available else None
        self._gauges: dict[str, Any] = {}
        if self.available:
            self._build_gauges()

    def _build_gauges(self) -> None:
        self._gauges = {
            "fps": Gauge("ppe_monitor_fps", "Pipeline FPS", registry=self._registry),
            "tracked_count": Gauge("ppe_monitor_tracked_count", "Tracked persons count", registry=self._registry),
            "active_violations": Gauge("ppe_monitor_active_violations", "Active violations count", registry=self._registry),
            "dropped_frames": Gauge("ppe_monitor_dropped_frames_total", "Dropped frames count", registry=self._registry),
            "verifier_calls_last_sec": Gauge("ppe_monitor_verifier_calls_per_sec", "Verifier calls per second", registry=self._registry),
            "compliance_rate": Gauge("ppe_monitor_compliance_rate_pct", "Compliance rate percent", registry=self._registry),
            "estimated_flops_per_sec": Gauge("ppe_monitor_estimated_flops_per_sec", "Estimated FLOP per second", registry=self._registry),
            "estimated_gflops_per_sec": Gauge("ppe_monitor_estimated_gflops_per_sec", "Estimated GFLOP per second", registry=self._registry),
            "estimated_tflops_per_sec": Gauge("ppe_monitor_estimated_tflops_per_sec", "Estimated TFLOP per second", registry=self._registry),
            "estimated_tops_per_sec": Gauge("ppe_monitor_estimated_tops_per_sec", "Estimated TOPS", registry=self._registry),
            "estimated_compute_utilization_pct": Gauge(
                "ppe_monitor_estimated_compute_utilization_pct",
                "Estimated compute utilization percent",
                registry=self._registry,
            ),
            "pose_infer_per_sec": Gauge("ppe_monitor_pose_infer_per_sec", "Pose inferences per second", registry=self._registry),
            "ppe_infer_per_sec": Gauge("ppe_monitor_ppe_infer_per_sec", "PPE inferences per second", registry=self._registry),
            "verifier_aux_infer_per_sec": Gauge(
                "ppe_monitor_verifier_aux_infer_per_sec",
                "Verifier auxiliary inferences per second",
                registry=self._registry,
            ),
            "verifier_crop_infer_per_sec": Gauge(
                "ppe_monitor_verifier_crop_infer_per_sec",
                "Verifier crop-model inferences per second",
                registry=self._registry,
            ),
            "verifier_ollama_calls_per_sec": Gauge(
                "ppe_monitor_verifier_ollama_calls_per_sec",
                "Verifier ollama calls per second",
                registry=self._registry,
            ),
            "process_rss_mb": Gauge("ppe_monitor_process_rss_mb", "Process RSS memory in MB", registry=self._registry),
            "process_vms_mb": Gauge("ppe_monitor_process_vms_mb", "Process VMS memory in MB", registry=self._registry),
            "system_memory_used_mb": Gauge("ppe_monitor_system_memory_used_mb", "System used memory in MB", registry=self._registry),
            "system_memory_total_mb": Gauge("ppe_monitor_system_memory_total_mb", "System total memory in MB", registry=self._registry),
            "system_memory_utilization_pct": Gauge(
                "ppe_monitor_system_memory_utilization_pct",
                "System memory utilization percent",
                registry=self._registry,
            ),
            "event_stream_dropped_total": Gauge(
                "ppe_monitor_event_stream_dropped_total",
                "Dropped event-stream writes count",
                registry=self._registry,
            ),
            "jetson_exporter_available": Gauge(
                "ppe_monitor_jetson_exporter_available",
                "Jetson exporter availability (1=up, 0=down)",
                registry=self._registry,
            ),
            "jetson_cpu_utilization_pct": Gauge(
                "ppe_monitor_jetson_cpu_utilization_pct",
                "Jetson CPU utilization percent from exporter",
                registry=self._registry,
            ),
            "jetson_gpu_utilization_pct": Gauge(
                "ppe_monitor_jetson_gpu_utilization_pct",
                "Jetson GPU utilization percent from exporter",
                registry=self._registry,
            ),
            "jetson_memory_utilization_pct": Gauge(
                "ppe_monitor_jetson_memory_utilization_pct",
                "Jetson memory utilization percent from exporter",
                registry=self._registry,
            ),
            "jetson_memory_used_mb": Gauge(
                "ppe_monitor_jetson_memory_used_mb",
                "Jetson memory used MB from exporter",
                registry=self._registry,
            ),
            "jetson_memory_total_mb": Gauge(
                "ppe_monitor_jetson_memory_total_mb",
                "Jetson memory total MB from exporter",
                registry=self._registry,
            ),
            "jetson_temperature_c": Gauge(
                "ppe_monitor_jetson_temperature_c",
                "Jetson temperature C from exporter",
                registry=self._registry,
            ),
            "jetson_power_w": Gauge(
                "ppe_monitor_jetson_power_w",
                "Jetson power watts from exporter",
                registry=self._registry,
            ),
            "jetson_fan_pwm_pct": Gauge(
                "ppe_monitor_jetson_fan_pwm_pct",
                "Jetson fan PWM percent from exporter",
                registry=self._registry,
            ),
        }

    def update(
        self,
        metrics: Mapping[str, Any],
        *,
        event_stream_dropped: int = 0,
        jetson: Optional[JetsonSnapshot] = None,
    ) -> None:
        if not self.available:
            return
        for key, gauge in self._gauges.items():
            if key == "event_stream_dropped_total":
                gauge.set(float(max(0, int(event_stream_dropped))))
                continue
            if key.startswith("jetson_"):
                continue
            if key not in metrics:
                continue
            value = metrics.get(key, 0.0)
            try:
                gauge.set(float(value))
            except Exception:
                gauge.set(0.0)
        self._update_jetson(jetson)

    def _update_jetson(self, jetson: Optional[JetsonSnapshot]) -> None:
        if jetson is None:
            self._gauges["jetson_exporter_available"].set(0.0)
            return
        self._gauges["jetson_exporter_available"].set(1.0 if jetson.available else 0.0)
        if not jetson.available:
            return
        self._gauges["jetson_cpu_utilization_pct"].set(float(jetson.cpu_utilization_pct))
        self._gauges["jetson_gpu_utilization_pct"].set(float(jetson.gpu_utilization_pct))
        self._gauges["jetson_memory_utilization_pct"].set(float(jetson.memory_utilization_pct))
        self._gauges["jetson_memory_used_mb"].set(float(jetson.memory_used_mb))
        self._gauges["jetson_memory_total_mb"].set(float(jetson.memory_total_mb))
        self._gauges["jetson_temperature_c"].set(float(jetson.temperature_c))
        self._gauges["jetson_power_w"].set(float(jetson.power_w))
        self._gauges["jetson_fan_pwm_pct"].set(float(jetson.fan_pwm_pct))

    def render(self) -> bytes:
        if not self.available:
            return b"# prometheus exporter disabled or prometheus_client unavailable\n"
        return generate_latest(self._registry)
