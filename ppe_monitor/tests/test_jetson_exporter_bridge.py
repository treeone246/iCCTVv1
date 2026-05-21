"""Tests for Jetson exporter bridge parsing and normalization."""

from app.jetson_exporter_bridge import JetsonExporterBridge, JetsonExporterConfig, parse_prometheus_text


def test_parse_prometheus_text_collects_values() -> None:
    raw = """
# HELP jetson_gpu_utilization GPU util
# TYPE jetson_gpu_utilization gauge
jetson_gpu_utilization 45
jetson_cpu_utilization{core="0"} 20
jetson_cpu_utilization{core="1"} 40
"""
    out = parse_prometheus_text(raw)
    assert out["jetson_gpu_utilization"] == [45.0]
    assert out["jetson_cpu_utilization"] == [20.0, 40.0]


def test_bridge_snapshot_uses_metric_map_and_averages(monkeypatch) -> None:
    bridge = JetsonExporterBridge(
        JetsonExporterConfig(
            enabled=True,
            url="http://127.0.0.1:9100/metrics",
            timeout_seconds=1.0,
            metric_map={},
        )
    )
    sample = """
jetson_gpu_utilization 55
jetson_cpu_utilization{core="0"} 30
jetson_cpu_utilization{core="1"} 50
jetson_memory_utilization 60
jetson_memory_used_mb 4096
jetson_memory_total_mb 8192
jetson_temperature_c 64
jetson_power_w 11.2
jetson_fan_pwm_pct 25
"""

    monkeypatch.setattr(bridge, "_fetch_text", lambda url, timeout_seconds: sample)
    snap = bridge.read_snapshot()
    assert snap.enabled is True
    assert snap.available is True
    assert snap.gpu_utilization_pct == 55.0
    assert snap.cpu_utilization_pct == 40.0
    assert snap.memory_utilization_pct == 60.0
    assert snap.memory_used_mb == 4096.0


def test_bridge_local_mode_uses_local_jtop_reader(monkeypatch) -> None:
    bridge = JetsonExporterBridge(
        JetsonExporterConfig(
            enabled=True,
            mode="local_jtop",
            url="http://127.0.0.1:9100/metrics",
            timeout_seconds=1.0,
            refresh_seconds=0.1,
            fallback_to_local_jtop=True,
            metric_map={},
        )
    )

    from app.jetson_exporter_bridge import JetsonSnapshot

    monkeypatch.setattr(
        bridge,
        "_read_from_local_jtop",
        lambda: JetsonSnapshot(
            enabled=True,
            available=True,
            source_url="local_jtop",
            cpu_utilization_pct=33.0,
            gpu_utilization_pct=44.0,
            memory_utilization_pct=55.0,
        ),
    )
    snap = bridge.read_snapshot()
    assert snap.available is True
    assert snap.source_url == "local_jtop"
    assert snap.cpu_utilization_pct == 33.0
