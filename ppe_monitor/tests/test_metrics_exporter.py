"""Tests for Prometheus metrics exporter helper."""

from app.metrics_exporter import PrometheusMetricsExporter


def test_metrics_exporter_render_returns_bytes() -> None:
    exporter = PrometheusMetricsExporter(enabled=True)
    exporter.update(
        {
            "fps": 20.0,
            "tracked_count": 2,
            "estimated_flops_per_sec": 1_000_000_000.0,
            "estimated_tops_per_sec": 1.0,
            "process_rss_mb": 512.0,
        },
        event_stream_dropped=0,
    )
    out = exporter.render()
    assert isinstance(out, (bytes, bytearray))
    assert len(out) > 0
