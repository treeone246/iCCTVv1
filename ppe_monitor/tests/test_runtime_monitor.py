"""Tests for runtime memory monitor helpers."""

from app.runtime_monitor import read_memory_snapshot


def test_memory_snapshot_disabled_is_zero() -> None:
    snap = read_memory_snapshot(enabled=False)
    assert snap.enabled is False
    assert snap.process_rss_mb == 0.0
    assert snap.system_memory_total_mb == 0.0


def test_memory_snapshot_enabled_has_non_negative_values() -> None:
    snap = read_memory_snapshot(enabled=True)
    assert snap.enabled is True
    assert snap.process_rss_mb >= 0.0
    assert snap.process_vms_mb >= 0.0
    assert snap.system_memory_used_mb >= 0.0
    assert snap.system_memory_total_mb >= 0.0
    assert snap.system_memory_utilization_pct >= 0.0
