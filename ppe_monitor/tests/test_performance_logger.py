"""Tests for periodic performance JSONL logger."""

import json
import time
from pathlib import Path

from app.jetson_exporter_bridge import JetsonSnapshot
from app.performance_logger import PerformanceLogWriter


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_performance_logger_writes_snapshot(tmp_path: Path) -> None:
    out = tmp_path / "perf.jsonl"
    writer = PerformanceLogWriter(
        {
            "performance_logging": {
                "enabled": True,
                "path": str(out),
                "interval_seconds": 0.1,
                "include_jetson": True,
                "include_raw_metrics": True,
            }
        }
    )
    writer.emit(
        frame_id=10,
        metrics={
            "fps": 5.5,
            "tracked_count": 2,
            "active_violations": 1,
            "estimated_tops_per_sec": 0.42,
            "process_rss_mb": 123.4,
        },
        jetson=JetsonSnapshot(enabled=True, available=True, cpu_utilization_pct=33.0),
        source="unit_test",
        timestamp=time.time(),
    )
    writer.close()

    lines = _read_lines(out)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "performance_snapshot"
    assert payload["frame_id"] == 10
    assert payload["summary"]["fps"] == 5.5
    assert payload["compute"]["tops_per_sec"] == 0.42
    assert payload["jetson"]["cpu_utilization_pct"] == 33.0
    assert payload["source"] == "unit_test"
    assert "metrics" in payload


def test_performance_logger_respects_interval(tmp_path: Path) -> None:
    out = tmp_path / "perf_interval.jsonl"
    writer = PerformanceLogWriter(
        {
            "performance_logging": {
                "enabled": True,
                "path": str(out),
                "interval_seconds": 3.0,
                "include_jetson": False,
                "include_raw_metrics": False,
            }
        }
    )
    t0 = time.time()
    writer.emit(frame_id=1, metrics={"fps": 1.0}, timestamp=t0)
    writer.emit(frame_id=2, metrics={"fps": 2.0}, timestamp=t0 + 0.5)
    writer.close()

    lines = _read_lines(out)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["frame_id"] == 1
    assert "jetson" not in payload
    assert "metrics" not in payload


def test_performance_logger_records_per_camera_fps(tmp_path: Path) -> None:
    out = tmp_path / "perf_cameras.jsonl"
    writer = PerformanceLogWriter(
        {
            "performance_logging": {
                "enabled": True,
                "path": str(out),
                "interval_seconds": 0.1,
            }
        }
    )
    writer.emit(
        frame_id=3,
        metrics={"fps": 12.3},
        backend="deepstream",
        source_id=1,
        camera_id="rig_floor_cam_02",
        input_fps=15.0,
        per_camera_fps={"rig_floor_cam_01": 14.2, "rig_floor_cam_02": 15.0},
        timestamp=time.time(),
    )
    writer.close()
    lines = _read_lines(out)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["backend"] == "deepstream"
    assert payload["camera_id"] == "rig_floor_cam_02"
    assert payload["source_id"] == 1
    assert payload["input_fps"] == 15.0
    assert payload["per_camera_fps"]["rig_floor_cam_01"] == 14.2
