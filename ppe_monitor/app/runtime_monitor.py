"""Runtime resource monitor for process/system memory snapshots."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


@dataclass
class MemorySnapshot:
    enabled: bool
    process_rss_mb: float = 0.0
    process_vms_mb: float = 0.0
    system_memory_used_mb: float = 0.0
    system_memory_total_mb: float = 0.0
    system_memory_utilization_pct: float = 0.0


def read_memory_snapshot(enabled: bool) -> MemorySnapshot:
    if not enabled:
        return MemorySnapshot(enabled=False)

    rss_mb = 0.0
    vms_mb = 0.0
    sys_used_mb = 0.0
    sys_total_mb = 0.0
    sys_util_pct = 0.0

    if psutil is not None:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        rss_mb = _to_mb(float(getattr(mem, "rss", 0.0)))
        vms_mb = _to_mb(float(getattr(mem, "vms", 0.0)))
        vm = psutil.virtual_memory()
        sys_used_mb = _to_mb(float(getattr(vm, "used", 0.0)))
        sys_total_mb = _to_mb(float(getattr(vm, "total", 0.0)))
        sys_util_pct = float(getattr(vm, "percent", 0.0))
    else:
        rss_mb = _linux_rss_mb_fallback()
        sys_used_mb, sys_total_mb = _linux_system_mem_fallback()
        if sys_total_mb > 0:
            sys_util_pct = (sys_used_mb / sys_total_mb) * 100.0

    return MemorySnapshot(
        enabled=True,
        process_rss_mb=round(rss_mb, 2),
        process_vms_mb=round(vms_mb, 2),
        system_memory_used_mb=round(sys_used_mb, 2),
        system_memory_total_mb=round(sys_total_mb, 2),
        system_memory_utilization_pct=round(sys_util_pct, 2),
    )


def _to_mb(bytes_value: float) -> float:
    return bytes_value / (1024.0 * 1024.0)


def _linux_rss_mb_fallback() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    kb = float(parts[1])
                    return kb / 1024.0
    except Exception:
        return 0.0
    return 0.0


def _linux_system_mem_fallback() -> tuple[float, float]:
    total_kb = 0.0
    available_kb = 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = float(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_kb = float(line.split()[1])
    except Exception:
        return 0.0, 0.0

    used_kb = max(0.0, total_kb - available_kb)
    return used_kb / 1024.0, total_kb / 1024.0
