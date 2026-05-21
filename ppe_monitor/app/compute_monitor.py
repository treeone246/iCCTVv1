"""Estimated compute power monitor for runtime model-inference activity.

This module provides a lightweight GFLOPS/s estimator based on:
- per-model estimated GFLOPs per inference
- observed inference calls per second

It reports *estimated* compute load, not exact hardware counters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComputeProfile:
    """Per-model inference compute estimates (GFLOPs per call)."""

    pose_gflops_per_infer: float = 8.5
    ppe_gflops_per_infer: float = 20.0
    verifier_aux_gflops_per_infer: float = 20.0
    device_peak_gflops: float = 0.0


@dataclass
class ComputeEstimate:
    """Current estimated compute usage snapshot."""

    enabled: bool
    pose_infer_per_sec: float = 0.0
    ppe_infer_per_sec: float = 0.0
    verifier_aux_infer_per_sec: float = 0.0
    pose_estimated_gflops_per_sec: float = 0.0
    ppe_estimated_gflops_per_sec: float = 0.0
    verifier_aux_estimated_gflops_per_sec: float = 0.0
    estimated_gflops_per_sec: float = 0.0
    estimated_tflops_per_sec: float = 0.0
    estimated_compute_utilization_pct: float = 0.0


def estimate_compute_usage(
    *,
    enabled: bool,
    profile: ComputeProfile,
    pose_infer_per_sec: float,
    ppe_infer_per_sec: float,
    verifier_aux_infer_per_sec: float,
) -> ComputeEstimate:
    if not enabled:
        return ComputeEstimate(enabled=False)

    pose_gflops_sec = max(0.0, pose_infer_per_sec) * max(0.0, profile.pose_gflops_per_infer)
    ppe_gflops_sec = max(0.0, ppe_infer_per_sec) * max(0.0, profile.ppe_gflops_per_infer)
    verifier_aux_gflops_sec = max(0.0, verifier_aux_infer_per_sec) * max(0.0, profile.verifier_aux_gflops_per_infer)
    total_gflops_sec = pose_gflops_sec + ppe_gflops_sec + verifier_aux_gflops_sec
    total_tflops_sec = total_gflops_sec / 1000.0

    utilization = 0.0
    if profile.device_peak_gflops > 0.0:
        utilization = min(100.0, (total_gflops_sec / profile.device_peak_gflops) * 100.0)

    return ComputeEstimate(
        enabled=True,
        pose_infer_per_sec=round(float(pose_infer_per_sec), 2),
        ppe_infer_per_sec=round(float(ppe_infer_per_sec), 2),
        verifier_aux_infer_per_sec=round(float(verifier_aux_infer_per_sec), 2),
        pose_estimated_gflops_per_sec=round(pose_gflops_sec, 2),
        ppe_estimated_gflops_per_sec=round(ppe_gflops_sec, 2),
        verifier_aux_estimated_gflops_per_sec=round(verifier_aux_gflops_sec, 2),
        estimated_gflops_per_sec=round(total_gflops_sec, 2),
        estimated_tflops_per_sec=round(total_tflops_sec, 4),
        estimated_compute_utilization_pct=round(utilization, 2),
    )
