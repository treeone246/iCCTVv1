"""Unit tests for estimated compute-power monitor helpers."""

from app.compute_monitor import ComputeProfile, estimate_compute_usage


def test_estimate_compute_usage_disabled_returns_zeroes() -> None:
    profile = ComputeProfile()
    out = estimate_compute_usage(
        enabled=False,
        profile=profile,
        pose_infer_per_sec=20.0,
        ppe_infer_per_sec=20.0,
        verifier_aux_infer_per_sec=20.0,
        verifier_crop_infer_per_sec=20.0,
        verifier_ollama_calls_per_sec=5.0,
    )
    assert out.enabled is False
    assert out.estimated_gflops_per_sec == 0.0
    assert out.estimated_tflops_per_sec == 0.0
    assert out.estimated_tops_per_sec == 0.0
    assert out.estimated_compute_utilization_pct == 0.0


def test_estimate_compute_usage_calculates_expected_values() -> None:
    profile = ComputeProfile(
        pose_gflops_per_infer=10.0,
        ppe_gflops_per_infer=20.0,
        verifier_aux_gflops_per_infer=30.0,
        verifier_crop_gflops_per_infer=40.0,
        verifier_ollama_gflops_per_infer=50.0,
        device_peak_gflops=1000.0,
    )
    out = estimate_compute_usage(
        enabled=True,
        profile=profile,
        pose_infer_per_sec=5.0,
        ppe_infer_per_sec=4.0,
        verifier_aux_infer_per_sec=2.0,
        verifier_crop_infer_per_sec=3.0,
        verifier_ollama_calls_per_sec=1.0,
    )
    # 5*10 + 4*20 + 2*30 + 3*40 + 1*50 = 360 GFLOPS/s
    assert out.enabled is True
    assert out.pose_estimated_gflops_per_sec == 50.0
    assert out.ppe_estimated_gflops_per_sec == 80.0
    assert out.verifier_aux_estimated_gflops_per_sec == 60.0
    assert out.verifier_crop_estimated_gflops_per_sec == 120.0
    assert out.verifier_ollama_estimated_gflops_per_sec == 50.0
    assert out.estimated_gflops_per_sec == 360.0
    assert out.estimated_tflops_per_sec == 0.36
    assert out.estimated_tops_per_sec == 0.36
    assert out.estimated_compute_utilization_pct == 36.0
