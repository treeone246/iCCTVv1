"""Unit tests for SCRFD face-gate config/init behavior."""

from app.face_gate import SCRFDFaceGate


def test_face_gate_disabled_returns_disabled_observation() -> None:
    gate = SCRFDFaceGate({"face_gate": {"enabled": False}})
    assert gate.enabled is False


def test_face_gate_plan_path_marks_not_ready() -> None:
    gate = SCRFDFaceGate(
        {
            "face_gate": {
                "enabled": True,
                "scrfd_model_path": "models/scrfd_2.5g_fp16.plan",
            }
        }
    )
    # Current implementation supports ONNX path for built-in inference.
    assert gate.enabled is True
    assert gate._ready is False
    assert "requires_onnx" in gate._init_error
