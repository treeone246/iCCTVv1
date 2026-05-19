"""Integration-style tests for behavior-agent orchestration safety and outputs."""

import json
from pathlib import Path

from app.ai_behavior_agent.agent import BehaviorAgentRunner
from app.ai_behavior_agent.ollama_client import OllamaGenerateResult


def _write_event(path: Path, track_id: int = 1) -> None:
    row = {
        "event_type": "ppe_observation",
        "timestamp": "2026-05-19T12:00:00Z",
        "track_id": track_id,
        "ppe": {
            "helmet": {
                "status": "VIOLATION",
                "status_stable": "INDETERMINATE",
                "positive_conf": 0.35,
                "negative_conf": 0.65,
                "reason": "direct_violation",
                "sm_stage": "VIOLATION_CANDIDATE",
                "alert_status": "CLEARED",
            }
        },
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _make_runner(tmp_path: Path) -> BehaviorAgentRunner:
    events = tmp_path / "detection_events.jsonl"
    _write_event(events, track_id=11)
    return BehaviorAgentRunner(
        events_jsonl=events,
        output_dir=tmp_path / "behavior_out",
        memory_path=tmp_path / "person_behavior_memory.json",
        host="http://127.0.0.1:11434",
        model="qwen3:4b",
        interval_seconds=5.0,
        max_recent_events=100,
        temperature=0.1,
        num_predict=768,
        num_ctx=4096,
        strict_json=True,
        update_memory=True,
        min_identity_confidence=0.65,
        save_training_records=True,
    )


def test_agent_handles_invalid_llm_response_without_memory_update(monkeypatch, tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    memory_path = Path(runner.memory_path)
    memory_path.write_text(json.dumps({"11": {"identity_confidence": 0.8}}), encoding="utf-8")
    original = memory_path.read_text(encoding="utf-8")

    def fake_generate(prompt: str) -> OllamaGenerateResult:
        return OllamaGenerateResult(ok=False, model="qwen3:4b", error="invalid_json_response")

    monkeypatch.setattr(runner.client, "generate_json", fake_generate)
    cycle = runner.run_cycle()
    assert cycle["ok"] is False
    assert cycle["memory_updated"] is False
    assert memory_path.read_text(encoding="utf-8") == original


def test_agent_writes_latest_and_history_with_expected_model(monkeypatch, tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)

    def fake_generate(prompt: str) -> OllamaGenerateResult:
        data = {
            "agent_type": "background_behavior_intelligence",
            "model": "should_be_overridden",
            "generated_at": "2026-05-19T12:00:05Z",
            "summary": "Repeated helmet instability on track 11.",
            "detected_patterns": [
                {
                    "type": "possible_identity_switch",
                    "description": "Track 11 changed behavior rapidly.",
                    "confidence": 0.62,
                    "track_ids": [11],
                }
            ],
            "memory_update_recommendations": [
                {"track_id": 11, "action": "flag_possible_identity_switch", "identity_confidence_delta": 0.02}
            ],
            "dashboard_insights": [{"title": "Identity risk", "detail": "Track 11 needs review."}],
            "training_data_suggestions": [{"track_id": 11, "label_hint": "possible_identity_switch"}],
        }
        return OllamaGenerateResult(ok=True, model="qwen3:4b", data=data)

    monkeypatch.setattr(runner.client, "generate_json", fake_generate)
    cycle = runner.run_cycle()
    assert cycle["ok"] is True

    latest_path = Path(cycle["paths"]["latest_path"])
    history_path = Path(cycle["paths"]["history_path"])
    assert latest_path.exists()
    assert history_path.exists()

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert latest["model"] == "qwen3:4b"
    assert history["model"] == "qwen3:4b"
