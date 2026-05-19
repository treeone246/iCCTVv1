"""Tests for standalone behavior-agent Ollama client."""

from pathlib import Path

from app.ai_behavior_agent.ollama_client import OllamaBehaviorClient


def _build_client(tmp_path: Path) -> OllamaBehaviorClient:
    return OllamaBehaviorClient(
        host="http://127.0.0.1:11434",
        model="qwen3:4b",
        timeout_seconds=1.0,
        output_dir=tmp_path,
    )


def test_ollama_client_parse_success(monkeypatch, tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    def fake_post_json(path: str, payload: dict) -> dict:
        assert path == "/api/generate"
        assert payload["model"] == "qwen3:4b"
        assert payload["think"] is False
        return {"response": "{\"summary\":\"ok\",\"detected_patterns\":[]}"}

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    result = client.generate_json("test prompt")
    assert result.ok is True
    assert result.model == "qwen3:4b"
    assert result.data is not None
    assert result.data.get("summary") == "ok"


def test_ollama_client_invalid_json_creates_debug_artifact(monkeypatch, tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    def fake_post_json(path: str, payload: dict) -> dict:
        return {"response": "not valid json"}

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    result = client.generate_json("test prompt")
    assert result.ok is False
    assert result.model == "qwen3:4b"
    debug_files = list((tmp_path / "debug").glob("invalid_response_*.txt"))
    assert debug_files, "Expected invalid JSON debug artifact to be written."
