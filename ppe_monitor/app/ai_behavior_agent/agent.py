"""Background behavior intelligence orchestrator and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .event_reader import read_recent_events
from .memory_reinforcer import apply_memory_recommendations
from .ollama_client import OllamaBehaviorClient
from .prompts import build_prompt
from .schemas import empty_agent_output, sanitize_agent_output
from .storage import BehaviorAgentStorage, safe_read_json, safe_write_json


def _log(event_type: str, **fields: Any) -> None:
    print(json.dumps({"event_type": event_type, **fields}, default=str))


class BehaviorAgentRunner:
    """Runs periodic event-stream analysis and safe memory reinforcement."""

    def __init__(
        self,
        *,
        events_jsonl: str | Path,
        output_dir: str | Path,
        memory_path: str | Path,
        host: str,
        model: str,
        interval_seconds: float,
        max_recent_events: int,
        temperature: float,
        num_predict: int,
        num_ctx: int,
        strict_json: bool,
        update_memory: bool,
        min_identity_confidence: float,
        save_training_records: bool,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.events_jsonl = Path(events_jsonl)
        self.output_dir = Path(output_dir)
        self.memory_path = Path(memory_path)
        self.host = host
        self.model = model
        self.interval_seconds = float(interval_seconds)
        self.max_recent_events = int(max_recent_events)
        self.strict_json = bool(strict_json)
        self.update_memory = bool(update_memory)
        self.min_identity_confidence = float(min_identity_confidence)
        self.save_training_records = bool(save_training_records)
        self.timeout_seconds = float(timeout_seconds)

        self.storage = BehaviorAgentStorage(
            output_dir=self.output_dir,
            save_training_records=self.save_training_records,
        )
        self.client = OllamaBehaviorClient(
            host=self.host,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            output_dir=self.output_dir,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], project_root: Path) -> "BehaviorAgentRunner":
        cfg = dict(config.get("behavior_agent", {}) or {})
        return cls(
            events_jsonl=_resolve_path(cfg.get("events_jsonl", "outputs/detection_events.jsonl"), project_root),
            output_dir=_resolve_path(cfg.get("output_dir", "outputs/behavior_agent"), project_root),
            memory_path=_resolve_path(cfg.get("memory_path", "outputs/person_behavior_memory.json"), project_root),
            host=str(cfg.get("host", "http://127.0.0.1:11434")),
            model=str(cfg.get("model", "qwen3:4b")),
            interval_seconds=float(cfg.get("interval_seconds", 5)),
            max_recent_events=int(cfg.get("max_recent_events", 100)),
            temperature=float(cfg.get("temperature", 0.1)),
            num_predict=int(cfg.get("num_predict", 768)),
            num_ctx=int(cfg.get("num_ctx", 4096)),
            strict_json=bool(cfg.get("strict_json", True)),
            update_memory=bool(cfg.get("update_memory", True)),
            min_identity_confidence=float(cfg.get("min_identity_confidence", 0.65)),
            save_training_records=bool(cfg.get("save_training_records", True)),
            timeout_seconds=float(cfg.get("timeout_seconds", 8.0)),
        )

    def run_cycle(self) -> Dict[str, Any]:
        events, time_window = read_recent_events(self.events_jsonl, self.max_recent_events)
        memory = safe_read_json(self.memory_path, default={})
        if not isinstance(memory, dict):
            memory = {}
        memory_before = json.dumps(memory, sort_keys=True)

        prompt = build_prompt(events=events, time_window=time_window, memory=memory, model=self.model)
        llm = self.client.generate_json(prompt)
        if not llm.ok:
            _log(
                "behavior_agent_cycle_skipped",
                model=self.model,
                reason=llm.error,
                event_count=time_window.get("event_count", 0),
            )
            return {
                "ok": False,
                "model": self.model,
                "error": llm.error,
                "event_count": int(time_window.get("event_count", 0) or 0),
                "memory_updated": False,
            }

        normalized = sanitize_agent_output(llm.data, model=self.model, time_window=time_window)
        if self.strict_json and not isinstance(normalized, dict):
            normalized = empty_agent_output(model=self.model, time_window=time_window)

        saved_paths = self.storage.save_insight(normalized)
        memory_result = apply_memory_recommendations(
            memory=memory,
            recommendations=list(normalized.get("memory_update_recommendations", [])),
            dashboard_insights=list(normalized.get("dashboard_insights", [])),
            training_suggestions=list(normalized.get("training_data_suggestions", [])),
            update_enabled=self.update_memory,
            min_identity_confidence=self.min_identity_confidence,
        )
        memory_after = json.dumps(memory_result.memory, sort_keys=True)
        memory_changed = memory_after != memory_before
        if self.update_memory and memory_changed:
            safe_write_json(self.memory_path, memory_result.memory)

        _log(
            "behavior_agent_cycle_complete",
            model=self.model,
            event_count=time_window.get("event_count", 0),
            applied_updates=memory_result.applied,
            rejected_updates=memory_result.rejected,
            latest_path=saved_paths.get("latest_path"),
        )
        return {
            "ok": True,
            "model": self.model,
            "event_count": int(time_window.get("event_count", 0) or 0),
            "memory_updated": bool(self.update_memory and memory_changed),
            "paths": saved_paths,
        }

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stopper = stop_event or threading.Event()
        while not stopper.is_set():
            try:
                self.run_cycle()
            except Exception as exc:
                _log("behavior_agent_cycle_error", model=self.model, error=str(exc))
            if stopper.wait(timeout=max(0.5, self.interval_seconds)):
                break


class BehaviorAgentService:
    """Optional background-thread wrapper for running the agent inside FastAPI lifespan."""

    def __init__(self, runner: BehaviorAgentRunner) -> None:
        self.runner = runner
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any], project_root: Path) -> "BehaviorAgentService":
        runner = BehaviorAgentRunner.from_config(config=config, project_root=project_root)
        return cls(runner=runner)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self.runner.run_forever,
            kwargs={"stop_event": self._stop},
            name="behavior-agent",
            daemon=True,
        )
        self._thread.start()
        _log("behavior_agent_service_started", model=self.runner.model, interval=self.runner.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        _log("behavior_agent_service_stopped", model=self.runner.model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background AI behavior intelligence agent.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to project config YAML.")
    parser.add_argument("--events-jsonl", type=str, default="", help="Override events JSONL path.")
    parser.add_argument("--interval", type=float, default=0.0, help="Override analysis interval seconds.")
    parser.add_argument("--once", action="store_true", help="Run one analysis cycle and exit.")
    parser.add_argument("--model", type=str, default="", help="Override Ollama model.")
    parser.add_argument("--host", type=str, default="", help="Override Ollama host.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent
    config = load_config(config_path)
    behavior_cfg = dict(config.get("behavior_agent", {}) or {})

    if args.events_jsonl:
        behavior_cfg["events_jsonl"] = args.events_jsonl
    if args.interval > 0:
        behavior_cfg["interval_seconds"] = float(args.interval)
    if args.model:
        behavior_cfg["model"] = args.model
    if args.host:
        behavior_cfg["host"] = args.host

    config = dict(config)
    config["behavior_agent"] = behavior_cfg

    runner = BehaviorAgentRunner.from_config(config=config, project_root=project_root)
    _log(
        "behavior_agent_start",
        model=runner.model,
        host=runner.host,
        events_jsonl=str(runner.events_jsonl),
        interval_seconds=runner.interval_seconds,
        once=bool(args.once),
    )

    if args.once:
        runner.run_cycle()
        return

    stopper = threading.Event()
    try:
        runner.run_forever(stop_event=stopper)
    except KeyboardInterrupt:
        stopper.set()


def _resolve_path(path_value: Any, project_root: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


if __name__ == "__main__":
    main()
