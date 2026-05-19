# Behavior Intelligence Agent

This document describes the background AI behavior intelligence feature added to `ppe_monitor`.

## Scope

- The real-time PPE pipeline remains the source of truth for safety alerts.
- The behavior agent is analytics-only and never overrides the state machine.
- The agent reads text event data only (no image input).

## Phase 1 Event Stream

- Producer: `app/pipeline.py` via `app/event_stream.py`
- Output: `outputs/detection_events.jsonl`
- Event type: `ppe_observation`
- Triggering:
  - raw item status change
  - stabilized item status change
  - state-machine stage change
  - per-track heartbeat

This writer is non-blocking:

- `process_frame` enqueues with `put_nowait`
- a daemon thread performs file I/O
- full queue drops oldest writes from pipeline perspective (never blocks)

## Phase 2 Agent Package

Package path:

- `app/ai_behavior_agent/`

Main modules:

- `event_reader.py`: last-N JSONL event reading, malformed line tolerant
- `prompts.py`: strict prompt + compact event summary
- `ollama_client.py`: `/api/generate`, `think: false`, retry, invalid JSON debug artifact
- `schemas.py`: strict output normalization + safe fallback schema
- `memory_reinforcer.py`: allowlisted memory updates only
- `storage.py`: latest/history JSON and optional training JSONL
- `agent.py`: orchestrator + CLI + optional background service

## Phase 3 API + Dashboard

Read-only FastAPI endpoints:

- `GET /api/behavior-agent/latest`
- `GET /api/behavior-agent/history`
- `GET /api/behavior-agent/memory`

Frontend:

- `static/index.html` and `static/dashboard.js` now include an **AI Behavior Intelligence** panel.

## Config Keys

In `config.yaml`:

- `event_stream.*`
- `behavior_agent.*`

Defaults:

- `event_stream.enabled: true`
- `behavior_agent.enabled: false`

## Run and Test

### 1) Live inference window (event stream generation)

```bash
python live_inference_window.py --source ./videos/simulationTest2.mp4
```

Confirm output file grows:

- `outputs/detection_events.jsonl`

### 2) One-shot behavior analysis

```bash
python -m app.ai_behavior_agent.agent \
  --config config.yaml \
  --events-jsonl outputs/detection_events.jsonl \
  --interval 5 \
  --once \
  --model qwen3:4b \
  --host http://127.0.0.1:11434
```

Confirm outputs:

- `outputs/behavior_agent/latest_behavior_insight.json`
- `outputs/behavior_agent/history/behavior_agent_<timestamp>.json`
- `outputs/person_behavior_memory.json` (only if safe updates were applied)

### 3) Unit tests

```bash
pytest -q
```
