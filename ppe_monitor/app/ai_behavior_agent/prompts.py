"""Prompt builder for the text-only behavior intelligence agent."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping


def build_prompt(
    *,
    events: List[Dict[str, Any]],
    time_window: Mapping[str, Any],
    memory: Mapping[str, Any],
    model: str,
) -> str:
    summary = _summarize_events(events)
    memory_text = _compact_json(memory, max_chars=6000)
    schema_template = {
        "agent_type": "background_behavior_intelligence",
        "model": model,
        "generated_at": "ISO-8601 UTC timestamp",
        "time_window": {"start": "ISO-8601 or null", "end": "ISO-8601 or null", "event_count": 0},
        "summary": "short paragraph",
        "detected_patterns": [
            {
                "type": "repeated_ppe_violation | possible_identity_switch | possible_false_violation | alert_flicker | zone_risk_pattern",
                "description": "what happened based only on event data",
                "confidence": 0.0,
                "track_ids": [1],
                "evidence": "brief evidence statement",
            }
        ],
        "memory_update_recommendations": [
            {
                "track_id": 1,
                "action": "flag_possible_identity_switch | set_review_needed | mark_violation_unconfirmed | append_anomaly_flag",
                "reason": "why this is safe and evidence-based",
                "identity_confidence_delta": 0.03,
                "anomaly_flag": "possible_identity_switch",
            }
        ],
        "dashboard_insights": [{"title": "Insight", "detail": "operational insight", "confidence": 0.0}],
        "training_data_suggestions": [{"track_id": 1, "label_hint": "possible_false_violation", "reason": "why"}],
    }

    return (
        "You are an AI Behavior Intelligence Agent for a CCTV PPE compliance system. "
        f"You are using {model} as a text reasoning model. Analyze only the provided event data and memory. "
        "Do not hallucinate. Do not make final safety decisions. Do not claim visual details unless they are "
        "explicitly present in the event data. Find repeated violations, possible false detections, possible "
        "identity switches, alert flicker, and zone-risk patterns. Return strict JSON only using the required schema. "
        "If evidence is weak, lower confidence and recommend human review instead of confirming a violation.\n\n"
        "Critical constraints:\n"
        "- You may not confirm a final safety violation.\n"
        "- You may not delete person memory.\n"
        "- You may not override the state machine.\n"
        "- You may not change detector thresholds or emergency/safety alerts.\n"
        "- Reserved-null fields (reid_similarity, zone, risk_context, person_memory_id) are unknown: never invent them.\n"
        "- Treat status vs status_stable divergence as possible detector flicker / false positives.\n"
        "- Treat repeated sm_stage ladder changes as potential alert spam/flicker signals.\n"
        "- Use positive_conf and negative_conf to down-rank low-confidence conclusions.\n\n"
        f"Time window metadata: {json.dumps(dict(time_window), ensure_ascii=True)}\n\n"
        f"Event summary:\n{summary}\n\n"
        f"Current behavior memory JSON:\n{memory_text}\n\n"
        "Return JSON object only (no markdown, no prose outside JSON), following this exact key structure:\n"
        f"{json.dumps(schema_template, ensure_ascii=True)}"
    )


def _summarize_events(events: Iterable[Mapping[str, Any]]) -> str:
    events_list = list(events)
    if not events_list:
        return "No events available in this window."

    track_item_counts: Dict[tuple[int, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    track_divergence: Dict[int, int] = defaultdict(int)
    low_conf_signals: Dict[int, int] = defaultdict(int)
    stage_changes: Dict[tuple[int, str], int] = defaultdict(int)
    prev_stage: Dict[tuple[int, str], str] = {}
    prev_raw: Dict[tuple[int, str], str] = {}
    raw_flips: Dict[tuple[int, str], int] = defaultdict(int)

    tail_lines: List[str] = []
    for event in events_list:
        track_id = _to_int(event.get("track_id"))
        timestamp = str(event.get("timestamp", ""))
        ppe = event.get("ppe", {})
        if not isinstance(ppe, Mapping):
            continue
        for item, details in ppe.items():
            if not isinstance(details, Mapping):
                continue
            raw = str(details.get("status", ""))
            stable = str(details.get("status_stable", ""))
            stage = str(details.get("sm_stage", ""))
            pos = _to_float(details.get("positive_conf"))
            neg = _to_float(details.get("negative_conf"))

            key = (track_id, str(item))
            track_item_counts[key][stable or raw or "UNKNOWN"] += 1
            if raw and stable and raw != stable:
                track_divergence[track_id] += 1
            if max(pos, neg) < 0.55:
                low_conf_signals[track_id] += 1
            if key in prev_stage and prev_stage[key] != stage:
                stage_changes[key] += 1
            if key in prev_raw and prev_raw[key] != raw:
                raw_flips[key] += 1
            prev_stage[key] = stage
            prev_raw[key] = raw

            tail_lines.append(
                f"{timestamp} track={track_id} item={item} raw={raw} stable={stable} "
                f"stage={stage} pos={pos:.3f} neg={neg:.3f}"
            )

    lines: List[str] = [f"events={len(events_list)}"]
    for (track_id, item), state_counts in sorted(track_item_counts.items()):
        counts = ", ".join(f"{k}:{v}" for k, v in sorted(state_counts.items()))
        flips = raw_flips.get((track_id, item), 0)
        stage_jump = stage_changes.get((track_id, item), 0)
        lines.append(
            f"track={track_id} item={item} states[{counts}] raw_flips={flips} stage_changes={stage_jump}"
        )

    for track_id in sorted(track_divergence):
        lines.append(
            f"track={track_id} divergence_count={track_divergence[track_id]} "
            f"low_conf_signals={low_conf_signals.get(track_id, 0)}"
        )

    lines.append("recent_event_tail:")
    for line in tail_lines[-40:]:
        lines.append(line)
    return "\n".join(lines)


def _compact_json(value: Mapping[str, Any], max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1
