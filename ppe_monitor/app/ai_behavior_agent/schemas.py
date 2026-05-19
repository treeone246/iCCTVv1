"""Schema helpers for behavior-agent output normalization and safety fallbacks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping

AGENT_TYPE = "background_behavior_intelligence"

PATTERN_TYPES = {
    "repeated_ppe_violation",
    "possible_identity_switch",
    "possible_false_violation",
    "alert_flicker",
    "zone_risk_pattern",
}


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_time_window(time_window: Mapping[str, Any] | None) -> Dict[str, Any]:
    default = {"start": None, "end": None, "event_count": 0}
    if not isinstance(time_window, Mapping):
        return default
    out = dict(default)
    start = time_window.get("start")
    end = time_window.get("end")
    count = time_window.get("event_count")
    out["start"] = str(start) if start else None
    out["end"] = str(end) if end else None
    try:
        out["event_count"] = max(0, int(count))
    except Exception:
        out["event_count"] = 0
    return out


def empty_agent_output(model: str, time_window: Mapping[str, Any] | None) -> Dict[str, Any]:
    return {
        "agent_type": AGENT_TYPE,
        "model": str(model),
        "generated_at": utc_iso_now(),
        "time_window": normalize_time_window(time_window),
        "summary": "",
        "detected_patterns": [],
        "memory_update_recommendations": [],
        "dashboard_insights": [],
        "training_data_suggestions": [],
    }


def sanitize_agent_output(
    raw: Any,
    *,
    model: str,
    time_window: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    out = empty_agent_output(model=model, time_window=time_window)
    if not isinstance(raw, Mapping):
        return out

    generated_at = raw.get("generated_at")
    if isinstance(generated_at, str) and generated_at.strip():
        out["generated_at"] = generated_at.strip()

    summary = raw.get("summary")
    if isinstance(summary, str):
        out["summary"] = summary.strip()

    out["detected_patterns"] = _sanitize_patterns(raw.get("detected_patterns"))
    out["memory_update_recommendations"] = _sanitize_object_list(raw.get("memory_update_recommendations"))
    out["dashboard_insights"] = _sanitize_dashboard(raw.get("dashboard_insights"))
    out["training_data_suggestions"] = _sanitize_object_list(raw.get("training_data_suggestions"))

    # Never trust model-declared routing identity/model hints.
    out["agent_type"] = AGENT_TYPE
    out["model"] = str(model)
    out["time_window"] = normalize_time_window(time_window)
    return out


def _sanitize_patterns(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        p_type = str(entry.get("type", "")).strip()
        if p_type not in PATTERN_TYPES:
            continue
        item: Dict[str, Any] = {"type": p_type}
        desc = entry.get("description")
        if isinstance(desc, str) and desc.strip():
            item["description"] = desc.strip()
        conf = _safe_float(entry.get("confidence"))
        if conf is not None:
            item["confidence"] = conf
        tracks = _safe_track_ids(entry.get("track_ids"))
        if tracks:
            item["track_ids"] = tracks
        evidence = entry.get("evidence")
        if isinstance(evidence, str) and evidence.strip():
            item["evidence"] = evidence.strip()
        out.append(item)
    return out


def _sanitize_dashboard(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for entry in value:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                out.append({"title": "Insight", "detail": text})
            continue
        if not isinstance(entry, Mapping):
            continue
        title = entry.get("title")
        detail = entry.get("detail")
        if not isinstance(title, str) or not title.strip():
            title = "Insight"
        if not isinstance(detail, str) or not detail.strip():
            continue
        insight: Dict[str, Any] = {"title": title.strip(), "detail": detail.strip()}
        conf = _safe_float(entry.get("confidence"))
        if conf is not None:
            insight["confidence"] = conf
        out.append(insight)
    return out


def _sanitize_object_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in value:
        if isinstance(entry, Mapping):
            out.append(dict(entry))
    return out


def _safe_track_ids(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    out: List[int] = []
    for entry in value:
        try:
            out.append(int(entry))
        except Exception:
            continue
    return out


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return round(number, 4)
