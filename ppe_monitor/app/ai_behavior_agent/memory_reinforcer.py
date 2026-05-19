"""Allowlisted, bounded behavior-memory updates from LLM recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping


DISALLOWED_HINTS = (
    "confirm_violation",
    "confirmed_violation",
    "final_violation",
    "final confirmed violation",
    "delete",
    "remove_memory",
    "override",
    "state_machine",
    "threshold",
    "cooldown",
    "emergency",
    "safety_alert",
)

ALLOWED_ACTIONS = {
    "flag_possible_identity_switch",
    "set_review_needed",
    "mark_violation_unconfirmed",
    "append_anomaly_flag",
    "store_dashboard_insight",
    "store_training_label",
}


@dataclass
class MemoryUpdateResult:
    memory: Dict[str, Any]
    applied: int = 0
    rejected: int = 0
    reasons: List[str] = field(default_factory=list)


def apply_memory_recommendations(
    *,
    memory: Mapping[str, Any] | None,
    recommendations: List[Dict[str, Any]],
    dashboard_insights: List[Dict[str, Any]],
    training_suggestions: List[Dict[str, Any]],
    update_enabled: bool,
    min_identity_confidence: float,
) -> MemoryUpdateResult:
    mutable_memory: Dict[str, Any] = dict(memory or {})
    if not update_enabled:
        return MemoryUpdateResult(memory=mutable_memory)

    result = MemoryUpdateResult(memory=mutable_memory)
    for rec in recommendations:
        if not isinstance(rec, Mapping):
            result.rejected += 1
            result.reasons.append("invalid_recommendation_type")
            continue
        if _contains_disallowed_text(rec):
            result.rejected += 1
            result.reasons.append("disallowed_intent")
            continue

        action = str(rec.get("action", rec.get("type", ""))).strip().lower()
        action = _normalize_action(action)
        if action not in ALLOWED_ACTIONS:
            result.rejected += 1
            result.reasons.append(f"unsupported_action:{action or 'empty'}")
            continue

        track_ids = _extract_track_ids(rec)
        if action in {"store_dashboard_insight", "store_training_label"} and not track_ids:
            track_ids = _infer_track_ids_from_payload(dashboard_insights, training_suggestions)

        if not track_ids:
            result.rejected += 1
            result.reasons.append("missing_track_id")
            continue

        for track_id in track_ids:
            entry = _ensure_track_memory(result.memory, track_id)
            if action == "flag_possible_identity_switch":
                delta = _clamp_float(rec.get("identity_confidence_delta"), low=0.01, high=0.15, default=0.03)
                entry["identity_confidence"] = max(
                    float(min_identity_confidence),
                    float(entry.get("identity_confidence", 1.0)) - delta,
                )
                entry["review_needed"] = True
                entry["possible_identity_switch"] = True
                entry["last_violation_status"] = "unconfirmed"
                _append_unique_flag(entry, "possible_identity_switch")
            elif action == "set_review_needed":
                entry["review_needed"] = True
            elif action == "mark_violation_unconfirmed":
                entry["last_violation_status"] = "unconfirmed"
            elif action == "append_anomaly_flag":
                flag = str(rec.get("anomaly_flag", "anomaly_detected")).strip() or "anomaly_detected"
                _append_unique_flag(entry, flag)
            elif action == "store_dashboard_insight":
                insight = _pick_latest_dashboard_insight(dashboard_insights)
                if insight:
                    entry["latest_dashboard_insight"] = insight
            elif action == "store_training_label":
                labels = _extract_training_labels(training_suggestions)
                if labels:
                    existing = entry.setdefault("suggested_training_labels", [])
                    for label in labels:
                        if label not in existing:
                            existing.append(label)
            result.applied += 1

    # Safe additive enrichment from current cycle outputs.
    latest = _pick_latest_dashboard_insight(dashboard_insights)
    if latest:
        for track_id in _infer_track_ids_from_payload(dashboard_insights, training_suggestions):
            entry = _ensure_track_memory(result.memory, track_id)
            entry["latest_dashboard_insight"] = latest

    labels = _extract_training_labels(training_suggestions)
    if labels:
        for track_id in _infer_track_ids_from_payload(dashboard_insights, training_suggestions):
            entry = _ensure_track_memory(result.memory, track_id)
            existing = entry.setdefault("suggested_training_labels", [])
            for label in labels:
                if label not in existing:
                    existing.append(label)

    return result


def _contains_disallowed_text(rec: Mapping[str, Any]) -> bool:
    flattened = " ".join(str(v).lower() for v in rec.values())
    return any(bad in flattened for bad in DISALLOWED_HINTS)


def _extract_track_ids(rec: Mapping[str, Any]) -> List[str]:
    track_ids: List[str] = []
    if "track_id" in rec:
        track = _track_to_key(rec.get("track_id"))
        if track is not None:
            track_ids.append(track)
    values = rec.get("track_ids")
    if isinstance(values, list):
        for entry in values:
            track = _track_to_key(entry)
            if track is not None and track not in track_ids:
                track_ids.append(track)
    return track_ids


def _infer_track_ids_from_payload(
    dashboard_insights: List[Dict[str, Any]],
    training_suggestions: List[Dict[str, Any]],
) -> List[str]:
    out: List[str] = []
    for item in dashboard_insights:
        track = _track_to_key(item.get("track_id")) if isinstance(item, Mapping) else None
        if track is not None and track not in out:
            out.append(track)
    for item in training_suggestions:
        track = _track_to_key(item.get("track_id")) if isinstance(item, Mapping) else None
        if track is not None and track not in out:
            out.append(track)
    return out


def _pick_latest_dashboard_insight(dashboard_insights: List[Dict[str, Any]]) -> str:
    if not dashboard_insights:
        return ""
    last = dashboard_insights[-1]
    if not isinstance(last, Mapping):
        return ""
    detail = last.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    title = last.get("title")
    if isinstance(title, str):
        return title.strip()
    return ""


def _extract_training_labels(training_suggestions: List[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    for item in training_suggestions:
        if not isinstance(item, Mapping):
            continue
        hint = item.get("label_hint")
        if not isinstance(hint, str):
            continue
        label = hint.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _ensure_track_memory(memory: MutableMapping[str, Any], track_id: str) -> Dict[str, Any]:
    entry = memory.get(track_id)
    if not isinstance(entry, dict):
        entry = {}
        memory[track_id] = entry
    entry.setdefault("identity_confidence", 1.0)
    entry.setdefault("review_needed", False)
    entry.setdefault("possible_identity_switch", False)
    entry.setdefault("last_violation_status", "unknown")
    entry.setdefault("anomaly_flags", [])
    entry.setdefault("suggested_training_labels", [])
    entry.setdefault("latest_dashboard_insight", "")
    return entry


def _append_unique_flag(entry: Dict[str, Any], flag: str) -> None:
    flags = entry.setdefault("anomaly_flags", [])
    if flag not in flags:
        flags.append(flag)


def _track_to_key(value: Any) -> str | None:
    try:
        return str(int(value))
    except Exception:
        return None


def _normalize_action(action: str) -> str:
    action = action.strip().lower()
    aliases = {
        "possible_identity_switch": "flag_possible_identity_switch",
        "identity_switch_suspected": "flag_possible_identity_switch",
        "flag_identity_switch": "flag_possible_identity_switch",
        "review_needed": "set_review_needed",
        "set_review": "set_review_needed",
        "unconfirm_violation": "mark_violation_unconfirmed",
        "mark_unconfirmed": "mark_violation_unconfirmed",
        "anomaly_flag": "append_anomaly_flag",
        "add_anomaly_flag": "append_anomaly_flag",
        "dashboard_insight": "store_dashboard_insight",
        "save_dashboard_insight": "store_dashboard_insight",
        "training_label": "store_training_label",
        "save_training_label": "store_training_label",
    }
    return aliases.get(action, action)


def _clamp_float(value: Any, *, low: float, high: float, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = default
    if out < low:
        return low
    if out > high:
        return high
    return out
