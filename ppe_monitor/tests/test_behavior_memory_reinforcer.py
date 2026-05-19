"""Safety tests for allowlisted behavior-memory updates."""

from app.ai_behavior_agent.memory_reinforcer import apply_memory_recommendations


def test_memory_reinforcer_applies_allowlisted_updates_only() -> None:
    memory = {"7": {"identity_confidence": 0.9, "review_needed": False, "anomaly_flags": []}}
    recommendations = [
        {
            "track_id": 7,
            "action": "flag_possible_identity_switch",
            "identity_confidence_delta": 0.05,
            "reason": "frequent track jumps",
        },
        {
            "track_id": 7,
            "action": "append_anomaly_flag",
            "anomaly_flag": "alert_flicker",
        },
        {
            "track_id": 7,
            "action": "change_thresholds",
            "reason": "override detector thresholds now",
        },
    ]
    result = apply_memory_recommendations(
        memory=memory,
        recommendations=recommendations,
        dashboard_insights=[{"track_id": 7, "detail": "Review swap risk around helmet."}],
        training_suggestions=[{"track_id": 7, "label_hint": "possible_false_violation"}],
        update_enabled=True,
        min_identity_confidence=0.65,
    )

    assert result.applied >= 2
    assert result.rejected >= 1
    updated = result.memory["7"]
    assert updated["review_needed"] is True
    assert updated["possible_identity_switch"] is True
    assert updated["last_violation_status"] == "unconfirmed"
    assert updated["identity_confidence"] >= 0.65
    assert "possible_identity_switch" in updated["anomaly_flags"]
    assert "alert_flicker" in updated["anomaly_flags"]
    assert updated["latest_dashboard_insight"] != ""
    assert "possible_false_violation" in updated["suggested_training_labels"]


def test_memory_reinforcer_rejects_final_confirmed_violation_action() -> None:
    result = apply_memory_recommendations(
        memory={"3": {"last_violation_status": "unknown"}},
        recommendations=[
            {
                "track_id": 3,
                "action": "mark_final_confirmed_violation",
                "reason": "confirm violation immediately",
            }
        ],
        dashboard_insights=[],
        training_suggestions=[],
        update_enabled=True,
        min_identity_confidence=0.65,
    )
    assert result.applied == 0
    assert result.rejected == 1
    assert result.memory["3"]["last_violation_status"] != "confirmed"
