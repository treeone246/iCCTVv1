"""Tests for low-confidence escalation to verifier."""

from types import SimpleNamespace

from app.pipeline import MonitoringPipeline
from app.schemas import Classification


def _make_pipeline_stub() -> MonitoringPipeline:
    p = object.__new__(MonitoringPipeline)
    p.low_conf_escalation_enabled = True
    p.low_conf_escalation_threshold = 0.60
    p.low_conf_escalation_items = {"gloves"}
    p.low_conf_escalation_thresholds = {"gloves": 0.60}
    return p


def test_gloves_low_conf_compliant_is_escalated() -> None:
    p = _make_pipeline_stub()
    bind = SimpleNamespace(bound=True)
    assert (
        p._should_force_verifier_on_low_conf(
            item="gloves",
            base_classification=Classification.COMPLIANT,
            bind=bind,
            positive_conf=0.52,
        )
        is True
    )


def test_gloves_high_conf_compliant_is_not_escalated() -> None:
    p = _make_pipeline_stub()
    bind = SimpleNamespace(bound=True)
    assert (
        p._should_force_verifier_on_low_conf(
            item="gloves",
            base_classification=Classification.COMPLIANT,
            bind=bind,
            positive_conf=0.85,
        )
        is False
    )


def test_non_gloves_not_escalated_by_rule() -> None:
    p = _make_pipeline_stub()
    bind = SimpleNamespace(bound=True)
    assert (
        p._should_force_verifier_on_low_conf(
            item="helmet",
            base_classification=Classification.COMPLIANT,
            bind=bind,
            positive_conf=0.10,
        )
        is False
    )
