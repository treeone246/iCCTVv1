"""Tests for simple PPE compliance memory anti-spam behavior."""

from __future__ import annotations

from app.ppe_memory import PPEMemoryConfig, PPEMemoryManager, PersonState


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def step(self, dt: float) -> None:
        self.t += dt


def _full_true_obs() -> dict:
    return {
        "helmet": True,
        "coverall": True,
        "gloves": True,
        "safety_glasses": True,
        "boots": True,
    }


def _update(manager: PPEMemoryManager, clock: _Clock, helmet_value, step: float = 0.1):
    obs = _full_true_obs()
    obs["helmet"] = helmet_value
    mem = manager.update_track("cam_a", 7, obs, bbox=(0, 0, 100, 200))
    clock.step(step)
    return mem


def test_flicker_protection(monkeypatch) -> None:
    import app.ppe_memory as pm

    clock = _Clock()
    monkeypatch.setattr(pm, "time", clock.now)

    cfg = PPEMemoryConfig(
        vote_window_frames=12,
        min_frames_for_decision=6,
        ok_ratio_threshold=0.75,
        violation_ratio_threshold=0.25,
        violation_confirm_sec=0.5,
        compliant_confirm_sec=0.3,
        alert_cooldown_sec=60.0,
    )
    manager = PPEMemoryManager(cfg)

    sequence = [True, False, True, False, None, True, False, True, True, None, True, True]
    states = []
    mem = None
    for val in sequence:
        mem = _update(manager, clock, val)
        states.append(mem.state)

    assert mem is not None
    assert PersonState.VIOLATION_CONFIRMED not in states
    assert mem.should_emit_alert() is False


def test_stable_complete_ppe_to_compliant_confirmed(monkeypatch) -> None:
    import app.ppe_memory as pm

    clock = _Clock()
    monkeypatch.setattr(pm, "time", clock.now)

    cfg = PPEMemoryConfig(
        vote_window_frames=30,
        min_frames_for_decision=10,
        compliant_confirm_sec=0.5,
    )
    manager = PPEMemoryManager(cfg)

    mem = None
    for _ in range(30):
        mem = manager.update_track("cam_a", 7, _full_true_obs(), bbox=(0, 0, 100, 200))
        clock.step(0.1)

    assert mem is not None
    assert mem.state == PersonState.COMPLIANT_CONFIRMED


def test_stable_violation_candidate_then_confirmed_and_single_alert(monkeypatch) -> None:
    import app.ppe_memory as pm

    clock = _Clock()
    monkeypatch.setattr(pm, "time", clock.now)

    cfg = PPEMemoryConfig(
        vote_window_frames=30,
        min_frames_for_decision=10,
        compliant_confirm_sec=0.4,
        violation_confirm_sec=0.6,
        alert_cooldown_sec=60.0,
    )
    manager = PPEMemoryManager(cfg)

    # Stabilize compliant first.
    mem = None
    for _ in range(25):
        mem = manager.update_track("cam_a", 7, _full_true_obs(), bbox=(0, 0, 100, 200))
        clock.step(0.1)
    assert mem is not None
    assert mem.state == PersonState.COMPLIANT_CONFIRMED

    states = []
    for _ in range(35):
        mem = _update(manager, clock, False)
        states.append(mem.state)

    assert PersonState.VIOLATION_CANDIDATE in states
    assert mem.state == PersonState.VIOLATION_CONFIRMED
    assert mem.should_emit_alert() is True


def test_alert_cooldown(monkeypatch) -> None:
    import app.ppe_memory as pm

    clock = _Clock()
    monkeypatch.setattr(pm, "time", clock.now)

    cfg = PPEMemoryConfig(alert_cooldown_sec=5.0)
    manager = PPEMemoryManager(cfg)
    mem = manager.get_or_create("cam_a", 9)
    mem.state = PersonState.VIOLATION_CONFIRMED

    assert mem.should_emit_alert() is True
    assert mem.should_emit_alert() is False

    clock.step(4.0)
    assert mem.should_emit_alert() is False

    clock.step(1.1)
    assert mem.should_emit_alert() is True


def test_recovery_to_compliant_confirmed_after_stable_evidence(monkeypatch) -> None:
    import app.ppe_memory as pm

    clock = _Clock()
    monkeypatch.setattr(pm, "time", clock.now)

    cfg = PPEMemoryConfig(
        vote_window_frames=30,
        min_frames_for_decision=10,
        compliant_confirm_sec=0.5,
        violation_confirm_sec=0.6,
    )
    manager = PPEMemoryManager(cfg)

    # Force a stable violation period.
    mem = None
    for _ in range(35):
        mem = _update(manager, clock, False)
    assert mem is not None
    assert mem.state in (PersonState.VIOLATION_CANDIDATE, PersonState.VIOLATION_CONFIRMED)

    # Recover with stable complete PPE.
    for _ in range(40):
        mem = manager.update_track("cam_a", 7, _full_true_obs(), bbox=(0, 0, 100, 200))
        clock.step(0.1)

    assert mem.state == PersonState.COMPLIANT_CONFIRMED
