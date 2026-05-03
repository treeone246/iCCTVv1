import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Scenario:
    name: str
    distance: str
    camera_angle: str
    lighting: str
    occlusion: str
    motion_blur: str
    bbox_scale_min: float
    bbox_scale_max: float
    conf_min: float
    conf_max: float
    noise_px: float
    miss_prob: float


@dataclass(frozen=True)
class ActivityProfile:
    name: str
    base_posture: str
    body_tilt: float
    torso_shift_y: float
    motion_x: float
    motion_y: float
    arm_swing: float
    leg_swing: float
    crouch_factor: float


COCO17_BASE = [
    (0.00, -0.42),
    (-0.04, -0.45),
    (0.04, -0.45),
    (-0.08, -0.42),
    (0.08, -0.42),
    (-0.12, -0.28),
    (0.12, -0.28),
    (-0.18, -0.05),
    (0.18, -0.05),
    (-0.22, 0.16),
    (0.22, 0.16),
    (-0.08, 0.04),
    (0.08, 0.04),
    (-0.09, 0.30),
    (0.09, 0.30),
    (-0.10, 0.50),
    (0.10, 0.50),
]


SCENARIOS = [
    Scenario(
        name="close_clear_front",
        distance="close",
        camera_angle="front",
        lighting="good",
        occlusion="none",
        motion_blur="low",
        bbox_scale_min=0.42,
        bbox_scale_max=0.60,
        conf_min=0.84,
        conf_max=0.98,
        noise_px=2.2,
        miss_prob=0.015,
    ),
    Scenario(
        name="medium_front",
        distance="medium",
        camera_angle="front",
        lighting="good",
        occlusion="none",
        motion_blur="low",
        bbox_scale_min=0.28,
        bbox_scale_max=0.40,
        conf_min=0.76,
        conf_max=0.92,
        noise_px=3.8,
        miss_prob=0.03,
    ),
    Scenario(
        name="far_front",
        distance="far",
        camera_angle="front",
        lighting="good",
        occlusion="none",
        motion_blur="low",
        bbox_scale_min=0.16,
        bbox_scale_max=0.24,
        conf_min=0.54,
        conf_max=0.80,
        noise_px=6.8,
        miss_prob=0.08,
    ),
    Scenario(
        name="medium_low_light",
        distance="medium",
        camera_angle="front",
        lighting="low",
        occlusion="none",
        motion_blur="low",
        bbox_scale_min=0.27,
        bbox_scale_max=0.38,
        conf_min=0.62,
        conf_max=0.84,
        noise_px=4.6,
        miss_prob=0.07,
    ),
    Scenario(
        name="medium_partial_occlusion",
        distance="medium",
        camera_angle="front",
        lighting="good",
        occlusion="partial",
        motion_blur="low",
        bbox_scale_min=0.27,
        bbox_scale_max=0.37,
        conf_min=0.58,
        conf_max=0.82,
        noise_px=5.2,
        miss_prob=0.11,
    ),
    Scenario(
        name="far_side_blur",
        distance="far",
        camera_angle="side",
        lighting="good",
        occlusion="partial",
        motion_blur="high",
        bbox_scale_min=0.15,
        bbox_scale_max=0.22,
        conf_min=0.42,
        conf_max=0.68,
        noise_px=8.0,
        miss_prob=0.16,
    ),
    Scenario(
        name="close_topdown",
        distance="close",
        camera_angle="top_down",
        lighting="good",
        occlusion="partial",
        motion_blur="low",
        bbox_scale_min=0.40,
        bbox_scale_max=0.56,
        conf_min=0.64,
        conf_max=0.88,
        noise_px=4.4,
        miss_prob=0.09,
    ),
]


ACTIVITY_PROFILES = {
    "standing": ActivityProfile(
        name="standing",
        base_posture="upright",
        body_tilt=0.0,
        torso_shift_y=0.0,
        motion_x=0.5,
        motion_y=0.3,
        arm_swing=0.01,
        leg_swing=0.005,
        crouch_factor=0.0,
    ),
    "walking": ActivityProfile(
        name="walking",
        base_posture="upright",
        body_tilt=0.02,
        torso_shift_y=0.0,
        motion_x=3.5,
        motion_y=0.8,
        arm_swing=0.05,
        leg_swing=0.07,
        crouch_factor=0.0,
    ),
    "sitting": ActivityProfile(
        name="sitting",
        base_posture="sitting",
        body_tilt=0.03,
        torso_shift_y=0.06,
        motion_x=0.6,
        motion_y=0.4,
        arm_swing=0.02,
        leg_swing=0.01,
        crouch_factor=0.35,
    ),
    "manual_working": ActivityProfile(
        name="manual_working",
        base_posture="upright",
        body_tilt=0.04,
        torso_shift_y=0.01,
        motion_x=1.1,
        motion_y=0.6,
        arm_swing=0.08,
        leg_swing=0.015,
        crouch_factor=0.03,
    ),
    "drilling_operation": ActivityProfile(
        name="drilling_operation",
        base_posture="upright",
        body_tilt=0.08,
        torso_shift_y=0.02,
        motion_x=1.4,
        motion_y=1.0,
        arm_swing=0.11,
        leg_swing=0.03,
        crouch_factor=0.06,
    ),
    "pipe_handling": ActivityProfile(
        name="pipe_handling",
        base_posture="upright",
        body_tilt=0.07,
        torso_shift_y=0.01,
        motion_x=2.2,
        motion_y=0.9,
        arm_swing=0.07,
        leg_swing=0.05,
        crouch_factor=0.02,
    ),
    "crouching": ActivityProfile(
        name="crouching",
        base_posture="crouch",
        body_tilt=0.05,
        torso_shift_y=0.10,
        motion_x=0.9,
        motion_y=0.7,
        arm_swing=0.03,
        leg_swing=0.01,
        crouch_factor=0.52,
    ),
    "bending": ActivityProfile(
        name="bending",
        base_posture="bend",
        body_tilt=0.45,
        torso_shift_y=0.06,
        motion_x=0.7,
        motion_y=0.5,
        arm_swing=0.02,
        leg_swing=0.01,
        crouch_factor=0.18,
    ),
    "falling": ActivityProfile(
        name="falling",
        base_posture="transition",
        body_tilt=0.04,
        torso_shift_y=0.0,
        motion_x=1.6,
        motion_y=4.8,
        arm_swing=0.08,
        leg_swing=0.08,
        crouch_factor=0.08,
    ),
    "lying": ActivityProfile(
        name="lying",
        base_posture="lying",
        body_tilt=1.52,
        torso_shift_y=0.10,
        motion_x=0.2,
        motion_y=0.2,
        arm_swing=0.005,
        leg_swing=0.005,
        crouch_factor=0.58,
    ),
}


SUPERVISED_LABELS = [
    "standing",
    "walking",
    "sitting",
    "manual_working",
    "drilling_operation",
    "pipe_handling",
    "crouching",
    "bending",
    "falling",
    "lying",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic pose activity data for drilling-like operations. "
            "Includes labeled and unlabeled sessions with varied camera scenarios."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/pose_dummy_drilling_activity_v2.json",
    )
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--frames-per-session", type=int, default=48)
    parser.add_argument("--labeled-sessions-per-class", type=int, default=18)
    parser.add_argument("--unlabeled-sessions", type=int, default=72)
    return parser.parse_args()


def _rotate(x: float, y: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return x * c - y * s, x * s + y * c


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _session_name(label: str, scenario_name: str, idx: int) -> str:
    return f"{label}_{scenario_name}_{idx:05d}"


def _apply_posture_adjustments(
    x: float,
    y: float,
    kpt_idx: int,
    profile: ActivityProfile,
    phase: float,
    progress: float,
) -> tuple[float, float]:
    if profile.base_posture == "sitting":
        if kpt_idx in (13, 14, 15, 16):
            y -= 0.15
            x += 0.02 if kpt_idx in (14, 16) else -0.02
    if profile.base_posture == "crouch":
        if kpt_idx in (13, 14):
            y -= 0.12
        if kpt_idx in (15, 16):
            y -= 0.26
        if kpt_idx in (11, 12):
            y -= 0.05
    if profile.base_posture == "bend":
        if kpt_idx in (0, 1, 2, 3, 4, 5, 6):
            y += 0.12
            x += 0.03 if kpt_idx in (2, 4, 6) else -0.03
    if profile.base_posture == "lying":
        y += 0.02
    if profile.base_posture == "transition":  # falling
        y += progress * 0.25

    if kpt_idx in (7, 9):
        y += profile.arm_swing * math.sin(phase)
    if kpt_idx in (8, 10):
        y -= profile.arm_swing * math.sin(phase)
    if kpt_idx in (13, 15):
        y -= profile.leg_swing * math.sin(phase)
    if kpt_idx in (14, 16):
        y += profile.leg_swing * math.sin(phase)

    y += profile.crouch_factor * 0.02
    return x, y


def _keypoint_conf(
    scenario: Scenario,
    activity: str,
    kpt_idx: int,
    rng: random.Random,
) -> float:
    c = rng.uniform(scenario.conf_min, scenario.conf_max)

    if kpt_idx in (9, 10, 15, 16):
        c -= rng.uniform(0.03, 0.12)
    if scenario.occlusion == "partial" and kpt_idx in (7, 8, 9, 10):
        c -= rng.uniform(0.07, 0.18)
    if scenario.lighting == "low":
        c -= rng.uniform(0.04, 0.10)
    if scenario.motion_blur == "high":
        c -= rng.uniform(0.06, 0.13)
    if activity in ("drilling_operation", "manual_working", "pipe_handling") and kpt_idx in (
        7,
        8,
        9,
        10,
    ):
        c -= rng.uniform(0.02, 0.08)
    if rng.random() < scenario.miss_prob:
        c = rng.uniform(0.01, 0.15)

    return round(_clip(c, 0.01, 0.99), 3)


def _simulate_frame(
    *,
    activity: str,
    scenario: Scenario,
    frame_index: int,
    frame_count: int,
    session_name: str,
    subject_id: str,
    camera_id: str,
    person_id: int,
    frame_w: int,
    frame_h: int,
    bbox_center_x: float,
    bbox_center_y: float,
    bbox_w: float,
    bbox_h: float,
    velocity_x: float,
    velocity_y: float,
    ts: float,
    rng: random.Random,
    labelled_activity: str,
) -> dict[str, Any]:
    profile = ACTIVITY_PROFILES[activity]
    progress = frame_index / max(1, frame_count - 1)
    phase = progress * 2.0 * math.pi

    cx = bbox_center_x + (velocity_x * frame_index) + rng.uniform(-1.2, 1.2)
    cy = bbox_center_y + (velocity_y * frame_index) + rng.uniform(-1.2, 1.2)

    if activity == "falling":
        cy += progress * frame_h * 0.16

    wobble = 1.0 + 0.025 * math.sin(frame_index / 4.0)
    if activity == "falling":
        wobble += 0.06 * progress
    if activity in ("lying", "sitting"):
        wobble *= 0.97

    bw = bbox_w * wobble
    bh = bbox_h * wobble

    cx = _clip(cx, bw * 0.6, frame_w - bw * 0.6)
    cy = _clip(cy, bh * 0.55, frame_h - bh * 0.55)

    x1 = _clip(cx - bw / 2.0, 0.0, frame_w - 1.0)
    y1 = _clip(cy - bh / 2.0, 0.0, frame_h - 1.0)
    x2 = _clip(cx + bw / 2.0, 0.0, frame_w - 1.0)
    y2 = _clip(cy + bh / 2.0, 0.0, frame_h - 1.0)

    keypoints = []
    for kpt_idx, (lx, ly) in enumerate(COCO17_BASE):
        x, y = _apply_posture_adjustments(
            lx,
            ly + profile.torso_shift_y,
            kpt_idx,
            profile,
            phase,
            progress,
        )

        theta = profile.body_tilt + rng.uniform(-0.02, 0.02)
        if activity == "falling":
            theta += progress * 1.45
        rx, ry = _rotate(x, y, theta)

        px = cx + (rx * bw) + rng.uniform(-scenario.noise_px, scenario.noise_px)
        py = cy + (ry * bh) + rng.uniform(-scenario.noise_px, scenario.noise_px)

        # Falling should look more unstable with scattered keypoints and abrupt noise bursts.
        if activity == "falling":
            if 0.30 <= progress <= 0.75:
                px += rng.uniform(-scenario.noise_px * 4.0, scenario.noise_px * 4.0)
                py += rng.uniform(-scenario.noise_px * 5.0, scenario.noise_px * 5.0)
            if rng.random() < 0.08:
                px += rng.uniform(-scenario.noise_px * 7.0, scenario.noise_px * 7.0)
                py += rng.uniform(-scenario.noise_px * 7.0, scenario.noise_px * 7.0)

        px = _clip(px, 0.0, frame_w - 1.0)
        py = _clip(py, 0.0, frame_h - 1.0)

        conf = _keypoint_conf(scenario, activity, kpt_idx, rng)
        if activity == "falling":
            if 0.30 <= progress <= 0.75:
                conf = round(_clip(conf - rng.uniform(0.08, 0.22), 0.01, 0.99), 3)
            if rng.random() < 0.06:
                conf = round(rng.uniform(0.01, 0.12), 3)
        keypoints.append(
            {
                "kpt_index": kpt_idx,
                "x": round(px, 2),
                "y": round(py, 2),
                "xn": round(px / frame_w, 6),
                "yn": round(py / frame_h, 6),
                "conf": conf,
            }
        )

    model_conf = rng.uniform(scenario.conf_min, scenario.conf_max)
    model_conf -= rng.uniform(0.01, 0.04) if activity in ("falling", "manual_working") else 0.0
    if activity == "falling" and 0.30 <= progress <= 0.75:
        model_conf -= rng.uniform(0.06, 0.18)
    model_conf = round(_clip(model_conf, 0.30, 0.99), 3)

    return {
        "session_name": session_name,
        "activity_label": labelled_activity,
        "synthetic_true_activity": activity,
        "subject_id": subject_id,
        "camera_id": camera_id,
        "scenario": scenario.name,
        "distance": scenario.distance,
        "camera_angle": scenario.camera_angle,
        "lighting": scenario.lighting,
        "occlusion": scenario.occlusion,
        "motion_blur": scenario.motion_blur,
        "frame_index": frame_index,
        "timestamp": round(ts, 3),
        "person_id": person_id,
        "class_id": 0,
        "class_label": "person",
        "model_confidence": model_conf,
        "bbox": {
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "w": round(max(0.0, x2 - x1), 2),
            "h": round(max(0.0, y2 - y1), 2),
        },
        "keypoints": keypoints,
    }


def _random_activity_for_unlabeled(rng: random.Random) -> str:
    picks = [
        "standing",
        "walking",
        "manual_working",
        "drilling_operation",
        "pipe_handling",
        "crouching",
        "bending",
    ]
    return rng.choice(picks)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    frame_w = args.frame_width
    frame_h = args.frame_height
    frames = args.frames_per_session

    subjects = [f"s{i:02d}" for i in range(1, 21)]
    cameras = [
        "cam_drill_floor_a",
        "cam_drill_floor_b",
        "cam_shaker_area",
        "cam_pipe_deck",
        "cam_control_room",
        "cam_stairs",
    ]

    samples: list[dict[str, Any]] = []
    session_meta: list[dict[str, Any]] = []

    base_ts = 1777420000.0
    session_counter = 0

    for label in SUPERVISED_LABELS:
        for _ in range(args.labeled_sessions_per_class):
            scenario = rng.choice(SCENARIOS)
            session_counter += 1

            subject = rng.choice(subjects)
            camera = rng.choice(cameras)
            person_id = rng.randint(1, 40)
            session_name = _session_name(label, scenario.name, session_counter)

            scale = rng.uniform(scenario.bbox_scale_min, scenario.bbox_scale_max)
            bbox_h = frame_h * scale
            width_factor = rng.uniform(0.43, 0.66)
            if scenario.camera_angle == "side":
                width_factor *= rng.uniform(0.73, 0.88)
            if label in ("pipe_handling", "walking"):
                width_factor *= 0.96
            bbox_w = bbox_h * width_factor

            cx = rng.uniform(bbox_w * 0.65, frame_w - bbox_w * 0.65)
            cy = rng.uniform(bbox_h * 0.58, frame_h - bbox_h * 0.58)
            profile = ACTIVITY_PROFILES[label]
            vx = rng.uniform(-profile.motion_x, profile.motion_x)
            vy = rng.uniform(-profile.motion_y, profile.motion_y)

            start_ts = base_ts + (session_counter * 71.0)
            session_meta.append(
                {
                    "session_name": session_name,
                    "activity_label": label,
                    "subject_id": subject,
                    "camera_id": camera,
                    "is_labeled": True,
                    "scenario": scenario.name,
                    "distance": scenario.distance,
                    "camera_angle": scenario.camera_angle,
                    "lighting": scenario.lighting,
                    "occlusion": scenario.occlusion,
                    "motion_blur": scenario.motion_blur,
                    "frames": frames,
                }
            )

            for fidx in range(frames):
                ts = start_ts + (fidx / 30.0)
                sample = _simulate_frame(
                    activity=label,
                    scenario=scenario,
                    frame_index=fidx,
                    frame_count=frames,
                    session_name=session_name,
                    subject_id=subject,
                    camera_id=camera,
                    person_id=person_id,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    bbox_center_x=cx,
                    bbox_center_y=cy,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    velocity_x=vx,
                    velocity_y=vy,
                    ts=ts,
                    rng=rng,
                    labelled_activity=label,
                )
                samples.append(sample)

    # Unlabeled sessions include transitions to mimic noisy real-world activity changes.
    for _ in range(args.unlabeled_sessions):
        scenario = rng.choice(SCENARIOS)
        session_counter += 1
        session_name = _session_name("unlabeled", scenario.name, session_counter)
        subject = rng.choice(subjects)
        camera = rng.choice(cameras)
        person_id = rng.randint(1, 40)

        scale = rng.uniform(scenario.bbox_scale_min, scenario.bbox_scale_max)
        bbox_h = frame_h * scale
        bbox_w = bbox_h * rng.uniform(0.42, 0.64)
        cx = rng.uniform(bbox_w * 0.65, frame_w - bbox_w * 0.65)
        cy = rng.uniform(bbox_h * 0.58, frame_h - bbox_h * 0.58)

        # Build 2-3 phases in each unlabeled session.
        phase_labels = [_random_activity_for_unlabeled(rng) for _ in range(rng.randint(2, 3))]
        cut_points = sorted(rng.sample(range(8, frames - 8), k=len(phase_labels) - 1))
        boundaries = [0] + cut_points + [frames]

        start_ts = base_ts + (session_counter * 71.0)
        session_meta.append(
            {
                "session_name": session_name,
                "activity_label": "unlabeled",
                "subject_id": subject,
                "camera_id": camera,
                "is_labeled": False,
                "scenario": scenario.name,
                "distance": scenario.distance,
                "camera_angle": scenario.camera_angle,
                "lighting": scenario.lighting,
                "occlusion": scenario.occlusion,
                "motion_blur": scenario.motion_blur,
                "phase_labels_hidden": phase_labels,
                "frames": frames,
            }
        )

        for phase_idx, phase_label in enumerate(phase_labels):
            f_start = boundaries[phase_idx]
            f_end = boundaries[phase_idx + 1]
            profile = ACTIVITY_PROFILES[phase_label]
            vx = rng.uniform(-profile.motion_x, profile.motion_x)
            vy = rng.uniform(-profile.motion_y, profile.motion_y)

            for fidx in range(f_start, f_end):
                ts = start_ts + (fidx / 30.0)
                sample = _simulate_frame(
                    activity=phase_label,
                    scenario=scenario,
                    frame_index=fidx,
                    frame_count=frames,
                    session_name=session_name,
                    subject_id=subject,
                    camera_id=camera,
                    person_id=person_id,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    bbox_center_x=cx,
                    bbox_center_y=cy,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    velocity_x=vx,
                    velocity_y=vy,
                    ts=ts,
                    rng=rng,
                    labelled_activity="unlabeled",
                )
                samples.append(sample)

    out = {
        "dataset_name": "pose_activity_drilling_scope_dummy",
        "version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_size": {"width": frame_w, "height": frame_h},
        "activities_supervised": SUPERVISED_LABELS,
        "activities_unlabeled_marker": "unlabeled",
        "notes": [
            "Synthetic data for pipeline prototyping only.",
            "Includes drilling-related worker activities and camera condition variations.",
            "Includes unlabeled sessions to test semi-supervised pseudo-label workflows.",
        ],
        "counts": {
            "num_sessions": len(session_meta),
            "num_samples": len(samples),
            "frames_per_session": frames,
            "num_labeled_sessions": len(
                [m for m in session_meta if m.get("is_labeled") is True]
            ),
            "num_unlabeled_sessions": len(
                [m for m in session_meta if m.get("is_labeled") is False]
            ),
        },
        "session_metadata": session_meta,
        "samples": samples,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Sessions: {out['counts']['num_sessions']}")
    print(f"Samples: {out['counts']['num_samples']}")
    print(f"Labeled sessions: {out['counts']['num_labeled_sessions']}")
    print(f"Unlabeled sessions: {out['counts']['num_unlabeled_sessions']}")


if __name__ == "__main__":
    main()
