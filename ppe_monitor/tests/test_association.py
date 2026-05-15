"""Unit tests for keypoint-aware PPE association and held checks."""

from app.association import AssociationEngine
from app.schemas import Classification


def make_config() -> dict:
    return {
        "association": {
            "keypoint_conf_floor": 0.4,
            "held_distance_ratio": 1.2,
            "held_items": ["helmet", "goggles", "gloves"],
            "helmet_keypoints": ["nose", "left_eye", "right_eye"],
            "goggles_keypoints": ["left_eye", "right_eye"],
            "gloves_keypoints": ["left_wrist", "right_wrist"],
            "boots_keypoints": ["left_ankle", "right_ankle"],
            "coverall_keypoints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            "helmet_distance_px": 90,
            "goggles_distance_px": 60,
            "gloves_distance_px": 75,
            "boots_distance_px": 85,
            "coverall_iou_threshold": 0.1,
        }
    }


def base_keypoints() -> tuple[dict, dict]:
    keypoints = {
        "nose": (100.0, 100.0),
        "left_eye": (95.0, 95.0),
        "right_eye": (105.0, 95.0),
        "left_wrist": (160.0, 180.0),
        "right_wrist": (180.0, 180.0),
        "left_ankle": (90.0, 260.0),
        "right_ankle": (120.0, 260.0),
        "left_shoulder": (80.0, 130.0),
        "right_shoulder": (120.0, 130.0),
        "left_hip": (85.0, 200.0),
        "right_hip": (115.0, 200.0),
    }
    conf = {name: 0.9 for name in keypoints}
    return keypoints, conf


def test_bound_correctly_returns_compliant() -> None:
    engine = AssociationEngine(make_config())
    keypoints, conf = base_keypoints()
    ppe = [{"label": "helmet", "bbox": (80.0, 70.0, 120.0, 120.0), "conf": 0.9}]
    classification, bind = engine.classify_item("helmet", keypoints, conf, ppe, (300, 300, 3))
    assert classification == Classification.COMPLIANT
    assert bind is not None and bind.bound


def test_bound_but_held_returns_violation() -> None:
    engine = AssociationEngine(make_config())
    keypoints, conf = base_keypoints()
    # Helmet-like box near wrists, far from head.
    ppe = [{"label": "helmet", "bbox": (150.0, 160.0, 190.0, 200.0), "conf": 0.9}]
    classification, bind = engine.classify_item("helmet", keypoints, conf, ppe, (300, 300, 3))
    assert classification == Classification.VIOLATION
    assert bind is not None and bind.held


def test_unbound_and_visible_returns_tentative() -> None:
    engine = AssociationEngine(make_config())
    keypoints, conf = base_keypoints()
    ppe = [{"label": "helmet", "bbox": (230.0, 40.0, 270.0, 80.0), "conf": 0.8}]
    classification, bind = engine.classify_item("helmet", keypoints, conf, ppe, (300, 300, 3))
    assert classification == Classification.VIOLATION_TENTATIVE
    assert bind is None


def test_unbound_and_invisible_returns_indeterminate() -> None:
    engine = AssociationEngine(make_config())
    keypoints, conf = base_keypoints()
    conf["nose"] = 0.1
    conf["left_eye"] = 0.1
    conf["right_eye"] = 0.1
    ppe = []
    classification, bind = engine.classify_item("helmet", keypoints, conf, ppe, (300, 300, 3))
    assert classification == Classification.INDETERMINATE
    assert bind is None
