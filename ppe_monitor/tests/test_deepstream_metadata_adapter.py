"""Tests for DeepStream metadata adapter without requiring pyds."""

from app.deepstream.metadata_adapter import DsFrameMeta, DsObjectMeta, adapt_frame


def test_adapt_frame_mixed_objects() -> None:
    frame = DsFrameMeta(
        frame_num=12,
        source_id=0,
        pts_ns=2_500_000_000,
        objects=[
            DsObjectMeta(
                class_id=0,
                class_label="person",
                confidence=0.99,
                bbox=(100.0, 50.0, 80.0, 200.0),
                object_id=42,
                source_id=0,
            ),
            DsObjectMeta(
                class_id=1,
                class_label="helmet",
                confidence=0.88,
                bbox=(120.0, 65.0, 30.0, 30.0),
                object_id=43,
                source_id=0,
            ),
        ],
    )
    out = adapt_frame(
        frame,
        camera_id="rig_floor_cam_01",
        label_map={},
        person_classes={"person"},
        ppe_classes={"helmet", "gloves"},
        alias_to_canonical={},
    )
    assert out.frame_id == 12
    assert out.source_id == 0
    assert len(out.persons) == 1
    assert out.persons[0].person_id == 42
    assert out.persons[0].bbox == (100.0, 50.0, 180.0, 250.0)
    assert len(out.ppe_detections) == 1
    assert out.ppe_detections[0].label == "helmet"
    assert out.ppe_detections[0].bbox == (120.0, 65.0, 150.0, 95.0)
    assert out.object_count == 2


def test_adapt_frame_unknown_class_is_ignored_for_ppe() -> None:
    frame = DsFrameMeta(
        frame_num=1,
        source_id=3,
        pts_ns=0,
        objects=[
            DsObjectMeta(
                class_id=99,
                class_label="random_obj",
                confidence=0.4,
                bbox=(1.0, 2.0, 3.0, 4.0),
                object_id=7,
                source_id=3,
            )
        ],
    )
    out = adapt_frame(
        frame,
        camera_id="cam_x",
        label_map={},
        person_classes={"person"},
        ppe_classes={"helmet"},
        alias_to_canonical={},
    )
    assert len(out.persons) == 0
    assert len(out.ppe_detections) == 0
    assert out.class_counts.get("random_obj") == 1
