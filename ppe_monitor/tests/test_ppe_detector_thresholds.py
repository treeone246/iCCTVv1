"""Tests for PPE detector per-item confidence thresholds."""

import numpy as np

from app.ppe_detector import YOLOPPEDetector


class _TensorWrap:
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _TensorWrap(xyxy)
        self.conf = _TensorWrap(conf)
        self.cls = _TensorWrap(cls)


class _Result:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeModel:
    def __init__(self, results):
        self._results = results

    def predict(self, **kwargs):
        _ = kwargs
        return self._results


def test_harness_uses_stricter_per_item_threshold() -> None:
    # class 0=harness, class 1=helmet
    xyxy = np.array(
        [
            [10.0, 10.0, 40.0, 40.0],  # harness 0.65 -> should be dropped
            [50.0, 50.0, 80.0, 80.0],  # helmet 0.30 -> should pass global 0.22
        ],
        dtype=np.float32,
    )
    conf = np.array([0.65, 0.30], dtype=np.float32)
    cls = np.array([0, 1], dtype=np.float32)
    names = {0: "harness", 1: "helmet"}
    model = _FakeModel([_Result(_Boxes(xyxy, conf, cls), names)])

    detector = YOLOPPEDetector(
        model=model,  # type: ignore[arg-type]
        conf_threshold=0.22,
        imgsz=640,
        label_aliases={},
        per_item_conf_thresholds={"harness": 0.70},
    )
    out = detector.detect(np.zeros((128, 128, 3), dtype=np.uint8))
    labels = [d.label for d in out]
    assert "harness" not in labels
    assert "helmet" in labels


def test_harness_passes_when_above_07() -> None:
    xyxy = np.array([[10.0, 10.0, 40.0, 40.0]], dtype=np.float32)
    conf = np.array([0.80], dtype=np.float32)
    cls = np.array([0], dtype=np.float32)
    names = {0: "harness"}
    model = _FakeModel([_Result(_Boxes(xyxy, conf, cls), names)])

    detector = YOLOPPEDetector(
        model=model,  # type: ignore[arg-type]
        conf_threshold=0.22,
        imgsz=640,
        label_aliases={},
        per_item_conf_thresholds={"harness": 0.70},
    )
    out = detector.detect(np.zeros((128, 128, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].label == "harness"
