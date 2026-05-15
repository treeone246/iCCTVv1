"""YOLOE verifier wrapper with fixed-class ONNX inference behavior."""

from typing import Optional

import numpy as np
from ultralytics import YOLO

from .schemas import VerifierResult, VerifierVerdict


class VerifierBase:
    """Base verifier interface."""

    def verify(self, crop: np.ndarray, expected_item: str) -> VerifierResult:
        raise NotImplementedError


class YOLOEVerifier(VerifierBase):
    """Verifier that checks expected PPE class existence on a person crop."""

    def __init__(self, model: YOLO, conf_threshold: float, imgsz: int) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def verify(self, crop: np.ndarray, expected_item: str) -> VerifierResult:
        results = self.model.predict(
            source=crop,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return VerifierResult(verdict=VerifierVerdict.VIOLATION, score=0.0, source="yoloe")

        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return VerifierResult(verdict=VerifierVerdict.VIOLATION, score=0.0, source="yoloe")

        names = result.names if isinstance(result.names, dict) else {}
        cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
        conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []

        best_match: Optional[float] = None
        for idx, class_id in enumerate(cls):
            label = str(names.get(int(class_id), int(class_id)))
            # TODO: swap to prompt-based inference once prompt-encoder export is usable in ONNX runtime.
            if label == expected_item:
                score = float(conf[idx]) if idx < len(conf) else 0.0
                if best_match is None or score > best_match:
                    best_match = score

        if best_match is not None and best_match >= self.conf_threshold:
            return VerifierResult(
                verdict=VerifierVerdict.COMPLIANT,
                score=best_match,
                source="yoloe",
            )

        return VerifierResult(verdict=VerifierVerdict.VIOLATION, score=0.0, source="yoloe")


class MockVerifier(VerifierBase):
    """Fallback verifier for no-model development mode."""

    def verify(self, crop: np.ndarray, expected_item: str) -> VerifierResult:
        return VerifierResult(verdict=VerifierVerdict.COMPLIANT, score=1.0, source="mock")
