"""Verifier backends for PPE confirmation (YOLOE and Ollama VLM)."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from ultralytics import YOLO

from .jpeg_utils import encode_jpeg_bytes
from .schemas import VerifierResult, VerifierVerdict


@dataclass
class VerifierContext:
    """Optional context passed from pipeline to verifier backends."""

    person_crop: Optional[np.ndarray] = None
    item_crop: Optional[np.ndarray] = None
    positive_conf: float = 0.0
    negative_conf: float = 0.0
    expected_item: str = ""


class VerifierBase:
    """Base verifier interface."""

    def verify(
        self,
        crop: np.ndarray,
        expected_item: str,
        context: Optional[VerifierContext] = None,
    ) -> VerifierResult:
        raise NotImplementedError


class YOLOEVerifier(VerifierBase):
    """Verifier that checks expected PPE class existence on a person crop."""

    def __init__(self, model: YOLO, conf_threshold: float, imgsz: int) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def verify(
        self,
        crop: np.ndarray,
        expected_item: str,
        context: Optional[VerifierContext] = None,
    ) -> VerifierResult:
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
        expected_norm = _normalize_label(expected_item)
        for idx, class_id in enumerate(cls):
            label = str(names.get(int(class_id), int(class_id)))
            if _normalize_label(label) == expected_norm:
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


class OllamaVLMClient:
    """Minimal Ollama /api/chat client for image verification."""

    def __init__(
        self,
        host: str,
        model: str,
        timeout_seconds: float = 8.0,
        temperature: float = 0.0,
        jpeg_backend: str = "auto",
        jpeg_quality: int = 85,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.jpeg_backend = str(jpeg_backend or "auto").lower()
        self.jpeg_quality = int(jpeg_quality)

    def classify(
        self,
        expected_item: str,
        positive_label: str,
        negative_label: str,
        person_crop: np.ndarray,
        item_crop: np.ndarray,
    ) -> VerifierResult:
        person_b64 = _encode_jpeg_base64(
            person_crop,
            quality=self.jpeg_quality,
            backend=self.jpeg_backend,
        )
        item_b64 = _encode_jpeg_base64(
            item_crop,
            quality=self.jpeg_quality,
            backend=self.jpeg_backend,
        )
        if person_b64 is None or item_b64 is None:
            return VerifierResult(verdict=VerifierVerdict.INDETERMINATE, score=0.0, source="ollama")

        prompt = (
            "You are a strict PPE classifier. "
            f"Target item: {expected_item}. "
            f"Positive class: '{positive_label}'. "
            f"Negative class: '{negative_label}'. "
            "Image #1 is the person ROI. Image #2 is a focused item ROI. "
            "Return JSON only with keys: decision, confidence, reason. "
            "decision must be one of: COMPLIANT, VIOLATION, INDETERMINATE. "
            "Use INDETERMINATE if uncertain, occluded, or blurred."
        )

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [person_b64, item_b64],
                }
            ],
        }

        try:
            raw = self._post_json("/api/chat", payload)
            content = str(raw.get("message", {}).get("content", ""))
            parsed = _parse_json_object(content)
            decision = str(parsed.get("decision", "INDETERMINATE")).upper()
            confidence = float(parsed.get("confidence", 0.0) or 0.0)
            reason = str(parsed.get("reason", "")).strip()
        except Exception:
            return VerifierResult(verdict=VerifierVerdict.INDETERMINATE, score=0.0, source="ollama")

        if decision == "COMPLIANT":
            return VerifierResult(verdict=VerifierVerdict.COMPLIANT, score=confidence, source="ollama")
        if decision == "VIOLATION":
            return VerifierResult(verdict=VerifierVerdict.VIOLATION, score=confidence, source="ollama")
        if reason:
            # Keep deterministic source tag while preserving uncertainty through verdict.
            _ = reason
        return VerifierResult(verdict=VerifierVerdict.INDETERMINATE, score=confidence, source="ollama")

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url=f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as res:
                body = res.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        return json.loads(body)


class HybridVerifier(VerifierBase):
    """Uses YOLOE as fast path and Ollama VLM when detections are ambiguous."""

    def __init__(
        self,
        yoloe: YOLOEVerifier,
        ollama: OllamaVLMClient,
        labels: Dict[str, Dict[str, str]],
        ambiguity_margin: float,
        low_conf_threshold: float,
        enable_vlm: bool = True,
    ) -> None:
        self.yoloe = yoloe
        self.ollama = ollama
        self.labels = labels
        self.ambiguity_margin = ambiguity_margin
        self.low_conf_threshold = low_conf_threshold
        self.enable_vlm = enable_vlm
        # Expose YOLO model so pipeline can still reuse it for ensemble detection.
        self.model = yoloe.model
        self.imgsz = yoloe.imgsz

    def verify(
        self,
        crop: np.ndarray,
        expected_item: str,
        context: Optional[VerifierContext] = None,
    ) -> VerifierResult:
        yoloe_result = self.yoloe.verify(crop, expected_item, context)
        if not self.enable_vlm or context is None:
            return yoloe_result

        label_spec = self.labels.get(expected_item, {})
        positive_label = label_spec.get("positive", expected_item)
        negative_label = label_spec.get("negative", f"not_{expected_item}")

        is_ambiguous = (
            context.negative_conf > 0.0
            and (
                abs(context.positive_conf - context.negative_conf) <= self.ambiguity_margin
                or context.positive_conf < self.low_conf_threshold
            )
        )

        # Escalate to VLM only when there is explicit ambiguity signal.
        if not is_ambiguous:
            return yoloe_result

        person_crop = context.person_crop if context.person_crop is not None else crop
        item_crop = context.item_crop if context.item_crop is not None else crop
        return self.ollama.classify(
            expected_item=expected_item,
            positive_label=positive_label,
            negative_label=negative_label,
            person_crop=person_crop,
            item_crop=item_crop,
        )


class MockVerifier(VerifierBase):
    """Fallback verifier for no-model development mode."""

    def verify(
        self,
        crop: np.ndarray,
        expected_item: str,
        context: Optional[VerifierContext] = None,
    ) -> VerifierResult:
        return VerifierResult(verdict=VerifierVerdict.COMPLIANT, score=1.0, source="mock")


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _encode_jpeg_base64(image: np.ndarray, quality: int = 85, backend: str = "auto") -> Optional[str]:
    if image is None or image.size == 0:
        return None
    encoded = encode_jpeg_bytes(image, quality=int(quality), backend=str(backend))
    if encoded is None:
        return None
    return base64.b64encode(encoded).decode("ascii")


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
