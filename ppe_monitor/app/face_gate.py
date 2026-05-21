"""SCRFD-based face visibility gate for goggles decisions.

Supports shadow mode and optional enforcement override.
Current inference backend: ONNX Runtime SCRFD model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


@dataclass
class FaceGateObservation:
    enabled: bool
    available: bool
    face_visible: bool
    best_confidence: float = 0.0
    face_count: int = 0
    best_bbox: Optional[Tuple[float, float, float, float]] = None
    reason: str = ""


class SCRFDFaceGate:
    """Face gate that uses SCRFD detections to decide face visibility."""

    def __init__(self, config: dict) -> None:
        fg_cfg = dict(config.get("face_gate", {}) or {})
        self.enabled = bool(fg_cfg.get("enabled", False))
        self.item = str(fg_cfg.get("target_item", "goggles"))
        self.shadow_mode = bool(fg_cfg.get("shadow_mode", True))
        self.enforce = bool(fg_cfg.get("enforce", False))
        self.log_enabled = bool(fg_cfg.get("log_enabled", True))
        self.log_every_n_frames = max(1, int(fg_cfg.get("log_every_n_frames", 10)))
        self.min_confidence = float(fg_cfg.get("min_confidence", 0.45))
        self.min_face_area_ratio = float(fg_cfg.get("min_face_area_ratio", 0.02))
        self.input_size = int(fg_cfg.get("input_size", 640))
        self.nms_iou = float(fg_cfg.get("nms_iou_threshold", 0.4))
        self.score_threshold = float(fg_cfg.get("score_threshold", 0.35))
        self.model_path = str(fg_cfg.get("scrfd_model_path", "models/det_2.5g.onnx"))

        self._session = None
        self._ready = False
        self._init_error = ""
        self._last_obs: Dict[int, FaceGateObservation] = {}
        self._last_ts: Dict[int, float] = {}
        self.cache_seconds = float(fg_cfg.get("cache_seconds", 0.2))

        if not self.enabled:
            return
        if ort is None:
            self._init_error = "onnxruntime_not_available"
            return

        model = Path(self.model_path)
        if not model.is_absolute():
            model = Path.cwd() / model
        model = model.resolve()
        if model.suffix.lower() != ".onnx":
            self._init_error = "scrfd_backend_requires_onnx_model"
            return
        if not model.exists():
            self._init_error = f"scrfd_model_missing:{model.as_posix()}"
            return

        providers = []
        try:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        except Exception:
            providers = ["CPUExecutionProvider"]

        try:
            self._session = ort.InferenceSession(model.as_posix(), providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            self._ready = True
        except Exception as exc:
            self._init_error = f"scrfd_session_init_failed:{exc}"
            self._ready = False

    def should_log(self, frame_id: int) -> bool:
        return self.log_enabled and (frame_id % self.log_every_n_frames == 0)

    def observe(self, frame: np.ndarray, person_id: int, person_bbox: Tuple[float, float, float, float]) -> FaceGateObservation:
        if not self.enabled:
            return FaceGateObservation(False, False, False, reason="face_gate_disabled")
        if not self._ready or self._session is None:
            return FaceGateObservation(True, False, False, reason=self._init_error or "scrfd_not_ready")

        now = time.monotonic()
        cached = self._last_obs.get(person_id)
        if cached is not None and (now - self._last_ts.get(person_id, 0.0)) < self.cache_seconds:
            return cached

        crop, offset = self._crop_person_roi(frame, person_bbox)
        if crop.size == 0:
            obs = FaceGateObservation(True, True, False, reason="empty_person_roi")
            self._cache(person_id, now, obs)
            return obs

        dets = self._run_scrfd(crop)
        face_count = len(dets)
        if face_count == 0:
            obs = FaceGateObservation(True, True, False, reason="no_face_detected")
            self._cache(person_id, now, obs)
            return obs

        dets = self._nms(dets, self.nms_iou)
        best = max(dets, key=lambda d: d[4])
        bx1, by1, bx2, by2, conf = best
        area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        crop_area = float(max(1, crop.shape[0] * crop.shape[1]))
        area_ratio = area / crop_area
        visible = (conf >= self.min_confidence) and (area_ratio >= self.min_face_area_ratio)

        ox, oy = offset
        best_global = (bx1 + ox, by1 + oy, bx2 + ox, by2 + oy)
        reason = "face_visible" if visible else "face_small_or_low_conf"
        obs = FaceGateObservation(
            enabled=True,
            available=True,
            face_visible=visible,
            best_confidence=float(conf),
            face_count=int(face_count),
            best_bbox=best_global,
            reason=reason,
        )
        self._cache(person_id, now, obs)
        return obs

    def _cache(self, person_id: int, ts: float, obs: FaceGateObservation) -> None:
        self._last_obs[person_id] = obs
        self._last_ts[person_id] = ts

    def _crop_person_roi(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = person_bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = bw * 0.08
        pad_y = bh * 0.08
        ix1 = int(max(0, min(w - 1, x1 - pad_x)))
        iy1 = int(max(0, min(h - 1, y1 - pad_y)))
        ix2 = int(max(1, min(w, x2 + pad_x)))
        iy2 = int(max(1, min(h, y2 + pad_y)))
        if ix2 <= ix1 or iy2 <= iy1:
            return np.empty((0, 0, 3), dtype=np.uint8), (float(ix1), float(iy1))
        return frame[iy1:iy2, ix1:ix2], (float(ix1), float(iy1))

    def _run_scrfd(self, crop_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        img = cv2.resize(crop_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        inp = img.astype(np.float32)
        inp = (inp - 127.5) / 128.0
        inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]

        outputs = self._session.run(None, {self._input_name: inp})  # type: ignore[attr-defined]
        out_map = {name: arr for name, arr in zip(self._output_names, outputs)}  # type: ignore[attr-defined]

        stride_to_scores: Dict[int, np.ndarray] = {}
        stride_to_bboxes: Dict[int, np.ndarray] = {}
        for name, arr in out_map.items():
            lname = name.lower()
            stride = _extract_stride(name) or _infer_stride_from_shape(arr.shape, self.input_size)
            if "score" in lname:
                stride_to_scores[stride] = arr
            elif "bbox" in lname:
                stride_to_bboxes[stride] = arr

        dets: List[Tuple[float, float, float, float, float]] = []
        sx = crop_bgr.shape[1] / float(self.input_size)
        sy = crop_bgr.shape[0] / float(self.input_size)
        for stride in sorted(set(stride_to_scores.keys()) & set(stride_to_bboxes.keys())):
            score_t = stride_to_scores[stride]
            bbox_t = stride_to_bboxes[stride]
            dets.extend(self._decode_stride_outputs(score_t, bbox_t, stride=stride, sx=sx, sy=sy))
        return dets

    def _decode_stride_outputs(
        self,
        score_t: np.ndarray,
        bbox_t: np.ndarray,
        *,
        stride: int,
        sx: float,
        sy: float,
    ) -> List[Tuple[float, float, float, float, float]]:
        dets: List[Tuple[float, float, float, float, float]] = []
        if score_t.ndim != 4 or bbox_t.ndim != 4:
            return dets
        # Typical SCRFD format:
        # score: [1, A, H, W] where A is anchors (often 2)
        # bbox : [1, A*4, H, W]
        _, s_c, s_h, s_w = score_t.shape
        _, b_c, b_h, b_w = bbox_t.shape
        if s_h != b_h or s_w != b_w or s_h <= 0 or s_w <= 0:
            return dets

        anchor_n = max(1, b_c // 4)
        score_view = score_t[0]
        if s_c == anchor_n:
            scores = score_view
        elif s_c == 1:
            scores = np.repeat(score_view, anchor_n, axis=0)
        else:
            scores = score_view[:anchor_n]

        bbox_view = bbox_t[0].reshape(anchor_n, 4, b_h, b_w)
        for a in range(anchor_n):
            for y in range(b_h):
                for x in range(b_w):
                    conf = float(scores[a, y, x])
                    if conf < self.score_threshold:
                        continue
                    l = float(bbox_view[a, 0, y, x]) * stride
                    t = float(bbox_view[a, 1, y, x]) * stride
                    r = float(bbox_view[a, 2, y, x]) * stride
                    b = float(bbox_view[a, 3, y, x]) * stride
                    cx = (x + 0.5) * stride
                    cy = (y + 0.5) * stride
                    x1 = (cx - l) * sx
                    y1 = (cy - t) * sy
                    x2 = (cx + r) * sx
                    y2 = (cy + b) * sy
                    if x2 <= x1 or y2 <= y1:
                        continue
                    dets.append((x1, y1, x2, y2, conf))
        return dets

    def _nms(self, dets: List[Tuple[float, float, float, float, float]], iou_thresh: float) -> List[Tuple[float, float, float, float, float]]:
        if not dets:
            return []
        dets_sorted = sorted(dets, key=lambda d: d[4], reverse=True)
        keep: List[Tuple[float, float, float, float, float]] = []
        while dets_sorted:
            best = dets_sorted.pop(0)
            keep.append(best)
            rem: List[Tuple[float, float, float, float, float]] = []
            for d in dets_sorted:
                if _iou(best, d) < iou_thresh:
                    rem.append(d)
            dets_sorted = rem
        return keep


def _extract_stride(name: str) -> Optional[int]:
    tail = name.split("_")[-1]
    try:
        stride = int(tail)
    except Exception:
        return None
    if stride in (8, 16, 32, 64):
        return stride
    return None


def _infer_stride_from_shape(shape: tuple[int, ...], input_size: int) -> int:
    if len(shape) != 4:
        return 8
    h = shape[2]
    if h <= 0:
        return 8
    stride = int(round(float(input_size) / float(h)))
    if stride in (8, 16, 32, 64):
        return stride
    return 8


def _iou(a: Tuple[float, float, float, float, float], b: Tuple[float, float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = a_area + b_area - inter
    if den <= 0.0:
        return 0.0
    return inter / den
