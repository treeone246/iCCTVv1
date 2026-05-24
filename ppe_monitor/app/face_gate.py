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

from .gpu_utils import cv2_cuda_available, get_cupy_module, torchvision_nms_available


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
        self.gpu_preprocess = bool(fg_cfg.get("gpu_preprocess", True))
        self.decode_backend = str(fg_cfg.get("decode_backend", "auto")).lower()
        self.nms_backend = str(fg_cfg.get("nms_backend", "auto")).lower()

        self._session = None
        self._ready = False
        self._init_error = ""
        self._last_obs: Dict[int, FaceGateObservation] = {}
        self._last_ts: Dict[int, float] = {}
        self.cache_seconds = float(fg_cfg.get("cache_seconds", 0.2))
        self._cv2_cuda_ready = self.gpu_preprocess and cv2_cuda_available()
        self._cupy_ready = get_cupy_module() is not None and self.decode_backend in {"auto", "cupy"}
        self._torchvision_nms_ready = torchvision_nms_available() and self.nms_backend in {"auto", "torchvision"}

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
        inp = self._preprocess_crop(crop_bgr)

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

        det_chunks: List[np.ndarray] = []
        sx = crop_bgr.shape[1] / float(self.input_size)
        sy = crop_bgr.shape[0] / float(self.input_size)
        for stride in sorted(set(stride_to_scores.keys()) & set(stride_to_bboxes.keys())):
            score_t = stride_to_scores[stride]
            bbox_t = stride_to_bboxes[stride]
            det_arr = self._decode_stride_outputs(score_t, bbox_t, stride=stride, sx=sx, sy=sy)
            if det_arr.size > 0:
                det_chunks.append(det_arr)
        if not det_chunks:
            return []
        dets = np.concatenate(det_chunks, axis=0)
        dets = self._nms(dets, self.nms_iou)
        if dets.size == 0:
            return []
        return [
            (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))
            for row in dets
        ]

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        img: np.ndarray
        if self._cv2_cuda_ready:
            try:
                gpu = cv2.cuda_GpuMat()
                gpu.upload(crop_bgr)
                gpu_resized = cv2.cuda.resize(
                    gpu,
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                img = gpu_resized.download()
            except Exception:
                img = cv2.resize(crop_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(crop_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        inp = img.astype(np.float32)
        inp = (inp - 127.5) / 128.0
        return np.transpose(inp, (2, 0, 1))[None, :, :, :]

    def _decode_stride_outputs(
        self,
        score_t: np.ndarray,
        bbox_t: np.ndarray,
        *,
        stride: int,
        sx: float,
        sy: float,
    ) -> np.ndarray:
        """Vectorized SCRFD stride decode with optional CuPy backend."""
        if score_t.ndim != 4 or bbox_t.ndim != 4:
            return np.empty((0, 5), dtype=np.float32)
        _, s_c, s_h, s_w = score_t.shape
        _, b_c, b_h, b_w = bbox_t.shape
        if s_h != b_h or s_w != b_w or s_h <= 0 or s_w <= 0:
            return np.empty((0, 5), dtype=np.float32)

        anchor_n = max(1, b_c // 4)
        score_view = score_t[0]
        if s_c == anchor_n:
            scores = score_view
        elif s_c == 1:
            scores = np.repeat(score_view, anchor_n, axis=0)
        else:
            scores = score_view[:anchor_n]

        bbox_view = bbox_t[0].reshape(anchor_n, 4, b_h, b_w)
        if self._cupy_ready and self.decode_backend in {"auto", "cupy"}:
            cp = get_cupy_module()
            if cp is not None and (anchor_n * b_h * b_w) >= 2048:
                try:
                    return self._decode_stride_outputs_cupy(cp, scores, bbox_view, stride, sx, sy)
                except Exception:
                    pass
        return self._decode_stride_outputs_numpy(scores, bbox_view, stride, sx, sy)

    def _decode_stride_outputs_numpy(
        self,
        scores: np.ndarray,
        bbox_view: np.ndarray,
        stride: int,
        sx: float,
        sy: float,
    ) -> np.ndarray:
        mask = scores >= self.score_threshold
        if not np.any(mask):
            return np.empty((0, 5), dtype=np.float32)

        a_idx, y_idx, x_idx = np.where(mask)
        conf = scores[a_idx, y_idx, x_idx].astype(np.float32)
        l = bbox_view[a_idx, 0, y_idx, x_idx].astype(np.float32) * float(stride)
        t = bbox_view[a_idx, 1, y_idx, x_idx].astype(np.float32) * float(stride)
        r = bbox_view[a_idx, 2, y_idx, x_idx].astype(np.float32) * float(stride)
        b = bbox_view[a_idx, 3, y_idx, x_idx].astype(np.float32) * float(stride)
        cx = (x_idx.astype(np.float32) + 0.5) * float(stride)
        cy = (y_idx.astype(np.float32) + 0.5) * float(stride)

        x1 = (cx - l) * float(sx)
        y1 = (cy - t) * float(sy)
        x2 = (cx + r) * float(sx)
        y2 = (cy + b) * float(sy)
        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return np.empty((0, 5), dtype=np.float32)

        return np.stack(
            [
                x1[valid],
                y1[valid],
                x2[valid],
                y2[valid],
                conf[valid],
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    def _decode_stride_outputs_cupy(
        self,
        cp: object,
        scores: np.ndarray,
        bbox_view: np.ndarray,
        stride: int,
        sx: float,
        sy: float,
    ) -> np.ndarray:
        xp = cp  # type: ignore[assignment]
        score_cp = xp.asarray(scores)
        bbox_cp = xp.asarray(bbox_view)
        mask = score_cp >= float(self.score_threshold)
        if int(mask.any().item()) == 0:
            return np.empty((0, 5), dtype=np.float32)

        a_idx, y_idx, x_idx = xp.where(mask)
        conf = score_cp[a_idx, y_idx, x_idx].astype(xp.float32)
        l = bbox_cp[a_idx, 0, y_idx, x_idx].astype(xp.float32) * float(stride)
        t = bbox_cp[a_idx, 1, y_idx, x_idx].astype(xp.float32) * float(stride)
        r = bbox_cp[a_idx, 2, y_idx, x_idx].astype(xp.float32) * float(stride)
        b = bbox_cp[a_idx, 3, y_idx, x_idx].astype(xp.float32) * float(stride)
        cx = (x_idx.astype(xp.float32) + xp.float32(0.5)) * float(stride)
        cy = (y_idx.astype(xp.float32) + xp.float32(0.5)) * float(stride)

        x1 = (cx - l) * float(sx)
        y1 = (cy - t) * float(sy)
        x2 = (cx + r) * float(sx)
        y2 = (cy + b) * float(sy)
        valid = (x2 > x1) & (y2 > y1)
        if int(valid.any().item()) == 0:
            return np.empty((0, 5), dtype=np.float32)

        out_cp = xp.stack(
            [
                x1[valid],
                y1[valid],
                x2[valid],
                y2[valid],
                conf[valid],
            ],
            axis=1,
        )
        return xp.asnumpy(out_cp).astype(np.float32, copy=False)

    def _nms(self, dets: np.ndarray, iou_thresh: float) -> np.ndarray:
        if dets.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        if self._torchvision_nms_ready:
            try:
                keep_idx = self._nms_torchvision(dets, iou_thresh)
                return dets[keep_idx]
            except Exception:
                if self.nms_backend == "torchvision":
                    return np.empty((0, 5), dtype=np.float32)
        keep_idx = self._nms_numpy(dets, iou_thresh)
        return dets[keep_idx]

    def _nms_torchvision(self, dets: np.ndarray, iou_thresh: float) -> np.ndarray:
        import torch
        from torchvision.ops import nms as tv_nms

        boxes = torch.as_tensor(dets[:, :4], dtype=torch.float32)
        scores = torch.as_tensor(dets[:, 4], dtype=torch.float32)
        keep = tv_nms(boxes, scores, float(iou_thresh))
        return keep.cpu().numpy().astype(np.int64, copy=False)

    def _nms_numpy(self, dets: np.ndarray, iou_thresh: float) -> np.ndarray:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            denom = areas[i] + areas[rest] - inter
            iou = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0.0)
            order = rest[iou < float(iou_thresh)]

        return np.asarray(keep, dtype=np.int64)


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
