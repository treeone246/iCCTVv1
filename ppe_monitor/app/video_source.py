"""Video source abstraction for webcam, file, and RTSP inputs."""

from typing import Optional, Tuple, Union

import cv2
import numpy as np


class VideoSource:
    """Thin wrapper around OpenCV VideoCapture with frame-drop helpers."""

    def __init__(self, source: Union[int, str], target_fps: float, drop_grab_limit: int = 3) -> None:
        self.source = source
        self.target_fps = target_fps
        self.drop_grab_limit = max(0, int(drop_grab_limit))
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_latest(self, requested_drops: int = 0) -> Tuple[Optional[np.ndarray], int]:
        if self.cap is None:
            raise RuntimeError("Video source not opened.")

        dropped = 0
        max_drops = min(max(0, requested_drops), self.drop_grab_limit)
        for _ in range(max_drops):
            if self.cap.grab():
                dropped += 1
            else:
                break

        ok, frame = self.cap.read()
        if not ok:
            return None, dropped
        return frame, dropped
