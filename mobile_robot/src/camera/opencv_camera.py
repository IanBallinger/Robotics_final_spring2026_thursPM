from __future__ import annotations

import time
from typing import Optional

import cv2

from .types import CameraFrame, CameraIntrinsics, StreamConfig


class OpenCVCamera:
    """Simple RGB camera wrapper around ``cv2.VideoCapture``."""

    def __init__(
        self,
        device_index: int = 0,
        config: StreamConfig = StreamConfig(),
        backend: Optional[int] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ):
        self.device_index = device_index
        self.config = config
        self.backend = backend
        self.intrinsics = intrinsics
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_id = 0

    def open(self) -> None:
        if self._capture is not None:
            return
        if self.backend is None:
            capture = cv2.VideoCapture(self.device_index)
        else:
            capture = cv2.VideoCapture(self.device_index, self.backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"failed to open camera device {self.device_index}")
        self._capture = capture

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "OpenCVCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self) -> CameraFrame:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("failed to capture frame from OpenCV camera")
        result = CameraFrame(
            color=frame,
            timestamp_s=time.time(),
            frame_id=self._frame_id,
            intrinsics=self.intrinsics,
        )
        self._frame_id += 1
        return result
