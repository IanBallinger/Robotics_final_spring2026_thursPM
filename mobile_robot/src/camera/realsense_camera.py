from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .types import CameraIntrinsics, DepthFrame, StreamConfig

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - depends on system hardware package
    rs = None


class RealSenseCamera:
    """Intel RealSense RGB-D camera wrapper.

    Requires ``pyrealsense2`` to be installed on the target system.
    """

    def __init__(
        self,
        color_config: StreamConfig = StreamConfig(),
        depth_config: StreamConfig = StreamConfig(),
        align_depth_to_color: bool = True,
    ):
        self.color_config = color_config
        self.depth_config = depth_config
        self.align_depth_to_color = align_depth_to_color

        self._pipeline = None
        self._align = None
        self._profile = None
        self._depth_scale = 1.0
        self._frame_id = 0

    def open(self) -> None:
        if rs is None:
            raise ImportError(
                "pyrealsense2 is required to use RealSenseCamera but is not installed"
            )
        if self._pipeline is not None:
            return

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self.color_config.width,
            self.color_config.height,
            rs.format.bgr8,
            self.color_config.fps,
        )
        config.enable_stream(
            rs.stream.depth,
            self.depth_config.width,
            self.depth_config.height,
            rs.format.z16,
            self.depth_config.fps,
        )

        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        self._pipeline = pipeline
        self._profile = profile
        self._align = rs.align(rs.stream.color) if self.align_depth_to_color else None

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._profile = None
            self._align = None

    def __enter__(self) -> "RealSenseCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def color_intrinsics(self) -> CameraIntrinsics:
        if rs is None or self._profile is None:
            raise RuntimeError("RealSense camera is not open")
        stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()
        return CameraIntrinsics(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            dist_coeffs=np.asarray(intr.coeffs, dtype=float),
        )

    def depth_intrinsics(self) -> CameraIntrinsics:
        if rs is None or self._profile is None:
            raise RuntimeError("RealSense camera is not open")
        stream = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = stream.get_intrinsics()
        return CameraIntrinsics(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            dist_coeffs=np.asarray(intr.coeffs, dtype=float),
        )

    def read(self, timeout_ms: int = 5000) -> DepthFrame:
        if self._pipeline is None:
            self.open()
        assert self._pipeline is not None

        frames = self._pipeline.wait_for_frames(timeout_ms)
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("failed to capture synchronized RealSense color/depth frames")

        color = np.asarray(color_frame.get_data())
        depth = np.asarray(depth_frame.get_data())
        timestamp_s = time.time()
        intrinsics = self.color_intrinsics() if self.align_depth_to_color else self.depth_intrinsics()

        result = DepthFrame(
            color=color,
            depth=depth,
            depth_scale=self._depth_scale,
            aligned_depth=self.align_depth_to_color,
            timestamp_s=timestamp_s,
            frame_id=self._frame_id,
            intrinsics=intrinsics,
        )
        self._frame_id += 1
        return result
