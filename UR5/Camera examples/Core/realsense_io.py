"""
realsense_io.py

Pure-Python RealSense D435i RGBD capture with depth aligned to color.

Outputs per frame:
  - rgb: uint8 HxWx3 (BGR by default, so it plays nicely with OpenCV)
  - depth_m: float32 HxW (meters), aligned to the rgb image
  - intrinsics: dict with fx, fy, cx, cy, width, height, dist_coeffs
  - timestamp_ms: float (sensor timestamp in milliseconds)

Dependencies:
  pip install pyrealsense2 numpy opencv-python

Notes:
  - This file does NOT do segmentation or pose estimation.
  - It focuses on clean, consistent RGB + depth aligned to RGB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise ImportError(
        "pyrealsense2 is not installed. Install it with:\n"
        "  pip install pyrealsense2\n"
        "On some systems you may need Intel RealSense SDK installed as well."
    ) from e


@dataclass
class RealSenseConfig:
    # Common D435i-friendly defaults
    color_width: int = 640
    color_height: int = 480
    color_fps: int = 30

    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30

    # If True, returns RGB as BGR (OpenCV-friendly). If False, returns RGB.
    output_bgr: bool = True

    # Depth scale will be read from the sensor; this is just a sanity clamp.
    min_depth_m: float = 0.15
    max_depth_m: float = 5.0

    # Simple depth filtering (helps a lot for stable point clouds)
    enable_depth_filters: bool = True
    enable_spatial_filter: bool = True
    enable_temporal_filter: bool = True
    enable_hole_filling: bool = True

    # Auto-exposure is usually fine; you can manually set exposure outside this file if needed.
    enable_advanced_mode: bool = False  # Placeholder for future use


class RealSenseIO:
    """
    Manages RealSense pipeline:
      - starts RGB + depth streams
      - aligns depth to color
      - returns numpy arrays + intrinsics per frame
    """

    def __init__(self, cfg: Optional[RealSenseConfig] = None) -> None:
        self.cfg = cfg or RealSenseConfig()

        self._pipeline: Optional[rs.pipeline] = None
        self._profile: Optional[rs.pipeline_profile] = None
        self._align: Optional[rs.align] = None

        self._depth_scale: Optional[float] = None
        self._intrinsics: Optional[Dict[str, float]] = None

        # Filters
        self._spatial = None
        self._temporal = None
        self._hole_filling = None

        self._started = False

    @property
    def depth_scale(self) -> float:
        if self._depth_scale is None:
            raise RuntimeError("RealSense pipeline not started, depth_scale not available.")
        return float(self._depth_scale)

    @property
    def intrinsics(self) -> Dict[str, float]:
        if self._intrinsics is None:
            raise RuntimeError("RealSense pipeline not started, intrinsics not available.")
        return dict(self._intrinsics)

    def start(self) -> None:
        if self._started:
            return

        self._pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(
            rs.stream.color,
            self.cfg.color_width,
            self.cfg.color_height,
            rs.format.rgb8,
            self.cfg.color_fps,
        )
        config.enable_stream(
            rs.stream.depth,
            self.cfg.depth_width,
            self.cfg.depth_height,
            rs.format.z16,
            self.cfg.depth_fps,
        )

        self._profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)

        # Get depth scale (z16 units to meters)
        dev = self._profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        # Get intrinsics for the color stream (since depth is aligned to color)
        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intrinsics = {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "width": float(intr.width),
            "height": float(intr.height),
            "dist_coeffs": [float(v) for v in list(intr.coeffs)],
        }

        # Initialize depth filters
        if self.cfg.enable_depth_filters:
            if self.cfg.enable_spatial_filter:
                self._spatial = rs.spatial_filter()
                # mild defaults; you can tune later
                self._spatial.set_option(rs.option.filter_magnitude, 2)
                self._spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
                self._spatial.set_option(rs.option.filter_smooth_delta, 20)
                self._spatial.set_option(rs.option.holes_fill, 0)

            if self.cfg.enable_temporal_filter:
                self._temporal = rs.temporal_filter()
                self._temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
                self._temporal.set_option(rs.option.filter_smooth_delta, 20)

            if self.cfg.enable_hole_filling:
                self._hole_filling = rs.hole_filling_filter()

        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        assert self._pipeline is not None
        self._pipeline.stop()
        self._pipeline = None
        self._profile = None
        self._align = None
        self._started = False

    def _apply_depth_filters(self, depth_frame: rs.depth_frame) -> rs.depth_frame:
        """
        Apply optional RealSense filters. Each filter returns a frame.
        """
        frame = depth_frame

        if self.cfg.enable_depth_filters:
            if self._spatial is not None:
                frame = self._spatial.process(frame)
            if self._temporal is not None:
                frame = self._temporal.process(frame)
            if self._hole_filling is not None:
                frame = self._hole_filling.process(frame)

        return frame

    def get_frame(
        self,
        timeout_ms: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:
        """
        Returns:
          rgb: uint8 HxWx3 (BGR if cfg.output_bgr else RGB)
          depth_m: float32 HxW (meters), aligned to rgb
          intrinsics: dict (fx, fy, cx, cy, width, height, dist_coeffs)
          timestamp_ms: float
        """
        if not self._started:
            raise RuntimeError("Call start() before get_frame().")
        assert self._pipeline is not None
        assert self._align is not None
        assert self._depth_scale is not None
        assert self._intrinsics is not None

        frames = self._pipeline.wait_for_frames(timeout_ms)
        aligned = self._align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to retrieve aligned color/depth frames.")

        # Apply filters in aligned space (aligned depth frame)
        depth_frame = self._apply_depth_filters(depth_frame)

        # Convert to numpy
        color = np.asanyarray(color_frame.get_data())  # RGB8
        depth_z16 = np.asanyarray(depth_frame.get_data()).astype(np.float32)  # z16 units

        # Convert depth to meters
        depth_m = depth_z16 * float(self._depth_scale)

        # Clamp depth to a sane range and set invalid to 0
        depth_m = np.where(
            (depth_m >= self.cfg.min_depth_m) & (depth_m <= self.cfg.max_depth_m),
            depth_m,
            0.0,
        ).astype(np.float32)

        # Color format conversion
        if self.cfg.output_bgr:
            # RGB -> BGR
            color = color[:, :, ::-1].copy()

        # Timestamp: use color timestamp (they are aligned, so either is fine)
        timestamp_ms = float(color_frame.get_timestamp())

        return color, depth_m, dict(self._intrinsics), timestamp_ms


def _demo() -> None:
    """
    Quick sanity test:
      - Opens camera
      - Shows rgb and depth preview (requires opencv-python)
    """
    import cv2

    cam = RealSenseIO()
    cam.start()
    print("Started RealSense.")
    print("Intrinsics:", cam.intrinsics)
    print("Depth scale (m per unit):", cam.depth_scale)

    try:
        while True:
            rgb, depth_m, intr, ts = cam.get_frame()

            # Depth visualization
            depth_vis = depth_m.copy()
            depth_vis[depth_vis == 0] = np.nan
            # scale for display
            depth_norm = np.nan_to_num(depth_vis / max(1e-6, cam.cfg.max_depth_m))
            depth_img = (np.clip(depth_norm, 0.0, 1.0) * 255.0).astype(np.uint8)

            cv2.imshow("RGB (BGR)", rgb)
            cv2.imshow("Depth (0..max_depth)", depth_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Stopped RealSense.")


if __name__ == "__main__":
    _demo()
