"""Camera interfaces, frame containers, and vision-space transforms.

This package provides:
- OpenCV camera capture for RGB/mono cameras
- Intel RealSense RGB-D capture when ``pyrealsense2`` is installed
- reusable image and geometry transforms for camera/world conversions

These utilities are intended to support downstream perception tasks such as
AprilTag pose estimation, tray detection, and dishwasher-handle localization.
"""

from .opencv_camera import OpenCVCamera
from .realsense_camera import RealSenseCamera
from .transforms import (
    apply_extrinsics,
    bgr_to_gray,
    camera_to_world,
    depth_pixel_to_world_point,
    depth_to_meters,
    depth_pixel_to_camera_point,
    pixels_to_world_on_plane,
    resize_image,
    undistort_image,
)
from .types import CameraExtrinsics, CameraFrame, CameraIntrinsics, DepthFrame, StreamConfig

__all__ = [
    "apply_extrinsics",
    "bgr_to_gray",
    "CameraExtrinsics",
    "CameraFrame",
    "CameraIntrinsics",
    "camera_to_world",
    "DepthFrame",
    "depth_pixel_to_world_point",
    "depth_to_meters",
    "depth_pixel_to_camera_point",
    "OpenCVCamera",
    "pixels_to_world_on_plane",
    "RealSenseCamera",
    "resize_image",
    "StreamConfig",
    "undistort_image",
]
