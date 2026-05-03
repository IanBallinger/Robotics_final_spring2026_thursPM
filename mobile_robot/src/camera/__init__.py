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
from .tray_detection import HSVBounds, TrayDetectionResult, hsv_mask, isolate_tray_and_find_centroid
from .types import CameraExtrinsics, CameraFrame, CameraIntrinsics, DepthFrame, StreamConfig
from .utils import (
    camera_to_arm_point,
    camera_to_robot_point,
    default_camera_config_path,
    load_camera_to_arm_extrinsics,
    load_camera_to_robot_extrinsics,
    robot_point_to_world,
    rotation_matrix_from_rpy,
)

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
    "HSVBounds",
    "hsv_mask",
    "isolate_tray_and_find_centroid",
    "camera_to_arm_point",
    "camera_to_robot_point",
    "default_camera_config_path",
    "load_camera_to_arm_extrinsics",
    "load_camera_to_robot_extrinsics",
    "OpenCVCamera",
    "pixels_to_world_on_plane",
    "RealSenseCamera",
    "resize_image",
    "robot_point_to_world",
    "rotation_matrix_from_rpy",
    "StreamConfig",
    "TrayDetectionResult",
    "undistort_image",
]
