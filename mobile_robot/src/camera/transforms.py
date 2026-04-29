from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from .types import CameraExtrinsics, CameraIntrinsics


def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def resize_image(
    image: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    *,
    scale: Optional[float] = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize by explicit ``(width, height)`` or uniform ``scale``."""
    if size is None and scale is None:
        raise ValueError("either size or scale must be provided")
    if size is not None and scale is not None:
        raise ValueError("provide only one of size or scale")
    if scale is not None:
        height, width = image.shape[:2]
        size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, size, interpolation=interpolation)



def undistort_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Remove lens distortion using OpenCV calibration parameters."""
    if intrinsics.dist_coeffs is None:
        return image.copy()
    return cv2.undistort(image, intrinsics.matrix(), intrinsics.dist_coeffs)



def depth_to_meters(depth_image: np.ndarray, depth_scale: float) -> np.ndarray:
    """Convert raw depth image values to meters."""
    return depth_image.astype(np.float32) * float(depth_scale)



def depth_pixel_to_camera_point(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Back-project one depth pixel into 3D camera coordinates."""
    if depth_m <= 0:
        raise ValueError("depth_m must be positive")
    x = (u - intrinsics.cx) * depth_m / intrinsics.fx
    y = (v - intrinsics.cy) * depth_m / intrinsics.fy
    return np.array([x, y, depth_m], dtype=float)



def apply_extrinsics(point_camera: Iterable[float], extrinsics: CameraExtrinsics) -> np.ndarray:
    """Transform a 3D point from the camera frame into the world frame."""
    p_c = np.asarray(point_camera, dtype=float).reshape(3)
    r = np.asarray(extrinsics.rotation_camera_to_world, dtype=float).reshape(3, 3)
    t = np.asarray(extrinsics.translation_camera_to_world, dtype=float).reshape(3)
    return r @ p_c + t



def camera_to_world(point_camera: Iterable[float], extrinsics: CameraExtrinsics) -> np.ndarray:
    """Alias for ``apply_extrinsics`` for readability in perception code."""
    return apply_extrinsics(point_camera, extrinsics)



def depth_pixel_to_world_point(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    """Back-project a depth pixel and transform it directly into world coordinates."""
    point_camera = depth_pixel_to_camera_point(u, v, depth_m, intrinsics)
    return camera_to_world(point_camera, extrinsics)



def pixels_to_world_on_plane(
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    *,
    plane_normal_world: Iterable[float] = (0.0, 0.0, 1.0),
    plane_offset_world: float = 0.0,
) -> np.ndarray:
    """Intersect a camera ray with a world plane.

    The plane is defined by ``n . X + d = 0`` in the world frame.
    For the common ground plane ``z = 0``, use the defaults.
    """
    ray_camera = np.array(
        [
            (u - intrinsics.cx) / intrinsics.fx,
            (v - intrinsics.cy) / intrinsics.fy,
            1.0,
        ],
        dtype=float,
    )
    ray_camera /= np.linalg.norm(ray_camera)

    r = np.asarray(extrinsics.rotation_camera_to_world, dtype=float).reshape(3, 3)
    t = np.asarray(extrinsics.translation_camera_to_world, dtype=float).reshape(3)
    ray_world = r @ ray_camera

    n = np.asarray(plane_normal_world, dtype=float).reshape(3)
    denom = float(n @ ray_world)
    if abs(denom) < 1e-9:
        raise ValueError("camera ray is parallel to plane")

    scale = -(float(n @ t) + float(plane_offset_world)) / denom
    if scale < 0:
        raise ValueError("plane intersection lies behind the camera")
    return t + scale * ray_world
