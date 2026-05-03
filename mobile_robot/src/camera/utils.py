from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import yaml

from .transforms import apply_extrinsics
from .types import CameraExtrinsics

if TYPE_CHECKING:
    from autonomy.mission_runner import Pose2D


DEFAULT_CAMERA_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "camera_config.yaml"
)


def default_camera_config_path() -> Path:
    return DEFAULT_CAMERA_CONFIG_PATH



def _load_named_camera_extrinsics(
    field_name: str,
    config_path: Optional[Path | str] = None,
) -> CameraExtrinsics:
    path = Path(config_path) if config_path is not None else default_camera_config_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    camera_raw = raw.get(field_name, {})

    if "translation" in camera_raw or "rotation_rpy" in camera_raw:
        translation_raw = camera_raw.get("translation", {})
        rotation_raw = camera_raw.get("rotation_rpy", {})
        tx = float(translation_raw.get("x", 0.0))
        ty = float(translation_raw.get("y", 0.0))
        tz = float(translation_raw.get("z", 0.0))
        roll = float(rotation_raw.get("roll", 0.0))
        pitch = float(rotation_raw.get("pitch", 0.0))
        yaw = float(rotation_raw.get("yaw", 0.0))
    else:
        tx = float(camera_raw.get("x", 0.0))
        ty = float(camera_raw.get("y", 0.0))
        tz = float(camera_raw.get("z", 0.0))
        roll = float(camera_raw.get("roll", 0.0))
        pitch = float(camera_raw.get("pitch", 0.0))
        yaw = float(camera_raw.get("yaw", 0.0))

    rotation = rotation_matrix_from_rpy(roll=roll, pitch=pitch, yaw=yaw)
    translation = np.array([tx, ty, tz], dtype=float)
    return CameraExtrinsics(
        rotation_camera_to_world=rotation,
        translation_camera_to_world=translation,
    )



def load_camera_to_robot_extrinsics(
    mission_config_path: Optional[Path | str] = None,
) -> CameraExtrinsics:
    return _load_named_camera_extrinsics("camera_to_robot", mission_config_path)



def load_camera_to_arm_extrinsics(
    config_path: Optional[Path | str] = None,
) -> CameraExtrinsics:
    return _load_named_camera_extrinsics("camera_to_arm", config_path)



def _camera_optical_to_nominal_frame(point_camera: np.ndarray) -> np.ndarray:
    """Convert camera optical-frame coordinates into the nominal forward/left/up frame."""
    p_c = np.asarray(point_camera, dtype=float).reshape(3)
    return np.array([p_c[2], -p_c[0], -p_c[1]], dtype=float)



def camera_to_robot_point(point_camera: np.ndarray, extrinsics: CameraExtrinsics) -> np.ndarray:
    """Transform a 3D point from the camera optical frame into the robot/body frame.

    Optical-frame convention from the RGB-D / AprilTag stack is assumed to be:
    - +x right
    - +y down
    - +z forward

    Robot/body-frame convention used in this repo is assumed to be:
    - +x forward
    - +y left
    - +z up

    The configured extrinsics are interpreted as the pose of this nominal
    forward/left/up camera frame in the robot frame.
    """
    point_camera_nominal = _camera_optical_to_nominal_frame(point_camera)
    return apply_extrinsics(point_camera_nominal, extrinsics)



def camera_to_arm_point(point_camera: np.ndarray, extrinsics: CameraExtrinsics) -> np.ndarray:
    """Transform a 3D point from the camera optical frame into the 2-link arm base frame.

    This uses the same optical-to-nominal camera conversion as ``camera_to_robot_point``:
    - camera optical: +x right, +y down, +z forward
    - nominal mounted-camera frame: +x forward, +y left, +z up

    The configured ``camera_to_arm`` extrinsics are interpreted as the pose of
    that nominal camera frame in the arm base frame.
    """
    point_camera_nominal = _camera_optical_to_nominal_frame(point_camera)
    return apply_extrinsics(point_camera_nominal, extrinsics)



def rotation_matrix_from_rpy(*, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx



def robot_point_to_world(point_robot: np.ndarray, robot_pose: "Pose2D") -> np.ndarray:
    x_r, y_r, z_r = np.asarray(point_robot, dtype=float).reshape(3)
    c = float(np.cos(robot_pose.heading))
    s = float(np.sin(robot_pose.heading))
    x_w = robot_pose.x + c * x_r - s * y_r
    y_w = robot_pose.y + s * x_r + c * y_r
    return np.array([x_w, y_w, z_r], dtype=float)
