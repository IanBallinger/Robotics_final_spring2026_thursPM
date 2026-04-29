from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from autonomy.mission_runner import Pose2D, default_tasks_path
from camera import (
    CameraExtrinsics,
    RealSenseCamera,
    StreamConfig,
    camera_to_world,
    depth_pixel_to_camera_point,
    depth_to_meters,
)
from camera.types import DepthFrame

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - depends on system package
    YOLO = None


@dataclass(frozen=True)
class PersonDetection:
    track_id: Optional[int]
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    center_uv: Tuple[int, int]
    depth_m: Optional[float]
    position_camera_m: Optional[np.ndarray]
    position_robot_m: Optional[np.ndarray]
    position_world_m: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PersonDetectionConfig:
    model_path: str = "yolo26n.pt"
    confidence_threshold: float = 0.5
    min_depth_m: float = 0.3
    max_depth_m: float = 6.0
    min_valid_depth_pixels: int = 20
    tracker: Optional[str] = "bytetrack.yaml"
    color_config: StreamConfig = StreamConfig(width=640, height=480, fps=15)
    depth_config: StreamConfig = StreamConfig(width=640, height=480, fps=15)
    align_depth_to_color: bool = True


class PersonDetector:
    """YOLO-based person detection with RealSense depth and camera-frame transforms."""

    def __init__(
        self,
        *,
        config: PersonDetectionConfig = PersonDetectionConfig(),
        camera: Optional[RealSenseCamera] = None,
        camera_extrinsics: Optional[CameraExtrinsics] = None,
        mission_config_path: Optional[Path | str] = None,
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics is required to use PersonDetector but is not installed"
            )

        self.config = config
        self.camera = camera or RealSenseCamera(
            color_config=config.color_config,
            depth_config=config.depth_config,
            align_depth_to_color=config.align_depth_to_color,
        )
        self.camera_extrinsics = camera_extrinsics or load_camera_to_robot_extrinsics(
            mission_config_path
        )
        self.model = YOLO(config.model_path)

    def open(self) -> None:
        self.camera.open()

    def close(self) -> None:
        self.camera.close()

    def __enter__(self) -> "PersonDetector":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def detect(
        self,
        frame: Optional[DepthFrame] = None,
        *,
        robot_pose: Optional[Pose2D] = None,
        persist_tracks: bool = True,
    ) -> List[PersonDetection]:
        if frame is None:
            frame = self.camera.read()
        if frame.depth is None or frame.intrinsics is None:
            raise ValueError("depth frame with camera intrinsics is required")

        if self.config.tracker:
            results = self.model.track(
                frame.color,
                classes=[0],
                conf=self.config.confidence_threshold,
                tracker=self.config.tracker,
                verbose=False,
                persist=persist_tracks,
            )
        else:
            results = self.model(
                frame.color,
                classes=[0],
                conf=self.config.confidence_threshold,
                verbose=False,
            )

        depth_m = depth_to_meters(frame.depth, frame.depth_scale)
        detections: List[PersonDetection] = []
        image_h, image_w = frame.color.shape[:2]

        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1 = max(0, min(x1, image_w - 1))
            x2 = max(0, min(x2, image_w))
            y1 = max(0, min(y1, image_h - 1))
            y2 = max(0, min(y2, image_h))
            if x2 <= x1 or y2 <= y1:
                continue

            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            track_id = int(box.id[0]) if box.id is not None else None
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            depth_value = self._estimate_depth(depth_m, x1, y1, x2, y2)
            point_camera = None
            point_robot = None
            point_world = None
            if depth_value is not None:
                point_camera = depth_pixel_to_camera_point(
                    cx, cy, depth_value, frame.intrinsics
                )
                point_robot = camera_to_world(point_camera, self.camera_extrinsics)
                if robot_pose is not None:
                    point_world = robot_point_to_world(point_robot, robot_pose)

            detections.append(
                PersonDetection(
                    track_id=track_id,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_uv=(cx, cy),
                    depth_m=depth_value,
                    position_camera_m=point_camera,
                    position_robot_m=point_robot,
                    position_world_m=point_world,
                )
            )

        return detections

    def detect_with_frame(
        self,
        frame: Optional[DepthFrame] = None,
        *,
        robot_pose: Optional[Pose2D] = None,
        persist_tracks: bool = True,
    ) -> tuple[List[PersonDetection], DepthFrame]:
        if frame is None:
            frame = self.camera.read()
        return self.detect(frame, robot_pose=robot_pose, persist_tracks=persist_tracks), frame

    def render_debug(
        self,
        frame: DepthFrame,
        detections: Sequence[PersonDetection],
    ) -> np.ndarray:
        display = frame.color.copy()
        if frame.depth is None:
            return display

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(frame.depth, alpha=0.03), cv2.COLORMAP_JET
        )

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            has_depth = det.depth_m is not None and det.position_robot_m is not None
            color = (0, 255, 0) if has_depth else (0, 165, 255)
            if has_depth:
                x_r, y_r, z_r = det.position_robot_m.tolist()
                label = (
                    f"id={det.track_id if det.track_id is not None else -1} "
                    f"d={det.depth_m:.2f}m robot=({x_r:+.2f},{y_r:+.2f},{z_r:+.2f})"
                )
            else:
                label = f"id={det.track_id if det.track_id is not None else -1} (no depth)"
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return np.hstack([display, depth_colormap])

    def _estimate_depth(
        self,
        depth_m: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Optional[float]:
        width = x2 - x1
        height = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        sample_half_w = max(2, width // 6)
        sample_half_h = max(2, height // 6)
        sx1 = max(0, cx - sample_half_w)
        sx2 = min(depth_m.shape[1], cx + sample_half_w)
        sy1 = max(0, cy - sample_half_h)
        sy2 = min(depth_m.shape[0], cy + sample_half_h)

        depth_crop = depth_m[sy1:sy2, sx1:sx2]
        valid = depth_crop[
            (depth_crop > self.config.min_depth_m)
            & (depth_crop < self.config.max_depth_m)
        ]
        if valid.size < self.config.min_valid_depth_pixels:
            return None
        return float(np.median(valid))


def load_camera_to_robot_extrinsics(
    mission_config_path: Optional[Path | str] = None,
) -> CameraExtrinsics:
    path = Path(mission_config_path) if mission_config_path is not None else default_tasks_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    camera_raw = raw.get("camera_to_robot", {})

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



def rotation_matrix_from_rpy(*, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx



def robot_point_to_world(point_robot: np.ndarray, robot_pose: Pose2D) -> np.ndarray:
    x_r, y_r, z_r = np.asarray(point_robot, dtype=float).reshape(3)
    c = float(np.cos(robot_pose.heading))
    s = float(np.sin(robot_pose.heading))
    x_w = robot_pose.x + c * x_r - s * y_r
    y_w = robot_pose.y + s * x_r + c * y_r
    return np.array([x_w, y_w, z_r], dtype=float)


if __name__ == "__main__":
    with PersonDetector() as detector:
        while True:
            detections, frame = detector.detect_with_frame()
            combined = detector.render_debug(frame, detections)
            cv2.imshow("Person detection", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
