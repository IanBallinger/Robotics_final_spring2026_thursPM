from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pupil_apriltags as apriltag

from camera import RealSenseCamera, bgr_to_gray, undistort_image
from camera.types import CameraFrame, CameraIntrinsics, DepthFrame, StreamConfig


class AprilTagPoseEst:
    """AprilTag pose estimation using frames from a RealSense camera.

    The estimated tag pose is returned in the camera frame as
    ``(rotation_matrix, translation_vector)``.
    """

    def __init__(
        self,
        *,
        realsense_camera: Optional[RealSenseCamera] = None,
        tag_size_m: float = 0.12,
        color_config: StreamConfig = StreamConfig(),
        depth_config: StreamConfig = StreamConfig(),
        align_depth_to_color: bool = True,
    ):
        self.at_detector = apriltag.Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.tag_size = float(tag_size_m)
        self.camera = realsense_camera or RealSenseCamera(
            color_config=color_config,
            depth_config=depth_config,
            align_depth_to_color=align_depth_to_color,
        )

        # key: tag id, value: (rotation matrix, translation vector)
        self.pose_estimate: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def open(self) -> None:
        self.camera.open()

    def close(self) -> None:
        self.camera.close()

    def __enter__(self) -> "AprilTagPoseEst":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _detect_april_tags(
        self,
        frame: CameraFrame,
        *,
        intrinsics: Optional[CameraIntrinsics] = None,
        undistort: bool = True,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        intrinsics = intrinsics or frame.intrinsics
        if intrinsics is None:
            raise ValueError("camera intrinsics are required for AprilTag pose estimation")

        image = frame.color
        if undistort:
            image = undistort_image(image, intrinsics)

        gray = bgr_to_gray(image)
        results = self.at_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
            tag_size=self.tag_size,
        )

        pose_estimate: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for result in results:
            pose_estimate[int(result.tag_id)] = (
                np.asarray(result.pose_R, dtype=float),
                np.asarray(result.pose_t, dtype=float),
            )

        self.pose_estimate = pose_estimate
        return pose_estimate

    def get_pose_estimate(
        self,
        frame: Optional[CameraFrame] = None,
        *,
        undistort: bool = True,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Estimate AprilTag poses.

        When ``frame`` is omitted, this method captures a fresh frame from the
        configured RealSense camera.
        """
        if frame is None:
            frame = self.camera.read()
        return self._detect_april_tags(frame, undistort=undistort)

    def get_pose_estimate_with_depth(
        self,
        frame: Optional[DepthFrame] = None,
        *,
        undistort: bool = True,
    ) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], DepthFrame]:
        """Return AprilTag poses along with the RealSense RGB-D frame used."""
        if frame is None:
            frame = self.camera.read()
        poses = self._detect_april_tags(frame, undistort=undistort)
        return poses, frame


if __name__ == "__main__":
    with AprilTagPoseEst() as april_tag_pose_est:
        while True:
            poses, frame = april_tag_pose_est.get_pose_estimate_with_depth()
            print({tag_id: (r.shape, t.reshape(-1).tolist()) for tag_id, (r, t) in poses.items()})
            if frame.frame_id is not None and frame.frame_id > 10:
                break
