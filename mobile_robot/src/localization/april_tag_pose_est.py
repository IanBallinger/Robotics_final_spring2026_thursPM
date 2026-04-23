import cv2
import numpy as np
import pupil_apriltags as apriltag
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def circular_mean(angles: List[float], weights: Optional[List[float]] = None) -> float:
    if len(angles) == 0:
        raise ValueError("angles must be non-empty")
    if weights is None:
        weights = [1.0] * len(angles)

    s = sum(w * np.sin(a) for a, w in zip(angles, weights))
    c = sum(w * np.cos(a) for a, w in zip(angles, weights))
    return float(np.arctan2(s, c))


class AprilTagGlobalPoseEstimator:
    """
    Global frame:
        x = horizontal floor axis
        y = forward floor axis
        z = vertical up axis

    Camera/OpenCV frame:
        x_cam = right
        y_cam = down
        z_cam = forward

    Tag map:
        x, y, z = tag center in global frame
        yaw     = tag facing direction in global x-y plane

    Tag yaw convention:
        yaw = 0       -> tag front faces +global x
        yaw = pi/2    -> tag front faces +global y
        yaw = pi      -> tag front faces -global x
        yaw = -pi/2   -> tag front faces -global y
    """

    def __init__(
        self,
        tag_map: Dict[int, Dict[str, float]],
        calibration_file: str = "camera_calibration_live.npz",
        tag_size: float = 0.17,
        camera_in_robot: Optional[Dict[str, float]] = None,
        family: str = "tag36h11",
    ):
        self.tag_map = tag_map
        self.tag_size = float(tag_size)
        self.family = family

        self._load_camera_calibration(calibration_file)

        self.detector = apriltag.Detector(
            families=self.family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        # robot/body frame: x forward, y left, z up
        if camera_in_robot is None:
            camera_in_robot = {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }

        self.camera_in_robot = {
            "x": float(camera_in_robot.get("x", 0.0)),
            "y": float(camera_in_robot.get("y", 0.0)),
            "z": float(camera_in_robot.get("z", 0.0)),
        }

        self._new_camera_matrix = None
        self._undistort_roi = None
        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_shape = None

    def _load_camera_calibration(self, calibration_file: str) -> None:
        calib_path = Path(calibration_file)

        if not calib_path.is_absolute():
            here = Path(__file__).resolve().parent
            candidate_here = here / calibration_file
            candidate_cwd = Path.cwd() / calibration_file

            if candidate_here.exists():
                calib_path = candidate_here
            elif candidate_cwd.exists():
                calib_path = candidate_cwd

        if not calib_path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {calib_path}. "
                f"Put camera_calibration_live.npz in the repo root or in src/localization/."
            )

        calibration_data = np.load(str(calib_path))
        self.camera_matrix = calibration_data["camera_matrix"]
        self.dist_coeffs = calibration_data["dist_coeffs"]

        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])

    def _prepare_undistortion(self, frame_shape: Tuple[int, int, int]) -> None:
        h, w = frame_shape[:2]
        shape_key = (h, w)

        if self._undistort_shape == shape_key:
            return

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            0,
            (w, h),
        )

        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_camera_matrix,
            (w, h),
            cv2.CV_16SC2,
        )

        self._new_camera_matrix = new_camera_matrix
        self._undistort_roi = roi
        self._undistort_map1 = map1
        self._undistort_map2 = map2
        self._undistort_shape = shape_key

    def _undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        self._prepare_undistortion(frame.shape)
        undistorted = cv2.remap(
            frame,
            self._undistort_map1,
            self._undistort_map2,
            interpolation=cv2.INTER_LINEAR,
        )
        return undistorted

    def _detect_tags(self, frame: np.ndarray) -> Tuple[List[Any], np.ndarray]:
        undistorted = self._undistort_frame(frame)
        display_frame = undistorted.copy()
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        fx = float(self._new_camera_matrix[0, 0])
        fy = float(self._new_camera_matrix[1, 1])
        cx = float(self._new_camera_matrix[0, 2])
        cy = float(self._new_camera_matrix[1, 2])

        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=self.tag_size,
        )
        return results, display_frame

    def _build_R_global_tag(self, tag_yaw: float) -> np.ndarray:
        """
        Build the 3x3 rotation matrix whose columns are tag axes in global coords.

        Tag-frame axes chosen here:
            tag +z : tag front / facing direction in global x-y
            tag +x : tag right direction in global x-y
            tag +y : downward on the printed tag = global -z

        This gives a right-handed frame:
            x_tag × y_tag = z_tag
        """
        z_tag_global = np.array(
            [np.cos(tag_yaw), np.sin(tag_yaw), 0.0],
            dtype=float,
        )   
        x_tag_global = np.array(
            [np.sin(tag_yaw), -np.cos(tag_yaw), 0.0],
            dtype=float,
        )
        y_tag_global = np.array(
            [0.0, 0.0, -1.0],
            dtype=float,
        )

        R_global_tag = np.column_stack((x_tag_global, y_tag_global, z_tag_global))
        return R_global_tag

    def _yaw_from_R_global_camera(self, R_global_camera: np.ndarray) -> float:
        """
        Compute robot/camera global yaw in the x-y plane from the camera forward axis.
        Camera forward axis is +z_cam.
        """
        forward_global = R_global_camera[:, 2]
        return wrap_angle(np.arctan2(forward_global[1], forward_global[0]))

    def _compute_robot_pose_from_detection(self, detection: Any) -> Optional[Dict[str, Any]]:
        tag_id = int(detection.tag_id)
        if tag_id not in self.tag_map:
            return None

        tag_pose = self.tag_map[tag_id]
        tag_x = float(tag_pose.get("x", 0.0))
        tag_y = float(tag_pose.get("y", 0.0))
        tag_z = float(tag_pose.get("z", 0.0))
        tag_yaw = float(tag_pose.get("yaw", 0.0))

        tag_origin_global = np.array([-tag_x, -tag_y, tag_z], dtype=float)

        # pupil_apriltags:
        #   R_camera_tag maps tag-frame vectors to camera-frame vectors
        #   t_camera_tag is tag origin expressed in camera frame
        R_camera_tag = np.asarray(detection.pose_R, dtype=float).reshape(3, 3)
        t_camera_tag = np.asarray(detection.pose_t, dtype=float).reshape(3)

        # Invert the tag pose to get camera pose in tag frame
        R_tag_camera = R_camera_tag.T
        p_camera_in_tag = -R_tag_camera @ t_camera_tag

        # Tag known orientation in global frame
        R_global_tag = self._build_R_global_tag(tag_yaw)

        # Camera pose in global frame
        p_camera_global = tag_origin_global + R_global_tag @ p_camera_in_tag
        R_global_camera = R_global_tag @ R_tag_camera

        yaw_robot_global = self._yaw_from_R_global_camera(R_global_camera)

        # Offset from robot origin to camera, expressed in robot frame
        camera_offset_robot = np.array(
            [
                self.camera_in_robot["x"],
                self.camera_in_robot["y"],
                self.camera_in_robot["z"],
            ],
            dtype=float,
        )

        # Assume robot/body frame yaw matches camera yaw in global x-y
        c = np.cos(yaw_robot_global)
        s = np.sin(yaw_robot_global)
        R_global_robot = np.array(
            [
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        camera_offset_global = R_global_robot @ camera_offset_robot
        p_robot_global = p_camera_global - camera_offset_global

        weight = 1.0 / max(np.linalg.norm(t_camera_tag), 1e-6)

        return {
            "tag_id": tag_id,
            "x": float(-p_robot_global[0]),
            "y": float(-p_robot_global[1]),
            "z": float(p_robot_global[2]),
            "yaw": float(yaw_robot_global + np.pi),  # flip to match your desired yaw convention
            "weight": weight,
        }

    def _fuse_measurements(self, per_tag_measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        weights = [m["weight"] for m in per_tag_measurements]
        weight_sum = sum(weights)

        if weight_sum <= 0:
            raise ValueError("Non-positive weight sum in fusion.")

        xs = [m["x"] for m in per_tag_measurements]
        ys = [m["y"] for m in per_tag_measurements]
        zs = [m["z"] for m in per_tag_measurements]
        yaws = [m["yaw"] for m in per_tag_measurements]

        x_fused = float(sum(w * x for w, x in zip(weights, xs)) / weight_sum)
        y_fused = float(sum(w * y for w, y in zip(weights, ys)) / weight_sum)
        z_fused = float(sum(w * z for w, z in zip(weights, zs)) / weight_sum)
        yaw_fused = circular_mean(yaws, weights)

        return {
            "x": x_fused,
            "y": y_fused,
            "z": z_fused,
            "yaw": yaw_fused,
            "tag_ids": [m["tag_id"] for m in per_tag_measurements],
            "num_tags": len(per_tag_measurements),
        }

    def process_frame(
        self,
        frame: np.ndarray,
        draw: bool = True
    ) -> Tuple[bool, Optional[Dict[str, Any]], np.ndarray]:
        results, debug_frame = self._detect_tags(frame)

        per_tag_measurements: List[Dict[str, Any]] = []

        for r in results:
            m = self._compute_robot_pose_from_detection(r)
            if m is None:
                continue

            per_tag_measurements.append(m)

            if draw:
                corners = r.corners.astype(int)

                for i in range(4):
                    cv2.line(
                        debug_frame,
                        tuple(corners[i]),
                        tuple(corners[(i + 1) % 4]),
                        (0, 255, 0),
                        2,
                    )

                center_xy = (int(r.center[0]), int(r.center[1]))
                cv2.putText(
                    debug_frame,
                    f"ID: {int(r.tag_id)}",
                    center_xy,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        if len(per_tag_measurements) == 0:
            if draw:
                cv2.putText(
                    debug_frame,
                    "No mapped AprilTags detected",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            return False, None, debug_frame

        fused = self._fuse_measurements(per_tag_measurements)

        if draw:
            cv2.putText(
                debug_frame,
                f"global x = {fused['x']:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                debug_frame,
                f"global y = {fused['y']:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                debug_frame,
                f"global z = {fused['z']:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                debug_frame,
                f"global yaw = {np.degrees(fused['yaw']):.1f} deg",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
            )

        return True, fused, debug_frame


if __name__ == "__main__":
    TAG_MAP = {
        4: {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
        }
    }

    CAMERA_IN_ROBOT = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }

    estimator = AprilTagGlobalPoseEstimator(
        tag_map=TAG_MAP,
        calibration_file="camera_calibration_live.npz",
        tag_size=0.17,
        camera_in_robot=CAMERA_IN_ROBOT,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        success, measurement, debug_frame = estimator.process_frame(frame, draw=True)

        if success:
            print(
                f"global x={measurement['x']:.3f}, "
                f"global y={measurement['y']:.3f}, "
                f"global z={measurement['z']:.3f}, "
                f"global yaw={np.degrees(measurement['yaw']):.1f} deg"
            )
        else:
            print("No mapped AprilTag detected.")

        cv2.imshow("AprilTag Global Pose Estimation", debug_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()