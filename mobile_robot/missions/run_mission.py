#!/usr/bin/env python
"""Simple waypoint-tracking mission runner using a finite state machine.

Mission logic:
- Each task specifies a goal pose in the map frame.
- The robot tracks that goal using the localization estimate and waypoint
  controller.
- AprilTag detections are used only as localization measurements when enabled.
- After the goal is reached, the task optionally executes a timed manipulator
  action / completion-condition hold.

This intentionally avoids global path planning and py_trees.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
import pupil_apriltags as apriltag

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from autonomy.mission_runner import (  # noqa: E402
    Pose2D,
    Task,
    default_camera_path,
    default_localization_path,
    default_tasks_path,
    evaluate_condition,
    load_map,
    load_tasks,
)
from camera import (  # noqa: E402
    RealSenseCamera,
    StreamConfig,
    bgr_to_gray,
    load_camera_to_robot_extrinsics,
)
from localization import (  # noqa: E402
    AprilTagMeasurement,
    ExtendedKalmanFilter2D,
    IMUMeasurement,
    WheelTwistMeasurement,
)
from localization.person_detection import PersonDetector  # noqa: E402
from serial_connection.elevator_serial_con import ElevatorSerialConnect  # noqa: E402
from serial_connection.serial_con import SerialConnect  # noqa: E402
from serial_connection.serialization import EncoderReading, IMUReading  # noqa: E402
from guidance.waypoint_controller import CascadedWaypointController, MapPoseVelocity, wrap_to_pi  # noqa: E402
from planning.a_star import Waypoint  # noqa: E402


DEBUG_WHEEL_ACKS = True


class MissionPhase(Enum):
    IDLE = auto()
    SEEK_GOAL = auto()
    EXECUTE_TASK = auto()
    STOP_FOR_OBSTACLE = auto()
    DONE = auto()
    ESTOP = auto()


@dataclass
class LocalizationConfig:
    filter_type: str
    use_ekf: bool
    initial_state: np.ndarray
    initial_covariance: np.ndarray
    process_noise: np.ndarray
    apriltag_measurement_noise: np.ndarray
    gyro_measurement_noise: np.ndarray
    wheel_twist_measurement_noise: np.ndarray
    use_imu_accel_in_prediction: bool
    apriltag_reinitialize_distance_m: float


@dataclass
class RuntimeConfig:
    control_rate_hz: float
    controller_v_max: float
    controller_omega_max: float
    tag_distance_kp: float
    tag_center_kp: float
    search_omega_rad_s: float
    align_only_center_error_px: float
    default_distance_tolerance_m: float
    default_center_tolerance_px: float
    default_settle_time_s: float
    person_detection_period_s: float
    obstacle_stop_forward_m: float
    obstacle_stop_lateral_m: float


@dataclass
class RealSenseDeviceConfig:
    serial_number: Optional[str] = None
    width: int = 640
    height: int = 480
    fps: int = 15


@dataclass
class TagDetection:
    tag_id: str
    center_x_px: float
    center_y_px: float
    tx_m: float
    ty_m: float
    tz_m: float
    pose_R: np.ndarray


@dataclass
class TelemetryPacket:
    timestamp: float
    phase: str
    task_index: int
    current_task: str
    goal_x: Optional[float]
    goal_y: Optional[float]
    goal_heading: Optional[float]
    goal_distance_error_m: Optional[float]
    goal_heading_error_rad: Optional[float]
    x: float
    y: float
    yaw: float
    vx_body: float
    vy_body: float
    wz: float
    deploy: bool
    manual_control: bool
    allstop: bool
    obstacle_blocked: bool


class MissionRuntime:
    def __init__(
        self,
        tasks_cfg_path: Path,
        localization_cfg_path: Path,
        camera_cfg_path: Path,
        serial_port: Optional[str],
        elevator_serial_port: Optional[str],
        disable_camera: bool,
        debug: bool,
        telemetry_host: Optional[str],
        telemetry_port: int,
        telemetry_rate_hz: float,
        control_host: str,
        control_port: int,
    ):
        self.tasks_path = Path(tasks_cfg_path)
        self.localization_cfg_path = Path(localization_cfg_path)
        self.camera_cfg_path = Path(camera_cfg_path)
        self.tasks = load_tasks(self.tasks_path)
        self.map = load_map(self.tasks_path)
        self.landmarks_by_id = {str(landmark.id): landmark for landmark in self.map.landmarks}
        self.camera_to_robot_extrinsics = load_camera_to_robot_extrinsics(self.camera_cfg_path)
        if not self.tasks:
            raise ValueError("mission config must define at least one task")

        (
            self.localization_config,
            self.runtime_config,
            self.apriltag_camera_config,
            self.person_camera_config,
        ) = self._load_config(
            self.tasks_path,
            self.localization_cfg_path,
            self.camera_cfg_path,
        )

        self.serial = SerialConnect(
            port=serial_port,
            tx_rate_hz=self.runtime_config.control_rate_hz,
            rx_publish_rate_hz=max(20.0, self.runtime_config.control_rate_hz),
            debug=debug,
        )
        self.elevator_serial: Optional[ElevatorSerialConnect] = None
        if elevator_serial_port:
            self.elevator_serial = ElevatorSerialConnect(
                port=elevator_serial_port,
                tx_rate_hz=self.runtime_config.control_rate_hz,
                rx_publish_rate_hz=max(20.0, self.runtime_config.control_rate_hz),
                debug=debug,
            )

        self.twist_helper = CascadedWaypointController(
            v_max=self.runtime_config.controller_v_max,
            omega_max=self.runtime_config.controller_omega_max,
        )
        self.localization_filter = ExtendedKalmanFilter2D(
            initial_state=self.localization_config.initial_state,
            initial_covariance=self.localization_config.initial_covariance,
            process_noise=self.localization_config.process_noise,
            apriltag_measurement_noise=self.localization_config.apriltag_measurement_noise,
            gyro_measurement_noise=self.localization_config.gyro_measurement_noise,
            wheel_twist_measurement_noise=self.localization_config.wheel_twist_measurement_noise,
        )
        print(f"Localization mode: {self.localization_config.filter_type}")

        self.apriltag_camera: Optional[RealSenseCamera] = None
        self.person_camera: Optional[RealSenseCamera] = None
        self.person_detector: Optional[PersonDetector] = None
        self.apriltag_detector = apriltag.Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        if not disable_camera:
            self.apriltag_camera = RealSenseCamera(
                color_config=StreamConfig(
                    width=self.apriltag_camera_config.width,
                    height=self.apriltag_camera_config.height,
                    fps=self.apriltag_camera_config.fps,
                ),
                depth_config=StreamConfig(
                    width=self.apriltag_camera_config.width,
                    height=self.apriltag_camera_config.height,
                    fps=self.apriltag_camera_config.fps,
                ),
                serial_number=self.apriltag_camera_config.serial_number,
            )
            self.apriltag_camera.open()

            self.person_camera = RealSenseCamera(
                color_config=StreamConfig(
                    width=self.person_camera_config.width,
                    height=self.person_camera_config.height,
                    fps=self.person_camera_config.fps,
                ),
                depth_config=StreamConfig(
                    width=self.person_camera_config.width,
                    height=self.person_camera_config.height,
                    fps=self.person_camera_config.fps,
                ),
                serial_number=self.person_camera_config.serial_number,
            )
            try:
                self.person_camera.open()
                self.person_detector = PersonDetector(
                    camera=self.person_camera,
                    mission_config_path=self.tasks_path,
                )
            except Exception as exc:  # pragma: no cover
                print(f"WARN: person camera/detector unavailable: {exc}")
                self.person_camera = None
                self.person_detector = None

        self.phase = MissionPhase.IDLE
        self._phase_before_obstacle_stop = MissionPhase.SEEK_GOAL
        self.task_index = 0
        self.phase_start_time = time.monotonic()
        self.target_reached_since: Optional[float] = None
        self.last_loop_time = time.monotonic()
        self.last_telemetry_time = 0.0
        self.telemetry_period = 0.0 if telemetry_rate_hz <= 0.0 else 1.0 / telemetry_rate_hz
        self.telemetry_host = telemetry_host
        self.telemetry_port = telemetry_port
        self.telemetry_sock: Optional[socket.socket] = None
        if telemetry_host:
            self.telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Telemetry UDP target: {telemetry_host}:{telemetry_port}")

        self.control_host = control_host
        self.control_port = control_port
        self.control_sock: Optional[socket.socket] = None
        if control_port > 0:
            self.control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.control_sock.bind((control_host, control_port))
            self.control_sock.setblocking(False)
            print(f"Control UDP listener: udp://{control_host}:{control_port}")

        self.deploy = False
        self.manual_control = False
        self.allstop = False
        self._last_wheel_mode: Optional[str] = None
        self._obstacle_blocked = False
        self._last_person_detection_time = -float("inf")
        self._active_arm_task_name: Optional[str] = None
        self._active_arm_waypoint_index = 0
        self._last_arm_waypoint_send_time = -float("inf")
        self._last_debug_cmd_line: Optional[str] = None
        self._last_debug_eff_line: Optional[str] = None
        self._last_debug_enc_line: Optional[str] = None
        self._last_debug_imu_line: Optional[str] = None
        self._last_debug_ack_meta_line: Optional[str] = None
        self._last_debug_applied_cmd_line: Optional[str] = None
        self._last_debug_rx_cmd_line: Optional[str] = None
        self._last_debug_imu_meta_line: Optional[str] = None

    @staticmethod
    def _load_config(
        tasks_path: Path,
        localization_path: Path,
        camera_path: Path,
    ) -> tuple[LocalizationConfig, RuntimeConfig, RealSenseDeviceConfig, RealSenseDeviceConfig]:
        with open(tasks_path, "r", encoding="utf-8") as f:
            mission_raw = yaml.safe_load(f)
        with open(localization_path, "r", encoding="utf-8") as f:
            localization_raw = yaml.safe_load(f)
        with open(camera_path, "r", encoding="utf-8") as f:
            camera_raw = yaml.safe_load(f)

        loc = localization_raw.get("localization", {})
        runtime = mission_raw.get("mission_runtime", {})
        init = loc.get("initial_state", {})
        filter_type = str(loc.get("filter", "ekf")).strip().lower()

        localization = LocalizationConfig(
            filter_type=filter_type,
            use_ekf=(filter_type == "ekf"),
            initial_state=np.array(
                [
                    float(init.get("x", 0.0)),
                    float(init.get("y", 0.0)),
                    float(init.get("yaw", 0.0)),
                    float(init.get("vx_body", 0.0)),
                    float(init.get("vy_body", 0.0)),
                    float(init.get("wz", 0.0)),
                    float(init.get("b_ax", 0.0)),
                    float(init.get("b_ay", 0.0)),
                    float(init.get("b_wz", 0.0)),
                ],
                dtype=float,
            ),
            initial_covariance=np.asarray(loc.get("initial_covariance"), dtype=float),
            process_noise=np.asarray(loc.get("process_noise"), dtype=float),
            apriltag_measurement_noise=np.asarray(loc.get("apriltag_measurement_noise", [[0.025, 0.0, 0.0], [0.0, 0.025, 0.0], [0.0, 0.0, 0.04]]), dtype=float),
            gyro_measurement_noise=np.asarray(loc.get("gyro_measurement_noise", [[0.15]]), dtype=float),
            wheel_twist_measurement_noise=np.asarray(
                loc.get("wheel_twist_measurement_noise", [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.08]]),
                dtype=float,
            ),
            use_imu_accel_in_prediction=bool(loc.get("use_imu_accel_in_prediction", False)),
            apriltag_reinitialize_distance_m=float(loc.get("apriltag_reinitialize_distance_m", 0.0)),
        )
        runtime_cfg = RuntimeConfig(
            control_rate_hz=float(runtime.get("control_rate_hz", 15.0)),
            controller_v_max=float(runtime.get("controller_v_max", 0.18)),
            controller_omega_max=float(runtime.get("controller_omega_max", 0.18)),
            tag_distance_kp=float(runtime.get("tag_distance_kp", 0.8)),
            tag_center_kp=float(runtime.get("tag_center_kp", 1.2)),
            search_omega_rad_s=float(runtime.get("search_omega_rad_s", 0.35)),
            align_only_center_error_px=float(runtime.get("align_only_center_error_px", 120.0)),
            default_distance_tolerance_m=float(runtime.get("default_distance_tolerance_m", 0.08)),
            default_center_tolerance_px=float(runtime.get("default_center_tolerance_px", 40.0)),
            default_settle_time_s=float(runtime.get("default_settle_time_s", 0.3)),
            person_detection_period_s=float(runtime.get("person_detection_period_s", 0.5)),
            obstacle_stop_forward_m=float(runtime.get("obstacle_stop_forward_m", 1.2)),
            obstacle_stop_lateral_m=float(runtime.get("obstacle_stop_lateral_m", 0.5)),
        )

        apriltag_camera_raw = camera_raw.get("apriltag_camera", {})
        person_camera_raw = camera_raw.get("person_camera", {})
        apriltag_camera = RealSenseDeviceConfig(
            serial_number=os.environ.get("APRILTAG_REALSENSE_SERIAL", apriltag_camera_raw.get("serial_number")),
            width=int(apriltag_camera_raw.get("width", 640)),
            height=int(apriltag_camera_raw.get("height", 480)),
            fps=int(apriltag_camera_raw.get("fps", 15)),
        )
        person_camera = RealSenseDeviceConfig(
            serial_number=os.environ.get("PERSON_REALSENSE_SERIAL", person_camera_raw.get("serial_number")),
            width=int(person_camera_raw.get("width", 640)),
            height=int(person_camera_raw.get("height", 480)),
            fps=int(person_camera_raw.get("fps", 15)),
        )
        return localization, runtime_cfg, apriltag_camera, person_camera

    def _current_task(self) -> Optional[Task]:
        if self.task_index >= len(self.tasks):
            return None
        return self.tasks[self.task_index]

    def _set_phase(self, phase: MissionPhase) -> None:
        if self.phase != phase:
            self.phase = phase
            self.phase_start_time = time.monotonic()
            self.target_reached_since = None
            if phase == MissionPhase.EXECUTE_TASK:
                self._active_arm_task_name = None

    def _zero_wheel_rates(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    def _body_twist_to_wheels(self, vx: float, omega: float) -> tuple[float, float, float, float]:
        return self.twist_helper.body_twist_to_wheel_rates(vx, omega)

    def _update_localization_from_imu(self, dt: float) -> None:
        packets = self.serial.read_packets(max_lines=128)
        imu_packets = [msg for msg in packets if isinstance(msg, IMUReading)]
        encoder_packets = [msg for msg in packets if isinstance(msg, EncoderReading)]

        if not self.localization_config.use_ekf:
            return

        if not imu_packets:
            self.localization_filter.predict(IMUMeasurement(ax=0.0, ay=0.0, wz=0.0), dt)
        else:
            imu_msg = imu_packets[-1]
            imu = IMUMeasurement(ax=0.0, ay=0.0, wz=imu_msg.gz)
            self.localization_filter.predict(imu, dt)
            self.localization_filter.update_imu(IMUMeasurement(wz=imu_msg.gz))

        if encoder_packets:
            enc = encoder_packets[-1]
            vx_body, vy_body, omega = self.twist_helper.wheel_rates_to_body_twist((enc.w1, enc.w2, enc.w3, enc.w4))
            self.localization_filter.update_wheel_twist(WheelTwistMeasurement(vx=vx_body, vy=vy_body, wz=omega))

    def _detect_tags(self) -> dict[str, TagDetection]:
        if self.apriltag_camera is None:
            return {}
        frame = self.apriltag_camera.read()
        if frame.intrinsics is None:
            return {}
        gray = bgr_to_gray(frame.color)
        results = self.apriltag_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[frame.intrinsics.fx, frame.intrinsics.fy, frame.intrinsics.cx, frame.intrinsics.cy],
            tag_size=0.09,
        )
        detections: dict[str, TagDetection] = {}
        for r in results:
            t = np.asarray(r.pose_t, dtype=float).reshape(3)
            detections[str(r.tag_id)] = TagDetection(
                tag_id=str(r.tag_id),
                center_x_px=float(r.center[0]),
                center_y_px=float(r.center[1]),
                tx_m=float(t[0]),
                ty_m=float(t[1]),
                tz_m=float(t[2]),
                pose_R=np.asarray(r.pose_R, dtype=float).reshape(3, 3),
            )

        if detections:
            debug_tags = ", ".join(
                f"id={tag_id} z={det.tz_m:.3f}m cx={det.center_x_px:.1f}px"
                for tag_id, det in sorted(detections.items())
            )
            print(f"[AprilTag] detected: {debug_tags}")
        else:
            print("[AprilTag] detected: none")
        return detections

    @staticmethod
    def _optical_to_nominal_rotation() -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )

    @staticmethod
    def _tag_world_rotation(heading: float) -> np.ndarray:
        outward = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=float)
        y_world = np.array([0.0, 0.0, -1.0], dtype=float)
        z_world = -outward
        x_world = np.cross(y_world, z_world)
        x_world /= max(np.linalg.norm(x_world), 1e-9)
        return np.column_stack((x_world, y_world, z_world))

    @staticmethod
    def _invert_transform(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r_inv = rotation.T
        t_inv = -r_inv @ translation.reshape(3)
        return r_inv, t_inv

    def _apriltag_measurement_from_detection(self, det: TagDetection) -> Optional[AprilTagMeasurement]:
        landmark = self.landmarks_by_id.get(det.tag_id)
        if landmark is None:
            return None

        r_robot_camera = (
            np.asarray(self.camera_to_robot_extrinsics.rotation_camera_to_world, dtype=float).reshape(3, 3)
            @ self._optical_to_nominal_rotation()
        )
        t_robot_camera = np.asarray(self.camera_to_robot_extrinsics.translation_camera_to_world, dtype=float).reshape(3)

        r_camera_tag = np.asarray(det.pose_R, dtype=float).reshape(3, 3)
        t_camera_tag = np.array([det.tx_m, det.ty_m, det.tz_m], dtype=float)

        r_robot_tag = r_robot_camera @ r_camera_tag
        t_robot_tag = r_robot_camera @ t_camera_tag + t_robot_camera
        r_tag_robot, t_tag_robot = self._invert_transform(r_robot_tag, t_robot_tag)

        r_world_tag = self._tag_world_rotation(float(landmark.heading))
        t_world_tag = np.array([float(landmark.point[0]), float(landmark.point[1]), 0.0], dtype=float)

        r_world_robot = r_world_tag @ r_tag_robot
        t_world_robot = r_world_tag @ t_tag_robot + t_world_tag
        yaw_world_robot = float(np.arctan2(r_world_robot[1, 0], r_world_robot[0, 0]))

        return AprilTagMeasurement(
            x=float(t_world_robot[0]),
            y=float(t_world_robot[1]),
            yaw=yaw_world_robot,
            covariance=self.localization_config.apriltag_measurement_noise,
        )

    def _update_localization_from_apriltags(self, detections: dict[str, TagDetection]) -> None:
        if not detections:
            return

        measurements: list[AprilTagMeasurement] = []
        for tag_id in sorted(detections.keys()):
            measurement = self._apriltag_measurement_from_detection(detections[tag_id])
            if measurement is not None:
                measurements.append(measurement)

        if not measurements:
            return

        mean_x = float(np.mean([m.x for m in measurements]))
        mean_y = float(np.mean([m.y for m in measurements]))
        mean_yaw = float(
            np.arctan2(
                np.mean([np.sin(m.yaw) for m in measurements]),
                np.mean([np.cos(m.yaw) for m in measurements]),
            )
        )

        if self.localization_config.use_ekf:
            self.localization_filter.update_apriltag(
                AprilTagMeasurement(
                    x=mean_x,
                    y=mean_y,
                    yaw=mean_yaw,
                    covariance=self.localization_config.apriltag_measurement_noise,
                )
            )
        else:
            state = self.localization_filter.get_state()
            state[0] = mean_x
            state[1] = mean_y
            state[2] = mean_yaw
            self.localization_filter.state = state

    def _person_blocking(self, now: float) -> bool:
        if self.person_detector is None:
            self._obstacle_blocked = False
            return False
        if (now - self._last_person_detection_time) < self.runtime_config.person_detection_period_s:
            return self._obstacle_blocked
        try:
            detections = self.person_detector.detect(robot_pose=None)
        except Exception as exc:  # pragma: no cover
            print(f"WARN: person detection update failed: {exc}")
            self._obstacle_blocked = False
            return False
        self._last_person_detection_time = now
        blocked = False
        for det in detections:
            if det.position_robot_m is None:
                continue
            x_r, y_r, _ = det.position_robot_m.tolist()
            if 0.0 < x_r <= self.runtime_config.obstacle_stop_forward_m and abs(y_r) <= self.runtime_config.obstacle_stop_lateral_m:
                blocked = True
                break
        self._obstacle_blocked = blocked
        return blocked

    def _update_manipulator_for_task(self, task: Task, active: bool, now: float) -> None:
        if self.elevator_serial is None:
            return
        if self._active_arm_task_name != task.name:
            self._active_arm_task_name = task.name
            self._active_arm_waypoint_index = 0
            self._last_arm_waypoint_send_time = -float("inf")
        if not active:
            return
        self.elevator_serial.send_height_cmd(task.desired_elevator_height_m)
        if task.arm_waypoints and ((now - self._last_arm_waypoint_send_time) >= max(0.0, task.arm_point_dwell_s)):
            waypoint = task.arm_waypoints[min(self._active_arm_waypoint_index, len(task.arm_waypoints) - 1)]
            self.elevator_serial.send_arm_cmd(waypoint.x, waypoint.y)
            self._last_arm_waypoint_send_time = now
            if self._active_arm_waypoint_index < len(task.arm_waypoints) - 1:
                self._active_arm_waypoint_index += 1
        self.elevator_serial.flush_tx()

    def _task_context(self, task: Task) -> dict:
        current_elevator_height_m = 0.0
        if self.elevator_serial is not None:
            for msg in self.elevator_serial.read_parsed(max_lines=32):
                current_elevator_height_m = float(msg.height_m)
        desired_elevator_height_m = float(task.desired_elevator_height_m)
        elevator_height_error_m = desired_elevator_height_m - current_elevator_height_m
        state = self.localization_filter.get_state()
        speed = float(np.hypot(state[3], state[4]))
        return {
            "current_task": task.name,
            "task_index": self.task_index,
            "desired_elevator_height_m": desired_elevator_height_m,
            "current_elevator_height_m": current_elevator_height_m,
            "elevator_height_error_m": elevator_height_error_m,
            "elevator_at_height": abs(elevator_height_error_m) <= 0.01,
            "robot_stopped": speed < 0.02,
            "task_elapsed_s": float(time.monotonic() - self.phase_start_time),
            "deploy": bool(self.deploy),
            "manual_control": bool(self.manual_control),
            "allstop": bool(self.allstop),
            "obstacle_blocking_path": bool(self._obstacle_blocked),
        }

    def _task_complete(self, task: Task) -> bool:
        if not task.completion_conditions:
            return True
        context = self._task_context(task)
        return all(evaluate_condition(expr, context) for expr in task.completion_conditions)

    def _advance_to_execute_task(self) -> None:
        task = self._current_task()
        if task is None:
            self._set_phase(MissionPhase.DONE)
            return
        self.target_reached_since = None
        self._set_phase(MissionPhase.EXECUTE_TASK)

    def _advance_task(self) -> None:
        self.task_index += 1
        if self.task_index >= len(self.tasks):
            self._set_phase(MissionPhase.DONE)
        else:
            self._set_phase(MissionPhase.SEEK_GOAL)

    def _seek_goal_command(self, task: Task, now: float) -> tuple[tuple[float, float, float, float], Optional[float], Optional[float]]:
        goal = task.goal
        if goal is None:
            self.target_reached_since = None
            return self._zero_wheel_rates(), None, None

        state = self.localization_filter.get_state()
        pose = MapPoseVelocity(
            x=float(state[0]),
            y=float(state[1]),
            heading=float(state[2]),
            vx=float(state[3] * np.cos(state[2]) - state[4] * np.sin(state[2])),
            vy=float(state[3] * np.sin(state[2]) + state[4] * np.cos(state[2])),
            heading_rate=float(state[5]),
        )
        command = self.twist_helper.compute(
            pose,
            Waypoint(xy=(float(goal.x), float(goal.y)), heading=float(goal.heading)),
            final_pose_mode=True,
        )

        position_error_m = float(np.hypot(goal.x - pose.x, goal.y - pose.y))
        heading_error_rad = float(wrap_to_pi(goal.heading - pose.heading))
        distance_tol = float(self.runtime_config.default_distance_tolerance_m)
        heading_tol = 0.15
        settle_time = float(self.runtime_config.default_settle_time_s)

        reached = position_error_m <= distance_tol and abs(heading_error_rad) <= heading_tol
        if reached:
            if self.target_reached_since is None:
                self.target_reached_since = now
            if (now - self.target_reached_since) >= settle_time:
                self._advance_to_execute_task()
                return self._zero_wheel_rates(), position_error_m, heading_error_rad
        else:
            self.target_reached_since = None

        return command.wheel_rates, position_error_m, heading_error_rad

    def _poll_control_socket(self) -> None:
        if self.control_sock is None:
            return
        while True:
            try:
                payload, addr = self.control_sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                message = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                print(f"WARN: invalid control packet from {addr}: {exc}")
                continue
            if "deploy" in message:
                self.deploy = bool(message["deploy"])
            if "manual_control" in message:
                self.manual_control = bool(message["manual_control"])
            if "allstop" in message:
                self.allstop = bool(message["allstop"])

    def _update_wheel_mode(self, autonomy_active: bool) -> None:
        desired_mode = "AUTONOMY" if autonomy_active else "JOYSTICK"
        if desired_mode == self._last_wheel_mode:
            return
        self.serial.send_raw_line(f"MODE,{desired_mode}", force=True)
        self._last_wheel_mode = desired_mode

    def _maybe_capture_wheel_debug_lines(self) -> None:
        if not DEBUG_WHEEL_ACKS:
            return
        raw_line = getattr(self.serial, "_latest_raw_line", None)
        if not raw_line:
            return
        if raw_line.startswith("CMD,"):
            self._last_debug_cmd_line = raw_line
        elif raw_line.startswith("EFF,"):
            self._last_debug_eff_line = raw_line
        elif raw_line.startswith("ENC,"):
            self._last_debug_enc_line = raw_line
        elif raw_line.startswith("IMU,"):
            self._last_debug_imu_line = raw_line
        elif raw_line.startswith("DBG,ACK_META,"):
            self._last_debug_ack_meta_line = raw_line
        elif raw_line.startswith("DBG,LATEST_APPLIED_CMD,"):
            self._last_debug_applied_cmd_line = raw_line
        elif raw_line.startswith("DBG,LATEST_RX_CMD,"):
            self._last_debug_rx_cmd_line = raw_line
        elif raw_line.startswith("DBG,IMU_META,"):
            self._last_debug_imu_meta_line = raw_line

    def _publish_telemetry(self, now: float, current_task: str, task: Optional[Task], position_error_m: Optional[float], heading_error_rad: Optional[float]) -> None:
        if self.telemetry_sock is None:
            return
        if self.telemetry_period > 0.0 and (now - self.last_telemetry_time) < self.telemetry_period:
            return
        state = self.localization_filter.get_state()
        packet = TelemetryPacket(
            timestamp=time.time(),
            phase=self.phase.name,
            task_index=int(self.task_index),
            current_task=current_task,
            goal_x=None if task is None or task.goal is None else float(task.goal.x),
            goal_y=None if task is None or task.goal is None else float(task.goal.y),
            goal_heading=None if task is None or task.goal is None else float(task.goal.heading),
            goal_distance_error_m=None if position_error_m is None else float(position_error_m),
            goal_heading_error_rad=None if heading_error_rad is None else float(heading_error_rad),
            x=float(state[0]),
            y=float(state[1]),
            yaw=float(state[2]),
            vx_body=float(state[3]),
            vy_body=float(state[4]),
            wz=float(state[5]),
            deploy=bool(self.deploy),
            manual_control=bool(self.manual_control),
            allstop=bool(self.allstop),
            obstacle_blocked=bool(self._obstacle_blocked),
        )
        self.telemetry_sock.sendto(json.dumps(asdict(packet)).encode("utf-8"), (self.telemetry_host, self.telemetry_port))
        self.last_telemetry_time = now

    def run(self, max_ticks: Optional[int] = None) -> None:
        period = 1.0 / self.runtime_config.control_rate_hz
        tick = 0
        try:
            while True:
                if max_ticks is not None and tick >= max_ticks:
                    break
                if self.phase == MissionPhase.DONE:
                    break

                now = time.monotonic()
                dt = max(now - self.last_loop_time, 1e-3)
                self.last_loop_time = now
                self._update_localization_from_imu(dt)
                self._maybe_capture_wheel_debug_lines()
                self._poll_control_socket()

                autonomy_active = self.deploy and self.manual_control and not self.allstop
                self._update_wheel_mode(autonomy_active)
                task = self._current_task()
                current_task_name = "DONE" if task is None else task.name
                wheel_rates = self._zero_wheel_rates()
                position_error_m: Optional[float] = None
                heading_error_rad: Optional[float] = None
                detections: dict[str, TagDetection] = {}

                if self.apriltag_camera is not None:
                    detections = self._detect_tags()
                    self._update_localization_from_apriltags(detections)

                if task is None:
                    self._set_phase(MissionPhase.DONE)
                elif self.allstop:
                    self._set_phase(MissionPhase.ESTOP)
                elif not autonomy_active:
                    self._set_phase(MissionPhase.IDLE)
                elif self.phase == MissionPhase.IDLE:
                    self._set_phase(MissionPhase.SEEK_GOAL)

                if autonomy_active and self.phase not in (MissionPhase.DONE, MissionPhase.ESTOP):
                    self._update_manipulator_for_task(task, active=(self.phase == MissionPhase.EXECUTE_TASK), now=now)

                    if self.phase == MissionPhase.SEEK_GOAL:
                        if self._person_blocking(now):
                            self._phase_before_obstacle_stop = self.phase
                            self._set_phase(MissionPhase.STOP_FOR_OBSTACLE)
                            wheel_rates = self._zero_wheel_rates()
                        else:
                            wheel_rates, position_error_m, heading_error_rad = self._seek_goal_command(task, now)
                    elif self.phase == MissionPhase.STOP_FOR_OBSTACLE:
                        wheel_rates = self._zero_wheel_rates()
                        if not self._person_blocking(now):
                            self._set_phase(self._phase_before_obstacle_stop)
                    elif self.phase == MissionPhase.EXECUTE_TASK:
                        wheel_rates = self._zero_wheel_rates()
                        if self._task_complete(task):
                            self._advance_task()

                if autonomy_active or self.allstop:
                    self.serial.send_wheel_cmd(*wheel_rates)
                    self.serial.flush_tx()

                self._publish_telemetry(now, current_task_name, task, position_error_m, heading_error_rad)

                state = self.localization_filter.get_state()
                print(
                    f"tick={tick:04d} phase={self.phase.name} task={current_task_name} "
                    f"pose=({state[0]:.3f},{state[1]:.3f},{state[2]:.3f}) "
                    f"goal_err={None if position_error_m is None else round(position_error_m,3)} heading_err={None if heading_error_rad is None else round(heading_error_rad,3)} "
                    f"obstacle={self._obstacle_blocked} deploy={self.deploy} manual={self.manual_control} allstop={self.allstop} "
                    f"wheel_rates={tuple(round(v,3) for v in wheel_rates)}"
                )
                if DEBUG_WHEEL_ACKS:
                    if self._last_debug_ack_meta_line is not None:
                        print(f"  wheel_ack_meta: {self._last_debug_ack_meta_line}")
                    if self._last_debug_rx_cmd_line is not None:
                        print(f"  wheel_rx_cmd: {self._last_debug_rx_cmd_line}")
                    if self._last_debug_applied_cmd_line is not None:
                        print(f"  wheel_applied_cmd: {self._last_debug_applied_cmd_line}")
                    if self._last_debug_cmd_line is not None:
                        print(f"  wheel_ack: {self._last_debug_cmd_line}")
                    if self._last_debug_eff_line is not None:
                        print(f"  wheel_eff: {self._last_debug_eff_line}")
                    if self._last_debug_enc_line is not None:
                        print(f"  wheel_enc: {self._last_debug_enc_line}")
                    if self._last_debug_imu_meta_line is not None:
                        print(f"  imu_meta: {self._last_debug_imu_meta_line}")
                    if self._last_debug_imu_line is not None:
                        print(f"  imu_raw: {self._last_debug_imu_line}")

                tick += 1
                sleep_dt = period - (time.monotonic() - now)
                if sleep_dt > 0.0:
                    time.sleep(sleep_dt)
        finally:
            for _ in range(5):
                self.serial.send_wheel_cmd(0.0, 0.0, 0.0, 0.0, force=True)
                self.serial.flush_tx(force=True)
                time.sleep(0.02)
            self.serial.send_raw_line("MODE,JOYSTICK", force=True)
            self.serial.close()
            if self.elevator_serial is not None:
                self.elevator_serial.close()
            if self.apriltag_camera is not None:
                self.apriltag_camera.close()
            if self.person_camera is not None:
                self.person_camera.close()
            if self.telemetry_sock is not None:
                self.telemetry_sock.close()
            if self.control_sock is not None:
                self.control_sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(default_tasks_path()), help="Path to mission_config.yaml")
    parser.add_argument("--camera", default=str(default_camera_path()), help="Path to camera_config.yaml")
    parser.add_argument("--localization", default=str(default_localization_path()), help="Path to localization_config.yaml")
    parser.add_argument("--port", default=None, help="Serial port for wheel controller ESP32")
    parser.add_argument("--elevator-port", default=None, help="Serial port for elevator controller ESP32")
    parser.add_argument("--disable-camera", action="store_true")
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--debug-serial", action="store_true")
    parser.add_argument("--telemetry-host", default=None, help="UDP host/IP for visualization telemetry")
    parser.add_argument("--telemetry-port", type=int, default=8765)
    parser.add_argument("--telemetry-rate-hz", type=float, default=10.0)
    parser.add_argument("--control-host", default="0.0.0.0", help="UDP bind host/IP for deploy/allstop control packets")
    parser.add_argument("--control-port", type=int, default=8766)
    args = parser.parse_args()

    runtime = MissionRuntime(
        tasks_cfg_path=Path(args.tasks),
        localization_cfg_path=Path(args.localization),
        camera_cfg_path=Path(args.camera),
        serial_port=args.port,
        elevator_serial_port=args.elevator_port,
        disable_camera=args.disable_camera,
        debug=args.debug_serial,
        telemetry_host=args.telemetry_host,
        telemetry_port=args.telemetry_port,
        telemetry_rate_hz=args.telemetry_rate_hz,
        control_host=args.control_host,
        control_port=args.control_port,
    )
    runtime.run(max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
