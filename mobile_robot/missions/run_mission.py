#!/usr/bin/env python
"""Hardware-facing waypoint mission runner.

Loads `config/tasks.yaml`, initializes:
- serial connection to the ESP32 wheel controller
- EKF/UKF localization
- optional AprilTag camera localization
- A* path planning
- waypoint controller
- mission tree execution
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import py_trees
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from autonomy.mission_runner import Pose2D, Task, default_tasks_path, load_map, load_tasks  # noqa: E402
from autonomy.trees.waypoint_mission import (  # noqa: E402
    MISSION_DONE,
    create_tree,
    initialise_blackboard,
)
from guidance.waypoint_controller import (  # noqa: E402
    CascadedWaypointController,
    MapPoseVelocity,
    MecanumCommand,
    wrap_to_pi,
)
from localization import (  # noqa: E402
    AprilTagMeasurement,
    ExtendedKalmanFilter2D,
    IMUMeasurement,
    UnscentedKalmanFilter2D,
)

from camera import RealSenseCamera  # noqa: E402
from localization.april_tag_pose_est import AprilTagPoseEst
from localization.person_detection import PersonDetector

from localization.map import Map  # noqa: E402
from planning.a_star import AStar, waypoints_from_polyline  # noqa: E402
from serial_connection.elevator_serial_con import ElevatorSerialConnect  # noqa: E402
from serial_connection.serial_con import SerialConnect  # noqa: E402


@dataclass
class PlannedTask:
    task: Task
    polyline: list[tuple[float, float]]
    waypoints: list
    waypoint_index: int = 0


@dataclass
class LocalizationConfig:
    filter_name: str
    initial_state: np.ndarray
    initial_covariance: np.ndarray
    process_noise: np.ndarray
    apriltag_measurement_noise: np.ndarray
    gyro_measurement_noise: np.ndarray


@dataclass
class RuntimeConfig:
    control_rate_hz: float
    waypoint_capture_radius: float
    controller_v_max: float
    controller_omega_max: float


@dataclass
class CameraToRobotTransform:
    """Camera extrinsics expressed in the robot/body frame.

    The mission config may provide either the legacy planar form
    ``{x, y, yaw}`` or the richer nested 3D form
    ``{translation: {x, y, z}, rotation_rpy: {roll, pitch, yaw}}``.

    Current AprilTag localization logic still uses the planar subset
    ``x, y, yaw`` while the full fields are available for perception modules.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class DynamicObstaclePacket:
    name: str
    center: list[float]
    radius: float


@dataclass
class TelemetryPacket:
    timestamp: float
    x: float
    y: float
    yaw: float
    vx_body: float
    vy_body: float
    wz: float
    cell: Optional[list[int]]
    current_task: str
    localization_ok: bool
    distance_to_goal: float
    heading_error: float
    dynamic_obstacles: list[DynamicObstaclePacket]


class MissionRuntime:
    def __init__(
        self,
        tasks_path: Path,
        serial_port: Optional[str],
        elevator_serial_port: Optional[str],
        disable_camera: bool,
        debug: bool,
        telemetry_host: Optional[str],
        telemetry_port: int,
        telemetry_rate_hz: float,
    ):
        self.tasks_path = Path(tasks_path)
        self.tasks = load_tasks(self.tasks_path)
        self.task_lookup = {task.name: task for task in self.tasks}
        self.map_ = load_map(self.tasks_path)
        self.localization_config, self.runtime_config, self.camera_to_robot = self._load_config(
            self.tasks_path
        )
        self.planned_tasks = self._plan_tasks(self.map_, self.tasks)
        self.planned_lookup = {planned.task.name: planned for planned in self.planned_tasks}
        self.landmark_lookup = {str(landmark.id): landmark for landmark in self.map_.landmarks}

        self.blackboard = initialise_blackboard(
            self.tasks[0].name,
            state={
                "mission_enabled": True,
                "localization_ok": True,
                "battery_voltage": 24.0,
                "path_ready": True,
                "obstacle_blocking_path": False,
                "zone_clear": True,
                "current_elevator_height_m": 0.0,
                "desired_elevator_height_m": float(self.tasks[0].desired_elevator_height_m),
                "elevator_height_error_m": float(self.tasks[0].desired_elevator_height_m),
                "elevator_at_height": True,
                "distance_to_goal": 1.0,
                "heading_error": 1.0,
                "tray_detected": False,
                "robot_stopped": False,
                "tray_released": False,
                "gripper_closed": True,
                "previous_task_complete": True,
            },
        )
        self.tree = create_tree(self.tasks)
        self.tree.setup(timeout=2.0)

        self.controller = CascadedWaypointController(
            v_max=self.runtime_config.controller_v_max,
            omega_max=self.runtime_config.controller_omega_max,
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
        self.localization_filter = self._create_localization_filter()

        self.shared_camera: Optional[RealSenseCamera] = None
        self.apriltag_estimator = None
        self.person_detector = None
        self._warned_raw_apriltag = False
        self._person_obstacle_prefix = "person_"
        self._person_obstacle_radius_m = 0.35
        self._dynamic_obstacle_packets: list[DynamicObstaclePacket] = []
        if not disable_camera:
            self.shared_camera = RealSenseCamera()
            self.shared_camera.open()
            self.apriltag_estimator = AprilTagPoseEst(realsense_camera=self.shared_camera)
            self.person_detector = PersonDetector(
                camera=self.shared_camera,
                mission_config_path=self.tasks_path,
            )

        self.map_figure = None
        self.map_axes = None
        self._setup_live_map_plot()

        self.last_loop_time = time.monotonic()
        self.telemetry_host = telemetry_host
        self.telemetry_port = telemetry_port
        self.telemetry_period = 0.0 if telemetry_rate_hz <= 0.0 else 1.0 / telemetry_rate_hz
        self.last_telemetry_time = 0.0
        self.telemetry_sock: Optional[socket.socket] = None
        if telemetry_host:
            self.telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Telemetry UDP target: {telemetry_host}:{telemetry_port}")

    @staticmethod
    def _plan_tasks(map_: Map, tasks: list[Task]) -> list[PlannedTask]:
        planner = AStar(map_)
        planned: list[PlannedTask] = []
        for task in tasks:
            start_xy = (task.start.x, task.start.y)
            goal_xy = (task.goal.x, task.goal.y)
            polyline = planner.generate_plan(start_xy, goal_xy)
            if not polyline:
                raise RuntimeError(f"A* found no path for task {task.name}")
            planned.append(
                PlannedTask(
                    task=task,
                    polyline=polyline,
                    waypoints=waypoints_from_polyline(polyline, end_heading=task.goal.heading),
                )
            )
        return planned

    @staticmethod
    def _load_config(
        tasks_path: Path,
    ) -> tuple[LocalizationConfig, RuntimeConfig, CameraToRobotTransform]:
        with open(tasks_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        loc = raw.get("localization", {})
        runtime = raw.get("mission_runtime", {})
        init = loc.get("initial_state", {})

        localization = LocalizationConfig(
            filter_name=str(loc.get("filter", "ekf")).lower(),
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
            apriltag_measurement_noise=np.asarray(loc.get("apriltag_measurement_noise"), dtype=float),
            gyro_measurement_noise=np.asarray(loc.get("gyro_measurement_noise"), dtype=float),
        )
        runtime_cfg = RuntimeConfig(
            control_rate_hz=float(runtime.get("control_rate_hz", 20.0)),
            waypoint_capture_radius=float(runtime.get("waypoint_capture_radius", 0.05)),
            controller_v_max=float(runtime.get("controller_v_max", 0.35)),
            controller_omega_max=float(runtime.get("controller_omega_max", 1.8)),
        )
        cam = raw.get("camera_to_robot", {})
        if "translation" in cam or "rotation_rpy" in cam:
            translation = cam.get("translation", {})
            rotation = cam.get("rotation_rpy", {})
            camera_to_robot = CameraToRobotTransform(
                x=float(translation.get("x", 0.0)),
                y=float(translation.get("y", 0.0)),
                z=float(translation.get("z", 0.0)),
                roll=float(rotation.get("roll", 0.0)),
                pitch=float(rotation.get("pitch", 0.0)),
                yaw=float(rotation.get("yaw", 0.0)),
            )
        else:
            camera_to_robot = CameraToRobotTransform(
                x=float(cam.get("x", 0.0)),
                y=float(cam.get("y", 0.0)),
                z=float(cam.get("z", 0.0)),
                roll=float(cam.get("roll", 0.0)),
                pitch=float(cam.get("pitch", 0.0)),
                yaw=float(cam.get("yaw", 0.0)),
            )
        return localization, runtime_cfg, camera_to_robot

    def _create_localization_filter(self):
        filter_cls = (
            UnscentedKalmanFilter2D
            if self.localization_config.filter_name == "ukf"
            else ExtendedKalmanFilter2D
        )
        return filter_cls(
            initial_state=self.localization_config.initial_state,
            initial_covariance=self.localization_config.initial_covariance,
            process_noise=self.localization_config.process_noise,
            apriltag_measurement_noise=self.localization_config.apriltag_measurement_noise,
            gyro_measurement_noise=self.localization_config.gyro_measurement_noise,
        )

    def _current_state_for_controller(self) -> MapPoseVelocity:
        est = self.localization_filter.get_state()
        yaw = float(est[2])
        vx_body = float(est[3])
        vy_body = float(est[4])
        wz = float(est[5])
        c = np.cos(yaw)
        s = np.sin(yaw)
        vx_world = vx_body * c - vy_body * s
        vy_world = vx_body * s + vy_body * c
        return MapPoseVelocity(
            x=float(est[0]),
            y=float(est[1]),
            heading=yaw,
            vx=float(vx_world),
            vy=float(vy_world),
            heading_rate=wz,
        )

    def _maybe_update_apriltag(self) -> None:
        if self.apriltag_estimator is None:
            return
        try:
            pose = self.apriltag_estimator.get_pose_estimate()
        except Exception as exc:  # pragma: no cover
            print(f"WARN: AprilTag update failed: {exc}")
            return
        measurement = self._coerce_global_apriltag_measurement(pose)
        if measurement is not None:
            self.localization_filter.update_apriltag(measurement)

    def _camera_optical_point_to_robot_xy(self, point_camera: np.ndarray) -> np.ndarray:
        p_c = np.asarray(point_camera, dtype=float).reshape(3)
        point_camera_nominal = np.array([p_c[2], -p_c[0], -p_c[1]], dtype=float)

        cr = np.cos(self.camera_to_robot.roll)
        sr = np.sin(self.camera_to_robot.roll)
        cp = np.cos(self.camera_to_robot.pitch)
        sp = np.sin(self.camera_to_robot.pitch)
        cy = np.cos(self.camera_to_robot.yaw)
        sy = np.sin(self.camera_to_robot.yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        rot = rz @ ry @ rx
        trans = np.array([self.camera_to_robot.x, self.camera_to_robot.y, self.camera_to_robot.z], dtype=float)
        point_robot = trans + rot @ point_camera_nominal
        return point_robot[:2]

    def _coerce_global_apriltag_measurement(
        self, pose: Any
    ) -> Optional[AprilTagMeasurement]:
        if pose is None:
            return None
        if isinstance(pose, AprilTagMeasurement):
            return pose
        if isinstance(pose, dict) and {"x", "y", "yaw"}.issubset(pose.keys()):
            return AprilTagMeasurement(
                x=float(pose["x"]),
                y=float(pose["y"]),
                yaw=float(pose["yaw"]),
                covariance=self.localization_config.apriltag_measurement_noise,
            )
        if (
            isinstance(pose, (tuple, list, np.ndarray))
            and len(pose) == 3
            and all(np.isscalar(v) for v in pose)
        ):
            return AprilTagMeasurement(
                x=float(pose[0]),
                y=float(pose[1]),
                yaw=float(pose[2]),
                covariance=self.localization_config.apriltag_measurement_noise,
            )

        if isinstance(pose, dict):
            measurements: list[tuple[float, float, float]] = []
            for tag_id, raw_tag_pose in pose.items():
                landmark = self.landmark_lookup.get(str(tag_id))
                if landmark is None:
                    continue
                if not isinstance(raw_tag_pose, (tuple, list)) or len(raw_tag_pose) != 2:
                    continue
                _, t = raw_tag_pose
                t = np.asarray(t, dtype=float).reshape(-1)
                if t.size < 3:
                    continue

                tag_offset_robot = self._camera_optical_point_to_robot_xy(t)

                robot_yaw = wrap_to_pi(landmark.heading + np.pi - self.camera_to_robot.yaw)
                c = np.cos(robot_yaw)
                s = np.sin(robot_yaw)
                rot_robot_world = np.array([[c, -s], [s, c]], dtype=float)
                robot_world = np.asarray(landmark.point, dtype=float) - rot_robot_world @ tag_offset_robot
                measurements.append((robot_world[0], robot_world[1], robot_yaw))

            if measurements:
                meas = np.asarray(measurements, dtype=float)
                yaw = float(np.arctan2(np.mean(np.sin(meas[:, 2])), np.mean(np.cos(meas[:, 2]))))
                return AprilTagMeasurement(
                    x=float(np.mean(meas[:, 0])),
                    y=float(np.mean(meas[:, 1])),
                    yaw=yaw,
                    covariance=self.localization_config.apriltag_measurement_noise,
                )

        if not self._warned_raw_apriltag:
            print(
                "WARN: AprilTag detections could not be converted into a global pose. "
                "Check tag IDs/landmark names, landmark headings, and camera_to_robot transform."
            )
            self._warned_raw_apriltag = True
        return None

    def _setup_live_map_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt

            plt.ion()
            self.map_figure, self.map_axes = plt.subplots()
            self.map_figure.canvas.manager.set_window_title("Mission Map")
        except Exception as exc:  # pragma: no cover
            print(f"WARN: could not initialize live map plot: {exc}")
            self.map_figure = None
            self.map_axes = None

    def _update_live_map_plot(self, state: MapPoseVelocity) -> None:
        if self.map_axes is None or self.map_figure is None:
            return
        self.map_.current_position = (state.x, state.y)
        self.map_.draw(self.map_axes, current_position=self.map_.current_position)
        self.map_axes.set_title("Mission Map with Dynamic Obstacles")
        self.map_figure.canvas.draw_idle()
        self.map_figure.canvas.flush_events()

    def _update_dynamic_person_obstacles(self) -> None:
        if self.person_detector is None:
            return

        est = self.localization_filter.get_state()
        robot_pose = Pose2D(x=float(est[0]), y=float(est[1]), heading=float(est[2]))

        try:
            detections = self.person_detector.detect(robot_pose=robot_pose)
        except Exception as exc:  # pragma: no cover
            print(f"WARN: person detection update failed: {exc}")
            return

        self.map_.clear_obstacles_by_prefix(self._person_obstacle_prefix)
        self._dynamic_obstacle_packets = []
        recognized_people = 0
        for det in detections:
            if det.position_world_m is None:
                continue
            center = (float(det.position_world_m[0]), float(det.position_world_m[1]))
            if self.map_.world_to_cell(center) is None:
                continue
            obstacle_name = f"{self._person_obstacle_prefix}{det.track_id if det.track_id is not None else recognized_people}"
            self.map_.add_circular_obstacle(center, self._person_obstacle_radius_m, obstacle_name)
            self._dynamic_obstacle_packets.append(
                DynamicObstaclePacket(
                    name=obstacle_name,
                    center=[center[0], center[1]],
                    radius=float(self._person_obstacle_radius_m),
                )
            )
            recognized_people += 1

        self.blackboard.set("obstacle_blocking_path", recognized_people > 0)

    def _path_intersects_dynamic_obstacle(
        self,
        task_name: str,
        state: MapPoseVelocity,
    ) -> bool:
        planned = self.planned_lookup[task_name]
        remaining_points = [(state.x, state.y)] + [
            wp.xy for wp in planned.waypoints[planned.waypoint_index :]
        ]
        if len(remaining_points) < 2:
            return False

        step = max(0.02, 0.5 * self.map_.resolution)
        for p0, p1 in zip(remaining_points[:-1], remaining_points[1:]):
            p0_arr = np.asarray(p0, dtype=float)
            p1_arr = np.asarray(p1, dtype=float)
            delta = p1_arr - p0_arr
            seg_len = float(np.linalg.norm(delta))
            if seg_len < 1e-9:
                sample_points = [tuple(p0_arr.tolist())]
            else:
                n = max(2, int(np.ceil(seg_len / step)) + 1)
                sample_points = [
                    tuple((p0_arr + alpha * delta).tolist())
                    for alpha in np.linspace(0.0, 1.0, n)
                ]

            for sample in sample_points:
                cell = self.map_.world_to_cell(sample)
                if cell is None:
                    continue
                if self.map_.cell_has_obstacle(cell[0], cell[1]):
                    return True
        return False

    def _zero_mecanum_command(self) -> MecanumCommand:
        return MecanumCommand(
            vx=0.0,
            vy=0.0,
            omega=0.0,
            wheel_rates=(0.0, 0.0, 0.0, 0.0),
        )

    def _update_localization_from_imu(self, dt: float) -> None:
        imu_msgs = self.serial.read_parsed(max_lines=128)
        if not imu_msgs:
            self.localization_filter.predict(IMUMeasurement(ax=0.0, ay=0.0, wz=0.0), dt)
            return

        imu_msg = imu_msgs[-1]
        ax_meas = imu_msg.ax
        ay_meas = imu_msg.ay
        imu = IMUMeasurement(ax=ax_meas, ay=ay_meas, wz=imu_msg.gz)
        self.localization_filter.predict(imu, dt)
        self.localization_filter.update_imu(imu)

    def _publish_telemetry(
        self,
        *,
        now: float,
        current_task: str,
        distance_to_goal: float,
        heading_error: float,
    ) -> None:
        if self.telemetry_sock is None:
            return
        if self.telemetry_period > 0.0 and (now - self.last_telemetry_time) < self.telemetry_period:
            return

        est = self.localization_filter.get_state()
        cell = self.map_.world_to_cell((float(est[0]), float(est[1])))
        packet = TelemetryPacket(
            timestamp=time.time(),
            x=float(est[0]),
            y=float(est[1]),
            yaw=float(est[2]),
            vx_body=float(est[3]),
            vy_body=float(est[4]),
            wz=float(est[5]),
            cell=None if cell is None else [int(cell[0]), int(cell[1])],
            current_task=current_task,
            localization_ok=True,
            distance_to_goal=float(distance_to_goal),
            heading_error=float(heading_error),
            dynamic_obstacles=list(self._dynamic_obstacle_packets),
        )
        payload = json.dumps(asdict(packet)).encode("utf-8")
        self.telemetry_sock.sendto(payload, (self.telemetry_host, self.telemetry_port))
        self.last_telemetry_time = now

    def _update_elevator_from_serial(self, current_task_name: str) -> None:
        task = self.task_lookup[current_task_name]
        self.blackboard.set("desired_elevator_height_m", float(task.desired_elevator_height_m))

        if self.elevator_serial is None:
            return

        self.elevator_serial.send_height_cmd(task.desired_elevator_height_m)
        self.elevator_serial.flush_tx()
        for msg in self.elevator_serial.read_parsed(max_lines=32):
            self.blackboard.set("current_elevator_height_m", float(msg.height_m))

    def _update_blackboard(self, current_task_name: str) -> tuple[float, float]:
        task = self.task_lookup[current_task_name]
        est = self.localization_filter.get_state()
        goal_error = float(np.hypot(task.goal.x - est[0], task.goal.y - est[1]))
        heading_error = float(wrap_to_pi(task.goal.heading - est[2]))
        speed = float(np.hypot(est[3], est[4]))
        current_elevator_height_m = float(self.blackboard.get("current_elevator_height_m") or 0.0)
        desired_elevator_height_m = float(task.desired_elevator_height_m)
        elevator_height_error_m = desired_elevator_height_m - current_elevator_height_m

        self.blackboard.set("distance_to_goal", goal_error)
        self.blackboard.set("heading_error", heading_error)
        self.blackboard.set("current_elevator_height_m", current_elevator_height_m)
        self.blackboard.set("desired_elevator_height_m", desired_elevator_height_m)
        self.blackboard.set("elevator_height_error_m", elevator_height_error_m)
        self.blackboard.set("elevator_at_height", abs(elevator_height_error_m) <= 0.01)
        self.blackboard.set("robot_stopped", speed < 0.02)
        self.blackboard.set("tray_detected", goal_error < 0.10)
        self.blackboard.set("tray_released", goal_error < 0.06)
        self.blackboard.set("gripper_closed", goal_error >= 0.06)
        return goal_error, heading_error

    def _active_goal_waypoint(self, task_name: str, state: MapPoseVelocity):
        planned = self.planned_lookup[task_name]
        goal_wp = planned.waypoints[planned.waypoint_index]
        rho = float(np.hypot(goal_wp.xy[0] - state.x, goal_wp.xy[1] - state.y))
        if rho < self.runtime_config.waypoint_capture_radius and planned.waypoint_index < len(planned.waypoints) - 1:
            planned.waypoint_index += 1
            goal_wp = planned.waypoints[planned.waypoint_index]
        return goal_wp

    def run(self, max_ticks: Optional[int] = None) -> None:
        period = 1.0 / self.runtime_config.control_rate_hz
        tick = 0
        try:
            while True:
                if max_ticks is not None and tick >= max_ticks:
                    break

                now = time.monotonic()
                dt = max(now - self.last_loop_time, 1e-3)
                self.last_loop_time = now

                current_task = self.blackboard.get("current_task")
                if current_task == MISSION_DONE:
                    break

                self._update_localization_from_imu(dt)
                self._maybe_update_apriltag()
                self._update_dynamic_person_obstacles()
                self._update_elevator_from_serial(current_task)

                state = self._current_state_for_controller()
                goal_wp = self._active_goal_waypoint(current_task, state)
                cmd = self.controller.compute(state, goal_wp)
                path_blocked = self._path_intersects_dynamic_obstacle(current_task, state)
                if path_blocked:
                    cmd = self._zero_mecanum_command()
                self.blackboard.set("obstacle_blocking_path", path_blocked)
                self.serial.send_wheel_cmd(*cmd.wheel_rates)
                self.serial.flush_tx()

                goal_error, heading_error = self._update_blackboard(current_task)
                self._publish_telemetry(
                    now=now,
                    current_task=current_task,
                    distance_to_goal=goal_error,
                    heading_error=heading_error,
                )
                self.tree.tick()
                new_task = self.blackboard.get("current_task")

                self._update_live_map_plot(state)

                print(
                    f"tick={tick:04d} task={new_task} pos=({state.x:.3f}, {state.y:.3f}) "
                    f"yaw={state.heading:.3f} goal_err={goal_error:.3f} "
                    f"heading_err={heading_error:.3f} wheel_rates={tuple(round(v, 3) for v in cmd.wheel_rates)}"
                )

                tick += 1
                sleep_dt = period - (time.monotonic() - now)
                if sleep_dt > 0.0:
                    time.sleep(sleep_dt)
        finally:
            self.serial.send_wheel_cmd(0.0, 0.0, 0.0, 0.0, force=True)
            self.serial.flush_tx(force=True)
            self.serial.close()
            if self.elevator_serial is not None:
                self.elevator_serial.close()
            if self.shared_camera is not None:
                self.shared_camera.close()
            if self.telemetry_sock is not None:
                self.telemetry_sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(default_tasks_path()), help="Path to tasks.yaml")
    parser.add_argument("--port", default=None, help="Serial port for wheel controller ESP32")
    parser.add_argument("--elevator-port", default=None, help="Serial port for elevator controller ESP32")
    parser.add_argument("--disable-camera", action="store_true")
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--debug-serial", action="store_true")
    parser.add_argument("--telemetry-host", default=None, help="UDP host/IP for visualization telemetry")
    parser.add_argument("--telemetry-port", type=int, default=8765)
    parser.add_argument("--telemetry-rate-hz", type=float, default=10.0)
    args = parser.parse_args()

    runtime = MissionRuntime(
        tasks_path=Path(args.tasks),
        serial_port=args.port,
        elevator_serial_port=args.elevator_port,
        disable_camera=args.disable_camera,
        debug=args.debug_serial,
        telemetry_host=args.telemetry_host,
        telemetry_port=args.telemetry_port,
        telemetry_rate_hz=args.telemetry_rate_hz,
    )
    runtime.run(max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
