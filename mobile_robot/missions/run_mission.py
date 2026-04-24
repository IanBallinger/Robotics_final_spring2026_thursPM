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

from autonomy.mission_runner import Task, default_tasks_path, load_map, load_tasks  # noqa: E402
from autonomy.trees.waypoint_mission import (  # noqa: E402
    MISSION_DONE,
    create_tree,
    initialise_blackboard,
)
from guidance.waypoint_controller import (  # noqa: E402
    CascadedWaypointController,
    MapPoseVelocity,
    wrap_to_pi,
)
from localization import (  # noqa: E402
    AprilTagMeasurement,
    ExtendedKalmanFilter2D,
    IMUMeasurement,
    UnscentedKalmanFilter2D,
)

try:  # noqa: E402
    from localization.april_tag_pose_est import AprilTagPoseEst
except Exception:  # pragma: no cover
    AprilTagPoseEst = None  # type: ignore[assignment]

from localization.map import Map  # noqa: E402
from planning.a_star import AStar, waypoints_from_polyline  # noqa: E402
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
    """Placeholder planar camera extrinsics.

    x, y are the camera origin expressed in robot/body coordinates.
    yaw is the camera yaw relative to the robot frame.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


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


class MissionRuntime:
    def __init__(
        self,
        tasks_path: Path,
        serial_port: Optional[str],
        disable_camera: bool,
        camera_index: int,
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
        self.landmark_lookup = {str(landmark.name): landmark for landmark in self.map_.landmarks}

        self.blackboard = initialise_blackboard(
            self.tasks[0].name,
            state={
                "mission_enabled": True,
                "localization_ok": True,
                "battery_voltage": 24.0,
                "path_ready": True,
                "obstacle_blocking_path": False,
                "zone_clear": True,
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
        self.localization_filter = self._create_localization_filter()

        self.apriltag_estimator = None
        self.camera = None
        self._warned_raw_apriltag = False
        if not disable_camera and AprilTagPoseEst is not None:
            try:
                import cv2

                self.apriltag_estimator = AprilTagPoseEst()
                self.camera = cv2.VideoCapture(camera_index)
                if not self.camera.isOpened():
                    print("WARN: could not open camera; continuing without AprilTag updates")
                    self.camera.release()
                    self.camera = None
                    self.apriltag_estimator = None
            except Exception as exc:  # pragma: no cover
                print(f"WARN: failed to initialize AprilTag stack: {exc}")
                self.apriltag_estimator = None
                self.camera = None
        elif not disable_camera:
            print("WARN: AprilTag dependencies unavailable; continuing without camera localization")

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
        camera_to_robot = CameraToRobotTransform(
            x=float(cam.get("x", 0.0)),
            y=float(cam.get("y", 0.0)),
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
        if self.apriltag_estimator is None or self.camera is None:
            return
        ret, frame = self.camera.read()
        if not ret:
            return

        pose = self.apriltag_estimator.get_pose_estimate(frame)
        measurement = self._coerce_global_apriltag_measurement(pose)
        if measurement is not None:
            self.localization_filter.update_apriltag(measurement)

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

                tx = float(t[0])
                tz = float(t[2])

                # camera +z = robot forward, camera +x = robot right
                camera_offset_robot = np.array([tz, -tx], dtype=float)

                # placeholder camera->robot transform from tasks.yaml
                c_cr = np.cos(self.camera_to_robot.yaw)
                s_cr = np.sin(self.camera_to_robot.yaw)
                rot_cr = np.array([[c_cr, -s_cr], [s_cr, c_cr]], dtype=float)
                tag_offset_robot = np.array(
                    [self.camera_to_robot.x, self.camera_to_robot.y], dtype=float
                ) + rot_cr @ camera_offset_robot

                c = np.cos(landmark.heading)
                s = np.sin(landmark.heading)
                rot_tag_world = np.array([[c, -s], [s, c]], dtype=float)
                robot_world = np.asarray(landmark.point, dtype=float) + rot_tag_world @ tag_offset_robot
                robot_yaw = wrap_to_pi(landmark.heading + np.pi - self.camera_to_robot.yaw)
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

    def _update_localization_from_imu(self, dt: float) -> None:
        imu_msgs = self.serial.read_parsed(max_lines=128)
        if not imu_msgs:
            self.localization_filter.predict(IMUMeasurement(ax=0.0, ay=0.0, wz=0.0), dt)
            return

        imu_msg = imu_msgs[-1]
        ax_meas = imu_msg.ax - 0.4
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
        )
        payload = json.dumps(asdict(packet)).encode("utf-8")
        self.telemetry_sock.sendto(payload, (self.telemetry_host, self.telemetry_port))
        self.last_telemetry_time = now

    def _update_blackboard(self, current_task_name: str) -> tuple[float, float]:
        task = self.task_lookup[current_task_name]
        est = self.localization_filter.get_state()
        goal_error = float(np.hypot(task.goal.x - est[0], task.goal.y - est[1]))
        heading_error = float(wrap_to_pi(task.goal.heading - est[2]))
        speed = float(np.hypot(est[3], est[4]))

        self.blackboard.set("distance_to_goal", goal_error)
        self.blackboard.set("heading_error", heading_error)
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

                state = self._current_state_for_controller()
                goal_wp = self._active_goal_waypoint(current_task, state)
                cmd = self.controller.compute(state, goal_wp)
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
            if self.camera is not None:
                self.camera.release()
            if self.telemetry_sock is not None:
                self.telemetry_sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(default_tasks_path()), help="Path to tasks.yaml")
    parser.add_argument("--port", default=None, help="Serial port for wheel controller ESP32")
    parser.add_argument("--camera-index", type=int, default=0)
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
        disable_camera=args.disable_camera,
        camera_index=args.camera_index,
        debug=args.debug_serial,
        telemetry_host=args.telemetry_host,
        telemetry_port=args.telemetry_port,
        telemetry_rate_hz=args.telemetry_rate_hz,
    )
    runtime.run(max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
