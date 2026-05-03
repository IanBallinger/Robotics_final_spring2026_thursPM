"""
UR5 Julia-backed trace replay task.

This task uses the existing UR5TaskInterface orchestration while delegating
path planning/smoothing to a Julia API service.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib import error, request

from rtde_control import RTDEControlInterface

from ur5_task_interface import UR5TaskInterface

# Keep demo_4_16 gripper helper importable even when this script is run from repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.join(_THIS_DIR, "demo_4_16")
if _DEMO_DIR not in sys.path:
    sys.path.append(_DEMO_DIR)

from robotiq_gripper_control import RobotiqGripper  # noqa: E402


@dataclass
class GripperEvent:
    timestamp: float
    open_state: bool


class JuliaReplayTask(UR5TaskInterface):
    """Replay task that asks Julia for planned waypoints and replays gripper events."""

    def __init__(
        self,
        robot_ip: str,
        trace_csv_path: str,
        julia_api_base: str = "http://127.0.0.1:8081",
        trace_side: str = "left",
        downsample: int = 4,
        gripper_state_column: str = "actual_digital_output_bits",
        gripper_bit_index: int = 0,
        gripper_closed_when_bit_set: bool = True,
        planning_timeout_s: float = 10.0,
        connect_immediately: bool = False,
    ):
        super().__init__(robot_ip=robot_ip, connect_immediately=connect_immediately)
        self.trace_csv_path = trace_csv_path
        self.julia_api_base = julia_api_base.rstrip("/")
        self.trace_side = trace_side.lower()
        self.downsample = max(1, int(downsample))
        self.gripper_state_column = gripper_state_column
        self.gripper_bit_index = max(0, int(gripper_bit_index))
        self.gripper_closed_when_bit_set = bool(gripper_closed_when_bit_set)
        self.planning_timeout_s = float(planning_timeout_s)

        self.pause_event = threading.Event()
        self.stop_event = threading.Event()

        self.trace_poses: List[List[float]] = []
        self.planned_poses: List[List[float]] = []
        self.gripper_events: List[GripperEvent] = []

        self.gripper_control: Optional[RTDEControlInterface] = None
        self.gripper: Optional[RobotiqGripper] = None

    def setup(self):
        if not os.path.exists(self.trace_csv_path):
            raise FileNotFoundError(f"Trace CSV not found: {self.trace_csv_path}")

        self.trace_poses = self._load_trace_poses(self.trace_csv_path)
        if not self.trace_poses:
            raise RuntimeError("Trace file did not contain usable TCP poses")

        self.planned_poses = self._request_julia_plan(self.trace_poses)
        if not self.planned_poses:
            self.planned_poses = self.trace_poses

        self.gripper_events = self._extract_gripper_events_from_trace_csv(self.trace_csv_path)

        # Home defaults to current joints to avoid unexpected reconfiguration moves.
        current_q = self.robot.get_joint_positions() if self.robot else None
        if current_q and len(current_q) == 6:
            self.home_position = list(current_q)

        self._init_gripper()

        print(
            f"[SETUP] JuliaReplayTask ready: {len(self.trace_poses)} raw poses, "
            f"{len(self.planned_poses)} planned poses, {len(self.gripper_events)} gripper events"
        )
        return True

    def perform_task_logic(self):
        if not self.planned_poses:
            print("[ERROR] No planned poses to replay")
            return False

        # Align gripper events to replay timeline using first event timestamp.
        event_idx = 0
        event_t0 = self.gripper_events[0].timestamp if self.gripper_events else 0.0
        replay_t0 = time.time()
        paused_once = False

        for i, pose in enumerate(self.planned_poses):
            if self.stop_event.is_set():
                print("[INFO] Replay stop requested")
                return False

            while self.pause_event.is_set():
                if not paused_once:
                    self.robot.stop_arm()
                    paused_once = True
                time.sleep(0.05)

            if paused_once:
                print("[INFO] Replay resumed")
                paused_once = False

            elapsed = time.time() - replay_t0
            event_idx = self._apply_gripper_events(event_idx, event_t0 + elapsed)

            ok = self.robot.move_linear_to_pose(pose, asynchronous=False)
            if not ok:
                print(f"[ERROR] Replay move failed at index {i}")
                return False

        # Apply any remaining gripper events at end-of-trace.
        self._apply_gripper_events(event_idx, float("inf"))
        print("[TASK] Replay completed")
        return True

    def cleanup(self):
        if self.gripper_control is not None:
            try:
                self.gripper_control.disconnect()
            except (RuntimeError, OSError):
                pass
        super().cleanup()

    def request_pause(self):
        self.pause_event.set()

    def request_resume(self):
        self.pause_event.clear()

    def request_stop(self):
        self.stop_event.set()

    def _load_trace_poses(self, csv_path: str) -> List[List[float]]:
        poses: List[List[float]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            required = [f"actual_TCP_pose_{i}" for i in range(6)]
            if not reader.fieldnames or not all(name in reader.fieldnames for name in required):
                raise ValueError("Trace CSV missing actual_TCP_pose_0..5 columns")

            for row_idx, row in enumerate(reader):
                if row_idx % self.downsample != 0:
                    continue
                try:
                    pose = [float(row[f"actual_TCP_pose_{i}"]) for i in range(6)]
                    poses.append(pose)
                except (TypeError, ValueError, KeyError):
                    continue

        return poses

    def _extract_gripper_events_from_trace_csv(self, csv_path: str) -> List[GripperEvent]:
        events: List[GripperEvent] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return events

            if self.gripper_state_column not in reader.fieldnames:
                print(
                    f"[WARN] Gripper state column '{self.gripper_state_column}' not found in CSV; "
                    "gripper replay disabled"
                )
                return events

            last_open_state: Optional[bool] = None
            for row in reader:
                try:
                    timestamp = float(row["timestamp"])
                    bitfield = int(float(row[self.gripper_state_column]))
                    bit_is_set = bool((bitfield >> self.gripper_bit_index) & 0x1)
                    is_closed = bit_is_set if self.gripper_closed_when_bit_set else (not bit_is_set)
                    is_open = not is_closed

                    if last_open_state is None or is_open != last_open_state:
                        events.append(GripperEvent(timestamp=timestamp, open_state=is_open))
                        last_open_state = is_open
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

        events.sort(key=lambda e: e.timestamp)
        return events

    def _request_julia_plan(self, raw_poses: List[List[float]]) -> List[List[float]]:
        payload = {
            "trace_csv_path": self.trace_csv_path,
            "trace_side": self.trace_side,
            "downsample": self.downsample,
            "raw_pose_count": len(raw_poses),
            "raw_poses": raw_poses,
        }

        for endpoint in ("/replay/plan", "/plan_replay"):
            response = self._post_json(endpoint, payload)
            if response is None:
                continue

            waypoints = response.get("waypoints", [])
            parsed: List[List[float]] = []
            for wp in waypoints:
                if isinstance(wp, list) and len(wp) >= 6:
                    try:
                        parsed.append([float(v) for v in wp[:6]])
                    except (TypeError, ValueError):
                        continue

            if parsed:
                print(f"[INFO] Julia planner returned {len(parsed)} waypoints from {endpoint}")
                return parsed

        print("[WARN] Julia planner unavailable or empty response; replaying raw trace")
        return raw_poses

    def _post_json(self, endpoint: str, payload: dict) -> Optional[dict]:
        url = f"{self.julia_api_base}{endpoint}"
        req = request.Request(
            url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.planning_timeout_s) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"[WARN] Julia API call failed ({url}): {exc}")
            return None

    def _init_gripper(self):
        try:
            self.gripper_control = RTDEControlInterface(self.robot_ip)
            self.gripper = RobotiqGripper(self.gripper_control)
            self.gripper.set_force(50)
            self.gripper.set_speed(100)
            print("[INFO] Gripper replay enabled")
        except (RuntimeError, OSError, AttributeError) as exc:
            self.gripper = None
            print(f"[WARN] Gripper replay disabled: {exc}")

    def _apply_gripper_events(self, event_idx: int, current_event_time: float) -> int:
        if self.gripper is None:
            return len(self.gripper_events)

        while event_idx < len(self.gripper_events):
            event = self.gripper_events[event_idx]
            if event.timestamp > current_event_time:
                break

            desired_open = event.open_state
            try:
                if desired_open:
                    self.gripper.open()
                else:
                    self.gripper.close()
            except (RuntimeError, OSError) as exc:
                print(f"[WARN] Gripper command failed: {exc}")
            event_idx += 1

        return event_idx
