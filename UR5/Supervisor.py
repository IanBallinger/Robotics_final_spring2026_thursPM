"""Supervisor

Unified orchestration module that combines:
- dual-arm recording (pose + camera streaming)
- gripper handling
- task-frame coordinate transforms
- collaborative subtask registry

This file intentionally keeps high-level logic readable for team editing.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from pynput.keyboard import Key, Listener
from arm import UR5Arm

# Keep compatibility with working gripper helper from demo prototype.
_THIS_DIR = Path(__file__).resolve().parent
_DEMO_DIR = _THIS_DIR / "demo_4_16"
if str(_DEMO_DIR) not in sys.path:
    sys.path.append(str(_DEMO_DIR))
from robotiq_gripper_control import RobotiqGripper  # noqa: E402


LEFT_ARM_IP = "192.168.1.101"
RIGHT_ARM_IP = "192.168.1.102"

DEFAULT_RTDE_VARIABLES = [
    "timestamp",
    "target_q",
    "target_qd",
    "target_qdd",
    "target_current",
    "target_moment",
    "actual_q",
    "actual_qd",
    "actual_current",
    "joint_control_output",
    "actual_TCP_pose",
    "actual_TCP_speed",
    "actual_TCP_force",
    "target_TCP_pose",
    "target_TCP_speed",
    "actual_digital_input_bits",
    "joint_temperatures",
    "actual_execution_time",
    "robot_mode",
    "joint_mode",
    "safety_mode",
    "actual_tool_accelerometer",
    "speed_scaling",
    "target_speed_fraction",
    "actual_momentum",
    "actual_main_voltage",
    "actual_robot_voltage",
    "actual_robot_current",
    "actual_joint_voltage",
    "actual_digital_output_bits",
    "runtime_state",
    "standard_analog_input0",
    "standard_analog_input1",
    "standard_analog_output0",
    "standard_analog_output1",
    "robot_status_bits",
    "safety_status_bits",
]

CAMERA_COLORS = {
    "purple": {"lower": np.array([156, 83, 0]), "upper": np.array([180, 176, 143])},
    "yellow": {"lower": np.array([13, 255, 120]), "upper": np.array([98, 255, 208])},
    "green": {"lower": np.array([53, 90, 128]), "upper": np.array([87, 180, 221])},
    "tan": {"lower": np.array([40, 71, 139]), "upper": np.array([56, 195, 255])},
    "red": {"lower": np.array([1, 180, 131]), "upper": np.array([3, 255, 236])},
}
CM_PIXEL = 54.0 / 275
CENTER_X = 337.5
CENTER_Y = 337.5
VISION_Z = 0.1


@dataclass
class Subtask:
    """Intended to be an editable subtask registration object. @team add your work here via `subtasks` folder."""

    name: str
    description: str
    runner: Callable[["Supervisor", dict], bool]


@dataclass
class SupervisorState:
    """Shared state for coordination endpoints and local controls."""

    returning: bool = False
    paused: bool = False
    status_messages: List[str] = field(default_factory=list)
    completed_tasks: List[int] = field(default_factory=list)


@dataclass(order=True)
class QueuedTask:
    """Priority-queue entry for autonomous mode (higher points run first)."""

    queue_weight: float
    sequence: int
    name: str = field(compare=False)
    task_id: str = field(compare=False, default="")
    base_points: float = field(compare=False, default=0.0)
    points_if_completed: Dict[str, float] = field(compare=False, default_factory=dict)
    prerequisites: List[str] = field(compare=False, default_factory=list)
    arm: str = field(compare=False, default="any")
    blocks_arms: List[str] = field(compare=False, default_factory=list)
    unblocks_arms: List[str] = field(compare=False, default_factory=list)
    score_token: str = field(compare=False, default="")
    max_score_count: int = field(compare=False, default=0)
    resource_action: str = field(compare=False, default="")
    resource: str = field(compare=False, default="")
    resource_item: str = field(compare=False, default="")
    params: dict = field(compare=False, default_factory=dict)


class Supervisor:
    """Main orchestration class for dual-arm collaboration."""

    def __init__(
        self,
        left_ip: str = LEFT_ARM_IP,
        right_ip: str = RIGHT_ARM_IP,
        subtasks_dir: Optional[Path] = None,
    ):
        self.left_ip = left_ip
        self.right_ip = right_ip
        self.state = SupervisorState()
        self._state_lock = threading.Lock()
        self.subtasks: Dict[str, Subtask] = {}
        self._task_heap: List[QueuedTask] = []
        self._task_sequence = 0
        self._completed_tokens: Set[str] = set()
        self._blocked_arms: Dict[str, str] = {}
        self._points_map: Dict[str, Dict[str, float]] = {}
        self._resource_constraints: Dict[str, dict] = {}
        self._resource_state: Dict[str, Dict[str, int]] = {}
        self._score_counts: Dict[str, int] = {}
        self._earned_points_total: float = 0.0
        self.subtasks_dir = subtasks_dir or (_THIS_DIR / "subtasks")
        self._ensure_subtasks_dir()
        self.reload_subtasks()

    # ===== Coordination API-like methods =====
    def mark_returning(self, message: str = "on our way back now, get ready for handoff"):
        with self._state_lock:
            self.state.returning = True
            self.state.status_messages.append(message)

    def mark_ready(self):
        with self._state_lock:
            self.state.returning = False

    def mark_complete(self, task_number: int):
        with self._state_lock:
            if task_number not in self.state.completed_tasks:
                self.state.completed_tasks.append(task_number)

    def pause(self):
        with self._state_lock:
            self.state.paused = True

    def play(self):
        with self._state_lock:
            self.state.paused = False

    # ===== Coordinate frame methodology (from keyboard_proto.py) =====
    @staticmethod
    def _matrix_to_rotvec(rot: np.ndarray) -> List[float]:
        """Convert 3x3 rotation matrix to axis-angle vector without SciPy dependency."""
        trace = float(np.trace(rot))
        cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        theta = float(np.arccos(cos_theta))
        if abs(theta) < 1e-9:
            return [0.0, 0.0, 0.0]

        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-9:
            # Degenerate ~pi case; use diagonal fallback.
            axis = np.sqrt(np.maximum((np.diag(rot) + 1.0) / 2.0, 0.0))
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            return (axis * theta).tolist()

        axis = np.array(
            [
                rot[2, 1] - rot[1, 2],
                rot[0, 2] - rot[2, 0],
                rot[1, 0] - rot[0, 1],
            ],
            dtype=float,
        ) / (2.0 * sin_theta)
        return (axis * theta).tolist()

    def compute_task_frames(self) -> Dict[str, List[float]]:
        """Compute left/right task frames matching the existing methodology."""
        dy_t = 0.225 / 2 + 0.540 / 2
        dz_t = -0.753

        dx_left = 0.090 / 2 + 0.010 + 0.110
        rot_left = np.array(
            [[0.707, 0, -0.707], [0, -1, 0], [-0.707, 0, -0.707]],
            dtype=float,
        )
        trans_left = np.matmul([dx_left, dy_t, dz_t], rot_left).tolist()
        task_frame_left = trans_left + self._matrix_to_rotvec(rot_left)

        dx_right = -(0.090 / 2 + 0.010 + 0.110)
        rot_right = np.array(
            [[0.707, 0, 0.707], [0, -1, 0], [0.707, 0, -0.707]],
            dtype=float,
        )
        trans_right = np.matmul([dx_right, dy_t, dz_t], rot_right).tolist()
        task_frame_right = trans_right + self._matrix_to_rotvec(rot_right)

        return {"left": task_frame_left, "right": task_frame_right}

    # ===== Recording + camera integration (merged from record.py) =====
    def _start_camera_thread(
        self,
        udp_socket: socket.socket,
        udp_target: tuple[str, int],
        stop_event: threading.Event,
        send_lock: threading.Lock,
        camera_index: int,
    ) -> threading.Thread:
        def worker():
            cap = cv2.VideoCapture(camera_index)
            kernel = np.ones((5, 5), np.uint8)
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                for color_name, rng in CAMERA_COLORS.items():
                    mask = cv2.inRange(hsv, rng["lower"], rng["upper"])
                    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
                    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)
                    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 500:
                            continue
                        x, y, w, h = cv2.boundingRect(cnt)
                        xc = x + w // 2
                        yc = y + h // 2
                        x_cm = (xc - CENTER_X) * CM_PIXEL
                        y_cm = (CENTER_Y - yc) * CM_PIXEL
                        packet = {
                            "position": [round(x_cm / 100.0, 4), round(y_cm / 100.0, 4), VISION_Z],
                            "color": color_name,
                            "x_cm": round(float(x_cm), 3),
                            "y_cm": round(float(y_cm), 3),
                        }
                        try:
                            with send_lock:
                                udp_socket.sendto(json.dumps(packet).encode("utf-8"), udp_target)
                        except OSError:
                            pass

                cv2.imshow("Camera (Supervisor)", frame)
                cv2.waitKey(3)

            cap.release()
            cv2.destroyAllWindows()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return t

    def run_recording_session(
        self,
        output_base: str = "robot_data",
        frequency: float = 500.0,
        stream_udp_host: str = "",
        stream_udp_port: int = 9999,
        no_robot: bool = False,
        camera_enabled: bool = True,
        camera_index: int = 2,
    ):
        """Run synchronized left/right recording with optional UDP + camera stream."""
        dt = 1.0 / frequency
        udp_socket: Optional[socket.socket] = None
        udp_target = None
        send_lock = threading.Lock()

        if stream_udp_host:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_target = (stream_udp_host, int(stream_udp_port))

        camera_stop = threading.Event()
        camera_thread = None
        if camera_enabled and udp_socket is not None and udp_target is not None:
            camera_thread = self._start_camera_thread(
                udp_socket, udp_target, camera_stop, send_lock, camera_index
            )

        if no_robot:
            print("[Supervisor] No-robot mode active; camera stream only.")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                camera_stop.set()
                if camera_thread:
                    camera_thread.join(timeout=2)
                if udp_socket:
                    udp_socket.close()
            return

        left_arm = UR5Arm(self.left_ip, frequency=frequency, verbose=False)
        right_arm = UR5Arm(self.right_ip, frequency=frequency, verbose=False)
        rtde_r_left = left_arm.rtde_receive
        rtde_r_right = right_arm.rtde_receive
        rtde_c_left = left_arm.rtde_control
        rtde_c_right = right_arm.rtde_control

        gripper_left = None
        gripper_right = None
        try:
            gripper_left = RobotiqGripper(rtde_c_left)
            gripper_right = RobotiqGripper(rtde_c_right)
            for g in (gripper_left, gripper_right):
                g.activate()
                g.set_force(50)
                g.set_speed(100)
                g.open()
        except Exception as exc:
            print(f"[Supervisor] Gripper init warning (should you be in remote mode?): {exc}")
            gripper_left = None
            gripper_right = None

        output_path = Path(output_base)
        left_output = str(output_path.with_name(f"{output_path.stem}_left.csv"))
        right_output = str(output_path.with_name(f"{output_path.stem}_right.csv"))

        rtde_r_left.startFileRecording(left_output, DEFAULT_RTDE_VARIABLES)
        rtde_r_right.startFileRecording(right_output, DEFAULT_RTDE_VARIABLES)
        print(f"[Supervisor] Recording LEFT -> {left_output}")
        print(f"[Supervisor] Recording RIGHT -> {right_output}")

        gripper_state = {"left": True, "right": True}
        gripper_lock = threading.Lock()
        recording_active = True

        def on_press(key):
            nonlocal recording_active
            try:
                if key.char == "l":
                    with gripper_lock:
                        gripper_state["left"] = not gripper_state["left"]
                elif key.char == "r":
                    with gripper_lock:
                        gripper_state["right"] = not gripper_state["right"]
            except AttributeError:
                if key == Key.delete:
                    recording_active = False

        listener = Listener(on_press=on_press)
        listener.start()

        try:
            i = 0
            while recording_active:
                start = time.time()

                left_ts = rtde_r_left.getTimestamp()
                right_ts = rtde_r_right.getTimestamp()
                left_pose = rtde_r_left.getActualTCPPose()
                right_pose = rtde_r_right.getActualTCPPose()

                if udp_socket and udp_target:
                    packet = {
                        "timestamp": left_ts,
                        "actual_TCP_pose": left_pose,
                        "left_timestamp": left_ts,
                        "right_timestamp": right_ts,
                        "left_actual_TCP_pose": left_pose,
                        "right_actual_TCP_pose": right_pose,
                    }
                    with send_lock:
                        udp_socket.sendto(json.dumps(packet).encode("utf-8"), udp_target)

                with gripper_lock:
                    l_open = gripper_state["left"]
                    r_open = gripper_state["right"]
                if gripper_left is not None:
                    gripper_left.open() if l_open else gripper_left.close()
                if gripper_right is not None:
                    gripper_right.open() if r_open else gripper_right.close()

                if i % 10 == 0:
                    sys.stdout.write("\r")
                    sys.stdout.write(f"{i:3d} samples.")
                    sys.stdout.flush()
                i += 1

                duration = time.time() - start
                if duration < dt:
                    time.sleep(dt - duration)

        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
            try:
                rtde_r_left.stopFileRecording()
            except Exception:
                pass
            try:
                rtde_r_right.stopFileRecording()
            except Exception:
                pass
            camera_stop.set()
            if camera_thread:
                camera_thread.join(timeout=2)
            if udp_socket:
                udp_socket.close()
            try:
                left_arm.disconnect()
            except Exception:
                pass
            try:
                right_arm.disconnect()
            except Exception:
                pass
            print("\n[Supervisor] Recording stopped.")

    # ===== Collaborative subtask plugin structure =====
    def _ensure_subtasks_dir(self):
        self.subtasks_dir.mkdir(parents=True, exist_ok=True)
        init_py = self.subtasks_dir / "__init__.py"
        if not init_py.exists():
            init_py.write_text(
                '"""Subtask plugin package. Add files with a register_subtasks(registry) function."""\n',
                encoding="utf-8",
            )

        example_py = self.subtasks_dir / "example_subtask.py"
        if not example_py.exists():
            example_py.write_text(
                """def register_subtasks(registry):
    # Keep task logic short and explicit for easy team edits.
    def _example(supervisor, params):
        print(\"[example_subtask] params:\", params)
        frames = supervisor.compute_task_frames()
        print(\"[example_subtask] task frames computed for left/right\")
        print(frames)
        return True

    registry[\"example\"] = {
        \"description\": \"Minimal collaborative subtask template\",
        \"runner\": _example,
    }
""",
                encoding="utf-8",
            )

    def reload_subtasks(self):
        self.subtasks = {}
        for plugin_path in sorted(self.subtasks_dir.glob("*.py")):
            if plugin_path.name == "__init__.py":
                continue
            module_name = f"ur5_subtask_{plugin_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            register = getattr(module, "register_subtasks", None)
            if register is None:
                continue

            registry: Dict[str, dict] = {}
            register(registry)
            for name, entry in registry.items():
                runner = entry.get("runner")
                description = entry.get("description", "")
                if callable(runner):
                    self.subtasks[name] = Subtask(name=name, description=description, runner=runner)

    def list_subtasks(self) -> List[str]:
        return [f"{name}: {subtask.description}" for name, subtask in sorted(self.subtasks.items())]

    def run_subtask(self, name: str, params: Optional[dict] = None) -> bool:
        params = params or {}
        if name not in self.subtasks:
            raise KeyError(f"Unknown subtask: {name}")
        return bool(self.subtasks[name].runner(self, params))

    # ===== Full-autonomy priority queue =====
    @staticmethod
    def _normalize_arm_list(raw_arms: Optional[List[str]]) -> List[str]:
        allowed = {"left", "right"}
        if raw_arms is None:
            return []
        normalized: List[str] = []
        for arm in raw_arms:
            arm_name = str(arm).strip().lower()
            if arm_name not in allowed:
                raise ValueError(f"Unknown arm '{arm_name}'. Expected one of: left, right")
            if arm_name not in normalized:
                normalized.append(arm_name)
        return normalized

    @staticmethod
    def _validate_arm_selector(arm: str) -> str:
        normalized = str(arm).strip().lower()
        if normalized not in {"any", "left", "right", "both"}:
            raise ValueError("arm must be one of: any, left, right, both")
        return normalized

    def _task_target_arms(self, task: QueuedTask) -> List[str]:
        if task.arm == "left":
            return ["left"]
        if task.arm == "right":
            return ["right"]
        if task.arm == "both":
            return ["left", "right"]
        return []

    def _prerequisites_met(self, task: QueuedTask) -> bool:
        return all(token in self._completed_tokens for token in task.prerequisites)

    def _missing_prerequisites(self, task: QueuedTask) -> List[str]:
        return [token for token in task.prerequisites if token not in self._completed_tokens]

    def _effective_points(self, task: QueuedTask) -> float:
        if not self._resource_action_allowed(task):
            return 0.0
        if not self._resource_gate_allowed(task):
            return 0.0
        if not self._arm_gate_allowed(task):
            return 0.0

        bonus_points = 0.0
        for token, delta in task.points_if_completed.items():
            if token in self._completed_tokens:
                bonus_points += float(delta)

        dynamic_base_points, max_score_count, from_points_map = self._score_config_for_task(task)
        score_token = self._resolve_score_token(task)
        if max_score_count > 0 and self._score_counts.get(score_token, 0) >= max_score_count:
            # Once points-map cap is depleted, keep autonomous progression by using inline points.
            if from_points_map:
                return float(task.base_points) + bonus_points
            return 0.0

        return float(dynamic_base_points) + bonus_points

    def _score_config_for_task(self, task: QueuedTask) -> Tuple[float, int, bool]:
        token = self._resolve_score_token(task)
        mapped = self._points_map.get(token)
        if mapped is not None:
            return (
                float(mapped.get("points", 0.0)),
                self._normalize_max_score_count(mapped.get("max_score_count", 0)),
                True,
            )
        return (float(task.base_points), task.max_score_count, False)

    @staticmethod
    def _normalize_max_score_count(value: Optional[int]) -> int:
        if value is None:
            return 0
        normalized = int(value)
        if normalized < 0:
            raise ValueError("max_score_count must be >= 0")
        return normalized

    @staticmethod
    def _normalize_score_token(value: Optional[str], fallback: str) -> str:
        token = str(value).strip() if value is not None else ""
        return token if token else fallback

    def _resolve_score_token(self, task: QueuedTask) -> str:
        if task.score_token:
            return task.score_token
        return f"task:{task.name}"

    def _record_score_if_applicable(self, task: QueuedTask, awarded_points: float):
        if awarded_points <= 0.0:
            return

        _, max_score_count, _ = self._score_config_for_task(task)
        token = self._resolve_score_token(task)
        if max_score_count > 0 and self._score_counts.get(token, 0) >= max_score_count:
            return

        self._score_counts[token] = self._score_counts.get(token, 0) + 1
        self._earned_points_total += awarded_points

    def configure_points_map(self, points_map: Optional[dict]):
        self._points_map = {}
        if not points_map:
            return
        if not isinstance(points_map, dict):
            raise ValueError("points_map must be a JSON object of score_token -> {points, max_score_count}")

        for token, entry in points_map.items():
            if not isinstance(entry, dict):
                raise ValueError(f"points_map entry '{token}' must be a JSON object")
            points = float(entry.get("points", 0.0))
            max_score_count = self._normalize_max_score_count(entry.get("max_score_count", 0))
            self._points_map[str(token)] = {
                "points": points,
                "max_score_count": float(max_score_count),
            }

    @staticmethod
    def _normalize_resource_action(value: Optional[str]) -> str:
        if value is None:
            return ""
        normalized = str(value).strip().lower()
        if normalized and normalized not in {"acquire", "release"}:
            raise ValueError("resource_action must be one of: acquire, release")
        return normalized

    def configure_resource_constraints(self, resource_constraints: Optional[dict]):
        self._resource_constraints = {}
        self._resource_state = {}
        if not resource_constraints:
            return
        if not isinstance(resource_constraints, dict):
            raise ValueError("resource_constraints must be a JSON object")

        for resource_name, entry in resource_constraints.items():
            if not isinstance(entry, dict):
                raise ValueError(f"resource_constraints entry '{resource_name}' must be a JSON object")

            capacity = int(entry.get("capacity", 0))
            if capacity <= 0:
                raise ValueError(f"resource '{resource_name}' must define capacity > 0")

            mutex_items_raw = entry.get("mutex_items", [])
            if not isinstance(mutex_items_raw, list):
                raise ValueError(f"resource '{resource_name}' mutex_items must be a JSON array")

            self._resource_constraints[str(resource_name)] = {
                "capacity": capacity,
                "mutex_items": {str(item) for item in mutex_items_raw},
            }
            self._resource_state[str(resource_name)] = {}

    def _resource_action_allowed(self, task: QueuedTask) -> bool:
        if not task.resource_action or not task.resource:
            return True

        cfg = self._resource_constraints.get(task.resource)
        if cfg is None:
            # Unknown resource means graph didn't declare constraints; don't block task.
            return True

        held = self._resource_state.get(task.resource, {})
        item = task.resource_item
        capacity = int(cfg.get("capacity", 0))
        mutex_items: Set[str] = set(cfg.get("mutex_items", set()))

        if task.resource_action == "acquire":
            total_held = sum(int(v) for v in held.values())
            if total_held >= capacity:
                return False

            if item and item in mutex_items:
                for held_item, count in held.items():
                    if held_item in mutex_items and held_item != item and int(count) > 0:
                        return False
            return True

        if task.resource_action == "release":
            if not item:
                return False
            return int(held.get(item, 0)) > 0

        return True

    def _resource_gate_allowed(self, task: QueuedTask) -> bool:
        """Optional gating for tasks that are not direct acquire/release actions.

        Use params:
        - requires_resource_available_for: <resource_name>
 sh        - requires_resource_occupied_for: <resource_name>
        - requires_resource_item: <item_name> (optional, defaults to target_label)
        """
        params = task.params or {}
        item = str(params.get("requires_resource_item", params.get("target_label", ""))).strip()

        resource_name = str(params.get("requires_resource_available_for", "")).strip()
        if resource_name:
            cfg = self._resource_constraints.get(resource_name)
            if cfg is None:
                # Graph requested a gate for unknown resource; treat as blocked so it's explicit.
                return False

            held = self._resource_state.get(resource_name, {})
            capacity = int(cfg.get("capacity", 0))
            mutex_items: Set[str] = set(cfg.get("mutex_items", set()))

            total_held = sum(int(v) for v in held.values())
            if total_held >= capacity:
                return False

            if item and item in mutex_items:
                for held_item, count in held.items():
                    if held_item in mutex_items and held_item != item and int(count) > 0:
                        return False

        occupied_resource_name = str(params.get("requires_resource_occupied_for", "")).strip()
        if occupied_resource_name:
            cfg = self._resource_constraints.get(occupied_resource_name)
            if cfg is None:
                return False

            held = self._resource_state.get(occupied_resource_name, {})
            total_held = sum(int(v) for v in held.values())
            if total_held <= 0:
                return False
            if item:
                if int(held.get(item, 0)) <= 0:
                    return False

        return True

    def _arm_gate_allowed(self, task: QueuedTask) -> bool:
        """Optional gating based on other arm block state.

        Use params:
        - requires_arms_unblocked: ["left", "right"]
        """
        params = task.params or {}
        required = params.get("requires_arms_unblocked", [])
        if not required:
            return True
        if not isinstance(required, list):
            raise ValueError("requires_arms_unblocked must be a JSON array")

        for arm in required:
            arm_name = str(arm).strip().lower()
            if arm_name not in {"left", "right"}:
                raise ValueError("requires_arms_unblocked supports only 'left' or 'right'")
            if arm_name in self._blocked_arms:
                return False
        return True

    def _apply_resource_action(self, task: QueuedTask):
        if not task.resource_action or not task.resource:
            return

        if task.resource not in self._resource_state:
            self._resource_state[task.resource] = {}

        held = self._resource_state[task.resource]
        item = task.resource_item
        if not item:
            return

        if task.resource_action == "acquire":
            held[item] = int(held.get(item, 0)) + 1
            return

        if task.resource_action == "release":
            current = int(held.get(item, 0))
            if current <= 1:
                held.pop(item, None)
            else:
                held[item] = current - 1

    def _is_task_runnable(self, task: QueuedTask) -> bool:
        if not self._prerequisites_met(task):
            return False

        for arm in self._task_target_arms(task):
            if arm in self._blocked_arms and arm not in task.unblocks_arms:
                return False
        if not self._resource_action_allowed(task):
            return False
        if not self._resource_gate_allowed(task):
            return False
        if not self._arm_gate_allowed(task):
            return False
        return True

    def _refresh_queue_weights(self):
        for queued in self._task_heap:
            queued.queue_weight = -self._effective_points(queued)
        heapq.heapify(self._task_heap)

    def queue_task(
        self,
        name: str,
        points: float = 0.0,
        params: Optional[dict] = None,
        task_id: Optional[str] = None,
        prerequisites: Optional[List[str]] = None,
        points_if_completed: Optional[Dict[str, float]] = None,
        arm: str = "any",
        blocks_arms: Optional[List[str]] = None,
        unblocks_arms: Optional[List[str]] = None,
        score_token: Optional[str] = None,
        max_score_count: int = 0,
        validate_subtask_exists: bool = True,
    ):
        if name == "total_replay":
            raise ValueError("'total_replay' is linear playback and is not allowed in autonomy mode")
        if validate_subtask_exists and name not in self.subtasks:
            raise KeyError(f"Unknown subtask: {name}")

        arm_selector = self._validate_arm_selector(arm)
        blocks = self._normalize_arm_list(blocks_arms)
        unblocks = self._normalize_arm_list(unblocks_arms)
        prereq_list = [str(token) for token in (prerequisites or [])]
        bonus_map = {str(token): float(delta) for token, delta in (points_if_completed or {}).items()}
        resolved_task_id = str(task_id) if task_id else f"{name}#{self._task_sequence}"
        normalized_score_token = self._normalize_score_token(score_token, fallback=f"task:{name}")
        normalized_max_score_count = self._normalize_max_score_count(max_score_count)
        params_dict = params or {}
        resource_action = self._normalize_resource_action(params_dict.get("resource_action"))
        resource_name = str(params_dict.get("resource", "")).strip()
        resource_item = str(params_dict.get("resource_item", "")).strip()
        queue_weight = -float(points)

        queued = QueuedTask(
            queue_weight=queue_weight,
            sequence=self._task_sequence,
            name=name,
            task_id=resolved_task_id,
            base_points=float(points),
            points_if_completed=bonus_map,
            prerequisites=prereq_list,
            arm=arm_selector,
            blocks_arms=blocks,
            unblocks_arms=unblocks,
            score_token=normalized_score_token,
            max_score_count=normalized_max_score_count,
            resource_action=resource_action,
            resource=resource_name,
            resource_item=resource_item,
            params=params_dict,
        )
        self._task_sequence += 1
        heapq.heappush(self._task_heap, queued)

    def _enqueue_close_after_open(self, opened_task: QueuedTask):
        """Queue a generic close-door task after an open-door task completes."""
        if opened_task.name != "open_microwave_door":
            return

        # Initial open already has explicit graph-level close tasks.
        if opened_task.task_id == "open_door":
            return

        # If there are pending microwave put tasks, keep the door open for loading.
        pending_put_ids = {"put_bowl", "put_plate"}
        if any(task_id not in self._completed_tokens for task_id in pending_put_ids):
            return

        prereqs = [opened_task.task_id]
        if opened_task.task_id == "door_open_for_unload":
            prereqs = ["bowl_to_tray"]
        elif opened_task.task_id == "door_open_for_plate_unload":
            prereqs = ["plate_to_tray"]

        self.queue_task(
            name="close_microwave_door",
            points=1.0,
            params={
                "close_force": 15,
                "resource_action": "release",
                "resource": "microwave_door",
                "resource_item": "open",
            },
            prerequisites=prereqs,
            arm="left",
            blocks_arms=[],
            unblocks_arms=["left"],
            score_token="close_microwave_door",
            max_score_count=3,
            validate_subtask_exists=False,
        )

    def run_autonomy(self, max_tasks: Optional[int] = None) -> List[Tuple[str, bool, float, str]]:
        """Run tasks in descending points order until queue is empty or max_tasks reached."""
        self._completed_tokens = set()
        self._blocked_arms = {}
        self._resource_state = {resource: {} for resource in self._resource_constraints}
        self._score_counts = {}
        self._earned_points_total = 0.0
        results: List[Tuple[str, bool, float, str]] = []
        executed = 0

        while self._task_heap:
            if max_tasks is not None and executed >= max_tasks:
                break

            self._refresh_queue_weights()
            deferred: List[QueuedTask] = []
            queued: Optional[QueuedTask] = None

            while self._task_heap:
                candidate = heapq.heappop(self._task_heap)
                if self._is_task_runnable(candidate):
                    queued = candidate
                    break
                deferred.append(candidate)

            for item in deferred:
                heapq.heappush(self._task_heap, item)

            if queued is None:
                break

            current_points = self._effective_points(queued)
            ok = self.run_subtask(queued.name, params=queued.params)
            if ok:
                self._record_score_if_applicable(queued, current_points)
                self._apply_resource_action(queued)
                self._completed_tokens.add(queued.task_id)
                self._completed_tokens.add(queued.name)
                for arm in queued.unblocks_arms:
                    self._blocked_arms.pop(arm, None)
                for arm in queued.blocks_arms:
                    self._blocked_arms[arm] = queued.task_id
                self._enqueue_close_after_open(queued)

            results.append((queued.name, ok, current_points, queued.task_id))
            executed += 1

        return results

    def _pending_blockers(self) -> List[dict]:
        blockers: List[dict] = []
        for task in sorted(self._task_heap, key=lambda t: t.sequence):
            missing = self._missing_prerequisites(task)
            blocked_arms = [
                arm
                for arm in self._task_target_arms(task)
                if arm in self._blocked_arms and arm not in task.unblocks_arms
            ]
            resource_blocked = not self._resource_action_allowed(task)
            resource_gate_blocked = not self._resource_gate_allowed(task)
            arm_gate_blocked = not self._arm_gate_allowed(task)
            if missing or blocked_arms or resource_blocked or resource_gate_blocked or arm_gate_blocked:
                blockers.append(
                    {
                        "task_id": task.task_id,
                        "name": task.name,
                        "missing_prerequisites": missing,
                        "blocked_arms": blocked_arms,
                        "resource_blocked": resource_blocked,
                        "resource_gate_blocked": resource_gate_blocked,
                        "arm_gate_blocked": arm_gate_blocked,
                        "effective_points": self._effective_points(task),
                    }
                )
        return blockers

    def simulate_autonomy(self, max_tasks: Optional[int] = None) -> dict:
        """Simulate scheduler progression without running any subtasks.

        Returns a report containing execution order, deadlock status, and blockers.
        """
        self._completed_tokens = set()
        self._blocked_arms = {}
        self._resource_state = {resource: {} for resource in self._resource_constraints}
        self._score_counts = {}
        self._earned_points_total = 0.0

        progression: List[dict] = []
        executed = 0

        while self._task_heap:
            if max_tasks is not None and executed >= max_tasks:
                break

            self._refresh_queue_weights()
            deferred: List[QueuedTask] = []
            queued: Optional[QueuedTask] = None

            while self._task_heap:
                candidate = heapq.heappop(self._task_heap)
                if self._is_task_runnable(candidate):
                    queued = candidate
                    break
                deferred.append(candidate)

            for item in deferred:
                heapq.heappush(self._task_heap, item)

            if queued is None:
                break

            current_points = self._effective_points(queued)
            self._record_score_if_applicable(queued, current_points)
            self._apply_resource_action(queued)
            self._completed_tokens.add(queued.task_id)
            self._completed_tokens.add(queued.name)
            for arm in queued.unblocks_arms:
                self._blocked_arms.pop(arm, None)
            for arm in queued.blocks_arms:
                self._blocked_arms[arm] = queued.task_id
            self._enqueue_close_after_open(queued)

            progression.append(
                {
                    "task_id": queued.task_id,
                    "name": queued.name,
                    "effective_points": current_points,
                    "arm": queued.arm,
                    "blocks_arms": list(queued.blocks_arms),
                    "unblocks_arms": list(queued.unblocks_arms),
                    "score_token": self._resolve_score_token(queued),
                    "score_count": self._score_counts.get(self._resolve_score_token(queued), 0),
                    "max_score_count": queued.max_score_count,
                }
            )
            executed += 1

        pending = len(self._task_heap)
        deadlocked = pending > 0
        blockers = self._pending_blockers() if deadlocked else []
        return {
            "executed": progression,
            "executed_count": len(progression),
            "pending_count": pending,
            "deadlocked": deadlocked,
            "blocked_arms": dict(self._blocked_arms),
            "blockers": blockers,
            "score_counts": dict(self._score_counts),
            "earned_points_total": self._earned_points_total,
        }

    def pending_autonomy_tasks(self) -> int:
        return len(self._task_heap)

    def scoring_status(self) -> dict:
        return {
            "score_counts": dict(self._score_counts),
            "earned_points_total": self._earned_points_total,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UR5 Supervisor")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Run merged pose+camera recorder")
    rec.add_argument("--output-base", type=str, default="robot_data")
    rec.add_argument("--frequency", type=float, default=500.0)
    rec.add_argument("--stream-udp-host", type=str, default="")
    rec.add_argument("--stream-udp-port", type=int, default=9999)
    rec.add_argument("--no-robot", action="store_true")
    rec.add_argument("--no-camera", action="store_true")
    rec.add_argument("--camera-index", type=int, default=2)

    sub.add_parser("list-subtasks", help="List collaborative subtasks")

    run_task = sub.add_parser("run-subtask", help="Execute a registered subtask")
    run_task.add_argument("name", type=str)
    run_task.add_argument("--params-json", type=str, default="{}")
    run_task.add_argument(
        "--autonomy-mode",
        action="store_true",
        help="Run selected + queued tasks using priority queue mode (mutually exclusive with total_replay)",
    )
    run_task.add_argument(
        "--autonomy-max-tasks",
        type=int,
        default=0,
        help="Optional cap on number of queued tasks executed in autonomy mode (0 means no cap)",
    )
    run_task.add_argument(
        "--autonomy-queue-json",
        type=str,
        default="[]",
        help=(
            "JSON array of extra queued tasks, e.g. "
            "'[{\"name\":\"example\",\"points\":5.0,\"prerequisites\":[\"open_door\"],\"arm\":\"left\",\"params\":{}}]'"
        ),
    )
    run_task.add_argument(
        "--autonomy-graph-file",
        type=str,
        default="",
        help=(
            "Path to master task-graph JSON file. When provided, "
            "primary_task/autonomy_queue/runtime are loaded from file and "
            "--params-json + --autonomy-queue-json are ignored."
        ),
    )
    run_task.add_argument(
        "--autonomy-simulate",
        action="store_true",
        help=(
            "Simulate autonomy task selection/progression without running subtasks. "
            "Useful for deadlock detection from prerequisites/blocking."
        ),
    )

    return parser


def main(argv: Optional[List[str]] = None):
    args = _build_parser().parse_args(argv)
    supervisor = Supervisor()

    if args.command == "record":
        supervisor.run_recording_session(
            output_base=args.output_base,
            frequency=args.frequency,
            stream_udp_host=args.stream_udp_host,
            stream_udp_port=args.stream_udp_port,
            no_robot=args.no_robot,
            camera_enabled=(not args.no_camera),
            camera_index=args.camera_index,
        )
        return

    if args.command == "list-subtasks":
        for row in supervisor.list_subtasks():
            print(row)
        return

    if args.command == "run-subtask":
        params = json.loads(args.params_json)
        if args.autonomy_mode:
            def _queue_item(item: dict, default_task_id: str):
                if not isinstance(item, dict) or "name" not in item:
                    raise ValueError("Each queued task must be an object with at least a 'name' field")
                supervisor.queue_task(
                    item["name"],
                    points=float(item.get("points", item.get("priority", 0.0))),
                    params=item.get("params", {}),
                    task_id=item.get("task_id", default_task_id),
                    prerequisites=item.get("prerequisites", []),
                    points_if_completed=item.get("points_if_completed", {}),
                    arm=item.get("arm", "any"),
                    blocks_arms=item.get("blocks_arms", []),
                    unblocks_arms=item.get("unblocks_arms", []),
                    score_token=item.get("score_token"),
                    max_score_count=item.get("max_score_count", 0),
                    validate_subtask_exists=(not args.autonomy_simulate),
                )

            max_tasks: Optional[int] = None
            if args.autonomy_graph_file:
                graph_path = Path(args.autonomy_graph_file)
                if not graph_path.exists():
                    raise FileNotFoundError(f"Autonomy graph file not found: {graph_path}")
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
                if not isinstance(graph, dict):
                    raise ValueError("Autonomy graph file root must be a JSON object")

                supervisor.configure_points_map(graph.get("points_map", {}))
                supervisor.configure_resource_constraints(graph.get("resource_constraints", {}))

                primary_task = graph.get("primary_task", {})
                if not isinstance(primary_task, dict) or "name" not in primary_task:
                    raise ValueError("Autonomy graph must include object 'primary_task' with a 'name'")
                if primary_task.get("name") == "total_replay":
                    raise ValueError("--autonomy-mode cannot be used with 'total_replay'")

                _queue_item(primary_task, default_task_id=f"{primary_task['name']}:primary")

                extra_queue = graph.get("autonomy_queue", [])
                if not isinstance(extra_queue, list):
                    raise ValueError("Autonomy graph 'autonomy_queue' must be a JSON array")
                for idx, item in enumerate(extra_queue):
                    _queue_item(item, default_task_id=f"queued:{idx}")

                runtime_cfg = graph.get("runtime", {})
                if not isinstance(runtime_cfg, dict):
                    runtime_cfg = {}
                if args.autonomy_max_tasks > 0:
                    max_tasks = args.autonomy_max_tasks
                else:
                    cfg_max = int(runtime_cfg.get("autonomy_max_tasks", 0))
                    max_tasks = cfg_max if cfg_max > 0 else None
            else:
                if args.name == "total_replay":
                    raise ValueError("--autonomy-mode cannot be used with 'total_replay'")

                # Queue selected task first by default.
                primary_points = float(params.pop("points", params.pop("priority", 0.0)))
                primary_task_id = str(params.pop("task_id", f"{args.name}:primary"))
                primary_prerequisites = params.pop("prerequisites", [])
                primary_points_if_completed = params.pop("points_if_completed", {})
                primary_arm = str(params.pop("arm", "any"))
                primary_blocks_arms = params.pop("blocks_arms", [])
                primary_unblocks_arms = params.pop("unblocks_arms", [])
                primary_score_token = params.pop("score_token", None)
                primary_max_score_count = params.pop("max_score_count", 0)

                supervisor.queue_task(
                    args.name,
                    points=primary_points,
                    params=params,
                    task_id=primary_task_id,
                    prerequisites=primary_prerequisites,
                    points_if_completed=primary_points_if_completed,
                    arm=primary_arm,
                    blocks_arms=primary_blocks_arms,
                    unblocks_arms=primary_unblocks_arms,
                    score_token=primary_score_token,
                    max_score_count=primary_max_score_count,
                )

                extra_queue = json.loads(args.autonomy_queue_json)
                if not isinstance(extra_queue, list):
                    raise ValueError("--autonomy-queue-json must be a JSON array")
                for item in extra_queue:
                    _queue_item(item, default_task_id="")

                max_tasks = args.autonomy_max_tasks if args.autonomy_max_tasks > 0 else None

            if args.autonomy_simulate:
                report = supervisor.simulate_autonomy(max_tasks=max_tasks)
                print("autonomy simulation mode: no subtasks were executed")
                for step_idx, step in enumerate(report["executed"], start=1):
                    print(
                        f"sim step {step_idx}: '{step['name']}' "
                        f"(id={step['task_id']}, points={step['effective_points']}, arm={step['arm']})"
                    )
                if report["deadlocked"]:
                    print(
                        f"simulation deadlock detected: {report['pending_count']} pending task(s), "
                        f"blocked_arms={report['blocked_arms']}"
                    )
                    for blocker in report["blockers"]:
                        print(
                            f"  pending '{blocker['name']}' (id={blocker['task_id']}): "
                            f"missing_prerequisites={blocker['missing_prerequisites']} "
                            f"blocked_arms={blocker['blocked_arms']} "
                            f"resource_blocked={blocker['resource_blocked']} "
                            f"resource_gate_blocked={blocker['resource_gate_blocked']} "
                            f"arm_gate_blocked={blocker['arm_gate_blocked']} "
                            f"effective_points={blocker['effective_points']}"
                        )
                else:
                    print(
                        f"simulation completed without deadlock: executed={report['executed_count']} "
                        f"pending={report['pending_count']}"
                    )
                print(
                    f"simulation scoring summary: total_points={report['earned_points_total']} "
                    f"score_counts={report['score_counts']}"
                )
            else:
                results = supervisor.run_autonomy(max_tasks=max_tasks)
                for task_name, ok, points, task_id in results:
                    print(f"autonomy task '{task_name}' (id={task_id}, points={points}) -> {ok}")
                pending = supervisor.pending_autonomy_tasks()
                if pending:
                    print(
                        f"autonomy halted with {pending} pending task(s): "
                        "likely blocked arms or unmet prerequisites"
                    )
                score_status = supervisor.scoring_status()
                print(
                    f"autonomy scoring summary: total_points={score_status['earned_points_total']} "
                    f"score_counts={score_status['score_counts']}"
                )
        else:
            ok = supervisor.run_subtask(args.name, params=params)
            print(f"subtask '{args.name}' -> {ok}")
        return


if __name__ == "__main__":
    main()
