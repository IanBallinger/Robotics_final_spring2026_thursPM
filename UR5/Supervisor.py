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
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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

try:
    from subtasks.example_subtask import _get_or_start_vision_feeds
except Exception:
    _get_or_start_vision_feeds = None

try:
    from task_graph_visualizer import TaskGraphStateVisualizer
except Exception:
    TaskGraphStateVisualizer = None  # type: ignore


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
    'purple': {'lower': np.array([156,  83,   0]), 'upper': np.array([180, 176, 143])},
    'yellow': {'lower': np.array([ 13, 255, 120]), 'upper': np.array([ 98, 255, 208])},
    'green':  {'lower': np.array([ 53,  90, 128]), 'upper': np.array([ 87, 180, 221])},
    'tan':    {'lower': np.array([ 40,  71, 139]), 'upper': np.array([ 56, 195, 255])},
    'blue':   {'lower': np.array([ 95, 105, 158]), 'upper': np.array([103, 255, 255])},
    'red':    {'lower': np.array([  1, 180, 131]), 'upper': np.array([  3, 255, 236])},
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


OPEN_MICROWAVE_TASK_TEMPLATE = {
    "name": "open_microwave_door",
    "points": 1.0,
    "arm": "left",
    "blocks_arms": ["left"],
    "unblocks_arms": [],
    "score_token": "open_microwave_door",
    "max_score_count": 3,
    "params": {
        "door_handle_pose_source": "vision",
        "safe_approach_offset_m": 0.1,
        "requires_resource_available_for": "table_area",
        "resource_action": "acquire",
        "resource": "table_area",
        "resource_item": "microwave_open",
    },
}

DOOR_OPEN_FOR_LOAD_TASK_ID = "door_open_for_load"
PRESS_STOP_FOR_BOWL_TASK_ID = "press_stop_for_bowl"
PRESS_STOP_FOR_PLATE_TASK_ID = "press_stop_for_plate"


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
        self._grasped_object_by_arm: Dict[str, Optional[str]] = {"left": None, "right": None}
        self._idle_wait_margin: float = 0.5
        self._max_consecutive_idle_per_arm: int = 2
        self._score_counts: Dict[str, int] = {}
        self._earned_points_total: float = 0.0
        self._completed_task_ids: Set[str] = set()
        self._task_catalog: Dict[str, Dict[str, Any]] = {}
        self._visual_step_counter: int = 0
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
        vision_params: Optional[dict] = None,
    ) -> threading.Thread:
        def worker():
            if _get_or_start_vision_feeds is not None:
                feeds = None
                try:
                    feeds = _get_or_start_vision_feeds(params=vision_params or {})
                except Exception as exc:
                    print(f"[Supervisor] Warning: failed to initialize shared vision feeds: {exc}")
                    feeds = None

                if feeds is not None:
                    while not stop_event.is_set():
                        snapshot = feeds.snapshot()
                        detections = []
                        frame_ts = time.time()

                        for cam_data in snapshot.values():
                            cam_spec_raw = cam_data.get("spec_key", -1)
                            cam_spec_key = int(cam_spec_raw) if cam_spec_raw is not None else -1
                            cam_axis_pair = tuple(cam_data.get("axis_pair", ("x", "y")))
                            targets = cam_data.get("targets", {})
                            for label, point in targets.items():
                                point_axis_pair = point.get("axis_pair", None)
                                axis_pair = tuple(point_axis_pair) if point_axis_pair is not None else cam_axis_pair
                                x = float(point.get("x", 0.0))
                                y = float(point.get("y", 0.0))
                                z = float(point.get("z", VISION_Z))
                                if len(axis_pair) >= 2 and axis_pair[1] == "z" and "z" in point:
                                    z = float(point.get("z", 0.0))

                                detections.append(
                                    {
                                        "label": str(label),
                                        "color": str(point.get("target_name", label)),
                                        "position": [round(x, 4), round(y, 4), round(z, 4)],
                                        "axis_pair": list(axis_pair),
                                        "camera_index": int(point.get("camera_index", -1)),
                                        "spec_key": int(point.get("spec_key", cam_spec_key)) if point.get("spec_key", cam_spec_key) is not None else -1,
                                    }
                                )

                        if detections:
                            packet = {
                                "packet_type": "vision_frame",
                                "timestamp": frame_ts,
                                "detections": detections,
                            }
                            try:
                                with send_lock:
                                    udp_socket.sendto(json.dumps(packet).encode("utf-8"), udp_target)
                            except OSError:
                                pass
                        time.sleep(0.03)

                    try:
                        feeds.stop()
                    except Exception:
                        pass
                    return

            # Fallback local vision path (kept for compatibility if shared feed import fails).
            cap = cv2.VideoCapture(camera_index)
            kernel = np.ones((5, 5), np.uint8)
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame_detections = []
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
                        frame_detections.append(
                            {
                                "label": color_name,
                                "color": color_name,
                                "position": [round(x_cm / 100.0, 4), round(y_cm / 100.0, 4), VISION_Z],
                                "axis_pair": ["x", "y"],
                                "camera_index": int(camera_index),
                                "spec_key": -1,
                            }
                        )

                if frame_detections:
                    packet = {
                        "packet_type": "vision_frame",
                        "timestamp": time.time(),
                        "detections": frame_detections,
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
            vision_params = {
                "vision_camera_scan_max_index": max(int(camera_index), 6),
            }
            camera_thread = self._start_camera_thread(
                udp_socket, udp_target, camera_stop, send_lock, camera_index, vision_params
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
        if name == "_idle_wait":
            wait_s = float(params.get("wait_s", 0.05))
            if wait_s > 0.0:
                time.sleep(wait_s)
            return True
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

    @staticmethod
    def _normalize_grasp_label(raw: Optional[str]) -> str:
        if raw is None:
            return ""
        label = str(raw).strip().lower()
        if not label:
            return ""

        mapping = {
            "microwavable_bowl": "bowl",
            "microwavable_plate": "plate",
            "cup_for_drink": "cup",
            "bottle_to_fill_cup": "bottle",
            "cup_containing_stirrer": "stirrer",
            "tray": "tray",
            "microwave": "microwave",
            "microwave_door": "microwave",
            "door": "microwave",
        }
        return mapping.get(label, label)

    def _infer_gripper_requirements(self, task: QueuedTask) -> dict:
        """Infer per-arm grasp compatibility for known task patterns."""
        params = task.params or {}
        name = task.name

        if name == "acquire_bowl":
            return {"requires_empty": True, "set_object": "bowl"}
        if name == "acquire_plate":
            return {"requires_empty": True, "set_object": "plate"}
        if name == "acquire_cup":
            return {"requires_empty": True, "set_object": "cup"}
        if name == "acquire_bottle":
            return {"requires_empty": True, "set_object": "bottle"}
        if name == "place_bowl_in_microwave":
            return {"requires_object": "bowl", "clear_object": True}
        if name == "place_plate_in_microwave":
            return {"requires_object": "plate", "clear_object": True}
        if name == "place_cup_on_tray":
            return {"requires_object": "cup", "clear_object": True}
        if name == "take_bowl_out_to_tray":
            return {"requires_empty": True}
        if name == "take_plate_out_to_tray":
            return {"requires_empty": True}
        if name == "pour_drink_into_cup":
            return {"requires_object": "bottle"}
        if name == "return_bottle":
            return {"requires_object": "bottle", "clear_object": True}

        explicit_required = self._normalize_grasp_label(params.get("requires_grasped_object"))
        explicit_set = self._normalize_grasp_label(params.get("set_grasped_object"))
        explicit_clear = bool(params.get("clear_grasped_object", False))
        requires_empty = bool(params.get("requires_empty_gripper", False))

        return {
            "requires_object": explicit_required,
            "set_object": explicit_set,
            "clear_object": explicit_clear,
            "requires_empty": requires_empty,
        }

    def _gripper_gate_allowed(self, task: QueuedTask) -> bool:
        target_arms = self._task_target_arms(task)
        if len(target_arms) != 1:
            # Only single-arm tasks participate in grasp compatibility.
            return True

        arm = target_arms[0]
        current = self._grasped_object_by_arm.get(arm)
        rules = self._infer_gripper_requirements(task)
        required = self._normalize_grasp_label(rules.get("requires_object"))
        requires_empty = bool(rules.get("requires_empty", False))

        if requires_empty and current is not None:
            return False
        if required and current != required:
            return False
        return True

    def _apply_gripper_state(self, task: QueuedTask):
        target_arms = self._task_target_arms(task)
        if len(target_arms) != 1:
            return

        arm = target_arms[0]
        rules = self._infer_gripper_requirements(task)
        set_object = self._normalize_grasp_label(rules.get("set_object"))
        clear_object = bool(rules.get("clear_object", False))

        if clear_object:
            self._grasped_object_by_arm[arm] = None
        if set_object:
            self._grasped_object_by_arm[arm] = set_object

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
        if not self._gripper_gate_allowed(task):
            return 0.0
        if self._close_defers_for_other_arm_only_branch(task):
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

    @staticmethod
    def _task_tokens(task: QueuedTask) -> Set[str]:
        return {task.task_id, task.name}

    @staticmethod
    def _required_unblocked_arms(task: QueuedTask) -> Set[str]:
        params = task.params or {}
        raw = params.get("requires_arms_unblocked", [])
        if not isinstance(raw, list):
            return set()

        required: Set[str] = set()
        for arm in raw:
            arm_name = str(arm).strip().lower()
            if arm_name in {"left", "right"}:
                required.add(arm_name)
        return required

    def _potential_points(self, task: QueuedTask) -> float:
        """Potential score contribution for priority planning.

        This intentionally ignores dynamic gates and prerequisite completion so we
        can estimate long-horizon value currently blocked behind dependencies.
        """
        dynamic_base_points, max_score_count, from_points_map = self._score_config_for_task(task)
        score_token = self._resolve_score_token(task)
        if max_score_count > 0 and self._score_counts.get(score_token, 0) >= max_score_count:
            if from_points_map:
                return float(task.base_points)
            return 0.0
        return float(dynamic_base_points)

    def _arm_interference_blocks_candidate(self, pending_task: QueuedTask, candidate: QueuedTask) -> bool:
        arm_busy = {"left": False, "right": False}
        dispatch = self._task_reserved_arms(pending_task, arm_busy)
        if dispatch is None:
            return False

        _, reserved_arms = dispatch
        for arm_name in reserved_arms:
            arm_busy[arm_name] = True

        if self._task_reserved_arms(candidate, arm_busy) is None:
            return True

        required_unblocked = self._required_unblocked_arms(candidate)
        return any(arm in required_unblocked for arm in reserved_arms)

    def _dependency_unlock_value(
        self,
        task: QueuedTask,
        memo: Dict[int, float],
        visiting: Set[int],
    ) -> float:
        seq = task.sequence
        if seq in memo:
            return memo[seq]
        if seq in visiting:
            return 0.0

        visiting.add(seq)
        value = self._potential_points(task)
        tokens = self._task_tokens(task)

        for candidate in self._task_heap:
            if candidate.sequence == seq:
                continue

            remaining_prereqs = [
                token for token in candidate.prerequisites if token not in self._completed_tokens
            ]
            if not remaining_prereqs:
                continue

            overlap = len(tokens.intersection(remaining_prereqs))
            if overlap <= 0:
                continue

            child_value = self._dependency_unlock_value(candidate, memo, visiting)
            if child_value <= 0.0:
                continue

            value += child_value * (float(overlap) / float(len(remaining_prereqs)))

        visiting.remove(seq)
        memo[seq] = value
        return value

    def _blocked_points_if_pending(self, task: QueuedTask) -> float:
        """Estimate points blocked while task remains incomplete.

        Includes:
        - Prerequisite lockup (direct and transitive descendants).
        - Arm-lane and requires_arms_unblocked interference while task runs.
        """
        blocked_points = 0.0
        tokens = self._task_tokens(task)
        memo: Dict[int, float] = {}

        for candidate in self._task_heap:
            if candidate.sequence == task.sequence:
                continue

            missing = self._missing_prerequisites(candidate)
            overlap = len(tokens.intersection(missing))
            if overlap > 0 and missing:
                unlocked_value = self._dependency_unlock_value(candidate, memo, set())
                if unlocked_value > 0.0:
                    blocked_points += unlocked_value * (float(overlap) / float(len(missing)))
                continue

            # Candidate is prereq-ready; check if running this task would interfere.
            if missing:
                continue
            if not self._arm_interference_blocks_candidate(task, candidate):
                continue

            interfered_value = self._dependency_unlock_value(candidate, memo, set())
            if interfered_value > 0.0:
                blocked_points += interfered_value

        return blocked_points

    def _priority_score(self, task: QueuedTask) -> float:
        """Priority score combines immediate and unlockable downstream points."""
        score = self._effective_points(task) + self._blocked_points_if_pending(task)
        if (
            self._is_microwave_close_task(task)
            and not self._close_defers_for_other_arm_only_branch(task)
            and self._is_task_runnable_core(task)
        ):
            # Prefer freeing shared resources when it does not cut off the
            # other arm's only runnable branch.
            score += 1.0
        return score

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
        - requires_resource_not_occupied_for: <resource_name>
        - requires_resource_item: <item_name> (optional, defaults to target_label)
        - requires_resource_not_item: <item_name> (optional, defaults to requires_resource_item)
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

            same_item_held = bool(item) and int(held.get(item, 0)) > 0

            total_held = sum(int(v) for v in held.values())
            if total_held >= capacity and not same_item_held:
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

        not_occupied_resource_name = str(params.get("requires_resource_not_occupied_for", "")).strip()
        if not_occupied_resource_name:
            cfg = self._resource_constraints.get(not_occupied_resource_name)
            if cfg is None:
                return False

            held = self._resource_state.get(not_occupied_resource_name, {})
            not_item = str(params.get("requires_resource_not_item", item)).strip()
            if not_item:
                if int(held.get(not_item, 0)) > 0:
                    return False
            else:
                total_held = sum(int(v) for v in held.values())
                if total_held > 0:
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

    def _apply_resource_action_on_start(self, task: QueuedTask):
        """Apply resource usage at execution start.

        Acquire actions reserve capacity immediately when the task is dispatched.
        """
        if task.resource_action != "acquire" or not task.resource:
            return

        if task.resource not in self._resource_state:
            self._resource_state[task.resource] = {}

        held = self._resource_state[task.resource]
        item = task.resource_item
        if not item:
            return

        held[item] = int(held.get(item, 0)) + 1

    def _apply_resource_action_on_end(self, task: QueuedTask):
        """Apply resource usage transition at execution end.

        Release actions free capacity only after the task finishes successfully.
        """
        if task.resource_action != "release" or not task.resource:
            return

        held = self._resource_state.get(task.resource, {})
        item = task.resource_item
        if not item:
            return

        current = int(held.get(item, 0))
        if current <= 1:
            held.pop(item, None)
        else:
            held[item] = current - 1

    def _rollback_resource_start_if_needed(self, task: QueuedTask):
        """Rollback start-phase acquire when the task does not complete successfully."""
        if task.resource_action != "acquire" or not task.resource:
            return

        held = self._resource_state.get(task.resource, {})
        item = task.resource_item
        if not item:
            return

        current = int(held.get(item, 0))
        if current <= 1:
            held.pop(item, None)
        else:
            held[item] = current - 1

    @staticmethod
    def _is_microwave_close_task(task: QueuedTask) -> bool:
        return (
            task.name == "close_microwave_door"
            and task.resource_action == "release"
            and task.resource == "table_area"
            and task.resource_item == "microwave_open"
        )

    @staticmethod
    def _task_requires_microwave_open(task: QueuedTask) -> bool:
        params = task.params or {}
        occupied = str(params.get("requires_resource_occupied_for", "")).strip()
        item = str(params.get("requires_resource_item", params.get("target_label", ""))).strip()
        return occupied == "table_area" and item == "microwave_open"

    def _is_task_runnable_core(self, task: QueuedTask) -> bool:
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
        if not self._gripper_gate_allowed(task):
            return False
        return True

    def _close_defers_for_other_arm_only_branch(self, task: QueuedTask) -> bool:
        """Delay close when it would block the other arm's only runnable branch."""
        if not self._is_microwave_close_task(task):
            return False

        target_arms = self._task_target_arms(task)
        if len(target_arms) != 1:
            return False

        close_arm = target_arms[0]
        other_arm = self._other_arm(close_arm)

        runnable_on_other: List[QueuedTask] = []
        for candidate in self._task_heap:
            if candidate.sequence == task.sequence:
                continue
            if not self._task_can_run_on_arm(candidate, other_arm):
                continue
            if self._is_task_runnable_core(candidate):
                runnable_on_other.append(candidate)

        if len(runnable_on_other) != 1:
            return False

        return self._task_requires_microwave_open(runnable_on_other[0])

    def _is_task_runnable(self, task: QueuedTask) -> bool:
        if not self._is_task_runnable_core(task):
            return False
        if self._close_defers_for_other_arm_only_branch(task):
            return False
        return True

    @staticmethod
    def _task_reserved_arms(task: QueuedTask, arm_busy: Dict[str, bool]) -> Optional[Tuple[str, List[str]]]:
        """Return (dispatch_arm, reserved_arms) if task can be dispatched now, else None.

        dispatch_arm indicates which worker thread executes the task.
        reserved_arms indicates which arm lanes are occupied for the task duration.
        """
        if task.arm == "left":
            if arm_busy.get("left", False):
                return None
            return ("left", ["left"])

        if task.arm == "right":
            if arm_busy.get("right", False):
                return None
            return ("right", ["right"])

        if task.arm == "both":
            if arm_busy.get("left", False) or arm_busy.get("right", False):
                return None
            # Execute bimanual task on one worker while reserving both arm lanes.
            return ("left", ["left", "right"])

        # task.arm == "any"
        if not arm_busy.get("left", False):
            return ("left", ["left"])
        if not arm_busy.get("right", False):
            return ("right", ["right"])
        return None

    @staticmethod
    def _other_arm(arm_name: str) -> str:
        return "right" if arm_name == "left" else "left"

    @staticmethod
    def _task_can_run_on_arm(task: QueuedTask, arm_name: str) -> bool:
        if task.arm == "both":
            return True
        if task.arm == "any":
            return True
        return task.arm == arm_name

    def _best_future_value_for_arm(self, arm_name: str) -> float:
        memo: Dict[int, float] = {}
        best = 0.0
        for candidate in self._task_heap:
            if not self._task_can_run_on_arm(candidate, arm_name):
                continue
            if self._is_task_runnable(candidate):
                continue
            value = self._dependency_unlock_value(candidate, memo, set())
            if value > best:
                best = value
        return best

    def _arm_has_runnable_work_now(self, arm_name: str, arm_busy: Dict[str, bool]) -> bool:
        if arm_busy.get(arm_name, False):
            return False

        for candidate in self._task_heap:
            if not self._is_task_runnable(candidate):
                continue

            dispatch = self._task_reserved_arms(candidate, arm_busy)
            if dispatch is None:
                continue

            dispatch_arm, _ = dispatch
            if dispatch_arm == arm_name:
                return True

        return False

    def _runnable_tokens_for_dispatch_arm(self, arm_name: str, arm_busy: Dict[str, bool]) -> Set[str]:
        tokens: Set[str] = set()
        if arm_busy.get(arm_name, False):
            return tokens

        for candidate in self._task_heap:
            if not self._is_task_runnable(candidate):
                continue

            dispatch = self._task_reserved_arms(candidate, dict(arm_busy))
            if dispatch is None:
                continue

            dispatch_arm, _ = dispatch
            if dispatch_arm != arm_name:
                continue

            tokens.update(self._task_tokens(candidate))

        return tokens

    def _imminent_cross_arm_unlock_conflict(
        self,
        arm_name: str,
        candidate: QueuedTask,
        arm_busy: Dict[str, bool],
    ) -> bool:
        """True when candidate should wait for near-term other-arm unlocks.

        If candidate commits to a grasp object that would conflict with a same-arm
        task likely to become runnable next (because its missing prerequisites can
        all be satisfied by tasks runnable now on the other arm), prefer idling.
        """
        candidate_rules = self._infer_gripper_requirements(candidate)
        candidate_set = self._normalize_grasp_label(candidate_rules.get("set_object"))
        if not candidate_set:
            return False

        other_arm = self._other_arm(arm_name)
        other_arm_tokens = self._runnable_tokens_for_dispatch_arm(other_arm, arm_busy)
        if not other_arm_tokens:
            return False

        for future in self._task_heap:
            if future.sequence == candidate.sequence:
                continue
            if not self._task_can_run_on_arm(future, arm_name):
                continue

            missing = self._missing_prerequisites(future)
            if not missing:
                continue
            if not set(missing).issubset(other_arm_tokens):
                continue

            rules = self._infer_gripper_requirements(future)
            requires_empty = bool(rules.get("requires_empty", False))
            requires_object = self._normalize_grasp_label(rules.get("requires_object"))

            conflict = requires_empty or (bool(requires_object) and requires_object != candidate_set)
            if conflict:
                return True

        return False

    def _build_idle_wait_task(self, arm_name: str, reason: str) -> QueuedTask:
        task = QueuedTask(
            queue_weight=0.0,
            sequence=self._task_sequence,
            name="_idle_wait",
            task_id=f"idle_wait:{arm_name}:{self._task_sequence}",
            base_points=0.0,
            points_if_completed={},
            prerequisites=[],
            arm=arm_name,
            blocks_arms=[],
            unblocks_arms=[],
            score_token="task:_idle_wait",
            max_score_count=0,
            resource_action="",
            resource="",
            resource_item="",
            params={"wait_s": 0.05, "idle_reason": reason},
        )
        self._task_sequence += 1
        return task

    def _select_idle_arm_for_busy_partner(
        self,
        arm_busy: Dict[str, bool],
        consecutive_idle_by_arm: Dict[str, int],
    ) -> Optional[str]:
        """Choose right arm to explicitly idle while left arm is busy.

        This keeps idle insertion targeted to the asymmetric coordination use-case
        where right should wait for left-side context actions.
        """
        if arm_busy.get("right", False):
            return None
        if consecutive_idle_by_arm.get("right", 0) >= self._max_consecutive_idle_per_arm:
            return None
        if arm_busy.get("left", False):
            return "right"
        return None

    def _competing_grasp_branch_priority(self, arm_name: str, candidate: QueuedTask) -> float:
        """Return best priority among competing grasp branches on same arm.

        A competing branch is another currently-runnable task on the same arm
        that would commit to a different held-object path.
        """
        candidate_rules = self._infer_gripper_requirements(candidate)
        candidate_set = self._normalize_grasp_label(candidate_rules.get("set_object"))
        if not candidate_set:
            return 0.0

        best_competitor = 0.0
        for other in self._task_heap:
            if other.sequence == candidate.sequence:
                continue
            if not self._task_can_run_on_arm(other, arm_name):
                continue
            if not self._is_task_runnable(other):
                continue

            other_rules = self._infer_gripper_requirements(other)
            other_set = self._normalize_grasp_label(other_rules.get("set_object"))
            other_requires_empty = bool(other_rules.get("requires_empty", False))
            other_requires = self._normalize_grasp_label(other_rules.get("requires_object"))

            conflict = False
            if other_set and other_set != candidate_set:
                conflict = True
            elif other_requires_empty:
                conflict = True
            elif other_requires and other_requires != candidate_set:
                conflict = True

            if conflict:
                best_competitor = max(best_competitor, self._priority_score(other))

        return best_competitor

    def _should_insert_idle_wait(
        self,
        arm_name: str,
        candidate: QueuedTask,
        arm_busy: Dict[str, bool],
        consecutive_idle_by_arm: Dict[str, int],
    ) -> bool:
        if candidate.name == "_idle_wait":
            return False

        if consecutive_idle_by_arm.get(arm_name, 0) >= self._max_consecutive_idle_per_arm:
            return False

        other_arm = self._other_arm(arm_name)
        other_arm_active = arm_busy.get(other_arm, False) or self._arm_has_runnable_work_now(
            other_arm, arm_busy
        )
        candidate_priority = self._priority_score(candidate)
        if other_arm_active:
            future_value = self._best_future_value_for_arm(arm_name)
            if future_value > (candidate_priority + self._idle_wait_margin):
                return True

        # Branch-commit ambiguity on the same arm can justify a short idle even
        # before the other arm is technically busy.
        competitor_priority = self._competing_grasp_branch_priority(arm_name, candidate)
        if competitor_priority > 0.0 and competitor_priority >= (candidate_priority - 0.25):
            return True

        if self._imminent_cross_arm_unlock_conflict(arm_name, candidate, arm_busy):
            return True

        return False

    def _refresh_queue_weights(self):
        for queued in self._task_heap:
            queued.queue_weight = -self._priority_score(queued)
        heapq.heapify(self._task_heap)

    def _reset_visual_tracking(self):
        self._completed_task_ids = set()
        self._visual_step_counter = 0

    def _catalog_task(self, queued: QueuedTask):
        self._task_catalog[queued.task_id] = {
            "task_id": queued.task_id,
            "name": queued.name,
            "arm": queued.arm,
            "sequence": queued.sequence,
            "prerequisites": list(queued.prerequisites),
            "resource_action": queued.resource_action,
            "resource": queued.resource,
            "resource_item": queued.resource_item,
        }

    def _build_visual_snapshot(
        self,
        event: str,
        arm_busy: Optional[Dict[str, bool]] = None,
        running_by_arm: Optional[Dict[str, Optional[str]]] = None,
        message: str = "",
        tick: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._visual_step_counter += 1
        arm_busy = dict(arm_busy or {"left": False, "right": False})
        running_by_arm = dict(running_by_arm or {"left": None, "right": None})

        queued_by_id: Dict[str, QueuedTask] = {task.task_id: task for task in self._task_heap}
        running_ids = {task_id for task_id in running_by_arm.values() if task_id}

        tasks = []
        for task_id, meta in sorted(self._task_catalog.items(), key=lambda item: item[1]["sequence"]):
            state = "pending"
            priority = 0.0
            effective = 0.0

            if task_id in running_ids:
                state = "running"
            elif task_id in self._completed_task_ids:
                state = "completed"
            elif task_id in queued_by_id:
                queued = queued_by_id[task_id]
                priority = self._priority_score(queued)
                effective = self._effective_points(queued)
                state = "runnable" if self._is_task_runnable(queued) else "blocked"

            tasks.append(
                {
                    "task_id": task_id,
                    "name": meta["name"],
                    "arm": meta["arm"],
                    "prerequisites": list(meta["prerequisites"]),
                    "state": state,
                    "priority_score": priority,
                    "effective_points": effective,
                }
            )

        return {
            "step": self._visual_step_counter,
            "event": event,
            "tick": tick,
            "timestamp": time.time(),
            "message": message,
            "pending_count": len(self._task_heap),
            "blocked_arms": dict(self._blocked_arms),
            "resource_state": {
                resource: {item: int(count) for item, count in held.items()}
                for resource, held in self._resource_state.items()
            },
            "resource_constraints": {
                resource: {
                    "capacity": int(cfg.get("capacity", 0)),
                    "mutex_items": sorted(str(item) for item in cfg.get("mutex_items", set())),
                }
                for resource, cfg in self._resource_constraints.items()
            },
            "arm_busy": arm_busy,
            "running_by_arm": running_by_arm,
            "completed_task_ids": sorted(self._completed_task_ids),
            "earned_points_total": self._earned_points_total,
            "score_counts": dict(self._score_counts),
            "tasks": tasks,
        }

    def _push_visual_snapshot(
        self,
        visualizer,
        event: str,
        arm_busy: Optional[Dict[str, bool]] = None,
        running_by_arm: Optional[Dict[str, Optional[str]]] = None,
        message: str = "",
        tick: Optional[int] = None,
    ):
        if visualizer is None:
            return
        snapshot = self._build_visual_snapshot(
            event=event,
            arm_busy=arm_busy,
            running_by_arm=running_by_arm,
            message=message,
            tick=tick,
        )
        visualizer.push_snapshot(snapshot)

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
        self._catalog_task(queued)
        heapq.heappush(self._task_heap, queued)

    @staticmethod
    def _is_microwave_open_task(task: QueuedTask) -> bool:
        return (
            task.name == "open_microwave_door"
            and task.resource_action == "acquire"
            and task.resource == "table_area"
            and task.resource_item == "microwave_open"
        )

    def _queue_has_microwave_open_task(self) -> bool:
        for task in self._task_heap:
            if self._is_microwave_open_task(task) and self._prerequisites_met(task):
                return True
        return False

    @staticmethod
    def _resource_holds_microwave_open(resource_state: Dict[str, Dict[str, int]]) -> bool:
        held = resource_state.get("table_area", {})
        return int(held.get("microwave_open", 0)) > 0

    def _task_ready_except_microwave_open_gate(self, task: QueuedTask) -> bool:
        """True if task is dispatch-ready except for missing microwave_open occupancy."""
        if task.name == "_idle_wait":
            return False
        if task.arm not in {"right", "both", "any"}:
            return False
        if not self._task_requires_microwave_open(task):
            return False
        missing_prerequisites = [
            token
            for token in self._missing_prerequisites(task)
            if token != DOOR_OPEN_FOR_LOAD_TASK_ID
        ]
        if missing_prerequisites:
            return False
        # Exclude the occupied gate itself; check the rest of core run constraints.
        for arm in self._task_target_arms(task):
            if arm in self._blocked_arms and arm not in task.unblocks_arms:
                return False
        if not self._resource_action_allowed(task):
            return False
        if not self._arm_gate_allowed(task):
            return False
        if not self._gripper_gate_allowed(task):
            return False
        return True

    def _enqueue_open_for_right_arm_if_needed(
        self,
        arm_busy: Dict[str, bool],
        running_by_arm: Optional[Dict[str, Optional[str]]] = None,
    ) -> bool:
        """Queue one canonical open-door task when a right-arm task needs microwave_open.

        Returns True when a task was enqueued.
        """
        if self._resource_holds_microwave_open(self._resource_state):
            return False
        if self._queue_has_microwave_open_task():
            return False

        if running_by_arm:
            for running_task_id in running_by_arm.values():
                if not running_task_id:
                    continue
                meta = self._task_catalog.get(running_task_id, {})
                if (
                    meta.get("name") == "open_microwave_door"
                    and str(meta.get("resource_action", "")) == "acquire"
                    and str(meta.get("resource", "")) == "table_area"
                    and str(meta.get("resource_item", "")) == "microwave_open"
                ):
                    return False

        for candidate in self._task_heap:
            if not self._task_ready_except_microwave_open_gate(candidate):
                continue

            dispatch = self._task_reserved_arms(candidate, dict(arm_busy))
            if dispatch is None:
                continue

            self.queue_task(
                name=OPEN_MICROWAVE_TASK_TEMPLATE["name"],
                points=float(OPEN_MICROWAVE_TASK_TEMPLATE["points"]),
                params=dict(OPEN_MICROWAVE_TASK_TEMPLATE["params"]),
                task_id=DOOR_OPEN_FOR_LOAD_TASK_ID,
                prerequisites=[],
                arm=str(OPEN_MICROWAVE_TASK_TEMPLATE["arm"]),
                blocks_arms=list(OPEN_MICROWAVE_TASK_TEMPLATE["blocks_arms"]),
                unblocks_arms=list(OPEN_MICROWAVE_TASK_TEMPLATE["unblocks_arms"]),
                score_token=str(OPEN_MICROWAVE_TASK_TEMPLATE["score_token"]),
                max_score_count=int(OPEN_MICROWAVE_TASK_TEMPLATE["max_score_count"]),
                validate_subtask_exists=False,
            )
            return True

        return False

    def _queue_has_microwave_close_task(self) -> bool:
        for task in self._task_heap:
            if self._is_microwave_close_task(task):
                return True
        return False

    def _enqueue_close_for_resource_release(self, completed_task: QueuedTask):
        """Queue close task whenever microwave_open is held and no close is queued.

        This is resource-driven and not tied to any specific production task.
        """
        if self._is_microwave_close_task(completed_task):
            return

        held = self._resource_state.get("table_area", {})
        if int(held.get("microwave_open", 0)) <= 0:
            return

        if self._queue_has_microwave_close_task():
            return

        # Keep microwave open while loading tasks are still pending.
        pending_put_ids = {"put_bowl", "put_plate"}
        if any(task_id not in self._completed_tokens for task_id in pending_put_ids):
            return

        self.queue_task(
            name="close_microwave_door",
            points=1.0,
            params={
                "close_force": 15,
                "resource_action": "release",
                "resource": "table_area",
                "resource_item": "microwave_open",
            },
            task_id=f"close_microwave_for_resource#{self._task_sequence}",
            prerequisites=[completed_task.task_id],
            arm="left",
            blocks_arms=[],
            unblocks_arms=["left"],
            score_token="close_microwave_door",
            max_score_count=3,
            validate_subtask_exists=False,
        )

    def _queue_has_task_id(self, task_id: str) -> bool:
        for task in self._task_heap:
            if task.task_id == task_id:
                return True
        return False

    def _enqueue_press_stop_for_food_if_needed(self, completed_task: QueuedTask):
        """Queue canonical press-stop tasks after food is placed in microwave.

        This reuses the same subtask name (`press_microwave_stop`) for each food
        without requiring duplicate static entries in the autonomy graph.
        """
        by_put_task = {
            "put_bowl": {
                "task_id": PRESS_STOP_FOR_BOWL_TASK_ID,
                "target_food": "microwavable_bowl",
                "prerequisites": ["close_door", "put_bowl"],
            },
            "put_plate": {
                "task_id": PRESS_STOP_FOR_PLATE_TASK_ID,
                "target_food": "microwavable_plate",
                "prerequisites": ["put_plate"],
            },
        }

        cfg = by_put_task.get(completed_task.task_id)
        if cfg is None:
            return

        task_id = str(cfg["task_id"])
        if task_id in self._completed_tokens or self._queue_has_task_id(task_id):
            return

        self.queue_task(
            name="press_microwave_stop",
            points=1.0,
            params={
                "button": "stop",
                "target_food": str(cfg["target_food"]),
                "requires_resource_available_for": "table_area",
                "requires_resource_item": "microwave_open",
                "requires_resource_not_occupied_for": "table_area",
                "requires_resource_not_item": "microwave_open",
            },
            task_id=task_id,
            prerequisites=list(cfg["prerequisites"]),
            arm="right",
            blocks_arms=[],
            unblocks_arms=[],
            score_token="press_microwave_stop_with_food_inside",
            max_score_count=2,
            validate_subtask_exists=False,
        )

    def _enqueue_dynamic_followups(self, completed_task: QueuedTask):
        self._enqueue_press_stop_for_food_if_needed(completed_task)
        self._enqueue_close_for_resource_release(completed_task)

    def run_autonomy(
        self,
        max_tasks: Optional[int] = None,
        parallel_arms: bool = True,
        visualizer=None,
    ) -> List[Tuple[str, bool, float, str]]:
        """Run autonomy tasks with optional parallel per-arm execution.

        When parallel_arms=True, a coordinator dispatches tasks to one worker thread
        per arm. The same prerequisite/resource/blocking rules are respected.
        """
        if not parallel_arms:
            return self._run_autonomy_sequential(max_tasks=max_tasks, visualizer=visualizer)

        self._completed_tokens = set()
        self._reset_visual_tracking()
        self._blocked_arms = {}
        self._resource_state = {resource: {} for resource in self._resource_constraints}
        self._grasped_object_by_arm = {"left": None, "right": None}
        self._score_counts = {}
        self._earned_points_total = 0.0

        results: List[Tuple[str, bool, float, str]] = []
        dispatched = 0
        completed = 0
        arm_busy: Dict[str, bool] = {"left": False, "right": False}
        running_by_arm: Dict[str, Optional[str]] = {"left": None, "right": None}
        consecutive_idle_by_arm: Dict[str, int] = {"left": 0, "right": 0}

        self._push_visual_snapshot(
            visualizer,
            event="start",
            arm_busy=arm_busy,
            running_by_arm=running_by_arm,
            message="autonomy run started",
        )

        work_queues = {
            "left": queue.Queue(),
            "right": queue.Queue(),
        }
        completion_queue: queue.Queue = queue.Queue()

        def _worker(arm_name: str):
            while True:
                payload = work_queues[arm_name].get()
                if payload is None:
                    work_queues[arm_name].task_done()
                    break

                task, points, reserved_arms = payload
                ok = False
                err = None
                try:
                    ok = self.run_subtask(task.name, params=task.params)
                except Exception as exc:
                    err = exc

                completion_queue.put((task, points, reserved_arms, ok, err))
                work_queues[arm_name].task_done()

        workers = {
            arm: threading.Thread(target=_worker, args=(arm,), daemon=True)
            for arm in ("left", "right")
        }
        for t in workers.values():
            t.start()

        pending_exception: Optional[Exception] = None
        try:
            while True:
                # Drain completed task notifications first so gates can open promptly.
                while True:
                    try:
                        task, current_points, reserved_arms, ok, err = completion_queue.get_nowait()
                    except queue.Empty:
                        break

                    for arm_name in reserved_arms:
                        arm_busy[arm_name] = False
                        running_by_arm[arm_name] = None

                    if err is not None:
                        self._rollback_resource_start_if_needed(task)
                        pending_exception = err
                        break

                    if ok:
                        self._record_score_if_applicable(task, current_points)
                        self._apply_resource_action_on_end(task)
                        self._apply_gripper_state(task)
                        if task.name != "_idle_wait":
                            self._completed_tokens.add(task.task_id)
                            self._completed_tokens.add(task.name)
                            self._completed_task_ids.add(task.task_id)
                            for arm in task.unblocks_arms:
                                self._blocked_arms.pop(arm, None)
                            for arm in task.blocks_arms:
                                self._blocked_arms[arm] = task.task_id
                            self._enqueue_dynamic_followups(task)
                    else:
                        self._rollback_resource_start_if_needed(task)

                    results.append((task.name, ok, current_points, task.task_id))
                    completed += 1
                    self._push_visual_snapshot(
                        visualizer,
                        event="complete",
                        arm_busy=arm_busy,
                        running_by_arm=running_by_arm,
                        message=f"completed {task.task_id} ok={ok}",
                    )

                if pending_exception is not None:
                    break

                self._enqueue_open_for_right_arm_if_needed(arm_busy, running_by_arm)

                # Stop dispatching when max-tasks cap reached; allow in-flight tasks to finish.
                dispatch_allowed = max_tasks is None or dispatched < max_tasks
                dispatched_any = False

                while dispatch_allowed and self._task_heap:
                    self._refresh_queue_weights()
                    deferred: List[QueuedTask] = []
                    selected: Optional[QueuedTask] = None
                    selected_dispatch: Optional[Tuple[str, List[str]]] = None

                    while self._task_heap:
                        candidate = heapq.heappop(self._task_heap)
                        if self._is_task_runnable(candidate):
                            dispatch_info = self._task_reserved_arms(candidate, arm_busy)
                            if dispatch_info is not None:
                                selected = candidate
                                selected_dispatch = dispatch_info
                                break
                        deferred.append(candidate)

                    for item in deferred:
                        heapq.heappush(self._task_heap, item)

                    if selected is None or selected_dispatch is None:
                        idle_arm = self._select_idle_arm_for_busy_partner(
                            arm_busy, consecutive_idle_by_arm
                        )
                        if idle_arm is not None:
                            selected = self._build_idle_wait_task(
                                idle_arm,
                                reason="no runnable task while partner arm is active",
                            )
                            selected_dispatch = (idle_arm, [idle_arm])
                        else:
                            break

                    if selected is None or selected_dispatch is None:
                        break

                    dispatch_arm, reserved_arms = selected_dispatch
                    if self._should_insert_idle_wait(
                        dispatch_arm,
                        selected,
                        arm_busy,
                        consecutive_idle_by_arm,
                    ):
                        idle_reason = (
                            "waiting for higher-value unlocks from other-arm progress"
                        )
                        heapq.heappush(self._task_heap, selected)
                        selected = self._build_idle_wait_task(dispatch_arm, reason=idle_reason)
                        reserved_arms = [dispatch_arm]

                    for arm_name in reserved_arms:
                        arm_busy[arm_name] = True
                        running_by_arm[arm_name] = selected.task_id

                    self._apply_resource_action_on_start(selected)

                    current_points = self._effective_points(selected)
                    work_queues[dispatch_arm].put((selected, current_points, reserved_arms))
                    dispatched += 1
                    dispatched_any = True
                    self._push_visual_snapshot(
                        visualizer,
                        event="dispatch",
                        arm_busy=arm_busy,
                        running_by_arm=running_by_arm,
                        message=f"dispatched {selected.task_id} on {dispatch_arm}",
                    )

                    if selected.name == "_idle_wait":
                        consecutive_idle_by_arm[dispatch_arm] = (
                            consecutive_idle_by_arm.get(dispatch_arm, 0) + 1
                        )
                    else:
                        consecutive_idle_by_arm[dispatch_arm] = 0

                    dispatch_allowed = max_tasks is None or dispatched < max_tasks

                # Exit condition: no queued tasks and no running tasks.
                running = any(arm_busy.values())
                if not self._task_heap and not running:
                    break

                # Deadlock condition: queue still has tasks, nothing running, nothing dispatched.
                if self._task_heap and not running and not dispatched_any:
                    break

                # Wait for at least one worker completion to avoid busy looping.
                if any(arm_busy.values()):
                    task, current_points, reserved_arms, ok, err = completion_queue.get()
                    for arm_name in reserved_arms:
                        arm_busy[arm_name] = False
                        running_by_arm[arm_name] = None

                    if err is not None:
                        self._rollback_resource_start_if_needed(task)
                        pending_exception = err
                        break

                    if ok:
                        self._record_score_if_applicable(task, current_points)
                        self._apply_resource_action_on_end(task)
                        self._apply_gripper_state(task)
                        if task.name != "_idle_wait":
                            self._completed_tokens.add(task.task_id)
                            self._completed_tokens.add(task.name)
                            self._completed_task_ids.add(task.task_id)
                            for arm in task.unblocks_arms:
                                self._blocked_arms.pop(arm, None)
                            for arm in task.blocks_arms:
                                self._blocked_arms[arm] = task.task_id
                            self._enqueue_dynamic_followups(task)
                    else:
                        self._rollback_resource_start_if_needed(task)

                    results.append((task.name, ok, current_points, task.task_id))
                    completed += 1
                    self._push_visual_snapshot(
                        visualizer,
                        event="complete",
                        arm_busy=arm_busy,
                        running_by_arm=running_by_arm,
                        message=f"completed {task.task_id} ok={ok}",
                    )

            if pending_exception is not None:
                raise pending_exception

            self._push_visual_snapshot(
                visualizer,
                event="final",
                arm_busy=arm_busy,
                running_by_arm=running_by_arm,
                message="autonomy run finished",
            )

            return results
        finally:
            for arm in ("left", "right"):
                work_queues[arm].put(None)
            for t in workers.values():
                t.join(timeout=1.0)

    def _run_autonomy_sequential(self, max_tasks: Optional[int] = None, visualizer=None) -> List[Tuple[str, bool, float, str]]:
        """Original sequential autonomy loop."""
        self._completed_tokens = set()
        self._reset_visual_tracking()
        self._blocked_arms = {}
        self._resource_state = {resource: {} for resource in self._resource_constraints}
        self._grasped_object_by_arm = {"left": None, "right": None}
        self._score_counts = {}
        self._earned_points_total = 0.0
        results: List[Tuple[str, bool, float, str]] = []
        executed = 0
        arm_busy: Dict[str, bool] = {"left": False, "right": False}
        running_by_arm: Dict[str, Optional[str]] = {"left": None, "right": None}

        self._push_visual_snapshot(
            visualizer,
            event="start",
            arm_busy=arm_busy,
            running_by_arm=running_by_arm,
            message="sequential autonomy run started",
        )

        while self._task_heap:
            if max_tasks is not None and executed >= max_tasks:
                break

            self._enqueue_open_for_right_arm_if_needed(arm_busy, running_by_arm)

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
            running_by_arm[queued.arm if queued.arm in {"left", "right"} else "left"] = queued.task_id
            self._apply_resource_action_on_start(queued)
            self._push_visual_snapshot(
                visualizer,
                event="dispatch",
                arm_busy=arm_busy,
                running_by_arm=running_by_arm,
                message=f"dispatched {queued.task_id}",
            )
            ok = self.run_subtask(queued.name, params=queued.params)
            if ok:
                self._record_score_if_applicable(queued, current_points)
                self._apply_resource_action_on_end(queued)
                self._apply_gripper_state(queued)
                self._completed_tokens.add(queued.task_id)
                self._completed_tokens.add(queued.name)
                self._completed_task_ids.add(queued.task_id)
                for arm in queued.unblocks_arms:
                    self._blocked_arms.pop(arm, None)
                for arm in queued.blocks_arms:
                    self._blocked_arms[arm] = queued.task_id
                self._enqueue_dynamic_followups(queued)
            else:
                self._rollback_resource_start_if_needed(queued)

            results.append((queued.name, ok, current_points, queued.task_id))
            executed += 1
            running_by_arm = {"left": None, "right": None}
            self._push_visual_snapshot(
                visualizer,
                event="complete",
                arm_busy=arm_busy,
                running_by_arm=running_by_arm,
                message=f"completed {queued.task_id} ok={ok}",
            )

        self._push_visual_snapshot(
            visualizer,
            event="final",
            arm_busy=arm_busy,
            running_by_arm=running_by_arm,
            message="sequential autonomy run finished",
        )

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
            gripper_gate_blocked = not self._gripper_gate_allowed(task)
            blocked_points_if_pending = self._blocked_points_if_pending(task)
            priority_score = self._priority_score(task)
            if (
                missing
                or blocked_arms
                or resource_blocked
                or resource_gate_blocked
                or arm_gate_blocked
                or gripper_gate_blocked
            ):
                blockers.append(
                    {
                        "task_id": task.task_id,
                        "name": task.name,
                        "missing_prerequisites": missing,
                        "blocked_arms": blocked_arms,
                        "resource_blocked": resource_blocked,
                        "resource_gate_blocked": resource_gate_blocked,
                        "arm_gate_blocked": arm_gate_blocked,
                        "gripper_gate_blocked": gripper_gate_blocked,
                        "grasped_by_arm": dict(self._grasped_object_by_arm),
                        "blocked_points_if_pending": blocked_points_if_pending,
                        "priority_score": priority_score,
                        "effective_points": self._effective_points(task),
                    }
                )
        return blockers

    def simulate_autonomy(self, max_tasks: Optional[int] = None, parallel_arms: bool = True, visualizer=None) -> dict:
        """Simulate scheduler progression without running subtasks.

        When parallel_arms=True, simulation mirrors the coordinated two-lane
        (left/right) dispatcher used by run_autonomy.
        """
        if not parallel_arms:
            return self._simulate_autonomy_sequential(max_tasks=max_tasks, visualizer=visualizer)

        self._completed_tokens = set()
        self._reset_visual_tracking()
        self._blocked_arms = {}
        self._resource_state = {resource: {} for resource in self._resource_constraints}
        self._grasped_object_by_arm = {"left": None, "right": None}
        self._score_counts = {}
        self._earned_points_total = 0.0

        progression: List[dict] = []
        dispatched = 0
        arm_busy: Dict[str, bool] = {"left": False, "right": False}
        running_by_arm: Dict[str, Optional[str]] = {"left": None, "right": None}
        consecutive_idle_by_arm: Dict[str, int] = {"left": 0, "right": 0}
        tick = 0

        self._push_visual_snapshot(
            visualizer,
            event="start",
            arm_busy=arm_busy,
            running_by_arm=running_by_arm,
            message="simulation started",
            tick=tick,
        )

        while self._task_heap or any(arm_busy.values()):
            if max_tasks is not None and dispatched >= max_tasks:
                break

            self._enqueue_open_for_right_arm_if_needed(arm_busy, running_by_arm)

            launched: List[Tuple[QueuedTask, float, str, List[str]]] = []
            dispatch_allowed = max_tasks is None or dispatched < max_tasks

            while dispatch_allowed and self._task_heap:
                self._refresh_queue_weights()
                deferred: List[QueuedTask] = []
                selected: Optional[QueuedTask] = None
                selected_dispatch: Optional[Tuple[str, List[str]]] = None

                while self._task_heap:
                    candidate = heapq.heappop(self._task_heap)
                    if self._is_task_runnable(candidate):
                        dispatch_info = self._task_reserved_arms(candidate, arm_busy)
                        if dispatch_info is not None:
                            selected = candidate
                            selected_dispatch = dispatch_info
                            break
                    deferred.append(candidate)

                for item in deferred:
                    heapq.heappush(self._task_heap, item)

                if selected is None or selected_dispatch is None:
                    idle_arm = self._select_idle_arm_for_busy_partner(
                        arm_busy, consecutive_idle_by_arm
                    )
                    if idle_arm is not None:
                        selected = self._build_idle_wait_task(
                            idle_arm,
                            reason="no runnable task while partner arm is active",
                        )
                        selected_dispatch = (idle_arm, [idle_arm])
                    else:
                        break

                if selected is None or selected_dispatch is None:
                    break

                dispatch_arm, reserved_arms = selected_dispatch
                if self._should_insert_idle_wait(
                    dispatch_arm,
                    selected,
                    arm_busy,
                    consecutive_idle_by_arm,
                ):
                    idle_reason = "waiting for higher-value unlocks from other-arm progress"
                    heapq.heappush(self._task_heap, selected)
                    selected = self._build_idle_wait_task(dispatch_arm, reason=idle_reason)
                    reserved_arms = [dispatch_arm]

                for arm_name in reserved_arms:
                    arm_busy[arm_name] = True
                    running_by_arm[arm_name] = selected.task_id

                self._apply_resource_action_on_start(selected)

                current_points = self._effective_points(selected)
                launched.append((selected, current_points, dispatch_arm, reserved_arms))
                dispatched += 1
                self._push_visual_snapshot(
                    visualizer,
                    event="dispatch",
                    arm_busy=arm_busy,
                    running_by_arm=running_by_arm,
                    message=f"sim dispatch {selected.task_id} on {dispatch_arm}",
                    tick=tick,
                )
                if selected.name == "_idle_wait":
                    consecutive_idle_by_arm[dispatch_arm] = (
                        consecutive_idle_by_arm.get(dispatch_arm, 0) + 1
                    )
                else:
                    consecutive_idle_by_arm[dispatch_arm] = 0
                dispatch_allowed = max_tasks is None or dispatched < max_tasks

            if not launched and self._task_heap and not any(arm_busy.values()):
                # Deadlock: tasks remain but none can be dispatched.
                break

            if launched:
                tick += 1

            # Complete all launched tasks for this simulation tick.
            for queued, current_points, dispatch_arm, reserved_arms in launched:
                self._record_score_if_applicable(queued, current_points)
                self._apply_resource_action_on_end(queued)
                self._apply_gripper_state(queued)
                if queued.name != "_idle_wait":
                    self._completed_tokens.add(queued.task_id)
                    self._completed_tokens.add(queued.name)
                    self._completed_task_ids.add(queued.task_id)
                    for arm in queued.unblocks_arms:
                        self._blocked_arms.pop(arm, None)
                    for arm in queued.blocks_arms:
                        self._blocked_arms[arm] = queued.task_id
                    self._enqueue_dynamic_followups(queued)

                for arm_name in reserved_arms:
                    arm_busy[arm_name] = False
                    running_by_arm[arm_name] = None

                progression.append(
                    {
                        "task_id": queued.task_id,
                        "name": queued.name,
                        "effective_points": current_points,
                        "blocked_points_if_pending": self._blocked_points_if_pending(queued),
                        "priority_score": self._priority_score(queued),
                        "arm": queued.arm,
                        "dispatch_arm": dispatch_arm,
                        "reserved_arms": list(reserved_arms),
                        "sim_tick": tick,
                        "blocks_arms": list(queued.blocks_arms),
                        "unblocks_arms": list(queued.unblocks_arms),
                        "score_token": self._resolve_score_token(queued),
                        "score_count": self._score_counts.get(self._resolve_score_token(queued), 0),
                        "max_score_count": queued.max_score_count,
                    }
                )
                self._push_visual_snapshot(
                    visualizer,
                    event="complete",
                    arm_busy=arm_busy,
                    running_by_arm=running_by_arm,
                    message=f"sim complete {queued.task_id}",
                    tick=tick,
                )

        pending = len(self._task_heap)
        deadlocked = pending > 0
        blockers = self._pending_blockers() if deadlocked else []
        self._push_visual_snapshot(
            visualizer,
            event="final",
            arm_busy=arm_busy,
            running_by_arm=running_by_arm,
            message="simulation finished",
            tick=tick,
        )
        return {
            "executed": progression,
            "executed_count": len(progression),
            "pending_count": pending,
            "deadlocked": deadlocked,
            "blocked_arms": dict(self._blocked_arms),
            "grasped_by_arm": dict(self._grasped_object_by_arm),
            "blockers": blockers,
            "score_counts": dict(self._score_counts),
            "earned_points_total": self._earned_points_total,
        }

    def _simulate_autonomy_sequential(self, max_tasks: Optional[int] = None, visualizer=None) -> dict:
        """Original sequential simulator semantics."""
        self._completed_tokens = set()
        self._reset_visual_tracking()
        self._blocked_arms = {}
        self._resource_state = {resource: {} for resource in self._resource_constraints}
        self._grasped_object_by_arm = {"left": None, "right": None}
        self._score_counts = {}
        self._earned_points_total = 0.0

        progression: List[dict] = []
        executed = 0
        arm_busy: Dict[str, bool] = {"left": False, "right": False}
        running_by_arm: Dict[str, Optional[str]] = {"left": None, "right": None}

        self._push_visual_snapshot(
            visualizer,
            event="start",
            arm_busy=arm_busy,
            running_by_arm=running_by_arm,
            message="sequential simulation started",
        )

        while self._task_heap:
            if max_tasks is not None and executed >= max_tasks:
                break

            self._enqueue_open_for_right_arm_if_needed(arm_busy, running_by_arm)

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
            running_by_arm[queued.arm if queued.arm in {"left", "right"} else "left"] = queued.task_id
            self._apply_resource_action_on_start(queued)
            self._push_visual_snapshot(
                visualizer,
                event="dispatch",
                arm_busy=arm_busy,
                running_by_arm=running_by_arm,
                message=f"sim dispatch {queued.task_id}",
            )
            self._record_score_if_applicable(queued, current_points)
            self._apply_resource_action_on_end(queued)
            self._apply_gripper_state(queued)
            self._completed_tokens.add(queued.task_id)
            self._completed_tokens.add(queued.name)
            self._completed_task_ids.add(queued.task_id)
            for arm in queued.unblocks_arms:
                self._blocked_arms.pop(arm, None)
            for arm in queued.blocks_arms:
                self._blocked_arms[arm] = queued.task_id
            self._enqueue_dynamic_followups(queued)

            progression.append(
                {
                    "task_id": queued.task_id,
                    "name": queued.name,
                    "effective_points": current_points,
                    "blocked_points_if_pending": self._blocked_points_if_pending(queued),
                    "priority_score": self._priority_score(queued),
                    "arm": queued.arm,
                    "blocks_arms": list(queued.blocks_arms),
                    "unblocks_arms": list(queued.unblocks_arms),
                    "score_token": self._resolve_score_token(queued),
                    "score_count": self._score_counts.get(self._resolve_score_token(queued), 0),
                    "max_score_count": queued.max_score_count,
                }
            )
            executed += 1
            running_by_arm = {"left": None, "right": None}
            self._push_visual_snapshot(
                visualizer,
                event="complete",
                arm_busy=arm_busy,
                running_by_arm=running_by_arm,
                message=f"sim complete {queued.task_id}",
            )

        pending = len(self._task_heap)
        deadlocked = pending > 0
        blockers = self._pending_blockers() if deadlocked else []
        self._push_visual_snapshot(
            visualizer,
            event="final",
            arm_busy=arm_busy,
            running_by_arm=running_by_arm,
            message="sequential simulation finished",
        )
        return {
            "executed": progression,
            "executed_count": len(progression),
            "pending_count": pending,
            "deadlocked": deadlocked,
            "blocked_arms": dict(self._blocked_arms),
            "grasped_by_arm": dict(self._grasped_object_by_arm),
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
            "Path to autonomy task-graph JSON file. When provided, "
            "autonomy_queue (and optional primary_task/runtime) are loaded from file and "
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
    run_task.add_argument(
        "--autonomy-visualizer",
        type=str,
        default="off",
        choices=["off", "auto", "simulate", "live"],
        help=(
            "Task-graph visualizer mode. 'simulate' records scheduler simulation timeline, "
            "'live' tracks real scheduler state with optional live-follow playback toggle. "
            "'auto' selects simulate/live based on --autonomy-simulate."
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
            visualizer = None
            visualizer_mode = "off"
            if args.autonomy_visualizer != "off":
                visualizer_mode = args.autonomy_visualizer
                if visualizer_mode == "auto":
                    visualizer_mode = "simulate" if args.autonomy_simulate else "live"

                if TaskGraphStateVisualizer is None:
                    print("warning: task visualizer unavailable; continuing without visualization")
                    visualizer_mode = "off"
                else:
                    try:
                        visualizer = TaskGraphStateVisualizer(
                            mode=visualizer_mode,
                            title=(
                                "UR5 Task Graph Visualizer (simulation)"
                                if visualizer_mode == "simulate"
                                else "UR5 Task Graph Visualizer (live scheduler)"
                            ),
                        )
                        visualizer.start()
                    except Exception as exc:
                        print(f"warning: failed to start task visualizer: {exc}")
                        visualizer = None
                        visualizer_mode = "off"

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

                queued_from_graph = 0

                primary_task = graph.get("primary_task", None)
                if primary_task is not None:
                    if not isinstance(primary_task, dict) or "name" not in primary_task:
                        raise ValueError("If provided, 'primary_task' must be an object with a 'name'")
                    if primary_task.get("name") == "total_replay":
                        raise ValueError("--autonomy-mode cannot be used with 'total_replay'")
                    _queue_item(primary_task, default_task_id=f"{primary_task['name']}:primary")
                    queued_from_graph += 1

                extra_queue = graph.get("autonomy_queue", [])
                if not isinstance(extra_queue, list):
                    raise ValueError("Autonomy graph 'autonomy_queue' must be a JSON array")
                for idx, item in enumerate(extra_queue):
                    _queue_item(item, default_task_id=f"queued:{idx}")
                    queued_from_graph += 1

                if queued_from_graph <= 0:
                    raise ValueError(
                        "Autonomy graph must define at least one task via 'autonomy_queue' "
                        "or optional 'primary_task'"
                    )

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
                report = supervisor.simulate_autonomy(max_tasks=max_tasks, visualizer=visualizer)
                print("autonomy simulation mode: no subtasks were executed")
                for step_idx, step in enumerate(report["executed"], start=1):
                    tick_info = f", tick={step['sim_tick']}" if "sim_tick" in step else ""
                    dispatch_info = f", dispatch={step['dispatch_arm']}" if "dispatch_arm" in step else ""
                    reserved_info = (
                        f", reserved={step['reserved_arms']}" if "reserved_arms" in step else ""
                    )
                    blocked_info = (
                        f", blocked_points={step.get('blocked_points_if_pending', 0.0)}"
                    )
                    priority_info = f", priority={step.get('priority_score', step['effective_points'])}"
                    print(
                        f"sim step {step_idx}: '{step['name']}' "
                        f"(id={step['task_id']}, points={step['effective_points']}, arm={step['arm']}"
                        f"{blocked_info}{priority_info}{tick_info}{dispatch_info}{reserved_info})"
                    )
                if report["deadlocked"]:
                    print(
                        f"simulation deadlock detected: {report['pending_count']} pending task(s), "
                        f"blocked_arms={report['blocked_arms']} "
                        f"grasped_by_arm={report.get('grasped_by_arm', {})}"
                    )
                    for blocker in report["blockers"]:
                        print(
                            f"  pending '{blocker['name']}' (id={blocker['task_id']}): "
                            f"missing_prerequisites={blocker['missing_prerequisites']} "
                            f"blocked_arms={blocker['blocked_arms']} "
                            f"resource_blocked={blocker['resource_blocked']} "
                            f"resource_gate_blocked={blocker['resource_gate_blocked']} "
                            f"arm_gate_blocked={blocker['arm_gate_blocked']} "
                            f"gripper_gate_blocked={blocker.get('gripper_gate_blocked', False)} "
                            f"grasped_by_arm={blocker.get('grasped_by_arm', {})} "
                            f"blocked_points={blocker.get('blocked_points_if_pending', 0.0)} "
                            f"priority={blocker.get('priority_score', blocker['effective_points'])} "
                            f"effective_points={blocker['effective_points']}"
                        )
                else:
                    print(
                        f"simulation completed without deadlock: executed={report['executed_count']} "
                        f"pending={report['pending_count']} "
                        f"grasped_by_arm={report.get('grasped_by_arm', {})}"
                    )
                print(
                    f"simulation scoring summary: total_points={report['earned_points_total']} "
                    f"score_counts={report['score_counts']}"
                )
            else:
                results = supervisor.run_autonomy(max_tasks=max_tasks, visualizer=visualizer)
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

            if visualizer is not None:
                if visualizer_mode == "live":
                    print("live scheduler visualizer remains open for playback; close it when finished")
                else:
                    print("simulation visualizer ready; close the window when finished reviewing playback")
                visualizer.wait_until_closed()
        else:
            ok = supervisor.run_subtask(args.name, params=params)
            print(f"subtask '{args.name}' -> {ok}")
        return


if __name__ == "__main__":
    main()
