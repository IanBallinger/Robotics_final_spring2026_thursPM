import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Ensure local UR5 modules resolve when launching from workspace root.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from arm import UR5Arm

_DEMO_DIR = _THIS_DIR / "demo_4_16"


def _load_robotiq_gripper_class():
    gripper_file = _DEMO_DIR / "robotiq_gripper_control.py"
    if not gripper_file.exists():
        return None
    # Ensure sibling imports like `robotiq_preamble` resolve when this file
    # is loaded via importlib from a different working directory.
    if str(_DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(_DEMO_DIR))
    spec = importlib.util.spec_from_file_location("robotiq_gripper_control_local", gripper_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[tuner] Warning: failed to load gripper control module: {exc}")
        return None
    return getattr(module, "RobotiqGripper", None)

try:
    from subtasks.example_subtask import _get_or_start_vision_feeds
except Exception:
    _get_or_start_vision_feeds = None

MOVEMENT_PRIMITIVES = [
    "move_linear_to_pose",
    "move_to_pose",
    "move_to_joint_position",
    "move_path",
]

EDIT_FIELDS = ["x", "y", "z", "rx", "ry", "rz"]
JOINT_FIELDS = [f"q_{i}" for i in range(6)]
FIELD_LIMITS = {
    "x": (-1.5, 1.5),
    "y": (-1.5, 1.5),
    "z": (-0.2, 1.5),
    "rx": (-math.pi, math.pi),
    "ry": (-math.pi, math.pi),
    "rz": (-math.pi, math.pi),
}

JOINT_LIMITS = UR5Arm.get_default_joint_limits_rad()
FIELD_RESOLUTION = {
    "x": 0.001,
    "y": 0.001,
    "z": 0.001,
    "rx": 0.001,
    "ry": 0.001,
    "rz": 0.001,
}

POSITION_SLIDER_HALF_RANGE_M = 0.05
ROTATION_SLIDER_HALF_RANGE_RAD = 0.1
GRIPPER_OPEN_MM_MAX = 85.0
DEPENDENT_ITEM_NA = "N/A"


def _mat_vec_mul_row(v, m):
    # Row-vector multiply: [x y z] * M
    return [
        v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
        v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
        v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
    ]


def _base_to_global_task_xyz(base_xyz, arm_side):
    # Mirrors live_plot_runner.jl base_to_global_task_xyz constants/orientation.
    dy_t = 0.225 / 2.0 + 0.540 / 2.0
    dz_t = -0.753

    if arm_side == "left":
        dx_t = 0.090 / 2.0 + 0.010 + 0.110
        r_task_to_base = [
            [0.707, 0.0, -0.707],
            [0.0, -1.0, 0.0],
            [-0.707, 0.0, -0.707],
        ]
    elif arm_side == "right":
        dx_t = -(0.090 / 2.0 + 0.010 + 0.110)
        r_task_to_base = [
            [0.707, 0.0, 0.707],
            [0.0, -1.0, 0.0],
            [0.707, 0.0, -0.707],
        ]
    else:
        return [float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])]

    trans_base_to_task = _mat_vec_mul_row([dx_t, dy_t, dz_t], r_task_to_base)
    p_rel = [
        float(base_xyz[0]) - trans_base_to_task[0],
        float(base_xyz[1]) - trans_base_to_task[1],
        float(base_xyz[2]) - trans_base_to_task[2],
    ]

    # p_task = (p_base - t) * R' for row-vector convention.
    r_t = [
        [r_task_to_base[0][0], r_task_to_base[1][0], r_task_to_base[2][0]],
        [r_task_to_base[0][1], r_task_to_base[1][1], r_task_to_base[2][1]],
        [r_task_to_base[0][2], r_task_to_base[1][2], r_task_to_base[2][2]],
    ]
    return _mat_vec_mul_row(p_rel, r_t)


def _row_global_xyz(row, arm_side):
    p = _arm_prefix(arm_side)
    gx = row.get(f"{p}_global_x")
    gy = row.get(f"{p}_global_y")
    gz = row.get(f"{p}_global_z")
    if gx is not None and gy is not None and gz is not None:
        g = [_try_float(gx, float("nan")), _try_float(gy, float("nan")), _try_float(gz, float("nan"))]
        if all(math.isfinite(v) for v in g):
            return g

    tx = row.get(f"{p}_task_x")
    ty = row.get(f"{p}_task_y")
    tz = row.get(f"{p}_task_z")
    if tx is not None and ty is not None and tz is not None:
        t = [_try_float(tx, float("nan")), _try_float(ty, float("nan")), _try_float(tz, float("nan"))]
        if all(math.isfinite(v) for v in t):
            return t

    base_xyz = [
        _try_float(row.get(f"{p}_x")),
        _try_float(row.get(f"{p}_y")),
        _try_float(row.get(f"{p}_z")),
    ]
    return _base_to_global_task_xyz(base_xyz, arm_side)


def _update_row_task_global_fields(row, arm_side):
    p = _arm_prefix(arm_side)
    base_xyz = [
        _try_float(row.get(f"{p}_x")),
        _try_float(row.get(f"{p}_y")),
        _try_float(row.get(f"{p}_z")),
    ]
    gx, gy, gz = _base_to_global_task_xyz(base_xyz, arm_side)
    row[f"{p}_task_x"] = f"{gx:.9f}"
    row[f"{p}_task_y"] = f"{gy:.9f}"
    row[f"{p}_task_z"] = f"{gz:.9f}"
    row[f"{p}_global_x"] = f"{gx:.9f}"
    row[f"{p}_global_y"] = f"{gy:.9f}"
    row[f"{p}_global_z"] = f"{gz:.9f}"


def _try_float(v, default=0.0):
    try:
        if v is None:
            return float(default)
        txt = str(v).strip()
        if txt == "" or txt.lower() == "nothing":
            return float(default)
        return float(txt)
    except Exception:
        return float(default)


def _safe_json_list(raw):
    txt = str(raw or "").strip()
    if not txt:
        return []
    try:
        parsed = json.loads(txt)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _distance3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _arm_prefix(arm_side):
    return "left" if arm_side == "left" else "right"


def _pose_from_row(row, arm_side):
    p = _arm_prefix(arm_side)
    return [
        _try_float(row.get(f"{p}_x")),
        _try_float(row.get(f"{p}_y")),
        _try_float(row.get(f"{p}_z")),
        _try_float(row.get(f"{p}_rx")),
        _try_float(row.get(f"{p}_ry")),
        _try_float(row.get(f"{p}_rz")),
    ]


def _row_has_arm_pose_data(row, arm_side):
    p = _arm_prefix(arm_side)
    keys = [f"{p}_x", f"{p}_y", f"{p}_z"]
    for k in keys:
        raw = row.get(k)
        txt = str(raw if raw is not None else "").strip().lower()
        if txt not in {"", "nothing"}:
            return True
    return False


def _q_from_row(row, arm_side):
    p = _arm_prefix(arm_side)
    keys = [f"{p}_q_{i}" for i in range(6)]
    vals = [_try_float(row.get(k), default=float("nan")) for k in keys]
    if any(not math.isfinite(v) for v in vals):
        return None
    return vals


def _set_q_in_row(row, arm_side, q_vals):
    if q_vals is None or len(q_vals) != 6:
        return
    p = _arm_prefix(arm_side)
    for i, q in enumerate(q_vals):
        row[f"{p}_q_{i}"] = f"{float(q):.9f}"


def _tracked_item_position(row, label):
    if not label:
        return None
    for item in _safe_json_list(row.get("tracked_items_json", "")):
        if not isinstance(item, dict):
            continue
        if str(item.get("label", "")).strip().lower() != str(label).strip().lower():
            continue
        pos = item.get("position")
        if isinstance(pos, list) and len(pos) >= 3:
            return [_try_float(pos[0]), _try_float(pos[1]), _try_float(pos[2])]
    return None


def _tracked_item_family_key(item):
    if not isinstance(item, dict):
        return "spec:-1"
    spec_raw = item.get("spec_key", None)
    if spec_raw is not None:
        try:
            return f"spec:{int(spec_raw)}"
        except Exception:
            pass
    cam_raw = item.get("camera_index", None)
    if cam_raw is not None:
        try:
            return f"cam:{int(cam_raw)}"
        except Exception:
            pass
    return "spec:-1"


def _tracked_item_with_label(row, label):
    if not label:
        return None
    for item in _safe_json_list(row.get("tracked_items_json", "")):
        if not isinstance(item, dict):
            continue
        if str(item.get("label", "")).strip().lower() != str(label).strip().lower():
            continue
        return item
    return None


def _tracked_item_position_with_offsets(row, label, family_offsets):
    item = _tracked_item_with_label(row, label)
    if not isinstance(item, dict):
        return None
    pos = item.get("position")
    if not (isinstance(pos, list) and len(pos) >= 3):
        return None
    base = [_try_float(pos[0]), _try_float(pos[1]), _try_float(pos[2])]
    fam = _tracked_item_family_key(item)
    dx, dy, dz = family_offsets.get(fam, (0.0, 0.0, 0.0))
    return [base[0] + float(dx), base[1] + float(dy), base[2] + float(dz)]


def _sensitivity_domain_key(row):
    label = str(row.get("dependent_item_label", "")).strip()
    if not label or label == DEPENDENT_ITEM_NA:
        return DEPENDENT_ITEM_NA
    return label.lower()


def _tracked_item_rgba(item, is_active):
    alpha = 1.0 if is_active else 0.4
    raw_color = str(item.get("color", "")).strip().lower()
    raw_label = str(item.get("label", "")).strip().lower()
    probe = f"{raw_color} {raw_label}"

    named = {
        "red": (0.85, 0.12, 0.12),
        "yellow": (0.92, 0.82, 0.12),
        "green": (0.14, 0.68, 0.24),
        "blue": (0.12, 0.36, 0.86),
        "purple": (0.56, 0.24, 0.76),
        "tan": (0.78, 0.64, 0.46),
        "orange": (0.95, 0.52, 0.12),
    }
    for key, rgb in named.items():
        if key in probe:
            return (rgb[0], rgb[1], rgb[2], alpha)

    # Try matplotlib-native color parsing (named color, hex, etc.).
    try:
        parsed = matplotlib.colors.to_rgba(raw_color if raw_color else raw_label)
        return (float(parsed[0]), float(parsed[1]), float(parsed[2]), alpha)
    except Exception:
        pass

    return (0.5, 0.5, 0.5, alpha)


def _tracked_item_is_red(item):
    raw_color = str(item.get("color", "")).strip().lower()
    raw_label = str(item.get("label", "")).strip().lower()
    return "red" in raw_color or "red" in raw_label


def _task_graph_default_dependent_label(task_graph_file, task_id):
    graph_path = Path(task_graph_file)
    if not task_id or not graph_path.exists():
        return ""
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
    if not isinstance(tasks, list):
        return ""

    tid = str(task_id).strip().lower()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if str(task.get("task_id", "")).strip().lower() != tid:
            continue
        params = task.get("params", {})
        if not isinstance(params, dict):
            return ""
        preferred = str(params.get("dependent_item_label", "")).strip()
        if preferred:
            return preferred
        fallback = str(params.get("target_label", "")).strip()
        if fallback:
            return fallback
        return ""
    return ""


def _load_waypoints(csv_path, task_id=""):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if task_id and str(row.get("task_id", "")).strip() != task_id:
                continue
            rows.append(row)

    def _row_idx(r):
        return int(_try_float(r.get("waypoint_index", 0), 0.0))

    rows.sort(key=_row_idx)
    return rows, list(rows[0].keys()) if rows else []


def _normalize_primitive(value, fallback="move_linear_to_pose"):
    candidate = str(value or "").strip()
    if candidate in MOVEMENT_PRIMITIVES:
        return candidate
    fb = str(fallback or "").strip()
    if fb in MOVEMENT_PRIMITIVES:
        return fb
    return MOVEMENT_PRIMITIVES[0]


class MockUR5Arm:
    """Offline mock that exchanges state with an external process via JSON file."""

    def __init__(self, state_file):
        self.state_file = Path(state_file)
        self._lock = threading.Lock()
        self._ensure_state()

    def _ensure_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self.state_file.exists():
            return
        init = {
            "timestamp": time.time(),
            "source": "python_mock_backend",
            "pose": [0.45, -0.2, 0.45, 2.2, -2.2, 0.0],
            "q": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            "gripper_open_pct": 100.0,
            "gripper_force_pct": 100.0,
            "last_command": "init",
        }
        self._write_state(init)

    def _read_state(self):
        with self._lock:
            try:
                with self.state_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    return {}
                return data
            except Exception:
                return {}

    def _write_state(self, payload):
        tmp = self.state_file.with_name(
            f"{self.state_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with self._lock:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.state_file)

    def _merge_and_write(self, updates):
        cur = self._read_state()
        cur.update(updates)
        cur["timestamp"] = time.time()
        self._write_state(cur)

    def disconnect(self):
        return None

    def move_linear_to_pose(self, pose, speed=None, acceleration=None, asynchronous=False):
        self._merge_and_write(
            {
                "pose": [float(v) for v in pose[:6]],
                "last_command": "move_linear_to_pose",
                "last_speed": speed,
                "last_acceleration": acceleration,
                "last_async": bool(asynchronous),
            }
        )
        return True

    def move_to_pose(self, pose, speed=None, acceleration=None, asynchronous=False):
        self._merge_and_write(
            {
                "pose": [float(v) for v in pose[:6]],
                "last_command": "move_to_pose",
                "last_speed": speed,
                "last_acceleration": acceleration,
                "last_async": bool(asynchronous),
            }
        )
        return True

    def move_to_joint_position(self, q, speed=None, acceleration=None, asynchronous=False):
        q_vals = [float(v) for v in q[:6]]
        self._merge_and_write(
            {
                "q": q_vals,
                "last_command": "move_to_joint_position",
                "last_speed": speed,
                "last_acceleration": acceleration,
                "last_async": bool(asynchronous),
            }
        )
        return True

    def move_path(self, waypoints, asynchronous=False):
        if waypoints:
            wp = waypoints[-1]
            pose = [float(v) for v in wp[:6]]
            self._merge_and_write({"pose": pose, "last_command": "move_path", "last_async": bool(asynchronous)})
        return True

    def get_end_effector_pose(self):
        data = self._read_state()
        pose = data.get("pose", [0.0] * 6)
        if not isinstance(pose, list) or len(pose) < 6:
            return [0.0] * 6
        return [float(v) for v in pose[:6]]

    def get_joint_positions(self):
        data = self._read_state()
        q = data.get("q", [0.0] * 6)
        if not isinstance(q, list) or len(q) < 6:
            return [0.0] * 6
        return [float(v) for v in q[:6]]

    def get_inverse_kinematics(self, pose, near_joints=None):
        _ = pose
        if near_joints is not None and len(near_joints) >= 6:
            return [float(v) for v in near_joints[:6]]
        return self.get_joint_positions()

    def set_gripper_open_pct(self, open_pct):
        self._merge_and_write({"gripper_open_pct": float(open_pct), "last_command": "set_gripper_open_pct"})
        return True

    def set_gripper_force_pct(self, force_pct):
        self._merge_and_write({"gripper_force_pct": float(force_pct), "last_command": "set_gripper_force_pct"})
        return True


class WaypointTuningRunnerUI:
    def __init__(self, args):
        self.args = args
        self.arm_side = args.arm_side
        self.arm_prefix = _arm_prefix(self.arm_side)

        self.current_file = Path(args.waypoints_csv)
        pattern = str(self.current_file.parent / "*.csv")
        self.file_list = sorted(Path(p) for p in glob.glob(pattern))
        if self.current_file not in self.file_list:
            self.file_list.append(self.current_file)
            self.file_list = sorted(self.file_list)
        self.file_index = self.file_list.index(self.current_file)

        self.source_rows = []
        self.edited_rows = []
        self.csv_header = []
        self._load_waypoint_file(self.current_file, task_id=args.task_id)

        self.current_idx = 0
        self._task_graph_default_label = _task_graph_default_dependent_label(
            args.task_graph_file,
            args.task_id,
        )

        self.vision_feeds = None
        self.vision_lock = threading.Lock()
        self._vision_start_attempted = False
        self._vision_start_error = ""
        self._ensure_vision_feeds(reason="startup")

        self.robot = None
        if args.mock_robot:
            self.robot = MockUR5Arm(args.mock_state_file)
        elif args.robot_ip and not args.dry_run:
            self.robot = UR5Arm(args.robot_ip, verbose=False)

        self.gripper = None
        self._gripper_activated = False
        self._gripper_status_reason = ""
        self._gripper_class = _load_robotiq_gripper_class()
        self._gripper_failures = 0
        if self.robot is not None and self._gripper_class is not None:
            try:
                self.gripper = self._gripper_class(self.robot.rtde_control)
                if not args.mock_robot:
                    try:
                        ok_activate = self.gripper.activate()
                        self._gripper_activated = True
                        ok_force = self.gripper.set_force(100)
                        ok_speed = self.gripper.set_speed(100)
                        pos_mm = self.gripper.current_pos_mm()
                        print(
                            "[tuner][gripper] init "
                            f"activate={ok_activate} set_force={ok_force} set_speed={ok_speed} pos_mm={pos_mm}"
                        )
                    except Exception as exc:
                        self._gripper_status_reason = f"initialization failed: {exc}"
                        print(f"[tuner] Gripper init warning: {exc}")
            except Exception as exc:
                self._gripper_status_reason = f"driver construction failed: {exc}"
                self.gripper = None
        elif self.robot is not None and self._gripper_class is None:
            self._gripper_status_reason = "gripper control module failed to load"

        self.root = tk.Tk()
        self.root.title("UR5 Waypoint Tuning Runner")
        self.root.geometry("1550x930")

        self.primitive_var = tk.StringVar(value=args.primitive)
        self.closed_loop_var = tk.BooleanVar(value=bool(args.closed_loop_vision))
        self.realtime_mode_var = tk.BooleanVar(value=False)
        self.object_label_var = tk.StringVar(value=args.object_label or "")
        self.object_label_combo = None
        self.family_var = tk.StringVar(value="spec:-1")
        self.family_combo = None
        self.family_offset_x_var = tk.DoubleVar(value=0.0)
        self.family_offset_y_var = tk.DoubleVar(value=0.0)
        self.family_offset_z_var = tk.DoubleVar(value=0.0)
        self.family_offset_map_by_arm = {"left": {}, "right": {}}
        self.step_sleep_var = tk.DoubleVar(value=float(args.play_step_sleep_s))
        self.file_label_var = tk.StringVar(value=str(self.current_file.name))
        self.gripper_open_pct_var = tk.DoubleVar(value=100.0)

        self.waypoint_slider = None
        self.value_vars = {}
        self.field_scales = {}
        self.entry_vars = {}
        self.delta_vars = {}
        self.nominal_vars = {}
        self.joint_vars = {}
        self.joint_entry_vars = {}
        self.joint_delta_vars = {}
        self.joint_nominal_vars = {}
        self.info_var = tk.StringVar(value="")
        self.distance_zero_wp_idx_by_domain = {}

        self._building = False
        self._play_thread = None
        self._stop_play = threading.Event()
        self._realtime_job = None
        self._gripper_state_lock = threading.Lock()
        self._gripper_worker_stop = threading.Event()
        self._gripper_worker_wakeup = threading.Event()
        self._gripper_next_pct = None
        self._gripper_last_forwarded_pct = None
        self._gripper_last_forward_ts = 0.0
        self._gripper_forward_period_s = 2.0
        self._mock_last_gripper_open_pct = None

        self._primitive_var_trace_id = self.primitive_var.trace_add("write", self._on_primitive_changed)

        self._build_ui()
        self._refresh_object_label_choices()
        self._refresh_family_choices()
        self._set_waypoint(0)

        self._gripper_io_lock = threading.Lock()
        self._gripper_worker_thread = threading.Thread(target=self._gripper_worker_loop, daemon=True)
        self._gripper_worker_thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _ensure_vision_feeds(self, reason=""):
        if self.vision_feeds is not None:
            return True
        if _get_or_start_vision_feeds is None:
            self._vision_start_error = "vision module unavailable"
            return False
        try:
            self.vision_feeds = _get_or_start_vision_feeds(
                params={
                    "task_graph_file": self.args.task_graph_file,
                    "vision_camera_scan_max_index": int(self.args.vision_camera_scan_max_index),
                }
            )
            self._vision_start_attempted = True
            self._vision_start_error = ""
            print(f"[tuner] Vision feeds ready (reason={reason or 'unspecified'})")
            return True
        except Exception as exc:
            self._vision_start_attempted = True
            self._vision_start_error = str(exc)
            self.vision_feeds = None
            print(f"[tuner] Warning: vision feeds unavailable (reason={reason or 'unspecified'}): {exc}")
            return False

    def _on_closed_loop_toggle(self):
        if not bool(self.closed_loop_var.get()):
            return
        if self._ensure_vision_feeds(reason="closed_loop_toggle"):
            self.info_var.set("Closed-loop vision enabled: camera feeds are running")
        else:
            msg = self._vision_start_error or "vision feeds unavailable"
            self.info_var.set(f"Closed-loop vision enabled, but cameras are unavailable: {msg}")

    def _load_waypoint_file(self, csv_path, task_id=""):
        src_rows, csv_header = _load_waypoints(str(csv_path), task_id=task_id)
        if not src_rows:
            raise RuntimeError(f"No waypoints found in {csv_path} for task_id='{task_id}'")
        fallback_primitive = _normalize_primitive(self.args.primitive)
        for row in src_rows:
            row["movement_primitive"] = _normalize_primitive(
                row.get("movement_primitive", ""),
                fallback=fallback_primitive,
            )
        self.current_file = Path(csv_path)
        self.csv_header = csv_header
        self.source_rows = src_rows
        self.edited_rows = deepcopy(src_rows)

    def _step_file(self, delta):
        if not self.file_list:
            return
        new_idx = (self.file_index + delta) % len(self.file_list)
        new_file = self.file_list[new_idx]
        try:
            self._load_waypoint_file(new_file, task_id=self.args.task_id)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return
        self.file_index = new_idx
        self.file_label_var.set(str(new_file.name))
        self.waypoint_slider.configure(to=max(0, len(self.edited_rows) - 1))
        self._refresh_object_label_choices()
        self._refresh_family_choices()
        self._set_waypoint(0)

    def _refresh_family_choices(self):
        families = set(self._active_family_offset_map().keys())
        for row in self.edited_rows:
            for item in _safe_json_list(row.get("tracked_items_json", "")):
                if not isinstance(item, dict):
                    continue
                families.add(_tracked_item_family_key(item))

        ordered = sorted(families)
        if not ordered:
            ordered = ["spec:-1"]

        if self.family_combo is not None:
            self.family_combo["values"] = ordered

        cur = str(self.family_var.get() or "").strip()
        if cur not in ordered:
            self.family_var.set(ordered[0])
        self._sync_family_offset_controls_from_selection()

    def _sync_family_offset_controls_from_selection(self):
        fam = str(self.family_var.get() or "").strip()
        off = self._active_family_offset_map().get(fam, (0.0, 0.0, 0.0))
        self.family_offset_x_var.set(float(off[0]))
        self.family_offset_y_var.set(float(off[1]))
        self.family_offset_z_var.set(float(off[2]))

    def _active_family_offset_map(self):
        arm = str(self.arm_side).strip().lower()
        if arm not in self.family_offset_map_by_arm:
            self.family_offset_map_by_arm[arm] = {}
        return self.family_offset_map_by_arm[arm]

    def _on_family_changed(self):
        self._sync_family_offset_controls_from_selection()
        self._refresh_plots()
        self._set_waypoint(self.current_idx)

    def _on_family_offset_changed(self):
        fam = str(self.family_var.get() or "").strip() or "spec:-1"
        self._active_family_offset_map()[fam] = (
            float(self.family_offset_x_var.get()),
            float(self.family_offset_y_var.get()),
            float(self.family_offset_z_var.get()),
        )
        self._refresh_plots()
        self._set_waypoint(self.current_idx)

    def _refresh_object_label_choices(self):
        labels = set()

        if self._task_graph_default_label:
            labels.add(self._task_graph_default_label)
        if self.args.object_label:
            labels.add(str(self.args.object_label).strip())

        for row in self.edited_rows:
            dep = str(row.get("dependent_item_label", "")).strip()
            if dep:
                labels.add(dep)
            for item in _safe_json_list(row.get("tracked_items_json", "")):
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).strip()
                if label:
                    labels.add(label)

        ordered = sorted(labels, key=lambda x: x.lower())
        values = [DEPENDENT_ITEM_NA] + ordered
        if self.object_label_combo is not None:
            self.object_label_combo["values"] = values

        current = str(self.object_label_var.get() or "").strip()
        if current == DEPENDENT_ITEM_NA:
            return
        if current and current in labels:
            return

        if self.args.object_label and str(self.args.object_label).strip() in labels:
            self.object_label_var.set(str(self.args.object_label).strip())
            return

        if self._task_graph_default_label and self._task_graph_default_label in labels:
            self.object_label_var.set(self._task_graph_default_label)
            return

        if ordered:
            self.object_label_var.set(ordered[0])
            return

        self.object_label_var.set(DEPENDENT_ITEM_NA)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=8)
        controls.grid(row=0, column=0, sticky="ns")

        viz = ttk.Frame(self.root, padding=8)
        viz.grid(row=0, column=1, sticky="nsew")
        viz.columnconfigure(0, weight=1)
        viz.rowconfigure(0, weight=1)
        viz.rowconfigure(1, weight=1)

        ttk.Label(controls, text=f"Task: {self.args.task_id or '(all)'}", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 6))
        ttk.Label(controls, text=f"Arm Side: {self.arm_side}").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Button(controls, text="Prev File", command=lambda: self._step_file(-1)).grid(row=1, column=2, sticky="ew", padx=(4, 2))
        ttk.Button(controls, text="Next File", command=lambda: self._step_file(1)).grid(row=1, column=3, sticky="ew", padx=(2, 2))
        ttk.Label(controls, textvariable=self.file_label_var, width=22).grid(row=1, column=4, sticky="w")

        ttk.Label(controls, text="Waypoint").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.waypoint_slider = tk.Scale(
            controls,
            from_=0,
            to=max(0, len(self.edited_rows) - 1),
            orient=tk.HORIZONTAL,
            length=360,
            command=lambda v: self._set_waypoint(int(float(v))),
        )
        self.waypoint_slider.grid(row=3, column=0, columnspan=4, sticky="ew")

        ttk.Button(controls, text="Prev", command=self._prev_waypoint).grid(row=4, column=0, pady=4, sticky="ew")
        ttk.Button(controls, text="Next", command=self._next_waypoint).grid(row=4, column=1, pady=4, sticky="ew")
        ttk.Button(controls, text="Exec Current", command=self._execute_current).grid(row=4, column=2, pady=4, sticky="ew")
        ttk.Button(controls, text="Save Tuned CSV", command=self._save_tuned_csv).grid(row=4, column=3, pady=4, sticky="ew")
        ttk.Button(controls, text="Save Over Current CSV", command=self._save_over_current_csv).grid(row=4, column=4, pady=4, sticky="ew")

        ttk.Label(controls, text="Movement Primitive").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.OptionMenu(controls, self.primitive_var, self.primitive_var.get(), *MOVEMENT_PRIMITIVES).grid(row=6, column=0, columnspan=2, sticky="ew")

        ttk.Checkbutton(
            controls,
            text="Closed-loop vision playback",
            variable=self.closed_loop_var,
            command=self._on_closed_loop_toggle,
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Checkbutton(controls, text="Real-time tuning mode", variable=self.realtime_mode_var).grid(row=7, column=4, sticky="w", pady=(6, 0))
        ttk.Label(controls, text="Dependent/object label").grid(row=8, column=0, sticky="w")
        self.object_label_combo = ttk.Combobox(
            controls,
            textvariable=self.object_label_var,
            width=28,
            state="readonly",
            values=[],
        )
        self.object_label_combo.grid(row=9, column=0, columnspan=5, sticky="ew")
        self.object_label_combo.bind("<<ComboboxSelected>>", lambda _ev: self._on_object_label_changed())
        ttk.Button(controls, text="Persist Sensitivity ->", command=self._persist_sensitivity_forward).grid(
            row=8, column=3, columnspan=2, sticky="ew", padx=(4, 0)
        )
        ttk.Button(controls, text="Persist Mode ->", command=self._persist_primitive_forward).grid(
            row=9, column=3, columnspan=2, sticky="ew", padx=(4, 0)
        )

        ttk.Label(controls, text="Camera spec family").grid(row=10, column=0, sticky="w", pady=(6, 0))
        self.family_combo = ttk.Combobox(
            controls,
            textvariable=self.family_var,
            width=18,
            state="readonly",
            values=["spec:-1"],
        )
        self.family_combo.grid(row=10, column=1, columnspan=2, sticky="ew")
        self.family_combo.bind("<<ComboboxSelected>>", lambda _ev: self._on_family_changed())

        ttk.Label(controls, text="Family XYZ offset [m]").grid(row=11, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=-0.3,
            to=0.3,
            orient=tk.HORIZONTAL,
            resolution=0.001,
            length=120,
            variable=self.family_offset_x_var,
            command=lambda _v: self._on_family_offset_changed(),
            label="dx",
        ).grid(row=11, column=1, sticky="ew")
        tk.Scale(
            controls,
            from_=-0.3,
            to=0.3,
            orient=tk.HORIZONTAL,
            resolution=0.001,
            length=120,
            variable=self.family_offset_y_var,
            command=lambda _v: self._on_family_offset_changed(),
            label="dy",
        ).grid(row=11, column=2, sticky="ew")
        tk.Scale(
            controls,
            from_=-0.3,
            to=0.3,
            orient=tk.HORIZONTAL,
            resolution=0.001,
            length=120,
            variable=self.family_offset_z_var,
            command=lambda _v: self._on_family_offset_changed(),
            label="dz",
        ).grid(row=11, column=3, sticky="ew")

        ttk.Label(controls, text="Play step sleep [s]").grid(row=12, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.step_sleep_var, width=8).grid(row=12, column=1, sticky="w")

        ttk.Button(controls, text="Play Open Loop", command=lambda: self._play_sequence(False)).grid(row=13, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="Play Closed Loop", command=lambda: self._play_sequence(True)).grid(row=13, column=2, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="Stop Play", command=self._stop_playback).grid(row=13, column=4, sticky="ew", pady=4)

        ttk.Button(controls, text="Add WP From Trace", command=self._add_waypoint_from_trace).grid(row=14, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(controls, text="Add WP From Robot", command=self._add_waypoint_from_robot).grid(row=14, column=2, columnspan=2, sticky="ew", pady=2)

        ttk.Button(controls, text="Set Dist Zero Here", command=self._set_distance_zero_here).grid(
            row=15, column=0, columnspan=4, sticky="ew", pady=2
        )

        row0 = 16
        ttk.Label(controls, text="Edit Pose (slider + text)", font=("Segoe UI", 10, "bold")).grid(row=row0, column=0, columnspan=5, sticky="w", pady=(10, 2))

        for i, field in enumerate(EDIT_FIELDS):
            rr = row0 + 1 + i
            if field in ("x", "y", "z"):
                low, high = (-POSITION_SLIDER_HALF_RANGE_M, POSITION_SLIDER_HALF_RANGE_M)
            else:
                low, high = (-ROTATION_SLIDER_HALF_RANGE_RAD, ROTATION_SLIDER_HALF_RANGE_RAD)
            res = FIELD_RESOLUTION[field]

            ttk.Label(controls, text=field).grid(row=rr, column=0, sticky="w")

            v = tk.DoubleVar(value=0.0)
            e = tk.StringVar(value="0.000")
            n = tk.StringVar(value="nom=0.000")
            d = tk.StringVar(value="Δ=0.000")

            self.value_vars[field] = v
            self.entry_vars[field] = e
            self.nominal_vars[field] = n
            self.delta_vars[field] = d

            scale = tk.Scale(
                controls,
                from_=low,
                to=high,
                orient=tk.HORIZONTAL,
                resolution=res,
                length=220,
                variable=v,
                command=lambda _vv, f=field: self._on_scale_change(f),
            )
            scale.grid(row=rr, column=1, sticky="ew")
            self.field_scales[field] = scale

            entry = ttk.Entry(controls, textvariable=e, width=10)
            entry.grid(row=rr, column=2, sticky="w")
            entry.bind("<Return>", lambda _ev, f=field: self._on_entry_change(f))
            entry.bind("<FocusOut>", lambda _ev, f=field: self._on_entry_change(f))

            ttk.Label(controls, textvariable=n, width=13).grid(row=rr, column=3, sticky="w")
            ttk.Label(controls, textvariable=d, width=13).grid(row=rr, column=4, sticky="w")

        joint_row0 = row0 + 8
        ttk.Label(controls, text="Joint Tuning [rad]", font=("Segoe UI", 10, "bold")).grid(row=joint_row0, column=0, columnspan=5, sticky="w", pady=(8, 2))
        for i, field in enumerate(JOINT_FIELDS):
            rr = joint_row0 + 1 + i
            low = float(JOINT_LIMITS[i][0])
            high = float(JOINT_LIMITS[i][1])

            ttk.Label(controls, text=field).grid(row=rr, column=0, sticky="w")
            v = tk.DoubleVar(value=0.0)
            e = tk.StringVar(value="0.000")
            n = tk.StringVar(value="nom=0.000")
            d = tk.StringVar(value="Δ=0.000")

            self.joint_vars[field] = v
            self.joint_entry_vars[field] = e
            self.joint_nominal_vars[field] = n
            self.joint_delta_vars[field] = d

            scale = tk.Scale(
                controls,
                from_=low,
                to=high,
                orient=tk.HORIZONTAL,
                resolution=0.001,
                length=220,
                variable=v,
                command=lambda _vv, f=field: self._on_joint_scale_change(f),
            )
            scale.grid(row=rr, column=1, sticky="ew")

            entry = ttk.Entry(controls, textvariable=e, width=10)
            entry.grid(row=rr, column=2, sticky="w")
            entry.bind("<Return>", lambda _ev, f=field: self._on_joint_entry_change(f))
            entry.bind("<FocusOut>", lambda _ev, f=field: self._on_joint_entry_change(f))

            ttk.Label(controls, textvariable=n, width=13).grid(row=rr, column=3, sticky="w")
            ttk.Label(controls, textvariable=d, width=13).grid(row=rr, column=4, sticky="w")

        ttk.Button(
            controls,
            text="Wrist3 +180deg ->",
            command=lambda: self._apply_wrist3_offset_forward(math.pi),
        ).grid(row=joint_row0 + 7, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(
            controls,
            text="Wrist3 -180deg ->",
            command=lambda: self._apply_wrist3_offset_forward(-math.pi),
        ).grid(row=joint_row0 + 7, column=2, columnspan=2, sticky="ew", pady=(4, 0), padx=(4, 0))

        gr_row = joint_row0 + 8
        ttk.Label(controls, text="Gripper").grid(row=gr_row, column=0, sticky="w", pady=(8, 0))
        ttk.Label(controls, text="Open %").grid(row=gr_row + 1, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            resolution=1,
            length=220,
            variable=self.gripper_open_pct_var,
            command=lambda _v: self._on_gripper_open_change(),
        ).grid(row=gr_row + 1, column=1, sticky="ew")
        ttk.Button(controls, text="Persist Gripper ->", command=self._persist_gripper_forward).grid(
            row=gr_row + 1, column=2, columnspan=2, sticky="ew", padx=(4, 0)
        )

        ttk.Label(controls, textvariable=self.info_var, wraplength=420, foreground="#003f5c").grid(row=gr_row + 2, column=0, columnspan=5, sticky="w", pady=(8, 2))

        fig = Figure(figsize=(11, 8), dpi=100)
        self.ax_trace = fig.add_subplot(211, projection="3d")
        self.ax_dist = fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(fig, master=viz)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, sticky="nsew")

    def _queue_gripper_open_pct(self, open_pct):
        # Single-slot overwrite queue: keep only the latest pending target.
        with self._gripper_state_lock:
            self._gripper_next_pct = int(max(0, min(100, int(open_pct))))
        self._gripper_worker_wakeup.set()

    def _wait_for_gripper_forward(self, target_pct, timeout_s=3.0):
        t0 = time.time()
        while (time.time() - t0) < float(timeout_s):
            with self._gripper_state_lock:
                last = self._gripper_last_forwarded_pct
                pending = self._gripper_next_pct
            if last is not None and int(last) == int(target_pct) and pending is None:
                return True
            time.sleep(0.01)
        return False

    def _gripper_worker_loop(self):
        while not self._gripper_worker_stop.is_set():
            self._gripper_worker_wakeup.wait(timeout=0.1)
            self._gripper_worker_wakeup.clear()
            if self._gripper_worker_stop.is_set():
                break

            while not self._gripper_worker_stop.is_set():
                with self._gripper_state_lock:
                    target_pct = self._gripper_next_pct
                    self._gripper_next_pct = None
                if target_pct is None:
                    break

                elapsed = time.time() - float(self._gripper_last_forward_ts)
                wait_s = float(self._gripper_forward_period_s) - elapsed
                if wait_s > 0:
                    if self._gripper_worker_stop.wait(timeout=wait_s):
                        break

                # Coalesce: if newer pushes arrived during the throttle wait,
                # overwrite with the latest single pending item.
                while not self._gripper_worker_stop.is_set():
                    with self._gripper_state_lock:
                        newer = self._gripper_next_pct
                        if newer is None:
                            break
                        target_pct = newer
                        self._gripper_next_pct = None

                self._send_gripper_open_pct(int(target_pct))
                with self._gripper_state_lock:
                    self._gripper_last_forwarded_pct = int(target_pct)
                    self._gripper_last_forward_ts = time.time()

    def _recover_gripper_session(self, reason):
        if self.robot is None or self.args.mock_robot:
            return False
        if self._gripper_class is None:
            self._gripper_status_reason = "gripper control module failed to load"
            return False
        try:
            self.gripper = self._gripper_class(self.robot.rtde_control)
            ok_activate = self.gripper.activate()
            self._gripper_activated = True
            ok_force = self.gripper.set_force(100)
            ok_speed = self.gripper.set_speed(100)
            pos_mm = self.gripper.current_pos_mm()
            print(
                "[tuner][gripper] recover "
                f"reason={reason} activate={ok_activate} set_force={ok_force} set_speed={ok_speed} pos_mm={pos_mm}"
            )
            self._gripper_status_reason = ""
            return True
        except Exception as exc:
            self._gripper_activated = False
            self._gripper_status_reason = f"recovery failed: {exc}"
            print(f"[tuner][gripper] recover_failed reason={reason} err={exc}")
            return False

    def _on_scale_change(self, field):
        if self._building:
            return
        v = float(self.value_vars[field].get())
        self.entry_vars[field].set(f"{v:.4f}")
        self._apply_field_to_row(field, v)
        self._schedule_realtime_update()

    def _on_entry_change(self, field):
        if self._building:
            return
        v = _try_float(self.entry_vars[field].get(), self.value_vars[field].get())
        self.value_vars[field].set(v)
        self.entry_vars[field].set(f"{v:.4f}")
        self._apply_field_to_row(field, v)
        self._schedule_realtime_update()

    def _on_joint_scale_change(self, field):
        if self._building:
            return
        v = float(self.joint_vars[field].get())
        self.joint_entry_vars[field].set(f"{v:.4f}")
        self._apply_joint_field_to_row(field, v)
        self._schedule_realtime_update()

    def _on_joint_entry_change(self, field):
        if self._building:
            return
        v = _try_float(self.joint_entry_vars[field].get(), self.joint_vars[field].get())
        self.joint_vars[field].set(v)
        self.joint_entry_vars[field].set(f"{v:.4f}")
        self._apply_joint_field_to_row(field, v)
        self._schedule_realtime_update()

    def _on_object_label_changed(self):
        row = self.edited_rows[self.current_idx]
        selected = str(self.object_label_var.get() or "").strip()
        row["dependent_item_label"] = "" if selected == DEPENDENT_ITEM_NA else selected
        self._refresh_plots()
        self._set_waypoint(self.current_idx)

    def _row_primitive(self, row):
        return _normalize_primitive(row.get("movement_primitive", ""), fallback=self.args.primitive)

    def _on_primitive_changed(self, *_args):
        if self._building:
            return
        if self.current_idx < 0 or self.current_idx >= len(self.edited_rows):
            return
        raw_selected = str(self.primitive_var.get() or "").strip()
        selected = _normalize_primitive(raw_selected, fallback=self.args.primitive)
        if raw_selected != selected:
            self.primitive_var.set(selected)
            return
        self.edited_rows[self.current_idx]["movement_primitive"] = selected

    def _persist_primitive_forward(self):
        primitive = _normalize_primitive(self.primitive_var.get(), fallback=self.args.primitive)
        start_idx = int(self.current_idx)
        for i in range(start_idx, len(self.edited_rows)):
            self.edited_rows[i]["movement_primitive"] = primitive

        self._set_waypoint(self.current_idx)
        self.info_var.set(
            f"Persisted mode {primitive} from waypoint {start_idx + 1} to {len(self.edited_rows)}"
        )

    def _persist_sensitivity_forward(self):
        selected = str(self.object_label_var.get() or "").strip()
        dep_value = "" if selected == DEPENDENT_ITEM_NA else selected
        start_idx = int(self.current_idx)
        for i in range(start_idx, len(self.edited_rows)):
            self.edited_rows[i]["dependent_item_label"] = dep_value

        self._set_waypoint(self.current_idx)
        shown = DEPENDENT_ITEM_NA if dep_value == "" else dep_value
        self.info_var.set(
            f"Persisted sensitivity {shown} from waypoint {start_idx + 1} to {len(self.edited_rows)}"
        )

    def _set_distance_zero_here(self):
        row = self.edited_rows[self.current_idx]
        dep = self._current_object_label(row)
        domain = _sensitivity_domain_key(row)
        item = _tracked_item_with_label(row, dep)
        if isinstance(item, dict):
            fam = _tracked_item_family_key(item)
            active_offsets = self._active_family_offset_map()
            off = active_offsets.get(fam, (0.0, 0.0, 0.0))
            pos = item.get("position")
            if isinstance(pos, list) and len(pos) >= 3:
                raw = [_try_float(pos[0]), _try_float(pos[1]), _try_float(pos[2])]
                adj = [raw[0] + off[0], raw[1] + off[1], raw[2] + off[2]]
                wp = _row_global_xyz(row, self.arm_side)
                correction = [wp[0] - adj[0], wp[1] - adj[1], wp[2] - adj[2]]
                new_off = (off[0] + correction[0], off[1] + correction[1], off[2] + correction[2])
                active_offsets[fam] = new_off
                self.family_var.set(fam)
                self._sync_family_offset_controls_from_selection()

        if domain != DEPENDENT_ITEM_NA:
            self.distance_zero_wp_idx_by_domain[domain] = int(self.current_idx)
        self._refresh_plots()
        self._set_waypoint(self.current_idx)

    def _on_gripper_open_change(self):
        if self._building:
            return
        open_pct = max(0, min(100, int(round(float(self.gripper_open_pct_var.get())))))
        row = self.edited_rows[self.current_idx]
        pct_col = f"{self.arm_prefix}_gripper_open_pct"
        legacy_col = f"{self.arm_prefix}_gripper_open"
        row[pct_col] = f"{float(open_pct):.1f}"
        row[legacy_col] = "1" if open_pct >= 50 else "0"

        if self.args.mock_robot and self.robot is not None:
            if self._mock_last_gripper_open_pct != int(open_pct):
                self.robot.set_gripper_open_pct(open_pct)
                self._mock_last_gripper_open_pct = int(open_pct)
                self.info_var.set(f"Gripper open set to {open_pct}% (mock)")
            return
        if self.gripper is None:
            reason = self._gripper_status_reason or "connect robot in remote mode and ensure gripper driver is loaded"
            self.info_var.set(f"Gripper unavailable: {reason}")
            return

        should_queue = True
        with self._gripper_state_lock:
            if self._gripper_next_pct is not None and int(self._gripper_next_pct) == int(open_pct):
                should_queue = False
            elif self._gripper_last_forwarded_pct is not None and int(self._gripper_last_forwarded_pct) == int(open_pct):
                should_queue = False
        if should_queue:
            self._queue_gripper_open_pct(open_pct)

    def _persist_gripper_forward(self):
        open_pct = max(0, min(100, int(round(float(self.gripper_open_pct_var.get())))))
        pct_col = f"{self.arm_prefix}_gripper_open_pct"
        legacy_col = f"{self.arm_prefix}_gripper_open"
        start_idx = int(self.current_idx)
        for i in range(start_idx, len(self.edited_rows)):
            row = self.edited_rows[i]
            row[pct_col] = f"{float(open_pct):.1f}"
            row[legacy_col] = "1" if open_pct >= 50 else "0"

        self._set_waypoint(self.current_idx)
        self.info_var.set(
            f"Persisted gripper {open_pct}% from waypoint {start_idx + 1} to {len(self.edited_rows)}"
        )

        if self.realtime_mode_var.get():
            self._apply_gripper_for_waypoint(self.current_idx, blocking=False)

    def _apply_wrist3_offset_forward(self, delta_rad):
        start_idx = int(self.current_idx)
        joint_idx = 5
        joint_field = JOINT_FIELDS[joint_idx]
        low = float(JOINT_LIMITS[joint_idx][0])
        high = float(JOINT_LIMITS[joint_idx][1])

        clamped = 0
        for i in range(start_idx, len(self.edited_rows)):
            row = self.edited_rows[i]
            q = _q_from_row(row, self.arm_side) or [0.0] * 6
            target = float(q[joint_idx]) + float(delta_rad)
            bounded = min(high, max(low, target))
            if not math.isclose(target, bounded, rel_tol=0.0, abs_tol=1e-12):
                clamped += 1
            q[joint_idx] = bounded
            _set_q_in_row(row, self.arm_side, q)

        self._set_waypoint(self.current_idx)
        self._schedule_realtime_update()

        sign = "+" if float(delta_rad) >= 0.0 else "-"
        msg = (
            f"Persisted wrist_3 {sign}180deg from waypoint {start_idx + 1} to {len(self.edited_rows)}"
        )
        if clamped > 0:
            msg += f" ({clamped} clamped to joint limits for {joint_field})"
        self.info_var.set(msg)

    def _send_gripper_open_pct(self, open_pct):
        # Gripper open percentage maps to 0..85mm.
        pos_mm = open_pct * GRIPPER_OPEN_MM_MAX / 100.0
        last_error = ""
        with self._gripper_io_lock:
            for attempt in (1, 2):
                try:
                    if self.gripper is None:
                        if not self._recover_gripper_session(reason="gripper_none"):
                            raise RuntimeError(self._gripper_status_reason or "gripper unavailable")

                    if not self._gripper_activated:
                        ok_activate = self.gripper.activate()
                        self._gripper_activated = True
                        ok_force = self.gripper.set_force(100)
                        ok_speed = self.gripper.set_speed(100)
                        print(
                            "[tuner][gripper] lazy_init "
                            f"activate={ok_activate} set_force={ok_force} set_speed={ok_speed}"
                        )

                    ok = self.gripper.move(int(round(pos_mm)))
                    if ok is False and open_pct >= 98:
                        ok = self.gripper.open()
                    elif ok is False and open_pct <= 2:
                        ok = self.gripper.close()

                    if ok is False:
                        raise RuntimeError("move command rejected")

                    pos_readback = self.gripper.current_pos_mm()
                    self._gripper_failures = 0
                    print(
                        "[tuner][gripper] move "
                        f"attempt={attempt} target_pct={open_pct} target_mm={int(round(pos_mm))} ok={ok} readback_mm={pos_readback}"
                    )
                    self.root.after(0, lambda: self.info_var.set(f"Gripper open set to {open_pct}%"))
                    return
                except Exception as exc:
                    last_error = str(exc)
                    self._gripper_failures += 1
                    print(
                        "[tuner][gripper] move_failed "
                        f"attempt={attempt} target_pct={open_pct} target_mm={int(round(pos_mm))} err={exc}"
                    )
                    self._gripper_activated = False
                    if attempt == 1:
                        self._recover_gripper_session(reason=f"move_failed:{exc}")

        self.root.after(
            0,
            lambda: self.info_var.set(
                f"Gripper command failed after retry ({self._gripper_failures} failures): {last_error}"
            ),
        )

    def _schedule_realtime_update(self):
        if not self.realtime_mode_var.get() or self.robot is None:
            return
        if self._realtime_job is not None:
            self.root.after_cancel(self._realtime_job)
        self._realtime_job = self.root.after(120, self._apply_realtime_update)

    def _apply_realtime_update(self):
        self._realtime_job = None
        try:
            row = self.edited_rows[self.current_idx]
            primitive = self._row_primitive(row)
            if primitive == "move_to_joint_position":
                q = _q_from_row(row, self.arm_side)
                if q is not None:
                    self.robot.move_to_joint_position(q)
            else:
                pose = [
                    _try_float(row.get(f"{self.arm_prefix}_x"), 0.0),
                    _try_float(row.get(f"{self.arm_prefix}_y"), 0.0),
                    _try_float(row.get(f"{self.arm_prefix}_z"), 0.0),
                    _try_float(row.get(f"{self.arm_prefix}_rx"), 0.0),
                    _try_float(row.get(f"{self.arm_prefix}_ry"), 0.0),
                    _try_float(row.get(f"{self.arm_prefix}_rz"), 0.0),
                ]
                if primitive == "move_to_pose":
                    self.robot.move_to_pose(pose)
                else:
                    self.robot.move_linear_to_pose(pose)
            self._pull_robot_state_into_waypoint()
        except Exception:
            pass

    def _set_waypoint(self, idx):
        idx = max(0, min(idx, len(self.edited_rows) - 1))
        self.current_idx = idx
        self.waypoint_slider.set(idx)
        self._building = True
        try:
            src = self.source_rows[idx]
            row = self.edited_rows[idx]
            for f in EDIT_FIELDS:
                col = f"{self.arm_prefix}_{f}"
                nominal = _try_float(src.get(col), 0.0)
                cur = _try_float(row.get(col), nominal)
                half_range = POSITION_SLIDER_HALF_RANGE_M if f in ("x", "y", "z") else ROTATION_SLIDER_HALF_RANGE_RAD
                scale = self.field_scales.get(f)
                if scale is not None:
                    scale.configure(from_=(nominal - half_range), to=(nominal + half_range))
                self.value_vars[f].set(cur)
                self.entry_vars[f].set(f"{cur:.4f}")
                self.nominal_vars[f].set(f"nom={nominal:.4f}")
                self.delta_vars[f].set(f"Δ={cur - nominal:+.4f}")

            src_q = _q_from_row(src, self.arm_side) or [0.0] * 6
            cur_q = _q_from_row(row, self.arm_side) or src_q
            for i, field in enumerate(JOINT_FIELDS):
                nominal_q = float(src_q[i])
                cur_qi = float(cur_q[i])
                self.joint_vars[field].set(cur_qi)
                self.joint_entry_vars[field].set(f"{cur_qi:.4f}")
                self.joint_nominal_vars[field].set(f"nom={nominal_q:.4f}")
                self.joint_delta_vars[field].set(f"Δ={cur_qi - nominal_q:+.4f}")

            self.gripper_open_pct_var.set(float(self._row_gripper_open_pct(row)))
            self.primitive_var.set(self._row_primitive(row))

            dep_row = str(row.get("dependent_item_label", "")).strip()
            if dep_row:
                self.object_label_var.set(dep_row)
            else:
                self.object_label_var.set(DEPENDENT_ITEM_NA)

            wp_idx = int(_try_float(row.get("waypoint_index", idx + 1), idx + 1))
            wp_name = str(row.get("waypoint_name", "")).strip() or f"wp_{wp_idx}"
            dep = self._current_object_label(row)
            dist_msg = "dist=-"
            if dep:
                obj = _tracked_item_position_with_offsets(row, dep, self._active_family_offset_map())
                if obj is not None:
                    row_xyz = _row_global_xyz(row, self.arm_side)
                    dist_msg = f"dist={_distance3(row_xyz, obj):.4f} m"
            zero_msg = ""
            domain = _sensitivity_domain_key(row)
            if domain != DEPENDENT_ITEM_NA and domain in self.distance_zero_wp_idx_by_domain:
                zero_msg = f" zero_ref_wp={int(self.distance_zero_wp_idx_by_domain[domain]) + 1}"
            self.info_var.set(
                f"Waypoint {idx + 1}/{len(self.edited_rows)} : index={wp_idx} name={wp_name} dependent={dep or '-'} {dist_msg}{zero_msg}"
            )
        finally:
            self._building = False

        self._refresh_plots()
        if self.realtime_mode_var.get():
            self._apply_gripper_for_waypoint(self.current_idx, blocking=False)

    def _prev_waypoint(self):
        self._set_waypoint(self.current_idx - 1)

    def _next_waypoint(self):
        self._set_waypoint(self.current_idx + 1)

    def _apply_field_to_row(self, field, value):
        row = self.edited_rows[self.current_idx]
        src = self.source_rows[self.current_idx]
        col = f"{self.arm_prefix}_{field}"
        row[col] = f"{value:.9f}"
        if field in ("x", "y", "z"):
            _update_row_task_global_fields(row, self.arm_side)
        nominal = _try_float(src.get(col), 0.0)
        self.delta_vars[field].set(f"Δ={value - nominal:+.4f}")
        self._refresh_plots()

    def _apply_joint_field_to_row(self, field, value):
        row = self.edited_rows[self.current_idx]
        src = self.source_rows[self.current_idx]
        joint_idx = int(field.split("_")[1])
        q = _q_from_row(row, self.arm_side) or [0.0] * 6
        q[joint_idx] = float(value)
        _set_q_in_row(row, self.arm_side, q)
        nominal_q = (_q_from_row(src, self.arm_side) or [0.0] * 6)[joint_idx]
        self.joint_delta_vars[field].set(f"Δ={float(value) - float(nominal_q):+.4f}")

    def _current_object_label(self, row):
        label = str(row.get("dependent_item_label", "")).strip()
        if not label or label == DEPENDENT_ITEM_NA:
            return ""
        return label

    def _row_gripper_open_pct(self, row):
        gripper_pct_col = f"{self.arm_prefix}_gripper_open_pct"
        gripper_legacy_col = f"{self.arm_prefix}_gripper_open"
        gripper_pct = _try_float(row.get(gripper_pct_col), float("nan"))
        if not math.isfinite(gripper_pct):
            legacy_open = _try_float(row.get(gripper_legacy_col), 1.0)
            gripper_pct = 100.0 if legacy_open >= 0.5 else 0.0
        return max(0, min(100, int(round(float(gripper_pct)))))

    def _apply_gripper_for_waypoint(self, idx, blocking=False):
        if idx < 0 or idx >= len(self.edited_rows):
            return
        open_pct = self._row_gripper_open_pct(self.edited_rows[idx])
        if self.args.mock_robot and self.robot is not None:
            if self._mock_last_gripper_open_pct != int(open_pct):
                self.robot.set_gripper_open_pct(open_pct)
                self._mock_last_gripper_open_pct = int(open_pct)
            return
        if self.gripper is None:
            return
        should_queue = True
        with self._gripper_state_lock:
            if self._gripper_next_pct is not None and int(self._gripper_next_pct) == int(open_pct):
                should_queue = False
            elif self._gripper_last_forwarded_pct is not None and int(self._gripper_last_forwarded_pct) == int(open_pct):
                should_queue = False
        if should_queue:
            self._queue_gripper_open_pct(open_pct)
        if blocking and (should_queue or self._gripper_next_pct is not None):
            self._wait_for_gripper_forward(open_pct, timeout_s=4.5)

    def _add_waypoint_from_trace(self):
        if not self.edited_rows:
            return
        src = deepcopy(self.edited_rows[self.current_idx])
        new_wp_idx = len(self.edited_rows) + 1
        src["waypoint_index"] = str(new_wp_idx)
        src["waypoint_name"] = f"trace_wp_{new_wp_idx}"
        self.edited_rows.insert(self.current_idx + 1, src)
        self.source_rows.insert(self.current_idx + 1, deepcopy(src))
        self.waypoint_slider.configure(to=max(0, len(self.edited_rows) - 1))
        self._set_waypoint(self.current_idx + 1)

    def _add_waypoint_from_robot(self):
        if self.robot is None:
            messagebox.showwarning("Robot Not Connected", "Connect robot (--robot-ip) to capture current robot state.")
            return
        if not self.edited_rows:
            return
        try:
            pose = self.robot.get_end_effector_pose()
            q = self.robot.get_joint_positions()
        except Exception as exc:
            messagebox.showerror("Robot Read Error", str(exc))
            return

        row = deepcopy(self.edited_rows[self.current_idx])
        new_wp_idx = len(self.edited_rows) + 1
        row["waypoint_index"] = str(new_wp_idx)
        row["waypoint_name"] = f"robot_wp_{new_wp_idx}"
        for i, f in enumerate(EDIT_FIELDS):
            row[f"{self.arm_prefix}_{f}"] = f"{float(pose[i]):.9f}"
        _update_row_task_global_fields(row, self.arm_side)
        _set_q_in_row(row, self.arm_side, q)

        self.edited_rows.insert(self.current_idx + 1, row)
        self.source_rows.insert(self.current_idx + 1, deepcopy(row))
        self.waypoint_slider.configure(to=max(0, len(self.edited_rows) - 1))
        self._set_waypoint(self.current_idx + 1)

    def _pull_robot_state_into_waypoint(self):
        if self.robot is None:
            return
        try:
            pose = self.robot.get_end_effector_pose()
            q = self.robot.get_joint_positions()
        except Exception:
            return
        row = self.edited_rows[self.current_idx]
        for i, f in enumerate(EDIT_FIELDS):
            row[f"{self.arm_prefix}_{f}"] = f"{float(pose[i]):.9f}"
        _update_row_task_global_fields(row, self.arm_side)
        _set_q_in_row(row, self.arm_side, q)
        self._set_waypoint(self.current_idx)

    def _build_execution_pose(self, idx, closed_loop=False):
        row = self.edited_rows[idx]
        pose = _pose_from_row(row, self.arm_side)

        if not closed_loop:
            return pose

        self._ensure_vision_feeds(reason="closed_loop_execution")

        dep = self._current_object_label(row)
        if not dep:
            return pose

        recorded_obj = _tracked_item_position(row, dep)
        live_obj = None
        if self.vision_feeds is not None:
            try:
                with self.vision_lock:
                    live = self.vision_feeds.get_target(dep)
                if live is not None:
                    lx = _try_float(live.get("x", 0.0))
                    ly = _try_float(live.get("y", 0.0))
                    lz = _try_float(live.get("z", live.get("y", 0.0)))
                    live_obj = [lx, ly, lz]
            except Exception:
                live_obj = None

        if recorded_obj is None or live_obj is None:
            return pose

        # Camera-frame correction for tuning only shifts planar XY target.
        # Keep Z from the waypoint trace unchanged for safer repeatability.
        dx = live_obj[0] - recorded_obj[0]
        dy = live_obj[1] - recorded_obj[1]

        pose[0] += dx
        pose[1] += dy
        return pose

    def _execute_current(self):
        if self.robot is None:
            messagebox.showinfo("Dry Run", "No robot connected (dry run or no --robot-ip).")
            return

        try:
            row = self.edited_rows[self.current_idx]
            primitive = self._row_primitive(row)
            pose = self._build_execution_pose(self.current_idx, closed_loop=self.closed_loop_var.get())
            q = _q_from_row(row, self.arm_side)
            self._execute_primitive(pose, q, primitive=primitive)
            self._apply_gripper_for_waypoint(self.current_idx, blocking=False)
        except Exception as exc:
            messagebox.showerror("Execution Error", str(exc))

    def _execute_primitive(self, pose, q, primitive=None):
        primitive = _normalize_primitive(primitive, fallback=self.primitive_var.get())
        speed = self.args.speed
        accel = self.args.acceleration

        if primitive == "move_linear_to_pose":
            ok = self.robot.move_linear_to_pose(pose, speed=speed, acceleration=accel, asynchronous=False)
        elif primitive == "move_to_pose":
            ok = self.robot.move_to_pose(pose, speed=speed, acceleration=accel, asynchronous=False)
        elif primitive == "move_to_joint_position":
            if q is None:
                q = self.robot.get_inverse_kinematics(pose)
            if q is None:
                raise RuntimeError("No valid q-position available for move_to_joint_position")
            ok = self.robot.move_to_joint_position(q, speed=speed, acceleration=accel, asynchronous=False)
        elif primitive == "move_path":
            wp = pose + [speed or self.args.default_tool_speed, accel or self.args.default_tool_acceleration]
            ok = self.robot.move_path([wp], asynchronous=False)
        else:
            raise RuntimeError(f"Unsupported primitive: {primitive}")

        if not ok:
            raise RuntimeError(f"Primitive failed: {primitive}")

    def _play_sequence(self, closed_loop):
        if self.robot is None:
            messagebox.showinfo("Dry Run", "No robot connected (dry run or no --robot-ip).")
            return

        if self._play_thread is not None and self._play_thread.is_alive():
            messagebox.showwarning("Playback", "Playback already running")
            return

        self._stop_play.clear()

        def _run():
            try:
                for idx in range(self.current_idx, len(self.edited_rows)):
                    if self._stop_play.is_set():
                        break
                    self.root.after(0, lambda i=idx: self._set_waypoint(i))
                    row = self.edited_rows[idx]
                    primitive = self._row_primitive(row)
                    pose = self._build_execution_pose(idx, closed_loop=closed_loop)
                    q = _q_from_row(row, self.arm_side)
                    self._execute_primitive(pose, q, primitive=primitive)
                    self._apply_gripper_for_waypoint(idx, blocking=True)
                    time.sleep(max(0.0, float(self.step_sleep_var.get())))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Playback Error", str(exc)))

        self._play_thread = threading.Thread(target=_run, daemon=True)
        self._play_thread.start()

    def _stop_playback(self):
        self._stop_play.set()

    def _distance_series(self):
        nominal = []
        tuned = []
        idxs = []
        row_idxs = []
        domains = []

        for i, (src, row) in enumerate(zip(self.source_rows, self.edited_rows)):
            dep = self._current_object_label(row)
            obj = _tracked_item_position_with_offsets(row, dep, self._active_family_offset_map())
            if obj is None:
                continue

            src_xyz = _row_global_xyz(src, self.arm_side)
            row_xyz = _row_global_xyz(row, self.arm_side)

            idxs.append(i + 1)
            row_idxs.append(i)
            domains.append(_sensitivity_domain_key(row))
            nominal.append(_distance3(src_xyz, obj))
            tuned.append(_distance3(row_xyz, obj))

        if self.distance_zero_wp_idx_by_domain:
            row_pos = {row_i: pos for pos, row_i in enumerate(row_idxs)}
            baseline_by_domain = {}
            for domain, baseline_row in self.distance_zero_wp_idx_by_domain.items():
                bpos = row_pos.get(int(baseline_row), None)
                if bpos is None:
                    continue
                baseline_by_domain[domain] = (nominal[bpos], tuned[bpos])

            if baseline_by_domain:
                for i, domain in enumerate(domains):
                    base = baseline_by_domain.get(domain, None)
                    if base is None:
                        continue
                    nominal[i] -= base[0]
                    tuned[i] -= base[1]

        return idxs, nominal, tuned

    def _refresh_plots(self):
        self.ax_trace.clear()
        self.ax_dist.clear()

        left_xyz = [_row_global_xyz(r, "left") for r in self.edited_rows]
        right_xyz = [_row_global_xyz(r, "right") for r in self.edited_rows]

        left_valid = [_row_has_arm_pose_data(r, "left") for r in self.edited_rows]
        right_valid = [_row_has_arm_pose_data(r, "right") for r in self.edited_rows]

        legend_handles = []
        ci = self.current_idx

        if any(left_valid):
            lx = [p[0] if v else float("nan") for p, v in zip(left_xyz, left_valid)]
            ly = [p[1] if v else float("nan") for p, v in zip(left_xyz, left_valid)]
            lz = [p[2] if v else float("nan") for p, v in zip(left_xyz, left_valid)]
            left_line = self.ax_trace.plot(
                lx, ly, lz, color="#1f77b4", linewidth=1.4, label="left trace"
            )[0]
            legend_handles.append(left_line)
            if 0 <= ci < len(left_xyz) and left_valid[ci]:
                self.ax_trace.scatter([left_xyz[ci][0]], [left_xyz[ci][1]], [left_xyz[ci][2]], color="#1f77b4", s=65)

        if any(right_valid):
            rx = [p[0] if v else float("nan") for p, v in zip(right_xyz, right_valid)]
            ry = [p[1] if v else float("nan") for p, v in zip(right_xyz, right_valid)]
            rz = [p[2] if v else float("nan") for p, v in zip(right_xyz, right_valid)]
            right_line = self.ax_trace.plot(
                rx, ry, rz, color="#ff7f0e", linewidth=1.4, label="right trace"
            )[0]
            legend_handles.append(right_line)
            if 0 <= ci < len(right_xyz) and right_valid[ci]:
                self.ax_trace.scatter([right_xyz[ci][0]], [right_xyz[ci][1]], [right_xyz[ci][2]], color="#ff7f0e", s=65)

        row = self.edited_rows[ci]
        dep = self._current_object_label(row)
        red_item_counter = 0
        active_offsets = self._active_family_offset_map()
        for item in _safe_json_list(row.get("tracked_items_json", "")):
            if not isinstance(item, dict):
                continue
            pos = item.get("position")
            if not isinstance(pos, list) or len(pos) < 3:
                continue
            label = str(item.get("label", "obj"))
            is_active = bool(dep) and label.strip().lower() == dep.strip().lower()
            col = _tracked_item_rgba(item, is_active)
            fam = _tracked_item_family_key(item)
            dx, dy, dz = active_offsets.get(fam, (0.0, 0.0, 0.0))
            px = _try_float(pos[0]) + float(dx)
            py = _try_float(pos[1]) + float(dy)
            pz = _try_float(pos[2]) + float(dz)
            self.ax_trace.scatter([px], [py], [pz], color=col, s=40)

            if _tracked_item_is_red(item):
                red_item_counter += 1
                text_label = f"{label} [red#{red_item_counter}]"
                self.ax_trace.text(px, py, pz + 0.015, text_label, color="#ffdddd", fontsize=8)

        self.ax_trace.set_title("Arm Traces + Current Keypoint + Tracked Objects")
        self.ax_trace.set_xlabel("X [m]")
        self.ax_trace.set_ylabel("Y [m]")
        self.ax_trace.set_zlabel("Z [m]")
        if legend_handles:
            self.ax_trace.legend(handles=legend_handles, loc="upper left")

        idxs, nominal, tuned = self._distance_series()
        if idxs:
            self.ax_dist.plot(idxs, nominal, color="#999999", linewidth=1.2, label="nominal distance")
            self.ax_dist.plot(idxs, tuned, color="#2ca02c", linewidth=1.5, label="tuned distance")
            if 1 <= ci + 1 <= len(self.edited_rows):
                self.ax_dist.axvline(ci + 1, color="#444444", linestyle="--", linewidth=0.9)
        self.ax_dist.set_title("Distance To Dependent Object Over Waypoints")
        self.ax_dist.set_xlabel("Waypoint #")
        self.ax_dist.set_ylabel("Distance [m]")
        handles, labels = self.ax_dist.get_legend_handles_labels()
        if handles and labels:
            self.ax_dist.legend(loc="upper right")
        self.ax_dist.grid(alpha=0.2)

        self.canvas.draw_idle()

    def _save_tuned_csv(self):
        src = Path(self.current_file)
        task = self.args.task_id or "all"
        out_dir = Path(self.args.output_dir) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{src.stem}_tuned_{task}_{self.arm_side}_{stamp}.csv"

        header = list(self.csv_header)
        for extra in ["tune_source_csv", "tune_saved_at_iso"]:
            if extra not in header:
                header.append(extra)
        gripper_pct_col = f"{self.arm_prefix}_gripper_open_pct"
        gripper_legacy_col = f"{self.arm_prefix}_gripper_open"
        for extra in [gripper_pct_col, gripper_legacy_col]:
            if extra not in header:
                header.append(extra)
        if "movement_primitive" not in header:
            header.append("movement_primitive")

        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for row in self.edited_rows:
                out = dict(row)
                out["tune_source_csv"] = str(src)
                out["tune_saved_at_iso"] = datetime.now().isoformat()
                writer.writerow(out)

        messagebox.showinfo("Saved", f"Tuned CSV written:\n{out_path}")

    def _save_over_current_csv(self):
        out_path = Path(self.current_file)
        if not messagebox.askyesno("Confirm Overwrite", f"Overwrite current CSV?\n{out_path}"):
            return

        header = list(self.csv_header)
        for extra in ["tune_source_csv", "tune_saved_at_iso"]:
            if extra not in header:
                header.append(extra)
        gripper_pct_col = f"{self.arm_prefix}_gripper_open_pct"
        gripper_legacy_col = f"{self.arm_prefix}_gripper_open"
        for extra in [gripper_pct_col, gripper_legacy_col]:
            if extra not in header:
                header.append(extra)
        if "movement_primitive" not in header:
            header.append("movement_primitive")

        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for row in self.edited_rows:
                out = dict(row)
                out["tune_source_csv"] = str(out_path)
                out["tune_saved_at_iso"] = datetime.now().isoformat()
                writer.writerow(out)

        messagebox.showinfo("Saved", f"Updated current CSV:\n{out_path}")

    def _on_close(self):
        self._stop_play.set()
        self._gripper_worker_stop.set()
        self._gripper_worker_wakeup.set()
        try:
            if self.robot is not None:
                self.robot.disconnect()
        except Exception:
            pass
        try:
            if self.vision_feeds is not None:
                self.vision_feeds.stop()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="UR5 waypoint tuning runner (non-destructive CSV editor + playback).")
    parser.add_argument("--waypoints-csv", required=True, help="Input waypoint CSV (source will never be mutated)")
    parser.add_argument("--task-id", default="", help="Optional task_id filter within the waypoint CSV")
    parser.add_argument("--arm-side", choices=["left", "right"], default="right")
    parser.add_argument("--robot-ip", default="", help="Robot IP for selected arm side; leave empty for dry editing")
    parser.add_argument("--mock-robot", action="store_true", help="Use file-backed offline mock robot instead of hardware")
    parser.add_argument("--mock-state-file", default="traces/mock_robot_state.json", help="Shared JSON state path used by --mock-robot")
    parser.add_argument("--dry-run", action="store_true", help="Disable robot playback even if --robot-ip is set")
    parser.add_argument("--primitive", default="move_linear_to_pose", choices=MOVEMENT_PRIMITIVES)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--acceleration", type=float, default=None)
    parser.add_argument("--default-tool-speed", type=float, default=0.1)
    parser.add_argument("--default-tool-acceleration", type=float, default=1.0)
    parser.add_argument("--closed-loop-vision", action="store_true", help="Enable closed-loop playback option using live target updates")
    parser.add_argument("--task-graph-file", default="UR5/master_task_graph.json")
    parser.add_argument("--vision-camera-scan-max-index", type=int, default=12)
    parser.add_argument("--object-label", default="", help="Override dependent object label for distance/closed-loop computations")
    parser.add_argument("--play-step-sleep-s", type=float, default=0.2)
    parser.add_argument("--output-dir", default="UR5/tuned_waypoints")
    return parser.parse_args()


def main():
    args = parse_args()
    ui = WaypointTuningRunnerUI(args)
    ui.run()


if __name__ == "__main__":
    main()
