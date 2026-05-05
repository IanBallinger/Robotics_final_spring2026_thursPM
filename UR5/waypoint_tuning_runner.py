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
    spec = importlib.util.spec_from_file_location("robotiq_gripper_control_local", gripper_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
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

        self.vision_feeds = None
        self.vision_lock = threading.Lock()
        if args.closed_loop_vision and _get_or_start_vision_feeds is not None:
            try:
                self.vision_feeds = _get_or_start_vision_feeds(
                    params={
                        "task_graph_file": args.task_graph_file,
                        "vision_camera_scan_max_index": int(args.vision_camera_scan_max_index),
                    }
                )
            except Exception as exc:
                print(f"[tuner] Warning: vision feeds unavailable: {exc}")
                self.vision_feeds = None

        self.robot = None
        if args.mock_robot:
            self.robot = MockUR5Arm(args.mock_state_file)
        elif args.robot_ip and not args.dry_run:
            self.robot = UR5Arm(args.robot_ip, verbose=False)

        self.gripper = None
        gripper_class = _load_robotiq_gripper_class()
        if self.robot is not None and gripper_class is not None:
            try:
                self.gripper = gripper_class(self.robot.rtde_control)
            except Exception:
                self.gripper = None

        self.root = tk.Tk()
        self.root.title("UR5 Waypoint Tuning Runner")
        self.root.geometry("1550x930")

        self.primitive_var = tk.StringVar(value=args.primitive)
        self.closed_loop_var = tk.BooleanVar(value=bool(args.closed_loop_vision))
        self.realtime_mode_var = tk.BooleanVar(value=False)
        self.object_label_var = tk.StringVar(value=args.object_label or "")
        self.step_sleep_var = tk.DoubleVar(value=float(args.play_step_sleep_s))
        self.file_label_var = tk.StringVar(value=str(self.current_file.name))
        self.gripper_open_pct_var = tk.DoubleVar(value=100.0)
        self.gripper_force_pct_var = tk.DoubleVar(value=100.0)

        self.waypoint_slider = None
        self.value_vars = {}
        self.entry_vars = {}
        self.delta_vars = {}
        self.nominal_vars = {}
        self.joint_vars = {}
        self.joint_entry_vars = {}
        self.joint_delta_vars = {}
        self.joint_nominal_vars = {}
        self.info_var = tk.StringVar(value="")

        self._building = False
        self._play_thread = None
        self._stop_play = threading.Event()
        self._realtime_job = None

        self._build_ui()
        self._set_waypoint(0)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_waypoint_file(self, csv_path, task_id=""):
        src_rows, csv_header = _load_waypoints(str(csv_path), task_id=task_id)
        if not src_rows:
            raise RuntimeError(f"No waypoints found in {csv_path} for task_id='{task_id}'")
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
        self._set_waypoint(0)

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

        ttk.Label(controls, text="Movement Primitive").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.OptionMenu(controls, self.primitive_var, self.primitive_var.get(), *MOVEMENT_PRIMITIVES).grid(row=6, column=0, columnspan=2, sticky="ew")

        ttk.Checkbutton(controls, text="Closed-loop vision playback", variable=self.closed_loop_var).grid(row=7, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Checkbutton(controls, text="Real-time tuning mode", variable=self.realtime_mode_var).grid(row=7, column=4, sticky="w", pady=(6, 0))
        ttk.Label(controls, text="Dependent/object label").grid(row=8, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.object_label_var, width=28).grid(row=9, column=0, columnspan=5, sticky="ew")

        ttk.Label(controls, text="Play step sleep [s]").grid(row=10, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.step_sleep_var, width=8).grid(row=10, column=1, sticky="w")

        ttk.Button(controls, text="Play Open Loop", command=lambda: self._play_sequence(False)).grid(row=11, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="Play Closed Loop", command=lambda: self._play_sequence(True)).grid(row=11, column=2, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="Stop Play", command=self._stop_playback).grid(row=11, column=4, sticky="ew", pady=4)

        ttk.Button(controls, text="Add WP From Trace", command=self._add_waypoint_from_trace).grid(row=12, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(controls, text="Add WP From Robot", command=self._add_waypoint_from_robot).grid(row=12, column=2, columnspan=2, sticky="ew", pady=2)

        row0 = 13
        ttk.Label(controls, text="Edit Pose (slider + text)", font=("Segoe UI", 10, "bold")).grid(row=row0, column=0, columnspan=5, sticky="w", pady=(10, 2))

        for i, field in enumerate(EDIT_FIELDS):
            rr = row0 + 1 + i
            low, high = FIELD_LIMITS[field]
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
        ttk.Label(controls, text="Force %").grid(row=gr_row + 2, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            resolution=1,
            length=220,
            variable=self.gripper_force_pct_var,
            command=lambda _v: self._on_gripper_force_change(),
        ).grid(row=gr_row + 2, column=1, sticky="ew")

        ttk.Label(controls, textvariable=self.info_var, wraplength=420, foreground="#003f5c").grid(row=gr_row + 3, column=0, columnspan=5, sticky="w", pady=(8, 2))

        fig = Figure(figsize=(11, 8), dpi=100)
        self.ax_trace = fig.add_subplot(211, projection="3d")
        self.ax_dist = fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(fig, master=viz)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, sticky="nsew")

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

    def _on_gripper_open_change(self):
        if not self.realtime_mode_var.get():
            return
        open_pct = max(0, min(100, int(round(float(self.gripper_open_pct_var.get())))))
        if self.args.mock_robot and self.robot is not None:
            self.robot.set_gripper_open_pct(open_pct)
            return
        if self.gripper is None:
            return
        # Robotiq command uses millimeters; map percentage to approx 0-85mm stroke.
        pos_mm = (100 - open_pct) * 85 / 100
        try:
            self.gripper.move(int(round(pos_mm)))
        except Exception:
            pass

    def _on_gripper_force_change(self):
        if not self.realtime_mode_var.get():
            return
        force_pct = max(0, min(100, int(round(float(self.gripper_force_pct_var.get())))))
        if self.args.mock_robot and self.robot is not None:
            self.robot.set_gripper_force_pct(force_pct)
            return
        if self.gripper is None:
            return
        try:
            self.gripper.set_force(force_pct)
        except Exception:
            pass

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
            primitive = str(self.primitive_var.get() or self.args.primitive)
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

            wp_idx = int(_try_float(row.get("waypoint_index", idx + 1), idx + 1))
            wp_name = str(row.get("waypoint_name", "")).strip() or f"wp_{wp_idx}"
            dep = self._current_object_label(row)
            self.info_var.set(f"Waypoint {idx + 1}/{len(self.edited_rows)} : index={wp_idx} name={wp_name} dependent={dep or '-'}")
        finally:
            self._building = False

        self._refresh_plots()

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
        label = str(self.object_label_var.get() or "").strip()
        if label:
            return label
        return str(row.get("dependent_item_label", "")).strip()

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

        dx = live_obj[0] - recorded_obj[0]
        dy = live_obj[1] - recorded_obj[1]
        dz = live_obj[2] - recorded_obj[2]

        pose[0] += dx
        pose[1] += dy
        pose[2] += dz
        return pose

    def _execute_current(self):
        if self.robot is None:
            messagebox.showinfo("Dry Run", "No robot connected (dry run or no --robot-ip).")
            return

        try:
            pose = self._build_execution_pose(self.current_idx, closed_loop=self.closed_loop_var.get())
            q = _q_from_row(self.edited_rows[self.current_idx], self.arm_side)
            self._execute_primitive(pose, q)
        except Exception as exc:
            messagebox.showerror("Execution Error", str(exc))

    def _execute_primitive(self, pose, q):
        primitive = self.primitive_var.get().strip()
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
                    pose = self._build_execution_pose(idx, closed_loop=closed_loop)
                    q = _q_from_row(self.edited_rows[idx], self.arm_side)
                    self._execute_primitive(pose, q)
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

        for i, (src, row) in enumerate(zip(self.source_rows, self.edited_rows), start=1):
            dep = self._current_object_label(row)
            obj = _tracked_item_position(row, dep)
            if obj is None:
                continue

            src_xyz = _row_global_xyz(src, self.arm_side)
            row_xyz = _row_global_xyz(row, self.arm_side)

            idxs.append(i)
            nominal.append(_distance3(src_xyz, obj))
            tuned.append(_distance3(row_xyz, obj))

        return idxs, nominal, tuned

    def _refresh_plots(self):
        self.ax_trace.clear()
        self.ax_dist.clear()

        left_xyz = [_row_global_xyz(r, "left") for r in self.edited_rows]
        right_xyz = [_row_global_xyz(r, "right") for r in self.edited_rows]
        lx = [p[0] for p in left_xyz]
        ly = [p[1] for p in left_xyz]
        lz = [p[2] for p in left_xyz]
        rx = [p[0] for p in right_xyz]
        ry = [p[1] for p in right_xyz]
        rz = [p[2] for p in right_xyz]

        self.ax_trace.plot(lx, ly, lz, color="#1f77b4", linewidth=1.4, label="left trace")
        self.ax_trace.plot(rx, ry, rz, color="#ff7f0e", linewidth=1.4, label="right trace")

        ci = self.current_idx
        self.ax_trace.scatter([lx[ci]], [ly[ci]], [lz[ci]], color="#1f77b4", s=65)
        self.ax_trace.scatter([rx[ci]], [ry[ci]], [rz[ci]], color="#ff7f0e", s=65)

        row = self.edited_rows[ci]
        dep = self._current_object_label(row)
        for item in _safe_json_list(row.get("tracked_items_json", "")):
            if not isinstance(item, dict):
                continue
            pos = item.get("position")
            if not isinstance(pos, list) or len(pos) < 3:
                continue
            label = str(item.get("label", "obj"))
            col = "#7f7f7f"
            if label.strip().lower() == dep.strip().lower() and dep:
                col = "#d62728"
            self.ax_trace.scatter([_try_float(pos[0])], [_try_float(pos[1])], [_try_float(pos[2])], color=col, s=40)

        self.ax_trace.set_title("Arm Traces + Current Keypoint + Tracked Objects")
        self.ax_trace.set_xlabel("X [m]")
        self.ax_trace.set_ylabel("Y [m]")
        self.ax_trace.set_zlabel("Z [m]")
        self.ax_trace.legend(loc="upper left")

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

        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for row in self.edited_rows:
                out = dict(row)
                out["tune_source_csv"] = str(src)
                out["tune_saved_at_iso"] = datetime.now().isoformat()
                writer.writerow(out)

        messagebox.showinfo("Saved", f"Tuned CSV written:\n{out_path}")

    def _on_close(self):
        self._stop_play.set()
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
    parser.add_argument("--vision-camera-scan-max-index", type=int, default=6)
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
