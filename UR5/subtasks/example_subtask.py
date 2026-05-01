import csv
import time
import math
from pathlib import Path #filesystem, not robot point paths

from arm import UR5Arm

try:
    from robotiq_gripper_control import RobotiqGripper
except Exception:
    RobotiqGripper = None


def _load_joint_trace(csv_path: Path):
    records = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ts = float(row["timestamp"])
                joints = [float(row[f"actual_q_{i}"]) for i in range(6)]
            except (KeyError, ValueError, TypeError):
                continue
            records.append((ts, joints))
    return records


def _estimate_trace_hz(records):
    if len(records) < 2:
        return None

    deltas = []
    prev_ts = records[0][0]
    for ts, _ in records[1:]:
        dt = float(ts) - float(prev_ts)
        if dt > 0:
            deltas.append(dt)
        prev_ts = ts

    if not deltas:
        return None

    # Median is robust to occasional timing jitter in recorded traces.
    deltas.sort()
    mid = len(deltas) // 2
    median_dt = deltas[mid] if len(deltas) % 2 == 1 else 0.5 * (deltas[mid - 1] + deltas[mid])
    if median_dt <= 0:
        return None
    return 1.0 / median_dt


def _decimate_records(records, target_hz):
    source_hz = _estimate_trace_hz(records)
    if source_hz is None or target_hz <= 0:
        return records, 1, source_hz

    if target_hz >= source_hz:
        return records, 1, source_hz

    step = max(1, int(math.floor(source_hz / target_hz)))
    return records[::step], step, source_hz


def _load_named_waypoints(csv_path: Path, task_id: str = "", arm_prefix: str = "right"):
    def _try_float(row, key):
        raw = row.get(key)
        if raw is None:
            return None
        txt = str(raw).strip()
        if txt == "" or txt.lower() == "nothing":
            return None
        try:
            return float(txt)
        except (ValueError, TypeError):
            return None

    def _extract_q_position(row):
        candidate_sets = [
            [f"{arm_prefix}_q_{i}" for i in range(6)],
            [f"q_position_{i}" for i in range(6)],
            [f"actual_q_{i}" for i in range(6)],
            [f"q_{i}" for i in range(6)],
        ]
        for keys in candidate_sets:
            values = [_try_float(row, key) for key in keys]
            if all(v is not None for v in values):
                return values
        return None

    def _try_bool(row, key):
        raw = row.get(key)
        if raw is None:
            return None
        txt = str(raw).strip().lower()
        if txt == "" or txt == "nothing":
            return None
        if txt in {"true", "1", "yes", "y", "open"}:
            return True
        if txt in {"false", "0", "no", "n", "closed", "close"}:
            return False
        return None

    waypoints = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_task_id = str(row.get("task_id", "")).strip()
            if task_id and row_task_id and row_task_id != task_id:
                continue

            try:
                idx = int(float(row.get("waypoint_index", "0")))
            except (ValueError, TypeError):
                continue

            tcp_position = [
                _try_float(row, f"{arm_prefix}_x"),
                _try_float(row, f"{arm_prefix}_y"),
                _try_float(row, f"{arm_prefix}_z"),
                _try_float(row, f"{arm_prefix}_rx"),
                _try_float(row, f"{arm_prefix}_ry"),
                _try_float(row, f"{arm_prefix}_rz"),
            ]
            if not all(v is not None for v in tcp_position):
                tcp_position = None

            q_position = _extract_q_position(row)

            if tcp_position is None and q_position is None:
                continue

            waypoints.append(
                {
                    "index": idx,
                    "name": str(row.get("waypoint_name", "")).strip(),
                    "tcp_position": tcp_position,
                    "q_position": q_position,
                    "gripper_open": _try_bool(row, f"{arm_prefix}_gripper_open"),
                }
            )

    waypoints.sort(key=lambda item: item["index"])
    return waypoints


def register_subtasks(registry):
    """Register simple team-editable subtasks."""

    def _example(supervisor, params):
        print("[example_subtask] params:", params)
        frames = supervisor.compute_task_frames()
        print("[example_subtask] task frames:", frames)
        return True

    registry["example"] = {
        "description": "Minimal collaborative subtask template",
        "runner": _example,
    }

    def _run_named_waypoint_task(
        supervisor,
        params,
        *,
        task_name,
        default_csv,
        default_task_id,
        arm_side,
    ):
        waypoints_csv = Path(params.get("named_waypoints_csv", default_csv))
        task_id = str(params.get("task_id", default_task_id))
        speed = params.get("speed", None)
        acceleration = params.get("acceleration", None)

        if not waypoints_csv.exists():
            raise FileNotFoundError(f"Named waypoints CSV not found: {waypoints_csv}")

        waypoints = _load_named_waypoints(waypoints_csv, task_id=task_id, arm_prefix=arm_side)
        if not waypoints:
            raise RuntimeError(
                f"No valid {arm_side}-arm waypoints found in {waypoints_csv} for task_id={task_id}"
            )

        arm_ip = supervisor.right_ip if arm_side == "right" else supervisor.left_ip
        arm = UR5Arm(arm_ip, verbose=False)
        gripper = None
        gripper_force = int(params.get("gripper_force", 50))
        gripper_speed = int(params.get("gripper_speed", 100))
        gripper_settle_s = float(params.get("gripper_settle_s", 0.15))
        current_gripper_open = None

        if RobotiqGripper is None:
            print(f"[{task_name}] robotiq_gripper_control not available; gripper waypoint safeguards disabled")
        else:
            try:
                gripper = RobotiqGripper(arm.rtde_control)
                gripper.activate()
                gripper.set_force(gripper_force)
                gripper.set_speed(gripper_speed)
                print(f"[{task_name}] Gripper controller ready (force={gripper_force}, speed={gripper_speed})")
            except Exception as exc:
                print(f"[{task_name}] Warning: failed to initialize gripper controller: {exc}")
                gripper = None

        try:
            print(f"[{task_name}] Replaying {len(waypoints)} waypoints from {waypoints_csv}")
            for wp in waypoints:
                target_gripper_open = wp.get("gripper_open", None)

                # Safeguard semantics:
                # - Release when moving away: open before motion.
                if gripper is not None and target_gripper_open is True and current_gripper_open is not True:
                    gripper.open()
                    current_gripper_open = True
                    if gripper_settle_s > 0:
                        time.sleep(gripper_settle_s)

                ok = arm.move_to_joint_position(
                    wp,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=False,
                )
                if not ok:
                    raise RuntimeError(
                        f"Failed at waypoint index={wp['index']} name={wp['name'] or '<unnamed>'}"
                    )

                # - Grasp when arriving: close after motion.
                if gripper is not None and target_gripper_open is False and current_gripper_open is not False:
                    gripper.close()
                    current_gripper_open = False
                    if gripper_settle_s > 0:
                        time.sleep(gripper_settle_s)

            arm.stop_arm(use_linear=False)
            print(f"[{task_name}] Waypoint replay complete")
            return True
        finally:
            arm.disconnect()

    def _register_stub_task(task_name, arm_side="right", default_csv=None):
        """Register a lightweight runner for graph tasks not fully implemented yet.

        Behavior:
        - If a matching waypoint CSV exists (or is supplied via params), replay it.
        - Otherwise, log a no-op stub execution and return success.
        """

        def _runner(supervisor, params):
            params = params or {}
            resolved_default_csv = default_csv or f"UR5/waypoints_{task_name}.csv"
            csv_path = Path(params.get("named_waypoints_csv", resolved_default_csv))

            # Allow explicit task_id override while using task_name as default.
            task_id = str(params.get("task_id", task_name))

            if csv_path.exists():
                print(f"[{task_name}] Using waypoint replay from {csv_path}")
                return _run_named_waypoint_task(
                    supervisor,
                    dict(params, named_waypoints_csv=str(csv_path), task_id=task_id),
                    task_name=task_name,
                    default_csv=str(csv_path),
                    default_task_id=task_id,
                    arm_side=str(params.get("arm_side", arm_side)),
                )

            print(
                f"[{task_name}] STUB runner: no waypoint CSV found at {csv_path}. "
                "Returning success (no-op)."
            )
            return True

        registry[task_name] = {
            "description": f"Stub runner for {task_name}; replays waypoints if CSV exists, else no-op success",
            "runner": _runner,
        }

    # Task names observed in simulator output (UR5/subtasks/Untitled-1.txt).
    # Includes existing acquire/open/close flows so all task runners are list-registered.
    for _task_name, _arm_side, _default_csv in [
        ("acquire_bowl", "right", "UR5/waypoints_acquire_bowl.csv"),
        ("open_microwave_door", "left", "UR5/waypoints_open_microwave_door.csv"),
        ("close_microwave_door", "left", "UR5/waypoints_close_microwave_door.csv"),
        ("place_bowl_in_microwave", "right", "UR5/waypoints_place_bowl_in_microwave.csv"),
        ("right_arm_safe_retract", "right", "UR5/waypoints_right_arm_safe_retract.csv"),
        ("press_microwave_stop", "right", "UR5/waypoints_press_microwave_stop.csv"),
        ("take_bowl_out_to_tray", "right", "UR5/waypoints_take_bowl_out_to_tray.csv"),
        ("acquire_plate", "right", "UR5/waypoints_acquire_plate.csv"),
        ("place_plate_in_microwave", "right", "UR5/waypoints_place_plate_in_microwave.csv"),
        ("take_plate_out_to_tray", "right", "UR5/waypoints_take_plate_out_to_tray.csv"),
        ("acquire_cup", "right", "UR5/waypoints_acquire_cup.csv"),
        ("acquire_bottle", "right", "UR5/waypoints_acquire_bottle.csv"),
        ("pour_drink_into_cup", "right", "UR5/waypoints_pour_drink_into_cup.csv"),
        ("place_cup_on_tray", "right", "UR5/waypoints_place_cup_on_tray.csv"),
    ]:
        if _task_name not in registry:
            _register_stub_task(_task_name, arm_side=_arm_side, default_csv=_default_csv)

    def _total_replay(supervisor, params):
        """Replay full dual-arm joint traces by stepping point-by-point (default 10 Hz)."""
        left_csv = Path(params.get("left_csv", "robot_data_left.csv"))
        right_csv = Path(params.get("right_csv", "robot_data_right.csv"))
        sample_hz = float(params.get("sample_hz", 10.0))
        time_scale = float(params.get("time_scale", 1.0))
        speed = params.get("speed", None)
        acceleration = params.get("acceleration", None)

        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        if time_scale <= 0:
            raise ValueError("time_scale must be > 0")

        left_records = _load_joint_trace(left_csv)
        right_records = _load_joint_trace(right_csv)
        if not left_records or not right_records:
            raise RuntimeError("Both left/right CSV files must contain timestamp + actual_q_* data")

        left_records, left_step, left_src_hz = _decimate_records(left_records, sample_hz)
        right_records, right_step, right_src_hz = _decimate_records(right_records, sample_hz)

        if left_step > 1 or right_step > 1:
            left_src_str = f"{left_src_hz:.2f}" if left_src_hz else "unknown"
            right_src_str = f"{right_src_hz:.2f}" if right_src_hz else "unknown"
            print(
                "[total_replay] Decimated traces for sample_hz "
                f"{sample_hz:.2f}: left step={left_step} (src_hz={left_src_str}), "
                f"right step={right_step} (src_hz={right_src_str})"
            )

        total_points = min(len(left_records), len(right_records))
        if total_points <= 0:
            raise RuntimeError("No replay points found in overlapping left/right trace lengths")

        dt_trace = 1.0 / sample_hz
        dt_wall = dt_trace / time_scale

        left_arm = UR5Arm(supervisor.left_ip, verbose=False)
        right_arm = UR5Arm(supervisor.right_ip, verbose=False)
        try:
            print(
                "[total_replay] Starting replay "
                f"{left_csv} + {right_csv} at {sample_hz:.2f} Hz "
                f"(time_scale={time_scale:.2f}, points={total_points})"
            )

            for idx in range(total_points):
                loop_start = time.time()
                left_joints = left_records[idx][1]
                right_joints = right_records[idx][1]

                ok_l = left_arm.move_to_joint_position(
                    left_joints,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=False,
                )
                ok_r = right_arm.move_to_joint_position(
                    right_joints,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=False,
                )
                if not ok_l or not ok_r:
                    raise RuntimeError(
                        f"Move command failed at index={idx} (left_ok={ok_l}, right_ok={ok_r})"
                    )

                sleep_for = dt_wall - (time.time() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # Ensure final state settles.
            left_arm.stop_arm(use_linear=False)
            right_arm.stop_arm(use_linear=False)
            print("[total_replay] Replay complete")
            return True
        finally:
            left_arm.disconnect()
            right_arm.disconnect()

    registry["total_replay"] = {
        "description": "Replay full left/right CSV joint traces point-by-point at requested sample rate (default 10 Hz)",
        "runner": _total_replay,
    }
