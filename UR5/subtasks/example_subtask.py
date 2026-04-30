import csv
import time
import math
from pathlib import Path #filesystem, not robot point paths

from arm import UR5Arm


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
