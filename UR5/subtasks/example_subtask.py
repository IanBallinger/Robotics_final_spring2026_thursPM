import csv
import time
from pathlib import Path #filesystem, not robot point paths

from arm import UR5Arm


def _load_pose_trace(csv_path: Path):
    records = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ts = float(row["timestamp"])
                pose = [float(row[f"actual_TCP_pose_{i}"]) for i in range(6)]
            except (KeyError, ValueError, TypeError):
                continue
            records.append((ts, pose))
    return records


def _nearest_pose(records, idx, target_ts):
    # Advance index monotonically because replay time is monotonic.
    while idx + 1 < len(records) and records[idx + 1][0] <= target_ts:
        idx += 1

    if idx + 1 < len(records):
        before_dt = abs(records[idx][0] - target_ts)
        after_dt = abs(records[idx + 1][0] - target_ts)
        if after_dt < before_dt:
            idx += 1

    return records[idx][1], idx


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
        """Replay full dual-arm traces at synchronized timestamps (default 10 Hz)."""
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

        left_records = _load_pose_trace(left_csv)
        right_records = _load_pose_trace(right_csv)
        if not left_records or not right_records:
            raise RuntimeError("Both left/right CSV files must contain timestamp + actual_TCP_pose_* data")

        start_ts = max(left_records[0][0], right_records[0][0])
        end_ts = min(left_records[-1][0], right_records[-1][0])
        if end_ts <= start_ts:
            raise RuntimeError("No overlapping timestamp interval between left/right traces")

        dt_trace = 1.0 / sample_hz
        dt_wall = dt_trace / time_scale

        left_arm = UR5Arm(supervisor.left_ip, verbose=False)
        right_arm = UR5Arm(supervisor.right_ip, verbose=False)
        try:
            left_idx = 0
            right_idx = 0
            target_ts = start_ts

            print(
                "[total_replay] Starting replay "
                f"{left_csv} + {right_csv} at {sample_hz:.2f} Hz "
                f"(time_scale={time_scale:.2f})"
            )

            wall_next = time.time()
            while target_ts <= end_ts:
                left_pose, left_idx = _nearest_pose(left_records, left_idx, target_ts)
                right_pose, right_idx = _nearest_pose(right_records, right_idx, target_ts)

                ok_l = left_arm.move_linear_to_pose(
                    left_pose,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=True,
                )
                ok_r = right_arm.move_linear_to_pose(
                    right_pose,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=True,
                )
                if not ok_l or not ok_r:
                    raise RuntimeError(
                        f"Move command failed at t={target_ts:.4f} (left_ok={ok_l}, right_ok={ok_r})"
                    )

                wall_next += dt_wall
                sleep_for = wall_next - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                target_ts += dt_trace

            # Ensure final state settles.
            left_arm.stop_arm(use_linear=True)
            right_arm.stop_arm(use_linear=True)
            print("[total_replay] Replay complete")
            return True
        finally:
            left_arm.disconnect()
            right_arm.disconnect()

    registry["total_replay"] = {
        "description": "Replay full left/right CSV traces in synchronized timestamp order (default 10 Hz)",
        "runner": _total_replay,
    }
