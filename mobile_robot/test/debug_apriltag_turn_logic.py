#!/usr/bin/env python3
"""Headless AprilTag left/right turn-logic debugger.

Use this over SSH on the Jetson to verify the logic:
- tag left of image center  -> robot should turn right
- tag right of image center -> robot should turn left

By default this script only prints what it sees and what turn it would command.
If ``--serial-port`` is provided, it will also send a simple in-place wheel command
so you can verify the physical turn direction on the robot.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pupil_apriltags as apriltag

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from camera import RealSenseCamera, StreamConfig, bgr_to_gray  # noqa: E402
from serial_connection.serial_con import SerialConnect  # noqa: E402


@dataclass
class SelectedDetection:
    tag_id: int
    center_x_px: float
    center_y_px: float
    tx_m: float
    tz_m: float


def choose_detection(results: list, desired_tag_id: Optional[int]) -> Optional[SelectedDetection]:
    if desired_tag_id is not None:
        for r in results:
            if int(r.tag_id) == desired_tag_id:
                return SelectedDetection(
                    tag_id=int(r.tag_id),
                    center_x_px=float(r.center[0]),
                    center_y_px=float(r.center[1]),
                    tx_m=float(r.pose_t[0, 0]),
                    tz_m=float(r.pose_t[2, 0]),
                )
        return None

    if not results:
        return None

    best = min(results, key=lambda r: float(r.pose_t[2, 0]))
    return SelectedDetection(
        tag_id=int(best.tag_id),
        center_x_px=float(best.center[0]),
        center_y_px=float(best.center[1]),
        tx_m=float(best.pose_t[0, 0]),
        tz_m=float(best.pose_t[2, 0]),
    )


def compute_turn_label(center_error_px: float, deadband_px: float) -> str:
    if abs(center_error_px) <= deadband_px:
        return "CENTERED"
    if center_error_px < 0.0:
        return "TURN RIGHT"
    return "TURN LEFT"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-id", type=int, default=None, help="Specific AprilTag ID to track. Default: closest visible tag.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--tag-size-m", type=float, default=0.09)
    parser.add_argument("--serial-number", type=str, default=os.environ.get("APRILTAG_REALSENSE_SERIAL"))
    parser.add_argument("--loop-hz", type=float, default=5.0)
    parser.add_argument("--deadband-px", type=float, default=20.0, help="Treat tag as centered inside this pixel band.")
    parser.add_argument("--kp", type=float, default=1.2, help="Same sign convention as run_mission.py omega_cmd = -kp * atan2(tx, tz).")
    parser.add_argument("--max-omega", type=float, default=0.18)
    parser.add_argument("--serial-port", type=str, default=None, help="If set, send a simple in-place wheel command to the ESP32.")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--turn-wheel-rate", type=float, default=2.0, help="Wheel command magnitude used with --serial-port.")
    args = parser.parse_args()

    detector = apriltag.Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    camera = RealSenseCamera(
        color_config=StreamConfig(width=args.width, height=args.height, fps=args.fps),
        depth_config=StreamConfig(width=args.width, height=args.height, fps=args.fps),
        align_depth_to_color=True,
        serial_number=args.serial_number,
    )

    serial_conn: Optional[SerialConnect] = None
    if args.serial_port:
        serial_conn = SerialConnect(
            port=args.serial_port,
            baudrate=args.baudrate,
            timeout=0.0,
            tx_rate_hz=max(args.loop_hz, 1.0),
            rx_publish_rate_hz=20.0,
            debug=False,
        )

    period_s = 1.0 / max(args.loop_hz, 1e-3)

    print("Starting AprilTag turn-logic debugger")
    print(f"  tag_id={args.tag_id if args.tag_id is not None else 'closest visible'}")
    print(f"  camera={args.width}x{args.height}@{args.fps}")
    print(f"  serial_number={args.serial_number}")
    print(f"  serial_port={args.serial_port}")
    print("Press Ctrl+C to stop.")

    try:
        camera.open()
        intr = camera.color_intrinsics()
        print(f"Image center cx={intr.cx:.1f}px cy={intr.cy:.1f}px")

        while True:
            t0 = time.monotonic()
            frame = camera.read()
            gray = bgr_to_gray(frame.color)
            results = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[intr.fx, intr.fy, intr.cx, intr.cy],
                tag_size=args.tag_size_m,
            )

            det = choose_detection(results, args.tag_id)
            if det is None:
                print("no matching tag detected")
                if serial_conn is not None:
                    serial_conn.send_wheel_cmd(0.0, 0.0, 0.0, 0.0, force=True)
                time.sleep(max(0.0, period_s - (time.monotonic() - t0)))
                continue

            center_error_px = det.center_x_px - intr.cx
            heading_error_rad = math.atan2(det.tx_m, max(det.tz_m, 1e-6))
            omega_cmd = -args.kp * heading_error_rad
            omega_cmd = float(np.clip(omega_cmd, -args.max_omega, args.max_omega))
            desired_turn = compute_turn_label(center_error_px, args.deadband_px)

            if omega_cmd > 1e-4:
                mission_turn_from_sign = "LEFT (if +omega means CCW on your robot)"
            elif omega_cmd < -1e-4:
                mission_turn_from_sign = "RIGHT (if -omega means CW on your robot)"
            else:
                mission_turn_from_sign = "NONE"

            print(
                f"tag={det.tag_id} "
                f"cx={det.center_x_px:7.1f}px center_err={center_error_px:+7.1f}px "
                f"tx={det.tx_m:+.3f}m tz={det.tz_m:+.3f}m "
                f"heading_err={heading_error_rad:+.3f}rad | "
                f"image_logic={desired_turn} | mission_omega={omega_cmd:+.3f} -> {mission_turn_from_sign}"
            )

            if serial_conn is not None:
                if desired_turn == "TURN RIGHT":
                    left = args.turn_wheel_rate
                    right = -args.turn_wheel_rate
                elif desired_turn == "TURN LEFT":
                    left = -args.turn_wheel_rate
                    right = args.turn_wheel_rate
                else:
                    left = 0.0
                    right = 0.0
                serial_conn.send_wheel_cmd(left, right, left, right, force=True)
                print(f"  sent wheel cmd: left={left:+.2f} right={right:+.2f}")

            dt = time.monotonic() - t0
            time.sleep(max(0.0, period_s - dt))

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if serial_conn is not None:
            try:
                serial_conn.send_wheel_cmd(0.0, 0.0, 0.0, 0.0, force=True)
            except Exception:
                pass
            serial_conn.close()
        camera.close()


if __name__ == "__main__":
    main()
