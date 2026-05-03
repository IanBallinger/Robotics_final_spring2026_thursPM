#!/usr/bin/env python3
"""Simple host-side elevator serial test.

Periodically alternates elevator height commands between 0.4 m and 0.7 m,
which the elevator ESP32 should receive as:

    ELV_CMD,0.4
    ELV_CMD,0.7

It also prints any parsed elevator measurements, raw serial lines, and
`ELV_CTRL_EFF,...` control-effort debug messages returned by the ESP32.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import serial

from serial_connection.elevator_serialization import serialize_elevator_cmd  # noqa: E402
from serial_connection.serialization import serialize_arm_cmd  # noqa: E402


ARM_BASE_X_M = 0.0
ARM_BASE_Y_M = 0.0
ARM_LINK_1_M = 0.18
ARM_LINK_2_M = 0.18
RAD_TO_DEG = 180.0 / math.pi


def inverse_kinematics_python(
    x_e: float,
    y_e: float,
    x_o: float = ARM_BASE_X_M,
    y_o: float = ARM_BASE_Y_M,
    l1: float = ARM_LINK_1_M,
    l2: float = ARM_LINK_2_M,
) -> tuple[float, float] | None:
    dx = x_e - x_o
    dy = y_e - y_o
    r2 = dx * dx + dy * dy
    r = math.sqrt(r2)
    if r <= 0.0 or r > l1 + l2 or r < abs(l1 - l2):
        return None

    cos_theta2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    sin_theta2 = math.sqrt(max(0.0, 1.0 - cos_theta2 * cos_theta2))
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta1 = math.atan2(dy, dx) - math.atan2(l2 * sin_theta2, l1 + l2 * cos_theta2)
    return theta1, theta2


def forward_kinematics_python(
    theta1: float,
    theta2: float,
    x_o: float = ARM_BASE_X_M,
    y_o: float = ARM_BASE_Y_M,
    l1: float = ARM_LINK_1_M,
    l2: float = ARM_LINK_2_M,
) -> tuple[float, float]:
    x = x_o + l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    y = y_o + l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    return x, y


def validate_arm_ik_point(x_e: float, y_e: float, tolerance_m: float = 1e-6) -> bool:
    solution = inverse_kinematics_python(x_e, y_e)
    if solution is None:
        print(f"ARM IK unreachable target: ({x_e:.4f}, {y_e:.4f})")
        return False

    theta1, theta2 = solution
    x_fk, y_fk = forward_kinematics_python(theta1, theta2)
    err = math.hypot(x_fk - x_e, y_fk - y_e)
    print(
        "ARM IK",
        f"target=({x_e:.4f}, {y_e:.4f})",
        f"theta1_deg={theta1 * RAD_TO_DEG:.2f}",
        f"theta2_deg={theta2 * RAD_TO_DEG:.2f}",
        f"fk=({x_fk:.4f}, {y_fk:.4f})",
        f"err={err:.8f} m",
    )
    return err <= tolerance_m



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=os.environ.get("ELEVATOR_SERIAL_PORT", "/dev/ttyESP_ELV"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--rate-hz", type=float, default=5.0, help="Setpoint send rate")
    parser.add_argument(
        "--switch-period-s",
        type=float,
        default=4.0,
        help="Seconds before toggling between 0.4 m and 0.7 m",
    )
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means run forever")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--arm-test",
        action="store_true",
        help="Also test arm IK and send ARM_CMD targets to the elevator controller.",
    )
    parser.add_argument(
        "--arm-switch-period-s",
        type=float,
        default=4.0,
        help="Seconds before toggling between arm target points.",
    )
    args = parser.parse_args()

    setpoints_m = (0.6,0.6)
    setpoint_index = 0
    current_setpoint_m = setpoints_m[setpoint_index]
    last_switch_t = time.monotonic()
    start_t = last_switch_t
    tx_period = 0.0 if args.rate_hz <= 0.0 else 1.0 / args.rate_hz
    next_tx_t = start_t

    # arm_targets_m = ((0.20, 0.10), (0.12, 0.18))
    arm_targets_m = ((0.0, 0.0), (np.pi/2, np.pi/2))
    arm_target_index = 0
    current_arm_target = arm_targets_m[arm_target_index]
    last_arm_switch_t = start_t

    # if args.arm_test:
    #     print("Validating arm IK test points before opening serial...")
    #     for x_e, y_e in arm_targets_m:
    #         if not validate_arm_ik_point(x_e, y_e):
    #             raise SystemExit("arm IK validation failed for one or more targets")

    with serial.Serial(args.port, args.baudrate, timeout=0.01) as ser:
        print(f"Sending elevator setpoints on {args.port} at {args.rate_hz:.2f} Hz")
        print(f"Alternating between {setpoints_m[0]:.3f} m and {setpoints_m[1]:.3f} m every {args.switch_period_s:.2f} s")
        if args.arm_test:
            print(
                f"Alternating arm targets between {arm_targets_m[0]} m and {arm_targets_m[1]} m every {args.arm_switch_period_s:.2f} s"
            )

        while True:
            now = time.monotonic()
            elapsed = now - start_t
            if args.duration_s > 0.0 and elapsed >= args.duration_s:
                break

            # if now - last_switch_t >= args.switch_period_s:
            #     setpoint_index = (setpoint_index + 1) % len(setpoints_m)
            #     current_setpoint_m = setpoints_m[setpoint_index]
            #     last_switch_t = now
            #     print(f"Switched setpoint -> {current_setpoint_m:.3f} m")

            if args.arm_test and now - last_arm_switch_t >= args.arm_switch_period_s:
                arm_target_index = (arm_target_index + 1) % len(arm_targets_m)
                current_arm_target = arm_targets_m[arm_target_index]
                last_arm_switch_t = now
                print(f"Switched arm target -> {current_arm_target}")

            if now >= next_tx_t:
                # payload = serialize_elevator_cmd(current_setpoint_m)
                # ser.write(payload.encode("ascii"))
                # print(f"TX {payload.strip()}")

                if args.arm_test:
                    arm_payload = serialize_arm_cmd(*current_arm_target)
                    ser.write(arm_payload.encode("ascii"))
                    print(f"TX {arm_payload.strip()}")

                ser.flush()
                next_tx_t = now + max(tx_period, 0.01)

            while ser.in_waiting > 0:
                raw = ser.readline()
                if not raw:
                    break
                line = raw.decode("ascii", errors="replace").strip()
                if line:
                    print(f"RX {line}")

            time.sleep(0.01)


if __name__ == "__main__":
    main()
