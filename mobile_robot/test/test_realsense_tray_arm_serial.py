#!/usr/bin/env python3
"""Interactive tray-to-arm serial command test.

This combines:
- HSV tray selection from the RealSense color image
- tray centroid estimation with depth
- camera-frame -> arm-frame transform
- streaming planar end-effector commands over serial as ``ARM_CMD,x,y``

Typical use:
    python mobile_robot/test/test_realsense_tray_arm_serial.py --port /dev/ttyESP_ELV

Workflow:
- Left click on the tray color in the live image.
- Tune HSV tolerances if needed.
- The script finds the tray centroid, converts it to arm coordinates,
  projects that 3D point onto a chosen arm plane, and sends the resulting
  planar target over serial.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import serial

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from camera import (  # noqa: E402
    HSVBounds,
    RealSenseCamera,
    StreamConfig,
    camera_to_arm_point,
    depth_pixel_to_camera_point,
    depth_to_meters,
    isolate_tray_and_find_centroid,
    load_camera_to_arm_extrinsics,
)
from control_center.arm_visualizer import (  # noqa: E402
    ArmGeometry,
    EndEffectorTarget,
    create_two_link_arm_figure,
    inverse_kinematics,
    parse_arm_ack_line,
)
from serial_connection.elevator_serialization import parse_elevator_line  # noqa: E402
from serial_connection.serialization import serialize_arm_cmd  # noqa: E402

WINDOW_COLOR = "RealSense Color"
WINDOW_MASK = "HSV Mask"
WINDOW_MASKED = "Masked Result"
WINDOW_TRAY = "Tray Detection"
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class HSVSelection:
    h: int = 90
    s: int = 128
    v: int = 128
    h_tol: int = 15
    s_tol: int = 60
    v_tol: int = 60


class TrayArmCommanderUI:
    def __init__(self) -> None:
        self.selection = HSVSelection()
        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_hsv: Optional[np.ndarray] = None
        self.latest_mouse_xy: tuple[int, int] = (0, 0)
        self.has_user_selection = False
        self.last_centroid_xy: Optional[tuple[int, int]] = None
        self.last_area_px: float = 0.0
        self.last_point_camera_m: Optional[np.ndarray] = None
        self.last_point_arm_m: Optional[np.ndarray] = None
        self.last_planar_target_m: Optional[tuple[float, float]] = None
        self.last_tx_payload: Optional[str] = None
        self.last_status: str = "click tray color to arm"

    def create_ui(self) -> None:
        cv2.namedWindow(WINDOW_COLOR, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_MASKED, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_TRAY, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_COLOR, 960, 540)
        cv2.resizeWindow(WINDOW_MASK, 480, 360)
        cv2.resizeWindow(WINDOW_MASKED, 960, 540)
        cv2.resizeWindow(WINDOW_TRAY, 960, 540)

        cv2.createTrackbar("H", WINDOW_MASK, self.selection.h, 179, lambda _v: None)
        cv2.createTrackbar("S", WINDOW_MASK, self.selection.s, 255, lambda _v: None)
        cv2.createTrackbar("V", WINDOW_MASK, self.selection.v, 255, lambda _v: None)
        cv2.createTrackbar("H tol", WINDOW_MASK, self.selection.h_tol, 90, lambda _v: None)
        cv2.createTrackbar("S tol", WINDOW_MASK, self.selection.s_tol, 255, lambda _v: None)
        cv2.createTrackbar("V tol", WINDOW_MASK, self.selection.v_tol, 255, lambda _v: None)
        cv2.setMouseCallback(WINDOW_COLOR, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        self.latest_mouse_xy = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN or self.latest_hsv is None or self.latest_bgr is None:
            return
        if not (0 <= x < self.latest_hsv.shape[1] and 0 <= y < self.latest_hsv.shape[0]):
            return
        h, s, v = self.latest_hsv[y, x]
        b, g, r = self.latest_bgr[y, x]
        self.selection.h = int(h)
        self.selection.s = int(s)
        self.selection.v = int(v)
        self.has_user_selection = True
        self.sync_to_trackbars()
        print(
            f"Clicked pixel ({x}, {y}) | BGR=({int(b)}, {int(g)}, {int(r)}) | "
            f"HSV=({int(h)}, {int(s)}, {int(v)})"
        )

    def sync_from_trackbars(self) -> None:
        self.selection.h = cv2.getTrackbarPos("H", WINDOW_MASK)
        self.selection.s = cv2.getTrackbarPos("S", WINDOW_MASK)
        self.selection.v = cv2.getTrackbarPos("V", WINDOW_MASK)
        self.selection.h_tol = cv2.getTrackbarPos("H tol", WINDOW_MASK)
        self.selection.s_tol = cv2.getTrackbarPos("S tol", WINDOW_MASK)
        self.selection.v_tol = cv2.getTrackbarPos("V tol", WINDOW_MASK)

    def sync_to_trackbars(self) -> None:
        cv2.setTrackbarPos("H", WINDOW_MASK, int(self.selection.h))
        cv2.setTrackbarPos("S", WINDOW_MASK, int(self.selection.s))
        cv2.setTrackbarPos("V", WINDOW_MASK, int(self.selection.v))
        cv2.setTrackbarPos("H tol", WINDOW_MASK, int(self.selection.h_tol))
        cv2.setTrackbarPos("S tol", WINDOW_MASK, int(self.selection.s_tol))
        cv2.setTrackbarPos("V tol", WINDOW_MASK, int(self.selection.v_tol))

    def reset(self) -> None:
        self.selection = HSVSelection(h=90, s=128, v=128, h_tol=90, s_tol=255, v_tol=255)
        self.has_user_selection = False
        self.last_status = "reset; click tray color to arm"
        self.sync_to_trackbars()

    def current_bounds(self) -> HSVBounds:
        self.sync_from_trackbars()
        h, s, v = int(self.selection.h), int(self.selection.s), int(self.selection.v)
        ht, st, vt = int(self.selection.h_tol), int(self.selection.s_tol), int(self.selection.v_tol)
        s_min, s_max = max(0, s - st), min(255, s + st)
        v_min, v_max = max(0, v - vt), min(255, v + vt)
        h_lo, h_hi = h - ht, h + ht
        if h_lo < 0:
            h_lo += 180
        if h_hi > 179:
            h_hi -= 180
        return HSVBounds(lower=(h_lo, s_min, v_min), upper=(h_hi, s_max, v_max))

    def annotate(self, image: np.ndarray) -> np.ndarray:
        annotated = image.copy()
        if self.latest_hsv is not None:
            x, y = self.latest_mouse_xy
            x = max(0, min(image.shape[1] - 1, x))
            y = max(0, min(image.shape[0] - 1, y))
            h, s, v = self.latest_hsv[y, x]
            b, g, r = image[y, x]
            cv2.circle(annotated, (x, y), 5, (255, 255, 255), 1)
            cv2.putText(
                annotated,
                f"cursor ({x},{y}) BGR=({int(b)},{int(g)},{int(r)}) HSV=({int(h)},{int(s)},{int(v)})",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            (
                f"mask center=({self.selection.h},{self.selection.s},{self.selection.v}) "
                f"tol=(±{self.selection.h_tol},±{self.selection.s_tol},±{self.selection.v_tol})"
            ),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            self.last_status,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if self.last_centroid_xy is not None else (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        if self.last_point_camera_m is not None:
            p = self.last_point_camera_m
            cv2.putText(
                annotated,
                f"camera xyz=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if self.last_point_arm_m is not None:
            p = self.last_point_arm_m
            cv2.putText(
                annotated,
                f"arm xyz=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m",
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if self.last_planar_target_m is not None:
            tx, ty = self.last_planar_target_m
            cv2.putText(
                annotated,
                f"arm target xy=({tx:+.3f}, {ty:+.3f}) m",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if self.last_tx_payload is not None:
            cv2.putText(
                annotated,
                f"last tx: {self.last_tx_payload.strip()}",
                (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            "left click=pick tray color | p=print bounds | r=reset | q=quit",
            (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def print_current_range(self) -> None:
        bounds = self.current_bounds()
        print(
            "Current HSV selection:\n"
            f"  center: H={self.selection.h}, S={self.selection.s}, V={self.selection.v}\n"
            f"  tolerance: dH={self.selection.h_tol}, dS={self.selection.s_tol}, dV={self.selection.v_tol}\n"
            f"  bounds: lower={bounds.lower}, upper={bounds.upper}"
        )


def estimate_depth_at_pixel(depth_m: np.ndarray, x: int, y: int, window_radius: int = 2) -> Optional[float]:
    y1 = max(0, y - window_radius)
    y2 = min(depth_m.shape[0], y + window_radius + 1)
    x1 = max(0, x - window_radius)
    x2 = min(depth_m.shape[1], x + window_radius + 1)
    crop = depth_m[y1:y2, x1:x2]
    valid = crop[crop > 0.0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def extract_planar_target(point_arm: np.ndarray, axis_a: str, axis_b: str) -> tuple[float, float]:
    return float(point_arm[AXIS_TO_INDEX[axis_a]]), float(point_arm[AXIS_TO_INDEX[axis_b]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=os.environ.get("ELEVATOR_SERIAL_PORT", "/dev/ttyESP_ELV"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rate-hz", type=float, default=5.0, help="ARM_CMD send rate")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means run forever")
    parser.add_argument("--min-area", type=float, default=1000.0)
    parser.add_argument("--open-kernel", type=int, default=5)
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument(
        "--camera-config",
        default=os.path.join(REPO_ROOT, "mobile_robot", "config", "camera_config.yaml"),
    )
    parser.add_argument("--depth-window-radius", type=int, default=2)
    parser.add_argument("--arm-x-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--arm-y-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--visualize-arm", action="store_true")
    parser.add_argument("--link-1-m", type=float, default=0.18)
    parser.add_argument("--link-2-m", type=float, default=0.18)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.arm_x_axis == args.arm_y_axis:
        raise SystemExit("--arm-x-axis and --arm-y-axis must be different")

    ui = TrayArmCommanderUI()
    ui.create_ui()
    camera_extrinsics = load_camera_to_arm_extrinsics(args.camera_config)
    camera = RealSenseCamera(
        color_config=StreamConfig(width=args.width, height=args.height, fps=args.fps),
        depth_config=StreamConfig(width=args.width, height=args.height, fps=args.fps),
        align_depth_to_color=True,
    )

    geometry = ArmGeometry(link_1_m=args.link_1_m, link_2_m=args.link_2_m)
    plotter = None
    plt = None
    fig = None
    latest_arm_ack = None
    latest_elevator_height_m = None
    latest_raw_line = None
    if args.visualize_arm:
        plt, fig, _, plotter = create_two_link_arm_figure(
            geometry,
            title=f"Tray arm serial ({args.arm_x_axis},{args.arm_y_axis}) plane",
        )
        plt.ion()
        plt.show(block=False)

    start_t = time.monotonic()
    tx_period = 0.0 if args.rate_hz <= 0.0 else 1.0 / args.rate_hz
    next_tx_t = start_t
    current_target_xy: Optional[tuple[float, float]] = None

    try:
        camera.open()
        with serial.Serial(args.port, args.baudrate, timeout=0.01) as ser:
            print(f"Streaming tray targets to {args.port} at {args.rate_hz:.2f} Hz")
            print("Left click the tray color to start tracking and sending ARM_CMD targets.")

            while True:
                now = time.monotonic()
                if args.duration_s > 0.0 and (now - start_t) >= args.duration_s:
                    break
                if args.visualize_arm and plt is not None and fig is not None and not plt.fignum_exists(fig.number):
                    break

                frame = camera.read()
                bgr = frame.color
                ui.latest_bgr = bgr
                ui.latest_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                ui.last_centroid_xy = None
                ui.last_area_px = 0.0
                ui.last_point_camera_m = None
                ui.last_point_arm_m = None
                ui.last_planar_target_m = None
                current_target_xy = None

                result = None
                if ui.has_user_selection:
                    bounds = ui.current_bounds()
                    result = isolate_tray_and_find_centroid(
                        bgr,
                        bounds,
                        min_area_px=args.min_area,
                        open_kernel_size=args.open_kernel,
                        close_kernel_size=args.close_kernel,
                    )
                    ui.last_centroid_xy = result.centroid_xy
                    ui.last_area_px = result.area_px

                    if result.centroid_xy is not None and frame.depth is not None and frame.intrinsics is not None:
                        depth_m = depth_to_meters(frame.depth, frame.depth_scale)
                        cx, cy = result.centroid_xy
                        depth_value = estimate_depth_at_pixel(
                            depth_m,
                            cx,
                            cy,
                            window_radius=args.depth_window_radius,
                        )
                        if depth_value is not None:
                            point_camera = depth_pixel_to_camera_point(cx, cy, depth_value, frame.intrinsics)
                            point_arm = camera_to_arm_point(point_camera, camera_extrinsics)
                            target_xy = extract_planar_target(point_arm, args.arm_x_axis, args.arm_y_axis)
                            ui.last_point_camera_m = point_camera
                            ui.last_point_arm_m = point_arm
                            ui.last_planar_target_m = target_xy
                            current_target_xy = target_xy
                            ui.last_status = (
                                f"tray centroid={result.centroid_xy} area={result.area_px:.0f} "
                                f"target=({target_xy[0]:+.3f}, {target_xy[1]:+.3f})"
                            )
                        else:
                            ui.last_status = "tray found but no valid depth at centroid"
                    else:
                        ui.last_status = "tray not found"
                else:
                    ui.last_status = "click tray color to arm"

                if current_target_xy is not None:
                    ik = inverse_kinematics(EndEffectorTarget(*current_target_xy), geometry)
                    if ik is None:
                        ui.last_status = (
                            f"unreachable target=({current_target_xy[0]:+.3f}, {current_target_xy[1]:+.3f})"
                        )
                    elif now >= next_tx_t:
                        payload = serialize_arm_cmd(*current_target_xy)
                        ser.write(payload.encode("ascii"))
                        ser.flush()
                        ui.last_tx_payload = payload
                        next_tx_t = now + max(tx_period, 0.01)
                        if args.debug:
                            print(
                                f"TX {payload.strip()} | camera={np.round(ui.last_point_camera_m, 4).tolist()} "
                                f"arm={np.round(ui.last_point_arm_m, 4).tolist()}"
                            )
                elif now >= next_tx_t:
                    next_tx_t = now + max(tx_period, 0.01)

                while ser.in_waiting > 0:
                    raw = ser.readline()
                    if not raw:
                        break
                    line = raw.decode("ascii", errors="replace").strip()
                    if not line:
                        continue
                    latest_raw_line = line
                    if args.debug:
                        print(f"RX {line}")
                    arm_ack = parse_arm_ack_line(line)
                    if arm_ack is not None:
                        latest_arm_ack = arm_ack
                        continue
                    elv = parse_elevator_line(line)
                    if elv is not None:
                        latest_elevator_height_m = float(elv.height_m)

                annotated = ui.annotate(bgr)
                cv2.imshow(WINDOW_COLOR, annotated)
                if result is not None:
                    cv2.imshow(WINDOW_MASK, result.mask)
                    cv2.imshow(WINDOW_MASKED, result.masked_color)
                    cv2.imshow(WINDOW_TRAY, result.annotated_color)
                else:
                    blank = np.zeros(bgr.shape[:2], dtype=np.uint8)
                    cv2.imshow(WINDOW_MASK, blank)
                    cv2.imshow(WINDOW_MASKED, np.zeros_like(bgr))
                    cv2.imshow(WINDOW_TRAY, bgr)

                if plotter is not None:
                    cmd_a = current_target_xy[0] if current_target_xy is not None else None
                    cmd_b = current_target_xy[1] if current_target_xy is not None else None
                    plotter.update(
                        command_space="xy",
                        target_cmd_a=cmd_a,
                        target_cmd_b=cmd_b,
                        latest_ack=latest_arm_ack,
                        latest_height_m=latest_elevator_height_m,
                        latest_raw_line=latest_raw_line,
                    )
                    plt.pause(0.001)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    ui.print_current_range()
                if key == ord("r"):
                    ui.reset()

                time.sleep(0.005)
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
