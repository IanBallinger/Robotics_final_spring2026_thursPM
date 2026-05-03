#!/usr/bin/env python3
"""Interactive RealSense HSV mask tuner for tray isolation.

Usage:
    python mobile_robot/test/test_realsense_hsv_mask.py

Controls:
    - Left click on the color image to sample an HSV value from that pixel.
    - Adjust the center HSV value and tolerances with trackbars.
    - The script will isolate the largest tray-colored region and draw its centroid.
    - Press `p` to print the current HSV bounds.
    - Press `r` to reset to a wide-open mask.
    - Press `q` to quit.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

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


WINDOW_COLOR = "RealSense Color"
WINDOW_MASK = "HSV Mask"
WINDOW_MASKED = "Masked Result"
WINDOW_TRAY = "Tray Detection"


@dataclass
class HSVSelection:
    h: int = 90
    s: int = 128
    v: int = 128
    h_tol: int = 15
    s_tol: int = 60
    v_tol: int = 60


class HSVMaskTuner:
    def __init__(self) -> None:
        self.selection = HSVSelection()
        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_hsv: Optional[np.ndarray] = None
        self.latest_mouse_xy: tuple[int, int] = (0, 0)
        self.last_centroid_xy: Optional[tuple[int, int]] = None
        self.last_area_px: float = 0.0
        self.last_point_camera_m: Optional[np.ndarray] = None
        self.last_point_arm_m: Optional[np.ndarray] = None

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
        self.sync_to_trackbars()

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        self.latest_mouse_xy = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN or self.latest_hsv is None or self.latest_bgr is None:
            return
        if y < 0 or x < 0 or y >= self.latest_hsv.shape[0] or x >= self.latest_hsv.shape[1]:
            return

        h, s, v = self.latest_hsv[y, x]
        b, g, r = self.latest_bgr[y, x]
        self.selection.h = int(h)
        self.selection.s = int(s)
        self.selection.v = int(v)
        self.sync_to_trackbars()
        print(
            f"Clicked pixel ({x}, {y}) | "
            f"BGR=({int(b)}, {int(g)}, {int(r)}) | "
            f"HSV=({int(h)}, {int(s)}, {int(v)})"
        )

    def current_bounds(self) -> HSVBounds:
        self.sync_from_trackbars()

        h = int(self.selection.h)
        s = int(self.selection.s)
        v = int(self.selection.v)
        h_tol = int(self.selection.h_tol)
        s_tol = int(self.selection.s_tol)
        v_tol = int(self.selection.v_tol)

        s_min = max(0, s - s_tol)
        s_max = min(255, s + s_tol)
        v_min = max(0, v - v_tol)
        v_max = min(255, v + v_tol)

        h_lo = h - h_tol
        h_hi = h + h_tol
        if h_lo < 0:
            h_lo += 180
        if h_hi > 179:
            h_hi -= 180

        return HSVBounds(
            lower=(int(h_lo), int(s_min), int(v_min)),
            upper=(int(h_hi), int(s_max), int(v_max)),
        )

    def annotate(self, image: np.ndarray) -> np.ndarray:
        if self.latest_hsv is None:
            return image

        x, y = self.latest_mouse_xy
        x = max(0, min(image.shape[1] - 1, x))
        y = max(0, min(image.shape[0] - 1, y))
        h, s, v = self.latest_hsv[y, x]
        b, g, r = image[y, x]

        annotated = image.copy()
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
        status = "tray not found"
        status_color = (0, 165, 255)
        if self.last_centroid_xy is not None:
            status = f"tray centroid={self.last_centroid_xy} area={self.last_area_px:.0f}"
            status_color = (0, 255, 0)
        cv2.putText(
            annotated,
            status,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
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
        cv2.putText(
            annotated,
            "left click to sample HSV | p=print | r=reset | q=quit",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def print_current_range(self) -> None:
        h = self.selection.h
        s = self.selection.s
        v = self.selection.v
        ht = self.selection.h_tol
        st = self.selection.s_tol
        vt = self.selection.v_tol
        bounds = self.current_bounds()
        print(
            "Current HSV selection:\n"
            f"  center: H={h}, S={s}, V={v}\n"
            f"  tolerance: dH={ht}, dS={st}, dV={vt}\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--min-area", type=float, default=1000.0)
    parser.add_argument("--open-kernel", type=int, default=5)
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument(
        "--camera-config",
        default=os.path.join(REPO_ROOT, "mobile_robot", "config", "camera_config.yaml"),
        help="Path to config file containing camera_to_arm extrinsics.",
    )
    parser.add_argument("--depth-window-radius", type=int, default=2)
    args = parser.parse_args()

    tuner = HSVMaskTuner()
    tuner.create_ui()
    camera_extrinsics = load_camera_to_arm_extrinsics(args.camera_config)

    camera = RealSenseCamera(
        color_config=StreamConfig(width=args.width, height=args.height, fps=args.fps),
        depth_config=StreamConfig(width=args.width, height=args.height, fps=args.fps),
        align_depth_to_color=True,
    )

    try:
        camera.open()
        print("RealSense HSV tuner running")
        print("Left click on the tray color to sample HSV and isolate it.")
        print("Press 'p' to print the current range, 'r' to reset, 'q' to quit.")

        while True:
            frame = camera.read()
            bgr = frame.color
            tuner.latest_bgr = bgr
            tuner.latest_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            bounds = tuner.current_bounds()
            result = isolate_tray_and_find_centroid(
                bgr,
                bounds,
                min_area_px=args.min_area,
                open_kernel_size=args.open_kernel,
                close_kernel_size=args.close_kernel,
            )
            tuner.last_centroid_xy = result.centroid_xy
            tuner.last_area_px = result.area_px
            tuner.last_point_camera_m = None
            tuner.last_point_arm_m = None

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
                    tuner.last_point_camera_m = point_camera
                    tuner.last_point_arm_m = point_arm
                    print(
                        f"tray centroid px={result.centroid_xy} area_px={result.area_px:.0f} "
                        f"camera_xyz=({point_camera[0]:+.3f}, {point_camera[1]:+.3f}, {point_camera[2]:+.3f}) m "
                        f"arm_xyz=({point_arm[0]:+.3f}, {point_arm[1]:+.3f}, {point_arm[2]:+.3f}) m"
                    )

            annotated = tuner.annotate(bgr)

            cv2.imshow(WINDOW_COLOR, annotated)
            cv2.imshow(WINDOW_MASK, result.mask)
            cv2.imshow(WINDOW_MASKED, result.masked_color)
            cv2.imshow(WINDOW_TRAY, result.annotated_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                tuner.print_current_range()
            if key == ord("r"):
                tuner.reset()

    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
