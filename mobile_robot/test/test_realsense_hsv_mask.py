#!/usr/bin/env python3
"""Interactive RealSense HSV mask tuner for tray isolation.

Usage:
    python mobile_robot/test/test_realsense_hsv_mask.py

Controls:
    - Left click on the color image to sample an HSV value from that pixel.
    - Adjust the center HSV value and tolerances with trackbars.
    - Press `p` to print the current HSV center/range.
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

from camera import RealSenseCamera, StreamConfig  # noqa: E402


WINDOW_COLOR = "RealSense Color"
WINDOW_MASK = "HSV Mask"
WINDOW_MASKED = "Masked Result"


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

    def create_ui(self) -> None:
        cv2.namedWindow(WINDOW_COLOR, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_MASKED, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_COLOR, 960, 540)
        cv2.resizeWindow(WINDOW_MASK, 480, 360)
        cv2.resizeWindow(WINDOW_MASKED, 960, 540)

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
        if event != cv2.EVENT_LBUTTONDOWN or self.latest_hsv is None:
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

    def build_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        self.latest_hsv = hsv_image
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

        if 0 <= h_lo and h_hi <= 179:
            lower = np.array([h_lo, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_hi, s_max, v_max], dtype=np.uint8)
            return cv2.inRange(hsv_image, lower, upper)

        if h_lo < 0:
            lower_a = np.array([0, s_min, v_min], dtype=np.uint8)
            upper_a = np.array([h_hi, s_max, v_max], dtype=np.uint8)
            lower_b = np.array([180 + h_lo, s_min, v_min], dtype=np.uint8)
            upper_b = np.array([179, s_max, v_max], dtype=np.uint8)
            return cv2.bitwise_or(
                cv2.inRange(hsv_image, lower_a, upper_a),
                cv2.inRange(hsv_image, lower_b, upper_b),
            )

        lower_a = np.array([h_lo, s_min, v_min], dtype=np.uint8)
        upper_a = np.array([179, s_max, v_max], dtype=np.uint8)
        lower_b = np.array([0, s_min, v_min], dtype=np.uint8)
        upper_b = np.array([h_hi - 179, s_max, v_max], dtype=np.uint8)
        return cv2.bitwise_or(
            cv2.inRange(hsv_image, lower_a, upper_a),
            cv2.inRange(hsv_image, lower_b, upper_b),
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
        cv2.putText(
            annotated,
            "left click to sample HSV | p=print | r=reset | q=quit",
            (10, 75),
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
        print(
            "Current HSV selection:\n"
            f"  center: H={h}, S={s}, V={v}\n"
            f"  tolerance: dH={ht}, dS={st}, dV={vt}\n"
            f"  naive bounds: H=[{h-ht}, {h+ht}], S=[{s-st}, {s+st}], V=[{v-vt}, {v+vt}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    tuner = HSVMaskTuner()
    tuner.create_ui()

    camera = RealSenseCamera(
        color_config=StreamConfig(width=args.width, height=args.height, fps=args.fps),
        depth_config=StreamConfig(width=args.width, height=args.height, fps=args.fps),
        align_depth_to_color=True,
    )

    try:
        camera.open()
        print("RealSense HSV tuner running")
        print("Left click on the color image to sample HSV from a pixel.")
        print("Press 'p' to print the current range, 'r' to reset, 'q' to quit.")

        while True:
            frame = camera.read()
            bgr = frame.color
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            tuner.latest_bgr = bgr

            mask = tuner.build_mask(hsv)
            masked = cv2.bitwise_and(bgr, bgr, mask=mask)
            annotated = tuner.annotate(bgr)

            cv2.imshow(WINDOW_COLOR, annotated)
            cv2.imshow(WINDOW_MASK, mask)
            cv2.imshow(WINDOW_MASKED, masked)

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
