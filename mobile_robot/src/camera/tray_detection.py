from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVBounds:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass
class TrayDetectionResult:
    mask: np.ndarray
    masked_color: np.ndarray
    annotated_color: np.ndarray
    contour: Optional[np.ndarray]
    centroid_xy: Optional[tuple[int, int]]
    area_px: float


def hsv_mask(image_bgr: np.ndarray, bounds: HSVBounds) -> np.ndarray:
    """Return a binary mask for an HSV range.

    Supports hue wrap-around, e.g. lower hue > upper hue for red-like colors.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.asarray(bounds.lower, dtype=np.uint8)
    upper = np.asarray(bounds.upper, dtype=np.uint8)

    h_lo, s_lo, v_lo = [int(v) for v in lower]
    h_hi, s_hi, v_hi = [int(v) for v in upper]

    if h_lo <= h_hi:
        return cv2.inRange(hsv, lower, upper)

    lower_a = np.array([0, s_lo, v_lo], dtype=np.uint8)
    upper_a = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    lower_b = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper_b = np.array([179, s_hi, v_hi], dtype=np.uint8)
    return cv2.bitwise_or(
        cv2.inRange(hsv, lower_a, upper_a),
        cv2.inRange(hsv, lower_b, upper_b),
    )


def isolate_tray_and_find_centroid(
    image_bgr: np.ndarray,
    bounds: HSVBounds,
    *,
    min_area_px: float = 1000.0,
    open_kernel_size: int = 5,
    close_kernel_size: int = 9,
) -> TrayDetectionResult:
    """Isolate the tray by HSV thresholding and find its centroid.

    Steps:
    - threshold in HSV
    - morphological open/close to clean noise
    - find external contours
    - keep the largest contour above ``min_area_px``
    - draw contour + centroid on the color image
    """
    mask = hsv_mask(image_bgr, bounds)

    if open_kernel_size > 1:
        kernel_open = np.ones((open_kernel_size, open_kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    if close_kernel_size > 1:
        kernel_close = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = image_bgr.copy()
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    if not contours:
        return TrayDetectionResult(
            mask=mask,
            masked_color=masked,
            annotated_color=annotated,
            contour=None,
            centroid_xy=None,
            area_px=0.0,
        )

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < float(min_area_px):
        return TrayDetectionResult(
            mask=mask,
            masked_color=masked,
            annotated_color=annotated,
            contour=None,
            centroid_xy=None,
            area_px=area,
        )

    moments = cv2.moments(contour)
    centroid_xy: Optional[tuple[int, int]]
    if abs(moments["m00"]) < 1e-9:
        centroid_xy = None
    else:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        centroid_xy = (cx, cy)

    cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 200, 0), 2)

    if centroid_xy is not None:
        cv2.circle(annotated, centroid_xy, 6, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"centroid=({centroid_xy[0]}, {centroid_xy[1]})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"area_px={area:.0f}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return TrayDetectionResult(
        mask=mask,
        masked_color=masked,
        annotated_color=annotated,
        contour=contour,
        centroid_xy=centroid_xy,
        area_px=area,
    )
