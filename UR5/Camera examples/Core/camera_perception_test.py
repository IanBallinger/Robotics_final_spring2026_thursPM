"""
run_perception.py

Pure-Python perception runner:
  - Captures RGB + aligned depth from Intel RealSense (realsense_io.py)
  - Segments objects (segmentation.py)
  - Estimates CAD-free 6D pose proxy in CAMERA frame from mask + depth (pose_from_depth.py)
  - Tracks objects over time for stable IDs and smoothing
  - Visualizes masks, bboxes, and pose axes on the RGB image

Dependencies:
  pip install pyrealsense2 numpy opencv-python

Files expected in Core/:
  - Core/realsense_io.py
  - Core/segmentation.py
  - Core/pose_from_depth.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Core.realsense_io import RealSenseIO, RealSenseConfig
from Core.segmentation import Segmenter, Segmentation, DepthCCConfig, YoloSegConfig
from Core.pose_from_depth import (
    PoseFromDepthConfig,
    PoseEstimate,
    estimate_pose_from_mask_depth,
    project_points_to_image,
)

# -----------------------------
# User-editable configuration
# -----------------------------

@dataclass
class RunConfig:
    # Segmentation mode: "depth_cc" (fast) or "yolo" (optional dependency ultralytics)
    segmentation_mode: str = "depth_cc"

    # Tracking
    max_tracks: int = 20
    track_max_age_s: float = 1.0
    assoc_iou_min: float = 0.10
    assoc_dist_max_m: float = 0.25

    # Smoothing
    smooth_pos_alpha: float = 0.35   # 0=no update, 1=no smoothing
    smooth_rot_alpha: float = 0.35   # for quaternion nlerp

    # Output filtering
    min_pose_confidence: float = 0.10
    max_objects_to_draw: int = 8

    # Debug visualization
    draw_masks: bool = True
    draw_axes: bool = True
    axis_length_m: float = 0.10
    show_depth_preview: bool = False
    show_axis_legend: bool = False

    # Printing
    print_every_n_frames: int = 10


# Quick toggle for segmentation backend
SEGMENTATION_MODE = "yolo"  # "depth_cc" or "yolo"

# You can tune these as you like
RS_CFG = RealSenseConfig(
    color_width=640, color_height=480, color_fps=30,
    depth_width=640, depth_height=480, depth_fps=30,
    output_bgr=True,
    min_depth_m=0.15,
    max_depth_m=4.0,
    enable_depth_filters=True,
    enable_spatial_filter=True,
    enable_temporal_filter=True,
    enable_hole_filling=True,
)

DEPTH_CC_CFG = DepthCCConfig(
    min_depth_m=0.15,
    max_depth_m=2.0,
    use_auto_foreground=True,
    foreground_percentile=35.0,
    morph_kernel=5,
    min_area_px=900,
    max_instances=10,
)

_YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n-seg.pt")

YOLO_CFG = YoloSegConfig(
    model=_YOLO_MODEL_PATH,
    conf_thres=0.25,
    iou_thres=0.5,
    max_det=10,
    classes=None,
)

POSE_CFG = PoseFromDepthConfig(
    max_points=5000,
    min_points=250,
    min_depth_m=0.15,
    max_depth_m=4.0,
    enable_outlier_filter=True,
    outlier_radius_m=0.10,
    stabilize_axes=True,
    enforce_right_handed=True,
    symmetry_ratio_thresh=1.25,
)

RUN_CFG = RunConfig(
    segmentation_mode=SEGMENTATION_MODE,
    max_tracks=20,
    track_max_age_s=1.0,
    assoc_iou_min=0.10,
    assoc_dist_max_m=0.25,
    smooth_pos_alpha=0.35,
    smooth_rot_alpha=0.35,
    min_pose_confidence=0.10,
    max_objects_to_draw=8,
    draw_masks=True,
    draw_axes=True,
    axis_length_m=0.10,
    show_depth_preview=False,
    show_axis_legend=False,
    print_every_n_frames=10,
)

# -----------------------------
# Tracking data structures
# -----------------------------

@dataclass
class Track:
    track_id: int
    label: str
    score: float

    position_m: np.ndarray          # (3,)
    quaternion_xyzw: np.ndarray     # (4,)
    rotation_matrix: np.ndarray     # (3,3)

    bbox_xyxy: Tuple[int, int, int, int]
    last_seen_time: float
    num_points: int
    confidence: float
    flags: Dict[str, bool]


# -----------------------------
# Utility functions
# -----------------------------

def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    b_area = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (q / n).astype(np.float64)


def _quat_nlerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Normalized linear interpolation with shortest-path handling.
    q are in (x,y,z,w).
    """
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)

    # Shortest path: if dot < 0, negate q1
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1

    q = (1.0 - alpha) * q0 + alpha * q1
    return _quat_normalize(q)


def _smooth_pose(
    pos_old: np.ndarray,
    q_old: np.ndarray,
    pos_new: np.ndarray,
    q_new: np.ndarray,
    pos_alpha: float,
    rot_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pos = (1.0 - pos_alpha) * pos_old + pos_alpha * pos_new
    q = _quat_nlerp(q_old, q_new, rot_alpha)
    return pos.astype(np.float64), q.astype(np.float64)


def _draw_mask_overlay(img_bgr: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = img_bgr.copy()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[mask] = np.array(color_bgr, dtype=np.uint8)
    out = cv2.addWeighted(out, 1.0, overlay, alpha, 0.0)
    return out


def _draw_axes(
    img_bgr: np.ndarray,
    intrinsics: Dict[str, float],
    position_m: np.ndarray,
    rotation_matrix: np.ndarray,
    axis_length_m: float,
) -> None:
    origin = position_m.reshape(1, 3).astype(np.float32)
    R = rotation_matrix.astype(np.float64)

    axes_pts = np.stack(
        [
            origin[0],
            origin[0] + axis_length_m * R[:, 0],
            origin[0] + axis_length_m * R[:, 1],
            origin[0] + axis_length_m * R[:, 2],
        ],
        axis=0,
    ).astype(np.float32)

    uv = project_points_to_image(axes_pts, intrinsics)
    uv_i = np.round(uv).astype(int)

    o = tuple(uv_i[0].tolist())
    px = tuple(uv_i[1].tolist())
    py = tuple(uv_i[2].tolist())
    pz = tuple(uv_i[3].tolist())

    # OpenCV is BGR
    cv2.line(img_bgr, o, px, (0, 0, 255), 2)   # x
    cv2.line(img_bgr, o, py, (0, 255, 0), 2)   # y
    cv2.line(img_bgr, o, pz, (255, 0, 0), 2)   # z


def _random_color_from_id(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973 + 12345)
    c = rng.integers(low=50, high=230, size=(3,), dtype=np.int32)
    return int(c[0]), int(c[1]), int(c[2])


# -----------------------------
# Association and tracking
# -----------------------------

def _associate_tracks(
    tracks: List[Track],
    segs: List[Segmentation],
    poses: List[PoseEstimate],
    now_s: float,
    cfg: RunConfig,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """
    Returns:
      matches: dict track_index -> detection_index
      unmatched_tracks: list of track indices
      unmatched_dets: list of detection indices
    """
    if not tracks or not segs:
        return {}, list(range(len(tracks))), list(range(len(segs)))

    # Build cost matrix based on distance and IoU gate
    M = len(tracks)
    N = len(segs)
    cost = np.full((M, N), fill_value=np.inf, dtype=np.float64)

    for ti, tr in enumerate(tracks):
        for di, (s, p) in enumerate(zip(segs, poses)):
            iou = _bbox_iou(tr.bbox_xyxy, s.bbox_xyxy)
            dist = float(np.linalg.norm(tr.position_m - p.position_m))
            if iou < cfg.assoc_iou_min and dist > cfg.assoc_dist_max_m:
                continue
            # Weighted cost: prefer close distance, then higher IoU
            cost[ti, di] = dist - 0.10 * iou

    matches: Dict[int, int] = {}
    used_tracks = set()
    used_dets = set()

    # Greedy assignment (good enough for small counts)
    while True:
        ti, di = np.unravel_index(np.argmin(cost), cost.shape)
        best = float(cost[ti, di])
        if not np.isfinite(best):
            break
        if ti in used_tracks or di in used_dets:
            cost[ti, di] = np.inf
            continue
        matches[ti] = di
        used_tracks.add(ti)
        used_dets.add(di)
        cost[ti, :] = np.inf
        cost[:, di] = np.inf

    unmatched_tracks = [i for i in range(M) if i not in used_tracks]
    unmatched_dets = [j for j in range(N) if j not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


def _prune_tracks(tracks: List[Track], now_s: float, cfg: RunConfig) -> List[Track]:
    keep = []
    for tr in tracks:
        if (now_s - tr.last_seen_time) <= cfg.track_max_age_s:
            keep.append(tr)
    # Keep at most max_tracks by recency
    keep.sort(key=lambda t: t.last_seen_time, reverse=True)
    return keep[: cfg.max_tracks]


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cam = RealSenseIO(RS_CFG)
    cam.start()

    window_name = "Perception"

    def _init_segmenter(mode: str) -> Segmenter:
        mode = mode.strip().lower()
        if mode == "depth_cc":
            return Segmenter(mode="depth_cc", depth_cc_cfg=DEPTH_CC_CFG)
        if mode == "yolo":
            return Segmenter(mode="yolo", yolo_cfg=YOLO_CFG)
        raise ValueError(f"Unknown segmentation mode: {mode}")

    segmenter = _init_segmenter(RUN_CFG.segmentation_mode)

    tracks: List[Track] = []
    next_track_id = 1
    frame_idx = 0
    select_mode = False
    selected_track_id: Optional[int] = None
    roi_mode = False
    roi_dragging = False
    roi_start: Optional[Tuple[int, int]] = None
    roi_rect: Optional[Tuple[int, int, int, int]] = None
    roi_rect_live: Optional[Tuple[int, int, int, int]] = None
    last_tracks: List[Track] = []
    last_segs: List[Segmentation] = []
    last_track_id_to_det: Dict[int, int] = {}
    last_frame_shape: Optional[Tuple[int, int]] = None

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal selected_track_id
        nonlocal roi_dragging, roi_start, roi_rect, roi_rect_live

        if last_frame_shape is not None:
            h, w = last_frame_shape
            x = int(np.clip(x, 0, w - 1))
            y = int(np.clip(y, 0, h - 1))

        if event == cv2.EVENT_RBUTTONDOWN:
            selected_track_id = None
            if roi_mode:
                roi_dragging = False
                roi_start = None
                roi_rect = None
                roi_rect_live = None
            return

        if roi_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_dragging = True
                roi_start = (x, y)
                roi_rect_live = None
                return
            if event == cv2.EVENT_MOUSEMOVE and roi_dragging and roi_start is not None:
                x0, y0 = roi_start
                x1, y1 = min(x0, x), min(y0, y)
                x2, y2 = max(x0, x), max(y0, y)
                roi_rect_live = (x1, y1, x2, y2)
                return
            if event == cv2.EVENT_LBUTTONUP and roi_dragging and roi_start is not None:
                x0, y0 = roi_start
                x1, y1 = min(x0, x), min(y0, y)
                x2, y2 = max(x0, x), max(y0, y)
                roi_dragging = False
                roi_start = None
                roi_rect_live = None
                if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                    roi_rect = (x1, y1, x2, y2)
                else:
                    roi_rect = None
                return
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not last_tracks:
            return

        # Prefer mask hit-test when available; otherwise fallback to bbox.
        for tr in last_tracks:
            det_idx = last_track_id_to_det.get(tr.track_id)
            if det_idx is not None and 0 <= det_idx < len(last_segs):
                mask = last_segs[det_idx].mask
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x]):
                    selected_track_id = tr.track_id
                    return

        for tr in last_tracks:
            x1, y1, x2, y2 = tr.bbox_xyxy
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_track_id = tr.track_id
                return

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse)

    print("Running perception. Press 'q' or ESC to quit.")

    try:
        while True:
            rgb, depth_m, intr, ts_ms = cam.get_frame()
            now_s = time.time()
            frame_idx += 1
            last_frame_shape = rgb.shape[:2]

            # 1) segment
            if roi_rect is None:
                segs = segmenter.segment(rgb, depth_m)
            else:
                h, w = depth_m.shape[:2]
                x1, y1, x2, y2 = roi_rect
                x1 = int(np.clip(x1, 0, w - 1))
                x2 = int(np.clip(x2, 0, w - 1))
                y1 = int(np.clip(y1, 0, h - 1))
                y2 = int(np.clip(y2, 0, h - 1))
                if x2 <= x1 or y2 <= y1:
                    segs = []
                else:
                    rgb_roi = rgb[y1 : y2 + 1, x1 : x2 + 1]
                    depth_roi = depth_m[y1 : y2 + 1, x1 : x2 + 1]
                    segs_roi = segmenter.segment(rgb_roi, depth_roi)
                    segs = []
                    roi_h, roi_w = depth_roi.shape[:2]
                    for s in segs_roi:
                        mask_roi = s.mask.astype(bool)
                        if mask_roi.shape != (roi_h, roi_w):
                            mask_roi = mask_roi[:roi_h, :roi_w]
                        mask_full = np.zeros((h, w), dtype=bool)
                        mask_full[y1 : y1 + mask_roi.shape[0], x1 : x1 + mask_roi.shape[1]] = mask_roi
                        bx1, by1, bx2, by2 = s.bbox_xyxy
                        segs.append(
                            Segmentation(
                                mask=mask_full,
                                bbox_xyxy=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                                label=s.label,
                                score=float(s.score),
                            )
                        )

            # 2) pose estimates (camera frame)
            poses: List[PoseEstimate] = []
            for s in segs:
                # If we have an existing track match later, we will pass prev_R then.
                pose = estimate_pose_from_mask_depth(
                    mask=s.mask,
                    depth_m=depth_m,
                    intrinsics=intr,
                    cfg=POSE_CFG,
                    prev_rotation_matrix=None,
                )
                poses.append(pose)

            # Filter low-confidence detections early
            keep_idx = [i for i, p in enumerate(poses) if p.confidence >= RUN_CFG.min_pose_confidence]
            segs = [segs[i] for i in keep_idx]
            poses = [poses[i] for i in keep_idx]

            # 3) associate to tracks
            matches, unmatched_tracks, unmatched_dets = _associate_tracks(tracks, segs, poses, now_s, RUN_CFG)

            # 4) update matched tracks with smoothing and axis stabilization
            for ti, di in matches.items():
                tr = tracks[ti]
                s = segs[di]
                p = poses[di]

                # Re-estimate with prev rotation matrix to reduce PCA flips
                p_refined = estimate_pose_from_mask_depth(
                    mask=s.mask,
                    depth_m=depth_m,
                    intrinsics=intr,
                    cfg=POSE_CFG,
                    prev_rotation_matrix=tr.rotation_matrix,
                )

                pos_s, q_s = _smooth_pose(
                    tr.position_m,
                    tr.quaternion_xyzw,
                    p_refined.position_m,
                    p_refined.quaternion_xyzw,
                    RUN_CFG.smooth_pos_alpha,
                    RUN_CFG.smooth_rot_alpha,
                )

                tr.position_m = pos_s
                tr.quaternion_xyzw = q_s
                tr.rotation_matrix = p_refined.rotation_matrix
                tr.bbox_xyxy = s.bbox_xyxy
                tr.last_seen_time = now_s
                tr.label = s.label
                tr.score = float(s.score)
                tr.num_points = int(p_refined.num_points)
                tr.confidence = float(p_refined.confidence)
                tr.flags = dict(p_refined.flags)

            # 5) create new tracks for unmatched detections
            for di in unmatched_dets:
                s = segs[di]
                p = poses[di]

                tr = Track(
                    track_id=next_track_id,
                    label=s.label,
                    score=float(s.score),
                    position_m=p.position_m.astype(np.float64),
                    quaternion_xyzw=p.quaternion_xyzw.astype(np.float64),
                    rotation_matrix=p.rotation_matrix.astype(np.float64),
                    bbox_xyxy=s.bbox_xyxy,
                    last_seen_time=now_s,
                    num_points=int(p.num_points),
                    confidence=float(p.confidence),
                    flags=dict(p.flags),
                )
                next_track_id += 1
                tracks.append(tr)

            # Cache detection index per track_id for this frame before pruning/reordering
            track_id_to_det: Dict[int, int] = {}
            for ti, di in matches.items():
                if 0 <= ti < len(tracks):
                    track_id_to_det[tracks[ti].track_id] = di

            # 6) prune old tracks
            tracks = _prune_tracks(tracks, now_s, RUN_CFG)

            # 7) visualization
            vis = rgb.copy()

            # Draw tracks ordered by recency
            tracks_sorted = sorted(tracks, key=lambda t: t.last_seen_time, reverse=True)
            last_tracks = tracks_sorted[: RUN_CFG.max_objects_to_draw]
            last_segs = segs
            last_track_id_to_det = track_id_to_det

            tracks_to_draw = last_tracks
            if select_mode and selected_track_id is not None:
                tracks_to_draw = [tr for tr in last_tracks if tr.track_id == selected_track_id]

            for k, tr in enumerate(tracks_to_draw):
                color = _random_color_from_id(tr.track_id)

                # Optional: mask overlay only if we can find a current matching detection
                if RUN_CFG.draw_masks:
                    # find if this track was matched this frame
                    det_idx = track_id_to_det.get(tr.track_id)
                    if det_idx is not None:
                        vis = _draw_mask_overlay(vis, segs[det_idx].mask, color, alpha=0.30)

                x1, y1, x2, y2 = tr.bbox_xyxy
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                amb = tr.flags.get("rotation_ambiguous", False)
                few = tr.flags.get("too_few_points", False)
                txt = f"id={tr.track_id} {tr.label} conf={tr.confidence:.2f}"
                if amb:
                    txt += " amb"
                if few:
                    txt += " few"

                cv2.putText(
                    vis,
                    txt,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                if RUN_CFG.draw_axes:
                    _draw_axes(vis, intr, tr.position_m, tr.rotation_matrix, RUN_CFG.axis_length_m)

            if select_mode:
                if selected_track_id is None:
                    mode_txt = "mode: select (click object)"
                else:
                    mode_txt = f"mode: select (id={selected_track_id})"
            else:
                mode_txt = "mode: all (press 't' to select)"
            cv2.putText(
                vis,
                mode_txt,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            seg_txt = f"seg: {RUN_CFG.segmentation_mode} (y/yolo, d/depth)"
            cv2.putText(
                vis,
                seg_txt,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if RUN_CFG.show_axis_legend:
                cv2.putText(
                    vis,
                    "axes: x=red, y=green, z=blue (camera frame)",
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if roi_rect is None:
                roi_txt = "roi: off (press 'r')"
            else:
                roi_txt = "roi: set (press 'r' to edit, right-click to clear)"
            cv2.putText(
                vis,
                roi_txt,
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            rect = roi_rect_live or roi_rect
            if rect is not None:
                rx1, ry1, rx2, ry2 = rect
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

            if RUN_CFG.show_depth_preview:
                depth_vis = depth_m.copy()
                depth_vis[depth_vis == 0] = np.nan
                depth_norm = np.nan_to_num(depth_vis / max(1e-6, RS_CFG.max_depth_m))
                depth_img = (np.clip(depth_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
                cv2.imshow("Depth", depth_img)

            cv2.imshow(window_name, vis)

            # 8) print poses (occasionally)
            if RUN_CFG.print_every_n_frames > 0 and (frame_idx % RUN_CFG.print_every_n_frames == 0):
                print("\nTracks:")
                tracks_to_print = tracks_sorted[: RUN_CFG.max_objects_to_draw]
                if select_mode and selected_track_id is not None:
                    tracks_to_print = [tr for tr in tracks_to_print if tr.track_id == selected_track_id]
                for tr in tracks_to_print:
                    p = tr.position_m
                    q = tr.quaternion_xyzw
                    print(
                        f"  id={tr.track_id:<3d} label={tr.label:<12s} "
                        f"p=[{p[0]: .3f},{p[1]: .3f},{p[2]: .3f}] "
                        f"q=[{q[0]: .3f},{q[1]: .3f},{q[2]: .3f},{q[3]: .3f}] "
                        f"conf={tr.confidence:.2f} pts={tr.num_points}"
                    )

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("t"):
                select_mode = not select_mode
            if key == ord("r"):
                roi_mode = not roi_mode
            if key == ord("y"):
                RUN_CFG.segmentation_mode = "yolo"
                segmenter = _init_segmenter(RUN_CFG.segmentation_mode)
            if key == ord("d"):
                RUN_CFG.segmentation_mode = "depth_cc"
                segmenter = _init_segmenter(RUN_CFG.segmentation_mode)
            if key == ord("l"):
                RUN_CFG.show_axis_legend = not RUN_CFG.show_axis_legend

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
