"""
pose_from_depth.py

Compute a CAD-free 6D pose proxy of a segmented object in the CAMERA frame using:
  - depth + intrinsics to lift mask pixels into 3D points
  - PCA to define a consistent orientation frame from the point cloud

This gives a geometric pose that is often good enough for grasping and coarse placing.
It is not a "true" object pose for symmetric objects (cups, bottles, plates), so we
also output ambiguity flags.

Dependencies:
  pip install numpy

Optional (not required):
  pip install opencv-python  (only if you want the demo drawing utilities)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class PoseEstimate:
    position_m: np.ndarray            # (3,) in camera frame
    quaternion_xyzw: np.ndarray       # (4,) in camera frame
    rotation_matrix: np.ndarray       # (3,3)
    num_points: int
    confidence: float
    flags: Dict[str, bool]


@dataclass
class PoseFromDepthConfig:
    # Mask to point cloud
    max_points: int = 5000           # subsample if too many pixels
    min_points: int = 200            # below this, pose is unreliable

    # Depth validity
    min_depth_m: float = 0.15
    max_depth_m: float = 5.0

    # Outlier removal (simple radial filter around median)
    enable_outlier_filter: bool = True
    outlier_radius_m: float = 0.08   # keep points within this radius of median point

    # PCA stability
    enforce_right_handed: bool = True
    stabilize_axes: bool = True      # attempt to keep axes consistent (reduces flips)

    # Symmetry / ambiguity heuristics
    symmetry_ratio_thresh: float = 1.25  # if eigenvalue ratios are close, rotation is ambiguous

    # Confidence shaping
    conf_points_scale: int = 1500    # confidence rises with points up to this many
    conf_planarity_bonus: float = 0.1


# -----------------------------
# Core math helpers
# -----------------------------

def _rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (x,y,z,w).
    Assumes R is a proper rotation matrix.
    """
    # Robust conversion
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([x, y, z, w], dtype=np.float64)
    # Normalize
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (q / n).astype(np.float64)


def _make_right_handed(R: np.ndarray) -> np.ndarray:
    """
    Ensure rotation matrix is right-handed with det +1 by flipping the third axis if needed.
    """
    if np.linalg.det(R) < 0.0:
        R = R.copy()
        R[:, 2] *= -1.0
    return R


def _stabilize_pca_axes(V: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """
    PCA eigenvectors are unique up to sign flips. This function reduces random flips:
      - If reference is provided, align each axis to have positive dot with reference axis.
      - Otherwise, enforce a convention using axis directions relative to camera axes.

    V is 3x3 where columns are axes.
    """
    V = V.copy()

    if reference is not None:
        for k in range(3):
            if np.dot(V[:, k], reference[:, k]) < 0.0:
                V[:, k] *= -1.0
    else:
        # Simple camera-frame convention: make z-axis point away from camera if possible
        # (camera looks along +z in optical frame convention; but your points are already
        # in that frame, so positive z is forward). Encourage axis 2 to have positive z.
        if V[2, 2] < 0.0:
            V[:, 2] *= -1.0
        # Encourage x-axis to point to the right (positive x)
        if V[0, 0] < 0.0:
            V[:, 0] *= -1.0
        # Then recompute the remaining axis to keep orthonormality
        V[:, 1] = np.cross(V[:, 2], V[:, 0])
        n1 = np.linalg.norm(V[:, 1])
        if n1 > 1e-12:
            V[:, 1] /= n1

    return V


# -----------------------------
# Mask to 3D points
# -----------------------------

def mask_to_points_cam(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Dict[str, float],
    cfg: Optional[PoseFromDepthConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Convert mask + depth into 3D points in camera coordinates.

    intrinsics must contain: fx, fy, cx, cy (width/height are optional).
    Returns Nx3 float array in meters.
    """
    cfg = cfg or PoseFromDepthConfig()
    rng = rng or np.random.default_rng()

    if mask.dtype != bool:
        mask = mask.astype(bool)

    if depth_m.shape[:2] != mask.shape[:2]:
        raise ValueError("mask and depth_m must have the same HxW shape.")

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    # Valid depth within mask
    z = depth_m
    valid = mask & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m) & (z > 0.0)

    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Subsample if too many
    if ys.size > cfg.max_points:
        idx = rng.choice(ys.size, size=cfg.max_points, replace=False)
        ys = ys[idx]
        xs = xs[idx]

    z_s = z[ys, xs].astype(np.float64)
    x_s = (xs.astype(np.float64) - cx) * z_s / fx
    y_s = (ys.astype(np.float64) - cy) * z_s / fy

    pts = np.stack([x_s, y_s, z_s], axis=1).astype(np.float32)

    if cfg.enable_outlier_filter and pts.shape[0] >= 20:
        med = np.median(pts, axis=0)
        d = np.linalg.norm(pts - med[None, :], axis=1)
        keep = d <= float(cfg.outlier_radius_m)
        pts = pts[keep]

    return pts


# -----------------------------
# Pose estimation from points
# -----------------------------

def estimate_pose_pca_cam(
    points_cam: np.ndarray,
    cfg: Optional[PoseFromDepthConfig] = None,
    prev_rotation_matrix: Optional[np.ndarray] = None,
) -> PoseEstimate:
    """
    Estimate a CAD-free pose proxy from points in camera frame using PCA.
    Returns PoseEstimate with position and quaternion in camera frame.
    """
    cfg = cfg or PoseFromDepthConfig()

    flags: Dict[str, bool] = {
        "too_few_points": False,
        "rotation_ambiguous": False,
        "nearly_planar": False,
    }

    n = int(points_cam.shape[0])
    if n < cfg.min_points:
        flags["too_few_points"] = True
        # Best-effort: position is median, orientation identity
        if n == 0:
            pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            pos = np.median(points_cam.astype(np.float64), axis=0)
        R = np.eye(3, dtype=np.float64)
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return PoseEstimate(
            position_m=pos,
            quaternion_xyzw=q,
            rotation_matrix=R,
            num_points=n,
            confidence=0.0,
            flags=flags,
        )

    pts = points_cam.astype(np.float64)
    pos = np.median(pts, axis=0)

    # Center
    X = pts - pos[None, :]

    # Covariance
    C = (X.T @ X) / max(1.0, float(n - 1))

    # Eigen decomposition
    evals, evecs = np.linalg.eigh(C)  # evals ascending
    order = np.argsort(evals)[::-1]   # descending by variance
    evals = evals[order]
    evecs = evecs[:, order]  # columns

    # Symmetry / ambiguity heuristics
    # If top two eigenvalues are close, yaw around the smallest axis can be unstable.
    r12 = float(evals[0] / max(evals[1], 1e-12))
    r23 = float(evals[1] / max(evals[2], 1e-12))
    if r12 < cfg.symmetry_ratio_thresh:
        flags["rotation_ambiguous"] = True

    # Planarity: if smallest eigenvalue is much smaller than others, point cloud is planar
    if r23 > 8.0:
        flags["nearly_planar"] = True

    V = evecs  # columns are principal axes

    if cfg.stabilize_axes:
        V = _stabilize_pca_axes(V, reference=prev_rotation_matrix if prev_rotation_matrix is not None else None)

    R = V
    if cfg.enforce_right_handed:
        R = _make_right_handed(R)

    q = _rotation_matrix_to_quaternion_xyzw(R)

    # Confidence heuristic
    conf_points = min(1.0, n / max(1.0, float(cfg.conf_points_scale)))
    conf = conf_points
    if flags["nearly_planar"]:
        conf = min(1.0, conf + cfg.conf_planarity_bonus)
    if flags["rotation_ambiguous"]:
        conf *= 0.7

    return PoseEstimate(
        position_m=pos,
        quaternion_xyzw=q,
        rotation_matrix=R,
        num_points=n,
        confidence=float(conf),
        flags=flags,
    )


def estimate_pose_from_mask_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Dict[str, float],
    cfg: Optional[PoseFromDepthConfig] = None,
    prev_rotation_matrix: Optional[np.ndarray] = None,
) -> PoseEstimate:
    """
    Convenience wrapper: mask + depth -> points -> PCA pose.
    """
    cfg = cfg or PoseFromDepthConfig()
    pts = mask_to_points_cam(mask, depth_m, intrinsics, cfg=cfg)
    return estimate_pose_pca_cam(pts, cfg=cfg, prev_rotation_matrix=prev_rotation_matrix)


# -----------------------------
# Minimal demo helpers (optional)
# -----------------------------

def project_points_to_image(
    points_cam: np.ndarray,
    intrinsics: Dict[str, float],
) -> np.ndarray:
    """
    Project Nx3 camera-frame points to Nx2 pixel coordinates.
    Returns float Nx2 with (u, v).
    """
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    pts = points_cam.astype(np.float64)
    z = np.clip(pts[:, 2], 1e-9, None)
    u = (pts[:, 0] * fx / z) + cx
    v = (pts[:, 1] * fy / z) + cy
    return np.stack([u, v], axis=1)


def _demo() -> None:
    """
    Quick demo:
      - reads RGBD frames using Core/realsense_io.py
      - segments using Core/segmentation.py depth connected-components
      - estimates pose for the largest segment
      - draws a simple axis overlay

    Requires:
      pip install opencv-python pyrealsense2 numpy
    """
    import cv2
    from Core.realsense_io import RealSenseIO
    from Core.segmentation import Segmenter

    cfg = PoseFromDepthConfig()
    cam = RealSenseIO()
    cam.start()
    seg = Segmenter(mode="depth_cc")

    prev_R = None

    try:
        while True:
            rgb, depth_m, intr, ts = cam.get_frame()
            segs = seg.segment(rgb, depth_m)

            vis = rgb.copy()

            if segs:
                s0 = segs[0]
                pose = estimate_pose_from_mask_depth(s0.mask, depth_m, intr, cfg=cfg, prev_rotation_matrix=prev_R)
                prev_R = pose.rotation_matrix

                x1, y1, x2, y2 = s0.bbox_xyxy
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw axes in image by projecting 3D axis endpoints
                origin = pose.position_m.reshape(1, 3)
                L = 0.10  # 10 cm axes
                R = pose.rotation_matrix
                axes = np.stack(
                    [
                        origin[0],
                        origin[0] + L * R[:, 0],  # x
                        origin[0] + L * R[:, 1],  # y
                        origin[0] + L * R[:, 2],  # z
                    ],
                    axis=0,
                ).astype(np.float32)

                uv = project_points_to_image(axes, intr)
                o = tuple(np.round(uv[0]).astype(int).tolist())
                px = tuple(np.round(uv[1]).astype(int).tolist())
                py = tuple(np.round(uv[2]).astype(int).tolist())
                pz = tuple(np.round(uv[3]).astype(int).tolist())

                # Colors: x red, y green, z blue (OpenCV uses BGR)
                cv2.line(vis, o, px, (0, 0, 255), 2)
                cv2.line(vis, o, py, (0, 255, 0), 2)
                cv2.line(vis, o, pz, (255, 0, 0), 2)

                txt = f"pts={pose.num_points} conf={pose.confidence:.2f} amb={pose.flags['rotation_ambiguous']}"
                cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("pose_from_depth demo", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
