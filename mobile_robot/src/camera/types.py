from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StreamConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Optional[np.ndarray] = None

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    rotation_camera_to_world: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=float)
    )
    translation_camera_to_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )


@dataclass
class CameraFrame:
    color: np.ndarray
    timestamp_s: float
    frame_id: Optional[int] = None
    intrinsics: Optional[CameraIntrinsics] = None


@dataclass
class DepthFrame(CameraFrame):
    depth: Optional[np.ndarray] = None
    depth_scale: float = 1.0
    aligned_depth: bool = True
