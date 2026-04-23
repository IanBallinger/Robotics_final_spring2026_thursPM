"""
segmentation.py

Segmentation backends for RGBD frames (pure Python).

Design goals:
  - Simple default that works on a laptop with no GPU: depth-based connected components.
  - Optional drop-in ML backends if you have them installed:
      * YOLOv8-seg (ultralytics)
      * SAM (segment_anything) for mask refinement given boxes

API:
  Segmenter.segment(rgb_bgr, depth_m, target_label=None) -> list[Segmentation]

Segmentation output is always:
  - mask: bool HxW
  - bbox_xyxy: (x1, y1, x2, y2)
  - label: str
  - score: float
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import cv2
except ImportError as e:
    raise ImportError("opencv-python is required: pip install opencv-python") from e


# -----------------------------
# Common output format
# -----------------------------

@dataclass
class Segmentation:
    mask: np.ndarray  # bool HxW
    bbox_xyxy: Tuple[int, int, int, int]
    label: str
    score: float


# -----------------------------
# Configs
# -----------------------------

@dataclass
class DepthCCConfig:
    """
    Depth connected-components segmentation.

    Works best when:
      - objects are closer than background
      - there is some depth separation
    """
    min_depth_m: float = 0.15
    max_depth_m: float = 2.0

    # Foreground depth selection:
    # If use_auto_foreground is True, it picks pixels closer than a percentile threshold.
    use_auto_foreground: bool = True
    foreground_percentile: float = 35.0  # lower means closer pixels

    # Cleanup
    morph_kernel: int = 5
    min_area_px: int = 800
    max_instances: int = 10


@dataclass
class YoloSegConfig:
    """
    Optional YOLOv8 segmentation backend.

    Requires:
      pip install ultralytics

    Note:
      The first run may download weights if you provide a model name like 'yolov8n-seg.pt'.
    """
    model: str = os.path.join(os.path.dirname(__file__), "yolov8n-seg.pt")
    conf_thres: float = 0.25
    iou_thres: float = 0.5
    max_det: int = 10
    classes: Optional[List[int]] = None  # list of class ids to keep


@dataclass
class SamRefineConfig:
    """
    Optional SAM mask refinement given bounding boxes.

    Requires:
      pip install segment-anything
    And a SAM checkpoint file.

    This is best used as a refinement stage:
      boxes from detector -> SAM masks
    """
    checkpoint_path: str = ""  # required
    model_type: str = "vit_b"
    device: str = "cpu"        # cpu is slow but works
    multimask_output: bool = False


# -----------------------------
# Depth connected-components backend (default)
# -----------------------------

class DepthCCSegmenter:
    def __init__(self, cfg: Optional[DepthCCConfig] = None):
        self.cfg = cfg or DepthCCConfig()

    def segment(self, rgb_bgr: np.ndarray, depth_m: np.ndarray) -> List[Segmentation]:
        h, w = depth_m.shape[:2]
        depth = depth_m.copy()

        # Valid depth mask
        valid = (depth >= self.cfg.min_depth_m) & (depth <= self.cfg.max_depth_m)

        if not np.any(valid):
            return []

        # Foreground selection
        if self.cfg.use_auto_foreground:
            vals = depth[valid]
            thr = np.percentile(vals, self.cfg.foreground_percentile)
            fg = valid & (depth <= thr)
        else:
            fg = valid

        # Convert to uint8 for morphology
        fg_u8 = (fg.astype(np.uint8) * 255)

        k = max(1, int(self.cfg.morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # Clean small holes and speckles
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Connected components on foreground
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)

        segs: List[Segmentation] = []
        # label 0 is background
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < self.cfg.min_area_px:
                continue

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + bw - 1)
            y2 = int(y + bh - 1)

            mask = (labels == i)

            # A simple score: fraction of valid depth pixels in the component
            comp_valid = valid & mask
            score = float(np.sum(comp_valid) / max(1, int(area)))

            segs.append(
                Segmentation(
                    mask=mask.astype(bool),
                    bbox_xyxy=(x1, y1, x2, y2),
                    label="unknown",
                    score=score,
                )
            )

        # Keep largest instances up to max_instances
        segs.sort(key=lambda s: int(np.sum(s.mask)), reverse=True)
        return segs[: self.cfg.max_instances]


# -----------------------------
# Optional YOLOv8-seg backend
# -----------------------------

class YoloSegSegmenter:
    def __init__(self, cfg: Optional[YoloSegConfig] = None):
        self.cfg = cfg or YoloSegConfig()
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise ImportError(
                "ultralytics is required for YOLO segmentation. Install:\n"
                "  pip install ultralytics"
            ) from e
        self._YOLO = YOLO
        self.model = self._YOLO(self.cfg.model)

    def segment(self, rgb_bgr: np.ndarray, depth_m: np.ndarray) -> List[Segmentation]:
        # ultralytics expects RGB
        rgb = rgb_bgr[:, :, ::-1].copy()

        results = self.model.predict(
            source=rgb,
            conf=self.cfg.conf_thres,
            iou=self.cfg.iou_thres,
            max_det=self.cfg.max_det,
            classes=self.cfg.classes,
            verbose=False,
        )

        if not results:
            return []

        r0 = results[0]
        segs: List[Segmentation] = []

        # If masks are missing, return empty
        if r0.masks is None or r0.boxes is None:
            return []

        masks = r0.masks.data  # (N, H, W) torch
        boxes = r0.boxes

        # Convert to numpy
        masks_np = masks.cpu().numpy().astype(bool)

        # Class names map
        names: Dict[int, str] = getattr(r0, "names", {}) or {}

        for i in range(masks_np.shape[0]):
            mask = masks_np[i]

            # Box
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy.tolist()]

            cls_id = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            label = names.get(cls_id, f"class_{cls_id}")

            segs.append(
                Segmentation(
                    mask=mask,
                    bbox_xyxy=(x1, y1, x2, y2),
                    label=label,
                    score=conf,
                )
            )

        return segs


# -----------------------------
# Optional SAM refinement backend
# -----------------------------

class SamRefineSegmenter:
    """
    SAM is most useful to refine masks given boxes.
    This class expects you to provide boxes from elsewhere.

    Use:
      segment_with_boxes(rgb_bgr, boxes_xyxy) -> masks

    It does NOT do detection.
    """
    def __init__(self, cfg: SamRefineConfig):
        if not cfg.checkpoint_path:
            raise ValueError("SamRefineConfig.checkpoint_path is required.")
        self.cfg = cfg

        try:
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        except Exception as e:
            raise ImportError(
                "segment-anything is required for SAM refinement. Install:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e

        sam = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint_path)
        sam.to(device=cfg.device)
        self.predictor = SamPredictor(sam)

    def segment_with_boxes(
        self,
        rgb_bgr: np.ndarray,
        boxes_xyxy: List[Tuple[int, int, int, int]],
        labels: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
    ) -> List[Segmentation]:
        rgb = rgb_bgr[:, :, ::-1].copy()  # SAM expects RGB
        self.predictor.set_image(rgb)

        segs: List[Segmentation] = []
        labels = labels or ["unknown"] * len(boxes_xyxy)
        scores = scores or [1.0] * len(boxes_xyxy)

        for (x1, y1, x2, y2), lab, sc in zip(boxes_xyxy, labels, scores):
            box = np.array([x1, y1, x2, y2], dtype=np.float32)

            masks, mask_scores, _ = self.predictor.predict(
                box=box,
                multimask_output=self.cfg.multimask_output,
            )

            if masks is None or len(masks) == 0:
                continue

            # Choose best mask
            if self.cfg.multimask_output:
                j = int(np.argmax(mask_scores))
                mask = masks[j].astype(bool)
                sc_out = float(mask_scores[j])
            else:
                mask = masks[0].astype(bool)
                sc_out = float(mask_scores[0]) if mask_scores is not None else float(sc)

            segs.append(
                Segmentation(
                    mask=mask,
                    bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    label=str(lab),
                    score=sc_out,
                )
            )

        return segs


# -----------------------------
# Unified Segmenter facade
# -----------------------------

class Segmenter:
    """
    A simple facade so your main loop can call one thing.

    Modes:
      - "depth_cc" (default): depth-based connected components (fast, laptop-friendly)
      - "yolo": YOLOv8 segmentation (optional dependency)
    """
    def __init__(
        self,
        mode: str = "depth_cc",
        depth_cc_cfg: Optional[DepthCCConfig] = None,
        yolo_cfg: Optional[YoloSegConfig] = None,
    ):
        mode = mode.strip().lower()
        self.mode = mode

        if mode == "depth_cc":
            self.backend = DepthCCSegmenter(depth_cc_cfg)
        elif mode == "yolo":
            self.backend = YoloSegSegmenter(yolo_cfg)
        else:
            raise ValueError(f"Unknown segmentation mode: {mode}")

    def segment(
        self,
        rgb_bgr: np.ndarray,
        depth_m: np.ndarray,
        target_label: Optional[str] = None,
    ) -> List[Segmentation]:
        segs = self.backend.segment(rgb_bgr, depth_m)

        if target_label is None:
            return segs

        # Filter by label if labels exist (YOLO mode).
        t = target_label.strip().lower()
        return [s for s in segs if s.label.strip().lower() == t]


# -----------------------------
# Minimal demo
# -----------------------------

def _demo() -> None:
    """
    Demo segmentation output using the depth_cc mode with a RealSense stream.
    Requires:
      - Core/realsense_io.py in same folder
      - opencv-python
      - pyrealsense2
    """
    from Core.realsense_io import RealSenseIO

    cam = RealSenseIO()
    cam.start()

    seg = Segmenter(mode="depth_cc")

    try:
        while True:
            rgb, depth_m, intr, ts = cam.get_frame()
            segs = seg.segment(rgb, depth_m)

            vis = rgb.copy()
            for i, s in enumerate(segs[:5]):
                x1, y1, x2, y2 = s.bbox_xyxy
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{i}:{s.label} {s.score:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("segmentation demo", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
