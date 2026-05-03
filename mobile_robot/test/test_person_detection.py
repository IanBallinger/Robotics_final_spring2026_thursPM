#!/usr/bin/env python3
"""Live test script for person detection and depth estimation."""

from __future__ import annotations

import argparse
import os
import sys

import cv2

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from localization.person_detection import (  # noqa: E402
    PersonDetectionConfig,
    PersonDetector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="yolo26n.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--mission-config",
        default=None,
        help="Optional path to mission_config.yaml for camera_to_robot extrinsics.",
    )
    args = parser.parse_args()

    config = PersonDetectionConfig(
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        tracker=args.tracker or None,
    )
    config = PersonDetectionConfig(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        min_depth_m=config.min_depth_m,
        max_depth_m=config.max_depth_m,
        min_valid_depth_pixels=config.min_valid_depth_pixels,
        tracker=config.tracker,
        color_config=type(config.color_config)(width=args.width, height=args.height, fps=args.fps),
        depth_config=type(config.depth_config)(width=args.width, height=args.height, fps=args.fps),
        align_depth_to_color=config.align_depth_to_color,
    )

    with PersonDetector(config=config, mission_config_path=args.mission_config) as detector:
        print("Running person detection test — press Q to quit")
        while True:
            detections, frame = detector.detect_with_frame()
            for det in detections:
                print(
                    "DET",
                    f"id={det.track_id}",
                    f"conf={det.confidence:.2f}",
                    f"bbox={det.bbox_xyxy}",
                    f"depth_m={None if det.depth_m is None else round(det.depth_m, 3)}",
                    f"robot={None if det.position_robot_m is None else np_round_tuple(det.position_robot_m)}",
                )

            combined = detector.render_debug(frame, detections)
            # cv2.imshow("Person detection test", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()



def np_round_tuple(values) -> tuple[float, ...]:
    return tuple(round(float(v), 3) for v in values)


if __name__ == "__main__":
    main()
