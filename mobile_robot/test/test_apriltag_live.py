import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
TEST_DIR = THIS_FILE.parent
PROJECT_ROOT = TEST_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.localization.april_tag_pose_est import AprilTagGlobalPoseEstimator


# ---------------------------------------------------------------------
# Tag map for current estimator
# Each tag:
#   x, y, z = tag center in global frame
#   yaw     = tag facing direction in global x-y plane
# ---------------------------------------------------------------------
TAG_MAP = {
    4: {
        "x": 0.4,
        "y": 0.0,
        "z": 0.0,
        "yaw": -np.pi / 2,  # facing negative y direction
    },

    5: {
        "x": 0.0,
        "y": -0.4,
        "z": 0.0,
        "yaw": 0.0,  # facing positive x direction
    }
}

CAMERA_IN_ROBOT = {
    "x": 0.0,   # no offset in any direction
    "y": 0.0,   # left offset
    "z": 0.0,   # up offset
}

CALIBRATION_FILE = str(PROJECT_ROOT / "src" / "localization" / "camera_calibration_live.npz")
TAG_SIZE = 0.17


def yaw_deg_360(yaw_rad: float) -> float:
    yaw_deg = np.degrees(yaw_rad) % 360.0
    if np.isclose(yaw_deg, 360.0) or yaw_deg >= 360.0:
        yaw_deg = 0.0
    return float(yaw_deg)


def draw_tag_orientation(ax, x: float, y: float, yaw: float, length: float = 0.18) -> None:
    hx = x + length * np.cos(yaw)
    hy = y + length * np.sin(yaw)
    ax.plot([x, hx], [y, hy], "-", linewidth=2)
    ax.plot([hx], [hy], ">", markersize=6)


def main():
    estimator = AprilTagGlobalPoseEstimator(
        tag_map=TAG_MAP,
        calibration_file=CALIBRATION_FILE,
        tag_size=TAG_SIZE,
        camera_in_robot=CAMERA_IN_ROBOT,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    plt.ion()
    fig, (ax_xy, ax_pos, ax_yaw) = plt.subplots(3, 1, figsize=(7, 10))

    # -----------------------------------------------------------------
    # XY trajectory plot
    # -----------------------------------------------------------------
    ax_xy.set_title("Live Global Position")
    ax_xy.set_xlabel("global x [m]")
    ax_xy.set_ylabel("global y [m]")
    ax_xy.grid(True)
    # ax_xy.axis("equal")

    tag_xs = [pose["x"] for pose in TAG_MAP.values()]
    tag_ys = [pose["y"] for pose in TAG_MAP.values()]
    x_min, x_max = min(tag_xs) - 5.0, max(tag_xs) + 5.0
    y_min, y_max = min(tag_ys) - 5.0, max(tag_ys) + 5.0
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)

    ax_xy.scatter(tag_xs, tag_ys, marker="s", s=100, label="AprilTags")

    for tag_id, pose in TAG_MAP.items():
        ax_xy.text(pose["x"], pose["y"], f"  ID {tag_id}", fontsize=9)
        draw_tag_orientation(ax_xy, pose["x"], pose["y"], pose["yaw"], length=0.20)

    trail_line, = ax_xy.plot([], [], "-", linewidth=1.5, label="Robot trail")
    robot_point, = ax_xy.plot([], [], "o", markersize=8, label="Robot")
    heading_line, = ax_xy.plot([], [], "-", linewidth=2, label="Robot heading")
    ax_xy.legend(loc="upper right")

    # -----------------------------------------------------------------
    # x, y, z live plot
    # -----------------------------------------------------------------
    ax_pos.set_title("Live Global Position States")
    ax_pos.set_xlabel("sample")
    ax_pos.set_ylabel("position [m]")
    ax_pos.grid(True)

    x_line, = ax_pos.plot([], [], label="x [m]")
    y_line, = ax_pos.plot([], [], label="y [m]")
    z_line, = ax_pos.plot([], [], label="z [m]")
    ax_pos.legend(loc="upper right")

    # -----------------------------------------------------------------
    # yaw live plot
    # -----------------------------------------------------------------
    ax_yaw.set_title("Live Global Yaw")
    ax_yaw.set_xlabel("sample")
    ax_yaw.set_ylabel("yaw [deg, 0-360)")
    ax_yaw.grid(True)

    yaw_line, = ax_yaw.plot([], [], label="yaw [deg, 0-360)")
    ax_yaw.legend(loc="upper right")

    max_hist = 200
    xs = deque(maxlen=max_hist)
    ys = deque(maxlen=max_hist)
    zs = deque(maxlen=max_hist)
    yaws_deg = deque(maxlen=max_hist)
    ts = deque(maxlen=max_hist)

    sample_idx = 0

    print("Press 'q' in the OpenCV window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # draw=False prevents duplicate overlay from april_tag_pose_est.py
        success, measurement, debug_frame = estimator.process_frame(frame, draw=False)

        if success:
            x = float(measurement["x"])
            y = float(measurement["y"])
            z = float(measurement["z"])

            yaw = float(measurement["yaw"])
            yaw_display_deg = yaw_deg_360(yaw)

            xs.append(x)
            ys.append(y)
            zs.append(z)
            yaws_deg.append(yaw_display_deg)
            ts.append(sample_idx)
            sample_idx += 1

            # ---------------------------------------------------------
            # Update XY plot
            # ---------------------------------------------------------
            trail_line.set_data(list(xs), list(ys))
            robot_point.set_data([x], [y])

            heading_len = 1.0
            hx = x + heading_len * np.cos(yaw)
            hy = y + heading_len * np.sin(yaw)
            heading_line.set_data([x, hx], [y, hy])

            # ---------------------------------------------------------
            # Update position time series
            # ---------------------------------------------------------
            x_line.set_data(list(ts), list(xs))
            y_line.set_data(list(ts), list(ys))
            z_line.set_data(list(ts), list(zs))

            ax_pos.set_xlim(max(0, sample_idx - max_hist), max(max_hist, sample_idx))

            pos_vals = list(xs) + list(ys) + list(zs)
            pos_vmin = min(pos_vals)
            pos_vmax = max(pos_vals)
            if abs(pos_vmax - pos_vmin) < 1e-8:
                pos_vmin -= 1.0
                pos_vmax += 1.0
            else:
                pos_pad = 0.1 * (pos_vmax - pos_vmin)
                pos_vmin -= pos_pad
                pos_vmax += pos_pad
            ax_pos.set_ylim(pos_vmin, pos_vmax)

            # ---------------------------------------------------------
            # Update yaw time series
            # ---------------------------------------------------------
            yaw_line.set_data(list(ts), list(yaws_deg))

            ax_yaw.set_xlim(max(0, sample_idx - max_hist), max(max_hist, sample_idx))

            yaw_vals = list(yaws_deg)
            yaw_vmin = min(yaw_vals)
            yaw_vmax = max(yaw_vals)
            if abs(yaw_vmax - yaw_vmin) < 1e-8:
                yaw_vmin -= 5.0
                yaw_vmax += 5.0
            else:
                yaw_pad = 0.1 * (yaw_vmax - yaw_vmin)
                yaw_vmin -= yaw_pad
                yaw_vmax += yaw_pad
            ax_yaw.set_ylim(yaw_vmin, yaw_vmax)

            # ---------------------------------------------------------
            # Draw detected tag outlines ourselves
            # ---------------------------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = estimator.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[estimator.fx, estimator.fy, estimator.cx, estimator.cy],
                tag_size=estimator.tag_size,
            )

            visible_ids = []
            for r in results:
                if int(r.tag_id) not in TAG_MAP:
                    continue

                visible_ids.append(int(r.tag_id))
                corners = r.corners.astype(int)
                for i in range(4):
                    cv2.line(
                        debug_frame,
                        tuple(corners[i]),
                        tuple(corners[(i + 1) % 4]),
                        (0, 255, 0),
                        2,
                    )

                center_xy = (int(r.center[0]), int(r.center[1]))
                cv2.putText(
                    debug_frame,
                    f"ID: {int(r.tag_id)}",
                    center_xy,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # ---------------------------------------------------------
            # Console output
            # ---------------------------------------------------------
            print(
                f"FUSED: global x={x:.3f}, global y={y:.3f}, global z={z:.3f}, "
                f"global yaw={yaw_display_deg:.1f} deg, tags={measurement['tag_ids']}"
            )

            # ---------------------------------------------------------
            # Single clean OpenCV overlay
            # ---------------------------------------------------------
            cv2.putText(
                debug_frame,
                f"global x = {x:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                debug_frame,
                f"global y = {y:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                debug_frame,
                f"global z = {z:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                debug_frame,
                f"global yaw = {yaw_display_deg:.1f} deg",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
            )
            cv2.putText(
                debug_frame,
                f"Visible tags: {visible_ids}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
        else:
            cv2.putText(
                debug_frame,
                "No mapped AprilTag detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("AprilTag Global Pose Test", debug_frame)

        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()