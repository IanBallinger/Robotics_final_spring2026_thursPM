import sys
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports


# ---------------------------------------------------------------------
# Make project imports work
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
TEST_DIR = THIS_FILE.parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.kf import ExtendedKalmanFilter
from src.localization.april_tag_pose_est import AprilTagGlobalPoseEstimator


# =========================
# User settings
# =========================
SERIAL_PORT = "/dev/tty.usbmodem2101"
BAUD_RATE = 115200
TIMEOUT = 1.0
PRINT_EVERY_N = 10
MAX_POINTS = 3000
PLOT_HISTORY = 200

# TAG_MAP = {
#     4: {"x": -1.17 + 1.10, "y": 0.62 + 0.93, "z": 0.0, "yaw": -np.pi / 2},
#     5: {"x": -1.17, "y": 0.62, "z": 0.0, "yaw": 0.0},
#     6: {"x": -1.17, "y": 0.62 + 0.59, "z": 0.0, "yaw": 0.0},
#     9 : {"x": -0.48, "y": -0.08, "z": 0.0, "yaw": np.pi / 2},
# }

TAG_MAP = {
    4: {"x": 0, "y": 0, "z": 0.0, "yaw": 0},
    5: {"x": 0, "y": -1.0, "z": 0.0, "yaw": 0},
}

CAMERA_IN_ROBOT = {"x": 0.0, "y": 0.0, "z": 0.0}
CALIBRATION_FILE = str(PROJECT_ROOT / "src" / "localization" / "camera_calibration_live.npz")
TAG_SIZE = 0.17  # physical tag size in meters

# State: [x, y, theta, vx_body, vy_body, omega]
Q = np.diag([1e-5, 1e-5, 1e-4, 1e-2, 1e-2, 1e-2])
P0 = np.diag([0.05, 0.05, 0.02, 0.5, 0.5, 0.1])
R_IMU_VEL = np.diag([0.2, 0.2])
R_TAG = np.diag([0.05, 0.05, 0.01])
R_IMU_VEL_OMEGA = np.diag([0.3, 0.3, 0.08])

USE_PSEUDO_IMU_VELOCITY_UPDATE = True
VELOCITY_DECAY = 0.995
STATIONARY_ACC_THRESH = 0.08
STATIONARY_GYRO_THRESH = 0.05

INITIAL_STATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# =========================
# Shared data
# =========================
data_lock = threading.Lock()
running = True

# Fused EKF history
fused_x_hist = deque(maxlen=MAX_POINTS)
fused_y_hist = deque(maxlen=MAX_POINTS)
fused_z_hist = deque(maxlen=MAX_POINTS)
fused_yaw_hist = deque(maxlen=MAX_POINTS)

# IMU-only dead reckoning history
imu_x_hist = deque(maxlen=MAX_POINTS)
imu_y_hist = deque(maxlen=MAX_POINTS)
imu_z_hist = deque(maxlen=MAX_POINTS)
imu_yaw_hist = deque(maxlen=MAX_POINTS)

# AprilTag measurement history
tag_x_hist = deque(maxlen=MAX_POINTS)
tag_y_hist = deque(maxlen=MAX_POINTS)
tag_z_hist = deque(maxlen=MAX_POINTS)
tag_yaw_hist = deque(maxlen=MAX_POINTS)

# raw IMU histories
imu_t_hist = deque(maxlen=MAX_POINTS)
ax_hist = deque(maxlen=MAX_POINTS)
ay_hist = deque(maxlen=MAX_POINTS)
wz_hist = deque(maxlen=MAX_POINTS)

latest_info_lines = deque(maxlen=20)
latest_visible_tag_ids = []
last_debug_text = "Waiting for data"


# =========================
# Utility helpers
# =========================
def list_ports():
    print("Available serial ports:")
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device} | {p.description}")


def parse_imu_line(line: str):
    """
    Expected line format:
    time_us\tax\tay\taz\twx\twy\twz
    """
    parts = line.strip().split()
    if len(parts) != 7:
        return None

    try:
        return {
            "time_us": int(parts[0]),
            "ax": round(float(parts[1]), 2),
            "ay": round(float(parts[2]), 2),
            "az": round(float(parts[3]), 2),
            "wx": round(float(parts[4]), 2),
            "wy": round(float(parts[5]), 2),
            "wz": round(float(parts[6]), 2),
        }
    except ValueError:
        return None


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


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


# =========================
# IMU serial thread
# =========================
def serial_reader_thread(ser, ekf):
    global running, last_debug_text

    last_time_us = None
    t0_us = None
    valid_count = 0

    vx_meas = 0.0
    vy_meas = 0.0

    imu_x = INITIAL_STATE[0]
    imu_y = INITIAL_STATE[1]
    imu_z = 0.0
    imu_yaw =  INITIAL_STATE[2]
    imu_vx_body = 0.0
    imu_vy_body = 0.0

    while running:
        try:
            raw = ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            data = parse_imu_line(line)
            if data is None:
                with data_lock:
                    latest_info_lines.append(line)
                print(f"[INFO] {line}")
                continue

            valid_count += 1

            time_us = data["time_us"]

            # Keep the mapping from the original IMU EKF test file
            ax_body = -data["ax"]
            ay_body = data["az"]
            wz = data["wy"]

            if t0_us is None:
                t0_us = time_us

            if last_time_us is None:
                last_time_us = time_us
                continue

            dt = (time_us - last_time_us) * 1e-6
            last_time_us = time_us

            if dt <= 0.0 or dt > 0.2:
                continue

            # EKF predict
            ekf.predict(ax_body=ax_body, ay_body=ay_body, omega_meas=wz, dt=dt)

            # Pseudo IMU velocity update
            if USE_PSEUDO_IMU_VELOCITY_UPDATE:
                vx_meas = VELOCITY_DECAY * vx_meas + ax_body * dt
                vy_meas = VELOCITY_DECAY * vy_meas + ay_body * dt

                if (
                    abs(ax_body) < STATIONARY_ACC_THRESH
                    and abs(ay_body) < STATIONARY_ACC_THRESH
                    and abs(wz) < STATIONARY_GYRO_THRESH
                ):
                    vx_meas = 0.0
                    vy_meas = 0.0

                ekf.update_imu_velocity_and_omega(
                    vx_body_meas=vx_meas,
                    vy_body_meas=vy_meas,
                    omega_meas=wz,
                )

            fused_state = ekf.get_state()
            fused_x, fused_y, fused_theta, _, _, _ = fused_state

            # IMU-only dead reckoning
            imu_vx_body = VELOCITY_DECAY * imu_vx_body + ax_body * dt
            imu_vy_body = VELOCITY_DECAY * imu_vy_body + ay_body * dt

            if (
                abs(ax_body) < STATIONARY_ACC_THRESH
                and abs(ay_body) < STATIONARY_ACC_THRESH
                and abs(wz) < STATIONARY_GYRO_THRESH
            ):
                imu_vx_body = 0.0
                imu_vy_body = 0.0

            imu_yaw = wrap_angle(imu_yaw + wz * dt)
            c = np.cos(imu_yaw)
            s = np.sin(imu_yaw)
            imu_x += (c * imu_vx_body - s * imu_vy_body) * dt
            imu_y += (s * imu_vx_body + c * imu_vy_body) * dt

            t_sec = (time_us - t0_us) * 1e-6

            with data_lock:
                fused_x_hist.append(float(fused_x))
                fused_y_hist.append(float(fused_y))
                fused_z_hist.append(0.0)
                fused_yaw_hist.append(float(fused_theta))

                imu_x_hist.append(float(imu_x))
                imu_y_hist.append(float(imu_y))
                imu_z_hist.append(float(imu_z))
                imu_yaw_hist.append(float(imu_yaw))

                imu_t_hist.append(float(t_sec))
                ax_hist.append(float(ax_body))
                ay_hist.append(float(ay_body))
                wz_hist.append(float(wz))

                last_debug_text = (
                    f"EKF x={fused_x:.2f}, y={fused_y:.2f}, yaw={yaw_deg_360(fused_theta):.1f} deg | "
                    f"IMU x={imu_x:.2f}, y={imu_y:.2f}, yaw={yaw_deg_360(imu_yaw):.1f} deg"
                )

            if valid_count % PRINT_EVERY_N == 0:
                rate_hz = 1.0 / dt if dt > 0 else 0.0
                print(
                    f"IMU t={t_sec:7.2f} s | dt={dt:7.4f} s ({rate_hz:6.1f} Hz) | "
                    f"EKF x={fused_x:7.3f}, y={fused_y:7.3f}, yaw={yaw_deg_360(fused_theta):6.1f} deg | "
                    f"IMU x={imu_x:7.3f}, y={imu_y:7.3f}, yaw={yaw_deg_360(imu_yaw):6.1f} deg | "
                    f"ax={ax_body:7.3f}, ay={ay_body:7.3f}, wz={wz:7.3f}"
                )

        except Exception as e:
            print(f"[SERIAL ERROR] {e}")
            time.sleep(0.1)


# =========================
# Main camera + plot loop
# =========================
def main():
    global running, latest_visible_tag_ids, last_debug_text

    list_ports()
    print(f"\nOpening {SERIAL_PORT} @ {BAUD_RATE}")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2.0)

    ekf = ExtendedKalmanFilter(
        dt=0.01,
        initial_state=INITIAL_STATE,
        initial_covariance=P0,
        process_noise=Q,
        R_imu_vel=R_IMU_VEL,
        R_tag=R_TAG,
        R_imu_vel_omega=R_IMU_VEL_OMEGA,
    )

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

    thread = threading.Thread(target=serial_reader_thread, args=(ser, ekf), daemon=True)
    thread.start()

    plt.ion()
    fig, ((ax_xy, ax_pos), (ax_yaw, ax_imu)) = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(right=0.80, wspace=0.35, hspace=0.30)

    # XY plot
    ax_xy.set_title("Global XY: EKF vs IMU vs AprilTag")
    ax_xy.set_xlabel("global x [m]")
    ax_xy.set_ylabel("global y [m]")
    ax_xy.grid(True)

    tag_xs_static = [pose["x"] for pose in TAG_MAP.values()]
    tag_ys_static = [pose["y"] for pose in TAG_MAP.values()]
    ax_xy.scatter(tag_xs_static, tag_ys_static, marker="s", s=100, label="AprilTags in map")
    for tag_id, pose in TAG_MAP.items():
        ax_xy.text(pose["x"], pose["y"], f"  ID {tag_id}", fontsize=9)
        draw_tag_orientation(ax_xy, pose["x"], pose["y"], pose["yaw"], length=0.20)

    fused_trail_line, = ax_xy.plot([], [], "-", linewidth=1.8, label="EKF estimate")
    imu_trail_line, = ax_xy.plot([], [], "--", linewidth=1.5, label="IMU only")
    fused_point, = ax_xy.plot([], [], "o", markersize=7, label="EKF current")
    imu_point, = ax_xy.plot([], [], "x", markersize=7, label="IMU current")
    tag_point, = ax_xy.plot([], [], "s", markersize=7, label="AprilTag meas")
    fused_heading_line, = ax_xy.plot([], [], "-", linewidth=2, label="EKF heading")
    ax_xy.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # Position plot
    ax_pos.set_title("Position states")
    ax_pos.set_xlabel("sample")
    ax_pos.set_ylabel("position [m]")
    ax_pos.grid(True)
    fused_x_line, = ax_pos.plot([], [], label="EKF x")
    fused_y_line, = ax_pos.plot([], [], label="EKF y")
    imu_x_line, = ax_pos.plot([], [], label="IMU x")
    imu_y_line, = ax_pos.plot([], [], label="IMU y")
    tag_x_line, = ax_pos.plot([], [], linestyle="None", marker="o", markersize=4, label="Tag x")
    tag_y_line, = ax_pos.plot([], [], linestyle="None", marker="x", markersize=4, label="Tag y")
    ax_pos.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # Yaw plot
    ax_yaw.set_title("Yaw")
    ax_yaw.set_xlabel("sample")
    ax_yaw.set_ylabel("yaw [deg, 0-360)")
    ax_yaw.grid(True)
    fused_yaw_line, = ax_yaw.plot([], [], label="EKF yaw")
    imu_yaw_line, = ax_yaw.plot([], [], label="IMU yaw")
    tag_yaw_line, = ax_yaw.plot([], [], linestyle="None", marker="o", markersize=4, label="Tag yaw")
    ax_yaw.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # IMU plot
    ax_imu.set_title("Raw IMU signals")
    ax_imu.set_xlabel("time [s]")
    ax_imu.set_ylabel("value")
    ax_imu.grid(True)
    ax_line, = ax_imu.plot([], [], label="ax_body [m/s^2]")
    ay_line, = ax_imu.plot([], [], label="ay_body [m/s^2]")
    wz_line, = ax_imu.plot([], [], label="wz [rad/s]")
    ax_imu.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    print("Press 'q' in the OpenCV window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        success, measurement, debug_frame = estimator.process_frame(frame, draw=False)

        # Draw visible tag outlines using the same undistorted image path used by the estimator
        gray = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2GRAY)
        results = estimator.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[
                float(estimator._new_camera_matrix[0, 0]),
                float(estimator._new_camera_matrix[1, 1]),
                float(estimator._new_camera_matrix[0, 2]),
                float(estimator._new_camera_matrix[1, 2]),
            ],
            tag_size=estimator.tag_size,
        )

        visible_ids = []
        for r in results:
            if int(r.tag_id) not in TAG_MAP:
                continue
            visible_ids.append(int(r.tag_id))
            corners = r.corners.astype(int)
            for i in range(4):
                cv2.line(debug_frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
            center_xy = (int(r.center[0]), int(r.center[1]))
            cv2.putText(debug_frame, f"ID: {int(r.tag_id)}", center_xy,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        with data_lock:
            latest_visible_tag_ids = visible_ids

        if success:
            tag_x = float(measurement["x"])
            tag_y = float(measurement["y"])
            tag_z = float(measurement["z"])
            tag_yaw = float(measurement["yaw"])

            ekf.update_apriltag_pose(x_meas=tag_x, y_meas=tag_y, theta_meas=tag_yaw)
            fused_state = ekf.get_state()

            with data_lock:
                tag_x_hist.append(tag_x)
                tag_y_hist.append(tag_y)
                tag_z_hist.append(tag_z)
                tag_yaw_hist.append(tag_yaw)

                if len(fused_x_hist) > 0:
                    fused_x_hist[-1] = float(fused_state[0])
                    fused_y_hist[-1] = float(fused_state[1])
                    fused_z_hist[-1] = float(tag_z)
                    fused_yaw_hist[-1] = float(fused_state[2])
                else:
                    fused_x_hist.append(float(fused_state[0]))
                    fused_y_hist.append(float(fused_state[1]))
                    fused_z_hist.append(float(tag_z))
                    fused_yaw_hist.append(float(fused_state[2]))

                last_debug_text = (
                    f"EKF x={fused_state[0]:.2f}, y={fused_state[1]:.2f}, z={tag_z:.2f}, "
                    f"yaw={yaw_deg_360(fused_state[2]):.1f} deg | tags={measurement['tag_ids']}"
                )

            print(
                f"TAG: x={tag_x:.3f}, y={tag_y:.3f}, z={tag_z:.3f}, yaw={yaw_deg_360(tag_yaw):.1f} deg, "
                f"tags={measurement['tag_ids']} | "
                f"FUSED: x={fused_state[0]:.3f}, y={fused_state[1]:.3f}, yaw={yaw_deg_360(fused_state[2]):.1f} deg"
            )
        else:
            with data_lock:
                last_debug_text = "No mapped AprilTag detected"

        with data_lock:
            fused_xs = list(fused_x_hist)[-PLOT_HISTORY:]
            fused_ys = list(fused_y_hist)[-PLOT_HISTORY:]
            fused_yaws = [yaw_deg_360(v) for v in list(fused_yaw_hist)[-PLOT_HISTORY:]]

            imu_xs = list(imu_x_hist)[-PLOT_HISTORY:]
            imu_ys = list(imu_y_hist)[-PLOT_HISTORY:]
            imu_yaws = [yaw_deg_360(v) for v in list(imu_yaw_hist)[-PLOT_HISTORY:]]

            tag_xs = list(tag_x_hist)[-PLOT_HISTORY:]
            tag_ys = list(tag_y_hist)[-PLOT_HISTORY:]
            tag_yaws = [yaw_deg_360(v) for v in list(tag_yaw_hist)[-PLOT_HISTORY:]]

            imu_ts = list(imu_t_hist)[-PLOT_HISTORY:]
            axs_vals = list(ax_hist)[-PLOT_HISTORY:]
            ays_vals = list(ay_hist)[-PLOT_HISTORY:]
            wz_vals = list(wz_hist)[-PLOT_HISTORY:]

            info_lines = list(latest_info_lines)
            status_text = last_debug_text
            visible_ids_copy = list(latest_visible_tag_ids)

        # XY plot
        fused_trail_line.set_data(fused_xs, fused_ys)
        imu_trail_line.set_data(imu_xs, imu_ys)

        if fused_xs:
            fused_point.set_data([fused_xs[-1]], [fused_ys[-1]])
            yaw_rad = np.radians(fused_yaws[-1])
            heading_len = 0.5
            hx = fused_xs[-1] + heading_len * np.cos(yaw_rad)
            hy = fused_ys[-1] + heading_len * np.sin(yaw_rad)
            fused_heading_line.set_data([fused_xs[-1], hx], [fused_ys[-1], hy])
        else:
            fused_point.set_data([], [])
            fused_heading_line.set_data([], [])

        if imu_xs:
            imu_point.set_data([imu_xs[-1]], [imu_ys[-1]])
        else:
            imu_point.set_data([], [])

        if tag_xs:
            tag_point.set_data([tag_xs[-1]], [tag_ys[-1]])
        else:
            tag_point.set_data([], [])

        all_xy = []
        if fused_xs:
            all_xy.extend(zip(fused_xs, fused_ys))
        if imu_xs:
            all_xy.extend(zip(imu_xs, imu_ys))
        if tag_xs:
            all_xy.extend(zip(tag_xs, tag_ys))
        all_xy.extend(zip(tag_xs_static, tag_ys_static))

        if all_xy:
            all_x = [p[0] for p in all_xy]
            all_y = [p[1] for p in all_xy]
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            pad_x = max(0.5, 0.1 * max(1.0, x_max - x_min))
            pad_y = max(0.5, 0.1 * max(1.0, y_max - y_min))
            ax_xy.set_xlim(x_min - pad_x, x_max + pad_x)
            ax_xy.set_ylim(y_min - pad_y, y_max + pad_y)
            ax_xy.set_aspect("equal", adjustable="box")

        # Position plot with independent x-axes per source length
        fused_samples = list(range(len(fused_xs)))
        imu_samples = list(range(len(imu_xs)))
        tag_samples = list(range(len(tag_xs)))

        fused_x_line.set_data(fused_samples, fused_xs)
        fused_y_line.set_data(fused_samples, fused_ys)
        imu_x_line.set_data(imu_samples, imu_xs)
        imu_y_line.set_data(imu_samples, imu_ys)
        tag_x_line.set_data(tag_samples, tag_xs)
        tag_y_line.set_data(tag_samples, tag_ys)

        pos_vals = fused_xs + fused_ys + imu_xs + imu_ys + tag_xs + tag_ys
        max_pos_len = max([len(fused_samples), len(imu_samples), len(tag_samples), 1])
        ax_pos.set_xlim(0, max(PLOT_HISTORY, max_pos_len))
        if pos_vals:
            pos_min = min(pos_vals)
            pos_max = max(pos_vals)
            if abs(pos_max - pos_min) < 1e-8:
                pos_min -= 1.0
                pos_max += 1.0
            else:
                pad = 0.1 * (pos_max - pos_min)
                pos_min -= pad
                pos_max += pad
            ax_pos.set_ylim(pos_min, pos_max)

        # Yaw plot
        fused_yaw_samples = list(range(len(fused_yaws)))
        imu_yaw_samples = list(range(len(imu_yaws)))
        tag_yaw_samples = list(range(len(tag_yaws)))
        fused_yaw_line.set_data(fused_yaw_samples, fused_yaws)
        imu_yaw_line.set_data(imu_yaw_samples, imu_yaws)
        tag_yaw_line.set_data(tag_yaw_samples, tag_yaws)

        yaw_vals = fused_yaws + imu_yaws + tag_yaws
        max_yaw_len = max([len(fused_yaw_samples), len(imu_yaw_samples), len(tag_yaw_samples), 1])
        ax_yaw.set_xlim(0, max(PLOT_HISTORY, max_yaw_len))
        if yaw_vals:
            yaw_min = min(yaw_vals)
            yaw_max = max(yaw_vals)
            if abs(yaw_max - yaw_min) < 1e-8:
                yaw_min -= 5.0
                yaw_max += 5.0
            else:
                pad = 0.1 * (yaw_max - yaw_min)
                yaw_min -= pad
                yaw_max += pad
            ax_yaw.set_ylim(yaw_min, yaw_max)

        # IMU plot
        ax_line.set_data(imu_ts, axs_vals)
        ay_line.set_data(imu_ts, ays_vals)
        wz_line.set_data(imu_ts, wz_vals)
        if imu_ts:
            ax_imu.set_xlim(max(0.0, imu_ts[0]), max(1.0, imu_ts[-1]))
            imu_vals = axs_vals + ays_vals + wz_vals
            if imu_vals:
                vmin = min(imu_vals)
                vmax = max(imu_vals)
                if abs(vmax - vmin) < 1e-8:
                    vmin -= 1.0
                    vmax += 1.0
                else:
                    pad = 0.1 * (vmax - vmin)
                    vmin -= pad
                    vmax += pad
                ax_imu.set_ylim(vmin, vmax)
        else:
            ax_imu.cla()
            ax_imu.set_title("Raw IMU signals")
            ax_imu.set_xlabel("time [s]")
            ax_imu.set_ylabel("value")
            ax_imu.grid(True)
            if info_lines:
                ax_imu.text(0.5, 0.5, "No parsed IMU samples yet\nLast non-data line:\n" + info_lines[-1],
                            ha="center", va="center", transform=ax_imu.transAxes)

        # OpenCV overlay
        cv2.putText(debug_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(debug_frame, f"Visible tags: {visible_ids_copy}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.imshow("Fused IMU + AprilTag EKF Test", debug_frame)

        fig.canvas.draw()
        fig.canvas.flush_events()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    running = False
    time.sleep(0.2)
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    print("Closing serial port.")
    ser.close()


if __name__ == "__main__":
    main()
