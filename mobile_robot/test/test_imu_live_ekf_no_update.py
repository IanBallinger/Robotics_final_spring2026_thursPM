import os
import sys
import time
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
import serial.tools.list_ports


# -------------------------------------------------------------------
# Import EKF from mobile_robot/src/localization/kf.py
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from localization.kf import ExtendedKalmanFilter


# =========================
# User settings
# =========================
SERIAL_PORT = "/dev/tty.usbmodem101"   # change this
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.1

MAX_POINTS = 3000
PRINT_INTERVAL = 1.0

Q = np.diag([
    1e-5,
    1e-5,
    1e-4,
    1e-2,
    1e-2,
    1e-2,
])

# P0 = np.diag([
#     1e-6,
#     1e-6,
#     1e-6,
#     1e-3,
#     1e-3,
#     1e-3,
# ])

P0 = np.zeros((6, 6))

R_IMU_VEL = np.diag([0.05, 0.05])
R_TAG = np.diag([0.05, 0.05, 0.01])


# =========================
# Shared data
# =========================
data_lock = threading.Lock()
running = True

traj_x = deque(maxlen=MAX_POINTS)
traj_y = deque(maxlen=MAX_POINTS)

t_hist = deque(maxlen=MAX_POINTS)
ax_hist = deque(maxlen=MAX_POINTS)
ay_hist = deque(maxlen=MAX_POINTS)
wz_hist = deque(maxlen=MAX_POINTS)

latest_info_lines = deque(maxlen=20)


def list_ports():
    print("Available serial ports:")
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device} | {p.description}")


def parse_imu_line(line: str):
    """
    Expected format:
        time_us ax ay az wx wy wz
    """
    parts = line.strip().split()
    if len(parts) != 7:
        return None

    try:
        return {
            "time_us": int(parts[0]),
            "ax": float(parts[1]),
            "ay": float(parts[2]),
            "az": float(parts[3]),
            "wx": float(parts[4]),
            "wy": float(parts[5]),
            "wz": float(parts[6]),
        }
    except ValueError:
        return None


def serial_reader_thread(ser, ekf):
    global running

    last_time_us = None
    t0_us = None
    last_print_wall = time.time()

    while running:
        try:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            data = parse_imu_line(line)
            if data is None:
                with data_lock:
                    latest_info_lines.append(line)
                continue

            time_us = data["time_us"]
            ax_body = data["ax"]
            ay_body = data["ay"]
            wz = data["wz"]

            if t0_us is None:
                t0_us = time_us

            if last_time_us is None:
                last_time_us = time_us
                continue

            dt = (time_us - last_time_us) * 1e-6
            last_time_us = time_us

            if dt <= 0.0 or dt > 0.2:
                continue

            ekf.predict(
                ax_body=ax_body,
                ay_body=ay_body,
                omega_meas=wz,
                dt=dt,
            )

            state = ekf.get_state()
            px, py, theta, vx_body, vy_body, omega = state

            t_sec = (time_us - t0_us) * 1e-6

            with data_lock:
                traj_x.append(px)
                traj_y.append(py)

                t_hist.append(t_sec)
                ax_hist.append(ax_body)
                ay_hist.append(ay_body)
                wz_hist.append(wz)

            now_wall = time.time()
            if now_wall - last_print_wall >= PRINT_INTERVAL:
                print(
                    f"t={t_sec:7.2f} s | "
                    f"x={px:8.3f}, y={py:8.3f}, theta={np.degrees(theta):7.2f} deg | "
                    f"vx_b={vx_body:7.3f}, vy_b={vy_body:7.3f}, omega={omega:7.3f}"
                )
                last_print_wall = now_wall

        except Exception as e:
            print(f"[SERIAL ERROR] {e}")
            time.sleep(0.1)


def main():
    global running

    list_ports()
    print(f"\nOpening {SERIAL_PORT} @ {BAUD_RATE}")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)

    # Give the board time to reset after opening serial
    time.sleep(2.5)

    ekf = ExtendedKalmanFilter(
        dt=0.01,
        initial_state=np.zeros(6),
        initial_covariance=P0,
        process_noise=Q,
        R_imu_vel=R_IMU_VEL,
        R_tag=R_TAG,
        R_imu_vel_omega=np.diag([0.05, 0.05, 0.02]),
    )

    thread = threading.Thread(target=serial_reader_thread, args=(ser, ekf), daemon=True)
    thread.start()

    fig, (ax_traj_plot, ax_imu_plot) = plt.subplots(2, 1, figsize=(8, 9))

    def update_plot(_frame):
        with data_lock:
            xs = list(traj_x)
            ys = list(traj_y)

            ts = list(t_hist)
            axs = list(ax_hist)
            ays = list(ay_hist)
            wzs = list(wz_hist)

            info_lines = list(latest_info_lines)

        # trajectory
        ax_traj_plot.clear()
        ax_traj_plot.set_title("Live EKF Estimated Vehicle Motion (Global Coordinates)")
        ax_traj_plot.set_xlabel("x [m]")
        ax_traj_plot.set_ylabel("y [m]")
        ax_traj_plot.grid(True)

        if xs:
            ax_traj_plot.plot(xs, ys, linewidth=2)
            ax_traj_plot.scatter([xs[0]], [ys[0]], marker="o", s=40, label="start")
            ax_traj_plot.scatter([xs[-1]], [ys[-1]], marker="x", s=60, label="current")
            ax_traj_plot.legend(loc="best")

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            pad_x = max(0.5, 0.1 * max(1.0, x_max - x_min))
            pad_y = max(0.5, 0.1 * max(1.0, y_max - y_min))

            ax_traj_plot.set_xlim(x_min - pad_x, x_max + pad_x)
            ax_traj_plot.set_ylim(y_min - pad_y, y_max + pad_y)
            ax_traj_plot.set_aspect("equal", adjustable="box")
        else:
            ax_traj_plot.set_xlim(-1, 1)
            ax_traj_plot.set_ylim(-1, 1)
            ax_traj_plot.set_aspect("equal", adjustable="box")
            ax_traj_plot.text(0, 0, "No parsed trajectory data yet", ha="center")

        # imu
        ax_imu_plot.clear()
        ax_imu_plot.set_title("Live IMU Readings")
        ax_imu_plot.set_xlabel("time [s]")
        ax_imu_plot.set_ylabel("value")
        ax_imu_plot.grid(True)

        if ts:
            ax_imu_plot.plot(ts, axs, label="ax [m/s²]")
            ax_imu_plot.plot(ts, ays, label="ay [m/s²]")
            ax_imu_plot.plot(ts, wzs, label="omega [rad/s]")
            ax_imu_plot.legend(loc="best")
            ax_imu_plot.set_xlim(0, max(ts[-1], 1.0))
        else:
            msg = "No parsed IMU samples yet"
            if info_lines:
                msg += "\nLast non-data line:\n" + info_lines[-1]
            ax_imu_plot.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_imu_plot.transAxes)

    ani = FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)

    def on_close(_event):
        global running
        running = False
        print("Closing serial port.")
        ser.close()

    fig.canvas.mpl_connect("close_event", on_close)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()