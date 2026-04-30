#!/usr/bin/env python
"""Test the 2D EKF with IMU-only updates and visualize the estimate.

Modes
-----
1. Simulation mode (default)
   Simulates a robot moving in the plane with synthetic IMU measurements.
2. Live serial mode (`--port ...`)
   Reads IMU data from the ESP32 over the existing serial connection layer and
   runs the filter online with visualization.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from localization import (  # noqa: E402
    ExtendedKalmanFilter2D,
    IMUMeasurement,
)
from serial_connection.serial_con import SerialConnect  # noqa: E402


def make_filter(filter_name: str, gyro_noise: float):
    if filter_name != "ekf":
        raise ValueError(f"unsupported localization filter '{filter_name}'; only 'ekf' is supported")
    return ExtendedKalmanFilter2D(
        initial_state=np.zeros(6),
        initial_covariance=np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
        process_noise=np.diag([5e-4, 5e-4, 5e-4, 8e-3, 8e-3, 4e-3]),
        gyro_measurement_noise=np.array([[gyro_noise**2]], dtype=float),
    )


def simulate_truth_step(state: np.ndarray, ax_body: float, ay_body: float, wz: float, dt: float) -> np.ndarray:
    x, y, yaw, vx_body, vy_body, _ = state
    c = np.cos(yaw)
    s = np.sin(yaw)

    x_next = x + dt * (c * vx_body - s * vy_body)
    y_next = y + dt * (s * vx_body + c * vy_body)
    yaw_next = yaw + dt * wz
    vx_next = vx_body + dt * ax_body
    vy_next = vy_body + dt * ay_body
    return np.array([x_next, y_next, yaw_next, vx_next, vy_next, wz], dtype=float)


def commanded_motion(t: float) -> tuple[float, float, float]:
    ax = 0.0
    ay = 0.0
    wz = 0.0

    if t < 2.0:
        ax = 0.35
    elif t < 4.0:
        ax = -0.20
    elif t < 6.0:
        ay = 0.25
    elif t < 8.5:
        wz = 0.45
        ax = 0.10
    elif t < 11.0:
        wz = -0.35
        ay = -0.15
    else:
        ax = -0.05
        ay = 0.05
        wz = 0.10

    return ax, ay, wz


def setup_sim_plot(title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.28)
    axes = {
        "xy": fig.add_subplot(gs[0, 0]),
        "poserr": fig.add_subplot(gs[0, 1]),
        "yaw": fig.add_subplot(gs[1, 0]),
        "vel": fig.add_subplot(gs[1, 1]),
        "accel": fig.add_subplot(gs[2, 0]),
        "gyro": fig.add_subplot(gs[2, 1]),
    }
    fig.suptitle(title, fontsize=14)
    return plt, fig, axes


def setup_live_plot(title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.28)
    axes = {
        "xy": fig.add_subplot(gs[0, 0]),
        "yaw": fig.add_subplot(gs[0, 1]),
        "vel": fig.add_subplot(gs[1, 0]),
        "accel": fig.add_subplot(gs[1, 1]),
        "gyro": fig.add_subplot(gs[2, 0]),
        "speed": fig.add_subplot(gs[2, 1]),
    }
    fig.suptitle(title, fontsize=14)
    return plt, fig, axes


def run_simulation(args) -> None:
    rng = np.random.default_rng(args.seed)
    filt = make_filter(args.filter, args.gyro_noise)

    truth = np.zeros(6, dtype=float)
    t_values = np.arange(0.0, args.duration + args.dt, args.dt)

    truth_hist = np.zeros((len(t_values), 6), dtype=float)
    est_hist = np.zeros((len(t_values), 6), dtype=float)
    accel_hist = np.zeros((len(t_values), 2), dtype=float)
    gyro_hist = np.zeros(len(t_values), dtype=float)

    truth_hist[0] = truth
    est_hist[0] = filt.get_state()

    for k in range(1, len(t_values)):
        t = t_values[k - 1]
        ax_true, ay_true, wz_true = commanded_motion(t)
        truth = simulate_truth_step(truth, ax_true, ay_true, wz_true, args.dt)

        imu = IMUMeasurement(
            ax=ax_true + rng.normal(0.0, args.accel_noise),
            ay=ay_true + rng.normal(0.0, args.accel_noise),
            wz=wz_true + rng.normal(0.0, args.gyro_noise),
        )

        filt.predict(imu, args.dt)
        filt.update_imu(imu)

        truth_hist[k] = truth
        est_hist[k] = filt.get_state()
        accel_hist[k] = [imu.ax, imu.ay]
        gyro_hist[k] = imu.wz

    pos_err = np.linalg.norm(est_hist[:, :2] - truth_hist[:, :2], axis=1)
    yaw_err = np.unwrap(est_hist[:, 2]) - np.unwrap(truth_hist[:, 2])

    print(f"filter={args.filter}")
    print(f"final true pose: x={truth_hist[-1,0]:.3f}, y={truth_hist[-1,1]:.3f}, yaw={truth_hist[-1,2]:.3f}")
    print(f"final est  pose: x={est_hist[-1,0]:.3f}, y={est_hist[-1,1]:.3f}, yaw={est_hist[-1,2]:.3f}")
    print(f"final position error: {float(pos_err[-1]):.3f} m")
    print(f"final yaw error: {float(yaw_err[-1]):.3f} rad")

    plt, fig, ax = setup_sim_plot(f"IMU-only localization test ({args.filter.upper()})")

    ax["xy"].plot(truth_hist[:, 0], truth_hist[:, 1], label="true", linewidth=2.0)
    ax["xy"].plot(est_hist[:, 0], est_hist[:, 1], label=f"{args.filter.upper()} estimate", linewidth=1.8, linestyle="--")
    ax["xy"].scatter([truth_hist[0, 0]], [truth_hist[0, 1]], c="green", s=60, label="start")
    ax["xy"].scatter([truth_hist[-1, 0]], [truth_hist[-1, 1]], c="red", s=60, label="true end")
    ax["xy"].scatter([est_hist[-1, 0]], [est_hist[-1, 1]], c="purple", s=60, label="est end")
    ax["xy"].set_title("XY trajectory")
    ax["xy"].set_xlabel("x (m)")
    ax["xy"].set_ylabel("y (m)")
    ax["xy"].axis("equal")
    ax["xy"].grid(True, alpha=0.3)
    ax["xy"].legend(fontsize=8)

    ax["poserr"].plot(t_values, pos_err, color="tab:red")
    ax["poserr"].set_title("Position error")
    ax["poserr"].set_xlabel("t (s)")
    ax["poserr"].set_ylabel("||p_est - p_true|| (m)")
    ax["poserr"].grid(True, alpha=0.3)

    ax["yaw"].plot(t_values, np.unwrap(truth_hist[:, 2]), label="true yaw")
    ax["yaw"].plot(t_values, np.unwrap(est_hist[:, 2]), "--", label="estimated yaw")
    ax["yaw"].plot(t_values, yaw_err, ":", label="yaw error")
    ax["yaw"].set_title("Yaw")
    ax["yaw"].set_xlabel("t (s)")
    ax["yaw"].set_ylabel("rad")
    ax["yaw"].grid(True, alpha=0.3)
    ax["yaw"].legend(fontsize=8)

    ax["vel"].plot(t_values, truth_hist[:, 3], label="true vx_body")
    ax["vel"].plot(t_values, est_hist[:, 3], "--", label="est vx_body")
    ax["vel"].plot(t_values, truth_hist[:, 4], label="true vy_body")
    ax["vel"].plot(t_values, est_hist[:, 4], "--", label="est vy_body")
    ax["vel"].set_title("Body-frame velocity states")
    ax["vel"].set_xlabel("t (s)")
    ax["vel"].set_ylabel("m/s")
    ax["vel"].grid(True, alpha=0.3)
    ax["vel"].legend(fontsize=8)

    ax["accel"].plot(t_values, accel_hist[:, 0], label="imu ax")
    ax["accel"].plot(t_values, accel_hist[:, 1], label="imu ay")
    ax["accel"].set_title("IMU acceleration inputs")
    ax["accel"].set_xlabel("t (s)")
    ax["accel"].set_ylabel("m/s²")
    ax["accel"].grid(True, alpha=0.3)
    ax["accel"].legend(fontsize=8)

    ax["gyro"].plot(t_values, gyro_hist, label="imu wz")
    ax["gyro"].plot(t_values, truth_hist[:, 5], "--", label="true wz")
    ax["gyro"].set_title("Yaw-rate measurement")
    ax["gyro"].set_xlabel("t (s)")
    ax["gyro"].set_ylabel("rad/s")
    ax["gyro"].grid(True, alpha=0.3)
    ax["gyro"].legend(fontsize=8)

    fig.tight_layout()
    if args.no_show:
        fig.savefig(args.save_path, dpi=150)
        print(f"wrote {args.save_path}")
    else:
        plt.show()


def run_live_serial(args) -> None:
    plt, fig, ax = setup_live_plot(f"Live IMU-only localization ({args.filter.upper()})")
    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    filt = make_filter(args.filter, args.gyro_noise)

    times = [0.0]
    est_hist = [filt.get_state()]
    accel_hist = [[0.0, 0.0]]
    gyro_hist = [0.0]

    last_time = time.monotonic()
    t0 = last_time

    with SerialConnect(
        port=args.port,
        tx_rate_hz=0.0,
        rx_publish_rate_hz=args.rx_publish_rate_hz,
        debug=args.debug_serial,
    ) as serial_con:
        print(f"Reading IMU from serial port: {serial_con.port}")
        try:
            while True:
                now = time.monotonic()
                dt = max(now - last_time, 1e-3)
                last_time = now

                msgs = serial_con.read_parsed(max_lines=128)
                if msgs:
                    imu_msg = msgs[-1]
                    imu = IMUMeasurement(ax=imu_msg.ax, ay=imu_msg.ay, wz=imu_msg.gz)
                    filt.predict(imu, dt)
                    filt.update_imu(imu)
                    accel_hist.append([imu.ax, imu.ay])
                    gyro_hist.append(imu.wz)
                else:
                    imu = IMUMeasurement(ax=0.0, ay=0.0, wz=0.0)
                    filt.predict(imu, dt)
                    accel_hist.append([0.0, 0.0])
                    gyro_hist.append(0.0)

                est = filt.get_state()
                est_hist.append(est)
                times.append(now - t0)

                est_arr = np.asarray(est_hist)
                accel_arr = np.asarray(accel_hist)
                gyro_arr = np.asarray(gyro_hist)
                t_arr = np.asarray(times)
                speed = np.hypot(est_arr[:, 3], est_arr[:, 4])

                ax["xy"].cla()
                ax["xy"].plot(est_arr[:, 0], est_arr[:, 1], color="tab:blue", label="estimated path")
                ax["xy"].scatter([est_arr[0, 0]], [est_arr[0, 1]], c="green", s=50, label="start")
                ax["xy"].scatter([est_arr[-1, 0]], [est_arr[-1, 1]], c="red", s=50, label="current")
                ax["xy"].set_title("Estimated XY trajectory")
                ax["xy"].set_xlabel("x (m)")
                ax["xy"].set_ylabel("y (m)")
                ax["xy"].axis("equal")
                ax["xy"].grid(True, alpha=0.3)
                ax["xy"].legend(fontsize=8)

                ax["yaw"].cla()
                ax["yaw"].plot(t_arr, np.unwrap(est_arr[:, 2]), label="estimated yaw")
                ax["yaw"].set_title("Estimated yaw")
                ax["yaw"].set_xlabel("t (s)")
                ax["yaw"].set_ylabel("rad")
                ax["yaw"].grid(True, alpha=0.3)
                ax["yaw"].legend(fontsize=8)

                ax["vel"].cla()
                ax["vel"].plot(t_arr, est_arr[:, 3], label="vx_body")
                ax["vel"].plot(t_arr, est_arr[:, 4], label="vy_body")
                ax["vel"].set_title("Estimated body-frame velocity")
                ax["vel"].set_xlabel("t (s)")
                ax["vel"].set_ylabel("m/s")
                ax["vel"].grid(True, alpha=0.3)
                ax["vel"].legend(fontsize=8)

                ax["accel"].cla()
                ax["accel"].plot(t_arr, accel_arr[:, 0], label="imu ax")
                ax["accel"].plot(t_arr, accel_arr[:, 1], label="imu ay")
                ax["accel"].set_title("Raw IMU acceleration")
                ax["accel"].set_xlabel("t (s)")
                ax["accel"].set_ylabel("m/s²")
                ax["accel"].grid(True, alpha=0.3)
                ax["accel"].legend(fontsize=8)

                ax["gyro"].cla()
                ax["gyro"].plot(t_arr, gyro_arr, label="imu wz")
                ax["gyro"].set_title("Raw IMU yaw rate")
                ax["gyro"].set_xlabel("t (s)")
                ax["gyro"].set_ylabel("rad/s")
                ax["gyro"].grid(True, alpha=0.3)
                ax["gyro"].legend(fontsize=8)

                ax["speed"].cla()
                ax["speed"].plot(t_arr, speed, label="|v_body|")
                ax["speed"].set_title("Estimated body speed")
                ax["speed"].set_xlabel("t (s)")
                ax["speed"].set_ylabel("m/s")
                ax["speed"].grid(True, alpha=0.3)
                ax["speed"].legend(fontsize=8)

                fig.tight_layout()
                if args.no_show:
                    fig.savefig(args.save_path, dpi=150)
                else:
                    plt.pause(0.001)

                if args.duration > 0.0 and (now - t0) >= args.duration:
                    break
                time.sleep(max(0.0, 1.0 / args.live_plot_rate_hz))
        except KeyboardInterrupt:
            pass

    print(
        "final est pose: "
        f"x={est_hist[-1][0]:.3f}, y={est_hist[-1][1]:.3f}, yaw={est_hist[-1][2]:.3f}"
    )
    if args.no_show:
        print(f"wrote {args.save_path}")
    else:
        plt.ioff()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", choices=["ekf"], default="ekf")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--accel-noise", type=float, default=0.03)
    parser.add_argument("--gyro-noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", default=None, help="Read live IMU data from this serial port")
    parser.add_argument("--rx-publish-rate-hz", type=float, default=20.0)
    parser.add_argument("--live-plot-rate-hz", type=float, default=10.0)
    parser.add_argument("--debug-serial", action="store_true")
    parser.add_argument(
        "--save-path",
        default=os.path.join(os.path.dirname(__file__), "test_imu_only_localization.png"),
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.port:
        run_live_serial(args)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()
