#!/usr/bin/env python
"""Simple UDP listener + plotter for IMU localization telemetry.

This is a lightweight test utility for viewing streamed pose telemetry from the
Jetson/laptop stack without running the full control center.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def setup_plot(title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_yaw = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    fig.suptitle(title, fontsize=14)
    return plt, fig, ax_xy, ax_yaw, ax_vel, ax_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 means run forever")
    parser.add_argument(
        "--save-path",
        default=os.path.join(os.path.dirname(__file__), "test_imu_listener_plotter.png"),
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    plt, fig, ax_xy, ax_yaw, ax_vel, ax_text = setup_plot("IMU telemetry listener")
    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, args.port))
    sock.settimeout(args.timeout)
    print(f"Listening for UDP telemetry on {args.bind_host}:{args.port}")

    t0 = time.monotonic()
    latest = None
    times: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    yaws: list[float] = []
    vxs: list[float] = []
    vys: list[float] = []
    wzs: list[float] = []

    try:
        while True:
            if args.max_seconds > 0.0 and (time.monotonic() - t0) >= args.max_seconds:
                break
            try:
                payload, addr = sock.recvfrom(65535)
                latest = json.loads(payload.decode("utf-8"))
                now = time.monotonic() - t0
                times.append(now)
                xs.append(float(latest.get("x", 0.0)))
                ys.append(float(latest.get("y", 0.0)))
                yaws.append(float(latest.get("yaw", 0.0)))
                vxs.append(float(latest.get("vx_body", 0.0)))
                vys.append(float(latest.get("vy_body", 0.0)))
                wzs.append(float(latest.get("wz", 0.0)))
                print(
                    f"rx {addr[0]}:{addr[1]} pose=({xs[-1]:.3f}, {ys[-1]:.3f}, {yaws[-1]:.3f}) "
                    f"task={latest.get('current_task')} cell={latest.get('cell')}"
                )
            except socket.timeout:
                pass

            if latest is None:
                continue

            ax_xy.cla()
            ax_xy.plot(xs, ys, color="tab:blue", label="estimated path")
            ax_xy.scatter([xs[0]], [ys[0]], c="green", s=50, label="start")
            ax_xy.scatter([xs[-1]], [ys[-1]], c="red", s=50, label="current")
            ax_xy.set_title("XY trajectory")
            ax_xy.set_xlabel("x (m)")
            ax_xy.set_ylabel("y (m)")
            ax_xy.axis("equal")
            ax_xy.grid(True, alpha=0.3)
            ax_xy.legend(fontsize=8)

            ax_yaw.cla()
            ax_yaw.plot(times, np.unwrap(np.asarray(yaws)), label="yaw")
            ax_yaw.plot(times, wzs, label="wz")
            ax_yaw.set_title("Yaw / yaw-rate")
            ax_yaw.set_xlabel("t (s)")
            ax_yaw.set_ylabel("rad, rad/s")
            ax_yaw.grid(True, alpha=0.3)
            ax_yaw.legend(fontsize=8)

            ax_vel.cla()
            ax_vel.plot(times, vxs, label="vx_body")
            ax_vel.plot(times, vys, label="vy_body")
            ax_vel.set_title("Body velocities")
            ax_vel.set_xlabel("t (s)")
            ax_vel.set_ylabel("m/s")
            ax_vel.grid(True, alpha=0.3)
            ax_vel.legend(fontsize=8)

            ax_text.cla()
            ax_text.axis("off")
            ax_text.text(
                0.02,
                0.98,
                "\n".join(
                    [
                        f"current_task: {latest.get('current_task')}",
                        f"cell: {latest.get('cell')}",
                        f"localization_ok: {latest.get('localization_ok')}",
                        f"distance_to_goal: {float(latest.get('distance_to_goal', float('nan'))):.3f}",
                        f"heading_error: {float(latest.get('heading_error', float('nan'))):.3f}",
                        f"x: {xs[-1]:.3f}",
                        f"y: {ys[-1]:.3f}",
                        f"yaw: {yaws[-1]:.3f}",
                    ]
                ),
                va="top",
                ha="left",
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

            fig.tight_layout()
            if args.no_show:
                fig.savefig(args.save_path, dpi=150)
            else:
                plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    if args.no_show:
        print(f"wrote {args.save_path}")
    else:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
