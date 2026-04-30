#!/usr/bin/env python
"""Laptop-side control center / visualization for Jetson mission telemetry.

Listens for UDP telemetry from `mobile_robot/missions/run_mission.py`, loads the
same `tasks.yaml`, and overlays the robot pose + current cell on the map.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from autonomy.mission_runner import default_tasks_path, load_map, load_tasks  # noqa: E402
from localization.map import Map  # noqa: E402


def draw_map_background(ax, map_: Map) -> None:
    xmax = map_._xmin + map_.nx * map_.resolution  # noqa: SLF001
    ymax = map_._ymin + map_.ny * map_.resolution  # noqa: SLF001
    display = np.ma.masked_where(~map_.cell_inside, map_.grid)
    ax.imshow(
        display.T,
        origin="lower",
        extent=(map_._xmin, xmax, map_._ymin, ymax),  # noqa: SLF001
        aspect="equal",
        interpolation="nearest",
        cmap="Greens",
        alpha=0.18,
    )
    if np.any(map_.obstacle_mask):
        obs_display = np.ma.masked_where(~map_.obstacle_mask, np.ones((map_.nx, map_.ny)))
        ax.imshow(
            obs_display.T,
            origin="lower",
            extent=(map_._xmin, xmax, map_._ymin, ymax),  # noqa: SLF001
            aspect="equal",
            interpolation="nearest",
            cmap="Reds",
            vmin=0,
            vmax=1,
            alpha=0.35,
        )
    ax.plot(map_._vertices[:, 0], map_._vertices[:, 1], color="k", linewidth=1.2)  # noqa: SLF001

    for obstacle in map_.obstacles:
        ax.plot(obstacle.boundary[:, 0], obstacle.boundary[:, 1], color="darkred", linewidth=1.0)

    for landmark in map_.landmarks:
        ax.scatter(
            landmark.point[0],
            landmark.point[1],
            s=45,
            c="tab:purple",
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        ax.arrow(
            landmark.point[0],
            landmark.point[1],
            0.08 * np.cos(landmark.heading),
            0.08 * np.sin(landmark.heading),
            head_width=0.03,
            head_length=0.03,
            fc="tab:purple",
            ec="tab:purple",
            length_includes_head=True,
            zorder=3,
        )
        ax.text(
            landmark.point[0] + 0.02,
            landmark.point[1] + 0.02,
            str(landmark.name),
            fontsize=8,
            color="tab:purple",
        )


def _draw_task_goals(ax, tasks: list[Any]) -> None:
    for task in tasks:
        ax.scatter(task.goal.x, task.goal.y, marker="*", s=140, c="tab:orange", edgecolors="black", linewidths=0.8)
        ax.text(task.goal.x + 0.03, task.goal.y + 0.03, task.name, fontsize=9, color="tab:orange")


def _draw_robot_heading_arrow(ax, x: float, y: float, yaw: float, length: float = 0.18) -> None:
    dx = length * float(np.cos(yaw))
    dy = length * float(np.sin(yaw))
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=0.05,
        head_length=0.05,
        fc="tab:blue",
        ec="tab:blue",
        linewidth=2.0,
        length_includes_head=True,
        zorder=8,
    )



def _apply_dynamic_obstacles(map_: Map, telemetry: dict[str, Any]) -> None:
    map_.clear_obstacles_by_prefix("person_")
    for obstacle in telemetry.get("dynamic_obstacles", []):
        center = obstacle.get("center")
        radius = obstacle.get("radius")
        name = str(obstacle.get("name", "person_unknown"))
        if center is None or radius is None or len(center) != 2:
            continue
        map_.add_circular_obstacle((float(center[0]), float(center[1])), float(radius), name)



def setup_plot(map_: Map, tasks: list[Any]):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for visualization") from exc

    fig, ax = plt.subplots(figsize=(11, 7))
    draw_map_background(ax, map_)
    _draw_task_goals(ax, tasks)

    (traj_line,) = ax.plot([], [], color="tab:cyan", linewidth=1.8, label="Jetson estimate", zorder=6)
    (robot_marker,) = ax.plot([], [], marker=(3, 0, 0), markersize=14, color="black", linestyle="None", zorder=8)
    cell_patch = plt.Rectangle((0, 0), map_.resolution, map_.resolution, fill=False, edgecolor="tab:blue", linewidth=2.0, visible=False, zorder=7)
    ax.add_patch(cell_patch)
    status_text = ax.text(0.02, 0.98, "waiting for telemetry...", transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title("Robot control center")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return plt, fig, ax, traj_line, robot_marker, cell_patch, status_text


def update_visualization(
    *,
    plt,
    fig,
    ax,
    map_: Map,
    tasks: list[Any],
    traj_x: list[float],
    traj_y: list[float],
    telemetry: dict[str, Any],
) -> None:
    x = float(telemetry["x"])
    y = float(telemetry["y"])
    yaw = float(telemetry["yaw"])

    _apply_dynamic_obstacles(map_, telemetry)

    ax.clear()
    draw_map_background(ax, map_)
    _draw_task_goals(ax, tasks)

    ax.plot(traj_x, traj_y, color="tab:cyan", linewidth=1.8, label="Jetson estimate", zorder=6)
    ax.plot([x], [y], marker="o", markersize=7, color="black", linestyle="None", zorder=8)
    _draw_robot_heading_arrow(ax, x, y, yaw)

    cell = telemetry.get("cell")
    if cell is not None and len(cell) == 2:
        i, j = int(cell[0]), int(cell[1])
        if map_.is_valid_cell(i, j):
            x0 = map_._xmin + i * map_.resolution  # noqa: SLF001
            y0 = map_._ymin + j * map_.resolution  # noqa: SLF001
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    map_.resolution,
                    map_.resolution,
                    fill=False,
                    edgecolor="tab:blue",
                    linewidth=2.0,
                    zorder=7,
                )
            )

    status_text = "\n".join(
        [
            f"task: {telemetry.get('current_task', 'unknown')}",
            f"pose: ({x:.3f}, {y:.3f}, {yaw:.3f})",
            f"cell: {telemetry.get('cell')}",
            f"distance_to_goal: {float(telemetry.get('distance_to_goal', float('nan'))):.3f}",
            f"heading_error: {float(telemetry.get('heading_error', float('nan'))):.3f}",
            f"localization_ok: {telemetry.get('localization_ok')}",
            f"dynamic_obstacles: {len(telemetry.get('dynamic_obstacles', []))}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        status_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        zorder=9,
    )
    ax.set_title("Robot control center")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    fig.canvas.draw_idle()
    plt.pause(0.001)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(default_tasks_path()), help="Path to tasks.yaml")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 means run forever")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--save-path",
        default=os.path.join(os.path.dirname(__file__), "control_center_view.png"),
    )
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    map_ = load_map(Path(args.tasks))
    plt, fig, ax, traj_line, robot_marker, cell_patch, status_text = setup_plot(map_, tasks)
    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, args.port))
    sock.settimeout(args.timeout)
    print(f"Listening for telemetry on udp://{args.bind_host}:{args.port}")

    traj_x: list[float] = []
    traj_y: list[float] = []
    latest: Optional[dict[str, Any]] = None
    t0 = time.monotonic()

    try:
        while True:
            if args.max_seconds > 0.0 and (time.monotonic() - t0) >= args.max_seconds:
                break
            try:
                payload, addr = sock.recvfrom(65535)
                latest = json.loads(payload.decode("utf-8"))
                traj_x.append(float(latest["x"]))
                traj_y.append(float(latest["y"]))
                print(
                    f"rx {addr[0]}:{addr[1]} task={latest.get('current_task')} "
                    f"pose=({latest.get('x'):.3f}, {latest.get('y'):.3f}, {latest.get('yaw'):.3f}) cell={latest.get('cell')}"
                )
            except socket.timeout:
                pass

            if latest is not None:
                update_visualization(
                    plt=plt,
                    fig=fig,
                    ax=ax,
                    map_=map_,
                    tasks=tasks,
                    traj_x=traj_x,
                    traj_y=traj_y,
                    telemetry=latest,
                )
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    if args.no_show:
        fig.savefig(args.save_path, dpi=150)
        print(f"wrote {args.save_path}")
    else:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
