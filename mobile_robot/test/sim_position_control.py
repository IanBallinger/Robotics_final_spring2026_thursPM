#!/usr/bin/env python
"""
Simulate Mecanum-body dynamics + ``CascadedWaypointController`` along an A* path
from a start grid cell to a goal grid cell (same map model as ``a_star`` demo).

Run from repo root:
  python mobile_robot/test/sim_position_control.py --start 0 0 --goal 14 2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from guidance.waypoint_controller import (  # noqa: E402
    CascadedWaypointController,
    MapPoseVelocity,
)
from localization.map import Map, Obstacle, Landmark  # noqa: E402
from planning.a_star import AStar, waypoints_from_polyline  # noqa: E402


def make_demo_map(resolution: float = 0.1) -> Map:
    """Same footprint/obstacles as ``planning/a_star.py`` __main__."""
    m = Map([(0, 0), (2, 0), (2, 0.5), (1, 0.5), (1, 1), (0, 1), (0, 0)], resolution)
    m.add_obstacle(Obstacle([(1, 0.5), (1, 1), (0, 1), (0, 0.5)], "table"))
    m.add_obstacle(Obstacle([(0.7, 0.4), (1, 0.4), (1, 0.7), (0.7, 0.7)], "shelf"))
    m.add_landmark(Landmark((0.5, 0.5), "at1", id="at1"))
    return m


def cell_start_pose(
    map: Map, i: int, j: int, heading: float
) -> tuple[float, float, float]:
    if not map.is_open_cell(i, j):
        raise ValueError(f"start cell ({i}, {j}) is not an open map cell")
    x, y = map.cell_center(i, j)
    return float(x), float(y), float(heading)


def integrate_step(
    x: float,
    y: float,
    psi: float,
    vx_body: float,
    vy_body: float,
    omega: float,
    vx_cmd: float,
    vy_cmd: float,
    omega_cmd: float,
    dt: float,
    tau_v: float,
    tau_w: float,
) -> tuple[float, float, float, float, float, float]:
    if tau_v > 0.0:
        a_v = min(1.0, dt / tau_v)
        vx_body_n = vx_body + a_v * (vx_cmd - vx_body)
        vy_body_n = vy_body + a_v * (vy_cmd - vy_body)
    else:
        vx_body_n = vx_cmd
        vy_body_n = vy_cmd

    if tau_w > 0.0:
        a_w = min(1.0, dt / tau_w)
        omega_n = omega + a_w * (omega_cmd - omega)
    else:
        omega_n = omega_cmd

    psi_n = psi + dt * omega_n
    c, s = np.cos(psi_n), np.sin(psi_n)
    vx_world_n = vx_body_n * c - vy_body_n * s
    vy_world_n = vx_body_n * s + vy_body_n * c
    x_n = x + dt * vx_world_n
    y_n = y + dt * vy_world_n
    return x_n, y_n, psi_n, vx_body_n, vy_body_n, omega_n


def run_simulation(
    *,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    start_heading: float,
    dt: float,
    max_time: float,
    capture_radius: float,
    v_des: float,
    omega_des: float,
    tau_v: float,
    tau_w: float,
    show_plot: bool,
) -> None:
    map = make_demo_map()
    si, sj = start_cell
    gi, gj = goal_cell
    x, y, psi = cell_start_pose(map, si, sj, start_heading)
    if not map.is_open_cell(gi, gj):
        raise ValueError(f"goal cell ({gi}, {gj}) is not an open map cell")

    gx, gy = map.cell_center(gi, gj)
    start_xy = (x, y)
    goal_xy = (float(gx), float(gy))

    planner = AStar(map)
    polyline = planner.generate_plan(start_xy, goal_xy)
    if not polyline:
        raise RuntimeError("A* found no path for the chosen start/goal cells")

    wpts = waypoints_from_polyline(polyline, end_heading=None)
    ctrl = CascadedWaypointController()

    wp_index = 0
    vx_body, vy_body, omega = 0.0, 0.0, 0.0
    traj_x = [x]
    traj_y = [y]

    t = 0.0
    n_steps = int(np.ceil(max_time / dt))

    times = [0.0]
    xs = [x]
    ys = [y]
    headings = [psi]
    vxs = [0.0]
    vys = [0.0]
    vx_bodys = [0.0]
    vy_bodys = [0.0]
    omegas = [0.0]
    vx_cmds: list[float] = [float("nan")]
    vy_cmds: list[float] = [float("nan")]
    omega_cmds: list[float] = [float("nan")]

    for _ in range(n_steps):
        goal_wp = wpts[wp_index]
        rho = float(np.hypot(goal_wp.xy[0] - x, goal_wp.xy[1] - y))
        if rho < capture_radius and wp_index < len(wpts) - 1:
            wp_index += 1
            goal_wp = wpts[wp_index]

        vx_world = vx_body * np.cos(psi) - vy_body * np.sin(psi)
        vy_world = vx_body * np.sin(psi) + vy_body * np.cos(psi)
        state = MapPoseVelocity(
            x=x,
            y=y,
            heading=psi,
            vx=float(vx_world),
            vy=float(vy_world),
            heading_rate=omega,
        )
        cmd = ctrl.compute(state, goal_wp, v_des, omega_des)

        x, y, psi, vx_body, vy_body, omega = integrate_step(
            x,
            y,
            psi,
            vx_body,
            vy_body,
            omega,
            cmd.vx,
            cmd.vy,
            cmd.omega,
            dt,
            tau_v,
            tau_w,
        )
        traj_x.append(x)
        traj_y.append(y)
        t += dt
        vx_world_n = float(vx_body * np.cos(psi) - vy_body * np.sin(psi))
        vy_world_n = float(vx_body * np.sin(psi) + vy_body * np.cos(psi))
        times.append(t)
        xs.append(x)
        ys.append(y)
        headings.append(psi)
        vxs.append(vx_world_n)
        vys.append(vy_world_n)
        vx_bodys.append(vx_body)
        vy_bodys.append(vy_body)
        omegas.append(omega)
        vx_cmds.append(cmd.vx)
        vy_cmds.append(cmd.vy)
        omega_cmds.append(cmd.omega)

        last = wpts[-1]
        if wp_index == len(wpts) - 1:
            if float(np.hypot(last.xy[0] - x, last.xy[1] - y)) < capture_radius:
                break

    print(f"plan vertices: {len(polyline)}, waypoints: {len(wpts)}")
    print(
        f"stopped at t={t:.2f}s, final cell {map.world_to_cell((x, y))}, pos=({x:.3f}, {y:.3f}), psi={psi:.3f}"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting") from e

    t_arr = np.asarray(times)
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    psi_arr = np.asarray(headings)
    vx_arr = np.asarray(vxs)
    vy_arr = np.asarray(vys)
    vx_body_arr = np.asarray(vx_bodys)
    vy_body_arr = np.asarray(vy_bodys)
    omega_arr = np.asarray(omegas)
    vx_cmd_arr = np.asarray(vx_cmds)
    vy_cmd_arr = np.asarray(vy_cmds)
    omega_cmd_arr = np.asarray(omega_cmds)
    v_mag = np.hypot(vx_arr, vy_arr)
    v_body_mag = np.hypot(vx_body_arr, vy_body_arr)

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.0, 1.0], hspace=0.38, wspace=0.28)
    ax_map = fig.add_subplot(gs[0, :])
    ax_xy = fig.add_subplot(gs[1, 0])
    ax_psi = fig.add_subplot(gs[1, 1])
    ax_vxy = fig.add_subplot(gs[2, 0])
    ax_om = fig.add_subplot(gs[2, 1])

    xmax = map._xmin + map.nx * map.resolution  # noqa: SLF001
    ymax = map._ymin + map.ny * map.resolution  # noqa: SLF001
    display = np.ma.masked_where(~map.cell_inside, map.grid)
    ax_map.imshow(
        display.T,
        origin="lower",
        extent=(map._xmin, xmax, map._ymin, ymax),  # noqa: SLF001
        aspect="equal",
        interpolation="nearest",
        cmap="Greens",
        alpha=0.25,
    )
    if np.any(map.obstacle_mask):
        obs_display = np.ma.masked_where(~map.obstacle_mask, np.ones((map.nx, map.ny)))
        ax_map.imshow(
            obs_display.T,
            origin="lower",
            extent=(map._xmin, xmax, map._ymin, ymax),  # noqa: SLF001
            aspect="equal",
            interpolation="nearest",
            cmap="Reds",
            vmin=0,
            vmax=1,
            alpha=0.35,
        )
    ax_map.plot(traj_x, traj_y, "b-", linewidth=1.5, label="sim trajectory")
    px = [p[0] for p in polyline]
    py = [p[1] for p in polyline]
    ax_map.plot(px, py, "k--", linewidth=1.0, alpha=0.7, label="A* polyline")
    ax_map.plot(start_xy[0], start_xy[1], "go", markersize=8, label="start")
    ax_map.plot(goal_xy[0], goal_xy[1], "r*", markersize=12, label="goal")
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.legend(loc="upper right")
    ax_map.set_title("Simulated position control (cell → cell)")
    ax_map.set_aspect("equal", adjustable="box")

    ax_xy.plot(t_arr, xs_arr, label="x (m)")
    ax_xy.plot(t_arr, ys_arr, label="y (m)")
    ax_xy.set_xlabel("t (s)")
    ax_xy.set_ylabel("position (m)")
    ax_xy.legend(loc="best", fontsize=8)
    ax_xy.set_title("MapPoseVelocity position")
    ax_xy.grid(True, alpha=0.3)

    ax_psi.plot(t_arr, np.unwrap(psi_arr), color="C2", label="heading ψ (unwrap)")
    ax_psi.set_xlabel("t (s)")
    ax_psi.set_ylabel("heading (rad)")
    ax_psi.legend(loc="best", fontsize=8)
    ax_psi.set_title("Heading")
    ax_psi.grid(True, alpha=0.3)

    ax_vxy.plot(t_arr, vx_arr, label="vx world")
    ax_vxy.plot(t_arr, vy_arr, label="vy world")
    ax_vxy.plot(t_arr, vx_body_arr, "--", alpha=0.8, label="vx body")
    ax_vxy.plot(t_arr, vy_body_arr, "--", alpha=0.8, label="vy body")
    ax_vxy.plot(t_arr, v_mag, color="0.35", linewidth=1.2, label="|v| world")
    ax_vxy.plot(
        t_arr, v_body_mag, color="0.55", linewidth=1.0, linestyle=":", label="|v| body"
    )
    m_cmd = np.isfinite(vx_cmd_arr)
    ax_vxy.plot(
        t_arr[m_cmd],
        vx_cmd_arr[m_cmd],
        "k--",
        alpha=0.65,
        linewidth=1.0,
        label="vx_cmd",
    )
    ax_vxy.plot(
        t_arr[m_cmd],
        vy_cmd_arr[m_cmd],
        color="k",
        linestyle=":",
        alpha=0.65,
        linewidth=1.0,
        label="vy_cmd",
    )
    ax_vxy.set_xlabel("t (s)")
    ax_vxy.set_ylabel("velocity (m/s)")
    ax_vxy.legend(loc="best", fontsize=8)
    ax_vxy.set_title("World/body velocity + commands")
    ax_vxy.grid(True, alpha=0.3)

    ax_om.plot(t_arr, omega_arr, label="heading_rate ω")
    m_w = np.isfinite(omega_cmd_arr)
    ax_om.plot(
        t_arr[m_w],
        omega_cmd_arr[m_w],
        "k--",
        alpha=0.65,
        linewidth=1.0,
        label="ω_cmd",
    )
    ax_om.set_xlabel("t (s)")
    ax_om.set_ylabel("angular rate (rad/s)")
    ax_om.legend(loc="best", fontsize=8)
    ax_om.set_title("Yaw rate + command")
    ax_om.grid(True, alpha=0.3)

    fig.align_labels()
    if show_plot:
        plt.show()
    else:
        out = os.path.join(os.path.dirname(__file__), "sim_position_control.png")
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, nargs=2, metavar=("I", "J"), default=[0, 0])
    p.add_argument("--goal", type=int, nargs=2, metavar=("I", "J"), default=[14, 3])
    p.add_argument("--heading0", type=float, default=0.0, help="initial yaw (rad)")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--max-time", type=float, default=1000.0)
    p.add_argument(
        "--capture", type=float, default=0.01, help="waypoint capture radius (m)"
    )
    p.add_argument(
        "--v-des", type=float, default=0.25, help="outer-loop speed cap (m/s)"
    )
    p.add_argument("--omega-des", type=float, default=0.0)
    p.add_argument(
        "--tau-v",
        type=float,
        default=0.12,
        help="first-order lag on v (s); 0 = instant",
    )
    p.add_argument(
        "--tau-w",
        type=float,
        default=0.12,
        help="first-order lag on omega (s); 0 = instant",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="save PNG next to this script instead of plt.show()",
    )
    args = p.parse_args()

    run_simulation(
        start_cell=(args.start[0], args.start[1]),
        goal_cell=(args.goal[0], args.goal[1]),
        start_heading=args.heading0,
        dt=args.dt,
        max_time=args.max_time,
        capture_radius=args.capture,
        v_des=args.v_des,
        omega_des=args.omega_des,
        tau_v=args.tau_v,
        tau_w=args.tau_w,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
