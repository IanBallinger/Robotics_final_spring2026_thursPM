#!/usr/bin/env python
"""Demo: run the YAML-driven waypoint mission tree with planning, dynamics, and plotting."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import py_trees
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from autonomy.mission_runner import (  # noqa: E402
    Task,
    default_tasks_path,
    load_map,
    load_tasks,
)
from autonomy.trees.waypoint_mission import (
    MISSION_DONE,
    create_tree,
    initialise_blackboard,
)  # noqa: E402
from guidance.waypoint_controller import (  # noqa: E402
    CascadedWaypointController,
    MapPoseVelocity,
    wrap_to_pi,
)
from localization import (  # noqa: E402
    AprilTagMeasurement,
    ExtendedKalmanFilter2D,
    IMUMeasurement,
    UnscentedKalmanFilter2D,
)
from localization.map import Map  # noqa: E402
from planning.a_star import AStar, waypoints_from_polyline  # noqa: E402
from sim_position_control import integrate_step  # noqa: E402
from serial_connection.serial_con import SerialConnect  # noqa: E402


@dataclass
class PlannedTask:
    task: Task
    polyline: list[tuple[float, float]]
    waypoints: list
    waypoint_index: int = 0
    achieved: bool = False


@dataclass
class LocalizationSimConfig:
    filter_name: str
    initial_state: np.ndarray
    initial_covariance: np.ndarray
    process_noise: np.ndarray
    apriltag_measurement_noise: np.ndarray
    gyro_measurement_noise: np.ndarray
    apriltag_period_s: float
    apriltag_dropout_prob: float
    imu_accel_noise_std: float
    imu_gyro_noise_std: float
    apriltag_position_noise_std: float
    apriltag_yaw_noise_std: float


def tick_and_print(
    tree: py_trees.trees.BehaviourTree, tick_index: int, *, print_tree: bool
):
    tree.tick()
    if not print_tree:
        return
    print(f"\n--- Tick {tick_index} ---")
    print(py_trees.display.unicode_tree(tree.root, show_status=True))


def plan_tasks(map_: Map, tasks: list[Task]) -> list[PlannedTask]:
    planner = AStar(map_)
    planned: list[PlannedTask] = []
    for task in tasks:
        start_xy = (task.start.x, task.start.y)
        goal_xy = (task.goal.x, task.goal.y)
        polyline = planner.generate_plan(start_xy, goal_xy)
        if not polyline:
            raise RuntimeError(f"A* found no path for task {task.name}")
        planned.append(
            PlannedTask(
                task=task,
                polyline=polyline,
                waypoints=waypoints_from_polyline(
                    polyline, end_heading=task.goal.heading
                ),
            )
        )
    return planned


def load_localization_config(tasks_path: str | os.PathLike[str]) -> LocalizationSimConfig:
    raw = yaml.safe_load(open(tasks_path, "r", encoding="utf-8"))
    loc = raw.get("localization", {})
    if not loc:
        raise ValueError("mission_config.yaml must define a 'localization' section")

    sim = loc.get("simulation", {})
    init = loc.get("initial_state", {})
    return LocalizationSimConfig(
        filter_name=str(loc.get("filter", "ekf")).lower(),
        initial_state=np.array(
            [
                float(init.get("x", 0.0)),
                float(init.get("y", 0.0)),
                float(init.get("yaw", 0.0)),
                float(init.get("vx_body", 0.0)),
                float(init.get("vy_body", 0.0)),
                float(init.get("wz", 0.0)),
            ],
            dtype=float,
        ),
        initial_covariance=np.asarray(loc.get("initial_covariance"), dtype=float),
        process_noise=np.asarray(loc.get("process_noise"), dtype=float),
        apriltag_measurement_noise=np.asarray(
            loc.get("apriltag_measurement_noise"), dtype=float
        ),
        gyro_measurement_noise=np.asarray(loc.get("gyro_measurement_noise"), dtype=float),
        apriltag_period_s=float(sim.get("apriltag_period_s", 0.2)),
        apriltag_dropout_prob=float(sim.get("apriltag_dropout_prob", 0.0)),
        imu_accel_noise_std=float(sim.get("imu_accel_noise_std", 0.0)),
        imu_gyro_noise_std=float(sim.get("imu_gyro_noise_std", 0.0)),
        apriltag_position_noise_std=float(sim.get("apriltag_position_noise_std", 0.0)),
        apriltag_yaw_noise_std=float(sim.get("apriltag_yaw_noise_std", 0.0)),
    )


def create_localization_filter(config: LocalizationSimConfig):
    filter_cls = (
        UnscentedKalmanFilter2D if config.filter_name == "ukf" else ExtendedKalmanFilter2D
    )
    return filter_cls(
        initial_state=config.initial_state,
        initial_covariance=config.initial_covariance,
        process_noise=config.process_noise,
        apriltag_measurement_noise=config.apriltag_measurement_noise,
        gyro_measurement_noise=config.gyro_measurement_noise,
    )


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
        obs_display = np.ma.masked_where(
            ~map_.obstacle_mask, np.ones((map_.nx, map_.ny))
        )
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
    for obstacle in map_.obstacles:
        ax.plot(
            obstacle.boundary[:, 0],
            obstacle.boundary[:, 1],
            color="darkred",
            linewidth=1.2,
        )
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
        ax.text(
            landmark.point[0] + 0.02,
            landmark.point[1] + 0.02,
            landmark.name,
            fontsize=8,
            color="tab:purple",
        )


def setup_plot(map_: Map, planned_tasks: list[PlannedTask]):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting") from e

    fig, ax = plt.subplots(figsize=(11, 7))
    draw_map_background(ax, map_)

    pending_color = "tab:orange"
    active_color = "tab:blue"
    achieved_color = "tab:green"
    path_colors = ["C0", "C1", "C4", "C5", "C6", "C7"]

    artists: dict[str, dict[str, object]] = {}
    for idx, planned in enumerate(planned_tasks):
        color = path_colors[idx % len(path_colors)]
        poly = np.asarray(planned.polyline)
        wpts = np.asarray([wp.xy for wp in planned.waypoints])
        ax.plot(
            poly[:, 0],
            poly[:, 1],
            linestyle="--",
            linewidth=1.6,
            color=color,
            alpha=0.85,
            label=f"path: {planned.task.name}",
        )
        waypoint_scatter = ax.scatter(
            wpts[:, 0],
            wpts[:, 1],
            s=34,
            c=pending_color,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )
        start_artist = ax.scatter(
            planned.task.start.x,
            planned.task.start.y,
            marker="s",
            s=90,
            c=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        goal_artist = ax.scatter(
            planned.task.goal.x,
            planned.task.goal.y,
            marker="*",
            s=180,
            c=pending_color,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )
        ax.text(
            planned.task.goal.x + 0.03,
            planned.task.goal.y + 0.03,
            planned.task.name,
            fontsize=9,
            color=color,
        )
        artists[planned.task.name] = {
            "waypoints": waypoint_scatter,
            "start": start_artist,
            "goal": goal_artist,
            "pending_color": pending_color,
            "active_color": active_color,
            "achieved_color": achieved_color,
        }

    (robot_line,) = ax.plot(
        [], [], "k-", linewidth=2.0, label="sim trajectory", zorder=7
    )
    (est_line,) = ax.plot(
        [], [], color="tab:cyan", linewidth=1.6, linestyle="-.", label="KF estimate", zorder=7
    )
    (robot_marker,) = ax.plot(
        [],
        [],
        marker=(3, 0, 0),
        markersize=14,
        color="black",
        linestyle="None",
        zorder=8,
    )

    ax.set_title("Waypoint mission demo with A* paths and simulated dynamics")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return plt, fig, ax, artists, robot_line, est_line, robot_marker


def set_task_artist_state(
    artists: dict[str, dict[str, object]], task_name: str, state: str
) -> None:
    entry = artists[task_name]
    if state == "active":
        color = entry["active_color"]
    elif state == "achieved":
        color = entry["achieved_color"]
    else:
        color = entry["pending_color"]
    entry["waypoints"].set_color(color)
    entry["goal"].set_color(color)


def update_plot(
    plt,
    fig,
    artists: dict[str, dict[str, object]],
    robot_line,
    est_line,
    robot_marker,
    trajectory_x: list[float],
    trajectory_y: list[float],
    est_x: list[float],
    est_y: list[float],
    heading: float,
    tasks: list[Task],
    active_task: str,
    completed_tasks: set[str],
    title_suffix: str,
    pause: float,
) -> None:
    for task in tasks:
        if task.name in completed_tasks:
            set_task_artist_state(artists, task.name, "achieved")
        elif task.name == active_task:
            set_task_artist_state(artists, task.name, "active")
        else:
            set_task_artist_state(artists, task.name, "pending")

    robot_line.set_data(trajectory_x, trajectory_y)
    est_line.set_data(est_x, est_y)
    robot_marker.set_data([trajectory_x[-1]], [trajectory_y[-1]])
    robot_marker.set_marker((3, 0, np.degrees(heading) - 90.0))
    fig.axes[0].set_title(
        "Waypoint mission demo with A* paths and simulated dynamics"
        f"\nactive={active_task}, completed={len(completed_tasks)}/{len(tasks)}{title_suffix}"
    )
    fig.canvas.draw_idle()
    plt.pause(pause)


def simulate_active_task(
    planned: PlannedTask,
    controller: CascadedWaypointController,
    x: float,
    y: float,
    psi: float,
    vx_body: float,
    vy_body: float,
    omega: float,
    *,
    dt: float,
    sim_time_per_tick: float,
    capture_radius: float,
    tau_v: float,
    tau_w: float,
) -> tuple[float, float, float, float, float, float]:
    n_steps = max(1, int(np.ceil(sim_time_per_tick / dt)))

    for _ in range(n_steps):
        goal_wp = planned.waypoints[planned.waypoint_index]
        rho = float(np.hypot(goal_wp.xy[0] - x, goal_wp.xy[1] - y))
        if rho < capture_radius and planned.waypoint_index < len(planned.waypoints) - 1:
            planned.waypoint_index += 1
            goal_wp = planned.waypoints[planned.waypoint_index]

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
        cmd = controller.compute(state, goal_wp)
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

    return x, y, psi, vx_body, vy_body, omega, cmd.wheel_rates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dt", type=float, default=0.01, help="simulation integration step (s)"
    )
    p.add_argument(
        "--sim-time-per-tick",
        type=float,
        default=0.25,
        help="simulated motion per behaviour-tree tick (s)",
    )
    p.add_argument("--max-ticks", type=int, default=240)
    p.add_argument(
        "--capture", type=float, default=0.05, help="waypoint capture radius (m)"
    )
    p.add_argument(
        "--v-des",
        type=float,
        default=0.25,
        help="controller translational speed cap (m/s)",
    )
    p.add_argument("--omega-des", type=float, default=0.0)
    p.add_argument("--tau-v", type=float, default=0.12, help="first-order lag on v (s)")
    p.add_argument(
        "--tau-w", type=float, default=0.12, help="first-order lag on omega (s)"
    )
    p.add_argument(
        "--tick-sleep",
        type=float,
        default=0.2,
        help="wall-clock pause between plot updates when showing",
    )
    p.add_argument(
        "--print-tree",
        action="store_true",
        help="print the full behaviour tree at every tick",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="save a PNG instead of opening an interactive window",
    )
    p.add_argument(
        "--save-path",
        default=os.path.join(
            os.path.dirname(__file__), "run_waypoint_mission_demo.png"
        ),
        help="output path used with --no-show",
    )
    args = p.parse_args()

    # setup serial connection to mobile robot
    # serial_con = SerialConnect(port="/dev/ttyESP_WHL")

    tasks_path = default_tasks_path()
    tasks = load_tasks(tasks_path)
    localization_config = load_localization_config(tasks_path)
    task_lookup = {task.name: task for task in tasks}
    map_ = load_map(tasks_path)
    map_.plot()
    planned_tasks = plan_tasks(map_, tasks)
    planned_lookup = {planned.task.name: planned for planned in planned_tasks}

    bb = initialise_blackboard(
        tasks[0].name,
        state={
            "mission_enabled": True,
            "localization_ok": True,
            "battery_voltage": 24.0,
            "path_ready": True,
            "obstacle_blocking_path": False,
            "zone_clear": True,
            "elevator_at_height": True,
            "distance_to_goal": 1.0,
            "heading_error": 1.0,
            "tray_detected": False,
            "robot_stopped": False,
            "tray_released": False,
            "gripper_closed": True,
            "previous_task_complete": True,
        },
    )

    tree = create_tree(tasks)
    tree.setup(timeout=2.0)

    controller = CascadedWaypointController()
    controller.kv_inner = 10
    controller.ky_inner = 10
    controller.komega_inner = 10
    controller.k_rho = 1
    controller.k_alpha = 1
    controller.k_heading = 1
    x = float(tasks[0].start.x)
    y = float(tasks[0].start.y)
    psi = float(tasks[0].start.heading)
    vx_body = 0.0
    vy_body = 0.0
    omega = 0.0
    trajectory_x = [x]
    trajectory_y = [y]
    completed_tasks: set[str] = set()

    kf = create_localization_filter(localization_config)
    est_state = kf.get_state()
    est_traj_x = [float(est_state[0])]
    est_traj_y = [float(est_state[1])]
    rng = np.random.default_rng(7)
    last_apriltag_update_t = 0.0
    sim_time = 0.0

    plt, fig, _, artists, robot_line, est_line, robot_marker = setup_plot(map_, planned_tasks)
    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    update_plot(
        plt,
        fig,
        artists,
        robot_line,
        est_line,
        robot_marker,
        trajectory_x,
        trajectory_y,
        est_traj_x,
        est_traj_y,
        psi,
        tasks,
        bb.get("current_task"),
        completed_tasks,
        title_suffix="",
        pause=0.001,
    )

    for i in range(1, args.max_ticks + 1):
        previous_task = bb.get("current_task")
        if previous_task == MISSION_DONE:
            break

        planned = planned_lookup[previous_task]
        # serial_con.read_parsed()
        prev_vx_body = vx_body
        prev_vy_body = vy_body
        prev_omega = omega

        x, y, psi, vx_body, vy_body, omega, wheel_rates = simulate_active_task(
            planned,
            controller,
            x,
            y,
            psi,
            vx_body,
            vy_body,
            omega,
            dt=args.dt,
            sim_time_per_tick=args.sim_time_per_tick,
            capture_radius=args.capture,
            tau_v=args.tau_v,
            tau_w=args.tau_w,
        )
        # serial_con.send_wheel_cmd(
        #     wheel_rates[0], wheel_rates[1], wheel_rates[2], wheel_rates[3]
        # )
        trajectory_x.append(x)
        trajectory_y.append(y)

        sim_time += args.sim_time_per_tick
        imu_dt = args.sim_time_per_tick
        ax_body = (vx_body - prev_vx_body) / max(imu_dt, 1e-6)
        ay_body = (vy_body - prev_vy_body) / max(imu_dt, 1e-6)
        imu_meas = IMUMeasurement(
            ax=ax_body + rng.normal(0.0, localization_config.imu_accel_noise_std),
            ay=ay_body + rng.normal(0.0, localization_config.imu_accel_noise_std),
            wz=omega + rng.normal(0.0, localization_config.imu_gyro_noise_std),
        )
        kf.predict(imu_meas, imu_dt)
        kf.update_imu(imu_meas)

        if sim_time - last_apriltag_update_t >= localization_config.apriltag_period_s:
            last_apriltag_update_t = sim_time
            if rng.random() >= localization_config.apriltag_dropout_prob:
                apriltag_meas = AprilTagMeasurement(
                    x=x + rng.normal(0.0, localization_config.apriltag_position_noise_std),
                    y=y + rng.normal(0.0, localization_config.apriltag_position_noise_std),
                    yaw=psi + rng.normal(0.0, localization_config.apriltag_yaw_noise_std),
                    covariance=localization_config.apriltag_measurement_noise,
                )
                kf.update_apriltag(apriltag_meas)

        est_state = kf.get_state()
        est_x = float(est_state[0])
        est_y = float(est_state[1])
        est_psi = float(est_state[2])
        est_traj_x.append(est_x)
        est_traj_y.append(est_y)

        active_task = task_lookup[previous_task]
        goal_error = float(np.hypot(active_task.goal.x - est_x, active_task.goal.y - est_y))
        heading_error = float(wrap_to_pi(active_task.goal.heading - est_psi))
        speed = float(np.hypot(est_state[3], est_state[4]))

        bb.set("distance_to_goal", goal_error)
        bb.set("heading_error", heading_error)
        bb.set("robot_stopped", speed < 0.02)
        bb.set("tray_detected", goal_error < 0.10)
        bb.set("tray_released", goal_error < 0.06)
        bb.set("gripper_closed", goal_error >= 0.06)

        tick_and_print(tree, tick_index=i, print_tree=args.print_tree)
        current_task = bb.get("current_task")
        print(
            f"current_task = {current_task}, true=({x:.3f}, {y:.3f}, {psi:.3f}), "
            f"est=({est_x:.3f}, {est_y:.3f}, {est_psi:.3f}), "
            f"goal_err={goal_error:.3f}, heading_err={heading_error:.3f}"
        )

        if current_task != previous_task:
            completed_tasks.add(previous_task)
            planned.achieved = True

        update_plot(
            plt,
            fig,
            artists,
            robot_line,
            est_line,
            robot_marker,
            trajectory_x,
            trajectory_y,
            est_traj_x,
            est_traj_y,
            est_psi,
            tasks,
            current_task,
            completed_tasks,
            title_suffix=f", t={i * args.sim_time_per_tick:.2f}s",
            pause=0.001,
        )

        if current_task == MISSION_DONE:
            break
        if not args.no_show:
            time.sleep(args.tick_sleep)

    if args.no_show:
        fig.savefig(args.save_path, dpi=150)
        print(f"wrote {args.save_path}")
    else:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
