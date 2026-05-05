from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ArmGeometry:
    base_x_m: float = 0.0
    base_y_m: float = 0.0
    link_1_m: float = 0.18
    link_2_m: float = 0.18

    @property
    def max_reach_m(self) -> float:
        return self.link_1_m + self.link_2_m

    @property
    def min_reach_m(self) -> float:
        return abs(self.link_1_m - self.link_2_m)


@dataclass(frozen=True)
class JointAngles:
    theta1_rad: float
    theta2_rad: float


@dataclass(frozen=True)
class EndEffectorTarget:
    x_m: float
    y_m: float


@dataclass(frozen=True)
class ArmAck:
    cmd_a: float
    cmd_b: float
    theta1_deg: float
    theta2_deg: float

    @property
    def joint_angles(self) -> JointAngles:
        return JointAngles(
            theta1_rad=math.radians(self.theta1_deg),
            theta2_rad=math.radians(self.theta2_deg),
        )


def forward_kinematics(joints: JointAngles, geometry: ArmGeometry) -> tuple[tuple[float, float], tuple[float, float]]:
    shoulder_x = geometry.base_x_m + geometry.link_1_m * math.cos(joints.theta1_rad)
    shoulder_y = geometry.base_y_m + geometry.link_1_m * math.sin(joints.theta1_rad)
    ee_x = shoulder_x + geometry.link_2_m * math.cos(joints.theta1_rad + joints.theta2_rad)
    ee_y = shoulder_y + geometry.link_2_m * math.sin(joints.theta1_rad + joints.theta2_rad)
    return (shoulder_x, shoulder_y), (ee_x, ee_y)


def inverse_kinematics(target: EndEffectorTarget, geometry: ArmGeometry) -> Optional[JointAngles]:
    dx = target.x_m - geometry.base_x_m
    dy = target.y_m - geometry.base_y_m
    r2 = dx * dx + dy * dy
    r = math.sqrt(r2)
    if r <= 0.0 or r > geometry.max_reach_m or r < geometry.min_reach_m:
        return None

    cos_theta2 = (r2 - geometry.link_1_m**2 - geometry.link_2_m**2) / (2.0 * geometry.link_1_m * geometry.link_2_m)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    sin_theta2 = math.sqrt(max(0.0, 1.0 - cos_theta2 * cos_theta2))
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta1 = math.atan2(dy, dx) - math.atan2(
        geometry.link_2_m * sin_theta2,
        geometry.link_1_m + geometry.link_2_m * cos_theta2,
    )
    return JointAngles(theta1_rad=theta1, theta2_rad=theta2)


def parse_arm_ack_line(line: str) -> Optional[ArmAck]:
    s = line.strip()
    if not s.startswith("ARM_ACK,"):
        return None
    parts = s.split(",")
    if len(parts) != 5:
        return None
    try:
        return ArmAck(
            cmd_a=float(parts[1]),
            cmd_b=float(parts[2]),
            theta1_deg=float(parts[3]),
            theta2_deg=float(parts[4]),
        )
    except ValueError:
        return None


class TwoLinkArmPlotter:
    def __init__(self, ax, geometry: ArmGeometry, *, title: str = "Two-link arm debugger"):
        self.ax = ax
        self.geometry = geometry
        self.title = title

        self._setup_axes()

        (self.workspace_outer,) = ax.plot([], [], "k--", alpha=0.25, linewidth=1.0)
        (self.workspace_inner,) = ax.plot([], [], "k:", alpha=0.25, linewidth=1.0)
        (self.target_arm_line,) = ax.plot([], [], color="tab:orange", linewidth=2.5, alpha=0.75, label="target")
        (self.applied_arm_line,) = ax.plot([], [], color="tab:blue", linewidth=3.0, label="applied")
        (self.target_point,) = ax.plot([], [], marker="x", color="tab:orange", markersize=8, linestyle="None")
        (self.applied_point,) = ax.plot([], [], marker="o", color="tab:blue", markersize=6, linestyle="None")
        (self.base_point,) = ax.plot([geometry.base_x_m], [geometry.base_y_m], marker="o", color="black", markersize=6, linestyle="None")
        self.status_text = ax.text(
            0.02,
            0.98,
            "waiting for commands...",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        self.ax.legend(loc="upper right", fontsize=8)
        self._draw_workspace()

    def _setup_axes(self) -> None:
        reach = self.geometry.max_reach_m + 0.05
        self.ax.set_title(self.title)
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(self.geometry.base_x_m - reach, self.geometry.base_x_m + reach)
        self.ax.set_ylim(self.geometry.base_y_m - reach, self.geometry.base_y_m + reach)
        self.ax.grid(True, alpha=0.25)

    def _draw_workspace(self) -> None:
        samples = 256
        theta = [2.0 * math.pi * i / (samples - 1) for i in range(samples)]
        outer_x = [self.geometry.base_x_m + self.geometry.max_reach_m * math.cos(t) for t in theta]
        outer_y = [self.geometry.base_y_m + self.geometry.max_reach_m * math.sin(t) for t in theta]
        self.workspace_outer.set_data(outer_x, outer_y)
        if self.geometry.min_reach_m > 0.0:
            inner_x = [self.geometry.base_x_m + self.geometry.min_reach_m * math.cos(t) for t in theta]
            inner_y = [self.geometry.base_y_m + self.geometry.min_reach_m * math.sin(t) for t in theta]
            self.workspace_inner.set_data(inner_x, inner_y)

    def _set_arm_line(self, line_artist, point_artist, joints: JointAngles, color: str) -> tuple[float, float]:
        (joint_x, joint_y), (ee_x, ee_y) = forward_kinematics(joints, self.geometry)
        line_artist.set_data(
            [self.geometry.base_x_m, joint_x, ee_x],
            [self.geometry.base_y_m, joint_y, ee_y],
        )
        line_artist.set_color(color)
        point_artist.set_data([ee_x], [ee_y])
        point_artist.set_color(color)
        return ee_x, ee_y

    def update(
        self,
        *,
        command_space: str,
        target_cmd_a: Optional[float],
        target_cmd_b: Optional[float],
        latest_ack: Optional[ArmAck],
        latest_height_m: Optional[float] = None,
        latest_raw_line: Optional[str] = None,
    ) -> None:
        target_text = "target: none"
        applied_text = "applied: none"

        if target_cmd_a is not None and target_cmd_b is not None:
            if command_space == "joint":
                target_joints = JointAngles(theta1_rad=float(target_cmd_a), theta2_rad=float(target_cmd_b))
                target_xy = self._set_arm_line(self.target_arm_line, self.target_point, target_joints, "tab:orange")
                target_text = (
                    f"target joint: ({target_cmd_a:.3f}, {target_cmd_b:.3f}) rad\n"
                    f"target ee: ({target_xy[0]:.3f}, {target_xy[1]:.3f}) m"
                )
            else:
                target = EndEffectorTarget(x_m=float(target_cmd_a), y_m=float(target_cmd_b))
                ik = inverse_kinematics(target, self.geometry)
                self.target_point.set_data([target.x_m], [target.y_m])
                if ik is not None:
                    self._set_arm_line(self.target_arm_line, self.target_point, ik, "tab:orange")
                    self.target_point.set_data([target.x_m], [target.y_m])
                    target_text = (
                        f"target ee: ({target.x_m:.3f}, {target.y_m:.3f}) m\n"
                        f"target joint: ({ik.theta1_rad:.3f}, {ik.theta2_rad:.3f}) rad"
                    )
                else:
                    self.target_arm_line.set_data([], [])
                    target_text = f"target ee: ({target.x_m:.3f}, {target.y_m:.3f}) m [unreachable]"
        else:
            self.target_arm_line.set_data([], [])
            self.target_point.set_data([], [])

        if latest_ack is not None:
            applied_xy = self._set_arm_line(
                self.applied_arm_line,
                self.applied_point,
                latest_ack.joint_angles,
                "tab:blue",
            )
            applied_text = (
                f"ack cmd: ({latest_ack.cmd_a:.3f}, {latest_ack.cmd_b:.3f})\n"
                f"ack joint: ({latest_ack.theta1_deg:.1f}, {latest_ack.theta2_deg:.1f}) deg\n"
                f"ack ee: ({applied_xy[0]:.3f}, {applied_xy[1]:.3f}) m"
            )
        else:
            self.applied_arm_line.set_data([], [])
            self.applied_point.set_data([], [])

        extras = []
        if latest_height_m is not None:
            extras.append(f"elevator: {latest_height_m:.3f} m")
        if latest_raw_line:
            extras.append(f"last rx: {latest_raw_line[:80]}")
        self.status_text.set_text("\n".join([target_text, applied_text, *extras]))


def create_two_link_arm_figure(geometry: ArmGeometry, *, title: str = "Two-link arm debugger"):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for the arm visualizer") from exc

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    plotter = TwoLinkArmPlotter(ax, geometry, title=title)
    fig.tight_layout()
    return plt, fig, ax, plotter


__all__ = [
    "ArmAck",
    "ArmGeometry",
    "EndEffectorTarget",
    "JointAngles",
    "TwoLinkArmPlotter",
    "create_two_link_arm_figure",
    "forward_kinematics",
    "inverse_kinematics",
    "parse_arm_ack_line",
]
