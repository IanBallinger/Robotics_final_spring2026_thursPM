from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from ..planning.a_star import Waypoint
except ImportError:
    from planning.a_star import Waypoint  # type: ignore[no-redef]


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class MapPoseVelocity:
    """World-frame pose and velocity estimate (meters, radians, seconds)."""

    x: float
    y: float
    heading: float
    vx: float
    vy: float
    heading_rate: float

    def body_velocity(self) -> Tuple[float, float]:
        """Return body-frame longitudinal/lateral velocity ``(V_Cξ, V_Cη)``."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        v_xi = self.vx * c + self.vy * s
        v_eta = -self.vx * s + self.vy * c
        return float(v_xi), float(v_eta)


@dataclass(frozen=True)
class MecanumCommand:
    """Body twist plus wheel angular-rate command for a Mecanum platform."""

    vx: float
    vy: float
    omega: float
    wheel_rates: Tuple[float, float, float, float]


class CascadedWaypointController:
    """
    Holonomic waypoint tracker for a Mecanum base.

    Outer loop:
        map-frame position error -> body-frame translational references
        heading error -> yaw-rate reference

    Inner loop:
        proportional velocity feedback on body longitudinal/lateral speed and
        yaw rate.

    The kinematic conversion between body twist and wheel speeds follows the
    exact Mecanum-wheel constraint relations in Zeidis & Zimmermann (2019).
    """

    def __init__(
        self,
        k_rho: float = 1.5,
        k_alpha: float = 2.5,
        k_heading: float = 2.0,
        align_turn_thresh: float = 0.45,
        v_max: float = 0.35,
        omega_max: float = 1.8,
        kv_inner: float = 0.9,
        ky_inner: float = 0.9,
        komega_inner: float = 0.9,
        wheel_radius: float = 0.06,
        longitudinal_half_extent: float = 0.2,
        lateral_half_extent: float = 0.2,
    ):
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_heading = k_heading
        self.align_turn_thresh = align_turn_thresh
        self.v_max = v_max
        self.omega_max = omega_max
        self.kv_inner = kv_inner
        self.ky_inner = ky_inner
        self.komega_inner = komega_inner
        self.wheel_radius = wheel_radius
        self.longitudinal_half_extent = longitudinal_half_extent
        self.lateral_half_extent = lateral_half_extent

    @property
    def yaw_moment_arm(self) -> float:
        """Return ``ρ + l`` from the paper's notation."""
        return self.longitudinal_half_extent + self.lateral_half_extent

    def body_twist_to_wheel_rates(
        self,
        vx_body: float,
        vy_body: float,
        omega: float,
    ) -> Tuple[float, float, float, float]:
        """
        Convert body twist to wheel angular rates using Eq. (11) rearranged.

        Wheel ordering is ``(φ̇1, φ̇2, φ̇3, φ̇4)``. The result satisfies
        ``φ̇1 + φ̇2 = φ̇3 + φ̇4`` exactly.
        """
        r = self.wheel_radius
        a = self.yaw_moment_arm
        w1 = (vx_body - vy_body - a * omega) / r
        w2 = (vx_body + vy_body + a * omega) / r
        w3 = (vx_body + vy_body - a * omega) / r
        w4 = (vx_body - vy_body + a * omega) / r
        return float(w1), float(w2), float(w3), float(w4)

    def wheel_rates_to_body_twist(
        self,
        wheel_rates: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Convert wheel rates to body twist using the paper's pseudoinverse form.

        This implements Eq. (36), which is also exact when the compatibility
        relation ``φ̇1 + φ̇2 = φ̇3 + φ̇4`` holds.
        """
        w1, w2, w3, w4 = (float(w) for w in wheel_rates)
        r = self.wheel_radius
        a = self.yaw_moment_arm
        vx_body = 0.25 * r * (w1 + w2 + w3 + w4)
        vy_body = 0.25 * r * (-w1 + w2 + w3 - w4)
        omega = 0.25 * r * (-w1 + w2 - w3 + w4) / a
        return float(vx_body), float(vy_body), float(omega)

    def compute(
        self,
        state: MapPoseVelocity,
        goal: Waypoint,
        v_des: float,
        omega_des: float,
    ) -> MecanumCommand:
        """
        Return a Mecanum body-twist + wheel-rate command.

        Parameters
        ----------
        state:
            Current world-frame pose/velocity estimate.
        goal:
            Desired map position and heading.
        v_des:
            Cap on translational speed magnitude.
        omega_des:
            Yaw-rate feedforward.
        """
        gx, gy = float(goal.xy[0]), float(goal.xy[1])
        dx_w, dy_w = gx - state.x, gy - state.y
        rho = float(np.hypot(dx_w, dy_w))

        c, s = np.cos(state.heading), np.sin(state.heading)
        ex_body = dx_w * c + dy_w * s
        ey_body = -dx_w * s + dy_w * c

        alpha = wrap_to_pi(float(np.arctan2(ey_body, ex_body))) if rho > 1e-9 else 0.0
        e_psi = wrap_to_pi(goal.heading - state.heading)

        v_cap = min(max(0.0, float(v_des)), self.v_max)
        vx_ref = self.k_rho * ex_body
        vy_ref = self.k_rho * ey_body
        # vx_ref, vy_ref = self._scale_planar_speed(vx_ref, vy_ref, v_cap)

        if rho < 1e-9:
            vx_ref = 0.0
            vy_ref = 0.0

        # If the waypoint lies far off the current body x-axis, allow the robot
        # to exploit lateral motion rather than forcing a unicycle-style turn-in-place.
        # The heading controller still aligns the body to the waypoint heading.
        omega_ref = self.k_alpha * alpha + self.k_heading * e_psi
        omega_ref = float(np.clip(omega_ref, -self.omega_max, self.omega_max))

        vx_meas, vy_meas = state.body_velocity()
        omega_meas = state.heading_rate

        vx_cmd = vx_ref + self.kv_inner * (vx_ref - vx_meas)
        vy_cmd = vy_ref + self.ky_inner * (vy_ref - vy_meas)
        omega_cmd = omega_ref + self.komega_inner * (omega_ref - omega_meas)

        # vx_cmd, vy_cmd = self._scale_planar_speed(vx_cmd, vy_cmd, self.v_max)
        omega_cmd = float(np.clip(omega_cmd, -self.omega_max, self.omega_max))

        wheel_rates = self.body_twist_to_wheel_rates(vx_cmd, vy_cmd, omega_cmd)
        return MecanumCommand(
            vx=vx_cmd, vy=vy_cmd, omega=omega_cmd, wheel_rates=wheel_rates
        )
