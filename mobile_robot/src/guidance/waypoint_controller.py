from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from planning.a_star import Waypoint


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
class DifferentialDriveCommand:
    """Body twist plus wheel angular-rate command.

    Wheel-rate tuple uses canonical order matching the ESP32 wheel controller:
    ``(w1, w2, w3, w4) = (left_front, right_front, left_rear, right_rear)``.
    """

    vx: float
    vy: float
    omega: float
    wheel_rates: Tuple[float, float, float, float]


class CascadedWaypointController:
    """
    Waypoint tracker for a 4-wheel two-side drive base.

    Outer loop:
        map-frame position/heading error -> forward-speed and yaw-rate references

    Inner loop:
        proportional velocity feedback on body longitudinal speed and yaw rate.

    Lateral body velocity is not commanded for this platform and is always zero.
    Left/right wheel commands are duplicated across front/rear wheels. For this
    project, the wheel-rate mapping intentionally ignores track width and treats
    turn command contribution as side-based rather than metric differential-drive
    kinematics.
    """

    def __init__(
        self,
        k_rho: float = 0.1,
        k_alpha: float = 0.1,
        k_heading: float = 0.1,
        align_turn_thresh: float = 0.45,
        v_max: float = 0.35,
        omega_max: float = 0.5,
        kv_inner: float = 0.9,
        ky_inner: float = 0.9,
        komega_inner: float = 0.9,
        wheel_radius: float = 0.06,
        track_width: float = 0.4,
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
        self.track_width = track_width

    def body_twist_to_wheel_rates(
        self,
        vx_body: float,
        omega: float,
    ) -> Tuple[float, float, float, float]:
        """Convert body command to per-side wheel angular rates.

        Wheel ordering is ``(left_front, right_front, left_rear, right_rear)``.
        ``vy_body`` is ignored because lateral body motion is not supported.

        Note: this mapping intentionally ignores track width and uses
        ``left = vx - omega`` and ``right = vx + omega`` before dividing by wheel
        radius. This matches the project's side-command semantics for four
        independent wheels.
        """
        r = self.wheel_radius
        w_left = (vx_body - omega) / r
        w_right = (vx_body + omega) / r

        # clamp wheel rates
        w_left = np.clip(float(w_left),-3.0,3.0)
        w_right = np.clip(float(w_right),-3.0,3.0)
        return w_left, w_left, w_right, w_right

    def wheel_rates_to_body_twist(
        self,
        wheel_rates: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float]:
        """Convert ESP32-reported wheel rates back to body-command space.

        Ground truth is ``serial_to_from_jet.cpp``:
        - host TX uses ``WHL_CMD,left,left,right,right``
        - the ESP32 flips signs internally on channels 2 and 3
        - encoder/debug output is reported in the ESP32 wheel-index order

        Therefore the measured tuple ``(w1, w2, w3, w4)`` should be interpreted as:
        - left side command  ~= ``(w1 - w2) / 2``
        - right side command ~= ``(w4 - w3) / 2``
        """
        w1, w2, w3, w4 = (float(w) for w in wheel_rates)
        r = self.wheel_radius
        w_left = 0.5 * (w1 - w2)
        w_right = 0.5 * (w4 - w3)
        vx_body = 0.5 * r * (w_left + w_right)
        omega = 0.5 * r * (w_right - w_left)
        return float(vx_body), 0.0, float(omega)

    def compute(
        self,
        state: MapPoseVelocity,
        goal: Waypoint,
        *,
        final_pose_mode: bool = False,
    ) -> DifferentialDriveCommand:
        """
        Return a differential-drive body-twist + wheel-rate command.

        Parameters
        ----------
        state:
            Current world-frame pose/velocity estimate.
        goal:
            Desired map position and heading.
        """
        gx, gy = float(goal.xy[0]), float(goal.xy[1])
        dx_w, dy_w = gx - state.x, gy - state.y
        rho = float(np.hypot(dx_w, dy_w))

        c, s = np.cos(state.heading), np.sin(state.heading)
        ex_body = dx_w * c + dy_w * s
        ey_body = -dx_w * s + dy_w * c

        alpha = wrap_to_pi(float(np.arctan2(ey_body, ex_body))) if rho > 1e-9 else 0.0
        e_psi = wrap_to_pi(goal.heading - state.heading)

        if final_pose_mode:
            # Final pose regulator: first approach the goal position, then settle
            # precisely onto the desired terminal heading once the robot is close.
            final_pos_radius = 0.08
            final_heading_radius = 0.03

            if rho <= final_heading_radius:
                vx_ref = 0.0
                omega_ref = self.k_heading * e_psi
            else:
                vx_ref = self.k_rho * ex_body
                if rho <= final_pos_radius:
                    vx_ref *= rho / final_pos_radius
                if abs(alpha) > self.align_turn_thresh:
                    vx_ref = 0.0
                    omega_ref = self.k_alpha * alpha
                else:
                    omega_ref = self.k_alpha * alpha
                    if rho <= final_pos_radius:
                        omega_ref += self.k_heading * e_psi
        else:
            vx_ref = self.k_rho * ex_body

            # Differential drive cannot translate laterally. If the waypoint lies
            # far off the body x-axis, prioritize turning in place before driving.
            if abs(alpha) > self.align_turn_thresh:
                vx_ref = 0.0

            omega_ref = self.k_alpha * alpha + self.k_heading * e_psi

        vx_ref = float(np.clip(vx_ref, -self.v_max, self.v_max))
        omega_ref = float(np.clip(omega_ref, -self.omega_max, self.omega_max))

        vx_meas, _ = state.body_velocity()
        omega_meas = state.heading_rate

        vx_cmd = vx_ref + self.kv_inner * (vx_ref - vx_meas)
        omega_cmd = omega_ref + self.komega_inner * (omega_ref - omega_meas)

        vx_cmd = float(np.clip(vx_cmd, -self.v_max, self.v_max))
        omega_cmd = float(np.clip(omega_cmd, -self.omega_max, self.omega_max))

        wheel_rates = self.body_twist_to_wheel_rates(vx_cmd, omega_cmd)
        return DifferentialDriveCommand(
            vx=vx_cmd, vy=0.0, omega=omega_cmd, wheel_rates=wheel_rates
        )
