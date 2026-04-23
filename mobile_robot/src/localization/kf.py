"""
Extended Kalman Filter (EKF) for mobile robot localization using:

- IMU local-frame acceleration: ax_body, ay_body
- IMU yaw rate: omega
- IMU estimated local-frame velocities: vx_body, vy_body
- AprilTag global pose: x, y, theta

State vector:
    x = [px, py, theta, vx_body, vy_body, omega]^T

where
    px, py     : global position
    theta      : global heading
    vx_body    : robot/body-frame x velocity
    vy_body    : robot/body-frame y velocity
    omega      : yaw rate

Process input:
    u = [ax_body, ay_body, omega_meas]^T

Prediction model:
    px_{k+1}    = px_k + (cos(theta_k)*vx_k - sin(theta_k)*vy_k) * dt
    py_{k+1}    = py_k + (sin(theta_k)*vx_k + cos(theta_k)*vy_k) * dt
    theta_{k+1} = theta_k + omega_meas * dt
    vx_{k+1}    = vx_k + ax_body * dt
    vy_{k+1}    = vy_k + ay_body * dt
    omega_{k+1} = omega_meas

Measurement models:
    IMU velocity estimate:
        z_imu_vel = [vx_body, vy_body]
        or
        z_imu_vel_omega = [vx_body, vy_body, omega]

    AprilTag pose:
        z_tag = [px, py, theta]
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


class ExtendedKalmanFilter:
    """EKF with body-frame velocity states and IMU acceleration-driven prediction."""

    N = 6

    def __init__(
        self,
        dt: float,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        R_imu_vel: np.ndarray,
        R_tag: Optional[np.ndarray] = None,
        R_imu_vel_omega: Optional[np.ndarray] = None,
    ):
        self.dt = float(dt)

        self.state = np.array(initial_state, dtype=float).flatten()
        assert self.state.shape == (self.N,), (
            f"State must be length {self.N}, got {self.state.shape}"
        )

        self.P = np.array(initial_covariance, dtype=float).reshape(self.N, self.N)
        self.Q = np.array(process_noise, dtype=float).reshape(self.N, self.N)

        self.R_imu_vel = np.array(R_imu_vel, dtype=float)
        self.R_tag = np.array(R_tag, dtype=float) if R_tag is not None else None
        self.R_imu_vel_omega = (
            np.array(R_imu_vel_omega, dtype=float)
            if R_imu_vel_omega is not None
            else None
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def theta(self) -> float:
        return float(self.state[2])

    @property
    def vx_body(self) -> float:
        return float(self.state[3])

    @property
    def vy_body(self) -> float:
        return float(self.state[4])

    @property
    def omega(self) -> float:
        return float(self.state[5])

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    def predict(
        self,
        ax_body: float,
        ay_body: float,
        omega_meas: float,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict using IMU local acceleration and measured yaw rate.

        Parameters
        ----------
        ax_body : float
            Local/body-frame x acceleration (m/s^2).
        ay_body : float
            Local/body-frame y acceleration (m/s^2).
        omega_meas : float
            Measured yaw rate (rad/s).
        dt : float or None
            Time step.
        """
        if dt is None:
            dt = self.dt

        px, py, theta, vx, vy, _ = self.state

        c = np.cos(theta)
        s = np.sin(theta)

        # pose propagation using body-frame velocities rotated into world frame
        px_next = px + (c * vx - s * vy) * dt
        py_next = py + (s * vx + c * vy) * dt
        theta_next = self._wrap_angle(theta + omega_meas * dt)

        # body-frame velocity propagation using body-frame acceleration
        vx_next = vx + ax_body * dt
        vy_next = vy + ay_body * dt
        omega_next = omega_meas

        self.state = np.array([
            px_next,
            py_next,
            theta_next,
            vx_next,
            vy_next,
            omega_next,
        ])

        # Jacobian F = df/dx
        F = np.array([
            [1.0, 0.0, (-s * vx - c * vy) * dt, c * dt, -s * dt, 0.0],
            [0.0, 1.0, ( c * vx - s * vy) * dt, s * dt,  c * dt, 0.0],
            [0.0, 0.0, 1.0,                       0.0,    0.0,    0.0],
            [0.0, 0.0, 0.0,                       1.0,    0.0,    0.0],
            [0.0, 0.0, 0.0,                       0.0,    1.0,    0.0],
            [0.0, 0.0, 0.0,                       0.0,    0.0,    0.0],
        ])

        # Optional control Jacobian if you want input-noise modeling later:
        # u = [ax_body, ay_body, omega_meas]
        # G = df/du
        # Here we fold uncertainty into additive Q for simplicity.

        self.P = F @ self.P @ F.T + self.Q
        self.state[2] = self._wrap_angle(self.state[2])

        return self.state.copy()

    # ------------------------------------------------------------------
    # UPDATE — IMU estimated local velocities only
    # ------------------------------------------------------------------
    def update_imu_velocity(
        self,
        vx_body_meas: float,
        vy_body_meas: float,
        R_imu_vel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse IMU-estimated body-frame velocity [vx_body, vy_body].
        """
        z = np.array([vx_body_meas, vy_body_meas])

        H = np.zeros((2, self.N))
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        z_hat = H @ self.state
        y_innov = z - z_hat

        R = (
            np.array(R_imu_vel, dtype=float).reshape(2, 2)
            if R_imu_vel is not None
            else self.R_imu_vel.reshape(2, 2)
        )

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y_innov).flatten()
        self.state[2] = self._wrap_angle(self.state[2])

        I = np.eye(self.N)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        return self.state.copy()

    # ------------------------------------------------------------------
    # UPDATE — IMU estimated local velocities + omega
    # ------------------------------------------------------------------
    def update_imu_velocity_and_omega(
        self,
        vx_body_meas: float,
        vy_body_meas: float,
        omega_meas: float,
        R_imu_vel_omega: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse IMU-estimated [vx_body, vy_body, omega].
        """
        z = np.array([vx_body_meas, vy_body_meas, omega_meas])

        H = np.zeros((3, self.N))
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        H[2, 5] = 1.0

        z_hat = H @ self.state
        y_innov = z - z_hat

        if R_imu_vel_omega is not None:
            R = np.array(R_imu_vel_omega, dtype=float).reshape(3, 3)
        elif self.R_imu_vel_omega is not None:
            R = self.R_imu_vel_omega.reshape(3, 3)
        else:
            # fallback: combine velocity noise with a default omega variance
            R2 = self.R_imu_vel.reshape(2, 2)
            R = np.array([
                [R2[0, 0], R2[0, 1], 0.0],
                [R2[1, 0], R2[1, 1], 0.0],
                [0.0,      0.0,      0.01],
            ])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y_innov).flatten()
        self.state[2] = self._wrap_angle(self.state[2])

        I = np.eye(self.N)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        return self.state.copy()

    # ------------------------------------------------------------------
    # UPDATE — AprilTag pose
    # ------------------------------------------------------------------
    def update_apriltag_pose(
        self,
        x_meas: float,
        y_meas: float,
        theta_meas: float,
        R_tag: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse AprilTag global pose [x, y, theta].
        """
        z = np.array([x_meas, y_meas, theta_meas])

        H = np.zeros((3, self.N))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        z_hat = H @ self.state
        y_innov = z - z_hat
        y_innov[2] = self._wrap_angle(y_innov[2])

        if R_tag is not None:
            R = np.array(R_tag, dtype=float).reshape(3, 3)
        elif self.R_tag is not None:
            R = self.R_tag.reshape(3, 3)
        else:
            R = np.diag([0.05, 0.05, 0.01])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y_innov).flatten()
        self.state[2] = self._wrap_angle(self.state[2])

        I = np.eye(self.N)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        return self.state.copy()

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def get_heading(self) -> float:
        return self.theta

    def get_body_velocity(self) -> Tuple[float, float]:
        return (self.vx_body, self.vy_body)

    def get_yaw_rate(self) -> float:
        return self.omega

    def __repr__(self) -> str:
        return (
            f"ExtendedKalmanFilter("
            f"x={self.x:.3f}, y={self.y:.3f}, "
            f"theta={np.degrees(self.theta):.1f}°, "
            f"v_body=({self.vx_body:.3f},{self.vy_body:.3f}), "
            f"omega={self.omega:.3f})"
        )