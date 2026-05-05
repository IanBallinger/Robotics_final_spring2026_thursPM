"""Local 2D EKF pose estimator inspired by robot_localization.

This module provides lightweight nonlinear filters for fusing:
- AprilTag global pose measurements: [x, y, yaw]
- IMU measurements:
  - body-frame linear accelerations ax, ay (used in prediction)
  - yaw rate wz (used directly in the process model for yaw propagation)
- Wheel encoder twist measurements:
  - body-frame vx, vy only
  - wheel-derived wz is intentionally not fused

State definition (9D):
    x = [px, py, yaw, vx_body, vy_body, wz, b_ax, b_ay, b_wz]

where:
- px, py are global/world position coordinates
- yaw is global heading
- vx_body, vy_body are body-frame translational velocities
- wz stores the latest yaw rate supplied to the process model
- b_ax, b_ay are accelerometer biases
- b_wz is retained for backward compatibility but is not used in yaw propagation

The process model is:
    px_{k+1} = px_k + dt * (cos(yaw) * vx_body - sin(yaw) * vy_body)
    py_{k+1} = py_k + dt * (sin(yaw) * vx_body + cos(yaw) * vy_body)
    yaw_{k+1}= yaw_k + dt * imu.wz
    vx_{k+1} = vx_k + dt * (ax_body - b_ax)
    vy_{k+1} = vy_k + dt * (ay_body - b_ay)
    wz_{k+1} = imu.wz
    b_ax_{k+1} = b_ax_k
    b_ay_{k+1} = b_ay_k
    b_wz_{k+1} = b_wz_k
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

STATE_DIM = 9
POSE_MEAS_DIM = 3
GYRO_MEAS_DIM = 1
WHEEL_TWIST_MEAS_DIM = 2

PX = 0
PY = 1
YAW = 2
VX = 3
VY = 4
WZ = 5
BAX = 6
BAY = 7
BWZ = 8


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class IMUMeasurement:
    """IMU sample used by the filters.

    Attributes
    ----------
    ax : float
        Body-frame x acceleration [m/s^2].
    ay : float
        Body-frame y acceleration [m/s^2].
    wz : float
        Measured yaw rate [rad/s].
    """

    ax: float = 0.0
    ay: float = 0.0
    wz: float = 0.0


@dataclass
class AprilTagMeasurement:
    """Global AprilTag pose estimate.

    Attributes
    ----------
    x : float
        Global x position [m].
    y : float
        Global y position [m].
    yaw : float
        Global yaw angle [rad].
    covariance : Optional[np.ndarray]
        Optional 3x3 measurement covariance for [x, y, yaw].
    """

    x: float
    y: float
    yaw: float
    covariance: Optional[np.ndarray] = None

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], dtype=float)


@dataclass
class WheelTwistMeasurement:
    """Body-frame velocity measurement derived from wheel encoders.

    Attributes
    ----------
    vx : float
        Body-frame longitudinal velocity [m/s].
    vy : float
        Body-frame lateral velocity [m/s]. For this platform this is expected to
        be zero, but it is kept for filter/state consistency.
    wz : float
        Optional wheel-derived yaw rate [rad/s]. This field is accepted for API
        compatibility but is intentionally ignored by the filter.
    covariance : Optional[np.ndarray]
        Optional 2x2 or 3x3 measurement covariance. Only the ``vx, vy`` block is
        used.
    """

    vx: float
    vy: float
    wz: float = 0.0
    covariance: Optional[np.ndarray] = None

    def as_vector(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)


class _BaseLocalizationFilter:
    """Common utilities shared by the localization filter implementation."""

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        apriltag_measurement_noise: Optional[np.ndarray] = None,
        gyro_measurement_noise: Optional[np.ndarray] = None,
        wheel_twist_measurement_noise: Optional[np.ndarray] = None,
    ):
        self.state = self._coerce_initial_state(initial_state)
        self.covariance = self._coerce_square_matrix(
            initial_covariance,
            default=np.eye(STATE_DIM, dtype=float) * 1e-2,
        )

        # Process noise is defined per second and scaled by dt in predict().
        self.process_noise = self._coerce_square_matrix(
            process_noise,
            default=np.diag([5e-3, 5e-3, 5e-3, 5e-2, 5e-2, 2e-2, 5e-4, 5e-4, 5e-4]),
        )

        self.apriltag_measurement_noise = (
            np.diag([2.5e-2, 2.5e-2, 4e-2])
            if apriltag_measurement_noise is None
            else np.asarray(apriltag_measurement_noise, dtype=float).reshape(
                POSE_MEAS_DIM, POSE_MEAS_DIM
            )
        )

        self.gyro_measurement_noise = (
            np.array([[2e-1]], dtype=float)
            if gyro_measurement_noise is None
            else np.asarray(gyro_measurement_noise, dtype=float).reshape(
                GYRO_MEAS_DIM, GYRO_MEAS_DIM
            )
        )

        if wheel_twist_measurement_noise is None:
            self.wheel_twist_measurement_noise = np.diag([2e-2, 2e-2])
        else:
            wheel_noise = np.asarray(wheel_twist_measurement_noise, dtype=float)
            if wheel_noise.shape == (WHEEL_TWIST_MEAS_DIM, WHEEL_TWIST_MEAS_DIM):
                self.wheel_twist_measurement_noise = wheel_noise.copy()
            elif wheel_noise.shape == (3, 3):
                self.wheel_twist_measurement_noise = wheel_noise[:2, :2].copy()
            else:
                raise ValueError(
                    "wheel_twist_measurement_noise must have shape (2, 2) or (3, 3)"
                )

    @staticmethod
    def _coerce_initial_state(initial_state: Optional[np.ndarray]) -> np.ndarray:
        if initial_state is None:
            return np.zeros(STATE_DIM, dtype=float)
        arr = np.asarray(initial_state, dtype=float).reshape(-1)
        if arr.size == STATE_DIM:
            return arr.copy()
        raise ValueError(f"initial_state must have length {STATE_DIM}")

    @staticmethod
    def _coerce_square_matrix(
        matrix: Optional[np.ndarray],
        *,
        default: np.ndarray,
    ) -> np.ndarray:
        if matrix is None:
            return np.asarray(default, dtype=float).copy()
        arr = np.asarray(matrix, dtype=float)
        if arr.shape == (STATE_DIM, STATE_DIM):
            return arr.copy()
        raise ValueError(
            f"matrix must have shape {(STATE_DIM, STATE_DIM)}"
        )

    @staticmethod
    def _process_model(state: np.ndarray, imu: IMUMeasurement, dt: float) -> np.ndarray:
        """Nonlinear motion model with body-frame velocity states."""
        x = np.asarray(state, dtype=float).copy()
        yaw = x[YAW]
        c = np.cos(yaw)
        s = np.sin(yaw)

        out = x.copy()
        out[PX] = x[PX] + dt * (c * x[VX] - s * x[VY])
        out[PY] = x[PY] + dt * (s * x[VX] + c * x[VY])
        out[YAW] = wrap_angle(x[YAW] + dt * imu.wz)
        out[VX] = x[VX] + dt * (imu.ax - x[BAX])
        out[VY] = x[VY] + dt * (imu.ay - x[BAY])
        out[WZ] = float(imu.wz)
        out[BAX] = x[BAX]
        out[BAY] = x[BAY]
        out[BWZ] = x[BWZ]
        return out

    @staticmethod
    def _process_jacobian(state: np.ndarray, dt: float) -> np.ndarray:
        """Jacobian of the nonlinear process model for the EKF."""
        yaw = state[YAW]
        vx = state[VX]
        vy = state[VY]
        c = np.cos(yaw)
        s = np.sin(yaw)

        F = np.eye(STATE_DIM, dtype=float)
        F[PX, YAW] = dt * (-s * vx - c * vy)
        F[PX, VX] = dt * c
        F[PX, VY] = -dt * s

        F[PY, YAW] = dt * (c * vx - s * vy)
        F[PY, VX] = dt * s
        F[PY, VY] = dt * c

        F[WZ, WZ] = 0.0
        F[VX, BAX] = -dt
        F[VY, BAY] = -dt
        return F

    @staticmethod
    def _pose_measurement_model(state: np.ndarray) -> np.ndarray:
        return np.array([state[PX], state[PY], wrap_angle(state[YAW])], dtype=float)

    @staticmethod
    def _gyro_measurement_model(state: np.ndarray) -> np.ndarray:
        return np.array([state[WZ] + state[BWZ]], dtype=float)

    @staticmethod
    def _wheel_twist_measurement_model(state: np.ndarray) -> np.ndarray:
        return np.array([state[VX], state[VY]], dtype=float)

    @staticmethod
    def _pose_measurement_matrix() -> np.ndarray:
        H = np.zeros((POSE_MEAS_DIM, STATE_DIM), dtype=float)
        H[0, PX] = 1.0
        H[1, PY] = 1.0
        H[2, YAW] = 1.0
        return H

    @staticmethod
    def _gyro_measurement_matrix() -> np.ndarray:
        H = np.zeros((GYRO_MEAS_DIM, STATE_DIM), dtype=float)
        H[0, WZ] = 1.0
        H[0, BWZ] = 1.0
        return H

    @staticmethod
    def _wheel_twist_measurement_matrix() -> np.ndarray:
        H = np.zeros((WHEEL_TWIST_MEAS_DIM, STATE_DIM), dtype=float)
        H[0, VX] = 1.0
        H[1, VY] = 1.0
        return H

    def reset(self, state: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        self.state = self._coerce_initial_state(state)
        self.covariance = self._coerce_square_matrix(
            covariance,
            default=np.eye(STATE_DIM, dtype=float) * 1e-2,
        )
        self.state[YAW] = wrap_angle(self.state[YAW])
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def get_covariance(self) -> np.ndarray:
        return self.covariance.copy()

    def pose(self) -> Tuple[float, float, float]:
        return float(self.state[PX]), float(self.state[PY]), float(self.state[YAW])

    def velocity_body(self) -> Tuple[float, float, float]:
        return float(self.state[VX]), float(self.state[VY]), float(self.state[WZ])

    def imu_biases(self) -> Tuple[float, float, float]:
        return float(self.state[BAX]), float(self.state[BAY]), float(self.state[BWZ])


class ExtendedKalmanFilter2D(_BaseLocalizationFilter):
    """Minimal 2D EKF for AprilTag + IMU + wheel-twist fusion."""

    def predict(self, imu: IMUMeasurement, dt: float) -> np.ndarray:
        if dt <= 0.0:
            return self.get_state()

        F = self._process_jacobian(self.state, dt)
        self.state = self._process_model(self.state, imu, dt)
        self.covariance = F @ self.covariance @ F.T + self.process_noise * dt
        self.state[YAW] = wrap_angle(self.state[YAW])
        return self.get_state()

    def update_apriltag(self, measurement: AprilTagMeasurement) -> np.ndarray:
        z = measurement.as_vector()
        z[2] = wrap_angle(z[2])
        h = self._pose_measurement_model(self.state)
        H = self._pose_measurement_matrix()
        R = (
            self.apriltag_measurement_noise
            if measurement.covariance is None
            else np.asarray(measurement.covariance, dtype=float).reshape(POSE_MEAS_DIM, POSE_MEAS_DIM)
        )

        innovation = z - h
        innovation[2] = wrap_angle(innovation[2])

        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.state[YAW] = wrap_angle(self.state[YAW])

        I = np.eye(STATE_DIM)
        self.covariance = (I - K @ H) @ self.covariance
        return self.get_state()

    def update_imu(self, imu: IMUMeasurement) -> np.ndarray:
        # IMU yaw rate is consumed directly in predict() for yaw propagation.
        # Keep the cached wz state in sync for downstream consumers, but do not
        # run a separate EKF measurement update on yaw rate.
        self.state[WZ] = float(imu.wz)
        self.state[YAW] = wrap_angle(self.state[YAW])
        return self.get_state()

    def update_wheel_twist(self, measurement: WheelTwistMeasurement) -> np.ndarray:
        z = measurement.as_vector()
        h = self._wheel_twist_measurement_model(self.state)
        H = self._wheel_twist_measurement_matrix()
        if measurement.covariance is None:
            R = self.wheel_twist_measurement_noise
        else:
            wheel_cov = np.asarray(measurement.covariance, dtype=float)
            if wheel_cov.shape == (WHEEL_TWIST_MEAS_DIM, WHEEL_TWIST_MEAS_DIM):
                R = wheel_cov
            elif wheel_cov.shape == (3, 3):
                R = wheel_cov[:2, :2]
            else:
                raise ValueError(
                    "wheel twist measurement covariance must have shape (2, 2) or (3, 3)"
                )

        innovation = z - h
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.state[YAW] = wrap_angle(self.state[YAW])

        I = np.eye(STATE_DIM)
        self.covariance = (I - K @ H) @ self.covariance
        return self.get_state()


__all__ = [
    "AprilTagMeasurement",
    "ExtendedKalmanFilter2D",
    "IMUMeasurement",
    "WheelTwistMeasurement",
    "wrap_angle",
]
