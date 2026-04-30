"""Local 2D EKF/UKF pose estimator inspired by robot_localization.

This module provides lightweight nonlinear filters for fusing:
- AprilTag global pose measurements: [x, y, yaw]
- IMU measurements:
  - body-frame linear accelerations ax, ay (used in prediction)
  - yaw rate wz (fused as a measurement update)

State definition (9D):
    x = [px, py, yaw, vx_body, vy_body, wz, b_ax, b_ay, b_wz]

where:
- px, py are global/world position coordinates
- yaw is global heading
- vx_body, vy_body are body-frame translational velocities
- wz is bias-corrected yaw rate
- b_ax, b_ay are accelerometer biases
- b_wz is gyro yaw-rate bias

The process model is:
    px_{k+1} = px_k + dt * (cos(yaw) * vx_body - sin(yaw) * vy_body)
    py_{k+1} = py_k + dt * (sin(yaw) * vx_body + cos(yaw) * vy_body)
    yaw_{k+1}= yaw_k + dt * wz
    vx_{k+1} = vx_k + dt * (ax_body - b_ax)
    vy_{k+1} = vy_k + dt * (ay_body - b_ay)
    wz_{k+1} = wz_k
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
WHEEL_TWIST_MEAS_DIM = 3

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
        Body-frame yaw rate [rad/s].
    covariance : Optional[np.ndarray]
        Optional 3x3 measurement covariance for [vx, vy, wz].
    """

    vx: float
    vy: float
    wz: float
    covariance: Optional[np.ndarray] = None

    def as_vector(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.wz], dtype=float)


class _BaseLocalizationFilter:
    """Common utilities shared by the EKF and UKF implementations."""

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

        self.wheel_twist_measurement_noise = (
            np.diag([2e-2, 2e-2, 8e-2])
            if wheel_twist_measurement_noise is None
            else np.asarray(wheel_twist_measurement_noise, dtype=float).reshape(
                WHEEL_TWIST_MEAS_DIM, WHEEL_TWIST_MEAS_DIM
            )
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
        out[YAW] = wrap_angle(x[YAW] + dt * x[WZ])
        out[VX] = x[VX] + dt * (imu.ax - x[BAX])
        out[VY] = x[VY] + dt * (imu.ay - x[BAY])
        out[WZ] = x[WZ]
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

        F[YAW, WZ] = dt
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
        return np.array([state[VX], state[VY], state[WZ]], dtype=float)

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
        H[2, WZ] = 1.0
        return H

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
        z = np.array([imu.wz], dtype=float)
        h = self._gyro_measurement_model(self.state)
        H = self._gyro_measurement_matrix()
        R = self.gyro_measurement_noise

        innovation = z - h
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.state[YAW] = wrap_angle(self.state[YAW])

        I = np.eye(STATE_DIM)
        self.covariance = (I - K @ H) @ self.covariance
        return self.get_state()

    def update_wheel_twist(self, measurement: WheelTwistMeasurement) -> np.ndarray:
        z = measurement.as_vector()
        h = self._wheel_twist_measurement_model(self.state)
        H = self._wheel_twist_measurement_matrix()
        R = (
            self.wheel_twist_measurement_noise
            if measurement.covariance is None
            else np.asarray(measurement.covariance, dtype=float).reshape(
                WHEEL_TWIST_MEAS_DIM, WHEEL_TWIST_MEAS_DIM
            )
        )

        innovation = z - h
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.state[YAW] = wrap_angle(self.state[YAW])

        I = np.eye(STATE_DIM)
        self.covariance = (I - K @ H) @ self.covariance
        return self.get_state()


class UnscentedKalmanFilter2D(_BaseLocalizationFilter):
    """Minimal 2D UKF for AprilTag + IMU + wheel-twist fusion."""

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        apriltag_measurement_noise: Optional[np.ndarray] = None,
        gyro_measurement_noise: Optional[np.ndarray] = None,
        wheel_twist_measurement_noise: Optional[np.ndarray] = None,
        alpha: float = 1e-1,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        super().__init__(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            process_noise=process_noise,
            apriltag_measurement_noise=apriltag_measurement_noise,
            gyro_measurement_noise=gyro_measurement_noise,
            wheel_twist_measurement_noise=wheel_twist_measurement_noise,
        )
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def _sigma_point_weights(self) -> Tuple[np.ndarray, np.ndarray, float]:
        n = STATE_DIM
        lam = self.alpha**2 * (n + self.kappa) - n

        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
        wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)
        return wm, wc, lam

    def _sigma_points(self) -> np.ndarray:
        wm, _, lam = self._sigma_point_weights()
        _ = wm  # suppress unused warning in pure Python linters

        n = STATE_DIM
        jitter = 1e-9 * np.eye(n)
        sqrt = np.linalg.cholesky((n + lam) * (self.covariance + jitter))

        sigma = np.zeros((2 * n + 1, n), dtype=float)
        sigma[0] = self.state
        for i in range(n):
            sigma[i + 1] = self.state + sqrt[:, i]
            sigma[n + i + 1] = self.state - sqrt[:, i]
            sigma[i + 1, YAW] = wrap_angle(sigma[i + 1, YAW])
            sigma[n + i + 1, YAW] = wrap_angle(sigma[n + i + 1, YAW])
        return sigma

    @staticmethod
    def _mean_from_sigma_points(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mean = np.zeros(points.shape[1], dtype=float)
        mean[:YAW] = np.sum(weights[:, None] * points[:, :YAW], axis=0)
        mean[VX:] = np.sum(weights[:, None] * points[:, VX:], axis=0)
        mean[PY] = np.sum(weights * points[:, PY])
        mean[PX] = np.sum(weights * points[:, PX])

        sin_sum = np.sum(weights * np.sin(points[:, YAW]))
        cos_sum = np.sum(weights * np.cos(points[:, YAW]))
        mean[YAW] = np.arctan2(sin_sum, cos_sum)
        return mean

    @staticmethod
    def _state_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        d[YAW] = wrap_angle(d[YAW])
        return d

    @staticmethod
    def _meas_diff_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        d[2] = wrap_angle(d[2])
        return d

    def predict(self, imu: IMUMeasurement, dt: float) -> np.ndarray:
        if dt <= 0.0:
            return self.get_state()

        wm, wc, _ = self._sigma_point_weights()
        sigma = self._sigma_points()
        propagated = np.array([self._process_model(sp, imu, dt) for sp in sigma])

        self.state = self._mean_from_sigma_points(propagated, wm)

        P = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        for i in range(propagated.shape[0]):
            dx = self._state_diff(propagated[i], self.state)
            P += wc[i] * np.outer(dx, dx)
        self.covariance = P + self.process_noise * dt
        self.state[YAW] = wrap_angle(self.state[YAW])
        return self.get_state()

    def update_apriltag(self, measurement: AprilTagMeasurement) -> np.ndarray:
        z = measurement.as_vector()
        z[2] = wrap_angle(z[2])
        R = (
            self.apriltag_measurement_noise
            if measurement.covariance is None
            else np.asarray(measurement.covariance, dtype=float).reshape(POSE_MEAS_DIM, POSE_MEAS_DIM)
        )

        wm, wc, _ = self._sigma_point_weights()
        sigma = self._sigma_points()
        meas_sigma = np.array([self._pose_measurement_model(sp) for sp in sigma])

        z_pred = np.zeros(POSE_MEAS_DIM, dtype=float)
        z_pred[0] = np.sum(wm * meas_sigma[:, 0])
        z_pred[1] = np.sum(wm * meas_sigma[:, 1])
        z_pred[2] = np.arctan2(
            np.sum(wm * np.sin(meas_sigma[:, 2])),
            np.sum(wm * np.cos(meas_sigma[:, 2])),
        )

        S = np.zeros((POSE_MEAS_DIM, POSE_MEAS_DIM), dtype=float)
        Pxz = np.zeros((STATE_DIM, POSE_MEAS_DIM), dtype=float)
        for i in range(meas_sigma.shape[0]):
            dz = self._meas_diff_pose(meas_sigma[i], z_pred)
            dx = self._state_diff(sigma[i], self.state)
            S += wc[i] * np.outer(dz, dz)
            Pxz += wc[i] * np.outer(dx, dz)
        S += R

        innovation = self._meas_diff_pose(z, z_pred)
        K = Pxz @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.state[YAW] = wrap_angle(self.state[YAW])
        self.covariance = self.covariance - K @ S @ K.T
        return self.get_state()

    def update_imu(self, imu: IMUMeasurement) -> np.ndarray:
        z = np.array([imu.wz], dtype=float)
        R = self.gyro_measurement_noise

        wm, wc, _ = self._sigma_point_weights()
        sigma = self._sigma_points()
        meas_sigma = np.array([self._gyro_measurement_model(sp) for sp in sigma])
        z_pred = np.sum(wm[:, None] * meas_sigma, axis=0)

        S = np.zeros((GYRO_MEAS_DIM, GYRO_MEAS_DIM), dtype=float)
        Pxz = np.zeros((STATE_DIM, GYRO_MEAS_DIM), dtype=float)
        for i in range(meas_sigma.shape[0]):
            dz = meas_sigma[i] - z_pred
            dx = self._state_diff(sigma[i], self.state)
            S += wc[i] * np.outer(dz, dz)
            Pxz += wc[i] * np.outer(dx, dz)
        S += R

        innovation = z - z_pred
        K = Pxz @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.state[YAW] = wrap_angle(self.state[YAW])
        self.covariance = self.covariance - K @ S @ K.T
        return self.get_state()

    def update_wheel_twist(self, measurement: WheelTwistMeasurement) -> np.ndarray:
        z = measurement.as_vector()
        R = (
            self.wheel_twist_measurement_noise
            if measurement.covariance is None
            else np.asarray(measurement.covariance, dtype=float).reshape(
                WHEEL_TWIST_MEAS_DIM, WHEEL_TWIST_MEAS_DIM
            )
        )

        wm, wc, _ = self._sigma_point_weights()
        sigma = self._sigma_points()
        meas_sigma = np.array([self._wheel_twist_measurement_model(sp) for sp in sigma])
        z_pred = np.sum(wm[:, None] * meas_sigma, axis=0)

        S = np.zeros((WHEEL_TWIST_MEAS_DIM, WHEEL_TWIST_MEAS_DIM), dtype=float)
        Pxz = np.zeros((STATE_DIM, WHEEL_TWIST_MEAS_DIM), dtype=float)
        for i in range(meas_sigma.shape[0]):
            dz = meas_sigma[i] - z_pred
            dx = self._state_diff(sigma[i], self.state)
            S += wc[i] * np.outer(dz, dz)
            Pxz += wc[i] * np.outer(dx, dz)
        S += R

        innovation = z - z_pred
        K = Pxz @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.state[YAW] = wrap_angle(self.state[YAW])
        self.covariance = self.covariance - K @ S @ K.T
        return self.get_state()


# Backward-compatible alias for the original placeholder class name.


__all__ = [
    "AprilTagMeasurement",
    "ExtendedKalmanFilter2D",
    "IMUMeasurement",
    "UnscentedKalmanFilter2D",
    "WheelTwistMeasurement",
    "wrap_angle",
]
