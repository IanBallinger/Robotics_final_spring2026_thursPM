from src.localization.kf import ExtendedKalmanFilter
import numpy as np

# --- parameters ---
R = 0.05        # wheel radius (m)
b = 0.2         # wheel base (m)
dt = 0.1        # timestep (s)

# --- initial state ---
x0 = np.zeros(6)   # [x, y, theta, x_dot, y_dot, omega]

# --- covariance ---
P0 = np.eye(6) * 0.1

# --- process noise ---
Q = np.eye(6) * 0.01

# --- input noise (wheel noise) ---
Q_u = np.eye(2) * 0.001

# --- measurement noise ---
R_imu = np.array([[0.01]])              # omega only
R_tag = np.diag([0.05, 0.05, 0.02])     # x, y, theta

# --- create EKF ---
ekf = ExtendedKalmanFilter(
    wheel_radius=R,
    wheel_base=b,
    dt=dt,
    initial_state=x0,
    initial_covariance=P0,
    process_noise=Q,
    input_noise=Q_u,
    R_imu=R_imu,
    R_tag=R_tag
)

# --- simulate loop ---
for k in range(50):

    # fake wheel increments (robot moving forward)
    d_r = 0.1
    d_l = 0.1

    # --- PREDICT ---
    ekf.predict(d_r, d_l)

    # --- IMU update (every step) ---
    omega_meas = 0.0 + np.random.randn() * 0.01
    ekf.update_imu_omega(omega_meas)

    # --- AprilTag update (every 5 steps) ---
    if k % 5 == 0:
        x_meas = ekf.x + np.random.randn() * 0.05
        y_meas = ekf.y + np.random.randn() * 0.05
        theta_meas = ekf.theta + np.random.randn() * 0.02

        ekf.update_apriltag_pose(x_meas, y_meas, theta_meas)

    # print state
    print(ekf.get_state())