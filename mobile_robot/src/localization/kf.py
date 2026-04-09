"""
Kalman Filter for localization
Inputs:
 - current position and velocity estimate
 - current covariance estimate
 - position estimates from april tags
 - odometry

Outputs:
 - updated position and velocity estimate
 - updated covariance estimate
 - estimate cell from map
"""

import numpy as np


class KalmanFilter:
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray):
        self.state = initial_state
        self.covariance = initial_covariance

    def predict(self, dt: float):

        pass
