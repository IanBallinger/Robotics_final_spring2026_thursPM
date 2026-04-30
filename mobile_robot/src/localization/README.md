# Localization filters

This directory now includes a lightweight local 2D EKF/UKF implementation in `kf.py` inspired by the architecture of `robot_localization`.

## State

The filter state is

```text
[x, y, yaw, vx_body, vy_body, wz, b_ax, b_ay, b_wz]
```

where:
- `x, y, yaw` are global pose
- `vx_body, vy_body` are body-frame translational velocities
- `wz` is bias-corrected yaw rate
- `b_ax, b_ay` are accelerometer bias states
- `b_wz` is gyro yaw-rate bias

## Inputs fused

### AprilTag pose update
Use global AprilTag pose estimates as a direct pose measurement:

```python
AprilTagMeasurement(x, y, yaw, covariance=None)
```

### IMU
Use body-frame accelerations in prediction and fuse gyro yaw-rate in update:

```python
IMUMeasurement(ax, ay, wz)
```

## Example

```python
import numpy as np
from localization import (
    AprilTagMeasurement,
    ExtendedKalmanFilter2D,
    IMUMeasurement,
)

kf = ExtendedKalmanFilter2D(
    initial_state=np.zeros(6),
    initial_covariance=np.eye(9) * 0.1,
)

imu = IMUMeasurement(ax=0.1, ay=0.0, wz=0.05)
kf.predict(imu, dt=0.02)
kf.update_imu(imu)

apriltag = AprilTagMeasurement(x=1.2, y=0.4, yaw=0.1)
kf.update_apriltag(apriltag)

print(kf.pose())
```

## Notes

- This is not a full port of the ROS `robot_localization` package.
- It is a compact local implementation for this repository.
- If you need bias estimation, 3D orientation, or full ROS message/config compatibility, those would need to be added separately.
