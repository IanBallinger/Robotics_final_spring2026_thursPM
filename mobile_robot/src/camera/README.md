# Camera package

This package centralizes camera access and camera-frame geometry utilities for the mobile robot.

## Scope

- Open an RGB camera with OpenCV
- Open an Intel RealSense camera and capture synchronized color + depth frames
- Apply common image transforms
- Convert depth pixels into 3D camera coordinates
- Convert camera-frame points into world coordinates
- Project image pixels onto a known world plane

## Main modules

- `opencv_camera.py`: generic USB / laptop camera capture
- `realsense_camera.py`: Intel RealSense RGB-D capture
- `transforms.py`: image-space and geometry-space transforms
- `types.py`: shared dataclasses for frames, intrinsics, and extrinsics

## Intended downstream use

This package is designed to support later perception modules for:

- AprilTag pose estimation
- tray detection
- dishwasher handle localization
- pixel-to-world coordinate conversion for grasping / interaction

## Example: USB camera

```python
from camera import OpenCVCamera, bgr_to_gray

with OpenCVCamera(device_index=0) as camera:
    frame = camera.read()
    gray = bgr_to_gray(frame.color)
```

## Example: RealSense depth back-projection

```python
from camera import RealSenseCamera, depth_pixel_to_camera_point, depth_to_meters

with RealSenseCamera() as camera:
    frame = camera.read()
    depth_m = depth_to_meters(frame.depth, frame.depth_scale)
    point_c = depth_pixel_to_camera_point(640, 360, depth_m[360, 640], frame.intrinsics)
```

## Notes

- `pyrealsense2` must be installed on the machine to use `RealSenseCamera`.
- Camera-to-world transforms require calibrated extrinsics supplied by the caller.
- For ground-plane reasoning, `pixels_to_world_on_plane(...)` assumes a plane model in the world frame.
