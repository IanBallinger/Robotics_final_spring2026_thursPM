# Pure-Python Perception Stack (Intel RealSense D435i)
Segmentation + CAD-free 6D Pose (relative to the camera)

This project is a minimal perception stack that:
- streams **RGB + depth** from a D435i
- **segments** objects in the RGBD scene
- computes a **CAD-free 6D pose proxy** for each object **in the camera frame**
- **tracks** objects across frames for stable IDs and smoothing
- shows a live debug view (boxes, optional masks, optional pose axes)

The system outputs poses **relative to the camera** only (no robot base/world transforms).

---

## Output
For each tracked object, the system outputs:
- `track_id` (stable ID)
- `label` (only meaningful in YOLO mode, otherwise `"unknown"`)
- `position_m` in camera frame (meters): `[x, y, z]`
- `quaternion_xyzw` in camera frame: `[qx, qy, qz, qw]`
- `confidence` (0 to 1)
- `flags`:
  - `rotation_ambiguous` (likely symmetry)
  - `too_few_points` (pose unreliable)
  - `nearly_planar` (object points look mostly planar)

Example console print:
id=1 label=unknown p=[ 0.121,-0.053, 0.681] q=[ 0.010, 0.707,-0.020, 0.706] conf=0.62 pts=1800


---

## What “6D pose” means in this project (no CAD models)
Because CAD models are not used, the orientation is a **geometric proxy**:
- **Position**: the median (robust centroid) of the object’s 3D points
- **Orientation**: PCA axes of the object’s point cloud (a consistent “shape frame”)

This is often sufficient for grasping and coarse placement.

Some objects are inherently ambiguous:
- cylinder/bottle: rotation about the long axis is not uniquely defined
- plate: yaw about the normal is ambiguous

When the pose is likely ambiguous, the system sets `rotation_ambiguous=True`. Downstream code should avoid over-trusting yaw for these objects.

---

## Camera coordinate convention
The 3D points are computed from depth using camera intrinsics:
- `z` is depth (forward from the camera)
- `x` increases to the right in the image
- `y` increases downward in the image

This matches common camera optical frame conventions when using pixel back-projection:
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth(u,v)


---

## File layout (4 files)
- `Core/realsense_io.py`  
  Captures frames from the D435i:
  - aligns depth to color
  - returns `rgb`, `depth_m`, `intrinsics`, `timestamp_ms`

- `Core/segmentation.py`  
  Provides segmentation masks:
  - default `depth_cc` = depth connected-components (fast and laptop-friendly)
  - optional `yolo` = YOLOv8 segmentation (requires `ultralytics`)
  - optional `SAM refinement` utility class (not used by default)

- `Core/pose_from_depth.py`  
  Pose estimation:
  - mask + depth + intrinsics → 3D points in camera frame
  - PCA-based pose proxy: `(position, quaternion)`
  - includes outlier filtering and ambiguity flags

- `Core/run_perception.py`  
  Main runner:
  - capture → segment → pose → track → visualize and print

---

## Requirements

### Hardware
- Intel RealSense **D435i**

### Software
- Python 3.9+ recommended

### Python dependencies (minimal)
```bash
pip install numpy opencv-python pyrealsense2
