# Camera Vision + Live Plotter Integration

## Overview

The recording setup now integrates camera vision detections into the Julia live plotter. This allows you to visualize both robot TCP poses and camera-detected object positions in real-time as a 3D scatter plot.

## Quick Start

### 1. Start the Julia Live Plotter

```bash
# Terminal 1: Start Julia with single thread (required for GLFW)
julia --threads 1 live_plot_runner.jl
```

The plotter listens on:
- **Port 9999**: TCP poses from robot arms (small red points)
- **Port 9998**: Camera vision detections (large blue square points)

### 2. Run Robot State Recording

```bash
# Terminal 2: Run record.py to record robot state and stream TCP poses to Julia
python UR5/demo_4_16/record.py --stream-udp-host 127.0.0.1 --stream-udp-port 9999
```

This script:
- Records all robot state data to `robot_data.csv`
- Streams TCP poses to port 9999 for live visualization
- Ready to receive vision data (use `--vision-udp-host` and `--vision-udp-port` to enable)

### 3. Send Vision Data

Choose one of these approaches:

#### Option A: Run morphOps_streaming.py (RECOMMENDED - Canonical Example)

```bash
# Terminal 3: Run camera vision detection with streaming
python UR5/Camera\ examples/morphOps_streaming.py
```

This is the **canonical camera example** that detects all 5 tuned colors:
- Purple, Yellow, Green, Tan, Red (HSV calibrated)
- Streams centroid + bounding box for each detection
- Outputs both pixel and centimeter coordinates
- No additional integration needed - UDP streaming is built-in

#### Option B: Run ur5_full_example.py (with Vision Integration)

```bash
# Terminal 3: Run depth camera treadmill example
python UR5/Camera\ examples/ur5_full\ example.py
```

For this example, add vision sending to ur5_full_example.py (around line 50):

```python
from vision_data_integrator import VisionDataIntegrator

# After pipeline initialization:
vision_integrator = VisionDataIntegrator(scale_cm_to_m=True)

# In main loop, after computing y_target:
if y_target is not None:
    vision_integrator.send_y_target_position(y_target, x=0.0, z=0.0)
```

#### Option C: Send Simulated Vision Data (for testing)

```bash
# Terminal 3: Run test sender
python UR5/demo_4_16/vision_pose_sender.py
```

This sends 10 test positions to the plotter.

## File Reference

### Core Files

| File | Purpose |
|------|---------|
| `live_plot_runner.jl` | Julia live plotter with dual UDP listeners |
| `record.py` | Robot state recording + TCP pose streaming frontend |
| `vision_pose_sender.py` | Simple UDP sender for vision positions |
| `vision_data_integrator.py` | Higher-level camera integration helper |

### Canonical Camera Vision Example

| File | Purpose |
|------|---------|
| **`UR5/Camera examples/morphOps_streaming.py`** | **CANONICAL**: Multi-color HSV detection with UDP streaming to port 9998 |
| `UR5/Camera examples/morphOps.py` | Original single-color tuning tool (reference) |
| `UR5/Camera examples/ur5_full example.py` | Depth camera treadmill example |
| `UR5/Camera examples/Core/` | Perception stack (segmentation, pose estimation) |

## Architecture

```
Camera Vision Detection       Robot State Recording (record.py)
        ↓                              ↓
vision_data_integrator      --stream-udp-host 127.0.0.1
        ↓                              ↓
    Port 9998                      Port 9999
         ↖                            ↗
            live_plot_runner.jl
                    ↓
            Visualization (3D scatter)
```

## Key Points

### Coordinate System
- **Workspace calibration**: Centimeters [cm] for accuracy
- **Julia plotter**: Expects meters [m]
- **Automatic conversion**: VisionPoseSender and VisionDataIntegrator handle this

### Data Format

**TCP Poses (Port 9999)** - from record.py:
```json
{
  "timestamp": 1234567890,
  "actual_TCP_pose": [x, y, z, rx, ry, rz]
}
```

**Vision Detections (Port 9998)** - from camera:
```json
{
  "position": [x, y, z],
  "packet_number": 123
}
```

### Visualization

- **TCP Poses**: Small red points (size 5), color = orientation (Rx,Ry,Rz)
- **Vision Detections**: Large blue squares (size 8), color = position magnitude
- **Plot limits**: Auto-scales to fit all data
- **History**: Maintains last 4000 points (default, configurable)

## Usage Examples

### Basic: Record only (no live streaming)
```bash
python UR5/demo_4_16/record.py
```

### With TCP pose streaming
```bash
python UR5/demo_4_16/record.py \
  --stream-udp-host 127.0.0.1 \
  --stream-udp-port 9999
```

### With both TCP and vision streaming
```bash
python UR5/demo_4_16/record.py \
  --stream-udp-host 127.0.0.1 \
  --stream-udp-port 9999 \
  --vision-udp-host 127.0.0.1 \
  --vision-udp-port 9998
```

### Different robot IP
```bash
python UR5/demo_4_16/record.py \
  --robot-ip 192.168.1.102 \
  --stream-udp-host 127.0.0.1
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Julia plotter crashes | Run with `julia --threads 1` (required for GLFW on Windows) |
| No TCP points appear | Use `--stream-udp-host` flag when running record.py |
| No vision points appear | Check port 9998 is open, verify vision sender is running |
| Mixed coordinate frames | Ensure cm→m conversion: divide cm values by 100 |
| High packet drop | Increase `--refresh-every` in Julia (fewer plot updates) |
| Socket already in use | Change port numbers with `--stream-udp-port` or `--vision-udp-port` |
| Connection refused | Check Julia plotter is running, verify host/port settings |

## Running All Components Together

```bash
# Terminal 1: Start Julia live plotter
julia --threads 1 live_plot_runner.jl

# Terminal 2: Start robot state recording with TCP streaming
python UR5/demo_4_16/record.py \
  --stream-udp-host 127.0.0.1 \
  --stream-udp-port 9999

# Terminal 3: Start canonical camera vision detection (RECOMMENDED)
python UR5/Camera\ examples/morphOps_streaming.py

# Now both TCP poses and detected object centroids appear in real-time
```

### Alternative: With Custom Control Script

```bash
# Terminal 1: Julia plotter
julia --threads 1 live_plot_runner.jl

# Terminal 2: Robot + recording
python UR5/demo_4_16/record.py \
  --stream-udp-host 127.0.0.1 \
  --stream-udp-port 9999 \
  --vision-udp-host 127.0.0.1 \
  --vision-udp-port 9998

# Terminal 3: Custom camera or detection logic
# (sends vision data to port 9998 via send_vision_pose() or VisionPoseSender)
```

## Notes

- Workspace is calibrated to accurate-enough [cm] precision
- **Canonical camera example**: `morphOps_streaming.py` detects 5 HSV-tuned colors
  - Purple, Yellow, Green, Tan, Red
  - Outputs centroid + bounding box for each detection
  - Streams directly to port 9998 (no additional integration needed)
- Vision detections are sent to port 9998
- Robot TCP poses are sent to port 9999 via record.py
- All data is visualized simultaneously in the Julia 3D scatter plot
- The live plotter updates automatically as new data arrives
- Robot state recording continues regardless of UDP streaming status
