# morphOps Streaming Integration Guide

## Overview

`morphOps_streaming.py` is the canonical camera vision example that streams detected object centroids and bounding boxes to the Julia live plotter. It detects 5 colored objects (purple, yellow, green, tan, red) using HSV thresholding with the exact calibrated values from the course.

## Key Features

### Detected Colors (HSV Tuned)
- **Purple**: H:[156-180], S:[83-176], V:[0-143]
- **Yellow**: H:[13-98], S:[255], V:[120-208]
- **Green**: H:[53-87], S:[90-180], V:[128-221]
- **Tan**: H:[40-56], S:[71-195], V:[139-255]
- **Red**: H:[1-3], S:[180-255], V:[131-236]

### Output Data
For each detected object, sends to port 9998:
```json
{
  "position": [x_m, y_m, 0.0],
  "color": "color_name",
  "x_cm": x_position,
  "y_cm": y_position,
  "bbox": {"x": px, "y": py, "w": pw, "h": ph}
}
```

### Coordinate System
- Origin at camera center (pixel 337.5, 337.5)
- Conversion: 54 pixels = 27.5 cm (CM_PIXEL = 54.0 / 275)
- X-axis: positive right, negative left
- Y-axis: positive up, negative down
- All positions output in both pixels (for images) and centimeters (for world frame)

## Usage

### Start Julia Live Plotter
```bash
julia --threads 1 live_plot_runner.jl
```

### Run Record.py (Robot State)
```bash
python UR5/demo_4_16/record.py \
  --stream-udp-host 127.0.0.1 \
  --stream-udp-port 9999
```

### Run morphOps Streaming (Camera Vision)
```bash
python UR5/Camera\ examples/morphOps_streaming.py
```

Now the Julia plotter displays:
- **Red/small points** (port 9999): Robot TCP poses
- **Blue/large squares** (port 9998): Detected object centroids

## Output in Terminal

The script prints detections every 30 frames:
```
[30] Detected 2 objects: purple (5.43, 12.21), green (-3.12, 8.54)
[60] Detected 1 objects: yellow (0.15, 6.78)
```

## Integration with Control Scripts

### From record.py
Call `send_vision_pose()` during control loop:
```python
from record import send_vision_pose

# In your control code:
if vision_socket and vision_target:
    send_vision_pose(vision_socket, vision_target, x_m, y_m, z_m, packet_num)
```

### From Control System
Modify ur5_full_example.py to use morphOps detections:
```python
import subprocess
import json
import socket

# Start morphOps in background
proc = subprocess.Popen(['python', 'UR5/Camera examples/morphOps_streaming.py'])

# Later in your code, read detection data as needed
# The UDP stream flows directly to Julia
```

## Files

| File | Purpose |
|------|---------|
| `morphOps_streaming.py` | **CANONICAL**: Multi-color detection with UDP streaming |
| `morphOps.py` | Original single-color tuning tool (reference) |
| `live_plot_runner.jl` | Julia visualizer receiving both TCP and vision data |
| `record.py` | Robot state recording (sends TCP poses to port 9999) |

## Coordinate Reference

```
Camera View (top-down):
     
    Y+
     |
X- --+-- X+
     |
    Y-

Center: (337.5, 337.5) pixels = (0, 0) cm

CM_PIXEL = 54.0 / 275 ≈ 0.196 cm/pixel
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Can't open camera" | Check camera is connected (usually /dev/video2), modify `cv2.VideoCapture(2)` if needed |
| No detections | Verify object colors match HSV tuning, adjust `cv2.contourArea()` threshold if objects too small |
| No UDP packets | Ensure Julia plotter is running and listening on port 9998 |
| Julia shows only red points | Camera script isn't running; start morphOps_streaming.py |
| Detections lag on plot | Increase Julia's `--refresh-every` parameter to reduce update frequency |

## Extending morphOps_streaming.py

### Add New Color Detection
```python
colors['custom'] = {
    'lower': np.array([h_low, s_low, v_low]),
    'upper': np.array([h_high, s_high, v_high])
}
```

### Change Detection Sensitivity
```python
# Increase minimum area threshold (default 500)
if cv2.contourArea(cnt) > 1000:  # Ignore smaller objects
```

### Modify Output Data
Edit the `send_detection()` function to include additional fields:
```python
packet["depth"] = depth_estimate  # Add if using depth camera
packet["confidence"] = detection_confidence
```

## Notes

- The morphOps tool was originally designed for single-color tuning (see original comments)
- This streaming version runs **all** color detections simultaneously
- Workspace calibrated to centimeters for accuracy
- Julia plotter auto-scales axes to fit all incoming data (TCP + vision)
- Maximum 4000 points kept in history per data stream
