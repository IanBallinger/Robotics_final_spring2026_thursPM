"""
Camera Vision to Julia Live Plotter Integration
===============================================

This example shows how to integrate camera vision detection (from ur5_full_example.py)
with the keyboard_proto.py recording setup to send detected object positions to the
Julia live plotter in real-time.

The workspace uses centimeters [cm] for calibration accuracy.
"""

import socket
import json
import numpy as np
from typing import Optional

class VisionDataIntegrator:
    """Integrates camera vision detections with the live plotter."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9998, scale_cm_to_m: bool = True):
        """
        Initialize the vision data integrator.
        
        Args:
            host: IP address of Julia plotter
            port: UDP port for vision poses (default: 9998)
            scale_cm_to_m: If True, convert from cm (workspace calibration) to m (plotter)
        """
        self.host = host
        self.port = port
        self.scale_cm_to_m = scale_cm_to_m
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.packet_count = 0
        
    def send_detected_object(self, detection_result: dict):
        """
        Send a detected object from camera vision to the live plotter.
        
        Expected detection_result keys:
            'x': X position (cm or m depending on scale_cm_to_m)
            'y': Y position
            'z': Z position (depth)
            'confidence': Optional confidence score
            
        Args:
            detection_result: Dictionary with position data
        """
        try:
            x = float(detection_result.get('x', 0))
            y = float(detection_result.get('y', 0))
            z = float(detection_result.get('z', 0))
            
            # Convert from cm to m if needed
            if self.scale_cm_to_m:
                x, y, z = x / 100.0, y / 100.0, z / 100.0
            
            packet = {
                "position": [x, y, z],
                "packet_number": self.packet_count,
                "confidence": detection_result.get('confidence', 1.0)
            }
            
            msg = json.dumps(packet)
            self.sock.sendto(msg.encode(), (self.host, self.port))
            self.packet_count += 1
            return True
            
        except Exception as e:
            print(f"Failed to send detection: {e}")
            return False
    
    def send_y_target_position(self, y_target: float, x: float = 0.0, z: float = 0.0):
        """
        Convenience method for sending a y_target position (from ur5_full_example.py).
        
        This handles the common case where only Y position is detected from the treadmill.
        
        Args:
            y_target: Y position from camera (calibrated in meters or cm)
            x: X position (default 0)
            z: Z position/depth (default 0)
        """
        return self.send_detected_object({'x': x, 'y': y_target, 'z': z})
    
    def close(self):
        """Close the UDP socket."""
        try:
            self.sock.close()
        except:
            pass


# ============================================================================
# EXAMPLE: How to modify ur5_full_example.py to send data to Julia plotter
# ============================================================================

EXAMPLE_CODE = """
# In ur5_full_example.py, add at the top after other imports:
from vision_data_integrator import VisionDataIntegrator

# After initializing the pipeline, add:
vision_integrator = VisionDataIntegrator(scale_cm_to_m=True)

# In the main loop, after computing y_target, send it to the live plotter:
if y_target is not None:
    # y_target is already in meters from the ur5_full_example code
    vision_integrator.send_y_target_position(y_target, x=0.0, z=depth_frame.get_distance(640, 360))

# On exit, clean up:
vision_integrator.close()
"""

# ============================================================================
# EXAMPLE: How to integrate with keyboard_proto.py recording setup
# ============================================================================

INTEGRATION_NOTES = """
To integrate camera vision with keyboard_proto.py:

1. In keyboard_proto.py, the VisionPoseSender is already initialized on port 9998.

2. You can send simulated or real camera detections during the control loop.
   Inside the main control loop in keyboard_proto.py:
   
   # Example: Send a simulated detection based on robot position
   if vision_detection_available:
       detected_pos = compute_detection_from_camera()
       vision_sender.send_position(
           detected_pos[0] / 100.0,  # Convert cm to m
           detected_pos[1] / 100.0,
           detected_pos[2] / 100.0
       )

3. To use actual camera vision from ur5_full_example.py:
   - Run ur5_full_example.py which detects objects
   - Have it send detections via VisionDataIntegrator
   - Simultaneously run keyboard_proto.py for recording
   - Both will write to the Julia live plotter on port 9998

4. Start Julia live plotter:
   julia --threads 1 live_plot_runner.jl
   
   The plotter will show:
   - Red/small points: TCP poses from robot arms (port 9999)
   - Blue/large square points: Vision detections (port 9998)
"""

if __name__ == "__main__":
    print("Vision Data Integration Examples")
    print("=" * 60)
    print(EXAMPLE_CODE)
    print("\n" + "=" * 60)
    print(INTEGRATION_NOTES)
