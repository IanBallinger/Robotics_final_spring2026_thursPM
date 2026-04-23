"""
Vision Pose Sender - Sends camera-detected object positions to Julia live plotter
Sends JSON packets over UDP to port 9998 with position data.

Usage:
    - Import this module in your recording/control script
    - Create a VisionPoseSender instance
    - Call send_position(x, y, z) whenever you detect an object
    
Example:
    sender = VisionPoseSender(host="127.0.0.1", port=9998)
    sender.send_position(0.5, 0.3, 0.2)  # x, y, z in meters
"""

import socket
import json
import numpy as np
from typing import Tuple

class VisionPoseSender:
    """Sends camera-detected object positions to Julia live plotter via UDP."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9998):
        """
        Initialize the vision pose sender.
        
        Args:
            host: IP address of the Julia plotter (default: localhost)
            port: UDP port for vision poses (default: 9998)
        """
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.packet_count = 0
        print(f"Vision pose sender initialized. Sending to {host}:{port}")
    
    def send_position(self, x: float, y: float, z: float) -> bool:
        """
        Send a detected object position to the Julia plotter.
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters  
            z: Z coordinate in meters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create JSON packet with position
            packet = {
                "position": [float(x), float(y), float(z)],
                "packet_number": self.packet_count
            }
            
            msg = json.dumps(packet)
            self.sock.sendto(msg.encode(), (self.host, self.port))
            self.packet_count += 1
            return True
            
        except Exception as e:
            print(f"Failed to send vision position: {e}")
            return False
    
    def send_position_array(self, position: np.ndarray) -> bool:
        """
        Send a position from a numpy array [x, y, z].
        
        Args:
            position: 3-element numpy array or list
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pos = position.flatten()[:3]
            return self.send_position(float(pos[0]), float(pos[1]), float(pos[2]))
        except Exception as e:
            print(f"Failed to send position array: {e}")
            return False
    
    def close(self):
        """Close the UDP socket."""
        try:
            self.sock.close()
        except:
            pass


# Example usage
if __name__ == "__main__":
    import time
    
    sender = VisionPoseSender()
    
    # Send some test positions
    print("\nSending test positions to Julia plotter...")
    for i in range(10):
        x = 0.5 + 0.05 * np.sin(i * 0.5)
        y = 0.3 + 0.05 * np.cos(i * 0.5)
        z = 0.2 + 0.02 * i
        sender.send_position(x, y, z)
        print(f"  Sent position {i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
        time.sleep(0.2)
    
    sender.close()
    print("Done sending test positions.")
