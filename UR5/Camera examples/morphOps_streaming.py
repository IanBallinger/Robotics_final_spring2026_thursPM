# Enhanced morphOps.py with UDP streaming to Julia live plotter
# Original from UR5/Camera examples/morphOps.py
# Enhanced to stream detected object centroids to Julia visualizer on port 9998

import numpy as np
import cv2
import socket
import json
import time
from tkinter import *
import math

# UDP streaming setup
UDP_HOST = "127.0.0.1"
UDP_PORT = 9998
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_detection(x_cm, y_cm, color_name, bbox_info=None):
    """Send detected object centroid to Julia live plotter via UDP.
    
    Args:
        x_cm: X position in centimeters
        y_cm: Y position in centimeters
        color_name: Name of detected color for identification
        bbox_info: Optional dict with bounding box (x, y, w, h) in pixels
    """
    try:
        # Convert cm to meters for Julia plotter
        x_m = x_cm / 100.0
        y_m = y_cm / 100.0
        z_m = 0.0  # No depth from 2D camera
        
        packet = {
            "position": [x_m, y_m, z_m],
            "color": color_name,
            "x_cm": float(x_cm),
            "y_cm": float(y_cm)
        }
        
        if bbox_info:
            packet["bbox"] = bbox_info
        
        msg = json.dumps(packet)
        udp_socket.sendto(msg.encode("utf-8"), (UDP_HOST, UDP_PORT))
        return True
    except Exception as e:
        print(f"Failed to send detection: {e}")
        return False

tk = Tk()
l_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, lower', orient = HORIZONTAL)
l_h.pack()
u_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, upper', orient = HORIZONTAL)
u_h.pack()
u_h.set(255)
l_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, lower', orient = HORIZONTAL)
l_s.pack()
u_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, upper', orient = HORIZONTAL)
u_s.pack()
u_s.set(255)
l_v = Scale(tk, from_ = 0, to = 255, label = 'Value, lower', orient = HORIZONTAL)
l_v.pack()
u_v = Scale(tk, from_ = 0, to = 255, label = 'Value, upper', orient = HORIZONTAL)
u_v.pack()
u_v.set(255)


def main():
    """Main detection loop with multi-color support and UDP streaming."""
    # Open up the webcam
    cap = cv2.VideoCapture(2)
    
    # HSV color thresholds for all objects
    colors = {
        'purple': {
            'lower': np.array([156, 83, 0]),
            'upper': np.array([180, 176, 143])
        },
        'yellow': {
            'lower': np.array([13, 255, 120]),
            'upper': np.array([98, 255, 208])
        },
        'green': {
            'lower': np.array([53, 90, 128]),
            'upper': np.array([87, 180, 221])
        },
        'tan': {
            'lower': np.array([40, 71, 139]),
            'upper': np.array([56, 195, 255])
        },
        'red': {
            'lower': np.array([1, 180, 131]),
            'upper': np.array([3, 255, 236])
        }
    }
    
    # Coordinate conversion factor (pixels to cm)
    CM_PIXEL = 54.0 / 275
    
    # Camera center reference point (in pixels)
    CENTER_X = 337.5
    CENTER_Y = 337.5
    
    # Morphological kernel
    kernel = np.ones((5, 5), np.uint8)
    num_iterations = 3
    
    print("morphOps with UDP streaming started.")
    print(f"Streaming to {UDP_HOST}:{UDP_PORT}")
    print("Close window or press Ctrl+C to stop.")
    
    packet_count = 0
    
    try:
        while True:
            tk.update()
            
            # Read frame
            ret, cv_image = cap.read()
            if not ret:
                break
            
            # Convert to HSV
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Display original
            cv2.imshow("Original_Image", cv_image)
            cv2.waitKey(3)
            
            # Detect each color and send to plotter
            all_detections = []
            
            for color_name, color_range in colors.items():
                # Threshold for this color
                mask = cv2.inRange(
                    hsv_image,
                    color_range['lower'],
                    color_range['upper']
                )
                
                # Morphological operations
                opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=num_iterations)
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=num_iterations)
                
                # Find contours
                contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Process each contour
                for cnt in contours:
                    if cv2.contourArea(cnt) > 500:  # Filter small noise
                        x, y, w, h = cv2.boundingRect(cnt)
                        x_c = x + int(w / 2)
                        y_c = y + int(h / 2)
                        
                        # Convert to cm
                        x_cm = (x_c - CENTER_X) * CM_PIXEL
                        y_cm = (CENTER_Y - y_c) * CM_PIXEL
                        
                        # Draw on image
                        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(cv_image, (x_c, y_c), 4, (0, 255, 0), 2)
                        
                        text = f"{color_name}: x={x_cm:.2f}cm, y={y_cm:.2f}cm"
                        cv2.putText(
                            cv_image, text, (x_c - 50, y_c - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                        )
                        
                        # Store detection
                        detection = {
                            'color': color_name,
                            'x_cm': x_cm,
                            'y_cm': y_cm,
                            'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                            'area': float(cv2.contourArea(cnt))
                        }
                        all_detections.append(detection)
                        
                        # Send to Julia plotter
                        send_detection(
                            x_cm, y_cm, color_name,
                            bbox_info={'x': x, 'y': y, 'w': w, 'h': h}
                        )
            
            # Draw reference crosshair
            cv2.rectangle(cv_image, (335, 335), (340, 340), (255, 255, 0), 2)
            cv2.rectangle(cv_image, (113, 205), (561, 205), (255, 255, 0), 2)
            
            cv2.imshow('Bounding Box - All Colors', cv_image)
            cv2.waitKey(3)
            
            packet_count += 1
            if packet_count % 30 == 0 and all_detections:
                print(f"[{packet_count}] Detected {len(all_detections)} objects: "
                      + ", ".join([f"{d['color']} ({d['x_cm']:.1f}, {d['y_cm']:.1f})" 
                                   for d in all_detections]))
    
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        udp_socket.close()
        print("morphOps with UDP streaming stopped.")


if __name__ == '__main__':
    main()
