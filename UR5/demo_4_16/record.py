from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface
from robotiq_gripper_control import RobotiqGripper
from pynput.keyboard import Key, Listener
import time
import argparse
import sys
import json
import socket
import threading
import cv2
import numpy as np
import os

left_arm_ip = "192.168.1.101"
right_arm_ip = "192.168.1.102"

# Gripper control state (thread-safe)
gripper_state_L = True  # Left gripper, open initially
gripper_state_R = True  # Right gripper, open initially
gripper_state_lock = threading.Lock()
recording_active = True

def on_press(key):
    """Handle keyboard press events for gripper control."""
    global gripper_state_L, gripper_state_R, recording_active
    try:
        # "l" key toggles left gripper
        if key.char == 'l':
            with gripper_state_lock:
                gripper_state_L = not gripper_state_L
            print(f"Left gripper {'open' if gripper_state_L else 'close'}")
        # "r" key toggles right gripper
        elif key.char == 'r':
            with gripper_state_lock:
                gripper_state_R = not gripper_state_R
            print(f"Right gripper {'open' if gripper_state_R else 'close'}")
    except AttributeError:
        # Special keys like Delete don't have .char
        if key == Key.delete:
            recording_active = False

def on_release(key):
    """Handle keyboard release events."""
    pass

#this is a slightly modified example from ur_rtde docs. 
# explicitly stating this variables[] list is the only modification. 
variables = ["timestamp",
                  "target_q",
                  "target_qd",
                  "target_qdd",
                  "target_current",
                  "target_moment",
                  "actual_q",
                  "actual_qd",
                  "actual_current",
                  "joint_control_output",
                  "actual_TCP_pose",
                  "actual_TCP_speed",
                  "actual_TCP_force",
                  "target_TCP_pose",
                  "target_TCP_speed",
                  "actual_digital_input_bits",
                  "joint_temperatures",
                  "actual_execution_time",
                  "robot_mode",
                  "joint_mode",
                  "safety_mode",
                  "actual_tool_accelerometer",
                  "speed_scaling",
                  "target_speed_fraction",
                  "actual_momentum",
                  "actual_main_voltage",
                  "actual_robot_voltage",
                  "actual_robot_current",
                  "actual_joint_voltage",
                  "actual_digital_output_bits",
                  "runtime_state",
                  "standard_analog_input0",
                  "standard_analog_input1",
                  "standard_analog_output0",
                  "standard_analog_output1",
                  "robot_status_bits",
                  "safety_status_bits"]

# HSV color thresholds (tuned from morphOps_streaming.py)
CAMERA_COLORS = {
    'purple': {'lower': np.array([156,  83,   0]), 'upper': np.array([180, 176, 143])},
    'yellow': {'lower': np.array([ 13, 255, 120]), 'upper': np.array([ 98, 255, 208])},
    'green':  {'lower': np.array([ 53,  90, 128]), 'upper': np.array([ 87, 180, 221])},
    'tan':    {'lower': np.array([ 40,  71, 139]), 'upper': np.array([ 56, 195, 255])},
    'red':    {'lower': np.array([  1, 180, 131]), 'upper': np.array([  3, 255, 236])},
}
CM_PIXEL  = 54.0 / 275  # pixels -> cm
CENTER_X  = 337.5       # camera origin in pixels
CENTER_Y  = 337.5
VISION_Z  = 0.1         # fixed z: top-down camera (metres above table plane)


def run_camera_detection(udp_socket, udp_target, stop_event, udp_send_lock, camera_index=2):
    """Background thread: detect coloured objects and stream vision packets."""
    cap = cv2.VideoCapture(camera_index)
    kernel = np.ones((5, 5), np.uint8)
    iters = 3
    packet_count = 0
    print(f"Camera thread started (index {camera_index}). Streaming to {udp_target}.")
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for color_name, rng in CAMERA_COLORS.items():
                mask = cv2.inRange(hsv, rng['lower'], rng['upper'])
                opened  = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel, iterations=iters)
                closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iters)
                contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    if cv2.contourArea(cnt) < 500:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    xc = x + w // 2
                    yc = y + h // 2
                    x_cm = (xc - CENTER_X) * CM_PIXEL
                    y_cm = (CENTER_Y - yc) * CM_PIXEL

                    packet = {
                        "position": [round(x_cm / 100.0, 4),
                                     round(y_cm / 100.0, 4),
                                     VISION_Z],
                        "color": color_name,
                        "x_cm": round(float(x_cm), 3),
                        "y_cm": round(float(y_cm), 3),
                    }
                    try:
                        with udp_send_lock:
                            udp_socket.sendto(json.dumps(packet).encode("utf-8"), udp_target)
                    except Exception as e:
                        print(f"Vision UDP send error: {e}")

            cv2.imshow("Camera (record.py)", frame)
            cv2.waitKey(3)
            packet_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera thread stopped.")


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Record data example")
    parser.add_argument(
        "-ip",
        "--robot_ip",
        dest="ip",
        help="IP address of the LEFT UR robot",
        type=str,
        default=left_arm_ip,
        metavar="<LEFT robot IP address>")
    parser.add_argument(
        "--right-robot-ip",
        dest="right_ip",
        help="IP address of the RIGHT UR robot",
        type=str,
        default=right_arm_ip,
        metavar="<RIGHT robot IP address>")
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="data output (.csv) file to write to (default is \"robot_data.csv\"",
        type=str,
        default="robot_data.csv",
        metavar="<data output file>")
    parser.add_argument(
        "-f",
        "--frequency",
        dest="frequency",
        help="the frequency at which the data is recorded (default is 500Hz)",
        type=float,
        default=500.0,
        metavar="<frequency>")
    parser.add_argument(
        "--stream-udp-host",
        dest="stream_udp_host",
        help="optional host for UDP live stream (example: 127.0.0.1)",
        type=str,
        default="",
        metavar="<udp host>")
    parser.add_argument(
        "--stream-udp-port",
        dest="stream_udp_port",
        help="optional port for UDP live stream (default: 9999)",
        type=int,
        default=9999,
        metavar="<udp port>")
    parser.add_argument(
        "--no-robot",
        dest="no_robot",
        action="store_true",
        help="run without a robot connection (camera/vision only mode)")
    parser.add_argument(
        "--camera",
        dest="camera",
        action="store_true",
        help="enable camera detection thread alongside robot recording (default behavior)")
    parser.add_argument(
        "--no-camera",
        dest="no_camera",
        action="store_true",
        help="disable camera detection thread")
    parser.add_argument(
        "--camera-index",
        dest="camera_index",
        type=int,
        default=2,
        metavar="<camera index>",
        help="OpenCV camera device index (default: 0)")

    return parser.parse_args(args)

def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    global recording_active, gripper_state_L, gripper_state_R
    
    args = parse_args(args)
    dt = 1 / args.frequency

    udp_socket = None
    udp_target = None
    if args.stream_udp_host:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_target = (args.stream_udp_host, args.stream_udp_port)
    udp_send_lock = threading.Lock()

    # --- Camera thread (runs in both robot and no-robot modes when UDP target is set) ---
    camera_stop = threading.Event()
    camera_thread = None
    # Default: camera ON unless explicitly disabled via --no-camera.
    use_camera = not args.no_camera
    if use_camera and udp_socket and udp_target:
        camera_thread = threading.Thread(
            target=run_camera_detection,
            args=(udp_socket, udp_target, camera_stop, udp_send_lock, args.camera_index),
            daemon=True,
        )
        camera_thread.start()
    elif use_camera and not (udp_socket and udp_target):
        print("Warning: camera enabled but UDP target is missing; vision packets will not be sent.")

    # --- No-robot mode: camera-only, no RTDE/gripper/CSV ---
    if args.no_robot:
        print("Running in --no-robot mode. Camera vision only.")
        if not (udp_socket and udp_target):
            print("Warning: no --stream-udp-host specified; camera data will not be sent.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            camera_stop.set()
            if camera_thread:
                camera_thread.join(timeout=2)
            if udp_socket:
                udp_socket.close()
            print("No-robot mode stopped.")
        return

    # --- Full robot mode ---
    rtde_r_left = RTDEReceive(args.ip, args.frequency)
    rtde_r_right = RTDEReceive(args.right_ip, args.frequency)

    # Initialize gripper control interfaces
    try:
        rtde_c_L = RTDEControlInterface(args.ip)
        rtde_c_R = RTDEControlInterface(args.right_ip)
        
        gripper_L = RobotiqGripper(rtde_c_L)
        gripper_R = RobotiqGripper(rtde_c_R)
        
        # Activate grippers
        gripper_L.activate()
        gripper_R.activate()
        gripper_L.set_force(50)
        gripper_R.set_force(50)
        gripper_L.set_speed(100)
        gripper_R.set_speed(100)
        gripper_L.open()
        gripper_R.open()
        
        print("Grippers initialized. Use 'l' and 'r' keys to toggle left/right grippers.")
    except Exception as e:
        print(f"Warning: Could not initialize grippers: {e}")
        gripper_L = None
        gripper_R = None

    # Start keyboard listener in background thread
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    output_base, output_ext = os.path.splitext(args.output)
    if not output_ext:
        output_ext = ".csv"
    left_output = f"{output_base}_left{output_ext}"
    right_output = f"{output_base}_right{output_ext}"

    rtde_r_left.startFileRecording(left_output, variables)
    rtde_r_right.startFileRecording(right_output, variables)
    if udp_target:
        print(
            f"Data recording started (+ UDP stream to {udp_target[0]}:{udp_target[1]}), "
            "press [Ctrl-C] or Delete to end recording."
        )
    else:
        print("Data recording started, press [Ctrl-C] or Delete to end recording.")
    print(f"Saving LEFT arm to: {left_output}")
    print(f"Saving RIGHT arm to: {right_output}")
    i = 0
    prev_gripper_state_L = gripper_state_L
    prev_gripper_state_R = gripper_state_R
    
    try:
        while recording_active:
            start = time.time()

            left_timestamp = rtde_r_left.getTimestamp()
            right_timestamp = rtde_r_right.getTimestamp()
            left_pose = rtde_r_left.getActualTCPPose()
            right_pose = rtde_r_right.getActualTCPPose()

            if udp_socket and udp_target:
                packet = {
                    # Backward compatibility: keep original key as LEFT arm pose.
                    "timestamp": left_timestamp,
                    "actual_TCP_pose": left_pose,
                    "left_timestamp": left_timestamp,
                    "right_timestamp": right_timestamp,
                    "left_actual_TCP_pose": left_pose,
                    "right_actual_TCP_pose": right_pose,
                }
                with udp_send_lock:
                    udp_socket.sendto(json.dumps(packet).encode("utf-8"), udp_target)

            # Update gripper states if they changed
            with gripper_state_lock:
                curr_gripper_state_L = gripper_state_L
                curr_gripper_state_R = gripper_state_R
            
            if gripper_L is not None and curr_gripper_state_L != prev_gripper_state_L:
                if curr_gripper_state_L:
                    gripper_L.open()
                else:
                    gripper_L.close()
                prev_gripper_state_L = curr_gripper_state_L
            
            if gripper_R is not None and curr_gripper_state_R != prev_gripper_state_R:
                if curr_gripper_state_R:
                    gripper_R.open()
                else:
                    gripper_R.close()
                prev_gripper_state_R = curr_gripper_state_R

            if i % 10 == 0:
                sys.stdout.write("\r")
                sys.stdout.write("{:3d} samples.".format(i))
                sys.stdout.flush()
            end = time.time()
            duration = end - start

            if duration < dt:
                time.sleep(dt - duration)
            i += 1

    except KeyboardInterrupt:
        recording_active = False
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        try:
            rtde_r_left.stopFileRecording()
        except Exception:
            pass
        try:
            rtde_r_right.stopFileRecording()
        except Exception:
            pass
        camera_stop.set()
        if camera_thread:
            camera_thread.join(timeout=2)
        if udp_socket:
            udp_socket.close()
        print("\nData recording stopped.")


if __name__ == "__main__":
    main(sys.argv[1:])