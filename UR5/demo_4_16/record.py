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
        help="IP address of the UR robot",
        type=str,
        default=left_arm_ip,
        metavar="<IP address of the UR robot>")
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

    return parser.parse_args(args)

def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    global recording_active, gripper_state_L, gripper_state_R
    
    args = parse_args(args)
    dt = 1 / args.frequency
    rtde_r = RTDEReceive(args.ip, args.frequency)
    udp_socket = None
    udp_target = None

    if args.stream_udp_host:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_target = (args.stream_udp_host, args.stream_udp_port)

    # Initialize gripper control interfaces
    try:
        rtde_c_L = RTDEControlInterface(left_arm_ip)
        rtde_c_R = RTDEControlInterface(right_arm_ip)
        
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

    rtde_r.startFileRecording(args.output, variables)
    if udp_target:
        print(
            f"Data recording started (+ UDP stream to {udp_target[0]}:{udp_target[1]}), "
            "press [Ctrl-C] or Delete to end recording."
        )
    else:
        print("Data recording started, press [Ctrl-C] or Delete to end recording.")
    i = 0
    prev_gripper_state_L = gripper_state_L
    prev_gripper_state_R = gripper_state_R
    
    try:
        while recording_active:
            start = time.time()
            if udp_socket and udp_target:
                packet = {
                    "timestamp": rtde_r.getTimestamp(),
                    "actual_TCP_pose": rtde_r.getActualTCPPose(),
                }
                # Send line-delimited JSON as UTF-8 bytes.
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
        listener.stop()
        rtde_r.stopFileRecording()
        if udp_socket:
            udp_socket.close()
        print("\nData recording stopped.")


if __name__ == "__main__":
    main(sys.argv[1:])