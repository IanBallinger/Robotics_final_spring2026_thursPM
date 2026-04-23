from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time
import argparse
import sys
import json
import socket

left_arm_ip = "192.168.1.101"
right_arm_ip = "192.168.1.102"

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
    args = parse_args(args)
    dt = 1 / args.frequency
    rtde_r = RTDEReceive(args.ip, args.frequency)
    udp_socket = None
    udp_target = None

    if args.stream_udp_host:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_target = (args.stream_udp_host, args.stream_udp_port)

    rtde_r.startFileRecording(args.output, variables)
    if udp_target:
        print(
            f"Data recording started (+ UDP stream to {udp_target[0]}:{udp_target[1]}), "
            "press [Ctrl-C] to end recording."
        )
    else:
        print("Data recording started, press [Ctrl-C] to end recording.")
    i = 0
    try:
        while True:
            start = time.time()
            if udp_socket and udp_target:
                packet = {
                    "timestamp": rtde_r.getTimestamp(),
                    "actual_TCP_pose": rtde_r.getActualTCPPose(),
                }
                # Send line-delimited JSON as UTF-8 bytes.
                udp_socket.sendto(json.dumps(packet).encode("utf-8"), udp_target)

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
        rtde_r.stopFileRecording()
        if udp_socket:
            udp_socket.close()
        print("\nData recording stopped.")


if __name__ == "__main__":
    main(sys.argv[1:])