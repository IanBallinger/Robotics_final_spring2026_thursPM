from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface
from robotiq_gripper_control import RobotiqGripper
from pynput.keyboard import Key, Listener
import time
import argparse
import sys
import json
import math
import socket
import threading
import cv2
import numpy as np
import os
import re
import shutil

left_arm_ip = "192.168.1.101"
right_arm_ip = "192.168.1.102"

# Gripper control state (thread-safe)
gripper_state_L = True  # Left gripper, open initially
gripper_state_R = True  # Right gripper, open initially
gripper_state_lock = threading.Lock()
recording_active = True
pending_waypoint_marks = 0
waypoint_index = 0
waypoint_lock = threading.Lock()
vision_targets_lock = threading.Lock()
latest_target_positions = {}


def _mat_vec_mul_row(v, m):
    return (
        v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
        v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
        v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
    )


def base_to_global_task_xyz(base_xyz, arm_side):
    """Mirror Supervisor/global task-frame transform for both arms."""
    dy_t = 0.225 / 2 + 0.540 / 2
    dz_t = -0.753

    if arm_side == "left":
        dx_t = 0.090 / 2 + 0.010 + 0.110
        r_task_to_base = [
            [0.707, 0.0, -0.707],
            [0.0, -1.0, 0.0],
            [-0.707, 0.0, -0.707],
        ]
    else:
        dx_t = -(0.090 / 2 + 0.010 + 0.110)
        r_task_to_base = [
            [0.707, 0.0, 0.707],
            [0.0, -1.0, 0.0],
            [0.707, 0.0, -0.707],
        ]

    trans_base_to_task = _mat_vec_mul_row((dx_t, dy_t, dz_t), r_task_to_base)
    p_rel = (
        float(base_xyz[0]) - trans_base_to_task[0],
        float(base_xyz[1]) - trans_base_to_task[1],
        float(base_xyz[2]) - trans_base_to_task[2],
    )
    r_base_to_task = [
        [r_task_to_base[0][0], r_task_to_base[1][0], r_task_to_base[2][0]],
        [r_task_to_base[0][1], r_task_to_base[1][1], r_task_to_base[2][1]],
        [r_task_to_base[0][2], r_task_to_base[1][2], r_task_to_base[2][2]],
    ]
    return _mat_vec_mul_row(p_rel, r_base_to_task)

def on_press(key):
    """Handle keyboard press events for gripper control."""
    global gripper_state_L, gripper_state_R, recording_active, pending_waypoint_marks
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
        # "w" marks a waypoint snapshot to be named in Julia UI.
        elif key.char == 'w':
            with waypoint_lock:
                pending_waypoint_marks += 1
            print("Waypoint mark queued")
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
    'blue':   {'lower': np.array([ 95, 105, 158]), 'upper': np.array([103, 255, 255])},
    'red':    {'lower': np.array([  1, 180, 131]), 'upper': np.array([  3, 255, 236])},
}
CM_PIXEL  = 54.0 / 275  # pixels -> cm
CENTER_X  = 337.5       # camera origin in pixels
CENTER_Y  = 337.5
VISION_Z  = 0.1         # fixed z: top-down camera (metres above table plane)


def _send_stream_packet(stream_socket, stream_send_lock, packet):
    """Send one JSON packet over TCP as a newline-delimited message."""
    if stream_socket is None:
        return
    payload = (json.dumps(packet, separators=(",", ":")) + "\n").encode("utf-8")
    with stream_send_lock:
        stream_socket.sendall(payload)


def run_camera_detection(stream_socket, stream_target, stop_event, stream_send_lock, camera_index=2, color_to_target_label=None):
    """Background thread: detect coloured objects and stream vision packets."""
    cap = cv2.VideoCapture(camera_index)
    kernel = np.ones((5, 5), np.uint8)
    iters = 3
    packet_count = 0
    print(f"Camera thread started (index {camera_index}). Streaming to {stream_target} via TCP.")
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            frame_detections = []

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

                    resolved_label = color_to_target_label.get(color_name, color_name) if color_to_target_label else color_name
                    frame_detections.append(
                        {
                            "label": str(resolved_label),
                            "color": color_name,
                            "position": [round(x_cm / 100.0, 4), round(y_cm / 100.0, 4), VISION_Z],
                            "x_cm": round(float(x_cm), 3),
                            "y_cm": round(float(y_cm), 3),
                            "bbox_xywh": [int(x), int(y), int(w), int(h)],
                        }
                    )

            now_ts = time.time()
            with vision_targets_lock:
                latest_target_positions.clear()
                for det in frame_detections:
                    latest_target_positions[str(det["label"])] = {
                        "position": [float(det["position"][0]), float(det["position"][1]), float(det["position"][2])],
                        "timestamp": now_ts,
                        "color": det["color"],
                    }

            if frame_detections:
                packet = {
                    "packet_type": "vision_frame",
                    "timestamp": now_ts,
                    "detections": frame_detections,
                }
                try:
                    _send_stream_packet(stream_socket, stream_send_lock, packet)
                except Exception as e:
                    print(f"Vision TCP send error: {e}")

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
        help="DEPRECATED: output file argument is ignored; trace names are derived from task name",
        type=str,
        default="robot_data.csv",
        metavar="<data output file>")
    parser.add_argument(
        "-f",
        "--frequency",
        dest="frequency",
        help="the frequency at which the data is recorded (default is 30Hz)",
        type=float,
        default=30.0,
        metavar="<frequency>")
    parser.add_argument(
        "--stream-udp-host",
        dest="stream_udp_host",
        help="optional host for TCP live stream (legacy flag name; example: 127.0.0.1)",
        type=str,
        default="",
        metavar="<udp host>")
    parser.add_argument(
        "--stream-udp-port",
        dest="stream_udp_port",
        help="optional port for TCP live stream (legacy flag name; default: 9999)",
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
    parser.add_argument(
        "--task-graph-file",
        dest="task_graph_file",
        type=str,
        default="",
        metavar="<task graph json>",
        help="optional path to task graph JSON for task-aware waypoint logging")
    parser.add_argument(
        "--task-id",
        dest="task_id",
        type=str,
        default="",
        metavar="<task id>",
        help="task id from graph used to resolve dependent item label")
    parser.add_argument(
        "--named-waypoints-csv",
        dest="named_waypoints_csv",
        type=str,
        default="named_waypoints.csv",
        metavar="<named waypoints csv>",
        help="output CSV path Julia naming UI should write to")
    parser.add_argument(
        "--write-task-graph-labels",
        dest="write_task_graph_labels",
        action="store_true",
        help="write trace/waypoint CSV labels back into selected task params in graph")

    return parser.parse_args(args)


def _safe_task_name_for_filename(task_name, fallback_task_id=""):
    raw = str(task_name or "").strip()
    if not raw:
        raw = str(fallback_task_id or "").strip()
    if not raw:
        raw = "unnamed_task"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    safe = safe.strip("._-")
    return safe or "unnamed_task"


def load_task_context(task_graph_file, task_id):
    context = {
        "task_id": task_id,
        "task_name": str(task_id or ""),
        "dependent_item_label": "",
        "target_coloring": {},
    }
    if not task_graph_file or not task_id:
        return context

    try:
        with open(task_graph_file, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except Exception as exc:
        print(f"Warning: could not read task graph '{task_graph_file}': {exc}")
        return context

    context["target_coloring"] = dict(graph.get("target_coloring", {}))

    candidate_tasks = []
    primary = graph.get("primary_task")
    if isinstance(primary, dict):
        candidate_tasks.append(primary)
    queue = graph.get("autonomy_queue", [])
    if isinstance(queue, list):
        for item in queue:
            if isinstance(item, dict):
                candidate_tasks.append(item)

    selected = None
    for item in candidate_tasks:
        if str(item.get("task_id", "")) == task_id:
            selected = item
            break
    if selected is None:
        print(f"Warning: task_id '{task_id}' not found in graph '{task_graph_file}'")
        return context

    params = selected.get("params", {}) if isinstance(selected.get("params", {}), dict) else {}
    context["task_name"] = str(selected.get("name", ""))
    # Task dependency heuristic. Acquire tasks typically depend on target_label.
    if context["task_name"].startswith("acquire_"):
        context["dependent_item_label"] = str(params.get("target_label", ""))
    else:
        context["dependent_item_label"] = str(
            params.get("dependent_item_label")
            or params.get("target_label")
            or params.get("object_label")
            or params.get("source_label")
            or ""
        )
    return context


def write_graph_task_labels(task_graph_file, task_id, left_csv, right_csv, named_waypoints_csv, dependent_item_label):
    if not task_graph_file or not task_id:
        return
    try:
        with open(task_graph_file, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except Exception as exc:
        print(f"Warning: could not update graph labels, read failed: {exc}")
        return

    updated = False
    tasks = []
    if isinstance(graph.get("primary_task"), dict):
        tasks.append(graph["primary_task"])
    if isinstance(graph.get("autonomy_queue"), list):
        tasks.extend([item for item in graph["autonomy_queue"] if isinstance(item, dict)])

    for task in tasks:
        if str(task.get("task_id", "")) != task_id:
            continue
        params = task.get("params", {})
        if not isinstance(params, dict):
            params = {}
        params["pose_trace_csv_left"] = left_csv
        params["pose_trace_csv_right"] = right_csv
        params["named_waypoints_csv"] = named_waypoints_csv
        if dependent_item_label:
            params["dependent_item_label"] = dependent_item_label
        task["params"] = params
        updated = True
        break

    if not updated:
        print(f"Warning: graph label write skipped, task_id '{task_id}' not found")
        return

    try:
        with open(task_graph_file, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
            f.write("\n")
        print(f"Updated task graph labels for task_id '{task_id}'")
    except Exception as exc:
        print(f"Warning: could not write updated task graph: {exc}")


def _offset_and_distance(arm_pose, target_position):
    if not arm_pose or len(arm_pose) < 3 or not target_position or len(target_position) < 3:
        return None, None
    dx = float(target_position[0]) - float(arm_pose[0])
    dy = float(target_position[1]) - float(arm_pose[1])
    dz = float(target_position[2]) - float(arm_pose[2])
    return [dx, dy, dz], math.sqrt(dx * dx + dy * dy + dz * dz)

def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    global recording_active, gripper_state_L, gripper_state_R, pending_waypoint_marks, waypoint_index
    
    args = parse_args(args)
    dt = 1 / args.frequency
    task_context = load_task_context(args.task_graph_file, args.task_id)
    if args.output and args.output != "robot_data.csv":
        print("Warning: --output is deprecated and ignored. Using task-name-derived trace filenames.")
    target_coloring = task_context.get("target_coloring", {})
    color_to_target_label = {str(v): str(k) for k, v in target_coloring.items()}
    dependent_item_label = str(task_context.get("dependent_item_label", ""))
    if args.task_id:
        print(
            f"Task context: id={args.task_id}, name={task_context.get('task_name', '')}, "
            f"dependent_item_label={dependent_item_label or '<none>'}"
        )

    stream_socket = None
    stream_target = None
    if args.stream_udp_host:
        stream_target = (args.stream_udp_host, args.stream_udp_port)
        try:
            stream_socket = socket.create_connection(stream_target, timeout=5.0)
            stream_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            stream_socket.settimeout(None)
            print(f"Connected TCP stream to {stream_target[0]}:{stream_target[1]}")
        except Exception as exc:
            print(f"Warning: could not connect TCP stream to {stream_target[0]}:{stream_target[1]}: {exc}")
            stream_socket = None
            stream_target = None
    stream_send_lock = threading.Lock()

    # --- Camera thread (runs in both robot and no-robot modes when UDP target is set) ---
    camera_stop = threading.Event()
    camera_thread = None
    # Default: camera ON unless explicitly disabled via --no-camera.
    use_camera = not args.no_camera
    if use_camera and stream_socket and stream_target:
        camera_thread = threading.Thread(
            target=run_camera_detection,
            args=(stream_socket, stream_target, camera_stop, stream_send_lock, args.camera_index, color_to_target_label),
            daemon=True,
        )
        camera_thread.start()
    elif use_camera and not (stream_socket and stream_target):
        print("Warning: camera enabled but TCP stream target is missing; vision packets will not be sent.")

    # --- No-robot mode: camera-only, no RTDE/gripper/CSV ---
    if args.no_robot:
        print("Running in --no-robot mode. Camera vision only.")
        if not (stream_socket and stream_target):
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
            if stream_socket:
                stream_socket.close()
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

    safe_task_name = _safe_task_name_for_filename(task_context.get("task_name", ""), args.task_id)
    trace_base_name = f"trace_{safe_task_name}"
    left_output = f"{trace_base_name}_left.csv"
    right_output = f"{trace_base_name}_right.csv"

    traces_dir = os.path.join(".", "traces")
    os.makedirs(traces_dir, exist_ok=True)
    left_output_copy = os.path.join(traces_dir, f"{trace_base_name}_left.csv")
    right_output_copy = os.path.join(traces_dir, f"{trace_base_name}_right.csv")

    if args.write_task_graph_labels:
        write_graph_task_labels(
            args.task_graph_file,
            args.task_id,
            left_output,
            right_output,
            args.named_waypoints_csv,
            dependent_item_label,
        )

    rtde_r_left.startFileRecording(left_output, variables)
    rtde_r_right.startFileRecording(right_output, variables)
    if stream_target:
        print(
            f"Data recording started (+ TCP stream to {stream_target[0]}:{stream_target[1]}), "
            "press [Ctrl-C] or Delete to end recording."
        )
    else:
        print("Data recording started, press [Ctrl-C] or Delete to end recording.")
    print(f"Saving LEFT arm to: {left_output}")
    print(f"Saving RIGHT arm to: {right_output}")
    print(f"Will copy LEFT trace to: {left_output_copy}")
    print(f"Will copy RIGHT trace to: {right_output_copy}")
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
            left_q = rtde_r_left.getActualQ()
            right_q = rtde_r_right.getActualQ()
            left_task_xyz = base_to_global_task_xyz(left_pose[:3], "left")
            right_task_xyz = base_to_global_task_xyz(right_pose[:3], "right")

            with gripper_state_lock:
                curr_gripper_state_L = bool(gripper_state_L)
                curr_gripper_state_R = bool(gripper_state_R)

            if stream_socket and stream_target:
                packet = {
                    # Backward compatibility: keep original key as LEFT arm pose.
                    "timestamp": left_timestamp,
                    "actual_TCP_pose": left_pose,
                    "actual_q": left_q,
                    "left_timestamp": left_timestamp,
                    "right_timestamp": right_timestamp,
                    "left_actual_TCP_pose": left_pose,
                    "right_actual_TCP_pose": right_pose,
                    "left_actual_q": left_q,
                    "right_actual_q": right_q,
                    # Arm/base coordinates (explicit aliases for downstream consumers)
                    "left_base_xyz": [float(left_pose[0]), float(left_pose[1]), float(left_pose[2])],
                    "right_base_xyz": [float(right_pose[0]), float(right_pose[1]), float(right_pose[2])],
                    # Per-arm task/global coordinates used for moving-frame playback.
                    "left_task_xyz": [float(left_task_xyz[0]), float(left_task_xyz[1]), float(left_task_xyz[2])],
                    "right_task_xyz": [float(right_task_xyz[0]), float(right_task_xyz[1]), float(right_task_xyz[2])],
                    "left_global_xyz": [float(left_task_xyz[0]), float(left_task_xyz[1]), float(left_task_xyz[2])],
                    "right_global_xyz": [float(right_task_xyz[0]), float(right_task_xyz[1]), float(right_task_xyz[2])],
                    "left_gripper_open": curr_gripper_state_L,
                    "right_gripper_open": curr_gripper_state_R,
                }
                _send_stream_packet(stream_socket, stream_send_lock, packet)

            marks_to_send = 0
            with waypoint_lock:
                if pending_waypoint_marks > 0:
                    marks_to_send = pending_waypoint_marks
                    pending_waypoint_marks = 0

            for _ in range(marks_to_send):
                waypoint_index += 1
                dependent_snapshot = None
                tracked_items_snapshot = []
                if dependent_item_label:
                    with vision_targets_lock:
                        dependent_snapshot = latest_target_positions.get(dependent_item_label)
                with vision_targets_lock:
                    for label, data in latest_target_positions.items():
                        tracked_items_snapshot.append(
                            {
                                "label": str(label),
                                "position": [
                                    float(data.get("position", [0.0, 0.0, 0.0])[0]),
                                    float(data.get("position", [0.0, 0.0, 0.0])[1]),
                                    float(data.get("position", [0.0, 0.0, 0.0])[2]),
                                ],
                                "timestamp": float(data.get("timestamp", 0.0)),
                                "color": str(data.get("color", "")),
                            }
                        )

                dependent_position = dependent_snapshot.get("position") if dependent_snapshot else None
                left_offset, left_distance = _offset_and_distance(left_pose, dependent_position)
                right_offset, right_distance = _offset_and_distance(right_pose, dependent_position)

                waypoint_packet = {
                    "packet_type": "waypoint_mark",
                    "waypoint_index": waypoint_index,
                    "waypoint_mark_time": time.time(),
                    "task_id": args.task_id,
                    "task_name": task_context.get("task_name", ""),
                    "dependent_item_label": dependent_item_label,
                    "dependent_item_position": dependent_position,
                    "dependent_item_seen_time": dependent_snapshot.get("timestamp") if dependent_snapshot else None,
                    "left_actual_TCP_pose": left_pose,
                    "right_actual_TCP_pose": right_pose,
                    "left_actual_q": left_q,
                    "right_actual_q": right_q,
                    "left_task_xyz": [float(left_task_xyz[0]), float(left_task_xyz[1]), float(left_task_xyz[2])],
                    "right_task_xyz": [float(right_task_xyz[0]), float(right_task_xyz[1]), float(right_task_xyz[2])],
                    "left_global_xyz": [float(left_task_xyz[0]), float(left_task_xyz[1]), float(left_task_xyz[2])],
                    "right_global_xyz": [float(right_task_xyz[0]), float(right_task_xyz[1]), float(right_task_xyz[2])],
                    "left_gripper_open": curr_gripper_state_L,
                    "right_gripper_open": curr_gripper_state_R,
                    "left_distance_to_dependent_m": left_distance,
                    "right_distance_to_dependent_m": right_distance,
                    "left_offset_to_dependent_xyz": left_offset,
                    "right_offset_to_dependent_xyz": right_offset,
                    "tracked_items": tracked_items_snapshot,
                    "named_waypoints_csv": args.named_waypoints_csv,
                }
                if stream_socket and stream_target:
                    _send_stream_packet(stream_socket, stream_send_lock, waypoint_packet)
                print(
                    f"Waypoint #{waypoint_index} marked "
                    f"(dependent={dependent_item_label or 'none'}, "
                    f"left_dist={left_distance}, right_dist={right_distance})"
                )

            # Update gripper states if they changed
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
        if stream_socket:
            stream_socket.close()

        try:
            shutil.copy2(left_output, left_output_copy)
            print(f"Copied LEFT trace -> {left_output_copy}")
        except Exception as exc:
            print(f"Warning: could not copy LEFT trace to {left_output_copy}: {exc}")

        try:
            shutil.copy2(right_output, right_output_copy)
            print(f"Copied RIGHT trace -> {right_output_copy}")
        except Exception as exc:
            print(f"Warning: could not copy RIGHT trace to {right_output_copy}: {exc}")

        print("\nData recording stopped.")


if __name__ == "__main__":
    main(sys.argv[1:])