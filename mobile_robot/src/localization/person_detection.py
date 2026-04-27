import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def build_cam_to_robot_transform():
    cam_forward  = 0.0
    cam_sideways = 0.0
    cam_up       = 0.0
    cam_pitch_deg = 0.0
    cam_yaw_deg   = 0.0
    cam_roll_deg  = 0.0

    pitch = np.radians(cam_pitch_deg)
    yaw   = np.radians(cam_yaw_deg)
    roll  = np.radians(cam_roll_deg)

    Rx = np.array([[1, 0,            0           ],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])

    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [0,            1, 0           ],
                   [-np.sin(yaw), 0, np.cos(yaw)]])

    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0,             0,            1]])

    R = Ry @ Rx @ Rz
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [cam_forward, cam_sideways, cam_up]
    return T

T_cam_to_robot = build_cam_to_robot_transform() 

model = YOLO("yolo26n.pt")

# pipeline manages data flowing from the camera
pipeline = rs.pipeline()
# config tells it what you want
config = rs.config()
# we are asking for both a color stream and a depth stream, both at 640x480 res at 30 frames per second
# bgr8 means color pixels as blue/green/red bytes
# z16 means depth pixels as 16-bit integers (raw distance units)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# opens the camera and starts streaming
profile = pipeline.start(config)

# align is an object that will warp the depth frame so every depth pixel lines up exactly with its corresponding color pixel
# the two cameras are physically a few cm apart on device
align = rs.align(rs.stream.color)
# depth_scale is a small number that converts the raw integer depth values into meters
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
# intrinsics are the camera's optical properties - focal length and the center point of the image
# needed to convert a 2D pixel location into a real world 3D coord
intrinsics = profile.get_stream(rs.stream.color)\
    .as_video_stream_profile().get_intrinsics()

print("Running — press Q to quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        # convert the raw camera frame into numpy arrays
        color_image = np.asanyarray(aligned.get_color_frame().get_data())
        depth_image = np.asanyarray(aligned.get_depth_frame().get_data())

        # run yolo on image
        # classes[0] means only detect class 0 which is "person" in the COCO dataset YOLO is trained on
        # conf=0.5 means only keep detections where YOLO is at least 50% confident
        # verbose=False supresses per-frame console output
        # display is a copy of the color image that we will draw boxes onto - we copy it so we are not modifying the original
        # tracker - gives each person an ID
        # results = model(color_image, classes=[0], conf=0.5, verbose=False)
        results = model.track(color_image, classes=[0], conf=0.5, tracker="bytetrack.yaml", verbose=False, persist=True)
        display = color_image.copy()


        # results[0].boxes is a list of all the detections in this frame
        # For each one, box.xyxy[0] gives the bounding box as four coordinates — top-left corner (x1, y1) and bottom-right corner (x2, y2) in pixels.
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            person_id = int(box.id[0]) if box.id is not None else -1
            # crops the depth image to exactly the bounding box region
            # Multiplies by depth_scale to convert raw integers to meters
            depth_crop = depth_image[y1:y2, x1:x2].astype(float) * depth_scale
            # filters out bad reading: anything closer than 0.3m is sensor noise at very close range, farther than 6m is beyond reliable depth range
            valid = depth_crop[(depth_crop > 0.3) & (depth_crop < 6.0)]

            # if fewer than 20 valid depth pixels exist - person is wearing a reflective jacket ot too far away - flag and skip math
            if len(valid) < 20:
                label = "person (no depth)"
                color = (0, 165, 255)  # orange = bad depth
            else:
                # median is used instead of mean because the edges of the bounding box often capture background pixels which would skew an average
                # cx, cy is the center pixel of the bounding box
                Z = float(np.median(valid)) # robust distance extimate
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                # convert that center pixel + depth into a real 3D point (X, Y, Z) in meters in the camera's coordinate frame
                # Z is forward distance, X is left/right, Y is up/down
                point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], Z)
                X, Y, _ = point

                # ── STEP 3: transform camera→robot→world ──
                p_cam = np.array([X, Y, Z, 1.0])
                # @ is a matrix multiplication operator
                p_robot = T_cam_to_robot @ p_cam
                # p_world = T_robot_to_world @ p_robot  # uncomment when localization is available
                p_world = p_robot  # for now, robot frame = world frame

                label = f"id={person_id}  Z={Z:.2f}m  X={X:+.2f}  Y={Y:+.2f}"
                color = (0, 255, 0)  # green = good depth

                # ── STEP 4: update map ──
                # map.remove_obstacle(person_id)
                # map.add_circular_obstacle(
                #     center=(p_world[0], p_world[1]),
                #     radius=0.5,
                #     id=person_id
                # )

            # draws the bounding box rectangle and text label just above it onto the display image
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Converts the raw depth array into a colorized visualization — close objects appear red, far objects appear blue — purely for human debugging.
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        # np.hstack stitches the color frame and depth visualization side by side into one wide image
        combined = np.hstack([display, depth_colormap])
        cv2.imshow("YOLO + Depth test", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()