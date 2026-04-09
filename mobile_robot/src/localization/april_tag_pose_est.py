import cv2
from cv2.typing import MatLike
import numpy as np
import pupil_apriltags as apriltag
import time

from typing import Dict, Tuple


class AprilTagPoseEst:
    def __init__(self):
        self.at_detector = apriltag.Detector(
            families="tag36h11",  # or 'tag25h9', etc.
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.__load_camera_calibration()

        # key: tag id, value: (rotation matrix, translation vector)
        self.pose_estimate: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __load_camera_calibration(self):
        calibration_data = np.load("camera_calibration_live.npz")  # adjust filename
        self.camera_matrix = calibration_data["camera_matrix"]  # shape (3, 3)
        self.dist_coeffs = calibration_data[
            "dist_coeffs"
        ]  # shape (n,) typically (5,) or (8,)
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        self.tag_size = 0.10  # 10 cm

    def __detect_april_tags(self, frame: MatLike):

        # ------------------------------------------------------------------
        # 4. Undistort (optional but recommended)
        #    You can either:
        #    A) Undistort the entire image once, or
        #    B) Let the AprilTag detector handle radial distortion by
        #       passing camera_params directly (estimate_tag_pose=True).
        #
        # In practice, the built-in pose estimation in pupil_apriltags
        # uses the pinhole model with no distortion. So it's best to
        # either undistort the frame yourself or accept small distortion
        # errors if your lens is fairly undistorted or uses a cheap camera.
        # ------------------------------------------------------------------
        # Option A: Undistort the entire frame
        ##        h, w = frame.shape[:2]
        ##        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        ##            camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        ##        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        undistorted = frame  # This turns off the undistortion

        # Convert to grayscale
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        # 5. Detect AprilTags
        # ------------------------------------------------------------------
        # With pupil_apriltags, you can directly get pose estimation by
        # providing 'estimate_tag_pose=True' and camera_params + tag_size.
        # This automatically computes the rotation and translation of the tag.
        # ------------------------------------------------------------------
        results = self.at_detector.detect(
            frame,
            estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=self.tag_size,
        )

        # ------------------------------------------------------------------
        # 6. Process each detection
        # ------------------------------------------------------------------
        for r in results:
            # r.tag_id: the ID of the detected tag
            # r.corners: the (4,2) array of corner coordinates in the image
            # r.center:  the (x,y) coordinates of the tag center
            # r.pose_R, r.pose_t: pose of the tag in the camera frame
            #                    (right-handed coordinate system):
            #    - R is a 3x3 rotation matrix
            #    - t is a 3x1 translation vector
            #
            # The coordinate system by default: +x to the right, +y down,
            # and +z forward from the camera's perspective.
            #

            # ----------------------------------------
            # 6a. Extract Tag ID and corners
            # ----------------------------------------
            tag_id = r.tag_id
            R = r.pose_R  # 3×3 rotation matrix
            t = r.pose_t  # 3×1 translation vector

            self.pose_estimate[tag_id] = (R, t)

    def get_pose_estimate(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        self.__detect_april_tags(frame)
        return self.pose_estimate


if __name__ == "__main__":
    april_tag_pose_est = AprilTagPoseEst()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(0) # For MacOS or Linux, you may need to remove the cv2.CAP_DSHOW flag
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

    if not cap.isOpened():
        print("Error: Could not open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        april_tag_pose_est.get_pose_estimate(frame)
        print(april_tag_pose_est.pose_estimate)
        time.sleep(0.1)
