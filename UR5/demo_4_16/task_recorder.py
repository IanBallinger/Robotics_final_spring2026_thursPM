from pynput.keyboard import Key, Listener
import time
import numpy as np
from scipy.spatial.transform import Rotation
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq_gripper_control import RobotiqGripper
# from arm import UR5Arm

left_arm_ip = "192.168.1.101"
right_arm_ip = "192.168.1.102"

# LEFTARM = UR5Arm(left_arm_ip, verbose=False)
# RIGHTARM = UR5Arm(right_arm_ip, verbose=False)

gripperstateL = True # open initially
gripperstateR = True # open initially
compliant_mode = False
moveL = True
moveR = True

directions = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "q": False,
    "e": False,
    "g": False,
    "f": False,
    "c": False,
    "z": False,
    "x": False,
    "halt": False
}

def pressed(key):
    global gripperstateL
    global gripperstateR
    global compliant_mode
    global moveL
    global moveR
    if key == Key.delete:
        # Stop listener
        directions["halt"] = True
        return False
    
    if key.char in directions.keys() and not (directions[key.char]):
        directions[key.char] = True
        print(f"{key.char} pressed")
        if key.char == "g":
            if moveL:
                gripperstateL = not gripperstateL
            if moveR:
                gripperstateR = not gripperstateR
        if key.char == "z":
            moveL = not moveL
        if key.char == "x":
            moveR = not moveR
        if key.char == "c":
            compliant_mode = not compliant_mode
        if key.char == "f":
            print("joint angles (l, r)")
            print(rtde_r_L.getActualQ())
            print(rtde_r_R.getActualQ())
            print("end effector pos (l, r)")
            print(rtde_r_L.getActualTCPPose())
            print(rtde_r_R.getActualTCPPose())
            print("joint torques (l, r)")
            print(rtde_r_L.getJointTorques())
            print(rtde_r_R.getJointTorques())
            print("fwd. kin. (l, r)")
            print(rtde_c_L.getForwardKinematics())
            print(rtde_c_R.getForwardKinematics())
            print("inv. kin. (l, r)")
            print(rtde_c_L.getInverseKinematics())
            print(rtde_c_R.getInverseKinematics())


def released(key):
    if key.char in directions.keys():
        directions[key.char] = False
        print(f"{key.char} released")


with Listener(on_press = pressed, on_release = released) as listener:
    #### SETUP ####
    # Establish connections to both robot arms. "Left" and "right" are defined from
    # the robot's perspective, not the viewer's perspective.
    rtde_c_L = RTDEControlInterface(left_arm_ip)
    rtde_c_R = RTDEControlInterface(right_arm_ip)

    rtde_r_L = RTDEReceiveInterface(left_arm_ip)
    rtde_r_R = RTDEReceiveInterface(right_arm_ip)

    connection_tries = 0
    if not rtde_c_L.isConnected():
        while connection_tries < 3:
            rtde_c_L.reconnect()
            time.sleep(0.1)
            if rtde_c_L.isConnected():
                break
            connection_tries += 1

    if rtde_c_L.isConnected():
        print("Left robot connection successful!")
    else:
        print("Left robot connection not working")
        rtde_c_L.stopScript()

    connection_tries = 0
    if not rtde_c_R.isConnected():
        while connection_tries < 3:
            rtde_c_R.reconnect()
            time.sleep(0.1)
            if rtde_c_R.isConnected():
                break
            connection_tries += 1

    if rtde_c_R.isConnected():
        print("Right robot connection successful!")
    else:
        print("Right robot connection not working")
        rtde_c_R.stopScript()

    #### COORDINATE TRANSFORMATIONS ####
    tcp_offset = [0, 0, 0.174, 0, 0, 0] # offset from robot flange to gripper tip in output flange coordinate frame
    rtde_c_L.setTcp(tcp_offset)
    rtde_c_R.setTcp(tcp_offset)

    # Choose the top center of the table as the origin of the task coordinate frame.
    # X-axis points towards the right robot arm, y-axis points toward the microwave,
    # and z-axis points up. You may define your own task frame that works for your team.

    # Define coordinate transformations between this task frame and the robot base frames
    # as well as the camera coordinate frames:

    # LEFT ROBOT ARM
    # Translation in task coordinate frame from left robot base origin to center of table (hand-measured):
    dx_t = 0.090/2 + 0.010 + 0.110 # 1/2 vertical beam width + plate thickness + dist. to robot origin
    dy_t = 0.225/2 + 0.540/2 # half of mounting plate width + half of table depth
    dz_t = -0.753 # height from robot origin to top of table

    # Rotation from task coordinate frame to robot base frame in axis-angle format
    R = np.array([[ 0.707, 0, -0.707],
                [ 0,    -1,  0],
                [-0.707, 0, -0.707]])
    rot_base_to_task_L = Rotation.from_matrix(R).as_rotvec().tolist()

    # Rotate translation from task frame coordinates to robot base frame coordinates
    trans_base_to_task_L = np.matmul([dx_t, dy_t, dz_t], R).tolist()

    # Transformation from left robot base frame to task frame
    task_frame_L = trans_base_to_task_L + rot_base_to_task_L

    # RIGHT ROBOT ARM
    # Translation in task coordinate frame from right robot base origin to center of table (hand-measured):
    dx_t = -(0.090/2 + 0.010 + 0.110) # 1/2 vertical beam width + plate thickness + dist. to robot origin
    dy_t = 0.225/2 + 0.540/2 # half of mounting plate width + half of table depth
    dz_t = -0.753 # height from robot origin to top of table

    # Rotation from task coordinate frame to robot base frame in axis-angle format
    R = np.array([[ 0.707, 0, 0.707],
                    [ 0,    -1,  0],
                    [0.707, 0, -0.707]])
    rot_base_to_task_R = Rotation.from_matrix(R).as_rotvec().tolist()

    # Rotate translation from task frame coordinates to robot base frame coordinates
    trans_base_to_task_R = np.matmul([dx_t, dy_t, dz_t], R).tolist()

    # Transformation from right robot base frame to task frame
    task_frame_R = trans_base_to_task_R + rot_base_to_task_R

    #### FORCE MODE PARAMETERS ####
    # Define force mode parameters to move along task frame axes (same for both arms)
    selection_vector_x = [1, 0, 0, 0, 0, 0]
    wrench_neg_x = [-10, 0, 0, 0, 0, 0]
    wrench_pos_x = [10, 0, 0, 0, 0, 0]

    selection_vector_y = [0, 1, 0, 0, 0, 0]
    wrench_neg_y = [0, -10, 0, 0, 0, 0]
    wrench_pos_y = [0, 10, 0, 0, 0, 0]

    selection_vector_z = [0, 0, 1, 0, 0, 0]
    wrench_neg_z = [0, 0, -10, 0, 0, 0]
    wrench_pos_z = [0, 0, 10, 0, 0, 0]

    selection_vector_linear = [1, 1, 1, 0, 0, 0]
    selection_vector_torque = [0, 0, 0, 1, 1, 1]
    selection_vector_full = [1, 1, 1, 1, 1, 1]

    force_type = 2
    limits = [2, 2, 2, 1, 1, 1]

    ###############################################################################
    # LEFT ARM:
    # Initial joint position
    # Make sure that it is a safe position for the robot to move into without collsion!
    joint_q_L = np.radians([-48.24, -101.16, -107.03, -99.73, -120.90, 135.00])

    # Move left arm to initial joint position using joint position control
    print("Moving left arm to initial position using moveJ")
    # rtde_c_L.moveJ(joint_q_L)

    #### POSITION CONTROL ####
    TCP_vel = 0.2 # end effector velocity [m/s]
    TCP_accel = 0.25 # end effector acceleration [m/s^2]

    # Define target in task frame: 10 centimeters above the origin, gripper pointing down
    TCP_pose_L_task = [0, 0, 0.10, 0, 3.14, 0]

    # Transform target from task frame to robot base frame. If you want to think and
    # plan in the task frame, use this transformation to convert your target poses
    # to the robot base frame before sending them as commands to the robot.
    TCP_pose_L_base = rtde_c_L.poseTrans(task_frame_L, TCP_pose_L_task)

    # Move left arm to target position using moveL
    print("Moving left arm to target position using moveL")
    # rtde_c_L.moveL(TCP_pose_L_base, TCP_vel, TCP_accel, asynchronous=False)

    # Return to starting position
    print("Moving left arm to initial position using moveJ")
    # rtde_c_L.moveJ(joint_q_L)

    gripper_L = RobotiqGripper(rtde_c_L)
    gripper_L.activate()
    gripper_L.set_force(100)
    gripper_L.set_speed(100)
    gripper_L.open()

    gripper_R = RobotiqGripper(rtde_c_R)
    gripper_R.activate()
    gripper_R.set_force(50)
    gripper_R.set_speed(100)
    gripper_R.open()

    # FORCE CONTROL
    # Move along x axis of the task frame with force control, alternating between +x and -x every 2 seconds
    # Execute 500Hz control loop for 4 seconds, each cycle is 2ms

    prevstateL = True
    prevstateR = True
    while not directions["halt"]:
        # Begin timer for realtime control loop; this will ensure that each loop 
        # iteration takes 2ms regardless of how long the computations take
        t_startL = rtde_c_L.initPeriod()
        t_startR = rtde_c_R.initPeriod()

        if True not in directions.values():
            if compliant_mode:
                selection = selection_vector_full
                direction = [0, 0, 0, 0, 0, 0] #apply 0 force in all directions
                rtde_c_L.forceMode(task_frame_L, selection, direction, force_type, limits)
                rtde_c_R.forceMode(task_frame_R, selection, direction, force_type, limits)
            else:
                rtde_c_L.forceModeStop() #otherwise, fix the robot in place.
                rtde_c_R.forceModeStop()

        selection = selection_vector_full
        direction = direction = np.array([0, 0, 0, 0, 0, 0])
        
        if directions["a"]:
            direction += np.array(wrench_pos_x)
        if directions["d"]:
            direction += np.array(wrench_neg_x)
        if directions["w"]:
            direction += np.array(wrench_pos_y)
        if directions["s"]:
            direction += np.array(wrench_neg_y)
        if directions["q"]:
            direction += np.array(wrench_pos_z)
        if directions["e"]:
            direction += np.array(wrench_neg_z)
        
        if moveL:
            rtde_c_L.forceMode(task_frame_L, selection, direction, force_type, limits)
            if gripperstateL != prevstateL:
                if gripperstateL:
                    gripper_L.open()
                else:
                    gripper_L.close()
        if moveR:
            rtde_c_R.forceMode(task_frame_R, selection, direction, force_type, limits)
            if gripperstateR != prevstateR:
                if gripperstateR:
                    gripper_R.open()
                else:
                    gripper_R.close()
        
        prevstateL = gripperstateL
        prevstateR = gripperstateR
        # Wait until the next 2ms control cycle begins
        rtde_c_L.waitPeriod(t_startL)
        rtde_c_R.waitPeriod(t_startR)

    # Stop the program
    rtde_c_L.forceModeStop()
    listener.join()
