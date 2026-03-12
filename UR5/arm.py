"""
UR5 Arm Controller - Robotics Final Project

This module provides a clean interface to the Universal Robots UR5 arm using RTDE.
It wraps the low-level RTDE API with helper functions to simplify robot control,
feedback sensing, and safety checks.

Mostly translated from https://sdurobotics.gitlab.io/ur_rtde/api/api.html to provide docstrings.
"""

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO
from math import pi as pi

class UR5Arm:
    """
    High-level control interface for the UR5 robot arms.

    ideally this interface simplifies access to the ur_rtde api, and provides helpful docstrings
    
    Provides access to arm movements, sensing, and state monitoring, 
    while maintaining safety constraints and best practices for RTDE lib.
    """

    # Default arm configuration
    # TODO we should tune these
    DEFAULT_JOINT_SPEED = pi/20                         # rad/s   see examples/move_until_contact
    DEFAULT_JOINT_ACCELERATION = DEFAULT_JOINT_SPEED/20 # rad/s^2 arbitrary default
    DEFAULT_TOOL_SPEED = 0.01                           # m/s     arbitrary small ddefault
    DEFAULT_TOOL_ACCELERATION = DEFAULT_TOOL_SPEED/20   # m/s^2   arbitrary default

    def __init__(self, ip_address, frequency=500.0, verbose=False):
        """
        Initialize connection to the UR5 arm.

        Args:
            ip_address (str): IP address of the robot controller (e.g., "192.168.0.1")
            frequency (float): RTDE communication frequency in Hz. Default 500Hz for e-Series
            verbose (bool): Enable verbose output for debugging

        Raises:
            Exception: If connection to robot fails
        """
        self.ip_address = ip_address
        self.verbose = verbose
        
        # Initialize RTDE interfaces
        self.rtde_control = None
        self.rtde_receive = None
        self.rtde_io = None
        
        self._connect(frequency)

    def _connect(self, frequency):
        """
        Establish RTDE connections to the robot.

        Args:
            frequency (float): RTDE communication frequency in Hz

        Raises:
            Exception: If connection fails
        """
        try:
            self.rtde_control = RTDEControl(self.ip_address, frequency)
            self.rtde_receive = RTDEReceive(self.ip_address, frequency)
            self.rtde_io = RTDEIO(self.ip_address)
            
            if self.verbose:
                print(f"[INFO] Connected to UR5 at {self.ip_address}")
                print(f"[INFO] RTDE frequency: {frequency} Hz")
        except Exception as e:
            raise Exception(f"Failed to connect to robot at {self.ip_address}: {str(e)}")

    def disconnect(self):
        """
        Safely disconnect all RTDE interfaces from the robot.
        """
        if self.rtde_control:
            self.rtde_control.disconnect()
        if self.rtde_receive:
            self.rtde_receive.disconnect()
        if self.rtde_io:
            self.rtde_io.disconnect()
        if self.verbose:
            print("[INFO] Disconnected from UR5")

    def is_connected(self):
        """
        Check if all RTDE interfaces are connected to the robot.

        Returns:
            bool: True if all interfaces are connected, False otherwise
        """
        if not self.rtde_control or not self.rtde_receive or not self.rtde_io:
            return False
        return (self.rtde_control.isConnected() and 
                self.rtde_receive.isConnected() and 
                self.rtde_io.isConnected())

    # ========== MOVEMENT PRIMITIVES ==========

    def move_to_joint_position(self, joint_angles, speed=None, acceleration=None, 
                               asynchronous=False):
        """
        Move arm to target joint position (linear in joint-space).

        Args:
            joint_angles (list): Target joint angles in radians [q0, q1, q2, q3, q4, q5]
            speed (float): Joint speed of leading axis in rad/s. Uses default if None
            acceleration (float): Joint acceleration in rad/s^2. Uses default if None
            asynchronous (bool): If True, return immediately; if False, block until complete

        Returns:
            bool: True if move succeeded, False otherwise

        Raises:
            ValueError: If joint_angles don't have exactly 6 elements
        """
        if len(joint_angles) != 6:
            raise ValueError("joint_angles must contain exactly 6 values")
        
        speed = speed or self.DEFAULT_JOINT_SPEED
        acceleration = acceleration or self.DEFAULT_JOINT_ACCELERATION
        
        try:
            result = self.rtde_control.moveJ(joint_angles, speed, acceleration, asynchronous) #linearity not guaranteed
            if self.verbose:
                print(f"[MOVE_J] Target: {joint_angles}, Async: {asynchronous}")
            return result
        except Exception as e:
            print(f"[ERROR] moveJ failed: {str(e)}")
            return False

    def move_to_pose(self, pose, speed=None, acceleration=None, asynchronous=False):
        """
        Move arm to target pose using inverse kinematics (linear in joint-space).

        Args:
            pose (list): Target end_effector pose [x, y, z, rx, ry, rz] in meters and radians
            speed (float): Joint speed in rad/s. Uses default if None
            acceleration (float): Joint acceleration in rad/s^2. Uses default if None
            asynchronous (bool): If True, return immediately; if False, block until complete

        Returns:
            bool: True if move succeeded, False otherwise

        Raises:
            ValueError: If pose doesn't have exactly 6 elements
        """
        if len(pose) != 6:
            raise ValueError("pose must contain exactly 6 values [x, y, z, rx, ry, rz]")
        
        speed = speed or self.DEFAULT_JOINT_SPEED
        acceleration = acceleration or self.DEFAULT_JOINT_ACCELERATION
        
        try:
            result = self.rtde_control.moveJ_IK(pose, speed, acceleration, asynchronous) #linear in joint space
            if self.verbose:
                print(f"[MOVE_J_IK] Target pose: {pose}, Async: {asynchronous}")
            return result
        except Exception as e:
            print(f"[ERROR] moveJ_IK failed: {str(e)}")
            return False

    def move_linear_to_pose(self, pose, speed=None, acceleration=None, 
                           asynchronous=False):
        """
        Move arm to target pose with linear tool-space motion.

        Args:
            pose (list): Target end_effector pose [x, y, z, rx, ry, rz] in meters and radians
            speed (float): Tool speed in m/s. Uses default if None
            acceleration (float): Tool acceleration in m/s^2. Uses default if None
            asynchronous (bool): If True, return immediately; if False, block until complete

        Returns:
            bool: True if move succeeded, False otherwise

        Raises:
            ValueError: If pose doesn't have exactly 6 elements
        """
        if len(pose) != 6:
            raise ValueError("pose must contain exactly 6 values [x, y, z, rx, ry, rz]")
        
        speed = speed or self.DEFAULT_TOOL_SPEED
        acceleration = acceleration or self.DEFAULT_TOOL_ACCELERATION
        
        try:
            result = self.rtde_control.moveL(pose, speed, acceleration, asynchronous)
            if self.verbose:
                print(f"[MOVE_L] Target pose: {pose}, Async: {asynchronous}")
            return result
        except Exception as e:
            print(f"[ERROR] moveL failed: {str(e)}")
            return False

    def move_path(self, waypoints, asynchronous=False): #TODO
        """
        Execute a path through multiple waypoints.

        Each waypoint should include motion parameters: 
        [x, y, z, rx, ry, rz, speed, acceleration, blend_radius]



        Args:
            waypoints (list): List of waypoint configurations
            asynchronous (bool): If True, return immediately; if False, block until complete

        Returns:
            bool: True if path execution succeeded, False otherwise
        """
        try:
            # TODO: Implement path following. 
            # see https://sdurobotics.gitlab.io/ur_rtde/examples/examples.html#movel-path-with-blending-example
            # robustness principal:
            # if path is [[1,2,3,4,5,6],...] use movel(pose, speed, acc) => no blending
            # else use movel(path) (but check entry length == 9) => use user specified blending.
            #long story short: path entries list should be appended with vel, acc, and blend radius per entry
            if self.verbose:
                print(f"[PATH] Executing path with {len(waypoints)} waypoints")
            return True
        except Exception as e:
            print(f"[ERROR] path execution failed: {str(e)}")
            return False

    def stop_arm(self, deceleration=10.0, asynchronous=False, use_linear=True):
        """
        Stop all arm motion with controlled deceleration.

        Args:
            deceleration (float): Deceleration rate (m/s^2 for linear, rad/s^2 for joint)
            asynchronous (bool): If True, return immediately; if False, wait for stop
            use_linear (bool): If True, use linear deceleration; if False, use joint space

        Returns:
            bool: True if stop commanded successfully
        """
        try:
            if use_linear:
                self.rtde_control.stopL(deceleration, asynchronous) #linear in tool space
            else:
                self.rtde_control.stopJ(deceleration, asynchronous) #linear in joint space
            if self.verbose:
                print(f"[STOP] Arm stopped with deceleration {deceleration} m/s^2")
            return True
        except Exception as e:
            print(f"[ERROR] stop failed: {str(e)}")
            #TODO this may qualify as a requirement to auto-trigger e-stop.
            return False

    # ========== SENSING & FEEDBACK ==========

    def get_joint_positions(self):
        """
        Get current actual joint positions from encoders.

        Returns:
            list: Current joint angles [q0, q1, q2, q3, q4, q5] in radians,
                  or None if read failed
        """
        try:
            return self.rtde_receive.getActualQ()
        except Exception as e:
            print(f"[ERROR] Failed to read joint positions: {str(e)}")
            return None

    def get_joint_velocities(self):
        """
        Get current actual joint velocities.

        Returns:
            list: Current joint velocities [qd0, qd1, qd2, qd3, qd4, qd5] in rad/s,
                  or None if read failed
        """
        try:
            return self.rtde_receive.getActualQd()
        except Exception as e:
            print(f"[ERROR] Failed to read joint velocities: {str(e)}")
            return None

    def get_end_effector_pose(self):
        """
        Get current actual end_effector position and orientation.

        Returns:
            list: Current end_effector pose [x, y, z, rx, ry, rz] in meters and radians,
                  or None if read failed
        """
        try:
            return self.rtde_receive.getActualTCPPose()
        except Exception as e:
            print(f"[ERROR] Failed to read end_effector pose: {str(e)}")
            return None

    def get_end_effector_velocity(self):
        """
        Get current end_effector linear and angular velocity.

        Returns:
            list: Current end_effector velocity [vx, vy, vz, ωx, ωy, ωz] in m/s and rad/s,
                  or None if read failed
        """
        try:
            return self.rtde_receive.getActualTCPSpeed()
        except Exception as e:
            print(f"[ERROR] Failed to read end_effector velocity: {str(e)}")
            return None

    def get_joint_torques(self):
        """
        Get current joint torques measured at joints (gravity-compensated).

        Returns:
            list: Joint torques [τ0, τ1, τ2, τ3, τ4, τ5] in Nm,
                  or None if read failed
        """
        try:
            return self.rtde_control.getJointTorques()
        except Exception as e:
            print(f"[ERROR] Failed to read joint torques: {str(e)}")
            return None

    def get_joint_temperatures(self):
        """
        Get temperature of each joint.

        Returns:
            list: Joint temperatures in degrees Celsius, or None if read failed
        """
        try:
            return self.rtde_receive.getJointTemperatures()
        except Exception as e:
            print(f"[ERROR] Failed to read joint temperatures: {str(e)}")
            return None

    # ========== SAFETY & STATE MONITORING ==========

    def is_steady(self):
        """
        Check if arm is fully at rest and ready for external forces/tools.

        Returns:
            bool: True if arm is steady, False otherwise
        """
        try:
            return self.rtde_control.isSteady()
        except Exception as e:
            print(f"[ERROR] Failed to check steady state: {str(e)}")
            return False

    def is_program_running(self):
        """
        Check if a program is currently running on the controller.

        Returns:
            bool: True if program running, False otherwise
        """
        try:
            return self.rtde_control.isProgramRunning()
        except Exception as e:
            print(f"[ERROR] Failed to check program status: {str(e)}")
            return False

    def is_protected_stop_active(self):
        """
        Check if protective stop is currently active.

        Returns:
            bool: True if protective stop is active, False otherwise
        """
        try:
            return self.rtde_receive.isProtectiveStopped()
        except Exception as e:
            print(f"[ERROR] Failed to read protective stop state: {str(e)}")
            return False

    def is_emergency_stopped(self):
        """
        Check if emergency stop is active.

        Returns:
            bool: True if emergency stop is active, False otherwise
        """
        try:
            return self.rtde_receive.isEmergencyStopped()
        except Exception as e:
            print(f"[ERROR] Failed to read emergency stop state: {str(e)}")
            return False

    def get_robot_status(self):
        """
        Get comprehensive robot status flags.

        Returns:
            dict: Status information including power, program state, buttons, or None if failed
        """
        try:
            status = self.rtde_control.getRobotStatus()
            return {
                "power_on": bool(status & 0x01),
                "program_running": bool(status & 0x02),
                "teach_button_pressed": bool(status & 0x04),
                "power_button_pressed": bool(status & 0x08),
            }
        except Exception as e:
            print(f"[ERROR] Failed to read robot status: {str(e)}")
            return None

    # ========== KINEMATICS & TRANSFORMS ==========

    def get_forward_kinematics(self, joint_angles=None):
        """
        Calculate end_effector pose from joint angles using forward kinematics.

        Args:
            joint_angles (list): Joint angles in radians. Uses current position if None

        Returns:
            list: end_effector pose [x, y, z, rx, ry, rz] in meters and radians, or None if failed
        """
        try:
            if joint_angles is None:
                return self.rtde_control.getForwardKinematics()
            return self.rtde_control.getForwardKinematics(joint_angles)
        except Exception as e:
            print(f"[ERROR] Forward kinematics failed: {str(e)}")
            return None

    def get_inverse_kinematics(self, pose, near_joints=None):
        """
        Calculate joint angles from end_effector pose using inverse kinematics.

        Args:
            pose (list): Target pose [x, y, z, rx, ry, rz]
            near_joints (list): Preferred joint configuration. If provided, returns solution
                               closest to this configuration

        Returns:
            list: Joint angles [q0, q1, q2, q3, q4, q5] in radians, or None if no solution
        """
        try:
            if near_joints is None:
                return self.rtde_control.getInverseKinematics(pose)
            return self.rtde_control.getInverseKinematics(pose, near_joints)
        except Exception as e:
            print(f"[ERROR] Inverse kinematics failed: {str(e)}")
            return None

    def check_inverse_kinematics_valid(self, pose, near_joints=None):
        """
        Check if an inverse kinematics solution exists without calculating it.

        Args:
            pose (list): Target pose [x, y, z, rx, ry, rz]
            near_joints (list): Preferred configuration (optional)

        Returns:
            bool: True if a valid IK solution exists, False otherwise
        """
        try:
            return self.rtde_control.getInverseKinematicsHasSolution(pose, near_joints)
        except Exception as e:
            print(f"[ERROR] IK validation failed: {str(e)}")
            return False

    def pose_transform(self, pose_from, pose_from_to):
        """
        Transform a pose relative to another pose (composition of transforms).

        Args:
            pose_from (list): Reference pose [x, y, z, rx, ry, rz]
            pose_from_to (list): Transform relative to reference pose

        Returns:
            list: Resulting absolute pose, or None if failed
        """
        try:
            return self.rtde_control.poseTrans(pose_from, pose_from_to)
        except Exception as e:
            print(f"[ERROR] Pose transform failed: {str(e)}")
            return None

    # ========== FORCE/TORQUE SENSING & CONTROL ==========

    def set_payload(self, mass, center_of_gravity=None):
        """
        Set the mass and center of gravity of the tool payload.

        This must be updated when picking up or putting down objects.

        Args:
            mass (float): Payload mass in kilograms
            center_of_gravity (list): CoG as [x, y, z] displacement from tool flange in meters.
                                      If None, uses current CoG.

        Returns:
            bool: True if payload set successfully, False otherwise
        """
        try:
            if center_of_gravity is None:
                return self.rtde_control.setPayload(mass)
            return self.rtde_control.setPayload(mass, center_of_gravity)
        except Exception as e:
            print(f"[ERROR] Failed to set payload: {str(e)}")
            return False

    def set_target_payload(self, mass, center_of_gravity, inertia=None):
        """
        Set payload mass, center of gravity, and inertia for improved dynamics.

        Args:
            mass (float): Payload mass in kilograms
            center_of_gravity (list): CoG displacement [x, y, z] from tool flange in meters
            inertia (list): Inertia matrix [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] in kg·m^2.
                          If None, zero matrix is used.

        Returns:
            bool: True if payload set successfully, False otherwise
        """
        try:
            if inertia is None:
                inertia = [0.0] * 6
            return self.rtde_control.setTargetPayload(mass, center_of_gravity, inertia)
        except Exception as e:
            print(f"[ERROR] Failed to set target payload: {str(e)}")
            return False

    def zero_ft_sensor(self):
        """
        Zero the end_effector force/torque sensor by subtracting current measurement.

        Call this before performing force-sensitive tasks.

        Returns:
            bool: True if sensor zeroed successfully, False otherwise
        """
        try:
            return self.rtde_control.zeroFtSensor()
        except Exception as e:
            print(f"[ERROR] Failed to zero FT sensor: {str(e)}")
            return False

    # ========== DIGITAL I/O ==========

    def set_digital_output(self, output_id, signal_level):
        """
        Set a standard digital output pin high or low.

        Args:
            output_id (int): Output ID (0-7 for standard outputs)
            signal_level (bool): True for high/ON, False for low/OFF

        Returns:
            bool: True if set successfully, False otherwise
        """
        try:
            return self.rtde_io.setStandardDigitalOut(output_id, signal_level)
        except Exception as e:
            print(f"[ERROR] Failed to set digital output {output_id}: {str(e)}")
            return False

    def get_digital_input(self, input_id):
        """
        Read state of a standard digital input pin.

        Args:
            input_id (int): Input ID (0-7 for standard inputs)

        Returns:
            bool: True if input is high, False if low, or None if failed
        """
        try:
            return self.rtde_receive.getDigitalInState(input_id)
        except Exception as e:
            print(f"[ERROR] Failed to read digital input {input_id}: {str(e)}")
            return None

    # ========== TASK IMPLEMENTATION STUB ==========

    def execute_task(self):
        """
        Main task execution method - requires override.

        This stub should be replaced with the actual task implementation.

        """
        # TODO: Implement your task logic here
        print("[INFO] execute_task() not implemented - override in subclass")
        pass