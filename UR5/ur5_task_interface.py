"""
UR5 Task Template

tasks implement this template (specifically override setup() 
and perform_task_logic() ) for implementation.

The example here: Move to a series of 2 waypoints, 
then return to home position.
"""
import sys
from abc import ABC, abstractmethod
import time
import math
from arm import UR5Arm
PI = math.pi

#this is an interface. cannot be instantiated directly. 
# requires override of @abstractmethod decorated members
class UR5TaskInterface(ABC):
    """
    Template task.
    
    Inherit from this class and implement the execute() method to create
    task-specific control logic.
    """

    def __init__(self, robot_ip, connect_immediately=False):
        """
        Initialize the task and establish robot connection.

        Args:
            robot_ip (str): IP address of UR5 controller
        """
        self.robot_ip = robot_ip
        if connect_immediately: #useful for prototyping at task layer.
            self.robot = UR5Arm(self.robot_ip, verbose=True)
        self.home_position = None  # Define this in setup()
        self.waypoints = {}

    @abstractmethod
    def setup(self):
        """
        Configure task parameters and define key poses/positions.
        
        Override this method to define:
          - Home position
          - Target waypoints
          - Grasp poses
          - Any other task-specific parameters
        """
        # example code:
        # Define home position (placeholder values - adjust for your robot)
        # TODO pick some sane defaults, or default to current position
        self.home_position = [PI/2, PI/2, PI/2, PI/2, PI/2, 0.0]  # All joints at some angle

        # Example waypoints in p space (x, y, z, rx, ry, rz)
        self.waypoints = {
            #in the future, i'll make a way to pull these from the robot
            # via the api so we dont need to hardcode by hand.
            "pick_location": [0.3, 0.2, 0.5, PI, 0.0, 0.0],
            "place_location": [0.5, -0.3, 0.5, PI, 0.0, 0.0],
        }
        return True

    def verify_safety(self):
        """
        Verify robot is in safe state before executing task.

        Override this method to define extra task-readiness checks.

        Returns:
            bool: True if safe to proceed, False otherwise
        """
        # probably keep these.
        if not self.robot.is_connected():
            print("[ERROR] Robot not connected!")
            return False

        if self.robot.is_protected_stop_active():
            print("[ERROR] Protective stop is active!")
            return False

        if self.robot.is_emergency_stopped():
            print("[ERROR] Emergency stop is active!")
            return False

        # Check end-effector is ready (example)
        if not self.robot.is_steady():
            print("[WARNING] Robot not fully steady, waiting...")
            time.sleep(1) #could retry or fail-out, but for example purposes this is fine.

        print("[INFO] Safety checks passed")
        return True

    def move_to_home(self):
        """
        Move arm to home position.
        
        Override this with an early return to disable homing for the task.
        (maybe save the current position in that case)
        """
        print("[TASK] Moving to home position...")
        success = self.robot.move_to_joint_position(
            self.home_position,
            speed=UR5Arm.DEFAULT_JOINT_SPEED, #rad/sec
            acceleration=UR5Arm.DEFAULT_JOINT_ACCELERATION, #rad/(sec^2)
            asynchronous=False # let's standardize on synchronous method calls for now.
        )
        if success:
            print("[OK] Reached home position")
        else:
            print("[ERROR] Failed to reach home position")
        return success

    def move_to_waypoint(self, waypoint_name, use_linear=True): 
        #default case is to try linear motion, more intuitive for the programmer.
        """
        Move to a named waypoint.

        Args:
            waypoint_name (str): Name of waypoint from self.waypoints dict
            use_linear (bool): If True, try to use linear motion; if False, unconstrained.

        Returns:
            bool: True if move succeeded
        """
        if waypoint_name not in self.waypoints:
            print(f"[ERROR] Unknown waypoint: {waypoint_name}")
            return False

        pose = self.waypoints[waypoint_name]
        print(f"[TASK] Moving to {waypoint_name}: {pose}")

        if use_linear:
            success = self.robot.move_linear_to_pose(
                pose,
                speed=UR5Arm.DEFAULT_JOINT_SPEED, #rad/sec
                acceleration=UR5Arm.DEFAULT_JOINT_ACCELERATION, #rad/(sec^2)
                asynchronous=False
            )
        else:
            success = self.robot.move_to_pose(
                pose,
                speed=UR5Arm.DEFAULT_JOINT_SPEED, #rad/sec
                acceleration=UR5Arm.DEFAULT_JOINT_ACCELERATION, #rad/(sec^2)
                asynchronous=False
            )

        if success:
            # Read and display current state
            joint_pos = self.robot.get_joint_positions()
            end_effector_pose = self.robot.get_end_effector_pose()
            print(f"[OK] Current joint angles: {joint_pos}")
            print(f"[OK] Current p pose: {end_effector_pose}")
        else:
            print(f"[ERROR] Failed to reach waypoint {waypoint_name}")

        return success

    @abstractmethod
    def perform_task_logic(self):
        """
        Implement task-specific control logic here.
        """
        # TODO: Replace with actual task implementation

        print("\n[TASK] ===== EXECUTING EXAMPLE TASK =====")

        # Example 1: Move through waypoints
        print("\n--- Phase 1: Move to pick location ---")
        if not self.move_to_waypoint("pick_location", use_linear=True):
            return False

        # Simulate some work (e.g., grasp, force control, etc.)
        print("[TASK] Performing action at pick location...")
        time.sleep(2)

        # Example 2: Read sensor feedback
        print("\n--- Phase 2: Sensor feedback example ---")
        joint_torques = self.robot.get_joint_torques()
        end_effector_velocity = self.robot.get_end_effector_velocity()
        print(f"[INFO] Joint torques: {joint_torques}")
        print(f"[INFO] p velocity: {end_effector_velocity}")

        # Example 3: Move to place location
        print("\n--- Phase 3: Move to place location ---")
        if not self.move_to_waypoint("place_location", use_linear=True):
            return False

        print("[TASK] Performing action at place location...")
        time.sleep(2)

        print("\n[TASK] Main task completed successfully!")
        return True

    def cleanup(self):
        """
        Clean up resources and return robot to safe state.
        
        Override this to:
          - Return arm to home position
          - Disable end-effector/gripper
          - Save any data
          - Disconnect safely
        """
        print("\n[CLEANUP] Returning to home and cleaning up...")
        self.move_to_home()

        if self.robot.is_connected():
            self.robot.disconnect()
            print("[OK] Robot disconnected")

    def execute(self):
        """
        Main execution flow for the task.
        
        This orchestrates setup, verification, task execution, and cleanup.
        """
        if not self.robot:
            self.robot = UR5Arm(self.robot_ip, verbose=True)
        try:
            # Setup phase
            print("===== TASK STARTUP =====\n")
            self.setup()

            # Safety verification
            if not self.verify_safety():
                print("[FATAL] Safety verification failed!")
                return False

            # Move to home
            if not self.move_to_home():
                print("[FATAL] Failed to reach home position!")
                return False

            # Execute main task
            if not self.perform_task_logic():
                print("[FATAL] Task execution failed!")
                # Attempt to recover
                self.robot.stop_arm()
                time.sleep(1)
                return False

            # Cleanup
            self.cleanup()

            print("\n===== TASK COMPLETED SUCCESSFULLY =====")
            return True

        except KeyboardInterrupt:
            print("\n[INFO] Task interrupted by user")
            self.robot.stop_arm()
            self.cleanup()
            return False

        #catch the base case of Exception as well so we can safely halt.
        except Exception as e:
            print(f"\n[FATAL] Unexpected error: {str(e)}")
            self.robot.stop_arm()
            self.cleanup()
            return False


if __name__ == "__main__":
    # Example usage
    class ExampleTask(UR5TaskInterface):
        """small example of UR5TaskInterface usage"""
        def setup(self):
            return True
        def perform_task_logic(self):
            return False
        def cleanup(self):
            pass
    task = ExampleTask(robot_ip="192.168.0.1")
    try:
        if not task.setup():
            raise RuntimeError("broken setup step")
        if not task.execute():
            raise RuntimeError("broken execute step")

    except RuntimeError as error:
        print('Caught this error: ' + repr(error))

    finally:
        task.cleanup()

    sys.exit(0)
