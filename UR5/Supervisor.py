# drive the overarching program
# serve the webserver frontend to mobile-bot team, keep track of status and impending collaboration events
#  webapi endpoints: 
#   /ready (when the mobile bot wants a tray)
#   /returning (impending ready call, update task priorities to prepare for mobile bot quickly)
#   /complete/<num> (flag task number num as complete -> update downstream points possible per task and update weights)
#   /pause (gracefully pause all motions)
#   /play (gracefully resume all paused motions)
from multiprocessing import Process
from enum import Enum
import numpy as np
# from UR5.TaskOrder import TaskOrder
from UR5.ur5_task_interface import UR5TaskInterface
PI = np.pi

left_arm_ip = "1.0.0.1"
right_arm_ip = "1.0.0.2"

class Handedness(Enum):
    """control the coordinate frame of the robot waypoints"""
    LEFT = "left"
    RIGHT = "right"

COORDINATEFRAMES = {
    "global":{"x":[1,0,0], "y":[0,1,0], "z":[0,0,1]},
    Handedness.RIGHT:{"x":[1,0,0], "y":[0,1,0], "z":[0,0,1]},
    Handedness.LEFT:{"x":[1,0,0], "y":[0,1,0], "z":[0,0,1]},
}

class ExampleTaskL(UR5TaskInterface): #override these methods specifically to be able to instantiate.
    """ lightweight example task """
    waypoints = {}

    def __init__(self, handedness: Handedness):
        if not isinstance(handedness, Handedness):
            raise TypeError("handedness must be = Handedness.LEFT or .RIGHT")
        match handedness:
            case Handedness.LEFT:
                super().__init__(left_arm_ip)
            case Handedness.RIGHT:
                super().__init__(right_arm_ip)

    def setup(self):
        """ hold arm straight out from robot base """
        super().setup() #example code defined in the base class. you can extend it or replace it.
        self.waypoints = {"t-pose": [0.3, 0.2, 0.5, PI, 0.0, 0.0]}

    def perform_task_logic(self):
        pass

    def cleanup(self):
        pass

class ExampleTaskR(ExampleTaskL):
    def setup(self):
        super().setup()
        self.waypoints = {
            name: (-x, y, z, rx, ry, rz)
            for name, (x, y, z, rx, ry, rz) in self.waypoints.items()
        }


class Supervisor:
    """ example of running multiple arms """
    subtasks = []
    def __init__(self):
        self.subtasks = [
            ExampleTaskL(Handedness.LEFT),
            ExampleTaskR(Handedness.RIGHT),
        ]
        #just a small demo, should control 2 arms with the same movements.
        results = [ Process(target = task.execute) for task in self.subtasks ] 
        print(results)
