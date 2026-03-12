# drive the overarching program
# serve the webserver frontend to mobile-bot team, keep track of status and impending collaboration events
#  webapi endpoints: 
#   /ready (when the mobile bot wants a tray)
#   /returning (impending ready call, update task priorities to prepare for mobile bot quickly)
#   /complete/<num> (flag task number num as complete -> update downstream points possible per task and update weights)
#   /pause (gracefully pause all motions)
#   /play (gracefully resume all paused motions)
from multiprocessing import Process
import numpy as np

from UR5.TaskOrder import TaskOrder
from UR5.ur5_task_interface import UR5TaskInterface


import UR5.ur5_task_interface as ur5_task_interface

class ExampleTask(ur5_task_interface): #override these methods specifically to be able to instantiate.
    """ example task """
    def setup(self):
        super().setup() #example code defined in the base class. you can extend it or replace it.

    def perform_task_logic(self):
        super.perform_task_logic()

    def cleanup(self):
        super.cleanup()

class ExampleTask_mirrored(ExampleTask):
    def setup(self):
        super().setup() #example code defined in the base class. you can extend it or replace it.
        self.waypoints = [-1.0 * waypoint.x for waypoint in self.waypoints]
        pass

class Supervisor:
    #TODO manage the task queue
    left_arm = "1.0.0.1"
    right_arm = "1.0.0.2"
    subtasks = []
    def __init__(self):
        self.subtasks = [
            ExampleTask(self.right_arm),
            ExampleTask(self.right_arm),
        ]
        [ Process(target = task.execute) for task in self.subtasks ] #just a small demo, should control 2 arms with the same movements.
        return