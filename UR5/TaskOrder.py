#handle task selection strategy, initialize and maintain task list
#  use a priority queue
import heapq #priority queue
from ur5_task_interface import UR5TaskInterface


class ExampleTask(UR5TaskInterface): 
    """ prototyping sketch of behavior """
    #override these methods specifically to be able to instantiate tasks.
    def setup(self):
        pass
    def perform_task_logic(self):
        pass
    def cleanup(self):
        pass

#may ned to rafactor UR5TaskInterface to make collaborative tasks work.
class ExampleCollaborativeTask(UR5TaskInterface): 
    """ prototyping sketch of coordinated behavior """
    left_tasks = []
    right_tasks = []
    class SubtaskL(ExampleTask):
        """internal"""
    class SubtaskR(ExampleTask):
        """internal"""

    def __init__(self, left_ip = "1.0.0.1", right_ip = "1.0.0.2"):
        #TODO this is where the refactor needs to happen. super.__init__ assumes 1 robot.
        self.left_subtasks = [self.SubtaskL(left_ip)]
        self.right_subtasks = [self.SubtaskR(right_ip)]
    def setup(self):
        pass
    def perform_task_logic(self):
        pass
    def cleanup(self):
        pass

class TaskOrder:
    """ prototype. this is where automatic task selection happens """
    tasklist = []
    def __init__(self, tasks, left_ip=None, right_ip=None):
        #initialize tasks list here
        self.tasklist = [ #or something along these lines.
            #(points enabled, name/identifier, task object)
            (6+2, "microwave_plate", ExampleCollaborativeTask(left_ip, right_ip)),
            (6, "microwave_plate", ExampleTask(left_ip)),
            (6+2, "microwave_bowl", ExampleCollaborativeTask(left_ip, right_ip)),
            # etc....
            ]
        #min heap by default, heapq_max not supported until py3.14
        self.tasklist = { (1/item[1], item[2], item[3]) for item in self.tasklist }
        self.tasklist = heapq.heapify(tasks)
        return

    def next(self):
        "get the next task"
        return heapq.heappop(self.tasklist)
