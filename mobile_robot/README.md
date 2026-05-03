## Mobile Robot Documentation
We will be using pytrees, a Python-based behavior tree library to handle our stateful task assignment. Each task is a specific generated trajectory (can be waypoints or smooth paths or something in between) that is evaluated given the task and current state. The behavior tree will manage which task we are accomplishing; this decision will be made with a number of factors including:
 - which tasks have already been accomplished
 - state of the vehicle
 - status of HTTP API endpoints from UR5 robot
 - weighting of points (ie. if we can choose between two behaviors to accomplish at a certain time, do the one that will give us the most points)
 - operator input (in the worst case)

While each task is being completed, there should be two process ALWAYS running:
 - state estimation and localization: what is our position/velocity state at all times
 - collision avoidance: if we get too close (given the task, so this should be configurable), adjust the planned trajectory

