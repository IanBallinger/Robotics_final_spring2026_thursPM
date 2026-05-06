# Robotics_final_spring2026_thursPM

Note: ur_rtde only works with python 3.12 on windows. 

## Demo Video

<video controls src="./scheduler_demo.mp4" width="960">
	Your browser does not support embedded videos.
</video>

If the embedded player does not render on your platform, use this direct link: [scheduler_demo.mp4](./scheduler_demo.mp4)

## Getting Started:
from your terminal:

```bash
git clone https://github.com/IanBallinger/Robotics_final_spring2026_thursPM.git
cd Robotics_final_spring2026_thursPM
git submodule update --init --recursive
code .
```

Then (windows):
winget ships with win11, but you can also install it from [it's github releases page](https://github.com/microsoft/winget-cli/releases)
```bash
winget install Python.Python.3.12
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

(linux/macos):
You should have python 3.12 installed.
```bash
python3.12 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## UR5 Runbook

All commands below are from repo root.

Windows PowerShell:

	.\venv\Scripts\python.exe

Linux or macOS:

	./venv/bin/python

### 1) Quick sanity checks

List available subtasks:

	.\venv\Scripts\python.exe UR5/Supervisor.py list-subtasks

Run a trivial subtask:

	.\venv\Scripts\python.exe UR5/Supervisor.py run-subtask example --params-json "{}"

### 2) Safest first run (offline simulation)

No robot motion. Useful to verify graph logic and deadlocks.

	.\venv\Scripts\python.exe UR5/Supervisor.py run-subtask ignored --autonomy-mode --autonomy-simulate --autonomy-graph-file UR5/master_task_graph.json

Optional with visualizer:

	.\venv\Scripts\python.exe UR5/Supervisor.py run-subtask ignored --autonomy-mode --autonomy-simulate --autonomy-visualizer simulate --autonomy-graph-file UR5/master_task_graph.json

### 3) Live scheduler with controls (hardware path)

Starts paused with visualizer controls enabled.

	.\venv\Scripts\python.exe UR5/Supervisor.py run-subtask example --autonomy-mode --autonomy-visualizer live --autonomy-graph-file UR5/master_task_graph.json --autonomy-start-paused --autonomy-open-loop-default

Recommended safety toggles when hardware is partial:

- Disable gripper calls:

	--no-gripper

- Disable camera and vision feed startup:

	--no-camera

### 4) Single-arm test-rig mode

Use one UR controller for both logical lanes, and execute only one real arm lane while mocking the other.

	.\venv\Scripts\python.exe UR5/Supervisor.py run-subtask example --autonomy-mode --autonomy-visualizer live --autonomy-graph-file UR5/master_task_graph.json --autonomy-start-paused --autonomy-open-loop-default --single-arm-robot-ip 192.168.2.103 --single-arm-mode right --no-gripper --no-camera

Notes:

- single-arm-mode right means right lane is real, left lane is scheduler-mocked.
- single-arm-mode left means left lane is real, right lane is scheduler-mocked.

### 5) Waypoint tuning UI

Offline mock tuner (recommended):

	.\venv\Scripts\python.exe UR5/waypoint_tuning_runner.py --waypoints-csv UR5/waypoints_move_tray.csv --task-id move_tray --arm-side right --mock-robot --mock-state-file traces/mock_robot_state.json --no-camera

Hardware tuner:

	.\venv\Scripts\python.exe UR5/waypoint_tuning_runner.py --waypoints-csv UR5/waypoints_move_tray.csv --task-id move_tray --arm-side right --robot-ip 192.168.1.102 --no-camera

### 6) Recorder workflow (Python + Julia)

Start Julia live plot UI:

	julia --threads 1 live_plot_runner.jl --host 127.0.0.1 --port 9999 --named-waypoints-csv UR5/waypoints_acquire_bowl.csv

Start Python recorder:

	.\venv\Scripts\python.exe UR5/demo_4_16/record.py --stream-udp-host 127.0.0.1 --stream-udp-port 9999 --task-graph-file UR5/master_task_graph.json --task-id acquire_bowl --named-waypoints-csv UR5/waypoints_acquire_bowl.csv --write-task-graph-labels

Single-arm recorder mode:

	.\venv\Scripts\python.exe UR5/demo_4_16/record.py --stream-udp-host 127.0.0.1 --stream-udp-port 9999 --task-graph-file UR5/master_task_graph.json --task-id move_tray --named-waypoints-csv UR5/waypoints_move_tray.csv --write-task-graph-labels --single-arm-robot-ip 192.168.2.103 --no-gripper

Controls while recording:

- w queues a waypoint mark.
- l and r toggle gripper open or close states.
- Delete or Ctrl+C stops recording.

### 7) Troubleshooting notes

- If a task prints RTDE control script is not running, verify robot program state and clear any protective stop on the teach pendant.
- For camera-less setups, use no-camera on Supervisor, tuner, and recorder launch paths.
- For gripper-less setups, use no-gripper to skip all gripper activation and command calls.

## Mobile Robot

Source doc: mobile_robot/README.md

The mobile robot stack uses `pytrees` (Python behavior trees) for stateful task assignment.

Task selection factors include:

- which tasks have already been accomplished
- current vehicle state
- status of HTTP API endpoints from the UR5 robot
- weighted points (choose higher-value behavior when choices are available)
- operator input (fallback path)

Two processes are intended to run continuously while executing tasks:

- state estimation and localization: maintain current position and velocity state
- collision avoidance: enforce configurable proximity limits and adjust trajectory when needed
