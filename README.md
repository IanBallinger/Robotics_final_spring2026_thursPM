# Robotics_final_spring2026_thursPM

Note: ur_rtde only works with python 3.12 on windows. 

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

## Supervisor CLI Examples (UR5)

From the workspace root:

Windows (PowerShell):
```powershell
c:/Users/iballing/workspaces/robotics/venv/Scripts/python.exe UR5/Supervisor.py list-subtasks
```

Linux/macOS:
```bash
./venv/bin/python UR5/Supervisor.py list-subtasks
```

Run one subtask directly:
```powershell
c:/Users/iballing/workspaces/robotics/venv/Scripts/python.exe UR5/Supervisor.py run-subtask example --params-json "{}"
```

Run autonomy mode from JSON graph file (recommended):
```powershell
c:/Users/iballing/workspaces/robotics/venv/Scripts/python.exe UR5/Supervisor.py run-subtask ignored --autonomy-mode --autonomy-graph-file UR5/master_task_graph_example.json
```

Use the canonical graph (requires matching subtask implementations):
```powershell
c:/Users/iballing/workspaces/robotics/venv/Scripts/python.exe UR5/Supervisor.py run-subtask ignored --autonomy-mode --autonomy-graph-file UR5/master_task_graph.json
```

Optional override for max tasks from CLI:
```powershell
c:/Users/iballing/workspaces/robotics/venv/Scripts/python.exe UR5/Supervisor.py run-subtask ignored --autonomy-mode --autonomy-graph-file UR5/master_task_graph_example.json --autonomy-max-tasks 3
```

Simulate autonomy selection/progression (no robot actions, deadlock check):
```powershell
c:/Users/iballing/workspaces/robotics/venv/Scripts/python.exe UR5/Supervisor.py run-subtask ignored --autonomy-mode --autonomy-simulate --autonomy-graph-file UR5/master_task_graph.json
```

Expected output:
```text
autonomy simulation mode: no subtasks were executed
sim step 1: 'open_microwave_door' (id=open_door, points=1.0, arm=left)
sim step 2: 'acquire_bowl' (id=acquire_bowl, points=0.0, arm=right)
sim step 3: 'place_bowl_in_microwave' (id=put_bowl, points=1.0, arm=right)
sim step 4: 'right_arm_safe_retract' (id=right_retract, points=0.0, arm=right)
sim step 5: 'close_microwave_door' (id=close_door, points=1.0, arm=left)
sim step 6: 'press_microwave_stop' (id=press_stop_bowl, points=1.0, arm=right)
sim step 7: 'open_microwave_door' (id=door_open_for_unload, points=1.0, arm=left)
sim step 8: 'take_bowl_out_to_tray' (id=bowl_to_tray, points=1.0, arm=right)
sim step 9: 'acquire_plate' (id=acquire_plate, points=0.0, arm=right)
sim step 10: 'place_plate_in_microwave' (id=put_plate, points=1.0, arm=right)
sim step 11: 'close_microwave_door' (id=close_door_repeat, points=1.0, arm=left)
sim step 12: 'press_microwave_stop' (id=press_stop_plate, points=1.0, arm=right)
sim step 13: 'open_microwave_door' (id=door_open_for_plate_unload, points=1.0, arm=left)
sim step 14: 'take_plate_out_to_tray' (id=plate_to_tray, points=1.0, arm=right)
sim step 15: 'close_microwave_door' (id=close_microwave_door#18, points=1.0, arm=left)
sim step 16: 'acquire_cup' (id=acquire_cup, points=0.0, arm=right)
sim step 17: 'acquire_bottle' (id=acquire_bottle, points=0.0, arm=right)
sim step 18: 'pour_drink_into_cup' (id=pour_drink, points=1.0, arm=right)
sim step 19: 'place_cup_on_tray' (id=cup_on_tray, points=1.0, arm=right)
simulation completed without deadlock: executed=19 pending=0
simulation scoring summary: total_points=14.0 score_counts={'open_microwave_door': 3, 'place_bowl_in_microwave': 1, 'close_microwave_door': 3, 'press_microwave_stop_with_food_inside': 2, 'take_bowl_out_to_tray': 1, 'place_plate_in_microwave': 1, 'take_plate_out_to_tray': 1, 'pour_drink_from_bottle_into_cup': 1, 'place_cup_on_tray': 1}
```

Run an intentional deadlock simulation demo:
```powershell
c:/Users/iballing/workspaces/robotics/venv/Scripts/python.exe UR5/Supervisor.py run-subtask ignored --autonomy-mode --autonomy-simulate --autonomy-graph-file UR5/deadlock_task_graph_example.json
```

Score cap fields in graph tasks:
- score_token: groups repeated actions under one scoring bucket (for example, all door-open actions)
- max_score_count: maximum number of times points can be awarded for that score_token (0 means unlimited)

When max_score_count is reached, the task can still run for prerequisites and arm gating, but awarded points become 0 for that score_token.