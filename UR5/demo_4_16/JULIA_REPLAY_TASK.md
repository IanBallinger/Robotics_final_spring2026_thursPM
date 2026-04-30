# Julia Replay Task From Python Frontend

This prototype adds a Julia-backed replay path on top of the existing Python task interface.

## What Was Added

- `UR5/julia_replay_task.py`
  - `JuliaReplayTask` subclass of `UR5TaskInterface`
  - Reads `actual_TCP_pose_0..5` from recorded CSV traces
  - Calls Julia planning API (`/replay/plan` or `/plan_replay`)
  - Replays returned poses with `moveL`
  - Replays gripper open/close transitions from CSV digital-output bitfield

- `UR5/demo_4_16/python_frontend.py`
  - Lightweight HTTP frontend for replay orchestration and team status
  - Endpoints:
    - `POST /ready`
    - `POST /replay/start`
    - `POST /pause`
    - `POST /play`
    - `POST /stop`
    - `POST /status-message`
    - `POST /returning`
    - `POST /complete/<num>`
    - `GET /health`
    - `GET /state`

- `UR5/demo_4_16/record.py`
  - Recorder CSV already contains digital output state fields used by replay.

## Start Frontend

```powershell
python UR5/demo_4_16/python_frontend.py --host 0.0.0.0 --port 8090
```

## Start Julia Replay Server

```powershell
julia julia_replay_server.jl --host 127.0.0.1 --port 8081
```

Server file:
- `julia_replay_server.jl`

## Start Replay From Frontend

```powershell
curl -X POST http://127.0.0.1:8090/replay/start `
  -H "Content-Type: application/json" `
  -d '{
    "robot_ip": "192.168.1.101",
    "trace_csv_path": "c:/Users/iballing/workspaces/robotics/robot_data_left.csv",
    "julia_api_base": "http://127.0.0.1:8081",
    "trace_side": "left",
    "downsample": 4,
    "gripper_state_column": "actual_digital_output_bits",
    "gripper_bit_index": 0,
    "gripper_closed_when_bit_set": true
  }'
```

## Mobile Team Status Message Example

```powershell
curl -X POST http://127.0.0.1:8090/status-message `
  -H "Content-Type: application/json" `
  -d '{"source": "mobile", "message": "on our way back now, get ready for handoff"}'
```

## Julia API Contract (Expected)

The replay task posts JSON to:
- `POST /replay/plan` (preferred)
- `POST /plan_replay` (fallback)

Request body includes:
- `trace_csv_path`
- `trace_side`
- `downsample`
- `raw_pose_count`
- `raw_poses`

Response body should include:
- `waypoints`: array of 6D poses `[x,y,z,rx,ry,rz]`

If Julia planner is unavailable, the task replays the raw recorded trace poses.
