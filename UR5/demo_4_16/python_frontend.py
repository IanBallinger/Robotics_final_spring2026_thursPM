"""
Python frontend for task orchestration.

Provides simple HTTP endpoints for:
- mobile team status messaging
- pause/play controls
- launching a Julia-backed replay task

This process is intentionally independent from record.py so recording can run concurrently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UR5_DIR = os.path.dirname(_THIS_DIR)
if _UR5_DIR not in sys.path:
    sys.path.append(_UR5_DIR)

from julia_replay_task import JuliaReplayTask  # noqa: E402


@dataclass
class FrontendState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active_task: Optional[JuliaReplayTask] = None
    active_thread: Optional[threading.Thread] = None
    replay_status: str = "idle"
    last_error: str = ""
    returning: bool = False
    completed_tasks: List[int] = field(default_factory=list)
    status_messages: List[Dict[str, Any]] = field(default_factory=list)


STATE = FrontendState()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


def _run_replay_task(task: JuliaReplayTask):
    ok = False
    err = ""
    try:
        ok = task.execute()
    except RuntimeError as exc:
        err = str(exc)

    with STATE.lock:
        STATE.replay_status = "completed" if ok else "failed"
        STATE.last_error = err
        STATE.active_task = None
        STATE.active_thread = None


class FrontendHandler(BaseHTTPRequestHandler):
    server_version = "UR5Frontend/0.1"

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/health", "/state"):
            with STATE.lock:
                payload = {
                    "replay_status": STATE.replay_status,
                    "returning": STATE.returning,
                    "completed_tasks": list(STATE.completed_tasks),
                    "status_messages": list(STATE.status_messages[-30:]),
                    "last_error": STATE.last_error,
                }
            _json_response(self, 200, payload)
            return

        _json_response(self, 404, {"error": "unknown endpoint"})

    def do_POST(self):
        path = urlparse(self.path).path
        body = _read_json_body(self)

        if path == "/status-message":
            self._handle_status_message(body)
            return

        if path == "/returning":
            self._handle_returning(body)
            return

        if path in ("/ready", "/replay/start"):
            self._handle_replay_start(body)
            return

        if path == "/pause":
            self._handle_pause()
            return

        if path == "/play":
            self._handle_play()
            return

        if path == "/stop":
            self._handle_stop()
            return

        if path.startswith("/complete/"):
            self._handle_complete(path)
            return

        _json_response(self, 404, {"error": "unknown endpoint"})

    def _handle_status_message(self, body: Dict[str, Any]):
        message = str(body.get("message", "")).strip()
        source = str(body.get("source", "mobile"))
        if not message:
            _json_response(self, 400, {"error": "message is required"})
            return

        with STATE.lock:
            STATE.status_messages.append(
                {
                    "timestamp": time.time(),
                    "source": source,
                    "message": message,
                }
            )

        _json_response(self, 200, {"ok": True})

    def _handle_returning(self, body: Dict[str, Any]):
        message = str(body.get("message", "on our way back now, get ready for handoff"))
        with STATE.lock:
            STATE.returning = True
            STATE.status_messages.append(
                {
                    "timestamp": time.time(),
                    "source": "mobile",
                    "message": message,
                }
            )

        _json_response(self, 200, {"ok": True, "returning": True})

    def _handle_replay_start(self, body: Dict[str, Any]):
        robot_ip = str(body.get("robot_ip", "")).strip()
        trace_csv_path = str(body.get("trace_csv_path", "")).strip()
        julia_api_base = str(body.get("julia_api_base", "http://127.0.0.1:8081")).strip()
        trace_side = str(body.get("trace_side", "left")).strip().lower()
        downsample = int(body.get("downsample", 4))
        gripper_state_column = str(body.get("gripper_state_column", "actual_digital_output_bits")).strip()
        gripper_bit_index = int(body.get("gripper_bit_index", 0))
        gripper_closed_when_bit_set = bool(body.get("gripper_closed_when_bit_set", True))

        if not robot_ip:
            _json_response(self, 400, {"error": "robot_ip is required"})
            return
        if not trace_csv_path:
            _json_response(self, 400, {"error": "trace_csv_path is required"})
            return

        with STATE.lock:
            if STATE.active_thread is not None and STATE.active_thread.is_alive():
                _json_response(self, 409, {"error": "a replay task is already running"})
                return

            task = JuliaReplayTask(
                robot_ip=robot_ip,
                trace_csv_path=trace_csv_path,
                julia_api_base=julia_api_base,
                trace_side=trace_side,
                downsample=downsample,
                gripper_state_column=gripper_state_column,
                gripper_bit_index=gripper_bit_index,
                gripper_closed_when_bit_set=gripper_closed_when_bit_set,
                connect_immediately=False,
            )
            thread = threading.Thread(target=_run_replay_task, args=(task,), daemon=True)

            STATE.active_task = task
            STATE.active_thread = thread
            STATE.replay_status = "running"
            STATE.last_error = ""
            STATE.returning = False
            thread.start()

        _json_response(
            self,
            202,
            {
                "ok": True,
                "status": "running",
                "trace_csv_path": trace_csv_path,
                "gripper_state_column": gripper_state_column,
                "gripper_bit_index": gripper_bit_index,
                "gripper_closed_when_bit_set": gripper_closed_when_bit_set,
                "julia_api_base": julia_api_base,
            },
        )

    def _handle_pause(self):
        with STATE.lock:
            task = STATE.active_task
            if task is None:
                _json_response(self, 409, {"error": "no active replay task"})
                return
            task.request_pause()
            STATE.replay_status = "paused"

        _json_response(self, 200, {"ok": True, "status": "paused"})

    def _handle_play(self):
        with STATE.lock:
            task = STATE.active_task
            if task is None:
                _json_response(self, 409, {"error": "no active replay task"})
                return
            task.request_resume()
            STATE.replay_status = "running"

        _json_response(self, 200, {"ok": True, "status": "running"})

    def _handle_stop(self):
        with STATE.lock:
            task = STATE.active_task
            if task is None:
                _json_response(self, 409, {"error": "no active replay task"})
                return
            task.request_stop()
            STATE.replay_status = "stopping"

        _json_response(self, 200, {"ok": True, "status": "stopping"})

    def _handle_complete(self, path: str):
        _, _, raw_id = path.partition("/complete/")
        try:
            task_id = int(raw_id)
        except ValueError:
            _json_response(self, 400, {"error": "task id must be an integer"})
            return

        with STATE.lock:
            if task_id not in STATE.completed_tasks:
                STATE.completed_tasks.append(task_id)

        _json_response(self, 200, {"ok": True, "completed_tasks": list(STATE.completed_tasks)})


def main():
    parser = argparse.ArgumentParser(description="UR5 Python frontend for Julia replay tasks")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)
    print(f"Frontend listening on http://{args.host}:{args.port}")
    print("Endpoints: /health, /state, /ready, /replay/start, /pause, /play, /stop, /status-message, /returning, /complete/<num>")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
