"""
Load ``config/tasks.json`` and step through each task: follow the A* polyline
waypoint-by-waypoint, then finish when the elevator reaches its goal, then
advance to the next task.

This module does not talk to hardware; call ``current_robot_waypoint``,
``current_elevator_goal``, and ``step`` from your control loop.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

try:
    from ..guidance.waypoint_controller import wrap_to_pi
    from ..planning.a_star import AStar, GridPoint, Waypoint, waypoints_from_polyline
except ImportError:
    from guidance.waypoint_controller import wrap_to_pi  # type: ignore[no-redef]
    from planning.a_star import AStar, GridPoint, Waypoint, waypoints_from_polyline  # type: ignore[no-redef]


Phase = Literal["robot", "elevator"]


@dataclass(frozen=True)
class Task:
    name: str
    robot_start: GridPoint
    robot_start_heading: float
    robot_goal: GridPoint
    robot_goal_heading: float
    elevator_start: float
    elevator_goal: float


def default_tasks_path() -> Path:
    """``mobile_robot/config/tasks.json`` when this file lives in ``src/autonomy``."""
    return Path(__file__).resolve().parent.parent.parent / "config" / "tasks.json"


def load_tasks(path: Path | str) -> List[Task]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[Task] = []
    for row in raw["tasks"]:
        rs = tuple(float(x) for x in row["robot_start"])
        rg = tuple(float(x) for x in row["robot_goal"])
        out.append(
            Task(
                name=str(row["name"]),
                robot_start=rs,
                robot_start_heading=float(row.get("robot_start_heading", 0.0)),
                robot_goal=rg,
                robot_goal_heading=float(row.get("robot_goal_heading", 0.0)),
                elevator_start=float(row.get("elevator_start", 0.0)),
                elevator_goal=float(row.get("elevator_goal", 0.0)),
            )
        )
    return out


def _hypot(a: GridPoint, b: GridPoint) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


class MissionRunner:
    """
    For each task: visit every robot waypoint (A* + polyline tangents), then
    require the elevator to reach ``elevator_goal`` before the next task.
    """

    def __init__(
        self,
        planner: AStar,
        tasks: Sequence[Task],
        *,
        xy_tol: float = 0.08,
        heading_tol: float = 0.25,
        elev_tol: float = 0.02,
    ):
        self._tasks = list(tasks)
        self._xy_tol = xy_tol
        self._heading_tol = heading_tol
        self._elev_tol = elev_tol

        self._wps: List[List[Waypoint]] = []
        for t in self._tasks:
            poly = planner.generate_plan(t.robot_start, t.robot_goal)
            if len(poly) < 2:
                raise ValueError(f"task {t.name!r}: empty or trivial plan {poly!r}")
            self._wps.append(
                waypoints_from_polyline(poly, end_heading=t.robot_goal_heading)
            )

        self._task_i = 0
        self._wp_i = 0
        self._phase: Phase = "robot"

    @property
    def task_index(self) -> int:
        return self._task_i

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def current_task(self) -> Optional[Task]:
        if self._task_i >= len(self._tasks):
            return None
        return self._tasks[self._task_i]

    def is_complete(self) -> bool:
        return self._task_i >= len(self._tasks)

    def current_robot_waypoint(self) -> Optional[Waypoint]:
        """Waypoint to track in the robot phase; ``None`` in elevator phase or when done."""
        if self.is_complete() or self._phase != "robot":
            return None
        wps = self._wps[self._task_i]
        if self._wp_i >= len(wps):
            return None
        return wps[self._wp_i]

    def current_elevator_goal(self) -> Optional[float]:
        """Elevator setpoint in the elevator phase; ``None`` otherwise (hold / ignore)."""
        if self.is_complete() or self._phase != "elevator":
            return None
        return self._tasks[self._task_i].elevator_goal

    def _at_robot_waypoint(
        self, x: float, y: float, heading: float, wp: Waypoint
    ) -> bool:
        if _hypot((x, y), wp.xy) > self._xy_tol:
            return False
        return abs(wrap_to_pi(heading - wp.heading)) <= self._heading_tol

    def step(self, *, x: float, y: float, heading: float, elevator_pos: float) -> None:
        """Call each control tick with your estimated state; advances tasks when goals are met."""
        if self.is_complete():
            return

        task = self._tasks[self._task_i]
        if self._phase == "robot":
            wps = self._wps[self._task_i]
            if self._wp_i < len(wps):
                wp = wps[self._wp_i]
                if self._at_robot_waypoint(x, y, heading, wp):
                    self._wp_i += 1
            if self._wp_i >= len(wps):
                self._phase = "elevator"
            return

        if abs(elevator_pos - task.elevator_goal) <= self._elev_tol:
            self._task_i += 1
            self._wp_i = 0
            self._phase = "robot"


__all__ = ["MissionRunner", "Task", "default_tasks_path", "load_tasks"]
