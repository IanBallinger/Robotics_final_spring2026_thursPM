"""
Load ``config/mission_config.yaml`` and step through each task using boolean enter and
completion conditions.

This module is intentionally planner-agnostic and hardware-agnostic. The YAML
contains:
- one mission-level initial pose in the global frame
- one goal pose per task in the global frame
- enter conditions
- completion conditions

Task start poses are inferred sequentially: the first task starts at the
mission initial pose, and each later task starts at the previous task goal.

The runner advances to the next task only when:
1) the current task's enter conditions evaluate to ``True``
2) the current task's completion conditions evaluate to ``True``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from localization.map import Landmark, Map, Obstacle


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    heading: float


@dataclass(frozen=True)
class Task:
    name: str
    start: Pose2D
    goal: Pose2D
    desired_elevator_height_m: float
    enter_conditions: List[str]
    completion_conditions: List[str]


def default_tasks_path() -> Path:
    """``mobile_robot/config/mission_config.yaml`` when this file lives in ``src/autonomy``."""
    return Path(__file__).resolve().parent.parent.parent / "config" / "mission_config.yaml"


def _load_pose(raw: Mapping[str, Any]) -> Pose2D:
    pos = raw["position"]
    return Pose2D(
        x=float(pos["x"]),
        y=float(pos["y"]),
        heading=float(raw.get("heading", 0.0)),
    )


def _load_point(raw: Mapping[str, Any]) -> tuple[float, float]:
    return (float(raw["x"]), float(raw["y"]))


def load_map(path: Path | str) -> Map:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    map_raw = raw.get("map")
    if map_raw is None:
        raise ValueError("mission_config.yaml must define a 'map'")

    boundary_raw = map_raw.get("boundary")
    if boundary_raw is None:
        raise ValueError("mission_config.yaml map must define a 'boundary'")

    resolution = float(map_raw.get("resolution", 0.05))
    map_ = Map([_load_point(point) for point in boundary_raw], resolution)

    for landmark_raw in map_raw.get("landmarks", []):
        map_.add_landmark(
            Landmark(
                _load_point(landmark_raw["position"]),
                str(landmark_raw["name"]),
                id=str(landmark_raw.get("id", landmark_raw["name"])),
                heading=float(landmark_raw.get("heading", 0.0)),
            )
        )


    if map_raw.get("obstacles", []) is not None:
        for obstacle_raw in map_raw.get("obstacles", []):
            obstacle_boundary = obstacle_raw.get("boundary")
            if obstacle_boundary is None:
                raise ValueError(
                    f"map obstacle '{obstacle_raw.get('name', '<unnamed>')}' must define a 'boundary'"
                )
            map_.add_obstacle(
                Obstacle(
                    [_load_point(point) for point in obstacle_boundary],
                    str(obstacle_raw["name"]),
                )
            )

    return map_


def load_tasks(path: Path | str) -> List[Task]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    initial_state_raw = raw.get("initial_state")
    if initial_state_raw is None:
        raise ValueError("mission_config.yaml must define an 'initial_state'")

    out: List[Task] = []
    current_start = _load_pose(initial_state_raw)
    for row in raw["tasks"]:
        goal = _load_pose(row["goal"])
        out.append(
            Task(
                name=str(row["name"]),
                start=current_start,
                goal=goal,
                desired_elevator_height_m=float(row.get("desired_elevator_height_m", 0.0)),
                enter_conditions=[str(x) for x in row.get("enter_conditions", [])],
                completion_conditions=[
                    str(x) for x in row.get("completion_conditions", [])
                ],
            )
        )
        current_start = goal
    return out


def evaluate_condition(expression: str, context: Mapping[str, Any]) -> bool:
    """Evaluate a boolean task condition against a context dictionary."""
    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "min": min,
        "max": max,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
    }
    return bool(eval(expression, safe_globals, dict(context)))


class MissionRunner:
    """Sequential mission runner over YAML-defined tasks."""

    def __init__(self, tasks: Sequence[Task]):
        self._tasks = list(tasks)
        self._task_i = 0

    @property
    def task_index(self) -> int:
        return self._task_i

    @property
    def current_task(self) -> Optional[Task]:
        if self.is_complete():
            return None
        return self._tasks[self._task_i]

    def is_complete(self) -> bool:
        return self._task_i >= len(self._tasks)

    def _context_for_task(
        self, task: Task, context: Mapping[str, Any], *, previous_task_complete: bool
    ) -> Dict[str, Any]:
        merged = dict(context)
        merged.setdefault("current_task", task.name)
        merged.setdefault("task_index", self._task_i)
        merged.setdefault("previous_task_complete", previous_task_complete)
        merged.setdefault("desired_elevator_height_m", task.desired_elevator_height_m)
        return merged

    def conditions_met(
        self,
        expressions: Sequence[str],
        context: Mapping[str, Any],
        *,
        previous_task_complete: bool,
    ) -> bool:
        task = self.current_task
        if task is None:
            return False
        merged = self._context_for_task(
            task, context, previous_task_complete=previous_task_complete
        )
        return all(evaluate_condition(expr, merged) for expr in expressions)

    def can_enter(self, context: Mapping[str, Any]) -> bool:
        task = self.current_task
        if task is None:
            return False
        return self.conditions_met(
            task.enter_conditions,
            context,
            previous_task_complete=(self._task_i > 0),
        )

    def task_complete(self, context: Mapping[str, Any]) -> bool:
        task = self.current_task
        if task is None:
            return False
        return self.conditions_met(
            task.completion_conditions,
            context,
            previous_task_complete=(self._task_i > 0),
        )

    def current_start_pose(self) -> Optional[Pose2D]:
        task = self.current_task
        return None if task is None else task.start

    def current_goal_pose(self) -> Optional[Pose2D]:
        task = self.current_task
        return None if task is None else task.goal

    def step(self, context: Mapping[str, Any]) -> None:
        """Advance to the next task when the active task is enterable and complete."""
        task = self.current_task
        if task is None:
            return
        if not self.can_enter(context):
            return
        if not self.task_complete(context):
            return
        self._task_i += 1


__all__ = [
    "MissionRunner",
    "Pose2D",
    "Task",
    "default_tasks_path",
    "evaluate_condition",
    "load_map",
    "load_tasks",
]
