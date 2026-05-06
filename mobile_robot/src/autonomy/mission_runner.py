"""Minimal mission/task utilities for waypoint-tracking FSM missions.

Each task specifies a goal pose in the map frame plus optional manipulator and
completion-condition metadata. The hardware runtime is responsible for the
finite state machine and waypoint tracking. This module just loads the config
into small dataclasses.
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
class ArmWaypoint:
    x: float
    y: float


@dataclass(frozen=True)
class Task:
    name: str
    start: Optional[Pose2D]
    goal: Optional[Pose2D]
    desired_elevator_height_m: float = 0.0
    completion_conditions: List[str] | None = None
    arm_waypoints: List[ArmWaypoint] | None = None
    arm_point_dwell_s: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "completion_conditions", list(self.completion_conditions or [])
        )
        object.__setattr__(self, "arm_waypoints", list(self.arm_waypoints or []))


def default_tasks_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent / "config" / "mission_config.yaml"
    )


def default_camera_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent / "config" / "camera_config.yaml"
    )


def default_localization_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "config"
        / "localization_config.yaml"
    )


def _load_pose(raw: Mapping[str, Any]) -> Pose2D:
    pos = raw["position"]
    return Pose2D(
        x=float(pos["x"]),
        y=float(pos["y"]),
        heading=float(raw.get("heading", 0.0)),
    )


def _load_point(raw: Mapping[str, Any]) -> tuple[float, float]:
    return float(raw["x"]), float(raw["y"])


def _load_arm_waypoint(raw: Mapping[str, Any]) -> ArmWaypoint:
    return ArmWaypoint(x=float(raw["x"]), y=float(raw["y"]))


def load_map(path: Path | str) -> Map:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    map_raw = raw.get("map")
    if map_raw is None:
        raise ValueError("mission_config.yaml must define 'map'")

    boundary_raw = map_raw.get("boundary")
    if boundary_raw is None:
        raise ValueError("mission_config.yaml map must define 'boundary'")

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

    for obstacle_raw in map_raw.get("obstacles", []) or []:
        obstacle_boundary = obstacle_raw.get("boundary")
        if obstacle_boundary is None:
            raise ValueError(
                f"map obstacle '{obstacle_raw.get('name', '<unnamed>')}' must define 'boundary'"
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
        raise ValueError("mission_config.yaml must define 'initial_state'")

    load_map(path)

    tasks_raw = raw.get("tasks", [])
    if not tasks_raw:
        raise ValueError("mission_config.yaml must define at least one task")

    tasks: List[Task] = []
    current_start: Optional[Pose2D] = _load_pose(initial_state_raw)
    for row in tasks_raw:
        explicit_goal = row.get("goal")
        if explicit_goal is None:
            raise ValueError(f"task '{row.get('name', '<unnamed>')}' must define a goal pose")
        goal = _load_pose(explicit_goal)
        task = Task(
            name=str(row["name"]),
            start=current_start,
            goal=goal,
            desired_elevator_height_m=float(row.get("desired_elevator_height_m", 0.0)),
            completion_conditions=[
                str(x) for x in row.get("completion_conditions", [])
            ],
            arm_waypoints=[
                _load_arm_waypoint(point) for point in row.get("arm_waypoints", [])
            ],
            arm_point_dwell_s=float(row.get("arm_point_dwell_s", 0.5)),
        )
        tasks.append(task)
        current_start = goal if goal is not None else current_start

    return tasks


def evaluate_condition(expression: str, context: Mapping[str, Any]) -> bool:
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
    """Very small sequential task iterator kept for compatibility."""

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

    def current_start_pose(self) -> Optional[Pose2D]:
        task = self.current_task
        return None if task is None else task.start

    def current_goal_pose(self) -> Optional[Pose2D]:
        task = self.current_task
        return None if task is None else task.goal

    def task_complete(self, context: Mapping[str, Any]) -> bool:
        task = self.current_task
        if task is None:
            return False
        if not task.completion_conditions:
            return True
        merged: Dict[str, Any] = dict(context)
        merged.setdefault("current_task", task.name)
        merged.setdefault("task_index", self._task_i)
        return all(
            evaluate_condition(expr, merged) for expr in task.completion_conditions
        )

    def step(self, context: Mapping[str, Any]) -> None:
        if self.is_complete():
            return
        if self.task_complete(context):
            self._task_i += 1


__all__ = [
    "ArmWaypoint",
    "MissionRunner",
    "Pose2D",
    "Task",
    "default_camera_path",
    "default_localization_path",
    "default_tasks_path",
    "evaluate_condition",
    "load_map",
    "load_tasks",
]
