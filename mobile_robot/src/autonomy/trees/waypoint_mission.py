"""A simple py_trees mission runner built from ``config/mission_config.yaml``.

Each task becomes a subtree with:
- a gate on the active task name
- a check that all enter conditions are true
- a small sequence of placeholder waypoint behaviours
- a check that all completion conditions are true
- a transition to the next task (or DONE)

The waypoint behaviours are intentionally simple stand-ins while wiring up the
mission tree.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import py_trees

from ..behaviors import TickCounter
from ..mission_runner import Task, default_tasks_path, evaluate_condition, load_tasks


MISSION_DONE = "DONE"


class ConditionList(py_trees.behaviour.Behaviour):
    """Return SUCCESS when all expressions evaluate to True."""

    def __init__(self, name: str, expressions: Sequence[str]):
        super().__init__(name=name)
        self.expressions = list(expressions)
        self.blackboard = py_trees.blackboard.Blackboard()

    def update(self) -> py_trees.common.Status:
        context = {
            key.lstrip("/"): value for key, value in dict(self.blackboard.storage).items()
        }
        for expression in self.expressions:
            if not evaluate_condition(expression, context):
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


def _active_task_gate(task_name: str) -> py_trees.behaviour.Behaviour:
    return py_trees.behaviours.CheckBlackboardVariableValue(
        name=f"IsActiveTask={task_name}?",
        check=py_trees.common.ComparisonExpression(
            variable="current_task",
            value=task_name,
            operator=lambda a, b: a == b,
        ),
    )


def _advance_to(next_task: str) -> py_trees.behaviour.Behaviour:
    return py_trees.behaviours.SetBlackboardVariable(
        name=f"SetTask={next_task}",
        variable_name="current_task",
        variable_value=next_task,
        overwrite=True,
    )


def _task_subtree(task: Task, next_task: str) -> py_trees.behaviour.Behaviour:
    seq = py_trees.composites.Sequence(name=task.name, memory=False)
    seq.add_children(
        [
            _active_task_gate(task.name),
            ConditionList(name=f"CanEnter={task.name}", expressions=task.enter_conditions),
            TickCounter(name=f"DriveToStart:{task.name}", ticks_to_succeed=1),
            TickCounter(name=f"WaypointA:{task.name}", ticks_to_succeed=2),
            TickCounter(name=f"WaypointB:{task.name}", ticks_to_succeed=2),
            TickCounter(name=f"AlignAtGoal:{task.name}", ticks_to_succeed=1),
            ConditionList(
                name=f"TaskComplete={task.name}",
                expressions=task.completion_conditions,
            ),
            _advance_to(next_task),
            py_trees.behaviours.Running(name="YieldToNextTask"),
        ]
    )
    return seq


def create_tree(tasks: Sequence[Task] | None = None) -> py_trees.trees.BehaviourTree:
    if tasks is None:
        tasks = load_tasks(default_tasks_path())
    tasks = list(tasks)
    if not tasks:
        raise ValueError("at least one task is required")

    root = py_trees.composites.Selector(name="WaypointMission", memory=False)
    for i, task in enumerate(tasks):
        next_task = tasks[i + 1].name if i + 1 < len(tasks) else MISSION_DONE
        root.add_child(_task_subtree(task, next_task))

    done = py_trees.composites.Sequence(name="Done", memory=False)
    done.add_children(
        [
            _active_task_gate(MISSION_DONE),
            py_trees.behaviours.Success(name="MissionComplete"),
        ]
    )
    root.add_child(done)
    return py_trees.trees.BehaviourTree(root=root)


def initialise_blackboard(initial_task: str, state: Mapping[str, Any] | None = None):
    bb = py_trees.blackboard.Blackboard()
    bb.set("current_task", initial_task)
    for key, value in (state or {}).items():
        bb.set(key, value)
    return bb


__all__ = ["MISSION_DONE", "ConditionList", "create_tree", "initialise_blackboard"]
