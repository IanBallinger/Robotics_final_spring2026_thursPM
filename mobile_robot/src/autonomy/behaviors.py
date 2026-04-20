"""Custom autonomy behaviours."""

from __future__ import annotations

import py_trees


class TickCounter(py_trees.behaviour.Behaviour):
    """Return RUNNING until ticked enough times, then return SUCCESS.

    This is a lightweight placeholder behaviour for mission-tree wiring.
    The counter persists across repeated ticks and resets only when the
    behaviour is invalidated/interrupted.
    """

    def __init__(self, name: str, ticks_to_succeed: int):
        super().__init__(name=name)
        if ticks_to_succeed < 1:
            raise ValueError("ticks_to_succeed must be >= 1")
        self.ticks_to_succeed = int(ticks_to_succeed)
        self.ticks = 0

    def update(self) -> py_trees.common.Status:
        self.ticks += 1
        if self.ticks >= self.ticks_to_succeed:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            self.ticks = 0


__all__ = ["TickCounter"]
