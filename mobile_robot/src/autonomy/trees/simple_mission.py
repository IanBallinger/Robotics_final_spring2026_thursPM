"""A tiny py_trees behaviour tree for scaffolding & tests.

The idea is to model the overall task as phases:

  LOAD_TRAY -> DRIVE_TO_ZONE -> UNLOAD_TRAY -> DONE

A top-level Selector chooses which phase subtree is eligible based on a
blackboard key ("mission_phase"). Changing that key *externally* (operator,
planner, safety layer) will immediately switch which subtree gets ticked.
"""

from __future__ import annotations

import operator

import py_trees

from autonomy.behaviors.tick_counter import TickCounter


PHASE_LOAD = "LOAD_TRAY"
PHASE_DRIVE = "DRIVE_TO_ZONE"
PHASE_UNLOAD = "UNLOAD_TRAY"
PHASE_DONE = "DONE"


def _phase_gate(name: str, expected_phase: str) -> py_trees.behaviour.Behaviour:
    """Gate execution of a subtree based on the blackboard mission phase."""

    return py_trees.behaviours.CheckBlackboardVariableValue(
        name=name,
        check=py_trees.common.ComparisonExpression(
            variable="mission_phase",
            value=expected_phase,
            operator=operator.eq,
        ),
    )


def create_tree(*, auto_advance: bool = True) -> py_trees.trees.BehaviourTree:
    """Create a simple mission tree.

    Args:
        auto_advance: when True, each phase sets the next phase on SUCCESS.
            When False, phases never advance automatically (useful to test
            manual switching).
    """

    # --- Phase: Load tray -------------------------------------------------
    load_sequence = py_trees.composites.Sequence(name="LoadTray", memory=False)
    load_sequence.add_children(
        [
            _phase_gate("IsLoadPhase?", PHASE_LOAD),
            TickCounter("ApproachTray", ticks_to_succeed=2),
            TickCounter("PickTray", ticks_to_succeed=2),
        ]
    )
    if auto_advance:
        load_sequence.add_children(
            [
                py_trees.behaviours.SetBlackboardVariable(
                    name="SetPhase=DRIVE",
                    variable_name="mission_phase",
                    variable_value=PHASE_DRIVE,
                    overwrite=True,
                ),
                # Prevent the phase subtree from returning SUCCESS (which would make
                # the root Selector return SUCCESS and end the mission early).
                py_trees.behaviours.Running(name="YieldToNextPhase"),
            ]
        )

    # --- Phase: Drive -----------------------------------------------------
    drive_sequence = py_trees.composites.Sequence(name="DriveToZone", memory=False)
    drive_sequence.add_children(
        [
            _phase_gate("IsDrivePhase?", PHASE_DRIVE),
            TickCounter("Navigate", ticks_to_succeed=3),
        ]
    )
    if auto_advance:
        drive_sequence.add_children(
            [
                py_trees.behaviours.SetBlackboardVariable(
                    name="SetPhase=UNLOAD",
                    variable_name="mission_phase",
                    variable_value=PHASE_UNLOAD,
                    overwrite=True,
                ),
                py_trees.behaviours.Running(name="YieldToNextPhase"),
            ]
        )

    # --- Phase: Unload ----------------------------------------------------
    unload_sequence = py_trees.composites.Sequence(name="UnloadTray", memory=False)
    unload_sequence.add_children(
        [
            _phase_gate("IsUnloadPhase?", PHASE_UNLOAD),
            TickCounter("ApproachDropoff", ticks_to_succeed=2),
            TickCounter("ReleaseTray", ticks_to_succeed=1),
        ]
    )
    if auto_advance:
        unload_sequence.add_children(
            [
                py_trees.behaviours.SetBlackboardVariable(
                    name="SetPhase=DONE",
                    variable_name="mission_phase",
                    variable_value=PHASE_DONE,
                    overwrite=True,
                ),
                py_trees.behaviours.Running(name="YieldToNextPhase"),
            ]
        )

    # --- Done -------------------------------------------------------------
    done_sequence = py_trees.composites.Sequence(name="Done", memory=False)
    done_sequence.add_children(
        [
            _phase_gate("IsDonePhase?", PHASE_DONE),
            py_trees.behaviours.Success(name="MissionComplete"),
        ]
    )

    # --- Root: select which phase subtree runs ----------------------------
    root = py_trees.composites.Selector(name="MissionSelector", memory=False)
    root.add_children([load_sequence, drive_sequence, unload_sequence, done_sequence])

    tree = py_trees.trees.BehaviourTree(root=root)
    return tree


def initialise_blackboard(*, phase: str = PHASE_LOAD) -> py_trees.blackboard.Blackboard:
    bb = py_trees.blackboard.Blackboard()
    bb.set("mission_phase", phase)
    return bb
