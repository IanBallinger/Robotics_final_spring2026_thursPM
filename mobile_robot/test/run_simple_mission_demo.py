#!/usr/bin/env python
"""Demo: tick a tiny mission tree and manually switch phases.

This script is intentionally "no ROS" and uses placeholder actions.

It demonstrates two ways to switch between behaviours being executed:

1) Automatic phase advancement (each subtree sets the next phase on success)
2) Manual override (operator/safety sets mission_phase mid-execution)
"""

from __future__ import annotations

import os
import sys
import time

import py_trees

# Allow `from autonomy...` imports when running from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
sys.path.insert(0, SRC_DIR)

from autonomy.trees.simple_mission import (  # noqa: E402
    PHASE_DRIVE,
    PHASE_LOAD,
    PHASE_UNLOAD,
    create_tree,
    initialise_blackboard,
)


def tick_and_print(tree: py_trees.trees.BehaviourTree, *, tick_index: int):
    tree.tick()
    print(f"\n--- Tick {tick_index} ---")
    print(py_trees.display.unicode_tree(tree.root, show_status=True))


def main():
    bb = initialise_blackboard(phase=PHASE_LOAD)

    print("\n=== Demo A: auto-advance phases ===")
    tree = create_tree(auto_advance=True)
    tree.setup(timeout=2.0)

    for i in range(1, 15):
        tick_and_print(tree, tick_index=i)
        print(f"mission_phase = {bb.get('mission_phase')}")
        if bb.get("mission_phase") == "DONE":
            break
        time.sleep(1)

    print("\n=== Demo B: manual phase switching (override mid-run) ===")
    bb.set("mission_phase", PHASE_LOAD)
    tree = create_tree(auto_advance=False)
    tree.setup(timeout=2.0)

    for i in range(1, 8):
        if i == 3:
            # Force a switch while the LOAD subtree is still RUNNING.
            print("\n[override] setting mission_phase = UNLOAD_TRAY")
            bb.set("mission_phase", PHASE_UNLOAD)
        elif i == 6:
            print("\n[override] setting mission_phase = DRIVE_TO_ZONE")
            bb.set("mission_phase", PHASE_DRIVE)

        tick_and_print(tree, tick_index=i)
        print(f"mission_phase = {bb.get('mission_phase')}")
        time.sleep(1)


if __name__ == "__main__":
    main()
