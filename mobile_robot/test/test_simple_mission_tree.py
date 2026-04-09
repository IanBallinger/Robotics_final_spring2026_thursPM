import os
import sys
import unittest

import py_trees

# Allow `from autonomy...` imports when tests are run from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "mobile_robot", "src")
sys.path.insert(0, SRC_DIR)

from autonomy.trees.simple_mission import (  # noqa: E402
    PHASE_DONE,
    PHASE_DRIVE,
    PHASE_LOAD,
    PHASE_UNLOAD,
    create_tree,
    initialise_blackboard,
)


class TestSimpleMissionTree(unittest.TestCase):
    def setUp(self):
        # Ensure a blackboard exists and is initialised for each test.
        self.bb = initialise_blackboard(phase=PHASE_LOAD)

    def test_manual_phase_switching_changes_active_subtree(self):
        tree = create_tree(auto_advance=False)
        tree.setup(timeout=2.0)

        # Tick once: should be in load subtree.
        tree.tick()
        self.assertEqual(self.bb.get("mission_phase"), PHASE_LOAD)

        root = tree.root
        load, drive, unload, done = root.children
        self.assertEqual(load.status, py_trees.common.Status.RUNNING)

        # Force switch to unload while load is still RUNNING.
        self.bb.set("mission_phase", PHASE_UNLOAD)
        tree.tick()

        # With the selector, the load sequence should now fail its phase gate,
        # and the unload sequence should take over.
        self.assertEqual(load.status, py_trees.common.Status.FAILURE)
        self.assertEqual(unload.status, py_trees.common.Status.RUNNING)

        # Force switch to drive.
        self.bb.set("mission_phase", PHASE_DRIVE)
        tree.tick()
        self.assertEqual(drive.status, py_trees.common.Status.RUNNING)

    def test_auto_advance_runs_full_mission_to_done(self):
        self.bb.set("mission_phase", PHASE_LOAD)
        tree = create_tree(auto_advance=True)
        tree.setup(timeout=2.0)

        # Tick until the root reports SUCCESS or a safety max ticks is reached.
        for _ in range(50):
            tree.tick()
            if tree.root.status == py_trees.common.Status.SUCCESS:
                break

        self.assertEqual(self.bb.get("mission_phase"), PHASE_DONE)
        self.assertEqual(tree.root.status, py_trees.common.Status.SUCCESS)


if __name__ == "__main__":
    unittest.main()
