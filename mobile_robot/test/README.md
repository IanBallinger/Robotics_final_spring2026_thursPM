# py_trees scaffolding

This folder contains small behaviour tree examples + tests to learn how `py_trees`
works before connecting to real robot code.

## Files

- `run_simple_mission_demo.py` - ticks a simple mission tree and prints the tree
  with statuses. Demonstrates switching between subtrees by changing a blackboard
  variable (`mission_phase`).
- `test_simple_mission_tree.py` - `unittest` tests for phase switching and full
  mission progression.

## Run the demo

From repo root:

```bash
python mobile_robot/test/run_simple_mission_demo.py
```

## Run tests

```bash
python -m unittest mobile_robot.test.test_simple_mission_tree
```
