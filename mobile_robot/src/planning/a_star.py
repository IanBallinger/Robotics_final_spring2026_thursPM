from __future__ import annotations

import heapq
import os
import sys
from typing import Dict, List, Tuple, Union

import numpy as np

import math



try:
    from ..localization.map import Map, Obstacle, Landmark
except ImportError:
    # Allow direct execution: `python mobile_robot/src/planning/a_star.py`
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    from localization.map import Map, Obstacle, Landmark


GridPoint = Tuple[float, float]
Cell = Tuple[int, int]
Command = Union[str, float]


class AStar:
    def __init__(self, map: Map):
        self.map = map
        self.centers_inside = {
            tuple(center) for center in map.get_centers_inside().tolist()
        }
        self.clear()

    def clear(self):
        self.open_set: List[Tuple[float, GridPoint]] = []
        self.closed_set: set[GridPoint] = set()
        self.came_from: Dict[GridPoint, GridPoint] = {}
        self.g_score: Dict[GridPoint, float] = {}
        self.f_score: Dict[GridPoint, float] = {}

    def __heuristic(self, position: GridPoint, goal: GridPoint) -> float:
        return float(np.linalg.norm(np.array(position) - np.array(goal)))

    def __reconstruct_path(self, current: GridPoint) -> List[GridPoint]:            
        path = [current]
        while current in self.came_from:
            prev = current
            current = self.came_from[prev]
            path.append(current)
        path.reverse()
        return path

    def __cell_to_point(self, cell: Cell) -> GridPoint:
        return tuple(self.map.cell_center(*cell))

    def __get_neighbors(self, position: GridPoint) -> List[GridPoint]:
        cell = self.map.world_to_cell(position)
        if cell is None:
            return []

        neighbors: List[GridPoint] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                # no diagonal neighbors
                if di == 0 and dj == 0:
                    continue
                if di == -1 and dj == -1:
                    continue
                if di == 1 and dj == 1:
                    continue
                if di == -1 and dj == 1:
                    continue
                if di == 1 and dj == -1:
                    continue
                ni, nj = cell[0] + di, cell[1] + dj
                if self.map.is_open_cell(ni, nj):
                    neighbors.append(self.__cell_to_point((ni, nj)))
        return neighbors

    def generate_plan(self, start: GridPoint, goal: GridPoint) -> List[GridPoint]:
        self.clear()

        start_cell = self.map.world_to_cell(start)
        goal_cell = self.map.world_to_cell(goal)
        if start_cell is None:
            raise ValueError(f"start {start} is not in a valid map cell")
        if goal_cell is None:
            raise ValueError(f"goal {goal} is not in a valid map cell")

        start = self.__cell_to_point(start_cell)
        goal = self.__cell_to_point(goal_cell)

        self.g_score[start] = 0.0
        start_f = self.__heuristic(start, goal)
        self.f_score[start] = start_f
        heapq.heappush(self.open_set, (start_f, start))

        while self.open_set:
            _, current = heapq.heappop(self.open_set)
            if current in self.closed_set:
                continue

            if current == goal:
                return self.__reconstruct_path(current)

            self.closed_set.add(current)

            for neighbor in self.__get_neighbors(current):
                if neighbor in self.closed_set:
                    continue

                tentative_g_score = self.g_score[current] + self.__heuristic(
                    current, neighbor
                )
                if tentative_g_score < self.g_score.get(neighbor, np.inf):
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    neighbor_f = tentative_g_score + self.__heuristic(neighbor, goal)
                    self.f_score[neighbor] = neighbor_f
                    heapq.heappush(self.open_set, (neighbor_f, neighbor))

        return []
    
    def get_guidance_seq(
        self,
        start: GridPoint,
        start_hdg: float,
        goal: GridPoint,
        goal_hdg: float,
        t: float,
        m: float,
    ) -> List[Tuple[float, float]]:
        """
        Return a sequence of (forward_speed, turn_rate) commands.

        Assumptions:
        - each command is applied for exactly t seconds
        - moving forward one grid cell of length m uses v = m / t
        - turns are in-place with v = 0
        - first and last turns may be arbitrary angles
        - interior turns are determined by the 4-connected A* path
        """

        if t <= 0:
            raise ValueError("t must be positive")
        if m <= 0:
            raise ValueError("m must be positive")

        plan = self.generate_plan(start, goal)
        if len(plan) < 2:
            return []

        def wrap_to_pi(angle: float) -> float:
            return (angle + np.pi) % (2 * np.pi) - np.pi

        def step_heading(p0: GridPoint, p1: GridPoint) -> float:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            return float(np.arctan2(dy, dx))

        v_fwd = m / t
        vel_seq: List[Tuple[float, float]] = []

        path_hdgs = [step_heading(plan[i], plan[i + 1]) for i in range(len(plan) - 1)]

        # Initial heading alignment: arbitrary angle
        delta = wrap_to_pi(path_hdgs[0] - start_hdg)
        if not np.isclose(delta, 0.0):
            vel_seq.append((0.0, delta / t))

        # Forward motions + interior 90-deg turns
        vel_seq.append((v_fwd, 0.0))
        for prev_hdg, next_hdg in zip(path_hdgs[:-1], path_hdgs[1:]):
            delta = wrap_to_pi(next_hdg - prev_hdg)

            if np.isclose(delta, 0.0):
                vel_seq.append((v_fwd, 0.0))
            elif np.isclose(abs(delta), np.pi / 2):
                vel_seq.append((0.0, delta / t))
                vel_seq.append((v_fwd, 0.0))
            elif np.isclose(abs(delta), np.pi):
                raise ValueError(
                    f"Path contains a 180-degree reversal between segments: "
                    f"{prev_hdg} -> {next_hdg}"
                )
            else:
                raise ValueError(
                    f"Unexpected non-grid turn angle in path: {delta}"
                )

        # Final heading alignment: arbitrary angle
        delta = wrap_to_pi(goal_hdg - path_hdgs[-1])
        if not np.isclose(delta, 0.0):
            vel_seq.append((0.0, delta / t))

        return vel_seq


if __name__ == "__main__":
    map = Map([(0, 0), (2, 0), (2, 0.5), (1, 0.5), (1, 1), (0, 1), (0, 0)], 0.1)
    table = Obstacle([(1, 0.5), (1, 1), (0, 1), (0, 0.5)], "table")
    map.add_obstacle(table)
    shelf = Obstacle([(0.7, 0.4), (1, 0.4), (1, 0.7), (0.7, 0.7)], "shelf")
    map.add_obstacle(shelf)
    at1 = Landmark((0.5, 0.5), "at1")
    map.add_landmark(at1)

    astar = AStar(map)
    start = (0, 0)
    goal = (1.5, 0.3)
    plan = astar.generate_plan(start, goal)
    print(plan)

    map.set_plan(plan)
    print(astar.get_guidance_seq(start,0.2, goal,0.0, 1,0.5))
    map.plot()
