from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np

from localization.map import Landmark, Map, Obstacle


GridPoint = Tuple[float, float]
Cell = Tuple[int, int]
Command = Union[str, float]


@dataclass(frozen=True)
class Waypoint:
    """A map position (``GridPoint``) with desired body heading in radians."""

    xy: GridPoint
    heading: float


def waypoints_from_polyline(
    path: List[GridPoint],
    *,
    end_heading: float | None = None,
) -> List[Waypoint]:
    """
    One ``Waypoint`` per vertex: heading is the tangent toward the next vertex.

    The final waypoint uses ``end_heading`` when provided; otherwise it uses the
    last segment direction (same as the second-to-last vertex).
    """
    if not path:
        return []
    if len(path) == 1:
        h = 0.0 if end_heading is None else end_heading
        return [Waypoint(path[0], h)]

    out: List[Waypoint] = []
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        h = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        out.append(Waypoint(p0, h))

    p_prev, p_last = path[-2], path[-1]
    h_last = (
        float(end_heading)
        if end_heading is not None
        else float(np.arctan2(p_last[1] - p_prev[1], p_last[0] - p_prev[0]))
    )
    out.append(Waypoint(p_last, h_last))
    return out


class Planner(ABC):
    @abstractmethod
    def generate_plan(self, start: GridPoint, goal: GridPoint) -> List[GridPoint]:
        pass


class AStar(Planner):
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


class WaypointSequence:
    def __init__(
        self,
        start: Waypoint,
        goal: Waypoint,
        planner: Planner,
        capture_radius: float,
    ):
        self.start = start
        self.goal = goal
        self.planner = planner
        self.capture_radius = capture_radius
        self.plan = self.planner.generate_plan(self.start.xy, self.goal.xy)

    def generate_plan(self) -> List[GridPoint]:
        pass


__all__ = [
    "AStar",
    "Planner",
    "Waypoint",
    "WaypointSequence",
    "GridPoint",
    "Cell",
    "Command",
    "waypoints_from_polyline",
]


if __name__ == "__main__":
    map = Map([(0, 0), (2, 0), (2, 0.5), (1, 0.5), (1, 1), (0, 1), (0, 0)], 0.1)
    table = Obstacle([(1, 0.5), (1, 1), (0, 1), (0, 0.5)], "table")
    map.add_obstacle(table)
    shelf = Obstacle([(0.7, 0.4), (1, 0.4), (1, 0.7), (0.7, 0.7)], "shelf")
    map.add_obstacle(shelf)
    at1 = Landmark((0.5, 0.5), "at1", id="at1")
    map.add_landmark(at1)

    astar = AStar(map)
    start = (0, 0)
    goal = (1.5, 0.3)
    plan = astar.generate_plan(start, goal)
    print(plan)

    map.set_plan(plan)
    print(waypoints_from_polyline(plan, end_heading=0.0))
    map.plot()
