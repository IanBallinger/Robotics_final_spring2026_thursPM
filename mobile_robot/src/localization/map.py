from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MultipleLocator

# 2D grid over the axis-aligned bbox of a closed polygon; only cells whose
# centers fall inside the boundary are valid (cell_inside[i, j]).

GridPoint = Tuple[float, float]
Cell = Tuple[int, int]


def _closed_polygon_vertices(
    boundary: List[Tuple[float, float]],
) -> np.ndarray:
    """Return Nx2 float array with first vertex repeated at end if needed."""
    v = np.asarray(boundary, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("boundary must be a list of (x, y) points in meters")
    if len(v) < 3:
        raise ValueError("boundary needs at least 3 corners")
    if not np.allclose(v[0], v[-1]):
        v = np.vstack([v, v[0:1]])
    return v


class Obstacle:
    def __init__(self, points: List[Tuple[float, float]], name: str):
        self.points = np.asarray(points, dtype=float)
        self.boundary = _closed_polygon_vertices(self.points)
        self.name = name
        self._path = MplPath(self.boundary)

    def contains_point(self, position: Tuple[float, float]) -> bool:
        """True if ``position`` (meters) lies inside the obstacle polygon boundary."""
        xy = np.asarray(position, dtype=float).reshape(1, 2)
        return bool(self._path.contains_points(xy)[0])

    def contains_cell(self, grid_map: Map, i: int, j: int) -> bool:
        """True if the center of map cell (i, j) lies inside the obstacle."""
        if not grid_map.is_valid_cell(i, j):
            return False
        return self.contains_point(grid_map.cell_center(i, j))

    def covers_cell_centers(self, grid_map: Map) -> np.ndarray:
        """Bool array (nx, ny): True where the map cell center lies inside this obstacle."""
        inside = self._path.contains_points(grid_map._cell_centers).reshape(
            grid_map.nx, grid_map.ny
        )
        return inside & grid_map.cell_inside


class Landmark:
    def __init__(self, point: Tuple[float, float], name: str, heading: float = 0.0):
        self.point = point
        self.name = name
        self.heading = float(heading)

    def _get_delta_to_landmark(
        self, position: Tuple[float, float]
    ) -> Tuple[float, float]:
        return (self.point[0] - position[0], self.point[1] - position[1])

    def get_distance(self, position: Tuple[float, float]) -> Tuple[float, float]:
        delta = self._get_delta_to_landmark(position)
        return np.linalg.norm(delta)

    def get_angle(self, position: Tuple[float, float]) -> float:
        delta = self._get_delta_to_landmark(position)
        return np.arctan2(delta[1], delta[0])


class Map:
    """
    Occupancy-style grid aligned to x/y in meters. Lattice covers the bounding
    box of ``boundary``; cell (i, j) is valid iff its center lies inside the
    polygon (CCW closed boundary in meters). Cell (i, j) spans
    [xmin + i * r, xmin + (i + 1) * r) × [ymin + j * r, ymin + (j + 1) * r).
    """

    def __init__(self, boundary: List[Tuple[float, float]], resolution: float):
        if resolution <= 0:
            raise ValueError("resolution must be positive")

        self.boundary = list(boundary)
        self.resolution = float(resolution)
        self._vertices = _closed_polygon_vertices(boundary)

        xs = self._vertices[:, 0]
        ys = self._vertices[:, 1]
        self._xmin = float(np.min(xs))
        self._xmax = float(np.max(xs))
        self._ymin = float(np.min(ys))
        self._ymax = float(np.max(ys))

        self.nx = int(np.ceil((self._xmax - self._xmin) / self.resolution))
        self.ny = int(np.ceil((self._ymax - self._ymin) / self.resolution))
        if self.nx == 0 or self.ny == 0:
            raise ValueError("degenerate bounding box for boundary")

        cx = self._xmin + (np.arange(self.nx) + 0.5) * self.resolution
        cy = self._ymin + (np.arange(self.ny) + 0.5) * self.resolution
        xx, yy = np.meshgrid(cx, cy, indexing="ij")
        centers = np.column_stack([xx.ravel(), yy.ravel()])

        path = MplPath(self._vertices)
        inside = path.contains_points(centers).reshape(self.nx, self.ny)

        self.cell_inside = inside
        self._cell_centers = centers
        self.grid = np.zeros((self.nx, self.ny))
        self.obstacle_mask = np.zeros((self.nx, self.ny), dtype=bool)

        self.centers_inside = self.get_centers_inside()

        self.obstacles: List[Obstacle] = []
        self.landmarks = []

        self.current_position: GridPoint = (0, 0)

    def get_centers_inside(self) -> np.ndarray:
        xs = self._cell_centers.reshape(self.nx, self.ny, 2)[:, :, 0][self.cell_inside]
        ys = self._cell_centers.reshape(self.nx, self.ny, 2)[:, :, 1][self.cell_inside]
        return np.column_stack((xs, ys))

    def is_valid_cell(self, i: int, j: int) -> bool:
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return False
        return bool(self.cell_inside[i, j])

    def is_open_cell(self, i: int, j: int) -> bool:
        return self.is_valid_cell(i, j) and not self.obstacle_mask[i, j]

    def cell_has_obstacle(self, i: int, j: int) -> bool:
        """True iff lattice cell (i, j) is on the map and marked occupied by an obstacle."""
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return False
        return bool(self.obstacle_mask[i, j])

    def world_to_cell(self, position: GridPoint) -> Optional[Cell]:
        """Map a world point (m) to lattice indices, or None if outside map or invalid cell."""
        x, y = float(position[0]), float(position[1])
        i = int(np.floor((x - self._xmin) / self.resolution))
        j = int(np.floor((y - self._ymin) / self.resolution))
        if not self.is_valid_cell(i, j):
            return None
        return (i, j)

    def cell_center(self, i: int, j: int) -> GridPoint:
        if not self.is_valid_cell(i, j):
            raise IndexError(f"cell ({i}, {j}) is not a valid map cell")
        x = self._xmin + (i + 0.5) * self.resolution
        y = self._ymin + (j + 0.5) * self.resolution
        return (x, y)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)
        self.obstacle_mask |= obstacle.covers_cell_centers(self)

    def add_landmark(self, landmark: Landmark):
        self.landmarks.append(landmark)

    def set_plan(self, plan: List[GridPoint]) -> None:
        self.plan = plan
        for i in range(len(plan) - 1):
            cell_i = self.world_to_cell(plan[i])
            cell_j = self.world_to_cell(plan[i + 1])
            if cell_i is not None and cell_j is not None:
                self.grid[cell_i[0], cell_i[1]] = 1
                self.grid[cell_j[0], cell_j[1]] = 1

    def plot(self):
        _, ax = plt.subplots()
        xmax = self._xmin + self.nx * self.resolution
        ymax = self._ymin + self.ny * self.resolution
        display = np.ma.masked_where(~self.cell_inside, self.grid)
        ax.imshow(
            display.T,
            origin="lower",
            extent=(self._xmin, xmax, self._ymin, ymax),
            aspect="equal",
            interpolation="nearest",
        )
        ax.plot(
            self._vertices[:, 0],
            self._vertices[:, 1],
            color="k",
            linewidth=1.2,
        )

        # # plot current position
        # ax.plot(
        #     self.current_position[0],
        #     self.current_position[1],
        #     color="green",
        #     marker="o",
        # )
        # current_cell = np.array(self.world_to_cell(self.current_position))
        # if current_cell is not None:
        #     ax.imshow(
        #         current_cell.T,
        #         origin="lower",
        #     )

        # plot obstacles
        if np.any(self.obstacle_mask):
            obs_display = np.ma.masked_where(
                ~self.obstacle_mask, np.ones((self.nx, self.ny))
            )
            ax.imshow(
                obs_display.T,
                origin="lower",
                extent=(self._xmin, xmax, self._ymin, ymax),
                aspect="equal",
                interpolation="nearest",
                cmap="Reds",
                vmin=0,
                vmax=1,
                alpha=0.45,
            )
        for obstacle in self.obstacles:
            ax.plot(
                obstacle.boundary[:, 0],
                obstacle.boundary[:, 1],
                color="darkred",
                linewidth=1.0,
            )

        # plot landmarks
        for landmark in self.landmarks:
            ax.plot(
                landmark.point[0],
                landmark.point[1],
                color="blue",
                marker="o",
            )
            ax.arrow(
                landmark.point[0],
                landmark.point[1],
                0.08 * np.cos(landmark.heading),
                0.08 * np.sin(landmark.heading),
                head_width=0.03,
                head_length=0.03,
                fc="blue",
                ec="blue",
                length_includes_head=True,
            )
            ax.text(
                landmark.point[0] + 0.02,
                landmark.point[1] + 0.02,
                str(landmark.name),
                fontsize=8,
                color="blue",
            )

        ax.xaxis.set_minor_locator(MultipleLocator(self.resolution))
        ax.yaxis.set_minor_locator(MultipleLocator(self.resolution))
        ax.xaxis.set_major_locator(MultipleLocator(self.resolution))
        ax.yaxis.set_major_locator(MultipleLocator(self.resolution))
        ax.grid(which="major", color="0.35", linewidth=0.55)
        ax.grid(which="minor", color="0.35", linewidth=0.35)
        ax.set_axisbelow(True)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.show()


if __name__ == "__main__":
    m = Map([(0, 0), (2, 0), (2, 0.5), (1, 0.5), (1, 1), (0, 1), (0, 0)], 0.1)
    table = Obstacle([(1, 0.5), (1, 1), (0, 1), (0, 0.5)], "table")
    m.add_obstacle(table)
    shelf = Obstacle([(0.7, 0.4), (1, 0.4), (1, 0.7), (0.7, 0.7)], "shelf")
    m.add_obstacle(shelf)
    at1 = Landmark((0.5, 0.5), "at1", heading=0.0)
    m.add_landmark(at1)

    m.plot()
