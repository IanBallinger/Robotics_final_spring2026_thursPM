import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MultipleLocator

# 2D grid over the axis-aligned bbox of a closed polygon; only cells whose
# centers fall inside the boundary are valid (cell_inside[i, j]).


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
        self.grid = np.zeros((self.nx, self.ny))

    def is_valid_cell(self, i: int, j: int) -> bool:
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return False
        return bool(self.cell_inside[i, j])

    def world_to_cell(self, position: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Map a world point (m) to lattice indices, or None if outside map or invalid cell."""
        x, y = float(position[0]), float(position[1])
        i = int(np.floor((x - self._xmin) / self.resolution))
        j = int(np.floor((y - self._ymin) / self.resolution))
        if not self.is_valid_cell(i, j):
            return None
        return (i, j)

    def cell_center(self, i: int, j: int) -> Tuple[float, float]:
        if not self.is_valid_cell(i, j):
            raise IndexError(f"cell ({i}, {j}) is not a valid map cell")
        x = self._xmin + (i + 0.5) * self.resolution
        y = self._ymin + (j + 0.5) * self.resolution
        return (x, y)

    def query(self, position: Tuple[float, float]) -> Optional[float]:
        """Value at the cell containing ``position``, or None if not in a valid cell."""
        idx = self.world_to_cell(position)
        if idx is None:
            return None
        i, j = idx
        return float(self.grid[i, j])

    def set_obstacle(self, points: List[Tuple[float, float]], name: str):
        pass

    def set_landmark(self, points: List[Tuple[float, float]], name: str):
        pass

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
        ax.xaxis.set_minor_locator(MultipleLocator(self.resolution))
        ax.yaxis.set_minor_locator(MultipleLocator(self.resolution))
        ax.xaxis.set_major_locator(MultipleLocator(self.resolution))
        ax.yaxis.set_major_locator(MultipleLocator(self.resolution))
        # Major ticks use AutoLocator (e.g. 0.5, 1.0); grid() defaults to major off unless enabled.
        ax.grid(which="major", color="0.35", linewidth=0.55)
        ax.grid(which="minor", color="0.35", linewidth=0.35)
        ax.set_axisbelow(True)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.show()


if __name__ == "__main__":
    m = Map([(0, 0), (2, 0), (2, 0.5), (1, 0.5), (1, 1), (0, 1), (0, 0)], 0.1)
    m.plot()
