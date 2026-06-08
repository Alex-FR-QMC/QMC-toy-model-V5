# -*- coding: utf-8 -*-
"""
mcq_v5.grid — Discrete support for fields, fluxes, metrics, operators.

V5-0a minimal implementation. No staggered layout. Cartesian grids only.
No hardcoding of 5x5x5 or 7x7x7. Boundary policy = zero-flux (Neumann).

Vocabulary is empirical/contractual. No phenomenology.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
import numpy as np


@dataclass(frozen=True, slots=True)
class Grid:
    """Discrete cartesian grid.

    Attributes
    ----------
    shape : tuple[int, ...]
        Shape of the grid (e.g. (5, 5, 5) for 6d compatibility, (7, 7, 7)
        for V5 exploratory).
    dx : tuple[float, ...]
        Spacing per axis. Always stored as a tuple matching dim.
    """

    shape: tuple[int, ...]
    dx: tuple[float, ...]

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def n_cells(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def cell_centers(self) -> tuple[np.ndarray, ...]:
        """Per-axis coordinates of cell centers.

        Returns one 1D array per axis. Convention: coords[i] = (i + 0.5) * dx
        (cell-centered).
        """
        return tuple(
            (np.arange(self.shape[a]) + 0.5) * self.dx[a]
            for a in range(self.dim)
        )

    def faces(self, axis: int) -> np.ndarray:
        """Per-axis coordinates of inter-cell faces.

        For an axis of N cells, returns N-1 face coordinates (interior faces only).
        Convention: face at boundary between cell i and i+1 is at (i + 1) * dx[axis].
        """
        if not 0 <= axis < self.dim:
            raise ValueError(f"axis {axis} out of range for dim {self.dim}")
        n = self.shape[axis]
        return (np.arange(n - 1) + 1.0) * self.dx[axis]

    def neighbors(self, index: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        """Yield in-bounds neighbor indices in all axes (forward and backward).

        Boundary cells yield fewer neighbors.
        """
        if len(index) != self.dim:
            raise ValueError(
                f"index has {len(index)} components, grid has {self.dim}"
            )
        for axis in range(self.dim):
            for delta in (-1, +1):
                new_idx = list(index)
                new_idx[axis] += delta
                if 0 <= new_idx[axis] < self.shape[axis]:
                    yield tuple(new_idx)

    def boundary_policy(self) -> str:
        """Return the boundary policy.

        V5-0a: zero-flux (Neumann). No other policy implemented at this stage.
        """
        return "zero_flux_neumann"

    def __post_init__(self):
        # Validation: shape and dx must have same length
        if len(self.shape) != len(self.dx):
            raise ValueError(
                f"shape {self.shape} and dx {self.dx} must have same dim"
            )
        # All shape components must be positive integers
        for s in self.shape:
            if not isinstance(s, (int, np.integer)) or s <= 0:
                raise ValueError(f"shape components must be positive int, got {s}")
        # All dx components must be positive floats
        for d in self.dx:
            if not isinstance(d, (int, float, np.floating)) or d <= 0:
                raise ValueError(f"dx components must be positive float, got {d}")


def make_grid(shape: tuple[int, ...], dx: float | tuple[float, ...] = 1.0) -> Grid:
    """Convenience constructor that broadcasts a scalar dx to a tuple."""
    if isinstance(dx, (int, float, np.floating)):
        dx_tuple = tuple(float(dx) for _ in shape)
    else:
        dx_tuple = tuple(float(d) for d in dx)
    return Grid(shape=tuple(int(s) for s in shape), dx=dx_tuple)
