# -*- coding: utf-8 -*-
"""Tests for mcq_v5.grid — Grid invariants."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcq_v5.grid import Grid, make_grid


# ===== Construction =====

def test_grid_5x5x5_construction():
    g = make_grid((5, 5, 5))
    assert g.shape == (5, 5, 5)
    assert g.dim == 3
    assert g.n_cells == 125


def test_grid_7x7x7_construction():
    g = make_grid((7, 7, 7))
    assert g.shape == (7, 7, 7)
    assert g.dim == 3
    assert g.n_cells == 343


def test_grid_2d_construction():
    g = make_grid((4, 6), dx=0.5)
    assert g.shape == (4, 6)
    assert g.dim == 2
    assert g.n_cells == 24


def test_grid_anisotropic_dx():
    g = make_grid((5, 5, 5), dx=(1.0, 2.0, 0.5))
    assert g.dx == (1.0, 2.0, 0.5)


# ===== Cell centers =====

def test_cell_centers_convention():
    """Cell-centered: coords[i] = (i + 0.5) * dx."""
    g = make_grid((5, 5, 5), dx=1.0)
    centers = g.cell_centers()
    assert len(centers) == 3
    for axis_centers in centers:
        assert axis_centers.shape == (5,)
        np.testing.assert_allclose(axis_centers, [0.5, 1.5, 2.5, 3.5, 4.5])


def test_cell_centers_anisotropic():
    g = make_grid((3, 4), dx=(2.0, 0.5))
    cx, cy = g.cell_centers()
    np.testing.assert_allclose(cx, [1.0, 3.0, 5.0])
    np.testing.assert_allclose(cy, [0.25, 0.75, 1.25, 1.75])


# ===== Faces =====

def test_faces_count():
    """N cells along axis => N-1 interior faces."""
    g = make_grid((5, 5, 5))
    for axis in range(3):
        faces = g.faces(axis)
        assert faces.shape == (4,)


def test_faces_positions():
    """Faces at boundaries between cells."""
    g = make_grid((5, 5, 5), dx=1.0)
    faces = g.faces(0)
    np.testing.assert_allclose(faces, [1.0, 2.0, 3.0, 4.0])


def test_faces_invalid_axis():
    g = make_grid((5, 5, 5))
    with pytest.raises(ValueError, match="out of range"):
        g.faces(3)
    with pytest.raises(ValueError, match="out of range"):
        g.faces(-1)


# ===== Neighbors =====

def test_neighbors_interior_cell():
    """Interior cell has 2 * dim neighbors."""
    g = make_grid((5, 5, 5))
    nbrs = list(g.neighbors((2, 2, 2)))
    assert len(nbrs) == 6


def test_neighbors_corner_cell():
    """Corner (0,0,0) has only dim neighbors."""
    g = make_grid((5, 5, 5))
    nbrs = list(g.neighbors((0, 0, 0)))
    assert len(nbrs) == 3
    assert (1, 0, 0) in nbrs
    assert (0, 1, 0) in nbrs
    assert (0, 0, 1) in nbrs


def test_neighbors_face_cell():
    """Face cell has 2*dim - 1 neighbors."""
    g = make_grid((5, 5, 5))
    nbrs = list(g.neighbors((0, 2, 2)))
    assert len(nbrs) == 5


def test_neighbors_index_dim_mismatch():
    g = make_grid((5, 5, 5))
    with pytest.raises(ValueError, match="components"):
        list(g.neighbors((1, 2)))


# ===== Boundary policy =====

def test_boundary_policy_v5_0a():
    """V5-0a: zero-flux Neumann."""
    g = make_grid((5, 5, 5))
    assert g.boundary_policy() == "zero_flux_neumann"


# ===== Validation =====

def test_invalid_shape_negative():
    with pytest.raises(ValueError, match="positive"):
        Grid(shape=(5, -1, 5), dx=(1.0, 1.0, 1.0))


def test_invalid_shape_zero():
    with pytest.raises(ValueError, match="positive"):
        Grid(shape=(5, 0, 5), dx=(1.0, 1.0, 1.0))


def test_invalid_dx_negative():
    with pytest.raises(ValueError, match="positive"):
        Grid(shape=(5, 5, 5), dx=(1.0, -1.0, 1.0))


def test_shape_dx_dim_mismatch():
    with pytest.raises(ValueError, match="same dim"):
        Grid(shape=(5, 5, 5), dx=(1.0, 1.0))


# ===== Immutability (frozen dataclass) =====

def test_grid_is_frozen():
    g = make_grid((5, 5, 5))
    with pytest.raises(Exception):
        g.shape = (7, 7, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
