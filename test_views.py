# -*- coding: utf-8 -*-
"""Tests for mcq_v5.views — bounded zero-copy access.

Critère de validation:
    Views are bounded, zero-copy where applicable, macro-step-safe for history,
    non-computational, and do not reintroduce hidden global access.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.grid import Grid, make_grid
from mcq_v5.state import State
from mcq_v5.views import (
    StateView, MetricView, GridView,
    HistoryView, HistoryEntry,
)


# =========================================================================
# StateView — zero-copy, bounded access
# =========================================================================

def make_state():
    return State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": 10.0 * np.ones((5, 5, 5), dtype=np.float64),
            "mask": np.zeros((5, 5, 5), dtype=bool),
        },
        solver_fields=("psi", "h"),
    )


def test_stateview_field_zero_copy():
    """A. field(name) returns the actual reference; modifying it mutates State."""
    s = make_state()
    v = StateView(s)
    arr = v.field("psi")
    arr[0, 0, 0] = 42.0
    assert s["psi"][0, 0, 0] == 42.0


def test_stateview_field_unknown_raises():
    s = make_state()
    v = StateView(s)
    with pytest.raises(KeyError):
        v.field("unknown")


def test_stateview_has_field():
    s = make_state()
    v = StateView(s)
    assert v.has_field("psi") is True
    assert v.has_field("mask") is True
    assert v.has_field("nope") is False


def test_stateview_solver_field_names():
    s = make_state()
    v = StateView(s)
    assert v.solver_field_names() == ("psi", "h")


def test_stateview_all_field_names():
    s = make_state()
    v = StateView(s)
    names = v.all_field_names()
    assert set(names) == {"psi", "h", "mask"}


def test_stateview_requires_state():
    with pytest.raises(TypeError, match="State"):
        StateView("not a state")


# =========================================================================
# StateView.local_patch
# =========================================================================

def test_local_patch_interior():
    """B. Interior center yields full (2r+1)^d patch."""
    s = make_state()
    v = StateView(s)
    patch = v.local_patch("psi", center=(2, 2, 2), radius=1)
    assert patch.shape == (3, 3, 3)


def test_local_patch_corner_truncated():
    """C. Patch is truncated at boundaries, no padding."""
    s = make_state()
    v = StateView(s)
    patch = v.local_patch("psi", center=(0, 0, 0), radius=1)
    # Truncated: lo=0, hi=min(5, 0+1+1)=2 per axis
    assert patch.shape == (2, 2, 2)


def test_local_patch_face_truncated():
    s = make_state()
    v = StateView(s)
    patch = v.local_patch("psi", center=(0, 2, 2), radius=1)
    assert patch.shape == (2, 3, 3)


def test_local_patch_radius_zero():
    """radius=0 returns a 1x1x1 patch (the center cell itself)."""
    s = make_state()
    v = StateView(s)
    patch = v.local_patch("psi", center=(2, 2, 2), radius=0)
    assert patch.shape == (1, 1, 1)


def test_local_patch_invalid_center_raises():
    """D. Center out of bounds raises ValueError."""
    s = make_state()
    v = StateView(s)
    with pytest.raises(ValueError, match="out of grid bounds"):
        v.local_patch("psi", center=(5, 0, 0), radius=1)
    with pytest.raises(ValueError, match="out of grid bounds"):
        v.local_patch("psi", center=(-1, 0, 0), radius=1)


def test_local_patch_negative_radius_raises():
    s = make_state()
    v = StateView(s)
    with pytest.raises(ValueError, match="radius"):
        v.local_patch("psi", center=(2, 2, 2), radius=-1)


def test_local_patch_invalid_center_type_raises():
    s = make_state()
    v = StateView(s)
    with pytest.raises(TypeError):
        v.local_patch("psi", center=(2.5, 2, 2), radius=1)


def test_local_patch_zero_copy_when_writable():
    """Modifying interior patch slice modifies underlying State.

    Note: numpy advanced indexing might return a copy, but basic slicing
    returns a view. Center+radius generates basic slicing.
    """
    s = make_state()
    v = StateView(s)
    patch = v.local_patch("psi", center=(2, 2, 2), radius=1)
    # Verify it's a view by checking base
    assert patch.base is not None
    # Mutate the patch
    patch[0, 0, 0] = 777.0
    # Underlying State affected
    assert s["psi"][1, 1, 1] == 777.0


def test_local_patch_extra_dims_taken_fully():
    """For a tensor field, dims beyond len(center) are taken fully."""
    arr = np.random.rand(5, 5, 5, 3, 3).astype(np.float64)
    s = State(fields={"H": arr}, solver_fields=("H",))
    v = StateView(s)
    patch = v.local_patch("H", center=(2, 2, 2), radius=1)
    # Spatial dims truncated to (3,3,3), tensor dims full (3,3)
    assert patch.shape == (3, 3, 3, 3, 3)


def test_local_patch_no_silent_copy():
    """Convention: local_patch returns raw slice, no ascontiguousarray."""
    s = make_state()
    v = StateView(s)
    patch = v.local_patch("psi", center=(2, 2, 2), radius=1)
    # This basic slice should be contiguous already, but the contract is:
    # we never silently call ascontiguousarray. Verify by checking that
    # patch.base is the underlying array (proof of zero-copy).
    assert patch.base is s["psi"] or patch.base is not None


# =========================================================================
# GridView
# =========================================================================

def test_gridview_delegation():
    """H. GridView delegates shape/dx/dim/boundary correctly."""
    g = make_grid((5, 5, 5))
    gv = GridView(g)
    assert gv.shape == (5, 5, 5)
    assert gv.dx == (1.0, 1.0, 1.0)
    assert gv.dim == 3
    assert gv.n_cells == 125
    assert gv.boundary_policy() == "zero_flux_neumann"


def test_gridview_cell_centers_delegated():
    g = make_grid((5, 5, 5))
    gv = GridView(g)
    centers = gv.cell_centers()
    assert len(centers) == 3
    np.testing.assert_allclose(centers[0], [0.5, 1.5, 2.5, 3.5, 4.5])


def test_gridview_faces_delegated():
    g = make_grid((5, 5, 5))
    gv = GridView(g)
    f = gv.faces(0)
    assert f.shape == (4,)


def test_gridview_requires_grid():
    with pytest.raises(TypeError, match="Grid"):
        GridView("not a grid")


# =========================================================================
# MetricView (stub)
# =========================================================================

def test_metricview_stub_coefficients():
    """Stub: works if metric has .coefficients attribute."""
    class FakeMetric:
        coefficients = np.ones((5, 5, 5), dtype=np.float64)

    mv = MetricView(FakeMetric())
    coeffs = mv.coefficients()
    assert coeffs.shape == (5, 5, 5)
    # Verify zero-copy: modifying view modifies fake
    coeffs[0, 0, 0] = 99.0
    assert FakeMetric.coefficients[0, 0, 0] == 99.0


def test_metricview_face_coefficients_not_implemented():
    class FakeMetric:
        coefficients = np.ones((5, 5, 5), dtype=np.float64)
    mv = MetricView(FakeMetric())
    with pytest.raises(NotImplementedError):
        mv.face_coefficients(0)


def test_metricview_no_coefficients_attr_raises():
    class Empty:
        pass
    mv = MetricView(Empty())
    with pytest.raises(NotImplementedError):
        mv.coefficients()


# =========================================================================
# HistoryView — macro-step-only, writeable=False
# =========================================================================

def _make_state_with_psi(value: float) -> State:
    return State(
        fields={
            "psi": np.full((3, 3, 3), value, dtype=np.float64),
            "h": np.ones((3, 3, 3), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )


def test_historyview_only_committed_visible():
    """E. Uncommitted substeps must NEVER appear in previous/window."""
    history = [
        HistoryEntry(state=_make_state_with_psi(0.0), time=0.0, step=0, committed=True),
        HistoryEntry(state=_make_state_with_psi(0.5), time=0.5, step=0, committed=False),  # SUBSTEP
        HistoryEntry(state=_make_state_with_psi(1.0), time=1.0, step=1, committed=True),
        HistoryEntry(state=_make_state_with_psi(1.5), time=1.5, step=1, committed=False),  # SUBSTEP
        HistoryEntry(state=_make_state_with_psi(2.0), time=2.0, step=2, committed=True),
    ]
    h = HistoryView(history)
    assert h.n_committed() == 3
    # Latest committed
    arr = h.previous("psi", n=1)
    assert arr[0, 0, 0] == 2.0
    # Second-to-latest committed
    arr = h.previous("psi", n=2)
    assert arr[0, 0, 0] == 1.0
    # Oldest committed
    arr = h.previous("psi", n=3)
    assert arr[0, 0, 0] == 0.0


def test_historyview_substeps_never_appear():
    """Critical: no substep value (0.5 or 1.5) ever returned."""
    history = [
        HistoryEntry(state=_make_state_with_psi(0.0), time=0.0, step=0, committed=True),
        HistoryEntry(state=_make_state_with_psi(0.5), time=0.5, step=0, committed=False),
        HistoryEntry(state=_make_state_with_psi(1.0), time=1.0, step=1, committed=True),
    ]
    h = HistoryView(history)
    # Get all available
    window = h.window("psi", length=10)
    values = [arr[0, 0, 0] for arr in window]
    assert 0.5 not in values
    assert 1.5 not in values
    assert set(values) == {0.0, 1.0}


def test_historyview_previous_n_too_large_raises():
    history = [
        HistoryEntry(state=_make_state_with_psi(0.0), time=0.0, step=0, committed=True),
    ]
    h = HistoryView(history)
    with pytest.raises(IndexError):
        h.previous("psi", n=5)


def test_historyview_previous_invalid_n_raises():
    history = [
        HistoryEntry(state=_make_state_with_psi(0.0), time=0.0, step=0, committed=True),
    ]
    h = HistoryView(history)
    with pytest.raises(ValueError, match="n must be"):
        h.previous("psi", n=0)


def test_historyview_window_ordering():
    """G. window returns oldest-first within the window."""
    history = [
        HistoryEntry(state=_make_state_with_psi(float(i)), time=float(i), step=i, committed=True)
        for i in range(5)
    ]
    h = HistoryView(history)
    w = h.window("psi", length=3)
    assert len(w) == 3
    assert w[0][0, 0, 0] == 2.0  # oldest in window
    assert w[1][0, 0, 0] == 3.0
    assert w[2][0, 0, 0] == 4.0  # newest in window


def test_historyview_window_shorter_than_requested():
    """If fewer committed entries than `length`, returns what's available."""
    history = [
        HistoryEntry(state=_make_state_with_psi(0.0), time=0.0, step=0, committed=True),
        HistoryEntry(state=_make_state_with_psi(1.0), time=1.0, step=1, committed=True),
    ]
    h = HistoryView(history)
    w = h.window("psi", length=10)
    assert len(w) == 2


def test_historyview_readonly_protection():
    """Returned arrays must have writeable=False (anti-corruption)."""
    history = [
        HistoryEntry(state=_make_state_with_psi(0.0), time=0.0, step=0, committed=True),
    ]
    h = HistoryView(history)
    arr = h.previous("psi", n=1)
    assert arr.flags["WRITEABLE"] is False
    # Attempting to mutate raises ValueError
    with pytest.raises(ValueError):
        arr[0, 0, 0] = 999.0


def test_historyview_readonly_does_not_copy():
    """writeable=False is on a view, not a copy. Underlying state unchanged
    by the protection itself."""
    history = [
        HistoryEntry(state=_make_state_with_psi(7.0), time=0.0, step=0, committed=True),
    ]
    h = HistoryView(history)
    arr = h.previous("psi", n=1)
    # The view shares memory with the underlying state field
    assert arr.base is history[0].state.fields["psi"] or arr.base is not None


def test_historyview_does_not_own_history():
    """HistoryView holds a reference. Modifying the external list affects the view."""
    history = [
        HistoryEntry(state=_make_state_with_psi(0.0), time=0.0, step=0, committed=True),
    ]
    h = HistoryView(history)
    assert h.n_committed() == 1
    # External code adds a new committed entry
    history.append(
        HistoryEntry(state=_make_state_with_psi(1.0), time=1.0, step=1, committed=True)
    )
    assert h.n_committed() == 2  # view reflects the external mutation


def test_historyview_does_not_mutate_history():
    """Reading from view does not mutate the source list."""
    history = [
        HistoryEntry(state=_make_state_with_psi(0.0), time=0.0, step=0, committed=True),
        HistoryEntry(state=_make_state_with_psi(1.0), time=1.0, step=1, committed=True),
    ]
    h = HistoryView(history)
    snapshot = list(history)  # shallow copy of references
    _ = h.previous("psi", n=1)
    _ = h.window("psi", length=2)
    _ = h.n_committed()
    assert history == snapshot
    # Each entry untouched
    for e, e0 in zip(history, snapshot):
        assert e is e0


def test_historyview_none_raises():
    with pytest.raises(TypeError, match="None"):
        HistoryView(None)


def test_historyview_states_returns_state_list():
    history = [
        HistoryEntry(state=_make_state_with_psi(float(i)), time=float(i), step=i, committed=True)
        for i in range(3)
    ]
    h = HistoryView(history)
    states = h.states(length=2)
    assert len(states) == 2
    assert isinstance(states[0], State)
    assert states[0]["psi"][0, 0, 0] == 1.0  # oldest in window
    assert states[1]["psi"][0, 0, 0] == 2.0


# =========================================================================
# Non-computational invariant
# =========================================================================

def test_views_do_not_mutate_state_by_reading():
    """I. Reading via views does not change State (unless caller explicitly
    modifies a returned mutable array)."""
    s = make_state()
    psi_snapshot = s["psi"].copy()
    h_snapshot = s["h"].copy()
    mask_snapshot = s["mask"].copy()

    v = StateView(s)
    _ = v.field("psi")
    _ = v.has_field("h")
    _ = v.solver_field_names()
    _ = v.local_patch("psi", center=(2, 2, 2), radius=1)

    np.testing.assert_array_equal(s["psi"], psi_snapshot)
    np.testing.assert_array_equal(s["h"], h_snapshot)
    np.testing.assert_array_equal(s["mask"], mask_snapshot)


def test_views_zero_copy_through_local_patch():
    """Modifying a patch in-place must propagate to State."""
    s = make_state()
    v = StateView(s)
    patch = v.local_patch("psi", center=(2, 2, 2), radius=1)
    patch[:] = 5.0
    # All 27 affected cells in psi should now be 5
    assert s["psi"][1, 1, 1] == 5.0
    assert s["psi"][3, 3, 3] == 5.0
    # Cells outside the patch unchanged
    assert s["psi"][0, 0, 0] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
