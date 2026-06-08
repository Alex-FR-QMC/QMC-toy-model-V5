# -*- coding: utf-8 -*-
"""Tests for mcq_v5.module — Module local container."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.state import State
from mcq_v5.metric import ScalarConformalMetric, TensorMetric
from mcq_v5.module import Module
from mcq_v5.views import StateView, MetricView


# ===== Helpers =====

def make_scalar_state(shape=(5, 5, 5)):
    return State(
        fields={
            "psi": np.ones(shape, dtype=np.float64),
            "h": 10.0 * np.ones(shape, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )


def make_scalar_metric(shape=(5, 5, 5)):
    return ScalarConformalMetric(np.ones(shape, dtype=np.float64))


def make_tensor_metric(grid_shape=(5, 5, 5), block_dim=3):
    I = np.eye(block_dim, dtype=np.float64)
    H = np.broadcast_to(I, grid_shape + (block_dim, block_dim))
    H = np.ascontiguousarray(H, dtype=np.float64)
    return TensorMetric(H)


def make_state_with_tensor_field(grid_shape=(5, 5, 5), block_dim=3):
    return State(
        fields={
            "psi": np.ones(grid_shape, dtype=np.float64),
            "H_field": np.ascontiguousarray(
                np.broadcast_to(
                    np.eye(block_dim, dtype=np.float64),
                    grid_shape + (block_dim, block_dim),
                ),
                dtype=np.float64,
            ),
        },
        solver_fields=("psi", "H_field"),
    )


# ===== A. Construction with ScalarConformalMetric =====

def test_module_construction_scalar_metric():
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m0", state=s, metric=m)
    assert mod.id == "m0"
    assert mod.state is s
    assert mod.metric is m
    assert mod.metadata == {}


def test_module_construction_int_id():
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id=42, state=s, metric=m)
    assert mod.id == 42


# ===== B. Construction with TensorMetric =====

def test_module_construction_tensor_metric():
    s = make_scalar_state()
    tm = make_tensor_metric()
    mod = Module(id="mT", state=s, metric=tm)
    assert mod.metric is tm
    # Spatial shape (5,5,5) compatible with state psi shape (5,5,5)


# ===== C. Invalid state type =====

def test_module_invalid_state_type():
    m = make_scalar_metric()
    with pytest.raises(TypeError, match="state"):
        Module(id="m", state="not a state", metric=m)


# ===== D. Invalid metric type =====

def test_module_invalid_metric_type():
    s = make_scalar_state()
    with pytest.raises(TypeError, match="metric"):
        Module(id="m", state=s, metric="not a metric")


def test_module_invalid_id_type():
    s = make_scalar_state()
    m = make_scalar_metric()
    with pytest.raises(TypeError, match="id"):
        Module(id=[1, 2, 3], state=s, metric=m)


def test_module_invalid_metadata_type():
    s = make_scalar_state()
    m = make_scalar_metric()
    with pytest.raises(TypeError, match="metadata"):
        Module(id="m", state=s, metric=m, metadata="not a dict")


# ===== E. Metadata defensive copy =====

def test_module_metadata_defensive_copy():
    """Modifying external metadata dict after construction must not affect module."""
    s = make_scalar_state()
    m = make_scalar_metric()
    meta = {"experiment": "X", "seed": 42}
    mod = Module(id="m", state=s, metric=m, metadata=meta)
    meta["seed"] = 999  # mutate external
    assert mod.metadata == {"experiment": "X", "seed": 42}
    # Adding new key externally
    meta["new"] = "value"
    assert "new" not in mod.metadata


# ===== F. state_view zero-copy (EXPLICIT TEST per audit) =====

def test_module_state_view_zero_copy_explicit():
    """Modifying via state_view().field('psi') must propagate to module.state.

    This is the EXPLICIT test required by audit point 7F:
    confirms link maintained via reference, not accidental defensive copy
    inside Module constructor.
    """
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m", state=s, metric=m)

    # The exact assertion required by audit
    mod.state_view().field("psi")[0, 0, 0] = 99.0
    assert mod.state["psi"][0, 0, 0] == 99.0


def test_module_state_view_returns_state_view():
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m", state=s, metric=m)
    sv = mod.state_view()
    assert isinstance(sv, StateView)


def test_module_state_kept_by_reference():
    """The state object inside the module IS the one passed in."""
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m", state=s, metric=m)
    assert mod.state is s  # identity, not equality


# ===== G. metric_view zero-copy =====

def test_module_metric_view_zero_copy():
    """Modifying coefficients via view must propagate to module.metric."""
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m", state=s, metric=m)
    mv = mod.metric_view()
    coeffs = mv.coefficients()
    coeffs[0, 0, 0] = 77.0
    assert m.h[0, 0, 0] == 77.0  # original metric mutated (zero-copy)


def test_module_metric_view_returns_metric_view():
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m", state=s, metric=m)
    mv = mod.metric_view()
    assert isinstance(mv, MetricView)


# ===== H. diagnostics non-mutating =====

def test_module_diagnostics_does_not_mutate():
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m", state=s, metric=m)

    psi_before = s["psi"].copy()
    h_before = m.h.copy()

    _ = mod.diagnostics()

    np.testing.assert_array_equal(s["psi"], psi_before)
    np.testing.assert_array_equal(m.h, h_before)


def test_module_diagnostics_content():
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m0", state=s, metric=m)
    d = mod.diagnostics()
    assert d["id"] == "m0"
    assert d["state_solver_fields"] == ("psi", "h")
    assert "metric_type" in d
    assert d["metric_type"] == "ScalarConformalMetric"
    assert "metric_diagnostics" in d
    assert "state_norm" in d


# ===== I. Shape mismatch state/metric =====

def test_module_shape_mismatch_scalar_metric_raises():
    """state psi shape (5,5,5), metric h shape (7,7,7) => ValueError."""
    s = make_scalar_state(shape=(5, 5, 5))
    m = make_scalar_metric(shape=(7, 7, 7))
    with pytest.raises(ValueError, match="spatial"):
        Module(id="m", state=s, metric=m)


def test_module_shape_mismatch_tensor_metric_raises():
    s = make_scalar_state(shape=(5, 5, 5))
    tm = make_tensor_metric(grid_shape=(7, 7, 7))
    with pytest.raises(ValueError, match="spatial"):
        Module(id="m", state=s, metric=tm)


# ===== J. Tensor field prefix compatibility =====

def test_module_tensor_solver_field_with_scalar_metric_compatible():
    """State has H_field shape (5,5,5,3,3), metric scalar shape (5,5,5).

    Spatial prefix (5,5,5) matches => accepted.
    """
    s = make_state_with_tensor_field(grid_shape=(5, 5, 5), block_dim=3)
    m = make_scalar_metric(shape=(5, 5, 5))
    mod = Module(id="m", state=s, metric=m)
    assert mod.state["H_field"].shape == (5, 5, 5, 3, 3)


def test_module_tensor_solver_field_with_tensor_metric_compatible():
    """State has H_field shape (5,5,5,3,3), metric tensor shape (5,5,5,3,3).

    Metric spatial prefix is (5,5,5). State psi (5,5,5) and H_field (5,5,5,3,3)
    both have prefix (5,5,5) => accepted.
    """
    s = make_state_with_tensor_field(grid_shape=(5, 5, 5), block_dim=3)
    tm = make_tensor_metric(grid_shape=(5, 5, 5), block_dim=3)
    mod = Module(id="m", state=s, metric=tm)
    assert mod.state["H_field"].shape == (5, 5, 5, 3, 3)


def test_module_field_ndim_too_low_raises():
    """A solver field with ndim < metric spatial dim must raise."""
    s = State(
        fields={"psi": np.ones((5,), dtype=np.float64)},
        solver_fields=("psi",),
    )
    m = make_scalar_metric(shape=(5, 5, 5))  # spatial dim 3 > psi.ndim 1
    with pytest.raises(ValueError, match="ndim"):
        Module(id="m", state=s, metric=m)


def test_module_field_prefix_partial_match_raises():
    """Field shape with same dim but different values raises."""
    s = State(
        fields={"psi": np.ones((5, 5, 7), dtype=np.float64)},
        solver_fields=("psi",),
    )
    m = make_scalar_metric(shape=(5, 5, 5))
    with pytest.raises(ValueError, match="spatial"):
        Module(id="m", state=s, metric=m)


# ===== K. No coupling/dynamics methods =====

def test_module_has_no_coupling_method():
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m", state=s, metric=m)
    assert not hasattr(mod, "compute_coupling")
    assert not hasattr(mod, "aggregate")
    assert not hasattr(mod, "step")
    assert not hasattr(mod, "evolve")
    assert not hasattr(mod, "flux")
    assert not hasattr(mod, "divergence")
    assert not hasattr(mod, "gradient")


# ===== id immutability (per audit) =====

def test_module_id_has_no_setter():
    """id is property-only, no setter (per audit ajout 1)."""
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="original", state=s, metric=m)
    with pytest.raises(AttributeError):
        mod.id = "modified"


def test_module_uses_slots():
    """Slots prevent ad-hoc attribute addition."""
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m", state=s, metric=m)
    with pytest.raises(AttributeError):
        mod.new_attribute = "something"


# ===== Convenience constructors =====

def test_module_with_state():
    s1 = make_scalar_state()
    s2 = make_scalar_state()
    m = make_scalar_metric()
    mod1 = Module(id="m", state=s1, metric=m, metadata={"x": 1})
    mod2 = mod1.with_state(s2)
    assert mod2.id == "m"
    assert mod2.state is s2
    assert mod2.metric is m
    assert mod2.metadata == {"x": 1}


def test_module_with_metric():
    s = make_scalar_state()
    m1 = make_scalar_metric()
    m2 = make_scalar_metric()
    mod1 = Module(id="m", state=s, metric=m1)
    mod2 = mod1.with_metric(m2)
    assert mod2.metric is m2
    assert mod2.state is s


# ===== Empty metadata default =====

def test_module_empty_metadata_default():
    s = make_scalar_state()
    m = make_scalar_metric()
    mod = Module(id="m", state=s, metric=m)
    assert mod.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
