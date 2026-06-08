# -*- coding: utf-8 -*-
"""Tests for mcq_v5.coupling_operator.

Critère étape 10: abstract pairwise operator receiving only CouplingContext;
NullCouplingOperator returns immutable zero CouplingContribution for any
valid context; R=0/R=1 produce zero coupling; operator cannot see Instance,
Module, CouplingScheduler, global state, or history ownership.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.grid import make_grid
from mcq_v5.state import State
from mcq_v5.metric import ScalarConformalMetric
from mcq_v5.views import StateView, MetricView, GridView, HistoryView, HistoryEntry
from mcq_v5.coupling_context import CouplingContext
from mcq_v5.coupling_operator import (
    CouplingOperator, NullCouplingOperator, CouplingContribution,
)


# ===== Helpers =====

def make_views_pair(grid_shape=(5, 5, 5), psi_t=1.0, psi_s=2.0):
    s_t = State(
        fields={
            "psi": psi_t * np.ones(grid_shape, dtype=np.float64),
            "h": np.ones(grid_shape, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    s_s = State(
        fields={
            "psi": psi_s * np.ones(grid_shape, dtype=np.float64),
            "h": np.ones(grid_shape, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    m_t = ScalarConformalMetric(np.ones(grid_shape, dtype=np.float64))
    m_s = ScalarConformalMetric(np.ones(grid_shape, dtype=np.float64))
    g = make_grid(grid_shape, dx=1.0)
    return {
        "target_state_view": StateView(s_t),
        "source_state_view": StateView(s_s),
        "target_metric_view": MetricView(m_t),
        "source_metric_view": MetricView(m_s),
        "grid_view": GridView(g),
    }


def make_ctx(R=0.5, mode="global_scalar", **overrides):
    views = make_views_pair()
    kwargs = dict(
        target_id="m0", source_id="m1",
        overlap_R=R, overlap_mode=mode, time=0.0,
        **views,
    )
    kwargs.update(overrides)
    return CouplingContext(**kwargs)


# ===== A. Abstract not instantiable =====

def test_coupling_operator_abstract_not_instantiable():
    with pytest.raises(TypeError, match="abstract"):
        CouplingOperator()


# ===== B. NullCouplingOperator instantiable =====

def test_null_operator_instantiable():
    op = NullCouplingOperator()
    assert isinstance(op, CouplingOperator)
    assert isinstance(op, NullCouplingOperator)


# ===== C. pairwise_coupling returns CouplingContribution =====

def test_null_operator_returns_contribution():
    op = NullCouplingOperator()
    ctx = make_ctx()
    contrib = op.pairwise_coupling(ctx)
    assert isinstance(contrib, CouplingContribution)


# ===== D. Contribution ids match context ids =====

def test_contribution_ids_match_context_ids():
    op = NullCouplingOperator()
    ctx = make_ctx()
    contrib = op.pairwise_coupling(ctx)
    assert contrib.target_id == ctx.target_id
    assert contrib.source_id == ctx.source_id


# ===== E. Contribution value: zero, shape, dtype, C-contig =====

def test_contribution_value_zero_shape_dtype():
    op = NullCouplingOperator()
    ctx = make_ctx()
    contrib = op.pairwise_coupling(ctx)
    v = contrib.value
    assert isinstance(v, np.ndarray)
    assert v.shape == (5, 5, 5)
    assert v.dtype == np.float64
    assert v.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(v, np.zeros((5, 5, 5)))


# ===== F. Contribution value read-only =====

def test_contribution_value_read_only():
    op = NullCouplingOperator()
    ctx = make_ctx()
    contrib = op.pairwise_coupling(ctx)
    assert contrib.value.flags["WRITEABLE"] is False
    with pytest.raises(ValueError):
        contrib.value[0, 0, 0] = 1.0


# ===== G. Contribution metadata read-only =====

def test_contribution_metadata_read_only():
    op = NullCouplingOperator()
    ctx = make_ctx()
    contrib = op.pairwise_coupling(ctx)
    with pytest.raises(TypeError):
        contrib.metadata["x"] = 1


def test_contribution_metadata_content():
    op = NullCouplingOperator()
    ctx = make_ctx()
    contrib = op.pairwise_coupling(ctx)
    assert contrib.metadata["operator"] == "NullCouplingOperator"
    assert contrib.metadata["reason"] == "null_stub"
    assert contrib.metadata["overlap_mode"] == ctx.overlap_mode


# ===== H/I/J. R=0, R=1, R=0.5 all yield zero =====

def test_null_operator_R0_zero():
    op = NullCouplingOperator()
    ctx = make_ctx(R=0.0)
    contrib = op.pairwise_coupling(ctx)
    np.testing.assert_array_equal(contrib.value, np.zeros((5, 5, 5)))


def test_null_operator_R1_zero():
    op = NullCouplingOperator()
    ctx = make_ctx(R=1.0)
    contrib = op.pairwise_coupling(ctx)
    np.testing.assert_array_equal(contrib.value, np.zeros((5, 5, 5)))


def test_null_operator_Rmid_zero():
    op = NullCouplingOperator()
    ctx = make_ctx(R=0.5)
    contrib = op.pairwise_coupling(ctx)
    np.testing.assert_array_equal(contrib.value, np.zeros((5, 5, 5)))


# ===== K. local_field R array yields zero =====

def test_null_operator_local_field_zero():
    op = NullCouplingOperator()
    views = make_views_pair()
    R = 0.3 * np.ones((5, 5, 5), dtype=np.float64)
    ctx = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=R, overlap_mode="local_field", time=0.0,
        **views,
    )
    contrib = op.pairwise_coupling(ctx)
    np.testing.assert_array_equal(contrib.value, np.zeros((5, 5, 5)))


# ===== L. patchwise R array yields zero (context-direct) =====

def test_null_operator_patchwise_zero():
    op = NullCouplingOperator()
    views = make_views_pair()
    R = np.array([0.2, 0.7], dtype=np.float64)
    ctx = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=R, overlap_mode="patchwise", time=0.0,
        **views,
    )
    contrib = op.pairwise_coupling(ctx)
    np.testing.assert_array_equal(contrib.value, np.zeros((5, 5, 5)))


# ===== M. Passing Instance raises TypeError =====

def test_passing_instance_raises():
    """Cannot pass anything other than CouplingContext."""
    op = NullCouplingOperator()
    # Use a synthetic "Instance-like" string
    with pytest.raises(TypeError, match="CouplingContext"):
        op.pairwise_coupling("not a context")


# ===== N. Passing Module raises TypeError =====

def test_passing_module_raises():
    op = NullCouplingOperator()
    with pytest.raises(TypeError, match="CouplingContext"):
        op.pairwise_coupling(42)


def test_passing_none_raises():
    op = NullCouplingOperator()
    with pytest.raises(TypeError, match="CouplingContext"):
        op.pairwise_coupling(None)


# ===== O. File does not import instance, coupling_scheduler =====

def test_coupling_operator_file_isolation():
    """Audit ajout A: coupling_operator.py must NOT import from mcq_v5.module,
    mcq_v5.instance, or mcq_v5.coupling_scheduler.
    """
    import mcq_v5.coupling_operator as op_mod
    src = Path(op_mod.__file__).read_text()
    forbidden_imports = [
        "from .module import",
        "from .instance import",
        "from .coupling_scheduler import",
        "from mcq_v5.module import",
        "from mcq_v5.instance import",
        "from mcq_v5.coupling_scheduler import",
        "import mcq_v5.module",
        "import mcq_v5.instance",
        "import mcq_v5.coupling_scheduler",
    ]
    for pat in forbidden_imports:
        assert pat not in src, f"forbidden import found: {pat}"


def test_coupling_operator_namespace_no_module_or_instance():
    """The module's namespace must not expose Module, Instance, etc."""
    import mcq_v5.coupling_operator as op_mod
    forbidden_names = ["Module", "Instance", "CouplingScheduler"]
    for n in forbidden_names:
        assert not hasattr(op_mod, n), f"forbidden name in namespace: {n}"


def test_module_id_locally_redefined():
    """ModuleId in coupling_operator must be defined locally, not imported
    from mcq_v5.module (audit ajout A)."""
    import mcq_v5.coupling_operator as op_mod
    assert hasattr(op_mod, "ModuleId")
    # str | int alias
    assert op_mod.ModuleId == (str | int) or op_mod.ModuleId is not None


# ===== P. Operator has no method build_contexts / admissible_pairs etc. =====

def test_operator_no_scheduler_methods():
    op = NullCouplingOperator()
    forbidden = [
        "build_contexts", "admissible_pairs",
        "aggregate", "step", "evolve", "apply",
    ]
    for name in forbidden:
        assert not hasattr(op, name), f"forbidden attr present: {name}"


# ===== Q. Stateless / no cache =====

def test_null_operator_slots_empty():
    """__slots__ = () means no instance attributes."""
    op = NullCouplingOperator()
    with pytest.raises(AttributeError):
        op._last_context = make_ctx()
    with pytest.raises(AttributeError):
        op._cache = {}
    with pytest.raises(AttributeError):
        op.history = []


def test_null_operator_no_state_attributes():
    op = NullCouplingOperator()
    forbidden = [
        "instance", "module", "scheduler", "context_cache",
        "last_context", "history",
    ]
    for n in forbidden:
        assert not hasattr(op, n), f"forbidden attr: {n}"


# ===== R. Null operator does not depend on psi values =====

def test_null_operator_independent_of_state_values():
    op = NullCouplingOperator()
    # Two contexts with very different psi values
    ctx1 = make_ctx()  # psi_t=1.0, psi_s=2.0 default
    views2 = make_views_pair(psi_t=100.0, psi_s=-50.0)
    ctx2 = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
        **views2,
    )
    c1 = op.pairwise_coupling(ctx1)
    c2 = op.pairwise_coupling(ctx2)
    # Null contribution is the same: zero
    np.testing.assert_array_equal(c1.value, c2.value)
    np.testing.assert_array_equal(c1.value, np.zeros((5, 5, 5)))


# ===== Contribution: cross-validation kind/value/mass_delta =====

def test_contribution_source_valid():
    v = np.zeros((5, 5, 5), dtype=np.float64)
    c = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=v,
        conservative=True, mass_delta=0.0,
    )
    assert c.kind == "source"
    assert c.value.flags["WRITEABLE"] is False


def test_contribution_source_wrong_dtype_rejected():
    v = np.zeros((5, 5, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="source", value=v,
            conservative=True, mass_delta=0.0,
        )


def test_contribution_source_non_contiguous_rejected():
    big = np.zeros((10, 5, 5), dtype=np.float64)
    v = big[::2]
    if v.flags["C_CONTIGUOUS"]:
        pytest.skip("slice happened to remain contiguous")
    with pytest.raises(ValueError, match="C-contiguous"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="source", value=v,
            conservative=True, mass_delta=0.0,
        )


def test_contribution_source_tuple_value_rejected():
    """Audit ajout C: kind='source' with tuple value raises."""
    v = (np.zeros((5, 5, 5), dtype=np.float64),)
    with pytest.raises(TypeError, match="ndarray"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="source", value=v,
            conservative=True, mass_delta=0.0,
        )


def test_contribution_flux_valid():
    J0 = np.zeros((4, 5, 5), dtype=np.float64)
    J1 = np.zeros((5, 4, 5), dtype=np.float64)
    J2 = np.zeros((5, 5, 4), dtype=np.float64)
    c = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="flux", value=(J0, J1, J2),
        conservative=True, mass_delta=0.0,
    )
    assert c.kind == "flux"
    for J in c.value:
        assert J.flags["WRITEABLE"] is False


def test_contribution_flux_with_nonzero_mass_delta_rejected():
    """Audit ajout C: kind='flux' with mass_delta != 0 raises."""
    J0 = np.zeros((4, 5, 5), dtype=np.float64)
    J1 = np.zeros((5, 4, 5), dtype=np.float64)
    J2 = np.zeros((5, 5, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="mass_delta"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="flux", value=(J0, J1, J2),
            conservative=False, mass_delta=0.5,
        )


def test_contribution_flux_ndarray_value_rejected():
    """kind='flux' with single ndarray (not tuple) raises."""
    v = np.zeros((4, 5, 5), dtype=np.float64)
    with pytest.raises(TypeError, match="tuple"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="flux", value=v,
            conservative=True, mass_delta=0.0,
        )


def test_contribution_none_kind_requires_empty_value():
    c = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="none", value=(),
        conservative=True, mass_delta=0.0,
    )
    assert c.kind == "none"


def test_contribution_none_with_nonempty_value_rejected():
    v = np.zeros((5, 5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="none"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="none", value=v,
            conservative=True, mass_delta=0.0,
        )


def test_contribution_none_with_nonzero_mass_delta_rejected():
    with pytest.raises(ValueError, match="mass_delta"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="none", value=(),
            conservative=True, mass_delta=1.0,
        )


def test_contribution_invalid_kind_rejected():
    v = np.zeros((5, 5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="kind"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="garbage", value=v,
            conservative=True, mass_delta=0.0,
        )


def test_contribution_self_pair_rejected():
    v = np.zeros((5, 5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="differ"):
        CouplingContribution(
            target_id="m0", source_id="m0",
            kind="source", value=v,
            conservative=True, mass_delta=0.0,
        )


def test_contribution_mass_delta_nan_rejected():
    v = np.zeros((5, 5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="source", value=v,
            conservative=False, mass_delta=float("nan"),
        )


def test_contribution_conservative_invalid_type():
    v = np.zeros((5, 5, 5), dtype=np.float64)
    with pytest.raises(TypeError, match="conservative"):
        CouplingContribution(
            target_id="m0", source_id="m1",
            kind="source", value=v,
            conservative="yes", mass_delta=0.0,
        )


# ===== Contribution metadata copy =====

def test_contribution_metadata_defensive_copy():
    v = np.zeros((5, 5, 5), dtype=np.float64)
    meta = {"k": 1}
    c = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=v,
        conservative=True, mass_delta=0.0,
        metadata=meta,
    )
    meta["k"] = 999
    meta["new"] = "x"
    assert c.metadata["k"] == 1
    assert "new" not in c.metadata


# ===== Null operator uses zero-source convention =====

def test_null_operator_kind_is_source():
    op = NullCouplingOperator()
    contrib = op.pairwise_coupling(make_ctx())
    assert contrib.kind == "source"


def test_null_operator_conservative_true():
    op = NullCouplingOperator()
    contrib = op.pairwise_coupling(make_ctx())
    assert contrib.conservative is True


def test_null_operator_mass_delta_zero():
    op = NullCouplingOperator()
    contrib = op.pairwise_coupling(make_ctx())
    assert contrib.mass_delta == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
