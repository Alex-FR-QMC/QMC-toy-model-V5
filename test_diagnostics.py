# -*- coding: utf-8 -*-
"""Tests for mcq_v5.diagnostics — observational layer."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.state import State
from mcq_v5.grid import make_grid
from mcq_v5.metric import ScalarConformalMetric
from mcq_v5.module import Module
from mcq_v5.instance import Instance
from mcq_v5.geometry import GeometryOperator
from mcq_v5.coupling_operator import CouplingContribution
from mcq_v5.coupling_aggregator import CouplingAggregator, AggregatedCoupling
from mcq_v5.diagnostics import (
    Diagnostics,
    STATUS_PASS, STATUS_WARN, STATUS_FAIL, STATUS_UNDEFINED,
)


# ===== Helpers =====

def make_module(mid, shape=(5, 5, 5)):
    s = State(
        fields={
            "psi": np.ones(shape, dtype=np.float64),
            "h": np.ones(shape, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    m = ScalarConformalMetric(np.ones(shape, dtype=np.float64))
    return Module(id=mid, state=s, metric=m)


# ===== A. Construction =====

def test_diagnostics_construction():
    d = Diagnostics()
    assert d.metadata == {}


def test_diagnostics_with_metadata():
    d = Diagnostics(metadata={"label": "test"})
    assert d.metadata == {"label": "test"}


def test_diagnostics_invalid_metadata():
    with pytest.raises(TypeError, match="metadata"):
        Diagnostics(metadata="not a dict")


# ===== B. mass_conservation PASS =====

def test_mass_conservation_pass():
    d = Diagnostics()
    before = np.ones((5, 5, 5), dtype=np.float64)
    after = np.ones((5, 5, 5), dtype=np.float64)
    result = d.mass_conservation(before, after)
    assert result["status"] == STATUS_PASS
    assert result["delta"] == 0.0


# ===== C. mass_conservation FAIL =====

def test_mass_conservation_fail():
    d = Diagnostics()
    before = np.ones((5, 5, 5), dtype=np.float64)
    after = 2.0 * np.ones((5, 5, 5), dtype=np.float64)
    result = d.mass_conservation(before, after, tol=1e-12)
    assert result["status"] == STATUS_FAIL
    assert result["delta"] == pytest.approx(125.0)


def test_mass_conservation_shape_mismatch_raises():
    d = Diagnostics()
    with pytest.raises(ValueError, match="shape"):
        d.mass_conservation(
            np.ones(5, dtype=np.float64),
            np.ones(7, dtype=np.float64),
        )


def test_mass_basic():
    d = Diagnostics()
    result = d.mass(2.0 * np.ones(10, dtype=np.float64))
    assert result["status"] == STATUS_PASS
    assert result["mass"] == 20.0


# ===== D. positivity PASS =====

def test_positivity_pass():
    d = Diagnostics()
    field = np.array([1.0, 2.0, 3.0])
    result = d.positivity(field, floor=0.0)
    assert result["status"] == STATUS_PASS
    assert result["n_violations"] == 0


def test_positivity_floor_inclusive_default():
    """Default: strict=False -> 0 is allowed as floor."""
    d = Diagnostics()
    field = np.array([0.0, 1.0, 2.0])
    result = d.positivity(field, floor=0.0, strict=False)
    assert result["status"] == STATUS_PASS


# ===== E. positivity FAIL =====

def test_positivity_fail():
    d = Diagnostics()
    field = np.array([1.0, -0.5, 2.0, -0.1])
    result = d.positivity(field, floor=0.0)
    assert result["status"] == STATUS_FAIL
    assert result["n_violations"] == 2
    assert result["min"] == -0.5


def test_positivity_strict_floor():
    """strict=True: 0 is a violation."""
    d = Diagnostics()
    field = np.array([0.0, 1.0])
    result = d.positivity(field, floor=0.0, strict=True)
    assert result["status"] == STATUS_FAIL
    assert result["n_violations"] == 1


# ===== F. underflow NONE =====

def test_underflow_none():
    d = Diagnostics()
    field = np.array([1.0, 0.5, 0.1])
    result = d.underflow(field, threshold=1e-300)
    assert result["status"] == STATUS_PASS
    assert result["n_underflow"] == 0


# ===== G. underflow WARN =====

def test_underflow_warn():
    d = Diagnostics()
    field = np.array([1.0, 1e-301, 0.5, 0.0])
    result = d.underflow(field, threshold=1e-300)
    assert result["status"] == STATUS_WARN
    assert result["n_underflow"] == 2


# ===== H. functional_profile basic stats =====

def test_functional_profile_basic():
    d = Diagnostics()
    field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = d.functional_profile(field)
    assert result["status"] == STATUS_PASS
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    assert result["finite"] is True
    assert result["n_nan"] == 0
    assert result["n_inf"] == 0


def test_functional_profile_with_thresholds():
    d = Diagnostics()
    field = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    result = d.functional_profile(field, thresholds={"high": 1.2})
    assert result["active_fractions"]["high"] == pytest.approx(0.4)


# ===== I. functional_profile detects NaN/Inf =====

def test_functional_profile_with_nan():
    d = Diagnostics()
    field = np.array([1.0, np.nan, 3.0])
    result = d.functional_profile(field)
    assert result["finite"] is False
    assert result["n_nan"] == 1
    assert result["status"] == STATUS_FAIL


def test_functional_profile_with_inf():
    d = Diagnostics()
    field = np.array([1.0, np.inf, 3.0])
    result = d.functional_profile(field)
    assert result["n_inf"] == 1
    assert result["status"] == STATUS_FAIL


def test_functional_profile_with_inf_has_finite_json_values():
    """Audit amendment: inf in field must not propagate to output stats.
    Stats computed on finite values only, JSON strict must succeed."""
    d = Diagnostics()
    field = np.array([1.0, np.inf, 3.0])
    result = d.functional_profile(field)
    assert result["status"] == STATUS_FAIL
    assert np.isfinite(result["max"])
    assert np.isfinite(result["mean"])
    assert np.isfinite(result["min"])
    # JSON strict: allow_nan=False rejects nan/inf
    json.dumps(result, allow_nan=False)
    assert result["non_finite_policy"] == "finite_values_only"


def test_functional_profile_with_neg_inf_has_finite_json_values():
    d = Diagnostics()
    field = np.array([1.0, -np.inf, 3.0])
    result = d.functional_profile(field)
    assert result["status"] == STATUS_FAIL
    assert np.isfinite(result["min"])
    assert np.isfinite(result["max"])
    json.dumps(result, allow_nan=False)


def test_functional_profile_with_nan_and_inf_strict_json():
    d = Diagnostics()
    field = np.array([1.0, np.nan, np.inf, 3.0])
    result = d.functional_profile(field)
    assert result["status"] == STATUS_FAIL
    assert np.isfinite(result["mean"])
    json.dumps(result, allow_nan=False)


def test_functional_profile_all_nan():
    d = Diagnostics()
    field = np.array([np.nan, np.nan])
    result = d.functional_profile(field)
    assert result["status"] == STATUS_UNDEFINED


# ===== J/K/L. h_grad_activity_profile classifications =====

def test_h_grad_disjoint():
    """h_active and grad_active disjoint (B2-like signature)."""
    d = Diagnostics()
    h_act = np.array([1.0, 1.0, 0.0, 0.0])
    g_act = np.array([0.0, 0.0, 1.0, 1.0])
    result = d.h_grad_activity_profile(h_act, g_act, h_threshold=0.5, grad_threshold=0.5)
    assert result["grad_status"] == "DISJOINT"
    assert result["frac_intersection_h_grad"] == 0.0
    assert result["jaccard_h_grad"] == 0.0


def test_h_grad_co_active():
    d = Diagnostics()
    h_act = np.array([1.0, 1.0, 0.0])
    g_act = np.array([1.0, 0.0, 0.0])
    result = d.h_grad_activity_profile(h_act, g_act, h_threshold=0.5, grad_threshold=0.5)
    assert result["grad_status"] == "CO_ACTIVE"
    assert result["jaccard_h_grad"] == pytest.approx(0.5)


def test_h_grad_empty_grad():
    d = Diagnostics()
    h_act = np.array([1.0, 1.0, 0.0])
    g_act = np.array([0.0, 0.0, 0.0])
    result = d.h_grad_activity_profile(h_act, g_act, h_threshold=0.5, grad_threshold=0.5)
    assert result["grad_status"] == "EMPTY"


def test_h_grad_inactive():
    d = Diagnostics()
    h_act = np.zeros(5)
    g_act = np.zeros(5)
    result = d.h_grad_activity_profile(h_act, g_act, h_threshold=0.5, grad_threshold=0.5)
    assert result["grad_status"] == "INACTIVE"


def test_h_grad_no_h_active():
    d = Diagnostics()
    h_act = np.zeros(5)
    g_act = np.ones(5)
    result = d.h_grad_activity_profile(h_act, g_act, h_threshold=0.5, grad_threshold=0.5)
    assert result["grad_status"] == "NO_H_ACTIVE"


# ===== M. Exact P5bis-A fields present =====

def test_h_grad_profile_has_canonical_fields():
    """P5bis-A inheritance: canonical fields must be present."""
    d = Diagnostics()
    h_act = np.ones(5)
    g_act = np.ones(5)
    result = d.h_grad_activity_profile(h_act, g_act, h_threshold=0.5, grad_threshold=0.5)
    canonical = [
        "frac_h_active",
        "frac_grad_active",
        "frac_intersection_h_grad",
        "jaccard_h_grad",
        "grad_status",
    ]
    for k in canonical:
        assert k in result, f"missing canonical field: {k}"


def test_h_grad_shape_mismatch_raises():
    d = Diagnostics()
    with pytest.raises(ValueError, match="shape"):
        d.h_grad_activity_profile(np.ones(5), np.ones(7), h_threshold=0.5, grad_threshold=0.5)


# ===== N. b3_like_degeneracy true on constant =====

def test_b3_like_degeneracy_constant_field():
    d = Diagnostics()
    field = 5.0 * np.ones((5, 5, 5), dtype=np.float64)
    result = d.b3_like_degeneracy(field)
    assert result["b3_like_degenerate"] is True
    assert result["status"] == STATUS_WARN


# ===== O. b3_like_degeneracy false on varied =====

def test_b3_like_degeneracy_varied_field():
    d = Diagnostics()
    field = np.linspace(0.0, 10.0, 125).reshape((5, 5, 5))
    result = d.b3_like_degeneracy(field)
    assert result["b3_like_degenerate"] is False
    assert result["status"] == STATUS_PASS


def test_b3_like_degeneracy_reason():
    d = Diagnostics()
    field = np.full((10,), 3.0)
    result = d.b3_like_degeneracy(field)
    assert result["reason"] in (
        "low_variance", "low_range", "low_variance_and_low_range"
    )


# ===== P. instance_profile reads but does not mutate =====

def test_instance_profile_reads_module_ids():
    d = Diagnostics()
    inst = Instance(id="I", modules={
        "m0": make_module("m0"),
        "m1": make_module("m1"),
    })
    result = d.instance_profile(inst)
    assert result["n_modules"] == 2
    assert set(result["module_ids"]) == {"m0", "m1"}
    assert result["status"] == STATUS_PASS


def test_instance_profile_does_not_mutate():
    d = Diagnostics()
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})
    psi_snap = m0.state["psi"].copy()
    h_snap = m0.metric.h.copy()
    _ = d.instance_profile(inst)
    np.testing.assert_array_equal(m0.state["psi"], psi_snap)
    np.testing.assert_array_equal(m0.metric.h, h_snap)


def test_instance_profile_invalid_type():
    d = Diagnostics()
    with pytest.raises(TypeError, match="Instance"):
        d.instance_profile("not an instance")


# ===== Q. contribution_profile counts =====

def test_contribution_profile_counts_by_kind():
    d = Diagnostics()
    c1 = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=np.zeros((5, 5, 5), dtype=np.float64),
        conservative=True, mass_delta=0.0,
    )
    c2 = CouplingContribution(
        target_id="m0", source_id="m2",
        kind="source", value=np.zeros((5, 5, 5), dtype=np.float64),
        conservative=False, mass_delta=0.5,
    )
    c3 = CouplingContribution(
        target_id="m1", source_id="m0",
        kind="none", value=(),
        conservative=True, mass_delta=0.0,
    )
    result = d.contribution_profile((c1, c2, c3))
    assert result["n_contributions"] == 3
    assert result["count_by_kind"] == {"source": 2, "none": 1}
    assert result["count_by_target"] == {"m0": 2, "m1": 1}
    assert result["n_conservative"] == 2
    assert result["n_non_conservative"] == 1
    assert result["declared_mass_delta_sum"] == pytest.approx(0.5)


# ===== R. aggregated_coupling_profile =====

def test_aggregated_coupling_profile():
    d = Diagnostics()
    geom = GeometryOperator(make_grid((5, 5, 5)))
    agg = CouplingAggregator(geometry=geom)
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source",
        value=np.zeros((5, 5, 5), dtype=np.float64),
        conservative=True, mass_delta=0.0,
    )
    aggregated = agg.aggregate((contrib,), policy="conservative")
    result = d.aggregated_coupling_profile(aggregated)
    assert "m0" in result["targets"]
    assert result["mass_delta_by_target"]["m0"] == 0.0
    assert result["policy_by_target"]["m0"] == "conservative"
    assert result["n_contributions_by_target"]["m0"] == 1


# ===== S. V5-2 stubs raise NotImplementedError =====

def test_R_ij_stub_raises():
    d = Diagnostics()
    with pytest.raises(NotImplementedError, match="R_ij"):
        d.R_ij_matrix()


def test_novelty_transmitted_stub_raises():
    d = Diagnostics()
    with pytest.raises(NotImplementedError, match="novelty"):
        d.novelty_transmitted()


def test_mass_delta_by_coupling_kind_stub_raises():
    d = Diagnostics()
    with pytest.raises(NotImplementedError, match="mass_delta"):
        d.mass_delta_by_coupling_kind()


def test_circulation_stub_raises():
    d = Diagnostics()
    with pytest.raises(NotImplementedError, match="circulation"):
        d.circulation()


# ===== T. No forbidden methods =====

def test_diagnostics_no_forbidden_methods():
    d = Diagnostics()
    forbidden = [
        "step", "aggregate", "build_contexts", "pairwise_coupling",
        "evolve", "gradient", "flux", "divergence", "apply",
    ]
    for n in forbidden:
        assert not hasattr(d, n), f"forbidden method: {n}"


# ===== U. No mutation of input arrays =====

def test_diagnostics_no_mutation():
    d = Diagnostics()
    field = np.array([1.0, 2.0, 3.0])
    snap = field.copy()
    _ = d.mass(field)
    _ = d.positivity(field)
    _ = d.underflow(field)
    _ = d.functional_profile(field)
    _ = d.b3_like_degeneracy(field)
    np.testing.assert_array_equal(field, snap)


# ===== V. No cache / no history =====

def test_diagnostics_no_history():
    d = Diagnostics()
    forbidden = ["history", "_history", "_cache", "_last_result"]
    for n in forbidden:
        assert not hasattr(d, n), f"forbidden attr: {n}"


def test_diagnostics_slots():
    d = Diagnostics()
    with pytest.raises(AttributeError):
        d._cache = {}


# ===== Robustesse NaN/Inf (audit ajout 1) =====

def test_mass_handles_nan_returns_undefined():
    d = Diagnostics()
    field = np.array([1.0, np.nan, 3.0])
    result = d.mass(field)
    assert result["status"] == STATUS_UNDEFINED


def test_mass_conservation_with_nan_returns_undefined():
    d = Diagnostics()
    before = np.array([1.0, np.nan, 3.0])
    after = np.array([1.0, 2.0, 3.0])
    result = d.mass_conservation(before, after)
    assert result["status"] == STATUS_UNDEFINED


def test_positivity_with_nan_returns_undefined():
    d = Diagnostics()
    field = np.array([1.0, np.nan, -1.0])
    result = d.positivity(field)
    assert result["status"] == STATUS_UNDEFINED


def test_underflow_with_nan_returns_undefined():
    d = Diagnostics()
    field = np.array([1.0, np.nan, 1e-301])
    result = d.underflow(field)
    assert result["status"] == STATUS_UNDEFINED


def test_b3_with_nan_returns_undefined():
    d = Diagnostics()
    field = np.array([1.0, np.nan, 3.0])
    result = d.b3_like_degeneracy(field)
    assert result["status"] == STATUS_UNDEFINED


def test_h_grad_with_nan_returns_undefined():
    d = Diagnostics()
    h_act = np.array([1.0, np.nan, 1.0])
    g_act = np.array([1.0, 1.0, 1.0])
    result = d.h_grad_activity_profile(h_act, g_act, h_threshold=0.5, grad_threshold=0.5)
    assert result["status"] == STATUS_UNDEFINED


# ===== JSON-serializable outputs (audit ajout 2) =====

def test_all_outputs_json_serializable():
    d = Diagnostics()
    field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    outputs = [
        d.mass(field),
        d.mass_conservation(field, field),
        d.positivity(field),
        d.underflow(field),
        d.functional_profile(field),
        d.b3_like_degeneracy(field),
        d.h_grad_activity_profile(field, field, h_threshold=0.5, grad_threshold=0.5),
    ]
    for o in outputs:
        s = json.dumps(o)  # must succeed
        assert isinstance(s, str)


def test_jaccard_is_native_float_type():
    d = Diagnostics()
    h_act = np.ones(5)
    g_act = np.ones(5)
    result = d.h_grad_activity_profile(h_act, g_act, h_threshold=0.5, grad_threshold=0.5)
    # Native python float, not numpy scalar
    assert type(result["jaccard_h_grad"]) is float
    assert type(result["frac_h_active"]) is float
    assert type(result["frac_grad_active"]) is float
    assert type(result["frac_intersection_h_grad"]) is float


def test_aggregated_profile_json_serializable():
    d = Diagnostics()
    geom = GeometryOperator(make_grid((5, 5, 5)))
    agg = CouplingAggregator(geometry=geom)
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source",
        value=np.zeros((5, 5, 5), dtype=np.float64),
        conservative=True, mass_delta=0.0,
    )
    aggregated = agg.aggregate((contrib,), policy="conservative")
    profile = d.aggregated_coupling_profile(aggregated)
    s = json.dumps(profile)
    assert isinstance(s, str)


# ===== State contributions / aggregated invalid input =====

def test_contribution_profile_invalid_input():
    d = Diagnostics()
    with pytest.raises(TypeError, match="tuple"):
        d.contribution_profile([])


def test_contribution_profile_invalid_element():
    d = Diagnostics()
    with pytest.raises(TypeError, match="CouplingContribution"):
        d.contribution_profile(("not a contribution",))


def test_aggregated_profile_invalid_input():
    d = Diagnostics()
    with pytest.raises(TypeError, match="dict"):
        d.aggregated_coupling_profile("not a dict")


# ===== File isolation =====

def test_diagnostics_file_does_not_import_active_objects():
    """diagnostics.py may import State, Instance, CouplingContribution,
    AggregatedCoupling (passive types) but NOT active operators."""
    import mcq_v5.diagnostics as diag_mod
    src = Path(diag_mod.__file__).read_text()
    forbidden_active_imports = [
        "from .geometry import GeometryOperator",
        "from .coupling_scheduler import CouplingScheduler",
        "from .coupling_operator import CouplingOperator",
        "from .coupling_operator import NullCouplingOperator",
        "from .solver import",
    ]
    for pat in forbidden_active_imports:
        assert pat not in src, f"forbidden import: {pat}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
