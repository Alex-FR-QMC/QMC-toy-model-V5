# -*- coding: utf-8 -*-
"""Tests for mcq_v5.coupling_aggregator.

Critère étape 11: routes CouplingContribution by kind, aggregates by
target_id under explicit mass policies, returns immutable
AggregatedCoupling objects, no Instance/Module/Scheduler/Context/Operator
access.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.grid import make_grid
from mcq_v5.geometry import GeometryOperator
from mcq_v5.coupling_operator import CouplingContribution
from mcq_v5.coupling_aggregator import CouplingAggregator, AggregatedCoupling


# ===== Helpers =====

def make_geom(shape=(5, 5, 5)):
    return GeometryOperator(make_grid(shape, dx=1.0))


def make_source_contrib(target, source, value_array):
    return CouplingContribution(
        target_id=target, source_id=source,
        kind="source", value=value_array,
        conservative=True, mass_delta=float(np.sum(value_array)),
    )


def make_zero_source(target, source, shape=(5, 5, 5)):
    return CouplingContribution(
        target_id=target, source_id=source,
        kind="source", value=np.zeros(shape, dtype=np.float64),
        conservative=True, mass_delta=0.0,
    )


# ===== A. Construction =====

def test_aggregator_construction_valid():
    geom = make_geom()
    agg = CouplingAggregator(geometry=geom)
    assert agg.geometry is geom
    assert agg.mass_tolerance == 1e-12
    assert agg.source_bound is None


def test_aggregator_with_policy_params():
    agg = CouplingAggregator(
        geometry=make_geom(),
        mass_tolerance=1e-10,
        source_bound=0.5,
        metadata={"k": "v"},
    )
    assert agg.mass_tolerance == 1e-10
    assert agg.source_bound == 0.5


# ===== B. Invalid geometry type =====

def test_aggregator_invalid_geometry_type():
    with pytest.raises(TypeError, match="GeometryOperator"):
        CouplingAggregator(geometry="not a geom")


def test_aggregator_invalid_mass_tolerance():
    with pytest.raises(ValueError, match="mass_tolerance"):
        CouplingAggregator(geometry=make_geom(), mass_tolerance=-0.1)


def test_aggregator_invalid_source_bound():
    with pytest.raises(ValueError, match="source_bound"):
        CouplingAggregator(geometry=make_geom(), source_bound=-1.0)


def test_aggregator_invalid_metadata():
    with pytest.raises(TypeError, match="metadata"):
        CouplingAggregator(geometry=make_geom(), metadata="not a dict")


# ===== C. Empty contributions =====

def test_aggregate_empty_returns_empty_dict():
    agg = CouplingAggregator(geometry=make_geom())
    result = agg.aggregate((), policy="conservative")
    assert result == {}


# ===== D. Zero source => zero RHS =====

def test_aggregate_zero_source():
    agg = CouplingAggregator(geometry=make_geom())
    contrib = make_zero_source("m0", "m1")
    result = agg.aggregate((contrib,), policy="conservative")
    assert "m0" in result
    np.testing.assert_array_equal(result["m0"].rhs, np.zeros((5, 5, 5)))
    assert result["m0"].mass_delta == 0.0


# ===== E. Non-zero source with source_allowed =====

def test_aggregate_nonzero_source_source_allowed():
    agg = CouplingAggregator(geometry=make_geom())
    src = np.zeros((5, 5, 5), dtype=np.float64)
    src[2, 2, 2] = 1.0
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=src,
        conservative=False, mass_delta=1.0,
    )
    result = agg.aggregate((contrib,), policy="source_allowed")
    assert result["m0"].mass_delta == 1.0


# ===== F. Non-zero source with conservative -> rejected =====

def test_aggregate_nonzero_source_conservative_rejected():
    agg = CouplingAggregator(geometry=make_geom())
    src = np.zeros((5, 5, 5), dtype=np.float64)
    src[2, 2, 2] = 1.0
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=src,
        conservative=True, mass_delta=1.0,  # conservative=True but sum=1.0
    )
    with pytest.raises(ValueError, match="conservative"):
        agg.aggregate((contrib,), policy="conservative")


def test_conservative_inflexible_rejects_non_conservative_flag():
    """Audit ajout 2: any conservative=False under 'conservative' policy
    raises immediately, even if sum happens to be zero."""
    agg = CouplingAggregator(geometry=make_geom())
    # A non-conservative contribution that sums to zero (e.g. +5 -5)
    src = np.zeros((5, 5, 5), dtype=np.float64)
    src[0, 0, 0] = 5.0
    src[4, 4, 4] = -5.0
    assert np.sum(src) == 0.0
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=src,
        conservative=False,  # explicitly not conservative
        mass_delta=0.0,
    )
    with pytest.raises(ValueError, match="conservative=False"):
        agg.aggregate((contrib,), policy="conservative")


# ===== G. bounded_source under bound =====

def test_aggregate_bounded_source_under_bound():
    agg = CouplingAggregator(geometry=make_geom(), source_bound=1.0)
    src = np.zeros((5, 5, 5), dtype=np.float64)
    src[2, 2, 2] = 0.5
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=src,
        conservative=False, mass_delta=0.5,
    )
    result = agg.aggregate((contrib,), policy="bounded_source")
    assert result["m0"].mass_delta == 0.5
    assert result["m0"].diagnostics["source_bound"] == 1.0


# ===== H. bounded_source above bound =====

def test_aggregate_bounded_source_above_bound_rejected():
    agg = CouplingAggregator(geometry=make_geom(), source_bound=0.3)
    src = np.zeros((5, 5, 5), dtype=np.float64)
    src[2, 2, 2] = 1.0
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=src,
        conservative=False, mass_delta=1.0,
    )
    with pytest.raises(ValueError, match="bounded_source"):
        agg.aggregate((contrib,), policy="bounded_source")


# ===== I. bounded_source without source_bound =====

def test_aggregate_bounded_source_without_bound_rejected():
    agg = CouplingAggregator(geometry=make_geom())  # source_bound=None
    contrib = make_zero_source("m0", "m1")
    with pytest.raises(ValueError, match="source_bound"):
        agg.aggregate((contrib,), policy="bounded_source")


# ===== J. diagnostic_only never raises =====

def test_aggregate_diagnostic_only_accepts_non_conservative():
    agg = CouplingAggregator(geometry=make_geom())
    src = 5.0 * np.ones((5, 5, 5), dtype=np.float64)
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=src,
        conservative=False, mass_delta=625.0,
    )
    result = agg.aggregate((contrib,), policy="diagnostic_only")
    assert "m0" in result
    d = result["m0"].diagnostics
    assert d["would_violate_conservative"] is True


def test_aggregate_diagnostic_only_flags_bounded_violation():
    agg = CouplingAggregator(geometry=make_geom(), source_bound=0.1)
    src = np.zeros((5, 5, 5), dtype=np.float64)
    src[2, 2, 2] = 1.0
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=src,
        conservative=False, mass_delta=1.0,
    )
    result = agg.aggregate((contrib,), policy="diagnostic_only")
    assert result["m0"].diagnostics["would_violate_bounded_source"] is True


# ===== K. Zero flux => zero RHS =====

def test_aggregate_zero_flux():
    agg = CouplingAggregator(geometry=make_geom())
    fluxes = (
        np.zeros((4, 5, 5), dtype=np.float64),
        np.zeros((5, 4, 5), dtype=np.float64),
        np.zeros((5, 5, 4), dtype=np.float64),
    )
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="flux", value=fluxes,
        conservative=True, mass_delta=0.0,
    )
    result = agg.aggregate((contrib,), policy="conservative")
    np.testing.assert_array_equal(result["m0"].rhs, np.zeros((5, 5, 5)))


# ===== L. Constant flux 1D matches divergence =====

def test_aggregate_constant_flux_1d():
    """In 1D, J=ones((4,)) on grid (5,) => RHS = [-1, 0, 0, 0, +1]."""
    geom = GeometryOperator(make_grid((5,), dx=1.0))
    agg = CouplingAggregator(geometry=geom)
    J = np.ones((4,), dtype=np.float64)
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="flux", value=(J,),
        conservative=True, mass_delta=0.0,
    )
    result = agg.aggregate((contrib,), policy="conservative")
    expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(result["m0"].rhs, expected)


# ===== M. Mixed = source + flux =====

def test_aggregate_mixed_combines_source_and_flux():
    """source=[1,0,0,0,0], J=[1,1,1,1] => flux_rhs=[-1,0,0,0,1] =>
    mixed_rhs = [0,0,0,0,1]."""
    geom = GeometryOperator(make_grid((5,), dx=1.0))
    agg = CouplingAggregator(geometry=geom)
    source = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    fluxes = (np.ones((4,), dtype=np.float64),)
    # mass_delta: source sums to 1, divergence sums to 0, total = 1
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="mixed", value=(source, fluxes),
        conservative=False, mass_delta=1.0,
    )
    result = agg.aggregate((contrib,), policy="source_allowed")
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(result["m0"].rhs, expected)


# ===== N. Multi-contributions same target =====

def test_aggregate_multiple_contributions_same_target():
    agg = CouplingAggregator(geometry=make_geom())
    src1 = np.zeros((5, 5, 5), dtype=np.float64)
    src1[2, 2, 2] = 1.0
    src2 = np.zeros((5, 5, 5), dtype=np.float64)
    src2[0, 0, 0] = 0.5
    c1 = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=src1,
        conservative=False, mass_delta=1.0,
    )
    c2 = CouplingContribution(
        target_id="m0", source_id="m2",
        kind="source", value=src2,
        conservative=False, mass_delta=0.5,
    )
    result = agg.aggregate((c1, c2), policy="source_allowed")
    assert "m0" in result
    assert result["m0"].n_contributions == 2
    assert result["m0"].diagnostics["source_ids"] == ("m1", "m2")
    # Combined sum: 1.5
    assert result["m0"].mass_delta == pytest.approx(1.5)


# ===== O. Multi-targets => multiple AggregatedCouplings =====

def test_aggregate_multi_targets():
    agg = CouplingAggregator(geometry=make_geom())
    src_m0 = np.zeros((5, 5, 5), dtype=np.float64)
    src_m0[0, 0, 0] = 1.0
    src_m1 = np.zeros((5, 5, 5), dtype=np.float64)
    src_m1[1, 1, 1] = 2.0
    c1 = CouplingContribution(
        target_id="m0", source_id="m2",
        kind="source", value=src_m0,
        conservative=False, mass_delta=1.0,
    )
    c2 = CouplingContribution(
        target_id="m1", source_id="m2",
        kind="source", value=src_m1,
        conservative=False, mass_delta=2.0,
    )
    result = agg.aggregate((c1, c2), policy="source_allowed")
    assert set(result.keys()) == {"m0", "m1"}
    assert result["m0"].mass_delta == 1.0
    assert result["m1"].mass_delta == 2.0


# ===== P. RHS read-only =====

def test_aggregated_rhs_read_only():
    agg = CouplingAggregator(geometry=make_geom())
    contrib = make_zero_source("m0", "m1")
    result = agg.aggregate((contrib,), policy="conservative")
    rhs = result["m0"].rhs
    assert rhs.flags["WRITEABLE"] is False
    with pytest.raises(ValueError):
        rhs[0, 0, 0] = 1.0


# ===== Q. Diagnostics read-only =====

def test_aggregated_diagnostics_read_only():
    agg = CouplingAggregator(geometry=make_geom())
    contrib = make_zero_source("m0", "m1")
    result = agg.aggregate((contrib,), policy="conservative")
    with pytest.raises(TypeError):
        result["m0"].diagnostics["x"] = 1


def test_aggregated_diagnostics_content():
    agg = CouplingAggregator(geometry=make_geom())
    contrib = make_zero_source("m0", "m1")
    result = agg.aggregate((contrib,), policy="conservative")
    d = result["m0"].diagnostics
    assert d["policy"] == "conservative"
    assert d["n_contributions"] == 1
    assert d["source_ids"] == ("m1",)
    assert d["mass_policy_violation"] is False


# ===== R. Invalid policy =====

def test_aggregate_invalid_policy():
    agg = CouplingAggregator(geometry=make_geom())
    contrib = make_zero_source("m0", "m1")
    with pytest.raises(ValueError, match="policy"):
        agg.aggregate((contrib,), policy="random_string")


# ===== S. Shape mismatch source =====

def test_aggregate_source_shape_mismatch():
    geom = GeometryOperator(make_grid((5, 5, 5)))
    agg = CouplingAggregator(geometry=geom)
    # Build a contribution with wrong shape (would-be-source)
    # CouplingContribution doesn't validate shape against grid, so this
    # should be rejected by the aggregator's _route step.
    bad_src = np.zeros((7, 7, 7), dtype=np.float64)
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=bad_src,
        conservative=True, mass_delta=0.0,
    )
    with pytest.raises(ValueError, match="shape"):
        agg.aggregate((contrib,), policy="conservative")


# ===== T. Invalid contribution type in tuple =====

def test_aggregate_invalid_contribution_in_tuple():
    agg = CouplingAggregator(geometry=make_geom())
    with pytest.raises(TypeError, match="CouplingContribution"):
        agg.aggregate(("not a contribution",), policy="conservative")


def test_aggregate_contributions_must_be_tuple():
    agg = CouplingAggregator(geometry=make_geom())
    contrib = make_zero_source("m0", "m1")
    with pytest.raises(TypeError, match="tuple"):
        agg.aggregate([contrib], policy="conservative")


# ===== U. Aggregator has no scheduler/operator methods =====

def test_aggregator_no_forbidden_methods():
    agg = CouplingAggregator(geometry=make_geom())
    forbidden = [
        "pairwise_coupling", "build_contexts", "admissible_pairs",
        "step", "evolve", "compute_coupling", "compute_overlap",
    ]
    for n in forbidden:
        assert not hasattr(agg, n), f"forbidden method: {n}"


# ===== V. File does not import instance/module/scheduler/context/operator =====

def test_aggregator_file_isolation():
    import mcq_v5.coupling_aggregator as agg_mod
    src = Path(agg_mod.__file__).read_text()
    forbidden_imports = [
        "from .instance import",
        "from .module import Module",  # ModuleId may be from coupling_operator
        "from .coupling_scheduler import",
        "from .coupling_context import",
        "from mcq_v5.instance import",
        "from mcq_v5.coupling_scheduler import",
        "from mcq_v5.coupling_context import",
    ]
    for pat in forbidden_imports:
        assert pat not in src, f"forbidden import: {pat}"


def test_aggregator_namespace_no_forbidden_names():
    import mcq_v5.coupling_aggregator as agg_mod
    forbidden = ["Instance", "Module", "CouplingScheduler", "CouplingContext"]
    for n in forbidden:
        assert not hasattr(agg_mod, n), f"forbidden in namespace: {n}"


# ===== W. No cache / __slots__ =====

def test_aggregator_uses_slots():
    agg = CouplingAggregator(geometry=make_geom())
    with pytest.raises(AttributeError):
        agg._last_contributions = None
    with pytest.raises(AttributeError):
        agg._cache = {}


def test_aggregator_no_temporal_attributes():
    agg = CouplingAggregator(geometry=make_geom())
    forbidden = [
        "history", "_history", "_last_result", "_contexts",
        "_instance",
    ]
    for n in forbidden:
        assert not hasattr(agg, n)


# ===== Audit ajout 1: NaN/Inf final guard =====

def test_aggregate_rejects_nan_in_result():
    """If a contribution contains NaN, the aggregated RHS must be detected
    as NaN/Inf and rejected before locking."""
    agg = CouplingAggregator(geometry=make_geom())
    bad = np.zeros((5, 5, 5), dtype=np.float64)
    bad[2, 2, 2] = np.nan
    # Build a contribution with mass_delta finite but content NaN.
    # CouplingContribution doesn't currently check for NaN in source value;
    # the aggregator must catch it after summation.
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=bad,
        conservative=False, mass_delta=0.0,  # spurious
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        agg.aggregate((contrib,), policy="source_allowed")


def test_aggregate_rejects_inf_in_result():
    agg = CouplingAggregator(geometry=make_geom())
    bad = np.zeros((5, 5, 5), dtype=np.float64)
    bad[1, 1, 1] = np.inf
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="source", value=bad,
        conservative=False, mass_delta=0.0,
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        agg.aggregate((contrib,), policy="source_allowed")


# ===== Audit ajout 3: only target_ids with contributions appear =====

def test_aggregator_only_returns_targets_with_contributions():
    """If only m0 receives contributions, m1 and m2 must NOT appear in
    the returned dict (audit ajout 3).
    """
    agg = CouplingAggregator(geometry=make_geom())
    contrib = make_zero_source("m0", "m1")
    result = agg.aggregate((contrib,), policy="conservative")
    assert set(result.keys()) == {"m0"}
    assert "m1" not in result
    assert "m2" not in result


# ===== Conservative passes when conservative=True and sum<=tol =====

def test_conservative_passes_when_truly_conservative():
    """A truly conservative flux (sum=0) passes the conservative policy."""
    geom = GeometryOperator(make_grid((5,), dx=1.0))
    agg = CouplingAggregator(geometry=geom)
    J = np.ones((4,), dtype=np.float64)  # constant flux => RHS sum = 0
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="flux", value=(J,),
        conservative=True, mass_delta=0.0,
    )
    result = agg.aggregate((contrib,), policy="conservative")
    assert abs(result["m0"].mass_delta) < 1e-12


# ===== None kind contribution =====

def test_aggregate_none_kind_contributes_zero():
    agg = CouplingAggregator(geometry=make_geom())
    contrib = CouplingContribution(
        target_id="m0", source_id="m1",
        kind="none", value=(),
        conservative=True, mass_delta=0.0,
    )
    result = agg.aggregate((contrib,), policy="conservative")
    np.testing.assert_array_equal(result["m0"].rhs, np.zeros((5, 5, 5)))


# ===== Mass policy stored in AggregatedCoupling =====

def test_aggregated_records_policy():
    agg = CouplingAggregator(geometry=make_geom())
    contrib = make_zero_source("m0", "m1")
    result = agg.aggregate((contrib,), policy="source_allowed")
    assert result["m0"].mass_policy == "source_allowed"


# ===== AggregatedCoupling immutability =====

def test_aggregated_coupling_frozen():
    rhs = np.zeros((5, 5, 5), dtype=np.float64)
    agg = AggregatedCoupling(
        target_id="m0", rhs=rhs, mass_delta=0.0,
        mass_policy="conservative", n_contributions=1,
    )
    with pytest.raises(Exception):
        agg.target_id = "modified"


def test_aggregated_coupling_dtype_validation():
    rhs = np.zeros((5, 5, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        AggregatedCoupling(
            target_id="m0", rhs=rhs, mass_delta=0.0,
            mass_policy="conservative", n_contributions=1,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
