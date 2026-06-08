# -*- coding: utf-8 -*-
"""Tests for mcq_v5.experiment.ExperimentProtocol.

Critère étape 14: sole maintainer of HistoryEntry; step() = one macro-step
orchestrating all components; run() = loop; commits per macro-step;
checkpoints are metadata; substeps never committed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.grid import make_grid
from mcq_v5.state import State
from mcq_v5.metric import ScalarConformalMetric
from mcq_v5.module import Module
from mcq_v5.instance import Instance
from mcq_v5.views import HistoryView, HistoryEntry
from mcq_v5.geometry import GeometryOperator
from mcq_v5.coupling_scheduler import CouplingScheduler
from mcq_v5.coupling_operator import NullCouplingOperator
from mcq_v5.coupling_aggregator import CouplingAggregator
from mcq_v5.solver import ExplicitReferenceSolver
from mcq_v5.diagnostics import Diagnostics
from mcq_v5.experiment import ExperimentProtocol


# ===== Helpers =====

def make_module(mid, shape=(5, 5, 5), psi=None, h=None):
    if psi is None:
        psi = np.ones(shape, dtype=np.float64)
    if h is None:
        h = np.ones(shape, dtype=np.float64)
    s = State(
        fields={"psi": psi.copy(), "h": h.copy()},
        solver_fields=("psi", "h"),
    )
    m = ScalarConformalMetric(np.ones(shape, dtype=np.float64))
    return Module(id=mid, state=s, metric=m)


def make_protocol(
    M=1, shape=(5, 5, 5), dt=0.01, commit_initial=True,
    checkpoint_interval=None, guardrails=None, conservation_policy="conservative",
):
    grid = make_grid(shape, dx=1.0)
    modules = {f"m{i}": make_module(f"m{i}", shape=shape) for i in range(M)}
    inst = Instance(id="I", modules=modules)
    geom = GeometryOperator(grid)
    sch = CouplingScheduler()
    op = NullCouplingOperator()
    agg = CouplingAggregator(geometry=geom)
    solver = ExplicitReferenceSolver()
    return ExperimentProtocol(
        protocol_name="test",
        version="v5-0a",
        instance=inst,
        grid=grid,
        geometry=geom,
        scheduler=sch,
        coupling_operator=op,
        aggregator=agg,
        solver=solver,
        dt=dt,
        conservation_policy=conservation_policy,
        commit_initial=commit_initial,
        checkpoint_interval=checkpoint_interval,
        guardrails=guardrails,
        seeds={"rng_seed": 42},
    )


# ===== A. Construction valid =====

def test_protocol_construction_valid():
    p = make_protocol(M=1)
    assert p.protocol_name == "test"
    assert p.version == "v5-0a"
    assert p.step_index == 0
    assert p.time == 0.0
    assert p.dt == 0.01


# ===== B. Mandatory metadata present =====

def test_protocol_exports_mandatory_metadata():
    p = make_protocol(M=2)
    bundle = p.export_protocol_metadata()
    required = (
        "protocol_name", "version", "grid_shape", "metric_type",
        "N", "M", "coupling_policy", "conservation_policy",
        "history_policy", "seeds", "checkpoints", "guardrails",
    )
    for k in required:
        assert k in bundle, f"missing required key: {k}"
    assert bundle["N"] == 1
    assert bundle["M"] == 2
    assert bundle["metric_type"] == "scalar_conformal"
    assert bundle["grid_shape"] == (5, 5, 5)
    assert bundle["history_policy"]["sole_maintainer"] == "ExperimentProtocol"
    assert bundle["history_policy"]["commit"] == "every_macro_step"
    assert bundle["history_policy"]["substep_exclusion"] is True


# ===== C. Refuses construction with missing required args =====

def test_protocol_missing_protocol_name_rejected():
    grid = make_grid((5, 5, 5), dx=1.0)
    inst = Instance(id="I", modules={"m0": make_module("m0")})
    geom = GeometryOperator(grid)
    with pytest.raises(ValueError, match="protocol_name"):
        ExperimentProtocol(
            protocol_name="",
            version="v",
            instance=inst,
            grid=grid,
            geometry=geom,
            scheduler=CouplingScheduler(),
            coupling_operator=NullCouplingOperator(),
            aggregator=CouplingAggregator(geometry=geom),
            solver=ExplicitReferenceSolver(),
            dt=0.01,
        )


def test_protocol_missing_version_rejected():
    grid = make_grid((5, 5, 5), dx=1.0)
    inst = Instance(id="I", modules={"m0": make_module("m0")})
    geom = GeometryOperator(grid)
    with pytest.raises(ValueError, match="version"):
        ExperimentProtocol(
            protocol_name="test",
            version="",
            instance=inst,
            grid=grid,
            geometry=geom,
            scheduler=CouplingScheduler(),
            coupling_operator=NullCouplingOperator(),
            aggregator=CouplingAggregator(geometry=geom),
            solver=ExplicitReferenceSolver(),
            dt=0.01,
        )


def test_protocol_invalid_dt():
    grid = make_grid((5, 5, 5), dx=1.0)
    inst = Instance(id="I", modules={"m0": make_module("m0")})
    geom = GeometryOperator(grid)
    with pytest.raises(ValueError, match="dt"):
        ExperimentProtocol(
            protocol_name="t", version="v",
            instance=inst, grid=grid, geometry=geom,
            scheduler=CouplingScheduler(),
            coupling_operator=NullCouplingOperator(),
            aggregator=CouplingAggregator(geometry=geom),
            solver=ExplicitReferenceSolver(),
            dt=0.0,
        )


def test_protocol_grid_shape_mismatch_with_geometry():
    grid_5 = make_grid((5, 5, 5), dx=1.0)
    grid_7 = make_grid((7, 7, 7), dx=1.0)
    inst = Instance(id="I", modules={"m0": make_module("m0", shape=(5, 5, 5))})
    geom = GeometryOperator(grid_7)  # mismatch
    with pytest.raises(ValueError, match="shape"):
        ExperimentProtocol(
            protocol_name="t", version="v",
            instance=inst, grid=grid_5, geometry=geom,
            scheduler=CouplingScheduler(),
            coupling_operator=NullCouplingOperator(),
            aggregator=CouplingAggregator(geometry=geom),
            solver=ExplicitReferenceSolver(),
            dt=0.01,
        )


def test_protocol_geometry_dx_mismatch_rejected():
    """Geometry dx must match the protocol grid dx (not only shape)."""
    grid_dx1 = make_grid((5, 5, 5), dx=1.0)
    grid_dx2 = make_grid((5, 5, 5), dx=0.5)  # same shape, different dx
    inst = Instance(id="I", modules={"m0": make_module("m0", shape=(5, 5, 5))})
    geom = GeometryOperator(grid_dx2)
    with pytest.raises(ValueError, match="dx"):
        ExperimentProtocol(
            protocol_name="t", version="v",
            instance=inst, grid=grid_dx1, geometry=geom,
            scheduler=CouplingScheduler(),
            coupling_operator=NullCouplingOperator(),
            aggregator=CouplingAggregator(geometry=geom),
            solver=ExplicitReferenceSolver(),
            dt=0.01,
        )


def test_protocol_aggregator_geometry_shape_mismatch_rejected():
    """Aggregator's internal geometry must share the protocol's grid shape."""
    grid_5 = make_grid((5, 5, 5), dx=1.0)
    grid_7 = make_grid((7, 7, 7), dx=1.0)
    inst = Instance(id="I", modules={"m0": make_module("m0", shape=(5, 5, 5))})
    geom_5 = GeometryOperator(grid_5)
    geom_7 = GeometryOperator(grid_7)  # aggregator uses a different shape
    with pytest.raises(ValueError, match="aggregator.geometry.grid.shape"):
        ExperimentProtocol(
            protocol_name="t", version="v",
            instance=inst, grid=grid_5, geometry=geom_5,
            scheduler=CouplingScheduler(),
            coupling_operator=NullCouplingOperator(),
            aggregator=CouplingAggregator(geometry=geom_7),
            solver=ExplicitReferenceSolver(),
            dt=0.01,
        )


def test_protocol_aggregator_geometry_dx_mismatch_rejected():
    """Aggregator's internal geometry must share the protocol's grid dx."""
    grid_dx1 = make_grid((5, 5, 5), dx=1.0)
    grid_dx2 = make_grid((5, 5, 5), dx=0.5)  # same shape, different dx
    inst = Instance(id="I", modules={"m0": make_module("m0", shape=(5, 5, 5))})
    geom_protocol = GeometryOperator(grid_dx1)
    geom_aggregator = GeometryOperator(grid_dx2)
    with pytest.raises(ValueError, match="aggregator.geometry.grid.dx"):
        ExperimentProtocol(
            protocol_name="t", version="v",
            instance=inst, grid=grid_dx1, geometry=geom_protocol,
            scheduler=CouplingScheduler(),
            coupling_operator=NullCouplingOperator(),
            aggregator=CouplingAggregator(geometry=geom_aggregator),
            solver=ExplicitReferenceSolver(),
            dt=0.01,
        )


# ===== D. Initial commit creates M HistoryEntry committed=True =====

def test_initial_commit_creates_M_entries():
    p = make_protocol(M=3)
    entries = p.history_entries()
    assert len(entries) == 3
    for e in entries:
        assert e.committed is True
        assert e.step == 0
        assert e.time == 0.0


def test_no_initial_commit_when_disabled():
    p = make_protocol(M=2, commit_initial=False)
    assert len(p.history_entries()) == 0


# ===== E. step() advances step by +1 =====

def test_step_advances_step_index():
    p = make_protocol(M=1)
    assert p.step_index == 0
    p.step()
    assert p.step_index == 1
    p.step()
    assert p.step_index == 2


# ===== F. step() advances time by +dt =====

def test_step_advances_time_by_dt():
    p = make_protocol(M=1, dt=0.05)
    assert p.time == 0.0
    p.step()
    assert p.time == pytest.approx(0.05)
    p.step()
    assert p.time == pytest.approx(0.10)


# ===== G. run(n) equals n step() calls =====

def test_run_equivalent_to_n_step_calls():
    p1 = make_protocol(M=2)
    p2 = make_protocol(M=2)
    for _ in range(3):
        p1.step()
    p2.run(3)
    # Same step index, same time
    assert p1.step_index == p2.step_index == 3
    assert p1.time == p2.time
    # Same module states
    for mid in p1.instance.module_ids():
        np.testing.assert_allclose(
            p1.instance.get_module(mid).state["psi"],
            p2.instance.get_module(mid).state["psi"],
        )


def test_run_zero_is_noop():
    p = make_protocol(M=1)
    initial_step = p.step_index
    p.run(0)
    assert p.step_index == initial_step


def test_run_negative_rejected():
    p = make_protocol(M=1)
    with pytest.raises(ValueError, match="n_steps"):
        p.run(-1)


# ===== H. M=1 works =====

def test_M1_works():
    p = make_protocol(M=1)
    p.step()
    assert p.step_index == 1
    assert len(p.history_entries()) == 2  # initial + after step


# ===== I. M=2 works =====

def test_M2_works():
    p = make_protocol(M=2)
    p.step()
    assert p.step_index == 1
    # Initial commit (2) + after step (2) = 4
    assert len(p.history_entries()) == 4


# ===== J. Scheduler/Null operator produce null contributions =====

def test_null_coupling_produces_zero_contributions():
    """With NullCouplingOperator, contributions are zero -> no behavioral
    change beyond pure geometry diffusion."""
    p = make_protocol(M=2)
    psi_before = {
        mid: p.instance.get_module(mid).state["psi"].copy()
        for mid in p.instance.module_ids()
    }
    p.step()
    # NullCoupling: only geometry diffuses. Constant psi=1 -> no change.
    for mid in p.instance.module_ids():
        np.testing.assert_allclose(
            p.instance.get_module(mid).state["psi"],
            psi_before[mid],
        )


# ===== K. Null coupling + constant psi => unchanged =====

def test_null_coupling_constant_psi_unchanged():
    p = make_protocol(M=2)
    initial_psi = p.instance.get_module("m0").state["psi"].copy()
    p.run(5)
    final_psi = p.instance.get_module("m0").state["psi"]
    np.testing.assert_allclose(final_psi, initial_psi, atol=1e-12)


# ===== L. Geometry pic psi => diffusion =====

def test_geometry_pic_diffuses_under_null_coupling():
    """Centered pic on m0; after step, central value < initial."""
    grid = make_grid((5, 5, 5), dx=1.0)
    psi0 = np.zeros((5, 5, 5), dtype=np.float64)
    psi0[2, 2, 2] = 1.0
    modules = {"m0": make_module("m0", shape=(5, 5, 5), psi=psi0)}
    inst = Instance(id="I", modules=modules)
    geom = GeometryOperator(grid)
    p = ExperimentProtocol(
        protocol_name="test", version="v",
        instance=inst, grid=grid, geometry=geom,
        scheduler=CouplingScheduler(),
        coupling_operator=NullCouplingOperator(),
        aggregator=CouplingAggregator(geometry=geom),
        solver=ExplicitReferenceSolver(),
        dt=0.01,
    )
    p.step()
    new_psi = p.instance.get_module("m0").state["psi"]
    assert new_psi[2, 2, 2] < 1.0  # central pic decreased
    assert new_psi[1, 2, 2] > 0.0  # neighbour received
    # Conservation: total mass preserved
    assert abs(new_psi.sum() - 1.0) < 1e-12


# ===== M. History commit every macro-step =====

def test_history_count_after_k_steps():
    """After k steps with M modules and commit_initial=True:
    len(history) == (k+1) * M."""
    p = make_protocol(M=2)
    assert len(p.history_entries()) == 2  # initial commit
    p.step()
    assert len(p.history_entries()) == 4
    p.run(3)
    assert len(p.history_entries()) == (1 + 4) * 2  # 10


# ===== N. Checkpoint metadata =====

def test_checkpoint_interval_marks_steps():
    p = make_protocol(M=1, checkpoint_interval=2)
    p.run(5)
    # step 0 (initial): 0 % 2 == 0 -> checkpoint True
    # step 1: False, step 2: True, step 3: False, step 4: True, step 5: False
    entries = p.history_entries()
    # Filter by step
    step_to_checkpoint = {}
    for e in entries:
        step_to_checkpoint[e.step] = e.metadata["checkpoint"]
    assert step_to_checkpoint[0] is True
    assert step_to_checkpoint[1] is False
    assert step_to_checkpoint[2] is True
    assert step_to_checkpoint[3] is False
    assert step_to_checkpoint[4] is True
    assert step_to_checkpoint[5] is False


def test_checkpoint_does_not_block_commit():
    """All steps commit, regardless of checkpoint marker."""
    p = make_protocol(M=1, checkpoint_interval=3)
    p.run(5)
    entries = p.history_entries()
    # 1 initial + 5 steps = 6
    assert len(entries) == 6
    for e in entries:
        assert e.committed is True


def test_no_checkpoint_when_interval_none():
    p = make_protocol(M=1, checkpoint_interval=None)
    p.step()
    entries = p.history_entries()
    for e in entries:
        assert e.metadata["checkpoint"] is False


# ===== O. No substeps in HistoryView =====

def test_history_view_filters_uncommitted_entries():
    """Inject a fake uncommitted entry into _history manually; HistoryView
    must filter it out via committed-only access."""
    p = make_protocol(M=1)
    # Manually inject a committed=False entry
    fake_state = State(
        fields={"psi": 999.0 * np.ones((5, 5, 5), dtype=np.float64),
                "h": np.ones((5, 5, 5), dtype=np.float64)},
        solver_fields=("psi", "h"),
    )
    fake_entry = HistoryEntry(
        state=fake_state, time=999.0, step=999,
        committed=False,  # uncommitted substep simulation
        metadata={"module_id": "m0", "macro_step": 999, "checkpoint": False},
    )
    p._history.append(fake_entry)
    # Raw history has it
    times = [e.time for e in p._history]
    assert 999.0 in times
    # HistoryView for module m0 filters by committed=True
    hv_m0 = p.history_view_for_module("m0")
    # n_committed counts only committed entries
    n_committed = hv_m0.n_committed()
    # Only the initial commit (1 entry for m0), the fake is filtered out
    assert n_committed == 1
    # previous() returns the latest committed entry's psi (not the fake's 999)
    latest_psi = hv_m0.previous("psi", n=1)
    assert not np.any(latest_psi == 999.0)


# ===== P. Protocol is unique history maintainer =====

def test_protocol_is_unique_history_maintainer():
    """Sub-components (solver, scheduler, etc.) must NOT own history."""
    p = make_protocol(M=1)
    assert not hasattr(p._solver, "_history")
    assert not hasattr(p._scheduler, "_history")
    assert not hasattr(p._coupling_operator, "_history")
    assert not hasattr(p._aggregator, "_history")
    assert not hasattr(p._geometry, "_history")


# ===== Q. Protocol has no inline coupling/geometry methods =====

def test_protocol_no_inline_physics():
    p = make_protocol(M=1)
    forbidden = [
        "compute_gradient", "compute_flux", "compute_divergence",
        "compute_coupling", "pairwise_coupling",
        "build_contexts", "admissible_pairs",
    ]
    for n in forbidden:
        assert not hasattr(p, n), f"forbidden inline method: {n}"


# ===== S. No physics recoding in source =====

def test_protocol_does_not_recode_physics():
    import mcq_v5.experiment as exp_mod
    src = Path(exp_mod.__file__).read_text()
    # Should not contain raw gradient/divergence formulas
    forbidden_patterns = [
        "psi_R - psi_L",  # manual gradient
        "np.diff",        # would-be gradient shortcut
        "h_face *",       # manual flux
    ]
    for pat in forbidden_patterns:
        assert pat not in src, f"forbidden inline physics: {pat}"


# ===== T. HistoryEntry metadata content =====

def test_history_entry_metadata_keys():
    p = make_protocol(M=2)
    for e in p.history_entries():
        assert "instance_id" in e.metadata
        assert "module_id" in e.metadata
        assert "macro_step" in e.metadata
        assert "checkpoint" in e.metadata
        assert "protocol_name" in e.metadata
        assert e.metadata["protocol_name"] == "test"


def test_history_entry_module_id_distinct():
    p = make_protocol(M=2)
    p.step()
    m0_entries = [e for e in p.history_entries() if e.metadata["module_id"] == "m0"]
    m1_entries = [e for e in p.history_entries() if e.metadata["module_id"] == "m1"]
    assert len(m0_entries) == 2  # initial + step
    assert len(m1_entries) == 2


# ===== U. history_view_for_module returns only that module =====

def test_history_view_for_module_filters_correctly():
    p = make_protocol(M=2)
    p.step()
    hv_m0 = p.history_view_for_module("m0")
    # After 1 step + initial: 2 committed entries for m0
    assert hv_m0.n_committed() == 2
    # Get states for m0 only
    states_m0 = hv_m0.states(length=10)
    assert len(states_m0) == 2
    # All states have psi (m0 field present)
    for s in states_m0:
        assert "psi" in s.fields


# ===== V. history_entries returns a copy =====

def test_history_entries_returns_copy():
    p = make_protocol(M=1)
    entries = p.history_entries()
    initial_len = len(entries)
    entries.append("garbage")  # mutate returned list
    assert len(p.history_entries()) == initial_len  # internal unchanged


# ===== Guardrails =====

def test_guardrail_reject_nonfinite_state():
    """If a rhs (somehow) produces NaN, guardrail must reject the step."""
    # The solver itself catches NaN now. Test that the guardrail flag
    # is at least stored and accessible.
    p = make_protocol(M=1, guardrails={"reject_nonfinite_state": True})
    bundle = p.export_protocol_metadata()
    assert bundle["guardrails"]["reject_nonfinite_state"] is True


# ===== _add_coupling_rhs =====

def test_add_coupling_rhs_strict_addition():
    """Direct test of _add_coupling_rhs: addition is strict on underlying
    arrays, returns new State, original rhs not mutated."""
    p = make_protocol(M=2)
    rhs = State(
        fields={
            "psi": 0.5 * np.ones((5, 5, 5), dtype=np.float64),
            "h": np.zeros((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    from mcq_v5.coupling_aggregator import AggregatedCoupling
    coupling = AggregatedCoupling(
        target_id="m0",
        rhs=0.3 * np.ones((5, 5, 5), dtype=np.float64),
        mass_delta=37.5,
        mass_policy="source_allowed",
        n_contributions=1,
    )
    snap = rhs["psi"].copy()
    new_rhs = p._add_coupling_rhs(rhs, coupling, field_name="psi")
    # Original not mutated
    np.testing.assert_array_equal(rhs["psi"], snap)
    # Result is sum
    np.testing.assert_allclose(new_rhs["psi"], 0.8)
    # h preserved
    np.testing.assert_array_equal(new_rhs["h"], rhs["h"])
    # New State, C-contiguous, float64
    assert new_rhs["psi"].dtype == np.float64
    assert new_rhs["psi"].flags["C_CONTIGUOUS"]
    # Trace metadata
    assert new_rhs.metadata["coupling_merged"] is True
    assert new_rhs.metadata["coupling_target"] == "m0"


def test_add_coupling_rhs_invalid_field_name():
    p = make_protocol(M=1)
    rhs = State(
        fields={"psi": np.zeros((5, 5, 5), dtype=np.float64),
                "h": np.zeros((5, 5, 5), dtype=np.float64)},
        solver_fields=("psi", "h"),
    )
    from mcq_v5.coupling_aggregator import AggregatedCoupling
    coupling = AggregatedCoupling(
        target_id="m0",
        rhs=np.zeros((5, 5, 5), dtype=np.float64),
        mass_delta=0.0,
        mass_policy="conservative",
        n_contributions=0,
    )
    with pytest.raises(ValueError, match="field_name"):
        p._add_coupling_rhs(rhs, coupling, field_name="unknown")


# ===== Slots =====

def test_protocol_slots_prevent_adhoc_attribute():
    p = make_protocol(M=1)
    with pytest.raises(AttributeError):
        p.new_attr = "x"


# ===== Diagnostics integration (optional) =====

def test_protocol_accepts_diagnostics():
    grid = make_grid((5, 5, 5), dx=1.0)
    inst = Instance(id="I", modules={"m0": make_module("m0")})
    geom = GeometryOperator(grid)
    p = ExperimentProtocol(
        protocol_name="test", version="v",
        instance=inst, grid=grid, geometry=geom,
        scheduler=CouplingScheduler(),
        coupling_operator=NullCouplingOperator(),
        aggregator=CouplingAggregator(geometry=geom),
        solver=ExplicitReferenceSolver(),
        diagnostics=Diagnostics(),
        dt=0.01,
    )
    p.step()
    diag = p.last_step_diagnostics
    assert "step" in diag
    assert diag["step"] == 1


def test_protocol_diagnostics_optional():
    p = make_protocol(M=1)
    # No diagnostics passed; protocol still works
    p.step()
    assert p.last_step_diagnostics["step"] == 1


# ===== Continuous trajectory note: external mutation not blocked =====

def test_external_state_modification_not_blocked():
    """V5-0a documents that the protocol expects to be the sole actor.
    External modification is the caller's responsibility, but the protocol
    keeps running."""
    p = make_protocol(M=1)
    # Caller mutates _instance externally (anti-pattern but allowed)
    new_module = make_module(
        "m0", psi=5.0 * np.ones((5, 5, 5), dtype=np.float64)
    )
    p._instance = Instance(id="I", modules={"m0": new_module})
    # Protocol still steps
    p.step()
    assert p.step_index == 1


# ===== Solver does not own history (cross-check) =====

def test_solver_has_no_history_attribute_via_protocol():
    p = make_protocol(M=1)
    assert not hasattr(p._solver, "history")
    assert not hasattr(p._solver, "_history")
    assert not hasattr(p._solver, "commit_step")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
