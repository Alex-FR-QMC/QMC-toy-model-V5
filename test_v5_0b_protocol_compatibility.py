# -*- coding: utf-8 -*-
"""
V5-0b §3.B — Protocol compatibility harness.

Strict minimal audit. Verifies that ExperimentProtocol under
NullCouplingOperator reproduces the pure geometric dynamics with no
side-effects: no cross-module interaction, no checkpoint-induced
divergence, no history corruption, JSON-strict exportable metadata.

Permitted verdict for this file:
    PROTOCOL_COMPAT_PASS
        meaning: ExperimentProtocol + NullCouplingOperator preserves
        pure geometric dynamics under the tested conditions.

Forbidden inscriptions:
    Long-horizon reproductions (P5bis, P4 strict, h-dynamics) are
    explicitly OUT_OF_SCOPE for this file. See V5-0b protocol document
    §3.C for the boundary statement. The test
    `test_verdict_is_strictly_scoped` enforces this in code.

No modification of mcq_v5.*. Reference computations (when needed) are
done by re-running mcq_v5 components directly in a controlled isolated
loop, NOT by an external reimplementation (which is the responsibility
of §3.A geometry compatibility).
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.grid import make_grid
from mcq_v5.state import State
from mcq_v5.metric import ScalarConformalMetric
from mcq_v5.module import Module
from mcq_v5.instance import Instance
from mcq_v5.views import HistoryEntry
from mcq_v5.geometry import GeometryOperator
from mcq_v5.coupling_scheduler import CouplingScheduler
from mcq_v5.coupling_operator import NullCouplingOperator
from mcq_v5.coupling_aggregator import CouplingAggregator
from mcq_v5.solver import ExplicitReferenceSolver
from mcq_v5.diagnostics import Diagnostics
from mcq_v5.experiment import ExperimentProtocol


# =========================================================================
# Local helpers (no impact on mcq_v5.*)
# =========================================================================

def _make_module(mid, shape=(5, 5, 5), psi=None, h=None):
    if psi is None:
        psi = np.ones(shape, dtype=np.float64)
    if h is None:
        h = np.ones(shape, dtype=np.float64)
    state = State(
        fields={
            "psi": np.ascontiguousarray(psi, dtype=np.float64),
            "h": np.ascontiguousarray(h, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    metric = ScalarConformalMetric(np.ascontiguousarray(h, dtype=np.float64))
    return Module(id=mid, state=state, metric=metric)


def make_protocol(
    M=1,
    shape=(5, 5, 5),
    dt=0.01,
    psi_by_module=None,
    h_by_module=None,
    checkpoint_interval=None,
    commit_initial=True,
):
    """Helper local. Builds an ExperimentProtocol with NullCouplingOperator
    over M modules. psi_by_module / h_by_module: dict[module_id, ndarray].
    """
    grid = make_grid(shape, dx=1.0)
    modules = {}
    for i in range(M):
        mid = f"m{i}"
        psi = psi_by_module.get(mid) if psi_by_module else None
        h = h_by_module.get(mid) if h_by_module else None
        modules[mid] = _make_module(mid, shape=shape, psi=psi, h=h)
    instance = Instance(id="I", modules=modules)
    geometry = GeometryOperator(grid)
    scheduler = CouplingScheduler()
    operator = NullCouplingOperator()
    aggregator = CouplingAggregator(geometry=geometry)
    solver = ExplicitReferenceSolver()
    diagnostics = Diagnostics()
    return ExperimentProtocol(
        protocol_name="v5_0b_protocol_compat",
        version="v5-0b",
        instance=instance,
        grid=grid,
        geometry=geometry,
        scheduler=scheduler,
        coupling_operator=operator,
        aggregator=aggregator,
        solver=solver,
        diagnostics=diagnostics,
        dt=dt,
        conservation_policy="conservative",
        checkpoint_interval=checkpoint_interval,
        commit_initial=commit_initial,
    )


def _isolated_geometry_loop(
    psi_init: np.ndarray,
    h: np.ndarray,
    shape: tuple,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Reference loop: GeometryOperator.apply + ExplicitReferenceSolver.step,
    direct, no scheduler/coupling/aggregator. Used to verify that
    ExperimentProtocol + NullCoupling adds NOTHING.
    """
    grid = make_grid(shape, dx=1.0)
    geom = GeometryOperator(grid)
    solver = ExplicitReferenceSolver()
    state = State(
        fields={
            "psi": np.ascontiguousarray(psi_init, dtype=np.float64),
            "h": np.ascontiguousarray(h, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    metric = ScalarConformalMetric(np.ascontiguousarray(h, dtype=np.float64))
    for _ in range(n_steps):
        rhs = geom.apply(state, metric, field_name="psi")
        state = solver.step(state, rhs, dt)
    return state["psi"]


# =========================================================================
# B.1 — M=1 constant psi unchanged after n steps
# =========================================================================

def test_B1_M1_constant_psi_unchanged():
    psi0 = 2.5 * np.ones((5, 5, 5), dtype=np.float64)
    p = make_protocol(M=1, psi_by_module={"m0": psi0.copy()})
    psi_initial = p.instance.get_module("m0").state["psi"].copy()
    p.run(10)
    psi_final = p.instance.get_module("m0").state["psi"]
    np.testing.assert_allclose(psi_final, psi_initial, atol=1e-12, rtol=0)
    # History length: (n_steps + 1) * M = 11 * 1
    assert len(p.history_entries()) == 11


# =========================================================================
# B.2 — M=1 pic central, comparison with isolated geometry loop
# =========================================================================

def test_B2_M1_pic_matches_isolated_geometry_loop():
    """ExperimentProtocol + NullCoupling must add NOTHING to a pure
    GeometryOperator + Solver loop."""
    psi0 = np.zeros((5, 5, 5), dtype=np.float64)
    psi0[2, 2, 2] = 1.0
    n_steps = 5
    dt = 0.01

    # Protocol with NullCoupling
    p = make_protocol(M=1, psi_by_module={"m0": psi0.copy()}, dt=dt)
    p.run(n_steps)
    psi_protocol = p.instance.get_module("m0").state["psi"]

    # Isolated reference loop
    h = np.ones((5, 5, 5), dtype=np.float64)
    psi_ref = _isolated_geometry_loop(psi0, h, (5, 5, 5), dt, n_steps)

    np.testing.assert_allclose(psi_protocol, psi_ref, atol=1e-12, rtol=0)


def test_B2_diffusion_actually_happened():
    """Sanity check: the pic decreased and neighbours received mass."""
    psi0 = np.zeros((5, 5, 5), dtype=np.float64)
    psi0[2, 2, 2] = 1.0
    p = make_protocol(M=1, psi_by_module={"m0": psi0.copy()}, dt=0.01)
    p.run(5)
    psi_final = p.instance.get_module("m0").state["psi"]
    assert psi_final[2, 2, 2] < 1.0  # central pic decreased
    assert psi_final[1, 2, 2] > 0.0  # neighbour received
    # Conservation: total mass preserved
    assert abs(psi_final.sum() - 1.0) < 1e-12


# =========================================================================
# B.3.i — M=2 NullCoupling: m1 constant stays constant
# =========================================================================

def test_B3i_M2_m1_constant_stays_constant():
    """m0 has a pic, m1 is constant. Under NullCoupling, m1 must remain
    strictly constant (no cross-module effect)."""
    psi_m0 = np.zeros((5, 5, 5), dtype=np.float64)
    psi_m0[2, 2, 2] = 1.0
    psi_m1 = 0.7 * np.ones((5, 5, 5), dtype=np.float64)
    p = make_protocol(
        M=2,
        psi_by_module={"m0": psi_m0.copy(), "m1": psi_m1.copy()},
        dt=0.01,
    )
    m1_initial = p.instance.get_module("m1").state["psi"].copy()
    p.run(5)
    m1_final = p.instance.get_module("m1").state["psi"]
    np.testing.assert_allclose(m1_final, m1_initial, atol=1e-12, rtol=0)


# =========================================================================
# B.3.ii — M=2 NullCoupling: m0 evolves as if isolated in M=1
# =========================================================================

def test_B3ii_M2_m0_evolves_as_in_M1_isolated():
    """Two protocols, same initial m0 state:
       p_multi : M=2, m0=pic, m1=constant
       p_iso   : M=1, m0=pic
    After n_steps, p_multi.m0.psi must equal p_iso.m0.psi (bit-for-bit
    when feasible, atol=1e-12 otherwise).
    """
    psi_m0 = np.zeros((5, 5, 5), dtype=np.float64)
    psi_m0[2, 2, 2] = 1.0
    psi_m1_const = 0.5 * np.ones((5, 5, 5), dtype=np.float64)

    p_multi = make_protocol(
        M=2,
        psi_by_module={"m0": psi_m0.copy(), "m1": psi_m1_const.copy()},
        dt=0.01,
    )
    p_iso = make_protocol(
        M=1,
        psi_by_module={"m0": psi_m0.copy()},
        dt=0.01,
    )
    p_multi.run(5)
    p_iso.run(5)

    psi_multi = p_multi.instance.get_module("m0").state["psi"]
    psi_iso = p_iso.instance.get_module("m0").state["psi"]
    np.testing.assert_allclose(psi_multi, psi_iso, atol=1e-12, rtol=0)


# =========================================================================
# B.4 — History (k+1) * M
# =========================================================================

def test_B4_history_count():
    M = 3
    k = 4
    p = make_protocol(M=M)
    p.run(k)
    entries = p.history_entries()
    assert len(entries) == (k + 1) * M


def test_B4_history_all_committed():
    p = make_protocol(M=3)
    p.run(4)
    for e in p.history_entries():
        assert e.committed is True


def test_B4_history_view_per_module():
    p = make_protocol(M=3)
    p.run(4)
    for mid in p.instance.module_ids():
        hv = p.history_view_for_module(mid)
        # k + 1 = 5 committed entries per module
        assert hv.n_committed() == 5


def test_B4_history_anti_aliasing_temporal():
    """Audit ajout 1: HistoryEntry must be a true archive, not a list
    of pointers to a mutating array. Verify that step-0 psi and step-k
    psi are DISTINCT objects in memory."""
    psi0 = np.zeros((5, 5, 5), dtype=np.float64)
    psi0[2, 2, 2] = 1.0
    p = make_protocol(M=1, psi_by_module={"m0": psi0.copy()}, dt=0.01)
    p.run(3)
    entries = p.history_entries()
    # Filter for m0 only (M=1 so all entries are m0)
    m0_entries = [
        e for e in entries
        if e.metadata.get("module_id") == "m0"
    ]
    psi_step0 = m0_entries[0].state["psi"]
    psi_stepk = m0_entries[-1].state["psi"]
    # Different ndarray objects (distinct memory)
    assert psi_step0 is not psi_stepk
    # Values differ (diffusion happened)
    assert not np.array_equal(psi_step0, psi_stepk)
    # Step-0 is still the original pic
    assert psi_step0[2, 2, 2] == 1.0
    # Step-k decreased
    assert psi_stepk[2, 2, 2] < 1.0


# =========================================================================
# B.5 — Checkpoint neutral
# =========================================================================

def test_B5_checkpoint_does_not_affect_dynamics():
    """Two protocols, identical initial state:
       p_none : checkpoint_interval=None
       p_ckpt : checkpoint_interval=2
    After n_steps, states must be identical bit-for-bit."""
    psi0 = np.zeros((5, 5, 5), dtype=np.float64)
    psi0[2, 2, 2] = 1.0
    p_none = make_protocol(
        M=1, psi_by_module={"m0": psi0.copy()},
        dt=0.01, checkpoint_interval=None,
    )
    p_ckpt = make_protocol(
        M=1, psi_by_module={"m0": psi0.copy()},
        dt=0.01, checkpoint_interval=2,
    )
    p_none.run(6)
    p_ckpt.run(6)
    np.testing.assert_array_equal(
        p_none.instance.get_module("m0").state["psi"],
        p_ckpt.instance.get_module("m0").state["psi"],
    )


def test_B5_checkpoint_only_changes_metadata():
    """Metadata 'checkpoint' must be True at step 0, 2, 4 with
    interval=2, and False elsewhere. p_none must have all False."""
    p_none = make_protocol(M=1, dt=0.01, checkpoint_interval=None)
    p_ckpt = make_protocol(M=1, dt=0.01, checkpoint_interval=2)
    p_none.run(5)
    p_ckpt.run(5)

    for e in p_none.history_entries():
        assert e.metadata["checkpoint"] is False

    step_to_ckpt = {
        e.step: e.metadata["checkpoint"]
        for e in p_ckpt.history_entries()
    }
    assert step_to_ckpt[0] is True
    assert step_to_ckpt[1] is False
    assert step_to_ckpt[2] is True
    assert step_to_ckpt[3] is False
    assert step_to_ckpt[4] is True
    assert step_to_ckpt[5] is False


# =========================================================================
# B.6 — JSON strict metadata
# =========================================================================

_REQUIRED_BUNDLE_KEYS = (
    "protocol_name", "version", "grid_shape", "metric_type",
    "N", "M", "coupling_policy", "conservation_policy",
    "history_policy", "seeds", "checkpoints", "guardrails",
)


def test_B6_metadata_json_strict_allow_nan_false():
    """export_protocol_metadata() must be JSON-serializable with
    allow_nan=False (no inf, no nan)."""
    p = make_protocol(M=2)
    bundle = p.export_protocol_metadata()
    # Must succeed without TypeError or ValueError
    s = json.dumps(bundle, allow_nan=False)
    assert isinstance(s, str)
    # Audit ajout 2: full round-trip
    reloaded = json.loads(s)
    # Verify round-trip preserves required keys
    for k in _REQUIRED_BUNDLE_KEYS:
        assert k in reloaded, f"missing after round-trip: {k}"


def test_B6_metadata_required_keys():
    p = make_protocol(M=2, checkpoint_interval=3)
    bundle = p.export_protocol_metadata()
    for k in _REQUIRED_BUNDLE_KEYS:
        assert k in bundle, f"missing required key: {k}"
    # Specific values
    assert bundle["N"] == 1
    assert bundle["M"] == 2
    assert bundle["metric_type"] == "scalar_conformal"
    assert bundle["history_policy"]["sole_maintainer"] == "ExperimentProtocol"
    assert bundle["history_policy"]["commit"] == "every_macro_step"
    assert bundle["history_policy"]["substep_exclusion"] is True


def test_B6_metadata_json_strict_after_run():
    """After running steps, metadata still JSON-strict (current_step
    advances but all values remain finite/serializable)."""
    p = make_protocol(M=1, checkpoint_interval=2)
    p.run(7)
    bundle = p.export_protocol_metadata()
    s = json.dumps(bundle, allow_nan=False)
    reloaded = json.loads(s)
    assert reloaded["checkpoints"]["current_step"] == 7


# =========================================================================
# Verdict aggregation
# =========================================================================

def test_protocol_compatibility_verdict_PASS():
    """If this test passes, PROTOCOL_COMPAT_PASS is inscriptible.

    Reruns a tight set of canonical cases to inscribe the verdict
    structurally. No external reporting.
    """
    failures = []

    # Case 1: M=1 constant unchanged
    p = make_protocol(M=1)
    psi0 = p.instance.get_module("m0").state["psi"].copy()
    p.run(5)
    psiN = p.instance.get_module("m0").state["psi"]
    if not np.allclose(psiN, psi0, atol=1e-12, rtol=0):
        failures.append("M1_constant_unchanged")

    # Case 2: M=1 pic matches isolated loop
    psi_pic = np.zeros((5, 5, 5), dtype=np.float64)
    psi_pic[2, 2, 2] = 1.0
    p = make_protocol(M=1, psi_by_module={"m0": psi_pic.copy()})
    p.run(3)
    psi_proto = p.instance.get_module("m0").state["psi"]
    h = np.ones((5, 5, 5), dtype=np.float64)
    psi_ref = _isolated_geometry_loop(psi_pic, h, (5, 5, 5), 0.01, 3)
    if not np.allclose(psi_proto, psi_ref, atol=1e-12, rtol=0):
        failures.append("M1_pic_matches_isolated_loop")

    # Case 3: M=2 NullCoupling m1 constant stays constant
    p = make_protocol(
        M=2,
        psi_by_module={
            "m0": psi_pic.copy(),
            "m1": 0.5 * np.ones((5, 5, 5), dtype=np.float64),
        },
    )
    m1_init = p.instance.get_module("m1").state["psi"].copy()
    p.run(3)
    m1_final = p.instance.get_module("m1").state["psi"]
    if not np.allclose(m1_final, m1_init, atol=1e-12, rtol=0):
        failures.append("M2_m1_constant_stays")

    # Case 4: history count
    p = make_protocol(M=2)
    p.run(3)
    if len(p.history_entries()) != (3 + 1) * 2:
        failures.append("history_count")

    # Case 5: JSON strict
    p = make_protocol(M=2, checkpoint_interval=2)
    try:
        json.dumps(p.export_protocol_metadata(), allow_nan=False)
    except (TypeError, ValueError) as e:
        failures.append(f"json_strict: {e}")

    assert failures == [], f"PROTOCOL_COMPAT_FAIL: {failures}"


# =========================================================================
# Anti-glissement: forbidden statements not part of this verdict
# =========================================================================

def test_verdict_is_strictly_scoped():
    """Confirm in code that this file's only verdict is PROTOCOL_COMPAT.
    No reference to long-horizon reproductions is inscribed here.

    The patterns are built via concatenation to avoid self-detection
    (the test would otherwise find its own forbidden-list strings).
    """
    src = Path(__file__).read_text()
    # Build forbidden patterns via concatenation to avoid self-collision
    forbidden = [
        "V5" + " " + "reproduces" + " " + "P5bis",
        "P4" + " " + "strict" + " " + "validated",
        "h-dyn" + "amics" + " " + "reproduced",
        "h-dyn" + "amics" + " " + "validated",
        "P5" + "bis" + " " + "validated",
    ]
    for pat in forbidden:
        assert pat not in src, f"forbidden inscription found: {pat}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
