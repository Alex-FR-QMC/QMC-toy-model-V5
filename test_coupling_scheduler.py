# -*- coding: utf-8 -*-
"""Tests for mcq_v5.coupling_scheduler.CouplingScheduler.

Critère étape 9: only object reading Instance, produces CouplingContext
bounded objects, applies epsilon admissibility, no coupling logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.grid import Grid, make_grid
from mcq_v5.state import State
from mcq_v5.metric import ScalarConformalMetric
from mcq_v5.module import Module
from mcq_v5.instance import Instance
from mcq_v5.views import GridView, StateView, MetricView, HistoryView, HistoryEntry
from mcq_v5.coupling_context import CouplingContext
from mcq_v5.coupling_scheduler import CouplingScheduler


# ===== Helpers =====

def make_module(mid, shape=(5, 5, 5), psi_val=1.0):
    s = State(
        fields={
            "psi": psi_val * np.ones(shape, dtype=np.float64),
            "h": np.ones(shape, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    m = ScalarConformalMetric(np.ones(shape, dtype=np.float64))
    return Module(id=mid, state=s, metric=m)


def make_instance(n=3):
    modules = {f"m{i}": make_module(f"m{i}", psi_val=float(i + 1)) for i in range(n)}
    return Instance(id="I", modules=modules)


def make_grid_view(shape=(5, 5, 5)):
    return GridView(make_grid(shape, dx=1.0))


# ===== A. Construction =====

def test_scheduler_default_construction():
    sch = CouplingScheduler()
    assert sch.overlap_mode == "global_scalar"
    assert sch.epsilon_R_min == 0.0
    assert sch.epsilon_R_max == 0.0


def test_scheduler_with_policy():
    sch = CouplingScheduler(epsilon_R_min=0.1, epsilon_R_max=0.1)
    assert sch.epsilon_R_min == 0.1
    assert sch.epsilon_R_max == 0.1


# ===== B. Epsilon validation =====

def test_scheduler_epsilon_min_negative_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        CouplingScheduler(epsilon_R_min=-0.1)


def test_scheduler_epsilon_min_ge_1_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        CouplingScheduler(epsilon_R_min=1.0)


def test_scheduler_epsilon_max_negative_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        CouplingScheduler(epsilon_R_max=-0.01)


def test_scheduler_epsilon_max_ge_1_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        CouplingScheduler(epsilon_R_max=1.5)


def test_scheduler_epsilon_nan_rejected():
    with pytest.raises(ValueError, match="finite"):
        CouplingScheduler(epsilon_R_min=float("nan"))


def test_scheduler_epsilon_invalid_type():
    with pytest.raises(TypeError, match="numeric"):
        CouplingScheduler(epsilon_R_min="0.1")


def test_scheduler_epsilon_dead_window_rejected():
    """Audit ajout 1: epsilon_R_min + epsilon_R_max >= 1 must be rejected
    to prevent silent empty admissibility window.
    """
    with pytest.raises(ValueError, match="window"):
        CouplingScheduler(epsilon_R_min=0.6, epsilon_R_max=0.6)
    with pytest.raises(ValueError, match="window"):
        CouplingScheduler(epsilon_R_min=0.5, epsilon_R_max=0.5)


def test_scheduler_epsilon_boundary_window_just_valid():
    """Sum just below 1.0 is accepted."""
    sch = CouplingScheduler(epsilon_R_min=0.4, epsilon_R_max=0.5)
    assert sch.epsilon_R_min == 0.4
    assert sch.epsilon_R_max == 0.5


def test_scheduler_invalid_overlap_mode():
    with pytest.raises(ValueError, match="overlap_mode"):
        CouplingScheduler(overlap_mode="garbage")


def test_scheduler_invalid_overlap_provider_type():
    with pytest.raises(TypeError, match="overlap_provider"):
        CouplingScheduler(overlap_provider="not callable")


def test_scheduler_invalid_history_provider_type():
    with pytest.raises(TypeError, match="history_provider"):
        CouplingScheduler(history_provider="not callable")


def test_scheduler_invalid_metadata_type():
    with pytest.raises(TypeError, match="metadata"):
        CouplingScheduler(metadata="not a dict")


# ===== C/D/E. admissible_pairs for M=1, M=2, M=3 =====

def test_admissible_pairs_M1_empty():
    sch = CouplingScheduler()
    inst = make_instance(n=1)
    pairs = sch.admissible_pairs(inst)
    assert pairs == ()


def test_admissible_pairs_M2_two_ordered():
    sch = CouplingScheduler()
    inst = make_instance(n=2)
    pairs = sch.admissible_pairs(inst)
    assert len(pairs) == 2
    assert set(pairs) == {("m0", "m1"), ("m1", "m0")}


def test_admissible_pairs_M3_six_ordered():
    sch = CouplingScheduler()
    inst = make_instance(n=3)
    pairs = sch.admissible_pairs(inst)
    assert len(pairs) == 6
    expected = {
        ("m0", "m1"), ("m0", "m2"),
        ("m1", "m0"), ("m1", "m2"),
        ("m2", "m0"), ("m2", "m1"),
    }
    assert set(pairs) == expected


# ===== F. No self-pairs =====

def test_admissible_pairs_no_self_pair():
    sch = CouplingScheduler()
    inst = make_instance(n=3)
    pairs = sch.admissible_pairs(inst)
    for (t, s) in pairs:
        assert t != s


# ===== G. Determinism via sorted ids =====

def test_admissible_pairs_deterministic_sorted():
    """Audit ajout 3: deterministic via sorted(module_ids)."""
    sch = CouplingScheduler()
    # Insert in non-sorted order
    modules = {
        "m2": make_module("m2"),
        "m0": make_module("m0"),
        "m1": make_module("m1"),
    }
    inst = Instance(id="I", modules=modules)
    pairs = sch.admissible_pairs(inst)
    # First few pairs by sorted target id
    assert pairs[0] == ("m0", "m1")
    assert pairs[1] == ("m0", "m2")
    assert pairs[2] == ("m1", "m0")
    assert pairs[3] == ("m1", "m2")
    assert pairs[4] == ("m2", "m0")
    assert pairs[5] == ("m2", "m1")


def test_admissible_pairs_independent_of_insertion_order():
    """Same modules, different insert order -> same pair sequence."""
    sch = CouplingScheduler()
    inst1 = Instance(id="I", modules={
        f"m{i}": make_module(f"m{i}") for i in [2, 1, 0]
    })
    inst2 = Instance(id="I", modules={
        f"m{i}": make_module(f"m{i}") for i in [0, 2, 1]
    })
    assert sch.admissible_pairs(inst1) == sch.admissible_pairs(inst2)


# ===== H. Epsilon min excludes low R =====

def test_admissibility_excludes_low_R():
    """R = 0.05 with epsilon_R_min = 0.1 -> excluded."""
    def low_R(t, s, inst):
        return 0.05
    sch = CouplingScheduler(
        epsilon_R_min=0.1, epsilon_R_max=0.0, overlap_provider=low_R,
    )
    inst = make_instance(n=2)
    assert sch.admissible_pairs(inst) == ()


def test_admissibility_includes_R_above_min():
    """R = 0.5 with epsilon_R_min = 0.1 -> included."""
    sch = CouplingScheduler(epsilon_R_min=0.1, epsilon_R_max=0.0)
    inst = make_instance(n=2)
    assert len(sch.admissible_pairs(inst)) == 2


# ===== I. Epsilon max excludes near-one R =====

def test_admissibility_excludes_high_R():
    """R = 0.99 with epsilon_R_max = 0.1 -> excluded (1-R = 0.01 < 0.1)."""
    def high_R(t, s, inst):
        return 0.99
    sch = CouplingScheduler(
        epsilon_R_min=0.0, epsilon_R_max=0.1, overlap_provider=high_R,
    )
    inst = make_instance(n=2)
    assert sch.admissible_pairs(inst) == ()


def test_admissibility_includes_R_below_max():
    """R = 0.5 with epsilon_R_max = 0.1 -> included (1-R = 0.5 >= 0.1)."""
    sch = CouplingScheduler(epsilon_R_min=0.0, epsilon_R_max=0.1)
    inst = make_instance(n=2)
    assert len(sch.admissible_pairs(inst)) == 2


# ===== J. R=0 and R=1 excluded under positive epsilons =====

def test_admissibility_R0_excluded_with_positive_eps_min():
    def r0(t, s, inst):
        return 0.0
    sch = CouplingScheduler(epsilon_R_min=0.01, overlap_provider=r0)
    inst = make_instance(n=2)
    assert sch.admissible_pairs(inst) == ()


def test_admissibility_R1_excluded_with_positive_eps_max():
    def r1(t, s, inst):
        return 1.0
    sch = CouplingScheduler(epsilon_R_max=0.01, overlap_provider=r1)
    inst = make_instance(n=2)
    assert sch.admissible_pairs(inst) == ()


# ===== K. R=0/R=1 included if epsilons zero =====

def test_admissibility_R0_included_with_zero_epsilons():
    def r0(t, s, inst):
        return 0.0
    sch = CouplingScheduler(
        epsilon_R_min=0.0, epsilon_R_max=0.0, overlap_provider=r0,
    )
    inst = make_instance(n=2)
    assert len(sch.admissible_pairs(inst)) == 2


def test_admissibility_R1_included_with_zero_epsilons():
    def r1(t, s, inst):
        return 1.0
    sch = CouplingScheduler(
        epsilon_R_min=0.0, epsilon_R_max=0.0, overlap_provider=r1,
    )
    inst = make_instance(n=2)
    assert len(sch.admissible_pairs(inst)) == 2


# ===== L. build_contexts returns tuple of CouplingContext =====

def test_build_contexts_returns_tuple_of_contexts():
    sch = CouplingScheduler()
    inst = make_instance(n=2)
    gv = make_grid_view()
    contexts = sch.build_contexts(inst, gv, time=0.0)
    assert isinstance(contexts, tuple)
    assert len(contexts) == 2
    for c in contexts:
        assert isinstance(c, CouplingContext)


# ===== M. Contexts contain views, not raw modules =====

def test_contexts_contain_views_not_raw_modules():
    sch = CouplingScheduler()
    inst = make_instance(n=2)
    gv = make_grid_view()
    contexts = sch.build_contexts(inst, gv, time=0.0)
    for c in contexts:
        assert isinstance(c.target_state_view, StateView)
        assert isinstance(c.source_state_view, StateView)
        assert isinstance(c.target_metric_view, MetricView)
        assert isinstance(c.source_metric_view, MetricView)
        assert isinstance(c.grid_view, GridView)


# ===== N. CouplingOperator not imported / not referenced =====

def test_scheduler_does_not_import_coupling_operator():
    """coupling_scheduler must not actually import coupling_operator.

    The docstring may MENTION CouplingOperator to explain anti-centralization;
    what matters is the absence of an import statement.
    """
    import mcq_v5.coupling_scheduler as sch_mod
    src = Path(sch_mod.__file__).read_text()
    # No real import of coupling_operator module
    forbidden_imports = [
        "from .coupling_operator import",
        "from mcq_v5.coupling_operator import",
        "import mcq_v5.coupling_operator",
        "import coupling_operator",
    ]
    for pat in forbidden_imports:
        assert pat not in src, f"forbidden import found: {pat}"
    # Module should not have CouplingOperator in its namespace
    assert not hasattr(sch_mod, "CouplingOperator")


# ===== O. Scheduler has no compute_coupling / aggregate / step =====

def test_scheduler_no_coupling_methods():
    sch = CouplingScheduler()
    forbidden = [
        "compute_coupling", "aggregate", "apply", "step", "evolve",
        "compute_novelty", "compute_self_form",
    ]
    for name in forbidden:
        assert not hasattr(sch, name), f"forbidden attr present: {name}"


# ===== P. Scheduler reads Instance, but CouplingContext doesn't =====

def test_contexts_do_not_have_instance_attribute():
    sch = CouplingScheduler()
    inst = make_instance(n=2)
    gv = make_grid_view()
    contexts = sch.build_contexts(inst, gv, time=0.0)
    for c in contexts:
        assert not hasattr(c, "instance")
        assert not hasattr(c, "_instance")
        assert not hasattr(c, "module")
        assert not hasattr(c, "_module")


# ===== Q. Metadata policy exported in each context =====

def test_contexts_have_policy_metadata():
    sch = CouplingScheduler(epsilon_R_min=0.1, epsilon_R_max=0.2)
    inst = make_instance(n=2)
    gv = make_grid_view()
    contexts = sch.build_contexts(inst, gv, time=1.5)
    for c in contexts:
        assert c.metadata["created_by"] == "CouplingScheduler.build_contexts"
        assert c.metadata["overlap_mode"] == "global_scalar"
        assert c.metadata["epsilon_R_min"] == 0.1
        assert c.metadata["epsilon_R_max"] == 0.2
        assert c.metadata["pair_policy"] == "ordered_pairs_i_ne_j"
        assert c.metadata["R_eff_for_admissibility"] == 0.5  # default provider


def test_scheduler_metadata_propagated():
    """Scheduler-level metadata propagated into each context."""
    sch = CouplingScheduler(metadata={"experiment": "X"})
    inst = make_instance(n=2)
    gv = make_grid_view()
    contexts = sch.build_contexts(inst, gv, time=0.0)
    for c in contexts:
        assert c.metadata["experiment"] == "X"


# ===== R. local_field provider returns array, context preserves mode =====

def test_local_field_provider_returns_array():
    def array_provider(t, s, inst):
        return 0.5 * np.ones((5, 5, 5), dtype=np.float64)

    sch = CouplingScheduler(
        overlap_mode="local_field", overlap_provider=array_provider,
    )
    inst = make_instance(n=2)
    gv = make_grid_view()
    contexts = sch.build_contexts(inst, gv, time=0.0)
    for c in contexts:
        assert c.overlap_mode == "local_field"
        assert isinstance(c.overlap_R, np.ndarray)
        assert c.overlap_R.shape == (5, 5, 5)


# ===== S. local_field admissibility uses mean reduction =====

def test_local_field_admissibility_uses_mean():
    """Provider returns array with mean 0.05; epsilon_R_min = 0.1 -> excluded."""
    def low_mean_provider(t, s, inst):
        arr = np.full((5, 5, 5), 0.05, dtype=np.float64)
        return arr

    sch = CouplingScheduler(
        overlap_mode="local_field",
        epsilon_R_min=0.1, epsilon_R_max=0.0,
        overlap_provider=low_mean_provider,
    )
    inst = make_instance(n=2)
    gv = make_grid_view()
    # No pair should be admissible
    assert sch.admissible_pairs(inst) == ()
    assert sch.build_contexts(inst, gv, time=0.0) == ()


def test_local_field_admissibility_mean_in_metadata():
    def array_provider(t, s, inst):
        return 0.5 * np.ones((5, 5, 5), dtype=np.float64)

    sch = CouplingScheduler(
        overlap_mode="local_field", overlap_provider=array_provider,
    )
    inst = make_instance(n=2)
    gv = make_grid_view()
    contexts = sch.build_contexts(inst, gv, time=0.0)
    for c in contexts:
        assert c.metadata["overlap_reduction"] == "mean_for_pair_admissibility"


# ===== T. patchwise raises NotImplementedError =====

def test_patchwise_admissible_pairs_raises():
    sch = CouplingScheduler(overlap_mode="patchwise")
    inst = make_instance(n=2)
    with pytest.raises(NotImplementedError, match="patchwise"):
        sch.admissible_pairs(inst)


def test_patchwise_build_contexts_raises():
    sch = CouplingScheduler(overlap_mode="patchwise")
    inst = make_instance(n=2)
    gv = make_grid_view()
    with pytest.raises(NotImplementedError, match="patchwise"):
        sch.build_contexts(inst, gv, time=0.0)


# ===== U. No history ownership =====

def test_scheduler_no_history_attribute():
    """Audit guard from étape 6: Instance/Scheduler must not own history."""
    sch = CouplingScheduler()
    forbidden = [
        "history", "_history", "commit_step", "append_step",
        "record", "snapshot", "save_state", "tick",
    ]
    for name in forbidden:
        assert not hasattr(sch, name), f"forbidden temporal attr: {name}"


# ===== V. Stateless: no cache of contexts / last instance =====

def test_scheduler_no_context_cache():
    sch = CouplingScheduler()
    forbidden = ["_last_instance", "_cache", "_contexts", "_last_contexts"]
    for name in forbidden:
        assert not hasattr(sch, name), f"forbidden cache attr: {name}"


def test_scheduler_repeated_calls_consistent():
    """Calling build_contexts on same instance twice yields equivalent results."""
    sch = CouplingScheduler()
    inst = make_instance(n=3)
    gv = make_grid_view()
    c1 = sch.build_contexts(inst, gv, time=1.0)
    c2 = sch.build_contexts(inst, gv, time=1.0)
    assert len(c1) == len(c2)
    for x, y in zip(c1, c2):
        assert x.target_id == y.target_id
        assert x.source_id == y.source_id
        assert x.overlap_R == y.overlap_R
        assert x.overlap_mode == y.overlap_mode


# ===== Audit ajout 2: history_provider signature (no Instance) =====

def test_history_provider_called_with_pair_only():
    """history_provider must be (target_id, source_id), not (target, source, instance)."""
    calls = []

    def hist_provider(target_id, source_id):
        calls.append((target_id, source_id))
        return None

    sch = CouplingScheduler(history_provider=hist_provider)
    inst = make_instance(n=2)
    gv = make_grid_view()
    _ = sch.build_contexts(inst, gv, time=0.0)
    # Two admissible ordered pairs called the provider
    assert len(calls) == 2
    for (t, s) in calls:
        assert t != s


def test_history_provider_can_return_history_view():
    hist_list = [
        HistoryEntry(
            state=State(
                fields={"psi": np.ones((5, 5, 5), dtype=np.float64)},
                solver_fields=("psi",),
            ),
            time=0.0, step=0, committed=True,
        ),
    ]
    hv = HistoryView(hist_list)

    def hist_provider(target_id, source_id):
        return hv

    sch = CouplingScheduler(history_provider=hist_provider)
    inst = make_instance(n=2)
    gv = make_grid_view()
    contexts = sch.build_contexts(inst, gv, time=0.0)
    for c in contexts:
        assert c.allowed_history is hv


# ===== Audit ajout 4: scheduler does not validate provider array shape =====

def test_scheduler_delegates_shape_validation_to_context():
    """Scheduler must not preempt CouplingContext shape validation.

    A provider returning a wrong-shape array should cause CouplingContext
    to raise (not the scheduler).
    """
    def bad_shape_provider(t, s, inst):
        return 0.5 * np.ones((7, 7, 7), dtype=np.float64)  # not 5x5x5

    sch = CouplingScheduler(
        overlap_mode="local_field", overlap_provider=bad_shape_provider,
    )
    inst = make_instance(n=2)
    gv = make_grid_view(shape=(5, 5, 5))
    with pytest.raises(ValueError, match="shape"):
        sch.build_contexts(inst, gv, time=0.0)


# ===== build_contexts validation =====

def test_build_contexts_invalid_instance_type():
    sch = CouplingScheduler()
    gv = make_grid_view()
    with pytest.raises(TypeError, match="instance"):
        sch.build_contexts("not an instance", gv, time=0.0)


def test_build_contexts_invalid_grid_view_type():
    sch = CouplingScheduler()
    inst = make_instance(n=2)
    with pytest.raises(TypeError, match="grid_view"):
        sch.build_contexts(inst, "not a grid view", time=0.0)


def test_build_contexts_invalid_time_type():
    sch = CouplingScheduler()
    inst = make_instance(n=2)
    gv = make_grid_view()
    with pytest.raises(TypeError, match="time"):
        sch.build_contexts(inst, gv, time="0.0")


def test_build_contexts_time_nan_rejected():
    sch = CouplingScheduler()
    inst = make_instance(n=2)
    gv = make_grid_view()
    with pytest.raises(ValueError, match="finite"):
        sch.build_contexts(inst, gv, time=float("nan"))


def test_admissible_pairs_invalid_instance_type():
    sch = CouplingScheduler()
    with pytest.raises(TypeError, match="instance"):
        sch.admissible_pairs("not an instance")


# ===== Pair time propagation =====

def test_contexts_carry_provided_time():
    sch = CouplingScheduler()
    inst = make_instance(n=2)
    gv = make_grid_view()
    contexts = sch.build_contexts(inst, gv, time=2.5)
    for c in contexts:
        assert c.time == 2.5


# ===== Slots restrict ad-hoc attrs =====

def test_scheduler_uses_slots():
    sch = CouplingScheduler()
    with pytest.raises(AttributeError):
        sch.new_attr = "x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
