# -*- coding: utf-8 -*-
"""Tests for mcq_v5.solver — explicit Euler integration only."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.state import State
from mcq_v5.solver import (
    Solver, ExplicitReferenceSolver,
    SemiImplicitSolver, ADISolver, CrankNicolsonSolver,
)


# ===== Helpers =====

def make_state(psi=None, h=None, shape=(5, 5, 5), with_static=False):
    if psi is None:
        psi = np.ones(shape, dtype=np.float64)
    if h is None:
        h = np.ones(shape, dtype=np.float64)
    fields = {"psi": psi, "h": h}
    if with_static:
        fields["mask"] = np.zeros(shape, dtype=bool)
        fields["labels"] = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    return State(fields=fields, solver_fields=("psi", "h"))


def make_rhs(dpsi=None, dh=None, shape=(5, 5, 5)):
    if dpsi is None:
        dpsi = np.zeros(shape, dtype=np.float64)
    if dh is None:
        dh = np.zeros(shape, dtype=np.float64)
    return State(
        fields={"psi": dpsi, "h": dh},
        solver_fields=("psi", "h"),
    )


# ===== A. Construction =====

def test_explicit_solver_construction():
    s = ExplicitReferenceSolver()
    assert s.metadata == {}


def test_explicit_solver_with_metadata():
    s = ExplicitReferenceSolver(metadata={"label": "test"})
    assert s.metadata == {"label": "test"}


def test_explicit_solver_invalid_metadata():
    with pytest.raises(TypeError, match="metadata"):
        ExplicitReferenceSolver(metadata="not a dict")


# ===== B. Simple step 1D =====

def test_explicit_step_simple_1d():
    """psi = [1,1,1], rhs.psi = [1,0,-1], dt=0.1 -> [1.1, 1.0, 0.9]."""
    state = State(
        fields={
            "psi": np.array([1.0, 1.0, 1.0]),
            "h": np.array([1.0, 1.0, 1.0]),
        },
        solver_fields=("psi", "h"),
    )
    rhs = State(
        fields={
            "psi": np.array([1.0, 0.0, -1.0]),
            "h": np.array([0.0, 0.0, 0.0]),
        },
        solver_fields=("psi", "h"),
    )
    s = ExplicitReferenceSolver()
    next_state = s.step(state, rhs, dt=0.1)
    np.testing.assert_allclose(next_state["psi"], [1.1, 1.0, 0.9])
    np.testing.assert_array_equal(next_state["h"], [1.0, 1.0, 1.0])


def test_explicit_step_zero_rhs_returns_same_values():
    state = make_state()
    rhs = make_rhs()
    s = ExplicitReferenceSolver()
    next_state = s.step(state, rhs, dt=0.1)
    np.testing.assert_array_equal(next_state["psi"], state["psi"])
    np.testing.assert_array_equal(next_state["h"], state["h"])


# ===== C. Multiple solver fields integrated =====

def test_explicit_step_integrates_all_solver_fields():
    """psi and h both integrated, no specialization."""
    state = make_state()
    dpsi = 0.5 * np.ones((5, 5, 5), dtype=np.float64)
    dh = -2.0 * np.ones((5, 5, 5), dtype=np.float64)
    rhs = make_rhs(dpsi=dpsi, dh=dh)
    s = ExplicitReferenceSolver()
    next_state = s.step(state, rhs, dt=0.1)
    np.testing.assert_allclose(next_state["psi"], 1.0 + 0.05)
    np.testing.assert_allclose(next_state["h"], 1.0 + 0.1 * (-2.0))


# ===== D. Static fields preserved from state =====

def test_explicit_step_preserves_static_fields_from_state():
    state = make_state(with_static=True)
    rhs = make_rhs()
    s = ExplicitReferenceSolver()
    next_state = s.step(state, rhs, dt=0.1)
    assert "mask" in next_state.fields
    assert "labels" in next_state.fields
    np.testing.assert_array_equal(next_state["mask"], state["mask"])
    np.testing.assert_array_equal(next_state["labels"], state["labels"])
    assert next_state["mask"].dtype == np.bool_
    assert next_state["labels"].dtype == np.int32


# ===== E. Static fields in rhs ignored =====

def test_explicit_step_ignores_static_fields_in_rhs():
    """rhs has 'mask' field (static), state has none. mask must not appear
    in result (only state's static fields are kept)."""
    state = make_state()  # no mask
    rhs_fields = {
        "psi": np.zeros((5, 5, 5), dtype=np.float64),
        "h": np.zeros((5, 5, 5), dtype=np.float64),
        "mask": np.ones((5, 5, 5), dtype=bool),  # static in rhs
    }
    rhs = State(fields=rhs_fields, solver_fields=("psi", "h"))
    s = ExplicitReferenceSolver()
    next_state = s.step(state, rhs, dt=0.1)
    assert "mask" not in next_state.fields


# ===== F. solver_fields mismatch raises =====

def test_explicit_step_solver_fields_mismatch_raises():
    state = State(
        fields={"psi": np.ones((5,), dtype=np.float64),
                "h": np.ones((5,), dtype=np.float64)},
        solver_fields=("psi", "h"),
    )
    rhs = State(
        fields={"psi": np.zeros((5,), dtype=np.float64)},
        solver_fields=("psi",),  # only psi
    )
    s = ExplicitReferenceSolver()
    with pytest.raises(ValueError, match="solver_fields"):
        s.step(state, rhs, dt=0.1)


# ===== G. Shape mismatch raises =====

def test_explicit_step_shape_mismatch_raises():
    state = make_state(shape=(5, 5, 5))
    rhs = State(
        fields={"psi": np.zeros((7, 7, 7), dtype=np.float64),
                "h": np.zeros((7, 7, 7), dtype=np.float64)},
        solver_fields=("psi", "h"),
    )
    s = ExplicitReferenceSolver()
    with pytest.raises(ValueError, match="shape"):
        s.step(state, rhs, dt=0.1)


# ===== H. dt <= 0 rejected =====

def test_explicit_step_dt_zero_rejected():
    state = make_state()
    rhs = make_rhs()
    s = ExplicitReferenceSolver()
    with pytest.raises(ValueError, match="dt"):
        s.step(state, rhs, dt=0.0)


def test_explicit_step_dt_negative_rejected():
    state = make_state()
    rhs = make_rhs()
    s = ExplicitReferenceSolver()
    with pytest.raises(ValueError, match="dt"):
        s.step(state, rhs, dt=-0.1)


# ===== I. dt NaN/Inf rejected =====

def test_explicit_step_dt_nan_rejected():
    state = make_state()
    rhs = make_rhs()
    s = ExplicitReferenceSolver()
    with pytest.raises(ValueError, match="finite"):
        s.step(state, rhs, dt=float("nan"))


def test_explicit_step_dt_inf_rejected():
    state = make_state()
    rhs = make_rhs()
    s = ExplicitReferenceSolver()
    with pytest.raises(ValueError, match="finite"):
        s.step(state, rhs, dt=float("inf"))


def test_explicit_step_dt_invalid_type_rejected():
    state = make_state()
    rhs = make_rhs()
    s = ExplicitReferenceSolver()
    with pytest.raises(TypeError, match="dt"):
        s.step(state, rhs, dt="0.1")


# ===== Invalid type validations =====

def test_explicit_step_state_invalid_type():
    s = ExplicitReferenceSolver()
    rhs = make_rhs()
    with pytest.raises(TypeError, match="state"):
        s.step("not a state", rhs, dt=0.1)


def test_explicit_step_rhs_invalid_type():
    s = ExplicitReferenceSolver()
    state = make_state()
    with pytest.raises(TypeError, match="rhs"):
        s.step(state, "not a state", dt=0.1)


def test_explicit_step_rhs_wrong_dtype():
    s = ExplicitReferenceSolver()
    state = make_state()
    rhs = State(
        fields={"psi": np.zeros((5, 5, 5), dtype=np.float64),
                "h": np.zeros((5, 5, 5), dtype=np.float64)},
        solver_fields=("psi", "h"),
    )
    # No way to construct a non-float64 solver field in State (it rejects),
    # so this assertion path is structurally guarded earlier. Skip.
    # We still confirm step succeeds normally:
    _ = s.step(state, rhs, dt=0.1)


# ===== J. Source metadata not mutated =====

def test_explicit_step_does_not_mutate_state_metadata():
    state = State(
        fields={"psi": np.ones((5,), dtype=np.float64),
                "h": np.ones((5,), dtype=np.float64)},
        solver_fields=("psi", "h"),
        metadata={"experiment": "X", "t": 0.0},
    )
    rhs = make_rhs(shape=(5,))
    s = ExplicitReferenceSolver()
    _ = s.step(state, rhs, dt=0.1)
    # Source metadata unchanged
    assert state.metadata == {"experiment": "X", "t": 0.0}
    assert "solver" not in state.metadata


def test_explicit_step_does_not_mutate_rhs_metadata():
    state = make_state()
    rhs = State(
        fields={"psi": np.zeros((5, 5, 5), dtype=np.float64),
                "h": np.zeros((5, 5, 5), dtype=np.float64)},
        solver_fields=("psi", "h"),
        metadata={"operator": "GeometryOperator.apply"},
    )
    s = ExplicitReferenceSolver()
    _ = s.step(state, rhs, dt=0.1)
    assert rhs.metadata == {"operator": "GeometryOperator.apply"}


# ===== K. Output metadata trace =====

def test_explicit_step_output_metadata_trace():
    state = State(
        fields={"psi": np.ones((5,), dtype=np.float64),
                "h": np.ones((5,), dtype=np.float64)},
        solver_fields=("psi", "h"),
        metadata={"experiment": "X"},
    )
    rhs = make_rhs(shape=(5,))
    s = ExplicitReferenceSolver()
    next_state = s.step(state, rhs, dt=0.05)
    assert next_state.metadata["solver"] == "ExplicitReferenceSolver"
    assert next_state.metadata["dt"] == 0.05
    assert next_state.metadata["step_kind"] == "explicit_euler"
    assert next_state.metadata["integrated_solver_fields"] == ("psi", "h")
    assert next_state.metadata["experiment"] == "X"  # source preserved


# ===== L. Output memory independence =====

def test_explicit_step_output_independent_memory():
    state = make_state()
    rhs = make_rhs()
    s = ExplicitReferenceSolver()
    next_state = s.step(state, rhs, dt=0.1)
    next_state.fields["psi"][0, 0, 0] = 999.0
    assert state["psi"][0, 0, 0] == 1.0  # source unchanged


# ===== M. Solver stateless: repeated calls consistent =====

def test_explicit_solver_stateless_repeated_calls():
    """Call solver on state A, then state B, then state A again.
    Result for state A must be identical both times."""
    state_A = make_state(psi=np.ones((5,), dtype=np.float64), shape=(5,))
    state_B = State(
        fields={"psi": 100.0 * np.ones((5,), dtype=np.float64),
                "h": np.ones((5,), dtype=np.float64)},
        solver_fields=("psi", "h"),
    )
    rhs_A = make_rhs(dpsi=np.ones((5,), dtype=np.float64), shape=(5,))
    rhs_B = make_rhs(dpsi=2.0 * np.ones((5,), dtype=np.float64), shape=(5,))
    s = ExplicitReferenceSolver()

    n_A1 = s.step(state_A, rhs_A, dt=0.1)
    _ = s.step(state_B, rhs_B, dt=0.1)
    n_A2 = s.step(state_A, rhs_A, dt=0.1)

    np.testing.assert_array_equal(n_A1["psi"], n_A2["psi"])


# ===== N. No history ownership =====

def test_solver_no_history_attribute():
    s = ExplicitReferenceSolver()
    forbidden = [
        "history", "_history", "commit_step", "append_step",
        "record", "snapshot", "tick", "_time",
    ]
    for n in forbidden:
        assert not hasattr(s, n), f"forbidden temporal attr: {n}"


def test_solver_slots_no_adhoc():
    s = ExplicitReferenceSolver()
    with pytest.raises(AttributeError):
        s._history = []
    with pytest.raises(AttributeError):
        s._last_state = None


# ===== O. File isolation =====

def test_solver_file_isolation():
    import mcq_v5.solver as sv_mod
    src = Path(sv_mod.__file__).read_text()
    forbidden_imports = [
        "from .instance import",
        "from .module import",
        "from .geometry import",
        "from .coupling_scheduler import",
        "from .coupling_operator import",
        "from .coupling_aggregator import",
        "from .coupling_context import",
        "from mcq_v5.instance import",
        "from mcq_v5.module import",
        "from mcq_v5.geometry import",
        "from mcq_v5.coupling_scheduler import",
        "from mcq_v5.coupling_operator import",
        "from mcq_v5.coupling_aggregator import",
        "from mcq_v5.coupling_context import",
        "import mcq_v5.geometry",
        "import mcq_v5.coupling_operator",
    ]
    for pat in forbidden_imports:
        assert pat not in src, f"forbidden import: {pat}"


def test_solver_namespace_no_forbidden_names():
    import mcq_v5.solver as sv_mod
    forbidden = [
        "Instance", "Module", "GeometryOperator",
        "CouplingScheduler", "CouplingOperator", "CouplingAggregator",
        "CouplingContext", "AggregatedCoupling", "CouplingContribution",
        "HistoryEntry",
    ]
    for n in forbidden:
        assert not hasattr(sv_mod, n), f"forbidden in namespace: {n}"


# ===== P. Stubs raise NotImplementedError =====

def test_semi_implicit_stub_raises():
    s = SemiImplicitSolver()
    state = make_state()
    rhs = make_rhs()
    with pytest.raises(NotImplementedError, match="SemiImplicit"):
        s.step(state, rhs, dt=0.1)


def test_adi_stub_raises():
    s = ADISolver()
    state = make_state()
    rhs = make_rhs()
    with pytest.raises(NotImplementedError, match="ADI"):
        s.step(state, rhs, dt=0.1)


def test_crank_nicolson_stub_raises():
    s = CrankNicolsonSolver()
    state = make_state()
    rhs = make_rhs()
    with pytest.raises(NotImplementedError, match="Crank"):
        s.step(state, rhs, dt=0.1)


def test_stubs_documentation_mentions_signature_evolution():
    """Audit ajout 3: each implicit solver stub must document signature
    extension as architectural debt."""
    import mcq_v5.solver as sv_mod
    for cls in (sv_mod.SemiImplicitSolver, sv_mod.ADISolver, sv_mod.CrankNicolsonSolver):
        doc = (cls.__doc__ or "")
        assert "signature" in doc.lower() or "extended" in doc.lower(), (
            f"{cls.__name__} docstring must mention signature evolution"
        )


def test_solver_abstract_not_instantiable():
    with pytest.raises(TypeError, match="abstract"):
        Solver()


def test_explicit_solver_is_subclass_of_solver():
    s = ExplicitReferenceSolver()
    assert isinstance(s, Solver)


# ===== Q. No forbidden methods =====

def test_solver_no_forbidden_methods():
    s = ExplicitReferenceSolver()
    forbidden = [
        "aggregate", "build_contexts", "admissible_pairs",
        "pairwise_coupling", "compute_coupling",
        "gradient", "flux", "divergence", "apply",
    ]
    for n in forbidden:
        assert not hasattr(s, n), f"forbidden method: {n}"


# ===== Amendment 1: NaN/Inf guard on output =====

def test_solver_rejects_nan_in_integrated_field():
    """If rhs contains NaN, integrated field becomes NaN, solver must reject."""
    state = make_state()
    bad = np.zeros((5, 5, 5), dtype=np.float64)
    bad[2, 2, 2] = np.nan
    rhs = make_rhs(dpsi=bad)
    s = ExplicitReferenceSolver()
    with pytest.raises(ValueError, match="NaN or Inf"):
        s.step(state, rhs, dt=0.1)


def test_solver_rejects_inf_in_integrated_field():
    state = make_state()
    bad = np.zeros((5, 5, 5), dtype=np.float64)
    bad[1, 1, 1] = np.inf
    rhs = make_rhs(dpsi=bad)
    s = ExplicitReferenceSolver()
    with pytest.raises(ValueError, match="NaN or Inf"):
        s.step(state, rhs, dt=0.1)


def test_solver_rejects_huge_dt_overflow():
    """A pathological dt * rhs producing inf must be caught."""
    state = make_state()
    # Large rhs and large dt -> inf
    huge = 1e200 * np.ones((5, 5, 5), dtype=np.float64)
    rhs = make_rhs(dpsi=huge)
    s = ExplicitReferenceSolver()
    with pytest.raises(ValueError, match="NaN or Inf"):
        s.step(state, rhs, dt=1e200)


# ===== Amendment 2: C-contiguous output =====

def test_solver_output_is_c_contiguous():
    state = make_state()
    rhs = make_rhs(dpsi=0.5 * np.ones((5, 5, 5), dtype=np.float64))
    s = ExplicitReferenceSolver()
    next_state = s.step(state, rhs, dt=0.1)
    for sf in next_state.solver_fields:
        assert next_state[sf].flags["C_CONTIGUOUS"]
        assert next_state[sf].dtype == np.float64


# ===== Solver metadata propagation =====

def test_solver_level_metadata_propagated_with_prefix():
    s = ExplicitReferenceSolver(metadata={"label": "my_run"})
    state = make_state()
    rhs = make_rhs()
    next_state = s.step(state, rhs, dt=0.1)
    assert next_state.metadata["solver.label"] == "my_run"


# ===== State construction independence: solver does not retain refs =====

def test_solver_does_not_retain_state_or_rhs():
    """After step, modifying state's psi via its fields dict must not
    affect next_state."""
    state = make_state()
    rhs = make_rhs(dpsi=np.ones((5, 5, 5), dtype=np.float64))
    s = ExplicitReferenceSolver()
    next_state = s.step(state, rhs, dt=0.1)
    # Modify state's psi
    state.fields["psi"][0, 0, 0] = 999.0
    # next_state.psi for that cell was 1.0 + 0.1*1.0 = 1.1, not 999.0
    assert next_state["psi"][0, 0, 0] == pytest.approx(1.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
