# -*- coding: utf-8 -*-
"""
mcq_v5.solver — Time integration only.

Absolute principle:
    Solver defines NO physics.
    Solver calls NO GeometryOperator.
    Solver calls NO CouplingOperator / CouplingScheduler / CouplingAggregator.
    Solver does NOT read Instance.modules.
    Solver integrates only: State_current + dt * RHS_State -> State_next.

The chain of responsibility outside Solver:

    spatial dynamics     -> GeometryOperator
    pair construction    -> CouplingScheduler
    pairwise coupling    -> CouplingOperator
    routing aggregation  -> CouplingAggregator
    time integration     -> Solver (this module)
    history / protocol   -> ExperimentProtocol (later)

HistoryEntry creation is the responsibility of ExperimentProtocol, NOT
Solver. Solver returns a new State and never appends to any history.

Critère étape 12:
    ExplicitReferenceSolver performs only explicit Euler integration:
    State_next = State + dt * RHS_State.
    It defines no physics, reads no Instance/Module, owns no history,
    does not call geometry/coupling objects, preserves static fields from
    state, and leaves HistoryEntry creation to ExperimentProtocol.
    SemiImplicitSolver, ADISolver, and CrankNicolsonSolver are documented
    stubs.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
import numpy as np

from .state import State


# =========================================================================
# Abstract Solver
# =========================================================================

class Solver(ABC):
    """Abstract time integrator.

    Subclasses must implement step(state, rhs, dt) -> State.

    Note on signature evolution:
        The current abstract signature is designed for explicit solvers
        (Euler, RK*, ...) that only need a State RHS. Future implicit
        solvers (Semi-Implicit, Crank-Nicolson, ADI) will likely require
        access to the linear operator (or a sparse matrix representation)
        to invert (I - alpha*dt*L)psi = ... systems. For V5-0a, the
        signature is kept minimal; the abstract `step` may be extended
        when implicit solvers land.
    """

    __slots__ = ()

    @abstractmethod
    def step(self, state: State, rhs: State, dt: float) -> State:
        ...


# =========================================================================
# ExplicitReferenceSolver — Euler explicit
# =========================================================================

class ExplicitReferenceSolver(Solver):
    """Explicit Euler integrator.

    Convention:
        state_next[solver_field] = state[solver_field] + dt * rhs[solver_field]

    Static fields (non-solver) are preserved from `state`. Static fields
    appearing in `rhs` are ignored. Static fields appearing only in
    `state` are kept.

    The solver does NOT:
    - call any geometry/coupling object
    - read Instance.modules
    - maintain a history list
    - track global time (dt is passed per call)
    """

    __slots__ = ("_metadata",)

    def __init__(self, metadata: dict | None = None):
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be dict or None")
        self._metadata = dict(metadata) if metadata else {}

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def step(self, state: State, rhs: State, dt: float) -> State:
        """One explicit Euler step.

        Validation:
        - state is State, rhs is State
        - state.solver_fields == rhs.solver_fields
        - per solver field: same shape, dtype float64, C-contiguous
        - dt numeric, finite, > 0 (dt == 0 rejected to avoid spurious steps)

        Returns:
            A new State whose solver_fields are integrated and whose
            static fields are preserved from `state`. Metadata includes
            solver trace.
        """
        # --- Type validation ---
        if not isinstance(state, State):
            raise TypeError(f"state must be State, got {type(state).__name__}")
        if not isinstance(rhs, State):
            raise TypeError(f"rhs must be State, got {type(rhs).__name__}")

        # --- dt validation ---
        if not isinstance(dt, (int, float, np.floating)):
            raise TypeError(f"dt must be numeric, got {type(dt).__name__}")
        dt_f = float(dt)
        if not math.isfinite(dt_f):
            raise ValueError(f"dt must be finite, got {dt}")
        if dt_f <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")

        # --- solver_fields agreement ---
        if state.solver_fields != rhs.solver_fields:
            raise ValueError(
                f"solver_fields mismatch: state {state.solver_fields} "
                f"vs rhs {rhs.solver_fields}"
            )

        # --- per-field validation ---
        for sf in state.solver_fields:
            s_arr = state.fields[sf]
            r_arr = rhs.fields[sf]
            if s_arr.shape != r_arr.shape:
                raise ValueError(
                    f"shape mismatch on {sf!r}: state {s_arr.shape} "
                    f"vs rhs {r_arr.shape}"
                )
            if r_arr.dtype != np.float64:
                raise TypeError(
                    f"rhs field {sf!r}: dtype must be float64, got {r_arr.dtype}"
                )
            if not r_arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    f"rhs field {sf!r}: must be C-contiguous"
                )

        # --- Integrate solver_fields ---
        new_fields: dict = {}
        for sf in state.solver_fields:
            s_arr = state.fields[sf]
            r_arr = rhs.fields[sf]
            integrated = s_arr + dt_f * r_arr
            # Ensure C-contiguity explicitly (audit ajout 2: cast if needed)
            if not integrated.flags["C_CONTIGUOUS"]:
                integrated = np.ascontiguousarray(integrated)
            if integrated.dtype != np.float64:
                # Should not happen with float64 inputs, but defensive
                integrated = integrated.astype(np.float64)
            # --- Audit ajout 1: NaN/Inf guard on result ---
            if not np.all(np.isfinite(integrated)):
                raise ValueError(
                    f"integrated field {sf!r} contains NaN or Inf "
                    f"(CFL violation or rhs explosion suspected, dt={dt_f})"
                )
            new_fields[sf] = integrated

        # --- Preserve static fields from `state` (not from `rhs`) ---
        for fn, arr in state.fields.items():
            if fn not in state.solver_fields:
                new_fields[fn] = arr.copy()

        # --- Build metadata trace ---
        out_metadata = dict(state.metadata) if state.metadata else {}
        out_metadata["solver"] = "ExplicitReferenceSolver"
        out_metadata["dt"] = dt_f
        out_metadata["step_kind"] = "explicit_euler"
        out_metadata["integrated_solver_fields"] = tuple(state.solver_fields)
        # Merge solver-level metadata (e.g. label for the run)
        for k, v in self._metadata.items():
            out_metadata.setdefault(f"solver.{k}", v)

        return State(
            fields=new_fields,
            solver_fields=state.solver_fields,
            metadata=out_metadata,
        )

    def __repr__(self) -> str:
        return "ExplicitReferenceSolver()"


# =========================================================================
# Stubs for future implicit solvers
# =========================================================================

class SemiImplicitSolver(Solver):
    """V5-0a stub for semi-implicit time integration.

    Note: Future implicit solvers will likely require access to the
    GeometryOperator or a sparse matrix representation, meaning the
    abstract `step` signature may be extended. For V5-0a, this remains
    an architectural placeholder.
    """

    __slots__ = ()

    def step(self, state: State, rhs: State, dt: float) -> State:
        raise NotImplementedError(
            "SemiImplicitSolver is a V5-0a stub. Implicit time integration "
            "is deferred to a later phase and may require an extended "
            "signature (e.g., access to GeometryOperator or a linear "
            "operator representation)."
        )


class ADISolver(Solver):
    """V5-0a stub for Alternating Direction Implicit integration.

    Note: Future implicit solvers will likely require access to the
    GeometryOperator or a sparse matrix representation, meaning the
    abstract `step` signature may be extended. For V5-0a, this remains
    an architectural placeholder.
    """

    __slots__ = ()

    def step(self, state: State, rhs: State, dt: float) -> State:
        raise NotImplementedError(
            "ADISolver is a V5-0a stub. ADI integration is deferred to a "
            "later phase and may require an extended signature."
        )


class CrankNicolsonSolver(Solver):
    """V5-0a stub for Crank-Nicolson time integration.

    Note: Future implicit solvers will likely require access to the
    GeometryOperator or a sparse matrix representation, meaning the
    abstract `step` signature may be extended. For V5-0a, this remains
    an architectural placeholder.
    """

    __slots__ = ()

    def step(self, state: State, rhs: State, dt: float) -> State:
        raise NotImplementedError(
            "CrankNicolsonSolver is a V5-0a stub. Crank-Nicolson is "
            "deferred to a later phase and may require an extended "
            "signature (e.g., access to GeometryOperator)."
        )
