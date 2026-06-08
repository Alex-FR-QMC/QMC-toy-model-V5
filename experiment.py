# -*- coding: utf-8 -*-
"""
mcq_v5.experiment — ExperimentProtocol orchestrator.

Last V5-0a link. Orchestrates existing components without inventing
physics. The chain:

    GeometryOperator
    -> CouplingScheduler
    -> CouplingOperator
    -> CouplingAggregator
    -> Solver
    -> commit HistoryEntry

ExperimentProtocol is the SOLE maintainer of committed HistoryEntry
objects. Instance does not own history. Solver does not own history.
HistoryView is constructed FROM ExperimentProtocol._history (filtered).

History policy:
- Every accepted macro-step commits one HistoryEntry per module
  with committed=True.
- Checkpoints are METADATA markers (set via checkpoint_interval),
  NOT the condition for committing.
- Substeps (future RK4/ADI/CN) are NEVER committed and never appear
  in HistoryView.

Continuous-trajectory note:
    ExperimentProtocol assumes it is the only actor mutating the
    instance/state. If a caller modifies protocol._instance or any
    module's state externally between step() calls, the history will
    keep incrementing as if nothing happened — but semantic continuity
    is the caller's responsibility, not the protocol's.

Initial commit:
    commit_initial=True (default) captures the instance EXACTLY as
    passed to the constructor (no stochastic alteration at __init__
    in V5-0a). Time = 0.0, step = 0.

Critère étape 14:
    ExperimentProtocol is the sole maintainer of committed HistoryEntry
    objects; step() performs exactly one macro-step by orchestrating
    GeometryOperator, CouplingScheduler, CouplingOperator,
    CouplingAggregator and Solver; run(n_steps) is only a loop over
    step(); every accepted macro-step commits HistoryEntry objects for
    each module, while checkpoints are metadata/export markers, not the
    condition for history; substeps are never committed and never
    visible to HistoryView; mandatory protocol metadata are declared
    and exported.
"""

from __future__ import annotations
from typing import Any
import math
import numpy as np

from .state import State
from .grid import Grid
from .module import Module, ModuleId
from .instance import Instance
from .views import GridView, HistoryView, HistoryEntry
from .geometry import GeometryOperator
from .coupling_scheduler import CouplingScheduler
from .coupling_operator import CouplingOperator
from .coupling_aggregator import CouplingAggregator, AggregatedCoupling
from .solver import Solver
from .diagnostics import Diagnostics


# Mandatory protocol metadata keys
_REQUIRED_METADATA_KEYS = (
    "protocol_name",
    "version",
    "grid_shape",
    "metric_type",
    "N",
    "M",
    "coupling_policy",
    "conservation_policy",
    "history_policy",
    "seeds",
    "checkpoints",
    "guardrails",
)


class ExperimentProtocol:
    """Orchestrator. Sole maintainer of committed HistoryEntry.

    Construction:
        protocol_name, version : identification (required, str)
        instance : starting Instance
        grid : Grid for spatial operations
        geometry, scheduler, coupling_operator, aggregator, solver :
            pre-built components
        diagnostics : optional Diagnostics object
        dt : macro-step length, > 0
        conservation_policy : mass policy forwarded to aggregator
        checkpoint_interval : if not None, every k-th step gets
            metadata["checkpoint"] = True (commit happens regardless)
        seeds, guardrails : optional dicts for protocol metadata
        commit_initial : whether to commit step 0 at construction
            (default True)
        metadata : optional extra metadata

    The protocol does NOT create internal components. It orchestrates
    objects already constructed by the caller.
    """

    __slots__ = (
        "_protocol_name",
        "_version",
        "_instance",
        "_grid",
        "_grid_view",
        "_geometry",
        "_scheduler",
        "_coupling_operator",
        "_aggregator",
        "_solver",
        "_diagnostics",
        "_dt",
        "_conservation_policy",
        "_checkpoint_interval",
        "_seeds",
        "_guardrails",
        "_metadata",
        "_history",
        "_step",
        "_time",
        "_last_step_diagnostics",
    )

    def __init__(
        self,
        *,
        protocol_name: str,
        version: str,
        instance: Instance,
        grid: Grid,
        geometry: GeometryOperator,
        scheduler: CouplingScheduler,
        coupling_operator: CouplingOperator,
        aggregator: CouplingAggregator,
        solver: Solver,
        dt: float,
        diagnostics: Diagnostics | None = None,
        conservation_policy: str = "conservative",
        checkpoint_interval: int | None = None,
        seeds: dict | None = None,
        guardrails: dict | None = None,
        commit_initial: bool = True,
        metadata: dict | None = None,
    ):
        # --- Required string ids ---
        if not isinstance(protocol_name, str) or not protocol_name:
            raise ValueError("protocol_name must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise ValueError("version must be a non-empty string")

        # --- Component types ---
        if not isinstance(instance, Instance):
            raise TypeError("instance must be Instance")
        if not isinstance(grid, Grid):
            raise TypeError("grid must be Grid")
        if not isinstance(geometry, GeometryOperator):
            raise TypeError("geometry must be GeometryOperator")
        if not isinstance(scheduler, CouplingScheduler):
            raise TypeError("scheduler must be CouplingScheduler")
        if not isinstance(coupling_operator, CouplingOperator):
            raise TypeError("coupling_operator must be CouplingOperator")
        if not isinstance(aggregator, CouplingAggregator):
            raise TypeError("aggregator must be CouplingAggregator")
        if not isinstance(solver, Solver):
            raise TypeError("solver must be Solver")
        if diagnostics is not None and not isinstance(diagnostics, Diagnostics):
            raise TypeError("diagnostics must be Diagnostics or None")

        # --- dt ---
        if not isinstance(dt, (int, float, np.floating)):
            raise TypeError("dt must be numeric")
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt}")

        # --- conservation_policy ---
        if not isinstance(conservation_policy, str):
            raise TypeError("conservation_policy must be str")

        # --- checkpoint_interval ---
        if checkpoint_interval is not None:
            if not isinstance(checkpoint_interval, int) or checkpoint_interval <= 0:
                raise ValueError("checkpoint_interval must be positive int or None")

        # --- optional dicts ---
        if seeds is not None and not isinstance(seeds, dict):
            raise TypeError("seeds must be dict or None")
        if guardrails is not None and not isinstance(guardrails, dict):
            raise TypeError("guardrails must be dict or None")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be dict or None")

        # --- Compatibility ---
        # GeometryOperator and the aggregator's internal GeometryOperator
        # must share the SAME grid (shape AND dx). This prevents silent
        # divergence when CouplingAggregator routes kind="flux" via its
        # own geometry: a mismatched dx would convert flux to RHS with a
        # different finite-volume convention than the protocol's geometry.
        if tuple(geometry.grid.shape) != tuple(grid.shape):
            raise ValueError(
                f"geometry.grid.shape {geometry.grid.shape} != grid.shape "
                f"{grid.shape}"
            )
        if tuple(geometry.grid.dx) != tuple(grid.dx):
            raise ValueError(
                f"geometry.grid.dx {geometry.grid.dx} != grid.dx {grid.dx}"
            )
        if tuple(aggregator.geometry.grid.shape) != tuple(grid.shape):
            raise ValueError(
                f"aggregator.geometry.grid.shape "
                f"{aggregator.geometry.grid.shape} != grid.shape {grid.shape}"
            )
        if tuple(aggregator.geometry.grid.dx) != tuple(grid.dx):
            raise ValueError(
                f"aggregator.geometry.grid.dx "
                f"{aggregator.geometry.grid.dx} != grid.dx {grid.dx}"
            )

        # --- Store ---
        self._protocol_name = protocol_name
        self._version = version
        self._instance = instance
        self._grid = grid
        self._grid_view = GridView(grid)
        self._geometry = geometry
        self._scheduler = scheduler
        self._coupling_operator = coupling_operator
        self._aggregator = aggregator
        self._solver = solver
        self._diagnostics = diagnostics
        self._dt = float(dt)
        self._conservation_policy = conservation_policy
        self._checkpoint_interval = checkpoint_interval
        self._seeds = dict(seeds) if seeds else {}
        self._guardrails = dict(guardrails) if guardrails else {}
        self._metadata = dict(metadata) if metadata else {}

        # --- History list (sole maintainer) ---
        self._history: list[HistoryEntry] = []
        self._step = 0
        self._time = 0.0
        self._last_step_diagnostics: dict = {}

        # --- Initial commit ---
        # Captures the instance exactly as passed (no stochastic alteration
        # at construction in V5-0a).
        if commit_initial:
            self._commit_history()

    # =====================================================================
    # Read-only accessors
    # =====================================================================

    @property
    def protocol_name(self) -> str:
        return self._protocol_name

    @property
    def version(self) -> str:
        return self._version

    @property
    def instance(self) -> Instance:
        return self._instance

    @property
    def time(self) -> float:
        return self._time

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def last_step_diagnostics(self) -> dict:
        return dict(self._last_step_diagnostics)

    # =====================================================================
    # Mandatory protocol metadata
    # =====================================================================

    def export_protocol_metadata(self) -> dict:
        """Return the full mandatory metadata bundle (all keys required)."""
        # Determine metric_type
        metric_types = set()
        for mid in self._instance.module_ids():
            m = self._instance.get_module(mid)
            metric_types.add(type(m.metric).__name__)
        if len(metric_types) == 0:
            metric_type = "none"
        elif len(metric_types) == 1:
            mt = next(iter(metric_types))
            if mt == "ScalarConformalMetric":
                metric_type = "scalar_conformal"
            elif mt == "TensorMetric":
                metric_type = "tensor_stub"
            else:
                metric_type = mt
        else:
            metric_type = "mixed"

        coupling_policy = {
            "operator": type(self._coupling_operator).__name__,
            "overlap_mode": self._scheduler.overlap_mode,
            "epsilon_R_min": self._scheduler.epsilon_R_min,
            "epsilon_R_max": self._scheduler.epsilon_R_max,
        }

        history_policy = {
            "commit": "every_macro_step",
            "checkpoint_interval": self._checkpoint_interval,
            "substep_exclusion": True,
            "sole_maintainer": "ExperimentProtocol",
        }

        bundle = {
            "protocol_name": self._protocol_name,
            "version": self._version,
            "grid_shape": tuple(int(s) for s in self._grid.shape),
            "metric_type": metric_type,
            "N": 1,  # V5-0a: single instance
            "M": int(self._instance.n_modules()),
            "coupling_policy": coupling_policy,
            "conservation_policy": self._conservation_policy,
            "history_policy": history_policy,
            "seeds": dict(self._seeds),
            "checkpoints": {
                "interval": self._checkpoint_interval,
                "current_step": self._step,
            },
            "guardrails": dict(self._guardrails),
            "dt": self._dt,
            "extra_metadata": dict(self._metadata),
        }
        # Sanity: all required keys present
        for k in _REQUIRED_METADATA_KEYS:
            if k not in bundle:
                raise RuntimeError(f"missing required metadata key: {k}")
        return bundle

    # =====================================================================
    # step / run
    # =====================================================================

    def step(self) -> Instance:
        """Perform exactly one macro-step.

        Orchestration:
            1. Geometry RHS per module
            2. Build CouplingContexts via scheduler
            3. Pairwise coupling via operator
            4. Aggregate by target via aggregator
            5. Merge geom + coupling RHS per module
            6. Integrate via solver
            7. Build next Instance
            8. Advance time / step
            9. Commit HistoryEntry per module (committed=True)
        """
        current_instance = self._instance

        # 1. Geometry RHS per module
        geom_rhs_by_module: dict = {}
        for mid in current_instance.module_ids():
            module = current_instance.get_module(mid)
            geom_rhs_by_module[mid] = self._geometry.apply(
                module.state, module.metric, field_name="psi",
            )

        # 2. Coupling contexts (scheduler is the only one reading Instance)
        contexts = self._scheduler.build_contexts(
            current_instance, self._grid_view, time=self._time,
        )

        # 3. Pairwise coupling (operator only sees ctx)
        contributions = tuple(
            self._coupling_operator.pairwise_coupling(ctx) for ctx in contexts
        )

        # 4. Aggregation by target
        aggregated = self._aggregator.aggregate(
            contributions, policy=self._conservation_policy,
        )

        # 5/6. Merge RHS per module + integrate
        next_modules: dict = {}
        for mid in current_instance.module_ids():
            module = current_instance.get_module(mid)
            rhs = geom_rhs_by_module[mid]
            if mid in aggregated:
                rhs = self._add_coupling_rhs(
                    rhs, aggregated[mid], field_name="psi",
                )
            next_state = self._solver.step(module.state, rhs, self._dt)

            # Optional guardrail: reject NaN/Inf in solver_fields
            if self._guardrails.get("reject_nonfinite_state", False):
                for sf in next_state.solver_fields:
                    if not np.all(np.isfinite(next_state[sf])):
                        raise ValueError(
                            f"guardrail violation: module {mid!r} solver "
                            f"field {sf!r} contains NaN/Inf"
                        )

            next_modules[mid] = module.with_state(next_state)

        # 7. New Instance
        self._instance = current_instance.with_modules(next_modules)

        # 8. Advance time
        self._step += 1
        self._time += self._dt

        # 9. Commit HistoryEntry for each module
        self._commit_history()

        # Optional last-step diagnostics
        self._last_step_diagnostics = {
            "n_contributions": int(len(contributions)),
            "n_aggregated_targets": int(len(aggregated)),
            "step": int(self._step),
            "time": float(self._time),
        }

        return self._instance

    def run(self, n_steps: int) -> Instance:
        """Run n_steps macro-steps. n_steps == 0 is a no-op."""
        if not isinstance(n_steps, int):
            raise TypeError("n_steps must be int")
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        for _ in range(n_steps):
            self.step()
        return self._instance

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _add_coupling_rhs(
        self,
        rhs: State,
        coupling: AggregatedCoupling,
        field_name: str = "psi",
    ) -> State:
        """Add aggregated coupling RHS to the geometry RHS for one field.

        Strict numpy operation on underlying arrays. Returns a NEW State
        (immutability of original rhs is preserved). Other solver fields
        and static fields are preserved as-is.
        """
        if not isinstance(rhs, State):
            raise TypeError("rhs must be State")
        if not isinstance(coupling, AggregatedCoupling):
            raise TypeError("coupling must be AggregatedCoupling")
        if field_name not in rhs.solver_fields:
            raise ValueError(
                f"field_name {field_name!r} not in rhs.solver_fields "
                f"{rhs.solver_fields}"
            )
        rhs_field = rhs.fields[field_name]
        if rhs_field.shape != coupling.rhs.shape:
            raise ValueError(
                f"shape mismatch: rhs[{field_name!r}] {rhs_field.shape} "
                f"vs coupling.rhs {coupling.rhs.shape}"
            )

        # Strict numpy addition; coupling.rhs is writeable=False but
        # arithmetic produces a new writeable array.
        new_field = rhs_field + coupling.rhs
        if not new_field.flags["C_CONTIGUOUS"]:
            new_field = np.ascontiguousarray(new_field)
        if new_field.dtype != np.float64:
            new_field = new_field.astype(np.float64)

        # Reconstruct fields dict
        new_fields: dict = {}
        for sf in rhs.solver_fields:
            if sf == field_name:
                new_fields[sf] = new_field
            else:
                new_fields[sf] = rhs.fields[sf].copy()
        # Static fields
        for fn, arr in rhs.fields.items():
            if fn not in rhs.solver_fields:
                new_fields[fn] = arr.copy()

        # Metadata trace
        new_metadata = dict(rhs.metadata) if rhs.metadata else {}
        new_metadata["coupling_merged"] = True
        new_metadata["coupling_target"] = str(coupling.target_id)
        new_metadata["coupling_policy"] = str(coupling.mass_policy)

        return State(
            fields=new_fields,
            solver_fields=rhs.solver_fields,
            metadata=new_metadata,
        )

    def _commit_history(self) -> None:
        """Commit one HistoryEntry per module at the current (step, time)."""
        is_checkpoint = False
        if self._checkpoint_interval is not None:
            is_checkpoint = (self._step % self._checkpoint_interval) == 0
        for mid in self._instance.module_ids():
            module = self._instance.get_module(mid)
            entry = HistoryEntry(
                state=module.state,
                time=self._time,
                step=self._step,
                committed=True,  # ALWAYS True from this method
                metadata={
                    "instance_id": str(self._instance.id),
                    "module_id": str(mid),
                    "macro_step": int(self._step),
                    "checkpoint": bool(is_checkpoint),
                    "protocol_name": self._protocol_name,
                },
            )
            self._history.append(entry)

    # =====================================================================
    # History access
    # =====================================================================

    def history_entries(self) -> list[HistoryEntry]:
        """Return a copy of the internal history list."""
        return list(self._history)

    def history_view(self) -> HistoryView:
        """Return a HistoryView over the entire history."""
        return HistoryView(list(self._history))

    def history_view_for_module(self, module_id: ModuleId) -> HistoryView:
        """Return a HistoryView filtered to one module_id.

        Filtering reads metadata['module_id']. Source of truth remains
        self._history (this returns a filtered copy, not a separate list).
        """
        entries = [
            e for e in self._history
            if e.metadata.get("module_id") == str(module_id)
        ]
        return HistoryView(entries)

    def __repr__(self) -> str:
        return (
            f"ExperimentProtocol(name={self._protocol_name!r}, "
            f"version={self._version!r}, step={self._step}, "
            f"time={self._time}, M={self._instance.n_modules()})"
        )
