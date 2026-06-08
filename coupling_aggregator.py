# -*- coding: utf-8 -*-
"""
mcq_v5.coupling_aggregator — Routes and aggregates CouplingContributions.

Responsibility:
    CouplingContribution[] → routing by kind → RHS per target_id

Non-responsibilities:
- pairwise_coupling (belongs to CouplingOperator)
- novelty / self_form
- scheduler / build_contexts
- Instance / Module reading
- CouplingContext reading
- Solver step

Routing by kind:
- kind="source":  rhs += contribution.value
- kind="flux":    rhs += GeometryOperator.divergence(contribution.value)
                  (recall divergence returns -div(J), conservative RHS)
- kind="mixed":   rhs += source + GeometryOperator.divergence(fluxes)
- kind="none":    no contribution to RHS

Aggregation grouped by target_id (NOT by source_id).

Mass policies:
- "conservative":   all contributions must have conservative=True;
                    abs(sum(rhs)) <= mass_tolerance enforced redundantly
- "source_allowed": non-conservative sources accepted, mass_delta reported
- "bounded_source": abs(sum(rhs)) <= source_bound enforced
- "diagnostic_only": never raises on mass violation, reports flags

The aggregator returns AggregatedCoupling only for target_ids that
received at least one contribution. Modules with no contribution are
absent from the returned dict — handling absence is the Solver's job.

Critère étape 11:
    CouplingAggregator routes CouplingContribution by kind, aggregates
    by target_id under explicit mass policies, returns immutable
    AggregatedCoupling objects, and never reads Instance, Module,
    Scheduler, CouplingContext, or CouplingOperator.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping
import numpy as np

from .geometry import GeometryOperator
from .coupling_operator import CouplingContribution, ModuleId


MassPolicy = Literal[
    "conservative",
    "source_allowed",
    "bounded_source",
    "diagnostic_only",
]
_VALID_POLICIES = (
    "conservative", "source_allowed", "bounded_source", "diagnostic_only",
)


@dataclass(frozen=True, slots=True)
class AggregatedCoupling:
    """Immutable result of aggregating contributions for one target_id.

    Fields
    ------
    target_id : ModuleId
    rhs : np.ndarray
        Total RHS contribution from all coupling sources to this target.
        Shape grid.shape, float64, C-contiguous, writeable=False.
    mass_delta : float
        np.sum(rhs). Reported regardless of policy.
    mass_policy : str
        The policy under which this aggregate was computed.
    n_contributions : int
        Number of CouplingContribution objects that contributed.
    diagnostics : Mapping[str, Any]
        Read-only diagnostics (mass violations, source ids included, etc.).
    """

    target_id: ModuleId
    rhs: np.ndarray
    mass_delta: float
    mass_policy: str
    n_contributions: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # rhs validation
        if not isinstance(self.rhs, np.ndarray):
            raise TypeError(f"rhs must be ndarray, got {type(self.rhs).__name__}")
        if self.rhs.dtype != np.float64:
            raise TypeError(f"rhs dtype must be float64, got {self.rhs.dtype}")
        if not self.rhs.flags["C_CONTIGUOUS"]:
            raise ValueError("rhs must be C-contiguous")
        # Lock writeable=False with a defensive copy
        r = self.rhs.copy()
        r.flags.writeable = False
        object.__setattr__(self, "rhs", r)

        # mass_delta
        if not isinstance(self.mass_delta, (int, float, np.floating)):
            raise TypeError("mass_delta must be numeric")
        object.__setattr__(self, "mass_delta", float(self.mass_delta))

        # n_contributions
        if not isinstance(self.n_contributions, int) or self.n_contributions < 0:
            raise ValueError("n_contributions must be non-negative int")

        # diagnostics MappingProxyType
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be Mapping")
        proxy = MappingProxyType(dict(self.diagnostics))
        object.__setattr__(self, "diagnostics", proxy)


class CouplingAggregator:
    """Routes and aggregates CouplingContributions into RHS per target.

    Constructor parameters:
        geometry: GeometryOperator used for flux→divergence routing.
        mass_tolerance: tolerance for 'conservative' policy
            (default 1e-12).
        source_bound: bound for 'bounded_source' policy
            (None => bounded_source forbidden unless overridden at call).
        metadata: optional dict, defensively copied.
    """

    __slots__ = (
        "_geometry",
        "_mass_tolerance",
        "_source_bound",
        "_metadata",
    )

    def __init__(
        self,
        geometry: GeometryOperator,
        mass_tolerance: float = 1e-12,
        source_bound: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not isinstance(geometry, GeometryOperator):
            raise TypeError(
                f"geometry must be GeometryOperator, got {type(geometry).__name__}"
            )
        if not isinstance(mass_tolerance, (int, float, np.floating)):
            raise TypeError("mass_tolerance must be numeric")
        if float(mass_tolerance) < 0.0:
            raise ValueError("mass_tolerance must be >= 0")
        if source_bound is not None:
            if not isinstance(source_bound, (int, float, np.floating)):
                raise TypeError("source_bound must be numeric or None")
            if float(source_bound) < 0.0:
                raise ValueError("source_bound must be >= 0")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be dict or None")

        self._geometry = geometry
        self._mass_tolerance = float(mass_tolerance)
        self._source_bound = (
            float(source_bound) if source_bound is not None else None
        )
        self._metadata = dict(metadata) if metadata else {}

    @property
    def geometry(self) -> GeometryOperator:
        return self._geometry

    @property
    def mass_tolerance(self) -> float:
        return self._mass_tolerance

    @property
    def source_bound(self) -> float | None:
        return self._source_bound

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    # === Routing per contribution kind ===

    def _route(self, contribution: CouplingContribution) -> np.ndarray:
        """Route one contribution to its grid-shape RHS array."""
        grid_shape = self._geometry.grid.shape
        kind = contribution.kind
        if kind == "source":
            arr = contribution.value
            if tuple(arr.shape) != tuple(grid_shape):
                raise ValueError(
                    f"source contribution shape {arr.shape} != grid shape "
                    f"{grid_shape}"
                )
            # Return a writable copy so it can be accumulated downstream
            return arr.copy()
        elif kind == "flux":
            # GeometryOperator.divergence validates tuple length & shapes,
            # returns float64 C-contiguous of shape grid.shape
            return self._geometry.divergence(contribution.value)
        elif kind == "mixed":
            src, flx = contribution.value
            if tuple(src.shape) != tuple(grid_shape):
                raise ValueError(
                    f"mixed source shape {src.shape} != grid shape {grid_shape}"
                )
            div_part = self._geometry.divergence(flx)
            return src.copy() + div_part
        elif kind == "none":
            return np.zeros(grid_shape, dtype=np.float64)
        else:
            raise ValueError(f"unknown kind {kind!r}")

    # === Policy enforcement ===

    @staticmethod
    def _check_all_conservative(
        contributions: list[CouplingContribution],
        target_id: ModuleId,
    ) -> None:
        """Audit ajout 2: 'conservative' is inflexible. Any contribution
        with conservative=False triggers immediate rejection."""
        for c in contributions:
            if not c.conservative:
                raise ValueError(
                    f"conservative policy: contribution from {c.source_id!r} "
                    f"to {target_id!r} declares conservative=False"
                )

    def _enforce_policy(
        self,
        target_id: ModuleId,
        rhs: np.ndarray,
        contributions: list[CouplingContribution],
        policy: MassPolicy,
    ) -> dict[str, Any]:
        """Apply mass policy. Return diagnostics dict (may raise)."""
        mass_delta = float(np.sum(rhs))
        diagnostics: dict[str, Any] = {
            "mass_delta": mass_delta,
            "n_contributions": len(contributions),
            "source_ids": tuple(c.source_id for c in contributions),
            "policy": policy,
        }

        # Per-policy enforcement
        if policy == "conservative":
            # Pre-check (inflexible)
            self._check_all_conservative(contributions, target_id)
            # Redundant numerical check
            if abs(mass_delta) > self._mass_tolerance:
                raise ValueError(
                    f"conservative policy: |mass_delta|={abs(mass_delta)} "
                    f"exceeds tolerance {self._mass_tolerance} for target {target_id!r}"
                )
            diagnostics["mass_policy_violation"] = False

        elif policy == "source_allowed":
            diagnostics["mass_policy_violation"] = False  # never violates by design

        elif policy == "bounded_source":
            if self._source_bound is None:
                raise ValueError(
                    "bounded_source policy requires source_bound to be set "
                    "at aggregator construction"
                )
            if abs(mass_delta) > self._source_bound:
                raise ValueError(
                    f"bounded_source policy: |mass_delta|={abs(mass_delta)} "
                    f"exceeds source_bound {self._source_bound} for target {target_id!r}"
                )
            diagnostics["mass_policy_violation"] = False
            diagnostics["source_bound"] = self._source_bound

        elif policy == "diagnostic_only":
            # Never raises on mass violation
            would_violate_conservative = (
                abs(mass_delta) > self._mass_tolerance
                or any(not c.conservative for c in contributions)
            )
            would_violate_bounded_source = (
                self._source_bound is not None
                and abs(mass_delta) > self._source_bound
            )
            diagnostics["mass_policy_violation"] = False  # by definition
            diagnostics["would_violate_conservative"] = would_violate_conservative
            diagnostics["would_violate_bounded_source"] = (
                would_violate_bounded_source
            )

        else:
            raise ValueError(f"unknown policy {policy!r}")

        return diagnostics

    # === Public API ===

    def aggregate(
        self,
        contributions: tuple[CouplingContribution, ...],
        policy: MassPolicy = "conservative",
    ) -> dict[ModuleId, AggregatedCoupling]:
        """Aggregate contributions by target_id, return dict per target.

        Only target_ids that receive at least one contribution appear in
        the result. Modules with no incoming contribution are absent.
        """
        # Validation
        if not isinstance(contributions, tuple):
            raise TypeError(
                f"contributions must be tuple, got {type(contributions).__name__}"
            )
        if policy not in _VALID_POLICIES:
            raise ValueError(
                f"policy must be one of {_VALID_POLICIES}, got {policy!r}"
            )
        for i, c in enumerate(contributions):
            if not isinstance(c, CouplingContribution):
                raise TypeError(
                    f"contributions[{i}] must be CouplingContribution, "
                    f"got {type(c).__name__}"
                )

        # Empty: return empty dict (no inventing zero RHS for modules)
        if len(contributions) == 0:
            return {}

        # Group contributions by target_id
        by_target: dict[ModuleId, list[CouplingContribution]] = {}
        for c in contributions:
            by_target.setdefault(c.target_id, []).append(c)

        # Aggregate per target
        result: dict[ModuleId, AggregatedCoupling] = {}
        grid_shape = self._geometry.grid.shape
        for target_id, target_contribs in by_target.items():
            # Accumulate routed RHS
            rhs = np.zeros(grid_shape, dtype=np.float64)
            for c in target_contribs:
                routed = self._route(c)
                rhs = rhs + routed

            # --- Audit ajout 1: final NaN/Inf guard ---
            # Mathematical stability check after summation, before locking.
            if not np.all(np.isfinite(rhs)):
                raise ValueError(
                    f"Aggregated RHS for target {target_id!r} contains "
                    f"NaN or Inf"
                )

            # Apply policy (may raise)
            diagnostics = self._enforce_policy(
                target_id, rhs, target_contribs, policy
            )

            # Build immutable AggregatedCoupling
            agg = AggregatedCoupling(
                target_id=target_id,
                rhs=rhs,
                mass_delta=float(np.sum(rhs)),
                mass_policy=policy,
                n_contributions=len(target_contribs),
                diagnostics=diagnostics,
            )
            result[target_id] = agg

        return result

    def __repr__(self) -> str:
        return (
            f"CouplingAggregator(geometry={self._geometry!r}, "
            f"mass_tolerance={self._mass_tolerance}, "
            f"source_bound={self._source_bound})"
        )
