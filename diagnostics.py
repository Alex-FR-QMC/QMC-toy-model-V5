# -*- coding: utf-8 -*-
"""
mcq_v5.diagnostics — Observational layer.

Absolute principle:
    Diagnostics OBSERVE. They do not modify dynamics, do not step,
    do not aggregate, do not schedule, do not couple.

Diagnostics may READ:
- State
- Instance
- ndarray fluxes / coupling outputs
- CouplingContribution
- AggregatedCoupling

Diagnostics NEVER:
- mutate the read objects
- compute the coupling itself
- step time
- import GeometryOperator, CouplingScheduler, CouplingOperator,
  CouplingAggregator, or Solver as active objects (only the result
  dataclass AggregatedCoupling is imported)

Output convention:
    Each diagnostic returns a plain dict with:
        "name": <diagnostic name>
        "status": one of "PASS", "WARN", "FAIL", "UNDEFINED",
                  "NOT_IMPLEMENTED"
    Plus diagnostic-specific keys.

All scalar values in returned dicts are coerced to native Python types
(float, int, bool, str, tuple) to guarantee JSON-serializability by
downstream ExperimentProtocol.

NaN/Inf policy:
    Diagnostics handle non-finite inputs by returning status="UNDEFINED"
    (or "FAIL" if appropriate). They never propagate NaN silently and
    never raise on non-finite input.

Vocabulary:
    diagnostic, profile, status, B3-like, deferred, observational.
    Avoid: validated MCQ, proves, cognitive collapse, Ch4 confirmed.

Critère étape 13:
    Diagnostics observes State, Instance, fluxes, CouplingContribution
    and AggregatedCoupling without mutating them; reports mass
    conservation, positivity, underflow, functional profiles, P5bis-A
    inherited h/grad activity fields, and B3-like degeneracy; declares
    V5-2 diagnostic interfaces as NotImplementedError; performs no
    dynamics, coupling, scheduling, aggregation, or solving.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from .state import State
from .instance import Instance
from .coupling_operator import CouplingContribution
from .coupling_aggregator import AggregatedCoupling


# Status constants
STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_UNDEFINED = "UNDEFINED"
STATUS_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class Diagnostics:
    """Passive observational tool.

    Holds optional metadata only. No cache, no history.
    """

    __slots__ = ("_metadata",)

    def __init__(self, metadata: dict | None = None):
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be dict or None")
        self._metadata = dict(metadata) if metadata else {}

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    # =====================================================================
    # mass / mass_conservation
    # =====================================================================

    def mass(self, field: np.ndarray) -> dict:
        """Total mass (sum) of a field. No volume weighting in V5-0a."""
        if not isinstance(field, np.ndarray):
            raise TypeError("field must be ndarray")
        if not np.all(np.isfinite(field)):
            return {
                "name": "mass",
                "mass": None,
                "status": STATUS_UNDEFINED,
                "reason": "non_finite_values",
            }
        m = float(np.sum(field))
        return {
            "name": "mass",
            "mass": m,
            "status": STATUS_PASS,
        }

    def mass_conservation(
        self,
        before: np.ndarray,
        after: np.ndarray,
        tol: float = 1e-12,
    ) -> dict:
        """Check that sum(after) == sum(before) within tol."""
        if not isinstance(before, np.ndarray) or not isinstance(after, np.ndarray):
            raise TypeError("before and after must be ndarrays")
        if before.shape != after.shape:
            raise ValueError(
                f"shape mismatch: before {before.shape} vs after {after.shape}"
            )
        if not (np.all(np.isfinite(before)) and np.all(np.isfinite(after))):
            return {
                "name": "mass_conservation",
                "status": STATUS_UNDEFINED,
                "reason": "non_finite_values",
                "tol": float(tol),
            }
        mass_before = float(np.sum(before))
        mass_after = float(np.sum(after))
        delta = mass_after - mass_before
        ref = max(abs(mass_before), float(tol))
        relative_delta = delta / ref
        status = STATUS_PASS if abs(delta) <= float(tol) else STATUS_FAIL
        return {
            "name": "mass_conservation",
            "mass_before": mass_before,
            "mass_after": mass_after,
            "delta": float(delta),
            "relative_delta": float(relative_delta),
            "tol": float(tol),
            "status": status,
        }

    # =====================================================================
    # positivity
    # =====================================================================

    def positivity(
        self,
        field: np.ndarray,
        floor: float = 0.0,
        strict: bool = False,
    ) -> dict:
        if not isinstance(field, np.ndarray):
            raise TypeError("field must be ndarray")
        if not np.all(np.isfinite(field)):
            return {
                "name": "positivity",
                "status": STATUS_UNDEFINED,
                "reason": "non_finite_values",
                "floor": float(floor),
                "strict": bool(strict),
            }
        if strict:
            violations = field <= float(floor)
        else:
            violations = field < float(floor)
        n_viol = int(violations.sum())
        n_total = int(field.size)
        status = STATUS_PASS if n_viol == 0 else STATUS_FAIL
        return {
            "name": "positivity",
            "min": float(np.min(field)),
            "floor": float(floor),
            "strict": bool(strict),
            "n_violations": n_viol,
            "frac_violations": (float(n_viol) / n_total) if n_total > 0 else 0.0,
            "status": status,
        }

    # =====================================================================
    # underflow
    # =====================================================================

    def underflow(
        self,
        field: np.ndarray,
        threshold: float = 1e-300,
    ) -> dict:
        if not isinstance(field, np.ndarray):
            raise TypeError("field must be ndarray")
        if not np.all(np.isfinite(field)):
            return {
                "name": "underflow",
                "status": STATUS_UNDEFINED,
                "reason": "non_finite_values",
                "threshold": float(threshold),
            }
        abs_field = np.abs(field)
        mask = abs_field < float(threshold)
        n_under = int(mask.sum())
        n_total = int(field.size)
        status = STATUS_PASS if n_under == 0 else STATUS_WARN
        return {
            "name": "underflow",
            "threshold": float(threshold),
            "n_underflow": n_under,
            "frac_underflow": (float(n_under) / n_total) if n_total > 0 else 0.0,
            "min_abs": float(np.min(abs_field)),
            "status": status,
        }

    # =====================================================================
    # functional_profile
    # =====================================================================

    def functional_profile(
        self,
        field: np.ndarray,
        thresholds: dict[str, float] | None = None,
    ) -> dict:
        if not isinstance(field, np.ndarray):
            raise TypeError("field must be ndarray")

        # Count non-finite components
        finite_mask = np.isfinite(field)
        n_finite = int(finite_mask.sum())
        n_nan = int(np.isnan(field).sum())
        n_inf = int(np.isinf(field).sum())
        all_finite = bool(n_finite == field.size)

        if not all_finite:
            # Use nan-safe statistics; if all NaN, return UNDEFINED
            if n_finite == 0:
                return {
                    "name": "functional_profile",
                    "shape": tuple(int(s) for s in field.shape),
                    "n_finite": 0,
                    "n_nan": n_nan,
                    "n_inf": n_inf,
                    "finite": False,
                    "status": STATUS_UNDEFINED,
                    "reason": "all_values_non_finite",
                }
            # Partial: compute stats on FINITE VALUES ONLY for JSON safety.
            # nanmin/nanmax do not handle inf; we filter explicitly.
            finite_values = field[finite_mask]
            return {
                "name": "functional_profile",
                "shape": tuple(int(s) for s in field.shape),
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
                "median": float(np.median(finite_values)),
                "std": float(np.std(finite_values)),
                "variance": float(np.var(finite_values)),
                "l2_norm": float(np.linalg.norm(finite_values)),
                "n_finite": n_finite,
                "n_nan": n_nan,
                "n_inf": n_inf,
                "finite": False,
                "non_finite_policy": "finite_values_only",
                "status": STATUS_FAIL,
            }

        # All finite path
        out = {
            "name": "functional_profile",
            "shape": tuple(int(s) for s in field.shape),
            "min": float(np.min(field)),
            "max": float(np.max(field)),
            "mean": float(np.mean(field)),
            "median": float(np.median(field)),
            "std": float(np.std(field)),
            "variance": float(np.var(field)),
            "l2_norm": float(np.linalg.norm(field)),
            "n_finite": n_finite,
            "n_nan": 0,
            "n_inf": 0,
            "finite": True,
            "status": STATUS_PASS,
        }
        if thresholds:
            active_fractions = {}
            for name, thr in thresholds.items():
                active_fractions[name] = float((field > float(thr)).mean())
            out["active_fractions"] = active_fractions
        return out

    # =====================================================================
    # h_grad_activity_profile (P5bis-A inherited)
    # =====================================================================

    def h_grad_activity_profile(
        self,
        h_activity: np.ndarray,
        grad_activity: np.ndarray,
        h_threshold: float,
        grad_threshold: float,
    ) -> dict:
        """Returns the canonical P5bis-A diagnostic fields:
        frac_h_active, frac_grad_active, frac_intersection_h_grad,
        jaccard_h_grad, grad_status.

        Inputs must be shape-aligned ndarrays (caller responsibility).
        """
        if not isinstance(h_activity, np.ndarray) or not isinstance(
            grad_activity, np.ndarray
        ):
            raise TypeError("h_activity and grad_activity must be ndarrays")
        if h_activity.shape != grad_activity.shape:
            raise ValueError(
                f"shape mismatch: h_activity {h_activity.shape} vs "
                f"grad_activity {grad_activity.shape}"
            )
        if not (np.all(np.isfinite(h_activity)) and np.all(np.isfinite(grad_activity))):
            return {
                "name": "h_grad_activity_profile",
                "status": STATUS_UNDEFINED,
                "reason": "non_finite_values",
            }

        h_active = h_activity > float(h_threshold)
        grad_active = grad_activity > float(grad_threshold)
        intersection = h_active & grad_active
        union = h_active | grad_active

        frac_h_active = float(h_active.mean())
        frac_grad_active = float(grad_active.mean())
        frac_intersection = float(intersection.mean())

        # Audit ajout 2: native float conversion for Jaccard
        union_sum = float(union.sum())
        inter_sum = float(intersection.sum())
        jaccard = inter_sum / union_sum if union_sum > 0.0 else 0.0

        # grad_status classification
        if frac_h_active == 0.0 and frac_grad_active == 0.0:
            grad_status = "INACTIVE"
        elif frac_h_active == 0.0 and frac_grad_active > 0.0:
            grad_status = "NO_H_ACTIVE"
        elif frac_grad_active == 0.0:
            grad_status = "EMPTY"
        elif frac_intersection == 0.0:
            grad_status = "DISJOINT"
        else:
            grad_status = "CO_ACTIVE"

        return {
            "name": "h_grad_activity_profile",
            "frac_h_active": frac_h_active,
            "frac_grad_active": frac_grad_active,
            "frac_intersection_h_grad": frac_intersection,
            "jaccard_h_grad": jaccard,
            "grad_status": grad_status,
            "status": STATUS_PASS,
        }

    # =====================================================================
    # b3_like_degeneracy
    # =====================================================================

    def b3_like_degeneracy(
        self,
        field: np.ndarray,
        variance_tol: float = 1e-14,
        range_tol: float = 1e-12,
    ) -> dict:
        """Detect B3-like degeneracy (uniform-like field).

        Strictly observational: 'B3-like', not 'B3 proven'.
        """
        if not isinstance(field, np.ndarray):
            raise TypeError("field must be ndarray")
        if not np.all(np.isfinite(field)):
            return {
                "name": "b3_like_degeneracy",
                "status": STATUS_UNDEFINED,
                "reason": "non_finite_values",
            }
        variance = float(np.var(field))
        rng = float(np.max(field) - np.min(field))
        low_var = variance <= float(variance_tol)
        low_range = rng <= float(range_tol)
        degenerate = bool(low_var or low_range)
        if degenerate:
            if low_var and low_range:
                reason = "low_variance_and_low_range"
            elif low_var:
                reason = "low_variance"
            else:
                reason = "low_range"
        else:
            reason = "none"
        status = STATUS_WARN if degenerate else STATUS_PASS
        return {
            "name": "b3_like_degeneracy",
            "variance": variance,
            "range": rng,
            "variance_tol": float(variance_tol),
            "range_tol": float(range_tol),
            "b3_like_degenerate": degenerate,
            "reason": reason,
            "status": status,
        }

    # =====================================================================
    # instance_profile
    # =====================================================================

    def instance_profile(self, instance: Instance) -> dict:
        """Read-only summary of an Instance.

        Delegates to Module.diagnostics(); does not compute coupling,
        does not aggregate, does not modify any module.
        """
        if not isinstance(instance, Instance):
            raise TypeError(
                f"instance must be Instance, got {type(instance).__name__}"
            )
        module_diagnostics: dict = {}
        for mid in instance.module_ids():
            module_diagnostics[str(mid)] = instance.get_module(mid).diagnostics()
        return {
            "name": "instance_profile",
            "n_modules": int(instance.n_modules()),
            "module_ids": tuple(instance.module_ids()),
            "module_diagnostics": module_diagnostics,
            "status": STATUS_PASS,
        }

    # =====================================================================
    # contribution_profile
    # =====================================================================

    def contribution_profile(
        self,
        contributions: tuple[CouplingContribution, ...],
    ) -> dict:
        if not isinstance(contributions, tuple):
            raise TypeError("contributions must be tuple")
        for i, c in enumerate(contributions):
            if not isinstance(c, CouplingContribution):
                raise TypeError(
                    f"contributions[{i}] must be CouplingContribution"
                )
        count_by_kind: dict = {}
        count_by_target: dict = {}
        declared_mass_delta_sum = 0.0
        n_conservative = 0
        n_non_conservative = 0
        for c in contributions:
            count_by_kind[c.kind] = count_by_kind.get(c.kind, 0) + 1
            t = str(c.target_id)
            count_by_target[t] = count_by_target.get(t, 0) + 1
            declared_mass_delta_sum += float(c.mass_delta)
            if c.conservative:
                n_conservative += 1
            else:
                n_non_conservative += 1
        return {
            "name": "contribution_profile",
            "n_contributions": int(len(contributions)),
            "count_by_kind": count_by_kind,
            "count_by_target": count_by_target,
            "declared_mass_delta_sum": float(declared_mass_delta_sum),
            "n_conservative": int(n_conservative),
            "n_non_conservative": int(n_non_conservative),
            "status": STATUS_PASS,
        }

    # =====================================================================
    # aggregated_coupling_profile
    # =====================================================================

    def aggregated_coupling_profile(
        self,
        aggregated: dict,
    ) -> dict:
        if not isinstance(aggregated, dict):
            raise TypeError("aggregated must be dict[ModuleId, AggregatedCoupling]")
        for k, v in aggregated.items():
            if not isinstance(v, AggregatedCoupling):
                raise TypeError(
                    f"aggregated[{k!r}] must be AggregatedCoupling"
                )
        mass_delta_by_target = {}
        policy_by_target = {}
        n_contributions_by_target = {}
        total_mass_delta = 0.0
        for tid, agg in aggregated.items():
            key = str(tid)
            mass_delta_by_target[key] = float(agg.mass_delta)
            policy_by_target[key] = str(agg.mass_policy)
            n_contributions_by_target[key] = int(agg.n_contributions)
            total_mass_delta += float(agg.mass_delta)
        return {
            "name": "aggregated_coupling_profile",
            "targets": tuple(str(k) for k in aggregated.keys()),
            "mass_delta_by_target": mass_delta_by_target,
            "policy_by_target": policy_by_target,
            "n_contributions_by_target": n_contributions_by_target,
            "total_mass_delta": float(total_mass_delta),
            "status": STATUS_PASS,
        }

    # =====================================================================
    # V5-2 interface declarations (deferred)
    # =====================================================================

    def R_ij_matrix(self, *args, **kwargs):
        raise NotImplementedError(
            "R_ij_matrix diagnostics deferred to V5-2"
        )

    def novelty_transmitted(self, *args, **kwargs):
        raise NotImplementedError(
            "novelty_transmitted diagnostics deferred to V5-2"
        )

    def mass_delta_by_coupling_kind(self, *args, **kwargs):
        raise NotImplementedError(
            "mass_delta_by_coupling_kind diagnostics deferred to V5-2"
        )

    def circulation(self, *args, **kwargs):
        raise NotImplementedError(
            "circulation diagnostics deferred to V5-2"
        )

    def __repr__(self) -> str:
        return "Diagnostics()"
