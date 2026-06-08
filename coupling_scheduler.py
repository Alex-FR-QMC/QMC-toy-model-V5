# -*- coding: utf-8 -*-
"""
mcq_v5.coupling_scheduler — Anti-centralization architectural pivot.

CouplingScheduler is the ONLY object permitted to read a full Instance
in order to produce pairwise CouplingContext bundles. It does NOT
compute the coupling itself — that is the responsibility of
CouplingOperator (step 10), which will receive only CouplingContext
and never Instance.

Responsibilities:
- enumerate ordered target<-source pairs from Instance.module_ids()
- query an overlap_provider for each pair
- apply epsilon admissibility policy (R_min, R_max)
- build CouplingContext objects via construction (no coupling logic)
- attach policy metadata to each context

Non-responsibilities:
- compute coupling
- compute novelty / self-form
- compute flux or source
- aggregate contributions
- step / evolve / run dynamics
- maintain a history list
- enforce inter-module shape uniformity (Instance tolerates heterogeneity)
- validate ndarray shape/dtype of provider output (delegated to
  CouplingContext.__post_init__)

Design rules:
- Stateless or near-stateless: policy and providers stored; no cache of
  contexts or last_instance.
- Pair determinism via sorted(instance.module_ids()).
- history_provider signature is (target_id, source_id) only, never Instance.

Critère étape 9:
    CouplingScheduler is the only object that reads an Instance to create
    pairwise CouplingContext objects; it constructs ordered target-source
    pairs, applies epsilon admissibility before CouplingOperator, exports
    overlap policy metadata, owns no history, computes no coupling, and
    never leaks Instance/Module objects into CouplingContext.
"""

from __future__ import annotations
from typing import Any, Callable
import math
import numpy as np

from .grid import Grid
from .module import ModuleId
from .instance import Instance
from .views import GridView, HistoryView
from .coupling_context import CouplingContext, OverlapMode, _VALID_OVERLAP_MODES


# Provider signatures (typed via aliases for clarity)
OverlapProvider = Callable[[ModuleId, ModuleId, Instance], "float | np.ndarray"]
HistoryProvider = Callable[[ModuleId, ModuleId], "HistoryView | None"]


def _default_overlap_provider(
    target_id: ModuleId,
    source_id: ModuleId,
    instance: Instance,
) -> float:
    """Dry-test placeholder: constant overlap R = 0.5 for any pair.

    This is NOT a measured recouvrement. It exists only so that V5-0a
    interface tests can run without depending on a real overlap measurement
    pipeline (which is deferred to V5-2).
    """
    return 0.5


class CouplingScheduler:
    """Anti-centralization pivot. Reads Instance, produces CouplingContexts.

    Construction parameters:
        overlap_mode: 'global_scalar' or 'local_field'.
            'patchwise' raises NotImplementedError at scheduling time
            (still accepted by CouplingContext as a container).
        epsilon_R_min: float in [0, 1). Pair omitted if R < epsilon_R_min.
        epsilon_R_max: float in [0, 1). Pair omitted if 1 - R < epsilon_R_max.
            Constraint: epsilon_R_min + epsilon_R_max < 1.0 to leave a
            non-empty admissibility window.
        overlap_provider: callable(target_id, source_id, instance) -> R.
            R must be float for global_scalar, np.ndarray for local_field.
            Defaults to _default_overlap_provider (constant 0.5, dry).
        history_provider: callable(target_id, source_id) -> HistoryView|None.
            Note: NOT given Instance, to enforce pairwise containment of
            history scope.
        metadata: optional dict, defensively copied; merged into each
            context's metadata under the key prefix 'scheduler.*'.
    """

    __slots__ = (
        "_overlap_mode",
        "_epsilon_R_min",
        "_epsilon_R_max",
        "_overlap_provider",
        "_history_provider",
        "_metadata",
    )

    def __init__(
        self,
        overlap_mode: OverlapMode = "global_scalar",
        epsilon_R_min: float = 0.0,
        epsilon_R_max: float = 0.0,
        overlap_provider: OverlapProvider | None = None,
        history_provider: HistoryProvider | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        # --- overlap_mode ---
        if overlap_mode not in _VALID_OVERLAP_MODES:
            raise ValueError(
                f"overlap_mode must be one of {_VALID_OVERLAP_MODES}, "
                f"got {overlap_mode!r}"
            )

        # --- epsilon types and ranges ---
        for name, eps in (("epsilon_R_min", epsilon_R_min),
                          ("epsilon_R_max", epsilon_R_max)):
            if not isinstance(eps, (int, float, np.floating)):
                raise TypeError(
                    f"{name} must be numeric, got {type(eps).__name__}"
                )
            if not math.isfinite(float(eps)):
                raise ValueError(f"{name} must be finite, got {eps}")
            if not (0.0 <= float(eps) < 1.0):
                raise ValueError(
                    f"{name} must be in [0, 1), got {eps}"
                )

        eps_sum = float(epsilon_R_min) + float(epsilon_R_max)
        if eps_sum >= 1.0:
            raise ValueError(
                f"Inconsistent epsilons: epsilon_R_min + epsilon_R_max = "
                f"{eps_sum} must be < 1.0 to allow a valid admissibility "
                f"window"
            )

        # --- providers ---
        if overlap_provider is None:
            overlap_provider = _default_overlap_provider
        if not callable(overlap_provider):
            raise TypeError("overlap_provider must be callable or None")
        if history_provider is not None and not callable(history_provider):
            raise TypeError("history_provider must be callable or None")

        # --- metadata ---
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be dict or None, got {type(metadata).__name__}"
            )

        # --- store ---
        self._overlap_mode = overlap_mode
        self._epsilon_R_min = float(epsilon_R_min)
        self._epsilon_R_max = float(epsilon_R_max)
        self._overlap_provider = overlap_provider
        self._history_provider = history_provider
        self._metadata = dict(metadata) if metadata else {}

    # === Read accessors ===

    @property
    def overlap_mode(self) -> OverlapMode:
        return self._overlap_mode

    @property
    def epsilon_R_min(self) -> float:
        return self._epsilon_R_min

    @property
    def epsilon_R_max(self) -> float:
        return self._epsilon_R_max

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    # === Pair enumeration ===

    def _ordered_pairs(
        self, instance: Instance
    ) -> tuple[tuple[ModuleId, ModuleId], ...]:
        """Deterministic ordered target<-source pairs from sorted module ids."""
        # Sort ids: mixing str and int could fail. We coerce to str for
        # sorting only (does not change the actual ids passed downstream).
        ids = list(instance.module_ids())
        try:
            ids_sorted = sorted(ids)
        except TypeError:
            # Mixed types (e.g. str and int): sort by str representation
            ids_sorted = sorted(ids, key=lambda x: (type(x).__name__, str(x)))
        pairs = []
        for target_id in ids_sorted:
            for source_id in ids_sorted:
                if target_id == source_id:
                    continue
                pairs.append((target_id, source_id))
        return tuple(pairs)

    # === Admissibility ===

    def _admissibility_scalar(self, R: float) -> bool:
        """Pair admissible iff R >= eps_min AND 1 - R >= eps_max."""
        if R < self._epsilon_R_min:
            return False
        if (1.0 - R) < self._epsilon_R_max:
            return False
        return True

    def _reduce_R(self, R) -> float:
        """Reduce R to a scalar for admissibility decision.

        - global_scalar: return float(R)
        - local_field: return mean(R), documented policy
        - patchwise: not supported by scheduler (raises elsewhere)
        """
        if self._overlap_mode == "global_scalar":
            return float(R)
        if self._overlap_mode == "local_field":
            if not isinstance(R, np.ndarray):
                raise TypeError(
                    f"local_field overlap_provider must return ndarray, "
                    f"got {type(R).__name__}"
                )
            return float(np.mean(R))
        # patchwise filtered upstream
        raise NotImplementedError(
            f"_reduce_R: unhandled overlap_mode {self._overlap_mode!r}"
        )

    def admissible_pairs(
        self, instance: Instance
    ) -> tuple[tuple[ModuleId, ModuleId], ...]:
        """Return ordered admissible (target, source) pairs.

        Calls overlap_provider for each candidate pair, reduces R if needed
        (for local_field, uses mean), and filters by epsilon policy.
        """
        if not isinstance(instance, Instance):
            raise TypeError(
                f"instance must be Instance, got {type(instance).__name__}"
            )
        if self._overlap_mode == "patchwise":
            raise NotImplementedError(
                "patchwise scheduling not implemented in V5-0a"
            )

        admissible = []
        for (target_id, source_id) in self._ordered_pairs(instance):
            R = self._overlap_provider(target_id, source_id, instance)
            R_eff = self._reduce_R(R)
            if self._admissibility_scalar(R_eff):
                admissible.append((target_id, source_id))
        return tuple(admissible)

    # === Context building ===

    def build_contexts(
        self,
        instance: Instance,
        grid_view: GridView,
        time: float,
    ) -> tuple[CouplingContext, ...]:
        """Build CouplingContext for each admissible pair.

        Parameters
        ----------
        instance : Instance
            Multi-module collection. Read here, never leaked into contexts.
        grid_view : GridView
            Spatial grid view passed to each context.
        time : float
            Current simulation time, propagated to each context.

        Returns
        -------
        tuple[CouplingContext, ...]
            One context per admissible ordered pair, in deterministic order.
        """
        if not isinstance(instance, Instance):
            raise TypeError(
                f"instance must be Instance, got {type(instance).__name__}"
            )
        if not isinstance(grid_view, GridView):
            raise TypeError(
                f"grid_view must be GridView, got {type(grid_view).__name__}"
            )
        if not isinstance(time, (int, float, np.floating)):
            raise TypeError(
                f"time must be numeric, got {type(time).__name__}"
            )
        if not math.isfinite(float(time)):
            raise ValueError(f"time must be finite, got {time}")

        if self._overlap_mode == "patchwise":
            raise NotImplementedError(
                "patchwise scheduling not implemented in V5-0a"
            )

        contexts = []
        for (target_id, source_id) in self._ordered_pairs(instance):
            R = self._overlap_provider(target_id, source_id, instance)
            R_eff = self._reduce_R(R)
            if not self._admissibility_scalar(R_eff):
                continue

            target = instance.get_module(target_id)
            source = instance.get_module(source_id)

            # history: provider takes (target_id, source_id) only, no Instance
            allowed_history = None
            if self._history_provider is not None:
                allowed_history = self._history_provider(target_id, source_id)

            # Per-context metadata: policy + reduction info
            ctx_metadata = dict(self._metadata)
            ctx_metadata["created_by"] = "CouplingScheduler.build_contexts"
            ctx_metadata["overlap_mode"] = self._overlap_mode
            ctx_metadata["epsilon_R_min"] = self._epsilon_R_min
            ctx_metadata["epsilon_R_max"] = self._epsilon_R_max
            ctx_metadata["pair_policy"] = "ordered_pairs_i_ne_j"
            if self._overlap_mode == "local_field":
                ctx_metadata["overlap_reduction"] = (
                    "mean_for_pair_admissibility"
                )
            ctx_metadata["R_eff_for_admissibility"] = R_eff

            ctx = CouplingContext(
                target_id=target_id,
                source_id=source_id,
                target_state_view=target.state_view(),
                source_state_view=source.state_view(),
                target_metric_view=target.metric_view(),
                source_metric_view=source.metric_view(),
                grid_view=grid_view,
                overlap_R=R,
                overlap_mode=self._overlap_mode,
                time=float(time),
                allowed_history=allowed_history,
                metadata=ctx_metadata,
            )
            contexts.append(ctx)

        return tuple(contexts)

    def __repr__(self) -> str:
        return (
            f"CouplingScheduler(overlap_mode={self._overlap_mode!r}, "
            f"epsilon_R_min={self._epsilon_R_min}, "
            f"epsilon_R_max={self._epsilon_R_max})"
        )
