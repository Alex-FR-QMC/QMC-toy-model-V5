# -*- coding: utf-8 -*-
"""
mcq_v5.coupling_context — Bounded pairwise context for modular coupling.

A CouplingContext is a frozen, immutable bundle passed to a CouplingOperator.
It contains:

- target_id, source_id (with target_id != source_id enforced)
- bounded views (StateView, MetricView, GridView)
- explicit overlap_R + overlap_mode metadata
- current time
- optional allowed_history (HistoryView)
- read-only metadata

It DOES NOT contain:

- Instance, Module, ExperimentProtocol, CouplingScheduler, Solver,
  Diagnostics
- compute_* / aggregate / apply methods
- raw State / Metric / Grid objects

It computes NOTHING. It is a passive package.

Production:
  CouplingContext is intended to be produced by CouplingScheduler (step 9).
  Direct construction is allowed in V5-0a tests only.

allowed_history cloisonment:
  In V5-0, the history reference passed via allowed_history may technically
  be the global history list, but the CouplingOperator is permitted to
  query (via previous/window) ONLY fields belonging to target_id or
  source_id. Accessing the history of any third module k from this context
  is a violation of pairwise local containment. V5-0a enforces this by
  documentation only; future steps may add active filtering.

Critère étape 8:
    CouplingContext is an immutable bounded pairwise context,
    containing only target/source ids, bounded views, explicit overlap data,
    time, optional allowed history, and local metadata;
    it computes no coupling, sees no Instance/Module/Experiment,
    and prepares global_scalar/local_field overlap modes without
    interpreting them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping
import math
import numpy as np

from .module import ModuleId
from .views import StateView, MetricView, GridView, HistoryView


OverlapMode = Literal["global_scalar", "local_field", "patchwise"]
_VALID_OVERLAP_MODES = ("global_scalar", "local_field", "patchwise")


@dataclass(frozen=True, slots=True)
class CouplingContext:
    """Immutable bounded pairwise context for one target <- source coupling.

    Fields
    ------
    target_id, source_id : ModuleId
        Identifiers of the target (recipient) and source (emitter) modules.
        target_id == source_id is rejected.
    target_state_view, source_state_view : StateView
        Bounded zero-copy views over the respective module states.
    target_metric_view, source_metric_view : MetricView
        Bounded views over the respective module metrics.
    grid_view : GridView
        Read-only view over the spatial grid.
    overlap_R : float | np.ndarray
        Pairwise overlap. Scalar in [0,1] if overlap_mode='global_scalar';
        ndarray of shape grid.shape with values in [0,1] if 'local_field';
        ndarray for 'patchwise' (semantics deferred, container only).
    overlap_mode : Literal['global_scalar', 'local_field', 'patchwise']
        Mode tag, must be exported in all V5 diagnostics.
    time : float
        Current simulation time.
    allowed_history : HistoryView | None
        Optional history reference. Cloisonnement: even if this reference
        is the global history, the CouplingOperator may only query fields
        of target_id or source_id from it. V5-0a enforces this by
        documentation; later versions may filter actively.
    metadata : Mapping[str, Any]
        Read-only metadata bundle (wrapped in MappingProxyType).
    """

    target_id: ModuleId
    source_id: ModuleId
    target_state_view: StateView
    source_state_view: StateView
    target_metric_view: MetricView
    source_metric_view: MetricView
    grid_view: GridView
    overlap_R: float | np.ndarray
    overlap_mode: OverlapMode
    time: float
    allowed_history: HistoryView | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # --- ids ---
        if not isinstance(self.target_id, (str, int)):
            raise TypeError(
                f"target_id must be str|int, got {type(self.target_id).__name__}"
            )
        if not isinstance(self.source_id, (str, int)):
            raise TypeError(
                f"source_id must be str|int, got {type(self.source_id).__name__}"
            )
        if self.target_id == self.source_id:
            raise ValueError(
                f"target_id and source_id must differ, got both = {self.target_id!r}"
            )

        # --- views ---
        if not isinstance(self.target_state_view, StateView):
            raise TypeError("target_state_view must be StateView")
        if not isinstance(self.source_state_view, StateView):
            raise TypeError("source_state_view must be StateView")
        if not isinstance(self.target_metric_view, MetricView):
            raise TypeError("target_metric_view must be MetricView")
        if not isinstance(self.source_metric_view, MetricView):
            raise TypeError("source_metric_view must be MetricView")
        if not isinstance(self.grid_view, GridView):
            raise TypeError("grid_view must be GridView")

        # --- time ---
        if not isinstance(self.time, (int, float, np.floating)):
            raise TypeError(
                f"time must be numeric, got {type(self.time).__name__}"
            )
        if not math.isfinite(float(self.time)):
            raise ValueError("time must be finite")

        # --- overlap_mode ---
        if self.overlap_mode not in _VALID_OVERLAP_MODES:
            raise ValueError(
                f"overlap_mode must be one of {_VALID_OVERLAP_MODES}, "
                f"got {self.overlap_mode!r}"
            )

        # --- overlap_R + overlap_mode coherence ---
        self._validate_overlap(self.overlap_R, self.overlap_mode, self.grid_view)

        # --- allowed_history type ---
        if self.allowed_history is not None and not isinstance(
            self.allowed_history, HistoryView
        ):
            raise TypeError(
                "allowed_history must be HistoryView or None"
            )

        # --- metadata: defensive copy + read-only proxy ---
        # frozen dataclass forbids assignment; use object.__setattr__
        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"metadata must be Mapping, got {type(self.metadata).__name__}"
            )
        proxy = MappingProxyType(dict(self.metadata))
        object.__setattr__(self, "metadata", proxy)

        # --- overlap_R immutability lock ---
        # Even though the dataclass is frozen, ndarray contents would remain
        # mutable. To guarantee that CouplingContext is a stable package
        # passed to the future CouplingOperator, lock overlap_R:
        # - global_scalar: normalize to native float (immutable by nature)
        # - local_field / patchwise: defensive copy + writeable=False
        if self.overlap_mode == "global_scalar":
            object.__setattr__(self, "overlap_R", float(self.overlap_R))
        else:
            r = self.overlap_R.copy()
            r.flags.writeable = False
            object.__setattr__(self, "overlap_R", r)

    # === Validation helpers ===

    @staticmethod
    def _validate_overlap(
        overlap_R: float | np.ndarray,
        overlap_mode: OverlapMode,
        grid_view: GridView,
    ) -> None:
        if overlap_mode == "global_scalar":
            if not isinstance(overlap_R, (int, float, np.floating)):
                raise TypeError(
                    f"overlap_R for 'global_scalar' must be scalar, "
                    f"got {type(overlap_R).__name__}"
                )
            r = float(overlap_R)
            if not math.isfinite(r):
                raise ValueError(f"overlap_R must be finite, got {r}")
            if not (0.0 <= r <= 1.0):
                raise ValueError(f"overlap_R must be in [0,1], got {r}")
        elif overlap_mode == "local_field":
            if not isinstance(overlap_R, np.ndarray):
                raise TypeError(
                    f"overlap_R for 'local_field' must be ndarray, "
                    f"got {type(overlap_R).__name__}"
                )
            if overlap_R.dtype != np.float64:
                raise TypeError(
                    f"overlap_R dtype must be float64, got {overlap_R.dtype}"
                )
            if not overlap_R.flags["C_CONTIGUOUS"]:
                raise ValueError("overlap_R must be C-contiguous")
            if tuple(overlap_R.shape) != tuple(grid_view.shape):
                raise ValueError(
                    f"overlap_R shape {overlap_R.shape} must match "
                    f"grid shape {grid_view.shape}"
                )
            if not np.all(np.isfinite(overlap_R)):
                raise ValueError("overlap_R must be finite everywhere")
            if not (np.all(overlap_R >= 0.0) and np.all(overlap_R <= 1.0)):
                raise ValueError("overlap_R values must be in [0,1]")
        elif overlap_mode == "patchwise":
            # Container only; semantics deferred to later versions.
            # Minimum: must be ndarray, float64, contiguous, finite, in [0,1].
            if not isinstance(overlap_R, np.ndarray):
                raise TypeError(
                    f"overlap_R for 'patchwise' must be ndarray (V5-0a "
                    f"container), got {type(overlap_R).__name__}"
                )
            if overlap_R.dtype != np.float64:
                raise TypeError(
                    f"overlap_R dtype must be float64, got {overlap_R.dtype}"
                )
            if not overlap_R.flags["C_CONTIGUOUS"]:
                raise ValueError("overlap_R must be C-contiguous")
            if not np.all(np.isfinite(overlap_R)):
                raise ValueError("overlap_R must be finite everywhere")
            if not (np.all(overlap_R >= 0.0) and np.all(overlap_R <= 1.0)):
                raise ValueError("overlap_R values must be in [0,1]")
        else:  # pragma: no cover (covered by overlap_mode check earlier)
            raise ValueError(f"unhandled overlap_mode {overlap_mode!r}")

    # === Inspective helpers (non-mutating, non-computational) ===

    def is_global_overlap(self) -> bool:
        return self.overlap_mode == "global_scalar"

    def is_local_overlap(self) -> bool:
        return self.overlap_mode == "local_field"

    def is_patchwise_overlap(self) -> bool:
        return self.overlap_mode == "patchwise"

    def overlap_shape(self) -> tuple[int, ...] | None:
        """Shape of overlap_R if ndarray, else None."""
        if isinstance(self.overlap_R, np.ndarray):
            return tuple(self.overlap_R.shape)
        return None
