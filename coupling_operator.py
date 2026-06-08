# -*- coding: utf-8 -*-
"""
mcq_v5.coupling_operator — Local pairwise operator on CouplingContext.

Absolute principle:

    CouplingOperator receives ONLY CouplingContext.
    CouplingOperator NEVER receives Instance.
    CouplingOperator NEVER receives Module.
    CouplingOperator NEVER receives CouplingScheduler.

For V5-0a, this module is intentionally isolated from mcq_v5.module.
The type alias ModuleId is redefined locally to avoid indirect coupling.

Provides:
- CouplingOperator (abstract)
- CouplingContribution (immutable output package)
- NullCouplingOperator (stub returning zero contribution, useful for
  6d-compat baseline runs and invariant tests)

Critère étape 10:
    CouplingOperator is an abstract pairwise operator that receives only
    CouplingContext; NullCouplingOperator returns an immutable zero
    CouplingContribution for any valid context; R=0 and R=1 produce zero
    coupling; the operator cannot see Instance, Module, CouplingScheduler,
    global state, or history ownership.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping
import numpy as np

# Local redefinition of ModuleId to enforce isolation from mcq_v5.module.
# This is a deliberate decoupling for V5-0a. If a shared types module is
# extracted later, this alias can be replaced by an import from there
# (but never from mcq_v5.module).
ModuleId = str | int

from .coupling_context import CouplingContext


# Allowed kinds for a coupling contribution
ContributionKind = Literal["source", "flux", "mixed", "none"]
_VALID_KINDS = ("source", "flux", "mixed", "none")


@dataclass(frozen=True, slots=True)
class CouplingContribution:
    """Immutable output package from a CouplingOperator.

    Fields
    ------
    target_id, source_id : ModuleId
        Copied from the CouplingContext that produced this contribution.
    kind : Literal["source", "flux", "mixed", "none"]
        Routing tag for the future CouplingAggregator.
    value : np.ndarray | tuple[np.ndarray, ...]
        Data payload.
        - kind="source": single ndarray, shape grid.shape, float64, C-contig,
          rendered writeable=False.
        - kind="flux": tuple of ndarray, one per axis (faces convention).
        - kind="mixed": tuple of (source_array, flux_tuple) — V5-0a accepts
          minimally; aggregator routing in step 11.
        - kind="none": value must be an empty tuple ().
    conservative : bool
        Whether the contribution claims to conserve mass.
    mass_delta : float
        Net mass injected (positive) or removed (negative). For kind="flux",
        must be exactly 0.0 (a pure flux generates no spontaneous mass).
    metadata : Mapping[str, Any]
        Read-only metadata bundle.

    Internal validation (audit Amendment B):
    - kind="source": value is single ndarray with strict invariants
      (float64, C-contiguous, writeable=False).
    - kind="flux": mass_delta must be 0.0.
    - if kind="source" and mass_delta == 0.0, conservative=True is allowed.
    """

    target_id: ModuleId
    source_id: ModuleId
    kind: ContributionKind
    value: np.ndarray | tuple
    conservative: bool
    mass_delta: float
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

        # --- kind ---
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {_VALID_KINDS}, got {self.kind!r}"
            )

        # --- mass_delta type ---
        if not isinstance(self.mass_delta, (int, float, np.floating)):
            raise TypeError(
                f"mass_delta must be numeric, got {type(self.mass_delta).__name__}"
            )
        if not np.isfinite(float(self.mass_delta)):
            raise ValueError("mass_delta must be finite")
        object.__setattr__(self, "mass_delta", float(self.mass_delta))

        # --- conservative type ---
        if not isinstance(self.conservative, bool):
            raise TypeError(
                f"conservative must be bool, got {type(self.conservative).__name__}"
            )

        # --- kind / value / mass_delta cross-validation ---
        if self.kind == "source":
            # value must be a single ndarray, strict invariants
            if not isinstance(self.value, np.ndarray):
                raise TypeError(
                    f"kind='source' requires value: ndarray, "
                    f"got {type(self.value).__name__}"
                )
            if self.value.dtype != np.float64:
                raise TypeError(
                    f"source value dtype must be float64, got {self.value.dtype}"
                )
            if not self.value.flags["C_CONTIGUOUS"]:
                raise ValueError("source value must be C-contiguous")
            # Lock writeable=False (defensive: copy if needed, then lock)
            v = self.value.copy()
            v.flags.writeable = False
            object.__setattr__(self, "value", v)

        elif self.kind == "flux":
            # value must be a tuple of ndarray (one per axis)
            if not isinstance(self.value, tuple):
                raise TypeError(
                    f"kind='flux' requires value: tuple of ndarray, "
                    f"got {type(self.value).__name__}"
                )
            for ax, J in enumerate(self.value):
                if not isinstance(J, np.ndarray):
                    raise TypeError(
                        f"flux value axis {ax} must be ndarray, got {type(J).__name__}"
                    )
                if J.dtype != np.float64:
                    raise TypeError(
                        f"flux value axis {ax} dtype must be float64, got {J.dtype}"
                    )
                if not J.flags["C_CONTIGUOUS"]:
                    raise ValueError(
                        f"flux value axis {ax} must be C-contiguous"
                    )
            # Lock writeable=False on each
            locked = []
            for J in self.value:
                jc = J.copy()
                jc.flags.writeable = False
                locked.append(jc)
            object.__setattr__(self, "value", tuple(locked))
            # Pure flux: mass_delta must be 0.0
            if self.mass_delta != 0.0:
                raise ValueError(
                    f"kind='flux' requires mass_delta == 0.0, got {self.mass_delta}"
                )

        elif self.kind == "mixed":
            # V5-0a accepts mixed minimally: tuple (source_array, flux_tuple)
            if not isinstance(self.value, tuple) or len(self.value) != 2:
                raise TypeError(
                    "kind='mixed' requires value: tuple (source_ndarray, "
                    "flux_tuple)"
                )
            src, flx = self.value
            if not isinstance(src, np.ndarray):
                raise TypeError("mixed value[0] (source) must be ndarray")
            if src.dtype != np.float64:
                raise TypeError("mixed source dtype must be float64")
            if not src.flags["C_CONTIGUOUS"]:
                raise ValueError("mixed source must be C-contiguous")
            if not isinstance(flx, tuple):
                raise TypeError("mixed value[1] (flux) must be tuple")
            for ax, J in enumerate(flx):
                if not isinstance(J, np.ndarray):
                    raise TypeError(f"mixed flux axis {ax} must be ndarray")
                if J.dtype != np.float64:
                    raise TypeError(f"mixed flux axis {ax} dtype must be float64")
                if not J.flags["C_CONTIGUOUS"]:
                    raise ValueError(f"mixed flux axis {ax} must be C-contiguous")
            # Lock writeable on all components
            src_c = src.copy()
            src_c.flags.writeable = False
            locked_flx = []
            for J in flx:
                jc = J.copy()
                jc.flags.writeable = False
                locked_flx.append(jc)
            object.__setattr__(self, "value", (src_c, tuple(locked_flx)))

        elif self.kind == "none":
            # value must be an empty tuple (exactly: not a ndarray, not None)
            if not (isinstance(self.value, tuple) and len(self.value) == 0):
                raise ValueError(
                    f"kind='none' requires value=(), got {type(self.value).__name__}"
                )
            if self.mass_delta != 0.0:
                raise ValueError(
                    f"kind='none' requires mass_delta=0.0, got {self.mass_delta}"
                )

        # --- metadata defensive copy + read-only proxy ---
        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"metadata must be Mapping, got {type(self.metadata).__name__}"
            )
        proxy = MappingProxyType(dict(self.metadata))
        object.__setattr__(self, "metadata", proxy)


# =========================================================================
# Abstract CouplingOperator
# =========================================================================

class CouplingOperator(ABC):
    """Abstract pairwise coupling operator.

    Receives ONLY CouplingContext via pairwise_coupling(ctx).
    Concrete subclasses must implement pairwise_coupling.

    __slots__ = () to enforce ad-hoc attribute prohibition in subclasses.
    """

    __slots__ = ()

    @abstractmethod
    def pairwise_coupling(self, ctx: CouplingContext) -> CouplingContribution:
        """Compute the coupling contribution for a single pair.

        Must NOT access Instance, Module, CouplingScheduler, or any global
        state. Must only read fields of ctx.
        """
        ...


# =========================================================================
# NullCouplingOperator (stub)
# =========================================================================

class NullCouplingOperator(CouplingOperator):
    """Stub operator returning a zero-source CouplingContribution.

    Useful for:
    - 6d-compat baseline runs (no modular coupling)
    - testing scheduler/aggregator wiring without physics
    - testing R=0 / R=1 invariants on the wiring

    NullCouplingOperator does NOT read ctx.target_state_view or
    ctx.source_state_view. Its output depends only on ctx ids, overlap_mode,
    and grid shape.
    """

    __slots__ = ()

    def pairwise_coupling(self, ctx: CouplingContext) -> CouplingContribution:
        if not isinstance(ctx, CouplingContext):
            raise TypeError(
                f"pairwise_coupling expects CouplingContext, got "
                f"{type(ctx).__name__}"
            )
        # Zero source on the grid; locked writeable=False inside __post_init__
        zero = np.zeros(ctx.grid_view.shape, dtype=np.float64)
        return CouplingContribution(
            target_id=ctx.target_id,
            source_id=ctx.source_id,
            kind="source",
            value=zero,
            conservative=True,
            mass_delta=0.0,
            metadata={
                "operator": "NullCouplingOperator",
                "reason": "null_stub",
                "overlap_mode": ctx.overlap_mode,
            },
        )
