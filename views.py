# -*- coding: utf-8 -*-
"""
mcq_v5.views — Bounded zero-copy access wrappers.

Design rules V5-0a:

1. Views are NOT defensive copies. They are bounded references.
2. No silent .copy(), no silent np.array(...), no np.ascontiguousarray.
3. local_patch returns the raw numpy slice (possibly non-contiguous).
   Convention: it is the caller's responsibility to handle contiguity if
   needed for downstream computation.
4. HistoryView accesses only macro-steps with committed=True.
5. HistoryView returns arrays with writeable=False (numpy flag) to prevent
   accidental history corruption, while staying zero-copy.
6. HistoryView does NOT own the history list. It receives a reference to
   an external list maintained by Instance / ExperimentProtocol.
7. No hidden global access. No cache. No phenomenology.

Critère de validation:
    Views are bounded, zero-copy where applicable, macro-step-safe for history,
    non-computational, and do not reintroduce hidden global access.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from .state import State
from .grid import Grid


# =========================================================================
# StateView
# =========================================================================

class StateView:
    """Bounded zero-copy view over a State.

    field(name) returns the actual array reference inside the State.
    Modifying the returned array mutates the underlying State.
    This is intentional: views are not defensive.

    local_patch returns a raw numpy slice. It may be non-contiguous.
    Caller is responsible for handling contiguity if doing downstream math.
    """

    __slots__ = ("_state",)

    def __init__(self, state: State):
        if not isinstance(state, State):
            raise TypeError(f"StateView requires State, got {type(state).__name__}")
        self._state = state

    def field(self, name: str) -> np.ndarray:
        """Return the actual array of `name` in the underlying State.

        Modifying the returned array mutates the State. This is by design:
        views are not defensive. Callers that must not mutate should treat
        the result as read-only by discipline.
        """
        if name not in self._state.fields:
            raise KeyError(f"field {name!r} not in State")
        return self._state.fields[name]

    def has_field(self, name: str) -> bool:
        return name in self._state.fields

    def solver_field_names(self) -> tuple[str, ...]:
        return self._state.solver_fields

    def all_field_names(self) -> tuple[str, ...]:
        return tuple(self._state.fields.keys())

    def local_patch(
        self,
        name: str,
        center: tuple[int, ...],
        radius: int,
    ) -> np.ndarray:
        """Extract a local patch around `center` with given `radius`.

        - Patch is truncated at grid boundaries (no padding).
        - Returns a raw numpy slice. May be non-contiguous.
        - Center indices out of grid bounds raise ValueError.
        - radius < 0 raises ValueError.

        Convention V5-0a: caller is responsible for ascontiguousarray if
        downstream operators require contiguity. StateView does not enforce
        it because that would mean a hidden copy.
        """
        if radius < 0:
            raise ValueError(f"radius must be >= 0, got {radius}")
        arr = self.field(name)
        if not isinstance(center, tuple):
            raise TypeError(f"center must be tuple, got {type(center).__name__}")
        # For 3D scalar fields, len(center) must match arr.ndim's spatial part.
        # We assume center indexes the first len(center) dims (spatial dims).
        if len(center) > arr.ndim:
            raise ValueError(
                f"center has {len(center)} components, array has {arr.ndim} dims"
            )
        # Validate center in-bounds
        for axis, c in enumerate(center):
            if not isinstance(c, (int, np.integer)):
                raise TypeError(
                    f"center[{axis}] must be int, got {type(c).__name__}"
                )
            if not 0 <= c < arr.shape[axis]:
                raise ValueError(
                    f"center[{axis}]={c} out of grid bounds [0, {arr.shape[axis]})"
                )
        # Build slices
        slices = []
        for axis, c in enumerate(center):
            lo = max(0, c - radius)
            hi = min(arr.shape[axis], c + radius + 1)
            slices.append(slice(lo, hi))
        # Remaining dims (e.g., tensor components) are taken fully
        for axis in range(len(center), arr.ndim):
            slices.append(slice(None))
        return arr[tuple(slices)]


# =========================================================================
# MetricView (stub — full implementation when Metric lands)
# =========================================================================

class MetricView:
    """Bounded view over a Metric. V5-0a stub.

    Real implementation arrives with metric.py. For V5-0a, MetricView wraps
    an arbitrary object exposing a `.coefficients` numpy array attribute or
    nothing. Methods that depend on full Metric semantics raise
    NotImplementedError.
    """

    __slots__ = ("_metric",)

    def __init__(self, metric: Any):
        # No isinstance check yet because Metric class is not implemented.
        # We accept any object; downstream methods will fail at use-time if
        # the metric lacks expected attributes.
        self._metric = metric

    def coefficients(self) -> np.ndarray:
        """Return the underlying metric coefficients array (zero-copy)."""
        if not hasattr(self._metric, "coefficients"):
            raise NotImplementedError(
                "MetricView.coefficients requires metric.coefficients attribute "
                "(implemented in V5-0a metric.py step)"
            )
        return self._metric.coefficients

    def face_coefficients(self, axis: int) -> np.ndarray:
        raise NotImplementedError(
            "MetricView.face_coefficients implemented when GeometryOperator lands"
        )

    def determinant(self) -> np.ndarray:
        raise NotImplementedError("MetricView.determinant not implemented in V5-0a stub")

    def inverse(self) -> np.ndarray:
        raise NotImplementedError("MetricView.inverse not implemented in V5-0a stub")


# =========================================================================
# GridView
# =========================================================================

class GridView:
    """Read-only view over a Grid.

    Grid is already frozen (immutable dataclass), so this is conceptually
    just a delegation layer for interface symmetry with other views.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid: Grid):
        if not isinstance(grid, Grid):
            raise TypeError(f"GridView requires Grid, got {type(grid).__name__}")
        self._grid = grid

    @property
    def shape(self) -> tuple[int, ...]:
        return self._grid.shape

    @property
    def dx(self) -> tuple[float, ...]:
        return self._grid.dx

    @property
    def dim(self) -> int:
        return self._grid.dim

    @property
    def n_cells(self) -> int:
        return self._grid.n_cells

    def cell_centers(self) -> tuple[np.ndarray, ...]:
        return self._grid.cell_centers()

    def faces(self, axis: int) -> np.ndarray:
        return self._grid.faces(axis)

    def boundary_policy(self) -> str:
        return self._grid.boundary_policy()


# =========================================================================
# HistoryEntry & HistoryView
# =========================================================================

@dataclass(slots=True)
class HistoryEntry:
    """One entry in the history list.

    Owned by Instance / ExperimentProtocol, not by HistoryView.
    """
    state: State
    time: float
    step: int
    committed: bool
    metadata: dict = field(default_factory=dict)


class HistoryView:
    """Read-only access to a list of HistoryEntry.

    Filters strictly to committed=True entries.

    Returns numpy arrays with writeable=False to prevent accidental
    mutation of historical states, while staying zero-copy.

    Does NOT own the history. The list is maintained externally by
    Instance / ExperimentProtocol. HistoryView holds a reference.
    """

    __slots__ = ("_history",)

    def __init__(self, history: Sequence[HistoryEntry]):
        # history is expected to be a list (or similar sequence) owned externally.
        # We do not copy it. We do not validate elements beyond minimum checks
        # at access time, to avoid coupling HistoryView to a specific container.
        if history is None:
            raise TypeError("history reference cannot be None")
        self._history = history

    def _committed_entries(self) -> list[HistoryEntry]:
        """All entries with committed=True, in original order."""
        return [e for e in self._history if e.committed]

    def n_committed(self) -> int:
        return sum(1 for e in self._history if e.committed)

    def previous(self, field_name: str, n: int = 1) -> np.ndarray:
        """Return the field array from the n-th most recent committed entry.

        n=1 means the latest committed entry, n=2 the one before, etc.
        Returns a numpy array with writeable=False (zero-copy locked).
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        committed = self._committed_entries()
        if n > len(committed):
            raise IndexError(
                f"previous({n}): only {len(committed)} committed entries"
            )
        entry = committed[-n]
        arr = entry.state.fields[field_name]
        return self._readonly_view(arr)

    def window(self, field_name: str, length: int) -> list[np.ndarray]:
        """Return the last `length` committed arrays for field_name.

        Order: oldest first within the window. All arrays writeable=False.
        If fewer than `length` committed entries exist, the returned list
        is shorter than `length` (documented behavior, no error).
        """
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        committed = self._committed_entries()
        slice_ = committed[-length:] if length <= len(committed) else committed
        return [self._readonly_view(e.state.fields[field_name]) for e in slice_]

    def states(self, length: int) -> list[State]:
        """Return the last `length` committed State objects.

        States themselves are not mutated; their fields are returned as
        regular references when accessed via state.fields[...]. Callers
        that want guaranteed read-only access should use previous/window.
        """
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        committed = self._committed_entries()
        slice_ = committed[-length:] if length <= len(committed) else committed
        return [e.state for e in slice_]

    @staticmethod
    def _readonly_view(arr: np.ndarray) -> np.ndarray:
        """Return a zero-copy view of arr with writeable=False.

        Any attempt to modify the returned array raises ValueError from numpy.
        """
        v = arr.view()
        v.flags.writeable = False
        return v
