# -*- coding: utf-8 -*-
"""
mcq_v5.state — State container with strict serialization layout.

Design rules verrouillées V5-0a (cf. V5_0a_interfaces_plan.md §3.3 + audit):

1. State.fields can hold any dtype (psi, h, masks, labels, etc.).
2. Only fields declared in solver_fields enter flatten_for_solver().
3. solver_fields must be np.float64 AND C-contiguous, asserted explicitly.
4. No silent dtype conversion. No silent np.ascontiguousarray.
5. flatten_for_solver() is canonical: all solver_fields, order = self.solver_fields.
6. flatten_subset(field_names) is partial, marks layout.is_partial = True.
7. unflatten_from_solver refuses partial layouts unless allow_partial=True + base_state.
8. Round-trip must be bijective for solver fields:
   shape-safe, dtype-aware, order-stable, field-isolated.
9. State is constructed by copy (defensive), exposed via views elsewhere.
10. metadata is never flattened, never enters the solver vector.

Critère validation:
    State flatten/unflatten is bijective for solver fields,
    shape-safe, dtype-aware, order-stable, field-isolated,
    partial-layout guarded, and static-field compatible.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True, slots=True)
class FieldSlice:
    """Describes a contiguous slice of the flat solver vector."""
    name: str
    start: int
    end: int
    shape: tuple[int, ...]
    dtype: np.dtype
    role: str = "solver"


@dataclass(frozen=True, slots=True)
class SolverLayout:
    """Layout describing how to flatten/unflatten a State for the solver."""
    field_slices: tuple[FieldSlice, ...]
    total_size: int
    solver_fields: tuple[str, ...]
    is_partial: bool = False

    def slice_for(self, name: str) -> FieldSlice:
        for fs in self.field_slices:
            if fs.name == name:
                return fs
        raise KeyError(f"field {name!r} not in layout")

    def __post_init__(self):
        # Names unique
        names = [fs.name for fs in self.field_slices]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names in layout: {names}")
        # Indices contiguous and non-overlapping, covering [0, total_size]
        if self.field_slices:
            sorted_slices = sorted(self.field_slices, key=lambda fs: fs.start)
            if sorted_slices[0].start != 0:
                raise ValueError(f"layout does not start at 0: {sorted_slices[0]}")
            cursor = 0
            for fs in sorted_slices:
                if fs.start != cursor:
                    raise ValueError(
                        f"layout has gap or overlap at {fs.name}: "
                        f"expected start={cursor}, got {fs.start}"
                    )
                if fs.end <= fs.start:
                    raise ValueError(f"invalid slice {fs.name}: end <= start")
                expected_size = int(np.prod(fs.shape))
                if fs.end - fs.start != expected_size:
                    raise ValueError(
                        f"slice {fs.name} size mismatch: "
                        f"shape {fs.shape} = {expected_size}, end-start = {fs.end - fs.start}"
                    )
                cursor = fs.end
            if cursor != self.total_size:
                raise ValueError(
                    f"layout total_size {self.total_size} != cumulative end {cursor}"
                )
        elif self.total_size != 0:
            raise ValueError(f"empty layout but total_size={self.total_size}")
        # solver_fields names must all be present in slices (if not partial)
        slice_names = set(fs.name for fs in self.field_slices)
        sf_set = set(self.solver_fields)
        if not self.is_partial:
            if slice_names != sf_set:
                raise ValueError(
                    f"non-partial layout: slice names {slice_names} != solver_fields {sf_set}"
                )
        else:
            if not slice_names.issubset(sf_set):
                raise ValueError(
                    f"partial layout: slice names {slice_names} not subset of "
                    f"solver_fields {sf_set}"
                )


class State:
    """Container of dynamical and static fields with strict serialization.

    State.fields can hold arrays of any dtype.
    Only fields declared in solver_fields enter flatten_for_solver().

    Construction is defensive (arrays are copied) to prevent silent aliasing
    with caller data.
    """

    __slots__ = ("_fields", "_solver_fields", "_metadata")

    def __init__(
        self,
        fields: dict[str, np.ndarray],
        solver_fields: tuple[str, ...],
        metadata: dict[str, Any] | None = None,
    ):
        # Validation
        if not isinstance(fields, dict):
            raise TypeError(f"fields must be a dict, got {type(fields).__name__}")
        for name, arr in fields.items():
            if not isinstance(name, str):
                raise TypeError(f"field name must be str, got {type(name).__name__}")
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"field {name!r}: must be np.ndarray, got {type(arr).__name__}"
                )
        if not isinstance(solver_fields, tuple):
            raise TypeError(
                f"solver_fields must be tuple, got {type(solver_fields).__name__}"
            )
        for sf in solver_fields:
            if sf not in fields:
                raise ValueError(
                    f"solver_fields entry {sf!r} not in fields keys {list(fields.keys())}"
                )
        # Solver field invariants: dtype + contiguity
        for sf in solver_fields:
            arr = fields[sf]
            if arr.dtype != np.float64:
                raise TypeError(
                    f"solver field {sf!r}: dtype must be float64, got {arr.dtype}"
                )
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    f"solver field {sf!r}: must be C-contiguous"
                )
        # Defensive copy of arrays
        self._fields = {name: arr.copy() for name, arr in fields.items()}
        self._solver_fields = tuple(solver_fields)
        self._metadata = dict(metadata) if metadata else {}

    @property
    def fields(self) -> dict[str, np.ndarray]:
        """Read access to fields dict. Modifying arrays here mutates State."""
        return self._fields

    @property
    def solver_fields(self) -> tuple[str, ...]:
        return self._solver_fields

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def get_field(self, name: str) -> np.ndarray:
        if name not in self._fields:
            raise KeyError(f"field {name!r} not in State")
        return self._fields[name]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.get_field(name)

    def with_field(self, name: str, value: np.ndarray) -> "State":
        """Return a new State with field `name` replaced by `value`.

        If name is a solver field, value must be float64 + C-contiguous.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(f"value must be np.ndarray, got {type(value).__name__}")
        new_fields = dict(self._fields)
        new_fields[name] = value
        # Reuse __init__ validation
        return State(
            fields=new_fields,
            solver_fields=self._solver_fields,
            metadata=self._metadata,
        )

    def copy(self) -> "State":
        return State(
            fields=self._fields,
            solver_fields=self._solver_fields,
            metadata=self._metadata,
        )

    def norm(self, fields: list[str] | None = None) -> float:
        """L2 norm over selected fields (default: all solver fields)."""
        if fields is None:
            fields = list(self._solver_fields)
        sq = 0.0
        for fn in fields:
            arr = self._fields[fn]
            sq += float(np.sum(arr * arr))
        return float(np.sqrt(sq))

    # === Serialization ===

    def flatten_for_solver(self) -> tuple[np.ndarray, SolverLayout]:
        """Canonical flatten of all solver_fields in declared order.

        Returns
        -------
        flat : np.ndarray (1D, float64)
            Concatenated arrays in C order, solver_fields order.
        layout : SolverLayout
            Bijective metadata.
        """
        return self._flatten(self._solver_fields, is_partial=False)

    def flatten_subset(
        self, field_names: tuple[str, ...]
    ) -> tuple[np.ndarray, SolverLayout]:
        """Partial flatten of a subset of solver_fields.

        The resulting layout is marked is_partial=True. Round-trip via
        unflatten_from_solver requires allow_partial=True AND base_state.
        """
        if not isinstance(field_names, tuple):
            raise TypeError(f"field_names must be tuple, got {type(field_names).__name__}")
        for fn in field_names:
            if fn not in self._solver_fields:
                raise ValueError(
                    f"flatten_subset: {fn!r} not in solver_fields {self._solver_fields}"
                )
        return self._flatten(field_names, is_partial=True)

    def _flatten(
        self, field_names: tuple[str, ...], is_partial: bool
    ) -> tuple[np.ndarray, SolverLayout]:
        slices = []
        cursor = 0
        chunks = []
        for fn in field_names:
            arr = self._fields[fn]
            # Re-assert dtype and contiguity at flatten time (defensive)
            if arr.dtype != np.float64:
                raise TypeError(
                    f"flatten: field {fn!r} dtype is {arr.dtype}, expected float64"
                )
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    f"flatten: field {fn!r} is not C-contiguous"
                )
            size = arr.size
            slices.append(FieldSlice(
                name=fn,
                start=cursor,
                end=cursor + size,
                shape=tuple(arr.shape),
                dtype=arr.dtype,
                role="solver",
            ))
            # ravel(order='C') gives a C-order flattened view if possible,
            # but since we will concatenate, we use a copy for safety.
            chunks.append(arr.ravel(order='C'))
            cursor += size
        if chunks:
            flat = np.concatenate(chunks).astype(np.float64, copy=False)
        else:
            flat = np.zeros(0, dtype=np.float64)
        layout = SolverLayout(
            field_slices=tuple(slices),
            total_size=cursor,
            solver_fields=self._solver_fields,
            is_partial=is_partial,
        )
        return flat, layout

    @classmethod
    def unflatten_from_solver(
        cls,
        flat: np.ndarray,
        layout: SolverLayout,
        base_state: "State | None" = None,
        allow_partial: bool = False,
    ) -> "State":
        """Reconstruct a State from a flat vector + layout.

        - layout complete (is_partial=False), base_state=None:
            reconstruct State containing only solver fields.
        - layout complete, base_state given:
            reconstruct State preserving non-solver fields from base_state.
            Requires layout.solver_fields == base_state.solver_fields and
            shape/dtype compatibility for all overridden fields.
        - layout partial, base_state given, allow_partial=True:
            modify solver-field subset, preserve all other fields from base_state.
            Requires shape/dtype compatibility for all overridden fields.
        - layout partial without base_state or without allow_partial:
            raise ValueError.
        """
        if not isinstance(flat, np.ndarray):
            raise TypeError(f"flat must be np.ndarray, got {type(flat).__name__}")
        if flat.dtype != np.float64:
            raise TypeError(f"flat dtype must be float64, got {flat.dtype}")
        if flat.shape != (layout.total_size,):
            raise ValueError(
                f"flat shape {flat.shape} != layout total_size ({layout.total_size},)"
            )

        if layout.is_partial:
            if base_state is None:
                raise ValueError(
                    "partial layout requires base_state for unflatten"
                )
            if not allow_partial:
                raise ValueError(
                    "partial layout requires explicit allow_partial=True"
                )

        # === Guard: layout/base_state compatibility ===
        # If base_state is provided, layout must be compatible with it.
        # This prevents accidentally applying a layout from State A to State B.
        if base_state is not None:
            if layout.is_partial:
                # Partial: every field in layout must be in base_state.solver_fields
                # AND match shape/dtype.
                for fs in layout.field_slices:
                    if fs.name not in base_state._solver_fields:
                        raise ValueError(
                            f"partial layout: field {fs.name!r} not in "
                            f"base_state.solver_fields {base_state._solver_fields}"
                        )
                    base_arr = base_state._fields[fs.name]
                    if base_arr.shape != fs.shape:
                        raise ValueError(
                            f"partial layout: field {fs.name!r} shape "
                            f"{fs.shape} != base_state shape {base_arr.shape}"
                        )
                    if base_arr.dtype != fs.dtype:
                        raise TypeError(
                            f"partial layout: field {fs.name!r} dtype "
                            f"{fs.dtype} != base_state dtype {base_arr.dtype}"
                        )
            else:
                # Full: layout.solver_fields must match base_state.solver_fields
                # exactly (same order). Shape/dtype must match per field.
                if layout.solver_fields != base_state._solver_fields:
                    raise ValueError(
                        f"full layout solver_fields {layout.solver_fields} "
                        f"!= base_state.solver_fields {base_state._solver_fields}"
                    )
                for fs in layout.field_slices:
                    base_arr = base_state._fields[fs.name]
                    if base_arr.shape != fs.shape:
                        raise ValueError(
                            f"full layout: field {fs.name!r} shape "
                            f"{fs.shape} != base_state shape {base_arr.shape}"
                        )
                    if base_arr.dtype != fs.dtype:
                        raise TypeError(
                            f"full layout: field {fs.name!r} dtype "
                            f"{fs.dtype} != base_state dtype {base_arr.dtype}"
                        )

        # Build the fields dict
        if base_state is not None:
            new_fields = {name: arr.copy() for name, arr in base_state._fields.items()}
            solver_fields = base_state._solver_fields
            metadata = base_state._metadata
        else:
            new_fields = {}
            solver_fields = layout.solver_fields
            metadata = {}

        # Override the slices present in layout
        for fs in layout.field_slices:
            chunk = flat[fs.start:fs.end]
            arr = chunk.reshape(fs.shape).astype(fs.dtype, copy=True)
            # Ensure C-contiguous result
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            new_fields[fs.name] = arr

        return cls(
            fields=new_fields,
            solver_fields=solver_fields,
            metadata=metadata,
        )

    # === Algebra (controlled, no magic) ===

    def scale(self, a: float) -> "State":
        """Multiply all solver fields by scalar a. Non-solver fields preserved."""
        new_fields = dict(self._fields)
        for fn in self._solver_fields:
            new_fields[fn] = self._fields[fn] * float(a)
        return State(
            fields=new_fields,
            solver_fields=self._solver_fields,
            metadata=self._metadata,
        )

    def add(self, other: "State") -> "State":
        """Add solver fields elementwise. Non-solver fields kept from self."""
        if self._solver_fields != other._solver_fields:
            raise ValueError(
                f"add: solver_fields mismatch {self._solver_fields} vs {other._solver_fields}"
            )
        new_fields = dict(self._fields)
        for fn in self._solver_fields:
            a = self._fields[fn]
            b = other._fields[fn]
            if a.shape != b.shape:
                raise ValueError(f"add: shape mismatch on {fn!r}")
            new_fields[fn] = a + b
        return State(
            fields=new_fields,
            solver_fields=self._solver_fields,
            metadata=self._metadata,
        )

    def subtract(self, other: "State") -> "State":
        if self._solver_fields != other._solver_fields:
            raise ValueError(
                f"subtract: solver_fields mismatch"
            )
        new_fields = dict(self._fields)
        for fn in self._solver_fields:
            a = self._fields[fn]
            b = other._fields[fn]
            if a.shape != b.shape:
                raise ValueError(f"subtract: shape mismatch on {fn!r}")
            new_fields[fn] = a - b
        return State(
            fields=new_fields,
            solver_fields=self._solver_fields,
            metadata=self._metadata,
        )

    def __repr__(self) -> str:
        return (
            f"State(solver_fields={self._solver_fields}, "
            f"all_fields={list(self._fields.keys())})"
        )
