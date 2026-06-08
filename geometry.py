# -*- coding: utf-8 -*-
"""
mcq_v5.geometry — Finite-volume conservative scalar conformal operator.

GeometryOperator computes:
- gradient on interior faces only
- flux J = -h_face * grad(field) (6d-compatible)
- divergence -div(J), the conservative RHS contribution
- apply: spatial RHS for one selected scalar solver field

It does NOT compute:
- modular coupling
- overlap R_ij
- scheduler decisions
- inter-modular aggregation
- coupling sources / sinks
- synchronisation
- h-dynamics
- metric updates

Design rules:

- STATELESS. GeometryOperator stores ONLY self._grid. No cache, no
  self._last_flux, no temporary arrays kept across calls. Pure functions
  of inputs.
- 6d-compatible convention:
    flux         J = -h_face * grad(psi)
    divergence   returns -div(J)
    dpsi/dt      = divergence(flux(psi, metric))
- Zero-flux Neumann at external boundaries (implicit: no exterior faces).
- TensorMetric rejected by flux/apply (NotImplementedError).

Critère étape 7:
    GeometryOperator computes scalar conformal finite-volume gradients,
    fluxes, and conservative RHS -div(J) under zero-flux Neumann boundaries,
    with 6d-compatible harmonic face coefficients, statelessly, and
    without modular coupling, metric dynamics, or solver logic.
"""

from __future__ import annotations
import numpy as np

from .grid import Grid
from .state import State
from .metric import Metric, ScalarConformalMetric, TensorMetric


class GeometryOperator:
    """Stateless scalar conformal finite-volume operator.

    Pure function of inputs. Configured by Grid at construction.
    No caching of any kind.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid: Grid):
        if not isinstance(grid, Grid):
            raise TypeError(
                f"GeometryOperator requires Grid, got {type(grid).__name__}"
            )
        self._grid = grid

    @property
    def grid(self) -> Grid:
        return self._grid

    # =====================================================================
    # gradient
    # =====================================================================

    def gradient(self, field: np.ndarray) -> tuple[np.ndarray, ...]:
        """Per-axis gradient on interior faces.

        Convention:
            grad_axis[k+1/2] = (field[k+1] - field[k]) / dx[axis]

        Shapes (for 3D grid (N0, N1, N2)):
            grad[0]: (N0-1, N1, N2)
            grad[1]: (N0, N1-1, N2)
            grad[2]: (N0, N1, N2-1)

        Returns a tuple of length grid.dim, one array per axis.

        Validation:
            field.dtype == np.float64
            field.flags['C_CONTIGUOUS']
            field.shape == grid.shape
        """
        self._validate_scalar_field(field)
        dim = self._grid.dim
        dx = self._grid.dx
        grads = []
        for axis in range(dim):
            slicer_left = [slice(None)] * dim
            slicer_right = [slice(None)] * dim
            slicer_left[axis] = slice(None, -1)
            slicer_right[axis] = slice(1, None)
            left = field[tuple(slicer_left)]
            right = field[tuple(slicer_right)]
            g = (right - left) / dx[axis]
            # Ensure C-contiguous output (slicing of contiguous arrays
            # typically remains contiguous along non-leading axes; force
            # if needed without silent reshape)
            if not g.flags["C_CONTIGUOUS"]:
                g = np.ascontiguousarray(g)
            grads.append(g)
        return tuple(grads)

    # =====================================================================
    # flux
    # =====================================================================

    def flux(
        self,
        field: np.ndarray,
        metric: Metric,
    ) -> tuple[np.ndarray, ...]:
        """Per-axis conservative flux J = -h_face * grad(field).

        Uses ScalarConformalMetric.face_coefficients(axis) for h_face
        (harmonic mean, 6d-compatible).

        TensorMetric is rejected: raises NotImplementedError. Tensorial
        flux belongs to V5-4 (TensorMetric implementation).

        Returns a tuple of length grid.dim.
        """
        if isinstance(metric, TensorMetric):
            raise NotImplementedError(
                "GeometryOperator.flux on TensorMetric: tensorial flux not "
                "implemented in V5-0a (deferred to V5-4)"
            )
        if not isinstance(metric, ScalarConformalMetric):
            raise TypeError(
                f"flux expects ScalarConformalMetric, got {type(metric).__name__}"
            )

        # Shape compatibility between metric and grid
        if tuple(metric.shape) != tuple(self._grid.shape):
            raise ValueError(
                f"metric shape {metric.shape} != grid shape {self._grid.shape}"
            )

        grads = self.gradient(field)
        fluxes = []
        for axis in range(self._grid.dim):
            h_face = metric.face_coefficients(axis)
            if h_face.shape != grads[axis].shape:
                raise ValueError(
                    f"shape mismatch on axis {axis}: h_face {h_face.shape} "
                    f"vs grad {grads[axis].shape}"
                )
            J = -h_face * grads[axis]
            if not J.flags["C_CONTIGUOUS"]:
                J = np.ascontiguousarray(J)
            fluxes.append(J)
        return tuple(fluxes)

    # =====================================================================
    # divergence
    # =====================================================================

    def divergence(
        self,
        fluxes: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """RETURNS -div(J), THE CONSERVATIVE RHS CONTRIBUTION, NOT +div(J).

        Sign convention: dpsi/dt = divergence(flux(psi, metric))

        In 1D with N cells and N-1 interior faces J[0..N-2]:
            dpsi[0]    = -J[0] / dx
            dpsi[i]    = (J[i-1] - J[i]) / dx       (1 <= i <= N-2)
            dpsi[N-1]  =  J[N-2] / dx

        External faces are zero-flux Neumann (not stored, implicit).
        Mass conserved to machine precision: sum(dpsi) ~= 0.

        Multi-D: contributions from each axis summed.

        Returns array of shape grid.shape, float64, C-contiguous.
        """
        dim = self._grid.dim
        dx = self._grid.dx
        if not isinstance(fluxes, tuple):
            raise TypeError(
                f"fluxes must be tuple, got {type(fluxes).__name__}"
            )
        if len(fluxes) != dim:
            raise ValueError(
                f"fluxes must have length {dim}, got {len(fluxes)}"
            )

        # Validate per-axis shapes
        grid_shape = self._grid.shape
        for axis, J in enumerate(fluxes):
            if not isinstance(J, np.ndarray):
                raise TypeError(
                    f"flux on axis {axis} must be ndarray, got {type(J).__name__}"
                )
            expected_shape = list(grid_shape)
            expected_shape[axis] -= 1
            if tuple(J.shape) != tuple(expected_shape):
                raise ValueError(
                    f"flux on axis {axis}: shape {J.shape}, "
                    f"expected {tuple(expected_shape)}"
                )
            if J.dtype != np.float64:
                raise TypeError(
                    f"flux on axis {axis}: dtype {J.dtype}, expected float64"
                )

        # Compute -div(J) per axis, accumulate
        dpsi = np.zeros(grid_shape, dtype=np.float64)

        for axis in range(dim):
            J = fluxes[axis]
            # Build per-axis contribution to dpsi
            # contrib[0]    = -J[0] / dx[axis]
            # contrib[i]    = (J[i-1] - J[i]) / dx[axis]   for 1 <= i <= N-2
            # contrib[N-1]  =  J[N-2] / dx[axis]
            contrib = np.zeros(grid_shape, dtype=np.float64)

            # Interior cells: J[i-1] - J[i]
            # Slicer for cells 1..N-2 (interior of axis)
            sl_interior = [slice(None)] * dim
            sl_interior[axis] = slice(1, -1)
            sl_J_left = [slice(None)] * dim
            sl_J_left[axis] = slice(None, -1)  # J[i-1] for i in [1..N-2] -> J[0..N-3]
            sl_J_right = [slice(None)] * dim
            sl_J_right[axis] = slice(1, None)  # J[i] for i in [1..N-2] -> J[1..N-2]
            # J_left[i] corresponds to J[i-1] for interior cell i
            # Actually: for cells 1..N-2 (length N-2), we need J[0..N-3] and J[1..N-2]
            # J has length N-1, so:
            sl_J_for_left = [slice(None)] * dim
            sl_J_for_left[axis] = slice(None, -1)  # J[0..N-2-1] = J[0..N-3]
            sl_J_for_right = [slice(None)] * dim
            sl_J_for_right[axis] = slice(1, None)  # J[1..N-2]
            J_left = J[tuple(sl_J_for_left)]   # length N-2 along axis
            J_right = J[tuple(sl_J_for_right)] # length N-2 along axis
            contrib[tuple(sl_interior)] = (J_left - J_right) / dx[axis]

            # Left boundary (cell 0): -J[0] / dx
            sl_left_bd = [slice(None)] * dim
            sl_left_bd[axis] = 0
            sl_J0 = [slice(None)] * dim
            sl_J0[axis] = 0
            contrib[tuple(sl_left_bd)] = -J[tuple(sl_J0)] / dx[axis]

            # Right boundary (cell N-1): J[N-2] / dx
            sl_right_bd = [slice(None)] * dim
            sl_right_bd[axis] = -1
            sl_Jlast = [slice(None)] * dim
            sl_Jlast[axis] = -1
            contrib[tuple(sl_right_bd)] = J[tuple(sl_Jlast)] / dx[axis]

            dpsi += contrib

        if not dpsi.flags["C_CONTIGUOUS"]:
            dpsi = np.ascontiguousarray(dpsi)
        return dpsi

    # =====================================================================
    # apply
    # =====================================================================

    def apply(
        self,
        state: State,
        metric: Metric,
        field_name: str = "psi",
    ) -> State:
        """Spatial RHS for selected scalar solver field.

        Computes d(field_name)/dt = divergence(flux(state[field_name], metric)).
        All other solver fields receive a typed zero derivative.
        Static fields (non-solver) are preserved from `state`.

        The returned State represents a time derivative (dpsi/dt), not an
        evolved state. The metadata of the returned State is a defensive
        copy of state.metadata with two trace keys added:
            metadata["operator"] = "GeometryOperator.apply"
            metadata["derivative_field"] = field_name
        """
        # --- Validation ---
        if not isinstance(state, State):
            raise TypeError(
                f"state must be State, got {type(state).__name__}"
            )
        if field_name not in state.fields:
            raise KeyError(f"field {field_name!r} not in state")
        if field_name not in state.solver_fields:
            raise ValueError(
                f"field {field_name!r} is not a solver field of state "
                f"(solver_fields={state.solver_fields})"
            )

        field = state.fields[field_name]
        if tuple(field.shape) != tuple(self._grid.shape):
            raise ValueError(
                f"state[{field_name!r}].shape {field.shape} != grid.shape "
                f"{self._grid.shape} (apply requires scalar field on grid)"
            )

        # --- Compute derivative ---
        fluxes = self.flux(field, metric)
        dfield = self.divergence(fluxes)

        # --- Build dstate ---
        # All solver fields present: typed zeros for those != field_name
        new_fields: dict[str, np.ndarray] = {}
        for sf in state.solver_fields:
            if sf == field_name:
                new_fields[sf] = dfield
            else:
                # Typed zero array, same shape as the source solver field,
                # float64, C-contiguous (zeros() is C-contiguous by default)
                src = state.fields[sf]
                z = np.zeros(src.shape, dtype=np.float64)
                new_fields[sf] = z

        # Preserve static fields by copy (they are not solver fields)
        for fn, arr in state.fields.items():
            if fn not in state.solver_fields:
                # Defensive copy via State construction anyway, but explicit here
                new_fields[fn] = arr.copy()

        # Metadata: defensive copy + trace
        derived_metadata = dict(state.metadata) if state.metadata else {}
        derived_metadata["operator"] = "GeometryOperator.apply"
        derived_metadata["derivative_field"] = field_name

        return State(
            fields=new_fields,
            solver_fields=state.solver_fields,
            metadata=derived_metadata,
        )

    # =====================================================================
    # Internal validation
    # =====================================================================

    def _validate_scalar_field(self, field: np.ndarray) -> None:
        if not isinstance(field, np.ndarray):
            raise TypeError(
                f"field must be np.ndarray, got {type(field).__name__}"
            )
        if field.dtype != np.float64:
            raise TypeError(
                f"field dtype must be float64, got {field.dtype}"
            )
        if not field.flags["C_CONTIGUOUS"]:
            raise ValueError("field must be C-contiguous")
        if tuple(field.shape) != tuple(self._grid.shape):
            raise ValueError(
                f"field shape {field.shape} != grid shape {self._grid.shape}"
            )

    def __repr__(self) -> str:
        return f"GeometryOperator(grid_shape={self._grid.shape})"
