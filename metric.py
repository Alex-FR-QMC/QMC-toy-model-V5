# -*- coding: utf-8 -*-
"""
mcq_v5.metric — Metric abstraction and concrete implementations.

V5-0a scope:

- Metric: abstract interface (geometry only, no dynamics).
- ScalarConformalMetric: 6d-compat scalar h(theta) with harmonic-mean
  face coefficients. Real implementation, usable immediately.
- TensorMetric: stub with SPD validation (eigvalsh on last two axes, fully
  vectorized). Methods that would imply dynamics or full Riemannian
  geometry raise NotImplementedError.

Strict responsibilities (V5-0a):

Metric provides geometry. It does NOT compute:
- gradient
- divergence
- flux
- diffusion step
- solver step
- coupling

These belong to GeometryOperator, Solver, CouplingOperator (later steps).

Design rules:
- No silent dtype conversion (no np.asarray(..., dtype=...)).
- No silent contiguity coercion (no np.ascontiguousarray).
- Defensive copy on construction (consistent with State).
- Validations vectorized (no Python spatial loops for invariants).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


# Smallest representable positive float64. Used as numerical floor for
# harmonic-mean denominators. Since h > 0 is enforced at construction,
# this never masks a physics violation; it only prevents pure-zero division.
EPS_TINY = float(np.finfo(np.float64).tiny)


# =========================================================================
# Abstract Metric
# =========================================================================

class Metric(ABC):
    """Abstract metric interface. Geometry only. No dynamics."""

    @abstractmethod
    def face_coefficients(self, axis: int) -> np.ndarray:
        """Return per-face metric coefficients along given axis."""
        ...

    @abstractmethod
    def inverse(self) -> np.ndarray:
        """Return the pointwise inverse metric (convention-dependent)."""
        ...

    @abstractmethod
    def determinant(self, mode: str = "default") -> np.ndarray:
        """Return the pointwise determinant under a documented convention."""
        ...

    @abstractmethod
    def volume_element(self) -> np.ndarray:
        """Return the pointwise volume element under a documented convention."""
        ...

    @abstractmethod
    def distance(
        self,
        field_a: np.ndarray,
        field_b: np.ndarray,
        mode: str = "metric_l2",
    ) -> float:
        """Compute a distance between two scalar fields under the metric."""
        ...

    @abstractmethod
    def update(self, *args, **kwargs) -> "Metric":
        """Return an updated metric. V5-0a: stubs raise NotImplementedError."""
        ...

    @abstractmethod
    def diagnostics(self) -> dict:
        """Return diagnostic scalars (no phenomenology)."""
        ...

    def view(self):
        """Return a MetricView wrapping this metric (zero-copy)."""
        from .views import MetricView
        return MetricView(self)


# =========================================================================
# ScalarConformalMetric
# =========================================================================

class ScalarConformalMetric(Metric):
    """Scalar conformal metric h(theta), 6d-compat.

    The metric is a strictly positive scalar field h on the grid.
    Face coefficients use harmonic mean (reproduces 6d engine convention).

    Convention V5-0a (placeholder, not full Riemannian semantics):
    - inverse() = 1 / h
    - determinant(mode="scalar_conformal_placeholder") = h ** dim
    - volume_element() = h ** (dim / 2)   (placeholder convention)
    - distance(a, b, mode="metric_l2") = sqrt(sum(h * (a - b)^2))

    These are NOT a strict Laplace-Beltrami implementation. They are
    documented placeholders for V5-0a diagnostics.
    """

    __slots__ = ("_h", "_h0", "_h_min", "_metadata")

    def __init__(
        self,
        h: np.ndarray,
        h0: float | np.ndarray | None = None,
        h_min: float = 1e-30,
        metadata: dict[str, Any] | None = None,
    ):
        # --- Validation of h ---
        if not isinstance(h, np.ndarray):
            raise TypeError(f"h must be np.ndarray, got {type(h).__name__}")
        if h.dtype != np.float64:
            raise TypeError(f"h dtype must be float64, got {h.dtype}")
        if not h.flags["C_CONTIGUOUS"]:
            raise ValueError("h must be C-contiguous")
        # Strict positivity (vectorized)
        if not np.all(h > 0):
            raise ValueError(
                f"h must be strictly positive everywhere; "
                f"min(h)={h.min()}, count_nonpositive={int(np.sum(h <= 0))}"
            )

        # --- Validation of h0 if ndarray ---
        if isinstance(h0, np.ndarray):
            if h0.dtype != np.float64:
                raise TypeError(f"h0 dtype must be float64, got {h0.dtype}")
            if not h0.flags["C_CONTIGUOUS"]:
                raise ValueError("h0 must be C-contiguous")
            if h0.shape != h.shape:
                raise ValueError(
                    f"h0 shape {h0.shape} must match h shape {h.shape}"
                )
            if not np.all(h0 > 0):
                raise ValueError("h0 must be strictly positive everywhere")
        elif h0 is None:
            pass
        elif isinstance(h0, (int, float, np.floating)):
            if float(h0) <= 0:
                raise ValueError(f"h0 scalar must be > 0, got {h0}")
        else:
            raise TypeError(
                f"h0 must be None, scalar, or np.ndarray, got {type(h0).__name__}"
            )

        # --- Validation of h_min ---
        if not isinstance(h_min, (int, float, np.floating)):
            raise TypeError(f"h_min must be scalar, got {type(h_min).__name__}")
        if float(h_min) <= 0:
            raise ValueError(f"h_min must be > 0, got {h_min}")

        # --- Defensive copy ---
        self._h = h.copy()
        if isinstance(h0, np.ndarray):
            self._h0 = h0.copy()
        else:
            self._h0 = h0
        self._h_min = float(h_min)
        self._metadata = dict(metadata) if metadata else {}

    # === Read accessors ===

    @property
    def h(self) -> np.ndarray:
        return self._h

    @property
    def coefficients(self) -> np.ndarray:
        """MetricView protocol: scalar coefficients = h."""
        return self._h

    @property
    def h0(self) -> float | np.ndarray | None:
        return self._h0

    @property
    def h_min(self) -> float:
        return self._h_min

    @property
    def shape(self) -> tuple[int, ...]:
        return self._h.shape

    @property
    def dim(self) -> int:
        return self._h.ndim

    @property
    def metadata(self) -> dict:
        return self._metadata

    # === Face coefficients (harmonic mean, 6d-compat) ===

    def face_coefficients(self, axis: int) -> np.ndarray:
        """Harmonic mean of h on faces along given axis.

        Convention identical to 6d engine:
            h_face = 2 * h_L * h_R / (h_L + h_R + EPS_TINY)

        EPS_TINY = np.finfo(np.float64).tiny prevents pure-zero division
        without masking physics (h > 0 enforced at construction).

        Shapes:
            axis=0 -> (N0-1, N1, N2, ...)
            axis=1 -> (N0, N1-1, N2, ...)
            axis=2 -> (N0, N1, N2-1, ...)
        """
        if not isinstance(axis, (int, np.integer)):
            raise TypeError(f"axis must be int, got {type(axis).__name__}")
        if not 0 <= axis < self._h.ndim:
            raise ValueError(
                f"axis {axis} out of range for h.ndim={self._h.ndim}"
            )
        # Build slices for left/right along axis
        slicer_left = [slice(None)] * self._h.ndim
        slicer_right = [slice(None)] * self._h.ndim
        slicer_left[axis] = slice(None, -1)
        slicer_right[axis] = slice(1, None)
        h_L = self._h[tuple(slicer_left)]
        h_R = self._h[tuple(slicer_right)]
        return 2.0 * h_L * h_R / (h_L + h_R + EPS_TINY)

    # === Pointwise geometric quantities (placeholder conventions V5-0a) ===

    def inverse(self) -> np.ndarray:
        """Pointwise 1 / h. Safe since h > 0 enforced at construction."""
        return 1.0 / self._h

    def determinant(self, mode: str = "scalar_conformal_placeholder") -> np.ndarray:
        """Pointwise determinant under documented placeholder convention.

        V5-0a placeholder: det = h ** dim. NOT a strict Riemannian determinant.
        """
        if mode != "scalar_conformal_placeholder":
            raise NotImplementedError(
                f"determinant mode {mode!r} not implemented in V5-0a"
            )
        return self._h ** self.dim

    def volume_element(self) -> np.ndarray:
        """Pointwise volume element (placeholder convention).

        V5-0a placeholder: vol = h ** (dim / 2). NOT a strict
        Riemannian volume element.
        """
        return self._h ** (self.dim / 2.0)

    # === Distance ===

    def distance(
        self,
        field_a: np.ndarray,
        field_b: np.ndarray,
        mode: str = "metric_l2",
    ) -> float:
        """Distance between two scalar fields under the metric.

        mode="metric_l2": sqrt(sum(h * (a - b)**2)).
        Other modes: NotImplementedError.
        """
        if mode != "metric_l2":
            raise NotImplementedError(f"distance mode {mode!r} not implemented")
        if not isinstance(field_a, np.ndarray) or not isinstance(field_b, np.ndarray):
            raise TypeError("field_a, field_b must be np.ndarray")
        if field_a.shape != field_b.shape:
            raise ValueError(
                f"shape mismatch: {field_a.shape} vs {field_b.shape}"
            )
        if field_a.shape != self._h.shape:
            raise ValueError(
                f"field shape {field_a.shape} does not match metric shape {self._h.shape}"
            )
        diff = field_a - field_b
        return float(np.sqrt(np.sum(self._h * diff * diff)))

    # === update() ===

    def update(self, *args, **kwargs):
        raise NotImplementedError(
            "metric dynamics not implemented in V5-0a"
        )

    # === Diagnostics ===

    def diagnostics(self) -> dict:
        """Diagnostic scalars on h. No phenomenology."""
        h = self._h
        return {
            "min_h": float(h.min()),
            "max_h": float(h.max()),
            "mean_h": float(h.mean()),
            "median_h": float(np.median(h)),
            "shape": tuple(h.shape),
            "dim": self.dim,
            "strictly_positive": bool(np.all(h > 0)),
            "n_below_h_min": int(np.sum(h < self._h_min)),
        }


# =========================================================================
# TensorMetric — stub with SPD validation
# =========================================================================

class TensorMetric(Metric):
    """Tensor metric H(theta), V5-0a stub.

    Shape contract: H of shape (..., dim, dim) where last two dims are square.
    For 3D grids: H shape is (Nx, Ny, Nz, 3, 3).

    Validations at construction (all vectorized):
    - dtype float64
    - C-contiguous
    - last two dims equal (square block)
    - symmetric within tolerance (atol on H - H.swapaxes(-1, -2))
    - SPD: eigvalsh(H) > 0 everywhere
    - eigenvalues within optional bounds

    All dynamic methods (face_coefficients, inverse, determinant, distance,
    update) raise NotImplementedError. Only diagnostics() and view() are
    operational.
    """

    __slots__ = ("_H", "_metadata", "_sym_atol")

    def __init__(
        self,
        H: np.ndarray,
        eig_bounds: tuple[float, float] | None = None,
        sym_atol: float = 1e-10,
        metadata: dict[str, Any] | None = None,
    ):
        # --- Validate dtype ---
        if not isinstance(H, np.ndarray):
            raise TypeError(f"H must be np.ndarray, got {type(H).__name__}")
        if H.dtype != np.float64:
            raise TypeError(f"H dtype must be float64, got {H.dtype}")
        if not H.flags["C_CONTIGUOUS"]:
            raise ValueError("H must be C-contiguous")

        # --- Validate shape: at least 2 dims, last two equal ---
        if H.ndim < 2:
            raise ValueError(
                f"H must have at least 2 dims, got ndim={H.ndim}"
            )
        d1, d2 = H.shape[-2], H.shape[-1]
        if d1 != d2:
            raise ValueError(
                f"H last two dims must be equal (square block), got {d1} vs {d2}"
            )

        # --- Validate symmetry (vectorized) ---
        H_T = H.swapaxes(-1, -2)
        asym = np.max(np.abs(H - H_T))
        if asym > sym_atol:
            raise ValueError(
                f"H not symmetric within sym_atol={sym_atol}, max |H - H.T| = {asym}"
            )

        # --- Validate SPD: eigvalsh on last two axes (vectorized) ---
        # eigvalsh broadcasts on leading dims, returns eigenvalues sorted ascending
        eigs = np.linalg.eigvalsh(H)
        min_eig = float(eigs.min())
        if min_eig <= 0:
            raise ValueError(
                f"H must be SPD (positive definite); min eigenvalue = {min_eig}"
            )

        # --- Optional eigenvalue bounds ---
        if eig_bounds is not None:
            lo, hi = eig_bounds
            if min_eig < lo:
                raise ValueError(
                    f"min eigenvalue {min_eig} below lower bound {lo}"
                )
            max_eig = float(eigs.max())
            if max_eig > hi:
                raise ValueError(
                    f"max eigenvalue {max_eig} above upper bound {hi}"
                )

        # --- Defensive copy ---
        self._H = H.copy()
        self._sym_atol = float(sym_atol)
        self._metadata = dict(metadata) if metadata else {}

    @property
    def H(self) -> np.ndarray:
        return self._H

    @property
    def coefficients(self) -> np.ndarray:
        """MetricView protocol: returns the H tensor."""
        return self._H

    @property
    def shape(self) -> tuple[int, ...]:
        return self._H.shape

    @property
    def block_dim(self) -> int:
        return self._H.shape[-1]

    @property
    def metadata(self) -> dict:
        return self._metadata

    # === Dynamic methods: stubs ===

    def face_coefficients(self, axis: int) -> np.ndarray:
        raise NotImplementedError(
            "TensorMetric.face_coefficients not implemented in V5-0a stub"
        )

    def inverse(self) -> np.ndarray:
        raise NotImplementedError(
            "TensorMetric.inverse not implemented in V5-0a stub"
        )

    def determinant(self, mode: str = "default") -> np.ndarray:
        raise NotImplementedError(
            "TensorMetric.determinant not implemented in V5-0a stub"
        )

    def volume_element(self) -> np.ndarray:
        raise NotImplementedError(
            "TensorMetric.volume_element not implemented in V5-0a stub"
        )

    def distance(
        self,
        field_a: np.ndarray,
        field_b: np.ndarray,
        mode: str = "metric_l2",
    ) -> float:
        raise NotImplementedError(
            "TensorMetric.distance not implemented in V5-0a stub"
        )

    def update(self, *args, **kwargs):
        raise NotImplementedError(
            "TensorMetric.update not implemented in V5-0a stub"
        )

    # === Diagnostics (operational) ===

    def diagnostics(self) -> dict:
        """Diagnostics on H: shape, eigenvalue bounds, symmetry residual.

        Eigenvalues are recomputed here (cheap on small grids; not cached
        to avoid stale state).
        """
        H = self._H
        eigs = np.linalg.eigvalsh(H)
        H_T = H.swapaxes(-1, -2)
        asym = float(np.max(np.abs(H - H_T)))
        return {
            "shape": tuple(H.shape),
            "block_dim": self.block_dim,
            "min_eigenvalue": float(eigs.min()),
            "max_eigenvalue": float(eigs.max()),
            "mean_eigenvalue": float(eigs.mean()),
            "max_asymmetry": asym,
            "spd": bool(eigs.min() > 0),
        }
