# -*- coding: utf-8 -*-
"""
mcq_v5.module — Local module container.

Module aggregates: id, state, metric, metadata.
Module exposes: state_view(), metric_view(), diagnostics().

Module does NOT compute:
- coupling
- flux
- divergence
- gradient
- dynamics
- aggregation
- synchronisation
- overlap R_ij

These responsibilities belong to later layers (GeometryOperator,
CouplingScheduler, CouplingOperator, CouplingAggregator).

Design rules:
- id is immutable after construction (no setter).
- state and metric are stored by reference (they already do defensive
  copy at their own construction).
- metadata is defensively copied at construction.
- Spatial compatibility between state solver fields and metric is
  validated at construction time.

Critère étape 5:
    Module is a local container with typed State/Metric references,
    bounded views, non-mutating diagnostics, and no coupling/dynamics logic.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from .state import State
from .metric import Metric, ScalarConformalMetric, TensorMetric
from .views import StateView, MetricView


# Type alias for module id (V5-0a: str | int permitted; no ModuleId class yet)
ModuleId = str | int


def _metric_spatial_shape(metric: Metric) -> tuple[int, ...]:
    """Return the spatial shape of the metric (excluding tensor block dims).

    - ScalarConformalMetric: metric.shape
    - TensorMetric: metric.shape[:-2] (block dims excluded)
    - Other concrete Metric subclasses: fall back on `coefficients.shape`
      with a conservative heuristic.
    """
    if isinstance(metric, ScalarConformalMetric):
        return tuple(metric.shape)
    if isinstance(metric, TensorMetric):
        return tuple(metric.shape[:-2])
    # Generic fallback: try to read coefficients
    if hasattr(metric, "coefficients") and isinstance(metric.coefficients, np.ndarray):
        coeffs = metric.coefficients
        # Conservative: assume scalar-like
        return tuple(coeffs.shape)
    raise TypeError(
        f"cannot infer spatial shape of metric of type {type(metric).__name__}"
    )


class Module:
    """Local container of one (State, Metric) pair with id and metadata.

    No dynamics. No coupling. Just an aggregator with bounded views.
    """

    __slots__ = ("_id", "_state", "_metric", "_metadata")

    def __init__(
        self,
        id: ModuleId,
        state: State,
        metric: Metric,
        metadata: dict[str, Any] | None = None,
    ):
        # --- Type validation ---
        if not isinstance(id, (str, int)):
            raise TypeError(
                f"id must be str or int, got {type(id).__name__}"
            )
        if not isinstance(state, State):
            raise TypeError(
                f"state must be State, got {type(state).__name__}"
            )
        if not isinstance(metric, Metric):
            raise TypeError(
                f"metric must be Metric, got {type(metric).__name__}"
            )
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be dict or None, got {type(metadata).__name__}"
            )

        # --- Spatial compatibility validation ---
        spatial_shape = _metric_spatial_shape(metric)
        self._validate_spatial_compatibility(state, spatial_shape)

        # --- Store ---
        # id, state, metric kept by reference (state/metric have own defensive copy)
        # metadata defensively copied
        self._id = id
        self._state = state
        self._metric = metric
        self._metadata = dict(metadata) if metadata else {}

    @staticmethod
    def _validate_spatial_compatibility(
        state: State,
        spatial_shape: tuple[int, ...],
    ) -> None:
        """Verify that every solver field of state has a spatial prefix
        equal to spatial_shape.

        - For scalar fields with shape == spatial_shape: accepted.
        - For tensor fields with shape (spatial_shape + extra_dims): accepted.
        - For mismatched leading dims: raise ValueError.
        """
        n_spatial = len(spatial_shape)
        for fn in state.solver_fields:
            arr = state.fields[fn]
            if arr.ndim < n_spatial:
                raise ValueError(
                    f"solver field {fn!r}: ndim={arr.ndim} < spatial dim {n_spatial}"
                )
            if tuple(arr.shape[:n_spatial]) != spatial_shape:
                raise ValueError(
                    f"solver field {fn!r}: spatial prefix {arr.shape[:n_spatial]} "
                    f"!= metric spatial shape {spatial_shape}"
                )

    # === Read accessors ===

    @property
    def id(self) -> ModuleId:
        """Module identifier. Immutable after construction (no setter)."""
        return self._id

    @property
    def state(self) -> State:
        return self._state

    @property
    def metric(self) -> Metric:
        return self._metric

    @property
    def metadata(self) -> dict:
        return self._metadata

    # === Views ===

    def state_view(self) -> StateView:
        """Return a bounded zero-copy view over the module's state."""
        return StateView(self._state)

    def metric_view(self) -> MetricView:
        """Return a bounded zero-copy view over the module's metric."""
        return self._metric.view()

    # === Diagnostics (non-mutating) ===

    def diagnostics(self) -> dict:
        """Return diagnostic scalars. Never mutates state nor metric."""
        return {
            "id": self._id,
            "state_solver_fields": tuple(self._state.solver_fields),
            "state_all_fields": tuple(self._state.fields.keys()),
            "state_norm": self._state.norm(),
            "metric_type": type(self._metric).__name__,
            "metric_diagnostics": self._metric.diagnostics(),
        }

    # === Optional convenience constructors ===

    def with_state(self, state: State) -> "Module":
        """Return a new Module with a different State (same id, metric, metadata)."""
        return Module(
            id=self._id,
            state=state,
            metric=self._metric,
            metadata=self._metadata,
        )

    def with_metric(self, metric: Metric) -> "Module":
        """Return a new Module with a different Metric (same id, state, metadata)."""
        return Module(
            id=self._id,
            state=self._state,
            metric=metric,
            metadata=self._metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Module(id={self._id!r}, "
            f"state_fields={tuple(self._state.fields.keys())}, "
            f"metric_type={type(self._metric).__name__})"
        )
