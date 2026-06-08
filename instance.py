# -*- coding: utf-8 -*-
"""
mcq_v5.instance — Multi-module typed container.

Instance is a collection of Module objects at present time t.
It is NOT a temporal manager.

Responsibilities:
- aggregate multiple Module objects under unique ids
- expose bounded accessors
- provide non-mutating diagnostics

NON-responsibilities (explicitly):
- coupling
- overlap R_ij
- scheduler
- aggregation
- flux / gradient / divergence
- dynamics / step / evolve
- synchronization / consensus
- mean_state / average_metric
- history list / commit_step / HistoryEntry maintenance
  (history is maintained by Solver or ExperimentProtocol, not Instance)
- inter-module spatial shape uniformity (deferred to CouplingScheduler
  or ExperimentProtocol)

Design rules:
- M = 1, 2, 3, ... handled uniformly, no special-case branches
- modules dict shallow-copied at construction (Module objects by reference)
- metadata defensively copied
- modules property returns a copy of the dict (prevents mutation)
- get_module(id) returns the actual Module reference

Critère étape 6:
    Instance is a typed multi-module container,
    accepting M=1, M=2, M=3 without special cases,
    copying container metadata defensively,
    preserving Module references,
    exposing bounded accessors,
    and containing no coupling, overlap, aggregation, or dynamics logic.
"""

from __future__ import annotations
from typing import Any

from .module import Module, ModuleId


class Instance:
    """Typed collection of Module objects at present time t.

    Instance represents the spatial state of the multi-module network at a
    single time. Temporal persistence (history of past states, commit
    semantics) is the responsibility of Solver or ExperimentProtocol.
    """

    __slots__ = ("_id", "_modules", "_metadata")

    def __init__(
        self,
        id: ModuleId,
        modules: dict[ModuleId, Module],
        metadata: dict[str, Any] | None = None,
    ):
        # --- Type validation ---
        if not isinstance(id, (str, int)):
            raise TypeError(
                f"id must be str or int, got {type(id).__name__}"
            )
        if not isinstance(modules, dict):
            raise TypeError(
                f"modules must be dict, got {type(modules).__name__}"
            )
        if len(modules) == 0:
            raise ValueError("modules dict must not be empty")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be dict or None, got {type(metadata).__name__}"
            )

        # --- Per-entry validation ---
        for key, module in modules.items():
            if not isinstance(key, (str, int)):
                raise TypeError(
                    f"modules key must be str or int, got {type(key).__name__}"
                )
            if not isinstance(module, Module):
                raise TypeError(
                    f"modules value for key {key!r} must be Module, "
                    f"got {type(module).__name__}"
                )
            if key != module.id:
                raise ValueError(
                    f"dict key {key!r} does not match module.id {module.id!r}"
                )

        # --- Store ---
        # modules: shallow-copy the dict, keep Module objects by reference
        # metadata: defensive copy
        self._id = id
        self._modules = dict(modules)
        self._metadata = dict(metadata) if metadata else {}

    # === Read accessors ===

    @property
    def id(self) -> ModuleId:
        """Instance identifier. Immutable after construction (no setter)."""
        return self._id

    @property
    def modules(self) -> dict[ModuleId, Module]:
        """Return a shallow copy of the modules dict.

        The copy prevents external code from mutating Instance's internal
        dict (e.g., instance.modules['new'] = m would not affect Instance).
        The Module objects themselves are NOT copied: modifying a Module via
        a previously-held reference still affects Instance.
        """
        return dict(self._modules)

    @property
    def metadata(self) -> dict:
        return self._metadata

    def module_ids(self) -> tuple[ModuleId, ...]:
        """Return the tuple of module ids, in insertion order."""
        return tuple(self._modules.keys())

    def n_modules(self) -> int:
        return len(self._modules)

    def get_module(self, id: ModuleId) -> Module:
        """Return the Module with given id by reference.

        Raises KeyError if id not present.
        """
        if id not in self._modules:
            raise KeyError(f"module id {id!r} not in Instance")
        return self._modules[id]

    # === Diagnostics (non-mutating) ===

    def diagnostics(self) -> dict:
        """Return diagnostic scalars. Never mutates modules or their state.

        Includes per-module diagnostics by delegation; also reports per-module
        spatial shapes as a passive observation (no enforcement).
        """
        per_module = {
            mid: module.diagnostics()
            for mid, module in self._modules.items()
        }
        per_module_spatial = {}
        for mid, module in self._modules.items():
            # Conservative read: ScalarConformalMetric has .shape;
            # TensorMetric has .shape but block dims trail. We just report
            # the raw metric shape here without interpretation.
            shape = getattr(module.metric, "shape", None)
            per_module_spatial[mid] = tuple(shape) if shape is not None else None
        return {
            "id": self._id,
            "n_modules": len(self._modules),
            "module_ids": tuple(self._modules.keys()),
            "module_diagnostics": per_module,
            "module_metric_shapes": per_module_spatial,
        }

    # === Optional convenience constructors ===

    def with_module(self, module: Module) -> "Instance":
        """Return a new Instance with `module` added or replacing same-id entry."""
        new_modules = dict(self._modules)
        new_modules[module.id] = module
        return Instance(
            id=self._id,
            modules=new_modules,
            metadata=self._metadata,
        )

    def with_modules(
        self, modules: dict[ModuleId, Module]
    ) -> "Instance":
        """Return a new Instance with a completely new modules dict."""
        return Instance(
            id=self._id,
            modules=modules,
            metadata=self._metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Instance(id={self._id!r}, "
            f"n_modules={len(self._modules)}, "
            f"module_ids={tuple(self._modules.keys())})"
        )
