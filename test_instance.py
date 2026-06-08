# -*- coding: utf-8 -*-
"""Tests for mcq_v5.instance — multi-module container, M=1/2/3 uniform.

Includes audit guard: Instance must NOT maintain a history list or any
temporal persistence attribute. That responsibility belongs to Solver
or ExperimentProtocol.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.state import State
from mcq_v5.metric import ScalarConformalMetric, TensorMetric
from mcq_v5.module import Module
from mcq_v5.instance import Instance


# ===== Helpers =====

def make_state(shape=(5, 5, 5), psi_val=1.0):
    return State(
        fields={
            "psi": psi_val * np.ones(shape, dtype=np.float64),
            "h": 10.0 * np.ones(shape, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )


def make_metric(shape=(5, 5, 5)):
    return ScalarConformalMetric(np.ones(shape, dtype=np.float64))


def make_module(mid, shape=(5, 5, 5), psi_val=1.0):
    return Module(
        id=mid,
        state=make_state(shape, psi_val),
        metric=make_metric(shape),
    )


# ===== A/B/C. Construction M = 1, 2, 3 =====

def test_instance_construction_M1():
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})
    assert inst.n_modules() == 1
    assert inst.module_ids() == ("m0",)


def test_instance_construction_M2():
    m0 = make_module("m0")
    m1 = make_module("m1")
    inst = Instance(id="I", modules={"m0": m0, "m1": m1})
    assert inst.n_modules() == 2
    assert set(inst.module_ids()) == {"m0", "m1"}


def test_instance_construction_M3():
    mods = {f"m{i}": make_module(f"m{i}") for i in range(3)}
    inst = Instance(id="I", modules=mods)
    assert inst.n_modules() == 3
    assert set(inst.module_ids()) == {"m0", "m1", "m2"}


def test_instance_M1_no_special_case():
    """M=1 must be handled exactly like M>1."""
    m0 = make_module("m0")
    inst1 = Instance(id="I", modules={"m0": m0})
    # diagnostics, get_module, module_ids must all work uniformly
    _ = inst1.diagnostics()
    assert inst1.get_module("m0") is m0
    assert inst1.module_ids() == ("m0",)


# ===== D. module_ids =====

def test_instance_module_ids_preserves_insertion_order():
    mods = {f"m{i}": make_module(f"m{i}") for i in [3, 1, 2, 0]}
    inst = Instance(id="I", modules=mods)
    assert inst.module_ids() == ("m3", "m1", "m2", "m0")


# ===== E. get_module returns exact reference =====

def test_instance_get_module_returns_exact_reference():
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})
    assert inst.get_module("m0") is m0


# ===== F. unknown id raises KeyError =====

def test_instance_get_module_unknown_raises():
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})
    with pytest.raises(KeyError, match="not in Instance"):
        inst.get_module("unknown")


# ===== G. External dict mutation does not affect Instance =====

def test_instance_external_dict_modification_no_effect():
    m0 = make_module("m0")
    m1 = make_module("m1")
    external = {"m0": m0, "m1": m1}
    inst = Instance(id="I", modules=external)

    # Mutate external dict
    external["new"] = make_module("new")
    del external["m0"]

    # Instance unaffected
    assert set(inst.module_ids()) == {"m0", "m1"}
    assert "new" not in inst.module_ids()


# ===== H. modules property returns copy, mutation has no effect on internals =====

def test_instance_modules_property_is_copy():
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})
    mods = inst.modules
    mods["fake"] = make_module("fake")
    # Internal state unchanged
    assert "fake" not in inst.module_ids()
    assert inst.n_modules() == 1


# ===== I. Module objects kept by reference =====

def test_instance_modules_kept_by_reference():
    """Modifying a Module via the original reference reflects in Instance."""
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})

    # Modify state via the original Module's view
    m0.state_view().field("psi")[0, 0, 0] = 42.0

    # Instance reflects the change
    assert inst.get_module("m0").state["psi"][0, 0, 0] == 42.0


# ===== J. Metadata defensive copy =====

def test_instance_metadata_defensive_copy():
    m0 = make_module("m0")
    meta = {"protocol": "P1", "seed": 7}
    inst = Instance(id="I", modules={"m0": m0}, metadata=meta)
    meta["seed"] = 999
    meta["new"] = "value"
    assert inst.metadata == {"protocol": "P1", "seed": 7}


def test_instance_metadata_empty_default():
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})
    assert inst.metadata == {}


# ===== K. Invalid metadata type =====

def test_instance_invalid_metadata_type():
    m0 = make_module("m0")
    with pytest.raises(TypeError, match="metadata"):
        Instance(id="I", modules={"m0": m0}, metadata="not a dict")


# ===== L. Empty modules =====

def test_instance_empty_modules_raises():
    with pytest.raises(ValueError, match="empty"):
        Instance(id="I", modules={})


# ===== M. Key/id mismatch =====

def test_instance_key_id_mismatch_raises():
    m0 = make_module("m0")
    with pytest.raises(ValueError, match="does not match"):
        Instance(id="I", modules={"wrong_key": m0})


# ===== N. Invalid module value =====

def test_instance_invalid_module_value_raises():
    with pytest.raises(TypeError, match="must be Module"):
        Instance(id="I", modules={"m0": "not a module"})


# ===== O. Invalid module key type =====

def test_instance_invalid_module_key_type_raises():
    m0 = make_module("m0")
    # The dict can hold any key, so we need to construct a dict with a
    # tuple key, then pass it through
    bad_modules = {(1, 2): m0}
    with pytest.raises(TypeError, match="modules key"):
        Instance(id="I", modules=bad_modules)


# ===== Invalid id type =====

def test_instance_invalid_id_type():
    m0 = make_module("m0")
    with pytest.raises(TypeError, match="id"):
        Instance(id=[1, 2], modules={"m0": m0})


# ===== Invalid modules dict type =====

def test_instance_invalid_modules_type():
    with pytest.raises(TypeError, match="modules must be dict"):
        Instance(id="I", modules=[1, 2, 3])


# ===== P. No coupling / dynamics / overlap methods =====

def test_instance_no_coupling_methods():
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})
    forbidden = [
        "compute_coupling",
        "compute_overlap",
        "aggregate",
        "global_state",
        "mean_state",
        "average_metric",
        "synchronize",
        "step",
        "evolve",
        "flux",
        "gradient",
        "divergence",
    ]
    for name in forbidden:
        assert not hasattr(inst, name), f"Instance should not have {name!r}"


# ===== Audit guard: no history / temporal persistence =====

def test_instance_no_history_attribute():
    """Instance must NOT maintain a history list or temporal state.

    History belongs to Solver / ExperimentProtocol.
    """
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})
    forbidden_temporal = [
        "history",
        "_history",
        "commit_step",
        "append_step",
        "record",
        "snapshot",
        "save_state",
        "previous_state",
        "history_view",
        "advance",
        "tick",
    ]
    for name in forbidden_temporal:
        assert not hasattr(inst, name), (
            f"Instance must not own temporal attribute {name!r}"
        )


def test_instance_uses_slots_no_adhoc_attributes():
    """__slots__ prevents ad-hoc attribute addition, including history-like."""
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})
    with pytest.raises(AttributeError):
        inst.history = []  # would be a temporal escape hatch
    with pytest.raises(AttributeError):
        inst.commit_step = lambda: None


# ===== id immutability =====

def test_instance_id_no_setter():
    m0 = make_module("m0")
    inst = Instance(id="original", modules={"m0": m0})
    with pytest.raises(AttributeError):
        inst.id = "modified"


# ===== Q. Diagnostics non-mutating =====

def test_instance_diagnostics_does_not_mutate():
    m0 = make_module("m0")
    m1 = make_module("m1")
    inst = Instance(id="I", modules={"m0": m0, "m1": m1})

    psi_m0_before = m0.state["psi"].copy()
    psi_m1_before = m1.state["psi"].copy()
    h_m0_before = m0.metric.h.copy()
    h_m1_before = m1.metric.h.copy()

    _ = inst.diagnostics()

    np.testing.assert_array_equal(m0.state["psi"], psi_m0_before)
    np.testing.assert_array_equal(m1.state["psi"], psi_m1_before)
    np.testing.assert_array_equal(m0.metric.h, h_m0_before)
    np.testing.assert_array_equal(m1.metric.h, h_m1_before)


# ===== R. Diagnostics content =====

def test_instance_diagnostics_content():
    m0 = make_module("m0")
    m1 = make_module("m1")
    inst = Instance(id="I", modules={"m0": m0, "m1": m1})
    d = inst.diagnostics()
    assert d["id"] == "I"
    assert d["n_modules"] == 2
    assert set(d["module_ids"]) == {"m0", "m1"}
    assert "module_diagnostics" in d
    assert set(d["module_diagnostics"].keys()) == {"m0", "m1"}
    assert "module_metric_shapes" in d
    assert d["module_metric_shapes"]["m0"] == (5, 5, 5)


# ===== Heterogeneous modules tolerated at Instance level =====

def test_instance_heterogeneous_shapes_tolerated():
    """Per audit point 8: Instance does NOT enforce inter-module shape
    uniformity. It is the job of CouplingScheduler later.
    """
    m_small = make_module("ms", shape=(5, 5, 5))
    m_large = make_module("ml", shape=(7, 7, 7))
    # Should not raise
    inst = Instance(id="I", modules={"ms": m_small, "ml": m_large})
    assert inst.n_modules() == 2
    d = inst.diagnostics()
    assert d["module_metric_shapes"]["ms"] == (5, 5, 5)
    assert d["module_metric_shapes"]["ml"] == (7, 7, 7)


def test_instance_heterogeneous_metric_types_tolerated():
    """Scalar metric and tensor metric modules can coexist in an Instance."""
    m_scalar = make_module("ms")
    # Tensor module
    I33 = np.eye(3, dtype=np.float64)
    H = np.broadcast_to(I33, (5, 5, 5, 3, 3))
    H = np.ascontiguousarray(H, dtype=np.float64)
    tm = TensorMetric(H)
    s_tensor = make_state(shape=(5, 5, 5))
    m_tensor = Module(id="mt", state=s_tensor, metric=tm)

    inst = Instance(id="I", modules={"ms": m_scalar, "mt": m_tensor})
    assert inst.n_modules() == 2


# ===== Convenience constructors =====

def test_instance_with_module_adds_or_replaces():
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})

    m1 = make_module("m1")
    inst2 = inst.with_module(m1)
    assert set(inst2.module_ids()) == {"m0", "m1"}
    # Original unchanged
    assert set(inst.module_ids()) == {"m0"}

    # Replacing same id
    m0_new = make_module("m0", psi_val=2.0)
    inst3 = inst.with_module(m0_new)
    assert inst3.get_module("m0") is m0_new


def test_instance_with_modules_replaces_collection():
    m0 = make_module("m0")
    inst = Instance(id="I", modules={"m0": m0})

    m1 = make_module("m1")
    m2 = make_module("m2")
    inst2 = inst.with_modules({"m1": m1, "m2": m2})
    assert set(inst2.module_ids()) == {"m1", "m2"}
    # Original unchanged
    assert set(inst.module_ids()) == {"m0"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
