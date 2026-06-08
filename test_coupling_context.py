# -*- coding: utf-8 -*-
"""Tests for mcq_v5.coupling_context.CouplingContext.

Critère étape 8: immutable bounded pairwise context, no coupling logic,
no Instance/Module/Experiment visibility, prepares overlap modes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.grid import Grid, make_grid
from mcq_v5.state import State
from mcq_v5.metric import ScalarConformalMetric
from mcq_v5.views import StateView, MetricView, GridView, HistoryView, HistoryEntry
from mcq_v5.coupling_context import CouplingContext


# ===== Helpers =====

def make_views_pair(grid_shape=(5, 5, 5)):
    """Make target/source views + grid_view for tests."""
    s_t = State(
        fields={
            "psi": np.ones(grid_shape, dtype=np.float64),
            "h": np.ones(grid_shape, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    s_s = State(
        fields={
            "psi": 2.0 * np.ones(grid_shape, dtype=np.float64),
            "h": np.ones(grid_shape, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    m_t = ScalarConformalMetric(np.ones(grid_shape, dtype=np.float64))
    m_s = ScalarConformalMetric(np.ones(grid_shape, dtype=np.float64))
    g = make_grid(grid_shape, dx=1.0)
    return {
        "target_state_view": StateView(s_t),
        "source_state_view": StateView(s_s),
        "target_metric_view": MetricView(m_t),
        "source_metric_view": MetricView(m_s),
        "grid_view": GridView(g),
    }


def make_ctx_global(R=0.5, **overrides):
    views = make_views_pair()
    kwargs = dict(
        target_id="m0",
        source_id="m1",
        overlap_R=R,
        overlap_mode="global_scalar",
        time=0.0,
        **views,
    )
    kwargs.update(overrides)
    return CouplingContext(**kwargs)


# ===== A. Global scalar valid =====

def test_ctx_global_scalar_R_05():
    ctx = make_ctx_global(R=0.5)
    assert ctx.target_id == "m0"
    assert ctx.source_id == "m1"
    assert ctx.overlap_R == 0.5
    assert ctx.overlap_mode == "global_scalar"
    assert ctx.is_global_overlap()
    assert not ctx.is_local_overlap()


def test_ctx_global_scalar_R_0():
    """R=0 must be allowed (for invariance tests downstream)."""
    ctx = make_ctx_global(R=0.0)
    assert ctx.overlap_R == 0.0


def test_ctx_global_scalar_R_1():
    """R=1 must be allowed (for invariance tests downstream)."""
    ctx = make_ctx_global(R=1.0)
    assert ctx.overlap_R == 1.0


# ===== B. Local field valid =====

def test_ctx_local_field_valid():
    views = make_views_pair()
    R = 0.3 * np.ones((5, 5, 5), dtype=np.float64)
    ctx = CouplingContext(
        target_id="m0",
        source_id="m1",
        overlap_R=R,
        overlap_mode="local_field",
        time=0.0,
        **views,
    )
    assert ctx.is_local_overlap()
    assert ctx.overlap_shape() == (5, 5, 5)


# ===== C. Patchwise as container =====

def test_ctx_patchwise_accepted_as_container():
    views = make_views_pair()
    R_patch = np.array([0.2, 0.7], dtype=np.float64)
    ctx = CouplingContext(
        target_id="m0",
        source_id="m1",
        overlap_R=R_patch,
        overlap_mode="patchwise",
        time=0.0,
        **views,
    )
    assert ctx.is_patchwise_overlap()


# ===== D. Global R out of [0,1] rejected =====

def test_ctx_global_R_negative_rejected():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        make_ctx_global(R=-0.1)


def test_ctx_global_R_above_one_rejected():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        make_ctx_global(R=1.5)


# ===== E. Global R NaN/Inf rejected =====

def test_ctx_global_R_nan_rejected():
    with pytest.raises(ValueError, match="finite"):
        make_ctx_global(R=float("nan"))


def test_ctx_global_R_inf_rejected():
    with pytest.raises(ValueError, match="finite"):
        make_ctx_global(R=float("inf"))


# ===== F. Local field wrong dtype =====

def test_ctx_local_wrong_dtype_rejected():
    views = make_views_pair()
    R = 0.5 * np.ones((5, 5, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        CouplingContext(
            target_id="m0",
            source_id="m1",
            overlap_R=R,
            overlap_mode="local_field",
            time=0.0,
            **views,
        )


# ===== G. Local field non-contiguous =====

def test_ctx_local_non_contiguous_rejected():
    views = make_views_pair()
    big = np.ones((10, 5, 5), dtype=np.float64)
    R = 0.5 * big[::2]
    if R.flags["C_CONTIGUOUS"]:
        pytest.skip("slice happened to remain contiguous")
    with pytest.raises(ValueError, match="C-contiguous"):
        CouplingContext(
            target_id="m0",
            source_id="m1",
            overlap_R=R,
            overlap_mode="local_field",
            time=0.0,
            **views,
        )


# ===== H. Local field shape mismatch =====

def test_ctx_local_shape_mismatch_rejected():
    views = make_views_pair()  # grid 5x5x5
    R = 0.5 * np.ones((7, 7, 7), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        CouplingContext(
            target_id="m0",
            source_id="m1",
            overlap_R=R,
            overlap_mode="local_field",
            time=0.0,
            **views,
        )


# ===== I. Local field values out of [0,1] =====

def test_ctx_local_values_out_of_range_rejected():
    views = make_views_pair()
    R = np.ones((5, 5, 5), dtype=np.float64)
    R[0, 0, 0] = 1.5  # out of range
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        CouplingContext(
            target_id="m0",
            source_id="m1",
            overlap_R=R,
            overlap_mode="local_field",
            time=0.0,
            **views,
        )


def test_ctx_local_values_negative_rejected():
    views = make_views_pair()
    R = 0.5 * np.ones((5, 5, 5), dtype=np.float64)
    R[2, 2, 2] = -0.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        CouplingContext(
            target_id="m0",
            source_id="m1",
            overlap_R=R,
            overlap_mode="local_field",
            time=0.0,
            **views,
        )


def test_ctx_local_nan_rejected():
    views = make_views_pair()
    R = 0.5 * np.ones((5, 5, 5), dtype=np.float64)
    R[1, 1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        CouplingContext(
            target_id="m0",
            source_id="m1",
            overlap_R=R,
            overlap_mode="local_field",
            time=0.0,
            **views,
        )


# ===== J. Invalid overlap_mode =====

def test_ctx_invalid_overlap_mode_rejected():
    views = make_views_pair()
    with pytest.raises(ValueError, match="overlap_mode"):
        CouplingContext(
            target_id="m0",
            source_id="m1",
            overlap_R=0.5,
            overlap_mode="random_string",
            time=0.0,
            **views,
        )


# ===== K. target_id == source_id rejected =====

def test_ctx_self_pair_rejected():
    views = make_views_pair()
    with pytest.raises(ValueError, match="differ"):
        CouplingContext(
            target_id="m0",
            source_id="m0",
            overlap_R=0.5,
            overlap_mode="global_scalar",
            time=0.0,
            **views,
        )


# ===== L. Wrong view types rejected =====

def test_ctx_state_view_wrong_type():
    views = make_views_pair()
    views["target_state_view"] = "not a StateView"
    with pytest.raises(TypeError, match="StateView"):
        CouplingContext(
            target_id="m0", source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
            **views,
        )


def test_ctx_metric_view_wrong_type():
    views = make_views_pair()
    views["source_metric_view"] = "not a MetricView"
    with pytest.raises(TypeError, match="MetricView"):
        CouplingContext(
            target_id="m0", source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
            **views,
        )


def test_ctx_grid_view_wrong_type():
    views = make_views_pair()
    views["grid_view"] = "not a GridView"
    with pytest.raises(TypeError, match="GridView"):
        CouplingContext(
            target_id="m0", source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
            **views,
        )


def test_ctx_raw_state_rejected():
    """Cannot pass raw State instead of StateView."""
    views = make_views_pair()
    raw_state = State(
        fields={"psi": np.ones((5, 5, 5), dtype=np.float64)},
        solver_fields=("psi",),
    )
    views["target_state_view"] = raw_state  # wrong type
    with pytest.raises(TypeError, match="StateView"):
        CouplingContext(
            target_id="m0", source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
            **views,
        )


# ===== M. allowed_history wrong type =====

def test_ctx_allowed_history_wrong_type_rejected():
    views = make_views_pair()
    with pytest.raises(TypeError, match="HistoryView"):
        CouplingContext(
            target_id="m0", source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
            allowed_history="not a HistoryView",
            **views,
        )


def test_ctx_allowed_history_none_ok():
    ctx = make_ctx_global()
    assert ctx.allowed_history is None


def test_ctx_allowed_history_valid_historyview():
    views = make_views_pair()
    hist_list = [
        HistoryEntry(
            state=State(
                fields={"psi": np.ones((5, 5, 5), dtype=np.float64)},
                solver_fields=("psi",),
            ),
            time=0.0, step=0, committed=True,
        ),
    ]
    hv = HistoryView(hist_list)
    ctx = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
        allowed_history=hv,
        **views,
    )
    assert ctx.allowed_history is hv


# ===== N. Slots respected =====

def test_ctx_slots_no_adhoc_attribute():
    """frozen + slots: any attribute assignment raises (TypeError or
    AttributeError depending on Python internals)."""
    ctx = make_ctx_global()
    with pytest.raises((TypeError, AttributeError, Exception)):
        ctx.new_attr = "x"


# ===== O. Frozen respected =====

def test_ctx_frozen_target_id():
    ctx = make_ctx_global()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        ctx.target_id = "changed"


def test_ctx_frozen_overlap_R():
    ctx = make_ctx_global()
    with pytest.raises(Exception):
        ctx.overlap_R = 0.9


# ===== P. No forbidden attributes =====

def test_ctx_no_global_attributes():
    """Audit: no Instance, Module, ExperimentProtocol, etc."""
    ctx = make_ctx_global()
    forbidden = [
        "instance", "module", "experiment", "scheduler",
        "solver", "diagnostics_object", "_instance", "_experiment",
        "compute_coupling", "compute_overlap", "compute_novelty",
        "aggregate", "apply",
    ]
    for name in forbidden:
        assert not hasattr(ctx, name), f"forbidden attr present: {name}"


def test_ctx_dataclass_field_set_minimal():
    """All fields used by tests are present; no surprises."""
    ctx = make_ctx_global()
    expected = {
        "target_id", "source_id",
        "target_state_view", "source_state_view",
        "target_metric_view", "source_metric_view",
        "grid_view",
        "overlap_R", "overlap_mode", "time",
        "allowed_history", "metadata",
    }
    for f in expected:
        assert hasattr(ctx, f), f"missing expected field: {f}"


# ===== Q. Metadata defensive / read-only =====

def test_ctx_metadata_defensive_copy_at_construction():
    """External dict mutation after construction must not affect ctx."""
    views = make_views_pair()
    meta = {"k": 1, "extra": "info"}
    ctx = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
        metadata=meta,
        **views,
    )
    meta["k"] = 999
    meta["new"] = "value"
    assert ctx.metadata["k"] == 1
    assert "new" not in ctx.metadata


def test_ctx_metadata_read_only_mappingproxy():
    """ctx.metadata must reject modifications (MappingProxyType)."""
    ctx = make_ctx_global()
    with pytest.raises(TypeError):
        ctx.metadata["new_key"] = "x"


def test_ctx_metadata_default_empty():
    ctx = make_ctx_global()
    assert dict(ctx.metadata) == {}


def test_ctx_metadata_invalid_type_rejected():
    views = make_views_pair()
    with pytest.raises(TypeError, match="Mapping"):
        CouplingContext(
            target_id="m0", source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
            metadata="not a mapping",
            **views,
        )


# ===== R. Overlap mode helpers =====

def test_helpers_global():
    ctx = make_ctx_global()
    assert ctx.is_global_overlap() is True
    assert ctx.is_local_overlap() is False
    assert ctx.is_patchwise_overlap() is False
    assert ctx.overlap_shape() is None


def test_helpers_local():
    views = make_views_pair()
    R = 0.3 * np.ones((5, 5, 5), dtype=np.float64)
    ctx = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=R, overlap_mode="local_field", time=0.0,
        **views,
    )
    assert ctx.is_global_overlap() is False
    assert ctx.is_local_overlap() is True
    assert ctx.is_patchwise_overlap() is False
    assert ctx.overlap_shape() == (5, 5, 5)


# ===== id type validation =====

def test_ctx_invalid_target_id_type():
    views = make_views_pair()
    with pytest.raises(TypeError, match="target_id"):
        CouplingContext(
            target_id=[1, 2],
            source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar", time=0.0,
            **views,
        )


# ===== time validation =====

def test_ctx_time_nan_rejected():
    views = make_views_pair()
    with pytest.raises(ValueError, match="finite"):
        CouplingContext(
            target_id="m0", source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar",
            time=float("nan"),
            **views,
        )


def test_ctx_time_inf_rejected():
    views = make_views_pair()
    with pytest.raises(ValueError, match="finite"):
        CouplingContext(
            target_id="m0", source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar",
            time=float("inf"),
            **views,
        )


def test_ctx_time_invalid_type():
    views = make_views_pair()
    with pytest.raises(TypeError, match="time"):
        CouplingContext(
            target_id="m0", source_id="m1",
            overlap_R=0.5, overlap_mode="global_scalar",
            time="0.0",
            **views,
        )


def test_ctx_time_valid_float_and_int():
    """time can be int or float."""
    views = make_views_pair()
    ctx_int = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=0.5, overlap_mode="global_scalar", time=3,
        **views,
    )
    assert ctx_int.time == 3
    views = make_views_pair()
    ctx_f = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=0.5, overlap_mode="global_scalar", time=3.14,
        **views,
    )
    assert ctx_f.time == 3.14


# ===== Amendement: overlap_R immutability =====

def test_ctx_overlap_R_defensive_copy_array():
    """A. External ndarray mutation after construction must not affect ctx."""
    views = make_views_pair()
    R = 0.5 * np.ones((5, 5, 5), dtype=np.float64)
    ctx = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=R, overlap_mode="local_field", time=0.0,
        **views,
    )
    R[0, 0, 0] = 0.9
    assert ctx.overlap_R[0, 0, 0] == 0.5


def test_ctx_overlap_R_array_read_only():
    """B. Modifying ctx.overlap_R[...] raises ValueError (writeable=False)."""
    views = make_views_pair()
    R = 0.5 * np.ones((5, 5, 5), dtype=np.float64)
    ctx = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=R, overlap_mode="local_field", time=0.0,
        **views,
    )
    assert ctx.overlap_R.flags["WRITEABLE"] is False
    with pytest.raises(ValueError):
        ctx.overlap_R[0, 0, 0] = 0.9


def test_ctx_overlap_R_global_scalar_normalized_to_float():
    """C. global_scalar overlap_R normalized to native float."""
    views = make_views_pair()
    ctx = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=np.float64(0.5),
        overlap_mode="global_scalar", time=0.0,
        **views,
    )
    assert isinstance(ctx.overlap_R, float)
    assert ctx.overlap_R == 0.5


def test_ctx_overlap_R_patchwise_also_readonly():
    """patchwise overlap_R is also writeable=False (audit hint)."""
    views = make_views_pair()
    R = np.array([0.2, 0.7], dtype=np.float64)
    ctx = CouplingContext(
        target_id="m0", source_id="m1",
        overlap_R=R, overlap_mode="patchwise", time=0.0,
        **views,
    )
    assert ctx.overlap_R.flags["WRITEABLE"] is False
    # External mutation also has no effect
    R[0] = 0.99
    assert ctx.overlap_R[0] == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
