# -*- coding: utf-8 -*-
"""Tests for mcq_v5.state — strict serialization layout.

Critère verrouillé:
    State flatten/unflatten is bijective for solver fields,
    shape-safe, dtype-aware, order-stable, field-isolated,
    partial-layout guarded, and static-field compatible.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.state import State, SolverLayout, FieldSlice


# ===== Helpers =====

def make_simple_state():
    return State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": 10.0 * np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )


def make_state_with_static():
    return State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": 10.0 * np.ones((5, 5, 5), dtype=np.float64),
            "underflow_mask": np.zeros((5, 5, 5), dtype=bool),
            "labels": np.arange(125, dtype=np.int32).reshape(5, 5, 5),
        },
        solver_fields=("psi", "h"),
    )


def make_state_with_tensor():
    rng = np.random.default_rng(42)
    H = rng.standard_normal((4, 4, 4, 3, 3)).astype(np.float64)
    return State(
        fields={
            "psi": rng.standard_normal((4, 4, 4)).astype(np.float64),
            "h": rng.standard_normal((4, 4, 4)).astype(np.float64),
            "H": np.ascontiguousarray(H),
        },
        solver_fields=("psi", "h", "H"),
    )


# ===== Construction =====

def test_simple_construction():
    s = make_simple_state()
    assert s.solver_fields == ("psi", "h")
    assert "psi" in s.fields
    assert "h" in s.fields
    assert s.fields["psi"].dtype == np.float64


def test_construction_with_static_fields():
    s = make_state_with_static()
    assert s.solver_fields == ("psi", "h")
    assert "underflow_mask" in s.fields
    assert "labels" in s.fields
    assert s.fields["underflow_mask"].dtype == np.bool_
    assert s.fields["labels"].dtype == np.int32


def test_solver_field_not_in_fields_raises():
    with pytest.raises(ValueError, match="solver_fields entry"):
        State(
            fields={"psi": np.ones(10, dtype=np.float64)},
            solver_fields=("psi", "h"),
        )


def test_solver_field_wrong_dtype_raises():
    """Critère 3: solver fields must be float64."""
    with pytest.raises(TypeError, match="float64"):
        State(
            fields={
                "psi": np.ones(10, dtype=np.float32),
            },
            solver_fields=("psi",),
        )


def test_solver_field_not_contiguous_raises():
    """Critère 3: solver fields must be C-contiguous."""
    base = np.ones((10, 10), dtype=np.float64)
    non_contig = base[:, ::2]  # strided slice
    assert not non_contig.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        State(
            fields={"psi": non_contig},
            solver_fields=("psi",),
        )


def test_static_field_can_have_any_dtype():
    """Critère 1: static fields free of dtype constraint."""
    s = State(
        fields={
            "psi": np.ones(10, dtype=np.float64),
            "mask": np.zeros(10, dtype=bool),
            "labels": np.zeros(10, dtype=np.int64),
        },
        solver_fields=("psi",),
    )
    assert s.fields["mask"].dtype == np.bool_
    assert s.fields["labels"].dtype == np.int64


def test_construction_defensive_copy():
    """State construction copies arrays (no aliasing with caller data)."""
    psi_external = np.ones((5, 5, 5), dtype=np.float64)
    s = State(fields={"psi": psi_external}, solver_fields=("psi",))
    s.fields["psi"][0, 0, 0] = 999.0
    # External not affected
    assert psi_external[0, 0, 0] == 1.0


# ===== Full round-trip (CRITÈRE PRINCIPAL) =====

def test_full_round_trip_simple():
    """Critère: bijective on solver fields."""
    s = make_simple_state()
    flat, layout = s.flatten_for_solver()

    assert flat.dtype == np.float64
    assert flat.shape == (125 * 2,)
    assert layout.is_partial == False
    assert layout.solver_fields == ("psi", "h")

    s2 = State.unflatten_from_solver(flat, layout)
    assert s2.solver_fields == s.solver_fields
    np.testing.assert_array_equal(s2["psi"], s["psi"])
    np.testing.assert_array_equal(s2["h"], s["h"])
    assert s2["psi"].dtype == np.float64
    assert s2["h"].dtype == np.float64


def test_full_round_trip_with_tensor_field():
    """Critère: shape-safe over rank-5 tensor."""
    s = make_state_with_tensor()
    flat, layout = s.flatten_for_solver()

    expected_size = 4**3 + 4**3 + 4**3 * 9  # psi + h + H(4,4,4,3,3)
    assert flat.shape == (expected_size,)

    s2 = State.unflatten_from_solver(flat, layout)
    np.testing.assert_array_equal(s2["psi"], s["psi"])
    np.testing.assert_array_equal(s2["h"], s["h"])
    np.testing.assert_array_equal(s2["H"], s["H"])
    assert s2["H"].shape == (4, 4, 4, 3, 3)


def test_round_trip_preserves_static_fields_with_base_state():
    """Critère: static-field compatible (via base_state)."""
    s = make_state_with_static()
    flat, layout = s.flatten_for_solver()
    s2 = State.unflatten_from_solver(flat, layout, base_state=s)
    # Solver fields reconstructed
    np.testing.assert_array_equal(s2["psi"], s["psi"])
    np.testing.assert_array_equal(s2["h"], s["h"])
    # Static fields preserved
    np.testing.assert_array_equal(s2["underflow_mask"], s["underflow_mask"])
    np.testing.assert_array_equal(s2["labels"], s["labels"])


def test_round_trip_without_base_state_drops_static():
    """Without base_state, static fields are not reconstructed."""
    s = make_state_with_static()
    flat, layout = s.flatten_for_solver()
    s2 = State.unflatten_from_solver(flat, layout)
    assert "psi" in s2.fields
    assert "h" in s2.fields
    assert "underflow_mask" not in s2.fields
    assert "labels" not in s2.fields


# ===== Anti-bavure =====

def test_anti_bleeding_modify_psi_only():
    """Critère: field-isolated. Modifier le flat sur la tranche psi
    ne doit pas affecter h."""
    s = make_simple_state()
    flat, layout = s.flatten_for_solver()
    # Modify psi slice
    psi_slice = layout.slice_for("psi")
    flat[psi_slice.start:psi_slice.end] = 42.0
    # Reconstruct
    s2 = State.unflatten_from_solver(flat, layout)
    assert np.all(s2["psi"] == 42.0)
    np.testing.assert_array_equal(s2["h"], s["h"])  # h unchanged


def test_anti_bleeding_modify_h_only():
    s = make_simple_state()
    flat, layout = s.flatten_for_solver()
    h_slice = layout.slice_for("h")
    flat[h_slice.start:h_slice.end] = -7.0
    s2 = State.unflatten_from_solver(flat, layout)
    assert np.all(s2["h"] == -7.0)
    np.testing.assert_array_equal(s2["psi"], s["psi"])  # psi unchanged


def test_anti_bleeding_tensor_field_isolated():
    s = make_state_with_tensor()
    flat, layout = s.flatten_for_solver()
    H_slice = layout.slice_for("H")
    flat[H_slice.start:H_slice.end] = 0.0
    s2 = State.unflatten_from_solver(flat, layout)
    assert np.all(s2["H"] == 0.0)
    np.testing.assert_array_equal(s2["psi"], s["psi"])
    np.testing.assert_array_equal(s2["h"], s["h"])


# ===== Memory independence =====

def test_unflatten_creates_independent_arrays():
    """Modifier s2['psi'] ne doit pas modifier s['psi']."""
    s = make_simple_state()
    flat, layout = s.flatten_for_solver()
    s2 = State.unflatten_from_solver(flat, layout)
    s2.fields["psi"][0, 0, 0] = 999.0
    # s unchanged
    assert s["psi"][0, 0, 0] == 1.0


def test_construction_breaks_aliasing():
    """Modifying caller's array after State construction does not affect State."""
    psi = np.ones((5, 5, 5), dtype=np.float64)
    s = State(fields={"psi": psi}, solver_fields=("psi",))
    psi[0, 0, 0] = 999.0
    assert s["psi"][0, 0, 0] == 1.0


# ===== Order stability =====

def test_dict_order_immunity():
    """Critère: order-stable. Same solver_fields, different insert order => same flat."""
    psi_data = np.ones((5, 5, 5), dtype=np.float64)
    h_data = 10.0 * np.ones((5, 5, 5), dtype=np.float64)
    s1 = State(
        fields={"psi": psi_data, "h": h_data},
        solver_fields=("psi", "h"),
    )
    s2 = State(
        fields={"h": h_data, "psi": psi_data},  # different insertion order
        solver_fields=("psi", "h"),
    )
    f1, l1 = s1.flatten_for_solver()
    f2, l2 = s2.flatten_for_solver()
    np.testing.assert_array_equal(f1, f2)
    assert l1.solver_fields == l2.solver_fields


def test_solver_fields_order_matters_in_flat():
    """Changing solver_fields order changes the flat layout."""
    psi_data = np.arange(5, dtype=np.float64)
    h_data = 100.0 + np.arange(5, dtype=np.float64)
    s_psi_first = State(
        fields={"psi": psi_data.copy(), "h": h_data.copy()},
        solver_fields=("psi", "h"),
    )
    s_h_first = State(
        fields={"psi": psi_data.copy(), "h": h_data.copy()},
        solver_fields=("h", "psi"),
    )
    f1, _ = s_psi_first.flatten_for_solver()
    f2, _ = s_h_first.flatten_for_solver()
    # First 5 elements differ
    np.testing.assert_array_equal(f1[:5], psi_data)
    np.testing.assert_array_equal(f2[:5], h_data)


# ===== Partial layout (CRITÈRE: partial-layout guarded) =====

def test_partial_flatten_marks_layout():
    s = make_simple_state()
    flat, layout = s.flatten_subset(("psi",))
    assert layout.is_partial == True
    assert flat.shape == (125,)
    # Layout contains only psi slice
    assert len(layout.field_slices) == 1
    assert layout.field_slices[0].name == "psi"


def test_partial_unflatten_without_base_raises():
    s = make_simple_state()
    flat, layout = s.flatten_subset(("psi",))
    with pytest.raises(ValueError, match="base_state"):
        State.unflatten_from_solver(flat, layout)


def test_partial_unflatten_without_allow_partial_raises():
    s = make_simple_state()
    flat, layout = s.flatten_subset(("psi",))
    with pytest.raises(ValueError, match="allow_partial"):
        State.unflatten_from_solver(flat, layout, base_state=s)


def test_partial_unflatten_with_base_and_allow():
    """Partial round-trip: psi changed, h preserved from base_state."""
    s = make_simple_state()
    flat, layout = s.flatten_subset(("psi",))
    flat[:] = 42.0  # change all psi values
    s2 = State.unflatten_from_solver(
        flat, layout, base_state=s, allow_partial=True
    )
    assert np.all(s2["psi"] == 42.0)
    np.testing.assert_array_equal(s2["h"], s["h"])  # h preserved


def test_partial_with_static_preserves_static():
    s = make_state_with_static()
    flat, layout = s.flatten_subset(("psi",))
    s2 = State.unflatten_from_solver(
        flat, layout, base_state=s, allow_partial=True
    )
    # psi reconstructed, h, mask, labels preserved
    np.testing.assert_array_equal(s2["psi"], s["psi"])
    np.testing.assert_array_equal(s2["h"], s["h"])
    np.testing.assert_array_equal(s2["underflow_mask"], s["underflow_mask"])
    np.testing.assert_array_equal(s2["labels"], s["labels"])


def test_flatten_subset_invalid_field_raises():
    s = make_simple_state()
    with pytest.raises(ValueError, match="not in solver_fields"):
        s.flatten_subset(("psi", "unknown_field"))


# ===== Layout validation =====

def test_layout_duplicate_names_raises():
    with pytest.raises(ValueError, match="duplicate"):
        SolverLayout(
            field_slices=(
                FieldSlice("psi", 0, 10, (10,), np.dtype(np.float64)),
                FieldSlice("psi", 10, 20, (10,), np.dtype(np.float64)),
            ),
            total_size=20,
            solver_fields=("psi",),
        )


def test_layout_gap_raises():
    with pytest.raises(ValueError, match="gap or overlap"):
        SolverLayout(
            field_slices=(
                FieldSlice("psi", 0, 10, (10,), np.dtype(np.float64)),
                FieldSlice("h", 15, 25, (10,), np.dtype(np.float64)),  # gap
            ),
            total_size=25,
            solver_fields=("psi", "h"),
        )


def test_layout_overlap_raises():
    with pytest.raises(ValueError, match="gap or overlap"):
        SolverLayout(
            field_slices=(
                FieldSlice("psi", 0, 15, (15,), np.dtype(np.float64)),
                FieldSlice("h", 10, 25, (15,), np.dtype(np.float64)),  # overlap
            ),
            total_size=25,
            solver_fields=("psi", "h"),
        )


def test_layout_size_mismatch_raises():
    with pytest.raises(ValueError, match="size mismatch"):
        SolverLayout(
            field_slices=(
                FieldSlice("psi", 0, 10, (5,), np.dtype(np.float64)),  # shape says 5, slice says 10
            ),
            total_size=10,
            solver_fields=("psi",),
        )


# ===== Unflatten validation =====

def test_unflatten_wrong_shape_raises():
    s = make_simple_state()
    _, layout = s.flatten_for_solver()
    bad_flat = np.zeros(layout.total_size + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        State.unflatten_from_solver(bad_flat, layout)


def test_unflatten_wrong_dtype_raises():
    s = make_simple_state()
    flat, layout = s.flatten_for_solver()
    bad_flat = flat.astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        State.unflatten_from_solver(bad_flat, layout)


# ===== Algebra =====

def test_scale():
    s = make_simple_state()
    s2 = s.scale(2.0)
    np.testing.assert_array_equal(s2["psi"], 2.0 * s["psi"])
    np.testing.assert_array_equal(s2["h"], 2.0 * s["h"])


def test_add():
    s = make_simple_state()
    delta = State(
        fields={
            "psi": 0.5 * np.ones((5, 5, 5), dtype=np.float64),
            "h": np.zeros((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    s2 = s.add(delta)
    np.testing.assert_array_equal(s2["psi"], 1.5 * np.ones((5, 5, 5)))
    np.testing.assert_array_equal(s2["h"], s["h"])


def test_subtract():
    s = make_simple_state()
    other = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    diff = s.subtract(other)
    np.testing.assert_array_equal(diff["psi"], np.zeros((5, 5, 5)))
    np.testing.assert_array_equal(diff["h"], 9.0 * np.ones((5, 5, 5)))


def test_norm_default_all_solver():
    s = make_simple_state()
    expected = float(np.sqrt(125 * 1 + 125 * 100))
    np.testing.assert_allclose(s.norm(), expected)


def test_norm_selected_fields():
    s = make_simple_state()
    n_psi = s.norm(["psi"])
    np.testing.assert_allclose(n_psi, float(np.sqrt(125.0)))


# ===== Diagnostics absence (metadata not serialized) =====

def test_metadata_not_in_flat():
    s = State(
        fields={"psi": np.ones(10, dtype=np.float64)},
        solver_fields=("psi",),
        metadata={"experiment": "X", "seed": 42},
    )
    flat, layout = s.flatten_for_solver()
    assert flat.shape == (10,)
    # Metadata preserved separately
    assert s.metadata == {"experiment": "X", "seed": 42}


# ===== Guard: layout/base_state compatibility (audit amendment) =====

def test_full_layout_incompatible_solver_fields_with_base():
    """A. Full layout with solver_fields=('psi','h') applied to base_state
    with solver_fields=('psi','H') must raise."""
    s_source = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": 10.0 * np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    flat, layout = s_source.flatten_for_solver()

    s_base_different = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "H": 5.0 * np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "H"),
    )

    with pytest.raises(ValueError, match="solver_fields"):
        State.unflatten_from_solver(flat, layout, base_state=s_base_different)


def test_full_layout_different_order_with_base():
    """Full layout with ('psi','h') applied to base with ('h','psi') must raise."""
    s_source = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": 10.0 * np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    flat, layout = s_source.flatten_for_solver()

    s_base_reordered = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": 10.0 * np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("h", "psi"),
    )

    with pytest.raises(ValueError, match="solver_fields"):
        State.unflatten_from_solver(flat, layout, base_state=s_base_reordered)


def test_partial_layout_incompatible_shape_with_base():
    """B. Partial layout from State A applied to State B with different psi shape."""
    s_source = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    flat, layout = s_source.flatten_subset(("psi",))

    s_base_different_shape = State(
        fields={
            "psi": np.ones((7, 7, 7), dtype=np.float64),  # different shape
            "h": np.ones((7, 7, 7), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )

    with pytest.raises(ValueError, match="shape"):
        State.unflatten_from_solver(
            flat, layout,
            base_state=s_base_different_shape,
            allow_partial=True,
        )


def test_partial_layout_field_not_in_base_solver_fields():
    """Partial layout referencing a field absent from base solver_fields."""
    s_source = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    flat, layout = s_source.flatten_subset(("h",))

    s_base_no_h = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "H": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "H"),  # no 'h'
    )

    with pytest.raises(ValueError, match="not in"):
        State.unflatten_from_solver(
            flat, layout,
            base_state=s_base_no_h,
            allow_partial=True,
        )


def test_full_layout_incompatible_shape_with_base():
    """Full layout with one field shape mismatching base_state."""
    s_source = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    flat, layout = s_source.flatten_for_solver()

    s_base_different = State(
        fields={
            "psi": np.ones((7, 7, 7), dtype=np.float64),
            "h": np.ones((7, 7, 7), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )

    with pytest.raises(ValueError, match="shape"):
        State.unflatten_from_solver(flat, layout, base_state=s_base_different)


# ===== Algebra preserves static fields (audit amendment) =====

def test_scale_preserves_static_fields():
    """D. scale() must not touch static fields (mask, labels, etc.)."""
    s = make_state_with_static()
    s2 = s.scale(2.0)
    # Solver fields scaled
    np.testing.assert_array_equal(s2["psi"], 2.0 * s["psi"])
    np.testing.assert_array_equal(s2["h"], 2.0 * s["h"])
    # Static fields preserved
    np.testing.assert_array_equal(s2["underflow_mask"], s["underflow_mask"])
    np.testing.assert_array_equal(s2["labels"], s["labels"])
    assert s2["underflow_mask"].dtype == np.bool_
    assert s2["labels"].dtype == np.int32


def test_add_preserves_static_fields():
    """add() must not touch static fields, kept from self."""
    s = make_state_with_static()
    delta = State(
        fields={
            "psi": 0.5 * np.ones((5, 5, 5), dtype=np.float64),
            "h": np.zeros((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    s2 = s.add(delta)
    np.testing.assert_array_equal(s2["psi"], 1.5 * np.ones((5, 5, 5)))
    np.testing.assert_array_equal(s2["h"], s["h"])
    np.testing.assert_array_equal(s2["underflow_mask"], s["underflow_mask"])
    np.testing.assert_array_equal(s2["labels"], s["labels"])


def test_subtract_preserves_static_fields():
    s = make_state_with_static()
    other = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": 5.0 * np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    diff = s.subtract(other)
    np.testing.assert_array_equal(diff["psi"], np.zeros((5, 5, 5)))
    np.testing.assert_array_equal(diff["h"], 5.0 * np.ones((5, 5, 5)))
    np.testing.assert_array_equal(diff["underflow_mask"], s["underflow_mask"])
    np.testing.assert_array_equal(diff["labels"], s["labels"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
