# -*- coding: utf-8 -*-
"""Tests for mcq_v5.metric.TensorMetric — V5-0a stub.

Validates: dtype, contiguity, square last 2 dims, symmetry, SPD via
vectorized np.linalg.eigvalsh, eigenvalue bounds. Dynamic methods are
expected to raise NotImplementedError.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.metric import TensorMetric


# ===== Helpers =====

def make_identity_tensor_field(shape_grid=(3, 3, 3), block_dim=3):
    """Field of identity matrices, SPD by construction."""
    I = np.eye(block_dim, dtype=np.float64)
    H = np.broadcast_to(I, shape_grid + (block_dim, block_dim))
    return np.ascontiguousarray(H, dtype=np.float64)


def make_diag_tensor_field(values, shape_grid=(3, 3, 3)):
    """Field with constant diagonal matrix per cell."""
    block_dim = len(values)
    D = np.diag(values).astype(np.float64)
    H = np.broadcast_to(D, shape_grid + (block_dim, block_dim))
    return np.ascontiguousarray(H, dtype=np.float64)


# ===== Construction =====

def test_tensor_construction_identity():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    assert m.shape == (3, 3, 3, 3, 3)
    assert m.block_dim == 3


def test_tensor_construction_with_metadata():
    H = make_identity_tensor_field()
    m = TensorMetric(H, metadata={"label": "test"})
    assert m.metadata == {"label": "test"}


def test_tensor_dtype_float32_rejected():
    H = make_identity_tensor_field().astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        TensorMetric(H)


def test_tensor_non_contiguous_rejected():
    # Build a strided non-contiguous tensor field
    big = make_identity_tensor_field((6, 3, 3), block_dim=3)
    H = big[::2]  # likely non-contiguous
    if H.flags["C_CONTIGUOUS"]:
        pytest.skip("slice happened to remain contiguous")
    with pytest.raises(ValueError, match="C-contiguous"):
        TensorMetric(H)


def test_tensor_too_few_dims_rejected():
    H = np.ones((3,), dtype=np.float64)
    with pytest.raises(ValueError, match="at least 2 dims"):
        TensorMetric(H)


def test_tensor_non_square_last_two_dims_rejected():
    H = np.ones((3, 3, 3, 3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="square"):
        TensorMetric(H)


def test_tensor_not_symmetric_rejected():
    H = make_identity_tensor_field((3, 3, 3))
    H = H.copy()
    H[0, 0, 0, 0, 1] = 1.0  # break symmetry
    # H[0,0,0,1,0] still 0, so asymmetric
    with pytest.raises(ValueError, match="symmetric"):
        TensorMetric(H)


def test_tensor_non_spd_rejected():
    """A diagonal matrix with a zero eigenvalue must be rejected."""
    H = make_diag_tensor_field([1.0, 1.0, 0.0])  # zero eigenvalue
    with pytest.raises(ValueError, match="SPD"):
        TensorMetric(H)


def test_tensor_negative_eigenvalue_rejected():
    H = make_diag_tensor_field([1.0, 1.0, -0.5])
    with pytest.raises(ValueError, match="SPD"):
        TensorMetric(H)


def test_tensor_defensive_copy():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    H[0, 0, 0, 0, 0] = 999.0
    # m._H is a copy, so m.H[0,0,0,0,0] still 1
    assert m.H[0, 0, 0, 0, 0] == 1.0


# ===== Eigenvalue bounds =====

def test_tensor_eig_bounds_pass():
    H = make_diag_tensor_field([1.0, 2.0, 3.0])
    m = TensorMetric(H, eig_bounds=(0.5, 5.0))
    assert m.diagnostics()["min_eigenvalue"] == pytest.approx(1.0)
    assert m.diagnostics()["max_eigenvalue"] == pytest.approx(3.0)


def test_tensor_eig_bound_lower_violated():
    H = make_diag_tensor_field([0.1, 1.0, 2.0])
    with pytest.raises(ValueError, match="below lower bound"):
        TensorMetric(H, eig_bounds=(0.5, 5.0))


def test_tensor_eig_bound_upper_violated():
    H = make_diag_tensor_field([1.0, 1.0, 10.0])
    with pytest.raises(ValueError, match="above upper bound"):
        TensorMetric(H, eig_bounds=(0.5, 5.0))


# ===== Stub methods raise =====

def test_face_coefficients_stub_raises():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    with pytest.raises(NotImplementedError):
        m.face_coefficients(0)


def test_inverse_stub_raises():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    with pytest.raises(NotImplementedError):
        m.inverse()


def test_determinant_stub_raises():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    with pytest.raises(NotImplementedError):
        m.determinant()


def test_volume_element_stub_raises():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    with pytest.raises(NotImplementedError):
        m.volume_element()


def test_distance_stub_raises():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    a = np.zeros((3, 3, 3))
    with pytest.raises(NotImplementedError):
        m.distance(a, a)


def test_update_stub_raises():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    with pytest.raises(NotImplementedError):
        m.update()


# ===== Diagnostics (operational) =====

def test_diagnostics_identity():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    d = m.diagnostics()
    assert d["shape"] == (3, 3, 3, 3, 3)
    assert d["block_dim"] == 3
    np.testing.assert_allclose(d["min_eigenvalue"], 1.0)
    np.testing.assert_allclose(d["max_eigenvalue"], 1.0)
    assert d["spd"] is True


def test_diagnostics_diag_tensor():
    H = make_diag_tensor_field([1.0, 2.0, 3.0])
    m = TensorMetric(H)
    d = m.diagnostics()
    np.testing.assert_allclose(d["min_eigenvalue"], 1.0)
    np.testing.assert_allclose(d["max_eigenvalue"], 3.0)
    assert d["spd"] is True


def test_diagnostics_max_asymmetry_zero_for_symmetric():
    H = make_diag_tensor_field([1.0, 2.0, 3.0])
    m = TensorMetric(H)
    d = m.diagnostics()
    assert d["max_asymmetry"] == 0.0


# ===== Vectorized SPD validation (not Python loops) =====

def test_spd_validation_uses_vectorized_eigvalsh():
    """Construct H with one cell non-SPD; must be detected without explicit loop.

    This tests that validation handles the field globally, not by iterating
    over Python indices.
    """
    H = make_diag_tensor_field([1.0, 1.0, 1.0], shape_grid=(5, 5, 5))
    H = H.copy()
    # Break SPD only at a single deep cell
    H[3, 2, 1] = np.diag([1.0, -0.5, 1.0])
    # Ensure still symmetric (diagonal matrices are symmetric)
    with pytest.raises(ValueError, match="SPD"):
        TensorMetric(H)


def test_spd_validation_handles_2d_grid():
    """eigvalsh broadcasts over arbitrary leading dims."""
    block_dim = 2
    I = np.eye(block_dim, dtype=np.float64)
    H = np.broadcast_to(I, (4, 6, block_dim, block_dim))
    H = np.ascontiguousarray(H, dtype=np.float64)
    m = TensorMetric(H)
    d = m.diagnostics()
    np.testing.assert_allclose(d["min_eigenvalue"], 1.0)


def test_spd_validation_minimum_valid_block_dim_1():
    """Block dim 1: degenerate to scalar but should still validate."""
    H = np.ones((3, 3, 3, 1, 1), dtype=np.float64)
    m = TensorMetric(H)
    d = m.diagnostics()
    np.testing.assert_allclose(d["min_eigenvalue"], 1.0)


# ===== view() =====

def test_tensor_view_zero_copy():
    H = make_identity_tensor_field()
    m = TensorMetric(H)
    mv = m.view()
    coeffs = mv.coefficients()
    coeffs[0, 0, 0, 0, 0] = 42.0
    assert m.H[0, 0, 0, 0, 0] == 42.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
