# -*- coding: utf-8 -*-
"""Tests for mcq_v5.metric.ScalarConformalMetric."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.metric import ScalarConformalMetric, EPS_TINY
from mcq_v5.views import MetricView


# ===== Construction =====

def test_scalar_construction_valid():
    h = np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h)
    assert m.shape == (5, 5, 5)
    assert m.dim == 3
    np.testing.assert_array_equal(m.h, h)


def test_scalar_construction_with_h0_scalar():
    h = np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h, h0=1.0)
    assert m.h0 == 1.0


def test_scalar_construction_with_h0_array_validated():
    """h0 ndarray must be float64, C-contiguous, same shape, > 0."""
    h = np.ones((5, 5, 5), dtype=np.float64)
    h0 = 0.8 * np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h, h0=h0)
    np.testing.assert_array_equal(m.h0, h0)


def test_scalar_h0_array_wrong_dtype_raises():
    h = np.ones((5, 5, 5), dtype=np.float64)
    h0 = np.ones((5, 5, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="h0 dtype"):
        ScalarConformalMetric(h, h0=h0)


def test_scalar_h0_array_non_contiguous_raises():
    h = np.ones((5, 5, 5), dtype=np.float64)
    big = np.ones((10, 5, 5), dtype=np.float64)
    h0 = big[::2]  # non-contiguous slice
    assert not h0.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="h0 must be C-contiguous"):
        ScalarConformalMetric(h, h0=h0)


def test_scalar_h0_array_shape_mismatch_raises():
    h = np.ones((5, 5, 5), dtype=np.float64)
    h0 = np.ones((7, 7, 7), dtype=np.float64)
    with pytest.raises(ValueError, match="h0 shape"):
        ScalarConformalMetric(h, h0=h0)


def test_scalar_h0_array_nonpositive_raises():
    h = np.ones((5, 5, 5), dtype=np.float64)
    h0 = np.ones((5, 5, 5), dtype=np.float64)
    h0[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="h0 must be strictly positive"):
        ScalarConformalMetric(h, h0=h0)


def test_scalar_h0_scalar_nonpositive_raises():
    h = np.ones((5, 5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="h0 scalar"):
        ScalarConformalMetric(h, h0=-1.0)


def test_scalar_dtype_float32_rejected():
    h = np.ones((5, 5, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        ScalarConformalMetric(h)


def test_scalar_non_contiguous_rejected():
    base = np.ones((10, 5, 5), dtype=np.float64)
    h_non_contig = base[::2]
    assert not h_non_contig.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        ScalarConformalMetric(h_non_contig)


def test_scalar_nonpositive_rejected():
    h = np.ones((5, 5, 5), dtype=np.float64)
    h[2, 2, 2] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        ScalarConformalMetric(h)


def test_scalar_negative_rejected():
    h = np.ones((5, 5, 5), dtype=np.float64)
    h[2, 2, 2] = -0.5
    with pytest.raises(ValueError, match="strictly positive"):
        ScalarConformalMetric(h)


def test_scalar_defensive_copy():
    """Modifying caller's h after construction does not affect metric."""
    h = np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h)
    h[0, 0, 0] = 999.0
    assert m.h[0, 0, 0] == 1.0


def test_scalar_h_min_nonpositive_raises():
    h = np.ones((5, 5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="h_min"):
        ScalarConformalMetric(h, h_min=0.0)


# ===== Face coefficients =====

def test_face_coefficients_constant_h():
    """h constant => faces constant equal to h."""
    h = 3.5 * np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h)
    for axis in range(3):
        fc = m.face_coefficients(axis)
        np.testing.assert_allclose(fc, 3.5)


def test_face_coefficients_shapes():
    """axis k => shape with k-th dim reduced by 1."""
    h = np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h)
    assert m.face_coefficients(0).shape == (4, 5, 5)
    assert m.face_coefficients(1).shape == (5, 4, 5)
    assert m.face_coefficients(2).shape == (5, 5, 4)


def test_face_coefficients_harmonic_mean_simple():
    """h_L=1, h_R=3 -> 2*1*3/(1+3) = 1.5."""
    h = np.ones((2, 1, 1), dtype=np.float64)
    h[0, 0, 0] = 1.0
    h[1, 0, 0] = 3.0
    m = ScalarConformalMetric(h)
    fc = m.face_coefficients(0)
    np.testing.assert_allclose(fc[0, 0, 0], 1.5)


def test_face_coefficients_harmonic_mean_2_2():
    """h_L=2, h_R=2 -> 2 (harmonic of equal = equal)."""
    h = 2.0 * np.ones((3, 1, 1), dtype=np.float64)
    m = ScalarConformalMetric(h)
    fc = m.face_coefficients(0)
    np.testing.assert_allclose(fc, 2.0)


def test_face_coefficients_invalid_axis_raises():
    h = np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h)
    with pytest.raises(ValueError, match="out of range"):
        m.face_coefficients(3)
    with pytest.raises(ValueError, match="out of range"):
        m.face_coefficients(-1)


def test_face_coefficients_axis_type_raises():
    h = np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h)
    with pytest.raises(TypeError):
        m.face_coefficients("x")


# ===== inverse / determinant / volume_element =====

def test_inverse_simple():
    h = 2.0 * np.ones((3, 3, 3), dtype=np.float64)
    m = ScalarConformalMetric(h)
    inv = m.inverse()
    np.testing.assert_allclose(inv, 0.5)


def test_determinant_placeholder_convention():
    """V5-0a placeholder: det = h ** dim."""
    h = 2.0 * np.ones((3, 3, 3), dtype=np.float64)
    m = ScalarConformalMetric(h)
    det = m.determinant()
    np.testing.assert_allclose(det, 8.0)  # 2 ** 3


def test_determinant_unknown_mode_raises():
    h = np.ones((3, 3, 3), dtype=np.float64)
    m = ScalarConformalMetric(h)
    with pytest.raises(NotImplementedError, match="determinant"):
        m.determinant(mode="riemannian")


def test_volume_element_placeholder():
    """V5-0a placeholder: vol = h ** (dim/2)."""
    h = 4.0 * np.ones((3, 3, 3), dtype=np.float64)
    m = ScalarConformalMetric(h)
    vol = m.volume_element()
    np.testing.assert_allclose(vol, 8.0)  # 4 ** (3/2) = 8


# ===== Distance =====

def test_distance_metric_l2_zero():
    h = np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h)
    a = np.zeros((5, 5, 5), dtype=np.float64)
    d = m.distance(a, a)
    assert d == 0.0


def test_distance_metric_l2_simple():
    h = np.ones((3,), dtype=np.float64)
    m = ScalarConformalMetric(h)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])
    d = m.distance(a, b)
    np.testing.assert_allclose(d, 1.0)


def test_distance_metric_l2_weighted_by_h():
    """sqrt(sum(h * (a-b)^2)) — h amplifies."""
    h = 4.0 * np.ones((3,), dtype=np.float64)
    m = ScalarConformalMetric(h)
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0])
    d = m.distance(a, b)
    np.testing.assert_allclose(d, 2.0)  # sqrt(4 * 1)


def test_distance_unknown_mode_raises():
    h = np.ones((3,), dtype=np.float64)
    m = ScalarConformalMetric(h)
    a = np.zeros(3)
    with pytest.raises(NotImplementedError):
        m.distance(a, a, mode="geodesic")


def test_distance_shape_mismatch_raises():
    h = np.ones((3,), dtype=np.float64)
    m = ScalarConformalMetric(h)
    a = np.zeros(3)
    b = np.zeros(5)
    with pytest.raises(ValueError, match="shape mismatch"):
        m.distance(a, b)


def test_distance_field_shape_mismatch_metric_raises():
    h = np.ones((3,), dtype=np.float64)
    m = ScalarConformalMetric(h)
    a = np.zeros(5)
    b = np.zeros(5)
    with pytest.raises(ValueError, match="does not match metric shape"):
        m.distance(a, b)


# ===== update / diagnostics =====

def test_update_raises_not_implemented():
    h = np.ones((3, 3, 3), dtype=np.float64)
    m = ScalarConformalMetric(h)
    with pytest.raises(NotImplementedError, match="V5-0a"):
        m.update()


def test_diagnostics_basic():
    h = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = ScalarConformalMetric(h)
    d = m.diagnostics()
    assert d["min_h"] == 1.0
    assert d["max_h"] == 4.0
    assert d["mean_h"] == 2.5
    assert d["shape"] == (2, 2)
    assert d["dim"] == 2
    assert d["strictly_positive"] is True
    assert d["n_below_h_min"] == 0


def test_diagnostics_below_h_min():
    h = np.array([1.0, 2.0, 3.0])
    m = ScalarConformalMetric(h, h_min=2.5)
    d = m.diagnostics()
    assert d["n_below_h_min"] == 2  # 1.0 and 2.0 below 2.5


# ===== view() integration =====

def test_view_returns_metric_view():
    h = np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h)
    mv = m.view()
    assert isinstance(mv, MetricView)
    coeffs = mv.coefficients()
    np.testing.assert_array_equal(coeffs, h)


def test_view_zero_copy():
    """Modifying via view modifies metric internals (proves zero-copy)."""
    h = np.ones((5, 5, 5), dtype=np.float64)
    m = ScalarConformalMetric(h)
    mv = m.view()
    coeffs = mv.coefficients()
    coeffs[0, 0, 0] = 99.0
    assert m.h[0, 0, 0] == 99.0


# ===== EPS_TINY =====

def test_eps_tiny_is_machine_minimum():
    """EPS_TINY must be np.finfo(np.float64).tiny, not arbitrary."""
    assert EPS_TINY == np.finfo(np.float64).tiny


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
