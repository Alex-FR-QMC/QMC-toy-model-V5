# -*- coding: utf-8 -*-
"""
V5-0b §3.A — Geometry compatibility harness.

Strict minimal compatibility audit. Verifies that
mcq_v5.GeometryOperator reproduces the 6d-compatible scalar conformal
finite-volume RHS within machine precision.

The reference 6d RHS function (`reference_6d_rhs_scalar_conformal_1d`
and `reference_6d_rhs_scalar_conformal_3d`) is coded LOCALLY in this
test file, using only NumPy primitives, and is INDEPENDENT of mcq_v5.*.
This isolation is required by the V5-0b protocol: any reuse of mcq_v5
code in the reference would defeat the audit.

Convention (from V5-0a, must match):
    h_face = harmonic_mean(h_left, h_right) = 2*h_l*h_r/(h_l+h_r)
    grad   = (psi_right - psi_left) / dx
    J      = -h_face * grad
    RHS    = -div(J)   (zero-flux Neumann external boundaries)

Tolerance: atol=1e-12, rtol=0.

Verdict: GEOMETRY_COMPAT_PASS strict if all cases pass; otherwise
inscribe per-cell deltas for audit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.grid import make_grid
from mcq_v5.state import State
from mcq_v5.metric import ScalarConformalMetric
from mcq_v5.geometry import GeometryOperator


# =========================================================================
# Reference 6d RHS — coded FROM SCRATCH, INDEPENDENT of mcq_v5
# =========================================================================

def _harmonic_mean_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise 2ab/(a+b). Assumes a,b > 0."""
    return 2.0 * a * b / (a + b)


def reference_6d_rhs_scalar_conformal_1d(
    psi: np.ndarray,
    h: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Reference RHS for 1D scalar conformal FV operator.

    Pure NumPy, no mcq_v5 import. Reproduces:
        h_face[k+1/2] = harmonic_mean(h[k], h[k+1])
        grad[k+1/2]   = (psi[k+1] - psi[k]) / dx
        J[k+1/2]      = -h_face * grad
        RHS[0]        = -J[0] / dx
        RHS[i]        = (J[i-1] - J[i]) / dx  (interior)
        RHS[-1]       =  J[-1] / dx
    """
    assert psi.ndim == 1 and h.ndim == 1 and psi.shape == h.shape
    N = psi.shape[0]
    h_face = _harmonic_mean_pair(h[:-1], h[1:])   # length N-1
    grad = (psi[1:] - psi[:-1]) / dx              # length N-1
    J = -h_face * grad                            # length N-1
    rhs = np.zeros(N, dtype=np.float64)
    # Interior cells: (J[i-1] - J[i]) / dx for i in [1..N-2]
    rhs[1:-1] = (J[:-1] - J[1:]) / dx
    # Boundaries (Neumann zero-flux: external faces have J=0 implicitly)
    rhs[0] = -J[0] / dx
    rhs[-1] = J[-1] / dx
    return rhs


def reference_6d_rhs_scalar_conformal_3d(
    psi: np.ndarray,
    h: np.ndarray,
    dx: tuple,
) -> np.ndarray:
    """Reference RHS for 3D scalar conformal FV operator.

    Pure NumPy, no mcq_v5 import. Applies the 1D contribution per axis
    and sums (axis-additivity is mathematical, mirrors the divergence
    structure of the operator).
    """
    assert psi.ndim == 3 and h.ndim == 3 and psi.shape == h.shape
    dx0, dx1, dx2 = dx
    rhs = np.zeros_like(psi, dtype=np.float64)

    # Axis 0
    h_face = _harmonic_mean_pair(h[:-1, :, :], h[1:, :, :])
    grad = (psi[1:, :, :] - psi[:-1, :, :]) / dx0
    J = -h_face * grad  # shape (N0-1, N1, N2)
    contrib = np.zeros_like(psi)
    contrib[1:-1, :, :] = (J[:-1, :, :] - J[1:, :, :]) / dx0
    contrib[0, :, :] = -J[0, :, :] / dx0
    contrib[-1, :, :] = J[-1, :, :] / dx0
    rhs += contrib

    # Axis 1
    h_face = _harmonic_mean_pair(h[:, :-1, :], h[:, 1:, :])
    grad = (psi[:, 1:, :] - psi[:, :-1, :]) / dx1
    J = -h_face * grad  # shape (N0, N1-1, N2)
    contrib = np.zeros_like(psi)
    contrib[:, 1:-1, :] = (J[:, :-1, :] - J[:, 1:, :]) / dx1
    contrib[:, 0, :] = -J[:, 0, :] / dx1
    contrib[:, -1, :] = J[:, -1, :] / dx1
    rhs += contrib

    # Axis 2
    h_face = _harmonic_mean_pair(h[:, :, :-1], h[:, :, 1:])
    grad = (psi[:, :, 1:] - psi[:, :, :-1]) / dx2
    J = -h_face * grad  # shape (N0, N1, N2-1)
    contrib = np.zeros_like(psi)
    contrib[:, :, 1:-1] = (J[:, :, :-1] - J[:, :, 1:]) / dx2
    contrib[:, :, 0] = -J[:, :, 0] / dx2
    contrib[:, :, -1] = J[:, :, -1] / dx2
    rhs += contrib

    return rhs


def _v5_rhs(psi: np.ndarray, h: np.ndarray, grid) -> np.ndarray:
    """Apply mcq_v5 GeometryOperator to (psi, h) and return RHS array."""
    op = GeometryOperator(grid)
    state = State(
        fields={
            "psi": np.ascontiguousarray(psi, dtype=np.float64),
            "h": np.ascontiguousarray(h, dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    metric = ScalarConformalMetric(np.ascontiguousarray(h, dtype=np.float64))
    dstate = op.apply(state, metric, field_name="psi")
    return dstate["psi"]


# =========================================================================
# §3.A.1 — Constant field
# =========================================================================

def test_geom_compat_A1_constant_field_1d():
    """A.1 1D — constant psi => RHS = 0 partout."""
    psi = 3.5 * np.ones(7, dtype=np.float64)
    h = np.ones(7, dtype=np.float64)
    grid = make_grid((7,), dx=1.0)
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_1d(psi, h, dx=1.0)
    np.testing.assert_allclose(v5_rhs, np.zeros(7), atol=1e-12, rtol=0)
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)


def test_geom_compat_A1_constant_field_3d():
    psi = 2.0 * np.ones((5, 5, 5), dtype=np.float64)
    h = np.ones((5, 5, 5), dtype=np.float64)
    grid = make_grid((5, 5, 5), dx=1.0)
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_3d(psi, h, dx=(1.0, 1.0, 1.0))
    np.testing.assert_allclose(v5_rhs, np.zeros((5, 5, 5)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)


# =========================================================================
# §3.A.2 — Centered pic
# =========================================================================

def test_geom_compat_A2_pic_central_1d():
    """A.2 1D — psi=[0,1,0], h=1, dx=1 => RHS = [1, -2, 1] (locked)."""
    psi = np.array([0.0, 1.0, 0.0])
    h = np.array([1.0, 1.0, 1.0])
    grid = make_grid((3,), dx=1.0)
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_1d(psi, h, dx=1.0)
    np.testing.assert_allclose(v5_rhs, [1.0, -2.0, 1.0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)


def test_geom_compat_A2_pic_central_3d():
    """3D centered pic [2,2,2]=1, rest 0, h=1, dx=1."""
    psi = np.zeros((5, 5, 5), dtype=np.float64)
    psi[2, 2, 2] = 1.0
    h = np.ones((5, 5, 5), dtype=np.float64)
    grid = make_grid((5, 5, 5), dx=1.0)
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_3d(psi, h, dx=(1.0, 1.0, 1.0))
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)
    # Specific: central cell receives -6, six neighbours receive +1 each
    assert v5_rhs[2, 2, 2] == pytest.approx(-6.0)
    for offset in [(1, 2, 2), (3, 2, 2), (2, 1, 2),
                   (2, 3, 2), (2, 2, 1), (2, 2, 3)]:
        assert v5_rhs[offset] == pytest.approx(1.0)


# =========================================================================
# §3.A.3 — Linear field under Neumann zero-flux
# =========================================================================

def test_geom_compat_A3_linear_1d_neumann_explicit():
    """A.3 1D corrigé — psi=[0,1,2,3,4], h=1, dx=1:
       grad=[1,1,1,1], J=[-1,-1,-1,-1],
       RHS = [1, 0, 0, 0, -1]   (NOT zero everywhere under Neumann).
    """
    psi = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    h = np.ones(5, dtype=np.float64)
    grid = make_grid((5,), dx=1.0)
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_1d(psi, h, dx=1.0)
    expected = np.array([1.0, 0.0, 0.0, 0.0, -1.0])
    np.testing.assert_allclose(v5_rhs, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)
    # Conservation
    assert abs(v5_rhs.sum()) < 1e-12


def test_geom_compat_A3_linear_3d_axis0():
    """3D linear along axis 0: interior RHS = 0, boundary contributions
    opposite, global sum = 0 along each (j,k) line."""
    psi = np.zeros((5, 3, 3), dtype=np.float64)
    for i in range(5):
        psi[i, :, :] = float(i)  # linear along axis 0
    h = np.ones((5, 3, 3), dtype=np.float64)
    grid = make_grid((5, 3, 3), dx=1.0)
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_3d(psi, h, dx=(1.0, 1.0, 1.0))
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)
    # Each (j,k) line: contributions only at i=0 and i=N-1
    for j in range(3):
        for k in range(3):
            line = v5_rhs[:, j, k]
            assert abs(line[1:-1]).max() < 1e-12  # interior zero
            assert line[0] == pytest.approx(1.0)
            assert line[-1] == pytest.approx(-1.0)
            assert abs(line.sum()) < 1e-12


# =========================================================================
# §3.A.4 — Random deterministic (3D, seeded)
# =========================================================================

def test_geom_compat_A4_random_deterministic_3d():
    """A.4 — random ψ on 3D grid, deterministic seed, h variable."""
    rng = np.random.default_rng(42)
    shape = (5, 5, 5)
    psi = rng.standard_normal(shape).astype(np.float64)
    psi = np.ascontiguousarray(psi)
    h = 1.0 + 0.5 * rng.uniform(size=shape).astype(np.float64)
    h = np.ascontiguousarray(h)
    dx = (1.0, 1.0, 1.0)
    grid = make_grid(shape, dx=dx[0])
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_3d(psi, h, dx=dx)
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)


def test_geom_compat_A4_random_anisotropic_3d():
    """Same but with anisotropic dx."""
    rng = np.random.default_rng(7)
    shape = (5, 5, 5)
    psi = np.ascontiguousarray(rng.standard_normal(shape).astype(np.float64))
    h = np.ascontiguousarray(
        1.0 + 0.3 * rng.uniform(size=shape).astype(np.float64)
    )
    dx = (1.0, 2.0, 0.5)
    grid = make_grid(shape, dx=dx)
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_3d(psi, h, dx=dx)
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)


# =========================================================================
# §3.A.5 — Variable h, deterministic
# =========================================================================

def test_geom_compat_A5_h_variable_1d():
    """h variable along axis: verify harmonic face coefficients."""
    psi = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    h = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    grid = make_grid((5,), dx=1.0)
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_1d(psi, h, dx=1.0)
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)


def test_geom_compat_A5_h_variable_3d():
    rng = np.random.default_rng(123)
    shape = (5, 5, 5)
    psi = np.ascontiguousarray(rng.standard_normal(shape).astype(np.float64))
    # Strongly variable h
    h = np.ascontiguousarray(
        0.5 + 5.0 * rng.uniform(size=shape).astype(np.float64)
    )
    grid = make_grid(shape, dx=1.0)
    v5_rhs = _v5_rhs(psi, h, grid)
    ref_rhs = reference_6d_rhs_scalar_conformal_3d(psi, h, dx=(1.0, 1.0, 1.0))
    np.testing.assert_allclose(v5_rhs, ref_rhs, atol=1e-12, rtol=0)


# =========================================================================
# §3.A.6 — Global conservation
# =========================================================================

def test_geom_compat_A6_conservation_constant():
    psi = 3.5 * np.ones((5, 5, 5), dtype=np.float64)
    h = np.ones((5, 5, 5), dtype=np.float64)
    grid = make_grid((5, 5, 5), dx=1.0)
    rhs = _v5_rhs(psi, h, grid)
    assert abs(rhs.sum()) < 1e-12


def test_geom_compat_A6_conservation_pic():
    psi = np.zeros((5, 5, 5), dtype=np.float64)
    psi[2, 2, 2] = 1.0
    h = np.ones((5, 5, 5), dtype=np.float64)
    grid = make_grid((5, 5, 5), dx=1.0)
    rhs = _v5_rhs(psi, h, grid)
    assert abs(rhs.sum()) < 1e-12


def test_geom_compat_A6_conservation_random():
    rng = np.random.default_rng(99)
    shape = (5, 5, 5)
    psi = np.ascontiguousarray(rng.standard_normal(shape).astype(np.float64))
    h = np.ascontiguousarray(
        1.0 + 0.5 * rng.uniform(size=shape).astype(np.float64)
    )
    grid = make_grid(shape, dx=1.0)
    rhs = _v5_rhs(psi, h, grid)
    assert abs(rhs.sum()) < 1e-12


def test_geom_compat_A6_conservation_linear_field():
    """Linear field already tested case-wise; verify global conservation."""
    psi = np.zeros((5, 5, 5), dtype=np.float64)
    for i in range(5):
        psi[i, :, :] = float(i)
    h = np.ones((5, 5, 5), dtype=np.float64)
    grid = make_grid((5, 5, 5), dx=1.0)
    rhs = _v5_rhs(psi, h, grid)
    assert abs(rhs.sum()) < 1e-12


# =========================================================================
# Verdict aggregation
# =========================================================================

def test_geometry_compatibility_verdict_PASS():
    """If this test passes, GEOMETRY_COMPAT_PASS is inscribed.

    Aggregates the canonical cases on 3D 5x5x5 grid with h=1, dx=1.
    No external reporting; this is a structural confirmation only.
    """
    grid = make_grid((5, 5, 5), dx=1.0)
    h = np.ones((5, 5, 5), dtype=np.float64)
    cases = {
        "constant": 2.0 * np.ones((5, 5, 5), dtype=np.float64),
    }
    psi_pic = np.zeros((5, 5, 5), dtype=np.float64)
    psi_pic[2, 2, 2] = 1.0
    cases["pic_central"] = psi_pic
    rng = np.random.default_rng(2026)
    cases["random"] = np.ascontiguousarray(
        rng.standard_normal((5, 5, 5)).astype(np.float64)
    )

    failures = []
    for name, psi in cases.items():
        v5_rhs = _v5_rhs(psi, h, grid)
        ref_rhs = reference_6d_rhs_scalar_conformal_3d(
            psi, h, dx=(1.0, 1.0, 1.0)
        )
        delta = np.abs(v5_rhs - ref_rhs).max()
        if delta > 1e-12:
            failures.append(f"{name}: max_delta={delta:.3e}")
    assert failures == [], (
        f"GEOMETRY_COMPAT_FAIL on: {failures}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
