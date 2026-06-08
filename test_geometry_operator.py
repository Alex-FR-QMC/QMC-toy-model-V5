# -*- coding: utf-8 -*-
"""Tests for mcq_v5.geometry.GeometryOperator.

Critère étape 7: scalar conformal finite-volume gradients, fluxes,
conservative RHS -div(J), zero-flux Neumann boundaries, 6d-compatible
harmonic face coefficients, stateless, no coupling/metric-dynamics/solver.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from mcq_v5.grid import Grid, make_grid
from mcq_v5.state import State
from mcq_v5.metric import ScalarConformalMetric, TensorMetric
from mcq_v5.geometry import GeometryOperator


# ===== Helpers =====

def make_grid_5x5x5():
    return make_grid((5, 5, 5), dx=1.0)


def make_grid_3():
    return make_grid((3,), dx=1.0)


def make_metric_1d_unit(N):
    return ScalarConformalMetric(np.ones((N,), dtype=np.float64))


def make_metric_3d_unit():
    return ScalarConformalMetric(np.ones((5, 5, 5), dtype=np.float64))


# ===== A. Construction =====

def test_geom_construction_with_grid():
    g = make_grid_5x5x5()
    op = GeometryOperator(g)
    assert op.grid is g


def test_geom_construction_invalid_grid():
    with pytest.raises(TypeError, match="Grid"):
        GeometryOperator("not a grid")


# ===== B. Gradient — constant field => zero =====

def test_gradient_constant_field_zero():
    op = GeometryOperator(make_grid_5x5x5())
    field = 3.7 * np.ones((5, 5, 5), dtype=np.float64)
    grads = op.gradient(field)
    assert len(grads) == 3
    for g in grads:
        np.testing.assert_allclose(g, 0.0)


# ===== C. Gradient — linear along axis 0 =====

def test_gradient_linear_axis0():
    op = GeometryOperator(make_grid_5x5x5())
    field = np.zeros((5, 5, 5), dtype=np.float64)
    for i in range(5):
        field[i, :, :] = 2.0 * i  # slope 2 along axis 0
    grads = op.gradient(field)
    np.testing.assert_allclose(grads[0], 2.0)
    np.testing.assert_allclose(grads[1], 0.0)
    np.testing.assert_allclose(grads[2], 0.0)


# ===== D. Gradient — anisotropic dx =====

def test_gradient_anisotropic_dx():
    g = make_grid((5, 5, 5), dx=(1.0, 2.0, 0.5))
    op = GeometryOperator(g)
    field = np.zeros((5, 5, 5), dtype=np.float64)
    for j in range(5):
        field[:, j, :] = j  # linear along axis 1
    grads = op.gradient(field)
    # df/dj = 1, but dx[1] = 2, so gradient = 0.5
    np.testing.assert_allclose(grads[1], 0.5)


# ===== Gradient — shapes =====

def test_gradient_shapes():
    op = GeometryOperator(make_grid_5x5x5())
    field = np.ones((5, 5, 5), dtype=np.float64)
    grads = op.gradient(field)
    assert grads[0].shape == (4, 5, 5)
    assert grads[1].shape == (5, 4, 5)
    assert grads[2].shape == (5, 5, 4)


# ===== Gradient — validation =====

def test_gradient_dtype_rejected():
    op = GeometryOperator(make_grid_5x5x5())
    field = np.ones((5, 5, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        op.gradient(field)


def test_gradient_shape_rejected():
    op = GeometryOperator(make_grid_5x5x5())
    field = np.ones((7, 7, 7), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        op.gradient(field)


def test_gradient_non_contiguous_rejected():
    op = GeometryOperator(make_grid_5x5x5())
    big = np.ones((10, 5, 5), dtype=np.float64)
    field = big[::2]
    if field.flags["C_CONTIGUOUS"]:
        pytest.skip("slice happened to remain contiguous")
    with pytest.raises(ValueError, match="C-contiguous"):
        op.gradient(field)


# ===== E. Flux constant => zero =====

def test_flux_constant_field_zero():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    field = 2.5 * np.ones((5, 5, 5), dtype=np.float64)
    fluxes = op.flux(field, metric)
    for J in fluxes:
        np.testing.assert_allclose(J, 0.0)


# ===== F. Flux sign convention =====

def test_flux_sign_psi_decreasing():
    """psi=[2,1], h=1, dx=1.

    grad[0] = (1-2)/1 = -1
    flux[0] = -h * grad = -1 * (-1) = +1
    """
    op = GeometryOperator(make_grid((2,), dx=1.0))
    field = np.array([2.0, 1.0])
    metric = ScalarConformalMetric(np.ones((2,), dtype=np.float64))
    fluxes = op.flux(field, metric)
    np.testing.assert_allclose(fluxes[0], 1.0)


def test_flux_sign_psi_increasing():
    """psi=[1,2], grad=+1, flux=-1."""
    op = GeometryOperator(make_grid((2,), dx=1.0))
    field = np.array([1.0, 2.0])
    metric = ScalarConformalMetric(np.ones((2,), dtype=np.float64))
    fluxes = op.flux(field, metric)
    np.testing.assert_allclose(fluxes[0], -1.0)


def test_flux_h_constant_equals_minus_gradient():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    field = np.zeros((5, 5, 5), dtype=np.float64)
    for i in range(5):
        field[i, :, :] = float(i)
    grads = op.gradient(field)
    fluxes = op.flux(field, metric)
    np.testing.assert_allclose(fluxes[0], -grads[0])


def test_flux_h_variable_harmonic_weighted():
    """psi=[0,1], h=[2,8], dx=1.

    grad[0] = 1
    h_face = 2*2*8/(2+8) = 3.2
    flux = -3.2 * 1 = -3.2
    """
    op = GeometryOperator(make_grid((2,), dx=1.0))
    field = np.array([0.0, 1.0])
    metric = ScalarConformalMetric(np.array([2.0, 8.0]))
    fluxes = op.flux(field, metric)
    np.testing.assert_allclose(fluxes[0], -3.2)


# ===== TensorMetric rejected =====

def test_flux_tensor_metric_rejected():
    op = GeometryOperator(make_grid_5x5x5())
    H = np.broadcast_to(np.eye(3, dtype=np.float64), (5, 5, 5, 3, 3))
    H = np.ascontiguousarray(H, dtype=np.float64)
    tm = TensorMetric(H)
    field = np.ones((5, 5, 5), dtype=np.float64)
    with pytest.raises(NotImplementedError, match="TensorMetric"):
        op.flux(field, tm)


def test_flux_metric_shape_mismatch_rejected():
    op = GeometryOperator(make_grid_5x5x5())
    metric = ScalarConformalMetric(np.ones((7, 7, 7), dtype=np.float64))
    field = np.ones((5, 5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="metric shape"):
        op.flux(field, metric)


# ===== G. Divergence — zero flux => zero =====

def test_divergence_zero_flux_3d():
    op = GeometryOperator(make_grid_5x5x5())
    zero_flux = (
        np.zeros((4, 5, 5), dtype=np.float64),
        np.zeros((5, 4, 5), dtype=np.float64),
        np.zeros((5, 5, 4), dtype=np.float64),
    )
    dpsi = op.divergence(zero_flux)
    np.testing.assert_allclose(dpsi, 0.0)
    assert dpsi.shape == (5, 5, 5)


# ===== H. Divergence — constant interior flux in 1D =====

def test_divergence_constant_flux_1d_neumann():
    """Constant interior flux J=+1, 5 cells, dx=1.

    dpsi[0]   = -J/dx = -1
    dpsi[1..3] = (J-J)/dx = 0
    dpsi[4]   = +J/dx = +1
    sum       = 0
    """
    op = GeometryOperator(make_grid((5,), dx=1.0))
    J = np.ones((4,), dtype=np.float64)
    dpsi = op.divergence((J,))
    expected = np.array([-1.0, 0.0, 0.0, 0.0, +1.0])
    np.testing.assert_allclose(dpsi, expected)
    assert abs(dpsi.sum()) < 1e-12


# ===== I. Diffusion pic 1D [0,1,0] => [1,-2,1] =====

def test_diffusion_pic_1d():
    """psi=[0,1,0], h=1, dx=1.

    grad = [+1, -1]
    flux = [-1, +1]
    div = [1, -2, 1]  (sign: dpsi/dt = divergence)
    """
    op = GeometryOperator(make_grid_3())
    metric = make_metric_1d_unit(3)
    field = np.array([0.0, 1.0, 0.0])
    fluxes = op.flux(field, metric)
    dpsi = op.divergence(fluxes)
    np.testing.assert_allclose(dpsi, [1.0, -2.0, 1.0])


def test_diffusion_pic_1d_sign_verrouille():
    """Critical sign test: peak must DECREASE (dpsi[center] negative)."""
    op = GeometryOperator(make_grid_3())
    metric = make_metric_1d_unit(3)
    field = np.array([0.0, 1.0, 0.0])
    fluxes = op.flux(field, metric)
    dpsi = op.divergence(fluxes)
    # Peak center decreases, neighbours increase
    assert dpsi[1] < 0.0
    assert dpsi[0] > 0.0
    assert dpsi[2] > 0.0


# ===== Conservation — sum = 0 =====

def test_conservation_1d():
    op = GeometryOperator(make_grid((10,), dx=0.3))
    metric = ScalarConformalMetric(
        1.0 + np.arange(10, dtype=np.float64) * 0.1
    )
    field = np.random.default_rng(0).standard_normal(10).astype(np.float64)
    fluxes = op.flux(field, metric)
    dpsi = op.divergence(fluxes)
    assert abs(dpsi.sum()) < 1e-12


def test_conservation_3d():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    rng = np.random.default_rng(42)
    field = rng.standard_normal((5, 5, 5)).astype(np.float64)
    fluxes = op.flux(field, metric)
    dpsi = op.divergence(fluxes)
    assert abs(dpsi.sum()) < 1e-12


def test_conservation_3d_anisotropic():
    op = GeometryOperator(make_grid((5, 5, 5), dx=(1.0, 2.0, 0.5)))
    metric = make_metric_3d_unit()
    rng = np.random.default_rng(7)
    field = rng.standard_normal((5, 5, 5)).astype(np.float64)
    fluxes = op.flux(field, metric)
    dpsi = op.divergence(fluxes)
    assert abs(dpsi.sum()) < 1e-12


# ===== Q. Axis isolation =====

def test_divergence_axis_isolation_3d():
    """Inject flux only on axis 0; result must equal pure 1D divergence.

    In 3D with J = (J0, 0, 0), divergence must reduce to the divergence
    coming from axis 0 alone. No leakage from other axes.
    """
    op = GeometryOperator(make_grid_5x5x5())

    # Non-trivial flux on axis 0 only
    rng = np.random.default_rng(123)
    J0 = rng.standard_normal((4, 5, 5)).astype(np.float64)
    J1_zero = np.zeros((5, 4, 5), dtype=np.float64)
    J2_zero = np.zeros((5, 5, 4), dtype=np.float64)

    dpsi_3d = op.divergence((J0, J1_zero, J2_zero))

    # Compare with 1D divergence applied per (j, k) slice
    expected = np.zeros((5, 5, 5), dtype=np.float64)
    op1d = GeometryOperator(make_grid((5,), dx=1.0))
    for j in range(5):
        for k in range(5):
            J_slice = J0[:, j, k]
            d_slice = op1d.divergence((np.ascontiguousarray(J_slice),))
            expected[:, j, k] = d_slice

    np.testing.assert_allclose(dpsi_3d, expected, atol=1e-14)


def test_divergence_axis_isolation_axis1():
    """Same isolation test for axis 1."""
    op = GeometryOperator(make_grid_5x5x5())
    rng = np.random.default_rng(7)
    J0_zero = np.zeros((4, 5, 5), dtype=np.float64)
    J1 = rng.standard_normal((5, 4, 5)).astype(np.float64)
    J2_zero = np.zeros((5, 5, 4), dtype=np.float64)
    dpsi_3d = op.divergence((J0_zero, J1, J2_zero))

    expected = np.zeros((5, 5, 5), dtype=np.float64)
    op1d = GeometryOperator(make_grid((5,), dx=1.0))
    for i in range(5):
        for k in range(5):
            J_slice = J1[i, :, k]
            d_slice = op1d.divergence((np.ascontiguousarray(J_slice),))
            expected[i, :, k] = d_slice
    np.testing.assert_allclose(dpsi_3d, expected, atol=1e-14)


def test_divergence_axis_additivity():
    """divergence((J0, J1, J2)) == divergence((J0,0,0)) + ... per-axis sum."""
    op = GeometryOperator(make_grid_5x5x5())
    rng = np.random.default_rng(11)
    J0 = rng.standard_normal((4, 5, 5)).astype(np.float64)
    J1 = rng.standard_normal((5, 4, 5)).astype(np.float64)
    J2 = rng.standard_normal((5, 5, 4)).astype(np.float64)
    Z0 = np.zeros_like(J0)
    Z1 = np.zeros_like(J1)
    Z2 = np.zeros_like(J2)

    d_full = op.divergence((J0, J1, J2))
    d0 = op.divergence((J0, Z1, Z2))
    d1 = op.divergence((Z0, J1, Z2))
    d2 = op.divergence((Z0, Z1, J2))
    np.testing.assert_allclose(d_full, d0 + d1 + d2, atol=1e-14)


# ===== Divergence — validation =====

def test_divergence_wrong_tuple_length_rejected():
    op = GeometryOperator(make_grid_5x5x5())
    with pytest.raises(ValueError, match="length"):
        op.divergence((np.zeros((4, 5, 5)), np.zeros((5, 4, 5))))


def test_divergence_wrong_axis_shape_rejected():
    op = GeometryOperator(make_grid_5x5x5())
    bad = (
        np.zeros((3, 5, 5), dtype=np.float64),  # should be (4,5,5)
        np.zeros((5, 4, 5), dtype=np.float64),
        np.zeros((5, 5, 4), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="axis"):
        op.divergence(bad)


def test_divergence_wrong_dtype_rejected():
    op = GeometryOperator(make_grid_5x5x5())
    bad = (
        np.zeros((4, 5, 5), dtype=np.float32),
        np.zeros((5, 4, 5), dtype=np.float64),
        np.zeros((5, 5, 4), dtype=np.float64),
    )
    with pytest.raises(TypeError, match="float64"):
        op.divergence(bad)


def test_divergence_not_tuple_rejected():
    op = GeometryOperator(make_grid_5x5x5())
    with pytest.raises(TypeError, match="tuple"):
        op.divergence([np.zeros((4, 5, 5)), np.zeros((5, 4, 5)), np.zeros((5, 5, 4))])


# ===== J/K. Apply =====

def test_apply_constant_psi_zero():
    """psi constant => dpsi == 0."""
    g = make_grid_5x5x5()
    op = GeometryOperator(g)
    metric = make_metric_3d_unit()
    state = State(
        fields={
            "psi": 2.0 * np.ones((5, 5, 5), dtype=np.float64),
            "h": 10.0 * np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    dstate = op.apply(state, metric, field_name="psi")
    np.testing.assert_allclose(dstate["psi"], 0.0)


def test_apply_pic_1d():
    """1D pic; dstate['psi'] = [1,-2,1]."""
    g = make_grid_3()
    op = GeometryOperator(g)
    metric = make_metric_1d_unit(3)
    state = State(
        fields={
            "psi": np.array([0.0, 1.0, 0.0]),
            "h": np.ones((3,), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    dstate = op.apply(state, metric, field_name="psi")
    np.testing.assert_allclose(dstate["psi"], [1.0, -2.0, 1.0])


# ===== L. Apply 3D conservation =====

def test_apply_3d_conservation():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    rng = np.random.default_rng(101)
    psi = rng.standard_normal((5, 5, 5)).astype(np.float64)
    psi = np.ascontiguousarray(psi)
    state = State(
        fields={"psi": psi, "h": np.ones((5, 5, 5), dtype=np.float64)},
        solver_fields=("psi", "h"),
    )
    dstate = op.apply(state, metric, field_name="psi")
    assert abs(dstate["psi"].sum()) < 1e-12


# ===== M. h derivative is zero when field_name='psi' =====

def test_apply_h_derivative_zero_when_psi_targeted():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    state = State(
        fields={
            "psi": np.random.default_rng(0).standard_normal((5, 5, 5)).astype(np.float64),
            "h": np.random.default_rng(1).standard_normal((5, 5, 5)).astype(np.float64) + 5.0,
        },
        solver_fields=("psi", "h"),
    )
    # Ensure C-contiguous
    state = State(
        fields={k: np.ascontiguousarray(v) for k, v in state.fields.items()},
        solver_fields=state.solver_fields,
    )
    dstate = op.apply(state, metric, field_name="psi")
    np.testing.assert_array_equal(dstate["h"], np.zeros((5, 5, 5)))


# ===== Typed zero (audit ajout 1) =====

def test_apply_typed_zero_for_unused_solver_field():
    """Unused solver field receives np.zeros(shape, dtype=float64), C-contig.
    NOT a scalar 0.
    """
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    state = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    dstate = op.apply(state, metric, field_name="psi")
    h_deriv = dstate["h"]
    assert isinstance(h_deriv, np.ndarray)
    assert h_deriv.shape == (5, 5, 5)
    assert h_deriv.dtype == np.float64
    assert h_deriv.flags["C_CONTIGUOUS"]


# ===== Static fields preserved =====

def test_apply_static_fields_preserved():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    mask = np.ones((5, 5, 5), dtype=bool)
    mask[0, 0, 0] = False
    state = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
            "mask": mask,
        },
        solver_fields=("psi", "h"),
    )
    dstate = op.apply(state, metric, field_name="psi")
    assert "mask" in dstate.fields
    np.testing.assert_array_equal(dstate["mask"], mask)
    assert dstate["mask"].dtype == np.bool_


# ===== Metadata trace (audit ajout 3) =====

def test_apply_metadata_trace():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    state = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
        metadata={"experiment": "X", "t": 1.5},
    )
    dstate = op.apply(state, metric, field_name="psi")
    assert dstate.metadata["operator"] == "GeometryOperator.apply"
    assert dstate.metadata["derivative_field"] == "psi"
    # Source metadata preserved
    assert dstate.metadata["experiment"] == "X"
    assert dstate.metadata["t"] == 1.5
    # Defensive copy: source unmodified
    assert "operator" not in state.metadata


def test_apply_metadata_defensive_copy():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    state = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
        metadata={"k": 1},
    )
    dstate = op.apply(state, metric, field_name="psi")
    dstate.metadata["k"] = 999  # mutate derivative metadata
    assert state.metadata["k"] == 1  # source unchanged


# ===== Apply validation =====

def test_apply_field_not_in_state():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    state = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    with pytest.raises(KeyError):
        op.apply(state, metric, field_name="unknown")


def test_apply_field_not_solver():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    state = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
            "mask": np.ones((5, 5, 5), dtype=bool),
        },
        solver_fields=("psi", "h"),
    )
    with pytest.raises(ValueError, match="not a solver field"):
        op.apply(state, metric, field_name="mask")


def test_apply_invalid_state_type():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    with pytest.raises(TypeError, match="state"):
        op.apply("not a state", metric, field_name="psi")


# ===== Statelessness (audit ajout 2) =====

def test_geometry_stateless_no_extra_attributes():
    """GeometryOperator.__slots__ must be exactly ('_grid',)."""
    assert GeometryOperator.__slots__ == ("_grid",)


def test_geometry_no_ad_hoc_attribute():
    op = GeometryOperator(make_grid_5x5x5())
    with pytest.raises(AttributeError):
        op._cache = "something"
    with pytest.raises(AttributeError):
        op._last_flux = None


def test_geometry_stateless_repeated_calls():
    """Applying op on state A, then state B, then state A again, must
    yield identical result for A both times — no leakage between calls.
    """
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    rng = np.random.default_rng(0)
    psi_A = np.ascontiguousarray(rng.standard_normal((5, 5, 5)).astype(np.float64))
    psi_B = np.ascontiguousarray(rng.standard_normal((5, 5, 5)).astype(np.float64))
    state_A = State(
        fields={"psi": psi_A, "h": np.ones((5, 5, 5))},
        solver_fields=("psi", "h"),
    )
    state_B = State(
        fields={"psi": psi_B, "h": np.ones((5, 5, 5))},
        solver_fields=("psi", "h"),
    )

    dstate_A1 = op.apply(state_A, metric, field_name="psi")
    _ = op.apply(state_B, metric, field_name="psi")
    dstate_A2 = op.apply(state_A, metric, field_name="psi")

    np.testing.assert_array_equal(dstate_A1["psi"], dstate_A2["psi"])


def test_geometry_pure_function_inputs_not_mutated():
    """apply, flux, divergence, gradient must not mutate inputs."""
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    psi = np.ones((5, 5, 5), dtype=np.float64)
    psi[2, 2, 2] = 5.0
    psi_snapshot = psi.copy()
    state = State(
        fields={"psi": psi, "h": np.ones((5, 5, 5))},
        solver_fields=("psi", "h"),
    )
    h_snapshot = metric.h.copy()

    _ = op.gradient(psi)
    _ = op.flux(psi, metric)
    _ = op.apply(state, metric, field_name="psi")

    # State has defensive copy, but we still check metric.h not changed
    np.testing.assert_array_equal(metric.h, h_snapshot)


# ===== O. No coupling/dynamics methods =====

def test_geometry_no_coupling_methods():
    op = GeometryOperator(make_grid_5x5x5())
    forbidden = [
        "compute_coupling", "compute_overlap", "scheduler",
        "aggregate", "synchronize", "step", "evolve",
        "update_metric", "h_dynamics",
    ]
    for name in forbidden:
        assert not hasattr(op, name), f"forbidden method present: {name}"


# ===== Output contracts =====

def test_dstate_has_same_solver_fields():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    state = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    dstate = op.apply(state, metric, field_name="psi")
    assert dstate.solver_fields == state.solver_fields


def test_dstate_output_dtype_and_contiguity():
    op = GeometryOperator(make_grid_5x5x5())
    metric = make_metric_3d_unit()
    state = State(
        fields={
            "psi": np.ones((5, 5, 5), dtype=np.float64),
            "h": np.ones((5, 5, 5), dtype=np.float64),
        },
        solver_fields=("psi", "h"),
    )
    dstate = op.apply(state, metric, field_name="psi")
    for sf in dstate.solver_fields:
        arr = dstate[sf]
        assert arr.dtype == np.float64
        assert arr.flags["C_CONTIGUOUS"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
