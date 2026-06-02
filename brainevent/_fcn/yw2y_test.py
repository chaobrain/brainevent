# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import pytest

import brainunit as u
from brainevent import fcnmv_yw2y  # top-level export
from brainevent._fcn.yw2y import fcnmv_yw2y as fcnmv_yw2y_module  # module-level export
from brainevent._test_util import (
    generate_fixed_conn_num_indices,
    allclose,
)

shape = (20, 40)
n_conn = 4


def _reference(w, indices, y, transpose):
    """Hand-rolled reference: out[i, k] = w[i, k] * y[row/col]."""
    w_full = jnp.broadcast_to(w, indices.shape)
    yv = y[indices] if transpose else y[:, None]
    return w_full * yv


def test_module_and_toplevel_are_same_object():
    # The dedicated module is the single source of truth; the top-level export
    # must be the very same function object.
    assert fcnmv_yw2y is fcnmv_yw2y_module


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_fcnmv_yw2y(homo_w, transpose):
    m, n = shape  # (20, 40)
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    if homo_w:
        w = jnp.asarray(1.5, dtype=jnp.float32)
    else:
        w = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
    y = jnp.arange(1, (m if not transpose else n) + 1, dtype=jnp.float32)

    out = fcnmv_yw2y(w, indices, y, shape=shape, transpose=transpose)

    assert out.shape == indices.shape
    expected = _reference(w, indices, y, transpose)
    assert allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('homo_w', [True, False])
def test_fcnmv_yw2y_homo_size1_array_equiv_scalar(homo_w):
    # A size-1 array weight must behave identically to a 0-d scalar weight.
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    y = jnp.arange(1, m + 1, dtype=jnp.float32)
    scalar = jnp.asarray(1.5, dtype=jnp.float32)
    size1 = jnp.asarray([1.5], dtype=jnp.float32)
    out_scalar = fcnmv_yw2y(scalar, indices, y, shape=shape, transpose=False)
    out_size1 = fcnmv_yw2y(size1, indices, y, shape=shape, transpose=False)
    assert out_scalar.shape == out_size1.shape == indices.shape
    assert allclose(out_scalar, out_size1, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------- #
# Units: output unit must equal unit(weights) * unit(y).
# ---------------------------------------------------------------------------- #

def _setup_unit_case():
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w_mant = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
    y_mant = jnp.arange(1, m + 1, dtype=jnp.float32)
    return indices, w_mant, y_mant


def test_fcnmv_yw2y_units_both():
    indices, w_mant, y_mant = _setup_unit_case()
    out = fcnmv_yw2y(w_mant * u.siemens, indices, y_mant * u.mV, shape=shape, transpose=False)
    expected = (w_mant * y_mant[:, None]) * (u.siemens * u.mV)
    assert u.math.allclose(out, expected)


def test_fcnmv_yw2y_units_weight_only():
    indices, w_mant, y_mant = _setup_unit_case()
    out = fcnmv_yw2y(w_mant * u.siemens, indices, y_mant, shape=shape, transpose=False)
    expected = (w_mant * y_mant[:, None]) * u.siemens
    assert u.math.allclose(out, expected)


def test_fcnmv_yw2y_units_y_only():
    indices, w_mant, y_mant = _setup_unit_case()
    out = fcnmv_yw2y(w_mant, indices, y_mant * u.mV, shape=shape, transpose=False)
    expected = (w_mant * y_mant[:, None]) * u.mV
    assert u.math.allclose(out, expected)


def test_fcnmv_yw2y_units_none_is_plain_array():
    indices, w_mant, y_mant = _setup_unit_case()
    out = fcnmv_yw2y(w_mant, indices, y_mant, shape=shape, transpose=False)
    assert not isinstance(out, u.Quantity)
    assert allclose(out, w_mant * y_mant[:, None], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------- #
# Differentiability.
# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_fcnmv_yw2y_grad(homo_w, transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    if homo_w:
        w = jnp.asarray(1.5, dtype=jnp.float32)
    else:
        w = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
    y = jnp.arange(1, (m if not transpose else n) + 1, dtype=jnp.float32)

    def loss(w_, y_):
        return jnp.sum(fcnmv_yw2y(w_, indices, y_, shape=shape, transpose=transpose) ** 2)

    def ref(w_, y_):
        return jnp.sum(_reference(w_, indices, y_, transpose) ** 2)

    gw, gy = jax.grad(loss, argnums=(0, 1))(w, y)
    rgw, rgy = jax.grad(ref, argnums=(0, 1))(w, y)
    assert allclose(gw, rgw, rtol=1e-4, atol=1e-4)
    assert allclose(gy, rgy, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('transpose', [True, False])
def test_fcnmv_yw2y_jvp(transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
    y = jnp.arange(1, (m if not transpose else n) + 1, dtype=jnp.float32)
    dw = jnp.ones_like(w)
    dy = jnp.ones_like(y)

    f = lambda w_, y_: fcnmv_yw2y(w_, indices, y_, shape=shape, transpose=transpose)
    out, out_dot = jax.jvp(f, (w, y), (dw, dy))

    # d(w * yv) = dw * yv + w * dyv
    yv = y[indices] if transpose else y[:, None]
    dyv = dy[indices] if transpose else dy[:, None]
    expected_dot = dw * yv + w * dyv
    assert allclose(out, _reference(w, indices, y, transpose), rtol=1e-5, atol=1e-5)
    assert allclose(out_dot, expected_dot, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('transpose', [True, False])
def test_fcnmv_yw2y_vjp(transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
    y = jnp.arange(1, (m if not transpose else n) + 1, dtype=jnp.float32)

    f = lambda w_, y_: fcnmv_yw2y(w_, indices, y_, shape=shape, transpose=transpose)
    out, vjp_fn = jax.vjp(f, w, y)
    cot = jnp.ones_like(out)
    gw, gy = vjp_fn(cot)

    # d/dw = yv ;  d/dy is a scatter-add of w into y's positions.
    yv = y[indices] if transpose else y[:, None]
    assert allclose(gw, jnp.broadcast_to(yv, indices.shape), rtol=1e-4, atol=1e-4)
    if transpose:
        expected_gy = jnp.zeros_like(y).at[indices].add(w)
    else:
        expected_gy = jnp.sum(w, axis=1)
    assert allclose(gy, expected_gy, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------- #
# jit / vmap.
# ---------------------------------------------------------------------------- #

def test_fcnmv_yw2y_jit():
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
    y = jnp.arange(1, m + 1, dtype=jnp.float32)

    fn = jax.jit(lambda w_, y_: fcnmv_yw2y(w_, indices, y_, shape=shape, transpose=False))
    out = fn(w, y)
    assert allclose(out, w * y[:, None], rtol=1e-5, atol=1e-5)


def test_fcnmv_yw2y_vmap():
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
    Y = jnp.arange(1, 3 * m + 1, dtype=jnp.float32).reshape(3, m)  # batch of 3

    out = jax.vmap(lambda y_: fcnmv_yw2y(w, indices, y_, shape=shape, transpose=False))(Y)
    assert out.shape == (3, *indices.shape)
    expected = w[None] * Y[:, :, None]
    assert allclose(out, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------- #
# Dtype promotion (no equal-dtype constraint).
# ---------------------------------------------------------------------------- #

def test_fcnmv_yw2y_dtype_promotion():
    # float16 weights * float32 y -> float32, robust regardless of x64 config.
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w = jnp.ones(indices.shape, dtype=jnp.float16)
    y = jnp.arange(1, m + 1, dtype=jnp.float32)
    out = fcnmv_yw2y(w, indices, y, shape=shape, transpose=False)
    assert out.dtype == jnp.float32
    assert allclose(out, y[:, None].astype(jnp.float32), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------- #
# Empty structures.
# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo_w', [True, False])
def test_fcnmv_yw2y_empty_rows(homo_w, transpose):
    # rows == 0: indices (0, n_conn).
    n = 40
    indices = jnp.zeros((0, n_conn), dtype=jnp.int32)
    shp = (0, n)
    w = jnp.asarray(1.5, dtype=jnp.float32) if homo_w else jnp.zeros((0, n_conn), dtype=jnp.float32)
    y = jnp.zeros(n if transpose else 0, dtype=jnp.float32)
    out = fcnmv_yw2y(w, indices, y, shape=shp, transpose=transpose)
    assert out.shape == (0, n_conn)


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo_w', [True, False])
def test_fcnmv_yw2y_empty_conn(homo_w, transpose):
    # n_conn == 0: indices (rows, 0).
    m, n = shape
    indices = jnp.zeros((m, 0), dtype=jnp.int32)
    w = jnp.asarray(1.5, dtype=jnp.float32) if homo_w else jnp.zeros((m, 0), dtype=jnp.float32)
    y = jnp.arange(1, (m if not transpose else n) + 1, dtype=jnp.float32)
    out = fcnmv_yw2y(w, indices, y, shape=shape, transpose=transpose)
    assert out.shape == (m, 0)


# ---------------------------------------------------------------------------- #
# Validation.
# ---------------------------------------------------------------------------- #

def test_fcnmv_yw2y_validation():
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w = jnp.ones(indices.shape, dtype=jnp.float32)

    # indices not 2D
    with pytest.raises(ValueError):
        fcnmv_yw2y(jnp.ones(n_conn, dtype=jnp.float32), jnp.arange(n_conn, dtype=jnp.int32),
                   jnp.ones(m, dtype=jnp.float32), shape=shape, transpose=False)
    # shape not length-2
    with pytest.raises(ValueError):
        fcnmv_yw2y(w, indices, jnp.ones(m, dtype=jnp.float32), shape=(m, n, 1), transpose=False)
    # non-floating weights
    with pytest.raises(ValueError):
        fcnmv_yw2y(jnp.ones(indices.shape, dtype=jnp.int32), indices,
                   jnp.ones(m, dtype=jnp.float32), shape=shape, transpose=False)
    # weight shape mismatch (and not size-1)
    with pytest.raises(ValueError):
        fcnmv_yw2y(jnp.ones((m, n_conn + 1), dtype=jnp.float32), indices,
                   jnp.ones(m, dtype=jnp.float32), shape=shape, transpose=False)
    # y not 1D
    with pytest.raises(ValueError):
        fcnmv_yw2y(w, indices, jnp.ones((m, 1), dtype=jnp.float32), shape=shape, transpose=False)
    # wrong y length for transpose=False (expects shape[0]=m)
    with pytest.raises(ValueError):
        fcnmv_yw2y(w, indices, jnp.ones(m + 3, dtype=jnp.float32), shape=shape, transpose=False)
    # wrong y length for transpose=True (expects shape[1]=n)
    with pytest.raises(ValueError):
        fcnmv_yw2y(w, indices, jnp.ones(n + 3, dtype=jnp.float32), shape=shape, transpose=True)
