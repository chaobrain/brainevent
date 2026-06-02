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
from brainevent import fcnmv_yw2y  # top-level export (exercises N1)
from brainevent._fcn.float import fcnmv, fcnmm
from brainevent._test_util import (
    generate_fixed_conn_num_indices,
    vector_fcn,
    matrix_fcn,
    fcn_vector,
    fcn_matrix,
    allclose,
)

shape = (20, 40)
n_conn = 4


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_fcnmv(homo_w, transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn)
    w = jnp.asarray(1.5) if homo_w else jnp.ones(indices.shape)
    if transpose:
        x = jnp.ones(m)
        y = fcnmv(w, indices, x, shape=shape, transpose=True)
        y_ref = vector_fcn(x, w, indices, shape)
    else:
        x = jnp.ones(n)
        y = fcnmv(w, indices, x, shape=shape, transpose=False)
        y_ref = fcn_vector(x, w, indices, shape)
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_fcnmm(homo_w, transpose):
    m, n, k = *shape, 8
    indices = generate_fixed_conn_num_indices(m, n, n_conn)
    w = jnp.asarray(1.5) if homo_w else jnp.ones(indices.shape)
    if transpose:
        x = jnp.ones((k, m))
        y = fcnmm(w, indices, x.T, shape=shape, transpose=True).T
        y_ref = matrix_fcn(x, w, indices, shape)
    else:
        x = jnp.ones((n, k))
        y = fcnmm(w, indices, x, shape=shape, transpose=False)
        y_ref = fcn_matrix(x, w, indices, shape)
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)


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
    w_full = jnp.broadcast_to(w, indices.shape)
    yv = y[:, None] if not transpose else y[indices]
    expected = w_full * yv
    assert allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_fcnmv_yw2y_units():
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w_mant = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
    y_mant = jnp.arange(1, m + 1, dtype=jnp.float32)
    w = w_mant * u.siemens
    y = y_mant * u.mV

    out = fcnmv_yw2y(w, indices, y, shape=shape, transpose=False)

    expected = (w_mant * y_mant[:, None]) * (u.siemens * u.mV)
    assert u.math.allclose(out, expected)


@pytest.mark.parametrize('transpose', [True, False])
def test_fcnmv_yw2y_grad(transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
    y = jnp.arange(1, (m if not transpose else n) + 1, dtype=jnp.float32)

    def loss(w_, y_):
        return jnp.sum(fcnmv_yw2y(w_, indices, y_, shape=shape, transpose=transpose) ** 2)

    def ref(w_, y_):
        yv = y_[:, None] if not transpose else y_[indices]
        return jnp.sum((w_ * yv) ** 2)

    gw, gy = jax.grad(loss, argnums=(0, 1))(w, y)
    rgw, rgy = jax.grad(ref, argnums=(0, 1))(w, y)
    assert allclose(gw, rgw, rtol=1e-4, atol=1e-4)
    assert allclose(gy, rgy, rtol=1e-4, atol=1e-4)


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


def test_fcnmv_yw2y_validation():
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn, replace=True)
    w = jnp.ones(indices.shape, dtype=jnp.float32)

    # wrong y length for transpose=False (expects shape[0]=m)
    with pytest.raises(ValueError):
        fcnmv_yw2y(w, indices, jnp.ones(m + 3, dtype=jnp.float32), shape=shape, transpose=False)
    # weight shape mismatch (and not size-1)
    with pytest.raises(ValueError):
        fcnmv_yw2y(jnp.ones((m, n_conn + 1), dtype=jnp.float32), indices,
                   jnp.ones(m, dtype=jnp.float32), shape=shape, transpose=False)
    # non-floating weights
    with pytest.raises(ValueError):
        fcnmv_yw2y(jnp.ones(indices.shape, dtype=jnp.int32), indices,
                   jnp.ones(m, dtype=jnp.float32), shape=shape, transpose=False)
