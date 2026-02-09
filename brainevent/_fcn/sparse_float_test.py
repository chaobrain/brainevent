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

import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

from brainevent._fcn.sparse_float import (
    spfloat_fcnmv, spfloat_fcnmv_p,
    spfloat_fcnmm, spfloat_fcnmm_p,
)
from brainevent._test_util import (
    generate_fixed_conn_num_indices,
    vector_fcn,
    fcn_vector,
    matrix_fcn,
    fcn_matrix,
    allclose,
    ones_like,
)

platform = jax.default_backend()
SPFLOAT_FCNMV_IMPLEMENTATIONS = tuple(spfloat_fcnmv_p.available_backends(platform))
SPFLOAT_FCNMM_IMPLEMENTATIONS = tuple(spfloat_fcnmm_p.available_backends(platform))

if platform == 'cpu':
    shapes = [
        (20, 40),
        (50, 30),
    ]
else:
    shapes = [
        (20, 40),
        (50, 30),
        (200, 400),
        (500, 300),
    ]


def _sparse_float_vector(size, prob=0.3):
    """Generate a sparse float vector (zeros and random positive floats)."""
    mask = brainstate.random.rand(size) < prob
    vals = brainstate.random.rand(size)
    return jnp.where(mask, vals, 0.0)


def _sparse_float_matrix(rows, cols, prob=0.3):
    """Generate a sparse float matrix (zeros and random positive floats)."""
    mask = brainstate.random.rand(rows, cols) < prob
    vals = brainstate.random.rand(rows, cols)
    return jnp.where(mask, vals, 0.0)


# ---------------------------------------------------------------------------
# Forward: spfloat_fcnmv  (vector @ sparse_matrix  and  sparse_matrix @ vector)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", SPFLOAT_FCNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
def test_spfloat_fcnmv_transpose(implementation, shape, homo_w):
    """vector @ sparse_matrix  (transpose=True)."""
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    weights = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
    x = _sparse_float_vector(m)

    result = spfloat_fcnmv(weights, indices, x, shape=(m, n), transpose=True, backend=implementation)
    expected = vector_fcn(x, weights, indices, (m, n))
    assert allclose(result, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, x, result, expected))


@pytest.mark.parametrize("implementation", SPFLOAT_FCNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
def test_spfloat_fcnmv_no_transpose(implementation, shape, homo_w):
    """sparse_matrix @ vector  (transpose=False)."""
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    weights = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
    v = _sparse_float_vector(n)

    result = spfloat_fcnmv(weights, indices, v, shape=(m, n), transpose=False, backend=implementation)
    expected = fcn_vector(v, weights, indices, (m, n))
    assert allclose(result, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, v, result, expected))


# ---------------------------------------------------------------------------
# Forward: spfloat_fcnmm  (matrix @ sparse_matrix  and  sparse_matrix @ matrix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", SPFLOAT_FCNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("k", [10])
def test_spfloat_fcnmm_transpose(implementation, shape, homo_w, k):
    """matrix @ sparse_matrix  (transpose=True)."""
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    weights = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
    X = _sparse_float_matrix(m, k)

    result = spfloat_fcnmm(weights, indices, X, shape=(m, n), transpose=True, backend=implementation)
    expected = matrix_fcn(X.T, weights, indices, (m, n)).T
    assert allclose(result, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, X, result, expected))


@pytest.mark.parametrize("implementation", SPFLOAT_FCNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("k", [10])
def test_spfloat_fcnmm_no_transpose(implementation, shape, homo_w, k):
    """sparse_matrix @ matrix  (transpose=False)."""
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    weights = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
    B = _sparse_float_matrix(n, k)

    result = spfloat_fcnmm(weights, indices, B, shape=(m, n), transpose=False, backend=implementation)
    expected = fcn_matrix(B, weights, indices, (m, n))
    assert allclose(result, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, B, result, expected))


# ---------------------------------------------------------------------------
# Gradient (VJP): spfloat_fcnmv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", SPFLOAT_FCNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_spfloat_fcnmv_vjp(implementation, shape, homo_w, transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    w = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
    x = brainstate.random.rand(m if transpose else n)

    def f(x, w):
        return spfloat_fcnmv(w, indices, x, shape=(m, n), transpose=transpose, backend=implementation).sum()

    def f_ref(x, w):
        if transpose:
            return vector_fcn(x, w, indices, (m, n)).sum()
        else:
            return fcn_vector(x, w, indices, (m, n)).sum()

    r1 = jax.jit(lambda x, w: jax.grad(f, argnums=(0, 1))(x, w))(x, w)
    r2 = jax.jit(lambda x, w: jax.grad(f_ref, argnums=(0, 1))(x, w))(x, w)

    assert allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
    assert allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, w, x, r1[0], r1[1], r2[0], r2[1]))


# ---------------------------------------------------------------------------
# Gradient (JVP): spfloat_fcnmv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", SPFLOAT_FCNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_spfloat_fcnmv_jvp(implementation, shape, homo_w, transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    w = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
    x = brainstate.random.rand(m if transpose else n)

    def f(x, w):
        return spfloat_fcnmv(w, indices, x, shape=(m, n), transpose=transpose, backend=implementation)

    def f_ref(x, w):
        if transpose:
            return vector_fcn(x, w, indices, (m, n))
        else:
            return fcn_vector(x, w, indices, (m, n))

    o1, r1 = jax.jit(
        lambda x, w: jax.jvp(f, (x, w), (ones_like(x), ones_like(w)))
    )(x, w)
    o2, r2 = jax.jit(
        lambda x, w: jax.jvp(f_ref, (x, w), (ones_like(x), ones_like(w)))
    )(x, w)

    assert allclose(o1, o2, rtol=1e-3, atol=1e-3)
    assert allclose(r1, r2, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, w, x, o1, r1, o2, r2))


# ---------------------------------------------------------------------------
# Gradient (VJP): spfloat_fcnmm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", SPFLOAT_FCNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("k", [10])
def test_spfloat_fcnmm_vjp(implementation, shape, homo_w, transpose, k):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    w = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
    x = brainstate.random.rand(m, k) if transpose else brainstate.random.rand(n, k)

    def f(x, w):
        return spfloat_fcnmm(w, indices, x, shape=(m, n), transpose=transpose, backend=implementation).sum()

    def f_ref(x, w):
        if transpose:
            return matrix_fcn(x.T, w, indices, (m, n)).T.sum()
        else:
            return fcn_matrix(x, w, indices, (m, n)).sum()

    r1 = jax.jit(lambda x, w: jax.grad(f, argnums=(0, 1))(x, w))(x, w)
    r2 = jax.jit(lambda x, w: jax.grad(f_ref, argnums=(0, 1))(x, w))(x, w)

    assert allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
    assert allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, w, x, r1[0], r1[1], r2[0], r2[1]))


# ---------------------------------------------------------------------------
# Gradient (JVP): spfloat_fcnmm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", SPFLOAT_FCNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("k", [10])
def test_spfloat_fcnmm_jvp(implementation, shape, homo_w, transpose, k):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    w = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
    x = brainstate.random.rand(m, k) if transpose else brainstate.random.rand(n, k)

    def f(x, w):
        return spfloat_fcnmm(w, indices, x, shape=(m, n), transpose=transpose, backend=implementation)

    def f_ref(x, w):
        if transpose:
            return matrix_fcn(x.T, w, indices, (m, n)).T
        else:
            return fcn_matrix(x, w, indices, (m, n))

    o1, r1 = jax.jit(
        lambda x, w: jax.jvp(f, (x, w), (ones_like(x), ones_like(w)))
    )(x, w)
    o2, r2 = jax.jit(
        lambda x, w: jax.jvp(f_ref, (x, w), (ones_like(x), ones_like(w)))
    )(x, w)

    assert allclose(o1, o2, rtol=1e-3, atol=1e-3)
    assert allclose(r1, r2, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, w, x, o1, r1, o2, r2))


# ---------------------------------------------------------------------------
# Batching (vmap): spfloat_fcnmv over spikes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", SPFLOAT_FCNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("batch_axis", [0, 1])
def test_spfloat_fcnmv_vmap_transpose(implementation, shape, homo_w, batch_size, batch_axis):
    """vmap over vector @ sparse_matrix  (transpose=True)."""
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    weights = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

    if batch_axis == 0:
        xs = _sparse_float_matrix(batch_size, m)
    else:
        xs = _sparse_float_matrix(m, batch_size)

    result = jax.jit(jax.vmap(
        lambda x: spfloat_fcnmv(weights, indices, x, shape=(m, n), transpose=True, backend=implementation),
        in_axes=batch_axis,
    ))(xs)
    expected = jax.jit(jax.vmap(
        lambda x: vector_fcn(x, weights, indices, (m, n)),
        in_axes=batch_axis,
    ))(xs)
    assert allclose(result, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, xs, result, expected))


@pytest.mark.parametrize("implementation", SPFLOAT_FCNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("batch_axis", [0, 1])
def test_spfloat_fcnmv_vmap_no_transpose(implementation, shape, homo_w, batch_size, batch_axis):
    """vmap over sparse_matrix @ vector  (transpose=False)."""
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    weights = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

    if batch_axis == 0:
        vs = _sparse_float_matrix(batch_size, n)
    else:
        vs = _sparse_float_matrix(n, batch_size)

    result = jax.jit(jax.vmap(
        lambda v: spfloat_fcnmv(weights, indices, v, shape=(m, n), transpose=False, backend=implementation),
        in_axes=batch_axis,
    ))(vs)
    expected = jax.jit(jax.vmap(
        lambda v: fcn_vector(v, weights, indices, (m, n)),
        in_axes=batch_axis,
    ))(vs)
    assert allclose(result, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, vs, result, expected))


# ---------------------------------------------------------------------------
# Batching (vmap): spfloat_fcnmm over matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", SPFLOAT_FCNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("k", [10])
@pytest.mark.parametrize("batch_axis", [0, 1, 2])
def test_spfloat_fcnmm_vmap_no_transpose(implementation, shape, homo_w, batch_size, k, batch_axis):
    """vmap over sparse_matrix @ matrix  (transpose=False)."""
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
    weights = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

    if batch_axis == 0:
        Bs = brainstate.random.rand(batch_size, n, k)
    elif batch_axis == 1:
        Bs = brainstate.random.rand(n, batch_size, k)
    else:
        Bs = brainstate.random.rand(n, k, batch_size)

    result = jax.jit(jax.vmap(
        lambda B: spfloat_fcnmm(weights, indices, B, shape=(m, n), transpose=False, backend=implementation),
        in_axes=batch_axis,
    ))(Bs)
    expected = jax.jit(jax.vmap(
        lambda B: fcn_matrix(B, weights, indices, (m, n)),
        in_axes=batch_axis,
    ))(Bs)
    assert allclose(result, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, Bs, result, expected))
