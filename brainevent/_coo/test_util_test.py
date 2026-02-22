# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._coo.test_util import vector_coo, coo_vector, matrix_coo, coo_matrix


def _dense_from_coo(row, col, data, shape):
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    data = np.asarray(data)
    dense = np.zeros(shape, dtype=data.dtype)
    for r, c, d in zip(row, col, data):
        dense[r, c] += d
    return dense


def _sample_coo(shape, nnz, *, seed=0, allow_duplicates=True):
    rng = np.random.default_rng(seed)
    m, n = shape
    if nnz == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    if allow_duplicates:
        row = rng.integers(0, m, size=nnz, dtype=np.int32)
        col = rng.integers(0, n, size=nnz, dtype=np.int32)
        return row, col
    flat = rng.choice(m * n, size=nnz, replace=False)
    row = (flat // n).astype(np.int32)
    col = (flat % n).astype(np.int32)
    return row, col


@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("allow_duplicates", [True, False])
def test_vector_coo_matches_dense(homo_w, allow_duplicates):
    shape = (6, 5)
    row, col = _sample_coo(shape, 12, seed=1, allow_duplicates=allow_duplicates)
    rng = np.random.default_rng(2)
    x = rng.normal(size=shape[0]).astype(np.float32)

    if homo_w:
        w = np.float32(1.25)
        data = np.full(row.shape, w, dtype=np.float32)
    else:
        data = rng.normal(size=row.shape[0]).astype(np.float32)
        w = data

    dense = _dense_from_coo(row, col, data, shape)
    y_ref = x @ dense
    y = vector_coo(jnp.asarray(x), w, row, col, shape)
    assert jnp.allclose(y, y_ref, rtol=1e-5, atol=1e-5)
    jax.block_until_ready((y,))


@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("allow_duplicates", [True, False])
def test_coo_vector_matches_dense(homo_w, allow_duplicates):
    shape = (6, 5)
    row, col = _sample_coo(shape, 12, seed=3, allow_duplicates=allow_duplicates)
    rng = np.random.default_rng(4)
    v = rng.normal(size=shape[1]).astype(np.float32)

    if homo_w:
        w = np.float32(0.5)
        data = np.full(row.shape, w, dtype=np.float32)
    else:
        data = rng.normal(size=row.shape[0]).astype(np.float32)
        w = data

    dense = _dense_from_coo(row, col, data, shape)
    y_ref = dense @ v
    y = coo_vector(jnp.asarray(v), w, row, col, shape)
    assert jnp.allclose(y, y_ref, rtol=1e-5, atol=1e-5)
    jax.block_until_ready((y,))


@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("allow_duplicates", [True, False])
def test_matrix_coo_matches_dense(homo_w, allow_duplicates):
    shape = (5, 4)
    row, col = _sample_coo(shape, 10, seed=5, allow_duplicates=allow_duplicates)
    rng = np.random.default_rng(6)
    xs = rng.normal(size=(3, shape[0])).astype(np.float32)

    if homo_w:
        w = np.float32(2.0)
        data = np.full(row.shape, w, dtype=np.float32)
    else:
        data = rng.normal(size=row.shape[0]).astype(np.float32)
        w = data

    dense = _dense_from_coo(row, col, data, shape)
    y_ref = xs @ dense
    y = matrix_coo(jnp.asarray(xs), w, row, col, shape)
    assert jnp.allclose(y, y_ref, rtol=1e-5, atol=1e-5)
    jax.block_until_ready((y,))


@pytest.mark.parametrize("homo_w", [True, False])
@pytest.mark.parametrize("allow_duplicates", [True, False])
def test_coo_matrix_matches_dense(homo_w, allow_duplicates):
    shape = (5, 4)
    row, col = _sample_coo(shape, 10, seed=7, allow_duplicates=allow_duplicates)
    rng = np.random.default_rng(8)
    xs = rng.normal(size=(shape[1], 3)).astype(np.float32)

    if homo_w:
        w = np.float32(1.75)
        data = np.full(row.shape, w, dtype=np.float32)
    else:
        data = rng.normal(size=row.shape[0]).astype(np.float32)
        w = data

    dense = _dense_from_coo(row, col, data, shape)
    y_ref = dense @ xs
    y = coo_matrix(jnp.asarray(xs), w, row, col, shape)
    assert jnp.allclose(y, y_ref, rtol=1e-5, atol=1e-5)
    jax.block_until_ready((y,))


def test_empty_coo_outputs_zeros():
    shape = (4, 3)
    row = np.array([], dtype=np.int32)
    col = np.array([], dtype=np.int32)

    x = jnp.arange(shape[0], dtype=jnp.float32)
    v = jnp.arange(shape[1], dtype=jnp.float32)
    xs_left = jnp.arange(2 * shape[0], dtype=jnp.float32).reshape(2, shape[0])
    xs_right = jnp.arange(shape[1] * 2, dtype=jnp.float32).reshape(shape[1], 2)

    out_vector = vector_coo(x, 1.0, row, col, shape)
    assert out_vector.shape == (shape[1],)
    assert jnp.all(out_vector == 0)

    out_coo_vector = coo_vector(v, jnp.array([], dtype=jnp.float32), row, col, shape)
    assert out_coo_vector.shape == (shape[0],)
    assert jnp.all(out_coo_vector == 0)

    out_matrix = matrix_coo(xs_left, 1.0, row, col, shape)
    assert out_matrix.shape == (xs_left.shape[0], shape[1])
    assert jnp.all(out_matrix == 0)

    out_coo_matrix = coo_matrix(xs_right, jnp.array([], dtype=jnp.float32), row, col, shape)
    assert out_coo_matrix.shape == (shape[0], xs_right.shape[1])
    assert jnp.all(out_coo_matrix == 0)
    jax.block_until_ready((x, v, xs_left, xs_right, out_vector, out_coo_vector, out_matrix, out_coo_matrix))


def test_matrix_coo_jit_matches_dense():
    shape = (6, 5)
    row, col = _sample_coo(shape, 8, seed=9, allow_duplicates=False)
    xs = jnp.arange(18, dtype=jnp.float32).reshape(3, shape[0]) / 10.0
    w = jnp.linspace(0.1, 0.8, num=row.shape[0], dtype=jnp.float32)

    dense = jnp.zeros(shape, dtype=jnp.float32).at[row, col].add(w)
    y_ref = xs @ dense

    f = jax.jit(lambda x: matrix_coo(x, w, row, col, shape))
    y = f(xs)
    assert jnp.allclose(y, y_ref, rtol=1e-5, atol=1e-5)
    jax.block_until_ready((xs, w, dense, y_ref, y))


def test_coo_matrix_jit_matches_dense():
    shape = (6, 5)
    row, col = _sample_coo(shape, 8, seed=10, allow_duplicates=False)
    xs = jnp.arange(15, dtype=jnp.float32).reshape(shape[1], 3) / 7.0
    w = jnp.linspace(0.2, 1.0, num=row.shape[0], dtype=jnp.float32)

    dense = jnp.zeros(shape, dtype=jnp.float32).at[row, col].add(w)
    y_ref = dense @ xs

    f = jax.jit(lambda x: coo_matrix(x, w, row, col, shape))
    y = f(xs)
    assert jnp.allclose(y, y_ref, rtol=1e-5, atol=1e-5)
    jax.block_until_ready((xs, w, dense, y_ref, y))


def test_matrix_coo_grad_matches_dense():
    shape = (4, 3)
    row, col = _sample_coo(shape, 6, seed=11, allow_duplicates=False)
    xs = jnp.arange(8, dtype=jnp.float32).reshape(2, shape[0]) / 5.0
    w = jnp.linspace(0.1, 0.6, num=row.shape[0], dtype=jnp.float32)

    def f(xs, w):
        return matrix_coo(xs, w, row, col, shape).sum()

    def f_ref(xs, w):
        dense = jnp.zeros(shape, dtype=xs.dtype).at[row, col].add(w)
        return (xs @ dense).sum()

    g = jax.grad(f, argnums=(0, 1))(xs, w)
    g_ref = jax.grad(f_ref, argnums=(0, 1))(xs, w)

    assert jnp.allclose(g[0], g_ref[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(g[1], g_ref[1], rtol=1e-5, atol=1e-5)
    jax.block_until_ready((xs, w, g, g_ref))


def test_coo_matrix_grad_matches_dense():
    shape = (4, 3)
    row, col = _sample_coo(shape, 6, seed=12, allow_duplicates=False)
    xs = jnp.arange(9, dtype=jnp.float32).reshape(shape[1], 3) / 4.0
    w = jnp.linspace(0.2, 0.7, num=row.shape[0], dtype=jnp.float32)

    def f(xs, w):
        return coo_matrix(xs, w, row, col, shape).sum()

    def f_ref(xs, w):
        dense = jnp.zeros(shape, dtype=xs.dtype).at[row, col].add(w)
        return (dense @ xs).sum()

    g = jax.grad(f, argnums=(0, 1))(xs, w)
    g_ref = jax.grad(f_ref, argnums=(0, 1))(xs, w)

    assert jnp.allclose(g[0], g_ref[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(g[1], g_ref[1], rtol=1e-5, atol=1e-5)
    jax.block_until_ready((xs, w, g, g_ref))
