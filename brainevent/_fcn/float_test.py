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


def _temp_bytes(fn, *args):
    stats = jax.jit(fn).lower(*args).compile().memory_analysis()
    if stats is None:
        pytest.skip('memory_analysis is not available on this backend')
    return stats.temp_size_in_bytes


@pytest.mark.parametrize('homo_w', [True, False])
def test_fcnmm_scatter_temporary_memory_is_bounded_under_vmap(homo_w, monkeypatch):
    """Regression: the scatter path built an ``(n_pre, n_conn, n)`` temporary.

    Under ``vmap`` over the weights that grew with the batch (5.7 GiB for
    shape (5000, 3000), 300 connections, n=32, batch 32) and ran GPUs out of
    memory. The column-chunked implementation bounds the temporary per batch
    element.
    """
    import brainevent._fcn.float as fcn_float
    monkeypatch.setattr(fcn_float, '_FCNMM_CHUNK_ELEMS', 1 << 16)   # small budget at test sizes
    transpose = True
    m, n_post, conn, k, batch = 2000, 1000, 200, 16, 8
    indices = generate_fixed_conn_num_indices(m, n_post, conn)
    w = jnp.ones((batch,)) * 1.5 if homo_w else jnp.ones((batch,) + indices.shape)
    x = jnp.ones((m, k)) if transpose else jnp.ones((n_post, k))

    def f(w, x):
        return jax.vmap(lambda wi: fcnmm(wi, indices, x, shape=(m, n_post), transpose=transpose))(w)

    naive = batch * m * conn * k * 4          # one (n_pre, n_conn, n) float32 buffer per batch element
    assert _temp_bytes(f, w, x) < naive // 4


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('conn', [1, 7, 25])
def test_fcnmm_chunked_matches_reference(homo_w, transpose, conn, monkeypatch):
    """Chunking (including a ragged last chunk) gives the reference result."""
    import brainevent._fcn.float as fcn_float
    monkeypatch.setattr(fcn_float, '_FCNMM_CHUNK_ELEMS', 2 * 40 * conn)   # 2 columns per chunk, ragged last chunk
    m, n_post, k = 40, 30, 5
    indices = generate_fixed_conn_num_indices(m, n_post, conn)
    w = jnp.asarray(1.5) if homo_w else jnp.arange(indices.size, dtype=jnp.float32).reshape(indices.shape) / 100
    if transpose:
        x = jnp.arange(m * k, dtype=jnp.float32).reshape(k, m) / 10
        y = fcnmm(w, indices, x.T, shape=(m, n_post), transpose=True).T
        y_ref = matrix_fcn(x, w, indices, (m, n_post))
    else:
        x = jnp.arange(n_post * k, dtype=jnp.float32).reshape(n_post, k) / 10
        y = fcnmm(w, indices, x, shape=(m, n_post), transpose=False)
        y_ref = fcn_matrix(x, w, indices, (m, n_post))
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
