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

"""Tests for the CSC column-scatter event-driven matmat ``csc_binary_matmat``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._fcn.binary import csc_binary_matmat, csc_binary_matvec


def _csc(rng, num_pre, num_post, per_col):
    """Build a CSC structure: ``indices`` (row ids) and ``indptr`` (col bounds)."""
    indptr = [0]
    indices = []
    for _ in range(num_post):
        rows = rng.choice(num_pre, size=per_col, replace=False)
        indices.extend(int(r) for r in rows)
        indptr.append(indptr[-1] + per_col)
    return (
        jnp.asarray(np.asarray(indices, np.int32)),
        jnp.asarray(np.asarray(indptr, np.int32)),
    )


def _dense(weights, indices, indptr, num_pre, num_post):
    W = np.zeros((num_pre, num_post), np.float64)
    homo = weights.size == 1
    for col in range(num_post):
        for pos in range(int(indptr[col]), int(indptr[col + 1])):
            W[int(indices[pos]), col] += float(weights[0] if homo else weights[pos])
    return jnp.asarray(W, jnp.float32)


def _mask(matrix):
    m = np.asarray(matrix)
    return jnp.asarray((m > 0) if m.dtype != np.bool_ else m, dtype=jnp.float32)


@pytest.mark.parametrize("homo", [True, False])
@pytest.mark.parametrize("ev", [jnp.bool_, jnp.float32])
@pytest.mark.parametrize("n", [1, 4])
def test_csc_matmat_matches_dense(homo, ev, n):
    rng = np.random.default_rng(0)
    num_pre, num_post, per_col = 6, 5, 2
    indices, indptr = _csc(rng, num_pre, num_post, per_col)
    nse = int(indices.shape[0])
    weights = jnp.ones(1, jnp.float32) if homo else jnp.asarray(rng.random(nse) + 0.5, jnp.float32)
    matrix = jnp.asarray(rng.random((num_post, n)) > 0.5, dtype=ev)
    got = csc_binary_matmat(weights, indices, indptr, matrix, shape=(num_pre, num_post))
    ref = _dense(weights, indices, indptr, num_pre, num_post) @ _mask(matrix)
    assert got.shape == (num_pre, n)
    assert jnp.allclose(got, ref, atol=1e-5), (homo, str(ev), n)


def test_csc_matmat_single_column_equals_matvec():
    rng = np.random.default_rng(1)
    num_pre, num_post, per_col = 6, 5, 2
    indices, indptr = _csc(rng, num_pre, num_post, per_col)
    nse = int(indices.shape[0])
    weights = jnp.asarray(rng.random(nse) + 0.5, jnp.float32)
    spikes = jnp.asarray(rng.random(num_post) > 0.5, dtype=jnp.bool_)
    got = csc_binary_matmat(weights, indices, indptr, spikes[:, None], shape=(num_pre, num_post))
    ref = csc_binary_matvec(weights, indices, indptr, spikes, shape=(num_pre, num_post))
    assert jnp.allclose(got[:, 0], ref, atol=1e-5)


@pytest.mark.parametrize("homo", [True, False])
def test_csc_matmat_grad_weights(homo):
    rng = np.random.default_rng(2)
    num_pre, num_post, per_col, n = 6, 5, 2, 4
    indices, indptr = _csc(rng, num_pre, num_post, per_col)
    nse = int(indices.shape[0])
    weights = jnp.asarray([0.7], jnp.float32) if homo else jnp.asarray(rng.random(nse) + 0.5, jnp.float32)
    matrix = jnp.asarray(rng.random((num_post, n)) > 0.5, dtype=jnp.float32)

    def f(w):
        return csc_binary_matmat(w, indices, indptr, matrix, shape=(num_pre, num_post)).sum()

    g = jax.grad(f)(weights)
    # Reference: gradient via dense reconstruction is the per-position activity sum.
    mask = _mask(matrix)
    col_ids = np.repeat(np.arange(num_post), np.diff(np.asarray(indptr)))
    per_pos = np.asarray(mask)[col_ids].sum(axis=1)  # [NNZ]
    gref = float(per_pos.sum()) if homo else jnp.asarray(per_pos, jnp.float32)
    assert jnp.allclose(g, gref, atol=1e-4), homo


def test_csc_matmat_grad_matrix():
    rng = np.random.default_rng(3)
    num_pre, num_post, per_col, n = 6, 5, 2, 4
    indices, indptr = _csc(rng, num_pre, num_post, per_col)
    nse = int(indices.shape[0])
    weights = jnp.asarray(rng.random(nse) + 0.5, jnp.float32)
    matrix = jnp.asarray(rng.random((num_post, n)), dtype=jnp.float32)

    def f(m):
        return csc_binary_matmat(weights, indices, indptr, m, shape=(num_pre, num_post)).sum()

    g = jax.grad(f)(matrix)
    # dL/dM = W^T @ ones: each (col, j) gets sum of weights scattered from col.
    W = np.asarray(_dense(weights, indices, indptr, num_pre, num_post))
    gref = jnp.asarray(W.T @ np.ones((num_pre, n), np.float32), jnp.float32)
    assert jnp.allclose(g, gref, atol=1e-4)


def test_csc_matmat_check_grads():
    from jax.test_util import check_grads
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(4)
        num_pre, num_post, per_col, n = 6, 5, 2, 3
        indices, indptr = _csc(rng, num_pre, num_post, per_col)
        nse = int(indices.shape[0])
        weights = jnp.asarray(rng.random(nse) + 0.5, jnp.float64)
        matrix = jnp.asarray(rng.random((num_post, n)) > 0.5, dtype=jnp.bool_)

        def f(w):
            return csc_binary_matmat(w, indices, indptr, matrix, shape=(num_pre, num_post)).sum()

        check_grads(f, (weights,), order=2, modes=['rev'])
    finally:
        jax.config.update("jax_enable_x64", False)


def test_csc_matmat_jit():
    rng = np.random.default_rng(5)
    num_pre, num_post, per_col, n = 6, 5, 2, 4
    indices, indptr = _csc(rng, num_pre, num_post, per_col)
    nse = int(indices.shape[0])
    weights = jnp.asarray(rng.random(nse) + 0.5, jnp.float32)
    matrix = jnp.asarray(rng.random((num_post, n)) > 0.5, dtype=jnp.bool_)
    f = jax.jit(lambda w: csc_binary_matmat(w, indices, indptr, matrix, shape=(num_pre, num_post)))
    got = f(weights)
    ref = _dense(weights, indices, indptr, num_pre, num_post) @ _mask(matrix)
    assert jnp.allclose(got, ref, atol=1e-5)


def test_csc_matmat_cuda_kernel_selects_col_scatter_names():
    import inspect
    import brainevent._fcn.binary as mod
    src = inspect.getsource(mod._csc_binary_matmat_cuda_kernel)
    # The builder selects the col-scatter FFI family and transposes the matrix.
    assert "binary_fcnmm_col_scatter" in src
    assert "matrix.T" in src

    from pathlib import Path
    cu = Path(mod.__file__).with_name("binary_fcnmm_col_scatter.cu").read_text()
    assert "binary_fcnmm_col_scatter_homo_bool_f32" in cu
    assert "binary_fcnmm_col_scatter_hetero_bool_f32" in cu
    assert "binary_fcnmm_col_scatter_homo_float_f32" in cu
    assert "binary_fcnmm_col_scatter_hetero_float_f32" in cu


def test_csc_matmat_import_needs_no_nvcc():
    import importlib
    import brainevent._fcn.binary as mod
    importlib.reload(mod)
    assert hasattr(mod, "csc_binary_matmat")
