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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._event.compact import _compact_1d_jax
from brainevent._event.compact_binary import CompactBinary
from brainevent._fcn.compact_binary import (
    compact_binary_fcnmv,
    compact_binary_fcnmm,
)
from brainevent._fcn.float import fcnmv, fcnmm
from brainevent._test_util import generate_fixed_conn_num_indices

platform = jax.default_backend()

# Smaller shapes on CPU to keep tests fast
if platform == 'cpu':
    SHAPES = [(20, 40), (30, 30), (40, 20)]
else:
    SHAPES = [(20, 40), (200, 400)]

N_CONN = 4
BATCH = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_indices(shape, n_conn=N_CONN):
    m, n = shape
    return generate_fixed_conn_num_indices(m, n, min(n_conn, n))


def _mk_homo_w(dtype=jnp.float32):
    return jnp.array([1.5], dtype=dtype)


def _mk_hetero_w(indices, dtype=jnp.float32, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(indices.shape).astype(np.float32), dtype=dtype)


def _mk_spikes(size, p=0.5, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random(size) < p, dtype=jnp.float32)


def _mk_matrix(rows, cols, p=0.5, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random((rows, cols)) < p, dtype=jnp.float32)


def _dense_mat(weights, indices, shape):
    """Build (n_pre, n_post) dense weight matrix from FCN format."""
    n_pre, n_post = shape
    rows = jnp.repeat(jnp.arange(n_pre, dtype=indices.dtype), indices.shape[1])
    cols = indices.reshape(-1)
    w = jnp.asarray(weights)
    if w.size == 1:
        vals = jnp.full(indices.size, w.reshape(())[()], dtype=w.dtype)
    else:
        vals = w.reshape(-1)
    return jnp.zeros(shape, dtype=w.dtype).at[rows, cols].add(vals)


def _ref_mv(weights, indices, spikes, shape, transpose):
    """Reference dense matmul for MV."""
    dense = _dense_mat(weights, indices, shape)
    s = jnp.asarray(spikes > 0, dtype=dense.dtype)
    if transpose:
        return s @ dense  # (n_pre,) @ (n_pre, n_post) -> (n_post,)
    return dense @ s  # (n_pre, n_post) @ (n_post,) -> (n_pre,)


def _ref_mm(weights, indices, matrix, shape, transpose):
    """Reference dense matmul for MM."""
    dense = _dense_mat(weights, indices, shape)
    M = jnp.asarray(matrix > 0, dtype=dense.dtype)
    if transpose:
        return dense.T @ M  # (n_post, n_pre) @ (n_pre, k) -> (n_post, k)
    return dense @ M  # (n_pre, n_post) @ (n_post, k) -> (n_pre, k)


def generate_cs_pairs(
    memory_limit: float = 5,
    homo_or_not: bool = True,
    scale_max: int = 2000,
    conn_max: int = 4000,
    _N: int = 4000,
    data_size: int = 4,
    num_points: int = 5,
    include_dense_ref: bool = False,
):
    """Generate ``(conn, m)`` pairs near the GPU memory boundary.

    Returns a list of ``(conn, m)`` tuples where ``m = scale * _N``, such
    that the estimated memory usage stays just below *memory_limit* GiB.
    Integer truncation (``int()``) is used so that values always lean
    toward the **smaller** side to avoid overflow.

    Memory model
    -------------
    * **Sparse arrays only** (``include_dense_ref=False``):
      ``m * conn * data_size * times``
      where *times* = 1 (homo, indices only) or 2 (hetero, indices + weights).

    * **With dense reference** (``include_dense_ref=True``):
      ``m * conn * data_size * times  +  m² * data_size``
      Use this when the test constructs a full ``(m, m)`` dense matrix for
      correctness comparison.

    Parameters
    ----------
    memory_limit : float
        GPU memory budget in GiB.
    homo_or_not : bool
        True  → only indices are stored (1× sparse budget).
        False → indices + weights are stored (2× sparse budget).
    scale_max : int
        Upper bound for the scale factor *s* (``m = s * _N``).
    conn_max : int
        Maximum number of connections per row.
    _N : int
        Base neuron count per scale unit.
    data_size : int
        Bytes per element (4 for float32 / int32).
    num_points : int
        Number of sample points to generate.
    include_dense_ref : bool
        If True, the budget also accounts for a dense ``(m, m)`` reference
        matrix of size ``m² * data_size``.
    """ 
    import math
    import numpy as np
    limit_bytes = memory_limit * (1024 ** 3)
    times = 1 if homo_or_not else 2

    # K = budget expressed in units of (_N * data_size) bytes
    K = limit_bytes / (_N * data_size)

    # --- valid scale range ---------------------------------------------------
    if include_dense_ref:
        # For conn > 0 we need  K - s² * _N > 0  →  s < sqrt(K / _N)
        s_max = min(scale_max, int(math.sqrt(K / _N)))
    else:
        # conn = K / (s * times) >= 1  →  s <= K / times
        s_max = min(scale_max, int(K / times))

    # s_min: when *not* including the dense ref, large s → small conn,
    # which is fine; but very small s → conn > conn_max, wasting budget.
    if not include_dense_ref:
        s_min = max(1, math.ceil(K / (times * conn_max)))
    else:
        s_min = 1

    if s_min > s_max:
        raise ValueError(
            f"No valid (conn, m) pairs: s_min={s_min} > s_max={s_max}. "
            f"Try increasing memory_limit or decreasing scale_max/conn_max."
        )

    s_samples = np.geomspace(s_min, s_max, num_points)

    valid_pairs = []
    seen = set()

    for s_val in s_samples:
        s_int = max(1, int(round(s_val)))
        s_int = min(s_int, s_max)  # clamp

        # Compute max conn (floor to stay below the boundary)
        if include_dense_ref:
            budget_for_sparse = K - s_int ** 2 * _N
            if budget_for_sparse <= 0:
                continue
            c_int = int(budget_for_sparse / (s_int * times))
        else:
            c_int = int(K / (s_int * times))

        c_int = min(c_int, conn_max)

        m = s_int * _N

        if c_int > 0 and c_int <= m and s_int <= scale_max:
            pair = (c_int, m)
            if pair not in seen:
                seen.add(pair)
                valid_pairs.append(pair)

    return valid_pairs

# ===========================================================================
# 1. Forward correctness -- fcnmv
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', SHAPES)
def test_compact_binary_fcnmv_forward(homo_w, transpose, shape):
    """compact_binary_fcnmv forward output matches dense reference."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)

    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    cb = CompactBinary.from_array(spikes)

    y = compact_binary_fcnmv(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose,
    )
    y_ref = _ref_mv(weights, indices, spikes, shape, transpose)

    assert y.shape == y_ref.shape
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3), (
        f"max diff={jnp.max(jnp.abs(y - y_ref)):.4e}  shape={shape}  "
        f"homo_w={homo_w}  transpose={transpose}"
    )


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', SHAPES)
def test_compact_binary_fcnmv_forward_all_zeros(homo_w, transpose, shape):
    """All-zero spikes -> zero output."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = jnp.zeros(spike_size, dtype=jnp.float32)
    cb = CompactBinary.from_array(spikes)

    y = compact_binary_fcnmv(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose,
    )
    assert jnp.allclose(y, jnp.zeros_like(y))


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', SHAPES)
def test_compact_binary_fcnmv_forward_all_ones(homo_w, transpose, shape):
    """All-one spikes -> same as dense row/col sums."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = jnp.ones(spike_size, dtype=jnp.float32)
    cb = CompactBinary.from_array(spikes)

    y = compact_binary_fcnmv(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose,
    )
    y_ref = _ref_mv(weights, indices, spikes, shape, transpose)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
def test_compact_binary_fcnmv_forward_in_large_scale(homo_w):
    """compact_binary_fcnmv forward matches dense reference at large scale."""
    import gc

    for conn, m in generate_cs_pairs(homo_or_not=homo_w, include_dense_ref=True):
        indices = generate_fixed_conn_num_indices(m, m, conn)
        weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)

        transpose = True
        shape = (m, m)
        spikes = _mk_spikes(m)
        cb = CompactBinary.from_array(spikes)

        y = compact_binary_fcnmv(
            weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
            shape=shape, transpose=transpose,
        )
        y_ref = _ref_mv(weights, indices, spikes, shape, transpose)

        assert y.shape == y_ref.shape
        assert jnp.allclose(y, y_ref, rtol=5e-2, atol=5e-2), (
            f"max diff={jnp.max(jnp.abs(y - y_ref)):.4e}  shape={shape}  "
            f"homo_w={homo_w}  conn={conn}"
        )
        jax.block_until_ready((indices, weights, y, y_ref))

        del indices, weights, spikes, cb, y, y_ref
        gc.collect()


# ===========================================================================
# 2. Forward correctness -- fcnmm
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', SHAPES)
def test_compact_binary_fcnmm_forward(homo_w, transpose, shape):
    """compact_binary_fcnmm forward output matches dense reference."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)

    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    cb = CompactBinary.from_array(matrix)

    y = compact_binary_fcnmm(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose, pack_axis=1,
    )
    y_ref = _ref_mm(weights, indices, matrix, shape, transpose)

    assert y.shape == y_ref.shape
    assert jnp.allclose(y, y_ref, rtol=2e-3, atol=2e-3), (
        f"max diff={jnp.max(jnp.abs(y - y_ref)):.4e}  shape={shape}  "
        f"homo_w={homo_w}  transpose={transpose}"
    )


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_compact_binary_fcnmm_forward_all_zeros(homo_w, transpose):
    """All-zero matrix -> zero output for MM."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = jnp.zeros((source_rows, BATCH), dtype=jnp.float32)
    cb = CompactBinary.from_array(matrix)

    y = compact_binary_fcnmm(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose, pack_axis=1,
    )
    assert jnp.allclose(y, jnp.zeros_like(y))


# ===========================================================================
# 3. JVP (forward-mode autodiff) -- fcnmv
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_compact_binary_fcnmv_jvp_weights(homo_w, transpose, shape):
    """JVP w.r.t. weights: tangent should equal compact_binary result with w_dot as weights."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    cb = CompactBinary.from_array(spikes)

    w_dot = jnp.ones_like(weights) * 0.1

    f = lambda w: compact_binary_fcnmv(
        w, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose,
    )
    primals_out, tangents_out = jax.jvp(f, (weights,), (w_dot,))

    # JVP rule: tangent = compact_binary_fcnmv with w_dot replacing weights
    expected_tangent = compact_binary_fcnmv(
        w_dot, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose,
    )
    assert jnp.allclose(tangents_out, expected_tangent, rtol=1e-3, atol=1e-3)
    # Primals should still be correct forward pass
    y_ref = _ref_mv(weights, indices, spikes, shape, transpose)
    assert jnp.allclose(primals_out, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_compact_binary_fcnmv_jvp_spikes(homo_w, transpose, shape):
    """JVP w.r.t. spikes: tangent should equal float fcnmv with spk_dot."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    cb = CompactBinary.from_array(spikes)

    spk_dot = jnp.ones_like(spikes) * 0.5

    f = lambda s: compact_binary_fcnmv(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, s,
        shape=shape, transpose=transpose,
    )
    primals_out, tangents_out = jax.jvp(f, (spikes,), (spk_dot,))

    # JVP rule for spikes: tangent = fcnmv(weights, indices, spk_dot, ...)
    expected_tangent = fcnmv(weights, indices, spk_dot, shape=shape, transpose=transpose)
    assert jnp.allclose(tangents_out, expected_tangent, rtol=1e-3, atol=1e-3)


# ===========================================================================
# 4. JVP (forward-mode autodiff) -- fcnmm
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_compact_binary_fcnmm_jvp_weights(homo_w, transpose, shape):
    """JVP w.r.t. weights for MM: tangent = compact_binary_mm with w_dot."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    cb = CompactBinary.from_array(matrix)

    w_dot = jnp.ones_like(weights) * 0.1

    f = lambda w: compact_binary_fcnmm(
        w, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose, pack_axis=1,
    )
    primals_out, tangents_out = jax.jvp(f, (weights,), (w_dot,))

    expected_tangent = compact_binary_fcnmm(
        w_dot, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose, pack_axis=1,
    )
    assert jnp.allclose(tangents_out, expected_tangent, rtol=1e-3, atol=1e-3)
    y_ref = _ref_mm(weights, indices, matrix, shape, transpose)
    assert jnp.allclose(primals_out, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_compact_binary_fcnmm_jvp_matrix(homo_w, transpose, shape):
    """JVP w.r.t. matrix for MM: tangent = float fcnmm with m_dot."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    cb = CompactBinary.from_array(matrix)

    m_dot = jnp.ones_like(matrix) * 0.5

    f = lambda mat: compact_binary_fcnmm(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, mat,
        shape=shape, transpose=transpose, pack_axis=1,
    )
    primals_out, tangents_out = jax.jvp(f, (matrix,), (m_dot,))

    # JVP rule: tangent = fcnmm(weights, indices, m_dot, ...)
    expected_tangent = fcnmm(weights, indices, m_dot, shape=shape, transpose=transpose)
    assert jnp.allclose(tangents_out, expected_tangent, rtol=1e-3, atol=1e-3)


# ===========================================================================
# 5. VJP / grad (reverse-mode autodiff) -- fcnmv
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_compact_binary_fcnmv_vjp_weights(homo_w, transpose, shape):
    """VJP w.r.t. weights: compare against expected formula."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    cb = CompactBinary.from_array(spikes)

    out_size = n if transpose else m
    ct = jnp.ones(out_size, dtype=jnp.float32)

    f = lambda w: compact_binary_fcnmv(
        w, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose,
    )
    _, vjp_fn = jax.vjp(f, weights)
    (ct_w,) = vjp_fn(ct)

    assert ct_w.shape == weights.shape

    # Verify against brute-force dense gradient
    f_dense = lambda w: _ref_mv(w, indices, spikes, shape, transpose)
    _, vjp_dense = jax.vjp(f_dense, weights)
    (ct_w_dense,) = vjp_dense(ct)
    assert jnp.allclose(ct_w, ct_w_dense, rtol=1e-3, atol=1e-3), (
        f"max diff={jnp.max(jnp.abs(ct_w - ct_w_dense)):.4e}"
    )


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_compact_binary_fcnmv_vjp_spikes(homo_w, transpose, shape):
    """VJP w.r.t. spikes: ct_spk = fcnmv(w, idx, ct, shape, not transpose)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    cb = CompactBinary.from_array(spikes)

    out_size = n if transpose else m
    ct = jnp.ones(out_size, dtype=jnp.float32)

    f = lambda s: compact_binary_fcnmv(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, s,
        shape=shape, transpose=transpose,
    )
    _, vjp_fn = jax.vjp(f, spikes)
    (ct_spk,) = vjp_fn(ct)

    # Expected: fcnmv(weights, indices, ct, shape=shape, transpose=not transpose)
    expected = fcnmv(weights, indices, ct, shape=shape, transpose=not transpose)
    assert ct_spk.shape == spikes.shape
    assert jnp.allclose(ct_spk, expected, rtol=1e-3, atol=1e-3), (
        f"max diff={jnp.max(jnp.abs(ct_spk - expected)):.4e}"
    )


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_compact_binary_fcnmv_grad_weights(homo_w, transpose, shape):
    """jax.grad of sum(output) w.r.t. weights."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    cb = CompactBinary.from_array(spikes)

    f = lambda w: jnp.sum(
        compact_binary_fcnmv(
            w, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
            shape=shape, transpose=transpose,
        )
    )
    g = jax.grad(f)(weights)
    assert g.shape == weights.shape
    assert jnp.isfinite(g).all()


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_compact_binary_fcnmv_grad_spikes(homo_w, transpose, shape):
    """jax.grad of sum(output) w.r.t. spikes."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    cb = CompactBinary.from_array(spikes)

    f = lambda s: jnp.sum(
        compact_binary_fcnmv(
            weights, indices, cb.packed, cb.active_ids, cb.n_active, s,
            shape=shape, transpose=transpose,
        )
    )
    g = jax.grad(f)(spikes)
    assert g.shape == spikes.shape
    assert jnp.isfinite(g).all()


# ===========================================================================
# 6. VJP / grad (reverse-mode autodiff) -- fcnmm
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_compact_binary_fcnmm_vjp_weights(homo_w, transpose, shape):
    """VJP w.r.t. weights for MM: compare against dense gradient."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    cb = CompactBinary.from_array(matrix)

    out_rows = n if transpose else m
    ct = jnp.ones((out_rows, BATCH), dtype=jnp.float32)

    f = lambda w: compact_binary_fcnmm(
        w, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose, pack_axis=1,
    )
    _, vjp_fn = jax.vjp(f, weights)
    (ct_w,) = vjp_fn(ct)

    assert ct_w.shape == weights.shape

    f_dense = lambda w: _ref_mm(w, indices, matrix, shape, transpose)
    _, vjp_dense = jax.vjp(f_dense, weights)
    (ct_w_dense,) = vjp_dense(ct)
    assert jnp.allclose(ct_w, ct_w_dense, rtol=1e-3, atol=1e-3), (
        f"max diff={jnp.max(jnp.abs(ct_w - ct_w_dense)):.4e}"
    )


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_compact_binary_fcnmm_vjp_matrix(homo_w, transpose, shape):
    """VJP w.r.t. matrix for MM: ct_mat = fcnmm(w, idx, ct, shape, not transpose)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    cb = CompactBinary.from_array(matrix)

    out_rows = n if transpose else m
    ct = jnp.ones((out_rows, BATCH), dtype=jnp.float32)

    f = lambda mat: compact_binary_fcnmm(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, mat,
        shape=shape, transpose=transpose, pack_axis=1,
    )
    _, vjp_fn = jax.vjp(f, matrix)
    (ct_mat,) = vjp_fn(ct)

    # Expected: fcnmm(weights, indices, ct, shape=shape, transpose=not transpose)
    expected = fcnmm(weights, indices, ct, shape=shape, transpose=not transpose)
    assert ct_mat.shape == matrix.shape
    assert jnp.allclose(ct_mat, expected, rtol=1e-3, atol=1e-3), (
        f"max diff={jnp.max(jnp.abs(ct_mat - expected)):.4e}"
    )


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_compact_binary_fcnmm_grad_weights(homo_w, transpose):
    """jax.grad of sum(output) w.r.t. weights for MM."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    cb = CompactBinary.from_array(matrix)

    f = lambda w: jnp.sum(
        compact_binary_fcnmm(
            w, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
            shape=shape, transpose=transpose, pack_axis=1,
        )
    )
    g = jax.grad(f)(weights)
    assert g.shape == weights.shape
    assert jnp.isfinite(g).all()


# ===========================================================================
# 7. vmap -- fcnmv
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_compact_binary_fcnmv_vmap_over_spikes(homo_w, transpose, shape):
    """vmap(compact_binary_fcnmv) over a batch of spike vectors."""
    m, n = shape
    B = 5
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n

    # Create a batch of spike vectors
    rng = np.random.default_rng(7)
    batch_spikes = jnp.asarray(rng.random((B, spike_size)) < 0.5, dtype=jnp.float32)
    batch_cbs = jax.vmap(CompactBinary.from_array)(batch_spikes)

    f = lambda packed, active_ids, n_active, value: compact_binary_fcnmv(
        weights, indices, packed, active_ids, n_active, value,
        shape=shape, transpose=transpose,
    )
    y_vmap = jax.vmap(f)(
        batch_cbs.packed, batch_cbs.active_ids, batch_cbs.n_active, batch_cbs.value,
    )

    # Compare against loop — compute per-element active_ids independently.
    # Under vmap, the 1D compaction batching rule produces merged (row-level)
    # active_ids, which are not per-element correct for MV scatter.
    # The MV batching rule recomputes them, so vmap results are correct.
    y_loop = jnp.stack([
        compact_binary_fcnmv(
            weights, indices,
            jax.tree.map(lambda x: x[i], batch_cbs).packed,
            *_compact_1d_jax(batch_spikes[i], jax_impl=True),
            jax.tree.map(lambda x: x[i], batch_cbs).value,
            shape=shape, transpose=transpose,
        )
        for i in range(B)
    ])
    assert y_vmap.shape == y_loop.shape
    assert jnp.allclose(y_vmap, y_loop, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_compact_binary_fcnmv_vmap_grad(homo_w, transpose):
    """vmap(grad(compact_binary_fcnmv)) over a batch of spike vectors."""
    shape = (20, 40)
    m, n = shape
    B = 4
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n

    rng = np.random.default_rng(8)
    batch_spikes = jnp.asarray(rng.random((B, spike_size)) < 0.5, dtype=jnp.float32)
    batch_cbs = jax.vmap(CompactBinary.from_array)(batch_spikes)

    def loss(packed, active_ids, n_active, value):
        return jnp.sum(
            compact_binary_fcnmv(
                weights, indices, packed, active_ids, n_active, value,
                shape=shape, transpose=transpose,
            )
        )

    # grad w.r.t. value (spikes), vmapped
    grad_fn = jax.grad(loss, argnums=3)
    grads = jax.vmap(grad_fn)(
        batch_cbs.packed, batch_cbs.active_ids, batch_cbs.n_active, batch_cbs.value,
    )
    assert grads.shape == batch_spikes.shape
    assert jnp.isfinite(grads).all()


# ===========================================================================
# 8. vmap -- fcnmm
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_compact_binary_fcnmm_vmap_over_matrix(homo_w, transpose, shape):
    """vmap(compact_binary_fcnmm) over a batch of matrices."""
    m, n = shape
    B = 3
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n

    rng = np.random.default_rng(9)
    # batch of matrices: (B, source_rows, BATCH)
    batch_matrix = jnp.asarray(rng.random((B, source_rows, BATCH)) < 0.5, dtype=jnp.float32)
    batch_cbs = jax.vmap(CompactBinary.from_array)(batch_matrix)

    f = lambda packed, active_ids, n_active, value: compact_binary_fcnmm(
        weights, indices, packed, active_ids, n_active, value,
        shape=shape, transpose=transpose, pack_axis=1,
    )
    y_vmap = jax.vmap(f)(
        batch_cbs.packed, batch_cbs.active_ids, batch_cbs.n_active, batch_cbs.value,
    )

    # Compare against loop
    y_loop = jnp.stack([
        compact_binary_fcnmm(
            weights, indices,
            jax.tree.map(lambda x: x[i], batch_cbs).packed,
            jax.tree.map(lambda x: x[i], batch_cbs).active_ids,
            jax.tree.map(lambda x: x[i], batch_cbs).n_active,
            jax.tree.map(lambda x: x[i], batch_cbs).value,
            shape=shape, transpose=transpose, pack_axis=1,
        )
        for i in range(B)
    ])
    assert y_vmap.shape == y_loop.shape
    assert jnp.allclose(y_vmap, y_loop, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
def test_compact_binary_fcnmm_vmap_grad(homo_w):
    """vmap(grad(compact_binary_fcnmm)) over a batch of matrices."""
    shape = (20, 40)
    m, n = shape
    B = 3
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = n  # gather mode

    rng = np.random.default_rng(10)
    batch_matrix = jnp.asarray(rng.random((B, source_rows, BATCH)) < 0.5, dtype=jnp.float32)
    batch_cbs = jax.vmap(CompactBinary.from_array)(batch_matrix)

    def loss(packed, active_ids, n_active, value):
        return jnp.sum(
            compact_binary_fcnmm(
                weights, indices, packed, active_ids, n_active, value,
                shape=shape, transpose=False, pack_axis=1,
            )
        )

    grad_fn = jax.grad(loss, argnums=3)
    grads = jax.vmap(grad_fn)(
        batch_cbs.packed, batch_cbs.active_ids, batch_cbs.n_active, batch_cbs.value,
    )
    assert grads.shape == batch_matrix.shape
    assert jnp.isfinite(grads).all()


# ===========================================================================
# 9. jax.jit compatibility
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_compact_binary_fcnmv_jit(homo_w, transpose):
    """compact_binary_fcnmv runs correctly under jax.jit."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    cb = CompactBinary.from_array(spikes)

    f_jit = jax.jit(
        lambda w, packed, active_ids, n_active, value: compact_binary_fcnmv(
            w, indices, packed, active_ids, n_active, value,
            shape=shape, transpose=transpose,
        ),
    )
    y_jit = f_jit(weights, cb.packed, cb.active_ids, cb.n_active, cb.value)
    y_ref = _ref_mv(weights, indices, spikes, shape, transpose)
    assert jnp.allclose(y_jit, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_compact_binary_fcnmm_jit(homo_w, transpose):
    """compact_binary_fcnmm runs correctly under jax.jit."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    cb = CompactBinary.from_array(matrix)

    f_jit = jax.jit(
        lambda w, packed, active_ids, n_active, value: compact_binary_fcnmm(
            w, indices, packed, active_ids, n_active, value,
            shape=shape, transpose=transpose, pack_axis=1,
        ),
    )
    y_jit = f_jit(weights, cb.packed, cb.active_ids, cb.n_active, cb.value)
    y_ref = _ref_mm(weights, indices, matrix, shape, transpose)
    assert jnp.allclose(y_jit, y_ref, rtol=1e-3, atol=1e-3)


# ===========================================================================
# 10. brainunit Quantity support
# ===========================================================================

@pytest.mark.parametrize('transpose', [True, False])
def test_compact_binary_fcnmv_quantity_homo(transpose):
    """compact_binary_fcnmv handles brainunit Quantity weights (homo)."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    cb = CompactBinary.from_array(spikes)

    w_q = jnp.array([1.5]) * u.mS  # homogeneous weights with unit
    y = compact_binary_fcnmv(
        w_q, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose,
    )

    # Should return a Quantity
    assert isinstance(y, u.Quantity)
    y_plain = compact_binary_fcnmv(
        jnp.array([1.5]), indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose,
    )
    assert jnp.allclose(u.get_mantissa(y), y_plain, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('transpose', [True, False])
def test_compact_binary_fcnmm_quantity_homo(transpose):
    """compact_binary_fcnmm handles brainunit Quantity weights (homo)."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    cb = CompactBinary.from_array(matrix)

    w_q = jnp.array([1.5]) * u.mS
    y = compact_binary_fcnmm(
        w_q, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose, pack_axis=1,
    )

    assert isinstance(y, u.Quantity)
    y_plain = compact_binary_fcnmm(
        jnp.array([1.5]), indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose, pack_axis=1,
    )
    assert jnp.allclose(u.get_mantissa(y), y_plain, rtol=1e-3, atol=1e-3)


# ===========================================================================
# 11. Output shape checks
# ===========================================================================

@pytest.mark.parametrize('shape', [(20, 40), (30, 30), (40, 20)])
def test_compact_binary_fcnmv_output_shape_gather(shape):
    """Gather mode output has shape (n_pre,)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    spikes = _mk_spikes(n)
    cb = CompactBinary.from_array(spikes)
    y = compact_binary_fcnmv(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=False,
    )
    assert y.shape == (m,)


@pytest.mark.parametrize('shape', [(20, 40), (30, 30), (40, 20)])
def test_compact_binary_fcnmv_output_shape_scatter(shape):
    """Scatter mode output has shape (n_post,)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    spikes = _mk_spikes(m)
    cb = CompactBinary.from_array(spikes)
    y = compact_binary_fcnmv(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=True,
    )
    assert y.shape == (n,)


@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_compact_binary_fcnmm_output_shape_gather(shape):
    """Gather mode MM output has shape (n_pre, n_batch)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    matrix = _mk_matrix(n, BATCH)
    cb = CompactBinary.from_array(matrix)
    y = compact_binary_fcnmm(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=False, pack_axis=1,
    )
    assert y.shape == (m, BATCH)


@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_compact_binary_fcnmm_output_shape_scatter(shape):
    """Scatter mode MM output has shape (n_post, n_batch)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    matrix = _mk_matrix(m, BATCH)
    cb = CompactBinary.from_array(matrix)
    y = compact_binary_fcnmm(
        weights, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=True, pack_axis=1,
    )
    assert y.shape == (n, BATCH)


# ===========================================================================
# 12. Consistency between MV and MM (single column MM == MV)
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_mv_mm_consistency(homo_w, transpose, shape):
    """Single-column fcnmm output matches fcnmv output."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)

    # MV
    cb_mv = CompactBinary.from_array(spikes)
    y_mv = compact_binary_fcnmv(
        weights, indices, cb_mv.packed, cb_mv.active_ids, cb_mv.n_active, cb_mv.value,
        shape=shape, transpose=transpose,
    )

    # MM with k=1
    matrix = spikes[:, None]  # (spike_size, 1)
    cb_mm = CompactBinary.from_array(matrix)
    y_mm = compact_binary_fcnmm(
        weights, indices, cb_mm.packed, cb_mm.active_ids, cb_mm.n_active, cb_mm.value,
        shape=shape, transpose=transpose, pack_axis=1,
    )

    assert jnp.allclose(y_mv, y_mm[:, 0], rtol=1e-3, atol=1e-3), (
        f"max diff={jnp.max(jnp.abs(y_mv - y_mm[:, 0])):.4e}"
    )


# ===========================================================================
# 13. Gradient consistency: MV grad matches dense reference via finite diff
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_compact_binary_fcnmv_grad_weights_finite_diff(homo_w, transpose):
    """VJP gradient w.r.t. weights matches finite-difference approximation."""
    shape = (10, 20)
    m, n = shape
    indices = _mk_indices(shape, n_conn=2)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices, seed=99)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size, seed=99)
    cb = CompactBinary.from_array(spikes)

    out_size = n if transpose else m
    ct = jnp.ones(out_size)

    f = lambda w: compact_binary_fcnmv(
        w, indices, cb.packed, cb.active_ids, cb.n_active, cb.value,
        shape=shape, transpose=transpose,
    )
    _, vjp_fn = jax.vjp(f, weights)
    (ct_w_analytic,) = vjp_fn(ct)

    # Finite difference
    eps = 1e-3
    ct_w_fd = jnp.zeros_like(weights)
    for idx in np.ndindex(*weights.shape):
        wp = weights.at[idx].add(eps)
        wm = weights.at[idx].add(-eps)
        fd = (jnp.dot(ct, f(wp)) - jnp.dot(ct, f(wm))) / (2 * eps)
        ct_w_fd = ct_w_fd.at[idx].set(fd)

    assert jnp.allclose(ct_w_analytic, ct_w_fd, rtol=1e-2, atol=1e-2), (
        f"analytic={ct_w_analytic}  fd={ct_w_fd}"
    )
