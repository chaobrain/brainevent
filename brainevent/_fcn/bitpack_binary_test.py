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

from brainevent._event.bitpack_binary import bitpack
from brainevent._fcn.bitpack_binary import (
    bitpack_binary_fcnmv,
    bitpack_binary_fcnmm,
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
        return s @ dense  # (n_pre,) @ (n_pre, n_post) → (n_post,)
    return dense @ s  # (n_pre, n_post) @ (n_post,) → (n_pre,)


def _ref_mm(weights, indices, matrix, shape, transpose):
    """Reference dense matmul for MM."""
    dense = _dense_mat(weights, indices, shape)
    M = jnp.asarray(matrix > 0, dtype=dense.dtype)
    if transpose:
        return dense.T @ M  # (n_post, n_pre) @ (n_pre, k) → (n_post, k)
    return dense @ M  # (n_pre, n_post) @ (n_post, k) → (n_pre, k)


def generate_cs_pairs(
    memory_limit: float = 5,
    homo_or_not: bool = True,
    scale_max: int = 2000,
    conn_max: int = 4000,
    _N: int = 4000,
    data_size: int = 4,
    num_points: int = 5,
    include_dense_ref: bool = False,
    max_conn: int = 3000,
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
        Maximum number of connections per row from the original search space.
    _N : int
        Base neuron count per scale unit.
    data_size : int
        Bytes per element (4 for float32 / int32).
    num_points : int
        Number of sample points to generate.
    include_dense_ref : bool
        If True, the budget also accounts for a dense ``(m, m)`` reference
        matrix of size ``m² * data_size``.
    max_conn : int
        Hard upper bound on returned ``conn`` values. Final ``conn`` will
        satisfy ``conn <= min(conn_max, max_conn)``.
    """
    import math
    import numpy as np

    limit_bytes = memory_limit * (1024 ** 3)
    times = 1 if homo_or_not else 2

    # Hard cap for conn
    conn_cap = min(conn_max, max_conn)
    if conn_cap < 1:
        raise ValueError(f"max_conn must be >= 1, got {max_conn}")

    # K = budget expressed in units of (_N * data_size) bytes
    K = limit_bytes / (_N * data_size)

    # --- valid scale range ---------------------------------------------------
    if include_dense_ref:
        # For conn > 0 we need  K - s² * _N > 0  →  s < sqrt(K / _N)
        s_max = min(scale_max, int(math.sqrt(K / _N)))
    else:
        # conn = K / (s * times) >= 1  →  s <= K / times
        s_max = min(scale_max, int(K / times))

    # s_min: when not including dense ref, small s may force conn too large.
    if not include_dense_ref:
        s_min = max(1, math.ceil(K / (times * conn_cap)))
    else:
        s_min = 1

    if s_min > s_max:
        raise ValueError(
            f"No valid (conn, m) pairs: s_min={s_min} > s_max={s_max}. "
            f"Try increasing memory_limit or decreasing scale_max/conn_max/max_conn."
        )

    s_samples = np.geomspace(s_min, s_max, num_points)

    valid_pairs = []
    seen = set()

    for s_val in s_samples:
        s_int = max(1, int(round(s_val)))
        s_int = min(s_int, s_max)

        # Compute max conn (floor to stay below the boundary)
        if include_dense_ref:
            budget_for_sparse = K - s_int ** 2 * _N
            if budget_for_sparse <= 0:
                continue
            c_int = int(budget_for_sparse / (s_int * times))
        else:
            c_int = int(K / (s_int * times))

        # Apply both original conn_max and new hard max_conn
        c_int = min(c_int, conn_cap)

        m = s_int * _N

        if c_int > 0 and c_int <= m and s_int <= scale_max:
            pair = (c_int, m)
            if pair not in seen:
                seen.add(pair)
                valid_pairs.append(pair)

    return valid_pairs

# ===========================================================================
# 1. Forward correctness — fcnmv
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', SHAPES)
def test_bitpack_fcnmv_forward(homo_w, transpose, shape):
    """bitpack_binary_fcnmv forward output matches dense reference."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)

    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    packed = bitpack(spikes, axis=0)

    y = bitpack_binary_fcnmv(
        weights, indices, packed, spikes, shape=shape, transpose=transpose,
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
def test_bitpack_fcnmv_forward_all_zeros(homo_w, transpose, shape):
    """All-zero spikes → zero output."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = jnp.zeros(spike_size, dtype=jnp.float32)
    packed = bitpack(spikes, axis=0)

    y = bitpack_binary_fcnmv(weights, indices, packed, spikes, shape=shape, transpose=transpose)
    assert jnp.allclose(y, jnp.zeros_like(y))


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', SHAPES)
def test_bitpack_fcnmv_forward_all_ones(homo_w, transpose, shape):
    """All-one spikes → same as dense row/col sums."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = jnp.ones(spike_size, dtype=jnp.float32)
    packed = bitpack(spikes, axis=0)

    y = bitpack_binary_fcnmv(weights, indices, packed, spikes, shape=shape, transpose=transpose)
    y_ref = _ref_mv(weights, indices, spikes, shape, transpose)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
def test_bitpack_fcnmv_forward_in_large_scale(homo_w):
    """bitpack_binary_fcnmv forward matches dense reference at large scale (transpose=False)."""
    import gc

    for conn, m in generate_cs_pairs(homo_or_not=homo_w, include_dense_ref=True):
        indices = generate_fixed_conn_num_indices(m, m, conn)
        weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)

        transpose = False
        shape = (m, m)
        spikes = _mk_spikes(m)
        packed = bitpack(spikes, axis=0)

        y = bitpack_binary_fcnmv(
            weights, indices, packed, spikes, shape=shape, transpose=transpose,
        )
        y_ref = _ref_mv(weights, indices, spikes, shape, transpose)

        assert y.shape == y_ref.shape
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3), (
            f"max diff={jnp.max(jnp.abs(y - y_ref)):.4e}  shape={shape}  "
            f"homo_w={homo_w}  conn={conn}"
        )
        jax.block_until_ready((indices, weights, y, y_ref))

        del indices, weights, spikes, packed, y, y_ref
        gc.collect()


# ===========================================================================
# 2. Forward correctness — fcnmm
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('pack_axis', [0, 1])
@pytest.mark.parametrize('shape', SHAPES)
def test_bitpack_fcnmm_forward(homo_w, transpose, pack_axis, shape):
    """bitpack_binary_fcnmm forward output matches dense reference."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)

    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    packed = bitpack(matrix, axis=pack_axis)

    y = bitpack_binary_fcnmm(
        weights, indices, packed, matrix, shape=shape, transpose=transpose, pack_axis=pack_axis,
    )
    y_ref = _ref_mm(weights, indices, matrix, shape, transpose)

    assert y.shape == y_ref.shape
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3), (
        f"max diff={jnp.max(jnp.abs(y - y_ref)):.4e}  shape={shape}  "
        f"homo_w={homo_w}  transpose={transpose}  pack_axis={pack_axis}"
    )


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_bitpack_fcnmm_forward_all_zeros(homo_w, transpose):
    """All-zero matrix → zero output for MM."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = jnp.zeros((source_rows, BATCH), dtype=jnp.float32)
    packed = bitpack(matrix, axis=1)

    y = bitpack_binary_fcnmm(weights, indices, packed, matrix, shape=shape, transpose=transpose)
    assert jnp.allclose(y, jnp.zeros_like(y))


# ===========================================================================
# 3. JVP (forward-mode autodiff) — fcnmv
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_bitpack_fcnmv_jvp_weights(homo_w, transpose, shape):
    """JVP w.r.t. weights: tangent should equal bitpack result with w_dot as weights."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    packed = bitpack(spikes, axis=0)

    w_dot = jnp.ones_like(weights) * 0.1

    f = lambda w: bitpack_binary_fcnmv(w, indices, packed, spikes,
                                       shape=shape, transpose=transpose)
    primals_out, tangents_out = jax.jvp(f, (weights,), (w_dot,))

    # JVP rule: tangent = bitpack_binary_fcnmv with w_dot replacing weights
    expected_tangent = bitpack_binary_fcnmv(
        w_dot, indices, packed, spikes, shape=shape, transpose=transpose
    )
    assert jnp.allclose(tangents_out, expected_tangent, rtol=1e-3, atol=1e-3)
    # Primals should still be correct forward pass
    y_ref = _ref_mv(weights, indices, spikes, shape, transpose)
    assert jnp.allclose(primals_out, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_bitpack_fcnmv_jvp_spikes(homo_w, transpose, shape):
    """JVP w.r.t. spikes: tangent should equal float fcnmv with spk_dot."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    packed = bitpack(spikes, axis=0)

    spk_dot = jnp.ones_like(spikes) * 0.5

    f = lambda s: bitpack_binary_fcnmv(weights, indices, packed, s,
                                       shape=shape, transpose=transpose)
    primals_out, tangents_out = jax.jvp(f, (spikes,), (spk_dot,))

    # JVP rule for spikes: tangent = fcnmv(weights, indices, spk_dot, ...)
    expected_tangent = fcnmv(weights, indices, spk_dot, shape=shape, transpose=transpose)
    assert jnp.allclose(tangents_out, expected_tangent, rtol=1e-3, atol=1e-3)


# ===========================================================================
# 4. JVP (forward-mode autodiff) — fcnmm
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_bitpack_fcnmm_jvp_weights(homo_w, transpose, shape):
    """JVP w.r.t. weights for MM: tangent = bitpack_mm with w_dot."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    packed = bitpack(matrix, axis=1)

    w_dot = jnp.ones_like(weights) * 0.1

    f = lambda w: bitpack_binary_fcnmm(w, indices, packed, matrix,
                                       shape=shape, transpose=transpose)
    primals_out, tangents_out = jax.jvp(f, (weights,), (w_dot,))

    expected_tangent = bitpack_binary_fcnmm(
        w_dot, indices, packed, matrix, shape=shape, transpose=transpose
    )
    assert jnp.allclose(tangents_out, expected_tangent, rtol=1e-3, atol=1e-3)
    y_ref = _ref_mm(weights, indices, matrix, shape, transpose)
    assert jnp.allclose(primals_out, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_bitpack_fcnmm_jvp_matrix(homo_w, transpose, shape):
    """JVP w.r.t. matrix for MM: tangent = float fcnmm with m_dot."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    packed = bitpack(matrix, axis=1)

    m_dot = jnp.ones_like(matrix) * 0.5

    f = lambda mat: bitpack_binary_fcnmm(weights, indices, packed, mat,
                                         shape=shape, transpose=transpose)
    primals_out, tangents_out = jax.jvp(f, (matrix,), (m_dot,))

    # JVP rule: tangent = fcnmm(weights, indices, m_dot, ...)
    expected_tangent = fcnmm(weights, indices, m_dot, shape=shape, transpose=transpose)
    assert jnp.allclose(tangents_out, expected_tangent, rtol=1e-3, atol=1e-3)


# ===========================================================================
# 5. VJP / grad (reverse-mode autodiff) — fcnmv
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_bitpack_fcnmv_vjp_weights(homo_w, transpose, shape):
    """VJP w.r.t. weights: compare against expected formula."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    packed = bitpack(spikes, axis=0)

    out_size = n if transpose else m
    ct = jnp.ones(out_size, dtype=jnp.float32)

    f = lambda w: bitpack_binary_fcnmv(w, indices, packed, spikes,
                                       shape=shape, transpose=transpose)
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
def test_bitpack_fcnmv_vjp_spikes(homo_w, transpose, shape):
    """VJP w.r.t. spikes: ct_spk = fcnmv(w, idx, ct, shape, not transpose)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    packed = bitpack(spikes, axis=0)

    out_size = n if transpose else m
    ct = jnp.ones(out_size, dtype=jnp.float32)

    f = lambda s: bitpack_binary_fcnmv(weights, indices, packed, s,
                                       shape=shape, transpose=transpose)
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
def test_bitpack_fcnmv_grad_weights(homo_w, transpose, shape):
    """jax.grad of sum(output) w.r.t. weights."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    packed = bitpack(spikes, axis=0)

    f = lambda w: jnp.sum(
        bitpack_binary_fcnmv(w, indices, packed, spikes, shape=shape, transpose=transpose)
    )
    g = jax.grad(f)(weights)
    assert g.shape == weights.shape
    assert jnp.isfinite(g).all()


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_bitpack_fcnmv_grad_spikes(homo_w, transpose, shape):
    """jax.grad of sum(output) w.r.t. spikes."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    packed = bitpack(spikes, axis=0)

    f = lambda s: jnp.sum(
        bitpack_binary_fcnmv(weights, indices, packed, s, shape=shape, transpose=transpose)
    )
    g = jax.grad(f)(spikes)
    assert g.shape == spikes.shape
    assert jnp.isfinite(g).all()


# ===========================================================================
# 6. VJP / grad (reverse-mode autodiff) — fcnmm
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_bitpack_fcnmm_vjp_weights(homo_w, transpose, shape):
    """VJP w.r.t. weights for MM: compare against dense gradient."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    packed = bitpack(matrix, axis=1)

    out_rows = n if transpose else m
    ct = jnp.ones((out_rows, BATCH), dtype=jnp.float32)

    f = lambda w: bitpack_binary_fcnmm(w, indices, packed, matrix,
                                       shape=shape, transpose=transpose)
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
def test_bitpack_fcnmm_vjp_matrix(homo_w, transpose, shape):
    """VJP w.r.t. matrix for MM: ct_mat = fcnmm(w, idx, ct, shape, not transpose)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    packed = bitpack(matrix, axis=1)

    out_rows = n if transpose else m
    ct = jnp.ones((out_rows, BATCH), dtype=jnp.float32)

    f = lambda mat: bitpack_binary_fcnmm(weights, indices, packed, mat,
                                         shape=shape, transpose=transpose)
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
def test_bitpack_fcnmm_grad_weights(homo_w, transpose):
    """jax.grad of sum(output) w.r.t. weights for MM."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    packed = bitpack(matrix, axis=1)

    f = lambda w: jnp.sum(
        bitpack_binary_fcnmm(w, indices, packed, matrix, shape=shape, transpose=transpose)
    )
    g = jax.grad(f)(weights)
    assert g.shape == weights.shape
    assert jnp.isfinite(g).all()


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_bitpack_fcnmm_grad_matrix(homo_w, transpose):
    """jax.grad of sum(output) w.r.t. matrix for MM."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    packed = bitpack(matrix, axis=1)

    f = lambda mat: jnp.sum(
        bitpack_binary_fcnmm(weights, indices, packed, mat, shape=shape, transpose=transpose)
    )
    g = jax.grad(f)(matrix)
    assert g.shape == matrix.shape
    assert jnp.isfinite(g).all()


# ===========================================================================
# 7. vmap — fcnmv
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_bitpack_fcnmv_vmap_over_spikes(homo_w, transpose, shape):
    """vmap(bitpack_binary_fcnmv) over a batch of spike vectors."""
    m, n = shape
    B = 5
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n

    # Create a batch of spike vectors
    rng = np.random.default_rng(7)
    batch_spikes = jnp.asarray(rng.random((B, spike_size)) < 0.5, dtype=jnp.float32)
    batch_packed = jax.vmap(lambda s: bitpack(s, axis=0))(batch_spikes)

    f = lambda p, s: bitpack_binary_fcnmv(
        weights, indices, p, s, shape=shape, transpose=transpose
    )
    y_vmap = jax.vmap(f)(batch_packed, batch_spikes)

    # Compare against loop
    y_loop = jnp.stack([
        bitpack_binary_fcnmv(
            weights, indices, batch_packed[i], batch_spikes[i],
            shape=shape, transpose=transpose
        )
        for i in range(B)
    ])
    assert y_vmap.shape == y_loop.shape
    assert jnp.allclose(y_vmap, y_loop, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_bitpack_fcnmv_vmap_grad(homo_w, transpose):
    """vmap(grad(bitpack_binary_fcnmv)) over a batch of spike vectors."""
    shape = (20, 40)
    m, n = shape
    B = 4
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n

    rng = np.random.default_rng(8)
    batch_spikes = jnp.asarray(rng.random((B, spike_size)) < 0.5, dtype=jnp.float32)
    batch_packed = jax.vmap(lambda s: bitpack(s, axis=0))(batch_spikes)

    def loss(p, s):
        return jnp.sum(
            bitpack_binary_fcnmv(weights, indices, p, s, shape=shape, transpose=transpose)
        )

    # grad w.r.t. spikes, vmapped
    grad_fn = jax.grad(loss, argnums=1)
    grads = jax.vmap(grad_fn)(batch_packed, batch_spikes)
    assert grads.shape == batch_spikes.shape
    assert jnp.isfinite(grads).all()


# ===========================================================================
# 8. vmap — fcnmm
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(20, 40)])
def test_bitpack_fcnmm_vmap_over_matrix(homo_w, transpose, shape):
    """vmap(bitpack_binary_fcnmm) over a batch of matrices (axis 1 = batch)."""
    m, n = shape
    B = 3
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n

    rng = np.random.default_rng(9)
    # batch of matrices: (B, source_rows, BATCH)
    batch_matrix = jnp.asarray(rng.random((B, source_rows, BATCH)) < 0.5, dtype=jnp.float32)
    batch_packed = jax.vmap(lambda mat: bitpack(mat, axis=1))(batch_matrix)

    f = lambda p, mat: bitpack_binary_fcnmm(
        weights, indices, p, mat, shape=shape, transpose=transpose
    )
    y_vmap = jax.vmap(f)(batch_packed, batch_matrix)

    # Compare against loop
    y_loop = jnp.stack([
        bitpack_binary_fcnmm(
            weights, indices, batch_packed[i], batch_matrix[i],
            shape=shape, transpose=transpose
        )
        for i in range(B)
    ])
    assert y_vmap.shape == y_loop.shape
    assert jnp.allclose(y_vmap, y_loop, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
def test_bitpack_fcnmm_vmap_grad(homo_w):
    """vmap(grad(bitpack_binary_fcnmm)) over a batch of matrices."""
    shape = (20, 40)
    m, n = shape
    B = 3
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = n  # gather mode

    rng = np.random.default_rng(10)
    batch_matrix = jnp.asarray(rng.random((B, source_rows, BATCH)) < 0.5, dtype=jnp.float32)
    batch_packed = jax.vmap(lambda mat: bitpack(mat, axis=1))(batch_matrix)

    def loss(p, mat):
        return jnp.sum(
            bitpack_binary_fcnmm(weights, indices, p, mat, shape=shape, transpose=False)
        )

    grad_fn = jax.grad(loss, argnums=1)
    grads = jax.vmap(grad_fn)(batch_packed, batch_matrix)
    assert grads.shape == batch_matrix.shape
    assert jnp.isfinite(grads).all()


# ===========================================================================
# 9. jax.jit compatibility
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_bitpack_fcnmv_jit(homo_w, transpose):
    """bitpack_binary_fcnmv runs correctly under jax.jit."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    packed = bitpack(spikes, axis=0)

    f_jit = jax.jit(
        lambda w, p, s: bitpack_binary_fcnmv(w, indices, p, s, shape=shape, transpose=transpose),
        static_argnums=(),
    )
    y_jit = f_jit(weights, packed, spikes)
    y_ref = _ref_mv(weights, indices, spikes, shape, transpose)
    assert jnp.allclose(y_jit, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_bitpack_fcnmm_jit(homo_w, transpose):
    """bitpack_binary_fcnmm runs correctly under jax.jit."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    packed = bitpack(matrix, axis=1)

    f_jit = jax.jit(
        lambda w, p, mat: bitpack_binary_fcnmm(
            w, indices, p, mat, shape=shape, transpose=transpose
        ),
    )
    y_jit = f_jit(weights, packed, matrix)
    y_ref = _ref_mm(weights, indices, matrix, shape, transpose)
    assert jnp.allclose(y_jit, y_ref, rtol=1e-3, atol=1e-3)


# ===========================================================================
# 10. brainunit Quantity support
# ===========================================================================

@pytest.mark.parametrize('transpose', [True, False])
def test_bitpack_fcnmv_quantity_homo(transpose):
    """bitpack_binary_fcnmv handles brainunit Quantity weights (homo)."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size)
    packed = bitpack(spikes, axis=0)

    w_q = jnp.array([1.5]) * u.mS  # homogeneous weights with unit
    y = bitpack_binary_fcnmv(w_q, indices, packed, spikes, shape=shape, transpose=transpose)

    # Should return a Quantity
    assert isinstance(y, u.Quantity)
    y_plain = bitpack_binary_fcnmv(
        jnp.array([1.5]), indices, packed, spikes, shape=shape, transpose=transpose
    )
    assert jnp.allclose(u.get_mantissa(y), y_plain, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('transpose', [True, False])
def test_bitpack_fcnmm_quantity_homo(transpose):
    """bitpack_binary_fcnmm handles brainunit Quantity weights (homo)."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)
    packed = bitpack(matrix, axis=1)

    w_q = jnp.array([1.5]) * u.mS
    y = bitpack_binary_fcnmm(w_q, indices, packed, matrix, shape=shape, transpose=transpose)

    assert isinstance(y, u.Quantity)
    y_plain = bitpack_binary_fcnmm(
        jnp.array([1.5]), indices, packed, matrix, shape=shape, transpose=transpose
    )
    assert jnp.allclose(u.get_mantissa(y), y_plain, rtol=1e-3, atol=1e-3)


# ===========================================================================
# 11. Output shape checks
# ===========================================================================

@pytest.mark.parametrize('shape', [(20, 40), (30, 30), (40, 20)])
def test_bitpack_fcnmv_output_shape_gather(shape):
    """Gather mode output has shape (n_pre,)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    spikes = _mk_spikes(n)
    packed = bitpack(spikes, axis=0)
    y = bitpack_binary_fcnmv(weights, indices, packed, spikes, shape=shape, transpose=False)
    assert y.shape == (m,)


@pytest.mark.parametrize('shape', [(20, 40), (30, 30), (40, 20)])
def test_bitpack_fcnmv_output_shape_scatter(shape):
    """Scatter mode output has shape (n_post,)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    spikes = _mk_spikes(m)
    packed = bitpack(spikes, axis=0)
    y = bitpack_binary_fcnmv(weights, indices, packed, spikes, shape=shape, transpose=True)
    assert y.shape == (n,)


@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_bitpack_fcnmm_output_shape_gather(shape):
    """Gather mode MM output has shape (n_pre, n_batch)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    matrix = _mk_matrix(n, BATCH)
    packed = bitpack(matrix, axis=1)
    y = bitpack_binary_fcnmm(weights, indices, packed, matrix, shape=shape, transpose=False)
    assert y.shape == (m, BATCH)


@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_bitpack_fcnmm_output_shape_scatter(shape):
    """Scatter mode MM output has shape (n_post, n_batch)."""
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    matrix = _mk_matrix(m, BATCH)
    packed = bitpack(matrix, axis=1)
    y = bitpack_binary_fcnmm(weights, indices, packed, matrix, shape=shape, transpose=True)
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
    packed_mv = bitpack(spikes, axis=0)
    y_mv = bitpack_binary_fcnmv(
        weights, indices, packed_mv, spikes, shape=shape, transpose=transpose
    )

    # MM with k=1
    matrix = spikes[:, None]  # (spike_size, 1)
    packed_mm = bitpack(matrix, axis=1)
    y_mm = bitpack_binary_fcnmm(
        weights, indices, packed_mm, matrix, shape=shape, transpose=transpose, pack_axis=1
    )

    assert jnp.allclose(y_mv, y_mm[:, 0], rtol=1e-3, atol=1e-3), (
        f"max diff={jnp.max(jnp.abs(y_mv - y_mm[:, 0])):.4e}"
    )


# ===========================================================================
# 13. Gradient consistency: MV grad matches dense reference via finite diff
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_bitpack_fcnmv_grad_weights_finite_diff(homo_w, transpose):
    """VJP gradient w.r.t. weights matches finite-difference approximation."""
    shape = (10, 20)
    m, n = shape
    indices = _mk_indices(shape, n_conn=2)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices, seed=99)
    spike_size = m if transpose else n
    spikes = _mk_spikes(spike_size, seed=99)
    packed = bitpack(spikes, axis=0)

    out_size = n if transpose else m
    ct = jnp.ones(out_size)

    f = lambda w: bitpack_binary_fcnmv(w, indices, packed, spikes,
                                       shape=shape, transpose=transpose)
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


# ===========================================================================
# 14. pack_axis=0 vs pack_axis=1 for MM
# ===========================================================================

@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_bitpack_fcnmm_pack_axis_consistency(homo_w, transpose):
    """Results with pack_axis=0 and pack_axis=1 should match dense reference."""
    shape = (20, 40)
    m, n = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
    source_rows = m if transpose else n
    matrix = _mk_matrix(source_rows, BATCH)

    y_ref = _ref_mm(weights, indices, matrix, shape, transpose)

    for axis in [0, 1]:
        packed = bitpack(matrix, axis=axis)
        y = bitpack_binary_fcnmm(
            weights, indices, packed, matrix,
            shape=shape, transpose=transpose, pack_axis=axis,
        )
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3), (
            f"pack_axis={axis}  max diff={jnp.max(jnp.abs(y - y_ref)):.4e}"
        )
