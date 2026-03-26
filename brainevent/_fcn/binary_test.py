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
import numpy as np
import pytest

from brainevent._fcn.binary import binary_fcnmv, binary_fcnmv_p, binary_fcnmm, binary_fcnmm_p
from brainevent._test_util import generate_fixed_conn_num_indices


platform = jax.default_backend()
FCNMV_IMPLEMENTATIONS = tuple(binary_fcnmv_p.available_backends(platform))
FCNMM_IMPLEMENTATIONS = tuple(binary_fcnmm_p.available_backends(platform))

if platform == 'cpu':
    SHAPES = (
        (20, 40),
        (50, 30)
    )
else:
    SHAPES = (
        (20, 40),
        # (50, 30),
        # (200, 400)
        (400, 200)
    )


def _implementation_params(implementations, op_name: str):
    if implementations:
        return [pytest.param(impl, id=impl) for impl in implementations]
    return [
        pytest.param(
            None,
            marks=pytest.mark.skip(reason=f'No {op_name} implementations on platform={platform}'),
            id=f'no-{op_name}',
        )
    ]


FCNMV_PARAMS = _implementation_params(FCNMV_IMPLEMENTATIONS, 'binary_fcnmv')
FCNMM_PARAMS = _implementation_params(FCNMM_IMPLEMENTATIONS, 'binary_fcnmm')


def _make_weights(indices, homo_w: bool):
    if homo_w:
        return jnp.asarray([1.5], dtype=jnp.float32)
    return braintools.init.Normal(0.0, 1.0)(indices.shape)


def _to_binary_events(events, dtype):
    events = jnp.asarray(events)
    if events.dtype == jnp.bool_:
        return jnp.asarray(events, dtype=dtype)
    return jnp.asarray(events > 0, dtype=dtype)


def _dense_from_fixed_conn(weights, indices, shape):
    n_pre, _ = shape
    rows = jnp.repeat(jnp.arange(n_pre, dtype=indices.dtype), indices.shape[1])
    cols = indices.reshape(-1)
    weights = jnp.asarray(weights)
    if weights.size == 1:
        value = jnp.reshape(weights, (-1,))[0]
        values = jnp.full((indices.size,), value, dtype=weights.dtype)
    else:
        values = weights.reshape(-1)
    return jnp.zeros(shape, dtype=weights.dtype).at[rows, cols].add(values)


def _mv_reference(weights, indices, events, shape, transpose):
    dense = _dense_from_fixed_conn(weights, indices, shape)
    events = _to_binary_events(events, dense.dtype)
    if transpose:
        return events @ dense
    return dense @ events


def _mm_reference(weights, indices, matrix, shape, transpose):
    dense = _dense_from_fixed_conn(weights, indices, shape)
    matrix = _to_binary_events(matrix, dense.dtype)
    if transpose:
        return jnp.matmul(dense.T, matrix, precision=jax.lax.Precision.HIGHEST)
    return jnp.matmul(dense, matrix, precision=jax.lax.Precision.HIGHEST)


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


@pytest.mark.parametrize('implementation', FCNMV_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
def test_binary_fcnmv_forward_matches_reference_in_large_scale(implementation, homo_w, event_dtype):
    import gc
    for conn, m in generate_cs_pairs(homo_or_not=homo_w, include_dense_ref=True):

        indices = generate_fixed_conn_num_indices(m, m, conn)
        weights = _make_weights(indices, homo_w)

        # In this large-scale test we always use square matrices with transpose=True.
        transpose = True
        shape = (m, m)
        event_size = m
        if event_dtype is bool:
            events = brainstate.random.rand(event_size) < 0.5
        else:
            raw = jnp.asarray(brainstate.random.rand(event_size), dtype=jnp.float32)
            events = jnp.where(raw > 0.4, raw, 0.0)

        y = binary_fcnmv(
            weights,
            indices,
            events,
            shape=(m, m),
            transpose=transpose,
            backend=implementation,
        )
        y_ref = _mv_reference(weights, indices, events, shape, transpose)
        
        assert jnp.allclose(y, y_ref, rtol=1e-2, atol=1e-2)
        
        jax.block_until_ready((indices, weights, events, y, y_ref))
        
        del indices, weights, events, y, y_ref
        gc.collect()

@pytest.mark.parametrize('implementation', FCNMV_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('shape', SHAPES)
def test_binary_fcnmv_forward_matches_reference(implementation, homo_w, transpose, event_dtype, shape):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)))
    weights = _make_weights(indices, homo_w)

    event_size = m if transpose else n
    if event_dtype is bool:
        events = brainstate.random.rand(event_size) < 0.5
    else:
        raw = jnp.asarray(brainstate.random.rand(event_size), dtype=jnp.float32)
        events = jnp.where(raw > 0.4, raw, 0.0)

    y = binary_fcnmv(
        weights,
        indices,
        events,
        shape=shape,
        transpose=transpose,
        backend=implementation,
    )
    y_ref = _mv_reference(weights, indices, events, shape, transpose)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, events, y, y_ref))


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
def test_binary_fcnmm_forward_matches_reference(implementation, homo_w, transpose, event_dtype, k, shape):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)))
    weights = _make_weights(indices, homo_w)

    n_rows = m if transpose else n
    if event_dtype is bool:
        matrix = brainstate.random.rand(n_rows, k) < 0.5
    else:
        raw = jnp.asarray(brainstate.random.rand(n_rows, k), dtype=jnp.float32)
        matrix = jnp.where(raw > 0.4, raw, 0.0)

    y = binary_fcnmm(
        weights,
        indices,
        matrix,
        shape=shape,
        transpose=transpose,
        backend=implementation,
    )
    y_ref = _mm_reference(weights, indices, matrix, shape, transpose)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, matrix, y, y_ref))

@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (50, 30)])
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_fcnmm_thresholds_float_events(implementation, homo_w, shape, k, transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)), replace=False)
    weights = _make_weights(indices, homo_w)

    rng = np.random.RandomState(42)
    n_rows = m if transpose else n
    raw = rng.rand(n_rows, k).astype(np.float32)
    mask = rng.rand(n_rows, k) > 0.5
    float_events = jnp.asarray(raw * mask)
    binary_events = jnp.asarray(float_events > 0, dtype=jnp.float32)

    y_float = binary_fcnmm(
        weights,
        indices,
        float_events,
        shape=shape,
        transpose=transpose,
        backend=implementation,
    )
    y_binary = binary_fcnmm(
        weights,
        indices,
        binary_events,
        shape=shape,
        transpose=transpose,
        backend=implementation,
    )
    assert jnp.allclose(y_float, y_binary, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, float_events, binary_events, y_float, y_binary))
