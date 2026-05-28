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


import inspect
from pathlib import Path

import brainstate
import braintools
import brainevent
import brainevent._fcn.binary as binary_mod
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._fcn.binary import binary_fcnmv, binary_fcnmv_p, binary_fcnmm, binary_fcnmm_p
from brainevent._misc import fixed_conn_num_to_csc
from brainevent._test_util import generate_fixed_conn_num_indices


platform = jax.default_backend()
FCNMV_IMPLEMENTATIONS = tuple(
    impl for impl in binary_fcnmv_p.available_backends(platform)
)
FCNMM_IMPLEMENTATIONS = tuple(
    impl for impl in binary_fcnmm_p.available_backends(platform)
)
FCNMV_COL_SCATTER_IMPLEMENTATIONS = tuple(
    impl for impl in FCNMV_IMPLEMENTATIONS
    if impl == 'cuda_raw'
)
FCNMV_NON_CUDA_IMPLEMENTATIONS = tuple(
    impl for impl in FCNMV_IMPLEMENTATIONS
    if impl != 'cuda_raw'
)

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
FCNMV_COL_SCATTER_PARAMS = _implementation_params(
    FCNMV_COL_SCATTER_IMPLEMENTATIONS,
    'binary_fcnmv-col-scatter',
)
FCNMV_NON_CUDA_PARAMS = _implementation_params(
    FCNMV_NON_CUDA_IMPLEMENTATIONS,
    'binary_fcnmv-non-cuda',
)

FCNMM_MAIN_IMPLEMENTATIONS = tuple(
    impl
    for impl in FCNMM_IMPLEMENTATIONS
)
FCNMM_MAIN_PARAMS = _implementation_params(FCNMM_MAIN_IMPLEMENTATIONS, 'binary_fcnmm-main')


def test_binary_fcnmv_col_scatter_cuda_names_match_dispatch_contract():
    cuda_kernel_source = inspect.getsource(binary_mod._binary_fcnmv_cuda_kernel)
    assert "binary_fcnmv_col_scatter.cu" in cuda_kernel_source
    assert "fcn_binary_mv_col_scatter" in cuda_kernel_source
    assert "binary_fcnmv_col_scatter" in cuda_kernel_source
    assert "binary_fcnmv_T.cu" not in cuda_kernel_source
    assert "fcn_binary_mv_t" not in cuda_kernel_source

    source_path = Path(binary_mod.__file__).with_name("binary_fcnmv_col_scatter.cu")
    source = source_path.read_text()
    assert "DEFINE_BFCNMV_COL_TPR_HOMO" in source
    assert "_bfcnmv_col_tpr_homo_kern" in source
    assert "FFI_BFCNMV_COL_HOMO" in source
    assert "n_blocks_tpr" in source
    assert "DEFINE_BS_" not in source
    assert "_bs_" not in source
    assert "FFI_BS" not in source
    assert "DEFINE_BS_CSC" not in source
    assert "_bs_csc" not in source
    assert "n_blocks_csc" not in source
    assert "binary_fcnmv_col_scatter_homo_bool_f32" in source
    assert "binary_fcnmv_scatter_homo_bool_f32" not in source


def test_binary_fcnmm_col_scatter_cuda_operator_names_are_not_primitive_dispatch():
    cuda_kernel_source = inspect.getsource(binary_mod._binary_fcnmm_cuda_kernel)
    assert "binary_fcnmm_col_scatter.cu" not in cuda_kernel_source
    assert "fcn_binary_mm_col_scatter" not in cuda_kernel_source
    assert "binary_fcnmm_col_scatter" not in cuda_kernel_source

    source_path = Path(binary_mod.__file__).with_name("binary_fcnmm_col_scatter.cu")
    assert source_path.exists()
    source = source_path.read_text()
    assert "binary_fcnmm_col_scatter.cu" in source
    assert "DEFINE_BFCNMM_COL_WARP_HOMO" in source
    assert "_bfcnmm_col_warp_homo_kern" in source
    assert "FFI_BFCNMM_COL_HOMO" in source
    assert "binary_fcnmm_col_scatter_homo_bool_f32" in source
    assert "binary_fcnmm_T.cu" not in source
    assert "binary_fcnmm_scatter_colmajor_homo_bool_f32" not in source
    assert "BSMM_COLMAJOR" not in source
    assert "_bsmm_colmajor" not in source
    assert "CSC" not in source


def test_binary_fcnmm_col_scatter_uses_wpr_batch_grid_contract():
    source_path = Path(binary_mod.__file__).with_name("binary_fcnmm_col_scatter.cu")
    source = source_path.read_text()

    assert "int batch = static_cast<int>(blockIdx.y)" in source
    assert "matrix_t[static_cast<size_t>(batch) * n_pre + col]" in source
    assert "output[static_cast<size_t>(batch) * n_post + row]" in source
    assert "for (int pos = start + lane; pos < end; pos += BFCNMM_COL_WARP_THREADS)" in source
    assert "dim3 grid(grid_x, n_batch)" in source
    assert "lane >= n_batch" not in source


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


def _mv_reference_from_col_major(col_weights, col_indices, col_indptr, events, shape):
    events = _to_binary_events(events, jnp.asarray(col_weights).dtype)
    out = np.zeros(shape[0], dtype=np.asarray(col_weights).dtype)
    col_weights_np = np.asarray(col_weights)
    col_indices_np = np.asarray(col_indices)
    col_indptr_np = np.asarray(col_indptr)
    events_np = np.asarray(events)
    homo = col_weights_np.size == 1
    scalar_weight = col_weights_np.reshape(-1)[0] if homo else None

    for col in range(shape[1]):
        if not events_np[col]:
            continue
        start = col_indptr_np[col]
        end = col_indptr_np[col + 1]
        for pos in range(start, end):
            row = col_indices_np[pos]
            out[row] += scalar_weight if homo else col_weights_np[pos]
    return jnp.asarray(out)


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


def _make_matrix(n_rows, k, event_dtype):
    if event_dtype is bool:
        return brainstate.random.rand(n_rows, k) < 0.5
    raw = jnp.asarray(brainstate.random.rand(n_rows, k), dtype=jnp.float32)
    return jnp.where(raw > 0.4, raw, 0.0)


def _make_deterministic_matrix(n_rows, k, event_dtype):
    values = jnp.arange(n_rows * k, dtype=jnp.float32).reshape(n_rows, k)
    active = ((values * 17 + 11).astype(jnp.int32) % 7) < 2
    if event_dtype is bool:
        return active
    return jnp.where(active, values / (n_rows * k + 1) + 0.125, 0.0)


def _make_deterministic_indices(n_pre, n_post, n_conn):
    rows = np.arange(n_pre, dtype=np.int32)[:, None]
    cols = np.arange(n_conn, dtype=np.int32)[None, :]
    return jnp.asarray((rows * 17 + cols * 31 + 7) % n_post, dtype=jnp.int32)


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

        c_int = min(c_int, conn_cap)

        m = s_int * _N

        if c_int > 0 and c_int <= m and s_int <= scale_max:
            pair = (c_int, m)
            if pair not in seen:
                seen.add(pair)
                valid_pairs.append(pair)

    return valid_pairs

@pytest.mark.parametrize('implementation', FCNMV_COL_SCATTER_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
def test_binary_fcnmv_forward_column_scatter_matches_reference_in_large_scale(implementation, homo_w):
    import gc

    # Keep the large-scale coverage focused on the column-scatter backend itself.
    # Float-valued events are stricter here because the operator must still treat
    # them as binary activity (active iff > 0).
    for conn, m in generate_cs_pairs(
        homo_or_not=homo_w,
        include_dense_ref=True,
        max_conn=600,
        num_points=3,
    ):
        indices = generate_fixed_conn_num_indices(m, m, conn)
        weights = _make_weights(indices, homo_w)
        col_weights, col_indices, col_indptr = fixed_conn_num_to_csc(weights, indices, shape=(m, m))

        raw = jnp.asarray(brainstate.random.rand(m), dtype=jnp.float32)
        events = jnp.where(raw > 0.4, raw, 0.0)
        shape = (m, m)

        y = binary_fcnmv(
            weights,
            indices,
            events,
            shape=shape,
            transpose=False,
            backend=implementation,
            col_weights=col_weights,
            col_indices=col_indices,
            col_indptr=col_indptr,
        )
        y_ref = _mv_reference(weights, indices, events, shape, transpose=False)

        assert y.shape == y_ref.shape
        assert jnp.allclose(y, y_ref, rtol=5e-2, atol=5e-2), (
            f"max diff={jnp.max(jnp.abs(y - y_ref)):.4e}  shape={shape}  "
            f"backend={implementation}  homo_w={homo_w}  conn={conn}"
        )

        jax.block_until_ready((indices, weights, events, col_weights, col_indices, col_indptr, y, y_ref))

        del indices, weights, events, col_weights, col_indices, col_indptr, y, y_ref
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

    if implementation == 'cuda_raw' and not transpose:
        with pytest.raises(ValueError, match='row-gather.*BitPackedBinary'):
            binary_fcnmv(
                weights,
                indices,
                events,
                shape=shape,
                transpose=transpose,
                backend=implementation,
            )
        return

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


@pytest.mark.parametrize('implementation', FCNMV_COL_SCATTER_PARAMS)
def test_binary_fcnmv_post_scatter_ignores_col_scatter_with_warning(implementation):
    m, n = 20, 40
    indices = generate_fixed_conn_num_indices(m, n, 4)
    weights = _make_weights(indices, homo_w=False)
    col_weights, col_indices, col_indptr = fixed_conn_num_to_csc(weights, indices, shape=(m, n))
    events = brainstate.random.rand(m) < 0.5

    with pytest.warns(UserWarning, match='col-scatter options.*post/scatter path.*fall back'):
        y = binary_fcnmv(
            weights,
            indices,
            events,
            shape=(m, n),
            transpose=True,
            backend=implementation,
            col_weights=col_weights,
            col_indices=col_indices,
            col_indptr=col_indptr,
        )
    y_ref = _mv_reference(weights, indices, events, (m, n), transpose=True)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('implementation', FCNMV_NON_CUDA_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_fcnmv_non_cuda_ignores_col_scatter_with_warning(implementation, transpose):
    m, n = 20, 40
    indices = generate_fixed_conn_num_indices(m, n, 4)
    weights = _make_weights(indices, homo_w=False)
    col_weights, col_indices, col_indptr = fixed_conn_num_to_csc(weights, indices, shape=(m, n))
    event_size = m if transpose else n
    events = brainstate.random.rand(event_size) < 0.5

    with pytest.warns(UserWarning, match='col.?scatter options.*(CUDA raw|this backend).*(normal|default) gather/scatter path'):
        y = binary_fcnmv(
            weights,
            indices,
            events,
            shape=(m, n),
            transpose=transpose,
            backend=implementation,
            col_weights=col_weights,
            col_indices=col_indices,
            col_indptr=col_indptr,
        )
    y_ref = _mv_reference(weights, indices, events, (m, n), transpose=transpose)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('implementation', FCNMV_COL_SCATTER_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('shape', SHAPES)
def test_binary_fcnmv_forward_column_scatter_matches_reference(implementation, homo_w, event_dtype, shape):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)))
    weights = _make_weights(indices, homo_w)
    col_weights, col_indices, col_indptr = fixed_conn_num_to_csc(weights, indices, shape=shape)

    if event_dtype is bool:
        events = brainstate.random.rand(n) < 0.5
    else:
        raw = jnp.asarray(brainstate.random.rand(n), dtype=jnp.float32)
        events = jnp.where(raw > 0.4, raw, 0.0)

    y = binary_fcnmv(
        weights,
        indices,
        events,
        shape=shape,
        transpose=False,
        backend=implementation,
        col_weights=col_weights,
        col_indices=col_indices,
        col_indptr=col_indptr,
    )
    y_ref = _mv_reference(weights, indices, events, shape, transpose=False)
    y_col = _mv_reference_from_col_major(col_weights, col_indices, col_indptr, events, shape)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(y_col, y_ref, rtol=1e-6, atol=1e-6)
    jax.block_until_ready((indices, weights, events, col_weights, col_indices, col_indptr, y, y_ref, y_col))


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('shape', SHAPES)
def test_column_major_mirror_has_expected_sizes(homo_w, shape):
    indices = generate_fixed_conn_num_indices(shape[0], shape[1], max(1, int(shape[1] * 0.1)))
    weights = _make_weights(indices, homo_w=homo_w)
    col_weights, col_indices, col_indptr = fixed_conn_num_to_csc(weights, indices, shape=shape)

    nnz = indices.size
    assert col_indices.ndim == 1
    assert col_indices.size == nnz
    assert col_indptr.ndim == 1
    assert col_indptr.shape == (shape[1] + 1,)
    assert int(np.asarray(col_indptr)[0]) == 0
    assert int(np.asarray(col_indptr)[-1]) == nnz
    if homo_w:
        assert jnp.asarray(col_weights).size == 1
    else:
        assert jnp.asarray(col_weights).size == nnz


@pytest.mark.parametrize('implementation', FCNMV_COL_SCATTER_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_binary_fcnmv_column_scatter_jvp_weights_matches_reference(implementation, homo_w, event_dtype, shape):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)))
    weights = _make_weights(indices, homo_w)
    _, col_indices, col_indptr = fixed_conn_num_to_csc(weights, indices, shape=shape)

    if event_dtype is bool:
        events = brainstate.random.rand(n) < 0.5
    else:
        raw = jnp.asarray(brainstate.random.rand(n), dtype=jnp.float32)
        events = jnp.where(raw > 0.4, raw, 0.0)

    w_dot = jnp.ones_like(weights) * 0.1

    f = lambda w: binary_fcnmv(
        w,
        indices,
        events,
        shape=shape,
        transpose=False,
        backend=implementation,
        col_weights=fixed_conn_num_to_csc(w, indices, shape=shape)[0],
        col_indices=col_indices,
        col_indptr=col_indptr,
    )
    primals_out, tangents_out = jax.jvp(f, (weights,), (w_dot,))

    expected_tangent = binary_fcnmv(
        w_dot,
        indices,
        events,
        shape=shape,
        transpose=False,
        backend=implementation,
        col_weights=fixed_conn_num_to_csc(w_dot, indices, shape=shape)[0],
        col_indices=col_indices,
        col_indptr=col_indptr,
    )
    y_ref = _mv_reference(weights, indices, events, shape, transpose=False)

    assert jnp.allclose(primals_out, y_ref, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(tangents_out, expected_tangent, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('implementation', FCNMV_COL_SCATTER_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_binary_fcnmv_column_scatter_vjp_weights_matches_dense(implementation, homo_w, event_dtype, shape):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)))
    weights = _make_weights(indices, homo_w)
    _, col_indices, col_indptr = fixed_conn_num_to_csc(weights, indices, shape=shape)

    if event_dtype is bool:
        events = brainstate.random.rand(n) < 0.5
    else:
        raw = jnp.asarray(brainstate.random.rand(n), dtype=jnp.float32)
        events = jnp.where(raw > 0.4, raw, 0.0)

    ct = jnp.ones(m, dtype=jnp.float32)

    f = lambda w: binary_fcnmv(
        w,
        indices,
        events,
        shape=shape,
        transpose=False,
        backend=implementation,
        col_weights=fixed_conn_num_to_csc(w, indices, shape=shape)[0],
        col_indices=col_indices,
        col_indptr=col_indptr,
    )
    _, vjp_fn = jax.vjp(f, weights)
    (ct_w,) = vjp_fn(ct)

    f_dense = lambda w: _mv_reference(w, indices, events, shape, transpose=False)
    _, vjp_dense = jax.vjp(f_dense, weights)
    (ct_w_dense,) = vjp_dense(ct)

    tol = 1e-3 if homo_w else 5e-2
    assert ct_w.shape == weights.shape
    assert jnp.allclose(ct_w, ct_w_dense, rtol=tol, atol=tol), (
        f"max diff={jnp.max(jnp.abs(ct_w - ct_w_dense)):.4e}"
    )


@pytest.mark.parametrize('implementation', FCNMV_COL_SCATTER_PARAMS)
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_binary_fcnmv_scatter_vjp_weights_float_activity_matches_dense(implementation, shape):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)))
    weights = _make_weights(indices, homo_w=False)

    raw = jnp.asarray(brainstate.random.rand(m), dtype=jnp.float32)
    events = jnp.where(raw > 0.4, raw, 0.0)
    ct = jnp.ones(n, dtype=jnp.float32)

    f = lambda w: binary_fcnmv(
        w,
        indices,
        events,
        shape=shape,
        transpose=True,
        backend=implementation,
    )
    _, vjp_fn = jax.vjp(f, weights)
    (ct_w,) = vjp_fn(ct)

    f_dense = lambda w: _mv_reference(w, indices, events, shape, transpose=True)
    _, vjp_dense = jax.vjp(f_dense, weights)
    (ct_w_dense,) = vjp_dense(ct)

    assert ct_w.shape == weights.shape
    assert jnp.allclose(ct_w, ct_w_dense, rtol=5e-2, atol=5e-2), (
        f"max diff={jnp.max(jnp.abs(ct_w - ct_w_dense)):.4e}"
    )


@pytest.mark.parametrize('implementation', FCNMM_MAIN_PARAMS)
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
