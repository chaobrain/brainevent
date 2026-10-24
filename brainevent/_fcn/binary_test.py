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


import contextlib
import inspect
import random

import brainstate
import braintools
import brainevent._fcn.binary as binary_mod
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._fcn.binary import (
    binary_fcnmv,
    binary_fcnmv_p,
    binary_fcnmm,
    binary_fcnmm_p,
)
from brainevent._error import KernelCompilationError
from brainevent._misc import fixed_conn_num_to_csc
from brainevent._op.util import event_factor
from brainevent._test_util import (
    check_event_op_grads,
    generate_fixed_conn_num_indices,
    jax_x64_enabled,
    requires_gpu,
)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Make the random connectivity / event draws in this module deterministic.

    Test data comes from unseeded ``brainstate.random`` / ``np.random`` draws and
    from ``generate_fixed_conn_num_indices`` (which uses Python's ``random`` to
    decide sampling-with-replacement). Without a fixed seed an occasional draw on
    a small problem pushes the kernel-vs-dense comparison past its tolerance,
    making the test order-dependently flaky. Seeding every RNG source per test
    removes that dependence."""
    random.seed(0x5EED)
    np.random.seed(0x5EED)
    brainstate.random.seed(0x5EED)


platform = jax.default_backend()
ELL_MV_IMPLEMENTATIONS = tuple(impl for impl in binary_fcnmv_p.available_backends(platform))
FCNMM_IMPLEMENTATIONS = tuple(impl for impl in binary_fcnmm_p.available_backends(platform))

if platform == 'cpu':
    SHAPES = (
        (20, 40),
        (50, 30)
    )
else:
    SHAPES = (
        (20, 40),
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


ELL_MV_PARAMS = _implementation_params(ELL_MV_IMPLEMENTATIONS, 'ell_binary_matvec')
FCNMM_PARAMS = _implementation_params(FCNMM_IMPLEMENTATIONS, 'binary_fcnmm')


# ---------------------------------------------------------------------------
# CUDA kernel-name dispatch contracts
# ---------------------------------------------------------------------------


def test_ell_binary_matvec_cuda_kernel_is_scatter_only():
    cuda_kernel_source = inspect.getsource(binary_mod._ell_binary_matvec_cuda_kernel)
    # ELL primitive loads only the row-major source and scatters; the
    # transpose=False row-gather path was removed and the unfavorable W @ s
    # direction now routes through the perm-fused CSR kernel.
    assert "binary_fcnmv.cu" in cuda_kernel_source
    assert "fcn_binary_mv.binary_fcnmv_scatter" in cuda_kernel_source
    assert "binary_fcnmv_col_scatter" not in cuda_kernel_source
    assert "fcn_binary_mv_col_scatter" not in cuda_kernel_source
    assert "NotImplementedError" in cuda_kernel_source


def test_binary_fcnmm_cuda_operator_names_are_not_col_scatter():
    # The FCN matmat CUDA kernel keeps the native ELL gather names, but its
    # scatter path is routed through SRAW rather than the old BSM scatter FFI.
    cuda_kernel_source = inspect.getsource(binary_mod._binary_fcnmm_cuda_kernel)
    sraw_kernel_source = inspect.getsource(binary_mod._binary_fcnmm_sraw_cuda_kernel)
    assert "binary_fcnmm.cu" in cuda_kernel_source
    assert "_binary_fcnmm_sraw_cuda_kernel" in cuda_kernel_source
    assert "binary_fcnmm_pack" in cuda_kernel_source
    assert "ell_binary_matmat_pack" not in cuda_kernel_source
    assert "binary_fcnmm.cu" in sraw_kernel_source
    assert "binary_fcnmm_sraw" in sraw_kernel_source
    assert "fcn_binary_mm." in sraw_kernel_source
    assert "binary_fcnmm_scatter" not in cuda_kernel_source
    assert "binary_fcnmm_scatter" not in sraw_kernel_source
    assert "binary_fcnmm_col_scatter.cu" not in cuda_kernel_source
    assert "fcn_binary_mm_col_scatter" not in cuda_kernel_source
    assert "binary_fcnmm_col_scatter" not in cuda_kernel_source


def test_binary_fcnmm_cuda_raw_transpose_true_out_shape_is_raw_batch_first():
    source = inspect.getsource(binary_mod.binary_fcnmm_p_call)
    assert binary_mod._binary_fcnmm_uses_raw_batch_first(transpose=True, backend='cuda_raw')
    assert not binary_mod._binary_fcnmm_uses_raw_batch_first(transpose=False, backend='cuda_raw')
    assert not binary_mod._binary_fcnmm_uses_raw_batch_first(transpose=True, backend='jax_raw')
    assert "(matrix.shape[1], n_post)" in source


def test_binary_fcnmm_transform_adapter_contract():
    helper_source = inspect.getsource(binary_mod._binary_fcnmm_uses_raw_batch_first)
    ct_shape_source = inspect.getsource(binary_mod._logical_transpose_true_shape_from_cotangent)
    matrix_source = inspect.getsource(binary_mod._binary_fcnmm_jvp_matrix)
    weights_source = inspect.getsource(binary_mod._binary_fcnmm_jvp_weights)
    transpose_source = inspect.getsource(binary_mod._binary_fcnmm_transpose_rule)
    combined = helper_source + ct_shape_source + matrix_source + weights_source + transpose_source
    assert "_maybe_transpose_to_expected" in combined
    assert "cuda_raw" in combined
    assert "expected_shape" in combined
    assert "_logical_transpose_true_shape_from_cotangent" in transpose_source


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _make_weights(indices, homo_w: bool):
    if homo_w:
        return jnp.asarray([1.5], dtype=jnp.float32)
    return braintools.init.Normal(0.0, 1.0)(indices.shape)


def _to_binary_events(events, dtype):
    return event_factor(jnp.asarray(events), dtype=dtype)


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


# ---------------------------------------------------------------------------
# ELL binary matvec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('implementation', ELL_MV_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('shape', SHAPES)
def test_ell_binary_matvec_forward_matches_reference(implementation, homo_w, transpose, event_dtype, shape):
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
        # The CUDA row-gather matvec was removed; W @ s goes through the
        # perm-fused CSR kernel instead. The kernel-construction failure is
        # wrapped in ``KernelCompilationError`` (the original ``NotImplementedError``
        # is preserved as ``__cause__`` and its text in the message), so accept
        # both the wrapper and the bare exception.
        with pytest.raises(
            (KernelCompilationError, NotImplementedError, ValueError),
            match='row-gather',
        ):
            binary_fcnmv(weights, indices, events, shape=shape, transpose=transpose, backend=implementation)
        return

    y = binary_fcnmv(weights, indices, events, shape=shape, transpose=transpose, backend=implementation)
    y_ref = _mv_reference(weights, indices, events, shape, transpose)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, events, y, y_ref))


# ---------------------------------------------------------------------------
# Column-major (CSC) mirror structure builder
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ELL scatter (transpose=True) autodiff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('implementation', ELL_MV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 40), (30, 30)])
def test_ell_binary_matvec_scatter_vjp_weights_float_activity_matches_dense(implementation, shape):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)))
    weights = _make_weights(indices, homo_w=False)

    raw = jnp.asarray(brainstate.random.rand(m), dtype=jnp.float32)
    events = jnp.where(raw > 0.4, raw, 0.0)
    ct = jnp.ones(n, dtype=jnp.float32)

    f = lambda w: binary_fcnmv(w, indices, events, shape=shape, transpose=True, backend=implementation)
    _, vjp_fn = jax.vjp(f, weights)
    (ct_w,) = vjp_fn(ct)

    f_dense = lambda w: _mv_reference(w, indices, events, shape, transpose=True)
    _, vjp_dense = jax.vjp(f_dense, weights)
    (ct_w_dense,) = vjp_dense(ct)

    assert ct_w.shape == weights.shape
    assert jnp.allclose(ct_w, ct_w_dense, rtol=5e-2, atol=5e-2), (
        f"max diff={jnp.max(jnp.abs(ct_w - ct_w_dense)):.4e}"
    )


# ---------------------------------------------------------------------------
# ELL binary matmat
# ---------------------------------------------------------------------------


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

    y = binary_fcnmm(weights, indices, matrix, shape=shape, transpose=transpose, backend=implementation)
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

    y_float = binary_fcnmm(weights, indices, float_events, shape=shape, transpose=transpose, backend=implementation)
    y_binary = binary_fcnmm(weights, indices, binary_events, shape=shape, transpose=transpose, backend=implementation)
    assert jnp.allclose(y_float, y_binary, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((indices, weights, float_events, binary_events, y_float, y_binary))


# ---------------------------------------------------------------------------
# Activity contract: an event is active iff ``x != 0`` (bool: ``x``) on every
# backend and for every event dtype (#199, #204).
# ---------------------------------------------------------------------------

EVENT_DTYPES = (
    'bool', 'int8', 'uint8', 'int16', 'int32',
    'float16', 'bfloat16', 'float32', 'float64',
)
_EVENT_PATTERN = np.array([2., -1., 0., 0.5, 0., 1., -3., 0.])
_ACT_SHAPE = (24, 16)


def _x64_if(flag):
    return jax_x64_enabled() if flag else contextlib.nullcontext()


def _make_events(shape, dtype_name):
    """Tile ``[2, -1, 0, 0.5, ...]`` to ``shape`` and cast to ``dtype_name``."""
    raw = np.resize(_EVENT_PATTERN, int(np.prod(shape))).reshape(shape)
    if dtype_name == 'bool':
        arr = raw != 0
    elif dtype_name.startswith(('int', 'uint')):
        arr = raw.astype(np.int64).astype(dtype_name)
    else:
        arr = raw
    return jnp.asarray(arr, dtype=jnp.dtype(dtype_name))


def _fixed_conn(shape, n_conn, homo_w, w_dtype='float32', seed=0):
    rng = np.random.default_rng(seed)
    indices = jnp.asarray(rng.integers(0, shape[1], (shape[0], n_conn)), dtype=jnp.int32)
    if homo_w:
        weights = jnp.asarray([1.5], dtype=jnp.dtype(w_dtype))
    else:
        weights = jnp.asarray(rng.standard_normal((shape[0], n_conn)), dtype=jnp.dtype(w_dtype))
    return weights, indices


def _np_dense(weights, indices, shape):
    w = np.asarray(weights, dtype=np.float64)
    idx = np.asarray(indices)
    dense = np.zeros(shape, dtype=np.float64)
    for i in range(idx.shape[0]):
        for k in range(idx.shape[1]):
            dense[i, idx[i, k]] += w[0] if w.size == 1 else w[i, k]
    return dense


def _np_active(events):
    return (np.asarray(events) != 0).astype(np.float64)


@pytest.mark.parametrize('implementation', ELL_MV_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype_name', EVENT_DTYPES)
def test_binary_fcnmv_event_activity_all_dtypes(implementation, homo_w, transpose, dtype_name):
    if implementation == 'cuda_raw' and not transpose:
        pytest.skip('CUDA row-gather matvec removed; W @ s routes through CSR indexed.')
    shape = _ACT_SHAPE
    with _x64_if(dtype_name == 'float64'):
        weights, indices = _fixed_conn(shape, 5, homo_w)
        events = _make_events((shape[0] if transpose else shape[1],), dtype_name)
        y = np.asarray(binary_fcnmv(weights, indices, events, shape=shape, transpose=transpose,
                                    backend=implementation))
    dense = _np_dense(weights, indices, shape)
    s = _np_active(events)
    expected = s @ dense if transpose else dense @ s
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype_name', EVENT_DTYPES)
def test_binary_fcnmm_event_activity_all_dtypes(implementation, homo_w, transpose, dtype_name):
    shape = _ACT_SHAPE
    with _x64_if(dtype_name == 'float64'):
        weights, indices = _fixed_conn(shape, 5, homo_w)
        matrix = _make_events((shape[0] if transpose else shape[1], 3), dtype_name)
        y = np.asarray(binary_fcnmm(weights, indices, matrix, shape=shape, transpose=transpose,
                                    backend=implementation))
    dense = _np_dense(weights, indices, shape)
    s = _np_active(matrix)
    expected = dense.T @ s if transpose else dense @ s
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('w_dtype,e_dtype', [('float32', 'float64'), ('float64', 'float32'),
                                             ('float64', 'float64'), ('float16', 'float32')])
def test_binary_fcnmm_mixed_weight_event_float_dtypes(implementation, homo_w, transpose, w_dtype, e_dtype):
    """``binary_fcnmm`` CUDA ``_float`` kernels read events of the *weight*
    dtype; any other float event dtype must be converted, not reinterpreted."""
    if implementation == 'numba' and w_dtype == 'float16':
        pytest.skip('numba has no float16 weight support')
    shape = _ACT_SHAPE
    with jax_x64_enabled():
        weights, indices = _fixed_conn(shape, 5, homo_w, w_dtype=w_dtype)
        matrix = _make_events((shape[0] if transpose else shape[1], 3), e_dtype)
        y = binary_fcnmm(weights, indices, matrix, shape=shape, transpose=transpose, backend=implementation)
        assert y.dtype == jnp.dtype(w_dtype)
        y = np.asarray(y, dtype=np.float64)
    dense = _np_dense(weights, indices, shape)
    s = _np_active(matrix)
    expected = dense.T @ s if transpose else dense @ s
    tol = 1e-2 if w_dtype == 'float16' else 1e-5
    np.testing.assert_allclose(y, expected, rtol=tol, atol=tol)


@requires_gpu
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('w_dtype,e_dtype', [('float32', 'float64'), ('float64', 'float32'),
                                             ('float32', 'float32'), ('float32', 'int8')])
def test_binary_fcnmm_packed_gather_event_dtypes_gpu(homo_w, w_dtype, e_dtype):
    """Large event matrices take the bit-packed gather path (> 1 MB)."""
    shape = (40, 600)
    n_batch = 2000
    with jax_x64_enabled():
        weights, indices = _fixed_conn(shape, 8, homo_w, w_dtype=w_dtype)
        matrix = _make_events((shape[1], n_batch), e_dtype)
        y = np.asarray(binary_fcnmm(weights, indices, matrix, shape=shape, transpose=False,
                                    backend='cuda_raw'), dtype=np.float64)
    expected = _np_dense(weights, indices, shape) @ _np_active(matrix)
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    'w_dtype,e_dtype,expected_sfx,converted',
    [
        ('float32', 'bool', '_bool', False),
        ('float32', 'float32', '_float', False),
        ('float64', 'float64', '_float', False),
        ('float32', 'float64', '_bool', True),
        ('float64', 'float32', '_bool', True),
        ('float32', 'int32', '_bool', True),
    ],
)
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_fcnmm_cuda_event_abi_uses_weight_dtype_as_native(
    monkeypatch, w_dtype, e_dtype, expected_sfx, converted, transpose
):
    captured = {}

    def fake_ffi_call(kernel_name, out_info):
        captured['kernel_name'] = kernel_name

        def run(*args):
            captured['events'] = args[-1]
            infos = out_info if isinstance(out_info, (list, tuple)) else (out_info,)
            return tuple(jnp.zeros(info.shape, info.dtype) for info in infos)

        return run

    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda *a, **kw: None)
    monkeypatch.setattr(binary_mod.jax.ffi, 'ffi_call', fake_ffi_call)
    with jax_x64_enabled():
        weights = jnp.ones((4, 2), dtype=jnp.dtype(w_dtype))
        indices = jnp.zeros((4, 2), dtype=jnp.int32)
        matrix = _make_events((4, 3), e_dtype)
        kernel = binary_mod._binary_fcnmm_cuda_kernel(
            transpose=transpose,
            weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
            matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
            indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
            outs=[jax.ShapeDtypeStruct((4, 3), weights.dtype)],
        )
        kernel(weights, indices, matrix)
    w_sfx = '_f32' if w_dtype == 'float32' else '_f64'
    assert expected_sfx + w_sfx in captured['kernel_name']
    if converted:
        assert captured['events'].dtype == jnp.bool_
        np.testing.assert_array_equal(np.asarray(captured['events']), np.asarray(matrix) != 0)
    else:
        assert captured['events'].dtype == matrix.dtype


# ---------------------------------------------------------------------------
# Gradients (#198): weight cotangents use the activity mask, event cotangents
# are straight-through.
# ---------------------------------------------------------------------------

_GRAD_SHAPE = (12, 10)


def _grad_events(n, batch=None):
    x = np.resize(np.array([2., -1., 0., 1., 0., 0.5, -3., 0.]), n)
    if batch is not None:
        x = np.stack([np.roll(x, i) for i in range(batch)], axis=1)
    return jnp.asarray(x, dtype=jnp.float32)


@pytest.mark.parametrize('implementation', ELL_MV_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_fcnmv_grads_use_activity_mask(implementation, homo_w, transpose):
    if implementation == 'cuda_raw' and not transpose:
        pytest.skip('CUDA row-gather matvec removed; W @ s routes through CSR indexed.')
    shape = _GRAD_SHAPE
    weights, indices = _fixed_conn(shape, 4, homo_w, seed=7)
    x = _grad_events(shape[0] if transpose else shape[1])

    def op(p, v):
        return binary_fcnmv(p, indices, v, shape=shape, transpose=transpose, backend=implementation)

    def dense_op(p, z):
        d = _dense_from_fixed_conn(p, indices, shape)
        return z @ d if transpose else d @ z

    check_event_op_grads(op, dense_op, weights, x, mode='binary')


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_fcnmm_grads_use_activity_mask(implementation, homo_w, transpose):
    shape = _GRAD_SHAPE
    weights, indices = _fixed_conn(shape, 4, homo_w, seed=8)
    x = _grad_events(shape[0] if transpose else shape[1], batch=3)

    def op(p, v):
        return binary_fcnmm(p, indices, v, shape=shape, transpose=transpose, backend=implementation)

    def dense_op(p, z):
        d = _dense_from_fixed_conn(p, indices, shape)
        return d.T @ z if transpose else d @ z

    check_event_op_grads(op, dense_op, weights, x, mode='binary')
