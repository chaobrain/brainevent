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
            # Mirror jax.ffi.ffi_call: a single result spec returns a bare array.
            if isinstance(out_info, (list, tuple)):
                return tuple(jnp.zeros(info.shape, info.dtype) for info in out_info)
            return jnp.zeros(out_info.shape, out_info.dtype)

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


# ---------------------------------------------------------------------------
# Weighted events (#201): ``weighted=True`` multiplies by the event value.
# ---------------------------------------------------------------------------

def _np_values(events):
    ev = events.astype(jnp.float32) if events.dtype == jnp.bfloat16 else events
    return np.asarray(ev, dtype=np.float64)


@pytest.mark.parametrize('implementation', ELL_MV_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype_name', EVENT_DTYPES)
def test_binary_fcnmv_weighted_all_dtypes(implementation, homo_w, transpose, dtype_name):
    if implementation == 'cuda_raw' and not transpose:
        pytest.skip('CUDA row-gather matvec removed; W @ s routes through CSR indexed.')
    shape = _ACT_SHAPE
    with _x64_if(dtype_name == 'float64'):
        weights, indices = _fixed_conn(shape, 5, homo_w)
        events = _make_events((shape[0] if transpose else shape[1],), dtype_name)
        y = np.asarray(binary_fcnmv(weights, indices, events, shape=shape, transpose=transpose,
                                    backend=implementation, weighted=True))
    dense = _np_dense(weights, indices, shape)
    x = _np_values(events)
    expected = x @ dense if transpose else dense @ x
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype_name', EVENT_DTYPES)
def test_binary_fcnmm_weighted_all_dtypes(implementation, homo_w, transpose, dtype_name):
    shape = _ACT_SHAPE
    with _x64_if(dtype_name == 'float64'):
        weights, indices = _fixed_conn(shape, 5, homo_w)
        matrix = _make_events((shape[0] if transpose else shape[1], 3), dtype_name)
        y = np.asarray(binary_fcnmm(weights, indices, matrix, shape=shape, transpose=transpose,
                                    backend=implementation, weighted=True))
    dense = _np_dense(weights, indices, shape)
    x = _np_values(matrix)
    expected = dense.T @ x if transpose else dense @ x
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('n_conn', [5, 40])
def test_binary_fcnmm_weighted_gather_warp_and_basic(implementation, homo_w, n_conn):
    """``n_conn > 32`` selects the CUDA basic gather kernel."""
    shape = (24, 64)
    weights, indices = _fixed_conn(shape, n_conn, homo_w)
    matrix = _make_events((shape[1], 5), 'int8')
    y = np.asarray(binary_fcnmm(weights, indices, matrix, shape=shape, transpose=False,
                                backend=implementation, weighted=True))
    expected = _np_dense(weights, indices, shape) @ _np_values(matrix)
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-4)


_WEIGHTED_GRAD_DTYPES = ('float32', 'int8', 'float16', 'float64')


def _weighted_grad_events(shape, dtype_name):
    tol = 2e-2 if dtype_name in ('float16', 'bfloat16') else 1e-4
    return _make_events(shape, dtype_name), tol


@pytest.mark.parametrize('implementation', ELL_MV_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype_name', _WEIGHTED_GRAD_DTYPES)
def test_binary_fcnmv_weighted_grads(implementation, homo_w, transpose, dtype_name):
    if implementation == 'cuda_raw' and not transpose:
        pytest.skip('CUDA row-gather matvec removed; W @ s routes through CSR indexed.')
    shape = _GRAD_SHAPE
    with _x64_if(dtype_name == 'float64'):
        weights, indices = _fixed_conn(shape, 4, homo_w, seed=17)
        x, tol = _weighted_grad_events((shape[0] if transpose else shape[1],), dtype_name)

        def op(p, v):
            return binary_fcnmv(p, indices, v, shape=shape, transpose=transpose,
                                backend=implementation, weighted=True)

        def dense_op(p, z):
            d = _dense_from_fixed_conn(p, indices, shape)
            z = z.astype(d.dtype)
            return z @ d if transpose else d @ z

        check_event_op_grads(op, dense_op, weights, x, mode='weighted', rtol=tol, atol=tol)


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype_name', _WEIGHTED_GRAD_DTYPES)
def test_binary_fcnmm_weighted_grads(implementation, homo_w, transpose, dtype_name):
    shape = _GRAD_SHAPE
    with _x64_if(dtype_name == 'float64'):
        weights, indices = _fixed_conn(shape, 4, homo_w, seed=18)
        x, tol = _weighted_grad_events((shape[0] if transpose else shape[1], 3), dtype_name)

        def op(p, v):
            return binary_fcnmm(p, indices, v, shape=shape, transpose=transpose,
                                backend=implementation, weighted=True)

        def dense_op(p, z):
            d = _dense_from_fixed_conn(p, indices, shape)
            z = z.astype(d.dtype)
            return d.T @ z if transpose else d @ z

        check_event_op_grads(op, dense_op, weights, x, mode='weighted', rtol=tol, atol=tol)


def test_binary_fcnmv_weighted_binary_unchanged():
    weights, indices = _fixed_conn((6, 5), 3, False, seed=3)
    x = jnp.asarray([2, -1, 0, 3, 0, 1], dtype=jnp.int8)
    dense = _np_dense(weights, indices, (6, 5))
    yb = binary_fcnmv(weights, indices, x, shape=(6, 5), transpose=True)
    yw = binary_fcnmv(weights, indices, x, shape=(6, 5), transpose=True, weighted=True)
    np.testing.assert_allclose(np.asarray(yb), _np_active(x) @ dense, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(yw), np.asarray(x, np.float64) @ dense, rtol=1e-5, atol=1e-5)


def _capture_fcn_ffi(monkeypatch):
    captured: dict = {'names': []}

    def fake_ffi_call(kernel_name, out_info):
        captured['names'].append(kernel_name)

        def run(*args):
            captured['events'] = args[-1]
            # Mirror jax.ffi.ffi_call: a single result spec returns a bare array.
            if isinstance(out_info, (list, tuple)):
                return tuple(jnp.zeros(info.shape, info.dtype) for info in out_info)
            return jnp.zeros(out_info.shape, out_info.dtype)

        return run

    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda *a, **kw: None)
    monkeypatch.setattr(binary_mod.jax.ffi, 'ffi_call', fake_ffi_call)
    return captured


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('w_dtype', ['float32', 'float16'])
@pytest.mark.parametrize('e_dtype', ['bool', 'int8', 'float32', 'float16'])
def test_binary_fcnmv_cuda_weighted_abi(monkeypatch, homo_w, w_dtype, e_dtype):
    captured = _capture_fcn_ffi(monkeypatch)
    weights = jnp.ones((1,) if homo_w else (4, 2), dtype=jnp.dtype(w_dtype))
    indices = jnp.zeros((4, 2), dtype=jnp.int32)
    spikes = _make_events((4,), e_dtype)
    kernel = binary_mod._ell_binary_matvec_cuda_kernel(
        transpose=True,
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        weighted=True,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        outs=[jax.ShapeDtypeStruct((3,), weights.dtype)],
    )
    kernel(weights, indices, spikes)
    mode = '_homo' if homo_w else '_hetero'
    w_sfx = '_f32' if w_dtype == 'float32' else '_f16'
    assert captured['names'] == [f'fcn_binary_mv.binary_fcnmv_scatter{mode}_wv{w_sfx}']
    assert captured['events'].dtype == jnp.dtype(w_dtype)
    np.testing.assert_array_equal(np.asarray(captured['events']),
                                  np.asarray(spikes.astype(jnp.dtype(w_dtype))))


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('n_conn,size', [(2, '_warp'), (40, '_basic')])
@pytest.mark.parametrize('e_dtype', ['int8', 'float32', 'float64'])
@pytest.mark.parametrize('n_batch', [3, 5000])
def test_binary_fcnmm_cuda_weighted_abi(monkeypatch, homo_w, transpose, n_conn, size, e_dtype, n_batch):
    """Weighted events select ``_wv`` kernels and never the bit-packed path."""
    captured = _capture_fcn_ffi(monkeypatch)
    with jax_x64_enabled():
        weights = jnp.ones((1,) if homo_w else (64, n_conn), dtype=jnp.float32)
        indices = jnp.zeros((64, n_conn), dtype=jnp.int32)
        matrix = _make_events((64, n_batch), e_dtype)
        kernel = binary_mod._binary_fcnmm_cuda_kernel(
            transpose=transpose,
            weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
            matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
            indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
            weighted=True,
            outs=[jax.ShapeDtypeStruct((64, n_batch), weights.dtype)],
        )
        kernel(weights, indices, matrix)
    mode = '_homo' if homo_w else '_hetero'
    if transpose:
        expected = f'fcn_binary_mm.binary_fcnmm_sraw{mode}_wv_f32'
    else:
        expected = f'fcn_binary_mm.binary_fcnmm_gather{mode}{size}_wv_f32'
    assert captured['names'] == [expected]
    assert captured['events'].dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(captured['events']), np.asarray(matrix, np.float32))


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_fcnmv_weighted_vmap(implementation, transpose):
    """vmap of the weighted matvec batches into the weighted matmat."""
    shape = _GRAD_SHAPE
    weights, indices = _fixed_conn(shape, 4, False, seed=19)
    xs = _make_events((3, shape[0] if transpose else shape[1]), 'int8')
    backend = None if implementation == 'cuda_raw' and not transpose else implementation
    f = lambda v: binary_fcnmv(weights, indices, v, shape=shape, transpose=transpose,
                               backend=backend, weighted=True)
    got = np.asarray(jax.vmap(f)(xs))
    dense = _np_dense(weights, indices, shape)
    x = np.asarray(xs, np.float64)
    expected = x @ dense if transpose else x @ dense.T
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-4)


@requires_gpu
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('w_dtype', ['float32', 'float64', 'float16', 'bfloat16'])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_fcnmm_weighted_large_gpu(homo_w, w_dtype, transpose):
    """Large event matrices (> 1 MB) would take the packed path in binary mode."""
    shape = (40, 600)
    n_batch = 2000
    with jax_x64_enabled():
        weights, indices = _fixed_conn(shape, 8, homo_w, w_dtype=w_dtype)
        rng = np.random.default_rng(5)
        rows = shape[0] if transpose else shape[1]
        x = rng.integers(-3, 4, size=(rows, n_batch)) * (rng.random((rows, n_batch)) < 0.1)
        matrix = jnp.asarray(x, dtype=jnp.float32)
        y = binary_fcnmm(weights, indices, matrix, shape=shape, transpose=transpose,
                         backend='cuda_raw', weighted=True)
        y = np.asarray(y.astype(jnp.float32), dtype=np.float64)
    w64 = weights.astype(jnp.float32) if w_dtype == 'bfloat16' else weights
    dense = _np_dense(w64, indices, shape)
    expected = dense.T @ x if transpose else dense @ x
    tol = 2e-2 if w_dtype in ('float16', 'bfloat16') else 1e-5
    rel = np.linalg.norm(y - expected) / np.linalg.norm(expected)
    assert rel < tol, f'relative error {rel}'


# ---------------------------------------------------------------------------
# Inactive events contribute exactly zero (NaN/Inf weight isolation)
# docs/specs/2026-10-24-inactive-row-nan-isolation.md
# ---------------------------------------------------------------------------

_ISO_BAD = [pytest.param(np.nan, id='nan'), pytest.param(np.inf, id='inf'),
            pytest.param(-np.inf, id='-inf')]
_ISO_KINDS = ['bool', 'float', 'weighted']
_ISO_SHAPE = (6, 6)
# Events 0, 2, 4 are active. Scatter (transpose=True): posts 2, 3, 5 are only
# reached from inactive rows. Gather (transpose=False): rows 4, 5 only read
# inactive events.
_ISO_INDICES = np.array([[0, 1], [2, 3], [1, 4], [5, 0], [1, 1], [3, 5]], dtype=np.int32)
_ISO_EVENTS = np.array([2., 0., -1., 0., 0.5, 0.])
# Batch columns: the first is ``_ISO_EVENTS``; the last is all-inactive.
_ISO_MATRIX = np.stack([_ISO_EVENTS, np.array([0., 1., 0., 0., 0., 2.]), np.zeros(6)], axis=1)


def _iso_events(values, kind):
    if kind == 'bool':
        return jnp.asarray(values != 0), False
    return jnp.asarray(values, dtype=jnp.float32), kind == 'weighted'


def _ell_terms(weights, indices, values, transpose, weighted):
    """Per-connection, per-batch-column select terms and their event values."""
    idx = np.asarray(indices)
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64).reshape(-1)
                        if np.size(weights) == 1 else np.asarray(weights, dtype=np.float64), idx.shape)
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[:, None] if vals.ndim == 1 else vals
    ev = np.broadcast_to(vals[:, None, :], idx.shape + vals.shape[1:]) if transpose else vals[idx]
    factor = ev if weighted else np.ones_like(ev)
    with np.errstate(invalid='ignore'):
        terms = np.where(ev != 0, w[:, :, None] * factor, 0.)
    return terms, ev, factor


def _ell_select_reference(weights, indices, values, shape, transpose, weighted):
    """``sum where(active, w * f, 0)`` over the ELL structure, float64, NaN-exact."""
    terms, _, _ = _ell_terms(weights, indices, values, transpose, weighted)
    if transpose:
        out = np.zeros((shape[1], terms.shape[-1]))
        with np.errstate(invalid='ignore'):  # Inf + -Inf
            np.add.at(out, np.asarray(indices), terms)
    else:
        out = terms.sum(1)
    return out[:, 0] if np.ndim(values) == 1 else out


def _iso_weights(homo, bad, transpose, active_of_entry):
    rng = np.random.default_rng(21)
    if homo:
        return jnp.asarray([bad], dtype=jnp.float32)
    w = rng.standard_normal(_ISO_INDICES.shape)
    w[~active_of_entry] = bad
    return jnp.asarray(w, dtype=jnp.float32)


def _entry_active(values, transpose):
    """Activity of the event each ELL entry meets (``values`` is 1-D)."""
    act = np.asarray(values) != 0
    return np.broadcast_to(act[:, None], _ISO_INDICES.shape) if transpose else act[_ISO_INDICES]


def _skip_cuda_gather_mv(implementation, transpose):
    if implementation == 'cuda_raw' and not transpose:
        pytest.skip('cuda_raw has no transpose=False matvec (routed to the CSR kernel)')


@pytest.mark.parametrize('implementation', ELL_MV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('kind', _ISO_KINDS)
@pytest.mark.parametrize('bad', _ISO_BAD)
def test_binary_fcnmv_nonfinite_weights_on_inactive_events_are_isolated(implementation, transpose, homo, kind, bad):
    _skip_cuda_gather_mv(implementation, transpose)
    weights = _iso_weights(homo, bad, transpose, _entry_active(_ISO_EVENTS, transpose))
    events, weighted = _iso_events(_ISO_EVENTS, kind)
    out = binary_fcnmv(weights, jnp.asarray(_ISO_INDICES), events, shape=_ISO_SHAPE, transpose=transpose,
                       backend=implementation, weighted=weighted)
    out = np.asarray(out)
    ref = _ell_select_reference(weights, _ISO_INDICES, _ISO_EVENTS, _ISO_SHAPE, transpose, weighted)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5, equal_nan=True)
    inactive_only = [2, 3, 5] if transpose else [4, 5]
    np.testing.assert_array_equal(out[inactive_only], 0.)
    if not homo:
        assert np.all(np.isfinite(out)), out


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('kind', _ISO_KINDS)
@pytest.mark.parametrize('bad', _ISO_BAD)
def test_binary_fcnmm_nonfinite_weights_on_inactive_events_are_isolated(implementation, transpose, homo, kind, bad):
    # Bad weights sit on the entries inactive in batch column 0; column 1 makes
    # some of them active (NaN/Inf legitimately propagates there).
    weights = _iso_weights(homo, bad, transpose, _entry_active(_ISO_EVENTS, transpose))
    matrix, weighted = _iso_events(_ISO_MATRIX, kind)
    out = binary_fcnmm(weights, jnp.asarray(_ISO_INDICES), matrix, shape=_ISO_SHAPE, transpose=transpose,
                       backend=implementation, weighted=weighted)
    out = np.asarray(out)
    ref = _ell_select_reference(weights, _ISO_INDICES, _ISO_MATRIX, _ISO_SHAPE, transpose, weighted)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(out[:, 2], 0.)  # all-inactive column
    if not homo:
        assert np.all(np.isfinite(out[:, 0])), out


@pytest.mark.parametrize('implementation', ELL_MV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('kind', _ISO_KINDS)
def test_binary_fcnmv_nan_weight_on_active_event_propagates(implementation, transpose, homo, kind):
    _skip_cuda_gather_mv(implementation, transpose)
    active = _entry_active(_ISO_EVENTS, transpose)
    if homo:
        weights = jnp.asarray([np.nan], dtype=jnp.float32)
    else:
        w = np.random.default_rng(22).standard_normal(_ISO_INDICES.shape)
        w[np.argwhere(active)[0][0], np.argwhere(active)[0][1]] = np.nan
        weights = jnp.asarray(w, dtype=jnp.float32)
    events, weighted = _iso_events(_ISO_EVENTS, kind)
    out = np.asarray(binary_fcnmv(weights, jnp.asarray(_ISO_INDICES), events, shape=_ISO_SHAPE,
                                  transpose=transpose, backend=implementation, weighted=weighted))
    ref = _ell_select_reference(weights, _ISO_INDICES, _ISO_EVENTS, _ISO_SHAPE, transpose, weighted)
    assert np.any(np.isnan(out))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5, equal_nan=True)


def _ell_weight_grad_reference(weights, indices, values, ct, transpose, weighted):
    """``dW[i, k] = sum_b where(active, f * ct[target, b], 0)``; homo: summed."""
    _, ev, factor = _ell_terms(np.ones(indices.shape), indices, values, transpose, weighted)
    ct = np.asarray(ct, dtype=np.float64)
    ct = ct[:, None] if ct.ndim == 1 else ct
    c = ct[np.asarray(indices)] if transpose else np.broadcast_to(ct[:, None, :], ev.shape)
    with np.errstate(invalid='ignore'):
        g = np.where(ev != 0, factor * c, 0.).sum(-1)
    return np.asarray([g.sum()]) if np.size(weights) == 1 else g


@pytest.mark.parametrize('implementation', ELL_MV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('kind', _ISO_KINDS)
def test_binary_fcnmv_weight_cotangent_of_inactive_events_is_exactly_zero(implementation, transpose, homo, kind):
    _skip_cuda_gather_mv(implementation, transpose)
    rng = np.random.default_rng(23)
    weights = jnp.asarray([1.5] if homo else rng.standard_normal(_ISO_INDICES.shape), dtype=jnp.float32)
    events, weighted = _iso_events(_ISO_EVENTS, kind)
    ct = rng.standard_normal(6)
    # A NaN cotangent on an output reached only through inactive events.
    ct[3 if transpose else 5] = np.nan
    _, vjp = jax.vjp(lambda w: binary_fcnmv(w, jnp.asarray(_ISO_INDICES), events, shape=_ISO_SHAPE,
                                            transpose=transpose, backend=implementation, weighted=weighted),
                     weights)
    (g,) = vjp(jnp.asarray(ct, dtype=jnp.float32))
    g = np.asarray(g)
    ref = _ell_weight_grad_reference(weights, _ISO_INDICES, _ISO_EVENTS, ct, transpose, weighted)
    assert np.all(np.isfinite(g)), g
    np.testing.assert_allclose(g, ref, rtol=1e-5, atol=1e-5)
    if not homo:
        np.testing.assert_array_equal(g[~_entry_active(_ISO_EVENTS, transpose)], 0.)


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('kind', _ISO_KINDS)
def test_binary_fcnmm_weight_cotangent_of_inactive_events_is_exactly_zero(implementation, transpose, homo, kind):
    rng = np.random.default_rng(24)
    weights = jnp.asarray([1.5] if homo else rng.standard_normal(_ISO_INDICES.shape), dtype=jnp.float32)
    matrix, weighted = _iso_events(_ISO_MATRIX, kind)
    ct = rng.standard_normal((6, _ISO_MATRIX.shape[1]))
    ct[:, 2] = np.nan  # batch column 2 is all-inactive
    ct[3 if transpose else 5, 0] = np.nan  # reached only through inactive events in column 0
    _, vjp = jax.vjp(lambda w: binary_fcnmm(w, jnp.asarray(_ISO_INDICES), matrix, shape=_ISO_SHAPE,
                                            transpose=transpose, backend=implementation, weighted=weighted),
                     weights)
    (g,) = vjp(jnp.asarray(ct, dtype=jnp.float32))
    g = np.asarray(g)
    ref = _ell_weight_grad_reference(weights, _ISO_INDICES, _ISO_MATRIX, ct, transpose, weighted)
    np.testing.assert_allclose(g, ref, rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('n_conn', [8, 40])
def test_binary_fcnmm_nan_isolation_large(implementation, transpose, homo, n_conn):
    """Exercises the warp / basic / bit-packed (> 1 MB events) CUDA gather variants."""
    shape = (64, 600)
    n_batch = 512
    weights, indices = _fixed_conn(shape, n_conn, homo, seed=3)
    rng = np.random.default_rng(25)
    rows = shape[0] if transpose else shape[1]
    values = (rng.random((rows, n_batch)) < 0.05).astype(np.float64)
    values[:, -1] = 0.  # all-inactive batch column
    if homo:
        weights = jnp.asarray([np.nan], dtype=jnp.float32)
    else:
        w = np.array(weights)
        # NaN on every entry whose event row is inactive in batch column 0.
        w[~_np_entry_active(values[:, 0], indices, transpose)] = np.nan
        weights = jnp.asarray(w)
    matrix = jnp.asarray(values != 0)
    out = np.asarray(binary_fcnmm(weights, indices, matrix, shape=shape, transpose=transpose,
                                  backend=implementation))
    ref = _ell_select_reference(weights, indices, values, shape, transpose, False)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, equal_nan=True)
    np.testing.assert_array_equal(out[:, -1], 0.)
    if not homo:
        assert np.all(np.isfinite(out[:, 0]))


def _np_entry_active(values_1d, indices, transpose):
    act = np.asarray(values_1d) != 0
    idx = np.asarray(indices)
    return np.broadcast_to(act[:, None], idx.shape) if transpose else act[idx]


@pytest.mark.parametrize('backend', FCNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_fcnmm_square_output_layout_and_grads(backend, homo, transpose):
    # Regression: with ``n_batch`` equal to the output size the cuda_raw
    # scatter layout was guessed from shapes, returning (and differentiating)
    # the transpose of the logical ``(n_out, n_batch)`` product.
    n_pre, n_post, n_conn = 5, 4, 2
    rng = np.random.default_rng(0)
    indices = jnp.asarray(
        np.stack([rng.choice(n_post, n_conn, replace=False) for _ in range(n_pre)]), jnp.int32
    )
    weights = jnp.asarray([1.5] if homo else rng.random((n_pre, n_conn)) + 0.5, jnp.float32)
    n_in, n_out = (n_pre, n_post) if transpose else (n_post, n_pre)
    matrix = jnp.asarray(rng.random((n_in, n_out)) > 0.4, jnp.float32)
    dense = jnp.zeros((n_pre, n_post), jnp.float32).at[
        jnp.repeat(jnp.arange(n_pre), n_conn), indices.ravel()
    ].add(jnp.broadcast_to(weights, indices.shape).ravel())
    dense_op = dense.T if transpose else dense

    def f(w, m):
        return binary_fcnmm(w, indices, m, shape=(n_pre, n_post), transpose=transpose, backend=backend)

    def ref(w, m):
        d = jnp.zeros((n_pre, n_post), jnp.float32).at[
            jnp.repeat(jnp.arange(n_pre), n_conn), indices.ravel()
        ].add(jnp.broadcast_to(w, indices.shape).ravel())
        return (d.T if transpose else d) @ m

    out = f(weights, matrix)
    assert out.shape == (n_out, n_out)
    assert jnp.allclose(out, dense_op @ matrix, atol=1e-5)
    assert jnp.allclose(jax.jit(f)(weights, matrix), dense_op @ matrix, atol=1e-5)

    ct = jnp.asarray(rng.random((n_out, n_out)), jnp.float32)
    _, vjp = jax.vjp(f, weights, matrix)
    _, vjp_ref = jax.vjp(ref, weights, matrix)
    for g, g_ref in zip(vjp(ct), vjp_ref(ct)):
        assert jnp.allclose(g, g_ref, atol=1e-4)

    w_dot = jnp.ones_like(weights)
    m_dot = jnp.asarray(rng.random(matrix.shape), jnp.float32)
    _, t = jax.jvp(f, (weights, matrix), (w_dot, m_dot))
    _, t_ref = jax.jvp(ref, (weights, matrix), (w_dot, m_dot))
    assert jnp.allclose(t, t_ref, atol=1e-4)

    batched = jnp.stack([matrix, 1.0 - matrix])
    got = jax.vmap(lambda m: f(weights, m))(batched)
    assert jnp.allclose(got, jnp.einsum('ij,bjk->bik', dense_op, batched), atol=1e-5)
