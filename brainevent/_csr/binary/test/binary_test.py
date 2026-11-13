# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_csr/binary/binary.py``, grouped by behavior."""

import hashlib
import importlib
import re
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
import brainevent._csr.binary.binary as binary_mod
import brainevent._csr.config as config_mod
from brainevent._csr.test._test_util import (
    EVENT_DTYPE_NAMES,
    EVENT_KINDS,
    NONFINITE_VALUES,
    WEIGHTED_EVENT_DTYPE_NAMES,
    assert_select_reference,
    csr_matrix,
    csr_vector,
    cuda_kwargs,
    event_dtype_context,
    event_mask,
    event_values,
    get_csr,
    heavy_row_csr_numpy,
    homo_ct_case,
    inactive_nonfinite_case,
    int64_structure,
    matrix_csr,
    prepared_config,
    random_csr_numpy,
    recording_ffi_call,
    requires_gpu_backend,
    run_python_isolated,
    shape_of,
    strip_hybrid_suffix,
    vector_csr,
)
from brainevent._csr.binary import initialize
from brainevent._csr.binary.binary import (
    _binary_csrmm_benchmark_data,
    _binary_csrmm_cusparse_kernel,
    _binary_csrmv_benchmark_data,
    _binary_csrmv_cusparse_kernel,
    _csrmv_batching,
    binary_csrmm,
    binary_csrmm_p,
    binary_csrmm_p_call,
    binary_csrmv,
    binary_csrmv_p,
    binary_csrmv_p_call,
)
from brainevent._csr.binary.binary_indexed import binary_csrmm_indexed_p_call, binary_csrmv_indexed_p_call
from brainevent._csr.config import CSRConfig, CSRDeviceInfo, get_device_info
from brainevent._op import load_cuda_file
from brainevent._test_util import check_event_op_grads, jax_x64_enabled, requires_gpu, with_jax_x64


# Direct CSR vector and matrix event primitives
# ---------------------------------------------

platform = jax.default_backend()


CSRMV_IMPLEMENTATIONS = tuple(binary_csrmv_p.available_backends(platform))


CSRMM_IMPLEMENTATIONS = tuple(binary_csrmm_p.available_backends(platform))


@contextmanager
def _jax_x64_enabled():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def _require_implementations(implementations, op_name: str):
    if not implementations:
        pytest.skip(f'No {op_name} implementation on platform={platform}')


def _heavy_row_structure():
    """2-row CSR whose first row is longer than ``tpr_threshold`` (it uses the task queue)."""
    heavy = CSRConfig().tpr_threshold + 1
    indices = jnp.concatenate([jnp.arange(heavy, dtype=jnp.int32), jnp.zeros(1, jnp.int32)])
    indptr = jnp.array([0, heavy, heavy + 1], dtype=jnp.int32)
    return indices, indptr, heavy


def _kernel_eqn(primitive, fn, *args):
    """The only ``primitive`` equation in the jaxpr of ``fn(*args)``."""
    # Match by name: reloading the module (some tests do) rebinds the primitive object.
    eqns = [e for e in jax.make_jaxpr(fn)(*args).jaxpr.eqns if e.primitive.name == primitive.primitive.name]
    assert len(eqns) == 1
    return eqns[0]


def _assert_task_capacity_param(eqn, indptr, max_row_nnz):
    params = eqn.params
    assert not {'task_begin_info', 'task_end_info', 'status_info'} & params.keys()
    assert params['max_row_nnz'] == max_row_nnz
    assert type(params['task_capacity']) is int
    assert params['task_capacity'] >= CSRConfig().mv_capacity(int(indptr[-1]), len(indptr) - 1, int(np.diff(indptr).max(initial=0))) > 0


def _vector_csr_api(x, data, indices, indptr, shape, implementation):
    return binary_csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=True,
        backend=implementation,
        config=prepared_config(indices, indptr, shape),
    )


def _csr_vector_api(x, data, indices, indptr, shape, implementation):
    return binary_csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
        config=prepared_config(indices, indptr, shape),
    )


def _matrix_csr_api(x, data, indices, indptr, shape, implementation):
    # x @ csr: csrmm expects input as [shape[0], cols] for transpose=True.
    return binary_csrmm(
        data,
        indices,
        indptr,
        x.T,
        shape=shape,
        transpose=True,
        backend=implementation,
    config=prepared_config(indices, indptr, shape)).T


def _csr_matrix_api(x, data, indices, indptr, shape, implementation):
    return binary_csrmm(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
        config=prepared_config(indices, indptr, shape),
    )


def _assert_hybrid_scratch_call(call, n_math, task_capacity, indptr_dtype):
    """A hybrid FFI call gets the math operands plus three scratch buffers made in the lowering."""
    _, out_info, ffi_kwargs, args, kwargs = call
    scratch = [((task_capacity,), indptr_dtype), ((task_capacity,), indptr_dtype), ((2,), jnp.int32)]
    assert len(args) == n_math + len(scratch)
    assert ffi_kwargs['input_output_aliases'] == {n_math + i: 1 + i for i in range(len(scratch))}
    assert kwargs == {'task_capacity': task_capacity}
    assert [(a.shape, a.dtype) for a in args[n_math:]] == [(s, jnp.dtype(d)) for s, d in scratch]
    assert [(o.shape, o.dtype) for o in out_info[1:]] == [(s, jnp.dtype(d)) for s, d in scratch]


_NT_BLOCK_NNZ_PER_ROW = 1024  # > 512 so nt_auto picks the block kernel


def _dense_csr_operands(rng, m, k, n, dtype, homo):
    """Build a CSR/dense pair whose average row length selects the block kernel."""
    indptr = np.arange(0, (m + 1) * _NT_BLOCK_NNZ_PER_ROW,
                       _NT_BLOCK_NNZ_PER_ROW, dtype=np.int32)
    indices = np.concatenate([
        np.sort(rng.choice(k, _NT_BLOCK_NNZ_PER_ROW, replace=False)).astype(np.int32)
        for _ in range(m)
    ])
    nnz = int(indptr[-1])
    weights = rng.standard_normal(1 if homo else nnz).astype(dtype)
    matrix = rng.random((k, n)) < 0.5
    return weights, indices, indptr, matrix


_CSRMV_TIER_NNZ_PER_ROW = [4, 15, 16, 64, 511, 512, 1024]


_ACT_M, _ACT_K, _ACT_N = 13, 9, 3


def _hp_matmul(a, b):
    """Full-precision float32 matmul (GPU defaults to TF32, ~1e-3 error)."""
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _activity_operands(homo, seed=0):
    rng = np.random.default_rng(seed)
    data, indices, indptr, dense = random_csr_numpy(rng, _ACT_M, _ACT_K)
    if homo:
        data = np.array([1.5], dtype=np.float32)
        dense = (dense != 0) * 1.5
    return jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr), dense


def _dense_from_data(data, indices, indptr, shape):
    """Differentiable dense form of a CSR matrix (homogeneous if ``data.size == 1``)."""
    rows = np.repeat(np.arange(shape[0]), np.diff(np.asarray(indptr)))
    cols = np.asarray(indices)
    vals = jnp.broadcast_to(data, (cols.size,)) if data.size == 1 else data
    return jnp.zeros(shape, dtype=data.dtype).at[rows, cols].add(vals)


_BAD_HYBRID_CALL = r'''
import sys
from pathlib import Path
import jax, jax.numpy as jnp
import brainevent._csr.binary.binary as binary_mod
from brainevent._op import load_cuda_file
from brainevent._csr.config import CSRConfig
cfg = CSRConfig()
mod = 'csr_hybrid_spmv_binary' + cfg.module_suffix()
root = Path(binary_mod.__file__).parent / 'hybrid_mv'
load_cuda_file(root / 'hybrid_spmv_binary.cu', name=mod, extra_include_paths=[str(root)],
               extra_cuda_cflags=cfg.compile_flags(), allow_cuda_graph=False)
cap = 4
weights = jnp.ones(3, jnp.float32)
indices = jnp.array([0, 1, 2], jnp.int32)
indptr = jnp.array([0, 2, 3], jnp.int32)
vector = jnp.array([1, 0], jnp.int32)  # wrong dtype: the _float kernel expects float32
tb = jnp.zeros(cap, jnp.int32); te = jnp.zeros(cap, jnp.int32); st = jnp.zeros(2, jnp.int32)
outs = (jax.ShapeDtypeStruct((3,), jnp.float32), jax.ShapeDtypeStruct((cap,), jnp.int32),
        jax.ShapeDtypeStruct((cap,), jnp.int32), jax.ShapeDtypeStruct((2,), jnp.int32))
try:
    r = jax.ffi.ffi_call(mod + '.binary_csrmv_wat_hybrid_hetero_f32_float', outs,
                         input_output_aliases={4: 1, 5: 2, 6: 3})(
        weights, indices, indptr, vector, tb, te, st, task_capacity=cap)
    jax.block_until_ready(r)
except Exception as e:
    print('RAISED', type(e).__name__, e)
    sys.exit(0)
print('NO ERROR')
sys.exit(3)
'''


def _weighted_tol(dtype_name):
    # float16/bfloat16 events hold exact values (2, -1, 0.5) and the product is
    # computed in the weight dtype, but the event cotangent/tangent checks are
    # rounded to the event dtype.
    if dtype_name in ('float16', 'bfloat16'):
        return dict(rtol=2e-2, atol=2e-2)
    return dict(rtol=1e-4, atol=1e-4)


def _nonfinite_mv(case, shape, transpose, backend):
    indptr = jnp.asarray(case.indptr)
    return binary_csrmv(
        jnp.asarray(case.data), jnp.asarray(case.indices), indptr, case.events,
        shape=shape,
        transpose=transpose, backend=backend, weighted=case.weighted,
        config=prepared_config(case.indices, case.indptr, shape),
    )


def _nonfinite_mm(case, shape, transpose, backend):
    indptr = jnp.asarray(case.indptr)
    return binary_csrmm(
        jnp.asarray(case.data), jnp.asarray(case.indices), indptr, case.events,
        shape=shape,
        transpose=transpose, backend=backend, weighted=case.weighted,
        config=prepared_config(case.indices, case.indptr, shape),
    )


def _repeated_entry_csr(m, k, nse, homo, seed=0):
    """Row-sorted CSR of ``nse`` entries drawn with replacement, and its dense sum."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.integers(0, m, nse))
    cols = rng.integers(0, k, nse)
    w = np.full(nse, 1.5) if homo else rng.standard_normal(nse)
    dense = np.zeros((m, k))
    np.add.at(dense, (rows, cols), w)
    data = jnp.asarray([1.5] if homo else w, jnp.float32)
    indptr = jnp.asarray(np.searchsorted(rows, np.arange(m + 1)), jnp.int32)
    return data, jnp.asarray(cols, jnp.int32), indptr, dense


@pytest.mark.parametrize('hint', [False, True])
def test_binary_csrmv_p_call_returns_one_output_and_static_task_capacity(hint):
    indices, indptr, heavy = _heavy_row_structure()
    max_row_nnz = heavy if hint else None
    weights = jnp.ones(indices.shape, jnp.float32)
    vector = jnp.array([True, True])

    def call(w, v):
        return binary_csrmv_p_call(w, indices, indptr, v, shape=(2, heavy), transpose=True,
                                   backend='jax_raw', max_row_nnz=max_row_nnz)

    out = call(weights, vector)
    assert len(out) == 1
    expected = np.ones(heavy, np.float32)
    expected[0] += 1.0
    np.testing.assert_allclose(np.asarray(out[0]), expected)

    eqn = _kernel_eqn(binary_csrmv_p, call, weights, vector)
    assert (len(eqn.invars), len(eqn.outvars)) == (5, 1)
    _assert_task_capacity_param(eqn, indptr, max_row_nnz)


def test_binary_csrmm_p_call_returns_one_output_without_mv_workspace():
    indices, indptr, heavy = _heavy_row_structure()
    weights = jnp.ones(indices.shape, jnp.float32)
    matrix = jnp.array([[True, False, True], [True, True, False]])

    def call(w, b):
        return binary_csrmm_p_call(w, indices, indptr, b, shape=(2, heavy), transpose=True,
                                   backend='jax_raw')

    out = call(weights, matrix)
    assert len(out) == 1
    expected = np.zeros((heavy, 3), np.float32)
    expected[:, :] = np.asarray(matrix[0], np.float32)
    expected[0] += np.asarray(matrix[1], np.float32)
    np.testing.assert_allclose(np.asarray(out[0]), expected)

    eqn = _kernel_eqn(binary_csrmm_p, call, weights, matrix)
    assert (len(eqn.invars), len(eqn.outvars)) == (5, 1)
    assert not {'task_capacity', 'max_row_nnz'} & eqn.params.keys()
    assert eqn.params['call_plan'] is None


@pytest.mark.slow
def test_binary_csrmv_benchmark_data_matches_p_call():
    config = next(_binary_csrmv_benchmark_data(platform='cpu'))
    assert len(config.args) == 4  # math operands only

    out = binary_csrmv_p_call(
        *config.args,
        **config.kernel_kwargs,
        backend='jax_raw',
    )

    assert len(out) == 1
    assert out[0].shape == (config.kernel_kwargs['shape'][0],)


@pytest.mark.slow
def test_binary_csrmm_benchmark_data_matches_p_call():
    config = _binary_csrmm_benchmark_data(platform='cpu')[0]
    assert len(config.args) == 4  # math operands only

    out = binary_csrmm_p_call(
        *config.args,
        **config.kernel_kwargs,
        backend='jax_raw',
    )

    assert len(out) == 1
    assert out[0].shape == (config.kernel_kwargs['shape'][0], config.args[-1].shape[1])


@pytest.mark.slow
def test_binary_csrmv_batching_vector_axis_0_uses_csrmm_fast_path(monkeypatch):
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    vector = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
    math_out = jnp.ones((1, 2), dtype=jnp.float32)
    calls = []

    def fake_csrmm_p_call(w, idx, ptr, b, *, shape, transpose, backend, weighted=False,
                          config=None):
        calls.append((w, idx, ptr, b, shape, transpose, backend, config))
        return (math_out,)

    monkeypatch.setattr(
        "brainevent._csr.binary.binary.binary_csrmm_p_call",
        fake_csrmm_p_call,
    )

    out, out_axes = _csrmv_batching(
        (weights, indices, indptr, vector, jnp.empty(0, jnp.uint32)),
        (None, None, None, 0, None),
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        max_row_nnz=2,
    )

    assert calls
    np.testing.assert_array_equal(calls[0][3], vector.T)
    assert isinstance(calls[0][7], CSRConfig)
    assert out == (math_out,)
    assert list(out_axes) == [1]


@pytest.mark.slow
def test_binary_csrmv_batching_vector_axis_1_uses_csrmm_fast_path(monkeypatch):
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    vector = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
    math_out = jnp.ones((1, 2), dtype=jnp.float32)
    calls = []

    def fake_csrmm_p_call(w, idx, ptr, b, *, shape, transpose, backend, weighted=False,
                          config=None):
        calls.append((w, idx, ptr, b, shape, transpose, backend, config))
        return (math_out,)

    monkeypatch.setattr(
        "brainevent._csr.binary.binary.binary_csrmm_p_call",
        fake_csrmm_p_call,
    )

    out, out_axes = _csrmv_batching(
        (weights, indices, indptr, vector, jnp.empty(0, jnp.uint32)),
        (None, None, None, 1, None),
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
    )

    assert calls
    assert calls[0][3] is vector
    assert isinstance(calls[0][7], CSRConfig)
    assert out == (math_out,)
    assert list(out_axes) == [1]


@pytest.mark.slow
def test_binary_csrmv_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        events = jnp.array([True, False, True])

        got = binary_csrmv(
            weights,
            indices,
            indptr,
            events,
            shape=(2, 3),
            backend='jax_raw',
            config=prepared_config(indices, indptr, (2, 3)),
        )

        assert jnp.allclose(got, jnp.array([3.0, 4.0], dtype=jnp.float32))


@pytest.mark.slow
def test_binary_csrmv_rejects_int64_indices_with_int32_indptr():
    with _jax_x64_enabled():
        weights = jnp.ones(2, dtype=jnp.float32)
        indices = jnp.array([0, 1], dtype=jnp.int64)
        indptr = jnp.array([0, 2], dtype=jnp.int32)
        events = jnp.ones(2, dtype=bool)

        with pytest.raises(AssertionError, match="Indices must be int32"):
            binary_csrmv(
                weights,
                indices,
                indptr,
                events,
                shape=(1, 2),
                backend='jax_raw',
                config=prepared_config(indices, indptr, (1, 2)),
            )


@pytest.mark.slow
def test_binary_csrmm_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        events = jnp.array([[True, False], [False, True], [True, True]])

        got = binary_csrmm(weights, indices, indptr, events, shape=(2, 3), backend='jax_raw', config=prepared_config(indices, indptr, (2, 3)))
        expected = jnp.array([[3.0, 2.0], [4.0, 7.0]], dtype=jnp.float32)

        assert jnp.allclose(got, expected)


@pytest.mark.slow
def test_binary_csrmm_rejects_unsigned_structure_dtype():
    weights = jnp.ones(2, dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.uint32)
    indptr = jnp.array([0, 2], dtype=jnp.uint32)
    events = jnp.ones((2, 1), dtype=bool)

    with pytest.raises(AssertionError, match="Indices must be int32"):
        binary_csrmm(weights, indices, indptr, events, shape=(1, 2), backend='jax_raw')


@pytest.mark.slow
def test_binary_csrmv_jax_csr_kernel_homo_bool_casts_indices_to_indptr_dtype():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        kernel = _binary_csrmv_cusparse_kernel(
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct((3,), jnp.bool_),
            (2, 3),
            False,
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            outs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        )

        got = kernel(
            jnp.array([2.0], dtype=jnp.float32),
            jnp.array([0, 2, 1, 2], dtype=jnp.int32),
            indptr,
            jnp.array([True, False, True]),
        )[0]

        assert jnp.allclose(got, jnp.array([4.0, 2.0], dtype=jnp.float32))


@pytest.mark.slow
def test_binary_csrmv_jax_csr_kernel_hetero_float_transpose():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        kernel = _binary_csrmv_cusparse_kernel(
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.float32),
            (2, 3),
            True,
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            outs=[jax.ShapeDtypeStruct((3,), jnp.float32)],
        )

        got = kernel(
            jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
            jnp.array([0, 2, 1, 2], dtype=jnp.int32),
            indptr,
            jnp.array([1.0, -1.0], dtype=jnp.float32),
        )[0]

        # Negative events are active (``x != 0``).
        assert jnp.allclose(got, jnp.array([1.0, 3.0, 6.0], dtype=jnp.float32))


@pytest.mark.slow
def test_binary_csrmm_jax_csr_kernel_homo_bool_casts_indices_to_indptr_dtype():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        kernel = _binary_csrmm_cusparse_kernel(
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct((3, 2), jnp.bool_),
            (2, 3),
            False,
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            outs=[jax.ShapeDtypeStruct((2, 2), jnp.float32)],
        )

        got = kernel(
            jnp.array([2.0], dtype=jnp.float32),
            jnp.array([0, 2, 1, 2], dtype=jnp.int32),
            indptr,
            jnp.array([[True, False], [False, True], [True, True]]),
        )[0]

        expected = jnp.array([[4.0, 2.0], [2.0, 4.0]], dtype=jnp.float32)
        assert jnp.allclose(got, expected)


@pytest.mark.slow
def test_binary_csrmm_jax_csr_kernel_hetero_float_transpose():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        kernel = _binary_csrmm_cusparse_kernel(
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((2, 2), jnp.float32),
            (2, 3),
            True,
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            outs=[jax.ShapeDtypeStruct((3, 2), jnp.float32)],
        )

        got = kernel(
            jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
            jnp.array([0, 2, 1, 2], dtype=jnp.int32),
            indptr,
            jnp.array([[1.0, -1.0], [-1.0, 1.0]], dtype=jnp.float32),
        )[0]

        # Negative events are active (``x != 0``), so every entry of B fires.
        expected = jnp.array([[1.0, 1.0], [3.0, 3.0], [6.0, 6.0]], dtype=jnp.float32)
        assert jnp.allclose(got, expected)


@pytest.mark.slow
class TestVectorCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        m, n = 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = vector_csr(x, data, indices, indptr, (m, n))

        for implementation in CSRMV_IMPLEMENTATIONS:
            y = _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((x, indptr, indices, y2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr_vmap_vector(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        with jax.checking_leaks():
            n_batch, m, n = 10, 20, 40
            xs = brainstate.random.rand(n_batch, m) < 0.1
            indptr, indices = get_csr(m, n, 0.1)

            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            y2 = brainstate.transform.vmap2(lambda x: vector_csr(x, data, indices, indptr, (m, n)))(xs)

            for implementation in CSRMV_IMPLEMENTATIONS:
                y = brainstate.transform.vmap2(
                    lambda x: _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
                )(xs)
                assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        m, n = 20, 40
        v = brainstate.random.rand(n) < 0.1
        indptr, indices = get_csr(m, n, 0.2)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = csr_vector(v, data, indices, indptr, (m, n))

        for implementation in CSRMV_IMPLEMENTATIONS:
            y = _csr_vector_api(v, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((v, indptr, indices, y2))

    def _test_vjp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)
        x = (x < 0.6).astype(float)

        indptr, indices = get_csr(n_in, n_out, 0.2, replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, shape, implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, shape, implementation)
            return r.sum()

        r = jax.grad(f_api, argnums=(0, 1))(x, w)

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, w)
        assert jnp.allclose(r[0], r2[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(r[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, homo_w, replace, transpose):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._test_vjp(
                implementation=implementation,
                homo_w=homo_w,
                replace=replace,
                transpose=transpose,
            )

    def _test_jvp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in if transpose else n_out)
        x = (x < 0.6).astype(float)

        indptr, indices = get_csr(n_in, n_out, 0.1, replace=replace)

        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, shape, implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, shape, implementation)
            return r

        o1, r1 = jax.jvp(f_api, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r

        o2, r2 = jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, o1, r1, o2, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, homo_w, replace, transpose):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._test_jvp(
                implementation=implementation,
                homo_w=homo_w,
                replace=replace,
                transpose=transpose,
            )


@pytest.mark.slow
class TestBatchingVectorCSR:
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        implementation = self._implementation
        if transpose:
            y1 = _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
            y2 = vector_csr(x, data, indices, indptr, (m, n))
        else:
            y1 = _csr_vector_api(x, data, indices, indptr, (m, n), implementation)
            y2 = csr_vector(x, data, indices, indptr, (m, n))
        return jnp.allclose(y1, y2)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda x: self._run(x, data, indices, indptr, m, n))(xs)
            assert jnp.all(res)

        jax.block_until_ready((xs, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda data: self._run(x, data, indices, indptr, m, n))(data)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda ind: self._run(x, data, ind, indptr, m, n))(indices)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, (m, n), implementation)
            return r.sum()

        r1 = jax.grad(f_api, argnums=(0, 1))(x, data)

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_vector(x, w, indices, indptr, shape=(m, n))
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, data)

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_vjp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda x: self._run_vjp(x, data, indices, indptr, m, n))(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_vjp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_vjp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_vjp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_vjp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, (m, n), implementation)
            return r

        r1 = jax.jvp(f_api, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_vector(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_jvp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda x: self._run_jvp(x, data, indices, indptr, m, n))(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_jvp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_jvp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_jvp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_jvp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))


@pytest.mark.slow
class TestMatrixCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = matrix_csr(x, data, indices, indptr, (m, n))

        for implementation in CSRMM_IMPLEMENTATIONS:
            y = _matrix_csr_api(x, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, y2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        m, n, k = 20, 40, 10
        matrix = brainstate.random.rand(n, k) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = csr_matrix(matrix, data, indices, indptr, (m, n))

        for implementation in CSRMM_IMPLEMENTATIONS:
            y = _csr_matrix_api(matrix, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2)

        jax.block_until_ready((matrix, indptr, indices, y2))


@pytest.mark.slow
class TestBatchingMatrixCSR:
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        implementation = self._implementation
        if transpose:
            y1 = _matrix_csr_api(x, data, indices, indptr, (m, n), implementation)
            y2 = matrix_csr(x, data, indices, indptr, (m, n))
        else:
            y1 = _csr_matrix_api(x, data, indices, indptr, (m, n), implementation)
            y2 = csr_matrix(x, data, indices, indptr, (m, n))
        return jnp.allclose(y1, y2)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_matrix(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        xs = brainstate.random.rand(b, k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda x: self._run(x, data, indices, indptr, m, n))(xs)
            assert jnp.all(res)

        jax.block_until_ready((xs, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda data: self._run(x, data, indices, indptr, m, n))(data)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            # Traced indices have no prepared plan: native MM uses the portable fallback.
            res = brainstate.transform.vmap2(lambda ind: self._run(x, data, ind, indptr, m, n))(indices)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _matrix_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_matrix_api(x, w, indices, indptr, (m, n), implementation)
            return r.sum()

        r1 = jax.grad(f_api, argnums=(0, 1))(x, data)

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_matrix(x, w, indices, indptr, shape=(m, n))
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, data)

        return r1, r2

    @pytest.mark.parametrize('transpose', [True, False])
    def test_vmap_matrix_vjp(self, transpose):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        if transpose:
            xs = brainstate.random.rand(b, n, m) < 0.1
        else:
            xs = brainstate.random.rand(b, n, k) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(
                lambda x: self._run_vjp(x, data, indices, indptr, m, n, transpose=transpose)
            )(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_vjp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_vjp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_vjp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            # Traced indices have no prepared plan: native MM uses the portable fallback.
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_vjp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _matrix_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_matrix_api(x, w, indices, indptr, (m, n), implementation)
            return r

        r1 = jax.jvp(f_api, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_matrix(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_jvp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        xs = brainstate.random.rand(b, k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda x: self._run_jvp(x, data, indices, indptr, m, n))(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_jvp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_jvp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_jvp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            # Traced indices have no prepared plan: native MM uses the portable fallback.
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_jvp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))


def test_binary_csrmv_gpu_cusparse_backend_names():
    backends = binary_csrmv_p.available_backends('gpu')

    assert 'cusparse' in backends
    # Legacy names were removed / renamed.
    assert 'BCOO_cusparse' not in backends
    assert 'JAX_cusparse' not in backends


def test_binary_csrmm_gpu_cusparse_backend_names():
    backends = binary_csrmm_p.available_backends('gpu')

    assert 'cusparse' in backends
    # Legacy names were removed / renamed.
    assert 'BCOO_cusparse' not in backends
    assert 'JAX_cusparse' not in backends


@pytest.mark.parametrize(
    'factory,args,kwargs',
    [
        (
            binary_mod._binary_csrmv_cuda_kernel,
            (shape_of(jnp.float32), shape_of(jnp.bool_), False),
            {'outs': [shape_of(jnp.float32)]},
        ),
        (
            binary_mod._binary_csrmm_cuda_kernel,
            (shape_of(jnp.float32), shape_of(jnp.bool_, (2, 1)), False),
            {'outs': [shape_of(jnp.float32, (1, 1))]},
        ),
    ],
)
def test_cuda_kernel_generators_reject_int64_indices_before_loading_cuda(factory, args, kwargs):
    call_kwargs = cuda_kwargs()
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match="indices with dtype int32"):
        factory(*args, **call_kwargs)


def test_binary_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(binary_mod, "load_cuda_file", lambda path, name, **kwargs: load_calls.append((path, name, kwargs)))
    monkeypatch.setattr(binary_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    with jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)
        task_capacity = binary_mod._task_capacity(indices, indptr, None)
        common = {**cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64), 'task_capacity': task_capacity}

        mv_kernel = binary_mod._binary_csrmv_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (2,)),
            False,
            **{**common, 'outs': [shape_of(jnp.float32)]},
        )
        mv_out = mv_kernel(jnp.array([2.0], dtype=jnp.float32), indices, indptr, jnp.array([True, False]))

        mv_t_kernel = binary_mod._binary_csrmv_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (1,)),
            True,
            **{**common, 'outs': [shape_of(jnp.float32)]},
        )
        mv_t_out = mv_t_kernel(jnp.array([2.0], dtype=jnp.float32), indices, indptr, jnp.array([True]))

        mm_nt_kernel = binary_mod._binary_csrmm_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (2, 1)),
            False,
            **{**common, 'outs': [shape_of(jnp.float32, (1, 1))]},
        )
        mm_nt_out = mm_nt_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32), indices, indptr, jnp.array([[True], [False]]), jnp.empty(0, jnp.uint32),
        )

        cfg = CSRConfig().prepare_structure(indices, indptr, shape=(1, 2),
            device_info=CSRDeviceInfo('gpu', 'test', 4 * 1024**2, 8))
        spec, mask = cfg.kernel_args()
        mm_t_kernel = binary_mod._binary_csrmm_cuda_kernel(
            shape_of(jnp.float32, (2,)), shape_of(jnp.float32, (1, 1)), True,
            **{**common, 'outs': [shape_of(jnp.float32, (2, 1))],
               'config_spec': spec, 'call_plan': cfg.mm_plan(1, jnp.float32)},
        )
        mm_t_out = mm_t_kernel(jnp.array([2.0, 3.0], jnp.float32), indices, indptr,
                              jnp.array([[1.0]], jnp.float32), mask)

    assert [strip_hybrid_suffix(name) for _, name, _ in load_calls] == [
        'csr_binary_csrmv',
        'csr_hybrid_spmv_binary',
        'csr_binary_csrmm',
        'csr_hybrid_mm_row_binary',
    ]
    assert [strip_hybrid_suffix(call[0]) for call in ffi_calls] == [
        'csr_binary_csrmv.binary_csrmv_nt_auto_homo_f32_bool',
        'csr_hybrid_spmv_binary.binary_csrmv_wat_hybrid_homo_f32_bool',
        'csr_binary_csrmm.binary_csrmm_nt_auto_hetero_f32_bool',
        'csr_hybrid_mm_row_binary.batched_spmm_v3_f32_float',
    ]
    # Kernels return only the math output, in the logical layout (the hybrid
    # mat-mat transposes its physical ``(n, k)`` result back).
    outs = (mv_out, mv_t_out, mm_nt_out, mm_t_out)
    assert [len(out) for out in outs] == [1, 1, 1, 1]
    assert [out[0].shape for out in outs] == [(2,), (2,), (1, 1), (2, 1)]
    # Direct kernels take the four math operands; hybrid ones add their own scratch.
    assert len(ffi_calls[0][3]) == 4 and len(ffi_calls[2][3]) == 4
    _assert_hybrid_scratch_call(ffi_calls[1], 4, task_capacity, jnp.int64)
    assert ffi_calls[3][3][3].dtype == jnp.uint32
    assert ffi_calls[3][4]["expected_sms"] == 8


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@with_jax_x64
def test_binary_csrmv_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    vector = jnp.array([True, False], dtype=jnp.bool_) if transpose else jnp.array([True, False, True])

    got = binary_csrmv(data, indices, indptr64, vector, shape=(2, 3), transpose=transpose,
                       backend='cuda_raw', config=prepared_config(indices, indptr64, (2, 3)))
    expected = binary_csrmv(data, indices, indptr32, vector, shape=(2, 3), transpose=transpose,
                            backend='jax_raw', config=prepared_config(indices, indptr32, (2, 3)))

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@with_jax_x64
def test_binary_csrmm_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    matrix = (
        jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        if transpose else
        jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    )

    got = binary_csrmm(data, indices, indptr64, matrix, shape=(2, 3), transpose=transpose,
                       backend='cuda_raw', config=prepared_config(indices, indptr64, (2, 3)))
    expected = binary_csrmm(data, indices, indptr32, matrix, shape=(2, 3), transpose=transpose,
                            backend='jax_raw', config=prepared_config(indices, indptr32, (2, 3)))

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu_backend
@pytest.mark.parametrize("event_dtype", [jnp.int8, jnp.uint8, jnp.float64])
def test_binary_csrmm_cuda_normalizes_numeric_events_to_bool_abi(event_dtype):
    """Route non-native numeric MM events through the boolean CUDA ABI."""
    with jax.enable_x64():
        data = jnp.array([2.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        matrix = jnp.array([[1, 0], [0, 1]], dtype=event_dtype)

        got = binary_csrmm(
            data,
            indices,
            indptr,
            matrix,
            shape=(2, 3),
            transpose=True,
            backend="cuda_raw",
            config=prepared_config(indices, indptr, (2, 3)),
        )
        expected = binary_csrmm(
            data,
            indices,
            indptr,
            matrix,
            shape=(2, 3),
            transpose=True,
            backend="jax_raw",
            config=prepared_config(indices, indptr, (2, 3)),
        )

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu_backend
@pytest.mark.slow
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmm_nt_block_shared_memory_is_large_enough(homo):
    """float64 ``nt_auto`` must not fault when it routes to the block kernel."""
    with jax_x64_enabled():
        rng = np.random.default_rng(0)
        m, k, n = 32, 2048, 64
        weights, indices, indptr, matrix = _dense_csr_operands(
            rng, m, k, n, np.float64, homo
        )
        kwargs = dict(shape=(m, k), transpose=False)
        args = (jnp.asarray(weights), jnp.asarray(indices),
                jnp.asarray(indptr), jnp.asarray(matrix))

        got = binary_csrmm(*args, backend='cuda_raw', **kwargs)
        expected = binary_csrmm(*args, backend='jax_raw', **kwargs)

        assert jnp.allclose(got, expected, rtol=1e-10, atol=1e-10)


@requires_gpu_backend
@pytest.mark.slow
@pytest.mark.parametrize('nnz_per_row', _CSRMV_TIER_NNZ_PER_ROW)
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmv_matches_reference_across_dispatch_tiers(nnz_per_row, homo):
    """Every ``nt_auto`` row-mapping strategy must agree with the reference."""
    with jax_x64_enabled():
        rng = np.random.default_rng(nnz_per_row)
        m, k = 256, 2048
        indptr = np.arange(0, (m + 1) * nnz_per_row, nnz_per_row, dtype=np.int32)
        indices = np.concatenate([
            np.sort(rng.choice(k, nnz_per_row, replace=False)).astype(np.int32)
            for _ in range(m)
        ])
        nnz = int(indptr[-1])
        weights = rng.standard_normal(1 if homo else nnz).astype(np.float64)
        vector = rng.random(k) < 0.5

        kwargs = dict(shape=(m, k), transpose=False)
        args = (jnp.asarray(weights), jnp.asarray(indices),
                jnp.asarray(indptr), jnp.asarray(vector))

        got = binary_csrmv(*args, backend='cuda_raw', **kwargs)
        expected = binary_csrmv(*args, backend='jax_raw', **kwargs)

        assert jnp.allclose(got, expected, rtol=1e-10, atol=1e-10)


@requires_gpu_backend
@pytest.mark.slow
def test_binary_csrmv_warp_tier_handles_ragged_and_empty_rows():
    """The warp kernel must tolerate empty rows and a row count off the tiling.

    One warp handles one row and the grid is strided, so a row count that is
    not a multiple of the warps-per-block, mixed with empty and over-long rows,
    is the case where an off-by-one in the row loop or a divergent lane in the
    shuffle reduction would show up.
    """
    with jax_x64_enabled():
        rng = np.random.default_rng(7)
        k = 1024
        # average stays in the warp tier while individual rows vary wildly
        row_lengths = np.array([0, 1, 33, 0, 200, 64, 0, 7, 512, 90, 0, 31, 63, 65, 128],
                               dtype=np.int32)
        m = row_lengths.size
        assert 16 <= row_lengths.sum() // m < 512, 'must land in the warp tier'
        indptr = np.concatenate([[0], np.cumsum(row_lengths)]).astype(np.int32)
        indices = np.concatenate(
            [np.sort(rng.choice(k, int(n), replace=False)).astype(np.int32)
             for n in row_lengths]
        ).astype(np.int32)
        weights = rng.standard_normal(int(indptr[-1])).astype(np.float64)
        vector = rng.random(k) < 0.5

        kwargs = dict(shape=(m, k), transpose=False)
        args = (jnp.asarray(weights), jnp.asarray(indices),
                jnp.asarray(indptr), jnp.asarray(vector))

        got = binary_csrmv(*args, backend='cuda_raw', **kwargs)
        expected = binary_csrmv(*args, backend='jax_raw', **kwargs)

        assert jnp.allclose(got, expected, rtol=1e-10, atol=1e-10)
        # empty rows must be written, not left uninitialised
        assert jnp.all(jnp.asarray(got)[row_lengths == 0] == 0.0)


def test_binary_csrmv_cuda_source_keeps_three_dispatch_tiers():
    """``nt_auto`` must keep the warp tier between the thread and block tiers.

    Source-level so it runs without a GPU. A two-tier dispatcher still computes
    the right answer, so the numeric tests above cannot catch its removal.
    """
    from pathlib import Path

    cu = Path(binary_mod.__file__).with_name('binary_csrmv.cu').read_text()
    for kind in ('homo', 'hetero'):
        assert f'DEFINE_CSRMV_NT_WARP_{kind.upper()}(' in cu
        assert f'_csrmv_nt_warp_{kind}_kern' in cu
    # the three-way ladder itself
    assert cu.count('if (avg_nnz < 16)') == 2
    assert cu.count('} else if (avg_nnz < 512) {') == 2
    assert 'avg_nnz <= 512' not in cu, 'two-tier dispatch came back'


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmv_event_is_active_iff_nonzero(backend, dtype_name, homo, transpose):
    """Negative and fractional events are active and weights are never scaled by them."""
    with event_dtype_context(dtype_name):
        data, indices, indptr, dense = _activity_operands(homo)
        x = event_values(dtype_name, _ACT_M if transpose else _ACT_K)
        got = binary_csrmv(
            data, indices, indptr, x, shape=(_ACT_M, _ACT_K), transpose=transpose,
            backend=backend,
            config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)),
        )
        mask = event_mask(x)
        expected = mask @ dense if transpose else dense @ mask
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmm_event_is_active_iff_nonzero(backend, dtype_name, homo, transpose):
    """Matrix events follow the same ``x != 0`` rule on every backend."""
    with event_dtype_context(dtype_name):
        data, indices, indptr, dense = _activity_operands(homo, seed=1)
        B = event_values(dtype_name, (_ACT_M if transpose else _ACT_K, _ACT_N), seed=1)
        got = binary_csrmm(
            data, indices, indptr, B, shape=(_ACT_M, _ACT_K), transpose=transpose,
            backend=backend,
            config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)),
        )
        mask = event_mask(B)
        expected = dense.T @ mask if transpose else dense @ mask
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
def test_binary_csrmv_nan_event_is_active(backend):
    data, indices, indptr, dense = _activity_operands(homo=False)
    x = jnp.zeros(_ACT_M, dtype=jnp.float32).at[jnp.array([0, 5])].set(jnp.nan)
    got = binary_csrmv(
        data, indices, indptr, x, shape=(_ACT_M, _ACT_K), transpose=True,
        backend=backend,
        config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)),
    )
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), event_mask(x) @ dense, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmv_gradients_use_event_mask(backend, homo, transpose):
    """#198: weight cotangents use the activity mask, event cotangents are straight-through."""
    data, indices, indptr, _ = _activity_operands(homo, seed=2)
    shape = (_ACT_M, _ACT_K)
    x = event_values('float32', _ACT_M if transpose else _ACT_K, seed=2)

    def op(d, v):
        return binary_csrmv(
            d, indices, indptr, v, shape=shape, transpose=transpose,
            backend=backend,
            config=prepared_config(indices, indptr, shape))

    def dense_op(d, z):
        W = _dense_from_data(d, indices, indptr, shape)
        return _hp_matmul(z, W) if transpose else _hp_matmul(W, z)

    check_event_op_grads(op, dense_op, data, x, mode='binary')


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmm_gradients_use_event_mask(backend, homo, transpose):
    """#198 for the matrix product (the hetero branch goes through SDDMM)."""
    data, indices, indptr, _ = _activity_operands(homo, seed=3)
    shape = (_ACT_M, _ACT_K)
    B = event_values('float32', (_ACT_M if transpose else _ACT_K, _ACT_N), seed=3)

    def op(d, v):
        return binary_csrmm(
            d, indices, indptr, v, shape=shape, transpose=transpose,
            backend=backend,
            config=prepared_config(indices, indptr, shape))

    def dense_op(d, z):
        W = _dense_from_data(d, indices, indptr, shape)
        return _hp_matmul(W.T, z) if transpose else _hp_matmul(W, z)

    check_event_op_grads(op, dense_op, data, B, mode='binary')


def test_issue_198_binary_csr_weight_gradient_regression():
    """Exact #198 repro: a count of 2 contributes 1 to the weight gradient, not 2."""
    import brainevent as be

    W = jnp.array([[1., 10.], [100., 0.], [0., 1000.]])
    csr = be.CSR.fromdense(W)
    mk = lambda d: be.CSR(
        (d, csr.indices, csr.indptr), shape=W.shape)

    x = jnp.array([0., 2., 1.])
    g = jax.grad(lambda d: (be.BinaryArray(x) @ mk(d)).sum())(csr.data)
    np.testing.assert_allclose(np.asarray(g), [0., 0., 1., 1.])

    X = jnp.stack([x, x])
    g_mm = jax.grad(lambda d: (be.BinaryArray(X) @ mk(d)).sum())(csr.data)
    np.testing.assert_allclose(np.asarray(g_mm), [0., 0., 2., 2.])

    # Negative events are active: forward and gradient both count them once.
    xn = jnp.array([-1., 1., 0.])
    assert float((be.BinaryArray(xn) @ csr).sum()) == 111.
    g_neg = jax.grad(lambda d: (be.BinaryArray(xn) @ mk(d)).sum())(csr.data)
    np.testing.assert_allclose(np.asarray(g_neg), [1., 1., 1., 0.])


@requires_gpu
def test_issue_199_float64_events_hetero_float64_csr_default_backend():
    """#199: float64 spikes @ hetero float64 CSR on GPU must not abort and must be correct."""
    import brainevent as be

    with jax_x64_enabled():
        rng = np.random.default_rng(0)
        n_pre, n_post, p = 1000, 1000, 0.1
        W = (rng.random((n_pre, n_post)) < p) * rng.normal(size=(n_pre, n_post))
        csr = be.CSR.fromdense(jnp.asarray(W, dtype=jnp.float64))
        spikes = jnp.asarray(rng.random(n_pre) < 0.05, dtype=jnp.float64)
        spikes = spikes.at[:3].set(jnp.array([2.0, -1.0, 0.5]))
        out = jax.jit(lambda s: be.BinaryArray(s) @ csr)(spikes).block_until_ready()
        assert out.dtype == jnp.float64
        ref = (np.asarray(spikes) != 0).astype(np.float64) @ W
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10, atol=1e-10)


@requires_gpu
def test_hybrid_cuda_dtype_mismatch_raises_python_exception():
    """A failing host-side check in a hybrid kernel raises instead of aborting the process."""
    proc = run_python_isolated(_BAD_HYBRID_CALL)
    output = proc.stdout + proc.stderr
    assert proc.returncode >= 0, f'process killed by signal {-proc.returncode}:\n{output}'
    assert proc.returncode == 0, output
    assert 'expects matching vector dtype' in output, output


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype_name', WEIGHTED_EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmv_weighted_matches_dense_and_grads(backend, dtype_name, homo, transpose):
    """Weighted mat-vec is the exact product ``x @ W`` with exact derivatives."""
    with event_dtype_context(dtype_name):
        data, indices, indptr, _ = _activity_operands(homo, seed=4)
        shape = (_ACT_M, _ACT_K)
        x = event_values(dtype_name, _ACT_M if transpose else _ACT_K, seed=4)

        def op(d, v):
            return binary_csrmv(d, indices, indptr, v, shape=shape, transpose=transpose, backend=backend,
                                weighted=True, config=prepared_config(indices, indptr, shape))

        def dense_op(d, z):
            # Events are cast to the weight dtype (the kernels' compute dtype).
            W = _dense_from_data(d, indices, indptr, shape)
            z = z.astype(d.dtype)
            return _hp_matmul(z, W) if transpose else _hp_matmul(W, z)

        check_event_op_grads(op, dense_op, data, x, mode='weighted',
                             check_vmap=dtype_name == 'float32', **_weighted_tol(dtype_name))


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype_name', WEIGHTED_EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmm_weighted_matches_dense_and_grads(backend, dtype_name, homo, transpose):
    """Weighted mat-mat (hetero weight cotangent through the SDDMM path)."""
    with event_dtype_context(dtype_name):
        data, indices, indptr, _ = _activity_operands(homo, seed=5)
        shape = (_ACT_M, _ACT_K)
        B = event_values(dtype_name, (_ACT_M if transpose else _ACT_K, _ACT_N), seed=5)

        def op(d, v):
            return binary_csrmm(d, indices, indptr, v, shape=shape, transpose=transpose, backend=backend,
                                weighted=True, config=prepared_config(indices, indptr, shape))

        def dense_op(d, z):
            W = _dense_from_data(d, indices, indptr, shape)
            z = z.astype(d.dtype)
            return _hp_matmul(W.T, z) if transpose else _hp_matmul(W, z)

        check_event_op_grads(op, dense_op, data, B, mode='weighted',
                             check_vmap=dtype_name == 'float32', **_weighted_tol(dtype_name))


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmv_weighted_vmap_over_events_matches_dense(backend, homo, transpose):
    """vmap over the event axis lowers to the weighted mat-mat primitive."""
    data, indices, indptr, dense = _activity_operands(homo, seed=6)
    shape = (_ACT_M, _ACT_K)
    xs = jnp.stack([event_values('float32', _ACT_M if transpose else _ACT_K, seed=s) for s in range(3)])

    f = jax.vmap(lambda v: binary_csrmv(data, indices, indptr, v, shape=shape, transpose=transpose,
                                        backend=backend,
                                        weighted=True, config=prepared_config(indices, indptr, shape)))
    got = np.asarray(f(xs), dtype=np.float64)
    xs64 = np.asarray(xs, dtype=np.float64)
    expected = xs64 @ dense if transpose else xs64 @ dense.T
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
def test_binary_csrmv_weighted_heavy_rows_use_row_value(backend, homo):
    """Rows longer than one hybrid task (> 4096 entries) are scaled by their own event value."""
    rng = np.random.default_rng(7)
    m, k = 6, 9000
    data, indices, indptr, dense = heavy_row_csr_numpy(rng, m, k, row_nnz=8500)
    if homo:
        dense = (dense != 0) * 1.5
        data = np.array([1.5], np.float32)
    x = jnp.array([2.0, 0.0, -1.0, 0.5, 3.0, 0.0], dtype=jnp.float32)
    indptr = jnp.asarray(indptr)
    got = binary_csrmv(jnp.asarray(data), jnp.asarray(indices), indptr, x, shape=(m, k), transpose=True,
                       backend=backend, weighted=True, config=prepared_config(jnp.asarray(indices), indptr, (m, k)))
    np.testing.assert_allclose(np.asarray(got, np.float64), np.asarray(x, np.float64) @ dense,
                               rtol=1e-4, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
def test_binary_csrmm_weighted_heavy_rows_use_row_value(backend, homo):
    rng = np.random.default_rng(8)
    m, k = 5, 9000
    data, indices, indptr, dense = heavy_row_csr_numpy(rng, m, k, row_nnz=8500)
    if homo:
        dense = (dense != 0) * 1.5
        data = np.array([1.5], np.float32)
    B = jnp.array([[2.0, 0.0], [0.0, -1.0], [-1.0, 0.5], [0.5, 0.0], [0.0, 3.0]], dtype=jnp.float32)
    indptr = jnp.asarray(indptr)
    got = binary_csrmm(jnp.asarray(data), jnp.asarray(indices), indptr, B, shape=(m, k), transpose=True,
                       backend=backend, weighted=True, config=prepared_config(jnp.asarray(indices), indptr, (m, k)))
    np.testing.assert_allclose(np.asarray(got, np.float64), dense.T @ np.asarray(B, np.float64),
                               rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('weight_dtype', [jnp.float32, jnp.float16])
@pytest.mark.parametrize('homo', [True, False])
def test_binary_cuda_generators_select_weighted_abi(monkeypatch, weight_dtype, homo):
    """``weighted=True`` picks the ``_wv`` symbols and casts events to the weight dtype in the lowering."""
    ffi_calls = []
    monkeypatch.setattr(binary_mod, "load_cuda_file", lambda path, name, **kwargs: None)
    monkeypatch.setattr(binary_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    task_capacity = binary_mod._task_capacity(indices, indptr, None)
    n_w = 1 if homo else 2
    weights = jnp.ones(n_w, dtype=weight_dtype)
    sfx = {jnp.float32: '_f32', jnp.float16: '_f16'}[weight_dtype]
    hs = '_homo' if homo else '_hetero'
    cases = [
        (binary_mod._binary_csrmv_cuda_kernel, (2,), False, (1,),
         f'csr_binary_csrmv.binary_csrmv_nt_auto{hs}{sfx}_wv'),
        (binary_mod._binary_csrmv_cuda_kernel, (1,), True, (2,),
         f'csr_hybrid_spmv_weighted.binary_csrmv_wat_hybrid{hs}{sfx}_wv'),
        (binary_mod._binary_csrmm_cuda_kernel, (2, 3), False, (1, 3),
         f'csr_binary_csrmm.binary_csrmm_nt_auto{hs}{sfx}_wv'),
        (binary_mod._binary_csrmm_cuda_kernel, (1, 3), True, (2, 3),
         f'csr_hybrid_mm_row_weighted.batched_spmm_v3_weighted{"_homo" if homo else ""}{sfx}'),
    ]
    cfg = CSRConfig().prepare_structure(indices, indptr, shape=(1, 2),
        device_info=CSRDeviceInfo('gpu', 'test', 4 * 1024**2, 8))
    spec, mask = cfg.kernel_args()
    for factory, ev_shape, transpose, out_shape, expected_name in cases:
        ffi_calls.clear()
        events = jnp.full(ev_shape, -3, dtype=jnp.int8)
        kernel = factory(
            shape_of(weight_dtype, (n_w,)), shape_of(jnp.int8, ev_shape), transpose,
            **{**cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int32),
               'outs': [shape_of(weight_dtype, out_shape)], 'weighted': True, 'task_capacity': task_capacity,
               'config_spec': spec, 'call_plan': cfg.mm_plan(3, weight_dtype, weighted=True)},
        )
        is_mm = len(ev_shape) == 2
        kernel(weights, indices, indptr, events, *([mask] if is_mm else []))
        assert len(ffi_calls) == 1
        name, _, _, args, _ = ffi_calls[0]
        assert strip_hybrid_suffix(name) == expected_name
        event_arg = args[4].T if is_mm and transpose else args[3]
        assert event_arg.dtype == weight_dtype
        np.testing.assert_array_equal(np.asarray(event_arg, np.float32), np.full(ev_shape, -3.0))


def test_binary_csrmv_weighted_integer_events_have_zero_cotangent():
    """Integer weighted events are not differentiable: the event cotangent is float0."""
    data, indices, indptr, _ = _activity_operands(False, seed=9)
    x = event_values('int8', _ACT_K, seed=9)
    f = lambda d, v: binary_csrmv(d, indices, indptr, v, shape=(_ACT_M, _ACT_K), backend='jax_raw',
                                  weighted=True, config=prepared_config(indices, indptr, (_ACT_M, _ACT_K))).sum()
    g_d, g_x = jax.grad(f, argnums=(0, 1), allow_int=True)(data, x)
    assert g_x.dtype == jax.dtypes.float0
    W = np.asarray(_dense_from_data(data, indices, indptr, (_ACT_M, _ACT_K)))
    rows = np.repeat(np.arange(_ACT_M), np.diff(np.asarray(indptr)))
    np.testing.assert_allclose(np.asarray(g_d), np.asarray(x, np.float32)[np.asarray(indices)], rtol=1e-6)
    assert W.shape == (_ACT_M, _ACT_K) and rows.size == data.size


def test_binary_csrmv_weighted_default_is_binary():
    """``weighted`` defaults to ``False``: values are binarised."""
    data, indices, indptr, dense = _activity_operands(False, seed=10)
    x = event_values('float32', _ACT_K, seed=10)
    binary = binary_csrmv(data, indices, indptr, x, shape=(_ACT_M, _ACT_K), backend='jax_raw', config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)))
    weighted = binary_csrmv(data, indices, indptr, x, shape=(_ACT_M, _ACT_K), backend='jax_raw',
                            weighted=True, config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)))
    np.testing.assert_allclose(np.asarray(binary, np.float64), dense @ event_mask(x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(weighted, np.float64), dense @ np.asarray(x, np.float64),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('weighted', [False, True])
def test_binary_csrmv_batching_forwards_weighted(monkeypatch, axis, weighted):
    """The mat-vec batching rule keeps the event mode when lowering to mat-mat."""
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    vector = jnp.array([[2.0, 0.0], [0.0, -1.0]], dtype=jnp.float32)
    seen = []

    def fake_csrmm_p_call(w, idx, ptr, b, *, shape, transpose, backend, weighted=False,
                          config=None):
        seen.append(weighted)
        return (jnp.ones((1, 2), dtype=jnp.float32),)

    monkeypatch.setattr("brainevent._csr.binary.binary.binary_csrmm_p_call", fake_csrmm_p_call)
    axes = (None, None, None, axis, None)
    _csrmv_batching(
        (weights, indices, indptr, vector, jnp.empty(0, jnp.uint32)),
        axes, shape=(1, 2), transpose=False, backend="jax_raw", weighted=weighted,
    )
    assert seen == [weighted]


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('bad', tuple(NONFINITE_VALUES))
def test_binary_csrmv_inactive_nonfinite_weights_contribute_zero(backend, transpose, homo, kind, bad):
    shape = (7, 6)
    case = inactive_nonfinite_case(0, *shape, transpose=transpose, homo=homo, kind=kind, bad=bad)
    assert_select_reference(_nonfinite_mv(case, shape, transpose, backend), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('bad', tuple(NONFINITE_VALUES))
def test_binary_csrmm_inactive_nonfinite_weights_contribute_zero(backend, transpose, homo, kind, bad):
    shape = (7, 6)
    case = inactive_nonfinite_case(
        1, *shape, transpose=transpose, homo=homo, kind=kind, bad=bad, n_batch=4,
    )
    assert_select_reference(_nonfinite_mm(case, shape, transpose, backend), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
def test_binary_csrmm_nonfinite_weight_isolated_per_event_column(backend, transpose, kind):
    # A NaN weight whose event is active in some columns only: the other
    # columns stay exact (cuSPARSE included: it falls back to the select path).
    shape = (7, 6)
    case = inactive_nonfinite_case(
        2, *shape, transpose=transpose, homo=False, kind=kind, bad='nan', n_batch=4, partial=True,
    )
    assert np.isnan(case.reference).any() and np.isfinite(case.reference).any()
    assert_select_reference(_nonfinite_mm(case, shape, transpose, backend), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('mm', [False, True])
def test_binary_csr_inactive_nonfinite_weights_heavy_rows(backend, homo, kind, mm):
    # Rows longer than the hybrid thread-per-row threshold (CUDA task path).
    shape = (4, 300)
    case = inactive_nonfinite_case(
        3, *shape, transpose=True, homo=homo, kind=kind, bad='nan',
        n_batch=3 if mm else None, heavy_row_nnz=200,
    )
    fn = _nonfinite_mm if mm else _nonfinite_mv
    if mm and backend not in CSRMM_IMPLEMENTATIONS:
        pytest.skip(f'{backend} has no csrmm kernel')
    assert_select_reference(fn(case, shape, True, backend), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('mm', [False, True])
def test_binary_csr_active_nan_weight_still_propagates(backend, transpose, mm):
    shape = (7, 6)
    case = inactive_nonfinite_case(
        4, *shape, transpose=transpose, homo=False, kind='bool', bad='nan',
        n_batch=3 if mm else None,
    )
    data = case.data.copy()
    e = int(np.flatnonzero(case.active[:, 0])[0])
    data[e] = np.nan
    case = case._replace(data=data)
    out = np.asarray((_nonfinite_mm if mm else _nonfinite_mv)(case, shape, transpose, backend))
    target = case.indices[e] if transpose else int(np.searchsorted(case.indptr, e, side='right') - 1)
    assert np.isnan(out[target] if not mm else out[target, 0])
    assert np.isfinite(out).sum() > 0


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('mm', [False, True])
def test_binary_csr_weight_cotangent_of_inactive_entries_is_exact_zero(backend, transpose, kind, mm):
    # A NaN output cotangent must not leak into the weight cotangent of an
    # entry whose event is inactive (select, not ``ct * 0``).
    shape = (7, 6)
    case = inactive_nonfinite_case(
        5, *shape, transpose=transpose, homo=False, kind=kind, bad='nan',
        n_batch=3 if mm else None,
    )
    data = np.where(np.isfinite(case.data), case.data, 1.0).astype(np.float32)
    indptr = jnp.asarray(case.indptr)
    op = binary_csrmm if mm else binary_csrmv
    config = prepared_config(case.indices, case.indptr, shape)

    def f(d):
        return op(d, jnp.asarray(case.indices), indptr, case.events, shape=shape,
                  transpose=transpose, backend=backend, weighted=case.weighted, config=config)

    out, vjp = jax.vjp(f, jnp.asarray(data))
    (ct_w,) = vjp(jnp.full_like(out, jnp.nan))
    ct_w = np.asarray(ct_w)
    inactive = case.inactive_edge
    assert inactive.any() and (~inactive).any()
    np.testing.assert_array_equal(ct_w[inactive], 0.0)
    assert np.isnan(ct_w[~inactive]).all()


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('mm', [False, True])
def test_binary_csr_homo_weight_cotangent_ignores_nan_on_idle_outputs(backend, transpose, kind, mm):
    # A NaN output cotangent at an output that received only inactive events
    # must not reach the homogeneous weight cotangent (``inner(ct, r)`` with
    # ``r == 0`` there would be NaN).
    if mm and backend not in CSRMM_IMPLEMENTATIONS:
        pytest.skip(f'{backend} has no csrmm kernel')
    shape = (7, 6)
    case = inactive_nonfinite_case(
        6, *shape, transpose=transpose, homo=True, kind=kind, bad='nan', n_batch=3 if mm else None,
    )
    ct, expected = homo_ct_case(case, transpose)
    indptr = jnp.asarray(case.indptr)
    op = binary_csrmm if mm else binary_csrmv
    config = prepared_config(case.indices, case.indptr, shape)

    def f(d):
        return op(d, jnp.asarray(case.indices), indptr, case.events, shape=shape,
                  transpose=transpose, backend=backend, weighted=case.weighted,
                  config=config)

    _, vjp = jax.vjp(f, jnp.ones(1, jnp.float32))
    (ct_w,) = vjp(ct)
    assert np.isfinite(expected)
    np.testing.assert_allclose(np.asarray(ct_w), [expected], rtol=1e-5)


@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_binary_csrmm_weight_cotangent_isolated_per_event_column(backend, transpose):
    # NaN cotangent only in the batch columns where an entry's event is
    # inactive: the entry's weight cotangent stays the finite active-column sum.
    shape = (7, 6)
    case = inactive_nonfinite_case(
        7, *shape, transpose=transpose, homo=False, kind='weighted', bad='nan', n_batch=4,
    )
    data = np.where(np.isfinite(case.data), case.data, 1.0).astype(np.float32)
    rows = np.repeat(np.arange(shape[0]), np.diff(case.indptr))
    out_idx = case.indices if transpose else rows
    ev = np.asarray(case.events, np.float64)
    # NaN at (out, b) only where every entry writing ``out`` is inactive in b.
    n_out = case.reference.shape[0]
    touched_active = np.zeros((n_out, 4), bool)
    np.logical_or.at(touched_active, out_idx, case.active)
    ct = np.where(touched_active, 1.5, np.nan).astype(np.float32)
    assert np.isnan(ct).any()
    with np.errstate(invalid='ignore'):
        expected = np.where(case.active, ev[case.edge_unit] * ct[out_idx], 0.0).sum(1)
    indptr = jnp.asarray(case.indptr)

    def f(d):
        return binary_csrmm(d, jnp.asarray(case.indices), indptr, case.events, shape=shape,
                            transpose=transpose, backend=backend, weighted=True, config=prepared_config(case.indices, case.indptr, shape))

    _, vjp = jax.vjp(f, jnp.asarray(data))
    (ct_w,) = vjp(jnp.asarray(ct))
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(np.asarray(ct_w), expected, rtol=1e-5)


@pytest.mark.parametrize('undefined', ['B', 'data'])
def test_csrmm_transpose_rule_symbolic_zero_cotangent(undefined):
    # A symbolic-zero cotangent must map to symbolic zeros, not be fed to a kernel.
    from jax.interpreters import ad
    data = jnp.asarray([1.0, 2.0, 3.0], jnp.float32)
    indices = jnp.asarray([0, 2, 1], jnp.int32)
    indptr = jnp.asarray([0, 2, 3], jnp.int32)
    B = jnp.ones((3, 4), jnp.bool_)
    ct = [ad.Zero(jax.core.ShapedArray((2, 4), jnp.float32))]
    args = [data, indices, indptr, B, jnp.empty(0, jnp.uint32)]
    position = 3 if undefined == 'B' else 0
    args[position] = ad.UndefinedPrimal(jax.core.ShapedArray(args[position].shape, args[position].dtype))
    out = binary_mod._csrmm_transpose_rule(ct, *args, shape=(2, 3), transpose=False, backend=None)
    assert len(out) == 5
    assert out[-1] is None
    assert type(out[position]) is ad.Zero
    assert out[position].aval.shape == args[position].aval.shape


@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('m,k,nse', [(2, 3, 8), (7, 5, 105), (6, 5, 20)])
def test_repeated_entries_add(backend, homo, transpose, weighted, m, k, nse):
    data, indices, indptr, dense = _repeated_entry_csr(m, k, nse, homo)
    op = dense.T if transpose else dense
    rng = np.random.default_rng(1)
    v = (rng.random(op.shape[1]) < 0.6) * rng.standard_normal(op.shape[1])
    B = (rng.random((op.shape[1], 4)) < 0.6) * rng.standard_normal((op.shape[1], 4))
    for fn, x, impls in ((binary_csrmv, v, CSRMV_IMPLEMENTATIONS), (binary_csrmm, B, CSRMM_IMPLEMENTATIONS)):
        if backend not in impls:
            continue
        out = fn(data, indices, indptr, jnp.asarray(x, jnp.float32), shape=(m, k),
                 transpose=transpose, backend=backend, weighted=weighted,
                     config=prepared_config(indices, indptr, (m, k)),
                 )
        events = x if weighted else (x != 0)
        assert np.allclose(np.asarray(out), op @ events, rtol=1e-4, atol=1e-4), (fn.__name__, backend)


@pytest.mark.parametrize('generator,jax_kernel,vector_shape', [
    (_binary_csrmv_cusparse_kernel, '_binary_csrmv_jax_kernel', (2,)),
    (_binary_csrmm_cusparse_kernel, '_binary_csrmm_jax_kernel', (2, 4)),
])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('nse,uses_cusparse', [(6, True), (7, False)])
def test_cusparse_kernel_skips_cusparse_beyond_matrix_size(
    monkeypatch, generator, jax_kernel, vector_shape, homo, nse, uses_cusparse,
):
    # Static routing, checked without a GPU: a (2, 3) matrix storing at most 6
    # entries goes to cuSPARSE; storing more takes the exact pure-JAX kernel.
    sentinel = object()
    monkeypatch.setattr(binary_mod, jax_kernel, lambda *a, **kw: sentinel)
    kernel = generator(
        jax.ShapeDtypeStruct((1 if homo else nse,), jnp.float32),
        jax.ShapeDtypeStruct(vector_shape, jnp.float32),
        (2, 3),
        True,
        indices_info=jax.ShapeDtypeStruct((nse,), jnp.int32),
        outs=[jax.ShapeDtypeStruct((3,) + vector_shape[1:], jnp.float32)],
    )
    assert (kernel is not sentinel) == uses_cusparse


# Packaged hybrid CUDA ABI
# ------------------------

def test_binary_hybrid_cuda_exports_use_csr_abi_names():
    csr_dir = Path(__file__).parents[1]
    value_types = ('f32', 'f64', 'f16', 'bf16')
    event_types = ('bool', 'float')
    expected_symbols = {
        'hybrid_mv/hybrid_spmv_binary.cu': {
            f'binary_csrmv_wat_hybrid_{mode}_{value}_{event}'
            for mode in ('hetero', 'homo') for value in value_types for event in event_types
        },
        'hybrid_mv/hybrid_spmv_weighted.cu': {
            f'binary_csrmv_wat_hybrid_{mode}_{value}_wv'
            for mode in ('hetero', 'homo') for value in value_types
        },
        'hybrid_mv/hybrid_spmv_col_major_binary.cu': {
            f'hybrid_spmv_col_major_binary_hetero_{value}_{event}'
            for value in value_types for event in event_types
        },
        'hybrid_mv/hybrid_spmv_col_major_weighted.cu': {
            f'hybrid_spmv_col_major_weighted_hetero_{value}' for value in value_types
        },
        'hybrid_mm/hybrid_batchedv3_spmm_binary.cu': {
            f'batched_spmm_v3{mode}_{value}_{event}'
            for mode in ('', '_homo') for value in value_types for event in event_types
        },
        'hybrid_mm/hybrid_batchedv3_spmm_weighted.cu': {
            f'batched_spmm_v3_weighted{mode}_{value}'
            for mode in ('', '_homo') for value in value_types
        },
        'hybrid_mm/hybrid_batchedv3_spmm_col_major_binary.cu': {
            f'batched_spmm_v3_col_major_binary_hetero_{value}_{event}'
            for value in value_types for event in event_types
        },
        'hybrid_mm/hybrid_batchedv3_spmm_col_major_weighted.cu': {
            f'batched_spmm_v3_col_major_weighted_hetero_{value}' for value in value_types
        },
    }
    for filename, symbols in expected_symbols.items():
        text = (csr_dir / filename).read_text()
        exported = set(re.findall(r'^// @BE ([A-Za-z0-9_]+)$', text, re.MULTILINE))
        assert exported == symbols, filename


# Prepared hybrid matrix adapter
# ------------------------------

def _mm_adapter_prepared(*, weighted=False, homo=False):
    indices = jnp.array([0, 2, 1], jnp.int32)
    indptr = jnp.array([0, 2, 3], jnp.int32)
    device = CSRDeviceInfo('gpu', 'TEST_GPU', l2_bytes=4096, sms=8)
    config = CSRConfig(device_info=device).prepare_structure(indices, indptr, shape=(2, 3))
    spec, mask = config.kernel_args()
    weights = jnp.ones((1 if homo else 3,), jnp.float32)
    events = jnp.array([[1, 0], [-2, 3]], jnp.float32)
    kwargs = dict(
        config_spec=spec, call_plan=config.mm_plan(2, weights.dtype, weighted=weighted),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(events.shape, events.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        out_info=jax.ShapeDtypeStruct((3, 2), weights.dtype), weighted=weighted,
    )
    return kwargs, (weights, indices, indptr, events, mask)


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('permuted,homo', [(False, False), (False, True), (True, False), (True, True)])
def test_adapter_preserves_layout_dtype_and_plan(monkeypatch, weighted, permuted, homo):
    params, args = _mm_adapter_prepared(weighted=weighted, homo=homo)
    modules, calls = [], []

    def load_module(source, name, **kwargs):
        modules.append((source, name, kwargs))

    def ffi_call(target, infos, **kwargs):
        def call(*operands, **attrs):
            calls.append((target, infos, operands, attrs))
            return (jnp.arange(6, dtype=jnp.float32).reshape(2, 3),)
        return call

    monkeypatch.setattr(binary_mod, 'load_cuda_file', load_module)
    monkeypatch.setattr(jax.ffi, 'ffi_call', ffi_call)
    kernel = binary_mod._make_hybrid_mm_kernel(**params, permuted=permuted)
    if permuted:
        args = (*args[:3], jnp.array([2, 0, 1], jnp.int32), *args[3:])
    actual, = kernel(*args)
    np.testing.assert_array_equal(actual, np.arange(6, dtype=np.float32).reshape(2, 3).T)
    target, infos, operands, attrs = calls[0]
    np.testing.assert_array_equal(operands[-1], args[-2].T)
    assert len(modules) == 1
    source, module, load_kwargs = modules[0]
    mode = 'weighted' if weighted else 'binary'
    layout = 'col_major' if permuted and not homo else 'row'
    assert source.name == f'hybrid_batchedv3_spmm_{"col_major_" if layout == "col_major" else ""}{mode}.cu'
    assert re.fullmatch(rf'csr_hybrid_mm_{layout}_{mode}_[0-9a-f]{{12}}', module)
    assert load_kwargs == dict(extra_include_paths=[str(source.parent)], allow_cuda_graph=True)
    assert target.startswith(module + '.')
    assert len(operands) == (6 if permuted and not homo else 5)
    assert ('_homo' in target) == homo
    assert infos[1].shape == (params['call_plan'].workspace_words,)
    assert attrs['task_capacity'] == params['call_plan'].task_capacity
    assert attrs['expected_sms'] == 8
    assert attrs['expected_l2_bytes'] == 4096


def test_adapter_requires_complete_plan_and_matching_shape(monkeypatch):
    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda *args, **kwargs: pytest.fail('must reject before compilation'))
    params, _ = _mm_adapter_prepared()
    with pytest.raises(ValueError, match='complete prepared CSRConfig'):
        binary_mod._make_hybrid_mm_kernel(**dict(params, call_plan=None))
    with pytest.raises(ValueError, match='operand structure and batch'):
        binary_mod._make_hybrid_mm_kernel(**dict(params, call_plan=replace(params['call_plan'], batch=7)))
    with pytest.raises(ValueError, match='value dtype'):
        binary_mod._make_hybrid_mm_kernel(**dict(params, call_plan=replace(params['call_plan'], dtype=np.dtype(np.float64))))
    with pytest.raises(ValueError, match='prepared config and invocation'):
        binary_mod._make_hybrid_mm_kernel(**dict(params, call_plan=replace(params['call_plan'], workspace_words=1)))


def test_empty_adapter_does_not_require_preparation_or_cuda(monkeypatch):
    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda *args, **kwargs: pytest.fail('empty operation must not compile'))
    params, args = _mm_adapter_prepared()
    kernel = binary_mod._make_hybrid_mm_kernel(**dict(
        params, config_spec=None, call_plan=None,
        indices_info=jax.ShapeDtypeStruct((0,), jnp.int32),
    ))
    np.testing.assert_array_equal(kernel(*args)[0], np.zeros((3, 2)))


@pytest.mark.parametrize('mask', [np.zeros(0, np.uint32), np.zeros(1, np.int32)])
def test_adapter_rejects_mismatched_mask_before_ffi(monkeypatch, mask):
    params, args = _mm_adapter_prepared()
    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda *args, **kwargs: None)

    def ffi_call(*args, **kwargs):
        return lambda *args, **kwargs: pytest.fail('invalid prepared mask must not reach FFI')

    monkeypatch.setattr(jax.ffi, 'ffi_call', ffi_call)
    kernel = binary_mod._make_hybrid_mm_kernel(**params)
    with pytest.raises(ValueError, match='long-row mask'):
        kernel(*args[:-1], jnp.asarray(mask))


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('permuted,homo', [(False, False), (False, True), (True, False)])
def test_prepared_adapter_cuda_repeated_calls(weighted, permuted, homo):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    lengths = np.array([0, 3, 65, 513], np.int32)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    indices_np = np.arange(int(offsets[-1]), dtype=np.int32) % 7
    indices, indptr = jnp.asarray(indices_np), jnp.asarray(offsets)
    device_info = config_mod.get_device_info(indices, indptr)
    config = CSRConfig(device_info=device_info).prepare_structure(indices, indptr, shape=(4, 7))
    spec, mask = config.kernel_args()
    weights = jnp.ones((1 if homo else indices.size,), jnp.float32) / 8
    if not homo:
        weights *= jnp.arange(indices.size) % 3 - 1
    events = jnp.array([[0, 1, -2], [1, 0, 2], [-2, 1, 0], [1, 0, -1]], jnp.float32)
    perm = jnp.arange(indices.size - 1, -1, -1, dtype=jnp.int32)
    params = dict(
        config_spec=spec, call_plan=config.mm_plan(3, jnp.float32, weighted),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(events.shape, events.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        out_info=jax.ShapeDtypeStruct((7, 3), jnp.float32),
        weighted=weighted, permuted=permuted,
    )
    kernel = jax.jit(binary_mod._make_hybrid_mm_kernel(**params))
    edge_rows = np.repeat(np.arange(4), lengths)
    for active in (True, False, True):
        current = events if active else jnp.zeros_like(events)
        operands = (weights, indices, indptr) + ((perm,) if permuted else ()) + (current, mask)
        actual, = kernel(*operands)
        factors = np.asarray(current)[edge_rows]
        if not weighted:
            factors = factors != 0
        values = np.full(indices.size, float(weights[0])) if homo else np.asarray(weights)[np.asarray(perm) if permuted else np.arange(indices.size)]
        expected = np.zeros((7, 3), np.float32)
        np.add.at(expected, indices_np, factors * values[:, None])
        np.testing.assert_array_equal(actual, expected)


def test_cuda_rejects_plan_prepared_for_other_device():
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    params, args = _mm_adapter_prepared()
    actual_device = config_mod.get_device_info(*args)
    wrong_device = replace(actual_device, sms=actual_device.sms + 1)
    params['config_spec'] = replace(params['config_spec'], device_info=wrong_device)
    kernel = jax.jit(binary_mod._make_hybrid_mm_kernel(**params))
    with pytest.raises(Exception, match='prepared CSRConfig device does not match'):
        jax.block_until_ready(kernel(*args))


# Native vector scratch reuse
# ---------------------------

def _mv_native_ordinary_hybrid_ffi(*, weighted, homogeneous, cols, task_capacity):
    root = Path(__file__).parents[1] / 'hybrid_mv'
    mode = 'weighted' if weighted else 'binary'
    source = root / f'hybrid_spmv_{mode}.cu'
    dependencies = (source, *sorted(root.glob('*.cuh')))
    digest = hashlib.sha256(b''.join(path.read_bytes() for path in dependencies)).hexdigest()[:12]
    module = f'csr_spmv_status_{mode}_{digest}'
    load_cuda_file(
        source, name=module, extra_include_paths=[str(root)],
        extra_cuda_cflags=CSRConfig().compile_flags(), allow_cuda_graph=False,
    )
    weight_kind = 'homo' if homogeneous else 'hetero'
    event_suffix = 'wv' if weighted else 'float'
    target = f'{module}.binary_csrmv_wat_hybrid_{weight_kind}_f32_{event_suffix}'
    infos = (jax.ShapeDtypeStruct((cols,), jnp.float32),
             jax.ShapeDtypeStruct((task_capacity,), jnp.int32),
             jax.ShapeDtypeStruct((task_capacity,), jnp.int32),
             jax.ShapeDtypeStruct((2,), jnp.int32))
    return jax.ffi.ffi_call(target, infos, input_output_aliases={4: 1, 5: 2, 6: 3})


@requires_gpu_backend
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('homogeneous', [False, True], ids=['hetero', 'homo'])
@pytest.mark.parametrize('rows,cols', [(0, 3), (2, 3), (2, 0), (0, 0)],
                         ids=['zero_rows', 'zero_edges', 'zero_output', 'all_empty'])
def test_ordinary_empty_structure_resets_dirty_status(weighted, homogeneous, rows, cols):
    ffi = _mv_native_ordinary_hybrid_ffi(weighted=weighted, homogeneous=homogeneous,
                              cols=cols, task_capacity=0)
    weights = jnp.ones((1 if homogeneous else 0,), jnp.float32)
    indices = jnp.empty((0,), jnp.int32)
    indptr = jnp.zeros((rows + 1,), jnp.int32)
    events = jnp.ones((rows,), jnp.float32)
    scratch = (jnp.empty((0,), jnp.int32), jnp.empty((0,), jnp.int32),
               jnp.array([7, 9], jnp.int32))
    output, _, _, status = ffi(weights, indices, indptr, events, *scratch, task_capacity=0)
    np.testing.assert_array_equal(output, np.zeros(cols, np.float32))
    np.testing.assert_array_equal(status, [0, 0])


@requires_gpu_backend
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('homogeneous', [False, True], ids=['hetero', 'homo'])
def test_ordinary_reused_scratch_tracks_active_silent_and_reactivated_rows(weighted, homogeneous):
    count = 129  # One long-row task at the default scheduling threshold.
    ffi = _mv_native_ordinary_hybrid_ffi(weighted=weighted, homogeneous=homogeneous,
                              cols=count, task_capacity=1)
    weights = jnp.ones((1,), jnp.float32) if homogeneous else jnp.arange(1, count + 1, dtype=jnp.float32)
    indices = jnp.arange(count, dtype=jnp.int32)
    indptr = jnp.array([0, count], jnp.int32)
    scratch = (jnp.full((1,), -1, jnp.int32), jnp.full((1,), -1, jnp.int32),
               jnp.array([7, 9], jnp.int32))
    for activity in (2.0, 0.0, 0.0, 3.0):
        output, *scratch = ffi(weights, indices, indptr, jnp.array([activity], jnp.float32),
                               *scratch, task_capacity=1)
        factor = activity if weighted else float(activity != 0)
        expected = np.broadcast_to(np.asarray(weights), (count,)) * factor
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(scratch[-1], [int(activity != 0), 0])


# Native matrix sources, ABI and repeated calls
# ---------------------------------------------

_mm_native_BINARY = Path(__file__).resolve().parents[1]


_mm_native_MM = _mm_native_BINARY / 'hybrid_mm'


_mm_native_INCLUDE = _mm_native_BINARY.parents[1] / 'include'


def test_hybrid_mm_local_includes_resolve():
    """Moving col_major files must not leave dependencies at their old paths."""
    missing = []
    for source in sorted(_mm_native_MM.rglob('*')):
        if source.suffix not in ('.cu', '.cuh'):
            continue
        for name in re.findall(r'^\s*#include\s+"([^"]+)"', source.read_text(), re.MULTILINE):
            if not any((root / name).is_file() for root in (source.parent, _mm_native_INCLUDE)):
                missing.append(f'{source.relative_to(_mm_native_BINARY)}: {name}')
    assert not missing, '\n'.join(missing)


@pytest.mark.parametrize('permuted,homo', [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('dtype,suffix', [
    (jnp.float16, 'f16'), (jnp.bfloat16, 'bf16'),
    (jnp.float32, 'f32'), (jnp.float64, 'f64'),
])
def test_hybrid_mm_ffi_short_long_rows_and_repeated_calls(permuted, homo, weighted, dtype, suffix):
    """Compile every source and check both scatter phases with a nonidentity perm.

    Three batches in groups of two exercise scratch reuse between groups;
    alternating active and silent calls checks the output overwrite contract.
    This direct FFI test does not depend on the future library plan adapter.
    """
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    with jax_x64_enabled():
        lengths = np.array([0, 3, 65, 513, 1025], dtype=np.int64)
        rows, cols, batch, chunk = len(lengths), 13, 3, 2
        offsets = np.r_[0, np.cumsum(lengths)]
        nnz = int(offsets[-1])
        indices = (np.arange(nnz, dtype=np.int32) * 7) % cols
        perm = np.arange(nnz - 1, -1, -1, dtype=np.int64)
        edge_rows = np.repeat(np.arange(rows), lengths)
        # group=4 -> threshold=64; the three long rows use 1+2+3 tasks.
        capacity = chunk * 6
        workspace_bytes = batch * 8 + 2 * 8 + capacity * 12
        if weighted:
            workspace_bytes += (capacity * jnp.dtype(dtype).itemsize + 3) // 4 * 4
        # Half values accumulate into a float32 output; short-row lists hold one group.
        accumulator = jnp.float64 if dtype == jnp.float64 else jnp.float32
        outputs = (
            jax.ShapeDtypeStruct((batch, cols), accumulator),
            jax.ShapeDtypeStruct(((workspace_bytes + 3) // 4,), jnp.uint32),
            jax.ShapeDtypeStruct((chunk * rows,), jnp.int32),
        )
        mode = 'weighted' if weighted else 'binary'
        layout = 'col_major' if permuted else 'row'
        source = _mm_native_MM / f'hybrid_batchedv3_spmm_{"col_major_" if permuted else ""}{mode}.cu'
        dependencies = (source, *sorted(_mm_native_MM.rglob('*.cuh')))
        digest = hashlib.sha256(b''.join(path.read_bytes() for path in dependencies)).hexdigest()[:12]
        name = f'csr_hybrid_mm_source_test_{layout}_{mode}_{digest}'
        load_cuda_file(source, name=name, extra_include_paths=[str(_mm_native_MM)], allow_cuda_graph=False)
        if permuted:
            symbol = f'batched_spmm_v3_col_major_{"weighted" if weighted else "binary"}_hetero_{suffix}'
        else:
            symbol = 'batched_spmm_v3' + ('_weighted' if weighted else '')
            symbol += ('_homo' if homo else '') + '_' + suffix
        if not weighted:
            symbol += '_float'
        structure = (jnp.asarray(indices), jnp.asarray(offsets, jnp.int64))
        if permuted:
            structure += (jnp.asarray(perm),)
        structure += (jnp.asarray([0b11100], jnp.uint32),)
        ffi = jax.ffi.ffi_call(name + '.' + symbol, outputs)

        def call(values, events):
            return ffi(values, *structure, events, group=4, long_threshold=64,
                       chunk=chunk, long_tasks=6, task_capacity=capacity,
                       expected_sms=0, expected_l2_bytes=0)[0]

        call = jax.jit(call)
        events = np.array([[0, -2, 0.5, 1, 0], [1, 0, 1, -2, 0.5],
                           [0, 0.5, 1, 0, -2]], dtype=np.float32)
        if homo:
            # Equal-magnitude terms keep every intermediate BF16 sum exact,
            # so this checks routing rather than atomic reduction roundoff.
            events = np.sign(events)
        for scale, active in [(1, True), (1, False), (-2, True)]:
            weights = np.array([0.03125], np.float32) if homo else (np.arange(nnz) % 7 - 3) / 32
            weights = jnp.asarray(weights * scale, dtype)
            current = jnp.asarray(events if active else np.zeros_like(events), dtype if weighted else jnp.float32)
            terms = (np.full(nnz, float(weights[0])) if homo else
                     np.asarray(weights, np.float64)[perm if permuted else np.arange(nnz)])
            expected = np.zeros((batch, cols), np.float64)
            for b in range(batch):
                event = np.asarray(current[b], np.float64)[edge_rows]
                selected = event != 0
                values = terms[selected] * event[selected] if weighted else terms[selected]
                np.add.at(expected[b], indices[selected], values)
            result = call(weights, current)
            assert result.dtype == accumulator
            result = result.astype(dtype)
            # The reference sums in float64, then observes the output dtype.
            expected = np.asarray(jnp.asarray(expected, dtype), np.float64)
            np.testing.assert_array_equal(np.asarray(result, np.float64), expected)


# Prepared configurations through JIT, vmap and autodiff
# ------------------------------------------------------

@pytest.fixture(autouse=False)
def _prepared_enable_x64():
    with jax_x64_enabled():
        yield


def _prepared_backend():
    return 'cuda_raw' if jax.default_backend() == 'gpu' else 'jax_raw'


def _prepared_case(dtype=jnp.float32, *, homo=False):
    lengths = np.array([0, 3, 9, 513], np.int32)
    ptr = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    idx = (np.arange(int(ptr[-1]), dtype=np.int32) * 7) % 9
    indices, indptr = jnp.asarray(idx), jnp.asarray(ptr)
    device = get_device_info(indices, indptr)
    # CPU tests inspect the pure launch plan without lowering a CUDA kernel.
    device = device or CSRDeviceInfo('gpu', 'PLAN_TEST', l2_bytes=4096, sms=8)
    config = CSRConfig(tpr_threshold=16, task_nnz=64, device_info=device).prepare_structure(
        indices, indptr, shape=(4, 9))
    values = np.ones(1) / 16 if homo else (np.arange(idx.size) % 3 - 1) / 16
    events = jnp.asarray([[0, 1, -2], [1, 0, 2], [-2, 1, 0], [1, 0, -1]], dtype)
    return SimpleNamespace(
        weights=jnp.asarray(values, dtype), indices=indices, indptr=indptr,
        perm=jnp.arange(idx.size - 1, -1, -1, dtype=jnp.int32),
        events=events, config=config, shape=(4, 9),
    )


def _prepared_call(case, weights, events, *, indexed=False, weighted=False, config=None,
          backend=None, indptr=None, public=False):
    mm = events.ndim == 2
    if public:
        function = (brainevent.binary_csrmm_indexed if mm else brainevent.binary_csrmv_indexed) if indexed else (
            brainevent.binary_csrmm if mm else brainevent.binary_csrmv)
    else:
        function = (binary_csrmm_indexed_p_call if mm else binary_csrmv_indexed_p_call) if indexed else (
            binary_csrmm_p_call if mm else binary_csrmv_p_call)
    operands = (weights, case.indices, case.indptr if indptr is None else indptr)
    if indexed:
        operands += (case.perm,)
    result = function(*operands, events, shape=case.shape, transpose=True,
                      weighted=weighted, backend=backend or _prepared_backend(), config=config)
    return result if public else result[0]


def _prepared_reference(case, weights, events, *, indexed=False, weighted=False, indptr=None):
    ptr = np.asarray(case.indptr if indptr is None else indptr)
    rows = np.repeat(np.arange(case.shape[0]), np.diff(ptr))
    values = np.asarray(weights, np.float64)
    if values.size == 1:
        values = np.full(case.indices.size, values[0])
    elif indexed:
        values = values[np.asarray(case.perm)]
    factors = np.asarray(events, np.float64)[rows]
    if not weighted:
        factors = factors != 0
    shape = (case.shape[1],) + np.shape(events)[1:]
    output = np.zeros(shape, np.float64)
    np.add.at(output, np.asarray(case.indices), factors * values.reshape((-1,) + (1,) * (factors.ndim - 1)))
    return np.asarray(jnp.asarray(output, weights.dtype), np.float64)


def _prepared_equations(value):
    if hasattr(value, 'eqns'):
        for equation in value.eqns:
            yield equation
            yield from _prepared_equations(equation.params)
    elif hasattr(value, 'jaxpr'):
        yield from _prepared_equations(value.jaxpr)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _prepared_equations(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _prepared_equations(child)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('indexed,homo', [(False, False), (False, True)])
@pytest.mark.parametrize('weighted', [False, True])
def test_public_mm_prepared_config_types_and_long_rows(dtype, indexed, homo, weighted):
    case = _prepared_case(dtype, homo=homo)
    function = jax.jit(lambda w, x, config: _prepared_call(
        case, w, x, indexed=indexed, weighted=weighted, config=config, public=True))
    for events in (case.events, jnp.zeros_like(case.events)):
        actual = function(case.weights, events, case.config)
        expected = _prepared_reference(case, case.weights, events, indexed=indexed, weighted=weighted)
        np.testing.assert_array_equal(np.asarray(actual, np.float64), expected)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('indexed', [False])
@pytest.mark.parametrize('axis', [0, 1])
def test_mv_vmap_reenters_mm_with_transformed_call_plan(indexed, axis):
    case = _prepared_case()
    data = case.events.T if axis == 0 else case.events
    function = jax.vmap(lambda x: _prepared_call(case, case.weights, x, indexed=indexed,
                                      weighted=True, config=case.config), in_axes=axis)
    graph_function = jax.vmap(lambda x: _prepared_call(
        case, case.weights, x, indexed=indexed, weighted=True,
        config=case.config, backend='cuda_raw'), in_axes=axis)
    graph = jax.make_jaxpr(graph_function)(data)
    plans = [eqn.params['call_plan'] for eqn in _prepared_equations(graph)
             if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert len(plans) == 1 and plans[0].batch == case.events.shape[1]
    assert plans[0] == case.config.mm_plan(3, case.weights.dtype, True)
    actual = jax.jit(function)(data)
    np.testing.assert_array_equal(actual, _prepared_reference(case, case.weights, case.events,
                                                    indexed=indexed, weighted=True).T)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('indexed', [False])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_mm_vmap_rederives_plan_for_folded_columns(indexed, axis):
    case = _prepared_case()
    original = jnp.stack([case.events, case.events * 2])
    data = jnp.moveaxis(original, 0, axis)
    function = jax.vmap(lambda x: _prepared_call(case, case.weights, x, indexed=indexed,
                                      weighted=True, config=case.config), in_axes=axis)
    graph_function = jax.vmap(lambda x: _prepared_call(
        case, case.weights, x, indexed=indexed, weighted=True,
        config=case.config, backend='cuda_raw'), in_axes=axis)
    graph = jax.make_jaxpr(graph_function)(data)
    plans = [eqn.params['call_plan'] for eqn in _prepared_equations(graph)
             if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert len(plans) == 1 and plans[0] == case.config.mm_plan(6, jnp.float32, True)
    assert case.config.mm_plan(3, jnp.float32, True).batch == 3
    expected = np.stack([_prepared_reference(case, case.weights, x, indexed=indexed, weighted=True) for x in original])
    np.testing.assert_array_equal(jax.jit(function)(data), expected)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('indexed', [False])
def test_weights_only_vmap_reuses_prepared_structure_in_scan(indexed):
    case = _prepared_case()
    weights = jnp.stack([case.weights, case.weights * 2])
    function = jax.vmap(lambda w: _prepared_call(case, w, case.events, indexed=indexed,
                                      weighted=True, config=case.config))
    graph_function = jax.vmap(lambda w: _prepared_call(
        case, w, case.events, indexed=indexed, weighted=True,
        config=case.config, backend='cuda_raw'))
    graph = jax.make_jaxpr(graph_function)(weights)
    assert any(eqn.primitive.name == 'scan' for eqn in _prepared_equations(graph))
    plans = [eqn.params['call_plan'] for eqn in _prepared_equations(graph)
             if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert plans and all(plan == case.config.mm_plan(3, jnp.float32, True) for plan in plans)
    expected = np.stack([_prepared_reference(case, w, case.events, indexed=indexed, weighted=True) for w in weights])
    np.testing.assert_array_equal(jax.jit(function)(weights), expected)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('indexed,homo', [(False, False), (False, True)])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('mm', [False, True])
def test_prepared_config_jvp_and_vjp(indexed, homo, weighted, mm):
    case = _prepared_case(homo=homo)
    events = case.events if mm else case.events[:, 0]
    function = lambda w, x: _prepared_call(case, w, x, indexed=indexed, weighted=weighted, config=case.config)
    dweights, devents = jnp.ones_like(case.weights) / 16, jnp.ones_like(events) / 4
    actual, tangent = jax.jit(lambda w, x: jax.jvp(function, (w, x), (dweights, devents)))(case.weights, events)
    expected = _prepared_reference(case, case.weights, events, indexed=indexed, weighted=weighted)
    expected_tangent = (_prepared_reference(case, dweights, events, indexed=indexed, weighted=weighted)
                        + _prepared_reference(case, case.weights, devents, indexed=indexed, weighted=True))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-6, atol=2e-6)

    cotangent = jnp.asarray(np.resize([-1., .5, 2.], actual.shape), jnp.float32)
    def gradients(w, x):
        _, pullback = jax.vjp(function, w, x)
        return pullback(cotangent)
    grad_weights, grad_events = jax.jit(gradients)(case.weights, events)
    rows = np.repeat(np.arange(4), np.diff(np.asarray(case.indptr)))
    factors = np.asarray(events)[rows]
    if not weighted:
        # Binary events: the weight gradient is the exact mask derivative.
        factors = (factors != 0).astype(np.float32)
    slot_grads = factors * np.asarray(cotangent)[np.asarray(case.indices)]
    if mm:
        slot_grads = slot_grads.sum(axis=1)
    expected_weights = np.zeros(case.weights.shape, np.float32)
    if homo:
        expected_weights[0] = slot_grads.sum()
    else:
        np.add.at(expected_weights, np.asarray(case.perm) if indexed else np.arange(rows.size), slot_grads)
    values = np.full(rows.size, float(case.weights[0])) if homo else np.asarray(case.weights)[
        np.asarray(case.perm) if indexed else np.arange(rows.size)]
    dense = np.zeros(case.shape, np.float32)
    np.add.at(dense, (rows, np.asarray(case.indices)), values)
    np.testing.assert_allclose(grad_weights, expected_weights, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(grad_events, dense @ np.asarray(cotangent), rtol=2e-6, atol=2e-6)


@pytest.mark.usefixtures('_prepared_enable_x64')
def test_config_is_safe_as_scan_carry_and_in_independent_jits():
    case = _prepared_case()
    original_spec = case.config.kernel_args()[0]
    def scan(config, events):
        def body(carry, event):
            weights, snapshot = carry
            value = _prepared_call(case, weights, event, weighted=True, config=snapshot)
            return (weights * 2, snapshot), value
        return jax.lax.scan(body, (case.weights, config), events)
    data = jnp.stack([case.events, case.events * 2])
    seven_columns = jnp.concatenate([case.events, case.events, case.events[:, :1]], axis=1)
    with jax.checking_leaks():
        (weights, config), values = jax.jit(scan)(case.config, data)
        fn1 = jax.jit(lambda cfg, x: _prepared_call(case, case.weights, x, weighted=True, config=cfg))
        fn2 = jax.jit(lambda cfg, x: _prepared_call(case, case.weights, x, weighted=True, config=cfg))
        first = fn1(case.config, case.events[:, :2])
        second = fn2(case.config, seven_columns)
        jax.block_until_ready((values, first, second))
    assert case.config.kernel_args()[0] == config.kernel_args()[0] == original_spec
    np.testing.assert_array_equal(weights, case.weights * 4)
    np.testing.assert_array_equal(values, np.stack([
        _prepared_reference(case, case.weights, data[0], weighted=True),
        _prepared_reference(case, case.weights * 2, data[1], weighted=True),
    ]))
    np.testing.assert_array_equal(first, _prepared_reference(case, case.weights, case.events[:, :2], weighted=True))
    np.testing.assert_array_equal(second, _prepared_reference(case, case.weights, seven_columns, weighted=True))


@pytest.mark.usefixtures('_prepared_enable_x64')
def test_raw_calls_do_not_load_or_prepare_implicit_config(monkeypatch):
    case = _prepared_case()
    def forbidden(*args, **kwargs):
        raise AssertionError('raw operators must not load persistence or prepare structure')
    monkeypatch.setattr(initialize, 'load_csr_config', forbidden)
    monkeypatch.setattr(CSRConfig, 'prepare_structure', forbidden)
    # Native MV uses constant defaults. Portable MM needs no prepared plan.
    mv = jax.jit(lambda x: _prepared_call(case, case.weights, x, config=None))(case.events[:, 0])
    mm = jax.jit(lambda x: _prepared_call(case, case.weights, x, config=None, backend='jax_raw'))(case.events)
    np.testing.assert_array_equal(mv, _prepared_reference(case, case.weights, case.events[:, 0]))
    np.testing.assert_array_equal(mm, _prepared_reference(case, case.weights, case.events))


@pytest.mark.usefixtures('_prepared_enable_x64')
def test_auto_hybrid_guard_does_not_block_native_gather():
    case = _prepared_case()
    function = jax.jit(lambda x: binary_csrmm_p_call(
        case.weights, case.indices, case.indptr, x, shape=case.shape,
        transpose=True, backend=None, config=None)[0])
    # On GPU the unprepared native scatter MM falls back to the portable path.
    np.testing.assert_array_equal(function(case.events), _prepared_reference(case, case.weights, case.events))
    x = jnp.asarray(np.resize([-2., 0., 1.], (case.shape[1], 2)), jnp.float32)
    gather = jax.jit(lambda value: binary_csrmm_p_call(
        case.weights, case.indices, case.indptr, value, shape=case.shape,
        transpose=False, backend=_prepared_backend(), weighted=True, config=None)[0])
    rows = np.repeat(np.arange(case.shape[0]), np.diff(np.asarray(case.indptr)))
    dense = np.zeros(case.shape, np.float32)
    np.add.at(dense, (rows, np.asarray(case.indices)), np.asarray(case.weights))
    np.testing.assert_array_equal(gather(x), dense @ np.asarray(x))


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('indexed', [False])
@pytest.mark.parametrize('backend,transpose,dtype,batch', [
    ('jax_raw', True, jnp.float8_e4m3fn, 3),
    ('jax_raw', False, jnp.float8_e4m3fn, 3),
    ('jax_raw', True, jnp.float32, 2**31),
    ('cuda_raw', False, jnp.float32, 2**31),
])
def test_nonhybrid_pcall_does_not_apply_hybrid_plan_limits(monkeypatch, indexed, backend, transpose, dtype, batch):
    """Abstract tracing needs no allocation and must not apply an unrelated plan's limits."""
    case = _prepared_case(dtype)
    def forbidden(*args, **kwargs):
        raise AssertionError('nonhybrid MM must not derive a hybrid call plan')
    monkeypatch.setattr(CSRConfig, 'mm_plan', forbidden)
    function = binary_csrmm_indexed_p_call if indexed else binary_csrmm_p_call
    operands = (case.weights, case.indices, case.indptr) + ((case.perm,) if indexed else ())
    rows = case.shape[0] if transpose else case.shape[1]
    graph = jax.make_jaxpr(lambda x: function(
        *operands, x, shape=case.shape, transpose=transpose, backend=backend,
        config=case.config)[0])(jax.ShapeDtypeStruct((rows, batch), dtype))
    equations = [eqn for eqn in _prepared_equations(graph)
                 if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert len(equations) == 1
    assert equations[0].params['call_plan'] is None
    assert graph.out_avals[0].dtype == jnp.dtype(dtype)
    assert graph.out_avals[0].shape == (case.shape[1] if transpose else case.shape[0], batch)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('indexed', [False])
@pytest.mark.parametrize('transpose', [False, True])
def test_prepared_float8_mm_remains_available_on_jax_backend(indexed, transpose):
    indices, indptr = jnp.array([0, 2], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    weights = jnp.array([1, 2], jnp.float8_e4m3fn)
    perm = jnp.array([1, 0], jnp.int32)
    config = CSRConfig(device_info=CSRDeviceInfo('gpu', 'PLAN_TEST', 4096, 8)).prepare_structure(
        indices, indptr, shape=(2, 3))
    events = jnp.array([[1, 0], [0, 1]] if transpose else [[1, 0], [0, 0], [0, 1]], jnp.float8_e4m3fn)
    function = binary_csrmm_indexed_p_call if indexed else binary_csrmm_p_call
    operands = (weights, indices, indptr) + ((perm,) if indexed else ())
    actual = jax.jit(lambda x: function(
        *operands, x, shape=(2, 3), transpose=transpose, backend='jax_raw', config=config)[0])(events)
    a, b = (2, 1) if indexed else (1, 2)
    expected = [[a, 0], [0, 0], [0, b]] if transpose else [[a, 0], [0, b]]
    assert actual.dtype == jnp.dtype(jnp.float8_e4m3fn)
    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('indexed', [False])
def test_structural_vmap_invalidates_exact_plan(indexed):
    case = _prepared_case()
    other = jnp.array([0, 0, 300, 400, case.indices.size], jnp.int32)
    pointers = jnp.stack([case.indptr, other])
    function = jax.vmap(lambda ptr: _prepared_call(
        case, case.weights, case.events, indexed=indexed, weighted=True,
        config=case.config, indptr=ptr, backend='jax_raw'))
    graph = jax.make_jaxpr(function)(pointers)
    equations = [eqn for eqn in _prepared_equations(graph)
                 if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert equations and all(eqn.params['call_plan'] is None for eqn in equations)
    assert all(eqn.params['config_spec'].structure is None for eqn in equations)
    expected = np.stack([_prepared_reference(case, case.weights, case.events, indexed=indexed,
                                    weighted=True, indptr=ptr) for ptr in pointers])
    np.testing.assert_array_equal(jax.jit(function)(pointers), expected)
    if jax.default_backend() == 'gpu':
        native = jax.vmap(lambda ptr: _prepared_call(
            case, case.weights, case.events, indexed=indexed, config=case.config, indptr=ptr))
        binary = np.stack([_prepared_reference(case, case.weights, case.events, indexed=indexed,
                                               indptr=ptr) for ptr in pointers])
        np.testing.assert_array_equal(jax.jit(native)(pointers), binary)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('indexed', [False])
def test_native_unprepared_mm_and_mv_vmap_use_portable_fallback(indexed, monkeypatch):
    if jax.default_backend() != 'gpu':
        pytest.skip('actual hybrid lowering requires GPU')
    case = _prepared_case()
    expected = _prepared_reference(case, case.weights, case.events, indexed=indexed)
    for config in (None, CSRConfig(device_info=case.config.device_info),
                   replace(case.config, device_info=None)):
        function = lambda x: _prepared_call(case, case.weights, x, indexed=indexed, config=config)
        jax.clear_caches()  # the warning is raised once, while lowering
        monkeypatch.setattr(binary_mod, '_UNPREPARED_MM_WARNED', set())
        with pytest.warns(brainevent.PerformanceWarning, match='no prepared CSRConfig'):
            np.testing.assert_array_equal(jax.jit(function)(case.events), expected)
        batched = jax.jit(jax.vmap(function))(case.events.T)
        np.testing.assert_array_equal(np.asarray(batched).T, expected)


@pytest.mark.usefixtures('_prepared_enable_x64')
def test_direct_native_primitive_rejects_tampered_call_plan():
    if jax.default_backend() != 'gpu':
        pytest.skip('actual hybrid lowering requires GPU')
    case = _prepared_case()
    graph = jax.make_jaxpr(lambda w, x: _prepared_call(case, w, x, config=case.config))(case.weights, case.events)
    eqn = next(e for e in _prepared_equations(graph) if e.primitive is binary_csrmm_p.primitive)
    params = dict(eqn.params)
    params['call_plan'] = replace(params['call_plan'], workspace_words=params['call_plan'].workspace_words + 1)
    mask = case.config.kernel_args()[1]
    function = jax.jit(lambda w, x: binary_csrmm_p(w, case.indices, case.indptr, x, mask, **params)[0])
    with pytest.raises(brainevent.KernelCompilationError, match='prepared config and invocation'):
        function(case.weights, case.events)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('empty', ['nnz', 'batch'])
def test_empty_mm_bypasses_preparation_and_cuda_loader(monkeypatch, empty):
    case = _prepared_case()
    if empty == 'nnz':
        case.indices = jnp.empty((0,), jnp.int32)
        case.indptr = jnp.zeros((case.shape[0] + 1,), jnp.int32)
        case.weights = jnp.ones((1,), jnp.float32)
    else:
        case.events = case.events[:, :0]
    def forbidden(*args, **kwargs):
        raise AssertionError('empty hybrid MM must not load CUDA sources')
    monkeypatch.setattr(binary_mod, 'load_cuda_file', forbidden)
    actual = jax.jit(lambda x: _prepared_call(case, case.weights, x, config=None))(case.events)
    np.testing.assert_array_equal(actual, np.zeros((case.shape[1], case.events.shape[1])))


# Sampled gradient dispatch and backend derivatives
# -------------------------------------------------

_sampled_dispatch = importlib.import_module('brainevent._sddmm.dispatch')


_sampled_float_csr = importlib.import_module('brainevent._csr.float.float')


_sampled_binary_csr = importlib.import_module('brainevent._csr.binary.binary')


_sampled_indexed_csr = importlib.import_module('brainevent._csr.binary.binary_indexed')


_sampled_CASES = ['float', 'binary', 'weighted', 'indexed_binary', 'indexed_weighted']


def _sampled_operands(case, transpose, batch, homo=False, backend='jax_raw'):
    indices = jnp.array([2, 0, 1, 1], jnp.int32)
    indptr = jnp.array([0, 2, 2, 4], jnp.int32)
    rows = jnp.array([0, 0, 2, 2], jnp.int32)
    # Two physical slots share one canonical weight.
    perm = jnp.array([2, 0, 1, 1], jnp.int32)
    weights = jnp.arange(1 if homo else 3 if case.startswith('indexed') else 4,
                          dtype=jnp.float32) + 1
    event_shape = (3 if transpose else 4,) + (() if batch is None else (batch,))
    events = jnp.asarray(np.resize([0., -2., 1., 3.], event_shape))
    if case == 'float':
        op = _sampled_float_csr.csrmv if batch is None else _sampled_float_csr.csrmm
        call = lambda w: op(w, indices, indptr, events, shape=(3, 4),
                            transpose=transpose, backend=backend)
    elif case.startswith('indexed'):
        op = _sampled_indexed_csr.binary_csrmv_indexed if batch is None else _sampled_indexed_csr.binary_csrmm_indexed
        call = lambda w: op(
            w, indices, indptr, perm, events, shape=(3, 4), transpose=transpose,
            weighted=case.endswith('weighted'),
            backend=backend)
    else:
        op = _sampled_binary_csr.binary_csrmv if batch is None else _sampled_binary_csr.binary_csrmm
        config = (prepared_config(indices, indptr, (3, 4))
                  if transpose and batch is not None and backend in (None, 'cuda_raw') else None)
        call = lambda w: op(
            w, indices, indptr, events, shape=(3, 4), transpose=transpose,
            weighted=case == 'weighted',
            backend=backend, config=config)
    return call, weights, events, rows, indices, perm


def _sampled_check_second_order_gradients(call, w):
    loss = lambda w: jnp.sum(call(w) ** 2)
    direction = jnp.arange(w.size, dtype=w.dtype) + .5
    got_fwd = jax.jit(lambda w: jax.jvp(jax.grad(loss), (w,), (direction,))[1])(w)
    got_rev = jax.jit(jax.grad(lambda w: jnp.vdot(jax.grad(loss)(w), direction)))(w)
    # The forward is linear in weights; its explicit basis matrix gives the
    # Hessian 2*J.T@J independently of the SDDMM transpose implementation.
    basis_outputs = jax.vmap(call)(jnp.eye(w.size, dtype=w.dtype)).reshape(w.size, -1)
    expected = 2 * basis_outputs @ (basis_outputs.T @ direction)
    np.testing.assert_allclose(got_fwd, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_rev, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('case,batch', [('binary', None), ('weighted', None), ('binary', 0), ('weighted', 0), ('binary', 1), ('weighted', 1), ('binary', 3), ('weighted', 3)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('nonfinite', [False, True])
def test_csr_consumers_dispatch_slot_gradients(monkeypatch, case, transpose, batch, homo, nonfinite):
    calls = []
    original = _sampled_dispatch.sddmm
    def record(a, b, indices, indptr=None, **kwargs):
        backend = kwargs.get('backend')
        calls.append((a.shape[1] if a.ndim == 2 else 1, indices.ndim,
                      None if indptr is None else indptr.shape,
                      kwargs.get('mode', 'float'), getattr(backend, 'requested', backend)))
        return original(a, b, indices, indptr, **kwargs)
    monkeypatch.setattr(_sampled_dispatch, 'sddmm', record)
    jax.clear_caches()
    call, w, events, rows, cols, perm = _sampled_operands(case, transpose, batch, homo)
    ct_shape = (4 if transpose else 3,) + (() if batch is None else (batch,))
    ct = jnp.asarray(np.resize([2., -.5, 3., 0.], ct_shape))
    if nonfinite and batch != 0:
        if batch is None:
            ct = ct.at[0].set(jnp.inf).at[1].set(jnp.nan)
        else:
            ct = ct.at[0, 0].set(jnp.inf).at[1, 0].set(jnp.nan)
    def pullback(w, ct):
        return jax.vjp(call, w)[1](ct)[0]
    actual = jax.jit(pullback)(w, ct)
    unit, out = (rows, cols) if transpose else (cols, rows)
    x, terms = events[unit], ct[out]
    if case == 'float' or case.endswith('weighted'):
        terms = x * terms
    if case != 'float':
        terms = jnp.where(x != 0, terms, 0.)
    per_slot = terms if batch is None else jnp.sum(terms, axis=1)
    if homo:
        expected = jnp.sum(per_slot).reshape(1)
    elif case.startswith('indexed'):
        expected = jnp.zeros_like(w).at[perm].add(per_slot)
    else:
        expected = per_slot
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    mode = 'float' if case == 'float' else 'weighted' if case.endswith('weighted') else 'binary'
    backend = None if case == 'float' else 'jax_raw'
    assert calls == [(1 if batch is None else batch, 1, (4,), mode, backend)]


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('batch', [None, 1, 3])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('indexed', [False])
def test_event_consumers_preserve_activity_after_cast(transpose, batch, weighted, indexed):
    # Binary activity reads the original events (1e-8 is active). Weighted activity is
    # tested after the cast to the cotangent dtype, as in CUDA and cuSPARSE: 1e-8
    # underflows to zero in float16 and contributes nothing.
    rows = jnp.array([0, 0, 1, 2], jnp.int32)
    cols = jnp.array([2, 0, 1, 1], jnp.int32)
    indptr = jnp.array([0, 2, 3, 4], jnp.int32)
    perm = jnp.array([2, 0, 3, 1], jnp.int32)
    size = 3 if transpose else 4
    events = jnp.array([0., 1e-8, -2., 3.], jnp.float32)[:size]
    if batch is not None:
        events = jnp.broadcast_to(events[:, None], (size, batch))
    ct_shape = (4 if transpose else 3,) + (() if batch is None else (batch,))
    ct = jnp.full(ct_shape, jnp.inf, jnp.float16)
    weights = jnp.ones(4, jnp.float16)
    if indexed:
        op = _sampled_indexed_csr.binary_csrmv_indexed if batch is None else _sampled_indexed_csr.binary_csrmm_indexed
        call = lambda w: op(w, cols, indptr, perm, events, shape=(3, 4), transpose=transpose,
                            weighted=weighted,
                            backend='jax_raw')
    else:
        op = _sampled_binary_csr.binary_csrmv if batch is None else _sampled_binary_csr.binary_csrmm
        call = lambda w: op(w, cols, indptr, events, shape=(3, 4), transpose=transpose,
                            weighted=weighted,
                            backend='jax_raw')
    actual = jax.jit(lambda w, c: jax.vjp(call, w)[1](c)[0])(weights, ct)
    unit, out = (rows, cols) if transpose else (cols, rows)
    x = events[unit]
    terms = x.astype(ct.dtype) * ct[out] if weighted else ct[out]
    active = x.astype(ct.dtype) != 0 if weighted else x != 0
    expected = jnp.where(active, terms, 0)
    if batch is not None:
        expected = jnp.sum(expected, axis=1)
    if indexed:
        expected = jnp.zeros_like(weights).at[perm].add(expected)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual.dtype == ct.dtype


@pytest.mark.parametrize('case,batch', [('binary', None), ('weighted', None), ('binary', 1), ('weighted', 1), ('binary', 3), ('weighted', 3)])
@pytest.mark.parametrize('transpose', [False, True])
def test_migrated_consumers_preserve_second_order_gradients(case, transpose, batch):
    call, w, _, _, _, _ = _sampled_operands(case, transpose, batch)
    _sampled_check_second_order_gradients(call, w)


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('indexed', [False])
def test_shared_mv_gradient_avoids_unit_weight_forward_overflow(transpose, indexed):
    indices = jnp.array([0, 0] if transpose else [0, 1], jnp.int32)
    indptr = jnp.array([0, 1, 2] if transpose else [0, 2], jnp.int32)
    shape = (2, 1) if transpose else (1, 2)
    events = jnp.full((2,), 60000, jnp.float16)
    weights = jnp.array([1e-4], jnp.float16)
    ct = jnp.array([1e-4], jnp.float16)
    if indexed:
        call = lambda w: _sampled_indexed_csr.binary_csrmv_indexed(
            w, indices, indptr, jnp.zeros(2, jnp.int32), events,
            shape=shape, transpose=transpose, weighted=True, backend='jax_raw')
    else:
        call = lambda w: _sampled_binary_csr.binary_csrmv(
            w, indices, indptr, events, shape=shape, transpose=transpose,
            weighted=True, backend='jax_raw')
    actual = jax.jit(lambda w, c: jax.vjp(call, w)[1](c)[0])(weights, ct)
    expected = jnp.sum(events * ct).reshape(1)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('case', ['binary', 'weighted'])
@pytest.mark.parametrize('batch', [None, 1, 3])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('selection', [
    'jax_raw', 'cuda_raw', 'default_jax', 'default_cuda',
    'cusparse', 'default_cusparse', 'global_cusparse',
])
def test_event_weight_gradient_follows_csr_backend(monkeypatch, case, batch, transpose, selection):
    if jax.default_backend() != 'gpu':
        pytest.skip('inspect the selected CSR and sampled CUDA backends on a GPU')
    indexed = case.startswith('indexed')
    module = _sampled_indexed_csr if indexed else _sampled_binary_csr
    name = 'binary_csrmv' if batch is None else 'binary_csrmm'
    primitive = getattr(module, name + ('_indexed_p' if indexed else '_p'))
    if selection.startswith('default_'):
        selected = {'default_cuda': 'cuda_raw', 'default_jax': 'jax_raw',
                    'default_cusparse': 'cusparse'}[selection]
        monkeypatch.setitem(primitive._defaults, 'gpu', selected)
        backend = None
    elif selection == 'global_cusparse':
        from brainevent import config
        selected = 'cusparse'
        monkeypatch.setitem(config._global_backends, 'gpu', selected)
        backend = None
    else:
        selected = backend = selection
    native = selected == 'cuda_raw' and transpose
    recorded = []
    original = _sampled_dispatch.sddmm

    def record(a, b, indices, indptr=None, **kwargs):
        request = kwargs.get('backend')
        recorded.append(getattr(request, 'requested', request))
        return original(a, b, indices, indptr, **kwargs)

    monkeypatch.setattr(_sampled_dispatch, 'sddmm', record)
    if native:
        def fail_coo(*args, **kwargs):
            raise AssertionError('native weight gradients must retain the CSR structure')
        leaf = 'mv.sddmv' if batch in (None, 1) else 'mm.sddmm'
        monkeypatch.setattr(importlib.import_module(f'brainevent._sddmm._{leaf}'), '_csr_to_coo', fail_coo)
    jax.clear_caches()
    call, w, events, rows, cols, perm = _sampled_operands(case, transpose, batch, backend=backend)
    shape = (4 if transpose else 3,) + (() if batch is None else (batch,))
    ct = jnp.asarray(np.resize([2., -.5, 3., 0.], shape), jnp.float32)
    fn = jax.jit(lambda weights, cotangent: jax.vjp(call, weights)[1](cotangent)[0])
    result = fn(w, ct)
    unit, out = (rows, cols) if transpose else (cols, rows)
    terms = events[unit] * ct[out] if case.endswith('weighted') else ct[out]
    expected = jnp.where(events[unit] != 0, terms, 0.)
    if batch is not None:
        expected = jnp.sum(expected, axis=1)
    if indexed:
        expected = jnp.zeros_like(w).at[perm].add(expected)
    np.testing.assert_allclose(result, expected)
    assert recorded == [backend]
    ir = fn.lower(w, ct).as_text()
    target = 'hybrid_sddmv_' if batch in (None, 1) else 'batched_v3_sddmm_'
    assert (target in ir) == native
    assert ('cusparse_sddmm_' in ir) == (selected == 'cusparse')
    if native:
        assert 'stablehlo.is_finite' not in ir


@pytest.mark.parametrize('case', ['binary', 'weighted'])
@pytest.mark.parametrize('batch', [None, 3])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('backend', [None, 'cuda_raw', 'cusparse'])
def test_native_event_consumers_preserve_second_order_gradients(monkeypatch, case, batch, transpose, backend):
    if jax.default_backend() != 'gpu':
        pytest.skip('native sampled derivatives require a GPU')
    module = _sampled_indexed_csr if case.startswith('indexed') else _sampled_binary_csr
    name = 'binary_csrmv' if batch is None else 'binary_csrmm'
    primitive = getattr(module, name + ('_indexed_p' if case.startswith('indexed') else '_p'))
    monkeypatch.setitem(primitive._defaults, 'gpu', 'cuda_raw')
    jax.clear_caches()
    call, w, _, _, _, _ = _sampled_operands(case, transpose, batch, backend=backend)
    if case.startswith('indexed'):
        # The native indexed forward requires an nnz-sized canonical buffer.
        # Keep the repeated perm entries and leave its fourth weight unused.
        w = jnp.arange(1, 5, dtype=w.dtype)
    _sampled_check_second_order_gradients(call, w)


@pytest.mark.parametrize('indexed', [False])
@pytest.mark.parametrize('batch', [None, 3])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('gpu_backend', ['cuda_raw', 'cusparse'])
def test_event_gradient_resolves_backend_on_compilation_platform(
    monkeypatch, indexed, batch, transpose, gpu_backend,
):
    if jax.default_backend() != 'gpu':
        pytest.skip('the default device must differ from the CPU compilation target')
    module = _sampled_indexed_csr if indexed else _sampled_binary_csr
    name = 'binary_csrmv' if batch is None else 'binary_csrmm'
    primitive = getattr(module, name + ('_indexed_p' if indexed else '_p'))
    monkeypatch.setitem(primitive._defaults, 'gpu', gpu_backend)
    monkeypatch.setitem(primitive._defaults, 'cpu', 'jax_raw' if indexed else 'numba')
    jax.clear_caches()
    case = 'indexed_weighted' if indexed else 'weighted'
    call, w, events, rows, cols, perm = _sampled_operands(case, transpose, batch, backend=None)
    shape = (4 if transpose else 3,) + (() if batch is None else (batch,))
    ct = jnp.ones(shape, jnp.float32)
    fn = jax.jit(lambda weights, cotangent: jax.vjp(call, weights)[1](cotangent)[0], backend='cpu')
    actual = fn(w, ct)
    unit = rows if transpose else cols
    expected = events[unit]
    if batch is not None:
        expected = expected.sum(axis=1)
    if indexed:
        expected = jnp.zeros_like(w).at[perm].add(expected)
    np.testing.assert_allclose(actual, expected)
    assert next(iter(actual.devices())).platform == 'cpu'
    ir = fn.lower(w, ct).as_text()
    assert 'hybrid_sddmv_' not in ir and 'batched_v3_sddmm_' not in ir
    assert 'cusparse_sddmm_' not in ir


@pytest.mark.parametrize('batch', [None, 3])
@pytest.mark.parametrize('default,global_backend,requested,native', [
    ('cuda_raw', 'jax_raw', None, False),
    ('jax_raw', 'cuda_raw', None, True),
    ('cuda_raw', 'cuda_raw', 'jax_raw', False),
    ('jax_raw', 'jax_raw', 'cuda_raw', True),
    ('cuda_raw', 'cusparse', None, False),
    ('cusparse', 'jax_raw', None, False),
    ('cusparse', 'cuda_raw', None, True),
    ('cusparse', 'cusparse', 'jax_raw', False),
    ('jax_raw', 'jax_raw', 'cusparse', False),
])
def test_event_gradient_respects_backend_priority(monkeypatch, batch, default, global_backend, requested, native):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA backend precedence requires a GPU')
    from brainevent import config
    primitive = _sampled_binary_csr.binary_csrmv_p if batch is None else _sampled_binary_csr.binary_csrmm_p
    monkeypatch.setitem(primitive._defaults, 'gpu', default)
    monkeypatch.setitem(config._global_backends, 'gpu', global_backend)
    jax.clear_caches()
    call, w, _, _, _, _ = _sampled_operands('binary', True, batch, backend=requested)
    shape = (4,) + (() if batch is None else (batch,))
    ct = jnp.ones(shape, jnp.float32)
    fn = jax.jit(lambda weights, cotangent: jax.vjp(call, weights)[1](cotangent)[0])
    result = fn(w, ct)
    reference, _, _, _, _, _ = _sampled_operands('binary', True, batch)
    expected = jax.vjp(reference, w)[1](ct)[0]
    np.testing.assert_allclose(result, expected)
    ir = fn.lower(w, ct).as_text()
    target = 'hybrid_sddmv_' if batch is None else 'batched_v3_sddmm_'
    assert (target in ir) == native
    selected = global_backend if requested is None else requested
    assert ('cusparse_sddmm_' in ir) == (selected == 'cusparse')


# Half-precision scatter accumulation
# -----------------------------------

# Integer sums above 256 (bf16) or 2048 (fp16) stop growing when every edge
# adds 1 in the storage dtype: x + 1 rounds back to x. A float32 accumulator
# rounded once at the end represents both chosen totals exactly.
_HALF_FAN_IN = {'bfloat16': 1000, 'float16': 3000}


def _half_fan_in_case(dtype_name, row_nnz, homo):
    """Every row connects to the same ``row_nnz`` columns with unit weight."""
    rows = _HALF_FAN_IN[dtype_name]
    dtype = jnp.dtype(dtype_name)
    indices = jnp.asarray(np.tile(np.arange(row_nnz, dtype=np.int32), rows))
    indptr = jnp.asarray(np.arange(rows + 1, dtype=np.int32) * row_nnz)
    weights = jnp.ones((1 if homo else rows * row_nnz,), dtype)
    return weights, indices, indptr, (rows, row_nnz)


@requires_gpu_backend
@pytest.mark.parametrize('matrix', [False, True], ids=['mv', 'mm'])
@pytest.mark.parametrize('row_nnz', [2, 600], ids=['short_rows', 'long_rows'])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('homo', [False, True], ids=['hetero', 'homo'])
@pytest.mark.parametrize('dtype_name', sorted(_HALF_FAN_IN))
def test_native_half_scatter_accumulates_in_float32(dtype_name, homo, weighted, row_nnz, matrix):
    weights, indices, indptr, shape = _half_fan_in_case(dtype_name, row_nnz, homo)
    rows = shape[0]
    events = jnp.ones((rows, 2) if matrix else (rows,), jnp.float32 if weighted else jnp.bool_)
    operation = binary_csrmm if matrix else binary_csrmv
    actual = operation(weights, indices, indptr, events, shape=shape, transpose=True,
                       weighted=weighted, backend='cuda_raw',
                       config=prepared_config(indices, indptr, shape))
    assert actual.dtype == weights.dtype
    expected = np.full(actual.shape, rows, np.float32)
    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)


# Undersized native task queues
# -----------------------------

def _overflow_case(lengths, cols, homo, weighted, batch=None, seed=0):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, np.int64)
    nnz = int(lengths.sum())
    indices = jnp.asarray((np.arange(nnz) * 7) % cols, jnp.int32)
    indptr = jnp.asarray(np.r_[0, np.cumsum(lengths)], jnp.int32)
    weights = jnp.asarray([0.5] if homo else rng.integers(-4, 5, nnz) / 4, jnp.float32)
    shape = (len(lengths),) if batch is None else (len(lengths), batch)
    if weighted:
        events = jnp.asarray(rng.integers(-2, 3, shape), jnp.float32)
    else:
        events = jnp.asarray(rng.random(shape) < 0.8)
    return weights, indices, indptr, events, (len(lengths), cols)


@requires_gpu_backend
@pytest.mark.parametrize('hint', [100, 129], ids=['no_queue', 'short_queue'])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('homo', [False, True], ids=['hetero', 'homo'])
def test_native_mv_understated_row_hint_keeps_every_edge(hint, weighted, homo):
    # Rows of 600 edges need two 512-edge tasks each; the hint sizes the queue
    # for none (<= tpr_threshold) or one task per heavy row.
    weights, indices, indptr, events, shape = _overflow_case([600, 600, 600, 5], 700, homo, weighted)
    events = events.at[:3].set(1 if weighted else True)
    kwargs = dict(shape=shape, transpose=True, weighted=weighted)
    actual = binary_csrmv(weights, indices, indptr, events, backend='cuda_raw', max_row_nnz=hint, **kwargs)
    expected = binary_csrmv(weights, indices, indptr, events, backend='jax_raw', **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@requires_gpu_backend
@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('homo', [False, True], ids=['hetero', 'homo'])
def test_native_mm_stale_config_keeps_every_edge(batch, weighted, homo):
    # The stale plan marks rows 0..3 long with one task each (4 tasks); the
    # current structure has the same dimensions but needs 4 + 1 + 1 + 1 tasks.
    stale = np.r_[[512] * 4, [1] * 1000]
    current = np.r_[[2045, 1, 1, 1], [1] * 1000]
    _, stale_indices, stale_indptr, _, shape = _overflow_case(stale, 2048, homo, weighted, batch)
    weights, indices, indptr, events, _ = _overflow_case(current, 2048, homo, weighted, batch)
    config = prepared_config(stale_indices, stale_indptr, shape)
    assert config.primary.long_tasks == 4
    kwargs = dict(shape=shape, transpose=True, weighted=weighted)
    actual = binary_csrmm(weights, indices, indptr, events, backend='cuda_raw', config=config, **kwargs)
    expected = binary_csrmm(weights, indices, indptr, events, backend='jax_raw', **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
