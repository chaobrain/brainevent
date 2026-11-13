# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_csr/binary/binary_indexed.py``, grouped by behavior."""

import hashlib
import importlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
import brainevent._csr.binary.binary as binary_mod
import brainevent._csr.binary.binary_indexed as binary_indexed_mod
from brainevent import BinaryArray, FixedNumPerPre
from brainevent._csr.test._test_util import (
    EVENT_DTYPE_NAMES,
    EVENT_KINDS,
    NONFINITE_VALUES,
    WEIGHTED_EVENT_DTYPE_NAMES,
    assert_select_reference,
    cuda_kwargs,
    event_dtype_context,
    event_mask,
    event_values,
    heavy_row_csr_numpy,
    homo_ct_case,
    inactive_nonfinite_case,
    int64_structure,
    prepared_config,
    random_csr_numpy,
    recording_ffi_call,
    requires_gpu_backend,
    run_python_isolated,
    shape_of,
    strip_hybrid_suffix,
)
from brainevent._csr.binary.binary import binary_csrmm, binary_csrmm_p_call, binary_csrmv, binary_csrmv_p_call
from brainevent._csr.binary.binary_indexed import (
    _csrmm_idx_batching,
    _idx_batching,
    binary_csrmm_indexed,
    binary_csrmm_indexed_p,
    binary_csrmm_indexed_p_call,
    binary_csrmv_indexed,
    binary_csrmv_indexed_p,
    binary_csrmv_indexed_p_call,
)
from brainevent._csr.config import CSRConfig, CSRDeviceInfo, get_device_info
from brainevent._op import load_cuda_file
from brainevent._test_util import check_event_op_grads, jax_x64_enabled, requires_gpu, with_jax_x64


# Indexed CSR vector and matrix event primitives
# ----------------------------------------------

@contextmanager
def _jax_x64_enabled():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def _structure(rng, m, k, nse):
    indices = rng.integers(0, k, size=nse).astype(np.int32)
    # contiguous indptr summing to nse
    counts = np.diff(np.sort(rng.integers(0, nse + 1, size=m - 1))) if m > 1 else np.array([], dtype=int)
    # simpler deterministic indptr
    base = nse // m
    rem = nse - base * m
    rows = np.full(m, base, dtype=int)
    rows[:rem] += 1
    indptr = np.concatenate([[0], np.cumsum(rows)]).astype(np.int32)
    perm = rng.permutation(nse).astype(np.int32)
    return jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(perm)


def _heavy_row_structure():
    """2-row CSR whose first row is longer than ``tpr_threshold``, with a reversing ``perm``."""
    heavy = CSRConfig().tpr_threshold + 1
    indices = jnp.concatenate([jnp.arange(heavy, dtype=jnp.int32), jnp.zeros(1, jnp.int32)])
    indptr = jnp.array([0, heavy, heavy + 1], dtype=jnp.int32)
    perm = jnp.arange(heavy + 1, dtype=jnp.int32)[::-1]
    return indices, indptr, perm, heavy


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


def _spy_on(monkeypatch, module, name):
    """Wrap ``module.name``; return the list of ``(args, kwargs)`` it gets called with."""
    calls = []
    real = getattr(module, name)

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


def _assert_fcn_indexed_calls(calls, conn, n_calls):
    # Every call reads the cached CSC mirror.
    csc_indptr, csc_indices, perm = conn.buffers["csc"]
    assert len(calls) == n_calls
    for args, _ in calls:
        assert args[1] is csc_indices and args[2] is csc_indptr and args[3] is perm


def _assert_hybrid_scratch_call(call, n_math, task_capacity, indptr_dtype):
    """A hybrid FFI call gets the math operands plus three scratch buffers made in the lowering."""
    _, out_info, ffi_kwargs, args, kwargs = call
    scratch = [((task_capacity,), indptr_dtype), ((task_capacity,), indptr_dtype), ((2,), jnp.int32)]
    assert len(args) == n_math + len(scratch)
    assert ffi_kwargs['input_output_aliases'] == {n_math + i: 1 + i for i in range(len(scratch))}
    assert kwargs == {'task_capacity': task_capacity}
    assert [(a.shape, a.dtype) for a in args[n_math:]] == [(s, jnp.dtype(d)) for s, d in scratch]
    assert [(o.shape, o.dtype) for o in out_info[1:]] == [(s, jnp.dtype(d)) for s, d in scratch]


_PLATFORM = jax.default_backend()


_IDX_MV_BACKENDS = tuple(binary_csrmv_indexed_p.available_backends(_PLATFORM))


_IDX_MM_BACKENDS = tuple(binary_csrmm_indexed_p.available_backends(_PLATFORM))


_ACT_M, _ACT_K, _ACT_N = 11, 8, 3


def _hp_matmul(a, b):
    """Full-precision float32 matmul (GPU defaults to TF32, ~1e-3 error)."""
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _indexed_operands(homo, seed=0):
    """Return ``(weights, indices, indptr, perm, rows)`` with a random slot permutation."""
    rng = np.random.default_rng(seed)
    _, indices, indptr, _ = random_csr_numpy(rng, _ACT_M, _ACT_K)
    nse = indices.size
    perm = rng.permutation(nse).astype(np.int32)
    weights = (np.array([1.5], np.float32) if homo
               else rng.standard_normal(nse).astype(np.float32))
    rows = np.repeat(np.arange(_ACT_M), np.diff(indptr))
    return jnp.asarray(weights), jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(perm), rows


def _indexed_dense(weights, indices, perm, rows):
    """Differentiable dense matrix with ``D[rows[j], indices[j]] = weights[perm[j]]``."""
    cols = np.asarray(indices)
    vals = jnp.broadcast_to(weights, (cols.size,)) if weights.size == 1 else weights[perm]
    return jnp.zeros((_ACT_M, _ACT_K), weights.dtype).at[rows, cols].add(vals)


_BAD_PERM_CALL = r'''
import sys
from pathlib import Path
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import brainevent._csr.binary.binary_indexed as mod
from brainevent._op import load_cuda_file
load_cuda_file(Path(mod.__file__).parent / 'binary_indexed_csrmv.cu', name='csr_binary_indexed_csrmv')
weights = jnp.ones(3, jnp.float32)
indices = jnp.array([0, 1, 2], jnp.int32)
indptr = jnp.array([0, 2, 3], jnp.int32)
perm = jnp.array([2, 0, 1], jnp.int64)  # must match indptr (int32)
vector = jnp.array([1, 0, 1], jnp.int8)
try:
    r = jax.ffi.ffi_call('csr_binary_indexed_csrmv.binary_csrmv_nt_auto_perm_hetero_f32_bool',
                         jax.ShapeDtypeStruct((2,), jnp.float32))(weights, indices, indptr, perm, vector)
    jax.block_until_ready(r)
except Exception as e:
    print('RAISED', type(e).__name__, e)
    sys.exit(0)
print('NO ERROR')
sys.exit(3)
'''


def _weighted_tol(dtype_name):
    # Event tangents/cotangents of float16/bfloat16 events are rounded to the event dtype.
    if dtype_name in ('float16', 'bfloat16'):
        return dict(rtol=2e-2, atol=2e-2)
    return dict(rtol=1e-4, atol=1e-4)


def _heavy_indexed(seed, m, k, row_nnz, homo):
    rng = np.random.default_rng(seed)
    _, indices, indptr, _ = heavy_row_csr_numpy(rng, m, k, row_nnz)
    nse = indices.size
    perm = rng.permutation(nse).astype(np.int32)
    weights = np.array([1.5], np.float32) if homo else rng.standard_normal(nse).astype(np.float32)
    rows = np.repeat(np.arange(m), np.diff(indptr))
    dense = np.zeros((m, k))
    dense[rows, indices] = weights[0] if homo else weights[perm]
    return jnp.asarray(weights), jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(perm), dense


def _nonfinite_indexed_operands(case, seed=0):
    """Store the CSR-order weights of ``case`` permuted: ``data[perm[j]]`` is slot ``j``."""
    nse = case.indices.size
    if case.data.size == 1:
        return jnp.asarray(case.data), jnp.zeros(nse, jnp.int32)
    perm = np.random.default_rng(seed).permutation(nse).astype(np.int32)
    data = np.empty_like(case.data)
    data[perm] = case.data
    return jnp.asarray(data), jnp.asarray(perm)


def _indexed_call(case, shape, transpose, backend, mm, data=None):
    stored, perm = _nonfinite_indexed_operands(case)
    data = stored if data is None else data
    indptr = jnp.asarray(case.indptr)
    op = binary_csrmm_indexed if mm else binary_csrmv_indexed
    return op(
        data, jnp.asarray(case.indices), indptr, perm, case.events, shape=shape,
        transpose=transpose,
        backend=backend, weighted=case.weighted,
        config=prepared_config(case.indices, case.indptr, shape),
    )


def test_indexed_mv_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
        events = jnp.array([True, False, True])

        got = binary_csrmv_indexed(
            weights, indices, indptr, perm, events, shape=(2, 3), backend='jax_raw',
            config=prepared_config(indices, indptr, (2, 3)))

        assert jnp.allclose(got, jnp.array([40.0, 20.0], dtype=jnp.float32))


@pytest.mark.parametrize('hint', [False, True])
def test_binary_csrmv_indexed_p_call_returns_one_output_and_static_task_capacity(hint):
    indices, indptr, perm, heavy = _heavy_row_structure()
    max_row_nnz = heavy if hint else None
    weights = jnp.arange(1, heavy + 2, dtype=jnp.float32)
    vector = jnp.array([True, True])

    def call(w, v):
        return binary_csrmv_indexed_p_call(w, indices, indptr, perm, v, shape=(2, heavy), transpose=True,
                                           backend="jax_raw", max_row_nnz=max_row_nnz)

    out = call(weights, vector)
    assert len(out) == 1
    ref = binary_csrmv(weights[perm], indices, indptr, vector, shape=(2, heavy), transpose=True,
                       backend="jax_raw", config=prepared_config(indices, indptr, (2, heavy)))
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(ref))

    eqn = _kernel_eqn(binary_csrmv_indexed_p, call, weights, vector)
    assert (len(eqn.invars), len(eqn.outvars)) == (6, 1)
    _assert_task_capacity_param(eqn, indptr, max_row_nnz)


def test_binary_csrmv_indexed_batching_folds_vectors_into_matmat():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    vectors = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
    spec, mask = CSRConfig().kernel_args()

    out, out_dims = _idx_batching(
        (weights, indices, indptr, perm, vectors, mask),
        (None, None, None, None, 0, None),
        config_spec=spec,
        outs=[jax.ShapeDtypeStruct((1,), weights.dtype)],
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vectors.shape[1:], vectors.dtype),
        max_row_nnz=None,
        task_capacity=binary_indexed_mod._task_capacity(indices, indptr, None),
    )

    assert len(out) == 1
    assert out[0].shape == (1, 2)
    # weights[perm] == [2, 1]: batch 0 hits column 0, batch 1 hits column 1.
    np.testing.assert_allclose(np.asarray(out[0]), [[2.0, 1.0]])
    assert list(out_dims) == [1]


def test_binary_csrmv_indexed_vmap_matches_materialized_weights():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    vectors = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)

    got = jax.vmap(
        lambda vector: binary_csrmv_indexed(
            weights,
            indices,
            indptr,
            perm,
            vector,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
            config=prepared_config(indices, indptr, (1, 2)),
        )
    )(vectors)
    ref = jax.vmap(
        lambda vector: binary_csrmv(
            weights[perm],
            indices,
            indptr,
            vector,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
            config=prepared_config(indices, indptr, (1, 2)),
        )
    )(vectors)

    assert got.shape == (2, 1)
    assert jnp.allclose(got, ref)


def test_binary_csrmm_indexed_p_call_returns_one_output_without_mv_workspace():
    indices, indptr, perm, heavy = _heavy_row_structure()
    weights = jnp.arange(1, heavy + 2, dtype=jnp.float32)
    matrix = jnp.array([[True, False, True], [True, True, False]])

    def call(w, b):
        return binary_csrmm_indexed_p_call(w, indices, indptr, perm, b, shape=(2, heavy), transpose=True,
                                           backend="jax_raw")

    out = call(weights, matrix)
    assert len(out) == 1
    ref = binary_csrmm(weights[perm], indices, indptr, matrix, shape=(2, heavy), transpose=True,
                       backend="jax_raw", config=prepared_config(indices, indptr, (2, heavy)))
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(ref))

    eqn = _kernel_eqn(binary_csrmm_indexed_p, call, weights, matrix)
    assert (len(eqn.invars), len(eqn.outvars)) == (6, 1)
    assert not {'task_capacity', 'max_row_nnz'} & eqn.params.keys()
    assert eqn.params['call_plan'] is None


@pytest.mark.parametrize(
    ("batch_axis", "input_shape", "output_shape", "output_axis"),
    [
        (0, (2, 2, 2), (1, 2, 2), 1),
        (1, (2, 2, 2), (1, 2, 2), 1),
        (2, (2, 2, 2), (1, 2, 2), 2),
    ],
)
def test_binary_csrmm_indexed_batching_folds_batch_into_columns(
    batch_axis,
    input_shape,
    output_shape,
    output_axis,
):
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    matrices = jnp.arange(np.prod(input_shape)).reshape(input_shape) % 2 == 0
    spec, mask = CSRConfig().kernel_args()

    out, out_dims = _csrmm_idx_batching(
        (weights, indices, indptr, perm, matrices, mask),
        (None, None, None, None, batch_axis, None),
        config_spec=spec,
        outs=[jax.ShapeDtypeStruct((1, 2), weights.dtype)],
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(matrices.shape[1:], matrices.dtype),
    )

    assert len(out) == 1
    assert out[0].shape == output_shape
    assert list(out_dims) == [output_axis]
    # Every batch slice equals the unbatched product.
    for b in range(input_shape[batch_axis]):
        ref = binary_csrmm(weights[perm], indices, indptr, jnp.take(matrices, b, axis=batch_axis),
                           shape=(1, 2), backend="jax_raw", config=prepared_config(indices, indptr, (1, 2)))
        np.testing.assert_allclose(np.asarray(jnp.take(out[0], b, axis=output_axis)), np.asarray(ref))


def test_binary_csrmm_indexed_vmap_matches_materialized_weights():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    matrices = jnp.array(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ],
        dtype=jnp.bool_,
    )

    got = jax.vmap(
        lambda matrix: binary_csrmm_indexed(
            weights,
            indices,
            indptr,
            perm,
            matrix,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
            config=prepared_config(indices, indptr, (1, 2)),
        )
    )(matrices)
    ref = jax.vmap(
        lambda matrix: binary_csrmm(
            weights[perm],
            indices,
            indptr,
            matrix,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
            config=prepared_config(indices, indptr, (1, 2)),
        )
    )(matrices)

    assert got.shape == (2, 1, 2)
    assert jnp.allclose(got, ref)


def test_binary_csrmm_indexed_weight_jvp_has_single_tangent():
    weights = jnp.array([10.0, 20.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    matrix = jnp.array([[True, False], [False, True]])

    def call(w):
        return binary_csrmm_indexed_p_call(
            w,
            indices,
            indptr,
            perm,
            matrix,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
        )

    primals, tangents = jax.jvp(call, (weights,), (jnp.ones_like(weights),))

    assert len(primals) == len(tangents) == 1
    assert jnp.allclose(primals[0], jnp.array([[20.0, 10.0]], dtype=jnp.float32))
    assert jnp.allclose(tangents[0], jnp.array([[1.0, 1.0]], dtype=jnp.float32))


def test_fcn_unfavorable_binary_matvec_uses_indexed_route(monkeypatch):
    import brainevent._fcn.main as fcn_main
    data = jnp.array([[1.0, 2.0], [3.0, 0.0]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    conn = FixedNumPerPre((data, indices), shape=(2, 2), backend="jax_raw").build_weight_indices()
    vector = jnp.array([False, True])
    expected = conn.todense() @ vector.astype(jnp.float32)
    calls = _spy_on(monkeypatch, fcn_main, "binary_csrmv_indexed")

    got = conn @ BinaryArray(vector)
    got_again = conn @ BinaryArray(vector)

    assert got.shape == (2,)
    assert jnp.allclose(got, expected)
    assert jnp.allclose(got_again, got)
    _assert_fcn_indexed_calls(calls, conn, 2)


def test_fcn_unfavorable_binary_matmat_uses_indexed_route(monkeypatch):
    import brainevent._fcn.main as fcn_main
    data = jnp.array([[1.0, 2.0], [3.0, 0.0]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    conn = FixedNumPerPre((data, indices), shape=(2, 2), backend="jax_raw").build_weight_indices()
    matrix = jnp.array([[False, True], [True, False]])
    expected = conn.todense() @ matrix.astype(jnp.float32)
    calls = _spy_on(monkeypatch, fcn_main, "binary_csrmm_indexed")

    got = conn @ BinaryArray(matrix)
    got_again = conn @ BinaryArray(matrix)

    assert got.shape == (2, 2)
    assert jnp.allclose(got, expected)
    assert jnp.allclose(got_again, got)
    _assert_fcn_indexed_calls(calls, conn, 2)


def test_indexed_mv_rejects_int64_indices_with_int32_indptr():
    with _jax_x64_enabled():
        weights = jnp.ones(2, dtype=jnp.float32)
        indices = jnp.array([0, 1], dtype=jnp.int64)
        indptr = jnp.array([0, 2], dtype=jnp.int32)
        perm = jnp.array([0, 1], dtype=jnp.int32)
        events = jnp.ones(2, dtype=bool)

        with pytest.raises(AssertionError, match="Indices must be int32"):
            binary_csrmv_indexed(
                weights, indices, indptr, perm, events, shape=(1, 2), backend='jax_raw',
                config=prepared_config(indices, indptr, (1, 2)))


def test_indexed_mm_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
        events = jnp.array([[True, False], [False, True], [True, True]])

        got = binary_csrmm_indexed(
            weights, indices, indptr, perm, events, shape=(2, 3), backend='jax_raw',
            config=prepared_config(indices, indptr, (2, 3)))
        expected = jnp.array([[40.0, 10.0], [20.0, 60.0]], dtype=jnp.float32)

        assert jnp.allclose(got, expected)


def test_indexed_mm_rejects_unsigned_structure_dtype():
    weights = jnp.ones(2, dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.uint32)
    indptr = jnp.array([0, 2], dtype=jnp.uint32)
    perm = jnp.array([0, 1], dtype=jnp.int32)
    events = jnp.ones((2, 1), dtype=bool)

    with pytest.raises(AssertionError, match="Indices must be int32"):
        binary_csrmm_indexed(
            weights, indices, indptr, perm, events, shape=(1, 2), backend='jax_raw'
        )


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("homo", [True, False])
@pytest.mark.parametrize("ev", [jnp.bool_, jnp.float32])
def test_indexed_matches_materialized(transpose, homo, ev):
    rng = np.random.default_rng(0)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.ones(1, jnp.float32) if homo else jnp.asarray(rng.random(nse), jnp.float32)
    vlen = m if transpose else k
    v = jnp.asarray(rng.random(vlen) > 0.5, dtype=ev)
    got = binary_csrmv_indexed(
        weights, indices, indptr, perm, v, shape=(m, k), transpose=transpose,
        config=prepared_config(indices, indptr, (m, k)))
    ref_w = weights if homo else weights[perm]
    ref = binary_csrmv(ref_w, indices, indptr, v, shape=(m, k), transpose=transpose, config=prepared_config(indices, indptr, (m, k)))
    assert jnp.allclose(got, ref, atol=1e-5), (transpose, homo, ev)


def test_indexed_jit():
    rng = np.random.default_rng(5)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.bool_)
    f = jax.jit(lambda w: binary_csrmv_indexed(w, indices, indptr, perm, v, shape=(m, k), config=prepared_config(indices, indptr, (m, k))))
    got = f(weights)
    ref = binary_csrmv(weights[perm], indices, indptr, v, shape=(m, k), config=prepared_config(indices, indptr, (m, k)))
    assert jnp.allclose(got, ref, atol=1e-5)


def test_indexed_grad_weights():
    rng = np.random.default_rng(1)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.float32)
    f = lambda w: binary_csrmv_indexed(
        w, indices, indptr, perm, v, shape=(m, k), transpose=False,
        config=prepared_config(indices, indptr, (m, k))).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmv(
        w[perm], indices, indptr, v, shape=(m, k), transpose=False,
        config=prepared_config(indices, indptr, (m, k))).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5)


def test_indexed_grad_weights_transpose():
    rng = np.random.default_rng(11)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(m) > 0.5, dtype=jnp.float32)
    f = lambda w: binary_csrmv_indexed(
        w, indices, indptr, perm, v, shape=(m, k), transpose=True,
        config=prepared_config(indices, indptr, (m, k))).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmv(
        w[perm], indices, indptr, v, shape=(m, k), transpose=True,
        config=prepared_config(indices, indptr, (m, k))).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5)


def test_indexed_check_grads_weights():
    # Only ``weights`` is genuinely differentiable: the event indicator
    # e(v) = (v > 0) is a step function, so the v-gradient is a surrogate
    # (identical routing to binary_csrmv_p) and is not finite-difference checkable.
    # With v held constant the output is exactly linear in weights.
    from jax.test_util import check_grads
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(2)
        m, k, nse = 4, 5, 9
        indices, indptr, perm = _structure(rng, m, k, nse)
        weights = jnp.asarray(rng.random(nse), jnp.float64)
        v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.bool_)
        f = lambda w: binary_csrmv_indexed(w, indices, indptr, perm, v, shape=(m, k), config=prepared_config(indices, indptr, (m, k))).sum()
        check_grads(f, (weights,), order=2, modes=['rev'])
    finally:
        jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("homo", [True, False])
@pytest.mark.parametrize("ev", [jnp.bool_, jnp.float32])
@pytest.mark.parametrize("n", [1, 3])
def test_indexed_mm_matches_materialized(transpose, homo, ev, n):
    rng = np.random.default_rng(0)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.ones(1, jnp.float32) if homo else jnp.asarray(rng.random(nse), jnp.float32)
    rows = m if transpose else k
    B = jnp.asarray(rng.random((rows, n)) > 0.5, dtype=ev)
    got = binary_csrmm_indexed(
        weights, indices, indptr, perm, B, shape=(m, k), transpose=transpose,
        config=prepared_config(indices, indptr, (m, k)))
    ref_w = weights if homo else weights[perm]
    ref = binary_csrmm(ref_w, indices, indptr, B, shape=(m, k), transpose=transpose, config=prepared_config(indices, indptr, (m, k)))
    assert jnp.allclose(got, ref, atol=1e-5), (transpose, homo, ev, n)


def test_indexed_mm_single_column_equals_mv():
    rng = np.random.default_rng(3)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.bool_)
    got = binary_csrmm_indexed(
        weights, indices, indptr, perm, v[:, None], shape=(m, k), transpose=False,
        config=prepared_config(indices, indptr, (m, k)))
    ref = binary_csrmv_indexed(
        weights, indices, indptr, perm, v, shape=(m, k), transpose=False,
        config=prepared_config(indices, indptr, (m, k)))
    assert jnp.allclose(got[:, 0], ref, atol=1e-5)


@pytest.mark.parametrize("transpose", [True, False])
def test_indexed_mm_grad_weights(transpose):
    rng = np.random.default_rng(1)
    m, k, nse, n = 4, 5, 9, 3
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    rows = m if transpose else k
    B = jnp.asarray(rng.random((rows, n)) > 0.5, dtype=jnp.float32)
    f = lambda w: binary_csrmm_indexed(
        w, indices, indptr, perm, B, shape=(m, k), transpose=transpose,
        config=prepared_config(indices, indptr, (m, k))).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmm(
        w[perm], indices, indptr, B, shape=(m, k), transpose=transpose,
        config=prepared_config(indices, indptr, (m, k))).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5), transpose


def test_indexed_mm_grad_weights_homo():
    rng = np.random.default_rng(8)
    m, k, nse, n = 4, 5, 9, 3
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray([0.7], jnp.float32)
    B = jnp.asarray(rng.random((k, n)) > 0.5, dtype=jnp.float32)
    f = lambda w: binary_csrmm_indexed(
        w, indices, indptr, perm, B, shape=(m, k), transpose=False,
        config=prepared_config(indices, indptr, (m, k))).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmm(
        w, indices, indptr, B, shape=(m, k), transpose=False,
        config=prepared_config(indices, indptr, (m, k))).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5)


def test_indexed_mm_jit():
    rng = np.random.default_rng(5)
    m, k, nse, n = 4, 5, 9, 3
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    B = jnp.asarray(rng.random((k, n)) > 0.5, dtype=jnp.bool_)
    f = jax.jit(lambda w: binary_csrmm_indexed(w, indices, indptr, perm, B, shape=(m, k), config=prepared_config(indices, indptr, (m, k))))
    got = f(weights)
    ref = binary_csrmm(weights[perm], indices, indptr, B, shape=(m, k), config=prepared_config(indices, indptr, (m, k)))
    assert jnp.allclose(got, ref, atol=1e-5)


def test_indexed_mm_check_grads_weights():
    from jax.test_util import check_grads
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(2)
        m, k, nse, n = 4, 5, 9, 3
        indices, indptr, perm = _structure(rng, m, k, nse)
        weights = jnp.asarray(rng.random(nse), jnp.float64)
        B = jnp.asarray(rng.random((k, n)) > 0.5, dtype=jnp.bool_)
        f = lambda w: binary_csrmm_indexed(w, indices, indptr, perm, B, shape=(m, k), config=prepared_config(indices, indptr, (m, k))).sum()
        check_grads(f, (weights,), order=2, modes=['rev'])
    finally:
        jax.config.update("jax_enable_x64", False)


def test_indexed_mm_nontransposed_source_retains_only_perm_gather_exports():
    import brainevent._csr.binary.binary_indexed as mod

    from pathlib import Path
    cu = Path(mod.__file__).with_name("binary_indexed_csrmm.cu").read_text()
    # ``nt_auto_perm`` is the only exported permuted entry point: it dispatches
    # internally to the warp and block device kernels, which therefore stay.
    assert "DEFINE_CSRMM_NT_WARP_PERM_HETERO" in cu
    assert "DEFINE_CSRMM_NT_BLOCK_PERM_HETERO" in cu
    assert "binary_csrmm_nt_auto_perm_hetero_f32_bool" in cu
    # The transpose route moved to the hybrid kernels, so the permuted
    # ``t_warp`` family has no caller and must not come back.
    assert "DEFINE_CSRMM_T_WARP_PERM_HETERO" not in cu
    assert "FFI_CSRMM_T_WARP_PERM_HETERO" not in cu
    assert "binary_csrmm_t_warp_perm_hetero" not in cu


def test_nontransposed_raw_cuda_perm_uses_indptr_dtype():
    from pathlib import Path

    csr_dir = Path(binary_indexed_mod.__file__).parent
    for filename in ("binary_indexed_csrmv.cu", "binary_indexed_csrmm.cu"):
        source = (csr_dir / filename).read_text()
        assert "const IndptrT*  __restrict__ perm" in source
        assert "static_cast<const IndptrT*>(perm.data_ptr())" in source
        assert "BE_CHECK(perm.dtype() == indptr.dtype())" in source
        assert "abort()" not in source
        assert "const int32_t*  __restrict__ perm" not in source


def test_import_brainevent_needs_no_nvcc():
    # Importing the module must not compile CUDA (load_cuda_file is lazy).
    import importlib
    import brainevent._csr.binary.binary_indexed as mod
    with pytest.warns(UserWarning, match='already registered') as caught:
        importlib.reload(mod)
    assert len(caught) == 2
    assert all(any(name in str(w.message) for w in caught) for name in
               ('binary_csrmv_indexed', 'binary_csrmm_indexed'))
    assert hasattr(mod, "binary_csrmm_indexed")


def test_indexed_mm_is_exported():
    import brainevent
    from brainevent import binary_csrmm_indexed
    assert "binary_csrmm_indexed" in brainevent.__all__
    assert "binary_csrmm_indexed_p" in brainevent.__all__
    assert binary_csrmm_indexed is brainevent.binary_csrmm_indexed


@pytest.mark.parametrize(
    'factory,args,kwargs',
    [
        (
            binary_indexed_mod._binary_csrmv_indexed_cuda_kernel,
            (shape_of(jnp.float32), shape_of(jnp.bool_), False),
            {'outs': [shape_of(jnp.float32)], 'perm_info': shape_of(jnp.int32)},
        ),
        (
            binary_indexed_mod._binary_csrmm_indexed_cuda_kernel,
            (shape_of(jnp.float32), shape_of(jnp.bool_, (2, 1)), False),
            {'outs': [shape_of(jnp.float32, (1, 1))], 'perm_info': shape_of(jnp.int32)},
        ),
    ],
)
def test_cuda_kernel_generators_reject_int64_indices_before_loading_cuda(factory, args, kwargs):
    call_kwargs = cuda_kwargs()
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match="indices with dtype int32"):
        factory(*args, **call_kwargs)


def test_binary_indexed_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(binary_indexed_mod, "load_cuda_file", lambda path, name, **kwargs: load_calls.append((path, name, kwargs)))
    monkeypatch.setattr(binary_indexed_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))
    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda path, name, **kwargs: load_calls.append((path, name, kwargs)))

    outs = []
    with jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)
        perm = jnp.array([1, 0], dtype=jnp.int32)
        config = CSRConfig().prepare_structure(
            indices, indptr, shape=(1, 2),
            device_info=CSRDeviceInfo('gpu', 'TEST_GPU', 4 * 1024 * 1024, 16),
        )
        spec, mask = config.kernel_args()
        task_capacity = binary_indexed_mod._task_capacity(indices, indptr, None)
        common = {
            **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
            'task_capacity': task_capacity, 'config_spec': spec,
            'call_plan': config.mm_plan(1, jnp.float32),
        }
        mv_outs = [shape_of(jnp.float32)]
        mm_nt_outs = [shape_of(jnp.float32, (1, 1))]
        mm_t_outs = [shape_of(jnp.float32, (2, 1))]

        mv_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (2,)),
            False,
            **{**common, 'perm_info': shape_of(jnp.int32, (2,))},
        )
        outs.append(mv_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32), indices, indptr, perm, jnp.array([True, False]),
        ))

        mv_t_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (1,)),
            True,
            **{**common, 'perm_info': shape_of(jnp.int64, (2,)), 'outs': mv_outs},
        )
        outs.append(mv_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32), indices, indptr, perm.astype(jnp.int64), jnp.array([True]),
        ))

        mv_t_homo_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (1,)),
            True,
            **{**common, 'perm_info': shape_of(jnp.int64, (2,)), 'outs': mv_outs},
        )
        outs.append(mv_t_homo_kernel(
            jnp.array([2.0], dtype=jnp.float32), indices, indptr, perm.astype(jnp.int64), jnp.array([True]),
        ))

        mm_nt_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (2, 1)),
            False,
            **{**common, 'outs': mm_nt_outs, 'perm_info': shape_of(jnp.int32, (2,))},
        )
        outs.append(mm_nt_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32), indices, indptr, perm, jnp.array([[True], [False]]), mask,
        ))

        mm_t_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (1, 1)),
            True,
            **{**common, 'outs': mm_t_outs, 'perm_info': shape_of(jnp.int64, (2,))},
        )
        outs.append(mm_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32), indices, indptr, perm.astype(jnp.int64), jnp.array([[True]]), mask,
        ))

        mm_t_homo_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (1, 1)),
            True,
            **{**common, 'outs': mm_t_outs, 'perm_info': shape_of(jnp.int64, (2,))},
        )
        outs.append(mm_t_homo_kernel(
            jnp.array([2.0], dtype=jnp.float32), indices, indptr, perm.astype(jnp.int64), jnp.array([[True]]), mask,
        ))

    assert [strip_hybrid_suffix(name) for _, name, _ in load_calls] == [
        'csr_binary_indexed_csrmv',
        'csr_hybrid_spmv_col_major_binary',
        'csr_hybrid_spmv_binary',
        'csr_binary_indexed_csrmm',
        'csr_hybrid_mm_col_major_binary',
        'csr_hybrid_mm_row_binary',
    ]
    assert [strip_hybrid_suffix(call[0]) for call in ffi_calls] == [
        'csr_binary_indexed_csrmv.binary_csrmv_nt_auto_perm_hetero_f32_bool',
        'csr_hybrid_spmv_col_major_binary.hybrid_spmv_col_major_binary_hetero_f32_bool',
        'csr_hybrid_spmv_binary.binary_csrmv_wat_hybrid_homo_f32_bool',
        'csr_binary_indexed_csrmm.binary_csrmm_nt_auto_perm_hetero_f32_bool',
        'csr_hybrid_mm_col_major_binary.batched_spmm_v3_col_major_binary_hetero_f32_bool',
        'csr_hybrid_mm_row_binary.batched_spmm_v3_homo_f32_bool',
    ]
    # Kernels return only the math output, in the logical layout.
    assert [len(out) for out in outs] == [1] * 6
    assert [out[0].shape for out in outs] == [(2,), (2,), (2,), (1, 1), (2, 1), (2, 1)]
    # The direct perm kernels take five math operands. Hybrid kernels append
    # their own scratch; the homogeneous ones drop ``perm`` (four operands).
    assert len(ffi_calls[0][3]) == 5 and len(ffi_calls[3][3]) == 5
    for i, n_math in ((1, 5), (2, 4)):
        _assert_hybrid_scratch_call(ffi_calls[i], n_math, task_capacity, jnp.int64)
    for i, mask_index in ((4, 4), (5, 3)):
        _, out_info, _, args, attributes = ffi_calls[i]
        assert args[2].dtype == jnp.int64
        assert args[mask_index] is mask
        assert args[-1].shape == (1, 1)
        assert out_info[0].shape == (1, 2)  # native BN; wrapper returns NB
        assert out_info[1].dtype == jnp.uint32
        assert out_info[2].dtype == jnp.int32
        assert attributes['long_tasks'] == config.primary.long_tasks


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@with_jax_x64
def test_binary_indexed_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    vector = jnp.array([True, False], dtype=jnp.bool_) if transpose else jnp.array([True, False, True])

    got = binary_csrmv_indexed(weights, indices, indptr64, perm, vector, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw', config=prepared_config(indices, indptr64, (2, 3)))
    expected = binary_csrmv_indexed(weights, indices, indptr32, perm, vector, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw', config=prepared_config(indices, indptr32, (2, 3)))

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@with_jax_x64
def test_binary_indexed_csrmm_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    matrix = (
        jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        if transpose else
        jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    )

    got = binary_csrmm_indexed(weights, indices, indptr64, perm, matrix, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw', config=prepared_config(indices, indptr64, (2, 3)))
    expected = binary_csrmm_indexed(weights, indices, indptr32, perm, matrix, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw', config=prepared_config(indices, indptr32, (2, 3)))

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mv_event_is_active_iff_nonzero(backend, dtype_name, homo, transpose):
    with event_dtype_context(dtype_name):
        weights, indices, indptr, perm, rows = _indexed_operands(homo)
        dense = np.asarray(_indexed_dense(weights, indices, perm, rows), dtype=np.float64)
        x = event_values(dtype_name, _ACT_M if transpose else _ACT_K)
        got = binary_csrmv_indexed(
            weights, indices, indptr, perm, x, shape=(_ACT_M, _ACT_K), transpose=transpose,
            backend=backend,
            config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)),
        )
        mask = event_mask(x)
        expected = mask @ dense if transpose else dense @ mask
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mm_event_is_active_iff_nonzero(backend, dtype_name, homo, transpose):
    with event_dtype_context(dtype_name):
        weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=1)
        dense = np.asarray(_indexed_dense(weights, indices, perm, rows), dtype=np.float64)
        B = event_values(dtype_name, (_ACT_M if transpose else _ACT_K, _ACT_N), seed=1)
        got = binary_csrmm_indexed(
            weights, indices, indptr, perm, B, shape=(_ACT_M, _ACT_K), transpose=transpose,
            backend=backend,
            config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)),
        )
        mask = event_mask(B)
        expected = dense.T @ mask if transpose else dense @ mask
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mv_gradients_use_event_mask(backend, homo, transpose):
    """#198: the perm-scattered weight cotangent uses the activity mask."""
    weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=2)
    x = event_values('float32', _ACT_M if transpose else _ACT_K, seed=2)

    def op(w, v):
        return binary_csrmv_indexed(
            w, indices, indptr, perm, v, shape=(_ACT_M, _ACT_K),
            transpose=transpose, backend=backend,
            config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)))

    def dense_op(w, z):
        D = _indexed_dense(w, indices, perm, rows)
        return _hp_matmul(z, D) if transpose else _hp_matmul(D, z)

    check_event_op_grads(op, dense_op, weights, x, mode='binary')


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mm_gradients_use_event_mask(backend, homo, transpose):
    """#198 for the indexed matrix product (SDDMM on the mask, scattered via perm)."""
    weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=3)
    B = event_values('float32', (_ACT_M if transpose else _ACT_K, _ACT_N), seed=3)

    def op(w, v):
        return binary_csrmm_indexed(
            w, indices, indptr, perm, v, shape=(_ACT_M, _ACT_K),
            transpose=transpose, backend=backend,
            config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)))

    def dense_op(w, z):
        D = _indexed_dense(w, indices, perm, rows)
        return _hp_matmul(D.T, z) if transpose else _hp_matmul(D, z)

    check_event_op_grads(op, dense_op, weights, B, mode='binary')


@requires_gpu
def test_indexed_cuda_perm_dtype_mismatch_raises_python_exception():
    """The perm/indptr dtype check throws (Python exception) instead of aborting."""
    proc = run_python_isolated(_BAD_PERM_CALL)
    output = proc.stdout + proc.stderr
    assert proc.returncode >= 0, f'process killed by signal {-proc.returncode}:\n{output}'
    assert proc.returncode == 0, output
    assert 'require perm dtype' in output, output


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('dtype_name', WEIGHTED_EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mv_weighted_matches_dense_and_grads(backend, dtype_name, homo, transpose):
    """Weighted indexed mat-vec: exact product, weight cotangent scattered via perm."""
    with event_dtype_context(dtype_name):
        weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=4)
        x = event_values(dtype_name, _ACT_M if transpose else _ACT_K, seed=4)

        def op(w, v):
            return binary_csrmv_indexed(w, indices, indptr, perm, v, shape=(_ACT_M, _ACT_K),
                                        transpose=transpose, backend=backend,
                                        weighted=True, config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)))

        def dense_op(w, z):
            D = _indexed_dense(w, indices, perm, rows)
            z = z.astype(w.dtype)
            return _hp_matmul(z, D) if transpose else _hp_matmul(D, z)

        check_event_op_grads(op, dense_op, weights, x, mode='weighted',
                             check_vmap=dtype_name == 'float32', **_weighted_tol(dtype_name))


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('dtype_name', WEIGHTED_EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mm_weighted_matches_dense_and_grads(backend, dtype_name, homo, transpose):
    with event_dtype_context(dtype_name):
        weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=5)
        B = event_values(dtype_name, (_ACT_M if transpose else _ACT_K, _ACT_N), seed=5)

        def op(w, v):
            return binary_csrmm_indexed(w, indices, indptr, perm, v, shape=(_ACT_M, _ACT_K),
                                        transpose=transpose, backend=backend,
                                        weighted=True, config=prepared_config(indices, indptr, (_ACT_M, _ACT_K)))

        def dense_op(w, z):
            D = _indexed_dense(w, indices, perm, rows)
            z = z.astype(w.dtype)
            return _hp_matmul(D.T, z) if transpose else _hp_matmul(D, z)

        check_event_op_grads(op, dense_op, weights, B, mode='weighted',
                             check_vmap=dtype_name == 'float32', **_weighted_tol(dtype_name))


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
def test_indexed_mv_weighted_heavy_rows_use_row_value(backend, homo):
    """Multi-chunk hybrid rows re-read their own event value (indexed scatter)."""
    weights, indices, indptr, perm, dense = _heavy_indexed(7, 6, 9000, 8500, homo)
    x = jnp.array([2.0, 0.0, -1.0, 0.5, 3.0, 0.0], dtype=jnp.float32)
    got = binary_csrmv_indexed(weights, indices, indptr, perm, x, shape=dense.shape, transpose=True,
                               backend=backend, weighted=True, config=prepared_config(indices, indptr, dense.shape))
    np.testing.assert_allclose(np.asarray(got, np.float64), np.asarray(x, np.float64) @ dense,
                               rtol=1e-4, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
def test_indexed_mm_weighted_heavy_rows_use_row_value(backend, homo):
    weights, indices, indptr, perm, dense = _heavy_indexed(8, 5, 9000, 8500, homo)
    B = jnp.array([[2.0, 0.0], [0.0, -1.0], [-1.0, 0.5], [0.5, 0.0], [0.0, 3.0]], dtype=jnp.float32)
    got = binary_csrmm_indexed(weights, indices, indptr, perm, B, shape=dense.shape, transpose=True,
                               backend=backend, weighted=True, config=prepared_config(indices, indptr, dense.shape))
    np.testing.assert_allclose(np.asarray(got, np.float64), dense.T @ np.asarray(B, np.float64),
                               rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('homo', [True, False])
def test_indexed_cuda_generators_select_weighted_abi(monkeypatch, homo):
    """Indexed CUDA kernels select weighted exports and cast events to the weight dtype."""
    ffi_calls = []
    monkeypatch.setattr(binary_indexed_mod, "load_cuda_file", lambda path, name, **kwargs: None)
    monkeypatch.setattr(binary_indexed_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))
    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda path, name, **kwargs: None)

    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    task_capacity = binary_indexed_mod._task_capacity(indices, indptr, None)
    n_w = 1 if homo else 2
    weights = jnp.ones(n_w, dtype=jnp.float32)
    config = CSRConfig().prepare_structure(
        indices, indptr, shape=(1, 2),
        device_info=CSRDeviceInfo('gpu', 'TEST_GPU', 4 * 1024 * 1024, 16),
    )
    spec, mask = config.kernel_args()
    if homo:
        names = [
            'csr_binary_csrmv.binary_csrmv_nt_auto_homo_f32_wv',
            'csr_hybrid_spmv_weighted.binary_csrmv_wat_hybrid_homo_f32_wv',
            'csr_binary_csrmm.binary_csrmm_nt_auto_homo_f32_wv',
            'csr_hybrid_mm_row_weighted.batched_spmm_v3_weighted_homo_f32',
        ]
    else:
        names = [
            'csr_binary_indexed_csrmv.binary_csrmv_nt_auto_perm_hetero_f32_wv',
            'csr_hybrid_spmv_col_major_weighted.hybrid_spmv_col_major_weighted_hetero_f32',
            'csr_binary_indexed_csrmm.binary_csrmm_nt_auto_perm_hetero_f32_wv',
            'csr_hybrid_mm_col_major_weighted.batched_spmm_v3_col_major_weighted_hetero_f32',
        ]
    cases = [
        (binary_indexed_mod._binary_csrmv_indexed_cuda_kernel, (2,), False, (1,)),
        (binary_indexed_mod._binary_csrmv_indexed_cuda_kernel, (1,), True, (2,)),
        (binary_indexed_mod._binary_csrmm_indexed_cuda_kernel, (2, 3), False, (1, 3)),
        (binary_indexed_mod._binary_csrmm_indexed_cuda_kernel, (1, 3), True, (2, 3)),
    ]
    for (factory, ev_shape, transpose, out_shape), expected in zip(cases, names):
        ffi_calls.clear()
        events = jnp.full(ev_shape, 4, dtype=jnp.uint8)
        kernel = factory(
            shape_of(jnp.float32, (n_w,)), shape_of(jnp.uint8, ev_shape), transpose,
            **{**cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int32),
               'outs': [shape_of(jnp.float32, out_shape)], 'weighted': True, 'task_capacity': task_capacity,
               'config_spec': spec, 'call_plan': config.mm_plan(3, jnp.float32, weighted=True)},
        )
        if len(ev_shape) == 2:
            kernel(weights, indices, indptr, perm, events, mask)
        else:
            kernel(weights, indices, indptr, perm, events)
        assert len(ffi_calls) == 1
        name, _, _, args, _ = ffi_calls[0]
        assert strip_hybrid_suffix(name) == expected
        # Homogeneous weights reuse the plain kernels, which take no ``perm``.
        hybrid_mm = len(ev_shape) == 2 and transpose
        event_arg = args[-1] if hybrid_mm else args[3] if homo else args[4]
        assert event_arg.dtype == jnp.float32
        expected_shape = ev_shape[::-1] if hybrid_mm else ev_shape
        np.testing.assert_array_equal(np.asarray(event_arg), np.full(expected_shape, 4.0))


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('bad', tuple(NONFINITE_VALUES))
def test_indexed_mv_inactive_nonfinite_weights_contribute_zero(backend, transpose, homo, kind, bad):
    shape = (7, 6)
    case = inactive_nonfinite_case(10, *shape, transpose=transpose, homo=homo, kind=kind, bad=bad)
    assert_select_reference(_indexed_call(case, shape, transpose, backend, mm=False), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('bad', tuple(NONFINITE_VALUES))
def test_indexed_mm_inactive_nonfinite_weights_contribute_zero(backend, transpose, homo, kind, bad):
    shape = (7, 6)
    case = inactive_nonfinite_case(
        11, *shape, transpose=transpose, homo=homo, kind=kind, bad=bad, n_batch=4,
    )
    assert_select_reference(_indexed_call(case, shape, transpose, backend, mm=True), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
def test_indexed_mm_nonfinite_weight_isolated_per_event_column(backend, transpose, kind):
    shape = (7, 6)
    case = inactive_nonfinite_case(
        12, *shape, transpose=transpose, homo=False, kind=kind, bad='nan', n_batch=4, partial=True,
    )
    assert_select_reference(_indexed_call(case, shape, transpose, backend, mm=True), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('mm', [False, True])
def test_indexed_inactive_nonfinite_weights_heavy_rows(backend, homo, kind, mm):
    if mm and backend not in _IDX_MM_BACKENDS:
        pytest.skip(f'{backend} has no indexed csrmm kernel')
    shape = (4, 300)
    case = inactive_nonfinite_case(
        13, *shape, transpose=True, homo=homo, kind=kind, bad='nan',
        n_batch=3 if mm else None, heavy_row_nnz=200,
    )
    assert_select_reference(_indexed_call(case, shape, True, backend, mm=mm), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
def test_indexed_active_nan_weight_still_propagates(backend, transpose):
    shape = (7, 6)
    case = inactive_nonfinite_case(14, *shape, transpose=transpose, homo=False, kind='bool', bad='nan')
    e = int(np.flatnonzero(case.active[:, 0])[0])
    data = case.data.copy()
    data[e] = np.nan
    out = np.asarray(_indexed_call(case._replace(data=data), shape, transpose, backend, mm=False))
    target = case.indices[e] if transpose else int(np.searchsorted(case.indptr, e, side='right') - 1)
    assert np.isnan(out[target])
    assert np.isfinite(out).any()


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('mm', [False, True])
def test_indexed_weight_cotangent_of_inactive_entries_is_exact_zero(backend, transpose, kind, mm):
    if mm and backend not in _IDX_MM_BACKENDS:
        pytest.skip(f'{backend} has no indexed csrmm kernel')
    shape = (7, 6)
    case = inactive_nonfinite_case(
        15, *shape, transpose=transpose, homo=False, kind=kind, bad='nan',
        n_batch=3 if mm else None,
    )
    case = case._replace(data=np.where(np.isfinite(case.data), case.data, 1.0).astype(np.float32))
    stored, perm = _nonfinite_indexed_operands(case)

    def f(d):
        return _indexed_call(case, shape, transpose, backend, mm=mm, data=d)

    out, vjp = jax.vjp(f, stored)
    (ct_stored,) = vjp(jnp.full_like(out, jnp.nan))
    ct_slot = np.asarray(ct_stored)[np.asarray(perm)]
    inactive = case.inactive_edge
    assert inactive.any() and (~inactive).any()
    np.testing.assert_array_equal(ct_slot[inactive], 0.0)
    assert np.isnan(ct_slot[~inactive]).all()


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('mm', [False, True])
def test_indexed_homo_weight_cotangent_ignores_nan_on_idle_outputs(backend, transpose, kind, mm):
    if mm and backend not in _IDX_MM_BACKENDS:
        pytest.skip(f'{backend} has no indexed csrmm kernel')
    shape = (7, 6)
    case = inactive_nonfinite_case(
        16, *shape, transpose=transpose, homo=True, kind=kind, bad='nan', n_batch=3 if mm else None,
    )
    ct, expected = homo_ct_case(case, transpose)

    def f(d):
        return _indexed_call(case, shape, transpose, backend, mm=mm, data=d)

    _, vjp = jax.vjp(f, jnp.ones(1, jnp.float32))
    (ct_w,) = vjp(ct)
    assert np.isfinite(expected)
    np.testing.assert_allclose(np.asarray(ct_w), [expected], rtol=1e-5)


# Indexed native short-column scheduling
# --------------------------------------

@requires_gpu_backend
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
def test_col_major_respects_configured_short_column_threshold(weighted):
    # A 129-edge column is short at the configured threshold, so no task buffer
    # is needed. Using the old private macro names silently drops this column.
    root = Path(__file__).parents[1] / 'hybrid_mv'
    mode = 'weighted' if weighted else 'binary'
    source = root / f'hybrid_spmv_col_major_{mode}.cu'
    header = root / 'hybrid_spmv_col_major_impl.cuh'
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[:12]
    module = f'csr_spmv_threshold_{mode}_{digest}'
    config = CSRConfig(tpr_threshold=256)
    load_cuda_file(
        source, name=module, extra_include_paths=[str(root)],
        extra_cuda_cflags=config.compile_flags(), allow_cuda_graph=False,
    )
    count = 129
    weights = jnp.arange(1, count + 1, dtype=jnp.float32)
    indices = jnp.arange(count, dtype=jnp.int32)
    indptr = jnp.array([0, count], dtype=jnp.int32)
    perm = jnp.arange(count - 1, -1, -1, dtype=jnp.int32)
    events = jnp.array([2.0], dtype=jnp.float32)
    scratch = (jnp.empty((0,), jnp.int32), jnp.empty((0,), jnp.int32),
               jnp.full((2,), -1, jnp.int32))
    out_infos = (jax.ShapeDtypeStruct((count,), jnp.float32),
                 *(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in scratch))
    suffix = 'f32' if weighted else 'f32_float'
    target = f'{module}.hybrid_spmv_col_major_{mode}_hetero_{suffix}'
    output, _, _, status = jax.ffi.ffi_call(
        target, out_infos, input_output_aliases={5: 1, 6: 2, 7: 3},
    )(weights, indices, indptr, perm, events, *scratch, task_capacity=0)
    expected = np.arange(count, 0, -1, dtype=np.float32) * (2 if weighted else 1)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(status, [0, 0])


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
@pytest.mark.parametrize('indexed,homo', [(True, False), (True, True)])
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
@pytest.mark.parametrize('indexed', [True])
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
@pytest.mark.parametrize('indexed', [True])
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
@pytest.mark.parametrize('indexed', [True])
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
@pytest.mark.parametrize('indexed,homo', [(True, False), (True, True)])
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
@pytest.mark.parametrize('indexed', [True])
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
@pytest.mark.parametrize('indexed', [True])
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
@pytest.mark.parametrize('indexed', [True])
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
        with pytest.raises(brainevent.KernelCompilationError, match='prepared CSRConfig'):
            jax.jit(native)(pointers)


@pytest.mark.usefixtures('_prepared_enable_x64')
@pytest.mark.parametrize('indexed', [True])
def test_native_unprepared_mm_and_mv_vmap_fail_at_hybrid_lowering(indexed):
    if jax.default_backend() != 'gpu':
        pytest.skip('actual hybrid lowering requires GPU')
    case = _prepared_case()
    for config in (None, CSRConfig(device_info=case.config.device_info),
                   replace(case.config, device_info=None)):
        function = lambda x: _prepared_call(case, case.weights, x, indexed=indexed, config=config)
        with pytest.raises(brainevent.KernelCompilationError, match='prepared CSRConfig'):
            jax.jit(function)(case.events)
        with pytest.raises(brainevent.KernelCompilationError, match='prepared CSRConfig'):
            jax.jit(jax.vmap(function))(case.events.T)


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
        call = lambda w: op(
            w, indices, indptr, events, shape=(3, 4), transpose=transpose,
            weighted=case == 'weighted',
            backend=backend)
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


@pytest.mark.parametrize('case,batch', [('indexed_binary', None), ('indexed_weighted', None), ('indexed_binary', 0), ('indexed_weighted', 0), ('indexed_binary', 1), ('indexed_weighted', 1), ('indexed_binary', 3), ('indexed_weighted', 3)])
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
@pytest.mark.parametrize('indexed', [True])
def test_event_consumers_preserve_activity_before_narrowing(transpose, batch, weighted, indexed):
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
    expected = jnp.where(x != 0, terms, 0)
    if batch is not None:
        expected = jnp.sum(expected, axis=1)
    if indexed:
        expected = jnp.zeros_like(weights).at[perm].add(expected)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual.dtype == ct.dtype


@pytest.mark.parametrize('case,batch', [('indexed_binary', None), ('indexed_weighted', None), ('indexed_binary', 1), ('indexed_weighted', 1), ('indexed_binary', 3), ('indexed_weighted', 3)])
@pytest.mark.parametrize('transpose', [False, True])
def test_migrated_consumers_preserve_second_order_gradients(case, transpose, batch):
    call, w, _, _, _, _ = _sampled_operands(case, transpose, batch)
    _sampled_check_second_order_gradients(call, w)


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('indexed', [True])
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


@pytest.mark.parametrize('case', ['indexed_binary', 'indexed_weighted'])
@pytest.mark.parametrize('batch', [None, 1, 3])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('selection', ['jax_raw', 'cuda_raw', 'default_jax', 'default_cuda'])
def test_event_weight_gradient_follows_csr_backend(monkeypatch, case, batch, transpose, selection):
    if jax.default_backend() != 'gpu':
        pytest.skip('inspect the selected CSR and sampled CUDA backends on a GPU')
    indexed = case.startswith('indexed')
    module = _sampled_indexed_csr if indexed else _sampled_binary_csr
    name = 'binary_csrmv' if batch is None else 'binary_csrmm'
    primitive = getattr(module, name + ('_indexed_p' if indexed else '_p'))
    if selection.startswith('default_'):
        selected = 'cuda_raw' if selection == 'default_cuda' else 'jax_raw'
        monkeypatch.setitem(primitive._defaults, 'gpu', selected)
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
    if native:
        assert 'stablehlo.is_finite' not in ir


@pytest.mark.parametrize('case', ['indexed_binary', 'indexed_weighted'])
# Raw CUDA scatter MM requires a prepared config; retain MV and gather MM here.
@pytest.mark.parametrize('transpose,batch', [(False, None), (False, 3), (True, None)])
@pytest.mark.parametrize('backend', [None, 'cuda_raw'])
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


@pytest.mark.parametrize('indexed', [True])
@pytest.mark.parametrize('batch', [None, 3])
@pytest.mark.parametrize('transpose', [False, True])
def test_event_gradient_resolves_backend_on_compilation_platform(monkeypatch, indexed, batch, transpose):
    if jax.default_backend() != 'gpu':
        pytest.skip('the default device must differ from the CPU compilation target')
    module = _sampled_indexed_csr if indexed else _sampled_binary_csr
    name = 'binary_csrmv' if batch is None else 'binary_csrmm'
    primitive = getattr(module, name + ('_indexed_p' if indexed else '_p'))
    monkeypatch.setitem(primitive._defaults, 'gpu', 'cuda_raw')
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
