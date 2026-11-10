# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_sddmm/_common.py``, grouped by behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._error import KernelCompilationError
from brainevent._sddmm import csr_sddmm_dweight, csr_sddmm_p, csr_sddmv_p
from brainevent._sddmm.dispatch import sddmm
from brainevent._test_util import jax_x64_enabled


# Shared cuSPARSE backend
# -----------------------

@pytest.mark.parametrize('primitive', [csr_sddmv_p, csr_sddmm_p], ids=['mv', 'mm'])
def test_cusparse_backend_registration(primitive):
    assert 'cusparse' in primitive.available_backends('gpu')
    assert 'cusparse' not in primitive.available_backends('cpu')
    assert 'cusparse' not in primitive.available_backends('tpu')


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
@pytest.mark.parametrize('transpose', [False, True])
def test_cusparse_sampled_product(batch, mode, transpose):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    left = jnp.asarray(np.resize([0., -2., 1., .5], (3, batch)), jnp.float32)
    right = jnp.asarray(np.resize([.5, 2., -1., 3., 0.], (batch, 4)), jnp.float32)
    indices = jnp.array([3, 1, 1, 0], jnp.int32)
    indptr = jnp.array([0, 3, 3, 4], jnp.int32)
    fn = jax.jit(lambda left, right, indices, indptr: sddmm(
        left, right, indices, indptr, mode=mode, transpose=transpose, backend='cusparse'))
    expected = sddmm(left, right, indices, indptr, mode=mode,
                     transpose=transpose, backend='jax_raw')
    np.testing.assert_allclose(fn(left, right, indices, indptr), expected, rtol=1e-6)
    assert 'csr_sddmm_cusparse.cusparse_sddmm_f32' in fn.lower(left, right, indices, indptr).as_text()


@pytest.mark.parametrize('batch', [1, 5])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('offset_dtype', [jnp.int32, jnp.int64])
def test_cusparse_value_and_offset_dtypes(batch, mode, value_dtype, offset_dtype):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    with jax_x64_enabled():
        left = jnp.asarray(np.resize([0., -2., 1., .5], (3, batch)), value_dtype)
        right = jnp.asarray(np.resize([.25, -.5, 1., 0.], (batch, 4)), value_dtype)
        indices = jnp.array([3, 1, 1, 0], jnp.int32)
        indptr = jnp.array([0, 3, 3, 4], offset_dtype)
        fn = jax.jit(lambda left, right, indices, indptr: sddmm(
            left, right, indices, indptr, mode=mode, backend='cusparse'))
        result = fn(left, right, indices, indptr)
        expected = sddmm(left, right, indices, indptr, mode=mode, backend='jax_raw')
        assert result.dtype == value_dtype
        np.testing.assert_allclose(np.asarray(result, np.float64), np.asarray(expected, np.float64), rtol=1e-6)
        suffix = 'f64' if value_dtype == jnp.float64 else 'f32'
        assert f'cusparse_sddmm_{suffix}' in fn.lower(left, right, indices, indptr).as_text()


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_cusparse_nonfinite_inactive_events(batch, transpose, weighted):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    events = jnp.zeros((3, batch), jnp.float32).at[:, 0].set(jnp.array([0., -2., 1.]))
    ct = jnp.full((3, batch), jnp.nan, jnp.float32).at[:, 0].set(jnp.array([jnp.inf, 3., jnp.nan]))
    indices, indptr = jnp.arange(3, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32)
    fn = jax.jit(lambda events, ct: csr_sddmm_dweight(
        events, ct, indices, indptr, transpose=transpose, weighted=weighted, backend='cusparse'))
    np.testing.assert_array_equal(fn(events, ct), [0., -6. if weighted else 3., np.nan])


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('event_dtype,value_dtype,tiny', [
    (jnp.float64, jnp.float32, 1e-200),
    (jnp.float32, jnp.float16, 1e-10),
    (jnp.float64, jnp.bfloat16, 1e-200),
])
def test_cusparse_weighted_activity_after_cast(batch, transpose, event_dtype, value_dtype, tiny):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    with jax_x64_enabled():
        events = jnp.zeros((3, batch), event_dtype).at[:, 0].set(
            jnp.array([tiny, -tiny, 2.], event_dtype))
        ct = jnp.full((3, batch), -jnp.inf, value_dtype).at[:, 0].set(
            jnp.array([jnp.inf, jnp.nan, 3.], value_dtype))
        indices, indptr = jnp.arange(3, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32)
        fn = jax.jit(lambda events, ct: csr_sddmm_dweight(
            events, ct, indices, indptr, transpose=transpose, weighted=True, backend='cusparse'))
        result = fn(events, ct)
        assert result.dtype == value_dtype
        np.testing.assert_array_equal(np.asarray(result, np.float64), [0., 0., 6.])


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('event_dtype,event_values', [
    (jnp.bool_, [False, True, False]),
    (jnp.int8, [0, -2, 3]),
    (jnp.uint64, [0, 2**63, 1]),
    (jnp.float64, [0., 1e-200, np.nan]),
])
def test_cusparse_binary_event_dtypes(batch, event_dtype, event_values):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    with jax_x64_enabled():
        events = jnp.asarray(np.resize(event_values, (3, batch)), event_dtype)
        ct = jnp.full((3, batch), 2., jnp.float32)
        indices, indptr = jnp.arange(3, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32)
        fn = jax.jit(lambda events, ct: csr_sddmm_dweight(
            events, ct, indices, indptr, transpose=True, backend='cusparse'))
        expected = csr_sddmm_dweight(events, ct, indices, indptr, transpose=True, backend='jax_raw')
        np.testing.assert_array_equal(fn(events, ct), expected)


@pytest.mark.parametrize('batch', [1, 5])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
def test_cusparse_runtime_patterns_repeated_calls_and_vmap(batch, mode):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    left = jnp.asarray(np.resize([0., -2., 1., .5], (4, batch)), jnp.float32)
    right = jnp.asarray(np.resize([.25, -.5, 1., 0.], (batch, 5)), jnp.float32)
    patterns = (
        (jnp.array([4, 1, 1, 0, 3, 2], jnp.int32), jnp.array([0, 4, 4, 5, 6], jnp.int32)),
        (jnp.array([2, 0, 4, 4, 1, 1], jnp.int32), jnp.array([0, 0, 2, 3, 6], jnp.int32)),
    )
    traces = []
    def call(left, right, indices, indptr):
        traces.append(True)
        return sddmm(left, right, indices, indptr, mode=mode, backend='cusparse')
    fn = jax.jit(call)
    for indices, indptr in patterns:
        expected = sddmm(left, right, indices, indptr, mode=mode, backend='jax_raw')
        np.testing.assert_allclose(fn(left, right, indices, indptr), expected, rtol=1e-6)
        np.testing.assert_array_equal(fn(jnp.zeros_like(left), right, indices, indptr), np.zeros(6))
    assert len(traces) == 1
    left_batch, right_batch = jnp.stack([left, left * 2]), jnp.stack([right, right * .5])
    index_batch = jnp.stack([pattern[0] for pattern in patterns])
    pointer_batch = jnp.stack([pattern[1] for pattern in patterns])
    result = jax.jit(jax.vmap(fn))(left_batch, right_batch, index_batch, pointer_batch)
    expected = jnp.stack([
        sddmm(left_batch[position], right_batch[position], indices, indptr, mode=mode, backend='jax_raw')
        for position, (indices, indptr) in enumerate(patterns)
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('nonfinite', [False, True])
def test_cusparse_shard_map(batch, mode, transpose, nonfinite):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    from jax.sharding import Mesh, PartitionSpec as P
    left = jnp.asarray(np.resize([0., 1., -2., .5], (2, batch)), jnp.float32)
    right = jnp.asarray(np.resize([1., -2., .5, 3.], (batch, 3)), jnp.float32)
    if nonfinite:
        if transpose:
            right = right.at[0, 0].set(jnp.inf)
        else:
            left = left.at[0, 0].set(jnp.inf)
    indices, indptr = jnp.array([2, 0, 0], jnp.int32), jnp.array([0, 1, 3], jnp.int32)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('s',))
    fn = jax.shard_map(
        lambda left, right: sddmm(left, right, indices, indptr, mode=mode,
                                  transpose=transpose, backend='cusparse'),
        mesh=mesh, in_specs=(P('s'), P('s')), out_specs=P('s'),
    )
    expected = sddmm(left, right, indices, indptr, mode=mode, transpose=transpose, backend='jax_raw')
    np.testing.assert_allclose(jax.jit(fn)(left, right), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize('batch,rows,cols,nnz', [(1, 3, 4, 0), (3, 0, 4, 0), (3, 3, 0, 0),
                                               (0, 3, 4, 3), (0, 0, 0, 0)])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
def test_cusparse_empty_shapes(batch, rows, cols, nnz, mode):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    left, right = jnp.ones((rows, batch), jnp.float32), jnp.ones((batch, cols), jnp.float32)
    indices = jnp.zeros(nnz, jnp.int32)
    indptr = jnp.zeros(rows + 1, jnp.int32).at[-1].set(nnz)
    fn = jax.jit(lambda left, right: sddmm(left, right, indices, indptr, mode=mode, backend='cusparse'))
    np.testing.assert_array_equal(fn(left, right), np.zeros(nnz))
    assert 'cusparse_sddmm_' not in fn.lower(left, right).as_text()


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
def test_cusparse_rejects_coo(batch, mode):
    left, right = jnp.ones((2, batch)), jnp.ones((batch, 2))
    with pytest.raises(NotImplementedError, match='require CSR'):
        sddmm(left, right, jnp.array([[0, 1]], jnp.int32), mode=mode, backend='cusparse')


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
def test_cusparse_rejects_excess_slots(batch, mode):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    left, right = jnp.ones((1, batch)), jnp.ones((batch, 1))
    indices, indptr = jnp.array([0, 0], jnp.int32), jnp.array([0, 2], jnp.int32)
    error_type = ValueError if mode == 'float' else KernelCompilationError
    with pytest.raises(error_type, match='nnz <= rows'):
        jax.jit(lambda: sddmm(left, right, indices, indptr, mode=mode, backend='cusparse'))()


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('left_dtype,right_dtype,result_dtype', [
    (jnp.float16, jnp.bfloat16, jnp.float32),
    (jnp.float32, jnp.float64, jnp.float64),
    (jnp.int32, jnp.float32, jnp.float32),
])
def test_cusparse_ordinary_dtype_promotion(batch, left_dtype, right_dtype, result_dtype):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    with jax_x64_enabled():
        left, right = jnp.full((2, batch), 2, left_dtype), jnp.full((batch, 2), .5, right_dtype)
        indices, indptr = jnp.array([1, 0], jnp.int32), jnp.arange(3, dtype=jnp.int32)
        result = jax.jit(lambda: sddmm(left, right, indices, indptr, backend='cusparse'))()
        assert result.dtype == result_dtype
        np.testing.assert_array_equal(result, [batch, batch])


@pytest.mark.parametrize('batch', [1, 3])
def test_cusparse_rejects_integer_result(batch):
    if jax.default_backend() != 'gpu':
        pytest.skip('cuSPARSE execution requires a GPU')
    left, right = jnp.ones((2, batch), jnp.int32), jnp.ones((batch, 2), jnp.int32)
    indices, indptr = jnp.array([1, 0], jnp.int32), jnp.arange(3, dtype=jnp.int32)
    with pytest.raises(TypeError, match='must promote to'):
        sddmm(left, right, indices, indptr, backend='cusparse')
