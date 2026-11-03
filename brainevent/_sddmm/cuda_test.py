"""CUDA assembly preserves direction dispatch and accepts runtime CSR patterns."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._sddmm import csr_sddmm_dweight, csr_sddmm_p, csr_sddmv_p
from brainevent._test_util import jax_x64_enabled


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('backend', [None, 'auto', 'cuda_raw'])
def test_cuda_registration_and_lowering(batch, weighted, backend):
    primitive = csr_sddmv_p if batch == 1 else csr_sddmm_p
    assert {'auto', 'cuda_raw', 'jax_raw'} <= set(primitive.available_backends('gpu'))
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    e = jnp.asarray(np.resize([0., -2., 1.], (2, batch)), jnp.float32)
    c = jnp.arange(3 * batch, dtype=jnp.float32).reshape(3, batch)
    i, p = jnp.array([2, 0, 0], jnp.int32), jnp.array([0, 1, 3], jnp.int32)
    fn = jax.jit(lambda e, c, i, p: csr_sddmm_dweight(
        e, c, i, p, transpose=True, weighted=weighted, backend=backend))
    expected = csr_sddmm_dweight(e, c, i, p, transpose=True, weighted=weighted, backend='jax_raw')
    np.testing.assert_array_equal(fn(e, c, i, p), expected)
    target = 'hybrid_sddmv_' if batch == 1 else 'batched_v3_sddmm_'
    assert target in fn.lower(e, c, i, p).as_text()


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('transpose,backend', [(True, 'jax_raw'), (False, 'jax_raw'), (False, 'auto')])
def test_portable_routes_do_not_load_cuda(monkeypatch, batch, transpose, backend):
    leaf = 'mv.sddmv' if batch == 1 else 'mm.sddmm'
    module = importlib.import_module(f'brainevent._sddmm._{leaf}')
    def fail_load(*args, **kwargs):
        raise AssertionError('portable execution must not load CUDA')
    monkeypatch.setattr(module, '_load_hybrid', fail_load)
    e = jnp.ones((2 if transpose else 3, batch), jnp.float32)
    c = jnp.full((3 if transpose else 2, batch), 2., jnp.float32)
    i, p = jnp.array([2, 0], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    fn = jax.jit(lambda e, c: csr_sddmm_dweight(
        e, c, i, p, transpose=transpose, weighted=True, backend=backend))
    np.testing.assert_array_equal(fn(e, c), [2 * batch, 2 * batch])
    ir = fn.lower(e, c).as_text()
    assert 'hybrid_sddmv_' not in ir and 'batched_v3_sddmm_' not in ir


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('backend', ['auto', 'cuda_raw'])
@pytest.mark.parametrize('event_dtype,value_dtype,tiny', [
    (jnp.float64, jnp.float32, 1e-200),
    (jnp.float32, jnp.float16, 1e-10),
    (jnp.float64, jnp.bfloat16, 1e-200),
])
def test_weighted_cuda_checks_activity_after_cast(batch, backend, event_dtype, value_dtype, tiny):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    with jax_x64_enabled():
        e = jnp.zeros((3, batch), event_dtype).at[:, 0].set(
            jnp.array([tiny, -tiny, 2.], event_dtype))
        c = jnp.full((3, batch), -jnp.inf, value_dtype).at[:, 0].set(
            jnp.array([jnp.inf, jnp.nan, 3.], value_dtype))
        i, p = jnp.arange(3, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32)
        fn = jax.jit(lambda e, c: csr_sddmm_dweight(
            e, c, i, p, transpose=True, weighted=True, backend=backend))
        result = fn(e, c)
        assert result.dtype == c.dtype
        # Narrowed events and exact zeros skip nonfinite cotangents; the
        # active third row still contributes its weighted finite value.
        np.testing.assert_array_equal(np.asarray(result, np.float64), [0., 0., 6.])
        ir = fn.lower(e, c).as_text()
        target = 'hybrid_sddmv_weighted_' if batch == 1 else 'batched_v3_sddmm_weighted_'
        assert target in ir
        assert 'stablehlo.is_finite' not in ir


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('offset_dtype', [jnp.int32, jnp.int64])
def test_mm_runtime_patterns_heavy_rows_and_repeated_calls(weighted, value_dtype, offset_dtype):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    with jax_x64_enabled():
        # 35 rows cross a mask word; 65 batches cross the adapter's batch chunk.
        rows, cols, batch, nnz = 35, 17, 65, 1539
        e = jnp.asarray(np.resize([0., 1., -2., .5], (rows, batch)), value_dtype)
        c = jnp.asarray(np.resize([.25, -.5, 1., 0.], (cols, batch)), value_dtype)
        i = jnp.arange(nnz, dtype=jnp.int32) % cols
        traces = []
        def call(e, c, i, p):
            traces.append(True)
            return csr_sddmm_dweight(e, c, i, p, transpose=True, weighted=weighted, backend='cuda_raw')
        fn = jax.jit(call)
        patterns = ([nnz] + [0] * 34, [513] * 3 + [0] * 32,
                    [0] * 31 + [513, 0, 0, 1026], [44] * 34 + [43])
        for lengths in patterns:
            p = jnp.asarray(np.r_[0, np.cumsum(lengths)], offset_dtype)
            r = np.repeat(np.arange(rows), lengths)
            ev = np.asarray(e, np.float64)[r]
            ct = np.asarray(c, np.float64)[np.asarray(i)]
            expected = np.where(ev != 0, ev * ct if weighted else ct, 0).sum(axis=1)
            expected = np.asarray(jnp.asarray(expected, value_dtype), np.float64)
            np.testing.assert_allclose(np.asarray(fn(e, c, i, p), np.float64), expected, rtol=1e-6)
            np.testing.assert_array_equal(fn(jnp.zeros_like(e), c, i, p), np.zeros(nnz))
        assert len(traces) == 1


@pytest.mark.parametrize('nnz,batch', [(0, 3), (2, 0), (513, 2)])
def test_mm_degenerate_shapes_and_single_row(nnz, batch):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    e, c = jnp.ones((1, batch), jnp.float32), jnp.full((2, batch), 2., jnp.float32)
    i, p = jnp.zeros(nnz, jnp.int32), jnp.array([0, nnz], jnp.int32)
    for weighted in (False, True):
        fn = jax.jit(lambda e, c: csr_sddmm_dweight(
            e, c, i, p, transpose=True, weighted=weighted, backend='cuda_raw'))
        np.testing.assert_array_equal(fn(e, c), np.full(nnz, 2 * batch))


def test_mm_out_of_range_batch_falls_back_under_auto():
    if jax.default_backend() != 'gpu':
        pytest.skip('GPU auto routing requires a GPU')
    e = jnp.ones((1, 65536), jnp.float32)
    i, p = jnp.array([0], jnp.int32), jnp.array([0, 1], jnp.int32)
    np.testing.assert_array_equal(csr_sddmm_dweight(e, e, i, p, transpose=True), [65536.])
    with pytest.raises(ValueError, match='65535'):
        csr_sddmm_dweight(e, e, i, p, transpose=True, backend='cuda_raw')
