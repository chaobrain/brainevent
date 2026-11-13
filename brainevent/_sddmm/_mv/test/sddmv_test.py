# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_sddmm/_mv/sddmv.py``, grouped by behavior."""

import importlib
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._sddmm import csr_sddmm_dweight, csr_sddmm_p, csr_sddmv_dweight, csr_sddmv_p
from brainevent._test_util import jax_x64_enabled


# Vector sampled products and scheduler snapshots
# -----------------------------------------------

sddmv = importlib.import_module('brainevent._sddmm._mv.sddmv')


@pytest.fixture(autouse=False)
def enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def inputs(value_dtype, event_dtype, transpose):
    indices = jnp.array([2, 0, 1, 3, 0], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 2, 5], dtype=jnp.int64)
    events = jnp.array([2, 0, 3] if transpose else [2, 0, 3, 0], dtype=event_dtype)
    ct = jnp.array([1.25, -2, 4, 0.5] if transpose else [1.25, -2, 4], dtype=value_dtype)
    return events, ct, indices, indptr


def reference(events, ct, indices, indptr, *, transpose, weighted):
    rows = jnp.repeat(jnp.arange(indptr.size - 1), jnp.diff(indptr),
                      total_repeat_length=indices.size)
    unit, out = (rows, indices) if transpose else (indices, rows)
    x = events[unit]
    # Weighted activity is tested after the cast to ct.dtype, binary before it.
    active = x.astype(ct.dtype) != 0 if weighted else x != 0
    value = x.astype(ct.dtype) * ct[out] if weighted else ct[out]
    return jnp.where(active, value, jnp.zeros_like(value))


def forward_backend(config):
    """Carry a scheduler snapshot as a CSR consumer does, with automatic selection."""
    from brainevent._csr.binary.binary import binary_csrmv_p
    from brainevent._sddmm.dispatch import _CSRBackend
    return _CSRBackend(binary_csrmv_p, None, config.kernel_args()[0])


def check_result(args, *, transpose, weighted, backend=None):
    fn = jax.jit(lambda e, c, i, p: csr_sddmv_dweight(
        e, c, i, p, transpose=transpose, weighted=weighted, backend=backend))
    actual = fn(*args)
    expected = reference(*args, transpose=transpose, weighted=weighted)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64),
                               np.asarray(expected, dtype=np.float64), rtol=2e-6, atol=0,
                               equal_nan=True)
    return fn, actual


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('event_dtype', [jnp.bool_, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                                        jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64,
                                        jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_value_event_types(value_dtype, event_dtype, transpose, weighted):
    check_result(inputs(value_dtype, event_dtype, transpose),
                 transpose=transpose, weighted=weighted)


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_nonfinite_cotangents(value_dtype, transpose, weighted):
    e, c, i, p = inputs(value_dtype, value_dtype, transpose)
    c = jnp.array([jnp.inf, jnp.nan, -jnp.inf, jnp.nan][:c.size], dtype=value_dtype)
    check_result((e, c, i, p), transpose=transpose, weighted=weighted)


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_jax_event_kernel_is_branch_free(transpose, weighted):
    # Nonfinite cotangents are masked per slot: no global isfinite reduction
    # and no conditional, so the backward pass runs one fused gather product.
    args = inputs(jnp.float32, jnp.float32, transpose)
    fn = jax.jit(lambda e, c, i, p: csr_sddmv_dweight(
        e, c, i, p, transpose=transpose, weighted=weighted, backend='jax_raw'))
    ir = fn.lower(*args).as_text()
    assert 'stablehlo.case' not in ir and 'stablehlo.if' not in ir


@pytest.mark.usefixtures('enable_x64')
def test_registration_and_direction_lowering():
    assert {'cuda_raw', 'jax_raw', 'cusparse'} <= set(csr_sddmv_p.available_backends('gpu'))
    if jax.default_backend() != 'gpu':
        pytest.skip('inspect CUDA custom calls on a GPU')
    for transpose in [True, False]:
        args = inputs(jnp.float32, jnp.bool_, transpose)
        fn, _ = check_result(args, transpose=transpose, weighted=False)
        ir = str(fn.lower(*args).compiler_ir())
        assert ('hybrid_sddmv_binary' in ir) == transpose
        assert 'csr_sddmv_cusparse' not in ir


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_explicit_jax_backend(transpose, weighted):
    check_result(inputs(jnp.float32, jnp.float32, transpose),
                 transpose=transpose, weighted=weighted, backend='jax_raw')


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
def test_empty_and_repeated_slots(transpose):
    e = jnp.array([2.], dtype=jnp.float32)
    c = jnp.array([3.], dtype=jnp.float32)
    for count in [0, 4]:
        i = jnp.zeros(count, dtype=jnp.int32)
        p = jnp.array([0, count], dtype=jnp.int64)
        check_result((e, c, i, p), transpose=transpose, weighted=True)


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_heavy_rows_and_repeated_calls(value_dtype, weighted):
    i = jnp.arange(5200, dtype=jnp.int32) % 17
    p = jnp.array([0, 0, 5000, 5200], dtype=jnp.int64)
    c = jnp.arange(17, dtype=value_dtype)
    e = jnp.array([0., 2., -3.], dtype=value_dtype)
    fn, _ = check_result((e, c, i, p), transpose=True, weighted=weighted)
    np.testing.assert_array_equal(fn(jnp.zeros_like(e), c, i, p), np.zeros(i.size))


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
def test_vmap_and_portable_backend(transpose):
    e, c, i, p = inputs(jnp.float32, jnp.float32, transpose)
    fn = lambda x, y: csr_sddmv_dweight(x, y, i, p, transpose=transpose, weighted=True)
    result = jax.jit(jax.vmap(fn))(jnp.stack([e, e * 2]), jnp.stack([c, c * 3]))
    expected = reference(e, c, i, p, transpose=transpose, weighted=True)
    np.testing.assert_allclose(result, jnp.stack([expected, expected * 6]))
    check_result((e, c, i, p), transpose=transpose, weighted=True, backend='jax_raw')


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('backend', [None, 'auto', 'cuda_raw', 'cusparse'])
def test_native_event_backends_differentiate_like_jax(transpose, weighted, backend):
    if backend in ('cuda_raw', 'cusparse') and jax.default_backend() != 'gpu':
        pytest.skip('native execution requires a GPU')
    if backend == 'cuda_raw' and not transpose:
        pytest.skip('cuda_raw requires transpose=True')
    e, c, i, p = inputs(jnp.float32, jnp.float32, transpose)

    def loss(x, y, b):
        return jnp.sum(csr_sddmv_dweight(x, y, i, p, transpose=transpose,
                                         weighted=weighted, backend=b) ** 2)

    actual = jax.jit(jax.grad(partial(loss, b=backend), argnums=(0, 1)))(e, c)
    expected = jax.grad(partial(loss, b='jax_raw'), argnums=(0, 1))(e, c)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-6)
    call = lambda y, b: csr_sddmv_dweight(e, y, i, p, transpose=transpose, weighted=weighted, backend=b)
    np.testing.assert_allclose(jax.jvp(partial(call, b=backend), (c,), (c * 2,))[1],
                               jax.jvp(partial(call, b='jax_raw'), (c,), (c * 2,))[1], rtol=1e-6)


@pytest.mark.usefixtures('enable_x64')
def test_validation():
    e, c, i, p = inputs(jnp.float32, jnp.float32, True)
    with pytest.raises(TypeError, match='events'):
        csr_sddmv_dweight(e.astype(jnp.complex64), c, i, p, transpose=True)
    with pytest.raises(ValueError, match='row count'):
        csr_sddmv_dweight(e[:-1], c, i, p, transpose=True)
    with pytest.raises(TypeError, match='indices'):
        csr_sddmv_dweight(e, c, i.astype(jnp.float32), p, transpose=True)


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_nonfinite_events_with_finite_ct(transpose, weighted):
    e = jnp.array([jnp.nan, jnp.inf, -jnp.inf, 0., -2.], dtype=jnp.float32)
    c = jnp.array([0., 0., -2., 3., 5.], dtype=jnp.float32)
    i = jnp.arange(5, dtype=jnp.int32)
    p = jnp.arange(6, dtype=jnp.int32)
    check_result((e, c, i, p), transpose=transpose, weighted=weighted)


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
def test_shard_map_keeps_results_per_shard(transpose):
    from jax.sharding import Mesh, PartitionSpec as P
    e, c, i, p = inputs(jnp.float32, jnp.float32, transpose)
    mesh = Mesh(np.array(jax.devices()[:1]), ('s',))
    fn = jax.shard_map(
        lambda x, y: csr_sddmv_dweight(x, y, i, p, transpose=transpose, weighted=True),
        mesh=mesh, in_specs=(P('s'), P('s')), out_specs=P('s'),
    )
    for values in [c, c.at[0].set(jnp.inf)]:
        result = jax.jit(fn)(e, values)
        expected = reference(e, values, i, p, transpose=transpose, weighted=True)
        np.testing.assert_allclose(result, expected, equal_nan=True)


@pytest.mark.usefixtures('enable_x64')
def test_scheduler_config_is_shared_by_capacity_and_compilation():
    from brainevent._csr.config import CSRConfig
    config = CSRConfig(block_size=128, fixed_scatter_blocks=32, tpr_threshold=16, task_nnz=64)
    e = jnp.array([2., -3.], dtype=jnp.float32)
    c = jnp.arange(31, dtype=jnp.float32)
    i = jnp.arange(279, dtype=jnp.int32) % 31
    p = jnp.array([0, 129, 279], dtype=jnp.int32)
    check_result((e, c, i, p), transpose=True, weighted=True, backend=forward_backend(config))


@pytest.mark.usefixtures('enable_x64')
def test_explicit_scheduler_snapshot_survives_primitive_binding():
    from brainevent._csr.config import CSRConfig
    config = CSRConfig(block_size=128, fixed_scatter_blocks=32, tpr_threshold=16, task_nnz=64)
    args = inputs(jnp.float32, jnp.float32, True)
    closed = jax.make_jaxpr(lambda e, c, i, p: csr_sddmv_dweight(
        e, c, i, p, transpose=True, weighted=True, backend=forward_backend(config)))(*args)
    def walk(jaxpr):  # the primitive sits inside the custom derivative of native calls
        for eqn in jaxpr.eqns:
            yield eqn
            for value in eqn.params.values():
                inner = getattr(value, 'jaxpr', value)
                if hasattr(inner, 'eqns'):
                    yield from walk(inner)

    equation = next(eqn for eqn in walk(closed.jaxpr) if eqn.primitive is csr_sddmv_p.primitive)
    spec = equation.params['config_spec']
    assert spec == config.kernel_args()[0]
    assert spec.mv_capacity(279, 2) > CSRConfig().mv_capacity(279, 2)


@pytest.mark.usefixtures('enable_x64')
def test_module_cache_ignores_matrix_geometry_and_device(monkeypatch, request):
    from brainevent._csr.config import CSRConfig, CSRDeviceInfo
    config = CSRConfig(tpr_threshold=16, task_nnz=64).prepare_structure(
        jnp.array([0, 1], jnp.int32), jnp.array([0, 2], jnp.int32), shape=(1, 2))
    spec = config.kernel_args()[0]
    calls = []
    monkeypatch.setattr(sddmv, 'load_cuda_file', lambda *args, **kwargs: calls.append(kwargs))
    sddmv._load_hybrid_cached.cache_clear()
    request.addfinalizer(sddmv._load_hybrid_cached.cache_clear)
    first = sddmv._load_hybrid(False, spec)
    for size in range(1, 10):
        other = replace(spec, structure=replace(spec.structure, cols=100 + size),
                        device_info=CSRDeviceInfo('gpu', str(size), 1024 * size, size))
        assert sddmv._load_hybrid(False, other) == first
    assert len(calls) == 1
    assert sddmv._load_hybrid_cached.cache_info().currsize == 1
    assert sddmv._load_hybrid(False, replace(spec, task_nnz=128)) != first
    assert len(calls) == 2
    assert sddmv._load_hybrid_cached.cache_parameters()['maxsize'] == 128


@pytest.mark.usefixtures('enable_x64')
def test_zero_sized_matrix_axes():
    for transpose in [False, True]:
        for rows, cols in [(0, 3), (3, 0), (0, 0)]:
            e = jnp.zeros(rows if transpose else cols, dtype=jnp.float32)
            c = jnp.zeros(cols if transpose else rows, dtype=jnp.float32)
            result = csr_sddmv_dweight(e, c, jnp.zeros(0, jnp.int32),
                                       jnp.zeros(rows + 1, jnp.int32), transpose=transpose)
            assert result.shape == (0,)


@pytest.mark.usefixtures('enable_x64')
def test_zero_ct_does_not_multiply_an_overflowed_event_sum():
    # Shared-weight consumers can sum these slots without first overflowing
    # a unit-weight forward reduction (four 1e38 events would sum to Inf).
    e = jnp.full(4, 1e38, dtype=jnp.float32)
    c = jnp.zeros(1, dtype=jnp.float32)
    i = jnp.zeros(4, dtype=jnp.int32)
    p = jnp.arange(5, dtype=jnp.int32)
    _, result = check_result((e, c, i, p), transpose=True, weighted=True)
    assert jnp.sum(result) == 0


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float32, jnp.float64])
def test_jax_product_precision(transpose, value_dtype):
    rng = np.random.default_rng(17)
    e = jnp.asarray(rng.normal(size=127), dtype=value_dtype)
    c = jnp.asarray(rng.normal(size=127), dtype=value_dtype)
    i = jnp.asarray(rng.integers(0, 127, size=127 * 5), dtype=jnp.int32)
    p = jnp.arange(128, dtype=jnp.int32) * 5
    check_result((e, c, i, p), transpose=transpose, weighted=True, backend='jax_raw')


@pytest.mark.usefixtures('enable_x64')
def test_nontransposed_default_needs_no_cuda_loader(monkeypatch):
    def fail_load(*args, **kwargs):
        raise AssertionError('the nontransposed JAX path must not load a CUDA extension')
    monkeypatch.setattr(sddmv, 'load_cuda_file', fail_load)
    args = inputs(jnp.float32, jnp.float32, False)
    fn, _ = check_result(args, transpose=False, weighted=True)
    ir = str(fn.lower(*args).compiler_ir())
    assert 'csr_sddmv_cusparse' not in ir
    assert 'hybrid_sddmv' not in ir


# Native registration, activity and portable execution
# ----------------------------------------------------

@pytest.mark.parametrize('batch', [1])
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


@pytest.mark.parametrize('batch', [1])
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


@pytest.mark.parametrize('batch', [1])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('backend', ['jax_raw', 'auto', 'cuda_raw', 'cusparse'])
@pytest.mark.parametrize('event_dtype,value_dtype,tiny', [
    (jnp.float64, jnp.float32, 1e-200),
    (jnp.float32, jnp.float16, 1e-10),
    (jnp.float64, jnp.bfloat16, 1e-200),
])
def test_weighted_activity_is_checked_after_cast(batch, transpose, backend, event_dtype, value_dtype, tiny):
    gpu = jax.default_backend() == 'gpu'
    if backend in ('cuda_raw', 'cusparse') and not gpu:
        pytest.skip('native execution requires a GPU')
    if backend == 'cuda_raw' and not transpose:
        pytest.skip('cuda_raw requires transpose=True')
    with jax_x64_enabled():
        e = jnp.zeros((3, batch), event_dtype).at[:, 0].set(
            jnp.array([tiny, -tiny, 2.], event_dtype))
        c = jnp.full((3, batch), -jnp.inf, value_dtype).at[:, 0].set(
            jnp.array([jnp.inf, jnp.nan, 3.], value_dtype))
        i, p = jnp.arange(3, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32)
        fn = jax.jit(lambda e, c: csr_sddmm_dweight(
            e, c, i, p, transpose=transpose, weighted=True, backend=backend))
        result = fn(e, c)
        assert result.dtype == c.dtype
        # Every backend agrees: events that narrow to zero in ct.dtype and
        # exact zeros skip nonfinite cotangents; the active third row still
        # contributes its weighted finite value.
        np.testing.assert_array_equal(np.asarray(result, np.float64), [0., 0., 6.])
        ir = fn.lower(e, c).as_text()
        if gpu and transpose and backend in ('auto', 'cuda_raw'):
            target = 'hybrid_sddmv_weighted_' if batch == 1 else 'batched_v3_sddmm_weighted_'
            assert target in ir
            assert 'stablehlo.is_finite' not in ir
