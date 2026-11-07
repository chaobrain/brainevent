"""Exercise CSR sampled gradients, including CUDA/JAX direction dispatch."""

import importlib
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._sddmm import csr_sddmv_dweight, csr_sddmv_p

sddmv = importlib.import_module('brainevent._sddmm._mv.sddmv')


@pytest.fixture(autouse=True)
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
    value = x.astype(ct.dtype) * ct[out] if weighted else ct[out]
    return jnp.where(x != 0, value, jnp.zeros_like(value))


def check_result(args, *, transpose, weighted, backend=None, config=None):
    fn = jax.jit(lambda e, c, i, p: csr_sddmv_dweight(
        e, c, i, p, transpose=transpose, weighted=weighted, backend=backend, config=config))
    actual = fn(*args)
    expected = reference(*args, transpose=transpose, weighted=weighted)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64),
                               np.asarray(expected, dtype=np.float64), rtol=2e-6, atol=0,
                               equal_nan=True)
    return fn, actual


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('event_dtype', [jnp.bool_, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                                        jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64,
                                        jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_value_event_types(value_dtype, event_dtype, transpose, weighted):
    check_result(inputs(value_dtype, event_dtype, transpose),
                 transpose=transpose, weighted=weighted)


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_nonfinite_cotangents(value_dtype, transpose, weighted):
    e, c, i, p = inputs(value_dtype, value_dtype, transpose)
    c = jnp.array([jnp.inf, jnp.nan, -jnp.inf, jnp.nan][:c.size], dtype=value_dtype)
    check_result((e, c, i, p), transpose=transpose, weighted=weighted)


@pytest.mark.parametrize('transpose', [False, True])
def test_jax_activity_is_checked_before_narrowing(transpose):
    e = jnp.array([1e-200, 0, -1e-200], dtype=jnp.float64)
    c = jnp.full(3, jnp.inf, dtype=jnp.float32)
    i = jnp.arange(3, dtype=jnp.int32)
    p = jnp.arange(4, dtype=jnp.int32)
    _, result = check_result((e, c, i, p), transpose=transpose, weighted=True, backend='jax_raw')
    assert jnp.isnan(result[0]) and result[1] == 0 and jnp.isnan(result[2])


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


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_explicit_jax_backend(transpose, weighted):
    check_result(inputs(jnp.float32, jnp.float32, transpose),
                 transpose=transpose, weighted=weighted, backend='jax_raw')


@pytest.mark.parametrize('transpose', [False, True])
def test_empty_and_repeated_slots(transpose):
    e = jnp.array([2.], dtype=jnp.float32)
    c = jnp.array([3.], dtype=jnp.float32)
    for count in [0, 4]:
        i = jnp.zeros(count, dtype=jnp.int32)
        p = jnp.array([0, count], dtype=jnp.int64)
        check_result((e, c, i, p), transpose=transpose, weighted=True)


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_heavy_rows_and_repeated_calls(value_dtype, weighted):
    i = jnp.arange(5200, dtype=jnp.int32) % 17
    p = jnp.array([0, 0, 5000, 5200], dtype=jnp.int64)
    c = jnp.arange(17, dtype=value_dtype)
    e = jnp.array([0., 2., -3.], dtype=value_dtype)
    fn, _ = check_result((e, c, i, p), transpose=True, weighted=weighted)
    np.testing.assert_array_equal(fn(jnp.zeros_like(e), c, i, p), np.zeros(i.size))


@pytest.mark.parametrize('transpose', [False, True])
def test_vmap_and_portable_backend(transpose):
    e, c, i, p = inputs(jnp.float32, jnp.float32, transpose)
    fn = lambda x, y: csr_sddmv_dweight(x, y, i, p, transpose=transpose, weighted=True)
    result = jax.jit(jax.vmap(fn))(jnp.stack([e, e * 2]), jnp.stack([c, c * 3]))
    expected = reference(e, c, i, p, transpose=transpose, weighted=True)
    np.testing.assert_allclose(result, jnp.stack([expected, expected * 6]))
    check_result((e, c, i, p), transpose=transpose, weighted=True, backend='jax_raw')


def test_sampled_operator_has_no_derivative_rule():
    e, c, i, p = inputs(jnp.float32, jnp.float32, True)
    with pytest.raises(NotImplementedError, match='Differentiation rule'):
        jax.jvp(lambda x: csr_sddmv_dweight(e, x, i, p, transpose=True), (c,), (c,))


def test_validation():
    e, c, i, p = inputs(jnp.float32, jnp.float32, True)
    with pytest.raises(TypeError, match='events'):
        csr_sddmv_dweight(e.astype(jnp.complex64), c, i, p, transpose=True)
    with pytest.raises(ValueError, match='row count'):
        csr_sddmv_dweight(e[:-1], c, i, p, transpose=True)
    with pytest.raises(TypeError, match='indices'):
        csr_sddmv_dweight(e, c, i.astype(jnp.float32), p, transpose=True)


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_nonfinite_events_with_finite_ct(transpose, weighted):
    e = jnp.array([jnp.nan, jnp.inf, -jnp.inf, 0., -2.], dtype=jnp.float32)
    c = jnp.array([0., 0., -2., 3., 5.], dtype=jnp.float32)
    i = jnp.arange(5, dtype=jnp.int32)
    p = jnp.arange(6, dtype=jnp.int32)
    check_result((e, c, i, p), transpose=transpose, weighted=weighted)


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


def test_scheduler_config_is_shared_by_capacity_and_compilation():
    from brainevent._csr.config import CSRConfig
    config = CSRConfig(block_size=128, fixed_scatter_blocks=32, tpr_threshold=16, task_nnz=64)
    e = jnp.array([2., -3.], dtype=jnp.float32)
    c = jnp.arange(31, dtype=jnp.float32)
    i = jnp.arange(279, dtype=jnp.int32) % 31
    p = jnp.array([0, 129, 279], dtype=jnp.int32)
    check_result((e, c, i, p), transpose=True, weighted=True, config=config)


def test_explicit_scheduler_snapshot_survives_primitive_binding():
    from brainevent._csr.config import CSRConfig
    config = CSRConfig(block_size=128, fixed_scatter_blocks=32, tpr_threshold=16, task_nnz=64)
    args = inputs(jnp.float32, jnp.float32, True)
    closed = jax.make_jaxpr(lambda e, c, i, p: csr_sddmv_dweight(
        e, c, i, p, transpose=True, weighted=True, config=config))(*args)
    equation = next(eqn for eqn in closed.jaxpr.eqns if eqn.primitive is csr_sddmv_p.primitive)
    spec = equation.params['config_spec']
    assert spec == config.kernel_args()[0]
    assert spec.mv_capacity(279, 2) > CSRConfig().mv_capacity(279, 2)


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


def test_zero_sized_matrix_axes():
    for transpose in [False, True]:
        for rows, cols in [(0, 3), (3, 0), (0, 0)]:
            e = jnp.zeros(rows if transpose else cols, dtype=jnp.float32)
            c = jnp.zeros(cols if transpose else rows, dtype=jnp.float32)
            result = csr_sddmv_dweight(e, c, jnp.zeros(0, jnp.int32),
                                       jnp.zeros(rows + 1, jnp.int32), transpose=transpose)
            assert result.shape == (0,)


def test_zero_ct_does_not_multiply_an_overflowed_event_sum():
    # Shared-weight consumers can sum these slots without first overflowing
    # a unit-weight forward reduction (four 1e38 events would sum to Inf).
    e = jnp.full(4, 1e38, dtype=jnp.float32)
    c = jnp.zeros(1, dtype=jnp.float32)
    i = jnp.zeros(4, dtype=jnp.int32)
    p = jnp.arange(5, dtype=jnp.int32)
    _, result = check_result((e, c, i, p), transpose=True, weighted=True)
    assert jnp.sum(result) == 0


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float32, jnp.float64])
def test_jax_product_precision(transpose, value_dtype):
    rng = np.random.default_rng(17)
    e = jnp.asarray(rng.normal(size=127), dtype=value_dtype)
    c = jnp.asarray(rng.normal(size=127), dtype=value_dtype)
    i = jnp.asarray(rng.integers(0, 127, size=127 * 5), dtype=jnp.int32)
    p = jnp.arange(128, dtype=jnp.int32) * 5
    check_result((e, c, i, p), transpose=transpose, weighted=True, backend='jax_raw')


def test_nontransposed_default_needs_no_cuda_loader(monkeypatch):
    def fail_load(*args, **kwargs):
        raise AssertionError('the nontransposed JAX path must not load a CUDA extension')
    monkeypatch.setattr(sddmv, 'load_cuda_file', fail_load)
    args = inputs(jnp.float32, jnp.float32, False)
    fn, _ = check_result(args, transpose=False, weighted=True)
    ir = str(fn.lower(*args).compiler_ir())
    assert 'csr_sddmv_cusparse' not in ir
    assert 'hybrid_sddmv' not in ir
