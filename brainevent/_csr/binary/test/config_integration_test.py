"""Configuration snapshots through public CSR calls and JAX transformations."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._csr.config import CSRConfig, CSRDeviceInfo
from brainevent._csr.binary import _hybrid_mm, hybrid_config
from brainevent._csr.binary.binary import binary_csrmv_p_call, binary_csrmm_p_call, binary_csrmm_p
from brainevent._csr.binary.binary_indexed import binary_csrmv_indexed_p_call, binary_csrmm_indexed_p_call
from brainevent._test_util import jax_x64_enabled


@pytest.fixture(autouse=True)
def enable_x64():
    with jax_x64_enabled():
        yield


def _backend():
    return 'cuda_raw' if jax.default_backend() == 'gpu' else 'jax_raw'


def _case(dtype=jnp.float32, *, homo=False):
    lengths = np.array([0, 3, 9, 513], np.int32)
    ptr = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    idx = (np.arange(int(ptr[-1]), dtype=np.int32) * 7) % 9
    indices, indptr = jnp.asarray(idx), jnp.asarray(ptr)
    device = _hybrid_mm.get_device_info(indices, indptr)
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


def _call(case, weights, events, *, indexed=False, weighted=False, config=None,
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
                      weighted=weighted, backend=backend or _backend(), config=config)
    return result if public else result[0]


def _reference(case, weights, events, *, indexed=False, weighted=False, indptr=None):
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


def _equations(value):
    if hasattr(value, 'eqns'):
        for equation in value.eqns:
            yield equation
            yield from _equations(equation.params)
    elif hasattr(value, 'jaxpr'):
        yield from _equations(value.jaxpr)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _equations(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _equations(child)


@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('indexed,homo', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('weighted', [False, True])
def test_public_mm_prepared_config_types_and_long_rows(dtype, indexed, homo, weighted):
    case = _case(dtype, homo=homo)
    function = jax.jit(lambda w, x, config: _call(
        case, w, x, indexed=indexed, weighted=weighted, config=config, public=True))
    for events in (case.events, jnp.zeros_like(case.events)):
        actual = function(case.weights, events, case.config)
        expected = _reference(case, case.weights, events, indexed=indexed, weighted=weighted)
        np.testing.assert_array_equal(np.asarray(actual, np.float64), expected)


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('axis', [0, 1])
def test_mv_vmap_reenters_mm_with_transformed_call_plan(indexed, axis):
    case = _case()
    data = case.events.T if axis == 0 else case.events
    function = jax.vmap(lambda x: _call(case, case.weights, x, indexed=indexed,
                                      weighted=True, config=case.config), in_axes=axis)
    graph_function = jax.vmap(lambda x: _call(
        case, case.weights, x, indexed=indexed, weighted=True,
        config=case.config, backend='cuda_raw'), in_axes=axis)
    graph = jax.make_jaxpr(graph_function)(data)
    plans = [eqn.params['call_plan'] for eqn in _equations(graph)
             if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert len(plans) == 1 and plans[0].batch == case.events.shape[1]
    assert plans[0] == case.config.mm_plan(3, case.weights.dtype, True)
    actual = jax.jit(function)(data)
    np.testing.assert_array_equal(actual, _reference(case, case.weights, case.events,
                                                    indexed=indexed, weighted=True).T)


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_mm_vmap_rederives_plan_for_folded_columns(indexed, axis):
    case = _case()
    original = jnp.stack([case.events, case.events * 2])
    data = jnp.moveaxis(original, 0, axis)
    function = jax.vmap(lambda x: _call(case, case.weights, x, indexed=indexed,
                                      weighted=True, config=case.config), in_axes=axis)
    graph_function = jax.vmap(lambda x: _call(
        case, case.weights, x, indexed=indexed, weighted=True,
        config=case.config, backend='cuda_raw'), in_axes=axis)
    graph = jax.make_jaxpr(graph_function)(data)
    plans = [eqn.params['call_plan'] for eqn in _equations(graph)
             if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert len(plans) == 1 and plans[0] == case.config.mm_plan(6, jnp.float32, True)
    assert case.config.mm_plan(3, jnp.float32, True).batch == 3
    expected = np.stack([_reference(case, case.weights, x, indexed=indexed, weighted=True) for x in original])
    np.testing.assert_array_equal(jax.jit(function)(data), expected)


@pytest.mark.parametrize('indexed', [False, True])
def test_weights_only_vmap_reuses_prepared_structure_in_scan(indexed):
    case = _case()
    weights = jnp.stack([case.weights, case.weights * 2])
    function = jax.vmap(lambda w: _call(case, w, case.events, indexed=indexed,
                                      weighted=True, config=case.config))
    graph_function = jax.vmap(lambda w: _call(
        case, w, case.events, indexed=indexed, weighted=True,
        config=case.config, backend='cuda_raw'))
    graph = jax.make_jaxpr(graph_function)(weights)
    assert any(eqn.primitive.name == 'scan' for eqn in _equations(graph))
    plans = [eqn.params['call_plan'] for eqn in _equations(graph)
             if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert plans and all(plan == case.config.mm_plan(3, jnp.float32, True) for plan in plans)
    expected = np.stack([_reference(case, w, case.events, indexed=indexed, weighted=True) for w in weights])
    np.testing.assert_array_equal(jax.jit(function)(weights), expected)


@pytest.mark.parametrize('indexed,homo', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('mm', [False, True])
def test_prepared_config_jvp_and_vjp(indexed, homo, weighted, mm):
    case = _case(homo=homo)
    events = case.events if mm else case.events[:, 0]
    function = lambda w, x: _call(case, w, x, indexed=indexed, weighted=weighted, config=case.config)
    dweights, devents = jnp.ones_like(case.weights) / 16, jnp.ones_like(events) / 4
    actual, tangent = jax.jit(lambda w, x: jax.jvp(function, (w, x), (dweights, devents)))(case.weights, events)
    expected = _reference(case, case.weights, events, indexed=indexed, weighted=weighted)
    expected_tangent = (_reference(case, dweights, events, indexed=indexed, weighted=weighted)
                        + _reference(case, case.weights, devents, indexed=indexed, weighted=True))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-6, atol=2e-6)

    cotangent = jnp.asarray(np.resize([-1., .5, 2.], actual.shape), jnp.float32)
    def gradients(w, x):
        _, pullback = jax.vjp(function, w, x)
        return pullback(cotangent)
    grad_weights, grad_events = jax.jit(gradients)(case.weights, events)
    rows = np.repeat(np.arange(4), np.diff(np.asarray(case.indptr)))
    factors = np.asarray(events)[rows] if weighted else np.asarray(events)[rows] != 0
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


def test_config_is_safe_as_scan_carry_and_in_independent_jits():
    case = _case()
    original_spec = case.config.kernel_args()[0]
    def scan(config, events):
        def body(carry, event):
            weights, snapshot = carry
            value = _call(case, weights, event, weighted=True, config=snapshot)
            return (weights * 2, snapshot), value
        return jax.lax.scan(body, (case.weights, config), events)
    data = jnp.stack([case.events, case.events * 2])
    seven_columns = jnp.concatenate([case.events, case.events, case.events[:, :1]], axis=1)
    with jax.checking_leaks():
        (weights, config), values = jax.jit(scan)(case.config, data)
        fn1 = jax.jit(lambda cfg, x: _call(case, case.weights, x, weighted=True, config=cfg))
        fn2 = jax.jit(lambda cfg, x: _call(case, case.weights, x, weighted=True, config=cfg))
        first = fn1(case.config, case.events[:, :2])
        second = fn2(case.config, seven_columns)
        jax.block_until_ready((values, first, second))
    assert case.config.kernel_args()[0] == config.kernel_args()[0] == original_spec
    np.testing.assert_array_equal(weights, case.weights * 4)
    np.testing.assert_array_equal(values, np.stack([
        _reference(case, case.weights, data[0], weighted=True),
        _reference(case, case.weights * 2, data[1], weighted=True),
    ]))
    np.testing.assert_array_equal(first, _reference(case, case.weights, case.events[:, :2], weighted=True))
    np.testing.assert_array_equal(second, _reference(case, case.weights, seven_columns, weighted=True))


def test_raw_calls_do_not_load_or_prepare_implicit_config(monkeypatch):
    case = _case()
    def forbidden(*args, **kwargs):
        raise AssertionError('raw operators must not load persistence or prepare structure')
    monkeypatch.setattr(hybrid_config, 'load_csr_config', forbidden)
    monkeypatch.setattr(CSRConfig, 'prepare_structure', forbidden)
    # Native MV uses constant defaults. Portable MM needs no prepared plan.
    mv = jax.jit(lambda x: _call(case, case.weights, x, config=None))(case.events[:, 0])
    mm = jax.jit(lambda x: _call(case, case.weights, x, config=None, backend='jax_raw'))(case.events)
    np.testing.assert_array_equal(mv, _reference(case, case.weights, case.events[:, 0]))
    np.testing.assert_array_equal(mm, _reference(case, case.weights, case.events))


def test_auto_hybrid_guard_does_not_block_native_gather():
    case = _case()
    function = jax.jit(lambda x: binary_csrmm_p_call(
        case.weights, case.indices, case.indptr, x, shape=case.shape,
        transpose=True, backend=None, config=None)[0])
    if jax.default_backend() == 'gpu':
        with pytest.raises(brainevent.KernelCompilationError, match='prepared CSRConfig'):
            function(case.events)
    else:
        np.testing.assert_array_equal(function(case.events), _reference(case, case.weights, case.events))
    x = jnp.asarray(np.resize([-2., 0., 1.], (case.shape[1], 2)), jnp.float32)
    gather = jax.jit(lambda value: binary_csrmm_p_call(
        case.weights, case.indices, case.indptr, value, shape=case.shape,
        transpose=False, backend=_backend(), weighted=True, config=None)[0])
    rows = np.repeat(np.arange(case.shape[0]), np.diff(np.asarray(case.indptr)))
    dense = np.zeros(case.shape, np.float32)
    np.add.at(dense, (rows, np.asarray(case.indices)), np.asarray(case.weights))
    np.testing.assert_array_equal(gather(x), dense @ np.asarray(x))


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('backend,transpose,dtype,batch', [
    ('jax_raw', True, jnp.float8_e4m3fn, 3),
    ('jax_raw', False, jnp.float8_e4m3fn, 3),
    ('jax_raw', True, jnp.float32, 2**31),
    ('cuda_raw', False, jnp.float32, 2**31),
])
def test_nonhybrid_pcall_does_not_apply_hybrid_plan_limits(monkeypatch, indexed, backend, transpose, dtype, batch):
    """Abstract tracing needs no allocation and must not apply an unrelated plan's limits."""
    case = _case(dtype)
    def forbidden(*args, **kwargs):
        raise AssertionError('nonhybrid MM must not derive a hybrid call plan')
    monkeypatch.setattr(CSRConfig, 'mm_plan', forbidden)
    function = binary_csrmm_indexed_p_call if indexed else binary_csrmm_p_call
    operands = (case.weights, case.indices, case.indptr) + ((case.perm,) if indexed else ())
    rows = case.shape[0] if transpose else case.shape[1]
    graph = jax.make_jaxpr(lambda x: function(
        *operands, x, shape=case.shape, transpose=transpose, backend=backend,
        config=case.config)[0])(jax.ShapeDtypeStruct((rows, batch), dtype))
    equations = [eqn for eqn in _equations(graph)
                 if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert len(equations) == 1
    assert equations[0].params['call_plan'] is None
    assert graph.out_avals[0].dtype == jnp.dtype(dtype)
    assert graph.out_avals[0].shape == (case.shape[1] if transpose else case.shape[0], batch)


@pytest.mark.parametrize('indexed', [False, True])
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


@pytest.mark.parametrize('indexed', [False, True])
def test_structural_vmap_invalidates_exact_plan(indexed):
    case = _case()
    other = jnp.array([0, 0, 300, 400, case.indices.size], jnp.int32)
    pointers = jnp.stack([case.indptr, other])
    function = jax.vmap(lambda ptr: _call(
        case, case.weights, case.events, indexed=indexed, weighted=True,
        config=case.config, indptr=ptr, backend='jax_raw'))
    graph = jax.make_jaxpr(function)(pointers)
    equations = [eqn for eqn in _equations(graph)
                 if eqn.primitive.name in ('binary_csrmm', 'binary_csrmm_indexed')]
    assert equations and all(eqn.params['call_plan'] is None for eqn in equations)
    assert all(eqn.params['config_spec'].structure is None for eqn in equations)
    expected = np.stack([_reference(case, case.weights, case.events, indexed=indexed,
                                    weighted=True, indptr=ptr) for ptr in pointers])
    np.testing.assert_array_equal(jax.jit(function)(pointers), expected)
    if jax.default_backend() == 'gpu':
        native = jax.vmap(lambda ptr: _call(
            case, case.weights, case.events, indexed=indexed, config=case.config, indptr=ptr))
        with pytest.raises(brainevent.KernelCompilationError, match='prepared CSRConfig'):
            jax.jit(native)(pointers)


@pytest.mark.parametrize('indexed', [False, True])
def test_native_unprepared_mm_and_mv_vmap_fail_at_hybrid_lowering(indexed):
    if jax.default_backend() != 'gpu':
        pytest.skip('actual hybrid lowering requires GPU')
    case = _case()
    for config in (None, CSRConfig(device_info=case.config.device_info),
                   replace(case.config, device_info=None)):
        function = lambda x: _call(case, case.weights, x, indexed=indexed, config=config)
        with pytest.raises(brainevent.KernelCompilationError, match='prepared CSRConfig'):
            jax.jit(function)(case.events)
        with pytest.raises(brainevent.KernelCompilationError, match='prepared CSRConfig'):
            jax.jit(jax.vmap(function))(case.events.T)


def test_direct_native_primitive_rejects_tampered_call_plan():
    if jax.default_backend() != 'gpu':
        pytest.skip('actual hybrid lowering requires GPU')
    case = _case()
    graph = jax.make_jaxpr(lambda w, x: _call(case, w, x, config=case.config))(case.weights, case.events)
    eqn = next(e for e in _equations(graph) if e.primitive is binary_csrmm_p.primitive)
    params = dict(eqn.params)
    params['call_plan'] = replace(params['call_plan'], workspace_words=params['call_plan'].workspace_words + 1)
    mask = case.config.kernel_args()[1]
    function = jax.jit(lambda w, x: binary_csrmm_p(w, case.indices, case.indptr, x, mask, **params)[0])
    with pytest.raises(brainevent.KernelCompilationError, match='prepared config and invocation'):
        function(case.weights, case.events)


@pytest.mark.parametrize('empty', ['nnz', 'batch'])
def test_empty_mm_bypasses_preparation_and_cuda_loader(monkeypatch, empty):
    case = _case()
    if empty == 'nnz':
        case.indices = jnp.empty((0,), jnp.int32)
        case.indptr = jnp.zeros((case.shape[0] + 1,), jnp.int32)
        case.weights = jnp.ones((1,), jnp.float32)
    else:
        case.events = case.events[:, :0]
    def forbidden(*args, **kwargs):
        raise AssertionError('empty hybrid MM must not load CUDA sources')
    monkeypatch.setattr(_hybrid_mm, 'load_module', forbidden)
    actual = jax.jit(lambda x: _call(case, case.weights, x, config=None))(case.events)
    np.testing.assert_array_equal(actual, np.zeros((case.shape[1], case.events.shape[1])))
