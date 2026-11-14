# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Contracts for the combined TCSR MM primitive and its shared CSR derivatives.

CPU cases install temporary mathematical lowerings only inside this module's
fixture. GPU cases always execute the production CUDA lowering.
"""

from functools import partial
import importlib
from pathlib import Path

import brainunit as u
import brainevent
import jax
import jax.numpy as jnp
from jax.interpreters import ad
import numpy as np
import pytest

from brainevent._csr.config import CSRConfig
from brainevent._test_util import jax_x64_enabled, requires_gpu


api = importlib.import_module('brainevent._tcsr.binary.binary')
planner = importlib.import_module('brainevent._tcsr.binary.combined.planner')
WEIGHT_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)


def _reference_kernel(*, shape, transpose, **kwargs):
    """Use canonical coordinates, independent of native tile traversal."""
    def kernel(weights, indices, indptr, events, local, offsets, perm):
        rows = jnp.repeat(jnp.arange(shape[0], dtype=indices.dtype),
                          jnp.diff(indptr), total_repeat_length=indices.size)
        unit, target = (rows, indices) if transpose else (indices, rows)
        dtype = jnp.float64 if weights.dtype == jnp.float64 else jnp.float32
        values = jnp.broadcast_to(weights, (indices.size,)).astype(dtype)
        selected = jnp.where(events[unit] != 0, values[:, None], 0)
        size = shape[1] if transpose else shape[0]
        result = jnp.zeros((size, events.shape[1]), dtype).at[target].add(selected)
        return (result.astype(weights.dtype),)
    return kernel


@pytest.fixture(scope='module')
def cpu_math_backend():
    """Exercise real transformations with temporary CPU mathematical leaves."""
    float_api = importlib.import_module('brainevent._csr.float.float')
    sampled_mm = importlib.import_module('brainevent._sddmm._mm.sddmm')
    sampled_mv = importlib.import_module('brainevent._sddmm._mv.sddmv')
    aliases = (
        (api.binary_csrmm_p, _reference_kernel),
        (float_api.csrmm_p, float_api._csrmm_jax_kernel),
        (sampled_mm.csr_sddmm_p,
         lambda **kw: partial(sampled_mm._jax_kernel, **kw)),
        (sampled_mv.csr_sddmv_p,
         lambda **kw: partial(sampled_mv._jax_kernel, **kw)),
    )
    with pytest.MonkeyPatch.context() as patch:
        for primitive, generator in aliases:
            patch.setattr(primitive, '_kernels',
                          {key: dict(value) for key, value in primitive._kernels.items()})
            patch.setattr(primitive, '_defaults', dict(primitive._defaults))
            patch.setattr(primitive, '_registered_platforms', set(primitive._registered_platforms))
            primitive.def_kernel('cuda_raw', 'cpu', generator)
        with jax.default_device(jax.devices('cpu')[0]):
            yield
    # apply_primitive caches executables as well as lowerings. Do not allow a
    # compiled temporary CPU implementation to escape this fixture.
    jax.clear_caches()


def _make_case(transpose=True, homo=False, dtype=jnp.float32,
               event_dtype=jnp.float32, offset_dtype=np.int32, batch=3,
               shape=(3, 5), indptr=None, indices=None):
    ptr = np.asarray([0, 3, 3, 6] if indptr is None else indptr, dtype=offset_dtype)
    cols = np.asarray([0, 2, 2, 1, 3, 4] if indices is None else indices, dtype=np.int32)
    rows = np.repeat(np.arange(shape[0]), np.diff(ptr))
    permutation = np.arange(cols.size)
    if transpose:
        traversal_ptr, targets = ptr, cols
        source, output = shape
    else:
        permutation = np.argsort(cols, kind='stable')
        targets = rows[permutation]
        traversal_ptr = np.concatenate(([0], np.cumsum(np.bincount(cols, minlength=shape[1]))))
        source, output = shape[::-1]
    width = planner.tile_schedule(output, batch, 8 if dtype == jnp.float64 else 4).tile_width
    bounds = np.arange((output + width - 1) // width + 1, dtype=np.int64) * width
    offsets = np.empty((source, bounds.size), dtype=offset_dtype)
    for row in range(source):
        begin, end = map(int, traversal_ptr[row:row + 2])
        offsets[row] = begin + np.searchsorted(targets[begin:end], bounds)
    values = np.resize(np.array([1.25, -2, .5, 3, -.75, 2], np.float64), cols.size)
    weights = jnp.asarray([1.25] if homo else values, dtype)
    events = np.resize(np.array([0, -2, .5, 1, 0, -1], np.float32), (source, batch))
    return dict(weights=weights, indices=jnp.asarray(cols), indptr=jnp.asarray(ptr),
                events=jnp.asarray(events if event_dtype != jnp.bool_ else events != 0, event_dtype),
                local_targets=jnp.asarray(targets % width, jnp.uint16),
                tile_offsets=jnp.asarray(offsets.ravel()),
                perm=jnp.asarray(permutation, offset_dtype) if not transpose and not homo else None,
                tile_width=width, shape=shape, transpose=transpose)


def _invoke(case, *, _lowlevel=False, **changes):
    args = dict(case, **changes)
    weights, indices, indptr, events = (args.pop(key) for key in
                                      ('weights', 'indices', 'indptr', 'events'))
    if _lowlevel:
        return api.binary_csrmm_p_call(weights, indices, indptr, events, **args)[0]
    return api.binary_csrmm(weights, indices, indptr, events, **args)


def _numpy_reference(case, events=None, weights=None, binary=True):
    events = np.asarray(case['events'] if events is None else events)
    weights = np.asarray(case['weights'] if weights is None else weights)
    ptr, cols = np.asarray(case['indptr']), np.asarray(case['indices'])
    dtype = np.float64 if weights.dtype == np.float64 else np.float32
    output = np.zeros((case['shape'][1 if case['transpose'] else 0], events.shape[1]), dtype)
    for row in range(case['shape'][0]):
        for edge in range(ptr[row], ptr[row + 1]):
            unit, target = (row, cols[edge]) if case['transpose'] else (cols[edge], row)
            value = weights[0] if weights.size == 1 else weights[edge]
            for batch in range(events.shape[1]):
                if binary:
                    if events[unit, batch] != 0:
                        output[target, batch] += value
                else:
                    output[target, batch] += value * events[unit, batch]
    return output


def _equations(jaxpr):
    """Every equation of a jaxpr, including those of nested sub-jaxprs."""
    for eqn in jaxpr.eqns:
        yield eqn
        for value in eqn.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(item, 'jaxpr', item)
                if hasattr(inner, 'eqns'):
                    yield from _equations(inner)


def _assert_close(actual, expected):
    dtype = jnp.asarray(actual).dtype
    tolerance = .025 if dtype == jnp.bfloat16 else (.003 if dtype == jnp.float16 else 2e-5)
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64),
                               np.asarray(expected, dtype=np.float64), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('dtype', WEIGHT_DTYPES)
@pytest.mark.parametrize('event_dtype', [jnp.bool_, jnp.float32])
def test_cpu_primitive_dtype_direction_and_shared_weight(cpu_math_backend, transpose, homo,
                                                       dtype, event_dtype):
    with jax_x64_enabled():
        case = _make_case(transpose, homo, dtype, event_dtype)
        result = jax.jit(lambda weights: _invoke(case, weights=weights))(case['weights'])
        assert result.dtype == dtype
        _assert_close(result, _numpy_reference(case))


@requires_gpu
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('dtype', WEIGHT_DTYPES)
@pytest.mark.parametrize('event_dtype', [jnp.bool_, jnp.float32])
@pytest.mark.parametrize('offset_dtype', [np.int32, np.int64])
def test_cuda_dtype_direction_offset_and_shared_weight(transpose, homo, dtype, event_dtype,
                                                      offset_dtype):
    with jax_x64_enabled(), jax.default_device(jax.devices('gpu')[0]):
        case = _make_case(transpose, homo, dtype, event_dtype, offset_dtype)
        result = jax.jit(lambda weights, events: _invoke(case, weights=weights, events=events))(
            case['weights'], case['events'])
        _assert_close(result, _numpy_reference(case))
        assert result.dtype == dtype


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [False, True])
def test_jvp_keeps_numeric_input_tangents(cpu_math_backend, transpose, homo):
    case = _make_case(transpose, homo)
    dw = jnp.arange(case['weights'].size, dtype=jnp.float32) + .25
    de = jnp.arange(case['events'].size, dtype=jnp.float32).reshape(case['events'].shape) / 7 - 1
    fn = lambda w, e: _invoke(case, weights=w, events=e)
    _, tangent = jax.jvp(fn, (case['weights'], case['events']), (dw, de))
    expected = _numpy_reference(case, weights=dw) + _numpy_reference(case, events=de, binary=False)
    _assert_close(tangent, expected)
    # The weight tangent must retain the tiled primitive rather than switch to
    # the hybrid CSR forward, which would ignore the tile operands.
    graph = jax.make_jaxpr(lambda w, d: jax.jvp(
        lambda x: _invoke(case, weights=x, _lowlevel=True), (w,), (d,))[1])(case['weights'], dw)
    assert any(eqn.primitive is api.binary_csrmm_p.primitive for eqn in graph.jaxpr.eqns)


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [False, True])
def test_vjp_returns_canonical_edge_order_and_homo_sum(cpu_math_backend, transpose, homo):
    case = _make_case(transpose, homo)
    shape = (case['shape'][1 if transpose else 0], case['events'].shape[1])
    ct = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 5 + .2
    loss = lambda w, e: jnp.sum(_invoke(case, weights=w, events=e) * ct)
    dw, de = jax.grad(loss, argnums=(0, 1))(case['weights'], case['events'])
    rows = np.repeat(np.arange(case['shape'][0]), np.diff(np.asarray(case['indptr'])))
    cols = np.asarray(case['indices'])
    unit, target = (rows, cols) if transpose else (cols, rows)
    events = np.asarray(case['events'])[unit]
    # Binary events: the weight gradient is the exact activity-mask derivative.
    expected = np.sum(np.where(events != 0, np.asarray(ct)[target], 0), axis=1)
    _assert_close(dw, expected.sum(keepdims=True) if homo else expected)
    backward_case = dict(case, transpose=not transpose)
    _assert_close(de, _numpy_reference(backward_case, events=ct, binary=False))


@pytest.mark.parametrize('transpose', [True, False])
def test_shared_weight_gradient_sums_slots_without_a_csr_forward(transpose):
    csr_binary = importlib.import_module('brainevent._csr.binary.binary')
    case = _make_case(transpose, homo=True)
    graph = jax.make_jaxpr(jax.grad(lambda w: jnp.sum(_invoke(case, weights=w))))(case['weights'])
    primitives = {eqn.primitive for eqn in _equations(graph.jaxpr)}
    # The CSR forward has no tile operands, so the tiled transpose rule must
    # not rerun it; the shared cotangent sums the per-slot cotangents instead.
    assert api.binary_csrmm_p.primitive in primitives
    assert csr_binary.binary_csrmm_p.primitive not in primitives


@pytest.mark.parametrize('transpose', [True, False])
def test_config_reaches_forward_tangent_and_batched_calls_as_mv_schedule(transpose):
    case = _make_case(transpose)
    config = CSRConfig(tpr_threshold=5)

    def specs(fn, *args):
        graph = jax.make_jaxpr(fn)(*args)
        return [eqn.params['config_spec'] for eqn in _equations(graph.jaxpr)
                if eqn.primitive is api.binary_csrmm_p.primitive]

    assert specs(lambda w: _invoke(case, weights=w), case['weights']) == [None]
    run = lambda w: _invoke(case, weights=w, config=config)
    assert specs(run, case['weights']) == [config.mv]
    tangent = jnp.ones_like(case['weights'])
    assert specs(lambda w, d: jax.jvp(run, (w,), (d,)), case['weights'], tangent) == [config.mv] * 2
    # Batching over structures keeps the schedule, which holds no structure state.
    keys = ('indices', 'indptr', 'tile_offsets')
    stacked = [jnp.stack([case[key]] * 2) for key in keys]
    batched = jax.vmap(lambda *arrays: _invoke(case, config=config, **dict(zip(keys, arrays))))
    assert specs(batched, *stacked) == [config.mv]


@pytest.mark.parametrize('transpose', [True, False])
def test_higher_weight_derivatives_match_selected_linear_map(cpu_math_backend, transpose):
    case = _make_case(transpose)
    basis = np.eye(case['weights'].size, dtype=np.float32)
    columns = np.stack([_numpy_reference(case, weights=column).ravel() for column in basis], axis=1)
    loss = lambda w: jnp.sum(_invoke(case, weights=w) ** 2)
    hessian = jax.jacfwd(jax.grad(loss))(case['weights'])
    _assert_close(hessian, 2 * columns.T @ columns)


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('undefined_operand', [0, 3])
def test_zero_cotangent_preserves_seven_operand_contract(transpose, undefined_operand):
    case = _make_case(transpose)
    perm = case['perm'] if case['perm'] is not None else jnp.empty(0, jnp.int32)
    operands = [case[key] for key in ('weights', 'indices', 'indptr', 'events',
                                     'local_targets', 'tile_offsets')] + [perm]
    value = operands[undefined_operand]
    expected_aval = jax.core.ShapedArray(value.shape, value.dtype)
    operands[undefined_operand] = ad.UndefinedPrimal(expected_aval)
    shape = (case['shape'][1 if transpose else 0], case['events'].shape[1])
    zero = ad.Zero(jax.core.ShapedArray(shape, case['weights'].dtype))
    result = api._csrmm_transpose_rule([zero], *operands, shape=case['shape'], transpose=transpose)
    assert len(result) == 7
    assert isinstance(result[undefined_operand], ad.Zero)
    assert result[undefined_operand].aval == expected_aval
    assert result[4:] == (None, None, None)


def test_bool_event_gradient_is_float0(cpu_math_backend):
    case = _make_case(event_dtype=jnp.bool_)
    result = jax.grad(lambda events: jnp.sum(_invoke(case, events=events)), allow_int=True)(case['events'])
    assert result.shape == case['events'].shape
    assert result.dtype == jax.dtypes.float0


@requires_gpu
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('dtype', WEIGHT_DTYPES)
def test_cuda_gradients_and_numeric_input_tangent(transpose, homo, dtype):
    with jax_x64_enabled(), jax.default_device(jax.devices('gpu')[0]):
        case = _make_case(transpose, homo, dtype=dtype)
        fn = lambda w, e: _invoke(case, weights=w, events=e)
        dw, de = jax.jit(jax.grad(lambda w, e: jnp.sum(fn(w, e)), argnums=(0, 1)))(
            case['weights'], case['events'])
        assert dw.dtype == case['weights'].dtype
        assert de.dtype == case['events'].dtype
        rows = np.repeat(np.arange(case['shape'][0]), np.diff(np.asarray(case['indptr'])))
        unit = rows if transpose else np.asarray(case['indices'])
        selected = np.asarray(case['events'])[unit]
        expected_dw = (selected != 0).sum(axis=1)
        _assert_close(dw, expected_dw.sum(keepdims=True) if homo else expected_dw)
        output_shape = (case['shape'][1 if transpose else 0], case['events'].shape[1])
        _assert_close(de, _numpy_reference(dict(case, transpose=not transpose),
                                           events=np.ones(output_shape, np.float32), binary=False))
        tangent = jnp.arange(case['events'].size, dtype=jnp.float32).reshape(case['events'].shape) / 9 - .4
        _, derivative = jax.jvp(lambda e: fn(case['weights'], e), (case['events'],), (tangent,))
        _assert_close(derivative, _numpy_reference(case, events=tangent.astype(dtype), binary=False))


@requires_gpu
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('config', [None, CSRConfig(), CSRConfig(tpr_threshold=5)],
                         ids=['none', 'default', 'tuned'])
def test_cuda_batch_one_weight_derivatives_accept_optional_host_config(transpose, config):
    with jax.default_device(jax.devices('gpu')[0]):
        case = _make_case(transpose, batch=1)
        fn = lambda weights: _invoke(case, weights=weights, config=config)
        tangent = jnp.arange(case['weights'].size, dtype=jnp.float32) / 5 + .2
        _, derivative = jax.jvp(fn, (case['weights'],), (tangent,))
        _assert_close(derivative, _numpy_reference(case, weights=tangent))
        gradient = jax.grad(lambda weights: jnp.sum(fn(weights)))(case['weights'])
        rows = np.repeat(np.arange(case['shape'][0]), np.diff(np.asarray(case['indptr'])))
        unit = rows if transpose else np.asarray(case['indices'])
        # Binary events: the weight gradient is the exact activity-mask derivative.
        _assert_close(
            gradient, np.sum(np.asarray(case['events'])[unit] != 0, axis=1))


def _wide_case(transpose, target, batch, dtype):
    if transpose:
        shape = (3, target)
        ptr = np.array([0, 2, 2, 4], np.int32)
        indices = [0, target // 2, target // 2, target - 1]
    else:
        shape = (target, 3)
        counts = np.zeros(target, np.int32)
        counts[[0, target // 2, target - 1]] = [1, 2, 1]
        ptr = np.concatenate((np.zeros(1, np.int32), np.cumsum(counts, dtype=np.int32)))
        indices = [0, 0, 2, 1]
    return _make_case(transpose, dtype=dtype, shape=shape, batch=batch, indptr=ptr, indices=indices)


@requires_gpu
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('target,batch,dtype,width', [
    (400000, 2, jnp.float32, 4288),
    (1200000, 4, jnp.float32, 24576),
    (1200000, 4, jnp.float64, 12288),
])
def test_cuda_wide_tiles_and_tail_output(transpose, target, batch, dtype, width):
    with jax_x64_enabled(), jax.default_device(jax.devices('gpu')[0]):
        case = _wide_case(transpose, target, batch, dtype)
        assert case['tile_width'] == width
        result = jax.jit(lambda weights: _invoke(case, weights=weights))(case['weights'])
        _assert_close(result, _numpy_reference(case))


@requires_gpu
@pytest.mark.parametrize('transpose', [True, False])
def test_cuda_vmap_keeps_inner_batch_tile_plan(transpose):
    with jax.default_device(jax.devices('gpu')[0]):
        case = _wide_case(transpose, 100000, 2, jnp.float32)
        assert case['tile_width'] == 1056
        weights = jnp.stack((case['weights'], case['weights'] * 2))
        events = jnp.stack((case['events'], jnp.zeros_like(case['events'])))
        run = jax.jit(jax.vmap(lambda w, e: _invoke(case, weights=w, events=e)))
        result = run(weights, events)
        _assert_close(result[0], _numpy_reference(case))
        _assert_close(result[1], np.zeros_like(result[1]))


@pytest.mark.parametrize('transpose', [True, False])
def test_vmap_and_nested_vmap_preserve_inner_batch(cpu_math_backend, transpose):
    case = _make_case(transpose, batch=2)
    events = jnp.stack((case['events'], -case['events'], jnp.zeros_like(case['events'])))
    weights = jnp.stack((case['weights'], case['weights'] * 2, case['weights'] * -1))
    fn = lambda w, e: _invoke(case, weights=w, events=e)
    result = jax.jit(jax.vmap(fn))(weights, events)
    for i in range(3):
        _assert_close(result[i], _numpy_reference(case, weights=weights[i], events=events[i]))
    nested = jax.jit(jax.vmap(jax.vmap(fn)))(jnp.stack((weights, weights)),
                                            jnp.stack((events, events)))
    _assert_close(nested[0], result)


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('config', [None, CSRConfig(tpr_threshold=5)], ids=['none', 'tuned'])
def test_vmap_pairs_each_canonical_structure_with_its_plan(cpu_math_backend, transpose, config):
    first = _make_case(transpose)
    second = _make_case(transpose, indices=[0, 1, 4, 0, 0, 3])
    cases = (first, second)
    keys = ('weights', 'indices', 'indptr', 'events', 'local_targets', 'tile_offsets')
    arrays = tuple(jnp.stack([case[key] for case in cases]) for key in keys)
    permutations = jnp.stack([case['perm'] if case['perm'] is not None else jnp.empty(0, jnp.int32)
                              for case in cases])
    def run(weights, indices, indptr, events, local, offsets, perm):
        return _invoke(first, weights=weights, indices=indices, indptr=indptr, events=events,
                       local_targets=local, tile_offsets=offsets, perm=perm, config=config)
    result = jax.jit(jax.vmap(run))(*arrays, permutations)
    for i, case in enumerate(cases):
        _assert_close(result[i], _numpy_reference(case))


def test_changing_batch_rejects_old_adaptive_tile_metadata():
    case = _make_case(batch=2, shape=(3, 100000))
    assert case['tile_width'] == 1056
    with pytest.raises(ValueError, match='tile_width'):
        jax.make_jaxpr(lambda: _invoke(case, events=jnp.zeros((3, 4), jnp.float32)))()


@pytest.mark.parametrize('transpose', [True, False])
def test_plan_arrays_and_perm_are_dynamic_primitive_operands(transpose):
    case = _make_case(transpose)
    perm = case['perm'] if case['perm'] is not None else jnp.empty(0, jnp.int32)
    def run(weights, indices, indptr, events, local, offsets, permutation):
        return _invoke(case, _lowlevel=True, weights=weights, indices=indices, indptr=indptr, events=events,
                       local_targets=local, tile_offsets=offsets, perm=permutation)
    arrays = tuple(case[key] for key in ('weights', 'indices', 'indptr', 'events',
                                       'local_targets', 'tile_offsets')) + (perm,)
    graph = jax.make_jaxpr(run)(*arrays)
    equations = [eqn for eqn in graph.jaxpr.eqns if eqn.primitive is api.binary_csrmm_p.primitive]
    assert len(equations) == 1
    assert len(equations[0].invars) == 7
    assert list(equations[0].invars[:6]) == list(graph.jaxpr.invars[:6])
    if not transpose:
        assert equations[0].invars[6] == graph.jaxpr.invars[6]
    assert not graph.consts


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape,batch,ptr,cols', [
    ((3, 5), 0, [0, 3, 3, 6], [0, 2, 2, 1, 3, 4]),
    ((3, 5), 2, [0, 0, 0, 0], []),
    ((0, 5), 2, [0], []),
    ((3, 0), 2, [0, 0, 0, 0], []),
])
def test_empty_dimensions_and_structure(cpu_math_backend, transpose, shape, batch, ptr, cols):
    case = _make_case(transpose, shape=shape, batch=batch, indptr=ptr, indices=cols)
    result = jax.jit(lambda: _invoke(case))()
    assert result.shape == (shape[1 if transpose else 0], batch)
    _assert_close(result, _numpy_reference(case))


@requires_gpu
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape,batch,ptr,cols', [
    ((3, 5), 0, [0, 3, 3, 6], [0, 2, 2, 1, 3, 4]),
    ((3, 5), 2, [0, 0, 0, 0], []),
    ((0, 5), 2, [0], []),
    ((3, 0), 2, [0, 0, 0, 0], []),
])
def test_cuda_empty_dimensions_and_structure(transpose, shape, batch, ptr, cols):
    with jax.default_device(jax.devices('gpu')[0]):
        case = _make_case(transpose, shape=shape, batch=batch, indptr=ptr, indices=cols)
        result = jax.jit(lambda: _invoke(case))()
        _assert_close(result, _numpy_reference(case))


@pytest.mark.parametrize('transpose', [True, False])
def test_inactive_events_isolate_nonfinite_weights(cpu_math_backend, transpose):
    case = _make_case(transpose)
    weights = jnp.array([jnp.nan, jnp.inf, -jnp.inf, 3, 4, 5], jnp.float32)
    events = jnp.zeros_like(case['events'])
    _assert_close(_invoke(case, weights=weights, events=events),
                  np.zeros((case['shape'][1 if transpose else 0], events.shape[1])))
    active = events.at[0, 0].set(jnp.nan)
    _assert_close(_invoke(case, events=active), _numpy_reference(case, events=active))


@pytest.mark.parametrize('field,value', [
    ('events', jnp.ones((3,), jnp.float32)),
    ('events', jnp.ones((3, 3), jnp.int32)),
    ('indices', jnp.zeros(6, jnp.float32)),
    ('indptr', jnp.zeros(4, jnp.uint32)),
    ('local_targets', jnp.zeros(6, jnp.int32)),
    ('local_targets', jnp.zeros(5, jnp.uint16)),
    ('tile_offsets', jnp.zeros(5, jnp.int32)),
    ('tile_offsets', jnp.zeros(18, jnp.uint32)),
    ('weights', jnp.ones(2, jnp.float32)),
    ('weights', jnp.ones(6, jnp.int32)),
    ('backend', 'jax_raw'),
    ('tile_width', 0),
    ('tile_width', 2),
])
def test_invalid_inputs_fail_before_lowering(field, value):
    case = _make_case()
    with pytest.raises((TypeError, ValueError, AssertionError)):
        jax.make_jaxpr(lambda: _invoke(case, **{field: value}))()


@pytest.mark.parametrize('perm', [jnp.arange(5, dtype=jnp.int32), jnp.arange(6, dtype=jnp.uint32)])
def test_col_major_rejects_invalid_perm(perm):
    case = _make_case(False)
    with pytest.raises((TypeError, ValueError, AssertionError)):
        jax.make_jaxpr(lambda: _invoke(case, perm=perm))()


@pytest.mark.parametrize('field', ['indptr', 'tile_offsets', 'perm'])
@pytest.mark.parametrize('lowlevel', [False, True])
def test_numpy_int64_metadata_cannot_be_silently_narrowed(field, lowlevel):
    previous = jax.config.jax_enable_x64
    try:
        jax.config.update('jax_enable_x64', False)
        case = _make_case(transpose=field != 'perm')
        value = np.asarray(case[field], dtype=np.int64)
        with pytest.raises(ValueError, match='preserve|x64|64'):
            jax.make_jaxpr(lambda: _invoke(case, _lowlevel=lowlevel, **{field: value}))()
    finally:
        jax.config.update('jax_enable_x64', previous)


def test_scalar_weight_units_and_registered_call(cpu_math_backend):
    case = _make_case(homo=True)
    result = _invoke(case, weights=case['weights'][0] * u.mV)
    _assert_close(u.get_mantissa(result), _numpy_reference(case))
    assert u.get_unit(result) == u.mV
    args = dict(case)
    weights, indices, indptr, events = (args.pop(key) for key in
                                      ('weights', 'indices', 'indptr', 'events'))
    outputs = api.binary_csrmm_p.call(weights, indices, indptr, events, **args)
    assert len(outputs) == 1
    _assert_close(outputs[0], _numpy_reference(case))
    assert api.binary_csrmm_p.available_backends('gpu') == ['cuda_raw']


def test_native_resource_dependencies_are_packaged_locally():
    root = Path(api.__file__).parent / 'combined'
    include_root = Path(brainevent.__file__).parent / 'include'
    for relative in ('combined_spmm_binary.cu', 'combined_kernels.cuh',
                     'combined_impl.cuh', 'combined_ffi.cuh', 'planner.h',
                     'col_major/combined_spmm_col_major_binary.cu',
                     'col_major/combined_col_major.cuh',
                     'col_major/combined_col_major_ffi.cuh'):
        assert (root / relative).is_file(), relative
    for relative in ('cuda_traits.cuh', 'op_traits.cuh'):
        assert (include_root / 'brainevent' / relative).is_file(), relative
    sources = [*root.rglob('*.cuh'), *root.rglob('*.cu'), *root.rglob('*.h')]
    for path in sources:
        if 'test' in path.relative_to(root).parts:
            continue
        for line in path.read_text().splitlines():
            if line.startswith('#include "'):
                dependency = line.split('"')[1]
                assert any((base / dependency).is_file() for base in (path.parent, include_root)), (
                    path, dependency)
