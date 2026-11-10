# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Weighted direct/homo SpMM: types, event semantics, precision and FFI contracts."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._tcsr.binary.combined import jax_ffi as api
from brainevent._tcsr.binary.combined.test.weighted_reference import matrix, reference

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="CUDA device required")
DTYPES = [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]


def ffi_call(input_values, input_events, plan, *, homo=False, **overrides):
    """Call the typed entry directly to exercise validation below Python wrappers."""
    values, events = input_values, input_events
    args = dict(values=values, local=jnp.asarray(plan.local_targets),
                offsets=jnp.asarray(plan.tile_offsets), events=events,
                active=max(1, plan.batch * plan.rows), scratch=api.workspace_words(plan.rows, plan.batch),
                width=plan.tile_width, code=api.VALUE_CODE[plan.value_dtype])
    args.update(overrides)
    outputs = (jax.ShapeDtypeStruct((plan.batch, plan.cols), values.dtype),
               jax.ShapeDtypeStruct((args['active'],), jnp.int32),
               jax.ShapeDtypeStruct((args['scratch'],), jnp.uint32))
    mode = '_homo' if homo else ''
    target = f"{api._load(True)}.combined_weighted{mode}_{api.VALUE_SUFFIX[values.dtype]}"
    return jax.ffi.ffi_call(target, outputs)(args['values'], args['local'], args['offsets'], args['events'],
                                            tile_width=args['width'], value_type=args['code'])


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
@pytest.mark.parametrize('offset_dtype', [np.int32, np.int64])
@pytest.mark.parametrize('homo', [False, True])
def test_dtype_matrix_and_runtime_updates(dtype, index_dtype, offset_dtype, homo):
    source, target, ptr = matrix()
    plan = api.prepare(ptr.astype(offset_dtype), target.astype(index_dtype), 24581, 3, dtype=dtype)
    events = jnp.asarray(((np.arange(3 * plan.rows).reshape(3, -1) % 7) - 3) / 4, dtype)
    call = jax.jit(lambda w, e: api.spmm_weighted(w, e, plan))
    for shift, event in [(0, events), (0, jnp.zeros_like(events)), (2, events * -2)]:
        values = jnp.asarray((np.arange(1 if homo else len(source)) % 7 - 3 + shift) / 4, dtype)
        out = call(values, event)
        expected = reference(source, target, np.asarray(values), np.asarray(event), plan.cols)
        np.testing.assert_array_equal(np.asarray(out), expected.astype(jnp.dtype(dtype)))
        assert out.dtype == jnp.dtype(dtype)
    if homo:
        np.testing.assert_array_equal(np.asarray(api.spmm_weighted(values[0], events, plan)),
                                      reference(source, target, values, events, plan.cols).astype(jnp.dtype(dtype)))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('homo', [False, True])
def test_nonfinite_event_semantics(dtype, homo):
    ptr = np.array([0, 1, 2, 3, 4, 4], np.int64)
    target = np.array([0, 1, 2, 3], np.int32)
    source = np.arange(4)
    plan = api.prepare(ptr, target, 5, 4, dtype=dtype)
    events = jnp.array([[0, -0., -2, 0.5, np.nan], [0, 0, np.nan, 0, 1],
                        [1, -2, 0, np.inf, 0], [0, 0, 0, 0, 0]], dtype)
    for weights in ([np.inf, np.nan] if homo else [None]):
        values = jnp.array([weights] if homo else [np.nan, np.inf, 2, 0], dtype)
        out = api.spmm_weighted(values, events, plan)
        np.testing.assert_array_equal(np.asarray(out).astype(np.float64),
                                      reference(source, target, values, events, 5))


@pytest.mark.parametrize('dtype', DTYPES)
def test_products_and_sums_use_accumulator_precision(dtype):
    # Rounding each low-precision product first would erase this residual.
    bit = 10 if dtype == jnp.float16 else 7 if dtype == jnp.bfloat16 else 20 if dtype == jnp.float32 else 35
    delta = 2. ** -bit
    values = jnp.array([1 + delta, -(1 + 2 * delta)], dtype)
    events = jnp.array([[1 + delta, 1]], dtype)
    plan = api.prepare(np.array([0, 1, 2], np.int32), np.array([0, 0], np.int32), 1, 1, dtype=dtype)
    acc_dtype = np.float64 if dtype == jnp.float64 else np.float32
    product = np.asarray(values).astype(acc_dtype) * np.asarray(events)[0].astype(acc_dtype)
    expected = np.array([[product.sum(dtype=acc_dtype)]], dtype=jnp.dtype(dtype))
    np.testing.assert_array_equal(np.asarray(api.spmm_weighted(values, events, plan)), expected)
    if dtype in (jnp.float16, jnp.bfloat16):
        assert expected[0, 0] != 0
    # f64 amplitudes must not be narrowed to float32.
    if dtype == jnp.float64:
        plan = api.prepare(np.array([0, 1], np.int32), np.array([0], np.int32), 1, 1, dtype=dtype)
        np.testing.assert_array_equal(np.asarray(api.spmm_weighted(jnp.ones(1, dtype), events[:, :1], plan)),
                                      np.asarray(events[:, :1]))


@pytest.mark.parametrize('dtype', DTYPES)
def test_ordered_rows_across_extraction_chunks(dtype):
    rows, batch, cols = 8201, 3, api.tile_width(dtype) + 1
    source = np.arange(rows)
    target = (source * 37) % cols
    plan = api.prepare(np.arange(rows + 1, dtype=np.int64), target.astype(np.int32), cols, batch, dtype=dtype)
    events = jnp.asarray((np.arange(batch * rows).reshape(batch, rows) % 5 - 2) / 2, dtype)
    weights = jnp.asarray((source % 7 - 3) / 4, dtype)
    out, selected, scratch = ffi_call(weights, events, plan)
    np.testing.assert_array_equal(np.asarray(out), reference(source, target, weights, events, cols))
    count = np.asarray(scratch)[:batch * 2].reshape(batch, 2)[:, 0]
    for b in range(batch):
        active = np.flatnonzero(np.asarray(events)[b] != 0)
        assert count[b] == len(active)
        np.testing.assert_array_equal(np.asarray(selected).reshape(batch, rows)[b, :count[b]], active)


@pytest.mark.parametrize('shape', [(3, 5, 2), (0, 5, 2), (3, 0, 2), (0, 0, 2), (3, 5, 0)])
@pytest.mark.parametrize('homo', [False, True])
def test_empty_overwrite(shape, homo):
    rows, cols, batch = shape
    plan = api.prepare(np.zeros(rows + 1, np.int32), np.empty(0, np.int64), cols, batch)
    values = jnp.array([np.nan] if homo else [], jnp.float32)
    out = api.spmm_weighted(values, jnp.ones((batch, rows), jnp.float32), plan)
    np.testing.assert_array_equal(np.asarray(out), np.zeros((batch, cols)))


@pytest.mark.parametrize('invalid', ['event_dtype', 'value_dtype', 'event_shape', 'weight_shape', 'tile_width'])
def test_python_validation_before_loading(monkeypatch, invalid):
    monkeypatch.setattr(api, '_load', lambda *args: pytest.fail('invalid input reached CUDA loading'))
    plan = api.prepare(np.array([0, 2], np.int32), np.array([0, 1], np.int32), 2, 1)
    values, events = jnp.ones(2, jnp.float32), jnp.ones((1, 1), jnp.float32)
    if invalid == 'event_dtype': events = events.astype(jnp.bool_)
    elif invalid == 'value_dtype': values = values.astype(jnp.float64)
    elif invalid == 'event_shape': events = events.ravel()
    elif invalid == 'weight_shape': values = values.reshape(1, 2)
    else: plan = replace(plan, tile_width=8192)
    with pytest.raises(ValueError):
        api.spmm_weighted(values, events, plan)


@pytest.mark.parametrize('invalid', ['event_dtype', 'event_rank', 'values', 'offsets', 'local', 'active', 'scratch', 'width', 'code'])
def test_ffi_validation(invalid):
    plan = api.prepare(np.array([0, 2], np.int64), np.array([0, 1], np.int32), 2, 1)
    values, events = jnp.ones(2, jnp.float32), jnp.ones((1, 1), jnp.float32)
    bad = dict(event_dtype={'events': events.astype(jnp.bool_)}, event_rank={'events': events.ravel()},
               values={'values': values[:1]}, offsets={'offsets': jnp.zeros(1, jnp.int64)},
               local={'local': jnp.zeros(2, jnp.int32)}, active={'active': 0}, scratch={'scratch': 0},
               width={'width': 8192}, code={'code': 3})[invalid]
    with pytest.raises(Exception, match='dtype|rank|length|plan|metadata|scratch'):
        jax.block_until_ready(ffi_call(values, events, plan, **bad))


def test_int32_without_x64():
    with jax.enable_x64(False):
        plan = api.prepare(np.array([0, 1], np.int32), np.array([2], np.int32), 4, 2)
        out = api.spmm_weighted(jnp.array([3], jnp.float32), jnp.array([[-2], [0]], jnp.float32), plan)
        np.testing.assert_array_equal(np.asarray(out), [[0, 0, -6, 0], [0, 0, 0, 0]])
