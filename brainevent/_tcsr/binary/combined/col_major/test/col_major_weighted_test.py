# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Weighted mirror execution against original coordinates and canonical weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._tcsr.binary.combined.col_major import jax_ffi as api
from brainevent._tcsr.binary.combined.col_major.test.col_major_test import make_matrix
from brainevent._tcsr.binary.combined.test.weighted_reference import reference

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != 'gpu', reason='CUDA device required')
DTYPES = [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
@pytest.mark.parametrize('offset_dtype', [np.int32, np.int64])
def test_types_jit_weights_and_permutation_updates(dtype, index_dtype, offset_dtype):
    rows, cols, (ptr, indices, perm) = make_matrix()
    plan = api.prepare(ptr.astype(offset_dtype), indices.astype(index_dtype), 24581, 3, dtype=dtype)
    device_perm = jnp.asarray(perm, offset_dtype)
    events = jnp.asarray([[0, -2, 0.5, 0, 1.5, -0.25, 1], [1, 0, -1, 2, 0, -0.5, 0],
                          [0, 0, 0, 0, 0, 0, 0]], dtype)
    call = jax.jit(lambda v, p, e: api.spmm_weighted(v, p, e, plan))
    for shift, event in [(0, events), (0, jnp.zeros_like(events)), (3, events * -2)]:
        values = jnp.asarray(((np.arange(len(rows)) + shift) % 7 - 3) / 4, dtype)
        out = call(values, device_perm, event)
        expected = reference(cols, rows, values, event, plan.cols)
        np.testing.assert_array_equal(np.asarray(out), expected.astype(jnp.dtype(dtype)))
        assert out.dtype == jnp.dtype(dtype)
    graph = str(jax.make_jaxpr(lambda v, p, e: api.spmm_weighted(v, p, e, plan))(values, device_perm, events))
    assert 'ffi_call' in graph and 'gather' not in graph
    # Runtime perm changes only weight addresses, never event/source-row addresses.
    reverse = device_perm[::-1]
    remapped = np.empty(len(rows))
    remapped[perm] = np.asarray(values)[np.asarray(reverse)]
    np.testing.assert_array_equal(np.asarray(call(values, reverse, events)),
                                  reference(cols, rows, remapped, events, plan.cols).astype(jnp.dtype(dtype)))


@pytest.mark.parametrize('dtype', DTYPES)
def test_nonfinite_values_and_inactive_edges(dtype):
    rows = np.arange(5)
    cols = np.array([2, 0, 1, 3, 0])
    perm = np.argsort(cols, kind='stable')
    plan = api.prepare(np.array([0, 2, 3, 4, 5], np.int64), rows[perm], 5, 4, dtype=dtype)
    values = jnp.array([0, np.nan, np.inf, -np.inf, 2], dtype)
    events = jnp.array([[0, 0, np.nan, 0], [0, -2, 0, 0], [1, 0, 0, -0.25],
                        [-0., 0, 0, 0]], dtype)
    out = api.spmm_weighted(values, jnp.asarray(perm, jnp.int64), events, plan)
    np.testing.assert_array_equal(np.asarray(out).astype(np.float64), reference(cols, rows, values, events, 5))


@pytest.mark.parametrize('dtype', DTYPES)
def test_tcsr_mirror_and_canonical_values(dtype):
    from brainevent import CSR, TCSR

    rows, cols, _ = make_matrix(m=37)
    ptr = np.r_[0, np.bincount(rows, minlength=37).cumsum()]
    weights = jnp.asarray((np.arange(len(rows)) % 7 - 3) / 4, dtype)
    matrix = TCSR.fromcsr(CSR(weights, cols, ptr, shape=(37, 7), indptr_dtype=np.int64))
    assert not matrix.has_tcsc_mirror
    matrix = matrix.build_weight_indices()
    mirror = matrix.buffers['csc']
    plan = api.prepare(np.asarray(mirror[0]), np.asarray(mirror[1]), 37, 2, dtype=dtype)
    events = jnp.array([[1, 0, -2, 0, 0.5, -0.25, 0], [0, -1, 0, 1.5, 0, 0, 2]], dtype)
    out = api.spmm_weighted(matrix.data, mirror[2], events, plan)
    np.testing.assert_array_equal(np.asarray(out), reference(cols, rows, weights, events, 37).astype(jnp.dtype(dtype)))
    assert matrix.build_weight_indices().buffers['csc'] is mirror


@pytest.mark.parametrize('shape', [(5, 3, 2), (5, 0, 2), (0, 3, 2), (0, 0, 2), (5, 3, 0)])
def test_empty_structure(shape):
    m, n, batch = shape
    plan = api.prepare(np.zeros(n + 1, np.int64), np.empty(0, np.int32), m, batch)
    out = api.spmm_weighted(jnp.empty(0, jnp.float32), jnp.empty(0, jnp.int64),
                            jnp.ones((batch, n), jnp.float32), plan)
    np.testing.assert_array_equal(np.asarray(out), np.zeros((batch, m)))


def test_one_edge_with_x64_disabled():
    with jax.enable_x64(False):
        plan = api.prepare(np.array([0, 0, 1], np.int32), np.array([2], np.int32), 4, 2)
        out = api.spmm_weighted(jnp.array([3], jnp.float32), jnp.array([0], jnp.int32),
                                jnp.array([[0, -2], [1, 0]], jnp.float32), plan)
        np.testing.assert_array_equal(np.asarray(out), [[0, 0, -6, 0], [0, 0, 0, 0]])


@pytest.mark.parametrize('invalid', ['event_dtype', 'value_dtype', 'event_rank', 'values', 'perm_dtype', 'perm_rank', 'perm_length'])
def test_python_rejects_before_loading(monkeypatch, invalid):
    monkeypatch.setattr(api, '_load', lambda *args: pytest.fail('invalid input reached CUDA loading'))
    plan = api.prepare(np.array([0, 2], np.int64), np.array([0, 1], np.int32), 2, 1)
    values, events, perm = jnp.ones(2, jnp.float32), jnp.ones((1, 1), jnp.float32), jnp.array([1, 0], jnp.int64)
    if invalid == 'event_dtype': events = events.astype(jnp.float64)
    elif invalid == 'value_dtype': values = values.astype(jnp.float64)
    elif invalid == 'event_rank': events = events.ravel()
    elif invalid == 'values': values = values[:1]
    elif invalid == 'perm_dtype': perm = perm.astype(jnp.int32)
    elif invalid == 'perm_rank': perm = perm.reshape(1, 2)
    else: perm = perm[:1]
    with pytest.raises(ValueError):
        api.spmm_weighted(values, perm, events, plan)


@pytest.mark.parametrize('invalid', ['event_dtype', 'perm_dtype', 'perm_length', 'values', 'offsets', 'active', 'scratch', 'width'])
def test_ffi_validation(invalid):
    plan = api.prepare(np.array([0, 2], np.int64), np.array([0, 1], np.int32), 2, 1)
    values, events, perm = jnp.ones(2, jnp.float32), jnp.ones((1, 1), jnp.float32), jnp.array([1, 0], jnp.int64)
    offsets = jnp.asarray(plan.tile_offsets)
    active, scratch, width = 1, api.workspace_words(1, 1), plan.tile_width
    if invalid == 'event_dtype': events = events.astype(jnp.bool_)
    elif invalid == 'perm_dtype': perm = perm.astype(jnp.int32)
    elif invalid == 'perm_length': perm = perm[:1]
    elif invalid == 'values': values = values[:1]
    elif invalid == 'offsets': offsets = offsets[:1]
    elif invalid == 'active': active = 0
    elif invalid == 'scratch': scratch = 0
    else: width = 8192
    outputs = (jax.ShapeDtypeStruct((1, 2), jnp.float32), jax.ShapeDtypeStruct((active,), jnp.int32),
               jax.ShapeDtypeStruct((scratch,), jnp.uint32))
    target = f'{api._load(True)}.combined_col_major_weighted_hetero_f32'
    with pytest.raises(Exception, match='dtype|length|plan|metadata|scratch'):
        jax.block_until_ready(jax.ffi.ffi_call(target, outputs)(values, jnp.asarray(plan.local_targets),
            offsets, perm, events, tile_width=width, value_type=0 + api._base.VALUE_CODE[plan.value_dtype]))
