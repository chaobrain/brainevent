# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Delay delivery inherited from CSR, using the TCSR canonical edge order."""

import inspect

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSR, TCSR, csr_delay_deliver_layout, delaycsr


_COLUMNS = np.array([2, 0, 2, 1, 0], np.int32)
_POINTERS = np.array([0, 3, 5], np.int32)
_SOURCE_DELAYS = np.array([2, 0, 1, 2, 1], np.int32)
_SORT_ORDER = np.array([1, 0, 2, 4, 3], np.int32)
_CANONICAL_DELAYS = _SOURCE_DELAYS[_SORT_ORDER]


def _case(mode='heterogeneous', *, unit=None):
    values = {'heterogeneous': [2., 3., 5., 7., 11.], 'singleton': [2.], 'scalar': 2.}[mode]
    data = jnp.asarray(values, jnp.float32)
    if unit is not None:
        data = data * unit
    original = CSR(data, _COLUMNS, _POINTERS, shape=(2, 3), backend='jax_raw')
    tiled = TCSR(original)
    canonical = CSR(tiled.data, tiled.indices, tiled.indptr, shape=tiled.shape, backend='jax_raw')
    return original, tiled, canonical


def _reference(source, delays, events, time, ring):
    result = np.array(ring, copy=True)
    weights = np.broadcast_to(np.asarray(u.get_mantissa(source.data)), (source.nse,))
    pointers, columns = np.asarray(source.indptr), np.asarray(source.indices)
    for row in range(source.shape[0]):
        if events[row] != 0:
            for edge in range(int(pointers[row]), int(pointers[row + 1])):
                result[(time + int(delays[edge])) % len(result), columns[edge]] += weights[edge]
    return result


@pytest.mark.parametrize('mode', ['heterogeneous', 'singleton', 'scalar'])
def test_delay_layout_inherits_keywords_metadata_and_source_isolation(mode):
    _, tiled, canonical = _case(mode)
    data, indices, indptr, cache = tiled.data, tiled.indices, tiled.indptr, tiled._cache
    geometries = dict(cache.tile_geometries)
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    parent = canonical.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)

    assert inspect.signature(TCSR.delay_layout) == inspect.signature(CSR.delay_layout)
    assert type(layout) is delaycsr and isinstance(layout, CSR)
    assert layout.backend == tiled.backend == 'jax_raw'
    assert layout.shape == tiled.shape and layout.L == 4
    assert layout.delay_steps == parent.delay_steps == (0, 1, 2)
    for name in ('data', 'indices', 'indptr', 'delay_count', 'delay_off', 'row_task_count'):
        np.testing.assert_array_equal(getattr(layout, name), getattr(parent, name))
    np.testing.assert_array_equal(layout.ring, np.zeros((4, 3), np.float32))
    assert layout._cache.tile_plans == {}
    assert tiled.data is data and tiled.indices is indices and tiled.indptr is indptr
    assert tiled._cache is cache
    assert cache.tile_geometries.keys() == geometries.keys()
    assert all(cache.tile_geometries[key] is value for key, value in geometries.items())
    if mode != 'heterogeneous':
        assert layout.data.shape == (1,)
    else:
        np.testing.assert_array_equal(layout.data, [3., 5., 2., 11., 7.])


@pytest.mark.parametrize('mode', ['heterogeneous', 'singleton'])
def test_delay_arrays_follow_tcsr_sorted_edges_and_deliver_additively(mode):
    original, tiled, _ = _case(mode)
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    ring = jnp.arange(12, dtype=jnp.float32).reshape((4, 3))
    events = jnp.array([-2., 0.], jnp.float32)
    initial_owned = layout.ring
    expected = _reference(original, _SOURCE_DELAYS, np.asarray(events), 3, ring)

    result = jax.jit(lambda prepared, state, x: csr_delay_deliver_layout(
        state, x, prepared, 3,
    ))(layout, ring, events)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(ring, np.arange(12, dtype=np.float32).reshape((4, 3)))
    assert layout.ring is initial_owned
    assert int(layout.head) == 0


@pytest.mark.parametrize('use_matmul', [False, True])
def test_delay_forward_and_event_matmul_match_explicit_ring_reference(use_matmul):
    original, tiled, canonical = _case()
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    parent = canonical.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    events = np.array([[1., 0.], [0., -2.], [-1., 3.], [0., 0.], [1., 1.]], np.float32)
    expected_ring = np.zeros((4, 3), np.float32)

    for time, x in enumerate(events):
        expected_ring[time % 4] = 0
        expected_ring = _reference(original, _SOURCE_DELAYS, x, time, expected_ring)
        if use_matmul:
            layout.head = time % 4
            parent.head = time % 4
            result = BinaryArray(jnp.asarray(x)) @ layout
            parent_result = BinaryArray(jnp.asarray(x)) @ parent
        else:
            result = layout.forward(x=BinaryArray(jnp.asarray(x)), t=time)
            parent_result = parent.forward(x=BinaryArray(jnp.asarray(x)), t=time)
        np.testing.assert_array_equal(result, expected_ring[(time + 1) % 4])
        np.testing.assert_array_equal(layout.ring, expected_ring)
        np.testing.assert_array_equal(result, parent_result)
        np.testing.assert_array_equal(layout.ring, parent.ring)
        assert int(layout.head) == time % 4


def test_delay_with_data_preserves_current_order_arrivals_and_jit_state():
    _, tiled, _ = _case()
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    layout.forward(x=jnp.array([True, False]), t=0)
    arrivals = np.asarray(layout.ring).copy()
    weights = layout.data * 3
    replaced = jax.jit(lambda prepared, data: prepared.with_data(data=data))(layout, weights)

    assert type(replaced) is delaycsr
    np.testing.assert_array_equal(replaced.data, weights)
    np.testing.assert_array_equal(replaced.ring, arrivals)
    np.testing.assert_array_equal(layout.data, [3., 5., 2., 11., 7.])
    np.testing.assert_array_equal(layout.ring, arrivals)
    np.testing.assert_array_equal(replaced.indices, layout.indices)
    np.testing.assert_array_equal(replaced.delay_count, layout.delay_count)
    assert replaced.backend == 'jax_raw'

    with pytest.raises(ValueError, match='shape'):
        layout.with_data(jnp.ones((1,), jnp.float32))
    with pytest.raises(TypeError, match='dtype'):
        layout.with_data(layout.data.astype(jnp.float16))


def test_delay_state_survives_jitted_scan_with_wrapped_time():
    original, tiled, _ = _case()
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    events = jnp.array([[True, False], [False, True], [True, True], [False, False], [True, False]])

    def step(prepared, inputs):
        time, x = inputs
        output = prepared.forward(x=x, t=time)
        return prepared, output

    final, outputs = jax.jit(lambda prepared, xs: jax.lax.scan(
        step, prepared, (jnp.arange(xs.shape[0], dtype=jnp.int32), xs),
    ))(layout, events)
    expected_ring = np.zeros((4, 3), np.float32)
    expected_outputs = []
    for time, x in enumerate(np.asarray(events)):
        expected_ring[time % 4] = 0
        expected_ring = _reference(original, _SOURCE_DELAYS, x, time, expected_ring)
        expected_outputs.append(expected_ring[(time + 1) % 4].copy())

    assert type(final) is delaycsr
    np.testing.assert_array_equal(outputs, expected_outputs)
    np.testing.assert_array_equal(final.ring, expected_ring)
    np.testing.assert_array_equal(layout.ring, np.zeros((4, 3), np.float32))
    assert int(final.head) == 0


def test_delay_preserves_weight_units_through_inherited_factory_and_delivery():
    original, tiled, _ = _case(unit=u.mV)
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    result = layout.deliver(ring=jnp.zeros((4, 3), jnp.float32) * u.mV,
                            events=jnp.array([True, True]), t=1)
    assert u.get_unit(layout.data) == u.mV
    assert u.get_unit(layout.ring) == u.mV
    assert u.get_unit(result) == u.mV
    expected = _reference(original, _SOURCE_DELAYS, [True, True], 1, np.zeros((4, 3), np.float32))
    np.testing.assert_array_equal(u.get_mantissa(result), expected)


@pytest.mark.parametrize('delays,n_slots', [([-1] * 5, 4), ([4] * 5, 4), ([1] * 4, 4)])
def test_delay_layout_retains_parent_validation(delays, n_slots):
    _, tiled, canonical = _case()
    for source in (tiled, canonical):
        with pytest.raises((ValueError, TypeError)):
            source.delay_layout(delays=np.asarray(delays, np.int32), n_slots=n_slots)
