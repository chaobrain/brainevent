# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Delay layouts inherited from CSR, built on the TCSR canonical edge order."""

import inspect

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import CSR, CSRDelayLayout, TCSR, BinaryArray, csr_delay_deliver_layout

_COLUMNS = np.array([2, 0, 2, 1, 0], np.int32)
_POINTERS = np.array([0, 3, 5], np.int32)
_SOURCE_DELAYS = np.array([2, 0, 1, 2, 1], np.int32)
# TCSR position k holds the original edge _SORT_ORDER[k] (columns ascending in each row)
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
def test_delay_layout_is_inherited_and_leaves_the_source_unchanged(mode):
    _, tiled, canonical = _case(mode)
    data, indices, indptr = tiled.data, tiled.indices, tiled.indptr
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    parent = canonical.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)

    assert inspect.signature(TCSR.delay_layout) == inspect.signature(CSR.delay_layout)
    assert type(layout) is CSRDelayLayout
    assert layout.shape == tiled.shape and layout.n_slots == 4
    for name in ('data', 'indices', 'indptr', 'delays', 'delay_ptr', 'perm'):
        np.testing.assert_array_equal(getattr(layout, name), getattr(parent, name))
    assert tiled.data is data and tiled.indices is indices and tiled.indptr is indptr
    if mode != 'heterogeneous':
        assert layout.data.shape == (1,)
    else:
        # sorted by (delay, column) within each row of the TCSR order
        np.testing.assert_array_equal(layout.data, [3., 5., 2., 11., 7.])
        np.testing.assert_array_equal(layout.original_data(), tiled.data)


@pytest.mark.parametrize('mode', ['heterogeneous', 'singleton'])
def test_delivery_matches_the_original_edges(mode):
    original, tiled, _ = _case(mode)
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    ring = jnp.arange(12, dtype=jnp.float32).reshape((4, 3))
    events = jnp.array([-2., 0.], jnp.float32)
    expected = _reference(original, _SOURCE_DELAYS, np.asarray(events), 3, ring)

    result = jax.jit(lambda prepared, state, x: csr_delay_deliver_layout(state, x, prepared, 3))(
        layout, ring, events)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(ring, np.arange(12, dtype=np.float32).reshape((4, 3)))


def test_jitted_scan_with_wrapped_time_matches_the_reference():
    original, tiled, _ = _case()
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    events = jnp.array([[True, False], [False, True], [True, True], [False, False], [True, False]])

    def step(ring, inputs):
        time, x = inputs
        ring = csr_delay_deliver_layout(ring, BinaryArray(x), layout, time)
        arrived = ring[(time + 1) % 4]
        return ring.at[time % 4].set(0.), arrived

    final, outputs = jax.jit(lambda ring, xs: jax.lax.scan(
        step, ring, (jnp.arange(xs.shape[0], dtype=jnp.int32), xs),
    ))(jnp.zeros((4, 3), jnp.float32), events)
    expected_ring = np.zeros((4, 3), np.float32)
    expected_outputs = []
    for time, x in enumerate(np.asarray(events)):
        expected_ring = _reference(original, _SOURCE_DELAYS, x, time, expected_ring)
        expected_outputs.append(expected_ring[(time + 1) % 4].copy())
        expected_ring[time % 4] = 0
    np.testing.assert_array_equal(outputs, expected_outputs)
    np.testing.assert_array_equal(final, expected_ring)


def test_with_data_takes_weights_in_the_tcsr_order():
    _, tiled, _ = _case()
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    replaced = jax.jit(lambda prepared, data: prepared.with_data(data))(layout, tiled.data * 3)
    assert type(replaced) is CSRDelayLayout
    np.testing.assert_array_equal(replaced.original_data(), tiled.data * 3)
    np.testing.assert_array_equal(replaced.data, layout.data * 3)
    np.testing.assert_array_equal(replaced.indices, layout.indices)


def test_weight_units_survive_the_inherited_factory_and_delivery():
    original, tiled, _ = _case(unit=u.mV)
    layout = tiled.delay_layout(delays=_CANONICAL_DELAYS, n_slots=4)
    result = layout.deliver(ring=jnp.zeros((4, 3), jnp.float32) * u.mV, events=jnp.array([True, True]), t=1)
    assert u.get_unit(layout.data) == u.mV
    assert u.get_unit(result) == u.mV
    expected = _reference(original, _SOURCE_DELAYS, [True, True], 1, np.zeros((4, 3), np.float32))
    np.testing.assert_array_equal(u.get_mantissa(result), expected)


@pytest.mark.parametrize('delays,n_slots', [([-1] * 5, 4), ([4] * 5, 4), ([1] * 4, 4)])
def test_delay_layout_keeps_the_parent_validation(delays, n_slots):
    _, tiled, canonical = _case()
    for source in (tiled, canonical):
        with pytest.raises((ValueError, TypeError)):
            source.delay_layout(delays=np.asarray(delays, np.int32), n_slots=n_slots)
