# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Event weight gradients follow the event type, with no separate option.

Binary events (``BinaryArray``, raw booleans, ``weighted=False``) multiply the
weights by their activity mask in the forward pass, so the exact weight
gradient is ``ct * (x != 0)``. Weighted events (``WeightedArray``,
``weighted=True``) multiply by their values, so the gradient is ``ct * x``.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSC, CSR, TCSR, WeightedArray
from brainevent._csr.config import CSRConfig
from brainevent._sddmm import dispatch

_INDICES = jnp.array([0, 2, 1, 2], jnp.int32)
_INDPTR = jnp.array([0, 2, 4], jnp.int32)
_ROWS = np.array([0, 0, 1, 1])
_PERM = jnp.array([2, 0, 3, 1], jnp.int32)
_EVENT_VALUES = [0., 2., -3., .5]
_DENSE = jnp.array([[1., 0., 2.], [0., 3., 4.]], jnp.float32)


def _low_level_op(matrix, indexed, lowlevel):
    api = importlib.import_module(
        'brainevent._csr.binary.binary_indexed'
        if indexed else 'brainevent._csr.binary.binary'
    )
    name = 'binary_csrmm' if matrix else 'binary_csrmv'
    name += '_indexed' if indexed else ''
    name += '_p_call' if lowlevel else ''
    return getattr(api, name)


def _expected_slot_gradient(events, ct, transpose, weighted, homogeneous, indexed):
    """Reference weight cotangent of ``vdot(op(W, events), ct)``."""
    cols = np.asarray(_INDICES)
    unit, target = (_ROWS, cols) if transpose else (cols, _ROWS)
    factor = np.asarray(events)[unit]
    if not weighted:
        factor = (factor != 0).astype(np.float32)
    slots = np.sum(factor * np.asarray(ct)[target], axis=1)
    expected = np.zeros(1 if homogeneous else 4, np.float32)
    owners = (np.zeros(4, int) if homogeneous else
              (np.asarray(_PERM) if indexed else np.arange(4)))
    np.add.at(expected, owners, slots)
    return expected


@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('matrix', [False, True], ids=['mv', 'mm'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('homogeneous', [False, True])
@pytest.mark.parametrize('lowlevel', [False, True])
def test_weight_gradient_follows_event_mode(
    monkeypatch, weighted, matrix, transpose, indexed, homogeneous, lowlevel,
):
    """Non-unit, negative and zero events separate masks from values."""
    op = _low_level_op(matrix, indexed, lowlevel)
    weights = jnp.array([1.] if homogeneous else [1., 2., 3., 4.], jnp.float32)
    batch = 2 if matrix else 1
    units, targets = (2, 3) if transpose else (3, 2)
    events = jnp.array(np.resize(_EVENT_VALUES, (units, batch)), jnp.float32)
    ct = jnp.array(np.resize([5., -2., 3., 7.], (targets, batch)), jnp.float32)
    expected = _expected_slot_gradient(events, ct, transpose, weighted, homogeneous, indexed)

    seen = []
    original = dispatch.sddmm

    def record(*args, **kwargs):
        seen.append(kwargs['mode'])
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatch, 'sddmm', record)

    def loss(data):
        args = (data, _INDICES, _INDPTR)
        if indexed:
            args += (_PERM,)
        output = op(*args, events if matrix else events[:, 0],
                    shape=(2, 3), transpose=transpose, backend='jax_raw',
                    weighted=weighted)
        if lowlevel:
            output = output[0]
        return jnp.vdot(output, ct if matrix else ct[:, 0])

    actual = jax.jit(jax.grad(loss))(weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert seen and set(seen) == {'weighted' if weighted else 'binary'}


def _event_backends():
    return ['jax_raw', None] + (['cuda_raw'] if jax.default_backend() == 'gpu' else [])


@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('matrix', [False, True], ids=['mv', 'mm'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('backend', _event_backends())
def test_shared_weight_gradient_matches_dense_and_ignores_idle_nonfinite_cotangent(
    weighted, matrix, transpose, indexed, backend,
):
    """A shared weight takes ``<ct, A_unit @ events>``; a NaN/Inf cotangent on an
    output that received only inactive events still contributes exactly 0."""
    op = _low_level_op(matrix, indexed, lowlevel=False)
    batch = 2 if matrix else 1
    # Row 0 holds columns {0, 2}, row 1 holds {1, 2}. These events leave one
    # output idle: row 1 (no transpose) or column 1 (transpose).
    events = np.array([[2., -3.], [0., 0.]] if transpose else [[2., -3.], [0., 0.], [0., 0.]], np.float32)
    idle = 1
    finite = np.array([[5., -2.], [7., 4.], [3., 1.]] if transpose else [[5., -2.], [7., 4.]], np.float32)
    events, finite = events[:, :batch], finite[:, :batch]

    def gradient(ct):
        def loss(data):
            args = (data, _INDICES, _INDPTR) + ((_PERM,) if indexed else ())
            x = jnp.asarray(events if matrix else events[:, 0])
            output = op(*args, x, shape=(2, 3), transpose=transpose, backend=backend, weighted=weighted)
            return jnp.vdot(output, jnp.asarray(ct if matrix else ct[:, 0]))
        return jax.jit(jax.grad(loss))(jnp.array([1.5], jnp.float32))

    expected = _expected_slot_gradient(events, finite, transpose, weighted, True, indexed)
    np.testing.assert_allclose(gradient(finite), expected, rtol=1e-6)
    for bad in (np.nan, np.inf):
        ct = finite.copy()
        ct[idle] = bad
        clean = finite.copy()
        clean[idle] = 0.
        expected = _expected_slot_gradient(events, clean, transpose, weighted, True, indexed)
        np.testing.assert_allclose(gradient(ct), expected, rtol=1e-6)


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('matrix', [False, True], ids=['mv', 'mm'])
def test_binary_weight_jacfwd_matches_jacrev(matrix, transpose, indexed):
    """Forward and reverse weight derivatives agree for non-binary event values."""
    op = _low_level_op(matrix, indexed, lowlevel=False)
    weights = jnp.array([1., 2., 3., 4.], jnp.float32)
    batch = 2 if matrix else 1
    units = 2 if transpose else 3
    events = jnp.array(np.resize(_EVENT_VALUES, (units, batch)), jnp.float32)
    if not matrix:
        events = events[:, 0]

    def product(data):
        args = (data, _INDICES, _INDPTR) + ((_PERM,) if indexed else ())
        return op(*args, events, shape=(2, 3), transpose=transpose, backend='jax_raw')

    forward = jax.jacfwd(product)(weights)
    reverse = jax.jacrev(product)(weights)
    np.testing.assert_allclose(forward, reverse, rtol=1e-6)
    # The Jacobian of a binary product contains only the activity mask.
    assert set(np.unique(np.asarray(forward))) <= {0., 1.}


def _matrix_backends():
    backends = ['jax_raw']
    if jax.default_backend() == 'gpu':
        backends += ['cuda_raw', 'cusparse']
    return backends


def _matrix_expected(matrix, events, ct, left, matrix_events, weighted):
    coo = matrix.tocoo()
    rows, cols = np.asarray(coo.row), np.asarray(coo.col)
    factor = np.asarray(events)
    if not weighted:
        factor = (factor != 0).astype(np.float32)
    cotangent = np.asarray(ct)
    if not matrix_events:
        unit, target = (rows, cols) if left else (cols, rows)
        return factor[unit] * cotangent[target]
    if left:
        return np.sum(factor[:, rows] * cotangent[:, cols], axis=0)
    return np.sum(factor[cols, :] * cotangent[rows, :], axis=1)


@pytest.mark.parametrize('factory', [CSR.fromdense, CSC.fromdense, TCSR.fromdense],
                         ids=['CSR', 'CSC', 'TCSR'])
@pytest.mark.parametrize('weighted', [False, True], ids=['BinaryArray', 'WeightedArray'])
@pytest.mark.parametrize('matrix_events', [False, True], ids=['mv', 'mm'])
@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('backend', _matrix_backends())
def test_matrix_weight_gradient_follows_event_type(
    factory, weighted, matrix_events, left, backend,
):
    matrix = factory(_DENSE, backend=backend)
    events = jnp.array([2., -0.5] if left else [2., 0., -3.], jnp.float32)
    if matrix_events:
        events = jnp.stack([events, -2 * events], axis=0 if left else 1)
    wrap = WeightedArray if weighted else BinaryArray

    def product(item):
        return wrap(events) @ item if left else item @ wrap(events)

    values = events if weighted else (events != 0).astype(jnp.float32)
    expected_forward = values @ _DENSE if left else _DENSE @ values
    np.testing.assert_allclose(product(matrix), expected_forward, rtol=1e-6)
    ct = jnp.array(np.resize([5., -2., 3., 7.], expected_forward.shape), jnp.float32)
    expected = _matrix_expected(matrix, events, ct, left, matrix_events, weighted)

    def loss(data):
        return jnp.vdot(product(matrix.with_data(data)), ct)

    actual = jax.jit(jax.grad(loss))(matrix.data)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize('factory', [CSR.fromdense, CSC.fromdense, TCSR.fromdense],
                         ids=['CSR', 'CSC', 'TCSR'])
def test_matrices_have_no_backward_algorithm(factory):
    matrix = factory(jnp.eye(2, dtype=jnp.float32), backend='jax_raw', config=CSRConfig())
    variants = [
        matrix, matrix.with_data(matrix.data), matrix.T,
        jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(matrix))),
    ]
    assert not any(hasattr(item, 'backward_algorithm') for item in variants)
    with pytest.raises(TypeError):
        factory(jnp.eye(2, dtype=jnp.float32), backend='jax_raw',
                backward_algorithm='binary')


@pytest.mark.parametrize('algorithm', ['bptt', 'pp_prop', 'binary', 'weighted_float'])
def test_tcsr_backward_algorithm_raises_migration_error(algorithm):
    csr = CSR.fromdense(jnp.eye(2, dtype=jnp.float32), backend='jax_raw')
    with pytest.raises(TypeError, match='WeightedArray'):
        TCSR(csr, backward_algorithm=algorithm)


def test_csr_constructor_rejects_backward_algorithm():
    with pytest.raises(TypeError):
        CSR(jnp.ones(4, jnp.float32), _INDICES, _INDPTR, shape=(2, 3),
            backward_algorithm='binary')
