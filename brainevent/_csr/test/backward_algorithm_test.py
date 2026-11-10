# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Default event weight gradients and explicit backward-mode API contracts."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSC, CSR, TCSR
from brainevent._csr.config import CSRConfig
from brainevent._sddmm import dispatch


@pytest.mark.parametrize('matrix', [False, True], ids=['mv', 'mm'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('homogeneous', [False, True])
@pytest.mark.parametrize('lowlevel', [False, True])
def test_default_weight_gradient_uses_event_values(
    monkeypatch, matrix, transpose, indexed, homogeneous, lowlevel,
):
    """Non-unit events distinguish numeric gradients from activity masks."""
    api = importlib.import_module(
        'brainevent._csr.binary.binary_indexed'
        if indexed else 'brainevent._csr.binary.binary'
    )
    name = 'binary_csrmm' if matrix else 'binary_csrmv'
    name += '_indexed' if indexed else ''
    name += '_p_call' if lowlevel else ''
    op = getattr(api, name)
    indices = jnp.array([0, 2, 1, 2], jnp.int32)
    indptr = jnp.array([0, 2, 4], jnp.int32)
    rows = np.array([0, 0, 1, 1])
    permutation = jnp.array([2, 0, 3, 1], jnp.int32)
    weights = jnp.array([1.] if homogeneous else [1., 2., 3., 4.], jnp.float32)
    batch = 2 if matrix else 1
    units, targets = (2, 3) if transpose else (3, 2)
    events = jnp.array(
        np.resize([0., 2., -3., .5], (units, batch)), jnp.float32,
    )
    ct = jnp.array(
        np.resize([5., -2., 3., 7.], (targets, batch)), jnp.float32,
    )
    cols = np.asarray(indices)
    unit, target = (rows, cols) if transpose else (cols, rows)
    slots = np.sum(np.asarray(events)[unit] * np.asarray(ct)[target], axis=1)
    expected = np.zeros(weights.shape, np.float32)
    np.add.at(expected, np.zeros(4, int) if homogeneous else
              (np.asarray(permutation) if indexed else np.arange(4)), slots)

    seen = []
    original = dispatch.sddmm

    def record(*args, **kwargs):
        seen.append(kwargs['mode'])
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatch, 'sddmm', record)

    def loss(data):
        args = (data, indices, indptr)
        if indexed:
            args += (permutation,)
        output = op(*args, events if matrix else events[:, 0],
                    shape=(2, 3), transpose=transpose, backend='jax_raw')
        if lowlevel:
            output = output[0]
        return jnp.vdot(output, ct if matrix else ct[:, 0])

    actual = jax.jit(jax.grad(loss))(weights)
    np.testing.assert_allclose(actual, expected)
    assert seen and set(seen) == {'weighted'}


@pytest.mark.parametrize(
    'factory', [CSR.fromdense, CSC.fromdense, TCSR.fromdense],
)
def test_matrix_factories_default_to_weighted_float_and_preserve_it(factory):
    matrix = factory(jnp.array([[1., 0., 2.], [0., 3., 0.]], jnp.float32),
                     backend='jax_raw', config=CSRConfig())
    leaves, tree = jax.tree_util.tree_flatten(matrix)
    variants = [
        matrix, matrix.with_data(matrix.data), matrix.T,
        matrix.tocsr(), matrix.tocsc(),
        jax.tree_util.tree_unflatten(tree, leaves),
    ]
    assert all(
        item.backward_algorithm == 'weighted_float' for item in variants
    )


@pytest.mark.parametrize('algorithm', ['binary', 'weighted_float'])
@pytest.mark.parametrize(
    'factory', [CSR.fromdense, CSC.fromdense, TCSR.fromdense],
)
def test_matrix_factories_accept_explicit_modes(factory, algorithm):
    matrix = factory(jnp.eye(2, dtype=jnp.float32), backend='jax_raw',
                     backward_algorithm=algorithm, config=CSRConfig())
    assert matrix.backward_algorithm == algorithm


@pytest.mark.parametrize(
    'factory', [CSR.fromdense, CSC.fromdense, TCSR.fromdense],
)
@pytest.mark.parametrize('matrix_events', [False, True], ids=['mv', 'mm'])
@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('backend', ['jax_raw', 'cuda_raw', 'cusparse'])
def test_default_matrix_weight_gradients_use_event_values(
    factory, matrix_events, left, backend,
):
    if backend != 'jax_raw' and jax.default_backend() != 'gpu':
        pytest.skip('native backward integration requires a GPU')
    dense = jnp.array([[1., 0., 2.], [0., 3., 4.]], jnp.float32)
    matrix = factory(dense, backend=backend)
    events = jnp.array([2., -0.5] if left else [2., 0., -3.], jnp.float32)
    if matrix_events:
        events = jnp.stack([events, -2 * events], axis=0 if left else 1)

    def product(item):
        event_array = BinaryArray(events)
        return event_array @ item if left else item @ event_array

    activity = events != 0
    expected_forward = activity @ dense if left else dense @ activity
    np.testing.assert_allclose(product(matrix), expected_forward)
    ct = jnp.array(
        np.resize([5., -2., 3., 7.], expected_forward.shape), jnp.float32,
    )
    coo = matrix.tocoo()
    rows, cols = np.asarray(coo.row), np.asarray(coo.col)
    factor = np.asarray(events)
    cotangent = np.asarray(ct)
    if not matrix_events:
        unit, target = (rows, cols) if left else (cols, rows)
        expected = factor[unit] * cotangent[target]
    elif left:
        expected = np.sum(factor[:, rows] * cotangent[:, cols], axis=0)
    else:
        expected = np.sum(factor[cols, :] * cotangent[rows, :], axis=1)

    def loss(data):
        return jnp.vdot(product(matrix.with_data(data)), ct)

    actual = jax.jit(jax.grad(loss))(matrix.data)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    'algorithm', ['bptt', 'pp_prop', 'ppprop', 'weighted', 'unknown'],
)
def test_legacy_and_unknown_backward_modes_are_rejected(algorithm):
    with pytest.raises(ValueError, match='binary.*weighted_float'):
        CSR.fromdense(jnp.eye(2, dtype=jnp.float32), backend='jax_raw',
                      backward_algorithm=algorithm, config=CSRConfig())
