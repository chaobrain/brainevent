# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""TCSR inherits the CSR operations and their public keyword contracts."""

import inspect

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import CSC, CSR, TCSR


def _pair(*, shared=False, unit=None):
    data = jnp.array([2.] if shared else [2., 3., 4., 5., 6.], jnp.float32)
    if unit is not None:
        data = data * unit
    source = CSR(
        data, jnp.array([3, 1, 1, 2, 0], jnp.int32),
        jnp.array([0, 3, 4, 5], jnp.int32), shape=(3, 4), backend='jax_raw',
    )
    tiled = TCSR(source)
    reference = CSR(
        tiled.data, tiled.indices, tiled.indptr, shape=tiled.shape,
        backend=tiled.backend, config=tiled.config,
    ).build_weight_indices()
    return tiled, reference


def _values(value):
    return np.asarray(u.get_mantissa(value))


def _assert_equal(actual, expected):
    assert u.get_unit(actual) == u.get_unit(expected)
    np.testing.assert_allclose(_values(actual), _values(expected), rtol=1e-5, atol=1e-6)


def test_inherited_keyword_names_defaults_and_call_kinds_match_csr():
    names = (
        'dt2t', 'dt2t_transposed', 'update_on_pre', 'update_on_post',
        'apply_on_pre', 'apply_on_post', 'solve', 'with_data', 'apply',
        'build_weight_indices', 'with_rebuilt_config', 'prepare_diag_add',
        'diag_add', 'transpose', 'tocsr', 'tocsc', 'tocoo', 'sum',
    )
    for name in names:
        def contract(cls):
            return tuple((p.name, p.kind, p.default)
                         for p in inspect.signature(getattr(cls, name)).parameters.values())

        assert contract(TCSR) == contract(CSR), name


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('with_units', [False, True])
def test_dt2t_keywords_canonical_edge_order_units_and_jit(transpose, with_units):
    tiled, reference = _pair()
    name = 'dt2t_transposed' if transpose else 'dt2t'
    values = jnp.arange(tiled.shape[1 if transpose else 0], dtype=jnp.float32) + .5
    weights = jnp.arange(tiled.nse, dtype=jnp.float32) + 1
    if with_units:
        values, weights = values * u.ms, weights * u.mV
    expected = getattr(reference, name)(y_dim_arr=values, w_dim_arr=weights)
    actual = jax.jit(lambda m, y, w: getattr(m, name)(
        y_dim_arr=y, w_dim_arr=w,
    ))(tiled, values, weights)
    _assert_equal(actual, expected)
    edge_ids = (np.asarray(tiled.indices) if transpose else
                np.repeat(np.arange(tiled.shape[0]), np.diff(np.asarray(tiled.indptr))))
    np.testing.assert_allclose(_values(actual), _values(weights) * _values(values)[edge_ids])


@pytest.mark.parametrize('transpose', [False, True])
def test_dt2t_requires_per_edge_values_like_csr(transpose):
    tiled, reference = _pair(shared=True)
    name = 'dt2t_transposed' if transpose else 'dt2t'
    y = jnp.ones(tiled.shape[1 if transpose else 0], jnp.float32)
    for matrix in (reference, tiled):
        with pytest.raises(AssertionError):
            getattr(matrix, name)(y_dim_arr=y, w_dim_arr=matrix.data)


@pytest.mark.parametrize('on_pre', [False, True])
@pytest.mark.parametrize('with_units', [False, True])
def test_additive_plasticity_keywords_clipping_units_and_tile_lifecycle(on_pre, with_units):
    unit = u.mV if with_units else None
    tiled, reference = _pair(unit=unit)
    spikes = jnp.array([2., 0., -1.], jnp.float32) if on_pre else jnp.array([0., -1., 2., 0.], jnp.float32)
    trace = jnp.array([.5, -1., 2., .25], jnp.float32) if on_pre else jnp.array([.5, -1., 2.], jnp.float32)
    lower, upper = jnp.float32(1), jnp.float32(6)
    if with_units:
        trace, lower, upper = trace * unit, lower * unit, upper * unit
    name = 'update_on_pre' if on_pre else 'update_on_post'
    kwargs = ({'pre_spike': spikes, 'post_trace': trace} if on_pre else
              {'pre_trace': trace, 'post_spike': spikes})
    kwargs.update(w_min=lower, w_max=upper)
    original = _values(tiled.data).copy()
    expected = getattr(reference, name)(**kwargs)
    eager = getattr(tiled, name)(**kwargs)
    assert type(eager) is TCSR and eager._cache is tiled._cache
    assert eager._cache.tile_geometries is tiled._cache.tile_geometries
    _assert_equal(eager.data, expected.data)
    compiled = jax.jit(lambda m: getattr(m, name)(**kwargs))(tiled)
    assert type(compiled) is TCSR
    _assert_equal(compiled.data, expected.data)
    assert compiled._cache.tile_geometries.keys() == tiled._cache.tile_geometries.keys()
    np.testing.assert_array_equal(_values(tiled.data), original)


@pytest.mark.parametrize('on_pre', [False, True])
def test_general_plasticity_explicit_pytree_edge_state_and_keyword_values(on_pre):
    tiled, reference = _pair(shared=True)
    state = {'weight': jnp.arange(tiled.nse, dtype=jnp.float32),
             'count': jnp.arange(tiled.nse, dtype=jnp.int32)}
    pre_values = {'x': jnp.array([1., 2., 3.], jnp.float32)}
    post_values = {'x': jnp.array([.5, 1., 1.5, 2.], jnp.float32)}
    spikes = jnp.array([True, False, True]) if on_pre else jnp.array([False, True, True, False])

    def rule(edge, pre, post, params):
        return {'weight': edge['weight'] + params * (pre['x'] + post['x']),
                'count': edge['count'] + 1}

    name = 'apply_on_pre' if on_pre else 'apply_on_post'
    keywords = {'pre_spike' if on_pre else 'post_spike': spikes}
    keywords.update(rule=rule, edge_state=state, pre_vals=pre_values,
                    post_vals=post_values, params=jnp.float32(.25))
    expected = getattr(reference, name)(**keywords)
    actual = jax.jit(lambda m: getattr(m, name)(**keywords))(tiled)
    for key in state:
        _assert_equal(actual[key], expected[key])
    rows = np.repeat(np.arange(tiled.shape[0]), np.diff(np.asarray(tiled.indptr)))
    columns = np.asarray(tiled.indices)
    active = np.asarray(spikes)[rows if on_pre else columns]
    proposed = _values(state['weight']) + .25 * (_values(pre_values['x'])[rows] + _values(post_values['x'])[columns])
    np.testing.assert_allclose(actual['weight'], np.where(active, proposed, state['weight']))
    np.testing.assert_array_equal(actual['count'], state['count'] + active)
    with pytest.raises(ValueError, match='homogeneous'):
        getattr(tiled, name)(rule=lambda e, pre, post, params: e, **{
            'pre_spike' if on_pre else 'post_spike': spikes,
        })


@pytest.mark.parametrize('on_pre', [False, True])
def test_general_plasticity_default_state_units_and_input_gradient(on_pre):
    tiled, reference = _pair(unit=u.mV)
    name = 'apply_on_pre' if on_pre else 'apply_on_post'
    spikes = jnp.array([True, False, True]) if on_pre else jnp.array([False, True, True, False])
    keywords = {'pre_spike' if on_pre else 'post_spike': spikes}
    rule = lambda edge, pre, post, params: edge + params
    expected = getattr(reference, name)(rule=rule, params=.5 * u.mV, **keywords)
    actual = getattr(tiled, name)(rule=rule, params=.5 * u.mV, **keywords)
    _assert_equal(actual, expected)
    plain = tiled.apply(u.get_mantissa)

    def loss(weights):
        return getattr(plain.with_data(weights), name)(
            rule=lambda e, pre, post, p: e * p, params=jnp.float32(2), **keywords,
        ).sum()

    gradient = jax.jit(jax.grad(loss))(plain.data)
    rows = np.repeat(np.arange(tiled.shape[0]), np.diff(np.asarray(tiled.indptr)))
    edge_neurons = rows if on_pre else np.asarray(tiled.indices)
    np.testing.assert_array_equal(gradient, np.where(np.asarray(spikes)[edge_neurons], 2., 1.))


@pytest.mark.parametrize('shared', [False, True])
def test_format_conversions_preserve_values_duplicates_units_and_parent_type_contract(shared):
    tiled, reference = _pair(shared=shared, unit=u.mV)
    assert tiled.tocsr() is tiled
    physical = tiled.tocsc()
    assert physical.shape == tiled.shape
    _assert_equal(physical.todense(), reference.todense())
    assert type(physical) is CSC and physical._cache.tile_geometries == {}
    coordinates, parent_coordinates = tiled.tocoo(), reference.tocoo()
    assert coordinates.shape == tiled.shape
    assert coordinates.data.size == tiled.nse
    _assert_equal(coordinates.data, parent_coordinates.data)
    np.testing.assert_array_equal(coordinates.row, parent_coordinates.row)
    np.testing.assert_array_equal(coordinates.col, parent_coordinates.col)
    # COO's downstream CUDA densifier overwrites duplicate coordinates for
    # both CSR and TCSR. Check the exported edges independently to distinguish
    # that existing limitation from a conversion dropping duplicate edges.
    dense = np.zeros(tiled.shape, dtype=np.float32)
    np.add.at(dense, (np.asarray(coordinates.row), np.asarray(coordinates.col)),
              _values(coordinates.data))
    np.testing.assert_allclose(dense, _values(reference.todense()))
    assert type(tiled.transpose(axes=None)) is CSC
    _assert_equal(tiled.transpose().todense(), reference.todense().T)
    assert type(tiled.T.T) is CSR


@pytest.mark.parametrize('with_units', [False, True])
def test_solve_inherits_keywords_units_and_jit_without_requiring_tcsr_override(with_units):
    dense = jnp.array([[4., 1., 0.], [1., 3., 0.], [0., 0., 2.]], jnp.float32)
    data = dense * u.siemens if with_units else dense
    rhs = jnp.array([1., 2., 3.], jnp.float32)
    if with_units:
        rhs = rhs * u.mA
    reference = CSR.fromdense(data, backend='jax_raw')
    tiled = TCSR(reference)
    expected = reference.solve(b=rhs, tol=1e-7, reorder=0)
    result = jax.jit(lambda m, b: m.solve(b=b, tol=1e-7, reorder=0))(tiled, rhs)
    _assert_equal(result, expected)
    np.testing.assert_allclose(_values(result), np.linalg.solve(np.asarray(dense), _values(rhs)), rtol=1e-5)
    assert tiled._cache.tile_geometries == {}  # solve never prepares event-MM tiles


def test_solve_rejects_invalid_shapes_as_parent():
    tiled, reference = _pair()
    for matrix in (tiled, reference):
        with pytest.raises(ValueError, match='square'):
            matrix.solve(b=jnp.ones(3))
    square = TCSR.fromdense(jnp.eye(3), backend='jax_raw')
    with pytest.raises(ValueError, match='vector'):
        square.solve(b=jnp.ones((3, 1)))
