# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

import brainstate
import jax.numpy as jnp

import brainevent
from brainevent._test_util import (
    generate_fixed_conn_num_indices,
    vector_fcn,
    matrix_fcn,
    fcn_vector,
    fcn_matrix,
    allclose,
    gen_events,
    ones_like,
)


class TestGenerateFixedConnNumIndices:
    def test_shape_with_replace(self):
        n_pre, n_post, n_conn = 10, 20, 5
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        assert indices.shape == (n_pre, n_conn)

    def test_shape_without_replace(self):
        n_pre, n_post, n_conn = 10, 20, 5
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=False)
        assert indices.shape == (n_pre, n_conn)

    def test_values_in_range(self):
        n_pre, n_post, n_conn = 10, 20, 5
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < n_post)

    def test_without_replace_unique_per_row(self):
        n_pre, n_post, n_conn = 5, 20, 10
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=False)
        for i in range(n_pre):
            row = indices[i]
            unique_vals = jnp.unique(row)
            assert len(unique_vals) == n_conn

    def test_custom_rng(self):
        n_pre, n_post, n_conn = 5, 10, 3
        rng = brainstate.random.RandomState(42)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True, rng=rng)
        assert indices.shape == (n_pre, n_conn)


class TestVectorFcn:
    def test_homogeneous_weights(self):
        n_pre, n_post, n_conn = 10, 20, 5
        x = jnp.ones(n_pre)
        weights = jnp.array(2.0)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = vector_fcn(x, weights, indices, shape)
        assert result.shape == (n_post,)

    def test_heterogeneous_weights(self):
        n_pre, n_post, n_conn = 10, 20, 5
        x = jnp.ones(n_pre)
        weights = jnp.ones((n_pre, n_conn))
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = vector_fcn(x, weights, indices, shape)
        assert result.shape == (n_post,)

    def test_with_event_array_input(self):
        n_pre, n_post, n_conn = 10, 20, 5
        x = brainevent.EventArray(jnp.ones(n_pre))
        weights = jnp.array(1.0)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = vector_fcn(x, weights, indices, shape)
        assert result.shape == (n_post,)

    def test_zero_vector(self):
        n_pre, n_post, n_conn = 10, 20, 5
        x = jnp.zeros(n_pre)
        weights = jnp.array(1.0)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = vector_fcn(x, weights, indices, shape)
        assert jnp.allclose(result, jnp.zeros(n_post))


class TestMatrixFcn:
    def test_homogeneous_weights(self):
        n_pre, n_post, n_conn = 10, 20, 5
        batch_size = 3
        xs = jnp.ones((batch_size, n_pre))
        weights = jnp.array(2.0)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = matrix_fcn(xs, weights, indices, shape)
        assert result.shape == (batch_size, n_post)

    def test_heterogeneous_weights(self):
        n_pre, n_post, n_conn = 10, 20, 5
        batch_size = 3
        xs = jnp.ones((batch_size, n_pre))
        weights = jnp.ones((n_pre, n_conn))
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = matrix_fcn(xs, weights, indices, shape)
        assert result.shape == (batch_size, n_post)

    def test_with_event_array_input(self):
        n_pre, n_post, n_conn = 10, 20, 5
        batch_size = 3
        xs = brainevent.EventArray(jnp.ones((batch_size, n_pre)))
        weights = jnp.array(1.0)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = matrix_fcn(xs, weights, indices, shape)
        assert result.shape == (batch_size, n_post)


class TestFcnVector:
    def test_homogeneous_weights(self):
        n_pre, n_post, n_conn = 10, 20, 5
        x = jnp.ones(n_post)
        weights = jnp.array(2.0)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = fcn_vector(x, weights, indices, shape)
        assert result.shape == (n_pre,)

    def test_heterogeneous_weights(self):
        n_pre, n_post, n_conn = 10, 20, 5
        x = jnp.ones(n_post)
        weights = jnp.ones((n_pre, n_conn))
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = fcn_vector(x, weights, indices, shape)
        assert result.shape == (n_pre,)

    def test_with_event_array_input(self):
        n_pre, n_post, n_conn = 10, 20, 5
        x = brainevent.EventArray(jnp.ones(n_post))
        weights = jnp.array(1.0)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = fcn_vector(x, weights, indices, shape)
        assert result.shape == (n_pre,)


class TestFcnMatrix:
    def test_homogeneous_weights(self):
        n_pre, n_post, n_conn = 10, 20, 5
        batch_size = 3
        xs = jnp.ones((n_post, batch_size))
        weights = jnp.array(2.0)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = fcn_matrix(xs, weights, indices, shape)
        assert result.shape == (n_pre, batch_size)

    def test_heterogeneous_weights(self):
        n_pre, n_post, n_conn = 10, 20, 5
        batch_size = 3
        xs = jnp.ones((n_post, batch_size))
        weights = jnp.ones((n_pre, n_conn))
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = fcn_matrix(xs, weights, indices, shape)
        assert result.shape == (n_pre, batch_size)

    def test_with_event_array_input(self):
        n_pre, n_post, n_conn = 10, 20, 5
        batch_size = 3
        xs = brainevent.EventArray(jnp.ones((n_post, batch_size)))
        weights = jnp.array(1.0)
        indices = generate_fixed_conn_num_indices(n_pre, n_post, n_conn, replace=True)
        shape = (n_pre, n_post)

        result = fcn_matrix(xs, weights, indices, shape)
        assert result.shape == (n_pre, batch_size)


class TestAllclose:
    def test_identical_arrays(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        assert allclose(x, y)

    def test_close_arrays(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.00001, 2.00001, 3.00001])
        assert allclose(x, y)

    def test_not_close_arrays(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.1, 2.1, 3.1])
        assert not allclose(x, y)

    def test_with_event_arrays(self):
        x = brainevent.EventArray(jnp.array([1.0, 2.0, 3.0]))
        y = brainevent.EventArray(jnp.array([1.0, 2.0, 3.0]))
        assert allclose(x, y)

    def test_mixed_event_and_jax_array(self):
        x = brainevent.EventArray(jnp.array([1.0, 2.0, 3.0]))
        y = jnp.array([1.0, 2.0, 3.0])
        assert allclose(x, y)

    def test_custom_tolerances(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.01, 2.01, 3.01])
        assert not allclose(x, y, rtol=1e-4, atol=1e-4)
        assert allclose(x, y, rtol=0.1, atol=0.1)


class TestGenEvents:
    def test_shape(self):
        shape = (10, 20)
        events = gen_events(shape)
        assert events.shape == shape

    def test_returns_event_array(self):
        events = gen_events((10,))
        assert isinstance(events, brainevent.EventArray)

    def test_asbool_true(self):
        events = gen_events((100,), asbool=True)
        assert events.dtype == jnp.bool_

    def test_asbool_false(self):
        events = gen_events((100,), asbool=False)
        assert events.dtype == jnp.float32 or events.dtype == jnp.float64

    def test_probability(self):
        shape = (10000,)
        prob = 0.3
        events = gen_events(shape, prob=prob, asbool=True)
        actual_prob = jnp.mean(events.data.astype(float))
        assert jnp.abs(actual_prob - prob) < 0.05

    def test_high_probability(self):
        events = gen_events((10000,), prob=0.9)
        actual_prob = jnp.mean(events.data.astype(float))
        assert actual_prob > 0.85

    def test_low_probability(self):
        events = gen_events((10000,), prob=0.1)
        actual_prob = jnp.mean(events.data.astype(float))
        assert actual_prob < 0.15


class TestOnesLike:
    def test_simple_array(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result = ones_like(x)
        assert jnp.allclose(result, jnp.ones(3))

    def test_2d_array(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = ones_like(x)
        assert jnp.allclose(result, jnp.ones((2, 2)))

    def test_preserves_shape(self):
        x = jnp.zeros((5, 10, 3))
        result = ones_like(x)
        assert result.shape == x.shape

    def test_preserves_dtype(self):
        x = jnp.array([1, 2, 3], dtype=jnp.int32)
        result = ones_like(x)
        assert result.dtype == jnp.int32

    def test_pytree_dict(self):
        x = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0, 4.0, 5.0])}
        result = ones_like(x)
        assert jnp.allclose(result['a'], jnp.ones(2))
        assert jnp.allclose(result['b'], jnp.ones(3))

    def test_pytree_nested(self):
        x = {'a': jnp.array([1.0]), 'b': {'c': jnp.array([2.0, 3.0])}}
        result = ones_like(x)
        assert jnp.allclose(result['a'], jnp.ones(1))
        assert jnp.allclose(result['b']['c'], jnp.ones(2))

    def test_pytree_list(self):
        x = [jnp.array([1.0, 2.0]), jnp.array([3.0])]
        result = ones_like(x)
        assert jnp.allclose(result[0], jnp.ones(2))
        assert jnp.allclose(result[1], jnp.ones(1))
