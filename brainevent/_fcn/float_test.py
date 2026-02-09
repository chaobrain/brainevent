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

import functools

import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

from brainevent._fcn.float import fcnmv, fcnmv_p, fcnmm_p, fcnmm_p_call
from brainevent._test_util import (
    generate_fixed_conn_num_indices,
    vector_fcn,
    matrix_fcn,
    fcn_vector,
    fcn_matrix,
    allclose,
    ones_like,
)

platform = jax.default_backend()
FCNMV_IMPLEMENTATIONS = tuple(fcnmv_p.available_backends(platform))
FCNMM_IMPLEMENTATIONS = tuple(fcnmm_p.available_backends(platform))
FCNMV_PARAMS = FCNMV_IMPLEMENTATIONS or (None,)
FCNMM_PARAMS = FCNMM_IMPLEMENTATIONS or (None,)

if platform == 'cpu':
    shapes = [
        (20, 40),
        (50, 30),
    ]
else:
    shapes = [
        (20, 40),
        (50, 30),
        (200, 400),
        (500, 300),
        (2000, 4000),
        (5000, 3000),
    ]


def _make_data(homo_w, shape):
    if homo_w:
        return jnp.asarray(1.5, dtype=jnp.float32)
    return braintools.init.Normal(0.0, 1.0)(shape)


def _vector_fcn_api(x, data, indices, shape, implementation):
    return fcnmv(
        data,
        indices,
        x,
        shape=shape,
        transpose=True,
        backend=implementation,
    )


def _fcn_vector_api(x, data, indices, shape, implementation):
    return fcnmv(
        data,
        indices,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
    )


def _matrix_fcn_api(x, data, indices, shape, implementation):
    return fcnmm_p_call(
        data,
        indices,
        x.T,
        shape=shape,
        transpose=True,
        backend=implementation,
    )[0].T


def _fcn_matrix_api(x, data, indices, shape, implementation):
    return fcnmm_p_call(
        data,
        indices,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
    )[0]


@pytest.mark.skipif(
    not FCNMV_IMPLEMENTATIONS,
    reason=f'No fcnmv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', FCNMV_PARAMS)
class TestVector:
    def _generate_x(self, shape, require_float=False):
        if isinstance(shape, (tuple, list)):
            yield brainstate.random.rand(*shape)
        else:
            yield brainstate.random.rand(shape)

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_vector_csr(self, implementation, replace, homo_w, shape):
        m, n = shape
        for x in self._generate_x(m):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
            data = _make_data(homo_w, indices.shape)
            y = _vector_fcn_api(x, data, indices, (m, n), implementation)
            y_true = vector_fcn(x, data, indices, (m, n))
            assert allclose(y, y_true, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, data, y, y_true))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_csr_vector(self, implementation, replace, homo_w, shape):
        m, n = shape
        for v in self._generate_x(n):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
            data = _make_data(homo_w, indices.shape)
            y = _fcn_vector_api(v, data, indices, (m, n), implementation)
            y_true = fcn_vector(v, data, indices, (m, n))
            assert allclose(y, y_true, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((v, indices, data, y, y_true))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_vjp(self, implementation, replace, transpose, homo_w, shape):
        n_in, n_out = shape
        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x, data):
            if transpose:
                r = _vector_fcn_api(x, data, indices, shape, implementation)
            else:
                r = _fcn_vector_api(x, data, indices, shape, implementation)
            return r.sum()

        def f_ref(x, data):
            if transpose:
                r = vector_fcn(x, data, indices, shape)
            else:
                r = fcn_vector(x, data, indices, shape)
            return r.sum()

        for x in self._generate_x(n_in if transpose else n_out, require_float=True):
            r1 = jax.jit(lambda x, data: jax.grad(f_api, argnums=(0, 1))(x, data))(x, w)
            r2 = jax.jit(lambda x, data: jax.grad(f_ref, argnums=(0, 1))(x, data))(x, w)
            assert allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, w, r1[0], r1[1], r2[0], r2[1]))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_jvp(self, implementation, replace, transpose, homo_w, shape):
        n_in, n_out = shape
        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x, data):
            if transpose:
                return _vector_fcn_api(x, data, indices, shape, implementation)
            return _fcn_vector_api(x, data, indices, shape, implementation)

        def f_ref(x, data):
            if transpose:
                return vector_fcn(x, data, indices, shape)
            return fcn_vector(x, data, indices, shape)

        for x in self._generate_x(n_in if transpose else n_out, require_float=True):
            o1, r1 = jax.jit(
                lambda x, data: jax.jvp(f_api, (x, data), (ones_like(x), ones_like(data)))
            )(x, w)
            o2, r2 = jax.jit(
                lambda x, data: jax.jvp(f_ref, (x, data), (ones_like(x), ones_like(data)))
            )(x, w)
            assert allclose(r1, r2, rtol=1e-3, atol=1e-3)
            assert allclose(o1, o2, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, w, o1, r1, o2, r2))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    def test_batching_weight(self, implementation, replace, homo_w, shape, batch_size):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
        data = (
            brainstate.random.rand(batch_size)
            if homo_w else
            braintools.init.Normal(0., 1.)((batch_size,) + indices.shape)
        )

        x_left = brainstate.random.rand(m)
        y = jax.jit(jax.vmap(
            lambda w: _vector_fcn_api(x_left, w, indices, (m, n), implementation)
        ))(data)
        y_true = jax.jit(jax.vmap(
            lambda w: vector_fcn(x_left, w, indices, (m, n))
        ))(data)
        assert allclose(y, y_true, rtol=1e-3, atol=1e-3)

        x_right = brainstate.random.rand(n)
        y = jax.jit(jax.vmap(
            lambda w: _fcn_vector_api(x_right, w, indices, (m, n), implementation)
        ))(data)
        y_true = jax.jit(jax.vmap(
            lambda w: fcn_vector(x_right, w, indices, (m, n))
        ))(data)
        assert allclose(y, y_true, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((indices, data, x_left, x_right, y, y_true))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('batch_axis', [0, 1])
    def test_batching_vector(self, implementation, replace, homo_w, shape, batch_size, batch_axis):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
        data = _make_data(homo_w, indices.shape)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_vector(x):
            y = _vector_fcn_api(x, data, indices, (m, n), implementation)
            y_true = vector_fcn(x, data, indices, (m, n))
            return y, y_true

        xs = brainstate.random.rand(batch_size, m) if batch_axis == 0 else brainstate.random.rand(m, batch_size)
        y, y_true = f_compare_vector(xs)
        assert allclose(y, y_true, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((indices, data, xs, y, y_true))


@pytest.mark.skipif(
    not FCNMM_IMPLEMENTATIONS,
    reason=f'No fcnmm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
class TestMatrix:
    def _generate_x(self, shape, require_float=False):
        if isinstance(shape, (tuple, list)):
            yield brainstate.random.rand(*shape)
        else:
            yield brainstate.random.rand(shape)

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('k', [10])
    def test_matrix_csr(self, implementation, replace, homo_w, shape, k):
        m, n = shape
        for x in self._generate_x([k, m]):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
            data = _make_data(homo_w, indices.shape)
            y = _matrix_fcn_api(x, data, indices, (m, n), implementation)
            y_true = matrix_fcn(x, data, indices, (m, n))
            assert allclose(y, y_true, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, data, y, y_true))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('k', [10])
    def test_csr_matrix(self, implementation, replace, homo_w, shape, k):
        m, n = shape
        for matrix in self._generate_x([n, k]):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
            data = _make_data(homo_w, indices.shape)
            y = _fcn_matrix_api(matrix, data, indices, (m, n), implementation)
            y_true = fcn_matrix(matrix, data, indices, (m, n))
            assert allclose(y, y_true, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((matrix, indices, data, y, y_true))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('k', [10])
    def test_vjp(self, implementation, replace, transpose, homo_w, shape, k):
        n_in, n_out = shape
        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x, data):
            if transpose:
                r = _matrix_fcn_api(x, data, indices, shape, implementation)
            else:
                r = _fcn_matrix_api(x, data, indices, shape, implementation)
            return r.sum()

        def f_ref(x, data):
            if transpose:
                r = matrix_fcn(x, data, indices, shape)
            else:
                r = fcn_matrix(x, data, indices, shape)
            return r.sum()

        for x in self._generate_x([k, n_in] if transpose else [n_out, k], require_float=True):
            r1 = jax.jit(lambda x, data: jax.grad(f_api, argnums=(0, 1))(x, data))(x, w)
            r2 = jax.jit(lambda x, data: jax.grad(f_ref, argnums=(0, 1))(x, data))(x, w)
            assert allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, w, r1[0], r1[1], r2[0], r2[1]))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('k', [10])
    def test_jvp(self, implementation, replace, transpose, homo_w, shape, k):
        n_in, n_out = shape
        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x, data):
            if transpose:
                return _matrix_fcn_api(x, data, indices, shape, implementation)
            return _fcn_matrix_api(x, data, indices, shape, implementation)

        def f_ref(x, data):
            if transpose:
                return matrix_fcn(x, data, indices, shape)
            return fcn_matrix(x, data, indices, shape)

        for x in self._generate_x((k, n_in) if transpose else (n_out, k), require_float=True):
            o1, r1 = jax.jit(
                lambda x, data: jax.jvp(f_api, (x, data), (ones_like(x), ones_like(data)))
            )(x, w)
            o2, r2 = jax.jit(
                lambda x, data: jax.jvp(f_ref, (x, data), (ones_like(x), ones_like(data)))
            )(x, w)
            assert allclose(r1, r2, rtol=1e-3, atol=1e-3)
            assert allclose(o1, o2, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, w, o1, r1, o2, r2))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('k', [32])
    def test_batching_weight(self, implementation, replace, homo_w, shape, batch_size, k):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)

        data = (
            brainstate.random.rand(batch_size)
            if homo_w else
            braintools.init.Normal(0., 1.)((batch_size,) + indices.shape)
        )

        x_left = brainstate.random.rand(k, m)
        y = jax.jit(jax.vmap(
            lambda w: _matrix_fcn_api(x_left, w, indices, (m, n), implementation)
        ))(data)
        y_true = jax.jit(jax.vmap(
            lambda w: matrix_fcn(x_left, w, indices, (m, n))
        ))(data)
        assert allclose(y, y_true, rtol=1e-3, atol=1e-3)

        x_right = brainstate.random.rand(n, k)
        y = jax.jit(jax.vmap(
            lambda w: _fcn_matrix_api(x_right, w, indices, (m, n), implementation)
        ))(data)
        y_true = jax.jit(jax.vmap(
            lambda w: fcn_matrix(x_right, w, indices, (m, n))
        ))(data)
        assert allclose(y, y_true, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((indices, data, x_left, x_right, y, y_true))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('k', [32])
    @pytest.mark.parametrize('batch_axis', [0, 1, 2])
    def test_batching_vector(self, implementation, replace, homo_w, shape, batch_size, k, batch_axis):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
        data = _make_data(homo_w, indices.shape)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_matrix(x):
            y = _matrix_fcn_api(x, data, indices, (m, n), implementation)
            y_true = matrix_fcn(x, data, indices, (m, n))
            return y, y_true

        if batch_axis == 0:
            x_shape = [batch_size, k, m]
        elif batch_axis == 1:
            x_shape = [k, batch_size, m]
        else:
            x_shape = [k, m, batch_size]
        xs = brainstate.random.rand(*x_shape)
        y, y_true = f_compare_matrix(xs)
        assert allclose(y, y_true, rtol=1e-3, atol=1e-3)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_fcn_matrix(x):
            y = _fcn_matrix_api(x, data, indices, (m, n), implementation)
            y_true = fcn_matrix(x, data, indices, (m, n))
            return y, y_true

        if batch_axis == 0:
            x_shape = [batch_size, n, k]
        elif batch_axis == 1:
            x_shape = [n, batch_size, k]
        else:
            x_shape = [n, k, batch_size]
        xs = brainstate.random.rand(*x_shape)
        y, y_true = f_compare_fcn_matrix(xs)
        assert allclose(y, y_true, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((indices, data, xs, y, y_true))
