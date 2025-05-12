# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
import jax
import jax.numpy as jnp
import pytest

import brainevent
from brainevent._fixed_conn_num_test_util import (
    generate_data,
    vector_csr,
    matrix_csr,
    csr_vector,
    csr_matrix,
)

if brainstate.environ.get_platform() == 'cpu':
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


class TestVector:
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_vector_csr(self, replace, homo_w, shape):
        m, n = shape
        x = brainstate.random.rand(m)
        indices = generate_data(m, n, int(n * 0.1), replace=replace)

        data = 1.5 if homo_w else brainstate.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        y1 = jax.jit(lambda: x @ csr)()
        y2 = jax.jit(lambda: csr.T @ x)()

        y_true = vector_csr(x, csr.data, indices, (m, n))
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_csr_vector(self, replace, homo_w, shape):
        m, n = 20, 40
        v = brainstate.random.rand(n)
        indices = generate_data(m, n, int(n * 0.1), replace=replace)
        data = 1.5 if homo_w else brainstate.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        y1 = jax.jit(lambda: csr @ v)()
        y2 = jax.jit(lambda: v @ csr.T)()
        y_true = csr_vector(v, csr.data, indices, (m, n))
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    def _test_vjp(self, homo_w, replace, transpose, shape):
        n_in, n_out = shape
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)

        indices = generate_data(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = 1.5 if homo_w else brainstate.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r.sum()

        r1 = jax.jit(lambda: jax.grad(f_brainevent, argnums=(0, 1))(x, w))()

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, shape)
            else:
                r = csr_vector(x, w, indices, shape)
            return r.sum()

        r2 = jax.jit(lambda: jax.grad(f_jax, argnums=(0, 1))(x, w))()

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_vjp(self, replace, transpose, homo_w, shape):
        self._test_vjp(homo_w=homo_w, replace=replace, transpose=transpose, shape=shape)

    def _test_jvp(self, homo_w, replace, transpose, shape):
        n_in, n_out = shape
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in if transpose else n_out)

        indices = generate_data(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = 1.5 if homo_w else brainstate.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r

        o1, r1 = jax.jit(
            lambda: jax.jvp(
                f_brainevent,
                (x, w),
                (jnp.ones_like(x), jnp.ones_like(w))
            )
        )()

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, shape)
            else:
                r = csr_vector(x, w, indices, shape)
            return r

        o2, r2 = jax.jit(lambda: jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w))))()

        assert (jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_jvp(self, replace, transpose, homo_w, shape):
        self._test_jvp(homo_w=homo_w, replace=replace, transpose=transpose, shape=shape)

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    def test_batching_weight(self, replace, homo_w, shape, batch_size):
        m, n = shape
        indices = generate_data(m, n, int(n * 0.1), replace=replace)

        data = (
            brainstate.random.rand(batch_size)
            if homo_w else
            brainstate.init.Normal()((batch_size,) + indices.shape)
        )

        @jax.jit
        @jax.vmap
        def f_compare_vector_csr(w):
            csr = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = x @ csr
            y2 = csr.T @ x
            y_true = vector_csr(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        x = brainstate.random.rand(m)
        y1, y2, y_true = f_compare_vector_csr(data)
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

        @jax.jit
        @jax.vmap
        def f_compare_csr_vector(w):
            csr = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = csr @ x
            y2 = x @ csr.T
            y_true = csr_vector(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        x = brainstate.random.rand(n)
        y1, y2, y_true = f_compare_csr_vector(data)
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('batch_axis', [0, 1])
    def test_batching_vector(self, replace, homo_w, shape, batch_size, batch_axis):
        m, n = shape
        indices = generate_data(m, n, int(n * 0.1), replace=replace)

        data = (
            1.5
            if homo_w else
            brainstate.init.Normal()(indices.shape)
        )

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_vector_csr(x):
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = x @ csr
            y2 = csr.T @ x
            y_true = vector_csr(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        xs = brainstate.random.random([batch_size, m] if batch_axis == 0 else [m, batch_size])
        y1, y2, y_true = f_compare_vector_csr(xs)
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_csr_vector(x):
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = csr @ x
            y2 = x @ csr.T
            y_true = csr_vector(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        xs = brainstate.random.random([batch_size, n] if batch_axis == 0 else [n, batch_size])
        y1, y2, y_true = f_compare_csr_vector(xs)
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))


class TestMatrix:
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('k', [10])
    def test_matrix_csr(self, replace, homo_w, shape, k):
        m, n = shape
        x = brainstate.random.rand(k, m)
        indices = generate_data(m, n, int(n * 0.1), replace=replace)
        data = 1.5 if homo_w else brainstate.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        y1 = jax.jit(lambda: x @ csr)()
        y2 = jax.jit(lambda: (csr.T @ x.T).T)()
        y_true = matrix_csr(x, csr.data, indices, (m, n))
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('k', [10])
    def test_csr_matrix(self, replace, homo_w, shape, k):
        m, n = shape
        matrix = brainstate.random.rand(n, k)
        indices = generate_data(m, n, int(n * 0.1), replace=replace)
        data = 1.5 if homo_w else brainstate.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        y1 = jax.jit(lambda: csr @ matrix)()
        y2 = jax.jit(lambda: (matrix.T @ csr.T).T)()
        y_true = csr_matrix(matrix, csr.data, indices, (m, n))
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    def _test_vjp(self, homo_w, replace, transpose, shape, k):
        n_in, n_out = shape
        shape = (n_in, n_out)
        x = brainstate.random.rand(k, n_in) if transpose else brainstate.random.rand(n_out, k)

        indices = generate_data(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = 1.5 if homo_w else brainstate.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r.sum()

        r1 = jax.jit(lambda x, w: jax.grad(f_brainevent, argnums=(0, 1))(x, w))(x, w)

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, shape)
            else:
                r = csr_matrix(x, w, indices, shape)
            return r.sum()

        r2 = jax.jit(lambda x, w: jax.grad(f_jax, argnums=(0, 1))(x, w))(x, w)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('k', [10])
    def test_vjp(self, replace, transpose, homo_w, shape, k):
        self._test_vjp(homo_w=homo_w, replace=replace, transpose=transpose, shape=shape, k=k)

    def _test_jvp(self, homo_w, replace, transpose, shape, k):
        n_in, n_out = shape
        shape = (n_in, n_out)
        x = brainstate.random.random((k, n_in) if transpose else (n_out, k))

        indices = generate_data(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = 1.5 if homo_w else brainstate.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r

        o1, r1 = jax.jit(
            lambda: jax.jvp(
                f_brainevent,
                (x, w),
                (jnp.ones_like(x), jnp.ones_like(w))
            )
        )()

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, shape)
            else:
                r = csr_matrix(x, w, indices, shape)
            return r

        o2, r2 = jax.jit(lambda: jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w))))()

        assert (jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('k', [10])
    def test_jvp(self, replace, transpose, homo_w, shape, k):
        self._test_jvp(homo_w=homo_w, replace=replace, transpose=transpose, shape=shape, k=k)

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('k', [32])
    def test_batching_weight(self, replace, homo_w, shape, batch_size, k):
        m, n = shape
        indices = generate_data(m, n, int(n * 0.1), replace=replace)

        data = (
            brainstate.random.rand(batch_size)
            if homo_w else
            brainstate.init.Normal()((batch_size,) + indices.shape)
        )

        @jax.jit
        @jax.vmap
        def f_compare_matrix_csr(w):
            csr = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = x @ csr
            y2 = (csr.T @ x.T).T
            y_true = matrix_csr(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        x = brainstate.random.rand(k, m)
        y1, y2, y_true = f_compare_matrix_csr(data)
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

        @jax.jit
        @jax.vmap
        def f_compare_csr_vector(w):
            csr = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = csr @ x
            y2 = (x.T @ csr.T).T
            y_true = csr_matrix(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        x = brainstate.random.rand(n, k)
        y1, y2, y_true = f_compare_csr_vector(data)
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('k', [32])
    @pytest.mark.parametrize('batch_axis', [0, 1, 2])
    def test_batching_vector(self, replace, homo_w, shape, batch_size, k, batch_axis):
        m, n = shape
        indices = generate_data(m, n, int(n * 0.1), replace=replace)

        data = (
            1.5
            if homo_w else
            brainstate.init.Normal()(indices.shape)
        )

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_vector_csr(x):
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = x @ csr
            y2 = (csr.T @ x.T).T
            y_true = matrix_csr(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        if batch_axis == 0:
            shape = [batch_size, k, m]
        elif batch_axis == 1:
            shape = [k, batch_size, m]
        else:
            shape = [k, m, batch_size]
        xs = brainstate.random.random(shape)
        y1, y2, y_true = f_compare_vector_csr(xs)
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_csr_vector(x):
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = csr @ x
            y2 = (x.T @ csr.T).T
            y_true = csr_matrix(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        if batch_axis == 0:
            shape = [batch_size, n, k]
        elif batch_axis == 1:
            shape = [n, batch_size, k]
        else:
            shape = [n, k, batch_size]
        xs = brainstate.random.random(shape)
        y1, y2, y_true = f_compare_csr_vector(xs)
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))
