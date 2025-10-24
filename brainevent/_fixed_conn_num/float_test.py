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
import pytest

import brainevent
from brainevent._test_util import (
    generate_fixed_conn_num_indices,
    vector_fcn,
    matrix_fcn,
    fcn_vector,
    fcn_matrix,
    allclose,
    ones_like,
)

brainevent.config.numba_environ_set(parallel_if_possible=False)

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


def _remove_event_array(x):
    if isinstance(x, brainevent.EventArray):
        return x.data
    return x


class TestVector:
    def _generate_x(self, shape, require_float=False):
        if isinstance(shape, (tuple, list)):
            yield brainstate.random.rand(*shape)
        else:
            yield brainstate.random.rand(shape)

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_vector_csr(self, replace, homo_w, shape):
        m, n = shape
        for x in self._generate_x(m):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)

            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = jax.jit(lambda: x @ csr)()
            y2 = jax.jit(lambda: csr.T @ x)()
            y3 = _remove_event_array(x) @ csr.todense()

            y_true = vector_fcn(x, csr.data, indices, (m, n))
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y3, y_true, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_csr_vector(self, replace, homo_w, shape):
        m, n = 20, 40
        for v in self._generate_x(n):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = jax.jit(lambda: csr @ v)()
            y2 = jax.jit(lambda: v @ csr.T)()
            y_true = fcn_vector(v, csr.data, indices, (m, n))
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    def _test_vjp(self, homo_w, replace, transpose, shape):
        n_in, n_out = shape
        shape = (n_in, n_out)

        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r.sum()

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = vector_fcn(x, w, indices, shape)
            else:
                r = fcn_vector(x, w, indices, shape)
            return r.sum()

        for x in self._generate_x(n_in if transpose else n_out, require_float=True):
            r1 = jax.jit(lambda x, w: jax.grad(f_brainevent, argnums=(0, 1))(x, w))(x, w)
            r2 = jax.jit(lambda x, w: jax.grad(f_jax, argnums=(0, 1))(x, w))(x, w)

            assert (allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
            assert (allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    def test_vjp(self, replace, transpose, homo_w, shape):
        self._test_vjp(homo_w=homo_w, replace=replace, transpose=transpose, shape=shape)

    def _test_jvp(self, homo_w, replace, transpose, shape):
        n_in, n_out = shape
        shape = (n_in, n_out)

        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = vector_fcn(x, w, indices, shape)
            else:
                r = fcn_vector(x, w, indices, shape)
            return r

        for x in self._generate_x(n_in if transpose else n_out, require_float=True):
            o1, r1 = jax.jit(
                lambda x, w: jax.jvp(
                    f_brainevent,
                    (x, w),
                    (ones_like(x), ones_like(w))
                )
            )(x, w)
            o2, r2 = jax.jit(
                lambda x, w: jax.jvp(
                    f_jax,
                    (x, w),
                    (ones_like(x), ones_like(w))
                )
            )(x, w)

            assert (allclose(r1, r2, rtol=1e-3, atol=1e-3))
            assert (allclose(o1, o2, rtol=1e-3, atol=1e-3))

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
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)

        data = (
            brainstate.random.rand(batch_size)
            if homo_w else
            braintools.init.Normal(0., 1.)((batch_size,) + indices.shape)
        )

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def f_compare_vector_csr(w, x):
            csr = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = x @ csr
            y2 = csr.T @ x
            y_true = vector_fcn(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        for x in self._generate_x(m):
            y1, y2, y_true = f_compare_vector_csr(data, x)
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def f_compare_csr_vector(w, x):
            csr = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = csr @ x
            y2 = x @ csr.T
            y_true = fcn_vector(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        for x in self._generate_x(n):
            y1, y2, y_true = f_compare_csr_vector(data, x)
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('batch_axis', [0, 1])
    def test_batching_vector(self, replace, homo_w, shape, batch_size, batch_axis):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)

        data = (
            1.5
            if homo_w else
            braintools.init.Normal(0., 1.)(indices.shape)
        )

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_vector_csr(x):
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = x @ csr
            y2 = csr.T @ x
            y_true = vector_fcn(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        for xs in self._generate_x([batch_size, m] if batch_axis == 0 else [m, batch_size]):
            y1, y2, y_true = f_compare_vector_csr(xs)
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_csr_vector(x):
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = csr @ x
            y2 = x @ csr.T
            y_true = fcn_vector(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        for xs in self._generate_x([batch_size, n] if batch_axis == 0 else [n, batch_size]):
            y1, y2, y_true = f_compare_csr_vector(xs)
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))


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
    def test_matrix_csr(self, replace, homo_w, shape, k):
        m, n = shape
        for x in self._generate_x([k, m]):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = jax.jit(lambda: x @ csr)()
            y2 = jax.jit(lambda: (csr.T @ x.T).T)()
            y_true = matrix_fcn(x, csr.data, indices, (m, n))
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('k', [10])
    def test_csr_matrix(self, replace, homo_w, shape, k):
        m, n = shape
        for matrix in self._generate_x([n, k]):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = jax.jit(lambda: csr @ matrix)()
            y2 = jax.jit(lambda: (matrix.T @ csr.T).T)()
            y_true = fcn_matrix(matrix, csr.data, indices, (m, n))
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    def _test_vjp(self, homo_w, replace, transpose, shape, k):
        n_in, n_out = shape
        shape = (n_in, n_out)

        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r.sum()

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = matrix_fcn(x, w, indices, shape)
            else:
                r = fcn_matrix(x, w, indices, shape)
            return r.sum()

        for x in self._generate_x([k, n_in] if transpose else [n_out, k], require_float=True):
            r1 = jax.jit(lambda x, w: jax.grad(f_brainevent, argnums=(0, 1))(x, w))(x, w)
            r2 = jax.jit(lambda x, w: jax.grad(f_jax, argnums=(0, 1))(x, w))(x, w)

            assert (allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
            assert (allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

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

        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1), replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = matrix_fcn(x, w, indices, shape)
            else:
                r = fcn_matrix(x, w, indices, shape)
            return r

        for x in self._generate_x((k, n_in) if transpose else (n_out, k), require_float=True):
            o1, r1 = jax.jit(
                lambda x, w: jax.jvp(
                    f_brainevent,
                    (x, w),
                    (ones_like(x), ones_like(w))
                )
            )(x, w)
            o2, r2 = jax.jit(
                lambda x, w: jax.jvp(f_jax, (x, w), (ones_like(x), ones_like(w)))
            )(x, w)

            assert (allclose(r1, r2, rtol=1e-3, atol=1e-3))
            assert (allclose(o1, o2, rtol=1e-3, atol=1e-3))

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
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)

        data = (
            brainstate.random.rand(batch_size)
            if homo_w else
            braintools.init.Normal(0., 1.)((batch_size,) + indices.shape)
        )

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def f_compare_matrix_csr(w, x):
            csr = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = x @ csr
            y2 = (csr.T @ x.T).T
            y_true = matrix_fcn(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        for x in self._generate_x([k, m]):
            y1, y2, y_true = f_compare_matrix_csr(data, x)
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def f_compare_csr_vector(w, x):
            csr = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = csr @ x
            y2 = (x.T @ csr.T).T
            y_true = fcn_matrix(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        for x in self._generate_x([n, k]):
            y1, y2, y_true = f_compare_csr_vector(data, x)
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('k', [32])
    @pytest.mark.parametrize('batch_axis', [0, 1, 2])
    def test_batching_vector(self, replace, homo_w, shape, batch_size, k, batch_axis):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)

        data = (
            1.5
            if homo_w else
            braintools.init.Normal(0., 1.)(indices.shape)
        )

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_vector_csr(x):
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = x @ csr
            y2 = (csr.T @ x.T).T
            y_true = matrix_fcn(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        if batch_axis == 0:
            shape = [batch_size, k, m]
        elif batch_axis == 1:
            shape = [k, batch_size, m]
        else:
            shape = [k, m, batch_size]
        for xs in self._generate_x(shape):
            y1, y2, y_true = f_compare_vector_csr(xs)
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_csr_vector(x):
            csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = csr @ x
            y2 = (x.T @ csr.T).T
            y_true = fcn_matrix(x, csr.data, indices, (m, n))
            return y1, y2, y_true

        if batch_axis == 0:
            shape = [batch_size, n, k]
        elif batch_axis == 1:
            shape = [n, batch_size, k]
        else:
            shape = [n, k, batch_size]
        for xs in self._generate_x(shape):
            y1, y2, y_true = f_compare_csr_vector(xs)
            assert (allclose(y1, y_true, rtol=1e-3, atol=1e-3))
            assert (allclose(y2, y_true, rtol=1e-3, atol=1e-3))
