# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
import braintools
import jax
import jax.numpy as jnp
import pytest

from brainevent._csr.binary import binary_csrmv, binary_csrmv_p, binary_csrmm, binary_csrmm_p
from brainevent._csr.test_util import get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix

platform = jax.default_backend()
CSRMV_IMPLEMENTATIONS = tuple(binary_csrmv_p.available_backends(platform))
CSRMM_IMPLEMENTATIONS = tuple(binary_csrmm_p.available_backends(platform))


def _require_implementations(implementations, op_name: str):
    if not implementations:
        pytest.skip(f'No {op_name} implementation on platform={platform}')


def _vector_csr_api(x, data, indices, indptr, shape, implementation):
    return binary_csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=True,
        backend=implementation,
    )


def _csr_vector_api(x, data, indices, indptr, shape, implementation):
    return binary_csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
    )


def _matrix_csr_api(x, data, indices, indptr, shape, implementation):
    # x @ csr: csrmm expects input as [shape[0], cols] for transpose=True.
    return binary_csrmm(
        data,
        indices,
        indptr,
        x.T,
        shape=shape,
        transpose=True,
        backend=implementation,
    ).T


def _csr_matrix_api(x, data, indices, indptr, shape, implementation):
    return binary_csrmm(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
    )


class TestVectorCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        m, n = 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = vector_csr(x, data, indices, indptr, (m, n))

        for implementation in CSRMV_IMPLEMENTATIONS:
            y = _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((x, indptr, indices, y2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr_vmap_vector(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        with jax.checking_leaks():
            n_batch, m, n = 10, 20, 40
            xs = brainstate.random.rand(n_batch, m) < 0.1
            indptr, indices = get_csr(m, n, 0.1)

            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            y2 = brainstate.transform.vmap2(lambda x: vector_csr(x, data, indices, indptr, (m, n)))(xs)

            for implementation in CSRMV_IMPLEMENTATIONS:
                y = brainstate.transform.vmap2(
                    lambda x: _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
                )(xs)
                assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        m, n = 20, 40
        v = brainstate.random.rand(n) < 0.1
        indptr, indices = get_csr(m, n, 0.2)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = csr_vector(v, data, indices, indptr, (m, n))

        for implementation in CSRMV_IMPLEMENTATIONS:
            y = _csr_vector_api(v, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((v, indptr, indices, y2))

    def _test_vjp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)
        x = (x < 0.6).astype(float)

        indptr, indices = get_csr(n_in, n_out, 0.2, replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, shape, implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, shape, implementation)
            return r.sum()

        r = jax.grad(f_api, argnums=(0, 1))(x, w)

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, w)
        assert jnp.allclose(r[0], r2[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(r[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, homo_w, replace, transpose):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._test_vjp(
                implementation=implementation,
                homo_w=homo_w,
                replace=replace,
                transpose=transpose,
            )

    def _test_jvp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in if transpose else n_out)
        x = (x < 0.6).astype(float)

        indptr, indices = get_csr(n_in, n_out, 0.1, replace=replace)

        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, shape, implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, shape, implementation)
            return r

        o1, r1 = jax.jvp(f_api, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r

        o2, r2 = jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, o1, r1, o2, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, homo_w, replace, transpose):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._test_jvp(
                implementation=implementation,
                homo_w=homo_w,
                replace=replace,
                transpose=transpose,
            )


class TestBatchingVectorCSR:
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        implementation = self._implementation
        if transpose:
            y1 = _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
            y2 = vector_csr(x, data, indices, indptr, (m, n))
        else:
            y1 = _csr_vector_api(x, data, indices, indptr, (m, n), implementation)
            y2 = csr_vector(x, data, indices, indptr, (m, n))
        return jnp.allclose(y1, y2)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda x: self._run(x, data, indices, indptr, m, n))(xs)
            assert jnp.all(res)

        jax.block_until_ready((xs, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda data: self._run(x, data, indices, indptr, m, n))(data)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda ind: self._run(x, data, ind, indptr, m, n))(indices)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, (m, n), implementation)
            return r.sum()

        r1 = jax.grad(f_api, argnums=(0, 1))(x, data)

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_vector(x, w, indices, indptr, shape=(m, n))
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, data)

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_vjp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda x: self._run_vjp(x, data, indices, indptr, m, n))(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_vjp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_vjp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_vjp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_vjp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, (m, n), implementation)
            return r

        r1 = jax.jvp(f_api, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_vector(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_jvp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda x: self._run_jvp(x, data, indices, indptr, m, n))(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_jvp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_jvp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_jvp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_jvp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))


class TestMatrixCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = matrix_csr(x, data, indices, indptr, (m, n))

        for implementation in CSRMM_IMPLEMENTATIONS:
            y = _matrix_csr_api(x, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, y2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        m, n, k = 20, 40, 10
        matrix = brainstate.random.rand(n, k) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = csr_matrix(matrix, data, indices, indptr, (m, n))

        for implementation in CSRMM_IMPLEMENTATIONS:
            y = _csr_matrix_api(matrix, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2)

        jax.block_until_ready((matrix, indptr, indices, y2))


class TestBatchingMatrixCSR:
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        implementation = self._implementation
        if transpose:
            y1 = _matrix_csr_api(x, data, indices, indptr, (m, n), implementation)
            y2 = matrix_csr(x, data, indices, indptr, (m, n))
        else:
            y1 = _csr_matrix_api(x, data, indices, indptr, (m, n), implementation)
            y2 = csr_matrix(x, data, indices, indptr, (m, n))
        return jnp.allclose(y1, y2)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_matrix(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        xs = brainstate.random.rand(b, k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda x: self._run(x, data, indices, indptr, m, n))(xs)
            assert jnp.all(res)

        jax.block_until_ready((xs, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda data: self._run(x, data, indices, indptr, m, n))(data)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda ind: self._run(x, data, ind, indptr, m, n))(indices)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _matrix_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_matrix_api(x, w, indices, indptr, (m, n), implementation)
            return r.sum()

        r1 = jax.grad(f_api, argnums=(0, 1))(x, data)

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_matrix(x, w, indices, indptr, shape=(m, n))
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, data)

        return r1, r2

    @pytest.mark.parametrize('transpose', [True, False])
    def test_vmap_matrix_vjp(self, transpose):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        if transpose:
            xs = brainstate.random.rand(b, n, m) < 0.1
        else:
            xs = brainstate.random.rand(b, n, k) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(
                lambda x: self._run_vjp(x, data, indices, indptr, m, n, transpose=transpose)
            )(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_vjp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_vjp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_vjp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_vjp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _matrix_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_matrix_api(x, w, indices, indptr, (m, n), implementation)
            return r

        r1 = jax.jvp(f_api, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_matrix(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_jvp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        xs = brainstate.random.rand(b, k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda x: self._run_jvp(x, data, indices, indptr, m, n))(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_jvp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_jvp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_jvp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_jvp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))
