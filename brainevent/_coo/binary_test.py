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
import numpy as np
import pytest

from brainevent._coo.binary import binary_coomv, binary_coomv_p, binary_coomm, binary_coomm_p
from brainevent._coo.test_util import _get_coo, vector_coo, matrix_coo, coo_vector, coo_matrix

PLATFORM = jax.default_backend()
COOMV_IMPLEMENTATIONS = tuple(binary_coomv_p.available_backends(PLATFORM))
COOMM_IMPLEMENTATIONS = tuple(binary_coomm_p.available_backends(PLATFORM))

if not COOMV_IMPLEMENTATIONS:
    pytest.skip(f'No binary_coomv implementation on platform={PLATFORM}', allow_module_level=True)
if not COOMM_IMPLEMENTATIONS:
    pytest.skip(f'No binary_coomm implementation on platform={PLATFORM}', allow_module_level=True)


class TestVectorCOO:
    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_coo(self, implementation, replace, homo_w):
        m, n = 20, 40
        x = brainstate.random.rand(m) < 0.1
        row, col = _get_coo(m, n, 0.1, replace=replace)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = binary_coomv(data, row, col, x, shape=(m, n), transpose=True, backend=implementation)
        y2 = vector_coo(x, data, row, col, (m, n))
        assert jax.block_until_ready(jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))
        jax.block_until_ready((x, row, col, y, y2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_coo_vmap_vector(self, implementation, homo_w):
        n_batch, m, n = 10, 20, 40
        xs = brainstate.random.rand(n_batch, m) < 0.1
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = jax.vmap(
            lambda x: binary_coomv(data, row, col, x, shape=(m, n), transpose=True, backend=implementation)
        )(xs)
        y2 = jax.vmap(lambda x: vector_coo(x, data, row, col, (m, n)))(xs)
        assert jax.block_until_ready(jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))
        jax.block_until_ready((xs, row, col, y, y2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coo_vector(self, implementation, replace, homo_w):
        m, n = 20, 40
        v = brainstate.random.rand(n) < 0.1
        row, col = _get_coo(m, n, 0.2, replace=replace)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = binary_coomv(data, row, col, v, shape=(m, n), transpose=False, backend=implementation)
        y2 = coo_vector(v, data, row, col, (m, n))
        assert jax.block_until_ready(jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))
        jax.block_until_ready((v, row, col, y, y2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_coomv_single_nnz(self, implementation, homo_w, transpose):
        m, n = 10, 20
        row = np.array([3], dtype=np.int32)
        col = np.array([7], dtype=np.int32)
        if transpose:
            x = jnp.zeros(m, dtype=jnp.bool_).at[3].set(True)
        else:
            x = jnp.zeros(n, dtype=jnp.bool_).at[7].set(True)
        data = 2.5 if homo_w else jnp.array([2.5])
        y = binary_coomv(data, row, col, x, shape=(m, n), transpose=transpose, backend=implementation)
        y2 = vector_coo(x, data, row, col, (m, n)) if transpose else coo_vector(x, data, row, col, (m, n))
        assert jax.block_until_ready(jnp.allclose(y, y2, atol=1e-6))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('transpose', [True, False])
    def test_coomv_empty(self, implementation, transpose):
        m, n = 10, 20
        row = np.array([], dtype=np.int32)
        col = np.array([], dtype=np.int32)
        x = jnp.ones(m if transpose else n, dtype=jnp.bool_)
        data = jnp.array([1.0])
        y = binary_coomv(data, row, col, x, shape=(m, n), transpose=transpose, backend=implementation)
        expected_size = n if transpose else m
        assert jax.block_until_ready(jnp.allclose(y, jnp.zeros(expected_size)))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('nnz', [32, 33])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coomv_block_boundary(self, implementation, nnz, homo_w):
        m, n = 100, 200
        rng = np.random.default_rng(42)
        row = rng.integers(0, m, size=nnz, dtype=np.int32)
        col = rng.integers(0, n, size=nnz, dtype=np.int32)
        x = jnp.asarray(rng.random(n) > 0.5, dtype=jnp.bool_)
        data = 1.5 if homo_w else jnp.asarray(rng.standard_normal(nnz), dtype=jnp.float32)
        y = binary_coomv(data, row, col, x, shape=(m, n), transpose=False, backend=implementation)
        y2 = coo_vector(x, data, row, col, (m, n))
        assert jax.block_until_ready(jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coomv_heavy_duplicates(self, implementation, homo_w):
        m, n, nnz = 5, 5, 200
        row = np.zeros(nnz, dtype=np.int32)
        col = np.zeros(nnz, dtype=np.int32)
        x = jnp.ones(n, dtype=jnp.bool_)
        data = 1.0 if homo_w else jnp.ones(nnz, dtype=jnp.float32)
        y = binary_coomv(data, row, col, x, shape=(m, n), transpose=False, backend=implementation)
        y2 = coo_vector(x, data, row, col, (m, n))
        assert jax.block_until_ready(jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('shape', [(10000, 5), (5, 10000)])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coomv_extreme_shapes(self, implementation, shape, homo_w):
        m, n = shape
        nnz = 50
        rng = np.random.default_rng(123)
        row = rng.integers(0, m, size=nnz, dtype=np.int32)
        col = rng.integers(0, n, size=nnz, dtype=np.int32)
        x = jnp.asarray(rng.random(n) > 0.5, dtype=jnp.bool_)
        data = 1.5 if homo_w else jnp.asarray(rng.standard_normal(nnz), dtype=jnp.float32)
        y = binary_coomv(data, row, col, x, shape=(m, n), transpose=False, backend=implementation)
        y2 = coo_vector(x, data, row, col, (m, n))
        assert jax.block_until_ready(jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    def _test_vjp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)
        x = (x < 0.6).astype(float)

        row, col = _get_coo(n_in, n_out, 0.2, replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)

        def f_api(x, w):
            r = binary_coomv(w, row, col, x, shape=shape, transpose=transpose, backend=implementation)
            return r.sum()

        r = jax.grad(f_api, argnums=(0, 1))(x, w)

        def f_jax(x, w):
            if transpose:
                r = vector_coo(x, w, row, col, shape=shape)
            else:
                r = coo_vector(x, w, row, col, shape=shape)
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, w)
        assert jax.block_until_ready(jnp.allclose(r[0], r2[0], rtol=1e-3, atol=1e-3))
        assert jax.block_until_ready(jnp.allclose(r[1], r2[1], rtol=1e-3, atol=1e-3))
        jax.block_until_ready((x, row, col, r, r2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, implementation, transpose, replace, homo_w):
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

        row, col = _get_coo(n_in, n_out, 0.1, replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)

        def f_api(x, w):
            r = binary_coomv(w, row, col, x, shape=shape, transpose=transpose, backend=implementation)
            return r

        o1, r1 = jax.jvp(f_api, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        def f_jax(x, w):
            if transpose:
                r = vector_coo(x, w, row, col, shape=shape)
            else:
                r = coo_vector(x, w, row, col, shape=shape)
            return r

        o2, r2 = jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jax.block_until_ready(jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3))
        assert jax.block_until_ready(jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3))
        jax.block_until_ready((x, row, col, o1, r1, o2, r2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, implementation, transpose, replace, homo_w):
        self._test_jvp(
            implementation=implementation,
            homo_w=homo_w,
            replace=replace,
            transpose=transpose,
        )


class TestMatrixCOO:
    @pytest.mark.parametrize('implementation', COOMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_coo(self, implementation, homo_w):
        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = binary_coomm(data, row, col, x.T, shape=(m, n), transpose=True, backend=implementation).T
        y2 = matrix_coo(x.astype(float), data, row, col, (m, n))
        assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((x, row, col, y, y2))

    @pytest.mark.parametrize('implementation', COOMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coo_matrix(self, implementation, homo_w):
        m, n, k = 20, 40, 10
        x = brainstate.random.rand(n, k) < 0.1
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = binary_coomm(data, row, col, x, shape=(m, n), transpose=False, backend=implementation)
        y2 = coo_matrix(x.astype(float), data, row, col, (m, n))
        assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((x, row, col, y, y2))
