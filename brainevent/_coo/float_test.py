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


from importlib.metadata import version, PackageNotFoundError

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._coo.float import coomv, coomm, coomv_p, coomm_p
from brainevent._coo.test_util import _get_coo, vector_coo, matrix_coo, coo_vector, coo_matrix

PLATFORM = jax.default_backend()
COOMV_IMPLEMENTATIONS = tuple(coomv_p.available_backends(PLATFORM))
COOMM_IMPLEMENTATIONS = tuple(coomm_p.available_backends(PLATFORM))

if not COOMV_IMPLEMENTATIONS:
    pytest.skip(f'No coomv implementation on platform={PLATFORM}', allow_module_level=True)
if not COOMM_IMPLEMENTATIONS:
    pytest.skip(f'No coomm implementation on platform={PLATFORM}', allow_module_level=True)


class TestVectorCOO:
    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_coo(self, implementation, homo_w):
        m, n = 20, 40
        x = brainstate.random.rand(m)
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = coomv(data, row, col, x, shape=(m, n), transpose=True, backend=implementation)
        y2 = vector_coo(x, data, row, col, (m, n))
        assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((x, row, col, y, y2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coo_vector(self, implementation, homo_w):
        m, n = 20, 40
        v = brainstate.random.rand(n)
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = coomv(data, row, col, v, shape=(m, n), transpose=False, backend=implementation)
        y2 = coo_vector(v, data, row, col, (m, n))
        assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((v, row, col, y, y2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_coo_vmap_vector(self, implementation, homo_w):
        n_batch, m, n = 10, 20, 40
        xs = brainstate.random.rand(n_batch, m)
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = jax.vmap(
            lambda x: coomv(data, row, col, x, shape=(m, n), transpose=True, backend=implementation)
        )(xs)
        y2 = jax.vmap(lambda x: vector_coo(x, data, row, col, (m, n)))(xs)

        assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((xs, row, col, y, y2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_coomv_single_nnz(self, implementation, homo_w, transpose):
        m, n = 10, 20
        row = np.array([3], dtype=np.int32)
        col = np.array([7], dtype=np.int32)
        x = jnp.zeros(m if transpose else n, dtype=jnp.float32)
        x = x.at[3 if transpose else 7].set(1.0)
        data = 2.5 if homo_w else jnp.array([2.5])
        y = coomv(data, row, col, x, shape=(m, n), transpose=transpose, backend=implementation)
        y2 = vector_coo(x, data, row, col, (m, n)) if transpose else coo_vector(x, data, row, col, (m, n))
        assert jax.block_until_ready(jnp.allclose(y, y2, atol=1e-6))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('transpose', [True, False])
    def test_coomv_empty(self, implementation, transpose):
        m, n = 10, 20
        row = np.array([], dtype=np.int32)
        col = np.array([], dtype=np.int32)
        x = jnp.ones(m if transpose else n, dtype=jnp.float32)
        data = jnp.array([1.0])
        y = coomv(data, row, col, x, shape=(m, n), transpose=transpose, backend=implementation)
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
        x = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
        data = 1.5 if homo_w else jnp.asarray(rng.standard_normal(nnz), dtype=jnp.float32)
        y = coomv(data, row, col, x, shape=(m, n), transpose=False, backend=implementation)
        y2 = coo_vector(x, data, row, col, (m, n))
        assert jax.block_until_ready(jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coomv_heavy_duplicates(self, implementation, homo_w):
        m, n, nnz = 5, 5, 200
        row = np.zeros(nnz, dtype=np.int32)
        col = np.zeros(nnz, dtype=np.int32)
        x = jnp.ones(n, dtype=jnp.float32)
        data = 1.0 if homo_w else jnp.ones(nnz, dtype=jnp.float32)
        y = coomv(data, row, col, x, shape=(m, n), transpose=False, backend=implementation)
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
        x = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
        data = 1.5 if homo_w else jnp.asarray(rng.standard_normal(nnz), dtype=jnp.float32)
        y = coomv(data, row, col, x, shape=(m, n), transpose=False, backend=implementation)
        y2 = coo_vector(x, data, row, col, (m, n))
        assert jax.block_until_ready(jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coomv_dtype(self, implementation, dtype, homo_w):
        with brainstate.environ.context(precision=64 if dtype == jnp.float64 else 32):
            m, n = 30, 50
            rng = np.random.default_rng(99)
            row = rng.integers(0, m, size=60, dtype=np.int32)
            col = rng.integers(0, n, size=60, dtype=np.int32)
            x = jnp.asarray(rng.standard_normal(n), dtype=dtype)
            data = jnp.asarray(1.5, dtype=dtype) if homo_w else jnp.asarray(rng.standard_normal(60), dtype=dtype)
            y = coomv(data, row, col, x, shape=(m, n), transpose=False, backend=implementation)
            y2 = coo_vector(x, data, row, col, (m, n))
            assert y.dtype == dtype
            assert jax.block_until_ready(jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    def _test_vjp(self, implementation, homo_w, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)

        row, col = _get_coo(n_in, n_out, 0.2)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)

        def f_api(x, w):
            if transpose:
                r = coomv(w, row, col, x, shape=shape, transpose=True, backend=implementation)
            else:
                r = coomv(w, row, col, x, shape=shape, transpose=False, backend=implementation)
            return r.sum()

        r = jax.grad(f_api, argnums=(0, 1))(x, w)

        def f_jax(x, w):
            if transpose:
                r = vector_coo(x, w, row, col, shape=shape)
            else:
                r = coo_vector(x, w, row, col, shape=shape)
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, w)
        assert jnp.allclose(r[0], r2[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(r[1], r2[1], rtol=1e-3, atol=1e-3)
        jax.block_until_ready((x, row, col, r, r2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, implementation, transpose, homo_w):
        self._test_vjp(
            implementation=implementation,
            homo_w=homo_w,

            transpose=transpose,
        )

    def _test_jvp(self, implementation, homo_w, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in if transpose else n_out)
        row, col = _get_coo(n_in, n_out, 0.1)

        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)

        def f_api(x, w):
            if transpose:
                r = coomv(w, row, col, x, shape=shape, transpose=True, backend=implementation)
            else:
                r = coomv(w, row, col, x, shape=shape, transpose=False, backend=implementation)
            return r

        o1, r1 = jax.jvp(f_api, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        def f_jax(x, w):
            if transpose:
                r = vector_coo(x, w, row, col, shape=shape)
            else:
                r = coo_vector(x, w, row, col, shape=shape)
            return r

        o2, r2 = jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((x, row, col, o1, r1, o2, r2))

    @pytest.mark.parametrize('implementation', COOMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, implementation, transpose, homo_w):
        self._test_jvp(
            implementation=implementation,
            homo_w=homo_w,
            transpose=transpose,
        )


class TestMatrixCOO:
    @pytest.mark.parametrize('implementation', COOMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_coo(self, implementation, homo_w):
        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m)
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = coomm(data, row, col, x.T, shape=(m, n), transpose=True, backend=implementation).T
        y2 = matrix_coo(x, data, row, col, (m, n))
        assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((x, row, col, y, y2))

    @pytest.mark.parametrize('implementation', COOMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coo_matrix(self, implementation, homo_w):
        m, n, k = 20, 40, 10
        x = brainstate.random.rand(n, k)
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        y = coomm(data, row, col, x, shape=(m, n), transpose=False, backend=implementation)
        y2 = coo_matrix(x, data, row, col, (m, n))
        assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((x, row, col, y, y2))
