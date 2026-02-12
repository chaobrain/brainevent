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

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

from brainevent._csr.float import csrmv, csrmv_p, csrmm, csrmm_p
from brainevent._csr.test_util import get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix

platform = jax.default_backend()
CSRMV_IMPLEMENTATIONS = tuple(csrmv_p.available_backends(platform))
CSRMM_IMPLEMENTATIONS = tuple(csrmm_p.available_backends(platform))


def _make_data(homo_w, shape):
    if homo_w:
        return jnp.asarray(1.5, dtype=jnp.float32)
    return braintools.init.Normal(0.0, 1.0)(shape)


def _vector_csr_api(x, data, indices, indptr, shape, implementation):
    return csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=True,
        backend=implementation,
    )


def _csr_vector_api(x, data, indices, indptr, shape, implementation):
    return csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
    )


def _matrix_csr_api(x, data, indices, indptr, shape, implementation):
    return csrmm(
        data,
        indices,
        indptr,
        x.T,
        shape=shape,
        transpose=True,
        backend=implementation,
    ).T


def _csr_matrix_api(x, data, indices, indptr, shape, implementation):
    return csrmm(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
    )


def _row_ids_from_indptr(indptr):
    indptr = jnp.asarray(indptr)
    counts = jnp.diff(indptr)
    return jnp.repeat(jnp.arange(counts.shape[0], dtype=indptr.dtype), counts)


@pytest.mark.skipif(
    not CSRMV_IMPLEMENTATIONS,
    reason=f'No csrmv implementation on platform={platform}',
)
class TestFloatCSRMV:
    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, implementation, homo_w):
        m, n = 20, 40
        x = brainstate.random.rand(m)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
        y_ref = vector_csr(x, data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, implementation, homo_w):
        m, n = 20, 40
        v = brainstate.random.rand(n)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = _csr_vector_api(v, data, indices, indptr, (m, n), implementation)
        y_ref = csr_vector(v, data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((v, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr_vmap_vector(self, implementation, homo_w):
        n_batch, m, n = 10, 20, 40
        xs = brainstate.random.rand(n_batch, m)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = brainstate.transform.vmap2(
            lambda x: _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
        )(xs)
        y_ref = brainstate.transform.vmap2(lambda x: vector_csr(x, data, indices, indptr, (m, n)))(xs)
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)

        indptr, indices = get_csr(n_in, n_out, 0.2, replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x_arg, w_arg):
            if transpose:
                return _vector_csr_api(x_arg, w_arg, indices, indptr, shape, implementation).sum()
            return _csr_vector_api(x_arg, w_arg, indices, indptr, shape, implementation).sum()

        def f_ref(x_arg, w_arg):
            if transpose:
                return vector_csr(x_arg, w_arg, indices, indptr, shape=shape).sum()
            return csr_vector(x_arg, w_arg, indices, indptr, shape=shape).sum()

        r = jax.grad(f_api, argnums=(0, 1))(x, w)
        r_ref = jax.grad(f_ref, argnums=(0, 1))(x, w)
        assert jnp.allclose(r[0], r_ref[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(r[1], r_ref[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, r, r_ref))

    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in if transpose else n_out)

        indptr, indices = get_csr(n_in, n_out, 0.1, replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x_arg, w_arg):
            if transpose:
                return _vector_csr_api(x_arg, w_arg, indices, indptr, shape, implementation)
            return _csr_vector_api(x_arg, w_arg, indices, indptr, shape, implementation)

        def f_ref(x_arg, w_arg):
            if transpose:
                return vector_csr(x_arg, w_arg, indices, indptr, shape=shape)
            return csr_vector(x_arg, w_arg, indices, indptr, shape=shape)

        o1, r1 = jax.jvp(f_api, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        o2, r2 = jax.jvp(f_ref, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, o1, r1, o2, r2))


@pytest.mark.skipif(
    not CSRMM_IMPLEMENTATIONS,
    reason=f'No csrmm implementation on platform={platform}',
)
class TestFloatCSRMM:
    @pytest.mark.parametrize('implementation', CSRMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, implementation, homo_w):
        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = _matrix_csr_api(x, data, indices, indptr, (m, n), implementation)
        y_ref = matrix_csr(x, data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, implementation, homo_w):
        m, n, k = 20, 40, 10
        x = brainstate.random.rand(n, k)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = _csr_matrix_api(x, data, indices, indptr, (m, n), implementation)
        y_ref = csr_matrix(x, data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, implementation, homo_w, replace, transpose):
        m, n, k = 20, 30, 8
        shape = (m, n)
        x = brainstate.random.rand(m, k) if transpose else brainstate.random.rand(n, k)

        indptr, indices = get_csr(m, n, 0.2, replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x_arg, w_arg):
            return csrmm(
                w_arg,
                indices,
                indptr,
                x_arg,
                shape=shape,
                transpose=transpose,
                backend=implementation,
            ).sum()

        def f_ref(x_arg, w_arg):
            if transpose:
                return matrix_csr(x_arg.T, w_arg, indices, indptr, shape).T.sum()
            return csr_matrix(x_arg, w_arg, indices, indptr, shape).sum()

        r = jax.grad(f_api, argnums=(0, 1))(x, w)
        r_ref = jax.grad(f_ref, argnums=(0, 1))(x, w)
        assert jnp.allclose(r[0], r_ref[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(r[1], r_ref[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, r, r_ref))

    @pytest.mark.parametrize('implementation', CSRMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, implementation, homo_w, replace, transpose):
        m, n, k = 20, 30, 8
        shape = (m, n)
        x = brainstate.random.rand(m, k) if transpose else brainstate.random.rand(n, k)

        indptr, indices = get_csr(m, n, 0.1, replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x_arg, w_arg):
            return csrmm(
                w_arg,
                indices,
                indptr,
                x_arg,
                shape=shape,
                transpose=transpose,
                backend=implementation,
            )

        def f_ref(x_arg, w_arg):
            if transpose:
                return matrix_csr(x_arg.T, w_arg, indices, indptr, shape).T
            return csr_matrix(x_arg, w_arg, indices, indptr, shape)

        o1, r1 = jax.jvp(f_api, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        o2, r2 = jax.jvp(f_ref, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, o1, r1, o2, r2))
