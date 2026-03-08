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

from brainevent._csr.float import csrmv, csrmm
from brainevent._csr.sparse_float import (
    spfloat_csrmv,
    spfloat_csrmv_p,
    spfloat_csrmm,
    spfloat_csrmm_p,
)
from brainevent._csr.test_util import get_csr

platform = jax.default_backend()
SPFLOAT_CSRMV_PARAMS = tuple(spfloat_csrmv_p.available_backends(platform))
SPFLOAT_CSRMM_PARAMS = tuple(spfloat_csrmm_p.available_backends(platform))


def _make_data(homo_w, shape):
    if homo_w:
        return jnp.asarray(1.5, dtype=jnp.float32)
    return braintools.init.Normal(0.0, 1.0)(shape)


def _mask(values):
    return jnp.where(values > 0.5, values, 0.0)


def _spfloat_csrmv_api(data, indices, indptr, v, shape, transpose, implementation):
    return spfloat_csrmv(
        data,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
        backend=implementation,
    )


def _spfloat_csrmm_api(data, indices, indptr, B, shape, transpose, implementation):
    return spfloat_csrmm(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
        backend=implementation,
    )


@pytest.mark.skipif(
    not SPFLOAT_CSRMV_PARAMS,
    reason=f'No spfloat_csrmv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', SPFLOAT_CSRMV_PARAMS)
class TestSparseFloatCSRMV:
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matvec(self, implementation, homo_w, transpose):
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = brainstate.random.rand(m if transpose else n)
        data = _make_data(homo_w, indices.shape)

        result = _spfloat_csrmv_api(data, indices, indptr, v, (m, n), transpose, implementation)
        expected = csrmv(data, indices, indptr, v, shape=(m, n), transpose=transpose, backend=implementation)
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((v, indptr, indices, data, result, expected))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matvec_masked_input(self, implementation, homo_w, transpose):
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = _mask(brainstate.random.rand(m if transpose else n))
        data = _make_data(homo_w, indices.shape)

        result = _spfloat_csrmv_api(data, indices, indptr, v, (m, n), transpose, implementation)
        expected = csrmv(data, indices, indptr, v, shape=(m, n), transpose=transpose, backend=implementation)
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((v, indptr, indices, data, result, expected))

    def test_scalar_weight_broadcast(self, implementation):
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = brainstate.random.rand(n)
        data = jnp.asarray(2.5, dtype=jnp.float32)

        result = _spfloat_csrmv_api(data, indices, indptr, v, (m, n), False, implementation)
        expected = csrmv(data, indices, indptr, v, shape=(m, n), transpose=False, backend=implementation)
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((v, indptr, indices, data, result, expected))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp_vector(self, implementation, homo_w, transpose):
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = _mask(brainstate.random.rand(m if transpose else n))
        data = _make_data(homo_w, indices.shape)

        def f_test(v_arg):
            return _spfloat_csrmv_api(data, indices, indptr, v_arg, (m, n), transpose, implementation).sum()

        def f_ref(v_arg):
            return csrmv(data, indices, indptr, v_arg, shape=(m, n), transpose=transpose, backend=implementation).sum()

        grad_test = jax.grad(f_test)(v)
        grad_ref = jax.grad(f_ref)(v)
        assert jnp.allclose(grad_test, grad_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((v, indptr, indices, data, grad_test, grad_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp_weights(self, implementation, homo_w, transpose):
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = _mask(brainstate.random.rand(m if transpose else n))
        data = _make_data(homo_w, indices.shape)

        def f_test(w_arg):
            return _spfloat_csrmv_api(w_arg, indices, indptr, v, (m, n), transpose, implementation).sum()

        def f_ref(w_arg):
            return csrmv(w_arg, indices, indptr, v, shape=(m, n), transpose=transpose, backend=implementation).sum()

        grad_test = jax.grad(f_test)(data)
        grad_ref = jax.grad(f_ref)(data)
        assert jnp.allclose(grad_test, grad_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((v, indptr, indices, data, grad_test, grad_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp_vector(self, implementation, homo_w, transpose):
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = _mask(brainstate.random.rand(m if transpose else n))
        data = _make_data(homo_w, indices.shape)
        v_dot = jnp.ones_like(v)

        def f_test(v_arg):
            return _spfloat_csrmv_api(data, indices, indptr, v_arg, (m, n), transpose, implementation)

        def f_ref(v_arg):
            return csrmv(data, indices, indptr, v_arg, shape=(m, n), transpose=transpose, backend=implementation)

        primal_test, tangent_test = jax.jvp(f_test, (v,), (v_dot,))
        primal_ref, tangent_ref = jax.jvp(f_ref, (v,), (v_dot,))
        assert jnp.allclose(primal_test, primal_ref, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((v, indptr, indices, data, primal_test, tangent_test, primal_ref, tangent_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp_weights(self, implementation, homo_w, transpose):
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = _mask(brainstate.random.rand(m if transpose else n))
        data = _make_data(homo_w, indices.shape)
        data_dot = jnp.ones_like(data)

        def f_test(w_arg):
            return _spfloat_csrmv_api(w_arg, indices, indptr, v, (m, n), transpose, implementation)

        def f_ref(w_arg):
            return csrmv(w_arg, indices, indptr, v, shape=(m, n), transpose=transpose, backend=implementation)

        primal_test, tangent_test = jax.jvp(f_test, (data,), (data_dot,))
        primal_ref, tangent_ref = jax.jvp(f_ref, (data,), (data_dot,))
        assert jnp.allclose(primal_test, primal_ref, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((indptr, indices, data, primal_test, tangent_test, primal_ref, tangent_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector(self, implementation, homo_w):
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        vs = _mask(brainstate.random.rand(b, n))
        data = _make_data(homo_w, indices.shape)

        f_test = lambda v: _spfloat_csrmv_api(data, indices, indptr, v, (m, n), False, implementation)
        f_ref = lambda v: csrmv(data, indices, indptr, v, shape=(m, n), transpose=False, backend=implementation)

        result = jax.vmap(f_test)(vs)
        expected = jax.vmap(f_ref)(vs)
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((vs, indptr, indices, data, result, expected))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_transpose(self, implementation, homo_w):
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        vs = _mask(brainstate.random.rand(b, m))
        data = _make_data(homo_w, indices.shape)

        f_test = lambda v: _spfloat_csrmv_api(data, indices, indptr, v, (m, n), True, implementation)
        f_ref = lambda v: csrmv(data, indices, indptr, v, shape=(m, n), transpose=True, backend=implementation)

        result = jax.vmap(f_test)(vs)
        expected = jax.vmap(f_ref)(vs)
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((vs, indptr, indices, data, result, expected))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, implementation, homo_w):
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = _mask(brainstate.random.rand(n))

        if homo_w:
            data = brainstate.random.rand(b)
        else:
            data = braintools.init.Normal(0.0, 1.0)((b,) + indices.shape)

        f_test = lambda w: _spfloat_csrmv_api(w, indices, indptr, v, (m, n), False, implementation)
        f_ref = lambda w: csrmv(w, indices, indptr, v, shape=(m, n), transpose=False, backend=implementation)

        result = jax.vmap(f_test)(data)
        expected = jax.vmap(f_ref)(data)
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((v, indptr, indices, data, result, expected))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vjp(self, implementation, homo_w):
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        vs = _mask(brainstate.random.rand(b, n))
        data = _make_data(homo_w, indices.shape)

        def f_test(v_arg, w_arg):
            return _spfloat_csrmv_api(w_arg, indices, indptr, v_arg, (m, n), False, implementation).sum()

        def f_ref(v_arg, w_arg):
            return csrmv(w_arg, indices, indptr, v_arg, shape=(m, n), transpose=False, backend=implementation).sum()

        grad_test = jax.vmap(lambda v: jax.grad(f_test, argnums=(0, 1))(v, data))(vs)
        grad_ref = jax.vmap(lambda v: jax.grad(f_ref, argnums=(0, 1))(v, data))(vs)
        assert jnp.allclose(grad_test[0], grad_ref[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(grad_test[1], grad_ref[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((vs, indptr, indices, data, grad_test, grad_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_jvp(self, implementation, homo_w):
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        vs = _mask(brainstate.random.rand(b, n))
        v_dots = jnp.ones_like(vs)
        data = _make_data(homo_w, indices.shape)

        f_test = lambda v: _spfloat_csrmv_api(data, indices, indptr, v, (m, n), False, implementation)
        f_ref = lambda v: csrmv(data, indices, indptr, v, shape=(m, n), transpose=False, backend=implementation)

        primal_test, tangent_test = jax.vmap(lambda v, vd: jax.jvp(f_test, (v,), (vd,)))(vs, v_dots)
        primal_ref, tangent_ref = jax.vmap(lambda v, vd: jax.jvp(f_ref, (v,), (vd,)))(vs, v_dots)
        assert jnp.allclose(primal_test, primal_ref, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((vs, indptr, indices, data, primal_test, tangent_test, primal_ref, tangent_ref))


@pytest.mark.skipif(
    not SPFLOAT_CSRMM_PARAMS,
    reason=f'No spfloat_csrmm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', SPFLOAT_CSRMM_PARAMS)
class TestSparseFloatCSRMM:
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat(self, implementation, homo_w, transpose):
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)
        B = brainstate.random.rand(m if transpose else n, k)
        data = _make_data(homo_w, indices.shape)

        result = _spfloat_csrmm_api(data, indices, indptr, B, (m, n), transpose, implementation)
        expected = csrmm(data, indices, indptr, B, shape=(m, n), transpose=transpose, backend=implementation)
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((B, indptr, indices, data, result, expected))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_masked_input(self, implementation, homo_w, transpose):
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)
        B = _mask(brainstate.random.rand(m if transpose else n, k))
        data = _make_data(homo_w, indices.shape)

        result = _spfloat_csrmm_api(data, indices, indptr, B, (m, n), transpose, implementation)
        expected = csrmm(data, indices, indptr, B, shape=(m, n), transpose=transpose, backend=implementation)
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((B, indptr, indices, data, result, expected))

    def test_scalar_weight_broadcast(self, implementation):
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)
        B = brainstate.random.rand(n, k)
        data = jnp.asarray(2.5, dtype=jnp.float32)

        result = _spfloat_csrmm_api(data, indices, indptr, B, (m, n), False, implementation)
        expected = csrmm(data, indices, indptr, B, shape=(m, n), transpose=False, backend=implementation)
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((B, indptr, indices, data, result, expected))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_vjp_B(self, implementation, homo_w, transpose):
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)
        B = _mask(brainstate.random.rand(m if transpose else n, k))
        data = _make_data(homo_w, indices.shape)

        def f_test(B_arg):
            return _spfloat_csrmm_api(data, indices, indptr, B_arg, (m, n), transpose, implementation).sum()

        def f_ref(B_arg):
            return csrmm(data, indices, indptr, B_arg, shape=(m, n), transpose=transpose, backend=implementation).sum()

        grad_test = jax.grad(f_test)(B)
        grad_ref = jax.grad(f_ref)(B)
        assert jnp.allclose(grad_test, grad_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((B, indptr, indices, data, grad_test, grad_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_vjp_weights(self, implementation, homo_w, transpose):
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)
        B = _mask(brainstate.random.rand(m if transpose else n, k))
        data = _make_data(homo_w, indices.shape)

        def f_test(w_arg):
            return _spfloat_csrmm_api(w_arg, indices, indptr, B, (m, n), transpose, implementation).sum()

        def f_ref(w_arg):
            return csrmm(w_arg, indices, indptr, B, shape=(m, n), transpose=transpose, backend=implementation).sum()

        grad_test = jax.grad(f_test)(data)
        grad_ref = jax.grad(f_ref)(data)
        assert jnp.allclose(grad_test, grad_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((B, indptr, indices, data, grad_test, grad_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_jvp_B(self, implementation, homo_w, transpose):
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)
        B = _mask(brainstate.random.rand(m if transpose else n, k))
        data = _make_data(homo_w, indices.shape)
        B_dot = jnp.ones_like(B)

        def f_test(B_arg):
            return _spfloat_csrmm_api(data, indices, indptr, B_arg, (m, n), transpose, implementation)

        def f_ref(B_arg):
            return csrmm(data, indices, indptr, B_arg, shape=(m, n), transpose=transpose, backend=implementation)

        primal_test, tangent_test = jax.jvp(f_test, (B,), (B_dot,))
        primal_ref, tangent_ref = jax.jvp(f_ref, (B,), (B_dot,))
        assert jnp.allclose(primal_test, primal_ref, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((B, indptr, indices, data, primal_test, tangent_test, primal_ref, tangent_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_jvp_weights(self, implementation, homo_w, transpose):
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)
        B = _mask(brainstate.random.rand(m if transpose else n, k))
        data = _make_data(homo_w, indices.shape)
        data_dot = jnp.ones_like(data)

        def f_test(w_arg):
            return _spfloat_csrmm_api(w_arg, indices, indptr, B, (m, n), transpose, implementation)

        def f_ref(w_arg):
            return csrmm(w_arg, indices, indptr, B, shape=(m, n), transpose=transpose, backend=implementation)

        primal_test, tangent_test = jax.jvp(f_test, (data,), (data_dot,))
        primal_ref, tangent_ref = jax.jvp(f_ref, (data,), (data_dot,))
        assert jnp.allclose(primal_test, primal_ref, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((B, indptr, indices, data, primal_test, tangent_test, primal_ref, tangent_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_matrix(self, implementation, homo_w):
        b, m, n, k = 10, 50, 30, 8
        indptr, indices = get_csr(m, n, 0.1)
        Bs = _mask(brainstate.random.rand(b, n, k))
        data = _make_data(homo_w, indices.shape)

        f_test = lambda B: _spfloat_csrmm_api(data, indices, indptr, B, (m, n), False, implementation)
        f_ref = lambda B: csrmm(data, indices, indptr, B, shape=(m, n), transpose=False, backend=implementation)

        result = jax.vmap(f_test)(Bs)
        expected = jax.vmap(f_ref)(Bs)
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((Bs, indptr, indices, data, result, expected))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_matrix_transpose(self, implementation, homo_w):
        b, m, n, k = 10, 50, 30, 8
        indptr, indices = get_csr(m, n, 0.1)
        Bs = _mask(brainstate.random.rand(b, m, k))
        data = _make_data(homo_w, indices.shape)

        f_test = lambda B: _spfloat_csrmm_api(data, indices, indptr, B, (m, n), True, implementation)
        f_ref = lambda B: csrmm(data, indices, indptr, B, shape=(m, n), transpose=True, backend=implementation)

        result = jax.vmap(f_test)(Bs)
        expected = jax.vmap(f_ref)(Bs)
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((Bs, indptr, indices, data, result, expected))
