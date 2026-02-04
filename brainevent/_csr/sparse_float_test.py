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

"""
Unit tests for sparse_float CSR operations.

Tests cover:
- Forward pass for matvec and matmat
- JVP (forward-mode AD) for vector and weight inputs
- VJP (reverse-mode AD) for vector and weight inputs
- Batching/vmap operations
"""

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

from brainevent._csr.sparse_float import (
    sparse_float_csrmv,
    sparse_float_csrmm,
)
from brainevent._csr.float import csrmv, csrmm
from brainevent._csr.test_util import get_csr


class TestSparseFloatCSRMV:
    """Test forward pass for masked float CSR matrix-vector multiplication."""

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matvec(self, homo_w, transpose):
        """Test basic CSR @ vector operation."""
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)

        # Create input vector
        if transpose:
            v = brainstate.random.rand(m)
        else:
            v = brainstate.random.rand(n)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        # Test implementation
        result = sparse_float_csrmv(
            data, indices, indptr, v,
            shape=(m, n), transpose=transpose
        )

        # Reference implementation
        expected = csrmv(
            data, indices, indptr, v,
            shape=(m, n), transpose=transpose
        )

        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matvec_masked_input(self, homo_w, transpose):
        """Test CSR @ vector with sparse (masked) input vectors."""
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)

        # Create sparse (masked) vector - has zeros
        if transpose:
            v = brainstate.random.rand(m)
        else:
            v = brainstate.random.rand(n)
        v = jnp.where(v > 0.5, v, 0.0)  # Make it masked

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        # Test implementation
        result = sparse_float_csrmv(
            data, indices, indptr, v,
            shape=(m, n), transpose=transpose
        )

        # Reference implementation
        expected = csrmv(
            data, indices, indptr, v,
            shape=(m, n), transpose=transpose
        )

        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_scalar_weight_broadcast(self):
        """Test with scalar weight broadcasting."""
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = brainstate.random.rand(n)

        # Scalar weight
        data = 2.5

        result = sparse_float_csrmv(
            data, indices, indptr, v,
            shape=(m, n), transpose=False
        )
        expected = csrmv(
            data, indices, indptr, v,
            shape=(m, n), transpose=False
        )

        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)


class TestSparseFloatCSRMM:
    """Test forward pass for masked float CSR matrix-matrix multiplication."""

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat(self, homo_w, transpose):
        """Test basic CSR @ matrix operation."""
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)

        # Create input matrix
        if transpose:
            B = brainstate.random.rand(m, k)
        else:
            B = brainstate.random.rand(n, k)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        # Test implementation
        result = sparse_float_csrmm(
            data, indices, indptr, B,
            shape=(m, n), transpose=transpose
        )

        # Reference implementation
        expected = csrmm(
            data, indices, indptr, B,
            shape=(m, n), transpose=transpose
        )

        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_masked_input(self, homo_w, transpose):
        """Test CSR @ matrix with sparse (masked) input matrices."""
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)

        # Create sparse (masked) matrix - has zeros
        if transpose:
            B = brainstate.random.rand(m, k)
        else:
            B = brainstate.random.rand(n, k)
        B = jnp.where(B > 0.5, B, 0.0)  # Make it masked

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        # Test implementation
        result = sparse_float_csrmm(
            data, indices, indptr, B,
            shape=(m, n), transpose=transpose
        )

        # Reference implementation
        expected = csrmm(
            data, indices, indptr, B,
            shape=(m, n), transpose=transpose
        )

        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_scalar_weight_broadcast(self):
        """Test with scalar weight broadcasting."""
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)
        B = brainstate.random.rand(n, k)

        # Scalar weight
        data = 2.5

        result = sparse_float_csrmm(
            data, indices, indptr, B,
            shape=(m, n), transpose=False
        )
        expected = csrmm(
            data, indices, indptr, B,
            shape=(m, n), transpose=False
        )

        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)


class TestSparseFloatGradients:
    """Test JVP and VJP gradient rules for masked float CSR operations."""

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp_vector(self, homo_w, transpose):
        """Test VJP with respect to input vector."""
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)

        if transpose:
            v = brainstate.random.rand(m)
        else:
            v = brainstate.random.rand(n)
        v = jnp.where(v > 0.5, v, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(v):
            return sparse_float_csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=transpose
            ).sum()

        def f_ref(v):
            return csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=transpose
            ).sum()

        grad_test = jax.grad(f_test)(v)
        grad_ref = jax.grad(f_ref)(v)

        assert jnp.allclose(grad_test, grad_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp_weights(self, homo_w, transpose):
        """Test VJP with respect to weights."""
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)

        if transpose:
            v = brainstate.random.rand(m)
        else:
            v = brainstate.random.rand(n)
        v = jnp.where(v > 0.5, v, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(w):
            return sparse_float_csrmv(
                w, indices, indptr, v,
                shape=(m, n), transpose=transpose
            ).sum()

        def f_ref(w):
            return csrmv(
                w, indices, indptr, v,
                shape=(m, n), transpose=transpose
            ).sum()

        grad_test = jax.grad(f_test)(data)
        grad_ref = jax.grad(f_ref)(data)

        assert jnp.allclose(grad_test, grad_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp_vector(self, homo_w, transpose):
        """Test JVP with respect to input vector."""
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)

        if transpose:
            v = brainstate.random.rand(m)
        else:
            v = brainstate.random.rand(n)
        v = jnp.where(v > 0.5, v, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        v_dot = jnp.ones_like(v)

        def f_test(v):
            return sparse_float_csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=transpose
            )

        def f_ref(v):
            return csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=transpose
            )

        primal_test, tangent_test = jax.jvp(f_test, (v,), (v_dot,))
        primal_ref, tangent_ref = jax.jvp(f_ref, (v,), (v_dot,))

        assert jnp.allclose(primal_test, primal_ref, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp_weights(self, homo_w, transpose):
        """Test JVP with respect to weights."""
        m, n = 50, 30
        indptr, indices = get_csr(m, n, 0.1)

        if transpose:
            v = brainstate.random.rand(m)
        else:
            v = brainstate.random.rand(n)
        v = jnp.where(v > 0.5, v, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        data_dot = jnp.ones_like(data)

        def f_test(w):
            return sparse_float_csrmv(
                w, indices, indptr, v,
                shape=(m, n), transpose=transpose
            )

        def f_ref(w):
            return csrmv(
                w, indices, indptr, v,
                shape=(m, n), transpose=transpose
            )

        primal_test, tangent_test = jax.jvp(f_test, (data,), (data_dot,))
        primal_ref, tangent_ref = jax.jvp(f_ref, (data,), (data_dot,))

        assert jnp.allclose(primal_test, primal_ref, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_vjp_B(self, homo_w, transpose):
        """Test VJP for matrix-matrix product with respect to B."""
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)

        if transpose:
            B = brainstate.random.rand(m, k)
        else:
            B = brainstate.random.rand(n, k)
        B = jnp.where(B > 0.5, B, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(B):
            return sparse_float_csrmm(
                data, indices, indptr, B,
                shape=(m, n), transpose=transpose
            ).sum()

        def f_ref(B):
            return csrmm(
                data, indices, indptr, B,
                shape=(m, n), transpose=transpose
            ).sum()

        grad_test = jax.grad(f_test)(B)
        grad_ref = jax.grad(f_ref)(B)

        assert jnp.allclose(grad_test, grad_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_vjp_weights(self, homo_w, transpose):
        """Test VJP for matrix-matrix product with respect to weights."""
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)

        if transpose:
            B = brainstate.random.rand(m, k)
        else:
            B = brainstate.random.rand(n, k)
        B = jnp.where(B > 0.5, B, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(w):
            return sparse_float_csrmm(
                w, indices, indptr, B,
                shape=(m, n), transpose=transpose
            ).sum()

        def f_ref(w):
            return csrmm(
                w, indices, indptr, B,
                shape=(m, n), transpose=transpose
            ).sum()

        grad_test = jax.grad(f_test)(data)
        grad_ref = jax.grad(f_ref)(data)

        assert jnp.allclose(grad_test, grad_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_jvp_B(self, homo_w, transpose):
        """Test JVP for matrix-matrix product with respect to B."""
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)

        if transpose:
            B = brainstate.random.rand(m, k)
        else:
            B = brainstate.random.rand(n, k)
        B = jnp.where(B > 0.5, B, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        B_dot = jnp.ones_like(B)

        def f_test(B):
            return sparse_float_csrmm(
                data, indices, indptr, B,
                shape=(m, n), transpose=transpose
            )

        def f_ref(B):
            return csrmm(
                data, indices, indptr, B,
                shape=(m, n), transpose=transpose
            )

        primal_test, tangent_test = jax.jvp(f_test, (B,), (B_dot,))
        primal_ref, tangent_ref = jax.jvp(f_ref, (B,), (B_dot,))

        assert jnp.allclose(primal_test, primal_ref, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_matmat_jvp_weights(self, homo_w, transpose):
        """Test JVP for matrix-matrix product with respect to weights."""
        m, n, k = 50, 30, 10
        indptr, indices = get_csr(m, n, 0.1)

        if transpose:
            B = brainstate.random.rand(m, k)
        else:
            B = brainstate.random.rand(n, k)
        B = jnp.where(B > 0.5, B, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        data_dot = jnp.ones_like(data)

        def f_test(w):
            return sparse_float_csrmm(
                w, indices, indptr, B,
                shape=(m, n), transpose=transpose
            )

        def f_ref(w):
            return csrmm(
                w, indices, indptr, B,
                shape=(m, n), transpose=transpose
            )

        primal_test, tangent_test = jax.jvp(f_test, (data,), (data_dot,))
        primal_ref, tangent_ref = jax.jvp(f_ref, (data,), (data_dot,))

        assert jnp.allclose(primal_test, primal_ref, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)


class TestSparseFloatBatching:
    """Test vmap/batching rules for masked float CSR operations."""

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector(self, homo_w):
        """Test vmap over input vectors (axis 0)."""
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        vs = brainstate.random.rand(b, n)
        vs = jnp.where(vs > 0.5, vs, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(v):
            return sparse_float_csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=False
            )

        def f_ref(v):
            return csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=False
            )

        result = jax.vmap(f_test)(vs)
        expected = jax.vmap(f_ref)(vs)

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_transpose(self, homo_w):
        """Test vmap over input vectors with transpose (axis 0)."""
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        vs = brainstate.random.rand(b, m)
        vs = jnp.where(vs > 0.5, vs, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(v):
            return sparse_float_csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=True
            )

        def f_ref(v):
            return csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=True
            )

        result = jax.vmap(f_test)(vs)
        expected = jax.vmap(f_ref)(vs)

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, homo_w):
        """Test vmap over weight data."""
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        v = brainstate.random.rand(n)
        v = jnp.where(v > 0.5, v, 0.0)

        if homo_w:
            data = brainstate.random.rand(b)
        else:
            data = braintools.init.Normal(0., 1.)((b,) + indices.shape)

        def f_test(w):
            return sparse_float_csrmv(
                w, indices, indptr, v,
                shape=(m, n), transpose=False
            )

        def f_ref(w):
            return csrmv(
                w, indices, indptr, v,
                shape=(m, n), transpose=False
            )

        result = jax.vmap(f_test)(data)
        expected = jax.vmap(f_ref)(data)

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_matrix(self, homo_w):
        """Test vmap over input matrices."""
        b, m, n, k = 10, 50, 30, 8
        indptr, indices = get_csr(m, n, 0.1)
        Bs = brainstate.random.rand(b, n, k)
        Bs = jnp.where(Bs > 0.5, Bs, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(B):
            return sparse_float_csrmm(
                data, indices, indptr, B,
                shape=(m, n), transpose=False
            )

        def f_ref(B):
            return csrmm(
                data, indices, indptr, B,
                shape=(m, n), transpose=False
            )

        result = jax.vmap(f_test)(Bs)
        expected = jax.vmap(f_ref)(Bs)

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_matrix_transpose(self, homo_w):
        """Test vmap over input matrices with transpose."""
        b, m, n, k = 10, 50, 30, 8
        indptr, indices = get_csr(m, n, 0.1)
        Bs = brainstate.random.rand(b, m, k)
        Bs = jnp.where(Bs > 0.5, Bs, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(B):
            return sparse_float_csrmm(
                data, indices, indptr, B,
                shape=(m, n), transpose=True
            )

        def f_ref(B):
            return csrmm(
                data, indices, indptr, B,
                shape=(m, n), transpose=True
            )

        result = jax.vmap(f_test)(Bs)
        expected = jax.vmap(f_ref)(Bs)

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vjp(self, homo_w):
        """Test combined vmap + VJP gradient."""
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        vs = brainstate.random.rand(b, n)
        vs = jnp.where(vs > 0.5, vs, 0.0)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(v, w):
            return sparse_float_csrmv(
                w, indices, indptr, v,
                shape=(m, n), transpose=False
            ).sum()

        def f_ref(v, w):
            return csrmv(
                w, indices, indptr, v,
                shape=(m, n), transpose=False
            ).sum()

        grad_test = jax.vmap(lambda v: jax.grad(f_test, argnums=(0, 1))(v, data))(vs)
        grad_ref = jax.vmap(lambda v: jax.grad(f_ref, argnums=(0, 1))(v, data))(vs)

        assert jnp.allclose(grad_test[0], grad_ref[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(grad_test[1], grad_ref[1], rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_jvp(self, homo_w):
        """Test combined vmap + JVP forward-mode AD."""
        b, m, n = 10, 50, 30
        indptr, indices = get_csr(m, n, 0.1)
        vs = brainstate.random.rand(b, n)
        vs = jnp.where(vs > 0.5, vs, 0.0)
        v_dots = jnp.ones_like(vs)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_test(v):
            return sparse_float_csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=False
            )

        def f_ref(v):
            return csrmv(
                data, indices, indptr, v,
                shape=(m, n), transpose=False
            )

        def jvp_fn_test(v, v_dot):
            return jax.jvp(f_test, (v,), (v_dot,))

        def jvp_fn_ref(v, v_dot):
            return jax.jvp(f_ref, (v,), (v_dot,))

        primal_test, tangent_test = jax.vmap(jvp_fn_test)(vs, v_dots)
        primal_ref, tangent_ref = jax.vmap(jvp_fn_ref)(vs, v_dots)

        assert jnp.allclose(primal_test, primal_ref, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(tangent_test, tangent_ref, rtol=1e-3, atol=1e-3)
