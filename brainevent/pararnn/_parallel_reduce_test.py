# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Tests for parallel reduction solvers."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from brainevent.pararnn._parallel_reduce import (
    parallel_reduce_diag,
    parallel_reduce_block_diag,
)

# Skip entire module if no GPU available
_has_gpu = any(d.platform == 'gpu' for d in jax.devices())
if not _has_gpu:
    pytest.skip("No GPU available", allow_module_level=True)


def _sequential_reduce_diag(jac, rhs):
    """Reference sequential solver for diagonal systems."""
    B, T, N = rhs.shape
    h = jnp.zeros((B, N), dtype=rhs.dtype)
    results = []
    for t in range(T):
        h = jac[:, t, :] * h + rhs[:, t, :]
        results.append(h)
    return jnp.stack(results, axis=1)


def _sequential_reduce_block_diag(jac, rhs):
    """Reference sequential solver for block-diagonal systems."""
    B, T, N, K = rhs.shape
    h = jnp.zeros((B, N, K), dtype=rhs.dtype)
    results = []
    for t in range(T):
        h = jnp.einsum('...ij,...j->...i', jac[:, t], h) + rhs[:, t]
        results.append(h)
    return jnp.stack(results, axis=1)


class TestParallelReduceDiag:
    """Tests for diagonal parallel reduction."""

    @pytest.mark.parametrize('T', [1, 2, 4, 8, 16, 32, 64, 128])
    def test_forward_matches_sequential(self, T):
        key = jr.PRNGKey(42)
        B, N = 2, 8
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5  # Keep Jacobians small for stability
        rhs = jr.normal(k2, (B, T, N))

        h_seq = _sequential_reduce_diag(jac, rhs)
        h_par = parallel_reduce_diag(jac, rhs)

        assert jnp.allclose(h_seq, h_par, atol=1e-5, rtol=1e-5), \
            f"Max diff: {jnp.max(jnp.abs(h_seq - h_par))}"

    @pytest.mark.parametrize('B', [1, 4])
    @pytest.mark.parametrize('N', [1, 8, 32])
    def test_various_shapes(self, B, N):
        key = jr.PRNGKey(123)
        T = 16
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5
        rhs = jr.normal(k2, (B, T, N))

        h_seq = _sequential_reduce_diag(jac, rhs)
        h_par = parallel_reduce_diag(jac, rhs)

        assert jnp.allclose(h_seq, h_par, atol=1e-5, rtol=1e-5)

    def test_zero_jacobian(self):
        """With zero Jacobians, h[t] = rhs[t]."""
        B, T, N = 2, 16, 4
        jac = jnp.zeros((B, T, N))
        rhs = jnp.ones((B, T, N))

        h = parallel_reduce_diag(jac, rhs)
        assert jnp.allclose(h, rhs)

    def test_identity_jacobian(self):
        """With jac=1 and rhs=0, h stays at 0."""
        B, T, N = 2, 16, 4
        jac = jnp.ones((B, T, N))
        rhs = jnp.zeros((B, T, N))

        h = parallel_reduce_diag(jac, rhs)
        assert jnp.allclose(h, jnp.zeros_like(h))

    def test_gradient(self):
        """Check gradient computation via jax.grad."""
        B, T, N = 2, 8, 4
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.3
        rhs = jr.normal(k2, (B, T, N))

        def loss(jac, rhs):
            h = parallel_reduce_diag(jac, rhs)
            return jnp.sum(h ** 2)

        grads = jax.grad(loss, argnums=(0, 1))(jac, rhs)
        assert grads[0].shape == jac.shape
        assert grads[1].shape == rhs.shape
        assert jnp.all(jnp.isfinite(grads[0]))
        assert jnp.all(jnp.isfinite(grads[1]))

    def test_jit_compatible(self):
        """Verify the reduction works under jax.jit."""
        B, T, N = 2, 16, 4
        key = jr.PRNGKey(7)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5
        rhs = jr.normal(k2, (B, T, N))

        h_eager = parallel_reduce_diag(jac, rhs)
        h_jit = jax.jit(parallel_reduce_diag)(jac, rhs)
        assert jnp.allclose(h_eager, h_jit, atol=1e-6)


class TestParallelReduceBlockDiag:
    """Tests for block-diagonal parallel reduction."""

    @pytest.mark.parametrize('K', [2, 3])
    @pytest.mark.parametrize('T', [1, 4, 16, 64])
    def test_forward_matches_sequential(self, K, T):
        key = jr.PRNGKey(42)
        B, N = 2, 4  # N blocks of size K
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        h_seq = _sequential_reduce_block_diag(jac, rhs)
        h_par = parallel_reduce_block_diag(jac, rhs)

        assert jnp.allclose(h_seq, h_par, atol=1e-4, rtol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(h_seq - h_par))}"

    def test_block_diag_2x2_gradient(self):
        """Check gradient for 2x2 block-diagonal reduction."""
        B, T, N, K = 2, 8, 4, 2
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        def loss(jac, rhs):
            h = parallel_reduce_block_diag(jac, rhs)
            return jnp.sum(h ** 2)

        grads = jax.grad(loss, argnums=(0, 1))(jac, rhs)
        assert grads[0].shape == jac.shape
        assert grads[1].shape == rhs.shape
        assert jnp.all(jnp.isfinite(grads[0]))
        assert jnp.all(jnp.isfinite(grads[1]))

    def test_block_diag_3x3_gradient(self):
        """Check gradient for 3x3 block-diagonal reduction."""
        B, T, N, K = 2, 8, 4, 3
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        def loss(jac, rhs):
            h = parallel_reduce_block_diag(jac, rhs)
            return jnp.sum(h ** 2)

        grads = jax.grad(loss, argnums=(0, 1))(jac, rhs)
        assert grads[0].shape == jac.shape
        assert grads[1].shape == rhs.shape
        assert jnp.all(jnp.isfinite(grads[0]))


class TestParallelReduceNonPowerOf2:
    """Test with sequence lengths that are not powers of 2."""

    @pytest.mark.parametrize('T', [3, 5, 7, 10, 13, 17, 33, 65, 100])
    def test_diag_non_power_of_2(self, T):
        key = jr.PRNGKey(42)
        B, N = 2, 8
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5
        rhs = jr.normal(k2, (B, T, N))

        h_seq = _sequential_reduce_diag(jac, rhs)
        h_par = parallel_reduce_diag(jac, rhs)

        assert jnp.allclose(h_seq, h_par, atol=1e-4, rtol=1e-4), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_seq - h_par))}"


def _sequential_reduce_diag_v1(jac, rhs):
    """Reference sequential solver for diagonal systems."""
    B, T, N = rhs.shape
    h = jnp.zeros((B, N), dtype=rhs.dtype)
    results = []
    for t in range(T):
        h = jac[:, t, :] * h + rhs[:, t, :]
        results.append(h)
    return jnp.stack(results, axis=1)


def _sequential_reduce_block2(jac, rhs):
    """Reference sequential solver for 2x2 block-diagonal systems."""
    B, T, N, K = rhs.shape
    h = jnp.zeros((B, N, K), dtype=rhs.dtype)
    results = []
    for t in range(T):
        h = jnp.einsum('...ij,...j->...i', jac[:, t], h) + rhs[:, t]
        results.append(h)
    return jnp.stack(results, axis=1)


class TestCUDAReduceDiag:
    """Tests for CUDA diagonal parallel reduction."""

    @pytest.mark.parametrize('T', [1, 4, 16, 64, 256, 1024])
    def test_forward_matches_sequential(self, T):
        key = jr.PRNGKey(42)
        B, N = 2, 8
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5
        rhs = jr.normal(k2, (B, T, N))

        h_seq = _sequential_reduce_diag_v1(jac, rhs)
        h_cuda = parallel_reduce_diag(jac, rhs, backend='cuda_raw')

        assert jnp.allclose(h_seq, h_cuda, atol=1e-4, rtol=1e-4), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_seq - h_cuda))}"

    def test_matches_jax_native(self):
        """CUDA result should match JAX associative_scan result."""
        B, T, N = 4, 64, 16
        key = jr.PRNGKey(123)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5
        rhs = jr.normal(k2, (B, T, N))

        h_jax = parallel_reduce_diag(jac, rhs, backend='jax_raw')
        h_cuda = parallel_reduce_diag(jac, rhs, backend='cuda_raw')

        assert jnp.allclose(h_jax, h_cuda, atol=1e-4, rtol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(h_jax - h_cuda))}"

    @pytest.mark.parametrize('T', [3, 7, 13, 33, 100])
    def test_non_power_of_2(self, T):
        """Test with sequence lengths that are not powers of 2."""
        B, N = 2, 8
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5
        rhs = jr.normal(k2, (B, T, N))

        h_seq = _sequential_reduce_diag_v1(jac, rhs)
        h_cuda = parallel_reduce_diag(jac, rhs, backend='cuda_raw')

        assert jnp.allclose(h_seq, h_cuda, atol=1e-4, rtol=1e-4), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_seq - h_cuda))}"

    def test_zero_jacobian(self):
        """With zero Jacobians, h[t] = rhs[t]."""
        B, T, N = 2, 16, 4
        jac = jnp.zeros((B, T, N))
        rhs = jnp.ones((B, T, N))

        h = parallel_reduce_diag(jac, rhs, backend='cuda_raw')
        assert jnp.allclose(h, rhs)

    @pytest.mark.parametrize('T', [4096, 8192])
    def test_large_sequence(self, T):
        """Test with large T to exercise multi-block path."""
        B, N = 1, 8
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.3
        rhs = jr.normal(k2, (B, T, N))

        h_jax = parallel_reduce_diag(jac, rhs, backend='jax_raw')
        h_cuda = parallel_reduce_diag(jac, rhs, backend='cuda_raw')

        assert jnp.allclose(h_jax, h_cuda, atol=1e-3, rtol=1e-3), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_jax - h_cuda))}"


class TestCUDAReduceBlock2:
    """Tests for CUDA 2x2 block-diagonal parallel reduction."""

    @pytest.mark.parametrize('T', [1, 4, 16, 64, 256])
    def test_forward_matches_sequential(self, T):
        key = jr.PRNGKey(42)
        B, N, K = 2, 4, 2
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        h_seq = _sequential_reduce_block2(jac, rhs)
        h_cuda = parallel_reduce_block_diag(jac, rhs, backend='cuda_raw')

        assert jnp.allclose(h_seq, h_cuda, atol=1e-4, rtol=1e-4), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_seq - h_cuda))}"

    def test_matches_jax_native(self):
        """CUDA result should match JAX associative_scan result."""
        B, T, N, K = 2, 32, 8, 2
        key = jr.PRNGKey(123)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        h_jax = parallel_reduce_block_diag(jac, rhs, backend='jax_raw')
        h_cuda = parallel_reduce_block_diag(jac, rhs, backend='cuda_raw')

        assert jnp.allclose(h_jax, h_cuda, atol=1e-4, rtol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(h_jax - h_cuda))}"

    @pytest.mark.parametrize('T', [3, 7, 15, 33])
    def test_non_power_of_2(self, T):
        """Test with sequence lengths that are not powers of 2."""
        B, N, K = 2, 4, 2
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        h_seq = _sequential_reduce_block2(jac, rhs)
        h_cuda = parallel_reduce_block_diag(jac, rhs, backend='cuda_raw')

        assert jnp.allclose(h_seq, h_cuda, atol=1e-4, rtol=1e-4), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_seq - h_cuda))}"

    @pytest.mark.parametrize('T', [2048, 4096])
    def test_large_sequence(self, T):
        """Test with large T to exercise multi-block path."""
        B, N, K = 1, 4, 2
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.2
        rhs = jr.normal(k2, (B, T, N, K))

        h_jax = parallel_reduce_block_diag(jac, rhs, backend='jax_raw')
        h_cuda = parallel_reduce_block_diag(jac, rhs, backend='cuda_raw')

        assert jnp.allclose(h_jax, h_cuda, atol=1e-3, rtol=1e-3), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_jax - h_cuda))}"


# =============================================================================
# AD (JVP / transpose / grad) tests for primitives
# =============================================================================

from brainevent.pararnn._parallel_reduce import (
    parallel_reduce_diag_p,
    parallel_reduce_block_diag_p,
)


def _diag_via_primitive(jac, rhs):
    """Call diagonal reduce through the XLACustomKernel primitive."""
    return parallel_reduce_diag_p(
        jac, rhs,
        rhs_info=jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        outs=[jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)],
        backend='jax_raw',
    )[0]


def _block_diag_via_primitive(jac, rhs):
    """Call block-diagonal reduce through the XLACustomKernel primitive."""
    return parallel_reduce_block_diag_p(
        jac, rhs,
        rhs_info=jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        outs=[jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)],
        backend='jax_raw',
    )[0]


class TestDiagPrimitiveAD:
    """Tests for JVP/transpose/grad of parallel_reduce_diag_p."""

    def test_grad_wrt_rhs(self):
        """jax.grad through diagonal reduce primitive w.r.t. rhs."""
        B, T, N = 2, 16, 4
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.3
        rhs = jr.normal(k2, (B, T, N))

        def loss(rhs):
            return jnp.sum(_diag_via_primitive(jac, rhs) ** 2)

        g = jax.grad(loss)(rhs)
        assert g.shape == rhs.shape
        assert jnp.all(jnp.isfinite(g))

    def test_grad_wrt_jac(self):
        """jax.grad through diagonal reduce primitive w.r.t. jac."""
        B, T, N = 2, 8, 4
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.3
        rhs = jr.normal(k2, (B, T, N))

        def loss(jac):
            return jnp.sum(_diag_via_primitive(jac, rhs) ** 2)

        g = jax.grad(loss)(jac)
        assert g.shape == jac.shape
        assert jnp.all(jnp.isfinite(g))

    def test_jvp_rhs_matches_native(self):
        """JVP w.r.t. rhs via primitive matches JAX native AD."""
        B, T, N = 1, 8, 4
        key = jr.PRNGKey(42)
        k1, k2, k3 = jr.split(key, 3)
        jac = jr.normal(k1, (B, T, N)) * 0.3
        rhs = jr.normal(k2, (B, T, N))
        rhs_dot = jr.normal(k3, (B, T, N))

        # Via primitive (exercises our JVP rule)
        def f_prim(rhs):
            return _diag_via_primitive(jac, rhs)
        _, h_dot_prim = jax.jvp(f_prim, (rhs,), (rhs_dot,))

        # Via native JAX (ground truth: associative_scan is natively differentiable)
        def f_native(rhs):
            return parallel_reduce_diag(jac, rhs)
        _, h_dot_native = jax.jvp(f_native, (rhs,), (rhs_dot,))

        assert jnp.allclose(h_dot_prim, h_dot_native, atol=1e-5), \
            f"Max diff: {jnp.max(jnp.abs(h_dot_prim - h_dot_native))}"

    def test_jvp_jac_matches_native(self):
        """JVP w.r.t. jac via primitive matches JAX native AD."""
        B, T, N = 1, 8, 4
        key = jr.PRNGKey(42)
        k1, k2, k3 = jr.split(key, 3)
        jac = jr.normal(k1, (B, T, N)) * 0.3
        rhs = jr.normal(k2, (B, T, N))
        jac_dot = jr.normal(k3, (B, T, N))

        def f_prim(jac):
            return _diag_via_primitive(jac, rhs)
        _, h_dot_prim = jax.jvp(f_prim, (jac,), (jac_dot,))

        def f_native(jac):
            return parallel_reduce_diag(jac, rhs)
        _, h_dot_native = jax.jvp(f_native, (jac,), (jac_dot,))

        assert jnp.allclose(h_dot_prim, h_dot_native, atol=1e-5), \
            f"Max diff: {jnp.max(jnp.abs(h_dot_prim - h_dot_native))}"

    def test_grad_matches_native(self):
        """VJP via primitive matches JAX native AD for both inputs."""
        B, T, N = 1, 8, 4
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.3
        rhs = jr.normal(k2, (B, T, N))

        def loss_prim(jac, rhs):
            return jnp.sum(_diag_via_primitive(jac, rhs) ** 2)

        def loss_native(jac, rhs):
            return jnp.sum(parallel_reduce_diag(jac, rhs) ** 2)

        g_prim = jax.grad(loss_prim, argnums=(0, 1))(jac, rhs)
        g_native = jax.grad(loss_native, argnums=(0, 1))(jac, rhs)

        assert jnp.allclose(g_prim[0], g_native[0], atol=1e-5), \
            f"jac grad max diff: {jnp.max(jnp.abs(g_prim[0] - g_native[0]))}"
        assert jnp.allclose(g_prim[1], g_native[1], atol=1e-5), \
            f"rhs grad max diff: {jnp.max(jnp.abs(g_prim[1] - g_native[1]))}"


class TestBlockDiagPrimitiveAD:
    """Tests for JVP/transpose/grad of parallel_reduce_block_diag_p."""

    def test_grad_wrt_rhs(self):
        """jax.grad through block-diagonal reduce primitive w.r.t. rhs."""
        B, T, N, K = 2, 8, 4, 2
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        def loss(rhs):
            return jnp.sum(_block_diag_via_primitive(jac, rhs) ** 2)

        g = jax.grad(loss)(rhs)
        assert g.shape == rhs.shape
        assert jnp.all(jnp.isfinite(g))

    def test_grad_wrt_jac(self):
        """jax.grad through block-diagonal reduce primitive w.r.t. jac."""
        B, T, N, K = 2, 8, 4, 2
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        def loss(jac):
            return jnp.sum(_block_diag_via_primitive(jac, rhs) ** 2)

        g = jax.grad(loss)(jac)
        assert g.shape == jac.shape
        assert jnp.all(jnp.isfinite(g))

    def test_jvp_rhs_matches_native(self):
        """JVP w.r.t. rhs via primitive matches JAX native AD."""
        B, T, N, K = 1, 8, 4, 2
        key = jr.PRNGKey(42)
        k1, k2, k3 = jr.split(key, 3)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))
        rhs_dot = jr.normal(k3, (B, T, N, K))

        def f_prim(rhs):
            return _block_diag_via_primitive(jac, rhs)
        _, h_dot_prim = jax.jvp(f_prim, (rhs,), (rhs_dot,))

        def f_native(rhs):
            return parallel_reduce_block_diag(jac, rhs)
        _, h_dot_native = jax.jvp(f_native, (rhs,), (rhs_dot,))

        assert jnp.allclose(h_dot_prim, h_dot_native, atol=1e-5), \
            f"Max diff: {jnp.max(jnp.abs(h_dot_prim - h_dot_native))}"

    def test_jvp_jac_matches_native(self):
        """JVP w.r.t. jac via primitive matches JAX native AD."""
        B, T, N, K = 1, 8, 4, 2
        key = jr.PRNGKey(42)
        k1, k2, k3 = jr.split(key, 3)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))
        jac_dot = jr.normal(k3, (B, T, N, K, K))

        def f_prim(jac):
            return _block_diag_via_primitive(jac, rhs)
        _, h_dot_prim = jax.jvp(f_prim, (jac,), (jac_dot,))

        def f_native(jac):
            return parallel_reduce_block_diag(jac, rhs)
        _, h_dot_native = jax.jvp(f_native, (jac,), (jac_dot,))

        assert jnp.allclose(h_dot_prim, h_dot_native, atol=1e-5), \
            f"Max diff: {jnp.max(jnp.abs(h_dot_prim - h_dot_native))}"

    def test_grad_matches_native(self):
        """VJP via primitive matches JAX native AD for both inputs."""
        B, T, N, K = 1, 8, 4, 2
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        def loss_prim(jac, rhs):
            return jnp.sum(_block_diag_via_primitive(jac, rhs) ** 2)

        def loss_native(jac, rhs):
            return jnp.sum(parallel_reduce_block_diag(jac, rhs) ** 2)

        g_prim = jax.grad(loss_prim, argnums=(0, 1))(jac, rhs)
        g_native = jax.grad(loss_native, argnums=(0, 1))(jac, rhs)

        assert jnp.allclose(g_prim[0], g_native[0], atol=1e-5), \
            f"jac grad max diff: {jnp.max(jnp.abs(g_prim[0] - g_native[0]))}"
        assert jnp.allclose(g_prim[1], g_native[1], atol=1e-5), \
            f"rhs grad max diff: {jnp.max(jnp.abs(g_prim[1] - g_native[1]))}"

    def test_3x3_grad_wrt_rhs(self):
        """jax.grad for K=3 block-diagonal reduce via primitive."""
        B, T, N, K = 1, 8, 4, 3
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.2
        rhs = jr.normal(k2, (B, T, N, K))

        def loss(rhs):
            return jnp.sum(_block_diag_via_primitive(jac, rhs) ** 2)

        g = jax.grad(loss)(rhs)
        assert g.shape == rhs.shape
        assert jnp.all(jnp.isfinite(g))
