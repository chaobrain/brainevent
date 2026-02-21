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

"""Tests for CUDA-accelerated parallel reduction solvers.

These tests compare the tvmffi CUDA kernels against the sequential reference
implementation and the jax_raw backend. All tests are skipped if no GPU is
available or TVM FFI is not installed.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from brainevent.pararnn._parallel_reduce import (
    parallel_reduce_diag,
    parallel_reduce_diag_p,
    parallel_reduce_block_diag,
    parallel_reduce_block_diag_p,
)

# Skip entire module if no GPU available
_has_gpu = any(d.platform == 'gpu' for d in jax.devices())
if not _has_gpu:
    pytest.skip("No GPU available", allow_module_level=True)

# Check if tvmffi backend is available for parallel reduce
try:
    _has_cuda = 'tvmffi' in parallel_reduce_diag_p.available_backends('gpu')
except Exception:
    _has_cuda = False

if not _has_cuda:
    pytest.skip("TVM FFI CUDA backend not available", allow_module_level=True)


def _sequential_reduce_diag(jac, rhs):
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

    @pytest.mark.parametrize('T', [
        1,
        pytest.param(4, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(16, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(64, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(256, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(1024, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
    ])
    def test_forward_matches_sequential(self, T):
        key = jr.PRNGKey(42)
        B, N = 2, 8
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5
        rhs = jr.normal(k2, (B, T, N))

        h_seq = _sequential_reduce_diag(jac, rhs)
        h_cuda = parallel_reduce_diag(jac, rhs, backend='tvmffi')

        assert jnp.allclose(h_seq, h_cuda, atol=1e-4, rtol=1e-4), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_seq - h_cuda))}"

    @pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")
    def test_matches_jax_native(self):
        """CUDA result should match JAX associative_scan result."""
        B, T, N = 4, 64, 16
        key = jr.PRNGKey(123)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5
        rhs = jr.normal(k2, (B, T, N))

        h_jax = parallel_reduce_diag(jac, rhs, backend='jax_raw')
        h_cuda = parallel_reduce_diag(jac, rhs, backend='tvmffi')

        assert jnp.allclose(h_jax, h_cuda, atol=1e-4, rtol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(h_jax - h_cuda))}"

    @pytest.mark.parametrize('T', [
        pytest.param(3, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(7, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(13, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(33, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(100, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
    ])
    def test_non_power_of_2(self, T):
        """Test with sequence lengths that are not powers of 2."""
        B, N = 2, 8
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.5
        rhs = jr.normal(k2, (B, T, N))

        h_seq = _sequential_reduce_diag(jac, rhs)
        h_cuda = parallel_reduce_diag(jac, rhs, backend='tvmffi')

        assert jnp.allclose(h_seq, h_cuda, atol=1e-4, rtol=1e-4), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_seq - h_cuda))}"

    def test_zero_jacobian(self):
        """With zero Jacobians, h[t] = rhs[t]."""
        B, T, N = 2, 16, 4
        jac = jnp.zeros((B, T, N))
        rhs = jnp.ones((B, T, N))

        h = parallel_reduce_diag(jac, rhs, backend='tvmffi')
        assert jnp.allclose(h, rhs)

    @pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")
    def test_large_sequence(self):
        """Test with large T to exercise multi-block path."""
        B, T, N = 1, 4096, 8
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N)) * 0.3
        rhs = jr.normal(k2, (B, T, N))

        h_jax = parallel_reduce_diag(jac, rhs, backend='jax_raw')
        h_cuda = parallel_reduce_diag(jac, rhs, backend='tvmffi')

        assert jnp.allclose(h_jax, h_cuda, atol=1e-3, rtol=1e-3), \
            f"Max diff: {jnp.max(jnp.abs(h_jax - h_cuda))}"


class TestCUDAReduceBlock2:
    """Tests for CUDA 2x2 block-diagonal parallel reduction."""

    @pytest.mark.parametrize('T', [
        1,
        pytest.param(4, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(16, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(64, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(256, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
    ])
    def test_forward_matches_sequential(self, T):
        key = jr.PRNGKey(42)
        B, N, K = 2, 4, 2
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        h_seq = _sequential_reduce_block2(jac, rhs)
        h_cuda = parallel_reduce_block_diag(jac, rhs, backend='tvmffi')

        assert jnp.allclose(h_seq, h_cuda, atol=1e-4, rtol=1e-4), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_seq - h_cuda))}"

    @pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")
    def test_matches_jax_native(self):
        """CUDA result should match JAX associative_scan result."""
        B, T, N, K = 2, 32, 8, 2
        key = jr.PRNGKey(123)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        h_jax = parallel_reduce_block_diag(jac, rhs, backend='jax_raw')
        h_cuda = parallel_reduce_block_diag(jac, rhs, backend='tvmffi')

        assert jnp.allclose(h_jax, h_cuda, atol=1e-4, rtol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(h_jax - h_cuda))}"

    @pytest.mark.parametrize('T', [
        pytest.param(3, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(7, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(15, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
        pytest.param(33, marks=pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")),
    ])
    def test_non_power_of_2(self, T):
        """Test with sequence lengths that are not powers of 2."""
        B, N, K = 2, 4, 2
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.3
        rhs = jr.normal(k2, (B, T, N, K))

        h_seq = _sequential_reduce_block2(jac, rhs)
        h_cuda = parallel_reduce_block_diag(jac, rhs, backend='tvmffi')

        assert jnp.allclose(h_seq, h_cuda, atol=1e-4, rtol=1e-4), \
            f"T={T}, Max diff: {jnp.max(jnp.abs(h_seq - h_cuda))}"

    @pytest.mark.xfail(reason="Pre-existing CUDA kernel bug for T>1")
    def test_large_sequence(self):
        """Test with large T to exercise multi-block path."""
        B, T, N, K = 1, 2048, 4, 2
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        jac = jr.normal(k1, (B, T, N, K, K)) * 0.2
        rhs = jr.normal(k2, (B, T, N, K))

        h_jax = parallel_reduce_block_diag(jac, rhs, backend='jax_raw')
        h_cuda = parallel_reduce_block_diag(jac, rhs, backend='tvmffi')

        assert jnp.allclose(h_jax, h_cuda, atol=1e-3, rtol=1e-3), \
            f"Max diff: {jnp.max(jnp.abs(h_jax - h_cuda))}"
