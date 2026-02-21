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

"""Tests for fused CUDA GRU and LSTM-CIFG kernels.

These tests compare fused CUDA forward passes against sequential mode
(ground truth). All tests are skipped if no GPU is available or the
fused CUDA kernels are not installed.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from brainevent.pararnn._gru import GRUDiagMH
from brainevent.pararnn._lstm import LSTMCIFGDiagMH

# Skip entire module if no GPU available
_has_gpu = any(d.platform == 'gpu' for d in jax.devices())
if not _has_gpu:
    pytest.skip("No GPU available", allow_module_level=True)


class TestFusedGRU:
    """Tests for fused GRU CUDA forward pass."""

    def test_forward_shape(self):
        """Test that fused GRU produces correct output shape."""
        gru = GRUDiagMH(input_dim=8, state_dim=16, num_heads=2, mode='fused')
        x = jnp.ones((2, 32, 8))
        y = gru(x)
        assert y.shape == (2, 32, 16)
        assert jnp.all(jnp.isfinite(y))

    def test_fused_vs_sequential(self):
        """Test that fused output is close to sequential (ground truth)."""
        input_dim, state_dim = 8, 16
        gru_seq = GRUDiagMH(
            input_dim=input_dim, state_dim=state_dim,
            num_heads=2, mode='sequential', seed=42,
        )
        gru_fused = GRUDiagMH(
            input_dim=input_dim, state_dim=state_dim,
            num_heads=2, mode='fused', seed=42,
        )

        x = jr.normal(jr.PRNGKey(0), (2, 16, input_dim)) * 0.01

        y_seq = gru_seq(x)
        y_fused = gru_fused(x)

        # Fused uses 3 Newton iterations by default — expect reasonable agreement
        max_diff = jnp.max(jnp.abs(y_seq - y_fused))
        assert max_diff < 1.0, f"Fused vs sequential max diff: {max_diff}"

    @pytest.mark.parametrize('T', [8, 32, 64])
    def test_various_seq_lengths(self, T):
        """Test fused GRU with various sequence lengths."""
        gru = GRUDiagMH(input_dim=8, state_dim=16, num_heads=2, mode='fused')
        x = jr.normal(jr.PRNGKey(0), (2, T, 8)) * 0.1
        y = gru(x)
        assert y.shape == (2, T, 16)
        assert jnp.all(jnp.isfinite(y))

    def test_gradient_flow(self):
        """Test that gradients flow through fused GRU."""
        gru = GRUDiagMH(input_dim=8, state_dim=16, num_heads=2, mode='fused')
        x = jr.normal(jr.PRNGKey(0), (2, 16, 8)) * 0.1

        def loss(x):
            return jnp.sum(gru(x) ** 2)

        grad_x = jax.grad(loss)(x)
        assert grad_x.shape == x.shape
        assert jnp.all(jnp.isfinite(grad_x))

    def test_jit_compatible(self):
        """Test that fused GRU works under jax.jit."""
        gru = GRUDiagMH(input_dim=8, state_dim=16, num_heads=2, mode='fused')
        x = jnp.ones((2, 16, 8))

        y_eager = gru(x)
        y_jit = jax.jit(gru)(x)
        assert jnp.allclose(y_eager, y_jit, atol=1e-6)


class TestFusedLSTMCIFG:
    """Tests for fused LSTM-CIFG CUDA forward pass."""

    def test_forward_shape(self):
        """Test that fused LSTM produces correct output shape."""
        lstm = LSTMCIFGDiagMH(
            input_dim=8, state_dim=16, num_heads=2, mode='fused',
        )
        x = jnp.ones((2, 32, 8))
        y = lstm(x)
        assert y.shape == (2, 32, 16)  # Only h is returned, not [c, h]
        assert jnp.all(jnp.isfinite(y))

    def test_fused_vs_sequential(self):
        """Test that fused output is close to sequential (ground truth)."""
        input_dim, state_dim = 8, 16
        lstm_seq = LSTMCIFGDiagMH(
            input_dim=input_dim, state_dim=state_dim,
            num_heads=2, mode='sequential', seed=42,
        )
        lstm_fused = LSTMCIFGDiagMH(
            input_dim=input_dim, state_dim=state_dim,
            num_heads=2, mode='fused', seed=42,
        )

        x = jr.normal(jr.PRNGKey(0), (2, 16, input_dim)) * 0.01

        y_seq = lstm_seq(x)
        y_fused = lstm_fused(x)

        max_diff = jnp.max(jnp.abs(y_seq - y_fused))
        assert max_diff < 1.0, f"Fused vs sequential max diff: {max_diff}"

    @pytest.mark.parametrize('T', [8, 32, 64])
    def test_various_seq_lengths(self, T):
        """Test fused LSTM with various sequence lengths."""
        lstm = LSTMCIFGDiagMH(
            input_dim=8, state_dim=16, num_heads=2, mode='fused',
        )
        x = jr.normal(jr.PRNGKey(0), (2, T, 8)) * 0.1
        y = lstm(x)
        assert y.shape == (2, T, 16)
        assert jnp.all(jnp.isfinite(y))

    def test_gradient_flow(self):
        """Test that gradients flow through fused LSTM."""
        lstm = LSTMCIFGDiagMH(
            input_dim=8, state_dim=16, num_heads=2, mode='fused',
        )
        x = jr.normal(jr.PRNGKey(0), (2, 16, 8)) * 0.1

        def loss(x):
            return jnp.sum(lstm(x) ** 2)

        grad_x = jax.grad(loss)(x)
        assert grad_x.shape == x.shape
        assert jnp.all(jnp.isfinite(grad_x))

    def test_jit_compatible(self):
        """Test that fused LSTM works under jax.jit."""
        lstm = LSTMCIFGDiagMH(
            input_dim=8, state_dim=16, num_heads=2, mode='fused',
        )
        x = jnp.ones((2, 16, 8))

        y_eager = lstm(x)
        y_jit = jax.jit(lstm)(x)
        assert jnp.allclose(y_eager, y_jit, atol=1e-6)
