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

"""Tests for LSTMCIFGDiagMH module."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from brainevent.pararnn._cell import apply_rnn
from brainevent.pararnn._init import INITIALIZERS
from brainevent.pararnn._lstm import LSTMCIFGDiagMH, LSTMCIFGDiagMHImpl
from brainevent.pararnn._newton import NewtonConfig
from brainevent.pararnn._nonlinearities import get_nonlinearity


def _make_lstm_params(key, input_dim=8, state_dim=16, num_heads=2):
    """Create LSTM parameters for testing."""
    head_input_dim = input_dim // num_heads
    head_state_dim = state_dim // num_heads

    k1, k2, k3, k4 = jr.split(key, 4)
    A = INITIALIZERS['xlstm'](k1, (3, state_dim), fan_in=state_dim)
    B = INITIALIZERS['xavier_uniform'](
        k2, (num_heads, head_input_dim, 3, head_state_dim),
        fan_in=head_input_dim, fan_out=head_state_dim,
    )
    C = INITIALIZERS['xlstm'](k3, (2, state_dim), fan_in=state_dim)
    b = INITIALIZERS['bias_minus_linspace'](k4, (3, state_dim), fan_in=0)

    nonlin_f, deriv_f = get_nonlinearity('sigmoid')
    nonlin_o, deriv_o = get_nonlinearity('sigmoid')
    nonlin_c, deriv_c = get_nonlinearity('tanh')
    nonlin_state, deriv_state = get_nonlinearity('tanh')

    return (A, B, C, b,
            nonlin_f, nonlin_o, nonlin_c, nonlin_state,
            deriv_f, deriv_o, deriv_c, deriv_state)


class TestLSTMCIFGDiagMHImpl:
    """Test the LSTM implementation directly."""

    def test_recurrence_step_shape(self):
        """Test that recurrence_step produces correct output shape."""
        B, T, input_dim, state_dim = 2, 16, 8, 16
        internal_dim = 2 * state_dim
        params = _make_lstm_params(jr.PRNGKey(0), input_dim, state_dim)

        x = jr.normal(jr.PRNGKey(1), (B, T, input_dim))
        h = jnp.zeros((B, T, internal_dim))
        h_new = LSTMCIFGDiagMHImpl.recurrence_step(x, h, *params)
        assert h_new.shape == (B, T, internal_dim)

    def test_sequential_mode(self):
        """Test sequential evaluation produces finite output."""
        B, T, input_dim, state_dim = 2, 16, 8, 16
        internal_dim = 2 * state_dim
        params = _make_lstm_params(jr.PRNGKey(0), input_dim, state_dim)
        x = jr.normal(jr.PRNGKey(1), (B, T, input_dim))

        y = apply_rnn(LSTMCIFGDiagMHImpl, x, internal_dim, 'sequential',
                      NewtonConfig(), *params)
        # post_process extracts h from [c, h], so output is state_dim
        assert y.shape == (B, T, state_dim)
        assert jnp.all(jnp.isfinite(y))

    def test_parallel_mode(self):
        """Test parallel evaluation produces finite output."""
        B, T, input_dim, state_dim = 2, 16, 8, 16
        internal_dim = 2 * state_dim
        params = _make_lstm_params(jr.PRNGKey(0), input_dim, state_dim)
        x = jr.normal(jr.PRNGKey(1), (B, T, input_dim))

        y = apply_rnn(LSTMCIFGDiagMHImpl, x, internal_dim, 'parallel',
                      NewtonConfig(max_its=5), *params)
        assert y.shape == (B, T, state_dim)
        assert jnp.all(jnp.isfinite(y))

    def test_sequential_vs_parallel_convergence(self):
        """More Newton iterations should reduce the gap between modes."""
        B, T, input_dim, state_dim = 2, 8, 8, 16
        internal_dim = 2 * state_dim
        params = _make_lstm_params(jr.PRNGKey(0), input_dim, state_dim)
        x = jr.normal(jr.PRNGKey(1), (B, T, input_dim)) * 0.01

        y_seq = apply_rnn(LSTMCIFGDiagMHImpl, x, internal_dim, 'sequential',
                          NewtonConfig(), *params)

        diff_5 = jnp.max(jnp.abs(
            y_seq - apply_rnn(LSTMCIFGDiagMHImpl, x, internal_dim, 'parallel',
                              NewtonConfig(max_its=5), *params)
        ))
        diff_20 = jnp.max(jnp.abs(
            y_seq - apply_rnn(LSTMCIFGDiagMHImpl, x, internal_dim, 'parallel',
                              NewtonConfig(max_its=20), *params)
        ))

        # More iterations -> better convergence
        assert diff_20 <= diff_5 + 1e-6, \
            f"20 its ({diff_20}) should be <= 5 its ({diff_5})"
        assert diff_20 < 1.0, f"20 iterations max diff: {diff_20}"

    def test_jacobians_shape(self):
        """Test Jacobian computation produces correct shape."""
        B, T, input_dim, state_dim = 2, 16, 8, 16
        internal_dim = 2 * state_dim
        params = _make_lstm_params(jr.PRNGKey(0), input_dim, state_dim)
        x = jr.normal(jr.PRNGKey(1), (B, T, input_dim))
        h = jr.normal(jr.PRNGKey(2), (B, T, internal_dim)) * 0.1

        jac = LSTMCIFGDiagMHImpl.compute_jacobians(h, x, *params)
        # Block-diagonal 2x2: (B, T, N, 2, 2) where N = state_dim
        assert jac.shape == (B, T, state_dim, 2, 2)

    def test_jacobians_bwd_shape(self):
        """Test backward Jacobian computation."""
        B, T, input_dim, state_dim = 2, 16, 8, 16
        internal_dim = 2 * state_dim
        params = _make_lstm_params(jr.PRNGKey(0), input_dim, state_dim)
        x = jr.normal(jr.PRNGKey(1), (B, T, input_dim))
        h = jr.normal(jr.PRNGKey(2), (B, T, internal_dim)) * 0.1

        jac_bwd = LSTMCIFGDiagMHImpl.compute_jacobians_bwd(h, x, *params)
        assert jac_bwd.shape == (B, T, state_dim, 2, 2)
        # First timestep should be zero
        assert jnp.allclose(jac_bwd[:, 0, :, :, :], 0.0)


class TestLSTMCIFGDiagMHGradients:
    """Test gradient computation for LSTM."""

    def test_parallel_gradient(self):
        """Test that gradients flow through parallel mode."""
        B, T, input_dim, state_dim = 2, 8, 8, 16
        internal_dim = 2 * state_dim
        params = _make_lstm_params(jr.PRNGKey(0), input_dim, state_dim)
        x = jr.normal(jr.PRNGKey(1), (B, T, input_dim)) * 0.1

        def loss(x, A, B_w, C, b_bias, *rest):
            y = apply_rnn(LSTMCIFGDiagMHImpl, x, internal_dim, 'parallel',
                          NewtonConfig(max_its=3),
                          A, B_w, C, b_bias, *rest)
            return jnp.sum(y ** 2)

        grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(x, *params)
        assert grads[0].shape == x.shape
        assert grads[1].shape == params[0].shape  # A
        assert grads[2].shape == params[1].shape  # B
        assert grads[3].shape == params[2].shape  # C
        assert grads[4].shape == params[3].shape  # b
        for i, g in enumerate(grads):
            assert jnp.all(jnp.isfinite(g)), f"Gradient {i} contains NaN/Inf"

    def test_sequential_gradient(self):
        """Test that gradients flow through sequential mode."""
        B, T, input_dim, state_dim = 2, 8, 8, 16
        internal_dim = 2 * state_dim
        params = _make_lstm_params(jr.PRNGKey(0), input_dim, state_dim)
        x = jr.normal(jr.PRNGKey(1), (B, T, input_dim)) * 0.1

        def loss(x, A, B_w, C, b_bias, *rest):
            y = apply_rnn(LSTMCIFGDiagMHImpl, x, internal_dim, 'sequential',
                          NewtonConfig(),
                          A, B_w, C, b_bias, *rest)
            return jnp.sum(y ** 2)

        grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(x, *params)
        for i, g in enumerate(grads):
            assert jnp.all(jnp.isfinite(g)), f"Gradient {i} contains NaN/Inf"


class TestLSTMCIFGDiagMHModule:
    """Test the brainstate.nn.Module wrapper."""

    def test_construction(self):
        """Test module construction."""
        lstm = LSTMCIFGDiagMH(input_dim=8, state_dim=16, num_heads=2)
        assert lstm.input_dim == 8
        assert lstm.state_dim == 16
        assert lstm.internal_state_dim == 32

    def test_forward(self):
        """Test module forward pass."""
        lstm = LSTMCIFGDiagMH(input_dim=8, state_dim=16, num_heads=2, mode='sequential')
        x = jnp.ones((2, 10, 8))
        y = lstm(x)
        assert y.shape == (2, 10, 16)  # Only h is returned, not [c, h]
        assert jnp.all(jnp.isfinite(y))

    @pytest.mark.parametrize('num_heads', [1, 2, 4])
    def test_multi_head(self, num_heads):
        """Test with different numbers of heads."""
        input_dim = 8
        state_dim = 16
        lstm = LSTMCIFGDiagMH(
            input_dim=input_dim, state_dim=state_dim,
            num_heads=num_heads, mode='sequential',
        )
        x = jnp.ones((2, 10, input_dim))
        y = lstm(x)
        assert y.shape == (2, 10, state_dim)

    def test_invalid_heads(self):
        """Test that invalid num_heads raises error."""
        with pytest.raises(ValueError, match="must divide"):
            LSTMCIFGDiagMH(input_dim=7, state_dim=16, num_heads=2)

    def test_jit_compatible(self):
        """Test that the module works under jax.jit."""
        lstm = LSTMCIFGDiagMH(input_dim=8, state_dim=16, num_heads=2, mode='sequential')
        x = jnp.ones((2, 10, 8))

        y_eager = lstm(x)
        y_jit = jax.jit(lstm)(x)
        assert jnp.allclose(y_eager, y_jit, atol=1e-6)
