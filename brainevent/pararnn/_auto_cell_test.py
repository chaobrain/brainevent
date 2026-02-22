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

"""Tests for the general-purpose parallel RNN API (parallel_rnn, AutoRNNCell).

Tests three cell types:
- Diagonal: element-wise gating → diagonal Jacobians
- Dense (Elman): full matrix recurrence → dense Jacobians
- Block-diagonal: grouped hidden states → block-diagonal Jacobians

Each cell is tested for:
- Forward correctness: parallel vs sequential agreement
- Gradient correctness: parallel vs sequential gradient agreement
- Auto-detection: correct Jacobian structure identification
- JIT compatibility
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from brainevent.pararnn._auto_cell import (
    parallel_rnn,
    AutoRNNCell,
    _detect_jacobian_structure,
    _infer_state_dim,
)
from brainevent.pararnn import NewtonConfig

# ============================================================================
# Cell function definitions
# ============================================================================

def _diag_cell(h_prev, x_t, a, w, b):
    """Diagonal cell: element-wise gating."""
    return jax.nn.sigmoid(a * h_prev + w * x_t + b)


def _elman_cell(h_prev, x_t, W_h, W_x, b):
    """Elman RNN: dense Jacobian."""
    return jax.nn.tanh(W_h @ h_prev + W_x @ x_t + b)


def _block_diag_cell(h_prev, x_t, A0, A1, A2, w, b):
    """Block-diagonal cell with 3 blocks of size 2 (state_dim=6)."""
    h_new = jnp.zeros_like(h_prev)
    h_new = h_new.at[:2].set(jax.nn.tanh(A0 @ h_prev[:2] + w[:2] * x_t[:2] + b[:2]))
    h_new = h_new.at[2:4].set(jax.nn.tanh(A1 @ h_prev[2:4] + w[2:4] * x_t[2:4] + b[2:4]))
    h_new = h_new.at[4:6].set(jax.nn.tanh(A2 @ h_prev[4:6] + w[4:6] * x_t[4:6] + b[4:6]))
    return h_new


# ============================================================================
# Helper: create test data
# ============================================================================

def _make_diag_data(key, N=4, D=4, B=2, T=8, scale=0.01):
    k1, k2, k3, k4 = jr.split(key, 4)
    a = jr.normal(k1, (N,)) * 0.1
    w = jr.normal(k2, (N,)) * 0.1
    b = jnp.zeros(N)
    x = jr.normal(k3, (B, T, D)) * scale
    return x, (a, w, b)


def _make_elman_data(key, N=4, D=3, B=2, T=8, scale=0.01):
    k1, k2, k3, k4 = jr.split(key, 4)
    W_h = jr.normal(k1, (N, N)) * 0.1
    W_x = jr.normal(k2, (N, D)) * 0.1
    b = jnp.zeros(N)
    x = jr.normal(k3, (B, T, D)) * scale
    return x, (W_h, W_x, b)


def _make_block_diag_data(key, N=6, D=6, B=2, T=8, scale=0.01):
    k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
    A0 = jr.normal(k1, (2, 2)) * 0.1
    A1 = jr.normal(k2, (2, 2)) * 0.1
    A2 = jr.normal(k3, (2, 2)) * 0.1
    w = jr.normal(k4, (N,)) * 0.1
    b = jnp.zeros(N)
    x = jr.normal(k5, (B, T, D)) * scale
    return x, (A0, A1, A2, w, b)


# ============================================================================
# Test: Jacobian structure auto-detection
# ============================================================================

class TestAutoDetection:
    """Tests for automatic Jacobian structure detection."""

    def test_detect_diagonal(self):
        x, (a, w, b) = _make_diag_data(jr.PRNGKey(0))
        struct, K = _detect_jacobian_structure(_diag_cell, 4, 4, (a, w, b))
        assert struct == 'diagonal'
        assert K == 1

    def test_detect_dense(self):
        x, (W_h, W_x, b) = _make_elman_data(jr.PRNGKey(0))
        struct, K = _detect_jacobian_structure(_elman_cell, 4, 3, (W_h, W_x, b))
        assert struct == 'dense'

    def test_detect_block_diagonal(self):
        x, params = _make_block_diag_data(jr.PRNGKey(0))
        struct, K = _detect_jacobian_structure(_block_diag_cell, 6, 6, params)
        assert struct == 'block_diagonal'
        assert K == 2


# ============================================================================
# Test: State dimension inference
# ============================================================================

class TestStateDimInference:
    """Tests for automatic state dimension inference."""

    def test_infer_elman(self):
        x, params = _make_elman_data(jr.PRNGKey(0), N=4, D=3)
        dim = _infer_state_dim(_elman_cell, x, params)
        assert dim == 4

    def test_infer_diagonal(self):
        x, params = _make_diag_data(jr.PRNGKey(0), N=8, D=8)
        dim = _infer_state_dim(_diag_cell, x, params)
        assert dim == 8


# ============================================================================
# Test: Diagonal cell
# ============================================================================

class TestDiagonalCell:
    """Tests for parallel_rnn with diagonal Jacobian structure."""

    def test_forward_shape(self):
        x, params = _make_diag_data(jr.PRNGKey(0))
        y = parallel_rnn(_diag_cell, x, *params, jacobian_structure='diagonal')
        assert y.shape == (2, 8, 4)
        assert jnp.all(jnp.isfinite(y))

    def test_forward_vs_sequential(self):
        x, params = _make_diag_data(jr.PRNGKey(0))
        y_par = parallel_rnn(_diag_cell, x, *params, jacobian_structure='diagonal')
        y_seq = parallel_rnn(_diag_cell, x, *params, mode='sequential')
        diff = jnp.max(jnp.abs(y_par - y_seq))
        assert diff < 1e-4, f"Forward diff too large: {diff}"

    def test_gradient_finite(self):
        x, (a, w, b) = _make_diag_data(jr.PRNGKey(0))

        def loss(a, w, b):
            return jnp.sum(parallel_rnn(
                _diag_cell, x, a, w, b, jacobian_structure='diagonal',
            ) ** 2)

        grads = jax.grad(loss, argnums=(0, 1, 2))(a, w, b)
        for g in grads:
            assert jnp.all(jnp.isfinite(g))

    def test_gradient_vs_sequential(self):
        x, (a, w, b) = _make_diag_data(jr.PRNGKey(0))
        nc = NewtonConfig(max_its=10)

        def loss_par(a, w, b):
            return jnp.sum(parallel_rnn(
                _diag_cell, x, a, w, b, jacobian_structure='diagonal',
                newton_config=nc,
            ) ** 2)

        def loss_seq(a, w, b):
            return jnp.sum(parallel_rnn(
                _diag_cell, x, a, w, b, mode='sequential',
            ) ** 2)

        g_par = jax.grad(loss_par, argnums=(0, 1, 2))(a, w, b)
        g_seq = jax.grad(loss_seq, argnums=(0, 1, 2))(a, w, b)
        for gp, gs in zip(g_par, g_seq):
            scale = jnp.maximum(jnp.max(jnp.abs(gs)), 1e-6)
            rel_diff = jnp.max(jnp.abs(gp - gs)) / scale
            assert rel_diff < 0.01, f"Gradient rel diff: {rel_diff}"

    @pytest.mark.parametrize('T', [4, 16, 32])
    def test_various_seq_lengths(self, T):
        x, params = _make_diag_data(jr.PRNGKey(0), T=T)
        y = parallel_rnn(_diag_cell, x, *params, jacobian_structure='diagonal')
        assert y.shape == (2, T, 4)
        assert jnp.all(jnp.isfinite(y))

    def test_jit(self):
        x, (a, w, b) = _make_diag_data(jr.PRNGKey(0))

        @jax.jit
        def run(x, a, w, b):
            return parallel_rnn(
                _diag_cell, x, a, w, b, jacobian_structure='diagonal',
            )

        y_eager = parallel_rnn(
            _diag_cell, x, a, w, b, jacobian_structure='diagonal',
        )
        y_jit = run(x, a, w, b)
        assert jnp.allclose(y_eager, y_jit, atol=1e-6)


# ============================================================================
# Test: Dense (Elman) cell
# ============================================================================

class TestDenseCell:
    """Tests for parallel_rnn with dense Jacobian structure."""

    def test_forward_shape(self):
        x, params = _make_elman_data(jr.PRNGKey(0))
        y = parallel_rnn(_elman_cell, x, *params, jacobian_structure='dense')
        assert y.shape == (2, 8, 4)
        assert jnp.all(jnp.isfinite(y))

    def test_forward_vs_sequential(self):
        x, params = _make_elman_data(jr.PRNGKey(0))
        y_par = parallel_rnn(_elman_cell, x, *params, jacobian_structure='dense')
        y_seq = parallel_rnn(_elman_cell, x, *params, mode='sequential')
        diff = jnp.max(jnp.abs(y_par - y_seq))
        assert diff < 1e-4, f"Forward diff too large: {diff}"

    def test_gradient_finite(self):
        x, (W_h, W_x, b) = _make_elman_data(jr.PRNGKey(0))

        def loss(W_h, W_x, b):
            return jnp.sum(parallel_rnn(
                _elman_cell, x, W_h, W_x, b, jacobian_structure='dense',
            ) ** 2)

        grads = jax.grad(loss, argnums=(0, 1, 2))(W_h, W_x, b)
        for g in grads:
            assert jnp.all(jnp.isfinite(g))

    def test_gradient_vs_sequential(self):
        x, (W_h, W_x, b) = _make_elman_data(jr.PRNGKey(0))
        nc = NewtonConfig(max_its=10)

        def loss_par(W_h, W_x, b):
            return jnp.sum(parallel_rnn(
                _elman_cell, x, W_h, W_x, b, jacobian_structure='dense',
                newton_config=nc,
            ) ** 2)

        def loss_seq(W_h, W_x, b):
            return jnp.sum(parallel_rnn(
                _elman_cell, x, W_h, W_x, b, mode='sequential',
            ) ** 2)

        g_par = jax.grad(loss_par, argnums=(0, 1, 2))(W_h, W_x, b)
        g_seq = jax.grad(loss_seq, argnums=(0, 1, 2))(W_h, W_x, b)
        for gp, gs in zip(g_par, g_seq):
            scale = jnp.maximum(jnp.max(jnp.abs(gs)), 1e-6)
            rel_diff = jnp.max(jnp.abs(gp - gs)) / scale
            assert rel_diff < 0.01, f"Gradient rel diff: {rel_diff}"

    @pytest.mark.parametrize('N', [2, 4, 8])
    def test_various_state_dims(self, N):
        x, params = _make_elman_data(jr.PRNGKey(0), N=N, D=3)
        y = parallel_rnn(_elman_cell, x, *params, jacobian_structure='dense')
        assert y.shape == (2, 8, N)
        assert jnp.all(jnp.isfinite(y))

    def test_jit(self):
        x, (W_h, W_x, b) = _make_elman_data(jr.PRNGKey(0))

        @jax.jit
        def run(x, W_h, W_x, b):
            return parallel_rnn(
                _elman_cell, x, W_h, W_x, b, jacobian_structure='dense',
            )

        y_eager = parallel_rnn(
            _elman_cell, x, W_h, W_x, b, jacobian_structure='dense',
        )
        y_jit = run(x, W_h, W_x, b)
        assert jnp.allclose(y_eager, y_jit, atol=1e-6)

    def test_jit_grad(self):
        x, (W_h, W_x, b) = _make_elman_data(jr.PRNGKey(0))

        @jax.jit
        def grad_fn(W_h, W_x, b):
            def loss(W_h, W_x, b):
                return jnp.sum(parallel_rnn(
                    _elman_cell, x, W_h, W_x, b, jacobian_structure='dense',
                ) ** 2)
            return jax.grad(loss, argnums=(0, 1, 2))(W_h, W_x, b)

        grads = grad_fn(W_h, W_x, b)
        for g in grads:
            assert jnp.all(jnp.isfinite(g))


# ============================================================================
# Test: Block-diagonal cell
# ============================================================================

class TestBlockDiagCell:
    """Tests for parallel_rnn with block-diagonal Jacobian structure."""

    def test_forward_shape(self):
        x, params = _make_block_diag_data(jr.PRNGKey(0))
        y = parallel_rnn(
            _block_diag_cell, x, *params,
            jacobian_structure='block_diagonal', block_size=2,
        )
        assert y.shape == (2, 8, 6)
        assert jnp.all(jnp.isfinite(y))

    def test_forward_vs_sequential(self):
        x, params = _make_block_diag_data(jr.PRNGKey(0))
        y_par = parallel_rnn(
            _block_diag_cell, x, *params,
            jacobian_structure='block_diagonal', block_size=2,
        )
        y_seq = parallel_rnn(_block_diag_cell, x, *params, mode='sequential')
        diff = jnp.max(jnp.abs(y_par - y_seq))
        assert diff < 1e-3, f"Forward diff too large: {diff}"

    def test_gradient_finite(self):
        x, (A0, A1, A2, w, b) = _make_block_diag_data(jr.PRNGKey(0))

        def loss(A0, A1, A2, w, b):
            return jnp.sum(parallel_rnn(
                _block_diag_cell, x, A0, A1, A2, w, b,
                jacobian_structure='block_diagonal', block_size=2,
            ) ** 2)

        grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(A0, A1, A2, w, b)
        for g in grads:
            assert jnp.all(jnp.isfinite(g))

    def test_gradient_vs_sequential(self):
        x, (A0, A1, A2, w, b) = _make_block_diag_data(jr.PRNGKey(0))
        nc = NewtonConfig(max_its=10)

        def loss_par(A0, A1, A2, w, b):
            return jnp.sum(parallel_rnn(
                _block_diag_cell, x, A0, A1, A2, w, b,
                jacobian_structure='block_diagonal', block_size=2,
                newton_config=nc,
            ) ** 2)

        def loss_seq(A0, A1, A2, w, b):
            return jnp.sum(parallel_rnn(
                _block_diag_cell, x, A0, A1, A2, w, b, mode='sequential',
            ) ** 2)

        g_par = jax.grad(loss_par, argnums=(0, 1, 2, 3, 4))(A0, A1, A2, w, b)
        g_seq = jax.grad(loss_seq, argnums=(0, 1, 2, 3, 4))(A0, A1, A2, w, b)
        for gp, gs in zip(g_par, g_seq):
            scale = jnp.maximum(jnp.max(jnp.abs(gs)), 1e-6)
            rel_diff = jnp.max(jnp.abs(gp - gs)) / scale
            assert rel_diff < 0.01, f"Gradient rel diff: {rel_diff}"


# ============================================================================
# Test: Auto mode (auto-detect + auto state_dim)
# ============================================================================

class TestAutoMode:
    """Tests for fully automatic parallel_rnn (auto-detect everything)."""

    def test_auto_elman(self):
        x, params = _make_elman_data(jr.PRNGKey(0))
        y_auto = parallel_rnn(_elman_cell, x, *params)
        y_seq = parallel_rnn(_elman_cell, x, *params, mode='sequential')
        diff = jnp.max(jnp.abs(y_auto - y_seq))
        assert diff < 1e-4, f"Auto Elman diff: {diff}"

    def test_auto_diagonal(self):
        x, params = _make_diag_data(jr.PRNGKey(0))
        y_auto = parallel_rnn(_diag_cell, x, *params)
        y_seq = parallel_rnn(_diag_cell, x, *params, mode='sequential')
        diff = jnp.max(jnp.abs(y_auto - y_seq))
        assert diff < 1e-4, f"Auto diagonal diff: {diff}"

    def test_auto_jit(self):
        x, (W_h, W_x, b) = _make_elman_data(jr.PRNGKey(0))

        @jax.jit
        def run(x, W_h, W_x, b):
            return parallel_rnn(_elman_cell, x, W_h, W_x, b)

        y = run(x, W_h, W_x, b)
        assert y.shape == (2, 8, 4)
        assert jnp.all(jnp.isfinite(y))

    def test_explicit_state_dim(self):
        x, params = _make_elman_data(jr.PRNGKey(0))
        y = parallel_rnn(
            _elman_cell, x, *params,
            state_dim=4, jacobian_structure='dense',
        )
        assert y.shape == (2, 8, 4)

    def test_newton_config(self):
        x, params = _make_elman_data(jr.PRNGKey(0))
        y = parallel_rnn(
            _elman_cell, x, *params,
            jacobian_structure='dense',
            newton_config=NewtonConfig(max_its=5, omega_sor=0.9),
        )
        assert y.shape == (2, 8, 4)
        assert jnp.all(jnp.isfinite(y))


# ============================================================================
# Test: Dense parallel reduction
# ============================================================================

class TestDenseParallelReduce:
    """Tests for parallel_reduce_dense function."""

    def test_forward_correctness(self):
        """Compare parallel scan vs sequential loop."""
        from brainevent.pararnn._parallel_reduce import _parallel_reduce_dense_jax

        key = jr.PRNGKey(0)
        B, T, N = 2, 8, 4
        jac = jr.normal(key, (B, T, N, N)) * 0.1
        rhs = jr.normal(jr.PRNGKey(1), (B, T, N)) * 0.1

        # Sequential reference
        h = jnp.zeros((B, N))
        hs = []
        for t in range(T):
            h = (jac[:, t] @ h[..., None]).squeeze(-1) + rhs[:, t]
            hs.append(h)
        h_seq = jnp.stack(hs, axis=1)

        h_par = _parallel_reduce_dense_jax(jac, rhs)
        diff = jnp.max(jnp.abs(h_seq - h_par))
        assert diff < 1e-5, f"Dense reduce forward diff: {diff}"

    def test_gradient_rhs(self):
        """Test gradient w.r.t. rhs (linear)."""
        from brainevent.pararnn import parallel_reduce_dense

        key = jr.PRNGKey(0)
        B, T, N = 2, 8, 3
        jac = jr.normal(key, (B, T, N, N)) * 0.1
        rhs = jr.normal(jr.PRNGKey(1), (B, T, N)) * 0.1

        def loss(rhs):
            return jnp.sum(parallel_reduce_dense(jac, rhs) ** 2)

        g = jax.grad(loss)(rhs)
        assert g.shape == rhs.shape
        assert jnp.all(jnp.isfinite(g))

    @pytest.mark.parametrize('N', [2, 4, 6])
    def test_various_dims(self, N):
        from brainevent.pararnn._parallel_reduce import _parallel_reduce_dense_jax

        key = jr.PRNGKey(0)
        jac = jr.normal(key, (2, 8, N, N)) * 0.1
        rhs = jr.normal(jr.PRNGKey(1), (2, 8, N)) * 0.1
        h = _parallel_reduce_dense_jax(jac, rhs)
        assert h.shape == (2, 8, N)
        assert jnp.all(jnp.isfinite(h))
