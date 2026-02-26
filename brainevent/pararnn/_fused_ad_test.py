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

"""Tests for JVP/transpose rules of fused GRU and LSTM-CIFG primitives.

Testing strategy:
  - Call each primitive directly (via XLACustomKernel.__call__) with
    ``backend='jax_raw'`` to exercise the registered JVP rules.
  - Compare against "native" JAX AD: calling the jax_raw kernel function
    directly (bypassing the primitive), so JAX differentiates through the
    pure JAX ops natively.
  - For fused forward primitives (GRU/LSTM), the JVP rule uses implicit
    differentiation at convergence. The native JAX AD differentiates
    through the Newton iteration. With small inputs (good convergence),
    these should agree closely.
  - For fused backward primitives (GRU/LSTM), the computation is linear
    in ``grad_y``, so primitive AD and native AD should agree exactly.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from brainevent.pararnn._fused import (
    fused_gru_diag_fwd_p,
    fused_gru_diag_bwd_p,
    fused_lstm_cifg_diag_fwd_p,
    fused_lstm_cifg_diag_bwd_p,
    _fused_gru_fwd_jax_kernel,
    _fused_gru_bwd_jax_kernel,
    _fused_lstm_fwd_jax_kernel,
    _fused_lstm_bwd_jax_kernel,
)

# Skip entire module if no GPU available
_has_gpu = any(d.platform == 'gpu' for d in jax.devices())
if not _has_gpu:
    pytest.skip("No GPU available", allow_module_level=True)


# =============================================================================
# Helpers: call primitives directly
# =============================================================================

def _gru_fwd_via_primitive(A, Bxpb):
    """Call fused GRU forward through the XLACustomKernel primitive."""
    B, T = Bxpb.shape[0], Bxpb.shape[1]
    N = A.shape[1]
    return fused_gru_diag_fwd_p(
        A, Bxpb,
        A_info=jax.ShapeDtypeStruct(A.shape, A.dtype),
        outs=[jax.ShapeDtypeStruct((B, T, N), A.dtype)],
        backend='jax_raw',
    )[0]


def _gru_fwd_native(A, Bxpb):
    """Call fused GRU forward via pure JAX (no primitive)."""
    kernel = _fused_gru_fwd_jax_kernel()
    return kernel(A, Bxpb)[0]


def _gru_bwd_via_primitive(grad_y, h, A, Bxpb):
    """Call fused GRU backward through the XLACustomKernel primitive."""
    return fused_gru_diag_bwd_p(
        grad_y, h, A, Bxpb,
        A_info=jax.ShapeDtypeStruct(A.shape, A.dtype),
        outs=[jax.ShapeDtypeStruct(grad_y.shape, grad_y.dtype)],
        backend='jax_raw',
    )[0]


def _gru_bwd_native(grad_y, h, A, Bxpb):
    """Call fused GRU backward via pure JAX (no primitive)."""
    kernel = _fused_gru_bwd_jax_kernel()
    return kernel(grad_y, h, A, Bxpb)[0]


def _lstm_fwd_via_primitive(A, Bxpb, C):
    """Call fused LSTM-CIFG forward through the XLACustomKernel primitive."""
    B, T = Bxpb.shape[0], Bxpb.shape[1]
    N = A.shape[1]
    return fused_lstm_cifg_diag_fwd_p(
        A, Bxpb, C,
        A_info=jax.ShapeDtypeStruct(A.shape, A.dtype),
        outs=[jax.ShapeDtypeStruct((B, T, 2, N), A.dtype)],
        backend='jax_raw',
    )[0]


def _lstm_fwd_native(A, Bxpb, C):
    """Call fused LSTM-CIFG forward via pure JAX (no primitive)."""
    kernel = _fused_lstm_fwd_jax_kernel()
    return kernel(A, Bxpb, C)[0]


def _lstm_bwd_via_primitive(grad_y, full_state, A, Bxpb, C):
    """Call fused LSTM-CIFG backward through the XLACustomKernel primitive."""
    B, T = grad_y.shape[0], grad_y.shape[1]
    N = A.shape[1]
    return fused_lstm_cifg_diag_bwd_p(
        grad_y, full_state, A, Bxpb, C,
        A_info=jax.ShapeDtypeStruct(A.shape, A.dtype),
        outs=[jax.ShapeDtypeStruct((B, T, 2, N), A.dtype)],
        backend='jax_raw',
    )[0]


def _lstm_bwd_native(grad_y, full_state, A, Bxpb, C):
    """Call fused LSTM-CIFG backward via pure JAX (no primitive)."""
    kernel = _fused_lstm_bwd_jax_kernel()
    return kernel(grad_y, full_state, A, Bxpb, C)[0]


# =============================================================================
# Fused GRU Forward AD Tests
# =============================================================================

class TestFusedGRUForwardAD:
    """Tests for JVP/grad of fused_gru_diag_fwd_p.

    The JVP rule uses implicit differentiation (assumes converged Newton).
    Native JAX AD differentiates through the Newton iteration. These agree
    when convergence is good. We use small inputs (scale=0.02) so that
    3 Newton iterations suffice.
    """

    # Small scale ensures 3 Newton iterations converge well.
    # Tolerance: implicit diff agrees with unrolled diff within _tol.
    _scale = 0.01
    _tol = 0.015

    def _make_data(self, key, B=2, T=8, N=4):
        k1, k2 = jr.split(key)
        A = jr.normal(k1, (3, N)) * self._scale
        Bxpb = jr.normal(k2, (B, T, 3, N)) * self._scale
        return A, Bxpb

    def test_grad_wrt_A(self):
        """jax.grad through fused GRU forward w.r.t. A produces finite results."""
        A, Bxpb = self._make_data(jr.PRNGKey(0))

        def loss(A):
            return jnp.sum(_gru_fwd_via_primitive(A, Bxpb) ** 2)

        g = jax.grad(loss)(A)
        assert g.shape == A.shape
        assert jnp.all(jnp.isfinite(g))

    def test_grad_wrt_Bxpb(self):
        """jax.grad through fused GRU forward w.r.t. Bxpb produces finite results."""
        A, Bxpb = self._make_data(jr.PRNGKey(0))

        def loss(Bxpb):
            return jnp.sum(_gru_fwd_via_primitive(A, Bxpb) ** 2)

        g = jax.grad(loss)(Bxpb)
        assert g.shape == Bxpb.shape
        assert jnp.all(jnp.isfinite(g))

    def test_jvp_A_matches_native(self):
        """JVP w.r.t. A via primitive matches native JAX AD."""
        A, Bxpb = self._make_data(jr.PRNGKey(42))
        A_dot = jr.normal(jr.PRNGKey(99), A.shape) * self._scale

        def f_prim(A):
            return _gru_fwd_via_primitive(A, Bxpb)

        _, h_dot_prim = jax.jvp(f_prim, (A,), (A_dot,))

        def f_native(A):
            return _gru_fwd_native(A, Bxpb)

        _, h_dot_native = jax.jvp(f_native, (A,), (A_dot,))

        max_diff = jnp.max(jnp.abs(h_dot_prim - h_dot_native))
        assert max_diff < self._tol, f"Max diff: {max_diff}"

    def test_jvp_Bxpb_matches_native(self):
        """JVP w.r.t. Bxpb via primitive matches native JAX AD."""
        A, Bxpb = self._make_data(jr.PRNGKey(42))
        Bxpb_dot = jr.normal(jr.PRNGKey(99), Bxpb.shape) * self._scale

        def f_prim(Bxpb):
            return _gru_fwd_via_primitive(A, Bxpb)

        _, h_dot_prim = jax.jvp(f_prim, (Bxpb,), (Bxpb_dot,))

        def f_native(Bxpb):
            return _gru_fwd_native(A, Bxpb)

        _, h_dot_native = jax.jvp(f_native, (Bxpb,), (Bxpb_dot,))

        max_diff = jnp.max(jnp.abs(h_dot_prim - h_dot_native))
        assert max_diff < self._tol, f"Max diff: {max_diff}"

    def test_grad_A_matches_native(self):
        """VJP (grad) w.r.t. A via primitive matches native JAX AD."""
        A, Bxpb = self._make_data(jr.PRNGKey(42))

        def loss_prim(A):
            return jnp.sum(_gru_fwd_via_primitive(A, Bxpb) ** 2)

        def loss_native(A):
            return jnp.sum(_gru_fwd_native(A, Bxpb) ** 2)

        g_prim = jax.grad(loss_prim)(A)
        g_native = jax.grad(loss_native)(A)

        max_diff = jnp.max(jnp.abs(g_prim - g_native))
        assert max_diff < self._tol, f"Max diff: {max_diff}"

    def test_grad_Bxpb_matches_native(self):
        """VJP (grad) w.r.t. Bxpb via primitive matches native JAX AD."""
        A, Bxpb = self._make_data(jr.PRNGKey(42))

        def loss_prim(Bxpb):
            return jnp.sum(_gru_fwd_via_primitive(A, Bxpb) ** 2)

        def loss_native(Bxpb):
            return jnp.sum(_gru_fwd_native(A, Bxpb) ** 2)

        g_prim = jax.grad(loss_prim)(Bxpb)
        g_native = jax.grad(loss_native)(Bxpb)

        max_diff = jnp.max(jnp.abs(g_prim - g_native))
        assert max_diff < self._tol, f"Max diff: {max_diff}"

    def test_grad_both_matches_native(self):
        """VJP (grad) w.r.t. both A and Bxpb matches native JAX AD."""
        A, Bxpb = self._make_data(jr.PRNGKey(42))

        def loss_prim(A, Bxpb):
            return jnp.sum(_gru_fwd_via_primitive(A, Bxpb) ** 2)

        def loss_native(A, Bxpb):
            return jnp.sum(_gru_fwd_native(A, Bxpb) ** 2)

        g_prim = jax.grad(loss_prim, argnums=(0, 1))(A, Bxpb)
        g_native = jax.grad(loss_native, argnums=(0, 1))(A, Bxpb)

        for i, name in enumerate(['A', 'Bxpb']):
            max_diff = jnp.max(jnp.abs(g_prim[i] - g_native[i]))
            assert max_diff < self._tol, f"{name} grad max diff: {max_diff}"

    def test_jit_grad(self):
        """jax.jit(jax.grad(...)) works through the primitive."""
        A, Bxpb = self._make_data(jr.PRNGKey(0))

        def loss(A, Bxpb):
            return jnp.sum(_gru_fwd_via_primitive(A, Bxpb) ** 2)

        g = jax.jit(jax.grad(loss, argnums=(0, 1)))(A, Bxpb)
        assert g[0].shape == A.shape
        assert g[1].shape == Bxpb.shape
        assert jnp.all(jnp.isfinite(g[0]))
        assert jnp.all(jnp.isfinite(g[1]))


# =============================================================================
# Fused GRU Backward AD Tests
# =============================================================================

class TestFusedGRUBackwardAD:
    """Tests for JVP/grad of fused_gru_diag_bwd_p.

    The backward primitive is LINEAR in grad_y but nonlinear in (h, A, Bxpb).
    Only grad_y has JVP/transpose rules.
    """

    def _make_data(self, key, B=2, T=8, N=4):
        k1, k2, k3 = jr.split(key, 3)
        A = jr.normal(k1, (3, N)) * 0.1
        Bxpb = jr.normal(k2, (B, T, 3, N)) * 0.1
        # Compute h from forward (native, no primitive)
        h = _gru_fwd_native(A, Bxpb)
        grad_y = jr.normal(k3, (B, T, N)) * 0.1
        return grad_y, h, A, Bxpb

    def test_grad_wrt_grad_y(self):
        """jax.grad w.r.t. grad_y produces finite results."""
        grad_y, h, A, Bxpb = self._make_data(jr.PRNGKey(0))

        def loss(grad_y):
            return jnp.sum(_gru_bwd_via_primitive(grad_y, h, A, Bxpb) ** 2)

        g = jax.grad(loss)(grad_y)
        assert g.shape == grad_y.shape
        assert jnp.all(jnp.isfinite(g))

    def test_jvp_grad_y_matches_native(self):
        """JVP w.r.t. grad_y via primitive matches native JAX AD."""
        grad_y, h, A, Bxpb = self._make_data(jr.PRNGKey(42))
        grad_y_dot = jr.normal(jr.PRNGKey(99), grad_y.shape) * 0.1

        def f_prim(grad_y):
            return _gru_bwd_via_primitive(grad_y, h, A, Bxpb)

        _, dl_dot_prim = jax.jvp(f_prim, (grad_y,), (grad_y_dot,))

        def f_native(grad_y):
            return _gru_bwd_native(grad_y, h, A, Bxpb)

        _, dl_dot_native = jax.jvp(f_native, (grad_y,), (grad_y_dot,))

        assert jnp.allclose(dl_dot_prim, dl_dot_native, atol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(dl_dot_prim - dl_dot_native))}"

    def test_grad_matches_native(self):
        """VJP (grad) w.r.t. grad_y matches native JAX AD."""
        grad_y, h, A, Bxpb = self._make_data(jr.PRNGKey(42))

        def loss_prim(grad_y):
            return jnp.sum(_gru_bwd_via_primitive(grad_y, h, A, Bxpb) ** 2)

        def loss_native(grad_y):
            return jnp.sum(_gru_bwd_native(grad_y, h, A, Bxpb) ** 2)

        g_prim = jax.grad(loss_prim)(grad_y)
        g_native = jax.grad(loss_native)(grad_y)

        assert jnp.allclose(g_prim, g_native, atol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(g_prim - g_native))}"

    def test_jit_grad(self):
        """jax.jit(jax.grad(...)) works through the primitive."""
        grad_y, h, A, Bxpb = self._make_data(jr.PRNGKey(0))

        def loss(grad_y):
            return jnp.sum(_gru_bwd_via_primitive(grad_y, h, A, Bxpb) ** 2)

        g = jax.jit(jax.grad(loss))(grad_y)
        assert g.shape == grad_y.shape
        assert jnp.all(jnp.isfinite(g))

    def test_linearity_in_grad_y(self):
        """Backward is linear in grad_y: f(a*gy) = a*f(gy)."""
        grad_y, h, A, Bxpb = self._make_data(jr.PRNGKey(0))
        alpha = 2.5

        dl1 = _gru_bwd_via_primitive(grad_y, h, A, Bxpb)
        dl2 = _gru_bwd_via_primitive(alpha * grad_y, h, A, Bxpb)

        assert jnp.allclose(alpha * dl1, dl2, atol=1e-5), \
            f"Max diff: {jnp.max(jnp.abs(alpha * dl1 - dl2))}"


# =============================================================================
# Fused LSTM-CIFG Forward AD Tests
# =============================================================================

class TestFusedLSTMForwardAD:
    """Tests for JVP/grad of fused_lstm_cifg_diag_fwd_p.

    Same implicit-diff vs unrolled-diff comparison as GRU forward.
    Small inputs ensure Newton convergence.
    """

    _scale = 0.01
    _tol = 0.015

    def _make_data(self, key, B=2, T=8, N=4):
        k1, k2, k3 = jr.split(key, 3)
        A = jr.normal(k1, (3, N)) * self._scale
        Bxpb = jr.normal(k2, (B, T, 3, N)) * self._scale
        C = jr.normal(k3, (2, N)) * self._scale
        return A, Bxpb, C

    def test_grad_wrt_A(self):
        """jax.grad through fused LSTM forward w.r.t. A produces finite results."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(0))

        def loss(A):
            return jnp.sum(_lstm_fwd_via_primitive(A, Bxpb, C) ** 2)

        g = jax.grad(loss)(A)
        assert g.shape == A.shape
        assert jnp.all(jnp.isfinite(g))

    def test_grad_wrt_Bxpb(self):
        """jax.grad through fused LSTM forward w.r.t. Bxpb produces finite results."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(0))

        def loss(Bxpb):
            return jnp.sum(_lstm_fwd_via_primitive(A, Bxpb, C) ** 2)

        g = jax.grad(loss)(Bxpb)
        assert g.shape == Bxpb.shape
        assert jnp.all(jnp.isfinite(g))

    def test_grad_wrt_C(self):
        """jax.grad through fused LSTM forward w.r.t. C produces finite results."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(0))

        def loss(C):
            return jnp.sum(_lstm_fwd_via_primitive(A, Bxpb, C) ** 2)

        g = jax.grad(loss)(C)
        assert g.shape == C.shape
        assert jnp.all(jnp.isfinite(g))

    def test_jvp_A_matches_native(self):
        """JVP w.r.t. A via primitive matches native JAX AD."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(42))
        A_dot = jr.normal(jr.PRNGKey(99), A.shape) * self._scale

        def f_prim(A):
            return _lstm_fwd_via_primitive(A, Bxpb, C)

        _, ch_dot_prim = jax.jvp(f_prim, (A,), (A_dot,))

        def f_native(A):
            return _lstm_fwd_native(A, Bxpb, C)

        _, ch_dot_native = jax.jvp(f_native, (A,), (A_dot,))

        max_diff = jnp.max(jnp.abs(ch_dot_prim - ch_dot_native))
        assert max_diff < self._tol, f"Max diff: {max_diff}"

    def test_jvp_Bxpb_matches_native(self):
        """JVP w.r.t. Bxpb via primitive matches native JAX AD."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(42))
        Bxpb_dot = jr.normal(jr.PRNGKey(99), Bxpb.shape) * self._scale

        def f_prim(Bxpb):
            return _lstm_fwd_via_primitive(A, Bxpb, C)

        _, ch_dot_prim = jax.jvp(f_prim, (Bxpb,), (Bxpb_dot,))

        def f_native(Bxpb):
            return _lstm_fwd_native(A, Bxpb, C)

        _, ch_dot_native = jax.jvp(f_native, (Bxpb,), (Bxpb_dot,))

        max_diff = jnp.max(jnp.abs(ch_dot_prim - ch_dot_native))
        assert max_diff < self._tol, f"Max diff: {max_diff}"

    def test_jvp_C_matches_native(self):
        """JVP w.r.t. C via primitive matches native JAX AD."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(42))
        C_dot = jr.normal(jr.PRNGKey(99), C.shape) * self._scale

        def f_prim(C):
            return _lstm_fwd_via_primitive(A, Bxpb, C)

        _, ch_dot_prim = jax.jvp(f_prim, (C,), (C_dot,))

        def f_native(C):
            return _lstm_fwd_native(A, Bxpb, C)

        _, ch_dot_native = jax.jvp(f_native, (C,), (C_dot,))

        max_diff = jnp.max(jnp.abs(ch_dot_prim - ch_dot_native))
        assert max_diff < self._tol, f"Max diff: {max_diff}"

    def test_grad_A_matches_native(self):
        """VJP (grad) w.r.t. A via primitive matches native JAX AD."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(42))

        def loss_prim(A):
            return jnp.sum(_lstm_fwd_via_primitive(A, Bxpb, C) ** 2)

        def loss_native(A):
            return jnp.sum(_lstm_fwd_native(A, Bxpb, C) ** 2)

        g_prim = jax.grad(loss_prim)(A)
        g_native = jax.grad(loss_native)(A)

        max_diff = jnp.max(jnp.abs(g_prim - g_native))
        assert max_diff < self._tol, f"Max diff: {max_diff}"

    def test_grad_Bxpb_matches_native(self):
        """VJP (grad) w.r.t. Bxpb via primitive matches native JAX AD."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(42))

        def loss_prim(Bxpb):
            return jnp.sum(_lstm_fwd_via_primitive(A, Bxpb, C) ** 2)

        def loss_native(Bxpb):
            return jnp.sum(_lstm_fwd_native(A, Bxpb, C) ** 2)

        g_prim = jax.grad(loss_prim)(Bxpb)
        g_native = jax.grad(loss_native)(Bxpb)

        max_diff = jnp.max(jnp.abs(g_prim - g_native))
        assert max_diff < self._tol, f"Max diff: {max_diff}"

    def test_grad_all_matches_native(self):
        """VJP (grad) w.r.t. all params matches native JAX AD."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(42))

        def loss_prim(A, Bxpb, C):
            return jnp.sum(_lstm_fwd_via_primitive(A, Bxpb, C) ** 2)

        def loss_native(A, Bxpb, C):
            return jnp.sum(_lstm_fwd_native(A, Bxpb, C) ** 2)

        g_prim = jax.grad(loss_prim, argnums=(0, 1, 2))(A, Bxpb, C)
        g_native = jax.grad(loss_native, argnums=(0, 1, 2))(A, Bxpb, C)

        for i, name in enumerate(['A', 'Bxpb', 'C']):
            max_diff = jnp.max(jnp.abs(g_prim[i] - g_native[i]))
            assert max_diff < self._tol, f"{name} grad max diff: {max_diff}"

    def test_jit_grad(self):
        """jax.jit(jax.grad(...)) works through the primitive."""
        A, Bxpb, C = self._make_data(jr.PRNGKey(0))

        def loss(A, Bxpb, C):
            return jnp.sum(_lstm_fwd_via_primitive(A, Bxpb, C) ** 2)

        g = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(A, Bxpb, C)
        assert g[0].shape == A.shape
        assert g[1].shape == Bxpb.shape
        assert g[2].shape == C.shape
        for i in range(3):
            assert jnp.all(jnp.isfinite(g[i]))


# =============================================================================
# Fused LSTM-CIFG Backward AD Tests
# =============================================================================

class TestFusedLSTMBackwardAD:
    """Tests for JVP/grad of fused_lstm_cifg_diag_bwd_p.

    The backward primitive is LINEAR in grad_y but nonlinear in
    (full_state, A, Bxpb, C). Only grad_y has JVP/transpose rules.
    """

    def _make_data(self, key, B=2, T=8, N=4):
        k1, k2, k3, k4 = jr.split(key, 4)
        A = jr.normal(k1, (3, N)) * 0.1
        Bxpb = jr.normal(k2, (B, T, 3, N)) * 0.1
        C = jr.normal(k3, (2, N)) * 0.1
        # Compute full_state from forward (native, no primitive)
        full_state = _lstm_fwd_native(A, Bxpb, C)
        grad_y = jr.normal(k4, (B, T, N)) * 0.1
        return grad_y, full_state, A, Bxpb, C

    def test_grad_wrt_grad_y(self):
        """jax.grad w.r.t. grad_y produces finite results."""
        grad_y, full_state, A, Bxpb, C = self._make_data(jr.PRNGKey(0))

        def loss(grad_y):
            dl_ch = _lstm_bwd_via_primitive(grad_y, full_state, A, Bxpb, C)
            return jnp.sum(dl_ch ** 2)

        g = jax.grad(loss)(grad_y)
        assert g.shape == grad_y.shape
        assert jnp.all(jnp.isfinite(g))

    def test_jvp_grad_y_matches_native(self):
        """JVP w.r.t. grad_y via primitive matches native JAX AD."""
        grad_y, full_state, A, Bxpb, C = self._make_data(jr.PRNGKey(42))
        grad_y_dot = jr.normal(jr.PRNGKey(99), grad_y.shape) * 0.1

        def f_prim(grad_y):
            return _lstm_bwd_via_primitive(grad_y, full_state, A, Bxpb, C)

        _, dl_dot_prim = jax.jvp(f_prim, (grad_y,), (grad_y_dot,))

        def f_native(grad_y):
            return _lstm_bwd_native(grad_y, full_state, A, Bxpb, C)

        _, dl_dot_native = jax.jvp(f_native, (grad_y,), (grad_y_dot,))

        assert jnp.allclose(dl_dot_prim, dl_dot_native, atol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(dl_dot_prim - dl_dot_native))}"

    def test_grad_matches_native(self):
        """VJP (grad) w.r.t. grad_y matches native JAX AD."""
        grad_y, full_state, A, Bxpb, C = self._make_data(jr.PRNGKey(42))

        def loss_prim(grad_y):
            return jnp.sum(
                _lstm_bwd_via_primitive(grad_y, full_state, A, Bxpb, C) ** 2
            )

        def loss_native(grad_y):
            return jnp.sum(
                _lstm_bwd_native(grad_y, full_state, A, Bxpb, C) ** 2
            )

        g_prim = jax.grad(loss_prim)(grad_y)
        g_native = jax.grad(loss_native)(grad_y)

        assert jnp.allclose(g_prim, g_native, atol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(g_prim - g_native))}"

    def test_jit_grad(self):
        """jax.jit(jax.grad(...)) works through the primitive."""
        grad_y, full_state, A, Bxpb, C = self._make_data(jr.PRNGKey(0))

        def loss(grad_y):
            return jnp.sum(
                _lstm_bwd_via_primitive(grad_y, full_state, A, Bxpb, C) ** 2
            )

        g = jax.jit(jax.grad(loss))(grad_y)
        assert g.shape == grad_y.shape
        assert jnp.all(jnp.isfinite(g))

    def test_linearity_in_grad_y(self):
        """Backward is linear in grad_y: f(a*gy) = a*f(gy)."""
        grad_y, full_state, A, Bxpb, C = self._make_data(jr.PRNGKey(0))
        alpha = 2.5

        dl1 = _lstm_bwd_via_primitive(grad_y, full_state, A, Bxpb, C)
        dl2 = _lstm_bwd_via_primitive(alpha * grad_y, full_state, A, Bxpb, C)

        assert jnp.allclose(alpha * dl1, dl2, atol=1e-5), \
            f"Max diff: {jnp.max(jnp.abs(alpha * dl1 - dl2))}"
