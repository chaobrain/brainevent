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

"""
Fused kernels for GRU and LSTM-CIFG parallel training.

Each fused kernel combines the Newton iteration, cell evaluation, Jacobian
computation, and parallel reduction into a single operation.

Two backends are provided:
- ``jax_raw``: Pure-JAX implementation with hardcoded sigmoid/tanh
  (matching the CUDA kernels' fixed nonlinearities). Works on all platforms.
- ``tvmffi``: CUDA kernels via TVM FFI (GPU only). Eliminates intermediate
  global memory round-trips for maximum performance.

Functions:
    fused_gru_diag_forward: Fused GRU forward pass.
    fused_gru_diag_backward: Fused GRU backward pass.
    fused_lstm_cifg_diag_forward: Fused LSTM-CIFG forward pass.
    fused_lstm_cifg_diag_backward: Fused LSTM-CIFG backward pass.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._op import XLACustomKernel
from brainevent.kernix import load_cuda_file
from ._parallel_reduce import _adjoint_reduce_diag

__all__ = [
    'fused_gru_diag_forward',
    'fused_gru_diag_backward',
    'fused_lstm_cifg_diag_forward',
    'fused_lstm_cifg_diag_backward',
    'fused_gru_diag_fwd_p',
    'fused_gru_diag_bwd_p',
    'fused_lstm_cifg_diag_fwd_p',
    'fused_lstm_cifg_diag_bwd_p',
]

_dtype_sfx = {
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
}


# =============================================================================
# Helper: roll state
# =============================================================================

def _roll_state(h, axis=-2):
    """Shift hidden states: h_prev[t] = h[t-1], h_prev[0] = 0."""
    h_prev = jnp.roll(h, shift=1, axis=axis)
    h_prev = h_prev.at[..., 0, :].set(0.0)
    return h_prev


# =============================================================================
# Helpers: GRU cell equations with hardcoded sigmoid/tanh
# =============================================================================

def _gru_step_batched(h_prev, A, Bxpb):
    """Batched GRU cell with hardcoded sigmoid/tanh.

    Args:
        h_prev: Previous hidden state, shape ``(B, T, N)``.
        A: Diagonal recurrence weights, shape ``(3, N)``.
        Bxpb: Precomputed B@x+b, shape ``(B, T, 3, N)``.

    Returns:
        h_new: New hidden state, shape ``(B, T, N)``.
    """
    pre_z = A[0, :] * h_prev + Bxpb[..., 0, :]
    pre_r = A[1, :] * h_prev + Bxpb[..., 1, :]
    z = jax.nn.sigmoid(pre_z)
    r = jax.nn.sigmoid(pre_r)
    pre_h = A[2, :] * h_prev * r + Bxpb[..., 2, :]
    h_new = jnp.tanh(pre_h)
    return z * h_new + (1.0 - z) * h_prev


def _gru_jacobian_batched(h, A, Bxpb):
    """Compute diagonal GRU Jacobians (negated) for Newton solve.

    Args:
        h: Current hidden state guess, shape ``(B, T, N)``.
        A: Diagonal recurrence weights, shape ``(3, N)``.
        Bxpb: Precomputed B@x+b, shape ``(B, T, 3, N)``.

    Returns:
        jac: Negated diagonal Jacobians, shape ``(B, T, N)``.
    """
    h_prev = _roll_state(h)

    pre_z = A[0, :] * h_prev + Bxpb[..., 0, :]
    pre_r = A[1, :] * h_prev + Bxpb[..., 1, :]
    z = jax.nn.sigmoid(pre_z)
    r = jax.nn.sigmoid(pre_r)
    pre_h = A[2, :] * h_prev * r + Bxpb[..., 2, :]
    h_candidate = jnp.tanh(pre_h)

    # sigmoid derivative: s(x)(1-s(x))
    grad_z = z * (1.0 - z)
    grad_r = r * (1.0 - r)
    # tanh derivative: 1 - tanh(x)^2
    grad_h = 1.0 - h_candidate ** 2

    J_z = A[0, :] * grad_z
    J_r = A[1, :] * grad_r
    J_h = A[2, :] * grad_h

    J_h = J_h * (r + h_prev * J_r)
    jac = (1.0 - z) + (h_candidate - h_prev) * J_z + z * J_h
    return -jac


def _gru_jacobian_bwd_batched(h, A, Bxpb):
    """Compute backward Jacobians for GRU (flipped, shifted)."""
    jac = _gru_jacobian_batched(h, A, Bxpb)
    jac = jnp.flip(jac, axis=-2)
    jac = jnp.roll(jac, shift=1, axis=-2)
    jac = jac.at[..., 0, :].set(0.0)
    return jac


# =============================================================================
# Helpers: LSTM-CIFG cell equations with hardcoded sigmoid/tanh
# =============================================================================

def _lstm_step_batched(ch_prev, A, Bxpb, C):
    """Batched LSTM-CIFG cell with hardcoded sigmoid/tanh.

    Args:
        ch_prev: Previous [c, h] state, shape ``(B, T, 2, N)``.
        A: Diagonal recurrence weights, shape ``(3, N)``.
        Bxpb: Precomputed B@x+b, shape ``(B, T, 3, N)``.
        C: Peephole connections, shape ``(2, N)``.

    Returns:
        ch_new: New [c, h] state, shape ``(B, T, 2, N)``.
    """
    c_prev = ch_prev[..., 0, :]
    h_prev = ch_prev[..., 1, :]

    pre_f = A[0, :] * h_prev + Bxpb[..., 0, :]
    pre_o = A[1, :] * h_prev + Bxpb[..., 1, :]
    pre_c = A[2, :] * h_prev + Bxpb[..., 2, :]

    f = jax.nn.sigmoid(pre_f + C[0, :] * c_prev)
    c_new_cand = jnp.tanh(pre_c)
    c_new = f * c_prev + (1.0 - f) * c_new_cand
    o = jax.nn.sigmoid(pre_o + C[1, :] * c_new)
    h_new = o * jnp.tanh(c_new)

    return jnp.stack([c_new, h_new], axis=-2)


def _lstm_jacobian_batched(ch, A, Bxpb, C):
    """Compute 2x2 block-diagonal LSTM-CIFG Jacobians (negated).

    Args:
        ch: Current [c, h] state guess, shape ``(B, T, 2, N)``.
        A: Diagonal recurrence weights, shape ``(3, N)``.
        Bxpb: Precomputed B@x+b, shape ``(B, T, 3, N)``.
        C: Peephole connections, shape ``(2, N)``.

    Returns:
        jac: Negated block-diagonal Jacobians, shape ``(B, T, N, 2, 2)``.
    """
    cc = ch[..., 0, :]
    ch_prev = _roll_state(ch, axis=-3)
    c_prev = ch_prev[..., 0, :]
    h_prev = ch_prev[..., 1, :]

    pre_f = A[0, :] * h_prev + Bxpb[..., 0, :] + C[0, :] * c_prev
    pre_o = A[1, :] * h_prev + Bxpb[..., 1, :] + C[1, :] * cc
    pre_c = A[2, :] * h_prev + Bxpb[..., 2, :]

    f = jax.nn.sigmoid(pre_f)
    o = jax.nn.sigmoid(pre_o)
    c = jnp.tanh(pre_c)

    grad_f = f * (1.0 - f)
    grad_o = o * (1.0 - o)
    grad_c = 1.0 - c ** 2

    Jh_f = A[0, :] * grad_f
    Jh_o = A[1, :] * grad_o
    Jh_c = A[2, :] * grad_c

    Jc_f = C[0, :] * grad_f
    Jc_o = C[1, :] * grad_o

    scc = jnp.tanh(cc)
    sdercc = 1.0 - scc ** 2
    o_sdercc = o * sdercc

    Jcc = -(Jc_f * (c_prev - c) + f)
    Jch = -(Jh_f * (c_prev - c) + (1.0 - f) * Jh_c)
    Jhc = Jcc * (Jc_o * scc + o_sdercc)
    Jhh = -(scc * (Jh_o - Jc_o * Jch) - o_sdercc * Jch)

    jac = jnp.stack([
        jnp.stack([Jcc, Jch], axis=-1),
        jnp.stack([Jhc, Jhh], axis=-1),
    ], axis=-2)

    return jac


def _lstm_jacobian_bwd_batched(ch, A, Bxpb, C):
    """Compute backward Jacobians for LSTM-CIFG (transposed, flipped)."""
    cc = ch[..., 0, :]
    ch_prev = _roll_state(ch, axis=-3)
    c_prev = ch_prev[..., 0, :]
    h_prev = ch_prev[..., 1, :]

    pre_f = A[0, :] * h_prev + Bxpb[..., 0, :] + C[0, :] * c_prev
    pre_o = A[1, :] * h_prev + Bxpb[..., 1, :] + C[1, :] * cc
    pre_c = A[2, :] * h_prev + Bxpb[..., 2, :]

    f = jax.nn.sigmoid(pre_f)
    o = jax.nn.sigmoid(pre_o)
    c = jnp.tanh(pre_c)

    grad_f = f * (1.0 - f)
    grad_o = o * (1.0 - o)
    grad_c = 1.0 - c ** 2

    Jh_f = A[0, :] * grad_f
    Jh_o = A[1, :] * grad_o
    Jh_c = A[2, :] * grad_c

    Jc_f = C[0, :] * grad_f
    Jc_o = C[1, :] * grad_o

    scc = jnp.tanh(cc)
    sdercc = 1.0 - scc ** 2
    o_sdercc = o * sdercc

    Jcc = -(Jc_f * (c_prev - c) + f)
    Jch = -(Jh_f * (c_prev - c) + (1.0 - f) * Jh_c)
    Jhc = Jcc * (Jc_o * scc + o_sdercc)
    Jhh = -(scc * (Jh_o - Jc_o * Jch) - o_sdercc * Jch)

    # Transposed: swap off-diag
    jacobians = jnp.stack(
        [
            jnp.stack([Jcc, Jhc], axis=-1),
            jnp.stack([Jch, Jhh], axis=-1),
        ],
        axis=-2
    )

    # Flip time, roll, zero first
    jacobians = jnp.flip(jacobians, axis=-4)
    jacobians = jnp.roll(jacobians, shift=1, axis=-4)
    jacobians = jacobians.at[..., 0, :, :, :].set(0.0)

    return jacobians


# =============================================================================
# Parallel reduce helpers (inline for fused jax_raw kernels)
# =============================================================================

def _parallel_reduce_diag_inline(jac, rhs):
    """Diagonal parallel reduction via associative_scan."""

    def combine(a, b):
        j_a, r_a = a
        j_b, r_b = b
        return (j_b * j_a, j_b * r_a + r_b)

    _, h = jax.lax.associative_scan(combine, (jac, rhs), axis=-2)
    return h


def _parallel_reduce_block_diag_inline(jac, rhs):
    """Block-diagonal parallel reduction via associative_scan."""

    def combine(a, b):
        j_a, r_a = a
        j_b, r_b = b
        j_new = jnp.einsum('...ij,...jk->...ik', j_b, j_a)
        r_new = jnp.einsum('...ij,...j->...i', j_b, r_a) + r_b
        return (j_new, r_new)

    _, h = jax.lax.associative_scan(combine, (jac, rhs), axis=1)
    return h


# =============================================================================
# Fused GRU Diagonal — Forward
# =============================================================================

def _fused_gru_fwd_jax_kernel(**kwargs):
    """jax_raw kernel: fused GRU forward (Newton + parallel reduce)."""
    max_its = kwargs.get('max_its', 3)
    omega = kwargs.get('omega', 1.0)

    def kernel(A, Bxpb):
        batch_size, seq_len = Bxpb.shape[0], Bxpb.shape[1]
        hidden_dim = A.shape[1]

        # Initial guess: one recurrence step from zeros
        h0 = jnp.zeros((batch_size, seq_len, hidden_dim), dtype=A.dtype)
        h = _gru_step_batched(_roll_state(h0), A, Bxpb)

        # Newton iterations
        def body(_, h):
            h_prev = _roll_state(h)
            h_new = _gru_step_batched(h_prev, A, Bxpb)
            rhs = -(h - h_new)
            jac = _gru_jacobian_batched(h, A, Bxpb)
            dh = _parallel_reduce_diag_inline(jac, rhs)
            return h + omega * dh

        h = jax.lax.fori_loop(0, max_its, body, h)
        return (h,)

    return kernel


def _fused_gru_fwd_cuda_kernel(**kwargs):
    """tvmffi kernel: fused GRU forward via CUDA."""
    load_cuda_file(
        Path(__file__).parent / '_fused_gru_diag.cu',
        name='fused_gru_diag',
    )
    out_info = kwargs['outs']
    sfx = _dtype_sfx[np.dtype(kwargs['A_info'].dtype)]
    kernel_name = f'fused_gru_diag.fused_fwd_gru_diag{sfx}'

    def kernel(A, Bxpb):
        return jax.ffi.ffi_call(kernel_name, out_info)(A, Bxpb)

    return kernel


# -- JVP / transpose for fused GRU forward -----------------------------------
#
# At convergence h satisfies h[t] = gru_step(h[t-1], A, Bxpb[t]).
# By the implicit function theorem:
#   h_dot[t] = J[t]*h_dot[t-1] + (∂step/∂A)·A_dot + (∂step/∂Bxpb)·Bxpb_dot
# where J[t] = ∂gru_step/∂h_prev = -jac_neg[t].
# This is a forward linear recurrence solved via parallel_reduce_diag.

def _fused_gru_fwd_jvp_A(A_dot, A, Bxpb, **kwargs):
    """JVP of fused GRU forward w.r.t. A (implicit differentiation)."""
    h = _fused_gru_fwd_compute(A, Bxpb, kwargs)
    return _fused_gru_fwd_tangent(h, A, Bxpb, A_dot=A_dot, Bxpb_dot=None)


def _fused_gru_fwd_jvp_Bxpb(Bxpb_dot, A, Bxpb, **kwargs):
    """JVP of fused GRU forward w.r.t. Bxpb (implicit differentiation)."""
    h = _fused_gru_fwd_compute(A, Bxpb, kwargs)
    return _fused_gru_fwd_tangent(h, A, Bxpb, A_dot=None, Bxpb_dot=Bxpb_dot)


def _fused_gru_fwd_compute(A, Bxpb, kwargs):
    """Recompute forward hidden states from primals."""
    batch_size, seq_len = Bxpb.shape[0], Bxpb.shape[1]
    hidden_dim = A.shape[1]
    out = fused_gru_diag_fwd_p.primitive.bind(A, Bxpb, **kwargs)
    return out[0]


def _fused_gru_fwd_tangent(h, A, Bxpb, A_dot, Bxpb_dot):
    """Compute tangent of fused GRU forward via implicit differentiation."""
    h_prev = _roll_state(h)

    # Cell Jacobian (un-negated): ∂gru_step/∂h_prev
    jac_neg = _gru_jacobian_batched(h, A, Bxpb)
    J = -jac_neg  # un-negate

    # ∂step/∂A · A_dot + ∂step/∂Bxpb · Bxpb_dot
    primals = []
    tangents = []
    if A_dot is not None:
        primals.append(A)
        tangents.append(A_dot)
    else:
        primals.append(A)
        tangents.append(jnp.zeros_like(A))
    if Bxpb_dot is not None:
        primals.append(Bxpb)
        tangents.append(Bxpb_dot)
    else:
        primals.append(Bxpb)
        tangents.append(jnp.zeros_like(Bxpb))

    def step_params(A_p, Bxpb_p):
        return _gru_step_batched(h_prev, A_p, Bxpb_p)

    _, s = jax.jvp(step_params, tuple(primals), tuple(tangents))

    # Solve linear recurrence: h_dot[t] = J[t]*h_dot[t-1] + s[t]
    h_dot = _parallel_reduce_diag_inline(J, s)
    return (h_dot,)


def _fused_gru_fwd_transpose(ct, A, Bxpb, **kwargs):
    """Transpose of fused GRU forward (VJP via adjoint solve)."""
    ct_h = ct[0]

    if type(ct_h) is ad.Zero:
        ct_A = ad.Zero(A) if ad.is_undefined_primal(A) else A
        ct_Bxpb = ad.Zero(Bxpb) if ad.is_undefined_primal(Bxpb) else Bxpb
        return ct_A, ct_Bxpb

    # Need concrete A and Bxpb for Jacobian and vjp computation.
    # In JAX's AD pipeline, at least one of A/Bxpb is undefined (tangent).
    # The JVP created a linear map from (A_dot, Bxpb_dot) to h_dot;
    # transpose of that map requires concrete A, Bxpb (known from forward).
    if ad.is_undefined_primal(A) and ad.is_undefined_primal(Bxpb):
        raise ValueError(
            "Cannot transpose fused_gru_diag_fwd w.r.t. both A and Bxpb "
            "simultaneously."
        )

    # Recompute h from whichever primals are concrete
    if not ad.is_undefined_primal(A) and not ad.is_undefined_primal(Bxpb):
        h = _fused_gru_fwd_compute(A, Bxpb, kwargs)
    elif ad.is_undefined_primal(A):
        # A is tangent, Bxpb is concrete — but we need A to compute h.
        # This shouldn't happen in standard AD (JVP creates linear map w.r.t. A,
        # needing concrete A). Raise error.
        raise ValueError(
            "Cannot transpose fused_gru_diag_fwd: A is undefined but needed "
            "for Jacobian computation."
        )
    else:
        raise ValueError(
            "Cannot transpose fused_gru_diag_fwd: Bxpb is undefined but needed "
            "for Jacobian computation."
        )

    h_prev = _roll_state(h)
    jac_neg = _gru_jacobian_batched(h, A, Bxpb)
    J = -jac_neg

    # Adjoint solve: λ[t] = J[t+1]*λ[t+1] + ct_h[t]
    lam = _adjoint_reduce_diag(J, ct_h)

    # Parameter gradients via vjp of cell step
    def step_params(A_p, Bxpb_p):
        return _gru_step_batched(h_prev, A_p, Bxpb_p)

    _, vjp_fn = jax.vjp(step_params, A, Bxpb)
    ct_A_val, ct_Bxpb_val = vjp_fn(lam)

    ct_A = ct_A_val if ad.is_undefined_primal(A) else A
    ct_Bxpb = ct_Bxpb_val if ad.is_undefined_primal(Bxpb) else Bxpb
    return ct_A, ct_Bxpb


fused_gru_diag_fwd_p = XLACustomKernel('fused_gru_diag_fwd')
fused_gru_diag_fwd_p.def_kernel('jax_raw', 'cpu', _fused_gru_fwd_jax_kernel)
fused_gru_diag_fwd_p.def_kernel('jax_raw', 'gpu', _fused_gru_fwd_jax_kernel)
fused_gru_diag_fwd_p.def_kernel('jax_raw', 'tpu', _fused_gru_fwd_jax_kernel)
fused_gru_diag_fwd_p.def_tvmffi_kernel('gpu', _fused_gru_fwd_cuda_kernel, asdefault=True)
fused_gru_diag_fwd_p.def_tags('pararnn', 'fused')
fused_gru_diag_fwd_p.def_jvp_rule2(_fused_gru_fwd_jvp_A, _fused_gru_fwd_jvp_Bxpb)
fused_gru_diag_fwd_p.def_transpose_rule(_fused_gru_fwd_transpose)


def fused_gru_diag_forward(
    A: jax.Array,
    Bxpb: jax.Array,
    *,
    backend: str = None,
) -> jax.Array:
    """Fused GRU forward pass.

    Runs the complete Newton-parallel-reduction solve. Input projection
    (Bxpb = B @ x + b) must be precomputed.

    Args:
        A: Diagonal recurrence weights, shape ``(3, hidden_dim)``.
        Bxpb: Precomputed input projection, shape ``(B, T, 3, hidden_dim)``.
        backend: ``'tvmffi'`` for CUDA, ``None`` for default (``jax_raw``).

    Returns:
        Hidden states ``h``, shape ``(B, T, hidden_dim)``.
    """
    batch_size, seq_len = Bxpb.shape[0], Bxpb.shape[1]
    hidden_dim = A.shape[1]
    return fused_gru_diag_fwd_p(
        A, Bxpb,
        A_info=jax.ShapeDtypeStruct(A.shape, A.dtype),
        outs=[jax.ShapeDtypeStruct((batch_size, seq_len, hidden_dim), A.dtype)],
        backend=backend,
    )[0]


# =============================================================================
# Fused GRU Diagonal — Backward
# =============================================================================

def _fused_gru_bwd_jax_kernel(**kwargs):
    """jax_raw kernel: fused GRU backward (transposed solve)."""

    def kernel(grad_y, h, A, Bxpb):
        rhs = jnp.flip(grad_y, axis=-2)
        jac_bwd = _gru_jacobian_bwd_batched(h, A, Bxpb)
        dl_dh = jnp.flip(_parallel_reduce_diag_inline(jac_bwd, rhs), axis=-2)
        return (dl_dh,)

    return kernel


def _fused_gru_bwd_cuda_kernel(**kwargs):
    """tvmffi kernel: fused GRU backward via CUDA."""
    load_cuda_file(
        Path(__file__).parent / '_fused_gru_diag.cu',
        name='fused_gru_diag',
    )
    out_info = kwargs['outs']
    sfx = _dtype_sfx[np.dtype(kwargs['A_info'].dtype)]
    kernel_name = f'fused_gru_diag.fused_bwd_gru_diag{sfx}'

    def kernel(grad_y, h, A, Bxpb):
        return jax.ffi.ffi_call(kernel_name, out_info)(grad_y, h, A, Bxpb)

    return kernel


# -- JVP / transpose for fused GRU backward ----------------------------------
#
# The backward primitive computes:
#   dl_dh = flip(S(jac_bwd, flip(grad_y)))
# where jac_bwd depends nonlinearly on (h, A, Bxpb) and S is a forward solve.
# The output is LINEAR in grad_y but NONLINEAR in (h, A, Bxpb).

def _fused_gru_bwd_jvp_grad_y(grad_y_dot, grad_y, h, A, Bxpb, **kwargs):
    """JVP w.r.t. grad_y (linear): run backward solve on perturbed gradient."""
    jac_bwd = _gru_jacobian_bwd_batched(h, A, Bxpb)
    rhs_dot = jnp.flip(grad_y_dot, axis=-2)
    dl_dh_dot = jnp.flip(_parallel_reduce_diag_inline(jac_bwd, rhs_dot), axis=-2)
    return (dl_dh_dot,)


def _fused_gru_bwd_transpose(ct, grad_y, h, A, Bxpb, **kwargs):
    """Transpose of fused GRU backward.

    Only supports transposing w.r.t. grad_y (the linear input).
    """
    ct_dl = ct[0]

    for name, val in [('h', h), ('A', A), ('Bxpb', Bxpb)]:
        if ad.is_undefined_primal(val):
            raise ValueError(
                f"Cannot transpose fused_gru_diag_bwd w.r.t. {name} "
                f"(nonlinear dependency through Jacobian computation)."
            )

    if ad.is_undefined_primal(grad_y):
        if type(ct_dl) is ad.Zero:
            return ad.Zero(grad_y), h, A, Bxpb
        # Transpose of: dl_dh = flip(S(jac_bwd, flip(grad_y)))
        # = flip(S(jac_bwd, flip(·)))
        # Adjoint: ct_grad_y = flip(S^T(jac_bwd, flip(ct_dl)))
        jac_bwd = _gru_jacobian_bwd_batched(h, A, Bxpb)
        from ._parallel_reduce import _adjoint_reduce_diag
        ct_grad_y = jnp.flip(_adjoint_reduce_diag(jac_bwd, jnp.flip(ct_dl, axis=-2)), axis=-2)
        return ct_grad_y, h, A, Bxpb

    return grad_y, h, A, Bxpb


fused_gru_diag_bwd_p = XLACustomKernel('fused_gru_diag_bwd')
fused_gru_diag_bwd_p.def_kernel('jax_raw', 'cpu', _fused_gru_bwd_jax_kernel)
fused_gru_diag_bwd_p.def_kernel('jax_raw', 'gpu', _fused_gru_bwd_jax_kernel)
fused_gru_diag_bwd_p.def_kernel('jax_raw', 'tpu', _fused_gru_bwd_jax_kernel)
fused_gru_diag_bwd_p.def_tvmffi_kernel('gpu', _fused_gru_bwd_cuda_kernel, asdefault=True)
fused_gru_diag_bwd_p.def_tags('pararnn', 'fused')
fused_gru_diag_bwd_p.def_jvp_rule2(_fused_gru_bwd_jvp_grad_y, None, None, None)
fused_gru_diag_bwd_p.def_transpose_rule(_fused_gru_bwd_transpose)


def fused_gru_diag_backward(
    grad_y: jax.Array,
    h: jax.Array,
    A: jax.Array,
    Bxpb: jax.Array,
    *,
    backend: str = None,
) -> jax.Array:
    """Fused GRU backward pass (transposed system solve).

    Solves the transposed bidiagonal system for dl/dh.

    Args:
        grad_y: Gradient w.r.t. output, shape ``(B, T, hidden_dim)``.
        h: Forward hidden states, shape ``(B, T, hidden_dim)``.
        A: Diagonal recurrence weights, shape ``(3, hidden_dim)``.
        Bxpb: Precomputed input projection, shape ``(B, T, 3, hidden_dim)``.
        backend: ``'tvmffi'`` for CUDA, ``None`` for default.

    Returns:
        dl/dh with shape ``(B, T, hidden_dim)``.
    """
    return fused_gru_diag_bwd_p(
        grad_y, h, A, Bxpb,
        A_info=jax.ShapeDtypeStruct(A.shape, A.dtype),
        outs=[jax.ShapeDtypeStruct(grad_y.shape, grad_y.dtype)],
        backend=backend,
    )[0]


# =============================================================================
# Fused LSTM-CIFG Diagonal — Forward
# =============================================================================

def _fused_lstm_fwd_jax_kernel(**kwargs):
    """jax_raw kernel: fused LSTM-CIFG forward (Newton + block-diag reduce)."""
    max_its = kwargs.get('max_its', 3)
    omega = kwargs.get('omega', 1.0)

    def kernel(A, Bxpb, C):
        batch_size, seq_len = Bxpb.shape[0], Bxpb.shape[1]
        state_dim = A.shape[1]

        # Initial guess: one recurrence step from zeros
        # ch shape: (B, T, 2, N) where 2 = [c, h]
        ch0 = jnp.zeros((batch_size, seq_len, 2, state_dim), dtype=A.dtype)
        ch = _lstm_step_batched(_roll_state(ch0, axis=-3), A, Bxpb, C)

        # Newton iterations
        def body(_, ch):
            ch_prev = _roll_state(ch, axis=-3)
            ch_new = _lstm_step_batched(ch_prev, A, Bxpb, C)
            # Residual: (B, T, 2, N)
            rhs = -(ch - ch_new)
            # Jacobian: (B, T, N, 2, 2)
            jac = _lstm_jacobian_batched(ch, A, Bxpb, C)
            # Reshape rhs from (B, T, 2, N) to (B, T, N, 2) for block reduce
            rhs_blocked = jnp.moveaxis(rhs, -2, -1)
            dch_blocked = _parallel_reduce_block_diag_inline(jac, rhs_blocked)
            # Reshape back: (B, T, N, 2) -> (B, T, 2, N)
            dch = jnp.moveaxis(dch_blocked, -1, -2)
            return ch + omega * dch

        ch = jax.lax.fori_loop(0, max_its, body, ch)
        return (ch,)

    return kernel


def _fused_lstm_fwd_cuda_kernel(**kwargs):
    """tvmffi kernel: fused LSTM-CIFG forward via CUDA."""
    load_cuda_file(
        Path(__file__).parent / '_fused_lstm_cifg_diag.cu',
        name='fused_lstm_cifg_diag',
    )
    out_info = kwargs['outs']
    sfx = _dtype_sfx[np.dtype(kwargs['A_info'].dtype)]
    kernel_name = f'fused_lstm_cifg_diag.fused_fwd_lstm_cifg_diag{sfx}'

    def kernel(A, Bxpb, C):
        return jax.ffi.ffi_call(kernel_name, out_info)(A, Bxpb, C)

    return kernel


# -- JVP / transpose for fused LSTM-CIFG forward -----------------------------
#
# At convergence ch satisfies ch[t] = lstm_step(ch[t-1], A, Bxpb[t], C).
# Implicit differentiation gives a 2x2 block-diagonal linear recurrence
# for the tangent.

def _fused_lstm_fwd_jvp_A(A_dot, A, Bxpb, C, **kwargs):
    """JVP of fused LSTM-CIFG forward w.r.t. A."""
    ch = _fused_lstm_fwd_compute(A, Bxpb, C, kwargs)
    return _fused_lstm_fwd_tangent(ch, A, Bxpb, C, A_dot=A_dot, Bxpb_dot=None, C_dot=None)


def _fused_lstm_fwd_jvp_Bxpb(Bxpb_dot, A, Bxpb, C, **kwargs):
    """JVP of fused LSTM-CIFG forward w.r.t. Bxpb."""
    ch = _fused_lstm_fwd_compute(A, Bxpb, C, kwargs)
    return _fused_lstm_fwd_tangent(ch, A, Bxpb, C, A_dot=None, Bxpb_dot=Bxpb_dot, C_dot=None)


def _fused_lstm_fwd_jvp_C(C_dot, A, Bxpb, C, **kwargs):
    """JVP of fused LSTM-CIFG forward w.r.t. C."""
    ch = _fused_lstm_fwd_compute(A, Bxpb, C, kwargs)
    return _fused_lstm_fwd_tangent(ch, A, Bxpb, C, A_dot=None, Bxpb_dot=None, C_dot=C_dot)


def _fused_lstm_fwd_compute(A, Bxpb, C, kwargs):
    """Recompute forward full state from primals."""
    out = fused_lstm_cifg_diag_fwd_p.primitive.bind(A, Bxpb, C, **kwargs)
    return out[0]


def _fused_lstm_fwd_tangent(ch, A, Bxpb, C, A_dot, Bxpb_dot, C_dot):
    """Compute tangent of fused LSTM forward via implicit differentiation."""
    ch_prev = _roll_state(ch, axis=-3)

    # Cell Jacobian (negated): shape (B, T, N, 2, 2)
    jac_neg = _lstm_jacobian_batched(ch, A, Bxpb, C)
    J = -jac_neg  # un-negate: ∂lstm_step/∂ch_prev

    # ∂step/∂params · params_dot
    primals = (A, Bxpb, C)
    tangents = (
        A_dot if A_dot is not None else jnp.zeros_like(A),
        Bxpb_dot if Bxpb_dot is not None else jnp.zeros_like(Bxpb),
        C_dot if C_dot is not None else jnp.zeros_like(C),
    )

    def step_params(A_p, Bxpb_p, C_p):
        return _lstm_step_batched(ch_prev, A_p, Bxpb_p, C_p)

    _, s = jax.jvp(step_params, primals, tangents)
    # s has shape (B, T, 2, N) — move to (B, T, N, 2) for block reduce
    s_blocked = jnp.moveaxis(s, -2, -1)

    # Solve: ch_dot[t] = J[t] @ ch_dot[t-1] + s[t]
    ch_dot_blocked = _parallel_reduce_block_diag_inline(J, s_blocked)
    ch_dot = jnp.moveaxis(ch_dot_blocked, -1, -2)
    return (ch_dot,)


def _fused_lstm_fwd_transpose(ct, A, Bxpb, C, **kwargs):
    """Transpose of fused LSTM-CIFG forward (VJP via adjoint solve)."""
    ct_ch = ct[0]

    if type(ct_ch) is ad.Zero:
        ct_A = ad.Zero(A) if ad.is_undefined_primal(A) else A
        ct_Bxpb = ad.Zero(Bxpb) if ad.is_undefined_primal(Bxpb) else Bxpb
        ct_C = ad.Zero(C) if ad.is_undefined_primal(C) else C
        return ct_A, ct_Bxpb, ct_C

    for name, val in [('A', A), ('Bxpb', Bxpb), ('C', C)]:
        if ad.is_undefined_primal(val):
            raise ValueError(
                f"Cannot transpose fused_lstm_cifg_diag_fwd: {name} is "
                f"undefined but needed for Jacobian computation."
            )

    ch = _fused_lstm_fwd_compute(A, Bxpb, C, kwargs)
    ch_prev = _roll_state(ch, axis=-3)
    jac_neg = _lstm_jacobian_batched(ch, A, Bxpb, C)
    J = -jac_neg

    # Adjoint solve: λ[t] = J[t+1]^T @ λ[t+1] + ct_ch[t]
    # ct_ch shape (B, T, 2, N) → (B, T, N, 2)
    ct_blocked = jnp.moveaxis(ct_ch, -2, -1)
    from ._parallel_reduce import _adjoint_reduce_block_diag
    lam_blocked = _adjoint_reduce_block_diag(J, ct_blocked)
    lam = jnp.moveaxis(lam_blocked, -1, -2)

    # Parameter gradients
    def step_params(A_p, Bxpb_p, C_p):
        return _lstm_step_batched(ch_prev, A_p, Bxpb_p, C_p)

    _, vjp_fn = jax.vjp(step_params, A, Bxpb, C)
    ct_A_val, ct_Bxpb_val, ct_C_val = vjp_fn(lam)

    ct_A = ct_A_val if ad.is_undefined_primal(A) else A
    ct_Bxpb = ct_Bxpb_val if ad.is_undefined_primal(Bxpb) else Bxpb
    ct_C = ct_C_val if ad.is_undefined_primal(C) else C
    return ct_A, ct_Bxpb, ct_C


fused_lstm_cifg_diag_fwd_p = XLACustomKernel('fused_lstm_cifg_diag_fwd')
fused_lstm_cifg_diag_fwd_p.def_kernel('jax_raw', 'cpu', _fused_lstm_fwd_jax_kernel)
fused_lstm_cifg_diag_fwd_p.def_kernel('jax_raw', 'gpu', _fused_lstm_fwd_jax_kernel)
fused_lstm_cifg_diag_fwd_p.def_kernel('jax_raw', 'tpu', _fused_lstm_fwd_jax_kernel)
fused_lstm_cifg_diag_fwd_p.def_tvmffi_kernel('gpu', _fused_lstm_fwd_cuda_kernel, asdefault=True)
fused_lstm_cifg_diag_fwd_p.def_tags('pararnn', 'fused')
fused_lstm_cifg_diag_fwd_p.def_jvp_rule2(_fused_lstm_fwd_jvp_A, _fused_lstm_fwd_jvp_Bxpb, _fused_lstm_fwd_jvp_C)
fused_lstm_cifg_diag_fwd_p.def_transpose_rule(_fused_lstm_fwd_transpose)


def fused_lstm_cifg_diag_forward(
    A: jax.Array,
    Bxpb: jax.Array,
    C: jax.Array,
    *,
    backend: str = None,
) -> jax.Array:
    """Fused LSTM-CIFG forward pass.

    Runs the complete Newton-parallel-reduction solve with 2x2 block-diagonal
    Jacobians.

    Args:
        A: Diagonal recurrence weights, shape ``(3, state_dim)``.
        Bxpb: Precomputed input projection, shape ``(B, T, 3, state_dim)``.
        C: Peephole connection weights, shape ``(2, state_dim)``.
        backend: ``'tvmffi'`` for CUDA, ``None`` for default (``jax_raw``).

    Returns:
        Full state ``[c, h]``, shape ``(B, T, 2, state_dim)``.
    """
    batch_size, seq_len = Bxpb.shape[0], Bxpb.shape[1]
    state_dim = A.shape[1]
    return fused_lstm_cifg_diag_fwd_p(
        A, Bxpb, C,
        A_info=jax.ShapeDtypeStruct(A.shape, A.dtype),
        outs=[jax.ShapeDtypeStruct((batch_size, seq_len, 2, state_dim), A.dtype)],
        backend=backend,
    )[0]


# =============================================================================
# Fused LSTM-CIFG Diagonal — Backward
# =============================================================================

def _fused_lstm_bwd_jax_kernel(**kwargs):
    """jax_raw kernel: fused LSTM-CIFG backward (transposed 2x2 block solve)."""

    def kernel(grad_y, full_state, A, Bxpb, C):
        # grad_y: (B, T, state_dim) — gradient w.r.t. h only
        # full_state: (B, T, 2, state_dim)
        # Build gradient w.r.t. [c, h]: dc=0, dh=grad_y
        grad_ch = jnp.stack([
            jnp.zeros_like(grad_y), grad_y
        ], axis=-2)  # (B, T, 2, N)

        rhs = jnp.flip(grad_ch, axis=-3)
        # Reshape: (B, T, 2, N) -> (B, T, N, 2)
        rhs_blocked = jnp.moveaxis(rhs, -2, -1)

        jac_bwd = _lstm_jacobian_bwd_batched(full_state, A, Bxpb, C)
        dl_blocked = _parallel_reduce_block_diag_inline(jac_bwd, rhs_blocked)
        # (B, T, N, 2) -> (B, T, 2, N)
        dl_ch = jnp.moveaxis(dl_blocked, -1, -2)
        dl_ch = jnp.flip(dl_ch, axis=-3)
        return (dl_ch,)

    return kernel


def _fused_lstm_bwd_cuda_kernel(**kwargs):
    """tvmffi kernel: fused LSTM-CIFG backward via CUDA."""
    load_cuda_file(
        Path(__file__).parent / '_fused_lstm_cifg_diag.cu',
        name='fused_lstm_cifg_diag',
    )
    out_info = kwargs['outs']
    sfx = _dtype_sfx[np.dtype(kwargs['A_info'].dtype)]
    kernel_name = f'fused_lstm_cifg_diag.fused_bwd_lstm_cifg_diag{sfx}'

    def kernel(grad_y, full_state, A, Bxpb, C):
        return jax.ffi.ffi_call(kernel_name, out_info)(grad_y, full_state, A, Bxpb, C)

    return kernel


# -- JVP / transpose for fused LSTM-CIFG backward ----------------------------
#
# The backward primitive computes:
#   dl_ch = flip(S_block(jac_bwd, flip(grad_ch)))
# where grad_ch = [0, grad_y], jac_bwd depends nonlinearly on
# (full_state, A, Bxpb, C), and S_block is a 2x2 block-diagonal solve.
# The output is LINEAR in grad_y but NONLINEAR in the other inputs.

def _fused_lstm_bwd_jvp_grad_y(grad_y_dot, grad_y, full_state, A, Bxpb, C,
                               **kwargs):
    """JVP w.r.t. grad_y (linear): run backward solve on perturbed gradient."""
    grad_ch_dot = jnp.stack([jnp.zeros_like(grad_y_dot), grad_y_dot], axis=-2)
    rhs_dot = jnp.flip(grad_ch_dot, axis=-3)
    rhs_dot_blocked = jnp.moveaxis(rhs_dot, -2, -1)

    jac_bwd = _lstm_jacobian_bwd_batched(full_state, A, Bxpb, C)
    dl_blocked = _parallel_reduce_block_diag_inline(jac_bwd, rhs_dot_blocked)
    dl_ch_dot = jnp.moveaxis(dl_blocked, -1, -2)
    dl_ch_dot = jnp.flip(dl_ch_dot, axis=-3)
    return (dl_ch_dot,)


def _fused_lstm_bwd_transpose(ct, grad_y, full_state, A, Bxpb, C, **kwargs):
    """Transpose of fused LSTM-CIFG backward.

    Only supports transposing w.r.t. grad_y (the linear input).
    """
    ct_dl = ct[0]

    for name, val in [('full_state', full_state), ('A', A),
                      ('Bxpb', Bxpb), ('C', C)]:
        if ad.is_undefined_primal(val):
            raise ValueError(
                f"Cannot transpose fused_lstm_cifg_diag_bwd w.r.t. {name} "
                f"(nonlinear dependency through Jacobian computation)."
            )

    if ad.is_undefined_primal(grad_y):
        if type(ct_dl) is ad.Zero:
            return ad.Zero(grad_y), full_state, A, Bxpb, C
        # Transpose of: dl_ch = flip(S_block(jac_bwd, moveaxis(flip(grad_ch))))
        # Adjoint: flip → S^T → moveaxis_back → flip → extract h component
        jac_bwd = _lstm_jacobian_bwd_batched(full_state, A, Bxpb, C)
        ct_flipped = jnp.flip(ct_dl, axis=-3)
        ct_blocked = jnp.moveaxis(ct_flipped, -2, -1)
        from ._parallel_reduce import _adjoint_reduce_block_diag
        lam_blocked = _adjoint_reduce_block_diag(jac_bwd, ct_blocked)
        lam = jnp.moveaxis(lam_blocked, -1, -2)
        lam = jnp.flip(lam, axis=-3)
        # grad_ch = [0, grad_y] → ct_grad_y = lam[..., 1, :]
        ct_grad_y = lam[..., 1, :]
        return ct_grad_y, full_state, A, Bxpb, C

    return grad_y, full_state, A, Bxpb, C


fused_lstm_cifg_diag_bwd_p = XLACustomKernel('fused_lstm_cifg_diag_bwd')
fused_lstm_cifg_diag_bwd_p.def_kernel('jax_raw', 'cpu', _fused_lstm_bwd_jax_kernel)
fused_lstm_cifg_diag_bwd_p.def_kernel('jax_raw', 'gpu', _fused_lstm_bwd_jax_kernel)
fused_lstm_cifg_diag_bwd_p.def_kernel('jax_raw', 'tpu', _fused_lstm_bwd_jax_kernel)
fused_lstm_cifg_diag_bwd_p.def_tvmffi_kernel('gpu', _fused_lstm_bwd_cuda_kernel, asdefault=True)
fused_lstm_cifg_diag_bwd_p.def_tags('pararnn', 'fused')
fused_lstm_cifg_diag_bwd_p.def_jvp_rule2(_fused_lstm_bwd_jvp_grad_y, None, None, None, None)
fused_lstm_cifg_diag_bwd_p.def_transpose_rule(_fused_lstm_bwd_transpose)


def fused_lstm_cifg_diag_backward(
    grad_y: jax.Array,
    full_state: jax.Array,
    A: jax.Array,
    Bxpb: jax.Array,
    C: jax.Array,
    *,
    backend: str = None,
) -> jax.Array:
    """Fused LSTM-CIFG backward pass.

    Solves the transposed 2x2 block-diagonal system for dl/d[c,h].

    Args:
        grad_y: Gradient w.r.t. output h only, shape ``(B, T, state_dim)``.
        full_state: Forward full state [c, h], shape ``(B, T, 2, state_dim)``.
        A: Diagonal recurrence weights, shape ``(3, state_dim)``.
        Bxpb: Precomputed input projection, shape ``(B, T, 3, state_dim)``.
        C: Peephole connection weights, shape ``(2, state_dim)``.
        backend: ``'tvmffi'`` for CUDA, ``None`` for default.

    Returns:
        dl/d[c,h] with shape ``(B, T, 2, state_dim)``.
    """
    batch_size, seq_len = grad_y.shape[0], grad_y.shape[1]
    state_dim = A.shape[1]
    return fused_lstm_cifg_diag_bwd_p(
        grad_y, full_state, A, Bxpb, C,
        A_info=jax.ShapeDtypeStruct(A.shape, A.dtype),
        outs=[jax.ShapeDtypeStruct((batch_size, seq_len, 2, state_dim), A.dtype)],
        backend=backend,
    )[0]
