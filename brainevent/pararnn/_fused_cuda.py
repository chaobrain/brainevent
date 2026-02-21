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
Fused CUDA kernels for GRU and LSTM-CIFG parallel training.

Each fused kernel combines the Newton iteration, cell evaluation, Jacobian
computation, and parallel reduction into a single GPU kernel launch,
eliminating intermediate global memory round-trips.

The Python wrappers handle:
- Precomputing Bxpb = B @ x + b (multi-head input projection)
- Calling the CUDA forward kernel
- Implementing ``jax.custom_vjp`` for the backward pass
- Computing parameter gradients from dl/dh in Python

Functions:
    fused_gru_diag_forward: Fused GRU forward pass via CUDA.
    fused_lstm_cifg_diag_forward: Fused LSTM-CIFG forward pass via CUDA.
"""

from pathlib import Path

import jax
import numpy as np

from brainevent._op.util import register_tvm_cuda_from_file

__all__ = [
    'fused_gru_diag_forward',
    'fused_gru_diag_backward',
    'fused_lstm_cifg_diag_forward',
    'fused_lstm_cifg_diag_backward',
    'fused_cuda_available',
]

_dtype_sfx = {
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
}

_fused_registered = False


def _ensure_registered():
    """Register fused CUDA kernels via TVM FFI (cached, idempotent)."""
    global _fused_registered
    if _fused_registered:
        return

    cu_dir = Path(__file__).parent

    register_tvm_cuda_from_file(
        module='fused_gru_diag',
        source=cu_dir / '_fused_gru_diag.cu',
    )
    register_tvm_cuda_from_file(
        module='fused_lstm_cifg_diag',
        source=cu_dir / '_fused_lstm_cifg_diag.cu',
    )
    _fused_registered = True


def fused_cuda_available() -> bool:
    """Check whether fused CUDA kernels are available."""
    try:
        _ensure_registered()
        return True
    except Exception:
        return False


# ============================================================================
# Fused GRU Diagonal
# ============================================================================

def fused_gru_diag_forward(
    A: jax.Array,
    Bxpb: jax.Array,
) -> jax.Array:
    """Fused GRU forward pass via CUDA.

    Runs the complete Newton-parallel-reduction solve in a single kernel
    launch. Input projection (Bxpb = B @ x + b) must be precomputed.

    Args:
        A: Diagonal recurrence weights, shape ``(3, hidden_dim)``.
        Bxpb: Precomputed input projection, shape ``(B, T, 3, hidden_dim)``.

    Returns:
        Hidden states ``h``, shape ``(B, T, hidden_dim)``.
    """
    _ensure_registered()
    batch_size, seq_len = Bxpb.shape[0], Bxpb.shape[1]
    hidden_dim = A.shape[1]
    sfx = _dtype_sfx[np.dtype(A.dtype)]
    kernel_name = f'fused_gru_diag.fused_fwd_gru_diag{sfx}'
    out_info = jax.ShapeDtypeStruct(
        (batch_size, seq_len, hidden_dim), A.dtype
    )
    return jax.ffi.ffi_call(kernel_name, out_info)(A, Bxpb)


def fused_gru_diag_backward(
    grad_y: jax.Array,
    h: jax.Array,
    A: jax.Array,
    Bxpb: jax.Array,
) -> jax.Array:
    """Fused GRU backward pass (transposed system solve) via CUDA.

    Solves the transposed bidiagonal system for dl/dh using a single CUDA
    kernel launch. The result is used to compute parameter gradients in Python.

    Args:
        grad_y: Gradient w.r.t. output, shape ``(B, T, hidden_dim)``.
        h: Forward hidden states, shape ``(B, T, hidden_dim)``.
        A: Diagonal recurrence weights, shape ``(3, hidden_dim)``.
        Bxpb: Precomputed input projection, shape ``(B, T, 3, hidden_dim)``.

    Returns:
        dl/dh with shape ``(B, T, hidden_dim)``.
    """
    _ensure_registered()
    sfx = _dtype_sfx[np.dtype(A.dtype)]
    kernel_name = f'fused_gru_diag.fused_bwd_gru_diag{sfx}'
    out_info = jax.ShapeDtypeStruct(grad_y.shape, grad_y.dtype)
    return jax.ffi.ffi_call(kernel_name, out_info)(grad_y, h, A, Bxpb)


# ============================================================================
# Fused LSTM-CIFG Diagonal
# ============================================================================

def fused_lstm_cifg_diag_forward(
    A: jax.Array,
    Bxpb: jax.Array,
    C: jax.Array,
) -> jax.Array:
    """Fused LSTM-CIFG forward pass via CUDA.

    Runs the complete Newton-parallel-reduction solve with 2x2 block-diagonal
    Jacobians in a single kernel launch.

    Args:
        A: Diagonal recurrence weights, shape ``(3, state_dim)``.
        Bxpb: Precomputed input projection, shape ``(B, T, 3, state_dim)``.
        C: Peephole connection weights, shape ``(2, state_dim)``.

    Returns:
        Full state ``[c, h]``, shape ``(B, T, 2, state_dim)``.
    """
    _ensure_registered()
    batch_size, seq_len = Bxpb.shape[0], Bxpb.shape[1]
    state_dim = A.shape[1]
    sfx = _dtype_sfx[np.dtype(A.dtype)]
    kernel_name = f'fused_lstm_cifg_diag.fused_fwd_lstm_cifg_diag{sfx}'
    out_info = jax.ShapeDtypeStruct(
        (batch_size, seq_len, 2, state_dim), A.dtype
    )
    return jax.ffi.ffi_call(kernel_name, out_info)(A, Bxpb, C)


def fused_lstm_cifg_diag_backward(
    grad_y: jax.Array,
    full_state: jax.Array,
    A: jax.Array,
    Bxpb: jax.Array,
    C: jax.Array,
) -> jax.Array:
    """Fused LSTM-CIFG backward pass via CUDA.

    Solves the transposed 2x2 block-diagonal system for dl/d[c,h].

    Args:
        grad_y: Gradient w.r.t. output h only, shape ``(B, T, state_dim)``.
        full_state: Forward full state [c, h], shape ``(B, T, 2, state_dim)``.
        A: Diagonal recurrence weights, shape ``(3, state_dim)``.
        Bxpb: Precomputed input projection, shape ``(B, T, 3, state_dim)``.
        C: Peephole connection weights, shape ``(2, state_dim)``.

    Returns:
        dl/d[c,h] with shape ``(B, T, 2, state_dim)``.
    """
    _ensure_registered()
    batch_size, seq_len = grad_y.shape[0], grad_y.shape[1]
    state_dim = A.shape[1]
    sfx = _dtype_sfx[np.dtype(A.dtype)]
    kernel_name = f'fused_lstm_cifg_diag.fused_bwd_lstm_cifg_diag{sfx}'
    out_info = jax.ShapeDtypeStruct(
        (batch_size, seq_len, 2, state_dim), A.dtype
    )
    return jax.ffi.ffi_call(kernel_name, out_info)(
        grad_y, full_state, A, Bxpb, C
    )
