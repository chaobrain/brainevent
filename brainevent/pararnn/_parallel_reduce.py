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
Parallel reduction solvers for lower bidiagonal linear systems.

Solves systems of the form:
    h[t] = jac[t] * h[t-1] + rhs[t]

using ``jax.lax.associative_scan`` with the monoid:
    (J_b, r_b) @ (J_a, r_a) = (J_b * J_a, J_b * r_a + r_b)

Three Jacobian structures are supported:
- **Diagonal**: jac is (B, T, N), element-wise multiplication.
- **Block-diagonal 2x2**: jac is (B, T, N, 2, 2), batched 2x2 matmul.
- **Block-diagonal 3x3**: jac is (B, T, N, 3, 3), batched 3x3 matmul.

When ``backend='cuda'`` is specified and TVM FFI is available, GPU-accelerated
CUDA kernels are used instead of ``jax.lax.associative_scan``.
"""

import jax
import jax.numpy as jnp

__all__ = [
    'parallel_reduce_diag',
    'parallel_reduce_diag_bwd',
    'parallel_reduce_block_diag',
    'parallel_reduce_block_diag_bwd',
]


# =============================================================================
# Diagonal Jacobian reduction
# =============================================================================

def _parallel_reduce_diag_jax(jac, rhs):
    """JAX-native diagonal reduction via associative_scan."""

    def combine(a, b):
        j_a, r_a = a
        j_b, r_b = b
        return (j_b * j_a, j_b * r_a + r_b)

    _, h = jax.lax.associative_scan(combine, (jac, rhs), axis=-2)
    return h


def parallel_reduce_diag(
    jac: jax.Array,
    rhs: jax.Array,
    backend: str = None,
) -> jax.Array:
    """Solve h[t] = jac[t]*h[t-1] + rhs[t] with diagonal Jacobians.

    Uses ``jax.lax.associative_scan`` for O(log T) parallel depth, or
    CUDA kernels when ``backend='cuda'``.

    Args:
        jac: Jacobian diagonals, shape ``(B, T, N)``.
        rhs: Right-hand side, shape ``(B, T, N)``.
        backend: ``'cuda'`` for GPU kernels, ``None`` for JAX native.

    Returns:
        Solution ``h`` with shape ``(B, T, N)``.
    """
    if backend == 'cuda':
        from ._parallel_reduce_cuda import parallel_reduce_diag_cuda
        return parallel_reduce_diag_cuda(jac, rhs)
    return _parallel_reduce_diag_jax(jac, rhs)


def parallel_reduce_diag_bwd(
    jac: jax.Array,
    rhs: jax.Array,
    backend: str = None,
) -> jax.Array:
    """Backward (transposed) solve for diagonal Jacobians.

    The Jacobian for backward is already prepared (flipped and shifted) by
    ``compute_jacobians_bwd``, so this function just does a forward scan.

    Args:
        jac: Prepared backward Jacobians, shape ``(B, T, N)``.
        rhs: Gradient right-hand side (flipped), shape ``(B, T, N)``.
        backend: ``'cuda'`` for GPU kernels, ``None`` for JAX native.

    Returns:
        Solution with shape ``(B, T, N)``.
    """
    return parallel_reduce_diag(jac, rhs, backend=backend)


# =============================================================================
# Block-diagonal Jacobian reduction
# =============================================================================

def _parallel_reduce_block_diag_jax(jac, rhs):
    """JAX-native block-diagonal reduction via associative_scan."""

    def combine(a, b):
        j_a, r_a = a
        j_b, r_b = b
        j_new = jnp.einsum('...ij,...jk->...ik', j_b, j_a)
        r_new = jnp.einsum('...ij,...j->...i', j_b, r_a) + r_b
        return (j_new, r_new)

    # axis=1 is the T axis for both jac (B,T,N,K,K) and rhs (B,T,N,K)
    _, h = jax.lax.associative_scan(combine, (jac, rhs), axis=1)
    return h


def parallel_reduce_block_diag(
    jac: jax.Array,
    rhs: jax.Array,
    backend: str = None,
) -> jax.Array:
    """Solve h[t] = jac[t] @ h[t-1] + rhs[t] with block-diagonal Jacobians.

    Args:
        jac: Block-diagonal Jacobians, shape ``(B, T, N, K, K)`` where K is
            the block size (2 or 3).
        rhs: Right-hand side, shape ``(B, T, N, K)``.
        backend: ``'cuda'`` for GPU kernels (K=2 only), ``None`` for JAX native.

    Returns:
        Solution ``h`` with shape ``(B, T, N, K)``.
    """
    if backend == 'cuda':
        k = jac.shape[-1]
        if k == 2:
            from ._parallel_reduce_cuda import parallel_reduce_block2_cuda
            return parallel_reduce_block2_cuda(jac, rhs)
        # K=3 not supported in CUDA, fall through to JAX
    return _parallel_reduce_block_diag_jax(jac, rhs)


def parallel_reduce_block_diag_bwd(
    jac: jax.Array,
    rhs: jax.Array,
    backend: str = None,
) -> jax.Array:
    """Backward (transposed) solve for block-diagonal Jacobians.

    Args:
        jac: Prepared backward Jacobians, shape ``(B, T, N, K, K)``.
        rhs: Gradient right-hand side, shape ``(B, T, N, K)``.
        backend: ``'cuda'`` for GPU kernels, ``None`` for JAX native.

    Returns:
        Solution with shape ``(B, T, N, K)``.
    """
    return parallel_reduce_block_diag(jac, rhs, backend=backend)
