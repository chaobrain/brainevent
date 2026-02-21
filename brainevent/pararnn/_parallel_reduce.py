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

Backends:
- Default (``backend=None``): ``jax.lax.associative_scan``, natively
  differentiable.
- ``tvmffi``: CUDA kernels via TVM FFI (GPU only), dispatched through
  ``XLACustomKernel``. Not differentiable — intended for use inside
  ``jax.custom_vjp`` where gradients are handled explicitly.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._op import XLACustomKernel, register_tvm_cuda_from_file

__all__ = [
    'parallel_reduce_diag',
    'parallel_reduce_diag_bwd',
    'parallel_reduce_diag_p',
    'parallel_reduce_block_diag',
    'parallel_reduce_block_diag_bwd',
    'parallel_reduce_block_diag_p',
]


# =============================================================================
# JAX-native implementations (differentiable)
# =============================================================================

def _parallel_reduce_diag_jax(jac, rhs):
    """JAX-native diagonal reduction via associative_scan."""

    def combine(a, b):
        j_a, r_a = a
        j_b, r_b = b
        return (j_b * j_a, j_b * r_a + r_b)

    _, h = jax.lax.associative_scan(combine, (jac, rhs), axis=-2)
    return h


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


# =============================================================================
# XLACustomKernel primitives (for tvmffi backend dispatch)
# =============================================================================

_dtype_sfx = {
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
}


def _reduce_diag_jax_kernel(**kwargs):
    """jax_raw kernel generator for diagonal parallel reduction."""

    def kernel(jac, rhs):
        return (_parallel_reduce_diag_jax(jac, rhs),)

    return kernel


def _reduce_diag_cuda_kernel(**kwargs):
    """tvmffi kernel generator for diagonal parallel reduction."""
    register_tvm_cuda_from_file(
        module='pararnn_reduce_diag',
        source=Path(__file__).parent / '_parallel_reduce_diag.cu',
    )
    out_info = kwargs['outs']
    sfx = _dtype_sfx[np.dtype(kwargs['rhs_info'].dtype)]
    kernel_name = f'pararnn_reduce_diag.pararnn_reduce_diag{sfx}'

    def kernel(jac, rhs):
        return jax.ffi.ffi_call(kernel_name, out_info)(jac, rhs)

    return kernel


parallel_reduce_diag_p = XLACustomKernel('pararnn_reduce_diag')
parallel_reduce_diag_p.def_kernel('jax_raw', 'cpu', _reduce_diag_jax_kernel)
parallel_reduce_diag_p.def_kernel('jax_raw', 'gpu', _reduce_diag_jax_kernel)
parallel_reduce_diag_p.def_kernel('jax_raw', 'tpu', _reduce_diag_jax_kernel)
parallel_reduce_diag_p.def_tvmffi_kernel('gpu', _reduce_diag_cuda_kernel)
parallel_reduce_diag_p.def_tags('pararnn', 'reduce')


def _reduce_block_diag_jax_kernel(**kwargs):
    """jax_raw kernel generator for block-diagonal parallel reduction."""

    def kernel(jac, rhs):
        return (_parallel_reduce_block_diag_jax(jac, rhs),)

    return kernel


def _reduce_block_diag_cuda_kernel(**kwargs):
    """tvmffi kernel generator for 2x2 block-diagonal parallel reduction."""
    register_tvm_cuda_from_file(
        module='pararnn_reduce_block2',
        source=Path(__file__).parent / '_parallel_reduce_block2.cu',
    )
    out_info = kwargs['outs']
    sfx = _dtype_sfx[np.dtype(kwargs['rhs_info'].dtype)]
    kernel_name = f'pararnn_reduce_block2.pararnn_reduce_block2{sfx}'

    def kernel(jac, rhs):
        return jax.ffi.ffi_call(kernel_name, out_info)(jac, rhs)

    return kernel


parallel_reduce_block_diag_p = XLACustomKernel('pararnn_reduce_block_diag')
parallel_reduce_block_diag_p.def_kernel('jax_raw', 'cpu', _reduce_block_diag_jax_kernel)
parallel_reduce_block_diag_p.def_kernel('jax_raw', 'gpu', _reduce_block_diag_jax_kernel)
parallel_reduce_block_diag_p.def_kernel('jax_raw', 'tpu', _reduce_block_diag_jax_kernel)
parallel_reduce_block_diag_p.def_tvmffi_kernel('gpu', _reduce_block_diag_cuda_kernel)
parallel_reduce_block_diag_p.def_tags('pararnn', 'reduce')


# =============================================================================
# Public API — default path uses JAX directly (differentiable)
# =============================================================================

def parallel_reduce_diag(
    jac: jax.Array,
    rhs: jax.Array,
    backend: str = None,
) -> jax.Array:
    """Solve h[t] = jac[t]*h[t-1] + rhs[t] with diagonal Jacobians.

    Uses ``jax.lax.associative_scan`` for O(log T) parallel depth.
    The default path is natively differentiable. When ``backend='tvmffi'``
    is specified, dispatches through ``XLACustomKernel`` to CUDA kernels
    (not differentiable — use inside ``jax.custom_vjp``).

    Args:
        jac: Jacobian diagonals, shape ``(B, T, N)``.
        rhs: Right-hand side, shape ``(B, T, N)``.
        backend: ``'tvmffi'`` for GPU kernels, ``None`` for JAX native.

    Returns:
        Solution ``h`` with shape ``(B, T, N)``.
    """
    if backend is not None and backend != 'jax_raw':
        return parallel_reduce_diag_p(
            jac, rhs,
            rhs_info=jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            outs=[jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)],
            backend=backend,
        )[0]
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
        backend: ``'tvmffi'`` for GPU kernels, ``None`` for default.

    Returns:
        Solution with shape ``(B, T, N)``.
    """
    return parallel_reduce_diag(jac, rhs, backend=backend)


def parallel_reduce_block_diag(
    jac: jax.Array,
    rhs: jax.Array,
    backend: str = None,
) -> jax.Array:
    """Solve h[t] = jac[t] @ h[t-1] + rhs[t] with block-diagonal Jacobians.

    The default path uses ``jax.lax.associative_scan`` (differentiable).
    When ``backend='tvmffi'`` is specified, dispatches through
    ``XLACustomKernel`` to CUDA kernels (K=2 only).

    Args:
        jac: Block-diagonal Jacobians, shape ``(B, T, N, K, K)`` where K is
            the block size (2 or 3).
        rhs: Right-hand side, shape ``(B, T, N, K)``.
        backend: ``'tvmffi'`` for GPU kernels (K=2 only), ``None`` for
            JAX native.

    Returns:
        Solution ``h`` with shape ``(B, T, N, K)``.
    """
    if backend is not None and backend != 'jax_raw':
        # CUDA only supports K=2
        k = jac.shape[-1]
        if backend == 'tvmffi' and k != 2:
            return _parallel_reduce_block_diag_jax(jac, rhs)
        return parallel_reduce_block_diag_p(
            jac, rhs,
            rhs_info=jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            outs=[jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)],
            backend=backend,
        )[0]
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
        backend: ``'tvmffi'`` for GPU kernels, ``None`` for default.

    Returns:
        Solution with shape ``(B, T, N, K)``.
    """
    return parallel_reduce_block_diag(jac, rhs, backend=backend)
