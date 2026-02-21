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
CUDA-accelerated parallel reduction solvers via TVM FFI.

Provides GPU-accelerated alternatives to the ``jax.lax.associative_scan``
implementations in ``_parallel_reduce.py``. Falls back to the JAX-native
implementation when TVM FFI is not available or when running on CPU.

Functions:
    parallel_reduce_diag_cuda: Diagonal Jacobian reduction on GPU.
    parallel_reduce_block2_cuda: 2x2 block-diagonal Jacobian reduction on GPU.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'parallel_reduce_diag_cuda',
    'parallel_reduce_block2_cuda',
    'cuda_available',
]

_dtype_sfx = {
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
}

_cuda_registered = False


def _ensure_registered():
    """Register CUDA kernels via TVM FFI (cached, idempotent)."""
    global _cuda_registered
    if _cuda_registered:
        return
    from brainevent._op.util import register_tvm_cuda_from_file
    cu_dir = Path(__file__).parent

    register_tvm_cuda_from_file(
        module='pararnn_reduce_diag',
        source=cu_dir / '_parallel_reduce_diag.cu',
    )
    register_tvm_cuda_from_file(
        module='pararnn_reduce_block2',
        source=cu_dir / '_parallel_reduce_block2.cu',
    )
    _cuda_registered = True


def cuda_available() -> bool:
    """Check whether TVM FFI CUDA backend is available."""
    try:
        _ensure_registered()
        return True
    except Exception:
        return False


def parallel_reduce_diag_cuda(jac: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve h[t] = jac[t]*h[t-1] + rhs[t] using CUDA parallel reduction.

    CUDA implementation of the 5-step hierarchical reduction algorithm
    (Thomas + PCR). Equivalent to ``parallel_reduce_diag`` but runs entirely
    on the GPU without JAX intermediate operations.

    Args:
        jac: Jacobian diagonals, shape ``(B, T, N)``.
        rhs: Right-hand side, shape ``(B, T, N)``.

    Returns:
        Solution ``h`` with shape ``(B, T, N)``.
    """
    _ensure_registered()
    sfx = _dtype_sfx[np.dtype(rhs.dtype)]
    kernel_name = f'pararnn_reduce_diag.pararnn_reduce_diag{sfx}'
    out_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
    return jax.ffi.ffi_call(kernel_name, out_info)(jac, rhs)


def parallel_reduce_block2_cuda(jac: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve h[t] = J[t]@h[t-1] + rhs[t] with 2x2 block-diagonal Jacobians.

    CUDA implementation of the 5-step hierarchical reduction algorithm
    with 2x2 matrix monoid.

    Args:
        jac: Block-diagonal Jacobians, shape ``(B, T, N, 2, 2)``.
        rhs: Right-hand side, shape ``(B, T, N, 2)``.

    Returns:
        Solution ``h`` with shape ``(B, T, N, 2)``.
    """
    _ensure_registered()
    sfx = _dtype_sfx[np.dtype(rhs.dtype)]
    kernel_name = f'pararnn_reduce_block2.pararnn_reduce_block2{sfx}'
    out_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
    return jax.ffi.ffi_call(kernel_name, out_info)(jac, rhs)
