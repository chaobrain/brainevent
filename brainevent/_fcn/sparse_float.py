# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


from typing import Optional, Tuple, Union

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, check_fixed_conn_num_shape, namescope
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, register_tvm_cuda_kernels
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._typing import MatrixShape
from brainevent.config import get_numba_parallel
from .float import fcnmv_p_call, fcnmm_p_call

__all__ = [
    'spfloat_fcnmv',
    'spfloat_fcnmv_p',
    'spfloat_fcnmm',
    'spfloat_fcnmm_p',
]


@namescope(static_argnames=['shape', 'transpose'])
def spfloat_fcnmv(
    weights,
    indices,
    spikes,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse-float event-driven matrix--vector product with fixed connection number.

    Computes ``y = W @ s`` (or ``y = W^T @ s`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``s`` is a sparse-float vector.  Non-zero entries of ``s``
    contribute their actual floating-point value (not just ``1``) to the
    accumulation, combining the sparsity benefit of event-driven processing
    with floating-point precision.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    spikes : jax.Array or u.Quantity
        Sparse-float vector.  Entries equal to zero are skipped; non-zero
        entries are multiplied by the corresponding weight.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool, optional
        If ``False`` (default), compute ``W @ s`` (fixed post-synaptic
        connections, gather mode).  If ``True``, compute ``W^T @ s``
        (fixed pre-synaptic connections, scatter mode).
    backend : str or None, optional
        Execution backend override (``'numba'``,
        ``'pallas'``, ``'tvmffi'``, or ``None`` for automatic selection).

    Returns
    -------
    jax.Array or u.Quantity
        Result vector.  Shape is ``(num_pre,)`` when ``transpose=False``
        or ``(num_post,)`` when ``transpose=True``.

    See Also
    --------
    spfloat_fcnmm : Sparse-float event-driven matrix--matrix product.
    binary_fcnmv : Binary event-driven variant (ignores spike values).
    fcnmv : Dense float variant (no event-driven skipping).

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    Unlike the binary variant (:func:`binary_fcnmv`) which treats non-zero
    entries as ``1``, this sparse-float variant preserves the actual
    floating-point values of the spike vector.  When ``transpose=False``
    (gather mode):

        ``y[i] = sum_{k=0}^{n_conn-1} weights[i, k] * s[indices[i, k]]``

    where only terms with ``s[indices[i, k]] != 0`` are accumulated.  For
    homogeneous weights (``weights`` has shape ``(1,)``):

        ``y[i] = w * sum_{k=0}^{n_conn-1} s[indices[i, k]]``

    When ``transpose=True`` (scatter mode):

        ``y[indices[i, k]] += weights[i, k] * s[i]``    for all ``i, k`` where ``s[i] != 0``

    This formulation is mathematically equivalent to :func:`fcnmv` but skips
    zero entries of ``s``, providing a speedup when the spike vector is sparse.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.sparse_float import spfloat_fcnmv
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)  # homogeneous
        >>> indices = jnp.array([[0, 1], [1, 2]])      # (2, 2)
        >>> spikes = jnp.array([0.0, 2.0, 3.0])
        >>> y = spfloat_fcnmv(weights, indices, spikes, shape=(2, 3))
        >>> print(y)
        [2. 5.]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, spikes, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    spikes, v_unit = u.split_mantissa_unit(spikes)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = spfloat_fcnmv_p_call(
        weights,
        indices,
        spikes,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * v_unit * w_unit)


def _spfloat_fcnmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, spikes, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0.:
                        wsp = w * sp
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += wsp
        else:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0.:
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += weights[i, j] * sp

    else:
        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, spikes, posts):
                w = weights[0]
                for i in numba.prange(indices.shape[0]):
                    r = 0.
                    for j in range(indices.shape[1]):
                        index = indices[i, j]
                        sp = spikes[index]
                        if sp != 0.:
                            r += sp
                    posts[i] = r * w
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, spikes, posts):
                for i in numba.prange(indices.shape[0]):
                    r = 0.
                    for j in range(indices.shape[1]):
                        index = indices[i, j]
                        sp = spikes[index]
                        if sp != 0.:
                            r += weights[i, j] * sp
                    posts[i] = r

    def kernel(weights, indices, spikes):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, spikes)

    return kernel


def _spfloat_fcnmv_pallas_kernel(
    transpose: int,
    shape: Tuple[int, int],
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    block_dim = generate_block_dim(indices_info.shape[1])

    if transpose:
        # Sparse Matrix: [k, m]
        # vector: [k]

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            vector_ref,  # [n_pre]
            _,
            out_ref,  # [n_post]
        ):
            i_row = pl.program_id(0)
            vector = vector_ref[i_row]

            @pl.when(vector != 0. if vector_ref.dtype != jnp.bool_ else vector)
            def run():
                if homo:
                    wv = weight_ref[0] * vector
                    homo_data = jnp.ones(block_dim, dtype=weight_info.dtype) * wv

                def loop_fn(i_col_block, _):
                    i_col = i_col_block * block_dim
                    mask = i_col + jnp.arange(block_dim) < n_conn
                    ind = index_ref[i_row, pl.dslice(i_col, block_dim)]
                    ind = jnp.where(mask, ind, 0)
                    if homo:
                        data = homo_data
                    else:
                        data = weight_ref[i_row, pl.dslice(i_col, block_dim)]
                        data = jnp.where(mask, data * vector, 0.0)
                    atomic_add(out_ref, ind, data, mask=mask)

                jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre,),
                input_output_aliases={3: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            out_info = kwargs['outs'][0]
            return fn(weights, indices, vector, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        # Sparse Matrix: [m, k]
        # vector: [k]

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            vector_ref,  # [n_post]
            out_ref,  # [n_pre]
        ):
            i_row = pl.program_id(0)

            def loop_fn(i_col_block, out):
                i_col = i_col_block * block_dim
                mask = i_col + jnp.arange(block_dim) < n_conn
                ind = index_ref[i_row, pl.dslice(i_col, block_dim)]
                ind = jnp.where(mask, ind, 0)
                vec = vector_ref[ind]
                vec = jnp.where(mask, vec, 0.0)
                if homo:
                    return out + jnp.sum(vec)
                else:
                    weight = weight_ref[i_row, pl.dslice(i_col, block_dim)]
                    weight = jnp.where(mask, weight * vec, 0.0)
                    return out + jnp.sum(weight)

            i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
            if homo:
                i_row_sum = i_row_sum * weight_ref[0]
            out_ref[i_row] = i_row_sum

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(_raw_kernel, grid=(n_pre,), out_shape=kwargs['outs'], backend='triton')
            return fn(weights, indices, vector)

    return kernel


def _spfloat_fcnmv_cuda_kernel(
    transpose: bool,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    """
    TVM FFI CUDA kernels for sparse-float FCN matrix-vector multiplication.

    Optimization strategy over the dense fcnmv:
      Gather (transpose=False):
        y[i] = sum_k w[i,k] * s[idx[i,k]]  (skip when s[idx[i,k]] == 0)
        - __ballot_sync detects all-zero 32-connection chunks → skip weight load
        - At 5% SNN firing rate: ~95% fewer FMA operations
      Scatter (transpose=True):
        y[idx[i,k]] += w[i,k] * s[i]  (entire row skipped when s[i] == 0)
        - Per-pre-neuron early exit: 95% of neurons inactive → 95% skip
        - __shfl_sync broadcasts s[i] to all 32 lanes in warp scatter variant
    """
    register_tvm_cuda_kernels(
        module='spfloat_fcnmv',
        functions=[
            'spfloat_fcnmv_gather_warp',
            'spfloat_fcnmv_gather_basic',
            'spfloat_fcnmv_gather_shared',
            'spfloat_fcnmv_gather_auto',
            'spfloat_fcnmv_scatter_basic',
            'spfloat_fcnmv_scatter_warp',
            'spfloat_fcnmv_scatter_auto',
        ],
        source_code=r"""
#include <cuda_runtime.h>
#include <cstdint>

// ===========================================================================
// Sparse-Float FCN Matrix-Vector (spfloat_fcnmv) CUDA Kernels
//
// KEY DIFFERENCE FROM fcnmv: The input vector (spikes) is SPARSE.
// Optimization strategies:
//   Gather (transpose=False): y[i] = sum_k w[i,k]*s[idx[i,k]]
//     - __ballot_sync detects all-zero 32-connection chunks, skips weight load
//     - At 5% SNN firing rate: ~95% fewer FMA operations
//   Scatter (transpose=True): y[idx[i,k]] += w[i,k]*s[i]
//     - Per-pre-neuron early exit when s[i] == 0
//     - At 5% SNN firing rate: ~95% of pre-neurons skipped entirely
//
// IMPORTANT: weights.data_ptr() returns a GPU device pointer.
// NEVER dereference on host. GPU threads read weights[0] (homo)
// or weights[row*n_conn+k] (hetero).
// ===========================================================================

__device__ __inline__ float spfloat_mv_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ===========================================================================
// GATHER kernels: y[i] = sum_k w[i,k] * s[idx[i,k]]  (skip s == 0)
// ===========================================================================

// Warp gather: one warp (32 threads) per output row.
// Processes 32 consecutive connections per ballot cycle.
// __ballot_sync detects all-zero chunks → skip weight load/multiply entirely.
// Dynamic shared mem: 0 bytes.  Best for n_conn <= 64.
__global__ void _spfloat_gather_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    // All 32 warp threads process 32 consecutive connections per ballot cycle.
    // Out-of-bounds k gives sp = 0 (contributes false to ballot, no divergence).
    for (int base = 0; base < n_conn; base += 32) {
        int k = base + threadIdx.x;
        float sp = (k < n_conn) ? vector[i_row[k]] : 0.0f;
        // __ballot_sync: if all 32 threads have zero spike, skip weight load
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0f);
        if (ballot && k < n_conn && sp != 0.0f)
            val += (is_homo ? weights[0] : w_row[k]) * sp;
    }
    val = spfloat_mv_warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = val;
}

// Basic gather: one block (256 threads) per output row.
// Each of 8 warps processes 32 consecutive connections with ballot-skip.
// Warp-coalesced access: consecutive lanes access consecutive connections.
// Dynamic shared mem: 32 * sizeof(float) for block reduction.
// Best for 64 < n_conn <= 512.
__global__ void _spfloat_gather_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];   // 32 floats for block reduction
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;

    // Warp-level loop: each warp processes 32 consecutive connections per iter.
    // All threads in a warp share the same warp_id → same base → enter/exit together.
    int warp_id = threadIdx.x >> 5;   // 0..7
    int lane    = threadIdx.x & 31;
    float val = 0.0f;
    for (int base = warp_id * 32; base < n_conn; base += blockDim.x) {
        int k = base + lane;
        float sp = (k < n_conn) ? vector[i_row[k]] : 0.0f;
        // Per-warp ballot: skip weight multiply when all 32 spikes are zero
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0f);
        if (ballot && k < n_conn && sp != 0.0f)
            val += (is_homo ? weights[0] : w_row[k]) * sp;
    }

    // Inline block reduction via dynamic shared memory
    val = spfloat_mv_warp_reduce_sum(val);
    if (lane == 0) smem_red[warp_id] = val;
    __syncthreads();
    int n_warps = blockDim.x >> 5;   // 8
    val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warp_id == 0) val = spfloat_mv_warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = val;
}

// Shared-memory gather: tiles idx+weights into shmem to reduce bandwidth.
// Per-warp ballot-skip for zero-spike tiles.
// Dynamic shared mem: blockDim.x*(sizeof(int32_t)+sizeof(float)) + 32*4 bytes.
// Best for n_conn > 512.
__global__ void _spfloat_gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_wt  = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    float*   s_red = reinterpret_cast<float*>(
        smem_raw + blockDim.x * (sizeof(int32_t) + sizeof(float)));

    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;

    float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        // Cooperatively load connection tile into shmem (coalesced access)
        if (k < n_conn) {
            s_idx[threadIdx.x] = i_row[k];
            s_wt[threadIdx.x]  = is_homo ? 1.0f : w_row[k];
        }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        // Per-warp ballot: skip if all spikes in this 32-element chunk are zero
        float sp = (threadIdx.x < tile) ? vector[s_idx[threadIdx.x]] : 0.0f;
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0f);
        if (ballot && threadIdx.x < tile && sp != 0.0f)
            val += s_wt[threadIdx.x] * sp;
        __syncthreads();
    }

    // Inline block reduction
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = spfloat_mv_warp_reduce_sum(val);
    if (lane == 0) s_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = spfloat_mv_warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// ===========================================================================
// SCATTER kernels: y[idx[i,k]] += w[i,k]*s[i]
// KEY OPTIMIZATION: entire pre-neuron skipped when s[i] == 0
// At 5% SNN firing rate: 95% of pre-neurons retired in < 10 instructions
// ===========================================================================

// Basic scatter: one block per pre-neuron.
// Early exit when s[row] == 0 → 95% of blocks retire immediately at 5% rate.
// Pre-computes homo_wsp = weights[0] * sp once per active neuron.
__global__ void _spfloat_scatter_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    float sp = vector[row];
    if (sp == 0.0f) return;   // EARLY EXIT: neuron inactive
    float homo_wsp = is_homo ? weights[0] * sp : 0.0f;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {
        float w_sp = is_homo ? homo_wsp : w_row[k] * sp;
        atomicAdd(&output[i_row[k]], w_sp);
    }
}

// Warp scatter: 8 warps per block (256 threads), one warp per pre-neuron.
// __shfl_sync broadcasts s[row] from lane 0 to all 32 lanes → entire warp
// skips if s[row] == 0, without any shared memory or serialised load.
// Best for small-to-medium n_conn (avoids atomicAdd launch overhead).
__global__ void _spfloat_scatter_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        // Lane 0 loads spike value; broadcast to all 32 lanes via __shfl_sync
        float sp = (lane_id == 0) ? vector[row] : 0.0f;
        sp = __shfl_sync(0xffffffff, sp, 0);   // broadcast from lane 0
        if (sp == 0.0f) continue;              // ENTIRE WARP SKIPS
        float homo_wsp = is_homo ? weights[0] * sp : 0.0f;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        for (int k = lane_id; k < n_conn; k += 32) {
            float w_sp = is_homo ? homo_wsp : w_row[k] * sp;
            atomicAdd(&output[i_row[k]], w_sp);
        }
    }
}

// ===========================================================================
// TVM FFI Entry Points
// Convention: args = (weights, indices, vector, output, stream)
// weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
// GPU threads read weights[0] (homo) or weights[row*n_conn+k] (hetero).
// ===========================================================================

void spfloat_fcnmv_gather_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    _spfloat_gather_warp_kern<<<n_pre, 32, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_gather_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _spfloat_gather_basic_kern<<<n_pre, 256, shm, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_gather_shared(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int threads = 256;
    size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
    _spfloat_gather_shared_kern<<<n_pre, threads, shm, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

// Auto-selects the best gather kernel based on n_conn.
void spfloat_fcnmv_gather_auto(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm_red = 32 * sizeof(float);
    if (n_conn <= 64) {
        // Small n_conn: one warp per row, ballot skip zero chunks
        _spfloat_gather_warp_kern<<<n_pre, 32, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else if (n_conn > 512) {
        // Large n_conn: shared-memory tiling amortises index/weight bandwidth
        int threads = 256;
        size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
        _spfloat_gather_shared_kern<<<n_pre, threads, shm, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        // Medium n_conn: one block per row, 8-warp ballot skip
        _spfloat_gather_basic_kern<<<n_pre, 256, shm_red, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

// Scatter entry points (output zeroed before kernel launch)

void spfloat_fcnmv_scatter_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    // One block per pre-neuron: 95% exit immediately at 5% firing rate
    _spfloat_scatter_basic_kern<<<n_pre, 256, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_scatter_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    // 256 threads = 8 warps; ceil(n_pre / 8) blocks
    int blocks = (n_pre + 7) / 8;
    _spfloat_scatter_warp_kern<<<blocks, 256, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

// Auto-selects the best scatter kernel based on n_conn.
void spfloat_fcnmv_scatter_auto(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    if (n_conn <= 32) {
        // Small n_conn: one warp per neuron, warp-level early exit
        int blocks = (n_pre + 7) / 8;
        _spfloat_scatter_warp_kern<<<blocks, 256, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        // Larger n_conn: one block per neuron, block-level early exit
        _spfloat_scatter_basic_kern<<<n_pre, 256, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}
""",
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]

    if transpose:
        # Scatter mode: y[idx[i,k]] += w[i,k] * s[i]  (skip when s[i] == 0)
        if n_conn <= 32:
            kernel_name = 'spfloat_fcnmv.spfloat_fcnmv_scatter_warp'
        else:
            kernel_name = 'spfloat_fcnmv.spfloat_fcnmv_scatter_auto'

        def kernel(weights, indices, vector):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, vector)

    else:
        # Gather mode: y[i] = sum_k w[i,k] * s[idx[i,k]]  (skip when s == 0)
        if n_conn <= 64:
            kernel_name = 'spfloat_fcnmv.spfloat_fcnmv_gather_warp'
        elif n_conn > 512:
            kernel_name = 'spfloat_fcnmv.spfloat_fcnmv_gather_shared'
        else:
            kernel_name = 'spfloat_fcnmv.spfloat_fcnmv_gather_basic'

        def kernel(weights, indices, vector):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, vector)

    return kernel


def _spfloat_fcnmv_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    n_pre, n_post = shape

    def kernel(weights, indices, vector):
        if transpose:
            # Scatter: y[indices[i,k]] += weights[i,k] * vector[i]
            masked = jnp.broadcast_to(vector[:, None] * weights, indices.shape)
            return jax.ops.segment_sum(masked.ravel(), indices.ravel(), num_segments=n_post),
        else:
            # Gather: y[i] = sum_k weights[i,k] * vector[indices[i,k]]
            if weights.ndim == 0 or weights.size == 1:
                w = weights.ravel()[0]
                return jax.vmap(lambda ind: w * jnp.sum(vector[ind]))(indices),
            else:
                return jax.vmap(lambda w, ind: jnp.sum(w * vector[ind]))(weights, indices),

    return kernel


def _spfloat_fcnmv_jvp_spikes(spk_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return fcnmv_p_call(weights, indices, spk_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _spfloat_fcnmv_jvp_weights(w_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return spfloat_fcnmv_p_call(w_dot, indices, spikes, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _spfloat_fcnmv_transpose_rule(ct, weights, indices, spikes, *, shape, transpose, weight_info, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # dL/dspk = dL/dy * dy/dspk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(spikes):
        if type(ct) is ad.Zero:
            ct_spk = ad.Zero(spikes)
        else:
            ct_spk = fcnmv_p_call(
                weights, indices, ct, shape=shape, transpose=not transpose, backend=kwargs['backend']
            )[0]
        return weights, indices, ct_spk

    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_gmax = ad.Zero(weights)
        elif homo:
            # scalar
            ct_gmax = spfloat_fcnmv_p_call(
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                spikes,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend']
            )
            ct_gmax = jnp.inner(ct, ct_gmax[0]).reshape(*weight_info.shape)
        else:
            if transpose:
                ct_gmax = jax.vmap(lambda v, ind: v * ct[ind])(spikes, indices)
            else:
                ct_gmax = jax.vmap(lambda c, ind: c * spikes[ind])(ct, indices)
        return ct_gmax, indices, spikes


def _spfloat_fcnmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = spfloat_fcnmm_p_call(
            args[0],
            args[1],
            args[2].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = spfloat_fcnmm_p_call(
            args[0],
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(spfloat_fcnmv_p, args, axes, **kwargs)


def _spfloat_fcnmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indices = jnp.asarray(np.random.randint(0, n_post, (n_pre, n_conn), dtype=np.int32))
            if homo:
                weights = jnp.ones(1, dtype=dtype)
            else:
                weights = jnp.ones((n_pre, n_conn), dtype=dtype)
            v_size = n_post if not transpose else n_pre
            vector_data = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            vector_index = jnp.asarray(
                np.sort(np.random.choice(v_size, min(v_size // 5, v_size), replace=False)),
                dtype=jnp.int32,
            )
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, vector_data),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def spfloat_fcnmv_p_call(
    weights,
    indices,
    spikes,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for sparse-float event-driven matrix--vector
    product with fixed connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba, Pallas, or TVM FFI) without performing any
    physical-unit bookkeeping.  It is typically called from
    :func:`spfloat_fcnmv` or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    spikes : jax.Array
        Sparse-float vector; zero entries are treated as inactive.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool, optional
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).  Default is ``False``.
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, ``'tvmffi'``, or
        ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result vector.

    See Also
    --------
    spfloat_fcnmv : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, spikes, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return spfloat_fcnmv_p(
        weights,
        indices,
        spikes,
        outs=[out],
        shape=shape,
        transpose=transpose,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        backend=backend,
    )


spfloat_fcnmv_p = XLACustomKernel(
    'spfloat_fcnmv',
    doc="""
Low-level XLA custom-kernel primitive for ``spfloat_fcnmv``.

This ``XLACustomKernel`` instance dispatches the fixed-connection matrix-vector
multiplication operation with sparse-float inputs to registered backends
(``numba``, ``pallas``, ``tvmffi``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. This sparse-float variant skips zero entries
in the input vector while preserving their actual floating-point values (unlike
the binary variant which treats all non-zero entries as 1).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``spfloat_fcnmv_p.available_backends(platform)``,
and the default backend can be configured with ``spfloat_fcnmv_p.set_default(platform, backend)``.

See Also
--------
spfloat_fcnmv : High-level user-facing function wrapper.
"""
)
spfloat_fcnmv_p.def_numba_kernel(_spfloat_fcnmv_numba_kernel)
spfloat_fcnmv_p.def_pallas_kernel('gpu', _spfloat_fcnmv_pallas_kernel)
spfloat_fcnmv_p.def_tvmffi_kernel('gpu', _spfloat_fcnmv_cuda_kernel)
spfloat_fcnmv_p.def_kernel('jax_raw', 'cpu', _spfloat_fcnmv_jax_kernel)
spfloat_fcnmv_p.def_kernel('jax_raw', 'gpu', _spfloat_fcnmv_jax_kernel)
spfloat_fcnmv_p.def_kernel('jax_raw', 'tpu', _spfloat_fcnmv_jax_kernel)
spfloat_fcnmv_p.def_jvp_rule2(_spfloat_fcnmv_jvp_weights, None, _spfloat_fcnmv_jvp_spikes, None)
spfloat_fcnmv_p.def_transpose_rule(_spfloat_fcnmv_transpose_rule)
spfloat_fcnmv_p.def_batching_rule(_spfloat_fcnmv_batching)
spfloat_fcnmv_p.def_call(spfloat_fcnmv_p_call)
spfloat_fcnmv_p.def_tags('fcn', 'sparse_float')
spfloat_fcnmv_p.def_benchmark_data(_spfloat_fcnmv_benchmark_data)


@namescope(static_argnames=['shape', 'transpose'])
def spfloat_fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse-float event-driven matrix--matrix product with fixed connection number.

    Computes ``Y = W @ M`` (or ``Y = W^T @ M`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``M`` is a dense matrix whose entries may be sparse-float
    values.  Non-zero entries of ``M`` contribute their actual
    floating-point value to the accumulation.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    matrix : jax.Array or u.Quantity
        Dense matrix to multiply with, of shape ``(k, n)`` where ``k``
        matches the appropriate sparse-matrix dimension.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool
        If ``False``, compute ``W @ M`` (fixed post-synaptic connections,
        gather mode).  If ``True``, compute ``W^T @ M`` (fixed
        pre-synaptic connections, scatter mode).
    backend : str or None, optional
        Execution backend override.

    Returns
    -------
    jax.Array or u.Quantity
        Result matrix of shape ``(num_pre, n)`` when ``transpose=False``
        or ``(num_post, n)`` when ``transpose=True``.

    See Also
    --------
    spfloat_fcnmv : Sparse-float event-driven matrix--vector product.
    binary_fcnmm : Binary event-driven variant.
    fcnmm : Dense float variant.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    Unlike the binary variant (:func:`binary_fcnmm`) which treats non-zero
    matrix entries as ``1``, this sparse-float variant preserves the actual
    floating-point values.  When ``transpose=False`` (gather mode):

        ``Y[i, j] = sum_{k=0}^{n_conn-1} weights[i, k] * M[indices[i, k], j]``

    where only terms with ``M[indices[i, k], j] != 0`` are accumulated.  For
    homogeneous weights (``weights`` has shape ``(1,)``):

        ``Y[i, j] = w * sum_{k=0}^{n_conn-1} M[indices[i, k], j]``

    When ``transpose=True`` (scatter mode):

        ``Y[indices[i, k], j] += weights[i, k] * M[i, j]``    for all ``i, k, j`` where ``M[i, j] != 0``

    This formulation is mathematically equivalent to :func:`fcnmm` but skips
    zero entries of ``M``, providing a speedup when the matrix is sparse.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.sparse_float import spfloat_fcnmm
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> matrix = jnp.array([[0.0, 1.0],
        ...                     [2.0, 0.0],
        ...                     [3.0, 4.0]])
        >>> y = spfloat_fcnmm(weights, indices, matrix, shape=(2, 3), transpose=False)
        >>> print(y)
        [[2. 1.]
         [5. 4.]]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = spfloat_fcnmm_p_call(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * m_unit * w_unit)


def _spfloat_fcnmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # fixed pre connection number
        #
        # CSR: [k, m]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, matrix, posts):
                posts[:] = 0.
                w = weights[0]
                for i_k in range(matrix.shape[0]):
                    wv = w * matrix[i_k]
                    for i_conn in range(indices.shape[1]):
                        posts[indices[i_k, i_conn]] += wv
        else:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, matrix, posts):
                posts[:] = 0.
                for i_k in range(matrix.shape[0]):
                    for i_conn in range(indices.shape[1]):
                        posts[indices[i_k, i_conn]] += weights[i_k, i_conn] * matrix[i_k]

    else:
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, matrix, posts):
                w = weights[0]
                for i_m in numba.prange(indices.shape[0]):
                    posts[i_m] = w * np.sum(matrix[indices[i_m]], axis=0)
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, matrix, posts):
                for i_m in numba.prange(indices.shape[0]):
                    posts[i_m] = weights[i_m] @ matrix[indices[i_m]]

    def kernel(weights, indices, matrix):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, matrix)

    return kernel


def _spfloat_fcnmm_pallas_kernel(
    shape: MatrixShape,
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    n_col = matrix_info.shape[1]

    if transpose:
        #
        # fixed pre connection number
        #
        # - CSR: [k, m]
        # - matrix: [k, n]
        #

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            matrix_ref,  # [k, n]
            _,
            out_ref,  # [n_post, n]
        ):
            i_k = pl.program_id(0)
            i_n = pl.program_id(1)
            b = matrix_ref[i_k, i_n]

            def loop_fn(j, _):
                i_m = index_ref[i_k, j]
                if homo:
                    val = weight_ref[0] * b
                else:
                    val = weight_ref[i_k, j] * b
                atomic_add(out_ref, (i_m, i_n), val)

            jax.lax.fori_loop(0, n_conn, loop_fn, None)

        def kernel(weights, indices, matrix):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre, n_col),
                input_output_aliases={3: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            out_info = kwargs['outs'][0]
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)
            return fn(weights, indices, matrix, placeholder)

    else:

        #
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            matrix_ref,  # [k, n]
            out_ref,  # [n_pre, n]
        ):
            i_m = pl.program_id(0)
            i_n = pl.program_id(1)

            def loop_fn(j, acc):
                i_k = index_ref[i_m, j]
                if homo:
                    return acc + matrix_ref[i_k, i_n]
                else:
                    return acc + weight_ref[i_m, j] * matrix_ref[i_k, i_n]

            result = jax.lax.fori_loop(0, n_conn, loop_fn, jnp.zeros((), dtype=matrix_ref.dtype))
            if homo:
                result = result * weight_ref[0]
            out_ref[i_m, i_n] = result

        def kernel(weights, indices, matrix):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre, n_col),
                out_shape=kwargs['outs'],
                backend='triton',
            )
            return fn(weights, indices, matrix)

    return kernel


def _spfloat_fcnmm_cuda_kernel(
    transpose: bool,
    indices_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    **kwargs
):
    """
    TVM FFI CUDA kernels for sparse-float FCN matrix-matrix multiplication.

    Optimization strategy over the dense fcnmm:
      Gather (transpose=False):
        Y[i,j] = sum_k w[i,k] * M[idx[i,k],j]  (skip when M[...] == 0)
        - Per-element zero check avoids FMA when M entry is zero
        - Shared-memory tiling reduces index/weight bandwidth for large n_conn
      Scatter (transpose=True):
        Y[idx[i,k],j] += w[i,k] * M[i,j]  (skip when M[i,j] == 0)
        - Cached scatter: loads M[i,j_tile] into shmem once, then uses warp
          ballot to detect all-zero tiles and skip ALL n_conn atomic scatter
          ops for that tile → 95% savings at 5% SNN firing rate
    """
    register_tvm_cuda_kernels(
        module='spfloat_fcnmm',
        functions=[
            'spfloat_fcnmm_gather_basic',
            'spfloat_fcnmm_gather_shared',
            'spfloat_fcnmm_gather_vec4',
            'spfloat_fcnmm_gather_auto',
            'spfloat_fcnmm_scatter_block',
            'spfloat_fcnmm_scatter_cached',
            'spfloat_fcnmm_scatter_warp',
            'spfloat_fcnmm_scatter_auto',
        ],
        source_code=r"""
#include <cuda_runtime.h>
#include <cstdint>

// ===========================================================================
// Sparse-Float FCN Matrix-Matrix (spfloat_fcnmm) CUDA Kernels
//
// KEY DIFFERENCE FROM fcnmm: skip zero entries in the input matrix M.
//
// Gather mode (transpose=False):
//   Y[i,j] = sum_k w[i,k] * M[idx[i,k],j]   (skip when M[...] == 0)
//   - Per-element zero check avoids FMA for zero M entries
//
// Scatter mode (transpose=True):
//   Y[idx[i,k],j] += w[i,k] * M[i,j]         (skip when M[i,j] == 0)
//   - TILE-LEVEL EARLY EXIT: if entire M[i, j_tile] is zero, skip all n_conn
//     atomic scatter operations for that tile using warp ballot
//   - At 5% SNN firing rate: ~95% of scatter tiles skipped entirely
//
// IMPORTANT: weights.data_ptr() returns a GPU device pointer.
// NEVER dereference on host.
// ===========================================================================

// ===========================================================================
// GATHER kernels: Y[i,j] = sum_k w[i,k] * M[idx[i,k],j]  (skip M == 0)
// ===========================================================================

// Basic gather: one thread per output element Y[i,j].
// Iterates over n_conn connections; skips FMA when M[idx,j] == 0.
// Grid: (n_pre, ceil(n_col/64)), Block: (64,)
__global__ void _spfloat_mm_gather_basic_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [k_dim, n_col]
    float*         __restrict__ output,    // [n_pre, n_col]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre || j >= n_col) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    float acc = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        float m_val = matrix[(size_t)idx_row[k] * n_col + j];
        if (m_val != 0.0f) {   // Skip zero M entries (event-driven optimization)
            float w = is_homo ? weights[0] : w_row[k];
            acc += w * m_val;
        }
    }
    output[(size_t)i * n_col + j] = acc;
}

// Shared-memory gather: tiles connection list into shmem to reduce bandwidth.
// Per-connection zero check for event-driven sparsity.
// Grid: (n_pre, ceil(n_col/64)), Block: (64,)
// Shared mem: SPFLOAT_MM_TK * (sizeof(int32_t) + sizeof(float)) bytes
#define SPFLOAT_MM_TK 128
__global__ void _spfloat_mm_gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ char smem_mm[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_mm);
    float*   s_w   = reinterpret_cast<float*>(smem_mm + SPFLOAT_MM_TK * sizeof(int32_t));

    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;

    float acc = 0.0f;
    for (int k0 = 0; k0 < n_conn; k0 += SPFLOAT_MM_TK) {
        int tile = (k0 + SPFLOAT_MM_TK < n_conn) ? SPFLOAT_MM_TK : (n_conn - k0);
        // Cooperatively load connection tile into shmem
        for (int t = threadIdx.x; t < tile; t += blockDim.x) {
            s_idx[t] = idx_row[k0 + t];
            s_w[t]   = is_homo ? 1.0f : w_row[k0 + t];
        }
        __syncthreads();
        if (j < n_col) {
            for (int t = 0; t < tile; t++) {
                float m_val = matrix[(size_t)s_idx[t] * n_col + j];
                if (m_val != 0.0f)   // Skip zero M entries
                    acc += s_w[t] * m_val;
            }
        }
        __syncthreads();
    }
    if (j < n_col)
        output[(size_t)i * n_col + j] = is_homo ? (weights[0] * acc) : acc;
}

// Vectorised gather: float4 loads for M and output when n_col % 4 == 0.
// Any non-zero in float4 group → process all 4 elements (coarse zero check).
// Grid: (n_pre, ceil(n_col/4 / 64)), Block: (64,)
__global__ void _spfloat_mm_gather_vec4_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,   // [k_dim, n_col], n_col%4==0
    float*         __restrict__ output,   // [n_pre, n_col]
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i   = blockIdx.x;
    int j4  = blockIdx.y * blockDim.x + threadIdx.x;  // float4 group index
    int nc4 = n_col >> 2;
    if (i >= n_pre || j4 >= nc4) return;

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const float4*  mat4    = reinterpret_cast<const float4*>(matrix);

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < n_conn; k++) {
        float  w = is_homo ? weights[0] : w_row[k];
        float4 m = mat4[(size_t)idx_row[k] * nc4 + j4];
        // Coarse zero check: if any component non-zero, accumulate all 4
        if (m.x != 0.0f || m.y != 0.0f || m.z != 0.0f || m.w != 0.0f) {
            acc.x += w * m.x;
            acc.y += w * m.y;
            acc.z += w * m.z;
            acc.w += w * m.w;
        }
    }
    reinterpret_cast<float4*>(output)[(size_t)i * nc4 + j4] = acc;
}

// Auto-selects the best gather kernel based on n_conn and n_col.
void spfloat_fcnmm_gather_auto_device(
    const float* d_w, const int32_t* d_idx, const float* d_mat, float* d_out,
    int n_pre, int n_conn, int n_col, int is_homo, cudaStream_t s
) {
    int BJ = 64;
    if (n_col % 4 == 0 && n_col >= 64) {
        dim3 grid(n_pre, (n_col / 4 + BJ - 1) / BJ);
        _spfloat_mm_gather_vec4_kern<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 128) {
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = SPFLOAT_MM_TK * (sizeof(int32_t) + sizeof(float));
        _spfloat_mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        _spfloat_mm_gather_basic_kern<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// ===========================================================================
// SCATTER kernels (transpose=True)
// Y[idx[i,k],j] += w[i,k] * M[i,j]   (Y pre-zeroed, skip when M[i,j]==0)
// KEY OPTIMIZATION: tile-level early exit using warp ballot when M[i,:] == 0
// ===========================================================================

// Block scatter: one block per pre-neuron, per-element zero check.
// Skips atomicAdd when m_val == 0 (avoids false writes for sparse M).
// Grid: (n_pre,), Block: (256,)
__global__ void _spfloat_mm_scatter_block_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [n_pre, n_col]
    float*         __restrict__ output,    // [n_post, n_col] (pre-zeroed)
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const float*   m_row   = matrix + (size_t)i * n_col;
    for (int k = 0; k < n_conn; k++) {
        int   tgt = idx_row[k];
        float w   = is_homo ? weights[0] : w_row[k];
        float* out_row = output + (size_t)tgt * n_col;
        for (int j = threadIdx.x; j < n_col; j += blockDim.x) {
            float m_val = m_row[j];
            if (m_val != 0.0f)   // Skip zero M entries (no-op atomicAdd)
                atomicAdd(&out_row[j], w * m_val);
        }
    }
}

// Cached scatter: 2D grid — one block per (pre-neuron, n_col tile).
// TILE-LEVEL EARLY EXIT: loads M[i, j_tile] into shmem, then uses warp ballot
// to detect all-zero tiles. If the tile is all-zero, skip ALL n_conn scatter
// operations for this (i, j_tile) combination.
// At 5% SNN firing rate: ~95% of blocks return after the shmem load + ballot.
// Grid: (n_pre, ceil(n_col/BJ)), Block: (BJ,)
// Shared mem: BJ * sizeof(float) + sizeof(unsigned)
#define SPFLOAT_MM_SCATTER_BJ 128
__global__ void _spfloat_mm_scatter_cached_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [n_pre, n_col]
    float*         __restrict__ output,    // [n_post, n_col] (pre-zeroed)
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ float s_m[];   // M[i, j_tile] tile cache
    __shared__ unsigned s_any_nz;    // Tile sparsity flag (static in __global__ is OK)

    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    // Load M[i, j] tile into shmem (single global read per element)
    float m_val = (j < n_col) ? matrix[(size_t)i * n_col + j] : 0.0f;
    s_m[threadIdx.x] = m_val;

    // Initialize sparsity flag
    if (threadIdx.x == 0) s_any_nz = 0u;
    __syncthreads();

    // Warp-ballot based tile sparsity check:
    // Each warp votes on its 32 elements; if any warp detects a non-zero value,
    // it atomically sets the shared flag.
    unsigned warp_nz = __ballot_sync(0xffffffff, m_val != 0.0f);
    if (warp_nz && ((threadIdx.x & 31) == 0))  // warp leader writes
        atomicOr(&s_any_nz, 1u);
    __syncthreads();

    // TILE-LEVEL EARLY EXIT: skip all n_conn scatter ops if tile is all-zero
    // This is the KEY optimization for sparse input matrices (SNNs at 5% rate)
    if (!s_any_nz) return;

    // Scatter for non-zero elements: each thread handles one j column
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    if (j < n_col && m_val != 0.0f) {
        for (int k = 0; k < n_conn; k++) {
            int   tgt = idx_row[k];
            float w   = is_homo ? weights[0] : w_row[k];
            atomicAdd(&output[(size_t)tgt * n_col + j], w * m_val);
        }
    }
}

// Warp scatter: grid-stride over (pre-neuron, connection) pairs.
// Per-element zero check for M[i,j]; avoids atomicAdd when m_val == 0.
// Grid: (min(4096, ceil(n_pre*n_conn/8)),), Block: (256,)
__global__ void _spfloat_mm_scatter_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    int n_warps = (gridDim.x * blockDim.x) >> 5;
    int n_pairs = n_pre * n_conn;

    for (int pair = wid; pair < n_pairs; pair += n_warps) {
        int i = pair / n_conn;
        int k = pair % n_conn;
        int   tgt = indices[(size_t)i * n_conn + k];
        float w   = is_homo ? weights[0] : weights[(size_t)i * n_conn + k];
        const float* m_row   = matrix + (size_t)i * n_col;
        float*       out_row = output + (size_t)tgt * n_col;
        for (int j = lane; j < n_col; j += 32) {
            float m_val = m_row[j];
            if (m_val != 0.0f)   // Skip zero M entries
                atomicAdd(&out_row[j], w * m_val);
        }
    }
}

// ===========================================================================
// TVM FFI Entry Points
// Convention: args = (weights, indices, matrix, output, stream)
// weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
// Gather:  matrix [n_post, n_col], output [n_pre, n_col]
// Scatter: matrix [n_pre,  n_col], output [n_post, n_col]
// ===========================================================================

void spfloat_fcnmm_gather_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    _spfloat_mm_gather_basic_kern<<<grid, BJ, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void spfloat_fcnmm_gather_shared(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    size_t shm = SPFLOAT_MM_TK * (sizeof(int32_t) + sizeof(float));
    _spfloat_mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void spfloat_fcnmm_gather_vec4(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int BJ4 = 64;
    dim3 grid(n_pre, (n_col / 4 + BJ4 - 1) / BJ4);
    _spfloat_mm_gather_vec4_kern<<<grid, BJ4, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void spfloat_fcnmm_gather_auto(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    spfloat_fcnmm_gather_auto_device(d_w, d_idx, d_mat, d_out, n_pre, n_conn, n_col, is_homo, s);
}

// --- Scatter entry points (output zeroed before kernel launch) ---

void spfloat_fcnmm_scatter_block(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(float), s);
    _spfloat_mm_scatter_block_kern<<<n_pre, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void spfloat_fcnmm_scatter_cached(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(float), s);
    int BJ = SPFLOAT_MM_SCATTER_BJ;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    // Shared mem: M[i] tile + sparsity flag (aligned)
    size_t shm = BJ * sizeof(float) + sizeof(unsigned);
    _spfloat_mm_scatter_cached_kern<<<grid, BJ, shm, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void spfloat_fcnmm_scatter_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(float), s);
    int n_pairs = n_pre * n_conn;
    int blocks  = min(4096, (n_pairs + 7) / 8);
    _spfloat_mm_scatter_warp_kern<<<blocks, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

// Auto-selects the best scatter kernel based on n_conn and n_col.
void spfloat_fcnmm_scatter_auto(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(float), s);

    if (n_col <= 64) {
        // Small n_col: warp-per-(i,k) maximises parallelism over connections
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _spfloat_mm_scatter_warp_kern<<<blocks, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        // Larger n_col: cached M[i] in shmem with tile-level sparsity check.
        // At 5% firing rate: 95% of blocks exit before any scatter op.
        int BJ = SPFLOAT_MM_SCATTER_BJ;
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = BJ * sizeof(float) + sizeof(unsigned);
        _spfloat_mm_scatter_cached_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}
""",
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    n_col = matrix_info.shape[1]

    if transpose:
        # Scatter mode: Y[idx[i,k],j] += w[i,k] * M[i,j]  (skip when M[i,j]==0)
        # Cached scatter with tile-level early exit is the default for good n_col
        kernel_name = 'spfloat_fcnmm.spfloat_fcnmm_scatter_auto'

        def kernel(weights, indices, matrix):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, matrix)

    else:
        # Gather mode: Y[i,j] = sum_k w[i,k] * M[idx[i,k],j]  (skip M==0)
        if n_col % 4 == 0 and n_col >= 64:
            kernel_name = 'spfloat_fcnmm.spfloat_fcnmm_gather_vec4'
        elif n_conn > 128:
            kernel_name = 'spfloat_fcnmm.spfloat_fcnmm_gather_shared'
        else:
            kernel_name = 'spfloat_fcnmm.spfloat_fcnmm_gather_auto'

        def kernel(weights, indices, matrix):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, matrix)

    return kernel


def _spfloat_fcnmm_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    n_pre, n_post = shape

    def kernel(weights, indices, matrix):
        if transpose:
            # Scatter: Y[indices[i,k], j] += weights[i,k] * matrix[i, j]
            n = matrix.shape[1]
            n_conn = indices.shape[1]
            M_exp = jnp.broadcast_to(matrix[:, None, :], (n_pre, n_conn, n))
            if weights.ndim == 0 or weights.size == 1:
                vals = weights.ravel()[0] * M_exp
            else:
                vals = weights[:, :, None] * M_exp
            return jax.ops.segment_sum(vals.reshape(-1, n), indices.ravel(), num_segments=n_post),
        else:
            # Gather: Y[i, j] = sum_k weights[i,k] * matrix[indices[i,k], j]
            if weights.ndim == 0 or weights.size == 1:
                w = weights.ravel()[0]
                return jax.vmap(lambda ind: w * jnp.sum(matrix[ind], axis=0))(indices),
            else:
                return jax.vmap(lambda w, ind: jnp.sum(w[:, None] * matrix[ind], axis=0))(weights, indices),

    return kernel


def _spfloat_fcnmm_jvp_matrix(matrix_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return fcnmm_p_call(weights, indices, matrix_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _spfloat_fcnmm_jvp_weights(weights_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return spfloat_fcnmm_p_call(
        weights_dot, indices, matrix, shape=shape, transpose=transpose, backend=kwargs['backend'],
    )


def _spfloat_fcnmm_transpose_rule(ct, weights, indices, matrix, *, shape, transpose, weight_info, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # dL/dspk = dL/dy * dy/dspk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(matrix):
        if type(ct) is ad.Zero:
            ct_vector = ad.Zero(matrix)

        else:
            ct_vector = fcnmm_p_call(
                weights,
                indices,
                ct,
                shape=shape,
                transpose=not transpose,
                backend=kwargs['backend'],
            )[0]

        return weights, indices, ct_vector
    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_weight = ad.Zero(weights)

        elif homo:
            ct_weight = spfloat_fcnmm_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                matrix,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )[0]
            ct_weight = jnp.sum(ct * ct_weight).reshape(*weight_info.shape)

        else:
            if transpose:
                # inputs: [k, n] @ [k, n_conn]
                # ct: [m, n]
                ct_weight = jax.vmap(lambda mat, ind: ct[ind] @ mat)(matrix, indices)
            else:
                # inputs: [m, n] @ [m, n_conn]
                # ct: [k, n]
                ct_weight = jax.vmap(lambda c, ind: (matrix[ind] @ c))(ct, indices)
        return ct_weight, indices, matrix


def _batching_base_fn(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = spfloat_fcnmm_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _spfloat_fcnmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 1):
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 2):
        return _batching_base_fn(args, axis=2, **kwargs)

    else:
        return general_batching_rule(spfloat_fcnmm_p, args, axes, **kwargs)


def _spfloat_fcnmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indices = jnp.asarray(np.random.randint(0, n_post, (n_pre, n_conn), dtype=np.int32))
            if homo:
                weights = jnp.ones(1, dtype=dtype)
            else:
                weights = jnp.ones((n_pre, n_conn), dtype=dtype)
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, B),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def spfloat_fcnmm_p_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for sparse-float event-driven matrix--matrix
    product with fixed connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba, Pallas, or TVM FFI) without performing any
    physical-unit bookkeeping.  It is typically called from
    :func:`spfloat_fcnmm` or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    matrix : jax.Array or u.Quantity
        Dense matrix to multiply with, of shape ``(k, n)``.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, ``'tvmffi'``, or
        ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result matrix.

    Notes
    -----
    The ``transpose=True`` path uses scatter-based accumulation via
    ``atomic_add`` on GPU backends, while ``transpose=False`` uses a
    gather-based reduction.

    See Also
    --------
    spfloat_fcnmm : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return spfloat_fcnmm_p(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=[out],
        backend=backend,
    )


spfloat_fcnmm_p = XLACustomKernel(
    'spfloat_fcnmm',
    doc="""
Low-level XLA custom-kernel primitive for ``spfloat_fcnmm``.

This ``XLACustomKernel`` instance dispatches the fixed-connection matrix-matrix
multiplication operation with sparse-float inputs to registered backends
(``numba``, ``pallas``, ``tvmffi``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. This sparse-float variant skips zero entries
in the input matrix while preserving their actual floating-point values (unlike
the binary variant which treats all non-zero entries as 1).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``spfloat_fcnmm_p.available_backends(platform)``,
and the default backend can be configured with ``spfloat_fcnmm_p.set_default(platform, backend)``.

See Also
--------
spfloat_fcnmm : High-level user-facing function wrapper.
"""
)
spfloat_fcnmm_p.def_numba_kernel(_spfloat_fcnmm_numba_kernel)
spfloat_fcnmm_p.def_pallas_kernel('gpu', _spfloat_fcnmm_pallas_kernel)
spfloat_fcnmm_p.def_tvmffi_kernel('gpu', _spfloat_fcnmm_cuda_kernel)
spfloat_fcnmm_p.def_kernel('jax_raw', 'cpu', _spfloat_fcnmm_jax_kernel)
spfloat_fcnmm_p.def_kernel('jax_raw', 'gpu', _spfloat_fcnmm_jax_kernel)
spfloat_fcnmm_p.def_kernel('jax_raw', 'tpu', _spfloat_fcnmm_jax_kernel)
spfloat_fcnmm_p.def_jvp_rule2(_spfloat_fcnmm_jvp_weights, None, _spfloat_fcnmm_jvp_matrix, None)
spfloat_fcnmm_p.def_transpose_rule(_spfloat_fcnmm_transpose_rule)
spfloat_fcnmm_p.def_batching_rule(_spfloat_fcnmm_batching)
spfloat_fcnmm_p.def_call(spfloat_fcnmm_p_call)
spfloat_fcnmm_p.def_tags('fcn', 'sparse_float')
spfloat_fcnmm_p.def_benchmark_data(_spfloat_fcnmm_benchmark_data)
