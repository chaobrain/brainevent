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

from typing import Optional, Union, Tuple, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, check_fixed_conn_num_shape, namescope
from brainevent._op import (
    general_batching_rule, XLACustomKernel, numba_kernel, register_tvm_cuda_kernels,
    BenchmarkConfig
)
from brainevent.config import get_numba_parallel

__all__ = [
    'fcnmv',
    'fcnmv_p',
    'fcnmm',
    'fcnmm_p',
]


@namescope(static_argnames=['shape', 'transpose'])
def fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse matrix--vector product with fixed connection number.

    Computes ``y = W @ v`` (or ``y = W^T @ v`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``v`` is a dense floating-point vector.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    vector : jax.Array or u.Quantity
        Dense vector to multiply with.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool
        If ``False``, compute ``W @ v`` (fixed post-synaptic connections,
        gather mode).  If ``True``, compute ``W^T @ v`` (fixed
        pre-synaptic connections, scatter mode).
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
    fcnmm : Float sparse matrix--matrix product with fixed connection number.
    binary_fcnmv : Event-driven (binary) variant.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    When ``transpose=False`` (gather mode), the matrix-vector product computes:

        ``y[i] = sum_{k=0}^{n_conn-1} weights[i, k] * v[indices[i, k]]``

    For homogeneous weights (``weights`` has shape ``(1,)``):

        ``y[i] = w * sum_{k=0}^{n_conn-1} v[indices[i, k]]``

    When ``transpose=True`` (scatter mode), the computation distributes each
    row's contributions to the target columns:

        ``y[indices[i, k]] += weights[i, k] * v[i]``    for all ``i, k``

    The computational cost is ``O(num_pre * n_conn)`` regardless of the number
    of post-synaptic neurons, making this efficient for sparse connectivity.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.float import fcnmv
        >>>
        >>> weights = jnp.array([[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float32)
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> vector = jnp.array([1.0, 2.0, 3.0])
        >>> y = fcnmv(weights, indices, vector, shape=(2, 3), transpose=False)
        >>> print(y)
        [2.5 9. ]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, vector, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    vector, v_unit = u.split_mantissa_unit(vector)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = fcnmv_p_call(
        weights,
        indices,
        vector,
        transpose=transpose,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * v_unit * w_unit)


def _fcnmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # fixed pre connection number
        if weight_info.size == 1:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, vector, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(vector.shape[0]):
                    wv = w * vector[i]
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += wv
        else:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, vector, posts):
                posts[:] = 0.
                for i in range(vector.shape[0]):
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += weights[i, j] * vector[i]

    else:
        # fixed post connection number
        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, vector, posts):
                w = weights[0]
                for i in numba.prange(indices.shape[0]):
                    posts[i] = w * np.sum(vector[indices[i]])
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, vector, posts):
                for i in numba.prange(indices.shape[0]):
                    posts[i] = np.sum(weights[i] * vector[indices[i]])

    def kernel(weights, indices, vector):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, vector)

    return kernel


def _fcnmv_pallas_kernel(
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    block_dim = generate_block_dim(indices_info.shape[1], maximum=128)

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
            if homo:
                wv = vector * weight_ref[0]
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
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)
            return fn(weights, indices, vector, placeholder)

    else:
        # Sparse Matrix: [m, k]
        # vector: [k]

        def _raw_kernel(
            weight_ref,  # [1]
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
                    weight = jnp.where(mask, weight, 0.0)
                    return out + jnp.sum(weight * vec)

            i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
            if homo:
                i_row_sum = i_row_sum * weight_ref[0]
            out_ref[i_row] = i_row_sum

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(_raw_kernel, grid=(n_pre,), out_shape=kwargs['outs'], backend='triton')
            return fn(weights, indices, vector)

    return kernel


def _fcnmv_cuda_kernel(
    transpose: bool,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    _FCN_MV_FLOAT_CUDA = register_tvm_cuda_kernels(
        module='fcnmv',
        functions=[
            'fcnmv_gather_auto', 'fcnmv_gather_warp',
            'fcnmv_gather_basic', 'fcnmv_gather_shared', 'fcnmv_gather_vec4',
            'fcnmv_scatter_auto', 'fcnmv_scatter_warp',
            'fcnmv_scatter_basic', 'fcnmv_scatter_gridstride',
        ],
        source_code=r"""
#include <cuda_runtime.h>
#include <cstdint>

// =========================================================================
// Warp reduction helper
// =========================================================================

__device__ __inline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =========================================================================
// Gather device kernels (transpose=False)
// y[i] = sum_k w[i,k] * v[idx[i,k]]
//
// weights: device pointer. is_homo=1 => shape (1,); is_homo=0 => shape (n_pre,n_conn)
// GPU threads read weights[0] for homo, weights[row*n_conn+k] for hetero.
// =========================================================================

// One warp (32 threads) per output row. Handles any n_conn via loop.
__global__ void _gather_warp_kern(
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
    for (int k = threadIdx.x; k < n_conn; k += 32)
        val += is_homo ? vector[i_row[k]] : (w_row[k] * vector[i_row[k]]);
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block (256 threads) per output row with inline block reduction.
// Uses 32*sizeof(float) of dynamic shared memory for the reduction scratchpad.
// Best for 32 < n_conn <= 512.
__global__ void _gather_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];  // 32 floats for block reduction
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        val += is_homo ? vector[i_row[k]] : (w_row[k] * vector[i_row[k]]);
    // Inline block reduction via dynamic shared memory
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block per row, cooperative tile-load of indices+weights into shared memory.
// Shared memory layout (dynamic, size = blockDim.x*(4+4) + 32*4 bytes):
//   [0 .. blockDim.x*4)              : int32_t s_idx[blockDim.x]
//   [blockDim.x*4 .. 2*blockDim.x*4) : float   s_wt[blockDim.x]
//   [2*blockDim.x*4 .. +32*4)        : float   s_red[32]  (block reduction)
// Best for n_conn > 512.
__global__ void _gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_wt  = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    float*   s_red = reinterpret_cast<float*>(smem_raw + blockDim.x * (sizeof(int32_t) + sizeof(float)));

    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;

    float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        if (k < n_conn) {
            s_idx[threadIdx.x] = i_row[k];
            s_wt[threadIdx.x]  = is_homo ? 1.0f : w_row[k];
        }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        if (threadIdx.x < tile)
            val += s_wt[threadIdx.x] * vector[s_idx[threadIdx.x]];
        __syncthreads();
    }

    // Inline block reduction using s_red from dynamic shared memory
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) s_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block per row, float4/int4 vectorised loads with inline block reduction.
// Uses 32*sizeof(float) of dynamic shared memory for the reduction scratchpad.
// Best when n_conn % 4 == 0.
__global__ void _gather_vec4_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];  // 32 floats for block reduction
    int row = blockIdx.x;
    if (row >= n_pre) return;
    size_t base = (size_t)row * n_conn;
    const int4*   i4 = reinterpret_cast<const int4*>(indices + base);
    const float4* w4 = is_homo ? nullptr : reinterpret_cast<const float4*>(weights + base);
    int n4 = n_conn >> 2;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n4; k += blockDim.x) {
        int4 idx = i4[k];
        if (!is_homo) {
            float4 ww = w4[k];
            val += ww.x * vector[idx.x] + ww.y * vector[idx.y]
                 + ww.z * vector[idx.z] + ww.w * vector[idx.w];
        } else {
            val += vector[idx.x] + vector[idx.y] + vector[idx.z] + vector[idx.w];
        }
    }
    // Scalar remainder (when n_conn % 4 != 0)
    for (int k = (n4 << 2) + threadIdx.x; k < n_conn; k += blockDim.x) {
        float v = vector[indices[base + k]];
        val += is_homo ? v : (weights[base + k] * v);
    }
    // Inline block reduction via dynamic shared memory
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// =========================================================================
// Scatter device kernels (transpose=True)
// y[idx[i,k]] += w[i,k] * v[i]   (output must be pre-zeroed)
// =========================================================================

// One block per pre-neuron, threads stride over n_conn.
__global__ void _scatter_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    float v = vector[row];
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], (is_homo ? weights[0] : w_row[k]) * v);
}

// 8 warps per block (256 threads), one warp per pre-neuron.
// Grid = ceil(n_pre / 8) blocks.
__global__ void _scatter_warp_kern(
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
        float v = vector[row];
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], (is_homo ? weights[0] : w_row[k]) * v);
    }
}

// Flat grid-stride over all (pre, conn) pairs. Maximises SM occupancy.
__global__ void _scatter_gs_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int total  = n_pre * n_conn;
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / n_conn;
        float w = is_homo ? weights[0] : weights[idx];
        atomicAdd(&output[indices[idx]], w * vector[row]);
    }
}

// =========================================================================
// TVM FFI Entry Points
// =========================================================================
// Convention: args = (weights, indices, vector, output, stream)
// weights: shape (1,) for homo or (n_pre, n_conn) for hetero, float32
// indices: shape (n_pre, n_conn), int32
// vector:  shape (n_post,) for gather, (n_pre,) for scatter, float32
// output:  shape (n_pre,) for gather, (n_post,) for scatter, float32
//
// IMPORTANT: weights.data_ptr() is a GPU device pointer.
// NEVER dereference it on the host. Pass it to kernels unchanged.
// GPU threads read weights[0] (homo) or weights[row*n_conn+k] (hetero).

// --- Gather entry points ---

void fcnmv_gather_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;   // metadata: host-safe
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());  // device ptr, not dereferenced
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    _gather_warp_kern<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_gather_basic(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _gather_basic_kern<<<n_pre, 256, shm, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_gather_shared(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int threads = 256;
    // Dynamic shared mem: s_idx[threads] + s_wt[threads] + s_red[32]
    size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
    _gather_shared_kern<<<n_pre, threads, shm, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_gather_vec4(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _gather_vec4_kern<<<n_pre, 256, shm, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

// Auto-selects the best gather kernel based on n_conn.
void fcnmv_gather_auto(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());

    size_t shm_red = 32 * sizeof(float);
    if (n_conn <= 32) {
        _gather_warp_kern<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else if (n_conn % 4 == 0 && n_conn >= 128) {
        _gather_vec4_kern<<<n_pre, 256, shm_red, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else if (n_conn > 512) {
        int threads = 256;
        size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
        _gather_shared_kern<<<n_pre, threads, shm, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else {
        _gather_basic_kern<<<n_pre, 256, shm_red, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    }
}

// --- Scatter entry points (output zeroed before kernel launch) ---

void fcnmv_scatter_basic(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    _scatter_basic_kern<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_scatter_warp(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    // 256 threads per block = 8 warps; grid = ceil(n_pre / 8)
    int blocks = (n_pre + 7) / 8;
    _scatter_warp_kern<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_scatter_gridstride(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    int blocks = min(1024, (n_pre * n_conn + 255) / 256);
    _scatter_gs_kern<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

// Auto-selects the best scatter kernel based on problem size.
void fcnmv_scatter_auto(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);

    if (n_conn <= 32) {
        // One warp per pre-neuron
        int blocks = (n_pre + 7) / 8;
        _scatter_warp_kern<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else if ((long long)n_pre * n_conn > 262144LL) {
        // Large problem: grid-stride maximises occupancy
        int blocks = min(1024, (n_pre * n_conn + 255) / 256);
        _scatter_gs_kern<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else {
        // Medium problem: one block per pre-neuron
        _scatter_basic_kern<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    }
}
""",
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]

    if transpose:
        # Scatter mode: y[idx[i,k]] += w[i,k] * v[i]
        if n_conn <= 32:
            kernel_name = 'fcnmv.fcnmv_scatter_warp'
        else:
            kernel_name = 'fcnmv.fcnmv_scatter_auto'

        def kernel(weights, indices, vector):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, vector)

    else:
        # Gather mode: y[i] = sum_k w[i,k] * v[idx[i,k]]
        if n_conn % 4 == 0 and n_conn >= 128:
            kernel_name = 'fcnmv.fcnmv_gather_vec4'
        elif n_conn <= 32:
            kernel_name = 'fcnmv.fcnmv_gather_warp'
        else:
            kernel_name = 'fcnmv.fcnmv_gather_auto'

        def kernel(weights, indices, vector):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, vector)

    return kernel


def _fcnmv_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    def kernel(weights, indices, vector):
        out, weights, n_pre, n_post = check_fixed_conn_num_shape(
            weights, indices, vector, shape, transpose, require_scalar_weight=True,
        )
        if transpose:
            masked_weights = jnp.broadcast_to(vector[:, None] * weights, indices.shape)
            return jax.ops.segment_sum(
                masked_weights.ravel(), indices.ravel(), num_segments=n_post
            ),

        else:
            scalar_weight = weights.ndim == 0
            if scalar_weight:
                return jax.vmap(lambda ind: weights * u.math.sum(vector[ind]))(indices),
            else:
                return jax.vmap(lambda w, ind: u.math.sum(w * vector[ind]))(weights, indices),

    return kernel


def _fcnmv_jvp_vector(spk_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return fcnmv_p_call(weights, indices, spk_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _fcnmv_jvp_weights(w_dot, weights, indices, vector, *, shape, transpose, **kwargs):
    return fcnmv_p_call(w_dot, indices, vector, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _fcnmv_transpose_rule(ct, weights, indices, vector, *, shape, transpose, weight_info, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # dL/dspk = dL/dy * dy/dspk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(vector):
        if type(ct) is ad.Zero:
            ct_vector = ad.Zero(vector)
        else:
            ct_vector = fcnmv_p_call(
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
            ct_weight = fcnmv_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                vector,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )[0]
            ct_weight = jnp.inner(ct, ct_weight).reshape(*weight_info.shape)

        else:
            if transpose:
                ct_weight = jax.vmap(lambda v, ind: v * ct[ind])(vector, indices)
            else:
                ct_weight = jax.vmap(lambda c, ind: c * vector[ind])(ct, indices)
        return ct_weight, indices, vector


def _fcnmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = fcnmm_p_call(
            args[0],
            args[1],
            args[2].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend']
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = fcnmm_p_call(
            args[0],
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend']
        )
        return r, [1]
    else:
        return general_batching_rule(fcnmv_p, args, axes, **kwargs)


def _fcnmv_benchmark_data(*, platform):
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
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, vector),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def fcnmv_p_call(
    weights: jax.Array,
    indices: jax.Array,
    vector: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for sparse matrix--vector product with fixed
    connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba, Pallas, or TVM FFI) without performing any
    physical-unit bookkeeping.  It is typically called from :func:`fcnmv`
    or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    vector : jax.Array
        Dense vector to multiply with.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``,
        ``'tvmffi'``, or ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result vector.

    See Also
    --------
    fcnmv : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, vector, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return fcnmv_p(
        weights,
        indices,
        vector,
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=[out],
        backend=backend,
    )


fcnmv_p = XLACustomKernel(
    'fixed_num_mv',
    doc="""
Low-level XLA custom-kernel primitive for ``fcnmv``.

This ``XLACustomKernel`` instance dispatches the fixed-connection matrix-vector
multiplication operation with floating-point weights to registered backends
(``numba``, ``pallas``, ``tvmffi``), using runtime shape/dtype metadata
provided by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. Unlike the binary variant, this operation
uses full floating-point weights and processes all entries (not just spikes).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``fcnmv_p.available_backends(platform)``,
and the default backend can be configured with ``fcnmv_p.set_default(platform, backend)``.

See Also
--------
fcnmv : High-level user-facing function wrapper.
"""
)
fcnmv_p.def_numba_kernel(_fcnmv_numba_kernel)
fcnmv_p.def_pallas_kernel('gpu', _fcnmv_pallas_kernel)
fcnmv_p.def_tvmffi_kernel('gpu', _fcnmv_cuda_kernel)
fcnmv_p.def_kernel('jax_raw', 'cpu', _fcnmv_jax_kernel)
fcnmv_p.def_kernel('jax_raw', 'gpu', _fcnmv_jax_kernel)
fcnmv_p.def_kernel('jax_raw', 'tpu', _fcnmv_jax_kernel)
fcnmv_p.def_jvp_rule2(_fcnmv_jvp_weights, None, _fcnmv_jvp_vector)
fcnmv_p.def_transpose_rule(_fcnmv_transpose_rule)
fcnmv_p.def_batching_rule(_fcnmv_batching)
fcnmv_p.def_call(fcnmv_p_call)
fcnmv_p.def_tags('fcn', 'float')
fcnmv_p.def_benchmark_data(_fcnmv_benchmark_data)


@namescope(static_argnames=['shape', 'transpose'])
def fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse matrix--matrix product with fixed connection number.

    Computes ``Y = W @ M`` (or ``Y = W^T @ M`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``M`` is a dense floating-point matrix.

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
    fcnmv : Float sparse matrix--vector product with fixed connection number.
    binary_fcnmm : Event-driven (binary) variant.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    When ``transpose=False`` (gather mode), each output element is:

        ``Y[i, j] = sum_{k=0}^{n_conn-1} weights[i, k] * M[indices[i, k], j]``

    For homogeneous weights (``weights`` has shape ``(1,)``):

        ``Y[i, j] = w * sum_{k=0}^{n_conn-1} M[indices[i, k], j]``

    When ``transpose=True`` (scatter mode), the computation distributes
    contributions to target rows:

        ``Y[indices[i, k], j] += weights[i, k] * M[i, j]``    for all ``i, k, j``

    The computational cost is ``O(num_pre * n_conn * n)`` where ``n`` is the
    number of columns in ``M``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.float import fcnmm
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)  # homogeneous
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> matrix = jnp.array([[1.0, 0.5],
        ...                     [2.0, 1.0],
        ...                     [3.0, 1.5]])
        >>> y = fcnmm(weights, indices, matrix, shape=(2, 3), transpose=False)
        >>> print(y)
        [[3.  1.5]
         [5.  2.5]]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = fcnmm_p_call(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * m_unit * w_unit)


def _fcnmm_numba_kernel(
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
                for i in range(matrix.shape[0]):
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += weights[i, j] * matrix[i]

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


def _fcnmm_pallas_kernel(
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

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


def _fcnmm_cuda_kernel(
    transpose: bool,
    indices_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    **kwargs
):
    register_tvm_cuda_kernels(
        module='fcnmm',
        functions=[
            'fcnmm_gather_auto', 'fcnmm_gather_basic',
            'fcnmm_gather_shared', 'fcnmm_gather_vec4',
            'fcnmm_scatter_auto', 'fcnmm_scatter_block',
            'fcnmm_scatter_cached', 'fcnmm_scatter_warp',
        ],
        source_code=r"""
#include <cuda_runtime.h>
#include <cstdint>

// =========================================================================
// FCN Matrix-Matrix product CUDA kernels
//
// Gather mode (transpose=False):
//   Y[i, j] = sum_{k=0}^{n_conn-1} w[i,k] * M[indices[i,k], j]
//   indices: [n_pre, n_conn], weights: [1] or [n_pre, n_conn]
//   M: [n_post, n_col],  Y: [n_pre, n_col]
//
// Scatter mode (transpose=True):
//   Y[indices[i,k], j] += w[i,k] * M[i, j]
//   M: [n_pre, n_col],  Y: [n_post, n_col]  (Y pre-zeroed at launch)
//
// IMPORTANT: weights.data_ptr() returns a GPU device pointer.
// NEVER dereference on host. GPU threads read weights[0] (homo)
// or weights[i*n_conn+k] (hetero).
// =========================================================================

// =========================================================================
// Gather kernels (transpose=False)
// =========================================================================

// Basic gather: one thread per output element Y[i,j].
// Grid: (n_pre, ceil(n_col / 64)), Block: (64,)
// Threads in a warp read consecutive j positions of M[same_row, j..j+31]
// giving coalesced loads within each k iteration.
__global__ void _mm_gather_basic_kern(
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
        float w = is_homo ? weights[0] : w_row[k];
        acc += w * matrix[(size_t)idx_row[k] * n_col + j];
    }
    output[(size_t)i * n_col + j] = acc;
}

// Shared-memory gather: tiles the index and weight arrays into shared memory
// to reduce global memory traffic for the connection lists.
// Grid: (n_pre, ceil(n_col / 64)), Block: (64,)
// Shared mem: MMTK * 8 bytes (idx tile + weight tile).
// Best when n_conn is large (> 128) and index/weight bandwidth dominates.
#define MMTK 128
__global__ void _mm_gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ char smem_mm[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_mm);
    float*   s_w   = reinterpret_cast<float*>(smem_mm + MMTK * sizeof(int32_t));

    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;

    float acc = 0.0f;
    for (int k0 = 0; k0 < n_conn; k0 += MMTK) {
        int tile = (k0 + MMTK < n_conn) ? MMTK : (n_conn - k0);
        // Cooperatively load the connection tile into shared memory.
        for (int t = threadIdx.x; t < tile; t += blockDim.x) {
            s_idx[t] = idx_row[k0 + t];
            s_w[t]   = is_homo ? 1.0f : w_row[k0 + t];
        }
        __syncthreads();
        // Accumulate contributions from this tile.
        if (j < n_col) {
            for (int t = 0; t < tile; t++)
                acc += s_w[t] * matrix[(size_t)s_idx[t] * n_col + j];
        }
        __syncthreads();
    }
    if (j < n_col)
        output[(size_t)i * n_col + j] = is_homo ? (weights[0] * acc) : acc;
}

// Vectorised gather: float4 loads for M and output when n_col % 4 == 0.
// Each thread processes 4 consecutive j columns simultaneously => 4x throughput.
// Grid: (n_pre, ceil(n_col/4 / 64)), Block: (64,)
// Best when n_col is divisible by 4 and >= 64 (memory-bandwidth bound).
__global__ void _mm_gather_vec4_kern(
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
        acc.x += w * m.x;
        acc.y += w * m.y;
        acc.z += w * m.z;
        acc.w += w * m.w;
    }
    reinterpret_cast<float4*>(output)[(size_t)i * nc4 + j4] = acc;
}

// =========================================================================
// Scatter kernels (transpose=True)
// Y[indices[i,k], j] += w[i,k] * M[i, j]   (Y pre-zeroed before launch)
// =========================================================================

// Block scatter: one block per pre-neuron, threads stride over j columns.
// For each i, sequentially iterates over n_conn connections and atomically
// accumulates to the target output row. Good for large n_col.
// Grid: (n_pre,), Block: (256,)
__global__ void _mm_scatter_block_kern(
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
        for (int j = threadIdx.x; j < n_col; j += blockDim.x)
            atomicAdd(&out_row[j], w * m_row[j]);
    }
}

// Cached scatter: 2D grid — one block per (pre-neuron, n_col tile).
// Tiles M[i] into shared memory once, then performs all n_conn atomic
// scatter operations from shmem. Eliminates repeated DRAM reads of M[i]
// and is especially efficient for large n_conn with moderate n_col.
// Grid: (n_pre, ceil(n_col / BLOCK_J)), Block: (BLOCK_J,)
// Shared mem: BLOCK_J * sizeof(float) bytes
#define MM_SCATTER_BJ 128
__global__ void _mm_scatter_cached_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [n_pre, n_col]
    float*         __restrict__ output,    // [n_post, n_col] (pre-zeroed)
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ float s_m[];  // cache one M[i] column tile
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    // Load M[i, j] tile into shared memory once.
    s_m[threadIdx.x] = (j < n_col) ? matrix[(size_t)i * n_col + j] : 0.0f;
    __syncthreads();

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;

    if (j < n_col) {
        // All connections from row i scatter to their targets using shmem value.
        float m_val = s_m[threadIdx.x];
        for (int k = 0; k < n_conn; k++) {
            int   tgt = idx_row[k];
            float w   = is_homo ? weights[0] : w_row[k];
            atomicAdd(&output[(size_t)tgt * n_col + j], w * m_val);
        }
    }
}

// Warp scatter: grid-stride over (pre-neuron, connection) pairs.
// Each warp handles one (i, k) pair; lanes stride over j columns.
// Maximises parallelism over both i and k dimensions. Good for small n_col.
// Grid: (min(4096, ceil(n_pre*n_conn/8)),), Block: (256,)  [8 warps per block]
__global__ void _mm_scatter_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // global warp id
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
        for (int j = lane; j < n_col; j += 32)
            atomicAdd(&out_row[j], w * m_row[j]);
    }
}

// =========================================================================
// TVM FFI Entry Points
// =========================================================================
// Convention: args = (weights, indices, matrix, output, stream)
// weights: [1] (homo) or [n_pre, n_conn] (hetero), float32
// indices: [n_pre, n_conn], int32
// Gather:  matrix [n_post, n_col], output [n_pre, n_col]
// Scatter: matrix [n_pre,  n_col], output [n_post, n_col]
//
// weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
// GPU threads read weights[0] (homo) or weights[i*n_conn+k] (hetero).

// --- Gather entry points ---

void fcnmm_gather_basic(
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
    _mm_gather_basic_kern<<<grid, BJ, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_gather_shared(
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
    size_t shm = MMTK * (sizeof(int32_t) + sizeof(float));
    _mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_gather_vec4(
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
    _mm_gather_vec4_kern<<<grid, BJ4, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

// Auto-selects the best gather kernel based on n_conn and n_col.
void fcnmm_gather_auto(
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
    if (n_col % 4 == 0 && n_col >= 64) {
        // Vectorised float4 path: 4x throughput for aligned n_col.
        dim3 grid(n_pre, (n_col / 4 + BJ - 1) / BJ);
        _mm_gather_vec4_kern<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 128) {
        // Shared-memory path: amortises index/weight bandwidth for large n_conn.
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = MMTK * (sizeof(int32_t) + sizeof(float));
        _mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        // Basic path: good general-purpose option.
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        _mm_gather_basic_kern<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// --- Scatter entry points (output zeroed before kernel launch) ---

void fcnmm_scatter_block(
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
    _mm_scatter_block_kern<<<n_pre, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_warp(
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
    _mm_scatter_warp_kern<<<blocks, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_cached(
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
    int BJ = MM_SCATTER_BJ;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    size_t shm = BJ * sizeof(float);
    _mm_scatter_cached_kern<<<grid, BJ, shm, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

// Auto-selects the best scatter kernel based on problem size.
void fcnmm_scatter_auto(
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
        // Small n_col: one warp per (i,k) pair maximises SM utilisation.
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _mm_scatter_warp_kern<<<blocks, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 32) {
        // Moderate-to-large n_conn: cached M[i] in shmem cuts DRAM reads.
        int BJ = MM_SCATTER_BJ;
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = BJ * sizeof(float);
        _mm_scatter_cached_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        // Small n_conn: simple block-per-pre-neuron with thread stride.
        _mm_scatter_block_kern<<<n_pre, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}
""",
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    n_col = matrix_info.shape[1]

    if transpose:
        # Scatter mode: Y[idx[i,k], j] += w[i,k] * M[i, j]
        kernel_name = 'fcnmm.fcnmm_scatter_auto'

        def kernel(weights, indices, matrix):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, matrix)

    else:
        # Gather mode: Y[i, j] = sum_k w[i,k] * M[idx[i,k], j]
        if n_col % 4 == 0 and n_col >= 64:
            kernel_name = 'fcnmm.fcnmm_gather_vec4'
        elif n_conn > 128:
            kernel_name = 'fcnmm.fcnmm_gather_shared'
        else:
            kernel_name = 'fcnmm.fcnmm_gather_auto'

        def kernel(weights, indices, matrix):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, matrix)

    return kernel


def _fcnmm_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    def kernel(weights, indices, matrix):
        out, weights, n_pre, n_post = check_fixed_conn_num_shape(
            weights, indices, matrix, shape, transpose, require_scalar_weight=True,
        )
        if transpose:
            # Scatter mode: Y[n_post, n]
            # Y[indices[i, l], :] += weights[i, l] * matrix[i, :]
            n = matrix.shape[1]
            n_conn = indices.shape[1]
            M_exp = jnp.broadcast_to(matrix[:, None, :], (n_pre, n_conn, n))
            if weights.ndim == 0:
                vals = weights * M_exp
            else:
                vals = weights[:, :, None] * M_exp
            return jax.ops.segment_sum(vals.reshape(-1, n), indices.ravel(), num_segments=n_post),

        else:
            # Gather mode: Y[n_pre, n]
            # Y[i, :] = sum_l weights[i, l] * matrix[indices[i, l], :]
            if weights.ndim == 0:
                return jax.vmap(lambda ind: weights * jnp.sum(matrix[ind], axis=0))(indices),
            else:
                return jax.vmap(lambda w, ind: jnp.sum(w[:, None] * matrix[ind], axis=0))(weights, indices),

    return kernel


def _fcnmm_jvp_matrix(matrix_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return fcnmm_p_call(weights, indices, matrix_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _fcnmm_jvp_weights(weights_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return fcnmm_p_call(weights_dot, indices, matrix, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _fcnmm_transpose_rule(ct, weights, indices, matrix, *, shape, transpose, weight_info, **kwargs):
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
            ct_weight = fcnmm_p_call(
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
    r = fcnmm_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _fcnmm_batching(args, axes, **kwargs):
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
        return general_batching_rule(fcnmm_p, args, axes, **kwargs)


def _fcnmm_benchmark_data(*, platform):
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


def fcnmm_p_call(
    weights: jax.Array,
    indices: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for sparse matrix--matrix product with fixed
    connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba or Pallas) without performing any
    physical-unit bookkeeping.  It is typically called from :func:`fcnmm`
    or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    matrix : jax.Array
        Dense matrix to multiply with, of shape ``(k, n)``.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, or
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
    fcnmm : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return fcnmm_p(
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


fcnmm_p = XLACustomKernel(
    'fixed_num_mm',
    doc="""
Low-level XLA custom-kernel primitive for ``fcnmm``.

This ``XLACustomKernel`` instance dispatches the fixed-connection matrix-matrix
multiplication operation with floating-point weights to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. Unlike the binary variant, this operation
uses full floating-point weights and processes all entries (not just spikes).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``fcnmm_p.available_backends(platform)``,
and the default backend can be configured with ``fcnmm_p.set_default(platform, backend)``.

See Also
--------
fcnmm : High-level user-facing function wrapper.
"""
)
fcnmm_p.def_numba_kernel(_fcnmm_numba_kernel)
fcnmm_p.def_pallas_kernel('gpu', _fcnmm_pallas_kernel)
fcnmm_p.def_tvmffi_kernel('gpu', _fcnmm_cuda_kernel)
fcnmm_p.def_kernel('jax_raw', 'cpu', _fcnmm_jax_kernel)
fcnmm_p.def_kernel('jax_raw', 'gpu', _fcnmm_jax_kernel)
fcnmm_p.def_kernel('jax_raw', 'tpu', _fcnmm_jax_kernel)
fcnmm_p.def_jvp_rule2(_fcnmm_jvp_weights, None, _fcnmm_jvp_matrix)
fcnmm_p.def_transpose_rule(_fcnmm_transpose_rule)
fcnmm_p.def_batching_rule(_fcnmm_batching)
fcnmm_p.def_call(fcnmm_p_call)
fcnmm_p.def_tags('fcn', 'float')
fcnmm_p.def_benchmark_data(_fcnmm_benchmark_data)
