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
    'binary_fcnmv',
    'binary_fcnmv_p',
    'binary_fcnmm',
    'binary_fcnmm_p',
]


@namescope(static_argnames=['shape', 'transpose'])
def binary_fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    spikes: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven sparse matrix--vector product with fixed connection number.

    Computes ``y = W @ s`` (or ``y = W^T @ s`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``s`` is a binary spike vector.  Only the connections to
    spiking neurons contribute to the result, making this operation
    event-driven.

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
        Binary spike vector.  Entries are treated as active when ``True``
        (boolean) or ``> 0`` (float).
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool, optional
        If ``False`` (default), compute ``W @ s`` (fixed post-synaptic
        connections).  If ``True``, compute ``W^T @ s`` (fixed
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
    binary_fcnmm : Event-driven sparse matrix--matrix product.
    fcnmv : Float (non-event-driven) sparse matrix--vector product.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    The event-driven matrix-vector product computes (when ``transpose=False``):

        ``y[i] = sum_{k=0}^{n_conn-1} weights[i, k] * 1_{spikes[indices[i, k]] active}``

    where "active" means ``True`` for boolean spikes or ``> 0`` for float spikes.
    For homogeneous weights (``weights`` has shape ``(1,)``):

        ``y[i] = w * sum_{k=0}^{n_conn-1} 1_{spikes[indices[i, k]] active}``

    When ``transpose=True``, the scatter-mode product computes:

        ``y[indices[i, k]] += weights[i, k] * 1_{spikes[i] active}``    for all ``i, k``

    The event-driven formulation skips inactive neurons entirely, making the
    computation efficient when the spike vector is sparse.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.binary import binary_fcnmv
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)  # homogeneous
        >>> indices = jnp.array([[0, 1], [1, 2]])      # (2, 2)
        >>> spikes = jnp.array([True, False, True])
        >>> y = binary_fcnmv(weights, indices, spikes, shape=(2, 3))
        >>> print(y)
        [1. 1.]
    """
    weights, w_unit = u.split_mantissa_unit(weights)
    spikes, v_unit = u.split_mantissa_unit(spikes)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = binary_fcnmv_p_call(
        weights,
        indices,
        spikes,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * v_unit * w_unit)


def _binary_fcnmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i] > 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w
        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i] > 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]

    else:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    w = weights[0]
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spikes[index]:
                                r += w
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    spk_bool = spikes > 0.
                    w = weights[0]
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spk_bool[index]:
                                r += w
                        posts[i] = r
        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spikes[index]:
                                r += weights[i, j]
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    spk_bool = spikes > 0.
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spk_bool[index]:
                                r += weights[i, j]
                        posts[i] = r

    def kernel(weights, indices, spikes):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, spikes)

    return kernel


def _binary_fcnmv_pallas_kernel(
    transpose: int,
    shape: Tuple[int, int],
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    if len(shape) > 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    block_dim = generate_block_dim(indices_info.shape[1], maximum=256)

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
            vector_is_bool = vector_ref.dtype == jnp.bool_

            @pl.when(vector if vector_is_bool else vector > 0.)
            def run():
                if homo:
                    wv = weight_ref[0]
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
                        data = jnp.where(mask, data, 0.0)
                    atomic_add(out_ref, ind, data, mask=mask)

                jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(_raw_kernel, grid=(n_pre,), input_output_aliases={3: 0},
                                out_shape=kwargs['outs'], backend='triton')
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
                if vector_ref.dtype == jnp.bool_:
                    vec = jnp.where(mask, vec, False)
                else:
                    vec = jnp.where(mask, vec, 0.0)
                if homo:
                    if vector_ref.dtype == jnp.bool_:
                        return out + jnp.sum(jnp.asarray(vec, dtype=out_ref.dtype))
                    else:
                        return out + jnp.sum((vec > 0.).astype(out_ref.dtype))
                else:
                    weight = weight_ref[i_row, pl.dslice(i_col, block_dim)]
                    weight = jnp.where(mask, weight, 0.0)
                    if vector_ref.dtype == jnp.bool_:
                        weight = jnp.where(vec, weight, 0.)
                    else:
                        weight = jnp.where(vec > 0., weight, 0.)
                    return out + jnp.sum(weight)

            i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
            if homo:
                i_row_sum = i_row_sum * weight_ref[0]
            out_ref[i_row] = i_row_sum

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(_raw_kernel, grid=(n_pre,), out_shape=kwargs['outs'], backend='triton')
            return fn(weights, indices, vector)

    return kernel


def _binary_fcnmv_cuda_kernel(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    """
    CUDA TVM FFI kernel generator for ``binary_fcnmv``.

    Implements the event-driven sparse matrix--vector product on GPU via
    NVRTC-compiled CUDA kernels registered through the TVM FFI infrastructure.

    Kernel variants
    ---------------
    **Gather mode** (``transpose=False``):
    y[i] = sum_k weights[i,k] * is_active(spikes[indices[i,k]])

    - ``_bool_warp``  : bool spikes, one warp (32 threads) per row — best for n_conn ≤ 32
    - ``_bool_basic`` : bool spikes, one block (256 threads) per row + block reduction — best for n_conn > 32
    - ``_float_warp`` : float spikes, one warp per row — best for n_conn ≤ 32
    - ``_float_basic``: float spikes, one block per row + block reduction — best for n_conn > 32

    **Scatter mode** (``transpose=True``):
    if is_active(spikes[i]):  output[indices[i,k]] += weights[i,k]  for all k

    - ``_bool_warp``  : bool spikes, 8-warps-per-block layout — best for n_conn ≤ 32
    - ``_bool_basic`` : bool spikes, one block per pre-neuron — best for n_conn > 32
    - ``_float_warp`` : float spikes, 8-warps-per-block layout — best for n_conn ≤ 32
    - ``_float_basic``: float spikes, one block per pre-neuron — best for n_conn > 32

    Optimization notes
    ------------------
    - Gather kernels use **branchless** computation (multiply by 0/1) to avoid
      warp divergence from irregular spike patterns.
    - Scatter kernels use **early block exit** (return/continue if spike inactive)
      to skip all-inactive pre-neurons, which is highly effective at the typical
      1–5 % firing rates found in spiking neural networks.
    - Shared memory is used for block reduction in ``_basic`` gather variants
      (``extern __shared__``, never static in ``__device__`` functions per
      NVRTC constraints).
    - Host C++ entry functions only read metadata (``ndim()``, ``size()``).
      ``data_ptr()`` is passed unchanged to device kernels and never
      dereferenced on the host.
    """
    register_tvm_cuda_kernels(
        module='binary_fcnmv',
        functions=[
            'binary_fcnmv_gather_bool_warp',
            'binary_fcnmv_gather_bool_basic',
            'binary_fcnmv_gather_float_warp',
            'binary_fcnmv_gather_float_basic',
            'binary_fcnmv_scatter_bool_warp',
            'binary_fcnmv_scatter_bool_basic',
            'binary_fcnmv_scatter_float_warp',
            'binary_fcnmv_scatter_float_basic',
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
// GATHER kernels  (transpose=False)
//
// y[i] = sum_{k=0}^{n_conn-1} weights[i,k] * is_active(spikes[indices[i,k]])
//
//   is_active(s) for bool spikes (uint8):  s != 0
//   is_active(s) for float spikes:         s > 0.0f
//
// For homogeneous weights (is_homo=1):
//   y[i] = weights[0] * count_k( is_active(spikes[indices[i,k]]) )
//
// Branchless formulation: val += (float)(is_active(s)) [* weight]
// avoids warp divergence from irregular spike patterns.
// =========================================================================

// ---- Gather / Bool spikes ----

// One warp (32 threads) per output row.
// Best when n_conn <= 32; threads stride with step 32 for larger n_conn.
__global__ void _bg_bool_warp_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const uint8_t* __restrict__ spikes,    // [n_post], bool stored as uint8
    float*         __restrict__ output,    // [n_pre]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += 32) {
        float s = (float)(spikes[i_row[k]] != 0);
        val += is_homo ? s : (s * w_row[k]);
    }
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block (256 threads) per output row with inline block reduction.
// Uses 32*sizeof(float) of dynamic shared memory for the reduction scratchpad.
// Best when n_conn > 32.
__global__ void _bg_bool_basic_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const uint8_t* __restrict__ spikes,    // [n_post]
    float*         __restrict__ output,    // [n_pre]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];   // 32 floats for block reduction
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {
        float s = (float)(spikes[i_row[k]] != 0);
        val += is_homo ? s : (s * w_row[k]);
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

// ---- Gather / Float spikes ----

// One warp (32 threads) per output row.
// Best when n_conn <= 32.
__global__ void _bg_float_warp_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ spikes,    // [n_post], float spikes
    float*         __restrict__ output,    // [n_pre]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += 32) {
        float s = (spikes[i_row[k]] > 0.0f) ? 1.0f : 0.0f;
        val += is_homo ? s : (s * w_row[k]);
    }
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block (256 threads) per output row with inline block reduction.
// Best when n_conn > 32.
__global__ void _bg_float_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ spikes,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {
        float s = (spikes[i_row[k]] > 0.0f) ? 1.0f : 0.0f;
        val += is_homo ? s : (s * w_row[k]);
    }
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
// SCATTER kernels  (transpose=True)
//
// For each active row i (is_active(spikes[i]) == true):
//   output[indices[i, k]] += weights[i, k]   for k = 0 .. n_conn-1
//
// Output buffer must be pre-zeroed; this is done via cudaMemsetAsync
// in the TVM FFI entry functions below.
//
// Key optimisation: early exit (return / continue) when a pre-neuron is
// inactive.  With typical SNN firing rates of 1-5 %, this skips 95-99 %
// of all blocks/warps entirely.
// =========================================================================

// ---- Scatter / Bool spikes ----

// 8 warps per block (256 threads), one warp per pre-neuron.
// Grid = ceil(n_pre / 8) blocks.
// All 32 threads in a warp handle the same row — no intra-warp divergence
// on the "skip if inactive" check.
// Best when n_conn <= 32.
__global__ void _bs_bool_warp_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const uint8_t* __restrict__ spikes,    // [n_pre]
    float*         __restrict__ output,    // [n_post]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        if (!spikes[row]) continue;   // all 32 threads in warp take same branch
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        float w0 = is_homo ? weights[0] : 0.0f;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
    }
}

// One block (256 threads) per pre-neuron.  Entire block exits early
// if the neuron is inactive.
// Best when n_conn > 32.
__global__ void _bs_bool_basic_kern(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (!spikes[row]) return;   // skip entire block if neuron is inactive
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? weights[0] : 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
}

// ---- Scatter / Float spikes ----

// 8 warps per block, one warp per pre-neuron.  Best when n_conn <= 32.
__global__ void _bs_float_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ spikes,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        if (!(spikes[row] > 0.0f)) continue;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        float w0 = is_homo ? weights[0] : 0.0f;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
    }
}

// One block per pre-neuron.  Best when n_conn > 32.
__global__ void _bs_float_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ spikes,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (!(spikes[row] > 0.0f)) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? weights[0] : 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
}

// =========================================================================
// TVM FFI Entry Points
//
// Convention: args = (weights, indices, spikes, output, stream)
//   weights : float32, shape (1,) for homo or (n_pre, n_conn) for hetero
//   indices : int32,   shape (n_pre, n_conn)
//   spikes  : gather → (n_post,);  scatter → (n_pre,)
//             bool variant   → uint8 pointer
//             float variant  → float32 pointer
//   output  : gather → (n_pre,) float32, written directly (no pre-zero)
//             scatter → (n_post,) float32, zeroed here via cudaMemsetAsync
//
// IMPORTANT: data_ptr() returns a GPU device memory pointer.
// NEVER dereference it in host C++ code (causes SIGSEGV).
// Pass it unchanged to device kernels; GPU threads read from it.
// Only ndim() and size() are host-safe metadata reads.
// =========================================================================

// ---- Gather / Bool ----

void binary_fcnmv_gather_bool_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;   // host-safe metadata
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());   // device ptr
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    _bg_bool_warp_kern<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_bool_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _bg_bool_basic_kern<<<n_pre, 256, shm, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

// ---- Gather / Float ----

void binary_fcnmv_gather_float_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_spk = static_cast<const float*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    _bg_float_warp_kern<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_float_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_spk = static_cast<const float*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _bg_float_basic_kern<<<n_pre, 256, shm, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

// ---- Scatter / Bool (output pre-zeroed via cudaMemsetAsync) ----

void binary_fcnmv_scatter_bool_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
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
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    int blocks = (n_pre + 7) / 8;
    _bs_bool_warp_kern<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_bool_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
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
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    _bs_bool_basic_kern<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

// ---- Scatter / Float ----

void binary_fcnmv_scatter_float_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
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
    const float*   d_spk = static_cast<const float*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    int blocks = (n_pre + 7) / 8;
    _bs_float_warp_kern<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_float_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
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
    const float*   d_spk = static_cast<const float*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    _bs_float_basic_kern<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}
""",
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)

    if transpose:
        # Scatter mode: if is_active(spikes[i]) → output[indices[i,k]] += weights[i,k]
        if is_bool_spike:
            kernel_name = (
                'binary_fcnmv.binary_fcnmv_scatter_bool_warp'
                if n_conn <= 32
                else 'binary_fcnmv.binary_fcnmv_scatter_bool_basic'
            )
        else:
            kernel_name = (
                'binary_fcnmv.binary_fcnmv_scatter_float_warp'
                if n_conn <= 32
                else 'binary_fcnmv.binary_fcnmv_scatter_float_basic'
            )
    else:
        # Gather mode: y[i] = sum_k weights[i,k] * is_active(spikes[indices[i,k]])
        if is_bool_spike:
            kernel_name = (
                'binary_fcnmv.binary_fcnmv_gather_bool_warp'
                if n_conn <= 32
                else 'binary_fcnmv.binary_fcnmv_gather_bool_basic'
            )
        else:
            kernel_name = (
                'binary_fcnmv.binary_fcnmv_gather_float_warp'
                if n_conn <= 32
                else 'binary_fcnmv.binary_fcnmv_gather_float_basic'
            )

    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel


def _binary_fcnmv_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    """Pure JAX reference implementation for benchmarking comparison."""
    n_pre, n_post = shape

    def kernel(weights, indices, spikes):
        # Convert spikes to float: bool→{0,1}, float→{0,1} based on >0
        if spikes.dtype == jnp.bool_:
            spk_f = spikes.astype(weights.dtype)
        else:
            spk_f = (spikes > 0).astype(weights.dtype)

        if transpose:
            # Scatter: y[indices[i,k]] += weights[i,k] * spk_f[i]
            masked = jnp.broadcast_to(spk_f[:, None] * weights, indices.shape)
            return jax.ops.segment_sum(
                masked.ravel(), indices.ravel(), num_segments=n_post
            ),
        else:
            # Gather: y[i] = sum_k weights[i,k] * spk_f[indices[i,k]]
            if weights.size == 1:
                w = weights[0]
                return jax.vmap(lambda ind: w * jnp.sum(spk_f[ind]))(indices),
            else:
                return jax.vmap(lambda w, ind: jnp.sum(w * spk_f[ind]))(weights, indices),

    return kernel


def _binary_fcnmv_jvp_spikes(spk_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return fcnmv_p_call(weights, indices, spk_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _binary_fcnmv_jvp_weights(w_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return binary_fcnmv_p_call(w_dot, indices, spikes, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _binary_fcnmv_transpose_rule(ct, weights, indices, spikes, *, shape, transpose, weight_info, **kwargs):
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
                weights, indices, ct, shape=shape, transpose=not transpose, backend=kwargs['backend'],
            )[0]
        return weights, indices, ct_spk

    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_gmax = ad.Zero(weights)
        elif homo:
            # scalar
            ct_gmax = binary_fcnmv_p_call(
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                spikes,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )
            ct_gmax = jnp.inner(ct, ct_gmax[0]).reshape(*weight_info.shape)
        else:
            if transpose:
                ct_gmax = jax.vmap(lambda v, ind: v * ct[ind])(spikes, indices)
            else:
                ct_gmax = jax.vmap(lambda c, ind: c * spikes[ind])(ct, indices)
        return ct_gmax, indices, spikes


def _binary_fcnmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_fcnmm_p_call(
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
        r = binary_fcnmm_p_call(
            args[0],
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(binary_fcnmv_p, args, axes, **kwargs)


def _binary_fcnmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                n_conn = max(1, int(n_post * prob))
                indices = jnp.asarray(np.random.randint(0, n_post, (n_pre, n_conn), dtype=np.int32))
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.ones((n_pre, n_conn), dtype=dtype)
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    spikes = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    spikes = jnp.asarray(np.random.rand(v_size), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                yield BenchmarkConfig(
                    name,
                    (weights, indices, spikes),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )


def binary_fcnmv_p_call(
    weights: jax.Array,
    indices: jax.Array,
    spikes: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for event-driven sparse matrix--vector product
    with fixed connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba, Pallas, or TVM FFI) without performing any
    physical-unit bookkeeping.  It is typically called from
    :func:`binary_fcnmv` or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    spikes : jax.Array
        Binary spike vector (boolean or float).
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
    binary_fcnmv : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, spikes, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return binary_fcnmv_p(
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


binary_fcnmv_p = XLACustomKernel(
    'binary_fcnmv',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_fcnmv``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven)
fixed-connection matrix-vector multiplication operation to registered backends
(``numba``, ``pallas``, ``tvmffi``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. The event-driven formulation only processes
active (spiking) neurons, skipping zero entries for efficiency.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_fcnmv_p.available_backends(platform)``,
and the default backend can be configured with ``binary_fcnmv_p.set_default(platform, backend)``.

See Also
--------
binary_fcnmv : High-level user-facing function wrapper.
"""
)
binary_fcnmv_p.def_numba_kernel(_binary_fcnmv_numba_kernel)
binary_fcnmv_p.def_pallas_kernel('gpu', _binary_fcnmv_pallas_kernel)
binary_fcnmv_p.def_tvmffi_kernel('gpu', _binary_fcnmv_cuda_kernel)
binary_fcnmv_p.def_kernel('jax_raw', 'cpu', _binary_fcnmv_jax_kernel)
binary_fcnmv_p.def_kernel('jax_raw', 'gpu', _binary_fcnmv_jax_kernel)
binary_fcnmv_p.def_jvp_rule2(_binary_fcnmv_jvp_weights, None, _binary_fcnmv_jvp_spikes, None)
binary_fcnmv_p.def_transpose_rule(_binary_fcnmv_transpose_rule)
binary_fcnmv_p.def_batching_rule(_binary_fcnmv_batching)
binary_fcnmv_p.def_call(binary_fcnmv_p_call)
binary_fcnmv_p.def_tags('fcn', 'binary')
binary_fcnmv_p.def_benchmark_data(_binary_fcnmv_benchmark_data)


@namescope(static_argnames=['shape', 'transpose'])
def binary_fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven sparse matrix--matrix product with fixed connection number.

    Computes ``Y = W @ M`` (or ``Y = W^T @ M`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``M`` is a dense binary event matrix.  Only the connections
    to active (spiking) entries contribute to the result.

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
        Dense binary event matrix of shape ``(k, n)`` where ``k`` matches
        the appropriate sparse-matrix dimension.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool
        If ``False``, compute ``W @ M`` (fixed post-synaptic connections).
        If ``True``, compute ``W^T @ M`` (fixed pre-synaptic connections,
        scatter mode).
    backend : str or None, optional
        Execution backend override.

    Returns
    -------
    jax.Array or u.Quantity
        Result matrix of shape ``(num_pre, n)`` when ``transpose=False``
        or ``(num_post, n)`` when ``transpose=True``.

    See Also
    --------
    binary_fcnmv : Event-driven sparse matrix--vector product.
    fcnmm : Float (non-event-driven) sparse matrix--matrix product.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    The event-driven matrix-matrix product applies column by column.  When
    ``transpose=False`` (gather mode), for each output element:

        ``Y[i, j] = sum_{k=0}^{n_conn-1} weights[i, k] * 1_{M[indices[i, k], j] active}``

    where "active" means ``True`` for boolean entries or ``> 0`` for float
    entries of ``M``.  For homogeneous weights (``weights`` has shape ``(1,)``):

        ``Y[i, j] = w * sum_{k=0}^{n_conn-1} 1_{M[indices[i, k], j] active}``

    When ``transpose=True`` (scatter mode), the computation distributes
    active entries to their target rows:

        ``Y[indices[i, k], j] += weights[i, k] * 1_{M[i, j] active}``    for all ``i, k, j``

    The event-driven formulation skips inactive entries of ``M``, making the
    computation efficient when the event matrix is sparse.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.binary import binary_fcnmm
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> matrix = jnp.array([[True, False],
        ...                     [False, True],
        ...                     [True, True]])
        >>> y = binary_fcnmm(weights, indices, matrix, shape=(2, 3), transpose=False)
        >>> print(y)
        [[1. 1.]
         [1. 2.]]
    """
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = binary_fcnmm_p_call(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * m_unit * w_unit)


def _binary_fcnmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
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
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i_k in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i_k])
                        for i_conn in range(indices.shape[1]):
                            posts[indices[i_k, i_conn], nonzero] += w
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i_k in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i_k] > 0.)
                        for i_conn in range(indices.shape[1]):
                            posts[indices[i_k, i_conn], nonzero] += w
        else:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    for i in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i])
                        for j in range(indices.shape[1]):
                            posts[indices[i, j], nonzero] += weights[i, j]
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    for i in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i] > 0.)
                        for j in range(indices.shape[1]):
                            posts[indices[i, j], nonzero] += weights[i, j]

    else:
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    w = weights[0]
                    for i_m in numba.prange(indices.shape[0]):
                        posts[i_m] = w * np.sum(matrix[indices[i_m]], axis=0)
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    w = weights[0]
                    for i_m in numba.prange(indices.shape[0]):
                        events = matrix[indices[i_m]] > 0.
                        posts[i_m] = w * np.sum(events, axis=0)
        else:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    for i_m in numba.prange(indices.shape[0]):
                        posts[i_m] = weights[i_m] @ (matrix[indices[i_m]]).astype(weights.dtype)
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    for i_m in numba.prange(indices.shape[0]):
                        events = (matrix[indices[i_m]] > 0.).astype(weights.dtype)
                        posts[i_m] = weights[i_m] @ events

    def kernel(weights, indices, matrix):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, matrix)

    return kernel


def _binary_fcnmm_pallas_kernel(
    shape: MatrixShape,
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    atomic_add = getattr(pl, "atomic_add", None)
    if atomic_add is None:
        from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    if len(shape) > 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    block_k = generate_block_dim(indices_info.shape[1], maximum=128)
    block_n = generate_block_dim(matrix_info.shape[1], maximum=128)

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
            out_ref,  # [n_pre, n]
        ):
            i_k = pl.program_id(0)
            i_n_block = pl.program_id(1)
            i_n_start = i_n_block * block_n
            i_n_mask = i_n_start + jnp.arange(block_n) < matrix_ref.shape[1]
            if homo:
                weight = jnp.full(block_k, weight_ref[0])

            def loop_fn(i_index_block, _):
                i_index_start = i_index_block * block_k
                i_index_mask = i_index_start + jnp.arange(block_k) < n_conn
                ind = index_ref[i_k, pl.dslice(i_index_start, block_k)]
                ind = jnp.where(i_index_mask, ind, 0)
                mat = matrix_ref[i_k, pl.dslice(i_n_start, block_n)]
                mat = jnp.where(i_n_mask, mat, 0.0)
                if matrix_ref.dtype != jnp.bool_:
                    mat = (mat > 0.).astype(weight_ref.dtype)
                else:
                    mat = jnp.asarray(mat, dtype=weight_ref.dtype)
                if homo:
                    A = weight
                else:
                    A = weight_ref[i_k, pl.dslice(i_index_start, block_k)]
                    A = jnp.where(i_index_mask, A, 0.0)
                data = A[:, None] * mat[None, :]
                atomic_add(out_ref, (ind, pl.dslice(i_n_start, block_n)), data,
                           mask=i_index_mask[:, None] & i_n_mask[None, :])

            jax.lax.fori_loop(0, pl.cdiv(n_conn, block_k), loop_fn, None)

        def kernel(weights, indices, matrix):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre, pl.cdiv(matrix_info.shape[1], block_n)),
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
            _,
            out_ref,  # [n_pre, n]
        ):
            i_m = pl.program_id(0)
            i_n_block = pl.program_id(1)
            i_n_start = i_n_block * block_n
            i_n_mask = i_n_start + jnp.arange(block_n) < matrix_ref.shape[1]

            def loop_fn(i_k_block, out):
                i_k_start = i_k_block * block_k
                i_k_mask = i_k_start + jnp.arange(block_k) < n_conn
                ind = index_ref[i_m, pl.dslice(i_k_start, block_k)]
                ind = jnp.where(i_k_mask, ind, 0)
                mat = matrix_ref[ind, pl.dslice(i_n_start, block_n)]
                mat = jnp.where(i_k_mask[:, None] & i_n_mask[None, :], mat, 0.0)
                if matrix_ref.dtype != jnp.bool_:
                    mat = (mat > 0.).astype(weight_ref.dtype)
                else:
                    mat = jnp.asarray(mat, dtype=weight_ref.dtype)
                if homo:
                    inc = mat.sum(axis=0)
                else:
                    weight = weight_ref[i_m, pl.dslice(i_k_start, block_k)]
                    weight = jnp.where(i_k_mask, weight, 0.0)
                    inc = (weight[:, None] * mat).sum(axis=0)
                return out + inc

            final_out = jax.lax.fori_loop(
                0,
                pl.cdiv(n_conn, block_k),
                loop_fn,
                jnp.zeros(block_n, dtype=weight_ref.dtype)
            )
            if homo:
                final_out = final_out * weight_ref[0]
            atomic_add(out_ref, (i_m, pl.dslice(i_n_start, block_n)), final_out, mask=i_n_mask)

        def kernel(weights, indices, matrix):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre, pl.cdiv(matrix_info.shape[1], block_n)),
                input_output_aliases={3: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            out_info = kwargs['outs'][0]
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)
            return fn(weights, indices, matrix, placeholder)

    return kernel


def _binary_fcnmm_jvp_matrix(matrix_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return fcnmm_p_call(weights, indices, matrix_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _binary_fcnmm_jvp_weights(weights_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return binary_fcnmm_p_call(
        weights_dot, indices, matrix, shape=shape, transpose=transpose, backend=kwargs['backend']
    )


def _binary_fcnmm_transpose_rule(ct, weights, indices, matrix, *, shape, transpose, weight_info, **kwargs):
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
                backend=kwargs['backend']
            )[0]

        return weights, indices, ct_vector
    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_weight = ad.Zero(weights)

        elif homo:
            ct_weight = binary_fcnmm_p_call(
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
    r = binary_fcnmm_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _binary_fcnmm_batching(args, axes, **kwargs):
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
        return general_batching_rule(binary_fcnmm_p, args, axes, **kwargs)


def _binary_fcnmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                n_conn = max(1, int(n_post * prob))
                indices = jnp.asarray(np.random.randint(0, n_post, (n_pre, n_conn), dtype=np.int32))
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.ones((n_pre, n_conn), dtype=dtype)
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    matrix = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    matrix = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, indices, matrix),
                        {'shape': (n_pre, n_post), 'transpose': transpose}
                    )
                )
    return configs


def binary_fcnmm_p_call(
    weights: jax.Array,
    indices: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for event-driven sparse matrix--matrix product
    with fixed connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba or Pallas) without performing any physical-unit
    bookkeeping.  It is typically called from :func:`binary_fcnmm` or from
    autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    matrix : jax.Array
        Dense binary event matrix of shape ``(k, n)``.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, or ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result matrix.

    See Also
    --------
    binary_fcnmm : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return binary_fcnmm_p(
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


binary_fcnmm_p = XLACustomKernel(
    'binary_fcnmm',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_fcnmm``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven)
fixed-connection matrix-matrix multiplication operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. The event-driven formulation only processes
active (spiking) entries, skipping zero entries for efficiency.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_fcnmm_p.available_backends(platform)``,
and the default backend can be configured with ``binary_fcnmm_p.set_default(platform, backend)``.

See Also
--------
binary_fcnmm : High-level user-facing function wrapper.
"""
)
binary_fcnmm_p.def_numba_kernel(_binary_fcnmm_numba_kernel)
binary_fcnmm_p.def_pallas_kernel('gpu', _binary_fcnmm_pallas_kernel)
binary_fcnmm_p.def_jvp_rule2(_binary_fcnmm_jvp_weights, None, _binary_fcnmm_jvp_matrix, None)
binary_fcnmm_p.def_transpose_rule(_binary_fcnmm_transpose_rule)
binary_fcnmm_p.def_batching_rule(_binary_fcnmm_batching)
binary_fcnmm_p.def_call(binary_fcnmm_p_call)
binary_fcnmm_p.def_tags('fcn', 'binary')
binary_fcnmm_p.def_benchmark_data(_binary_fcnmm_benchmark_data)
