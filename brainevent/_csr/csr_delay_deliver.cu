// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

/*
 * csr_delay_deliver.cu
 * ====================
 *
 * Fused per-edge-delay spike delivery into a post-side arrival ring (#203).
 *
 * Operation (in place on the aliased ring):
 *   for every row i with x[i] != 0, for every edge e in row i:
 *     ring[(t + delays[e]) mod L, indices[e]] += w_e * f(x[i])
 *   f = 1 (binary modes) or x[i] (weighted mode); w_e = weights[0] (homo)
 *   or weights[e] (hetero). mod is the floor modulo; t and d are reduced
 *   separately so t + d cannot overflow.
 *
 * Parallelism: thread-, warp- or block-per-row chosen from avg_nnz =
 * nse / n_pre (< 32, < 256, otherwise). Inactive rows exit early (warp
 * ballot in the thread variant). Collisions are resolved with atomics.
 *
 * Entry points:
 *   csr_delay_deliver_{wt}_{ev}
 *     wt in {f32, f64, f16, bf16}
 *     ev in {bool (int8 events), float (float32 events),
 *            weighted (events of the weight dtype)}
 *   Homogeneous / heterogeneous weights and int32 / int64 indptr are
 *   dispatched at runtime.
 *
 * Operands: ring (L, n_post), events (n_pre,), weights (1,) or (nse,),
 *   indices (nse,) int32, indptr (n_pre + 1,) int32/int64, delays (nse,)
 *   int32, t (1,) int32; output: ring (aliased to operand 0).
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// -------------------------------------------------------------------------
// Type helpers
// -------------------------------------------------------------------------

template <typename WT> struct DelayW;
template <> struct DelayW<float> {
    using acc = float;
    static __device__ __forceinline__ float read(float x) { return x; }
    static __device__ __forceinline__ void add(float* p, float v) { atomic_add_f32(p, v); }
};
template <> struct DelayW<double> {
    using acc = double;
    static __device__ __forceinline__ double read(double x) { return x; }
    static __device__ __forceinline__ void add(double* p, double v) { atomic_add_f64(p, v); }
};
template <> struct DelayW<__half> {
    using acc = float;
    static __device__ __forceinline__ float read(__half x) { return __half2float(x); }
    static __device__ __forceinline__ void add(__half* p, float v) { atomic_add_f16(p, v); }
};
template <> struct DelayW<__nv_bfloat16> {
    using acc = float;
    static __device__ __forceinline__ float read(__nv_bfloat16 x) { return __bfloat162float(x); }
    static __device__ __forceinline__ void add(__nv_bfloat16* p, float v) { atomic_add_bf16(p, v); }
};

__device__ __forceinline__ float delay_ev_value(int8_t x) { return static_cast<float>(x); }
__device__ __forceinline__ float delay_ev_value(float x) { return x; }
__device__ __forceinline__ double delay_ev_value(double x) { return x; }
__device__ __forceinline__ float delay_ev_value(__half x) { return __half2float(x); }
__device__ __forceinline__ float delay_ev_value(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ int delay_floor_mod(int a, int n) {
    int r = a % n;
    return (r < 0) ? r + n : r;
}

// Arrival slot of an edge with delay d, given t_mod = t mod L in [0, L).
__device__ __forceinline__ int delay_slot(int t_mod, int d, int L) {
    if ((unsigned)d >= (unsigned)L) d = delay_floor_mod(d, L);
    int s = t_mod + d;
    return (s >= L) ? s - L : s;
}

// Deliver edges [start + offset, end) with the given stride for one row.
template <typename IndptrT, bool HOMO, typename WT>
__device__ __forceinline__ void delay_deliver_row(
    WT* __restrict__ ring,
    const WT* __restrict__ weights,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ delays,
    IndptrT start, IndptrT end, int offset, int stride,
    typename DelayW<WT>::acc factor,
    int t_mod, int L, int n_post
) {
    using ACC = typename DelayW<WT>::acc;
    ACC w_homo = HOMO ? DelayW<WT>::read(weights[0]) * factor : ACC(0);
    for (IndptrT pos = start + offset; pos < end; pos += stride) {
        int col = __ldg(&indices[pos]);
        int s = delay_slot(t_mod, __ldg(&delays[pos]), L);
        ACC v = HOMO ? w_homo : DelayW<WT>::read(weights[pos]) * factor;
        DelayW<WT>::add(&ring[static_cast<int64_t>(s) * n_post + col], v);
    }
}

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------

template <typename IndptrT, bool HOMO, bool WEIGHTED, typename EVT, typename WT>
__global__ void __launch_bounds__(256) _delay_thread_kern(
    WT* __restrict__ ring, const EVT* __restrict__ events, const WT* __restrict__ weights,
    const int32_t* __restrict__ indices, const IndptrT* __restrict__ indptr,
    const int32_t* __restrict__ delays, const int32_t* __restrict__ t_ptr,
    int n_pre, int L, int n_post
) {
    using ACC = typename DelayW<WT>::acc;
    int row = (int)(blockIdx.x * (uint32_t)blockDim.x) + threadIdx.x;
    ACC xv = (row < n_pre) ? static_cast<ACC>(delay_ev_value(events[row])) : ACC(0);
    bool active = (row < n_pre) && (xv != ACC(0));
    if (__ballot_sync(0xFFFFFFFF, active) == 0) return;
    if (!active) return;
    int t_mod = delay_floor_mod(__ldg(t_ptr), L);
    delay_deliver_row<IndptrT, HOMO, WT>(
        ring, weights, indices, delays, __ldg(&indptr[row]), __ldg(&indptr[row + 1]), 0, 1,
        WEIGHTED ? xv : ACC(1), t_mod, L, n_post);
}

template <typename IndptrT, bool HOMO, bool WEIGHTED, typename EVT, typename WT>
__global__ void __launch_bounds__(256) _delay_warp_kern(
    WT* __restrict__ ring, const EVT* __restrict__ events, const WT* __restrict__ weights,
    const int32_t* __restrict__ indices, const IndptrT* __restrict__ indptr,
    const int32_t* __restrict__ delays, const int32_t* __restrict__ t_ptr,
    int n_pre, int L, int n_post
) {
    using ACC = typename DelayW<WT>::acc;
    int row = (int)(blockIdx.x * (blockDim.x / 32u)) + (int)(threadIdx.x / 32u);
    int lane = (int)(threadIdx.x & 31u);
    if (row >= n_pre) return;
    ACC xv = static_cast<ACC>(delay_ev_value(events[row]));
    if (!(xv != ACC(0))) return;
    int t_mod = delay_floor_mod(__ldg(t_ptr), L);
    delay_deliver_row<IndptrT, HOMO, WT>(
        ring, weights, indices, delays, __ldg(&indptr[row]), __ldg(&indptr[row + 1]), lane, 32,
        WEIGHTED ? xv : ACC(1), t_mod, L, n_post);
}

template <typename IndptrT, bool HOMO, bool WEIGHTED, typename EVT, typename WT>
__global__ void __launch_bounds__(256) _delay_block_kern(
    WT* __restrict__ ring, const EVT* __restrict__ events, const WT* __restrict__ weights,
    const int32_t* __restrict__ indices, const IndptrT* __restrict__ indptr,
    const int32_t* __restrict__ delays, const int32_t* __restrict__ t_ptr,
    int n_pre, int L, int n_post
) {
    using ACC = typename DelayW<WT>::acc;
    int row = (int)blockIdx.x;
    if (row >= n_pre) return;
    ACC xv = static_cast<ACC>(delay_ev_value(events[row]));
    if (!(xv != ACC(0))) return;
    int t_mod = delay_floor_mod(__ldg(t_ptr), L);
    delay_deliver_row<IndptrT, HOMO, WT>(
        ring, weights, indices, delays, __ldg(&indptr[row]), __ldg(&indptr[row + 1]),
        (int)threadIdx.x, (int)blockDim.x, WEIGHTED ? xv : ACC(1), t_mod, L, n_post);
}

template <typename IndptrT, bool HOMO, bool WEIGHTED, typename EVT, typename WT>
static void delay_launch(
    WT* ring, const EVT* events, const WT* weights, const int32_t* indices,
    const IndptrT* indptr, const int32_t* delays, const int32_t* t_ptr,
    int n_pre, int L, int n_post, int64_t nse, cudaStream_t s
) {
    int64_t avg_nnz = nse / n_pre;
    if (avg_nnz < 32) {
        int grid = (n_pre + 255) / 256;
        _delay_thread_kern<IndptrT, HOMO, WEIGHTED, EVT, WT><<<grid, 256, 0, s>>>(
            ring, events, weights, indices, indptr, delays, t_ptr, n_pre, L, n_post);
    } else if (avg_nnz < 256) {
        int grid = (n_pre + 7) / 8;
        _delay_warp_kern<IndptrT, HOMO, WEIGHTED, EVT, WT><<<grid, 256, 0, s>>>(
            ring, events, weights, indices, indptr, delays, t_ptr, n_pre, L, n_post);
    } else {
        _delay_block_kern<IndptrT, HOMO, WEIGHTED, EVT, WT><<<n_pre, 256, 0, s>>>(
            ring, events, weights, indices, indptr, delays, t_ptr, n_pre, L, n_post);
    }
}

// -------------------------------------------------------------------------
// FFI entry points
// -------------------------------------------------------------------------

#define FFI_CSR_DELAY_DELIVER(SUFFIX, WEIGHT_C_T, EVENT_C_T, WEIGHTED)                     \
void csr_delay_deliver##SUFFIX(                                                           \
    const BE::Tensor ring,                                                                \
    const BE::Tensor events,                                                              \
    const BE::Tensor weights,                                                             \
    const BE::Tensor indices,                                                             \
    const BE::Tensor indptr,                                                              \
    const BE::Tensor delays,                                                              \
    const BE::Tensor t,                                                                   \
    BE::Tensor out_ring,                                                                  \
    int64_t stream                                                                        \
) {                                                                                       \
    BE_CHECK_CSR_INDICES_INT32(indices);                                                  \
    BE_CHECK(delays.dtype() == BE::DType::Int32)                                          \
        << "[be] csr_delay_deliver requires int32 delays, got "                           \
        << BE::dtype_name(delays.dtype());                                                \
    BE_CHECK(t.dtype() == BE::DType::Int32)                                               \
        << "[be] csr_delay_deliver requires an int32 t, got " << BE::dtype_name(t.dtype()); \
    BE_CHECK(out_ring.ndim() == 2) << "[be] csr_delay_deliver ring must be 2-D";          \
    BE_CHECK(delays.size(0) == indices.size(0))                                           \
        << "[be] csr_delay_deliver delays and indices must have the same length";         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int L = static_cast<int>(out_ring.size(0));                                           \
    int n_post = static_cast<int>(out_ring.size(1));                                      \
    int n_pre = static_cast<int>(indptr.size(0)) - 1;                                     \
    int64_t nse = static_cast<int64_t>(indices.size(0));                                  \
    if (n_pre <= 0 || nse == 0 || L <= 0 || n_post <= 0) return;                          \
    bool homo = (weights.size(0) == 1);                                                   \
    WEIGHT_C_T* d_ring = static_cast<WEIGHT_C_T*>(out_ring.data_ptr());                   \
    const EVENT_C_T* d_ev = static_cast<const EVENT_C_T*>(events.data_ptr());             \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr());           \
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());               \
    const int32_t* d_del = static_cast<const int32_t*>(delays.data_ptr());                \
    const int32_t* d_t = static_cast<const int32_t*>(t.data_ptr());                       \
    BE_DISPATCH_CSR_INDPTR(indptr.dtype(), IndptrT, {                                     \
        const IndptrT* d_ipt = static_cast<const IndptrT*>(indptr.data_ptr());            \
        if (homo) {                                                                       \
            delay_launch<IndptrT, true, WEIGHTED, EVENT_C_T, WEIGHT_C_T>(                 \
                d_ring, d_ev, d_w, d_idx, d_ipt, d_del, d_t, n_pre, L, n_post, nse, s);   \
        } else {                                                                          \
            delay_launch<IndptrT, false, WEIGHTED, EVENT_C_T, WEIGHT_C_T>(                \
                d_ring, d_ev, d_w, d_idx, d_ipt, d_del, d_t, n_pre, L, n_post, nse, s);   \
        }                                                                                 \
    });                                                                                   \
}

// @BE csr_delay_deliver_f32_bool
FFI_CSR_DELAY_DELIVER(_f32_bool, float, int8_t, false)
// @BE csr_delay_deliver_f32_float
FFI_CSR_DELAY_DELIVER(_f32_float, float, float, false)
// @BE csr_delay_deliver_f32_weighted
FFI_CSR_DELAY_DELIVER(_f32_weighted, float, float, true)
// @BE csr_delay_deliver_f64_bool
FFI_CSR_DELAY_DELIVER(_f64_bool, double, int8_t, false)
// @BE csr_delay_deliver_f64_float
FFI_CSR_DELAY_DELIVER(_f64_float, double, float, false)
// @BE csr_delay_deliver_f64_weighted
FFI_CSR_DELAY_DELIVER(_f64_weighted, double, double, true)
// @BE csr_delay_deliver_f16_bool
FFI_CSR_DELAY_DELIVER(_f16_bool, __half, int8_t, false)
// @BE csr_delay_deliver_f16_float
FFI_CSR_DELAY_DELIVER(_f16_float, __half, float, false)
// @BE csr_delay_deliver_f16_weighted
FFI_CSR_DELAY_DELIVER(_f16_weighted, __half, __half, true)
// @BE csr_delay_deliver_bf16_bool
FFI_CSR_DELAY_DELIVER(_bf16_bool, __nv_bfloat16, int8_t, false)
// @BE csr_delay_deliver_bf16_float
FFI_CSR_DELAY_DELIVER(_bf16_float, __nv_bfloat16, float, false)
// @BE csr_delay_deliver_bf16_weighted
FFI_CSR_DELAY_DELIVER(_bf16_weighted, __nv_bfloat16, __nv_bfloat16, true)
