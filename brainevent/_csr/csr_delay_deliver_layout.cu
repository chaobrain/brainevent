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
 * csr_delay_deliver_layout.cu
 * ===========================
 *
 * Per-edge-delay spike delivery on a precomputed delay-major layout
 * (CSRDelayLayout, #203 / #205).
 *
 * The edges of every row are sorted by delay and delay_ptr[i * L + d] is
 * the first position of delay d in row i (the segment ends at
 * delay_ptr[i * L + d + 1], or indptr[i + 1] for d = L - 1). A step
 *
 *   for every active row i, for every d, for every edge e of segment (i, d):
 *     ring[(t + d) mod L, indices[e]] += w_e * f(x[i])
 *
 * runs in two kernels:
 *   1. compact the active rows (x != 0, NaN active) into scratch[0:count],
 *      scratch[n_pre] = count;
 *   2. a persistent grid takes the items (d, active row) in delay-major
 *      order, 16 lanes per item. Concurrent items share one ring row
 *      (t + d) mod L, which stays in L2 while it is updated.
 *
 * Entry points:
 *   csr_delay_deliver_layout_{wt}_{ev}
 *     wt in {f32, f64, f16, bf16}
 *     ev in {bool (int8 events), float (float32 events),
 *            weighted (events of the weight dtype)}
 *   Homogeneous / heterogeneous weights and int32 / int64 indptr (and
 *   delay_ptr, same dtype) are dispatched at runtime.
 *
 * Operands: ring (L, n_post), events (n_pre,), weights (1,) or (nse,),
 *   indices (nse,) int32, indptr (n_pre + 1,), delay_ptr (n_pre, L),
 *   t (1,) int32 in [0, L); outputs: ring (aliased to operand 0),
 *   scratch (n_pre + 1,) int32.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// -------------------------------------------------------------------------
// Type helpers
// -------------------------------------------------------------------------

template <typename WT> struct LayoutW;
template <> struct LayoutW<float> {
    using acc = float;
    static __device__ __forceinline__ float read(float x) { return x; }
    static __device__ __forceinline__ void add(float* p, float v) { atomic_add_f32(p, v); }
};
template <> struct LayoutW<double> {
    using acc = double;
    static __device__ __forceinline__ double read(double x) { return x; }
    static __device__ __forceinline__ void add(double* p, double v) { atomic_add_f64(p, v); }
};
template <> struct LayoutW<__half> {
    using acc = float;
    static __device__ __forceinline__ float read(__half x) { return __half2float(x); }
    static __device__ __forceinline__ void add(__half* p, float v) { atomic_add_f16(p, v); }
};
template <> struct LayoutW<__nv_bfloat16> {
    using acc = float;
    static __device__ __forceinline__ float read(__nv_bfloat16 x) { return __bfloat162float(x); }
    static __device__ __forceinline__ void add(__nv_bfloat16* p, float v) { atomic_add_bf16(p, v); }
};

__device__ __forceinline__ float layout_ev_value(int8_t x) { return static_cast<float>(x); }
__device__ __forceinline__ float layout_ev_value(float x) { return x; }
__device__ __forceinline__ double layout_ev_value(double x) { return x; }
__device__ __forceinline__ float layout_ev_value(__half x) { return __half2float(x); }
__device__ __forceinline__ float layout_ev_value(__nv_bfloat16 x) { return __bfloat162float(x); }

// Lanes per (delay, row) item. Segments average nse / (n_pre * L) edges.
#define LAYOUT_GROUP 16

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------

template <typename EVT>
__global__ void __launch_bounds__(256) _layout_compact_kern(
    const EVT* __restrict__ events, int32_t* __restrict__ scratch, int n_pre
) {
    int i = (int)(blockIdx.x * (uint32_t)blockDim.x) + threadIdx.x;
    if (i < n_pre && layout_ev_value(events[i]) != 0) {
        scratch[atomicAdd(&scratch[n_pre], 1)] = i;
    }
}

template <typename IndptrT, bool HOMO, bool WEIGHTED, typename EVT, typename WT>
__global__ void __launch_bounds__(256) _layout_deliver_kern(
    WT* __restrict__ ring, const EVT* __restrict__ events, const WT* __restrict__ weights,
    const int32_t* __restrict__ indices, const IndptrT* __restrict__ indptr,
    const IndptrT* __restrict__ delay_ptr, const int32_t* __restrict__ scratch,
    const int32_t* __restrict__ t_ptr, int n_pre, int L, int n_post
) {
    using ACC = typename LayoutW<WT>::acc;
    const int n_active = scratch[n_pre];
    const int64_t total = static_cast<int64_t>(n_active) * L;
    const int sub = (int)(threadIdx.x & (LAYOUT_GROUP - 1));
    const int64_t n_groups = (static_cast<int64_t>(gridDim.x) * blockDim.x) / LAYOUT_GROUP;
    const int t_mod = __ldg(t_ptr);
    const ACC w_homo = HOMO ? LayoutW<WT>::read(weights[0]) : ACC(0);
    for (int64_t item = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / LAYOUT_GROUP;
         item < total; item += n_groups) {
        // Delay-major: item -> (d, a) with a fastest, so resident groups share d.
        const int d = static_cast<int>(item / n_active);
        const int a = static_cast<int>(item - static_cast<int64_t>(d) * n_active);
        const int row = scratch[a];
        const int64_t base = static_cast<int64_t>(row) * L + d;
        const IndptrT start = __ldg(&delay_ptr[base]);
        const IndptrT end = (d + 1 < L) ? __ldg(&delay_ptr[base + 1]) : __ldg(&indptr[row + 1]);
        if (start >= end) continue;
        const ACC factor = WEIGHTED ? static_cast<ACC>(layout_ev_value(events[row])) : ACC(1);
        int slot = t_mod + d;
        if (slot >= L) slot -= L;
        WT* dst = ring + static_cast<int64_t>(slot) * n_post;
        for (IndptrT p = start + sub; p < end; p += LAYOUT_GROUP) {
            const ACC v = HOMO ? w_homo * factor : LayoutW<WT>::read(weights[p]) * factor;
            LayoutW<WT>::add(&dst[__ldg(&indices[p])], v);
        }
    }
}

static int layout_grid_size() {
    static int grid = 0;
    if (grid == 0) {
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        grid = (sms > 0 ? sms : 80) * 32;
    }
    return grid;
}

template <typename IndptrT, bool HOMO, bool WEIGHTED, typename EVT, typename WT>
static void layout_launch(
    WT* ring, const EVT* events, const WT* weights, const int32_t* indices,
    const IndptrT* indptr, const IndptrT* delay_ptr, int32_t* scratch, const int32_t* t_ptr,
    int n_pre, int L, int n_post, cudaStream_t s
) {
    cudaMemsetAsync(scratch + n_pre, 0, sizeof(int32_t), s);
    _layout_compact_kern<EVT><<<(n_pre + 255) / 256, 256, 0, s>>>(events, scratch, n_pre);
    _layout_deliver_kern<IndptrT, HOMO, WEIGHTED, EVT, WT><<<layout_grid_size(), 256, 0, s>>>(
        ring, events, weights, indices, indptr, delay_ptr, scratch, t_ptr, n_pre, L, n_post);
}

// -------------------------------------------------------------------------
// FFI entry points
// -------------------------------------------------------------------------

#define FFI_CSR_DELAY_DELIVER_LAYOUT(SUFFIX, WEIGHT_C_T, EVENT_C_T, WEIGHTED)                \
void csr_delay_deliver_layout##SUFFIX(                                                       \
    const BE::Tensor ring,                                                                   \
    const BE::Tensor events,                                                                 \
    const BE::Tensor weights,                                                                \
    const BE::Tensor indices,                                                                \
    const BE::Tensor indptr,                                                                 \
    const BE::Tensor delay_ptr,                                                              \
    const BE::Tensor t,                                                                      \
    BE::Tensor out_ring,                                                                     \
    BE::Tensor scratch,                                                                      \
    int64_t stream                                                                           \
) {                                                                                          \
    BE_CHECK_CSR_INDICES_INT32(indices);                                                     \
    BE_CHECK(delay_ptr.dtype() == indptr.dtype())                                            \
        << "[be] csr_delay_deliver_layout requires delay_ptr with the indptr dtype, got "    \
        << BE::dtype_name(delay_ptr.dtype());                                                \
    BE_CHECK(t.dtype() == BE::DType::Int32)                                                  \
        << "[be] csr_delay_deliver_layout requires an int32 t, got "                         \
        << BE::dtype_name(t.dtype());                                                        \
    BE_CHECK(scratch.dtype() == BE::DType::Int32)                                            \
        << "[be] csr_delay_deliver_layout requires an int32 scratch buffer";                 \
    BE_CHECK(out_ring.ndim() == 2) << "[be] csr_delay_deliver_layout ring must be 2-D";      \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
    int L = static_cast<int>(out_ring.size(0));                                              \
    int n_post = static_cast<int>(out_ring.size(1));                                         \
    int n_pre = static_cast<int>(indptr.size(0)) - 1;                                        \
    BE_CHECK(scratch.size(0) == static_cast<int64_t>(n_pre) + 1)                             \
        << "[be] csr_delay_deliver_layout scratch must have n_pre + 1 entries";              \
    BE_CHECK(delay_ptr.size(0) == static_cast<int64_t>(n_pre) * L)                           \
        << "[be] csr_delay_deliver_layout delay_ptr must have n_pre * L entries";            \
    int64_t nse = static_cast<int64_t>(indices.size(0));                                     \
    if (n_pre <= 0 || nse == 0 || L <= 0 || n_post <= 0) return;                             \
    bool homo = (weights.size(0) == 1);                                                      \
    WEIGHT_C_T* d_ring = static_cast<WEIGHT_C_T*>(out_ring.data_ptr());                      \
    const EVENT_C_T* d_ev = static_cast<const EVENT_C_T*>(events.data_ptr());                \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr());              \
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());                  \
    int32_t* d_scratch = static_cast<int32_t*>(scratch.data_ptr());                          \
    const int32_t* d_t = static_cast<const int32_t*>(t.data_ptr());                          \
    BE_DISPATCH_CSR_INDPTR(indptr.dtype(), IndptrT, {                                        \
        const IndptrT* d_ipt = static_cast<const IndptrT*>(indptr.data_ptr());               \
        const IndptrT* d_dptr = static_cast<const IndptrT*>(delay_ptr.data_ptr());           \
        if (homo) {                                                                          \
            layout_launch<IndptrT, true, WEIGHTED, EVENT_C_T, WEIGHT_C_T>(                   \
                d_ring, d_ev, d_w, d_idx, d_ipt, d_dptr, d_scratch, d_t, n_pre, L, n_post, s); \
        } else {                                                                             \
            layout_launch<IndptrT, false, WEIGHTED, EVENT_C_T, WEIGHT_C_T>(                  \
                d_ring, d_ev, d_w, d_idx, d_ipt, d_dptr, d_scratch, d_t, n_pre, L, n_post, s); \
        }                                                                                    \
    });                                                                                      \
}

// @BE csr_delay_deliver_layout_f32_bool arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_f32_bool, float, int8_t, false)
// @BE csr_delay_deliver_layout_f32_float arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_f32_float, float, float, false)
// @BE csr_delay_deliver_layout_f32_weighted arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_f32_weighted, float, float, true)
// @BE csr_delay_deliver_layout_f64_bool arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_f64_bool, double, int8_t, false)
// @BE csr_delay_deliver_layout_f64_float arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_f64_float, double, float, false)
// @BE csr_delay_deliver_layout_f64_weighted arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_f64_weighted, double, double, true)
// @BE csr_delay_deliver_layout_f16_bool arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_f16_bool, __half, int8_t, false)
// @BE csr_delay_deliver_layout_f16_float arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_f16_float, __half, float, false)
// @BE csr_delay_deliver_layout_f16_weighted arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_f16_weighted, __half, __half, true)
// @BE csr_delay_deliver_layout_bf16_bool arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_bf16_bool, __nv_bfloat16, int8_t, false)
// @BE csr_delay_deliver_layout_bf16_float arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_bf16_float, __nv_bfloat16, float, false)
// @BE csr_delay_deliver_layout_bf16_weighted arg arg arg arg arg arg arg ret ret stream
FFI_CSR_DELAY_DELIVER_LAYOUT(_bf16_weighted, __nv_bfloat16, __nv_bfloat16, true)
