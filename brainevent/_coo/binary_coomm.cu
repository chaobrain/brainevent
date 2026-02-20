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
 * binary_coomm.cu -- Event-Driven Binary COO Sparse Matrix-Matrix CUDA Kernels
 * =============================================================================
 *
 * Python API: brainevent.binary_coomm(data, row, col, B, *, shape, transpose, backend)
 *
 * Event-driven COO (Coordinate) sparse matrix-matrix product where the dense
 * matrix ``B`` contains binary events (spikes). Only entries whose corresponding
 * ``B`` element is active contribute to the output, enabling event-driven
 * sparsity exploitation beyond the structural sparsity of the COO matrix.
 *
 * Parameters
 * ----------
 * data : 1-D float tensor of shape [1] (homogeneous) or [nnz] (heterogeneous).
 * row  : 1-D int32 tensor of shape [nnz] -- row indices of the sparse matrix.
 * col  : 1-D int32 tensor of shape [nnz] -- column indices of the sparse matrix.
 * B    : 2-D tensor of shape [k_in, n] (NT) or [m, n] (T) -- spike event matrix.
 *        bool  (int8): active when != 0.
 *        float (f32):  active when  > 0.
 * output : 2-D float tensor of shape [m, n] (NT) or [k, n] (T).
 *
 * Semantics
 * ---------
 * NT mode (transpose=False):
 *   out[row[s], j] += data[s]   for each s where B[col[s], j] is active
 *   Sparse matrix shape: [m, k].  B shape: [k, n].  Output shape: [m, n].
 *
 * T mode (transpose=True):
 *   out[col[s], j] += data[s]   for each s where B[row[s], j] is active
 *   Sparse matrix shape: [m, k].  B shape: [m, n].  Output shape: [k, n].
 *
 * Kernel Variants
 * ---------------
 * Two complementary kernel families cover the parameter space:
 *
 * Variant 1 — Column-Tiled (CT):
 *   Grid: (ceil(nnz/BLOCK_K), ceil(n/BLOCK_N))    Block: (BLOCK_N=32)
 *   One warp per thread block. Serially iterates over BLOCK_K=32 NNZ entries;
 *   all 32 threads handle a BLOCK_N=32 tile of output columns in parallel.
 *   Uses ``__ballot_sync`` to skip atomic writes when all columns in a tile
 *   are inactive for a given NNZ entry — the key event-driven optimisation.
 *   Coalesced reads of B[src, n_tile] (32 consecutive elements per NNZ).
 *   Best for: large nnz, small/moderate n (n ≤ 64). Typical SNN regime.
 *
 * Variant 2 — Warp-Per-Entry (WPE):
 *   Grid: (ceil(nnz/WARPS_PER_BLOCK), ceil(n/32))    Block: (256=8 warps)
 *   Each warp handles a single NNZ entry and 32 consecutive output columns.
 *   No serial loop over NNZ: maximum parallelism across entries.
 *   Coalesced reads and writes to consecutive columns.
 *   Best for: large n (n > 64), smaller or moderate nnz.
 *
 * Homogeneous vs Heterogeneous weights:
 *   Detected at runtime: data.size(0)==1 → homogeneous, broadcast over all NNZ.
 *   The is_homo flag is warp-uniform, so there is no warp divergence.
 *
 * Dtype support:
 *   Weights and output: float32, float64, float16 (sm_70+), bfloat16 (sm_80+).
 *   f16 and bf16 accumulate in float32 for numerical stability; atomic writes
 *   back as f16/bf16 using native atomicAdd (sm_70+/sm_80+) or CAS fallback.
 *
 * Output initialization:
 *   The output buffer is zeroed via cudaMemsetAsync before the kernel launches.
 *
 * Index safety:
 *   All array indices involving product (row_idx * n) use int64_t to avoid
 *   int32 overflow when m, k, or n is large (e.g., 100K neurons × 10K columns).
 *
 * IMPORTANT: All data_ptr() calls return GPU device pointers.
 * NEVER dereference on the host. Pass to kernels unchanged.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// Warp-level reduction helpers
// ============================================================================

__device__ __inline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __inline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// Active-check predicates
// ============================================================================

#define IS_ACTIVE_BOOL(s)  ((s) != 0)
#define IS_ACTIVE_FLOAT(s) ((s) > 0.0f)

// ============================================================================
// Per-dtype conversion macros: READ converts WEIGHT_T -> ACC_T
// ============================================================================

#define READ_F32(x)   (x)
#define READ_F64(x)   (x)
#define READ_F16(x)   __half2float(x)
#define READ_BF16(x)  __bfloat162float(x)

// ============================================================================
// Per-dtype atomic-add helpers (accumulator value -> weight memory)
//
// For f32/f64: native atomicAdd, universally supported.
// For f16: native atomicAdd on sm_70+; CAS-based fallback for older arches.
// For bf16: native atomicAdd on sm_80+; CAS-based fallback.
// ============================================================================

__device__ __inline__ void atomic_add_f32(float* addr, float val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomic_add_f64(double* addr, double val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomic_add_f16(__half* addr, float val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, __float2half(val));
#else
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __half2float(*reinterpret_cast<__half*>(&h));
        __half new_h = __float2half(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}

__device__ __inline__ void atomic_add_bf16(__nv_bfloat16* addr, float val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, __float2bfloat16(val));
#else
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&h));
        __nv_bfloat16 new_h = __float2bfloat16(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}

// ============================================================================
// Tiling constants
// ============================================================================

#define COOMM_CT_BLOCK_K   32   // NNZ entries processed serially per CT block
#define COOMM_CT_BLOCK_N   32   // Output columns processed in parallel per CT block (= warp width)
#define COOMM_WPE_WARPS    8    // Warps per block for WPE variant (8 × 32 = 256 threads)
#define COOMM_WPE_COLS     32   // Output columns processed per warp per n-tile (= warp width)

// ============================================================================
// Variant 1: Column-Tiled (CT) — NT direction
//
// Semantics: out[row[s], j] += data[s]  for each s where B[col[s], j] active
//
// Grid:  (ceil(nnz / BLOCK_K), ceil(n / BLOCK_N))
// Block: (BLOCK_N=32,)  — one warp
//
// Thread t handles output column (blockIdx.y * BLOCK_N + t).
// Each block serializes over BLOCK_K NNZ entries. For each entry:
//   - All threads load B[col[s], my_col] in coalesced fashion.
//   - __ballot_sync skips atomicAdd when no column in the tile is active.
//   - atomicAdd to out[row[s], my_col] for active threads.
//
// Weight read is warp-uniform (same k for all threads), so data[s] hits L1.
// ============================================================================

#define DEFINE_COOMM_CT_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_ct_nt_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                          \
    const int32_t*  __restrict__ row,                                                           \
    const int32_t*  __restrict__ col,                                                           \
    const SPIKE_T*  __restrict__ B,     /* [k_in, n] row-major */                              \
    WEIGHT_T*                    out,   /* [m, n]    row-major */                              \
    int nnz, int n, int is_homo                                                                 \
) {                                                                                             \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                             \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                             \
    int t         = threadIdx.x;                                                                \
    int my_col    = col_start + t;                                                              \
    bool col_valid = (my_col < n);                                                              \
    /* Pre-load homo weight: data[0] always valid since nnz > 0 on launch */                   \
    ACC_T homo_w = READ_W(data[0]);                                                             \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                                \
    if (nnz_end > nnz) nnz_end = nnz;                                                          \
    for (int s = nnz_start; s < nnz_end; s++) {                                                \
        /* col[s] and row[s] are warp-uniform reads (same value for all 32 threads) */         \
        int src = col[s];   /* source row index in B */                                        \
        int dst = row[s];   /* destination row index in out */                                 \
        /* Coalesced read: 32 threads load 32 consecutive B columns */                         \
        SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                \
        bool active = IS_ACTIVE(spike) && col_valid;                                            \
        /* __ballot_sync: skip all atomics if no column is active in this tile */              \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                    \
        if (ballot == 0u) continue;                                                             \
        if (active) {                                                                           \
            ACC_T w = is_homo ? homo_w : READ_W(data[s]);                                      \
            ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w);                                  \
        }                                                                                       \
    }                                                                                           \
}

// ============================================================================
// Variant 1: Column-Tiled (CT) — T direction
//
// Semantics: out[col[s], j] += data[s]  for each s where B[row[s], j] active
//
// Identical structure to CT-NT, but src = row[s], dst = col[s].
// ============================================================================

#define DEFINE_COOMM_CT_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_ct_t_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                         \
    const int32_t*  __restrict__ row,                                                          \
    const int32_t*  __restrict__ col,                                                          \
    const SPIKE_T*  __restrict__ B,     /* [m, n] row-major */                                \
    WEIGHT_T*                    out,   /* [k, n] row-major */                                \
    int nnz, int n, int is_homo                                                                \
) {                                                                                            \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                            \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                            \
    int t         = threadIdx.x;                                                               \
    int my_col    = col_start + t;                                                             \
    bool col_valid = (my_col < n);                                                             \
    ACC_T homo_w = READ_W(data[0]);                                                            \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                               \
    if (nnz_end > nnz) nnz_end = nnz;                                                         \
    for (int s = nnz_start; s < nnz_end; s++) {                                               \
        int src = row[s];   /* source row index in B (T mode) */                              \
        int dst = col[s];   /* destination row index in out (T mode) */                       \
        SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;               \
        bool active = IS_ACTIVE(spike) && col_valid;                                           \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                   \
        if (ballot == 0u) continue;                                                            \
        if (active) {                                                                          \
            ACC_T w = is_homo ? homo_w : READ_W(data[s]);                                     \
            ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w);                                 \
        }                                                                                      \
    }                                                                                          \
}

// ============================================================================
// Variant 2: Warp-Per-Entry (WPE) — NT direction
//
// Semantics: out[row[s], j] += data[s]  for each s where B[col[s], j] active
//
// Grid:  (ceil(nnz / WARPS_PER_BLOCK), ceil(n / COLS_PER_WARP))
// Block: (WARPS_PER_BLOCK * 32 = 256,)  — 8 warps
//
// Each warp handles exactly one NNZ entry and 32 consecutive output columns.
// No serialisation over NNZ: maximum GPU utilisation when nnz is large and
// n is large. All 32 lanes write to consecutive out[dst, n_start..n_end),
// giving fully coalesced writes when the warp's dst is unique (typical case
// with random COO). The warp ballot skips all atomics when B[src, n_chunk]
// is entirely zero for this entry's n-tile.
// ============================================================================

#define DEFINE_COOMM_WPE_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_wpe_nt_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                           \
    const int32_t*  __restrict__ row,                                                            \
    const int32_t*  __restrict__ col,                                                            \
    const SPIKE_T*  __restrict__ B,     /* [k_in, n] row-major */                               \
    WEIGHT_T*                    out,   /* [m, n]    row-major */                               \
    int nnz, int n, int is_homo                                                                  \
) {                                                                                              \
    int warp_id  = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);              \
    int lane     = threadIdx.x & 31;                                                             \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                                \
    int my_col    = col_start + lane;                                                            \
    if (warp_id >= nnz) return;                                                                  \
    bool col_valid = (my_col < n);                                                               \
    int s   = warp_id;                                                                           \
    int src = col[s];   /* source row in B */                                                    \
    int dst = row[s];   /* destination row in out */                                             \
    /* All lanes in warp read consecutive B columns — coalesced */                               \
    SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                     \
    bool active = IS_ACTIVE(spike) && col_valid;                                                 \
    /* Warp ballot: if entire warp's n-tile is inactive, skip atomics */                         \
    uint32_t ballot = __ballot_sync(0xffffffff, active);                                         \
    if (ballot == 0u) return;                                                                    \
    if (active) {                                                                                 \
        ACC_T w = is_homo ? READ_W(data[0]) : READ_W(data[s]);                                  \
        /* All active lanes write to consecutive out[dst, my_col] — coalesced */                 \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w);                                       \
    }                                                                                            \
}

// ============================================================================
// Variant 2: Warp-Per-Entry (WPE) — T direction
//
// Semantics: out[col[s], j] += data[s]  for each s where B[row[s], j] active
// ============================================================================

#define DEFINE_COOMM_WPE_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_wpe_t_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                          \
    const int32_t*  __restrict__ row,                                                           \
    const int32_t*  __restrict__ col,                                                           \
    const SPIKE_T*  __restrict__ B,     /* [m, n] row-major */                                 \
    WEIGHT_T*                    out,   /* [k, n] row-major */                                 \
    int nnz, int n, int is_homo                                                                 \
) {                                                                                             \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);            \
    int lane      = threadIdx.x & 31;                                                           \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                               \
    int my_col    = col_start + lane;                                                           \
    if (warp_id >= nnz) return;                                                                 \
    bool col_valid = (my_col < n);                                                              \
    int s   = warp_id;                                                                          \
    int src = row[s];   /* source row in B (T mode) */                                         \
    int dst = col[s];   /* destination row in out (T mode) */                                  \
    SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                    \
    bool active = IS_ACTIVE(spike) && col_valid;                                                \
    uint32_t ballot = __ballot_sync(0xffffffff, active);                                        \
    if (ballot == 0u) return;                                                                   \
    if (active) {                                                                               \
        ACC_T w = is_homo ? READ_W(data[0]) : READ_W(data[s]);                                 \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w);                                      \
    }                                                                                           \
}

// ============================================================================
// Kernel instantiations: 4 dtypes x 2 spike types x 4 variants = 32 kernels
// ============================================================================

// ---- Float32: CT-NT ----
DEFINE_COOMM_CT_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_CT_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
// ---- Float32: CT-T ----
DEFINE_COOMM_CT_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_CT_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
// ---- Float32: WPE-NT ----
DEFINE_COOMM_WPE_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_WPE_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
// ---- Float32: WPE-T ----
DEFINE_COOMM_WPE_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_WPE_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)

// ---- Float64: CT-NT ----
DEFINE_COOMM_CT_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_CT_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
// ---- Float64: CT-T ----
DEFINE_COOMM_CT_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_CT_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
// ---- Float64: WPE-NT ----
DEFINE_COOMM_WPE_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_WPE_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
// ---- Float64: WPE-T ----
DEFINE_COOMM_WPE_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_WPE_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)

// ---- Float16 (accumulate in f32): CT-NT ----
DEFINE_COOMM_CT_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_CT_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
// ---- Float16: CT-T ----
DEFINE_COOMM_CT_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_CT_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
// ---- Float16: WPE-NT ----
DEFINE_COOMM_WPE_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_WPE_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
// ---- Float16: WPE-T ----
DEFINE_COOMM_WPE_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_WPE_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)

// ---- BFloat16 (accumulate in f32): CT-NT ----
DEFINE_COOMM_CT_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_CT_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
// ---- BFloat16: CT-T ----
DEFINE_COOMM_CT_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_CT_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
// ---- BFloat16: WPE-NT ----
DEFINE_COOMM_WPE_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_WPE_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
// ---- BFloat16: WPE-T ----
DEFINE_COOMM_WPE_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_WPE_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)


// ============================================================================
// TVM FFI Entry Point Macros
// ============================================================================
//
// Convention: args = (data, row_idx, col_idx, B, output, stream)
//   data     : [1] (homo) or [nnz] (hetero) -- weight values
//   row_idx  : [nnz], int32 -- row indices of the sparse matrix
//   col_idx  : [nnz], int32 -- column indices of the sparse matrix
//   B        : [k_in, n] (NT) or [m, n] (T) -- spike event matrix
//   output   : [m, n] (NT) or [k, n] (T) -- result (zero-initialized here)
//   stream   : CUDA stream handle (int64)
//
// IMPORTANT: data_ptr() returns GPU device pointers.
// NEVER dereference on the host. Only extract metadata (ndim, size).
//
// Output zeroing: cudaMemsetAsync zeros the output before the atomic kernel.
//
// CT dispatch: block=(32,), grid=(ceil(nnz/32), ceil(n/32)).
//   One warp serialises over 32 NNZ entries per n-column-tile.
//
// WPE dispatch: block=(256,), grid=(ceil(nnz/8), ceil(n/32)).
//   Each warp handles exactly one NNZ entry and 32 output columns.
//   256 threads = 8 warps per block → 8 NNZ entries per block.
// ============================================================================

// ---- FFI macro: CT-NT ----
#define FFI_COOMM_CT_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)               \
void binary_coomm_ct_nt##SUFFIX(                                                           \
    tvm::ffi::TensorView data,                                                             \
    tvm::ffi::TensorView row_idx,                                                          \
    tvm::ffi::TensorView col_idx,                                                          \
    tvm::ffi::TensorView B,                                                                \
    tvm::ffi::TensorView output,                                                           \
    int64_t stream                                                                         \
) {                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int nnz  = static_cast<int>(row_idx.size(0));                                         \
    int n    = static_cast<int>(B.size(1));        /* output columns */                   \
    int m    = static_cast<int>(output.size(0));   /* output rows */                      \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                           \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                     \
    /* Zero output before atomic scatter */                                                \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);                    \
    if (nnz == 0) return;                                                                  \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                                   \
    dim3 grid(                                                                             \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                                 \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                                 \
        1                                                                                  \
    );                                                                                     \
    _coomm_ct_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                                     \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                  \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                  \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                                      \
        d_out, nnz, n, is_homo                                                            \
    );                                                                                     \
}

// ---- FFI macro: CT-T ----
#define FFI_COOMM_CT_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)                \
void binary_coomm_ct_t##SUFFIX(                                                            \
    tvm::ffi::TensorView data,                                                             \
    tvm::ffi::TensorView row_idx,                                                          \
    tvm::ffi::TensorView col_idx,                                                          \
    tvm::ffi::TensorView B,                                                                \
    tvm::ffi::TensorView output,                                                           \
    int64_t stream                                                                         \
) {                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int nnz  = static_cast<int>(row_idx.size(0));                                         \
    int n    = static_cast<int>(B.size(1));                                               \
    int k_out = static_cast<int>(output.size(0));                                         \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                           \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                     \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);                \
    if (nnz == 0) return;                                                                  \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                                   \
    dim3 grid(                                                                             \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                                 \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                                 \
        1                                                                                  \
    );                                                                                     \
    _coomm_ct_t_kern##SUFFIX<<<grid, block, 0, s>>>(                                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                  \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                  \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                                      \
        d_out, nnz, n, is_homo                                                            \
    );                                                                                     \
}

// ---- FFI macro: WPE-NT ----
#define FFI_COOMM_WPE_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)              \
void binary_coomm_wpe_nt##SUFFIX(                                                          \
    tvm::ffi::TensorView data,                                                             \
    tvm::ffi::TensorView row_idx,                                                          \
    tvm::ffi::TensorView col_idx,                                                          \
    tvm::ffi::TensorView B,                                                                \
    tvm::ffi::TensorView output,                                                           \
    int64_t stream                                                                         \
) {                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int nnz   = static_cast<int>(row_idx.size(0));                                        \
    int n     = static_cast<int>(B.size(1));                                              \
    int m     = static_cast<int>(output.size(0));                                         \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                           \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                     \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);                    \
    if (nnz == 0) return;                                                                  \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);    /* 256 threads = 8 warps */               \
    dim3 grid(                                                                             \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                                   \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                                    \
        1                                                                                  \
    );                                                                                     \
    _coomm_wpe_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                                    \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                  \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                  \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                                      \
        d_out, nnz, n, is_homo                                                            \
    );                                                                                     \
}

// ---- FFI macro: WPE-T ----
#define FFI_COOMM_WPE_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)               \
void binary_coomm_wpe_t##SUFFIX(                                                           \
    tvm::ffi::TensorView data,                                                             \
    tvm::ffi::TensorView row_idx,                                                          \
    tvm::ffi::TensorView col_idx,                                                          \
    tvm::ffi::TensorView B,                                                                \
    tvm::ffi::TensorView output,                                                           \
    int64_t stream                                                                         \
) {                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int nnz   = static_cast<int>(row_idx.size(0));                                        \
    int n     = static_cast<int>(B.size(1));                                              \
    int k_out = static_cast<int>(output.size(0));                                         \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                           \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                     \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);                \
    if (nnz == 0) return;                                                                  \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                               \
    dim3 grid(                                                                             \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                                   \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                                    \
        1                                                                                  \
    );                                                                                     \
    _coomm_wpe_t_kern##SUFFIX<<<grid, block, 0, s>>>(                                     \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                  \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                  \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                                      \
        d_out, nnz, n, is_homo                                                            \
    );                                                                                     \
}

// ============================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// ============================================================================
// Naming convention:
//   binary_coomm_{variant}_{direction}_{weight_dtype}_{spike_type}
//   variant:   ct (column-tiled) | wpe (warp-per-entry)
//   direction: nt (transpose=False) | t (transpose=True)
//   dtype:     f32 | f64 | f16 | bf16
//   spikes:    bool (int8) | float (float32)
// ============================================================================

// ============================================================================
// CT-NT: Column-Tiled, Non-Transpose
// ============================================================================

// ---- Float32 ----
// @tvm_ffi binary_coomm_ct_nt_f32_bool
FFI_COOMM_CT_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomm_ct_nt_f32_float
FFI_COOMM_CT_NT(_f32_float, float,  float,  sizeof(float))

// ---- Float64 ----
// @tvm_ffi binary_coomm_ct_nt_f64_bool
FFI_COOMM_CT_NT(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomm_ct_nt_f64_float
FFI_COOMM_CT_NT(_f64_float, double, float,  sizeof(double))

// ---- Float16 ----
// @tvm_ffi binary_coomm_ct_nt_f16_bool
FFI_COOMM_CT_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomm_ct_nt_f16_float
FFI_COOMM_CT_NT(_f16_float, __half, float,  sizeof(__half))

// ---- BFloat16 ----
// @tvm_ffi binary_coomm_ct_nt_bf16_bool
FFI_COOMM_CT_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomm_ct_nt_bf16_float
FFI_COOMM_CT_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// ============================================================================
// CT-T: Column-Tiled, Transpose
// ============================================================================

// ---- Float32 ----
// @tvm_ffi binary_coomm_ct_t_f32_bool
FFI_COOMM_CT_T(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomm_ct_t_f32_float
FFI_COOMM_CT_T(_f32_float, float,  float,  sizeof(float))

// ---- Float64 ----
// @tvm_ffi binary_coomm_ct_t_f64_bool
FFI_COOMM_CT_T(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomm_ct_t_f64_float
FFI_COOMM_CT_T(_f64_float, double, float,  sizeof(double))

// ---- Float16 ----
// @tvm_ffi binary_coomm_ct_t_f16_bool
FFI_COOMM_CT_T(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomm_ct_t_f16_float
FFI_COOMM_CT_T(_f16_float, __half, float,  sizeof(__half))

// ---- BFloat16 ----
// @tvm_ffi binary_coomm_ct_t_bf16_bool
FFI_COOMM_CT_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomm_ct_t_bf16_float
FFI_COOMM_CT_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// ============================================================================
// WPE-NT: Warp-Per-Entry, Non-Transpose
// ============================================================================

// ---- Float32 ----
// @tvm_ffi binary_coomm_wpe_nt_f32_bool
FFI_COOMM_WPE_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomm_wpe_nt_f32_float
FFI_COOMM_WPE_NT(_f32_float, float,  float,  sizeof(float))

// ---- Float64 ----
// @tvm_ffi binary_coomm_wpe_nt_f64_bool
FFI_COOMM_WPE_NT(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomm_wpe_nt_f64_float
FFI_COOMM_WPE_NT(_f64_float, double, float,  sizeof(double))

// ---- Float16 ----
// @tvm_ffi binary_coomm_wpe_nt_f16_bool
FFI_COOMM_WPE_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomm_wpe_nt_f16_float
FFI_COOMM_WPE_NT(_f16_float, __half, float,  sizeof(__half))

// ---- BFloat16 ----
// @tvm_ffi binary_coomm_wpe_nt_bf16_bool
FFI_COOMM_WPE_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomm_wpe_nt_bf16_float
FFI_COOMM_WPE_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// ============================================================================
// WPE-T: Warp-Per-Entry, Transpose
// ============================================================================

// ---- Float32 ----
// @tvm_ffi binary_coomm_wpe_t_f32_bool
FFI_COOMM_WPE_T(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomm_wpe_t_f32_float
FFI_COOMM_WPE_T(_f32_float, float,  float,  sizeof(float))

// ---- Float64 ----
// @tvm_ffi binary_coomm_wpe_t_f64_bool
FFI_COOMM_WPE_T(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomm_wpe_t_f64_float
FFI_COOMM_WPE_T(_f64_float, double, float,  sizeof(double))

// ---- Float16 ----
// @tvm_ffi binary_coomm_wpe_t_f16_bool
FFI_COOMM_WPE_T(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomm_wpe_t_f16_float
FFI_COOMM_WPE_T(_f16_float, __half, float,  sizeof(__half))

// ---- BFloat16 ----
// @tvm_ffi binary_coomm_wpe_t_bf16_bool
FFI_COOMM_WPE_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomm_wpe_t_bf16_float
FFI_COOMM_WPE_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))
