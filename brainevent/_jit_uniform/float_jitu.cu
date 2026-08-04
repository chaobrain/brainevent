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
 * float_jitu.cu -- dense light-RNG JIT uniform materialization.
 *
 * jitu_mv_f32 matches CSR matrix_mode="mv"; jitu_mm_aw_t4_f32 matches CSR
 * matrix_mode="mm". The public transpose flag selects notrans/trans output.
 */

#include <cstdio>
#include <cstdlib>

#include "cuda_common.h"
#include "brainevent/common.h"

#define LIGHT_TARGET_CHUNKS 4
#define AW_T4_GROUP_SIZE 4
#define AW_T4_GROUPS_PER_WARP 8

__device__ __forceinline__ unsigned int fast_bounded_u32(
    unsigned int r,
    unsigned int bound
) {
    return __umulhi(r, bound);
}

__device__ __forceinline__ unsigned int mix32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ unsigned int light_rng_init_wpr(
    unsigned int seed,
    int row,
    int chunk_id,
    int lane
) {
    unsigned int x = seed ^ 0xd1b54a35U;
    x ^= (unsigned int)row * 0x85ebca6bU;
    x ^= (unsigned int)chunk_id * 0xc2b2ae35U;
    x ^= (unsigned int)lane * 0x27d4eb2dU;
    x = mix32(x);
    return x == 0U ? 0x6d2b79f5U : x;
}

__device__ __forceinline__ unsigned int light_rng_next(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x == 0U ? 0x6d2b79f5U : x;
    return *state;
}

__device__ __forceinline__ float hash_uniform01(
    unsigned int seed,
    int row,
    int col
) {
    unsigned int h = seed ^ 0xa0761d65U;
    h ^= (unsigned int)row * 0xe7037ed1U;
    h ^= (unsigned int)col * 0x8ebc6af1U;
    h = mix32(h);
    return (float)(h & 0x00ffffffU) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ unsigned int stationary_initial_q(
    unsigned int* state,
    unsigned int cl
) {
    /*
     * The inter-arrival skip is Uniform{1, ..., cl - 1}.  A stationary
     * renewal stream must start from the equilibrium residual distribution
     * P(q = r) = 2 * (cl - 1 - r) / (cl * (cl - 1)), r in [0, cl - 2].
     * Starting from Uniform{0, ..., cl - 1} creates a chunk-position ramp.
     */
    unsigned int n = cl - 1U;
    while (true) {
        unsigned int q = fast_bounded_u32(light_rng_next(state), n);
        unsigned int gate = fast_bounded_u32(light_rng_next(state), n);
        if (gate < n - q) return q;
    }
}

static int default_light_chunk_size(int k) {
    if (k <= 0) return 1;
    return (k + LIGHT_TARGET_CHUNKS - 1) / LIGHT_TARGET_CHUNKS;
}

#define DEFINE_JITU_MV(SFX, WEIGHT_T, ACC_T, READ_W, WRITE_W)                   \
__global__ void _jitu_mv##SFX##_kern(                                           \
    const WEIGHT_T* __restrict__ w_low,                                         \
    const WEIGHT_T* __restrict__ w_high,                                        \
    const int*      __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    WEIGHT_T*       __restrict__ output,                                        \
    int n_rows, int n_cols, int transpose, int chunk_size, int n_chunks         \
) {                                                                            \
    int lane = threadIdx.x & 31;                                                \
    int warp_id = threadIdx.x >> 5;                                             \
    int warps_per_block = blockDim.x >> 5;                                      \
    int row_block = (int)blockIdx.x;                                            \
    int chunk_id = (int)blockIdx.y;                                             \
    int row = row_block * warps_per_block + warp_id;                            \
    if (row >= n_rows || chunk_id >= n_chunks) return;                          \
    int chunk_start = chunk_id * chunk_size;                                    \
    if (chunk_start >= n_cols) return;                                          \
    int chunk_end = chunk_start + chunk_size;                                   \
    if (chunk_end > n_cols) chunk_end = n_cols;                                 \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);         \
    if (chunk_width == 0U) return;                                              \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                       \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                              \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                            \
    if (cl < 2U) cl = 2U;                                                       \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);                         \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);          \
    unsigned int q = stationary_initial_q(&rng, cl);                            \
    unsigned int local_j = (unsigned int)lane + 32U * q;                        \
    while (local_j < chunk_width) {                                             \
        int col = chunk_start + (int)local_j;                                   \
        float u01 = hash_uniform01(seed0, row, col);                            \
        ACC_T w = wlo + (ACC_T)u01 * range;                                     \
        size_t offset = transpose                                               \
            ? ((size_t)col * n_rows + row)                                      \
            : ((size_t)row * n_cols + col);                                     \
        output[offset] = WRITE_W(w);                                            \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);              \
        local_j = (unsigned int)lane + 32U * q;                                 \
    }                                                                           \
}

#define DEFINE_JITU_MM_AW_T4(SFX, WEIGHT_T, ACC_T, READ_W, WRITE_W)             \
__global__ void _jitu_mm_aw_t4##SFX##_kern(                                     \
    const WEIGHT_T* __restrict__ w_low,                                         \
    const WEIGHT_T* __restrict__ w_high,                                        \
    const int*      __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    WEIGHT_T*       __restrict__ output,                                        \
    int n_rows, int n_cols, int transpose, int chunk_size, int n_chunks         \
) {                                                                            \
    int lane = threadIdx.x & 31;                                                \
    int sub_lane = lane & (AW_T4_GROUP_SIZE - 1);                               \
    int group = lane >> 2;                                                      \
    int warp_id = threadIdx.x >> 5;                                             \
    int warps_per_block = blockDim.x >> 5;                                      \
    int chunk_id = (int)blockIdx.y;                                             \
    int warp_task = (int)blockIdx.x * warps_per_block + warp_id;                \
    int row = warp_task * AW_T4_GROUPS_PER_WARP + group;                        \
    if (row >= n_rows || chunk_id >= n_chunks) return;                          \
    int chunk_start = chunk_id * chunk_size;                                    \
    if (chunk_start >= n_cols) return;                                          \
    int chunk_end = chunk_start + chunk_size;                                   \
    if (chunk_end > n_cols) chunk_end = n_cols;                                 \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);         \
    if (chunk_width == 0U) return;                                              \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                       \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                              \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                            \
    if (cl < 2U) cl = 2U;                                                       \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);                         \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);      \
    unsigned int q = stationary_initial_q(&rng, cl);                            \
    unsigned int local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;       \
    while (local_j < chunk_width) {                                             \
        int col = chunk_start + (int)local_j;                                   \
        float u01 = hash_uniform01(seed0, row, col);                            \
        ACC_T w = wlo + (ACC_T)u01 * range;                                     \
        size_t offset = transpose                                               \
            ? ((size_t)col * n_rows + row)                                      \
            : ((size_t)row * n_cols + col);                                     \
        output[offset] = WRITE_W(w);                                            \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);              \
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;                \
    }                                                                           \
}

DEFINE_JITU_MV(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_JITU_MV(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_JITU_MV(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_JITU_MV(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

DEFINE_JITU_MM_AW_T4(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_JITU_MM_AW_T4(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_JITU_MM_AW_T4(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_JITU_MM_AW_T4(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

#define FFI_JITU(NAME, KERNEL, SFX, WEIGHT_T, ROW_GROUPS, TRANSPOSE)             \
void NAME##SFX(                                                                  \
    const BE::Tensor w_low,                                                      \
    const BE::Tensor w_high,                                                     \
    const BE::Tensor clen,                                                       \
    const BE::Tensor seed,                                                       \
    BE::Tensor output,                                                           \
    int n_rows,                                                                  \
    int n_cols,                                                                  \
    int chunk_size,                                                              \
    int64_t stream                                                               \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    if (n_rows <= 0 || n_cols <= 0) return;                                      \
    BE_CUDA_CHECK(cudaMemsetAsync(                                               \
        output.data_ptr(), 0, (size_t)n_rows * n_cols * sizeof(WEIGHT_T), s));   \
    if (chunk_size <= 0) return;                                                 \
    int n_chunks = (n_cols + chunk_size - 1) / chunk_size;                       \
    if (n_chunks <= 0) return;                                                   \
    int threads = 256;                                                           \
    int warps_per_block = threads / 32;                                          \
    int rows_per_block = warps_per_block * ROW_GROUPS;                           \
    int row_blocks = (n_rows + rows_per_block - 1) / rows_per_block;             \
    if (row_blocks > 2147483647 || n_chunks > 65535) {                           \
        fprintf(stderr, #NAME #SFX " grid overflow: row_blocks=%d n_chunks=%d\n", \
                row_blocks, n_chunks);                                           \
        abort();                                                                 \
    }                                                                            \
    dim3 blocks((unsigned int)row_blocks, (unsigned int)n_chunks, 1U);           \
    _jitu_##KERNEL##SFX##_kern<<<blocks, threads, 0, s>>>(                       \
        static_cast<const WEIGHT_T*>(w_low.data_ptr()),                          \
        static_cast<const WEIGHT_T*>(w_high.data_ptr()),                         \
        static_cast<const int*>(clen.data_ptr()),                                \
        static_cast<const int*>(seed.data_ptr()),                                \
        static_cast<WEIGHT_T*>(output.data_ptr()),                               \
        n_rows, n_cols, TRANSPOSE, chunk_size, n_chunks                          \
    );                                                                           \
    BE_CHECK_KERNEL_LAUNCH();                                                    \
}

// @BE jitu_mv_notrans_f32
FFI_JITU(jitu_mv_notrans, mv, _f32, float, 1, 0)
// @BE jitu_mv_notrans_f64
FFI_JITU(jitu_mv_notrans, mv, _f64, double, 1, 0)
// @BE jitu_mv_notrans_f16
FFI_JITU(jitu_mv_notrans, mv, _f16, __half, 1, 0)
// @BE jitu_mv_notrans_bf16
FFI_JITU(jitu_mv_notrans, mv, _bf16, __nv_bfloat16, 1, 0)

// @BE jitu_mv_trans_f32
FFI_JITU(jitu_mv_trans, mv, _f32, float, 1, 1)
// @BE jitu_mv_trans_f64
FFI_JITU(jitu_mv_trans, mv, _f64, double, 1, 1)
// @BE jitu_mv_trans_f16
FFI_JITU(jitu_mv_trans, mv, _f16, __half, 1, 1)
// @BE jitu_mv_trans_bf16
FFI_JITU(jitu_mv_trans, mv, _bf16, __nv_bfloat16, 1, 1)

// @BE jitu_mm_aw_t4_notrans_f32
FFI_JITU(jitu_mm_aw_t4_notrans, mm_aw_t4, _f32, float, AW_T4_GROUPS_PER_WARP, 0)
// @BE jitu_mm_aw_t4_notrans_f64
FFI_JITU(jitu_mm_aw_t4_notrans, mm_aw_t4, _f64, double, AW_T4_GROUPS_PER_WARP, 0)
// @BE jitu_mm_aw_t4_notrans_f16
FFI_JITU(jitu_mm_aw_t4_notrans, mm_aw_t4, _f16, __half, AW_T4_GROUPS_PER_WARP, 0)
// @BE jitu_mm_aw_t4_notrans_bf16
FFI_JITU(jitu_mm_aw_t4_notrans, mm_aw_t4, _bf16, __nv_bfloat16, AW_T4_GROUPS_PER_WARP, 0)

// @BE jitu_mm_aw_t4_trans_f32
FFI_JITU(jitu_mm_aw_t4_trans, mm_aw_t4, _f32, float, AW_T4_GROUPS_PER_WARP, 1)
// @BE jitu_mm_aw_t4_trans_f64
FFI_JITU(jitu_mm_aw_t4_trans, mm_aw_t4, _f64, double, AW_T4_GROUPS_PER_WARP, 1)
// @BE jitu_mm_aw_t4_trans_f16
FFI_JITU(jitu_mm_aw_t4_trans, mm_aw_t4, _f16, __half, AW_T4_GROUPS_PER_WARP, 1)
// @BE jitu_mm_aw_t4_trans_bf16
FFI_JITU(jitu_mm_aw_t4_trans, mm_aw_t4, _bf16, __nv_bfloat16, AW_T4_GROUPS_PER_WARP, 1)
