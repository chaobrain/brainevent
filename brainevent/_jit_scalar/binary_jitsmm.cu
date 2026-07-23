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
// =============================================================================

/*
 * light_rng-chunk-wpr-mm-aw-t4.cu - AW-Light MM backend with 4-thread WPR groups.
 *
 * MM = repeated MV: each column is an independent MV.
 *   - packed layout: [n][ceil(k/32)] column-major bitmask.
 *   - Grid 3D: (row_group_blocks, n_chunks, n).
 *   - One warp = 8 independent 4-thread (row, chunk_id, col) slices.
 *
 * Both notrans and trans orientations are provided.
 *
 * Restrictions: f32 weights, bool/int8 spikes, n <= 32.
 */

#include <cstdio>
#include <cstdlib>

#include "cuda_common.h"
#include "brainevent/common.h"

#define AW_T4_GROUP_SIZE 4
#define AW_T4_GROUPS_PER_WARP 8

// =========================================================================
// Bit-packed active check
// =========================================================================
#define IS_ACTIVE_PACKED(packed, idx) \
    ((__ldg(&(packed)[(idx) >> 5]) >> ((idx) & 31)) & 1U)

// =========================================================================
// Light RNG helpers
// =========================================================================

__device__ __forceinline__ unsigned int fast_bounded_u32(
    unsigned int r, unsigned int bound)
{
    return __umulhi(r, bound);
}

__device__ __forceinline__ unsigned int mix32(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ unsigned int light_rng_init_wpr(
    unsigned int seed, int row, int chunk_id, int lane)
{
    unsigned int x = seed ^ 0xd1b54a35U;
    x ^= (unsigned int)row * 0x85ebca6bU;
    x ^= (unsigned int)chunk_id * 0xc2b2ae35U;
    x ^= (unsigned int)lane * 0x27d4eb2dU;
    x = mix32(x);
    return x == 0U ? 0x6d2b79f5U : x;
}

__device__ __forceinline__ unsigned int light_rng_next(unsigned int *state)
{
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x == 0U ? 0x6d2b79f5U : x;
    return *state;
}

__device__ __forceinline__ float hash_scalar01(
    unsigned int seed, int row, int col)
{
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

__device__ __forceinline__ float group4_reduce_sum_f32(float value, int group)
{
    unsigned int mask = 0xFU << (group * AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 2, AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 1, AW_T4_GROUP_SIZE);
    return value;
}

__device__ __forceinline__ double group4_reduce_sum_f64(double value, int group)
{
    unsigned int mask = 0xFU << (group * AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 2, AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 1, AW_T4_GROUP_SIZE);
    return value;
}

// #########################################################################
// ##  Pack Kernel - column-major layout -> packed[n][ceil(k/32)]         ##
// #########################################################################

__global__ void _pack_kern(
    const int8_t *__restrict__ B,
    uint32_t *__restrict__ packed,
    int k, int n, int n_words)
{
    int word = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int col = (int)blockIdx.y;
    if (word >= n_words || col >= n)
        return;

    int base = word << 5;
    uint32_t bits = 0U;
#pragma unroll
    for (int b = 0; b < 32; ++b)
    {
        int j = base + b;
        if (j < k && __ldg(&B[(size_t)j * n + col]) != 0)
        {
            bits |= (1U << b);
        }
    }
    packed[(size_t)col * n_words + word] = bits;
}

// #########################################################################
// ##  AW-Light T4 Notrans                                  ##
// ##  Y[row, col] = sum_j w(row,j) * active(B[j, col])                  ##
// #########################################################################

__global__ void _notrans_f32_kern(
    const float *__restrict__ weight,
    const int *__restrict__ clen,
    const int *__restrict__ seed,
    const uint32_t *__restrict__ packed,
    float *__restrict__ output,
    int m, int k, int n, int n_words, int chunk_size, int n_chunks)
{
    int lane = threadIdx.x & 31;
    int sub_lane = lane & (AW_T4_GROUP_SIZE - 1);
    int group = lane >> 2;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int chunk_id = (int)blockIdx.y;
    int col = (int)blockIdx.z;
    int warp_task = (int)blockIdx.x * warps_per_block + warp_id;
    int row = warp_task * AW_T4_GROUPS_PER_WARP + group;
    if (row >= m || chunk_id >= n_chunks || col >= n)
        return;

    int chunk_start = chunk_id * chunk_size;
    if (chunk_start >= k)
        return;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > k)
        chunk_end = k;
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);
    if (chunk_width == 0U)
        return;

    float w = READ_F32(__ldg(&weight[0]));

    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U)
        cl = 2U;
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);

    const uint32_t *packed_col = packed + (size_t)col * n_words;

    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    float acc = 0.0f;
    while (local_j < chunk_width)
    {
        int j = chunk_start + (int)local_j;
        if (IS_ACTIVE_PACKED(packed_col, j))
        {
                        acc += w;
        }
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    }

    float row_acc = group4_reduce_sum_f32(acc, group);
    if (sub_lane == 0)
    {
        atomic_add_f32(&output[(size_t)row * n + col], row_acc);
    }
}

// #########################################################################
// ##  AW-Light T4 Trans                                ##
// ##  For active B[row, col], write w(row,j) to Y[j, col]              ##
// #########################################################################

__global__ void _trans_f32_kern(
    const float *__restrict__ weight,
    const int *__restrict__ clen,
    const int *__restrict__ seed,
    const uint32_t *__restrict__ packed,
    float *__restrict__ output,
    int m, int k, int n, int n_words, int chunk_size, int n_chunks)
{
    int lane = threadIdx.x & 31;
    int sub_lane = lane & (AW_T4_GROUP_SIZE - 1);
    int group = lane >> 2;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int chunk_id = (int)blockIdx.y;
    int col = (int)blockIdx.z;
    int warp_task = (int)blockIdx.x * warps_per_block + warp_id;
    int row = warp_task * AW_T4_GROUPS_PER_WARP + group;
    if (row >= m || chunk_id >= n_chunks || col >= n)
        return;

    const uint32_t *packed_col = packed + (size_t)col * n_words;
    if (!IS_ACTIVE_PACKED(packed_col, row))
        return;

    int chunk_start = chunk_id * chunk_size;
    if (chunk_start >= k)
        return;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > k)
        chunk_end = k;
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);
    if (chunk_width == 0U)
        return;

    float w = READ_F32(__ldg(&weight[0]));

    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U)
        cl = 2U;
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);

    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    while (local_j < chunk_width)
    {
        int j = chunk_start + (int)local_j;
        atomic_add_f32(&output[(size_t)j * n + col], w);
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    }
}

// #########################################################################
// ##  FFI Wrappers                                                        ##
// #########################################################################

// @BE pack
void pack(
    const BE::Tensor B,
    BE::Tensor packed,
    int k, int n, int n_words,
    int64_t stream)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (n_words <= 0 || n <= 0)
        return;

    int threads = 256;
    dim3 blocks((unsigned int)((n_words + threads - 1) / threads),
                (unsigned int)n, 1U);

    _pack_kern<<<blocks, threads, 0, s>>>(
        static_cast<const int8_t *>(B.data_ptr()),
        static_cast<uint32_t *>(packed.data_ptr()),
        k, n, n_words);
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE notrans_f32
void notrans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor packed,
    BE::Tensor output,
    int m, int k, int n, int n_words, int chunk_size,
    int64_t stream)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (m <= 0 || n <= 0)
        return;
    BE_CUDA_CHECK(cudaMemsetAsync(
        output.data_ptr(), 0, (size_t)m * n * sizeof(float), s));
    if (k <= 0 || chunk_size <= 0)
        return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0)
        return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int rows_per_block = warps_per_block * AW_T4_GROUPS_PER_WARP;
    int row_group_blocks = (m + rows_per_block - 1) / rows_per_block;
    if (row_group_blocks > 2147483647 || n_chunks > 65535 || n > 65535)
    {
        fprintf(stderr,
                "notrans_f32 grid overflow: "
                "row_group_blocks=%d n_chunks=%d n=%d\n",
                row_group_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_group_blocks,
                (unsigned int)n_chunks,
                (unsigned int)n);

    _notrans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float *>(weight.data_ptr()),
        static_cast<const int *>(clen.data_ptr()),
        static_cast<const int *>(seed.data_ptr()),
        static_cast<const uint32_t *>(packed.data_ptr()),
        static_cast<float *>(output.data_ptr()),
        m, k, n, n_words, chunk_size, n_chunks);
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE trans_f32
void trans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor packed,
    BE::Tensor output,
    int m, int k, int n, int n_words, int chunk_size,
    int64_t stream)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (m <= 0 || n <= 0)
        return;
    BE_CUDA_CHECK(cudaMemsetAsync(
        output.data_ptr(), 0, (size_t)k * n * sizeof(float), s));
    if (k <= 0 || chunk_size <= 0)
        return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0)
        return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int rows_per_block = warps_per_block * AW_T4_GROUPS_PER_WARP;
    int row_group_blocks = (m + rows_per_block - 1) / rows_per_block;
    if (row_group_blocks > 2147483647 || n_chunks > 65535 || n > 65535)
    {
        fprintf(stderr,
                "trans_f32 grid overflow: "
                "row_group_blocks=%d n_chunks=%d n=%d\n",
                row_group_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_group_blocks,
                (unsigned int)n_chunks,
                (unsigned int)n);

    _trans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float *>(weight.data_ptr()),
        static_cast<const int *>(clen.data_ptr()),
        static_cast<const int *>(seed.data_ptr()),
        static_cast<const uint32_t *>(packed.data_ptr()),
        static_cast<float *>(output.data_ptr()),
        m, k, n, n_words, chunk_size, n_chunks);
    BE_CHECK_KERNEL_LAUNCH();
}

#define DEFINE_BINARY_JITSMM_NOTRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD, GROUP_REDUCE) \
__global__ void _notrans##SFX##_kern(                                                        \
    const WEIGHT_T* __restrict__ weight,                                                      \
    const int*      __restrict__ clen,                                                        \
    const int*      __restrict__ seed,                                                        \
    const uint32_t* __restrict__ packed,                                                      \
    WEIGHT_T*       __restrict__ output,                                                      \
    int m, int k, int n, int n_words, int chunk_size, int n_chunks                            \
) {                                                                                          \
    int lane = threadIdx.x & 31;                                                              \
    int sub_lane = lane & (AW_T4_GROUP_SIZE - 1);                                             \
    int group = lane >> 2;                                                                    \
    int warp_id = threadIdx.x >> 5;                                                           \
    int warps_per_block = blockDim.x >> 5;                                                    \
    int chunk_id = (int)blockIdx.y;                                                           \
    int col = (int)blockIdx.z;                                                                \
    int warp_task = (int)blockIdx.x * warps_per_block + warp_id;                              \
    int row = warp_task * AW_T4_GROUPS_PER_WARP + group;                                      \
    if (row >= m || chunk_id >= n_chunks || col >= n) return;                                 \
    int chunk_start = chunk_id * chunk_size;                                                  \
    if (chunk_start >= k) return;                                                             \
    int chunk_end = chunk_start + chunk_size;                                                 \
    if (chunk_end > k) chunk_end = k;                                                         \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);                       \
    if (chunk_width == 0U) return;                                                            \
    ACC_T w = READ_W(__ldg(&weight[0]));                                                      \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                          \
    if (cl < 2U) cl = 2U;                                                                     \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);                                       \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);                    \
    const uint32_t* packed_col = packed + (size_t)col * n_words;                              \
    unsigned int q = stationary_initial_q(&rng, cl);                                          \
    unsigned int local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;                     \
    ACC_T acc = (ACC_T)0;                                                                     \
    while (local_j < chunk_width) {                                                           \
        int j = chunk_start + (int)local_j;                                                   \
        if (IS_ACTIVE_PACKED(packed_col, j)) acc += w;                                        \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);                            \
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;                              \
    }                                                                                         \
    ACC_T row_acc = GROUP_REDUCE(acc, group);                                                 \
    if (sub_lane == 0) ATOMIC_ADD(&output[(size_t)row * n + col], row_acc);                   \
}

#define DEFINE_BINARY_JITSMM_TRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD)    \
__global__ void _trans##SFX##_kern(                                             \
    const WEIGHT_T* __restrict__ weight,                                        \
    const int*      __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const uint32_t* __restrict__ packed,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n, int n_words, int chunk_size, int n_chunks              \
) {                                                                            \
    int lane = threadIdx.x & 31;                                                \
    int sub_lane = lane & (AW_T4_GROUP_SIZE - 1);                               \
    int group = lane >> 2;                                                      \
    int warp_id = threadIdx.x >> 5;                                             \
    int warps_per_block = blockDim.x >> 5;                                      \
    int chunk_id = (int)blockIdx.y;                                             \
    int col = (int)blockIdx.z;                                                  \
    int warp_task = (int)blockIdx.x * warps_per_block + warp_id;                \
    int row = warp_task * AW_T4_GROUPS_PER_WARP + group;                        \
    if (row >= m || chunk_id >= n_chunks || col >= n) return;                   \
    const uint32_t* packed_col = packed + (size_t)col * n_words;                \
    if (!IS_ACTIVE_PACKED(packed_col, row)) return;                             \
    int chunk_start = chunk_id * chunk_size;                                    \
    if (chunk_start >= k) return;                                               \
    int chunk_end = chunk_start + chunk_size;                                   \
    if (chunk_end > k) chunk_end = k;                                           \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);         \
    if (chunk_width == 0U) return;                                              \
    ACC_T w = READ_W(__ldg(&weight[0]));                                        \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                            \
    if (cl < 2U) cl = 2U;                                                       \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);                         \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);      \
    unsigned int q = stationary_initial_q(&rng, cl);                            \
    unsigned int local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;       \
    while (local_j < chunk_width) {                                             \
        int j = chunk_start + (int)local_j;                                     \
        ATOMIC_ADD(&output[(size_t)j * n + col], w);                            \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);              \
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;                \
    }                                                                           \
}

DEFINE_BINARY_JITSMM_NOTRANS(_f64, double, double, READ_F64, atomic_add_f64, group4_reduce_sum_f64)
DEFINE_BINARY_JITSMM_NOTRANS(_f16, __half, float, READ_F16, atomic_add_f16, group4_reduce_sum_f32)
DEFINE_BINARY_JITSMM_NOTRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16, group4_reduce_sum_f32)

DEFINE_BINARY_JITSMM_TRANS(_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BINARY_JITSMM_TRANS(_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BINARY_JITSMM_TRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

#define FFI_BINARY_JITSMM(NAME, KERNEL, SFX, WEIGHT_T, OUT_ROWS)             \
void NAME##SFX(                                                              \
    const BE::Tensor weight,                                                 \
    const BE::Tensor clen,                                                   \
    const BE::Tensor seed,                                                   \
    const BE::Tensor packed,                                                 \
    BE::Tensor output,                                                       \
    int m, int k, int n, int n_words, int chunk_size,                        \
    int64_t stream                                                           \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    if (m <= 0 || n <= 0) return;                                            \
    BE_CUDA_CHECK(cudaMemsetAsync(                                           \
        output.data_ptr(), 0, (size_t)OUT_ROWS * n * sizeof(WEIGHT_T), s));  \
    if (k <= 0 || chunk_size <= 0) return;                                   \
    int n_chunks = (k + chunk_size - 1) / chunk_size;                        \
    if (n_chunks <= 0) return;                                               \
    int threads = 256;                                                       \
    int warps_per_block = threads / 32;                                      \
    int rows_per_block = warps_per_block * AW_T4_GROUPS_PER_WARP;            \
    int row_group_blocks = (m + rows_per_block - 1) / rows_per_block;        \
    if (row_group_blocks > 2147483647 || n_chunks > 65535 || n > 65535) {    \
        fprintf(stderr, #NAME #SFX " grid overflow\n");                     \
        abort();                                                             \
    }                                                                        \
    dim3 blocks((unsigned int)row_group_blocks,                              \
                (unsigned int)n_chunks,                                      \
                (unsigned int)n);                                            \
    _##KERNEL##SFX##_kern<<<blocks, threads, 0, s>>>(                        \
        static_cast<const WEIGHT_T*>(weight.data_ptr()),                     \
        static_cast<const int*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                            \
        static_cast<const uint32_t*>(packed.data_ptr()),                     \
        static_cast<WEIGHT_T*>(output.data_ptr()),                           \
        m, k, n, n_words, chunk_size, n_chunks                               \
    );                                                                       \
    BE_CHECK_KERNEL_LAUNCH();                                                \
}

// @BE notrans_f64
FFI_BINARY_JITSMM(notrans, notrans, _f64, double, m)
// @BE notrans_f16
FFI_BINARY_JITSMM(notrans, notrans, _f16, __half, m)
// @BE notrans_bf16
FFI_BINARY_JITSMM(notrans, notrans, _bf16, __nv_bfloat16, m)

// @BE trans_f64
FFI_BINARY_JITSMM(trans, trans, _f64, double, k)
// @BE trans_f16
FFI_BINARY_JITSMM(trans, trans, _f16, __half, k)
// @BE trans_bf16
FFI_BINARY_JITSMM(trans, trans, _bf16, __nv_bfloat16, k)
