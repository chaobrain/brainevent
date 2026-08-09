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
 * binary_jitnmm.cu — stable binary MM AW-T4 backend with Normal weights (light RNG).
 *
 * MM = repeated MV: each column is an independent MV.
 *   - packed layout: [n][ceil(k/32)] column-major bitmask.
 *   - Grid 3D: (row_warp_blocks, n_chunks, n).
 *   - One warp = 8 independent 4-thread (row, chunk_id, col) slices.
 *
 * Both notrans and trans kernels are provided.
 *
 * Restrictions: f32 weights, bool/int8 spikes, n <= 32.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "cuda_common.h"
#include "brainevent/common.h"

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

__device__ __forceinline__ float hash_uniform01(
    unsigned int seed, int row, int col)
{
    unsigned int h = seed ^ 0xa0761d65U;
    h ^= (unsigned int)row * 0xe7037ed1U;
    h ^= (unsigned int)col * 0x8ebc6af1U;
    h = mix32(h);
    return (float)(h & 0x00ffffffU) * (1.0f / 16777216.0f);
}

// =========================================================================
// Acklam inverse normal CDF (probit) — rational approximation
// =========================================================================
__device__ __forceinline__ float hash_normal01(
    unsigned int seed, int row, int col)
{
    float u = hash_uniform01(seed, row, col);
    u = fmaxf(fminf(u, 1.0f - 1e-10f), 1e-10f);

    const float a1 = -39.696830f, a2 = 220.94609f, a3 = -275.92851f;
    const float a4 = 138.35775f, a5 = -30.664799f, a6 = 2.5066283f;
    const float b1 = -54.476099f, b2 = 161.58584f, b3 = -155.69898f;
    const float b4 = 66.801312f, b5 = -13.280681f;

    const float c1 = -0.007784894f, c2 = -0.32239646f, c3 = -2.4007583f;
    const float c4 = -2.5497325f, c5 = 4.3746641f, c6 = 2.9381640f;
    const float d1 = 0.007784696f, d2 = 0.32246713f, d3 = 2.4451342f;
    const float d4 = 3.7544087f;

    float z;
    if (u < 0.02425f) {
        float v = sqrtf(-2.0f * logf(u));
        z = (((((c1 * v + c2) * v + c3) * v + c4) * v + c5) * v + c6) /
            ((((d1 * v + d2) * v + d3) * v + d4) * v + 1.0f);
        z = -z;
    } else if (u > 0.97575f) {
        float v = sqrtf(-2.0f * logf(1.0f - u));
        z = (((((c1 * v + c2) * v + c3) * v + c4) * v + c5) * v + c6) /
            ((((d1 * v + d2) * v + d3) * v + d4) * v + 1.0f);
    } else {
        float v = u - 0.5f;
        float r = v * v;
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * v /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0f);
    }
    return z;
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
    const float *__restrict__ w_loc,
    const float *__restrict__ w_scale,
    const int *__restrict__ clen,
    const int *__restrict__ seed,
    const uint32_t *__restrict__ packed,
    float *__restrict__ output,
    int m, int k, int n, int n_words, int chunk_size, int n_chunks)
{
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int chunk_id = (int)blockIdx.y;
    int col = (int)blockIdx.z;
    int row_block = (int)blockIdx.x;
    int row = row_block * warps_per_block + warp_id;
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

    float loc = READ_F32(__ldg(&w_loc[0]));
    float scale = READ_F32(__ldg(&w_scale[0]));
    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U)
        cl = 2U;
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);

    const uint32_t *packed_col = packed + (size_t)col * n_words;

    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    float acc = 0.0f;
    while (local_j < chunk_width)
    {
        int j = chunk_start + (int)local_j;
        if (IS_ACTIVE_PACKED(packed_col, j))
        {
            float n01 = hash_normal01(seed0, row, j);
            acc += loc + n01 * scale;
        }
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }

    float row_acc = warp_reduce_sum_f32(acc);
    if (lane == 0)
    {
        atomic_add_f32(&output[(size_t)row * n + col], row_acc);
    }
}

// #########################################################################
// ##  AW-Light T4 Trans                                ##
// ##  For active B[row, col], write w(row,j) to Y[j, col]              ##
// #########################################################################

__global__ void _trans_f32_kern(
    const float *__restrict__ w_loc,
    const float *__restrict__ w_scale,
    const int *__restrict__ clen,
    const int *__restrict__ seed,
    const uint32_t *__restrict__ packed,
    float *__restrict__ output,
    int m, int k, int n, int n_words, int chunk_size, int n_chunks)
{
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int chunk_id = (int)blockIdx.y;
    int col = (int)blockIdx.z;
    int row_block = (int)blockIdx.x;
    int row = row_block * warps_per_block + warp_id;
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

    float loc = READ_F32(__ldg(&w_loc[0]));
    float scale = READ_F32(__ldg(&w_scale[0]));
    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U)
        cl = 2U;
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);

    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    while (local_j < chunk_width)
    {
        int j = chunk_start + (int)local_j;
        float n01 = hash_normal01(seed0, row, j);
        float w = loc + n01 * scale;
        atomic_add_f32(&output[(size_t)j * n + col], w);
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
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
    const BE::Tensor w_loc,
    const BE::Tensor w_scale,
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
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535 || n > 65535)
    {
        fprintf(stderr,
                "notrans_f32 grid overflow: "
                "row_warp_blocks=%d n_chunks=%d n=%d\n",
                row_warp_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks,
                (unsigned int)n_chunks,
                (unsigned int)n);

    _notrans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float *>(w_loc.data_ptr()),
        static_cast<const float *>(w_scale.data_ptr()),
        static_cast<const int *>(clen.data_ptr()),
        static_cast<const int *>(seed.data_ptr()),
        static_cast<const uint32_t *>(packed.data_ptr()),
        static_cast<float *>(output.data_ptr()),
        m, k, n, n_words, chunk_size, n_chunks);
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE trans_f32
void trans_f32(
    const BE::Tensor w_loc,
    const BE::Tensor w_scale,
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
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535 || n > 65535)
    {
        fprintf(stderr,
                "trans_f32 grid overflow: "
                "row_warp_blocks=%d n_chunks=%d n=%d\n",
                row_warp_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks,
                (unsigned int)n_chunks,
                (unsigned int)n);

    _trans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float *>(w_loc.data_ptr()),
        static_cast<const float *>(w_scale.data_ptr()),
        static_cast<const int *>(clen.data_ptr()),
        static_cast<const int *>(seed.data_ptr()),
        static_cast<const uint32_t *>(packed.data_ptr()),
        static_cast<float *>(output.data_ptr()),
        m, k, n, n_words, chunk_size, n_chunks);
    BE_CHECK_KERNEL_LAUNCH();
}

#define DEFINE_NOTRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD, WARP_REDUCE) \
__global__ void _notrans##SFX##_kern(                                          \
    const WEIGHT_T *__restrict__ w_loc,                                        \
    const WEIGHT_T *__restrict__ w_scale,                                      \
    const int *__restrict__ clen,                                              \
    const int *__restrict__ seed,                                              \
    const uint32_t *__restrict__ packed,                                       \
    WEIGHT_T *__restrict__ output,                                             \
    int m, int k, int n, int n_words, int chunk_size, int n_chunks)            \
{                                                                             \
    int lane = threadIdx.x & 31;                                               \
    int warp_id = threadIdx.x >> 5;                                            \
    int warps_per_block = blockDim.x >> 5;                                     \
    int chunk_id = (int)blockIdx.y;                                            \
    int col = (int)blockIdx.z;                                                 \
    int row_block = (int)blockIdx.x;               \
    int row = row_block * warps_per_block + warp_id;                       \
    if (row >= m || chunk_id >= n_chunks || col >= n) return;                  \
    int chunk_start = chunk_id * chunk_size;                                   \
    if (chunk_start >= k) return;                                              \
    int chunk_end = chunk_start + chunk_size;                                  \
    if (chunk_end > k) chunk_end = k;                                          \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);        \
    if (chunk_width == 0U) return;                                             \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                      \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                  \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                           \
    if (cl < 2U) cl = 2U;                                                      \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);                        \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);     \
    const uint32_t *packed_col = packed + (size_t)col * n_words;               \
    unsigned int q = stationary_initial_q(&rng, cl);                           \
    unsigned int local_j = (unsigned int)lane + 32U * q;      \
    ACC_T acc = (ACC_T)0;                                                      \
    while (local_j < chunk_width) {                                            \
        int j = chunk_start + (int)local_j;                                    \
        if (IS_ACTIVE_PACKED(packed_col, j)) {                                 \
            float n01 = hash_normal01(seed0, row, j);                          \
            acc += loc + (ACC_T)n01 * scale;                                   \
        }                                                                      \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);             \
        local_j = (unsigned int)lane + 32U * q;               \
    }                                                                          \
    ACC_T row_acc = WARP_REDUCE(acc);                                  \
    if (lane == 0) {                                                       \
        ATOMIC_ADD(&output[(size_t)row * n + col], row_acc);                   \
    }                                                                          \
}

#define DEFINE_TRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD)                \
__global__ void _trans##SFX##_kern(                                           \
    const WEIGHT_T *__restrict__ w_loc,                                       \
    const WEIGHT_T *__restrict__ w_scale,                                     \
    const int *__restrict__ clen,                                             \
    const int *__restrict__ seed,                                             \
    const uint32_t *__restrict__ packed,                                      \
    WEIGHT_T *__restrict__ output,                                            \
    int m, int k, int n, int n_words, int chunk_size, int n_chunks)           \
{                                                                            \
    int lane = threadIdx.x & 31;                                              \
    int warp_id = threadIdx.x >> 5;                                           \
    int warps_per_block = blockDim.x >> 5;                                    \
    int chunk_id = (int)blockIdx.y;                                           \
    int col = (int)blockIdx.z;                                                \
    int row_block = (int)blockIdx.x;              \
    int row = row_block * warps_per_block + warp_id;                      \
    if (row >= m || chunk_id >= n_chunks || col >= n) return;                 \
    const uint32_t *packed_col = packed + (size_t)col * n_words;              \
    if (!IS_ACTIVE_PACKED(packed_col, row)) return;                           \
    int chunk_start = chunk_id * chunk_size;                                  \
    if (chunk_start >= k) return;                                             \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > k) chunk_end = k;                                         \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);       \
    if (chunk_width == 0U) return;                                            \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                     \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                 \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                          \
    if (cl < 2U) cl = 2U;                                                     \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);                       \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);    \
    unsigned int q = stationary_initial_q(&rng, cl);                          \
    unsigned int local_j = (unsigned int)lane + 32U * q;     \
    while (local_j < chunk_width) {                                           \
        int j = chunk_start + (int)local_j;                                   \
        float n01 = hash_normal01(seed0, row, j);                             \
        ACC_T w = loc + (ACC_T)n01 * scale;                                   \
        ATOMIC_ADD(&output[(size_t)j * n + col], w);                          \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);            \
        local_j = (unsigned int)lane + 32U * q;              \
    }                                                                         \
}

DEFINE_NOTRANS(_f64, double, double, READ_F64, atomic_add_f64, warp_reduce_sum_f64)
DEFINE_NOTRANS(_f16, __half, float, READ_F16, atomic_add_f16, warp_reduce_sum_f32)
DEFINE_NOTRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16, warp_reduce_sum_f32)

DEFINE_TRANS(_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_TRANS(_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_TRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

#define FFI_JITNMM(NAME, KERNEL, SFX, WEIGHT_T, OUT_ROWS)                    \
void NAME##SFX(                                                              \
    const BE::Tensor w_loc,                                                   \
    const BE::Tensor w_scale,                                                 \
    const BE::Tensor clen,                                                    \
    const BE::Tensor seed,                                                    \
    const BE::Tensor packed,                                                  \
    BE::Tensor output,                                                        \
    int m, int k, int n, int n_words, int chunk_size,                         \
    int64_t stream)                                                           \
{                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    if (m <= 0 || n <= 0) return;                                             \
    BE_CUDA_CHECK(cudaMemsetAsync(                                            \
        output.data_ptr(), 0, (size_t)OUT_ROWS * n * sizeof(WEIGHT_T), s));   \
    if (k <= 0 || chunk_size <= 0) return;                                    \
    int n_chunks = (k + chunk_size - 1) / chunk_size;                         \
    if (n_chunks <= 0) return;                                                \
    int threads = 256;                                                        \
    int warps_per_block = threads / 32;                                       \
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;         \
    if (row_warp_blocks > 2147483647 || n_chunks > 65535 || n > 65535) {    \
        fprintf(stderr, #NAME #SFX " grid overflow\n");                      \
        abort();                                                              \
    }                                                                         \
    dim3 blocks((unsigned int)row_warp_blocks,                               \
                (unsigned int)n_chunks,                                       \
                (unsigned int)n);                                             \
    _##KERNEL##SFX##_kern<<<blocks, threads, 0, s>>>(                         \
        static_cast<const WEIGHT_T *>(w_loc.data_ptr()),                      \
        static_cast<const WEIGHT_T *>(w_scale.data_ptr()),                    \
        static_cast<const int *>(clen.data_ptr()),                            \
        static_cast<const int *>(seed.data_ptr()),                            \
        static_cast<const uint32_t *>(packed.data_ptr()),                     \
        static_cast<WEIGHT_T *>(output.data_ptr()),                           \
        m, k, n, n_words, chunk_size, n_chunks);                              \
    BE_CHECK_KERNEL_LAUNCH();                                                 \
}

// @BE notrans_f64
FFI_JITNMM(notrans, notrans, _f64, double, m)
// @BE notrans_f16
FFI_JITNMM(notrans, notrans, _f16, __half, m)
// @BE notrans_bf16
FFI_JITNMM(notrans, notrans, _bf16, __nv_bfloat16, m)

// @BE trans_f64
FFI_JITNMM(trans, trans, _f64, double, k)
// @BE trans_f16
FFI_JITNMM(trans, trans, _f16, __half, k)
// @BE trans_bf16
FFI_JITNMM(trans, trans, _bf16, __nv_bfloat16, k)
