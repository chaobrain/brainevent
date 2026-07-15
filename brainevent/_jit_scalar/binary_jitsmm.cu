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
 * Both gather (corder=True) and scatter (corder=False) are provided.
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

__device__ __forceinline__ unsigned int calibrated_chunk_clen_t4(
    unsigned int cl, int k, int chunk_size, int n_chunks)
{
    if (cl < 2U)
        cl = 2U;
    if (k <= 0 || chunk_size <= 0 || n_chunks <= 0)
        return cl;

    int full_chunks = k / chunk_size;
    int tail = k - full_chunks * chunk_size;
    if (full_chunks > n_chunks)
    {
        full_chunks = n_chunks;
        tail = 0;
    }

    unsigned int full_streams =
        (chunk_size < AW_T4_GROUP_SIZE) ? (unsigned int)chunk_size : AW_T4_GROUP_SIZE;
    unsigned long long stream_count =
        (unsigned long long)full_chunks * (unsigned long long)full_streams;
    if (tail > 0 && full_chunks < n_chunks)
        stream_count +=
            (unsigned long long)((tail < AW_T4_GROUP_SIZE) ? tail : AW_T4_GROUP_SIZE);
    if (stream_count == 0ULL)
        return cl;

    float width2 = 2.0f * (float)k;
    float target = width2 / (float)cl;
    float corrected = target + (float)stream_count * (1.0f / 3.0f);
    if (!(corrected > 0.0f))
        return cl;

    unsigned int eff = (unsigned int)(width2 / corrected + 0.5f);
    if (eff < 2U)
        eff = 2U;
    if (eff > cl)
        eff = cl;
    return eff;
}

__device__ __forceinline__ float group4_reduce_sum_f32(float value, int group)
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
// ##  AW-Light T4 Gather  (corder=True)                                  ##
// ##  Y[row, col] = sum_j w(row,j) * active(B[j, col])                  ##
// #########################################################################

__global__ void _gather_f32_kern(
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
    cl = calibrated_chunk_clen_t4(cl, k, chunk_size, n_chunks);
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);

    const uint32_t *packed_col = packed + (size_t)col * n_words;

    unsigned int q = fast_bounded_u32(light_rng_next(&rng), cl);
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
// ##  AW-Light T4 Scatter  (corder=False)                                ##
// ##  For active B[row, col], scatter w(row,j) to Y[j, col]              ##
// #########################################################################

__global__ void _scatter_f32_kern(
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
    cl = calibrated_chunk_clen_t4(cl, k, chunk_size, n_chunks);
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);

    unsigned int q = fast_bounded_u32(light_rng_next(&rng), cl);
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

// @BE gather_f32
void gather_f32(
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
                "gather_f32 grid overflow: "
                "row_group_blocks=%d n_chunks=%d n=%d\n",
                row_group_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_group_blocks,
                (unsigned int)n_chunks,
                (unsigned int)n);

    _gather_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float *>(weight.data_ptr()),
        static_cast<const int *>(clen.data_ptr()),
        static_cast<const int *>(seed.data_ptr()),
        static_cast<const uint32_t *>(packed.data_ptr()),
        static_cast<float *>(output.data_ptr()),
        m, k, n, n_words, chunk_size, n_chunks);
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE scatter_f32
void scatter_f32(
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
                "scatter_f32 grid overflow: "
                "row_group_blocks=%d n_chunks=%d n=%d\n",
                row_group_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_group_blocks,
                (unsigned int)n_chunks,
                (unsigned int)n);

    _scatter_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float *>(weight.data_ptr()),
        static_cast<const int *>(clen.data_ptr()),
        static_cast<const int *>(seed.data_ptr()),
        static_cast<const uint32_t *>(packed.data_ptr()),
        static_cast<float *>(output.data_ptr()),
        m, k, n, n_words, chunk_size, n_chunks);
    BE_CHECK_KERNEL_LAUNCH();
}
