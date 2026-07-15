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
 * float_jitsmm.cu -- dense-matrix light-RNG backends.
 *
 * jitsmm_gather/scatter_f32 use the AW-T4 MM random stream and match CSR
 * matrix_mode="mm".  jitsmm_mv_gather/scatter_f32 use the MV random stream
 * column-by-column and are only used by vmap(jitsmv).
 */

#include <cstdio>
#include <cstdlib>

#include "cuda_common.h"
#include "brainevent/common.h"

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

__device__ __forceinline__ float hash_scalar01(
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

__device__ __forceinline__ unsigned int calibrated_chunk_clen(
    unsigned int cl,
    int k,
    int chunk_size,
    int n_chunks
) {
    if (cl < 2U) cl = 2U;
    if (k <= 0 || chunk_size <= 0 || n_chunks <= 0) return cl;

    int full_chunks = k / chunk_size;
    int tail = k - full_chunks * chunk_size;
    if (full_chunks > n_chunks) {
        full_chunks = n_chunks;
        tail = 0;
    }

    unsigned int full_streams =
        (chunk_size < 32) ? (unsigned int)chunk_size : 32U;
    unsigned long long stream_count =
        (unsigned long long)full_chunks * (unsigned long long)full_streams;
    if (tail > 0 && full_chunks < n_chunks) {
        stream_count += (unsigned long long)((tail < 32) ? tail : 32);
    }
    if (stream_count == 0ULL) return cl;

    float width2 = 2.0f * (float)k;
    float target = width2 / (float)cl;
    float corrected = target + (float)stream_count * (1.0f / 3.0f);
    if (!(corrected > 0.0f)) return cl;

    unsigned int eff = (unsigned int)(width2 / corrected + 0.5f);
    if (eff < 2U) eff = 2U;
    if (eff > cl) eff = cl;
    return eff;
}

__device__ __forceinline__ unsigned int calibrated_chunk_clen_t4(
    unsigned int cl,
    int k,
    int chunk_size,
    int n_chunks
) {
    if (cl < 2U) cl = 2U;
    if (k <= 0 || chunk_size <= 0 || n_chunks <= 0) return cl;

    int full_chunks = k / chunk_size;
    int tail = k - full_chunks * chunk_size;
    if (full_chunks > n_chunks) {
        full_chunks = n_chunks;
        tail = 0;
    }

    unsigned int full_streams =
        (chunk_size < AW_T4_GROUP_SIZE) ? (unsigned int)chunk_size : AW_T4_GROUP_SIZE;
    unsigned long long stream_count =
        (unsigned long long)full_chunks * (unsigned long long)full_streams;
    if (tail > 0 && full_chunks < n_chunks) {
        stream_count +=
            (unsigned long long)((tail < AW_T4_GROUP_SIZE) ? tail : AW_T4_GROUP_SIZE);
    }
    if (stream_count == 0ULL) return cl;

    float width2 = 2.0f * (float)k;
    float target = width2 / (float)cl;
    float corrected = target + (float)stream_count * (1.0f / 3.0f);
    if (!(corrected > 0.0f)) return cl;

    unsigned int eff = (unsigned int)(width2 / corrected + 0.5f);
    if (eff < 2U) eff = 2U;
    if (eff > cl) eff = cl;
    return eff;
}

__device__ __forceinline__ float group4_reduce_sum_f32(float value, int group) {
    unsigned int mask = 0xFU << (group * AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 2, AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 1, AW_T4_GROUP_SIZE);
    return value;
}

__global__ void _mv_gather_f32_kern(
    const float* __restrict__ weight,
        const float* __restrict__ _w2,
    const int*   __restrict__ clen,
    const int*   __restrict__ seed,
    const float* __restrict__ B,
    float*       __restrict__ output,
    int m, int k, int n, int chunk_size, int n_chunks
) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_block = (int)blockIdx.x;
    int chunk_id = (int)blockIdx.y;
    int col_b = (int)blockIdx.z;
    int row = row_block * warps_per_block + warp_id;
    if (row >= m || chunk_id >= n_chunks || col_b >= n) return;

    int chunk_start = chunk_id * chunk_size;
    if (chunk_start >= k) return;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > k) chunk_end = k;
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);
    if (chunk_width == 0U) return;

    float w = READ_F32(__ldg(&weight[0]));
    
    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U) cl = 2U;
    cl = calibrated_chunk_clen(cl, k, chunk_size, n_chunks);
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);

    unsigned int q = fast_bounded_u32(light_rng_next(&rng), cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    float acc = 0.0f;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        acc += w * READ_F32(__ldg(&B[(size_t)j * n + col_b]));
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }

    float row_acc = warp_reduce_sum_f32(acc);
    if (lane == 0) {
        atomic_add_f32(&output[(size_t)row * n + col_b], row_acc);
    }
}

__global__ void _mv_scatter_f32_kern(
    const float* __restrict__ weight,
        const float* __restrict__ _w2,
    const int*   __restrict__ clen,
    const int*   __restrict__ seed,
    const float* __restrict__ B,
    float*       __restrict__ output,
    int m, int k, int n, int chunk_size, int n_chunks
) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_block = (int)blockIdx.x;
    int chunk_id = (int)blockIdx.y;
    int col_b = (int)blockIdx.z;
    int row = row_block * warps_per_block + warp_id;
    if (row >= m || chunk_id >= n_chunks || col_b >= n) return;

    float v = READ_F32(__ldg(&B[(size_t)row * n + col_b]));
    if (v == 0.0f) return;

    int chunk_start = chunk_id * chunk_size;
    if (chunk_start >= k) return;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > k) chunk_end = k;
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);
    if (chunk_width == 0U) return;

    float w = READ_F32(__ldg(&weight[0]));
    
    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U) cl = 2U;
    cl = calibrated_chunk_clen(cl, k, chunk_size, n_chunks);
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);

    unsigned int q = fast_bounded_u32(light_rng_next(&rng), cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        atomic_add_f32(&output[(size_t)j * n + col_b], w * v);
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }
}

__global__ void _mm_gather_f32_kern(
    const float* __restrict__ weight,
        const float* __restrict__ _w2,
    const int*   __restrict__ clen,
    const int*   __restrict__ seed,
    const float* __restrict__ B,
    float*       __restrict__ output,
    int m, int k, int n, int chunk_size, int n_chunks
) {
    int lane = threadIdx.x & 31;
    int sub_lane = lane & (AW_T4_GROUP_SIZE - 1);
    int group = lane >> 2;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int chunk_id = (int)blockIdx.y;
    int col_b = (int)blockIdx.z;
    int warp_task = (int)blockIdx.x * warps_per_block + warp_id;
    int row = warp_task * AW_T4_GROUPS_PER_WARP + group;
    if (row >= m || chunk_id >= n_chunks || col_b >= n) return;

    int chunk_start = chunk_id * chunk_size;
    if (chunk_start >= k) return;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > k) chunk_end = k;
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);
    if (chunk_width == 0U) return;

    float w = READ_F32(__ldg(&weight[0]));
    
    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U) cl = 2U;
    cl = calibrated_chunk_clen_t4(cl, k, chunk_size, n_chunks);
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);

    unsigned int q = fast_bounded_u32(light_rng_next(&rng), cl);
    unsigned int local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    float acc = 0.0f;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        acc += w * READ_F32(__ldg(&B[(size_t)j * n + col_b]));
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    }

    float row_acc = group4_reduce_sum_f32(acc, group);
    if (sub_lane == 0) {
        atomic_add_f32(&output[(size_t)row * n + col_b], row_acc);
    }
}

__global__ void _mm_scatter_f32_kern(
    const float* __restrict__ weight,
        const float* __restrict__ _w2,
    const int*   __restrict__ clen,
    const int*   __restrict__ seed,
    const float* __restrict__ B,
    float*       __restrict__ output,
    int m, int k, int n, int chunk_size, int n_chunks
) {
    int lane = threadIdx.x & 31;
    int sub_lane = lane & (AW_T4_GROUP_SIZE - 1);
    int group = lane >> 2;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int chunk_id = (int)blockIdx.y;
    int col_b = (int)blockIdx.z;
    int warp_task = (int)blockIdx.x * warps_per_block + warp_id;
    int row = warp_task * AW_T4_GROUPS_PER_WARP + group;
    if (row >= m || chunk_id >= n_chunks || col_b >= n) return;

    float v = READ_F32(__ldg(&B[(size_t)row * n + col_b]));
    if (v == 0.0f) return;

    int chunk_start = chunk_id * chunk_size;
    if (chunk_start >= k) return;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > k) chunk_end = k;
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);
    if (chunk_width == 0U) return;

    float w = READ_F32(__ldg(&weight[0]));
    
    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U) cl = 2U;
    cl = calibrated_chunk_clen_t4(cl, k, chunk_size, n_chunks);
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);

    unsigned int q = fast_bounded_u32(light_rng_next(&rng), cl);
    unsigned int local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        atomic_add_f32(&output[(size_t)j * n + col_b], w * v);
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    }
}

static void launch_mv_gather_f32(
    const BE::Tensor weight,
    const BE::Tensor _w2,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (m <= 0 || n <= 0) return;
    BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, (size_t)m * n * sizeof(float), s));
    if (k <= 0 || chunk_size <= 0) return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535 || n > 65535) {
        fprintf(stderr,
                "jitsmm_mv_gather_f32 grid overflow: row_warp_blocks=%d n_chunks=%d n=%d\n",
                row_warp_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, (unsigned int)n);

    _mv_gather_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(B.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, n, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

static void launch_mv_scatter_f32(
    const BE::Tensor weight,
    const BE::Tensor _w2,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (k <= 0 || n <= 0) return;
    BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, (size_t)k * n * sizeof(float), s));
    if (m <= 0 || chunk_size <= 0) return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535 || n > 65535) {
        fprintf(stderr,
                "jitsmm_mv_scatter_f32 grid overflow: row_warp_blocks=%d n_chunks=%d n=%d\n",
                row_warp_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, (unsigned int)n);

    _mv_scatter_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(B.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, n, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

static void launch_mm_gather_f32(
    const BE::Tensor weight,
    const BE::Tensor _w2,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (m <= 0 || n <= 0) return;
    BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, (size_t)m * n * sizeof(float), s));
    if (k <= 0 || chunk_size <= 0) return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int rows_per_block = warps_per_block * AW_T4_GROUPS_PER_WARP;
    int row_group_blocks = (m + rows_per_block - 1) / rows_per_block;
    if (row_group_blocks > 2147483647 || n_chunks > 65535 || n > 65535) {
        fprintf(stderr,
                "jitsmm_gather_f32 grid overflow: row_group_blocks=%d n_chunks=%d n=%d\n",
                row_group_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_group_blocks, (unsigned int)n_chunks, (unsigned int)n);

    _mm_gather_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(B.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, n, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

static void launch_mm_scatter_f32(
    const BE::Tensor weight,
    const BE::Tensor _w2,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (k <= 0 || n <= 0) return;
    BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, (size_t)k * n * sizeof(float), s));
    if (m <= 0 || chunk_size <= 0) return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int rows_per_block = warps_per_block * AW_T4_GROUPS_PER_WARP;
    int row_group_blocks = (m + rows_per_block - 1) / rows_per_block;
    if (row_group_blocks > 2147483647 || n_chunks > 65535 || n > 65535) {
        fprintf(stderr,
                "jitsmm_scatter_f32 grid overflow: row_group_blocks=%d n_chunks=%d n=%d\n",
                row_group_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_group_blocks, (unsigned int)n_chunks, (unsigned int)n);

    _mm_scatter_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(B.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, n, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE jitsmm_mv_gather_f32
void jitsmm_mv_gather_f32(
    const BE::Tensor weight,
    const BE::Tensor _w2,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    launch_mv_gather_f32(weight, weight, clen, seed, B, output, m, k, n, chunk_size, stream);
}

// @BE jitsmm_mv_scatter_f32
void jitsmm_mv_scatter_f32(
    const BE::Tensor weight,
    const BE::Tensor _w2,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    launch_mv_scatter_f32(weight, weight, clen, seed, B, output, m, k, n, chunk_size, stream);
}

// @BE jitsmm_gather_f32
void jitsmm_gather_f32(
    const BE::Tensor weight,
    const BE::Tensor _w2,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    launch_mm_gather_f32(weight, weight, clen, seed, B, output, m, k, n, chunk_size, stream);
}

// @BE jitsmm_scatter_f32
void jitsmm_scatter_f32(
    const BE::Tensor weight,
    const BE::Tensor _w2,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    launch_mm_scatter_f32(weight, weight, clen, seed, B, output, m, k, n, chunk_size, stream);
}
