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
 * float_jitnmv.cu -- dense-vector light-RNG WPR chunk backend.
 *
 * The generated row-major matrix is identical to CSR matrix_mode="mv" and
 * binary_jitumv, but the right operand is a dense float vector rather than a
 * packed event mask.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "cuda_common.h"
#include "brainevent/common.h"

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
    if (u < 0.02425f)
    {
        float v = sqrtf(-2.0f * logf(u));
        z = (((((c1 * v + c2) * v + c3) * v + c4) * v + c5) * v + c6) /
            ((((d1 * v + d2) * v + d3) * v + d4) * v + 1.0f);
        z = -z;
    }
    else if (u > 0.97575f)
    {
        float v = sqrtf(-2.0f * logf(1.0f - u));
        z = (((((c1 * v + c2) * v + c3) * v + c4) * v + c5) * v + c6) /
            ((((d1 * v + d2) * v + d3) * v + d4) * v + 1.0f);
    }
    else
    {
        float v = u - 0.5f;
        float r = v * v;
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * v /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0f);
    }
    return z;
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

__global__ void _gather_f32_kern(
    const float* __restrict__ w_loc,
    const float* __restrict__ w_scale,
    const int*   __restrict__ clen,
    const int*   __restrict__ seed,
    const float* __restrict__ vector,
    float*       __restrict__ output,
    int m, int k, int chunk_size, int n_chunks
) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_block = (int)blockIdx.x;
    int chunk_id = (int)blockIdx.y;
    int row = row_block * warps_per_block + warp_id;
    if (row >= m || chunk_id >= n_chunks) return;

    int chunk_start = chunk_id * chunk_size;
    if (chunk_start >= k) return;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > k) chunk_end = k;
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);
    if (chunk_width == 0U) return;

    float loc = READ_F32(__ldg(&w_loc[0]));
    float scale = READ_F32(__ldg(&w_scale[0]));
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
        float n01 = hash_normal01(seed0, row, j);
        float w = loc + n01 * scale;
        acc += w * READ_F32(__ldg(&vector[j]));
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }

    float row_acc = warp_reduce_sum_f32(acc);
    if (lane == 0) {
        atomic_add_f32(&output[row], row_acc);
    }
}

__global__ void _scatter_f32_kern(
    const float* __restrict__ w_loc,
    const float* __restrict__ w_scale,
    const int*   __restrict__ clen,
    const int*   __restrict__ seed,
    const float* __restrict__ vector,
    float*       __restrict__ output,
    int m, int k, int chunk_size, int n_chunks
) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_block = (int)blockIdx.x;
    int chunk_id = (int)blockIdx.y;
    int row = row_block * warps_per_block + warp_id;
    if (row >= m || chunk_id >= n_chunks) return;

    float v = READ_F32(__ldg(&vector[row]));
    if (v == 0.0f) return;

    int chunk_start = chunk_id * chunk_size;
    if (chunk_start >= k) return;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > k) chunk_end = k;
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);
    if (chunk_width == 0U) return;

    float loc = READ_F32(__ldg(&w_loc[0]));
    float scale = READ_F32(__ldg(&w_scale[0]));
    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U) cl = 2U;
    cl = calibrated_chunk_clen(cl, k, chunk_size, n_chunks);
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);

    unsigned int q = fast_bounded_u32(light_rng_next(&rng), cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        float n01 = hash_normal01(seed0, row, j);
        float w = loc + n01 * scale;
        atomic_add_f32(&output[j], w * v);
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }
}

static void launch_gather_f32(
    const BE::Tensor w_loc,
    const BE::Tensor w_scale,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor vector,
    BE::Tensor output,
    int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(output.size(0));
    int k = static_cast<int>(vector.size(0));
    if (m <= 0) return;
    BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, (size_t)m * sizeof(float), s));
    if (k <= 0 || chunk_size <= 0) return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535) {
        fprintf(stderr,
                "jitnmv_gather_f32 grid overflow: row_warp_blocks=%d n_chunks=%d\n",
                row_warp_blocks, n_chunks);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, 1U);

    _gather_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(w_loc.data_ptr()),
        static_cast<const float*>(w_scale.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

static void launch_scatter_f32(
    const BE::Tensor w_loc,
    const BE::Tensor w_scale,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor vector,
    BE::Tensor output,
    int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(vector.size(0));
    int k = static_cast<int>(output.size(0));
    if (k <= 0) return;
    BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, (size_t)k * sizeof(float), s));
    if (m <= 0 || chunk_size <= 0) return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535) {
        fprintf(stderr,
                "jitnmv_scatter_f32 grid overflow: row_warp_blocks=%d n_chunks=%d\n",
                row_warp_blocks, n_chunks);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, 1U);

    _scatter_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(w_loc.data_ptr()),
        static_cast<const float*>(w_scale.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE jitnmv_gather_f32
void jitnmv_gather_f32(
    const BE::Tensor w_loc,
    const BE::Tensor w_scale,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor vector,
    BE::Tensor output,
    int chunk_size,
    int64_t stream
) {
    launch_gather_f32(w_loc, w_scale, clen, seed, vector, output, chunk_size, stream);
}

// @BE jitnmv_scatter_f32
void jitnmv_scatter_f32(
    const BE::Tensor w_loc,
    const BE::Tensor w_scale,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor vector,
    BE::Tensor output,
    int chunk_size,
    int64_t stream
) {
    launch_scatter_f32(w_loc, w_scale, clen, seed, vector, output, chunk_size, stream);
}
