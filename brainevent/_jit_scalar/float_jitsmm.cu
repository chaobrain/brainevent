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
 * jitsmm_notrans/trans_f32 use the AW-T4 MM random stream and match CSR
 * matrix_mode="mm".  jitsmm_mv_notrans/trans_f32 use the MV random stream
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

__device__ __forceinline__ float group4_reduce_sum_f32(float value, int group) {
    unsigned int mask = 0xFU << (group * AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 2, AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 1, AW_T4_GROUP_SIZE);
    return value;
}

__device__ __forceinline__ double group4_reduce_sum_f64(double value, int group) {
    unsigned int mask = 0xFU << (group * AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 2, AW_T4_GROUP_SIZE);
    value += __shfl_down_sync(mask, value, 1, AW_T4_GROUP_SIZE);
    return value;
}

__global__ void _mv_notrans_f32_kern(
    const float* __restrict__ weight,
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
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);

    unsigned int q = stationary_initial_q(&rng, cl);
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

__global__ void _mv_trans_f32_kern(
    const float* __restrict__ weight,
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
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);

    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        atomic_add_f32(&output[(size_t)j * n + col_b], w * v);
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }
}

__global__ void _mm_notrans_f32_kern(
    const float* __restrict__ weight,
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
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);

    unsigned int q = stationary_initial_q(&rng, cl);
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

__global__ void _mm_trans_f32_kern(
    const float* __restrict__ weight,
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
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);

    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        atomic_add_f32(&output[(size_t)j * n + col_b], w * v);
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;
    }
}

static void launch_mv_notrans_f32(
    const BE::Tensor weight,
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
                "jitsmm_mv_notrans_f32 grid overflow: row_warp_blocks=%d n_chunks=%d n=%d\n",
                row_warp_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, (unsigned int)n);

    _mv_notrans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(B.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, n, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

static void launch_mv_trans_f32(
    const BE::Tensor weight,
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
                "jitsmm_mv_trans_f32 grid overflow: row_warp_blocks=%d n_chunks=%d n=%d\n",
                row_warp_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, (unsigned int)n);

    _mv_trans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(B.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, n, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

static void launch_mm_notrans_f32(
    const BE::Tensor weight,
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
                "jitsmm_notrans_f32 grid overflow: row_group_blocks=%d n_chunks=%d n=%d\n",
                row_group_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_group_blocks, (unsigned int)n_chunks, (unsigned int)n);

    _mm_notrans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(B.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, n, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

static void launch_mm_trans_f32(
    const BE::Tensor weight,
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
                "jitsmm_trans_f32 grid overflow: row_group_blocks=%d n_chunks=%d n=%d\n",
                row_group_blocks, n_chunks, n);
        abort();
    }
    dim3 blocks((unsigned int)row_group_blocks, (unsigned int)n_chunks, (unsigned int)n);

    _mm_trans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const float*>(B.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, n, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE jitsmm_mv_notrans_f32
void jitsmm_mv_notrans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    launch_mv_notrans_f32(weight, clen, seed, B, output, m, k, n, chunk_size, stream);
}

// @BE jitsmm_mv_trans_f32
void jitsmm_mv_trans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    launch_mv_trans_f32(weight, clen, seed, B, output, m, k, n, chunk_size, stream);
}

// @BE jitsmm_notrans_f32
void jitsmm_notrans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    launch_mm_notrans_f32(weight, clen, seed, B, output, m, k, n, chunk_size, stream);
}

// @BE jitsmm_trans_f32
void jitsmm_trans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor B,
    BE::Tensor output,
    int m, int k, int n, int chunk_size,
    int64_t stream
) {
    launch_mm_trans_f32(weight, clen, seed, B, output, m, k, n, chunk_size, stream);
}

#define DEFINE_JITSMM_MV_NOTRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD, WARP_REDUCE) \
__global__ void _mv_notrans##SFX##_kern(                                                \
    const WEIGHT_T* __restrict__ weight,                                                \
    const int*      __restrict__ clen,                                                  \
    const int*      __restrict__ seed,                                                  \
    const WEIGHT_T* __restrict__ B,                                                     \
    WEIGHT_T*       __restrict__ output,                                                \
    int m, int k, int n, int chunk_size, int n_chunks                                   \
) {                                                                                    \
    int lane = threadIdx.x & 31;                                                        \
    int warp_id = threadIdx.x >> 5;                                                     \
    int warps_per_block = blockDim.x >> 5;                                              \
    int row_block = (int)blockIdx.x;                                                    \
    int chunk_id = (int)blockIdx.y;                                                     \
    int col_b = (int)blockIdx.z;                                                        \
    int row = row_block * warps_per_block + warp_id;                                    \
    if (row >= m || chunk_id >= n_chunks || col_b >= n) return;                         \
    int chunk_start = chunk_id * chunk_size;                                            \
    if (chunk_start >= k) return;                                                       \
    int chunk_end = chunk_start + chunk_size;                                           \
    if (chunk_end > k) chunk_end = k;                                                   \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);                 \
    if (chunk_width == 0U) return;                                                      \
    ACC_T w = READ_W(__ldg(&weight[0]));                                                \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                    \
    if (cl < 2U) cl = 2U;                                                               \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);                                 \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);                  \
    unsigned int q = stationary_initial_q(&rng, cl);                                    \
    unsigned int local_j = (unsigned int)lane + 32U * q;                                \
    ACC_T acc = (ACC_T)0;                                                               \
    while (local_j < chunk_width) {                                                     \
        int j = chunk_start + (int)local_j;                                             \
        acc += w * READ_W(__ldg(&B[(size_t)j * n + col_b]));                            \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);                      \
        local_j = (unsigned int)lane + 32U * q;                                         \
    }                                                                                   \
    ACC_T row_acc = WARP_REDUCE(acc);                                                   \
    if (lane == 0) ATOMIC_ADD(&output[(size_t)row * n + col_b], row_acc);               \
}

#define DEFINE_JITSMM_MV_TRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD)        \
__global__ void _mv_trans##SFX##_kern(                                          \
    const WEIGHT_T* __restrict__ weight,                                        \
    const int*      __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const WEIGHT_T* __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n, int chunk_size, int n_chunks                           \
) {                                                                            \
    int lane = threadIdx.x & 31;                                                \
    int warp_id = threadIdx.x >> 5;                                             \
    int warps_per_block = blockDim.x >> 5;                                      \
    int row_block = (int)blockIdx.x;                                            \
    int chunk_id = (int)blockIdx.y;                                             \
    int col_b = (int)blockIdx.z;                                                \
    int row = row_block * warps_per_block + warp_id;                            \
    if (row >= m || chunk_id >= n_chunks || col_b >= n) return;                 \
    ACC_T v = READ_W(__ldg(&B[(size_t)row * n + col_b]));                       \
    if (v == (ACC_T)0) return;                                                  \
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
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);          \
    unsigned int q = stationary_initial_q(&rng, cl);                            \
    unsigned int local_j = (unsigned int)lane + 32U * q;                        \
    while (local_j < chunk_width) {                                             \
        int j = chunk_start + (int)local_j;                                     \
        ATOMIC_ADD(&output[(size_t)j * n + col_b], w * v);                      \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);              \
        local_j = (unsigned int)lane + 32U * q;                                 \
    }                                                                           \
}

#define DEFINE_JITSMM_MM_NOTRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD, GROUP_REDUCE) \
__global__ void _mm_notrans##SFX##_kern(                                                \
    const WEIGHT_T* __restrict__ weight,                                                \
    const int*      __restrict__ clen,                                                  \
    const int*      __restrict__ seed,                                                  \
    const WEIGHT_T* __restrict__ B,                                                     \
    WEIGHT_T*       __restrict__ output,                                                \
    int m, int k, int n, int chunk_size, int n_chunks                                   \
) {                                                                                    \
    int lane = threadIdx.x & 31;                                                        \
    int sub_lane = lane & (AW_T4_GROUP_SIZE - 1);                                       \
    int group = lane >> 2;                                                              \
    int warp_id = threadIdx.x >> 5;                                                     \
    int warps_per_block = blockDim.x >> 5;                                              \
    int chunk_id = (int)blockIdx.y;                                                     \
    int col_b = (int)blockIdx.z;                                                        \
    int warp_task = (int)blockIdx.x * warps_per_block + warp_id;                        \
    int row = warp_task * AW_T4_GROUPS_PER_WARP + group;                                \
    if (row >= m || chunk_id >= n_chunks || col_b >= n) return;                         \
    int chunk_start = chunk_id * chunk_size;                                            \
    if (chunk_start >= k) return;                                                       \
    int chunk_end = chunk_start + chunk_size;                                           \
    if (chunk_end > k) chunk_end = k;                                                   \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);                 \
    if (chunk_width == 0U) return;                                                      \
    ACC_T w = READ_W(__ldg(&weight[0]));                                                \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                    \
    if (cl < 2U) cl = 2U;                                                               \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);                                 \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, sub_lane);              \
    unsigned int q = stationary_initial_q(&rng, cl);                                    \
    unsigned int local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;               \
    ACC_T acc = (ACC_T)0;                                                               \
    while (local_j < chunk_width) {                                                     \
        int j = chunk_start + (int)local_j;                                             \
        acc += w * READ_W(__ldg(&B[(size_t)j * n + col_b]));                            \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);                      \
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;                        \
    }                                                                                   \
    ACC_T row_acc = GROUP_REDUCE(acc, group);                                           \
    if (sub_lane == 0) ATOMIC_ADD(&output[(size_t)row * n + col_b], row_acc);           \
}

#define DEFINE_JITSMM_MM_TRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD)        \
__global__ void _mm_trans##SFX##_kern(                                          \
    const WEIGHT_T* __restrict__ weight,                                        \
    const int*      __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const WEIGHT_T* __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n, int chunk_size, int n_chunks                           \
) {                                                                            \
    int lane = threadIdx.x & 31;                                                \
    int sub_lane = lane & (AW_T4_GROUP_SIZE - 1);                               \
    int group = lane >> 2;                                                      \
    int warp_id = threadIdx.x >> 5;                                             \
    int warps_per_block = blockDim.x >> 5;                                      \
    int chunk_id = (int)blockIdx.y;                                             \
    int col_b = (int)blockIdx.z;                                                \
    int warp_task = (int)blockIdx.x * warps_per_block + warp_id;                \
    int row = warp_task * AW_T4_GROUPS_PER_WARP + group;                        \
    if (row >= m || chunk_id >= n_chunks || col_b >= n) return;                 \
    ACC_T v = READ_W(__ldg(&B[(size_t)row * n + col_b]));                       \
    if (v == (ACC_T)0) return;                                                  \
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
        ATOMIC_ADD(&output[(size_t)j * n + col_b], w * v);                      \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);              \
        local_j = (unsigned int)sub_lane + AW_T4_GROUP_SIZE * q;                \
    }                                                                           \
}

DEFINE_JITSMM_MV_NOTRANS(_f64, double, double, READ_F64, atomic_add_f64, warp_reduce_sum_f64)
DEFINE_JITSMM_MV_NOTRANS(_f16, __half, float, READ_F16, atomic_add_f16, warp_reduce_sum_f32)
DEFINE_JITSMM_MV_NOTRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16, warp_reduce_sum_f32)

DEFINE_JITSMM_MV_TRANS(_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_JITSMM_MV_TRANS(_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_JITSMM_MV_TRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

DEFINE_JITSMM_MM_NOTRANS(_f64, double, double, READ_F64, atomic_add_f64, group4_reduce_sum_f64)
DEFINE_JITSMM_MM_NOTRANS(_f16, __half, float, READ_F16, atomic_add_f16, group4_reduce_sum_f32)
DEFINE_JITSMM_MM_NOTRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16, group4_reduce_sum_f32)

DEFINE_JITSMM_MM_TRANS(_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_JITSMM_MM_TRANS(_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_JITSMM_MM_TRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

#define FFI_JITSMM(NAME, KERNEL, SFX, WEIGHT_T, ROW_GROUPS, OUT_ROWS)          \
void NAME##SFX(                                                                \
    const BE::Tensor weight,                                                   \
    const BE::Tensor clen,                                                     \
    const BE::Tensor seed,                                                     \
    const BE::Tensor B,                                                        \
    BE::Tensor output,                                                         \
    int m, int k, int n, int chunk_size,                                       \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    if (OUT_ROWS <= 0 || n <= 0) return;                                       \
    BE_CUDA_CHECK(cudaMemsetAsync(                                             \
        output.data_ptr(), 0, (size_t)OUT_ROWS * n * sizeof(WEIGHT_T), s));    \
    if (k <= 0 || chunk_size <= 0) return;                                     \
    int n_chunks = (k + chunk_size - 1) / chunk_size;                          \
    if (n_chunks <= 0) return;                                                 \
    int threads = 256;                                                         \
    int warps_per_block = threads / 32;                                        \
    int rows_per_block = warps_per_block * ROW_GROUPS;                         \
    int row_blocks = (m + rows_per_block - 1) / rows_per_block;                \
    if (row_blocks > 2147483647 || n_chunks > 65535 || n > 65535) {            \
        fprintf(stderr, #NAME #SFX " grid overflow\n");                       \
        abort();                                                               \
    }                                                                          \
    dim3 blocks((unsigned int)row_blocks, (unsigned int)n_chunks, (unsigned int)n); \
    _##KERNEL##SFX##_kern<<<blocks, threads, 0, s>>>(                          \
        static_cast<const WEIGHT_T*>(weight.data_ptr()),                       \
        static_cast<const int*>(clen.data_ptr()),                              \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<const WEIGHT_T*>(B.data_ptr()),                            \
        static_cast<WEIGHT_T*>(output.data_ptr()),                             \
        m, k, n, chunk_size, n_chunks                                          \
    );                                                                         \
    BE_CHECK_KERNEL_LAUNCH();                                                  \
}

// @BE jitsmm_mv_notrans_f64
FFI_JITSMM(jitsmm_mv_notrans, mv_notrans, _f64, double, 1, m)
// @BE jitsmm_mv_notrans_f16
FFI_JITSMM(jitsmm_mv_notrans, mv_notrans, _f16, __half, 1, m)
// @BE jitsmm_mv_notrans_bf16
FFI_JITSMM(jitsmm_mv_notrans, mv_notrans, _bf16, __nv_bfloat16, 1, m)

// @BE jitsmm_mv_trans_f64
FFI_JITSMM(jitsmm_mv_trans, mv_trans, _f64, double, 1, k)
// @BE jitsmm_mv_trans_f16
FFI_JITSMM(jitsmm_mv_trans, mv_trans, _f16, __half, 1, k)
// @BE jitsmm_mv_trans_bf16
FFI_JITSMM(jitsmm_mv_trans, mv_trans, _bf16, __nv_bfloat16, 1, k)

// @BE jitsmm_notrans_f64
FFI_JITSMM(jitsmm_notrans, mm_notrans, _f64, double, AW_T4_GROUPS_PER_WARP, m)
// @BE jitsmm_notrans_f16
FFI_JITSMM(jitsmm_notrans, mm_notrans, _f16, __half, AW_T4_GROUPS_PER_WARP, m)
// @BE jitsmm_notrans_bf16
FFI_JITSMM(jitsmm_notrans, mm_notrans, _bf16, __nv_bfloat16, AW_T4_GROUPS_PER_WARP, m)

// @BE jitsmm_trans_f64
FFI_JITSMM(jitsmm_trans, mm_trans, _f64, double, AW_T4_GROUPS_PER_WARP, k)
// @BE jitsmm_trans_f16
FFI_JITSMM(jitsmm_trans, mm_trans, _f16, __half, AW_T4_GROUPS_PER_WARP, k)
// @BE jitsmm_trans_bf16
FFI_JITSMM(jitsmm_trans, mm_trans, _bf16, __nv_bfloat16, AW_T4_GROUPS_PER_WARP, k)
