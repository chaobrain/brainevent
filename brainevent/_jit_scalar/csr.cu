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
 * light_csr.cu -- materialize the light_rng-chunk-wpr logical matrix as CSR.
 *
 * This file intentionally mirrors the row-major matrix generator in
 * light_rng-chunk-wpr.cu.  One warp owns one (row, chunk_id) task and each lane
 * owns one residue class:
 *
 *   local_j = lane + 32 * q
 *
 * Count writes per-(row, chunk) counts.  Fill receives the exclusive
 * per-(row, chunk) offsets, replays the same streams, and writes deterministic
 * CSR slices without atomics.
 */

#include <cstdio>
#include <cstdlib>

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

__device__ __forceinline__ unsigned int warp_sum_u32(unsigned int value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ unsigned int warp_exclusive_prefix_u32(unsigned int value, int lane) {
    unsigned int inclusive = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned int other = __shfl_up_sync(0xffffffffU, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    return inclusive - value;
}

__device__ __forceinline__ unsigned int count_lane_connections(
    unsigned int seed0,
    int row,
    int chunk_id,
    int lane,
    unsigned int cl,
    unsigned int chunk_width
) {
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);
    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    unsigned int count = 0U;
    while (local_j < chunk_width) {
        count += 1U;
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }
    return count;
}

__global__ void _count_chunks_notrans_f32_kern(
    const int* __restrict__ clen,
    const int* __restrict__ seed,
    int*       __restrict__ chunk_counts,
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

    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U) cl = 2U;
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);

    unsigned int lane_count = count_lane_connections(
        seed0, row, chunk_id, lane, cl, chunk_width
    );
    unsigned int chunk_count = warp_sum_u32(lane_count);
    if (lane == 0) {
        chunk_counts[(size_t)row * n_chunks + chunk_id] = (int)chunk_count;
    }
}

__global__ void _fill_notrans_f32_kern(
    const float* __restrict__ weight,
    const int*   __restrict__ clen,
    const int*   __restrict__ seed,
    const int*   __restrict__ chunk_offsets,
    int*         __restrict__ indices,
    float*       __restrict__ data,
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

    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U) cl = 2U;
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);

    unsigned int lane_count = count_lane_connections(
        seed0, row, chunk_id, lane, cl, chunk_width
    );
    unsigned int lane_offset = warp_exclusive_prefix_u32(lane_count, lane);
    int base = __ldg(&chunk_offsets[(size_t)row * n_chunks + chunk_id]) + (int)lane_offset;

    float w = READ_F32(__ldg(&weight[0]));

    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);
    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    int write = 0;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        int pos = base + write;
        indices[pos] = j;
                data[pos] = w;
        write += 1;
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }
}


// #########################################################################
// ##  Trans count/fill pass -- write CSR(A.T)                            ##
// #########################################################################

__global__ void _count_chunks_trans_f32_kern(
    const int* __restrict__ clen,
    const int* __restrict__ seed,
    int*       __restrict__ row_counts,
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

    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U) cl = 2U;
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);

    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);
    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        atomicAdd(&row_counts[j], 1);
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }
}

__global__ void _fill_trans_f32_kern(
    const float* __restrict__ weight,
    const int*   __restrict__ clen,
    const int*   __restrict__ seed,
    const int*   __restrict__ indptr,
    int*         __restrict__ indices,
    float*       __restrict__ data,
    int*         __restrict__ cursor,
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

    unsigned int cl = (unsigned int)__ldg(&clen[0]);
    if (cl < 2U) cl = 2U;
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);
    float w = READ_F32(__ldg(&weight[0]));

    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);
    unsigned int q = stationary_initial_q(&rng, cl);
    unsigned int local_j = (unsigned int)lane + 32U * q;
    while (local_j < chunk_width) {
        int j = chunk_start + (int)local_j;
        int offset = atomicAdd(&cursor[j], 1);
        int pos = __ldg(&indptr[j]) + offset;
        indices[pos] = row;
        data[pos] = w;
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }
}

// #########################################################################
// ##  FFI Wrappers                                                        ##
// #########################################################################

// @BE count_chunks_trans_f32
void count_chunks_trans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    BE::Tensor row_counts,
    int n_rows,
    int n_cols,
    int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    (void)weight;
    if (n_rows <= 0 || n_cols <= 0 || chunk_size <= 0) return;
    int n_chunks = (n_cols + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;
    cudaMemsetAsync(row_counts.data_ptr(), 0, (size_t)n_cols * sizeof(int), s);

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535) {
        fprintf(stderr,
                "count_chunks_trans_f32 grid overflow: row_warp_blocks=%d n_chunks=%d\n",
                row_warp_blocks, n_chunks);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, 1U);

    _count_chunks_trans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<int*>(row_counts.data_ptr()),
        n_rows, n_cols, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE fill_trans_f32
void fill_trans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor indptr,
    BE::Tensor indices,
    BE::Tensor data,
    BE::Tensor cursor,
    int n_rows,
    int n_cols,
    int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (n_rows <= 0 || n_cols <= 0 || chunk_size <= 0) return;
    int n_chunks = (n_cols + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;
    cudaMemsetAsync(cursor.data_ptr(), 0, (size_t)n_cols * sizeof(int), s);

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535) {
        fprintf(stderr,
                "fill_trans_f32 grid overflow: row_warp_blocks=%d n_chunks=%d\n",
                row_warp_blocks, n_chunks);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, 1U);

    _fill_trans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const int*>(indptr.data_ptr()),
        static_cast<int*>(indices.data_ptr()),
        static_cast<float*>(data.data_ptr()),
        static_cast<int*>(cursor.data_ptr()),
        n_rows, n_cols, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE count_chunks_notrans_f32
void count_chunks_notrans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    BE::Tensor chunk_counts,
    int n_cols,
    int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    (void)weight;
    int m = static_cast<int>(chunk_counts.size(0));
    int n_chunks = static_cast<int>(chunk_counts.size(1));
    if (m == 0 || n_chunks == 0) return;
    if (n_cols <= 0 || chunk_size <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535) {
        fprintf(stderr,
                "count_chunks_notrans_f32 grid overflow: row_warp_blocks=%d n_chunks=%d\n",
                row_warp_blocks, n_chunks);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, 1U);

    _count_chunks_notrans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<int*>(chunk_counts.data_ptr()),
        m, n_cols, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE fill_notrans_f32
void fill_notrans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor chunk_offsets,
    BE::Tensor indices,
    BE::Tensor data,
    int n_cols,
    int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(chunk_offsets.size(0));
    int n_chunks = static_cast<int>(chunk_offsets.size(1));
    int nnz = static_cast<int>(indices.size(0));
    (void)nnz;
    if (m == 0 || n_chunks == 0) return;
    if (n_cols <= 0 || chunk_size <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535) {
        fprintf(stderr,
                "fill_notrans_f32 grid overflow: row_warp_blocks=%d n_chunks=%d\n",
                row_warp_blocks, n_chunks);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, 1U);

    _fill_notrans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const int*>(chunk_offsets.data_ptr()),
        static_cast<int*>(indices.data_ptr()),
        static_cast<float*>(data.data_ptr()),
        m, n_cols, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

#define DEFINE_FILL_NOTRANS(SFX, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
__global__ void _fill_notrans##SFX##_kern( \
    const WEIGHT_T* __restrict__ weight, \
    const int*      __restrict__ clen, \
    const int*      __restrict__ seed, \
    const int*      __restrict__ chunk_offsets, \
    int*            __restrict__ indices, \
    WEIGHT_T*       __restrict__ data, \
    int m, int k, int chunk_size, int n_chunks \
) { \
    int lane = threadIdx.x & 31; \
    int warp_id = threadIdx.x >> 5; \
    int warps_per_block = blockDim.x >> 5; \
    int row_block = (int)blockIdx.x; \
    int chunk_id = (int)blockIdx.y; \
    int row = row_block * warps_per_block + warp_id; \
    if (row >= m || chunk_id >= n_chunks) return; \
    int chunk_start = chunk_id * chunk_size; \
    if (chunk_start >= k) return; \
    int chunk_end = chunk_start + chunk_size; \
    if (chunk_end > k) chunk_end = k; \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start); \
    if (chunk_width == 0U) return; \
    unsigned int cl = (unsigned int)__ldg(&clen[0]); \
    if (cl < 2U) cl = 2U; \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]); \
    unsigned int lane_count = count_lane_connections( \
        seed0, row, chunk_id, lane, cl, chunk_width \
    ); \
    unsigned int lane_offset = warp_exclusive_prefix_u32(lane_count, lane); \
    int base = __ldg(&chunk_offsets[(size_t)row * n_chunks + chunk_id]) + (int)lane_offset; \
    ACC_T w = READ_W(__ldg(&weight[0])); \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane); \
    unsigned int q = stationary_initial_q(&rng, cl); \
    unsigned int local_j = (unsigned int)lane + 32U * q; \
    int write = 0; \
    while (local_j < chunk_width) { \
        int j = chunk_start + (int)local_j; \
        int pos = base + write; \
        indices[pos] = j; \
        data[pos] = WRITE_W(w); \
        write += 1; \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U); \
        local_j = (unsigned int)lane + 32U * q; \
    } \
}

#define DEFINE_FILL_TRANS(SFX, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
__global__ void _fill_trans##SFX##_kern( \
    const WEIGHT_T* __restrict__ weight, \
    const int*      __restrict__ clen, \
    const int*      __restrict__ seed, \
    const int*      __restrict__ indptr, \
    int*            __restrict__ indices, \
    WEIGHT_T*       __restrict__ data, \
    int*            __restrict__ cursor, \
    int m, int k, int chunk_size, int n_chunks \
) { \
    int lane = threadIdx.x & 31; \
    int warp_id = threadIdx.x >> 5; \
    int warps_per_block = blockDim.x >> 5; \
    int row_block = (int)blockIdx.x; \
    int chunk_id = (int)blockIdx.y; \
    int row = row_block * warps_per_block + warp_id; \
    if (row >= m || chunk_id >= n_chunks) return; \
    int chunk_start = chunk_id * chunk_size; \
    if (chunk_start >= k) return; \
    int chunk_end = chunk_start + chunk_size; \
    if (chunk_end > k) chunk_end = k; \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start); \
    if (chunk_width == 0U) return; \
    unsigned int cl = (unsigned int)__ldg(&clen[0]); \
    if (cl < 2U) cl = 2U; \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]); \
    ACC_T w = READ_W(__ldg(&weight[0])); \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane); \
    unsigned int q = stationary_initial_q(&rng, cl); \
    unsigned int local_j = (unsigned int)lane + 32U * q; \
    while (local_j < chunk_width) { \
        int j = chunk_start + (int)local_j; \
        int offset = atomicAdd(&cursor[j], 1); \
        int pos = __ldg(&indptr[j]) + offset; \
        indices[pos] = row; \
        data[pos] = WRITE_W(w); \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U); \
        local_j = (unsigned int)lane + 32U * q; \
    } \
}

DEFINE_FILL_NOTRANS(_f64, double, double, READ_F64, WRITE_F64)
DEFINE_FILL_NOTRANS(_f16, __half, float, READ_F16, WRITE_F16)
DEFINE_FILL_NOTRANS(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)

DEFINE_FILL_TRANS(_f64, double, double, READ_F64, WRITE_F64)
DEFINE_FILL_TRANS(_f16, __half, float, READ_F16, WRITE_F16)
DEFINE_FILL_TRANS(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)

#define DEFINE_COUNT_TRANS_WRAPPER(NAME, SFX, KERNEL) \
void NAME##SFX( \
    const BE::Tensor weight, \
    const BE::Tensor clen, \
    const BE::Tensor seed, \
    BE::Tensor row_counts, \
    int n_rows, \
    int n_cols, \
    int chunk_size, \
    int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    (void)weight; \
    if (n_rows <= 0 || n_cols <= 0 || chunk_size <= 0) return; \
    int n_chunks = (n_cols + chunk_size - 1) / chunk_size; \
    if (n_chunks <= 0) return; \
    cudaMemsetAsync(row_counts.data_ptr(), 0, (size_t)n_cols * sizeof(int), s); \
    int threads = 256; \
    int warps_per_block = threads / 32; \
    int row_blocks = (n_rows + warps_per_block - 1) / warps_per_block; \
    if (row_blocks > 2147483647 || n_chunks > 65535) { \
        fprintf(stderr, #NAME #SFX " grid overflow: row_blocks=%d n_chunks=%d\n", \
                row_blocks, n_chunks); \
        abort(); \
    } \
    dim3 blocks((unsigned int)row_blocks, (unsigned int)n_chunks, 1U); \
    KERNEL<<<blocks, threads, 0, s>>>( \
        static_cast<const int*>(clen.data_ptr()), \
        static_cast<const int*>(seed.data_ptr()), \
        static_cast<int*>(row_counts.data_ptr()), \
        n_rows, n_cols, chunk_size, n_chunks \
    ); \
    BE_CHECK_KERNEL_LAUNCH(); \
}

#define DEFINE_COUNT_NOTRANS_WRAPPER(NAME, SFX, KERNEL) \
void NAME##SFX( \
    const BE::Tensor weight, \
    const BE::Tensor clen, \
    const BE::Tensor seed, \
    BE::Tensor chunk_counts, \
    int n_cols, \
    int chunk_size, \
    int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    (void)weight; \
    int m = static_cast<int>(chunk_counts.size(0)); \
    int n_chunks = static_cast<int>(chunk_counts.size(1)); \
    if (m == 0 || n_chunks == 0) return; \
    if (n_cols <= 0 || chunk_size <= 0) return; \
    int threads = 256; \
    int warps_per_block = threads / 32; \
    int row_blocks = (m + warps_per_block - 1) / warps_per_block; \
    if (row_blocks > 2147483647 || n_chunks > 65535) { \
        fprintf(stderr, #NAME #SFX " grid overflow: row_blocks=%d n_chunks=%d\n", \
                row_blocks, n_chunks); \
        abort(); \
    } \
    dim3 blocks((unsigned int)row_blocks, (unsigned int)n_chunks, 1U); \
    KERNEL<<<blocks, threads, 0, s>>>( \
        static_cast<const int*>(clen.data_ptr()), \
        static_cast<const int*>(seed.data_ptr()), \
        static_cast<int*>(chunk_counts.data_ptr()), \
        m, n_cols, chunk_size, n_chunks \
    ); \
    BE_CHECK_KERNEL_LAUNCH(); \
}

#define DEFINE_FILL_NOTRANS_WRAPPER(NAME, SFX, KERNEL, WEIGHT_T) \
void NAME##SFX( \
    const BE::Tensor weight, \
    const BE::Tensor clen, \
    const BE::Tensor seed, \
    const BE::Tensor chunk_offsets, \
    BE::Tensor indices, \
    BE::Tensor data, \
    int n_cols, \
    int chunk_size, \
    int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int m = static_cast<int>(chunk_offsets.size(0)); \
    int n_chunks = static_cast<int>(chunk_offsets.size(1)); \
    int nnz = static_cast<int>(indices.size(0)); \
    (void)nnz; \
    if (m == 0 || n_chunks == 0) return; \
    if (n_cols <= 0 || chunk_size <= 0) return; \
    int threads = 256; \
    int warps_per_block = threads / 32; \
    int row_blocks = (m + warps_per_block - 1) / warps_per_block; \
    if (row_blocks > 2147483647 || n_chunks > 65535) { \
        fprintf(stderr, #NAME #SFX " grid overflow: row_blocks=%d n_chunks=%d\n", \
                row_blocks, n_chunks); \
        abort(); \
    } \
    dim3 blocks((unsigned int)row_blocks, (unsigned int)n_chunks, 1U); \
    KERNEL<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_T*>(weight.data_ptr()), \
        static_cast<const int*>(clen.data_ptr()), \
        static_cast<const int*>(seed.data_ptr()), \
        static_cast<const int*>(chunk_offsets.data_ptr()), \
        static_cast<int*>(indices.data_ptr()), \
        static_cast<WEIGHT_T*>(data.data_ptr()), \
        m, n_cols, chunk_size, n_chunks \
    ); \
    BE_CHECK_KERNEL_LAUNCH(); \
}

#define DEFINE_FILL_TRANS_WRAPPER(NAME, SFX, KERNEL, WEIGHT_T) \
void NAME##SFX( \
    const BE::Tensor weight, \
    const BE::Tensor clen, \
    const BE::Tensor seed, \
    const BE::Tensor indptr, \
    BE::Tensor indices, \
    BE::Tensor data, \
    BE::Tensor cursor, \
    int n_rows, \
    int n_cols, \
    int chunk_size, \
    int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    if (n_rows <= 0 || n_cols <= 0 || chunk_size <= 0) return; \
    int n_chunks = (n_cols + chunk_size - 1) / chunk_size; \
    if (n_chunks <= 0) return; \
    cudaMemsetAsync(cursor.data_ptr(), 0, (size_t)n_cols * sizeof(int), s); \
    int threads = 256; \
    int warps_per_block = threads / 32; \
    int row_blocks = (n_rows + warps_per_block - 1) / warps_per_block; \
    if (row_blocks > 2147483647 || n_chunks > 65535) { \
        fprintf(stderr, #NAME #SFX " grid overflow: row_blocks=%d n_chunks=%d\n", \
                row_blocks, n_chunks); \
        abort(); \
    } \
    dim3 blocks((unsigned int)row_blocks, (unsigned int)n_chunks, 1U); \
    KERNEL<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_T*>(weight.data_ptr()), \
        static_cast<const int*>(clen.data_ptr()), \
        static_cast<const int*>(seed.data_ptr()), \
        static_cast<const int*>(indptr.data_ptr()), \
        static_cast<int*>(indices.data_ptr()), \
        static_cast<WEIGHT_T*>(data.data_ptr()), \
        static_cast<int*>(cursor.data_ptr()), \
        n_rows, n_cols, chunk_size, n_chunks \
    ); \
    BE_CHECK_KERNEL_LAUNCH(); \
}
// @BE count_chunks_trans_f64
DEFINE_COUNT_TRANS_WRAPPER(count_chunks_trans, _f64, _count_chunks_trans_f32_kern)
// @BE count_chunks_trans_f16
DEFINE_COUNT_TRANS_WRAPPER(count_chunks_trans, _f16, _count_chunks_trans_f32_kern)
// @BE count_chunks_trans_bf16
DEFINE_COUNT_TRANS_WRAPPER(count_chunks_trans, _bf16, _count_chunks_trans_f32_kern)

// @BE fill_trans_f64
DEFINE_FILL_TRANS_WRAPPER(fill_trans, _f64, _fill_trans_f64_kern, double)
// @BE fill_trans_f16
DEFINE_FILL_TRANS_WRAPPER(fill_trans, _f16, _fill_trans_f16_kern, __half)
// @BE fill_trans_bf16
DEFINE_FILL_TRANS_WRAPPER(fill_trans, _bf16, _fill_trans_bf16_kern, __nv_bfloat16)

// @BE count_chunks_notrans_f64
DEFINE_COUNT_NOTRANS_WRAPPER(count_chunks_notrans, _f64, _count_chunks_notrans_f32_kern)
// @BE count_chunks_notrans_f16
DEFINE_COUNT_NOTRANS_WRAPPER(count_chunks_notrans, _f16, _count_chunks_notrans_f32_kern)
// @BE count_chunks_notrans_bf16
DEFINE_COUNT_NOTRANS_WRAPPER(count_chunks_notrans, _bf16, _count_chunks_notrans_f32_kern)

// @BE fill_notrans_f64
DEFINE_FILL_NOTRANS_WRAPPER(fill_notrans, _f64, _fill_notrans_f64_kern, double)
// @BE fill_notrans_f16
DEFINE_FILL_NOTRANS_WRAPPER(fill_notrans, _f16, _fill_notrans_f16_kern, __half)
// @BE fill_notrans_bf16
DEFINE_FILL_NOTRANS_WRAPPER(fill_notrans, _bf16, _fill_notrans_bf16_kern, __nv_bfloat16)
