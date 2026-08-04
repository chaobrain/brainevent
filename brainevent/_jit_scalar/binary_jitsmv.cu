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
 * light_rng-chunk-wpr.cu -- bit-packed light-RNG WPR chunk backend.
 *
 * Notrans and trans share one row-major matrix.  One warp owns one
 * (row, chunk_id) task, and each lane owns one residue class:
 *
 *   local_j = lane + 32 * q
 *
 * The connection stream is keyed by (seed, row, chunk_id, lane), while weights
 * are stateless hashes of (seed, row, col).  Inactive spikes do not change the
 * generated matrix.
 */

#include <cstdio>
#include <cstdlib>

#include "cuda_common.h"
#include "brainevent/common.h"

#define IS_ACTIVE_PACKED(packed, idx) \
    ((__ldg(&(packed)[(idx) >> 5]) >> ((idx) & 31)) & 1U)

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

__global__ void _pack_bool_kern(
    const int8_t* __restrict__ vector,
    uint32_t*     __restrict__ packed,
    int k
) {
    int word = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int base = word << 5;
    if (base >= k) return;

    uint32_t bits = 0U;
#pragma unroll
    for (int b = 0; b < 32; ++b) {
        int j = base + b;
        if (j < k && __ldg(&vector[j]) != 0) {
            bits |= (1U << b);
        }
    }
    packed[word] = bits;
}

__global__ void _notrans_f32_kern(
    const float*    __restrict__ weight,
    const int*      __restrict__ clen,
    const int*      __restrict__ seed,
    const uint32_t* __restrict__ packed,
    float*          __restrict__ output,
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
        if (IS_ACTIVE_PACKED(packed, j)) {
                        acc += w;
        }
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }

    float row_acc = warp_reduce_sum_f32(acc);
    if (lane == 0) {
        atomic_add_f32(&output[row], row_acc);
    }
}

__global__ void _trans_f32_kern(
    const float*    __restrict__ weight,
    const int*      __restrict__ clen,
    const int*      __restrict__ seed,
    const uint32_t* __restrict__ packed,
    float*          __restrict__ output,
    int m, int k, int chunk_size, int n_chunks
) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_block = (int)blockIdx.x;
    int chunk_id = (int)blockIdx.y;
    int row = row_block * warps_per_block + warp_id;
    if (row >= m || chunk_id >= n_chunks) return;
    if (!IS_ACTIVE_PACKED(packed, row)) return;

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
        atomic_add_f32(&output[j], w);
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);
        local_j = (unsigned int)lane + 32U * q;
    }
}

// @BE pack_bool
void pack_bool(
    const BE::Tensor vector,
    BE::Tensor packed,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int k = static_cast<int>(vector.size(0));
    int packed_words = static_cast<int>(packed.size(0));
    if (packed_words == 0) return;

    int threads = 256;
    int blocks = (packed_words + threads - 1) / threads;
    _pack_bool_kern<<<blocks, threads, 0, s>>>(
        static_cast<const int8_t*>(vector.data_ptr()),
        static_cast<uint32_t*>(packed.data_ptr()),
        k
    );
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE notrans_f32
void notrans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor packed,
    BE::Tensor output,
    int vector_size,
    int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(output.size(0));
    int k = vector_size;
    if (m == 0) return;
    BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, (size_t)m * sizeof(float), s));
    if (k <= 0 || chunk_size <= 0) return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535) {
        fprintf(stderr,
                "notrans_f32 grid overflow: row_warp_blocks=%d n_chunks=%d\n",
                row_warp_blocks, n_chunks);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, 1U);

    _notrans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const uint32_t*>(packed.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE trans_f32
void trans_f32(
    const BE::Tensor weight,
    const BE::Tensor clen,
    const BE::Tensor seed,
    const BE::Tensor packed,
    BE::Tensor output,
    int vector_size,
    int chunk_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = vector_size;
    int k = static_cast<int>(output.size(0));
    if (k == 0) return;
    BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, (size_t)k * sizeof(float), s));
    if (m <= 0 || chunk_size <= 0) return;

    int n_chunks = (k + chunk_size - 1) / chunk_size;
    if (n_chunks <= 0) return;

    int threads = 256;
    int warps_per_block = threads / 32;
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    if (row_warp_blocks > 2147483647 || n_chunks > 65535) {
        fprintf(stderr,
                "trans_f32 grid overflow: row_warp_blocks=%d n_chunks=%d\n",
                row_warp_blocks, n_chunks);
        abort();
    }
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, 1U);

    _trans_f32_kern<<<blocks, threads, 0, s>>>(
        static_cast<const float*>(weight.data_ptr()),
        static_cast<const int*>(clen.data_ptr()),
        static_cast<const int*>(seed.data_ptr()),
        static_cast<const uint32_t*>(packed.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        m, k, chunk_size, n_chunks
    );
    BE_CHECK_KERNEL_LAUNCH();
}

#define DEFINE_BINARY_JITSMV_NOTRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD, WARP_REDUCE) \
__global__ void _notrans##SFX##_kern(                                                       \
    const WEIGHT_T* __restrict__ weight,                                                     \
    const int*      __restrict__ clen,                                                       \
    const int*      __restrict__ seed,                                                       \
    const uint32_t* __restrict__ packed,                                                     \
    WEIGHT_T*       __restrict__ output,                                                     \
    int m, int k, int chunk_size, int n_chunks                                               \
) {                                                                                         \
    int lane = threadIdx.x & 31;                                                             \
    int warp_id = threadIdx.x >> 5;                                                          \
    int warps_per_block = blockDim.x >> 5;                                                   \
    int row_block = (int)blockIdx.x;                                                         \
    int chunk_id = (int)blockIdx.y;                                                          \
    int row = row_block * warps_per_block + warp_id;                                         \
    if (row >= m || chunk_id >= n_chunks) return;                                            \
    int chunk_start = chunk_id * chunk_size;                                                 \
    if (chunk_start >= k) return;                                                            \
    int chunk_end = chunk_start + chunk_size;                                                \
    if (chunk_end > k) chunk_end = k;                                                        \
    unsigned int chunk_width = (unsigned int)(chunk_end - chunk_start);                      \
    if (chunk_width == 0U) return;                                                           \
    ACC_T w = READ_W(__ldg(&weight[0]));                                                     \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2U) cl = 2U;                                                                    \
    unsigned int seed0 = (unsigned int)__ldg(&seed[0]);                                      \
    unsigned int rng = light_rng_init_wpr(seed0, row, chunk_id, lane);                       \
    unsigned int q = stationary_initial_q(&rng, cl);                                         \
    unsigned int local_j = (unsigned int)lane + 32U * q;                                     \
    ACC_T acc = (ACC_T)0;                                                                    \
    while (local_j < chunk_width) {                                                          \
        int j = chunk_start + (int)local_j;                                                  \
        if (IS_ACTIVE_PACKED(packed, j)) acc += w;                                           \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);                           \
        local_j = (unsigned int)lane + 32U * q;                                              \
    }                                                                                         \
    ACC_T row_acc = WARP_REDUCE(acc);                                                        \
    if (lane == 0) ATOMIC_ADD(&output[row], row_acc);                                        \
}

#define DEFINE_BINARY_JITSMV_TRANS(SFX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD)    \
__global__ void _trans##SFX##_kern(                                             \
    const WEIGHT_T* __restrict__ weight,                                        \
    const int*      __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const uint32_t* __restrict__ packed,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int chunk_size, int n_chunks                                  \
) {                                                                            \
    int lane = threadIdx.x & 31;                                                \
    int warp_id = threadIdx.x >> 5;                                             \
    int warps_per_block = blockDim.x >> 5;                                      \
    int row_block = (int)blockIdx.x;                                            \
    int chunk_id = (int)blockIdx.y;                                             \
    int row = row_block * warps_per_block + warp_id;                            \
    if (row >= m || chunk_id >= n_chunks) return;                               \
    if (!IS_ACTIVE_PACKED(packed, row)) return;                                 \
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
        ATOMIC_ADD(&output[j], w);                                              \
        q += 1U + fast_bounded_u32(light_rng_next(&rng), cl - 1U);              \
        local_j = (unsigned int)lane + 32U * q;                                 \
    }                                                                           \
}

DEFINE_BINARY_JITSMV_NOTRANS(_f64, double, double, READ_F64, atomic_add_f64, warp_reduce_sum_f64)
DEFINE_BINARY_JITSMV_NOTRANS(_f16, __half, float, READ_F16, atomic_add_f16, warp_reduce_sum_f32)
DEFINE_BINARY_JITSMV_NOTRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16, warp_reduce_sum_f32)

DEFINE_BINARY_JITSMV_TRANS(_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BINARY_JITSMV_TRANS(_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BINARY_JITSMV_TRANS(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

#define FFI_BINARY_JITSMV(NAME, KERNEL, SFX, WEIGHT_T)                       \
void NAME##SFX(                                                              \
    const BE::Tensor weight,                                                 \
    const BE::Tensor clen,                                                   \
    const BE::Tensor seed,                                                   \
    const BE::Tensor packed,                                                 \
    BE::Tensor output,                                                       \
    int vector_size,                                                         \
    int chunk_size,                                                          \
    int64_t stream                                                           \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    int m = #KERNEL[0] == 'n'                                                \
        ? static_cast<int>(output.size(0))                                   \
        : vector_size;                                                       \
    int k = #KERNEL[0] == 'n'                                                \
        ? vector_size                                                        \
        : static_cast<int>(output.size(0));                                  \
    int out_size = #KERNEL[0] == 'n' ? m : k;                                \
    if (out_size == 0) return;                                               \
    BE_CUDA_CHECK(cudaMemsetAsync(                                           \
        output.data_ptr(), 0, (size_t)out_size * sizeof(WEIGHT_T), s));      \
    if ((#KERNEL[0] == 'n' ? k : m) <= 0 || chunk_size <= 0) return;         \
    int n_chunks = (k + chunk_size - 1) / chunk_size;                        \
    if (n_chunks <= 0) return;                                               \
    int threads = 256;                                                       \
    int warps_per_block = threads / 32;                                      \
    int row_warp_blocks = (m + warps_per_block - 1) / warps_per_block;       \
    if (row_warp_blocks > 2147483647 || n_chunks > 65535) {                 \
        fprintf(stderr, #NAME #SFX " grid overflow\n");                     \
        abort();                                                             \
    }                                                                        \
    dim3 blocks((unsigned int)row_warp_blocks, (unsigned int)n_chunks, 1U);  \
    _##KERNEL##SFX##_kern<<<blocks, threads, 0, s>>>(                        \
        static_cast<const WEIGHT_T*>(weight.data_ptr()),                     \
        static_cast<const int*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                            \
        static_cast<const uint32_t*>(packed.data_ptr()),                     \
        static_cast<WEIGHT_T*>(output.data_ptr()),                           \
        m, k, chunk_size, n_chunks                                           \
    );                                                                       \
    BE_CHECK_KERNEL_LAUNCH();                                                \
}

// @BE notrans_f64
FFI_BINARY_JITSMV(notrans, notrans, _f64, double)
// @BE notrans_f16
FFI_BINARY_JITSMV(notrans, notrans, _f16, __half)
// @BE notrans_bf16
FFI_BINARY_JITSMV(notrans, notrans, _bf16, __nv_bfloat16)

// @BE trans_f64
FFI_BINARY_JITSMV(trans, trans, _f64, double)
// @BE trans_f16
FFI_BINARY_JITSMV(trans, trans, _f16, __half)
// @BE trans_bf16
FFI_BINARY_JITSMV(trans, trans, _bf16, __nv_bfloat16)
