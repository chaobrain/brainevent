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
 * binary_indexed_csrmm_hybrid.cu -- Indexed SRAW hybrid CSRMM
 * =============================================================================
 *
 * Heterogeneous SRAW backend for binary_csrmm_indexed with
 * transpose=True. Structural slot j reads weights[perm[j]], then scatters to
 * output[batch, indices[j]]. Scratch/task semantics match
 * binary_csrmm_sraw_hybrid.cu.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#ifndef BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK
#define BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(condition, message) \
    do                                                             \
    {                                                              \
        if (!(condition))                                          \
        {                                                          \
            std::fprintf(stderr, "%s\n", message);                \
            std::abort();                                          \
        }                                                          \
    } while (0)
#endif

#ifndef BE_HYBRID_BLOCK_SIZE
#define BE_HYBRID_BLOCK_SIZE 256
#endif

#ifndef BE_HYBRID_FIXED_SCATTER_BLOCKS
#define BE_HYBRID_FIXED_SCATTER_BLOCKS 2048
#endif

#ifndef BE_HYBRID_TPR_THRESHOLD
#define BE_HYBRID_TPR_THRESHOLD 128
#endif

#ifndef BE_HYBRID_TASK_NNZ
#define BE_HYBRID_TASK_NNZ 4096
#endif

namespace
{

    constexpr int kBlockSize = BE_HYBRID_BLOCK_SIZE;
    constexpr int kFixedScatterBlocks = BE_HYBRID_FIXED_SCATTER_BLOCKS;
    constexpr int kTprThreshold = BE_HYBRID_TPR_THRESHOLD;
    constexpr int kTaskNnz = BE_HYBRID_TASK_NNZ;
    constexpr int kStatusCountIndex = 0;
    constexpr int kStatusOverflowIndex = 1;

    static_assert(kBlockSize > 0, "BE_HYBRID_BLOCK_SIZE must be positive");
    static_assert(kBlockSize % 32 == 0, "BE_HYBRID_BLOCK_SIZE must be a multiple of 32");
    static_assert(kBlockSize <= 1024, "BE_HYBRID_BLOCK_SIZE must not exceed 1024");
    static_assert(kFixedScatterBlocks > 0, "BE_HYBRID_FIXED_SCATTER_BLOCKS must be positive");
    static_assert(kTprThreshold >= 0, "BE_HYBRID_TPR_THRESHOLD must be non-negative");
    static_assert(kTaskNnz > 0, "BE_HYBRID_TASK_NNZ must be positive");
    constexpr int64_t kInt32Max = 2147483647LL;

    static bool IsSignedIndexDType(BE::DType dtype)
    {
        return dtype == BE::DType::Int32 || dtype == BE::DType::Int64;
    }

    template <typename WeightT>
    struct HybridWeightTraits;

    template <>
    struct HybridWeightTraits<float>
    {
        using AccT = float;
        static constexpr BE::DType dtype = BE::DType::Float32;
        __device__ static AccT read(float value) { return READ_F32(value); }
        __device__ static void atomic_add(float *addr, AccT value) { atomic_add_f32(addr, value); }
    };

    template <>
    struct HybridWeightTraits<double>
    {
        using AccT = double;
        static constexpr BE::DType dtype = BE::DType::Float64;
        __device__ static AccT read(double value) { return READ_F64(value); }
        __device__ static void atomic_add(double *addr, AccT value) { atomic_add_f64(addr, value); }
    };

    template <>
    struct HybridWeightTraits<__half>
    {
        using AccT = float;
        static constexpr BE::DType dtype = BE::DType::Float16;
        __device__ static AccT read(__half value) { return READ_F16(value); }
        __device__ static void atomic_add(__half *addr, AccT value) { atomic_add_f16(addr, value); }
    };

    template <>
    struct HybridWeightTraits<__nv_bfloat16>
    {
        using AccT = float;
        static constexpr BE::DType dtype = BE::DType::BFloat16;
        __device__ static AccT read(__nv_bfloat16 value) { return READ_BF16(value); }
        __device__ static void atomic_add(__nv_bfloat16 *addr, AccT value) { atomic_add_bf16(addr, value); }
    };

    template <typename EventT>
    struct HybridEventTraits;

    template <>
    struct HybridEventTraits<int8_t>
    {
        static constexpr BE::DType dtype = BE::DType::Bool;
        __device__ static bool active(int8_t value) { return IS_ACTIVE_BOOL(value); }
    };

    template <>
    struct HybridEventTraits<float>
    {
        static constexpr BE::DType dtype = BE::DType::Float32;
        __device__ static bool active(float value) { return IS_ACTIVE_FLOAT(value); }
    };

    template <typename OffsetT>
    __device__ __forceinline__ int32_t TaskChunksForRowNnz(
        OffsetT nnz,
        int32_t *__restrict__ status)
    {
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            return 0;
        }

        OffsetT chunks =
            (nnz + static_cast<OffsetT>(kTaskNnz - 1)) / static_cast<OffsetT>(kTaskNnz);
        if (static_cast<int64_t>(chunks) > kInt32Max)
        {
            atomicExch(&status[kStatusOverflowIndex], 1);
            return 0;
        }
        return static_cast<int32_t>(chunks);
    }

    template <typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeIndexed(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        WeightT *__restrict__ output_col,
        OffsetT begin,
        OffsetT end)
    {
        using WeightTraits = HybridWeightTraits<WeightT>;
        using AccT = typename WeightTraits::AccT;
        for (OffsetT j = begin + static_cast<OffsetT>(threadIdx.x);
             j < end;
             j += static_cast<OffsetT>(blockDim.x))
        {
            IndexT idx = __ldg(&indices[j]);
            OffsetT weight_pos = __ldg(&perm[j]);
            AccT w = WeightTraits::read(__ldg(&weights[weight_pos]));
            WeightTraits::atomic_add(&output_col[idx], w);
        }
    }

    template <typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeThreadIndexed(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        WeightT *__restrict__ output_col,
        OffsetT begin,
        OffsetT end)
    {
        using WeightTraits = HybridWeightTraits<WeightT>;
        using AccT = typename WeightTraits::AccT;
        for (OffsetT j = begin; j < end; ++j)
        {
            IndexT idx = __ldg(&indices[j]);
            OffsetT weight_pos = __ldg(&perm[j]);
            AccT w = WeightTraits::read(__ldg(&weights[weight_pos]));
            WeightTraits::atomic_add(&output_col[idx], w);
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT>
    __global__ void IndexedCsrmmSrawHybridExtractKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        const OffsetT *__restrict__ perm,
        const EventT *__restrict__ matrix,
        WeightT *__restrict__ output_col,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        int32_t *__restrict__ status,
        int n_pre,
        int n_batch,
        int col,
        int task_capacity)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= n_pre)
        {
            return;
        }
        if (!HybridEventTraits<EventT>::active(
                __ldg(&matrix[static_cast<size_t>(row) * n_batch + col])))
        {
            return;
        }

        OffsetT row_begin = __ldg(&indptr[row]);
        OffsetT row_end = __ldg(&indptr[row + 1]);
        OffsetT nnz = row_end - row_begin;
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            ScatterRangeThreadIndexed(weights, indices, perm, output_col, row_begin, row_end);
            return;
        }

        int32_t n_chunks = TaskChunksForRowNnz(nnz, status);
        if (n_chunks <= 0)
        {
            return;
        }

        int32_t base = atomicAdd(&status[kStatusCountIndex], n_chunks);
        if (base < 0 || base > task_capacity || n_chunks > task_capacity - base)
        {
            atomicExch(&status[kStatusOverflowIndex], 1);
            return;
        }

        for (int32_t chunk = 0; chunk < n_chunks; ++chunk)
        {
            OffsetT begin = row_begin + static_cast<OffsetT>(chunk) * static_cast<OffsetT>(kTaskNnz);
            OffsetT end = begin + static_cast<OffsetT>(kTaskNnz);
            if (end > row_end)
            {
                end = row_end;
            }
            int32_t task = base + chunk;
            task_begin[task] = begin;
            task_end[task] = end;
        }
    }

    template <typename WeightT, typename IndexT, typename OffsetT>
    __global__ void IndexedCsrmmSrawHybridBlockKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        const OffsetT *__restrict__ task_begin,
        const OffsetT *__restrict__ task_end,
        const int32_t *__restrict__ status,
        int task_capacity,
        WeightT *__restrict__ output_col)
    {
        int active = __ldg(&status[kStatusCountIndex]);
        if (active > task_capacity)
        {
            active = task_capacity;
        }
        for (int task = blockIdx.x; task < active; task += gridDim.x)
        {
            OffsetT begin = __ldg(&task_begin[task]);
            OffsetT end = __ldg(&task_end[task]);
            ScatterRangeIndexed(weights, indices, perm, output_col, begin, end);
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT>
    static void LaunchIndexedCsrmmSrawHybrid(
        const WeightT *weights,
        const IndexT *indices,
        const OffsetT *indptr,
        const OffsetT *perm,
        const EventT *matrix,
        WeightT *output,
        OffsetT *task_begin,
        OffsetT *task_end,
        int32_t *status,
        int n_pre,
        int n_post,
        int n_batch,
        int64_t nnz,
        int task_capacity,
        cudaStream_t stream)
    {
        if (n_pre <= 0 || n_post <= 0 || n_batch <= 0 || nnz == 0)
        {
            return;
        }

        int extract_blocks = (n_pre + kBlockSize - 1) / kBlockSize;
        for (int col = 0; col < n_batch; ++col)
        {
            BE_CUDA_CHECK(cudaMemsetAsync(
                status, 0, static_cast<size_t>(2) * sizeof(int32_t), stream));
            WeightT *output_col = output + static_cast<size_t>(col) * n_post;
            IndexedCsrmmSrawHybridExtractKernel<WeightT, EventT, IndexT, OffsetT>
                <<<extract_blocks, kBlockSize, 0, stream>>>(
                    weights, indices, indptr, perm, matrix, output_col,
                    task_begin, task_end, status, n_pre, n_batch, col,
                    task_capacity);
            BE_CHECK_KERNEL_LAUNCH();

            if (task_capacity > 0)
            {
                int scatter_blocks = task_capacity < kFixedScatterBlocks
                                         ? task_capacity
                                         : kFixedScatterBlocks;
                IndexedCsrmmSrawHybridBlockKernel<WeightT, IndexT, OffsetT>
                    <<<scatter_blocks, kBlockSize, 0, stream>>>(
                        weights, indices, perm, task_begin, task_end, status,
                        task_capacity, output_col);
                BE_CHECK_KERNEL_LAUNCH();
            }
        }
    }

    static int ParseTaskCapacity(int64_t task_capacity)
    {
        BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(
            task_capacity >= 0,
            "binary_indexed_csrmm_sraw_hybrid expects non-negative task_capacity");
        BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(
            task_capacity <= static_cast<int64_t>(std::numeric_limits<int>::max()),
            "binary_indexed_csrmm_sraw_hybrid expects task_capacity inside int32 range");
        return static_cast<int>(task_capacity);
    }

} // namespace

#define DEFINE_BINARY_INDEXED_CSRMM_SRAW_HYBRID_HETERO(SUFFIX, WEIGHT_T, EVENT_T) \
void binary_indexed_csrmm_sraw_hybrid_hetero##SUFFIX(                  \
    const BE::Tensor weights,                                             \
    const BE::Tensor indices,                                             \
    const BE::Tensor indptr,                                              \
    const BE::Tensor perm,                                                \
    const BE::Tensor matrix,                                              \
    const BE::Tensor task_begin_scratch,                                  \
    const BE::Tensor task_end_scratch,                                    \
    const BE::Tensor status_scratch,                                      \
    BE::Tensor output,                                                    \
    BE::Tensor task_begin_output,                                         \
    BE::Tensor task_end_output,                                           \
    BE::Tensor status_output,                                             \
    int64_t task_capacity,                                                \
    int64_t stream)                                                       \
{                                                                         \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        weights.dtype() == HybridWeightTraits<WEIGHT_T>::dtype,           \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects matching weights dtype"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        IsSignedIndexDType(indices.dtype()),                              \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects indices=s32/s64"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        IsSignedIndexDType(indptr.dtype()) && perm.dtype() == indptr.dtype(), \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects indptr=s32/s64 and perm dtype to match indptr"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        matrix.dtype() == HybridEventTraits<EVENT_T>::dtype,              \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects matching matrix dtype"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        output.dtype() == HybridWeightTraits<WEIGHT_T>::dtype,            \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects matching output dtype"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        task_begin_scratch.dtype() == indptr.dtype() &&                   \
            task_end_scratch.dtype() == indptr.dtype() &&                 \
            task_begin_output.dtype() == indptr.dtype() &&                \
            task_end_output.dtype() == indptr.dtype() &&                  \
            status_scratch.dtype() == BE::DType::Int32 &&                 \
            status_output.dtype() == BE::DType::Int32,                    \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects status=s32 and task scratch dtype to match indptr"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        weights.ndim() == 1 && indices.ndim() == 1 && indptr.ndim() == 1 && \
            perm.ndim() == 1 && matrix.ndim() == 2 && output.ndim() == 2 && \
            task_begin_scratch.ndim() == 1 && task_end_scratch.ndim() == 1 && \
            status_scratch.ndim() == 1 && task_begin_output.ndim() == 1 && \
            task_end_output.ndim() == 1 && status_output.ndim() == 1,     \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects CSR/scratch rank-1 and matrix/output rank-2"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        weights.numel() == indices.numel() && perm.numel() == indices.numel(), \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects weights/perm/indices lengths to match"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        indptr.numel() == matrix.size(0) + 1,                             \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects indptr length to be matrix rows plus one"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        output.size(0) == matrix.size(1),                                 \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects output rows to equal matrix columns"); \
    int parsed_task_capacity = ParseTaskCapacity(task_capacity);           \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        task_begin_scratch.numel() == static_cast<size_t>(parsed_task_capacity) && \
            task_end_scratch.numel() == static_cast<size_t>(parsed_task_capacity) && \
            task_begin_output.numel() == static_cast<size_t>(parsed_task_capacity) && \
            task_end_output.numel() == static_cast<size_t>(parsed_task_capacity), \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects task scratch lengths to match task_capacity"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        status_scratch.numel() == 2 && status_output.numel() == 2,         \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects status length 2"); \
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);     \
    int n_pre = static_cast<int>(matrix.size(0));                          \
    int n_batch = static_cast<int>(matrix.size(1));                        \
    int n_post = static_cast<int>(output.size(1));                         \
    WEIGHT_T *out = output.data_ptr<WEIGHT_T>();                           \
    BE_CUDA_CHECK(cudaMemsetAsync(                                         \
        out, 0, static_cast<size_t>(output.size(0)) * output.size(1) * sizeof(WEIGHT_T), \
        cuda_stream));                                                     \
    if (n_pre <= 0 || n_batch <= 0 || n_post <= 0 || indices.numel() == 0) \
    {                                                                      \
        return;                                                            \
    }                                                                      \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        status_output.data_ptr<int32_t>() != nullptr,                      \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects non-null status scratch"); \
    BE_INDEXED_CSRMM_SRAW_HYBRID_CHECK(                                   \
        parsed_task_capacity == 0 ||                                       \
            (task_begin_output.data_ptr<void>() != nullptr &&              \
             task_end_output.data_ptr<void>() != nullptr),                 \
        "binary_indexed_csrmm_sraw_hybrid_hetero" #SUFFIX " expects non-null task scratch"); \
    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, { \
        LaunchIndexedCsrmmSrawHybrid<WEIGHT_T, EVENT_T, IndexT, OffsetT>(  \
            weights.data_ptr<const WEIGHT_T>(),                            \
            indices.data_ptr<const IndexT>(),                              \
            indptr.data_ptr<const OffsetT>(),                              \
            perm.data_ptr<const OffsetT>(),                                \
            matrix.data_ptr<const EVENT_T>(),                              \
            out,                                                           \
            task_begin_output.data_ptr<OffsetT>(),                         \
            task_end_output.data_ptr<OffsetT>(),                           \
            status_output.data_ptr<int32_t>(),                             \
            n_pre,                                                         \
            n_post,                                                        \
            n_batch,                                                       \
            static_cast<int64_t>(indices.numel()),                         \
            parsed_task_capacity,                                          \
            cuda_stream);                                                  \
    });                                                                    \
}

// @BE binary_indexed_csrmm_sraw_hybrid_hetero_f32_bool
DEFINE_BINARY_INDEXED_CSRMM_SRAW_HYBRID_HETERO(_f32_bool, float, int8_t)
// @BE binary_indexed_csrmm_sraw_hybrid_hetero_f32_float
DEFINE_BINARY_INDEXED_CSRMM_SRAW_HYBRID_HETERO(_f32_float, float, float)
// @BE binary_indexed_csrmm_sraw_hybrid_hetero_f64_bool
DEFINE_BINARY_INDEXED_CSRMM_SRAW_HYBRID_HETERO(_f64_bool, double, int8_t)
// @BE binary_indexed_csrmm_sraw_hybrid_hetero_f64_float
DEFINE_BINARY_INDEXED_CSRMM_SRAW_HYBRID_HETERO(_f64_float, double, float)
// @BE binary_indexed_csrmm_sraw_hybrid_hetero_f16_bool
DEFINE_BINARY_INDEXED_CSRMM_SRAW_HYBRID_HETERO(_f16_bool, __half, int8_t)
// @BE binary_indexed_csrmm_sraw_hybrid_hetero_f16_float
DEFINE_BINARY_INDEXED_CSRMM_SRAW_HYBRID_HETERO(_f16_float, __half, float)
// @BE binary_indexed_csrmm_sraw_hybrid_hetero_bf16_bool
DEFINE_BINARY_INDEXED_CSRMM_SRAW_HYBRID_HETERO(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_indexed_csrmm_sraw_hybrid_hetero_bf16_float
DEFINE_BINARY_INDEXED_CSRMM_SRAW_HYBRID_HETERO(_bf16_float, __nv_bfloat16, float)
