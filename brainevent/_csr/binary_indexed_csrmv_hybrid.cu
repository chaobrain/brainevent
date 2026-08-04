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
 * binary_indexed_csrmv_hybrid.cu -- Indexed hybrid CSR scatter
 * =============================================================================
 *
 * Heterogeneous CUDA backend for binary_csrmv_indexed with
 * transpose=True. Structural slot j reads weights[perm[j]], then scatters to
 * output[indices[j]]. Scratch/task semantics match
 * binary_csrmv_wat_hybrid.cu.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "xla/ffi/api/ffi.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace ffi = xla::ffi;

#ifndef BE_INDEXED_CSRMV_WAT_HYBRID_CHECK
#define BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(condition, message) \
    do                                                               \
    {                                                                \
        if (!(condition))                                            \
        {                                                            \
            std::fprintf(stderr, "%s\n", message);                  \
            std::abort();                                            \
        }                                                            \
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
        WeightT *__restrict__ output,
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
            WeightTraits::atomic_add(&output[idx], w);
        }
    }

    template <typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeThreadIndexed(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        WeightT *__restrict__ output,
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
            WeightTraits::atomic_add(&output[idx], w);
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT>
    __global__ void IndexedCsrmvWatHybridExtractKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        const OffsetT *__restrict__ perm,
        const EventT *__restrict__ vector,
        WeightT *__restrict__ output,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        int32_t *__restrict__ status,
        int m,
        int task_capacity)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= m || !HybridEventTraits<EventT>::active(__ldg(&vector[row])))
        {
            return;
        }

        OffsetT row_begin = __ldg(&indptr[row]);
        OffsetT row_end = __ldg(&indptr[row + 1]);
        OffsetT nnz = row_end - row_begin;
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            ScatterRangeThreadIndexed(weights, indices, perm, output, row_begin, row_end);
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
    __global__ void IndexedCsrmvWatHybridBlockKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        const OffsetT *__restrict__ task_begin,
        const OffsetT *__restrict__ task_end,
        const int32_t *__restrict__ status,
        int task_capacity,
        WeightT *__restrict__ output)
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
            ScatterRangeIndexed(weights, indices, perm, output, begin, end);
        }
    }

    static ffi::Error InternalCudaError(const char *context, cudaError_t error)
    {
        std::string message(context);
        message += ": ";
        message += cudaGetErrorString(error);
        return ffi::Error::Internal(message);
    }

    static ffi::Error InternalMessage(const char *message)
    {
        return ffi::Error::Internal(message);
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT>
    static ffi::Error LaunchIndexedCsrmvWatHybrid(
        const WeightT *weights,
        const IndexT *indices,
        const OffsetT *indptr,
        const OffsetT *perm,
        const EventT *vector,
        WeightT *output,
        OffsetT *task_begin,
        OffsetT *task_end,
        int32_t *device_status,
        int m,
        int k,
        int64_t nnz,
        int64_t task_capacity_attr,
        cudaStream_t stream)
    {
        if (task_capacity_attr < 0 ||
            task_capacity_attr > static_cast<int64_t>(std::numeric_limits<int>::max()))
        {
            return InternalMessage("task_capacity is outside int32 range");
        }
        int task_capacity = static_cast<int>(task_capacity_attr);

        cudaError_t status_code =
            cudaMemsetAsync(output, 0, static_cast<size_t>(k) * sizeof(WeightT), stream);
        if (status_code != cudaSuccess)
        {
            return InternalCudaError("cudaMemsetAsync(output) failed", status_code);
        }
        if (m <= 0 || nnz == 0)
        {
            return ffi::Error::Success();
        }
        if (device_status == nullptr)
        {
            return InternalMessage("device_status scratch buffer is null");
        }
        if (task_capacity > 0 && (task_begin == nullptr || task_end == nullptr))
        {
            return InternalMessage("hybrid task scratch buffer is null");
        }

        status_code = cudaMemsetAsync(
            device_status,
            0,
            static_cast<size_t>(2) * sizeof(int32_t),
            stream);
        if (status_code != cudaSuccess)
        {
            return InternalCudaError("cudaMemsetAsync(device_status) failed", status_code);
        }

        int extract_blocks = (m + kBlockSize - 1) / kBlockSize;
        IndexedCsrmvWatHybridExtractKernel<WeightT, EventT, IndexT, OffsetT>
            <<<extract_blocks, kBlockSize, 0, stream>>>(
                weights, indices, indptr, perm, vector, output,
                task_begin, task_end, device_status, m, task_capacity);
        status_code = cudaGetLastError();
        if (status_code != cudaSuccess)
        {
            return InternalCudaError(
                "IndexedCsrmvWatHybridExtractKernel launch failed",
                status_code);
        }

        if (task_capacity > 0)
        {
            int scatter_blocks = task_capacity < kFixedScatterBlocks
                                     ? task_capacity
                                     : kFixedScatterBlocks;
            IndexedCsrmvWatHybridBlockKernel<WeightT, IndexT, OffsetT>
                <<<scatter_blocks, kBlockSize, 0, stream>>>(
                    weights, indices, perm, task_begin, task_end, device_status,
                    task_capacity, output);
            status_code = cudaGetLastError();
            if (status_code != cudaSuccess)
            {
                return InternalCudaError(
                    "IndexedCsrmvWatHybridBlockKernel launch failed",
                    status_code);
            }
        }
        return ffi::Error::Success();
    }

} // namespace

#define DEFINE_BINARY_INDEXED_CSRMV_WAT_HYBRID_HETERO(SUFFIX, WEIGHT_T, EVENT_T) \
void binary_indexed_csrmv_wat_hybrid_hetero##SUFFIX(                    \
    const BE::Tensor weights,                                             \
    const BE::Tensor indices,                                             \
    const BE::Tensor indptr,                                              \
    const BE::Tensor perm,                                                \
    const BE::Tensor vector,                                              \
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
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        weights.dtype() == HybridWeightTraits<WEIGHT_T>::dtype,           \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects matching weights dtype"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        IsSignedIndexDType(indices.dtype()),                              \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects indices=s32/s64"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        IsSignedIndexDType(indptr.dtype()) && perm.dtype() == indptr.dtype(), \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects indptr=s32/s64 and perm dtype to match indptr"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        vector.dtype() == HybridEventTraits<EVENT_T>::dtype,              \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects matching vector dtype"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        task_begin_scratch.dtype() == indptr.dtype() &&                   \
            task_end_scratch.dtype() == indptr.dtype() &&                 \
            status_scratch.dtype() == BE::DType::Int32 &&                 \
            task_begin_output.dtype() == indptr.dtype() &&                \
            task_end_output.dtype() == indptr.dtype() &&                  \
            status_output.dtype() == BE::DType::Int32,                    \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects status=s32 and task scratch dtype to match indptr"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        output.dtype() == HybridWeightTraits<WEIGHT_T>::dtype,            \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects matching output dtype"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        weights.ndim() == 1 && indices.ndim() == 1 &&                     \
            indptr.ndim() == 1 && perm.ndim() == 1 && vector.ndim() == 1 && \
            task_begin_scratch.ndim() == 1 && task_end_scratch.ndim() == 1 && \
            status_scratch.ndim() == 1 && output.ndim() == 1 &&           \
            task_begin_output.ndim() == 1 && task_end_output.ndim() == 1 && \
            status_output.ndim() == 1,                                    \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects rank-1 buffers"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        weights.numel() == indices.numel() && perm.numel() == indices.numel(), \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects weights/perm/indices lengths to match"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        indptr.numel() == vector.numel() + 1,                              \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects indptr length to be vector length plus one"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        task_capacity >= 0,                                                \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects non-negative task_capacity"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        task_begin_scratch.numel() == static_cast<size_t>(task_capacity) && \
            task_end_scratch.numel() == static_cast<size_t>(task_capacity) && \
            task_begin_output.numel() == static_cast<size_t>(task_capacity) && \
            task_end_output.numel() == static_cast<size_t>(task_capacity), \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects task scratch lengths to match task_capacity"); \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        status_scratch.numel() == 2 && status_output.numel() == 2,         \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " expects status scratch length 2"); \
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);     \
    bool ok = true;                                                        \
    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, { \
        ffi::Error error = LaunchIndexedCsrmvWatHybrid<WEIGHT_T, EVENT_T, IndexT, OffsetT>( \
            weights.data_ptr<const WEIGHT_T>(),                            \
            indices.data_ptr<const IndexT>(),                              \
            indptr.data_ptr<const OffsetT>(),                              \
            perm.data_ptr<const OffsetT>(),                                \
            vector.data_ptr<const EVENT_T>(),                              \
            output.data_ptr<WEIGHT_T>(),                                   \
            task_begin_output.data_ptr<OffsetT>(),                         \
            task_end_output.data_ptr<OffsetT>(),                           \
            status_output.data_ptr<int32_t>(),                             \
            static_cast<int>(indptr.numel()) - 1,                          \
            static_cast<int>(output.numel()),                              \
            static_cast<int64_t>(indices.numel()),                         \
            task_capacity,                                                 \
            cuda_stream);                                                  \
        ok = error.success();                                              \
    });                                                                    \
    BE_INDEXED_CSRMV_WAT_HYBRID_CHECK(                                    \
        ok,                                                                \
        "binary_indexed_csrmv_wat_hybrid_hetero" #SUFFIX " launch failed"); \
}

// @BE binary_indexed_csrmv_wat_hybrid_hetero_f32_bool
DEFINE_BINARY_INDEXED_CSRMV_WAT_HYBRID_HETERO(_f32_bool, float, int8_t)
// @BE binary_indexed_csrmv_wat_hybrid_hetero_f32_float
DEFINE_BINARY_INDEXED_CSRMV_WAT_HYBRID_HETERO(_f32_float, float, float)
// @BE binary_indexed_csrmv_wat_hybrid_hetero_f64_bool
DEFINE_BINARY_INDEXED_CSRMV_WAT_HYBRID_HETERO(_f64_bool, double, int8_t)
// @BE binary_indexed_csrmv_wat_hybrid_hetero_f64_float
DEFINE_BINARY_INDEXED_CSRMV_WAT_HYBRID_HETERO(_f64_float, double, float)
// @BE binary_indexed_csrmv_wat_hybrid_hetero_f16_bool
DEFINE_BINARY_INDEXED_CSRMV_WAT_HYBRID_HETERO(_f16_bool, __half, int8_t)
// @BE binary_indexed_csrmv_wat_hybrid_hetero_f16_float
DEFINE_BINARY_INDEXED_CSRMV_WAT_HYBRID_HETERO(_f16_float, __half, float)
// @BE binary_indexed_csrmv_wat_hybrid_hetero_bf16_bool
DEFINE_BINARY_INDEXED_CSRMV_WAT_HYBRID_HETERO(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_indexed_csrmv_wat_hybrid_hetero_bf16_float
DEFINE_BINARY_INDEXED_CSRMV_WAT_HYBRID_HETERO(_bf16_float, __nv_bfloat16, float)
