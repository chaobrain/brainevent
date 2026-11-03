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
 * hybrid_sddmv_binary.cu -- Hybrid SDDMV relying on a caller-zeroed dweight
 * =============================================================================
 *
 * CUDA backend for the event-driven sampled product:
 *
 *   dweight[j] = ct[indices[j]]
 *     for active row where vector[row] is active
 *     for j in indptr[row]:indptr[row + 1]
 *   dweight[j] = 0 for every slot of an inactive row
 *
 * Strategy (default tuning values; configurable through HybridConfig):
 *   0. Clear status only; dweight arrives zeroed (aliased dweight_in).
 *   1. Extract active rows. Rows with nnz <= 128 are gathered immediately by
 *      one thread per row.
 *   2. Heavy active rows reserve contiguous block tasks once with atomicAdd.
 *      Each task stores begin/end offsets for up to 4096 slots.
 *   3. Launch a fixed number of 256-thread blocks. Each block reads the
 *      active task count on device and processes tasks with a grid-stride
 *      loop, avoiding D2H status reads and host synchronization.
 *
 * Integration contract:
 *
 *     ct             : f16/bf16/f32/f64[k] cotangent
 *     indices        : s32/s64[nnz] global columns in slot order
 *     indptr         : s32/s64[m + 1]
 *     vector         : bool/f32[m], plus int8/f64[m] for f64 ct; active means != 0
 *     task_begin/end : same dtype as indptr, task_capacity entries
 *     dweight_in     : ct dtype [nnz] caller-zeroed buffer aliased to dweight
 *     status         : s32[2] scratch/output {active_task_count, overflow_flag}
 *     dweight        : ct dtype [nnz] output
 *
 *   Entry points hybrid_sddmv_binary_{f16,bf16,f32,f64}_{bool,float}: the
 *   value suffix names the ct/dweight dtype, the event suffix the vector dtype.
 *   f64 additionally has _int8 and _double event entry points.
 *   Index dtypes are dispatched at run time.
 *
 *   task_capacity bounds the tasks required by all heavy rows. The Python
 *   caller computes this bound from shapes and the captured HybridConfig.
 */

#include "hybrid_sddmv.cuh"

#include <limits>

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
    constexpr int64_t kInt32Max = 2147483647LL;

    static_assert(kBlockSize > 0, "BE_HYBRID_BLOCK_SIZE must be positive");
    static_assert(kBlockSize % 32 == 0, "BE_HYBRID_BLOCK_SIZE must be a multiple of 32");
    static_assert(kBlockSize <= 1024, "BE_HYBRID_BLOCK_SIZE must not exceed 1024");
    static_assert(kFixedScatterBlocks > 0, "BE_HYBRID_FIXED_SCATTER_BLOCKS must be positive");
    static_assert(kTprThreshold >= 0, "BE_HYBRID_TPR_THRESHOLD must be non-negative");
    static_assert(kTaskNnz > 0, "BE_HYBRID_TASK_NNZ must be positive");

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

    // Block-wide: slot j -> (row, indices[j]); copy ct[indices[j]] to slot j.
    template <typename ValueT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void GatherRange(
        const ValueT *__restrict__ ct,
        const IndexT *__restrict__ indices,
        ValueT *__restrict__ dweight,
        OffsetT begin,
        OffsetT end)
    {
        for (OffsetT j = begin + static_cast<OffsetT>(threadIdx.x);
             j < end;
             j += static_cast<OffsetT>(blockDim.x))
        {
            IndexT col = __ldg(&indices[j]);
            dweight[j] = __ldg(&ct[col]);
        }
    }

    // One thread walks a whole short row.
    template <typename ValueT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void GatherRangeThread(
        const ValueT *__restrict__ ct,
        const IndexT *__restrict__ indices,
        ValueT *__restrict__ dweight,
        OffsetT begin,
        OffsetT end)
    {
        for (OffsetT j = begin; j < end; ++j)
        {
            IndexT col = __ldg(&indices[j]);
            dweight[j] = __ldg(&ct[col]);
        }
    }

    template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
    __global__ void HybridSddmvExtractKernel(
        const ValueT *__restrict__ ct,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        const EventT *__restrict__ vector,
        ValueT *__restrict__ dweight,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        int32_t *__restrict__ status,
        int m,
        int task_capacity)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= m || __ldg(&vector[row]) == EventT(0))
        {
            return;
        }

        OffsetT row_begin = __ldg(&indptr[row]);
        OffsetT row_end = __ldg(&indptr[row + 1]);
        OffsetT nnz = row_end - row_begin;
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            GatherRangeThread(ct, indices, dweight, row_begin, row_end);
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

    template <typename ValueT, typename IndexT, typename OffsetT>
    __global__ void HybridSddmvBlockKernel(
        const ValueT *__restrict__ ct,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ task_begin,
        const OffsetT *__restrict__ task_end,
        const int32_t *__restrict__ status,
        int task_capacity,
        ValueT *__restrict__ dweight)
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
            GatherRange(ct, indices, dweight, begin, end);
        }
    }

} // namespace

namespace hybrid_sddmv
{

    template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
    cudaError_t launch_binary(
        const ValueT *ct,
        const IndexT *indices,
        const OffsetT *indptr,
        const EventT *vector,
        ValueT *dweight,
        OffsetT *task_begin,
        OffsetT *task_end,
        int32_t *device_status,
        int m,
        int64_t nnz,
        int64_t task_capacity_attr,
        cudaStream_t stream)
    {
        if (m < 0 || nnz < 0 || task_capacity_attr < 0 ||
            task_capacity_attr > static_cast<int64_t>(std::numeric_limits<int>::max()))
        {
            return cudaErrorInvalidValue;
        }
        int task_capacity = static_cast<int>(task_capacity_attr);

        // dweight arrives zeroed from the caller (aliased dweight_in).
        cudaError_t status_code = cudaSuccess;
        size_t status_bytes = static_cast<size_t>(2) * sizeof(int32_t);
        if (device_status == nullptr)
        {
            return cudaErrorInvalidValue;
        }
        if (task_capacity > 0 && (task_begin == nullptr || task_end == nullptr))
        {
            return cudaErrorInvalidValue;
        }

        if (m > 0 && nnz > 0 &&
            (ct == nullptr || indices == nullptr || indptr == nullptr ||
             vector == nullptr || dweight == nullptr))
        {
            return cudaErrorInvalidValue;
        }

        status_code = cudaMemsetAsync(device_status, 0, status_bytes, stream);
        if (status_code != cudaSuccess)
        {
            return status_code;
        }
        if (m <= 0 || nnz == 0)
        {
            return cudaSuccess;
        }

        int extract_blocks = 1 + (m - 1) / kBlockSize;
        HybridSddmvExtractKernel<ValueT, EventT, IndexT, OffsetT>
            <<<extract_blocks, kBlockSize, 0, stream>>>(
                ct, indices, indptr, vector, dweight,
                task_begin, task_end, device_status, m, task_capacity);
        status_code = cudaGetLastError();
        if (status_code != cudaSuccess)
        {
            return status_code;
        }

        if (task_capacity > 0)
        {
            int scatter_blocks = task_capacity < kFixedScatterBlocks
                                     ? task_capacity
                                     : kFixedScatterBlocks;
            HybridSddmvBlockKernel<ValueT, IndexT, OffsetT>
                <<<scatter_blocks, kBlockSize, 0, stream>>>(
                    ct, indices, task_begin, task_end, device_status,
                    task_capacity, dweight);
            status_code = cudaGetLastError();
            if (status_code != cudaSuccess)
            {
                return status_code;
            }
        }
        return cudaSuccess;
    }

#define INSTANTIATE_HYBRID_BINARY(VALUE_T, EVENT_T, INDEX_T, OFFSET_T)       \
    template cudaError_t launch_binary<VALUE_T, EVENT_T, INDEX_T, OFFSET_T>( \
        const VALUE_T *, const INDEX_T *, const OFFSET_T *, const EVENT_T *, \
        VALUE_T *, OFFSET_T *, OFFSET_T *, int32_t *, int, int64_t, int64_t, cudaStream_t);
#define INSTANTIATE_HYBRID_INDICES(VALUE_T, EVENT_T)              \
    INSTANTIATE_HYBRID_BINARY(VALUE_T, EVENT_T, int32_t, int32_t) \
    INSTANTIATE_HYBRID_BINARY(VALUE_T, EVENT_T, int32_t, int64_t) \
    INSTANTIATE_HYBRID_BINARY(VALUE_T, EVENT_T, int64_t, int32_t) \
    INSTANTIATE_HYBRID_BINARY(VALUE_T, EVENT_T, int64_t, int64_t)

    INSTANTIATE_HYBRID_INDICES(__half, int8_t)
    INSTANTIATE_HYBRID_INDICES(__half, float)
    INSTANTIATE_HYBRID_INDICES(__nv_bfloat16, int8_t)
    INSTANTIATE_HYBRID_INDICES(__nv_bfloat16, float)
    INSTANTIATE_HYBRID_INDICES(float, int8_t)
    INSTANTIATE_HYBRID_INDICES(float, float)
    INSTANTIATE_HYBRID_INDICES(double, int8_t)
    INSTANTIATE_HYBRID_INDICES(double, float)
    INSTANTIATE_HYBRID_INDICES(double, double)
#undef INSTANTIATE_HYBRID_INDICES
#undef INSTANTIATE_HYBRID_BINARY

} // namespace hybrid_sddmv

#ifndef BE_SDDMV_STANDALONE
#include "cuda_common.h"
#include "brainevent/common.h"
#include "brainevent/dtypes.h"
#include "../_common/op_traits.cuh"
#include <cstdio>
#include <cstdlib>

using op_traits::ValueTraits;

#ifndef BE_HYBRID_SDDMV_CHECK
// Host-side check: throws BE::CheckError, which the generated FFI wrapper
// converts into an xla::ffi::Error (a Python exception) instead of aborting.
#define BE_HYBRID_SDDMV_CHECK(condition, message) \
    do                                            \
    {                                             \
        BE_CHECK(condition) << (message);         \
    } while (0)
#endif

#define DEFINE_HYBRID_SDDMV_BINARY(SUFFIX, VALUE_T, EVENT_T, EVENT_DTYPE)                                                                             \
    void hybrid_sddmv_binary##SUFFIX(                                                                                                    \
        const BE::Tensor ct,                                                                                                             \
        const BE::Tensor indices,                                                                                                        \
        const BE::Tensor indptr,                                                                                                         \
        const BE::Tensor vector,                                                                                                         \
        const BE::Tensor dweight_in,                                                                                                     \
        const BE::Tensor task_begin_scratch,                                                                                             \
        const BE::Tensor task_end_scratch,                                                                                               \
        const BE::Tensor status_scratch,                                                                                                 \
        BE::Tensor dweight,                                                                                                              \
        BE::Tensor task_begin_output,                                                                                                    \
        BE::Tensor task_end_output,                                                                                                      \
        BE::Tensor status_output,                                                                                                        \
        int64_t task_capacity,                                                                                                           \
        int64_t stream)                                                                                                                  \
    {                                                                                                                                    \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            ct.dtype() == ValueTraits<VALUE_T>::dtype &&                                                                                 \
                dweight.dtype() == ValueTraits<VALUE_T>::dtype,                                                                          \
            "hybrid_sddmv_binary" #SUFFIX " expects matching ct and dweight dtype");                                                     \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            op_traits::is_signed_index(indices.dtype()) && op_traits::is_signed_index(indptr.dtype()),                                   \
            "hybrid_sddmv_binary" #SUFFIX " expects indices/indptr=s32/s64");                                                            \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            vector.dtype() == EVENT_DTYPE,                                                                               \
            "hybrid_sddmv_binary" #SUFFIX " expects matching vector dtype");                                                             \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            task_begin_scratch.dtype() == indptr.dtype() &&                                                                              \
                task_end_scratch.dtype() == indptr.dtype() &&                                                                            \
                status_scratch.dtype() == BE::DType::Int32 &&                                                                            \
                task_begin_output.dtype() == indptr.dtype() &&                                                                           \
                task_end_output.dtype() == indptr.dtype() &&                                                                             \
                status_output.dtype() == BE::DType::Int32,                                                                               \
            "hybrid_sddmv_binary" #SUFFIX " expects status=s32 and task scratch dtype to match indptr");                                 \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            ct.ndim() == 1 && indices.ndim() == 1 && indptr.ndim() == 1 &&                                                               \
                vector.ndim() == 1 && dweight.ndim() == 1 &&                                                                             \
                task_begin_scratch.ndim() == 1 && task_end_scratch.ndim() == 1 &&                                                        \
                status_scratch.ndim() == 1 && task_begin_output.ndim() == 1 &&                                                           \
                task_end_output.ndim() == 1 && status_output.ndim() == 1,                                                                \
            "hybrid_sddmv_binary" #SUFFIX " expects rank-1 buffers");                                                                    \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            dweight_in.dtype() == ValueTraits<VALUE_T>::dtype && dweight_in.ndim() == 1 &&                                               \
                dweight_in.numel() == dweight.numel(),                                                                                   \
            "hybrid_sddmv_binary" #SUFFIX " expects dweight_in to match dweight");                                                       \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            dweight.numel() == indices.numel(),                                                                                          \
            "hybrid_sddmv_binary" #SUFFIX " expects dweight and indices lengths to match");                                              \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            indptr.numel() == vector.numel() + 1,                                                                                        \
            "hybrid_sddmv_binary" #SUFFIX " expects indptr length to be vector length plus one");                                        \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            task_capacity >= 0 &&                                                                                                        \
                task_begin_scratch.numel() == static_cast<size_t>(task_capacity) &&                                                      \
                task_end_scratch.numel() == static_cast<size_t>(task_capacity) &&                                                        \
                task_begin_output.numel() == static_cast<size_t>(task_capacity) &&                                                       \
                task_end_output.numel() == static_cast<size_t>(task_capacity),                                                           \
            "hybrid_sddmv_binary" #SUFFIX " expects task scratch lengths to match task_capacity");                                       \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            status_scratch.numel() == 2 && status_output.numel() == 2,                                                                   \
            "hybrid_sddmv_binary" #SUFFIX " expects status scratch length 2");                                                           \
        BE_HYBRID_SDDMV_CHECK(                                                                                                           \
            indptr.numel() - 1 <= static_cast<size_t>(std::numeric_limits<int>::max()),                                                  \
            "hybrid_sddmv_binary" #SUFFIX " expects at most INT32_MAX rows");                                                            \
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);                                                               \
        cudaError_t launch_status = cudaSuccess;                                                                                         \
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {                                                \
            cudaError_t error = hybrid_sddmv::launch_binary<VALUE_T, EVENT_T, IndexT, OffsetT>(                                          \
                ct.data_ptr<const VALUE_T>(),                                                                                            \
                indices.data_ptr<const IndexT>(),                                                                                        \
                indptr.data_ptr<const OffsetT>(),                                                                                        \
                vector.data_ptr<const EVENT_T>(),                                                                                        \
                dweight.data_ptr<VALUE_T>(),                                                                                             \
                task_begin_output.data_ptr<OffsetT>(),                                                                                   \
                task_end_output.data_ptr<OffsetT>(),                                                                                     \
                status_output.data_ptr<int32_t>(),                                                                                       \
                static_cast<int>(indptr.numel()) - 1,                                                                                    \
                static_cast<int64_t>(indices.numel()),                                                                                   \
                task_capacity,                                                                                                           \
                cuda_stream);                                                                                                            \
            launch_status = error;                                                                                                       \
        });                                                                                                                              \
        BE_CHECK(launch_status == cudaSuccess) << "hybrid_sddmv_binary" #SUFFIX " launch failed: " << cudaGetErrorString(launch_status); \
    }

// @BE hybrid_sddmv_binary_f16_bool
DEFINE_HYBRID_SDDMV_BINARY(_f16_bool, __half, int8_t, BE::DType::Bool)
// @BE hybrid_sddmv_binary_f16_float
DEFINE_HYBRID_SDDMV_BINARY(_f16_float, __half, float, BE::DType::Float32)
// @BE hybrid_sddmv_binary_bf16_bool
DEFINE_HYBRID_SDDMV_BINARY(_bf16_bool, __nv_bfloat16, int8_t, BE::DType::Bool)
// @BE hybrid_sddmv_binary_bf16_float
DEFINE_HYBRID_SDDMV_BINARY(_bf16_float, __nv_bfloat16, float, BE::DType::Float32)
// @BE hybrid_sddmv_binary_f32_bool
DEFINE_HYBRID_SDDMV_BINARY(_f32_bool, float, int8_t, BE::DType::Bool)
// @BE hybrid_sddmv_binary_f32_float
DEFINE_HYBRID_SDDMV_BINARY(_f32_float, float, float, BE::DType::Float32)
// @BE hybrid_sddmv_binary_f64_bool
DEFINE_HYBRID_SDDMV_BINARY(_f64_bool, double, int8_t, BE::DType::Bool)
// @BE hybrid_sddmv_binary_f64_float
DEFINE_HYBRID_SDDMV_BINARY(_f64_float, double, float, BE::DType::Float32)

// @BE hybrid_sddmv_binary_f64_int8
DEFINE_HYBRID_SDDMV_BINARY(_f64_int8, double, int8_t, BE::DType::Int8)
// @BE hybrid_sddmv_binary_f64_double
DEFINE_HYBRID_SDDMV_BINARY(_f64_double, double, double, BE::DType::Float64)

#endif // BE_SDDMV_STANDALONE
