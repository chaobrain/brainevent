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
 * hybrid_sddmv_weighted.cu -- Weighted hybrid SDDMV relying on a caller-zeroed dweight
 * =============================================================================
 *
 * Weighted counterpart of the binary file: same scheduling, but the event value
 * is carried into the product:
 *
 *   dweight[j] = vector[row] * ct[indices[j]]
 *     for active row where vector[row] != 0
 *     for j in indptr[row]:indptr[row + 1]
 *   dweight[j] = 0 for every slot of an inactive row
 *
 * Strategy (default tuning values; configurable through CSRConfig):
 *   0. Clear status only; dweight arrives zeroed (aliased dweight_in).
 *   1. Extract active rows. Each thread reads its event value once. Rows with
 *      nnz <= 128 are gathered immediately by that thread, scaled by the value.
 *   2. Heavy active rows reserve contiguous block tasks once with atomicAdd.
 *      Each task stores begin/end offsets for up to 4096 slots and, in the
 *      value queue, the event value read in step 1.
 *   3. Launch a fixed number of 256-thread blocks. Each block reads the
 *      active task count on device and processes tasks with a grid-stride
 *      loop, avoiding D2H status reads and host synchronization.
 *
 * Value queue: heavy tasks carry their event value in
 * task_value[task] instead of recovering the row, so the block kernel never
 * reads indptr or vector and needs no shared memory or block barriers.
 *
 * Integration contract:
 *
 *     ct             : f16/bf16/f32/f64[k] cotangent
 *     indices        : s32/s64[nnz] global columns in slot order
 *     indptr         : s32/s64[m + 1]
 *     vector         : same dtype as ct [m]; active means != 0
 *     task_begin/end : same dtype as indptr, task_capacity entries
 *     task_value     : ct dtype [task_capacity] event value queue
 *     dweight_in     : ct dtype [nnz] caller-zeroed buffer aliased to dweight
 *     status         : s32[2] scratch/output {active_task_count, overflow_flag}
 *     dweight        : ct dtype [nnz] output
 *
 *   task_capacity bounds the tasks required by all heavy rows. The Python
 *   caller computes this bound from shapes and the captured CSRConfig.
 */

#include "hybrid_sddmv.cuh"
#include "../_common/cuda_traits.cuh"

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

    // dweight[j] = value * ct[indices[j]] for j = first, first + stride, ... < end.
    // A short row is walked by one thread (stride 1); a task by a block.
    template <typename ValueT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void GatherScaled(
        const ValueT *__restrict__ ct,
        const IndexT *__restrict__ indices,
        ValueT *__restrict__ dweight,
        ValueT value,
        OffsetT first,
        OffsetT end,
        OffsetT stride)
    {
        using Traits = cuda_traits::ValueTraits<ValueT>;
        const auto scale = Traits::read(value);
        for (OffsetT j = first; j < end; j += stride)
        {
            dweight[j] = Traits::write(scale * Traits::read(__ldg(&ct[__ldg(&indices[j])])));
        }
    }

    template <typename ValueT, typename IndexT, typename OffsetT>
    __global__ void HybridSddmvExtractKernel(
        const ValueT *__restrict__ ct,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        const ValueT *__restrict__ vector,
        ValueT *__restrict__ dweight,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        ValueT *__restrict__ task_value,
        int32_t *__restrict__ status,
        int m,
        int task_capacity)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= m)
        {
            return;
        }
        ValueT value = __ldg(&vector[row]);
        if (!cuda_traits::EventTraits<ValueT>::active(value))
        {
            return;
        }

        OffsetT row_begin = __ldg(&indptr[row]);
        OffsetT row_end = __ldg(&indptr[row + 1]);
        OffsetT nnz = row_end - row_begin;
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            GatherScaled(ct, indices, dweight, value, row_begin, row_end, OffsetT(1));
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
            task_value[task] = value;
        }
    }

    template <typename ValueT, typename IndexT, typename OffsetT>
    __global__ void HybridSddmvBlockKernel(
        const ValueT *__restrict__ ct,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ task_begin,
        const OffsetT *__restrict__ task_end,
        const ValueT *__restrict__ task_value,
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
            GatherScaled(ct, indices, dweight, __ldg(&task_value[task]),
                         __ldg(&task_begin[task]) + static_cast<OffsetT>(threadIdx.x),
                         __ldg(&task_end[task]), static_cast<OffsetT>(blockDim.x));
        }
    }

} // namespace

namespace hybrid_sddmv
{

    template <typename IndexT, typename OffsetT, typename ValueT>
    cudaError_t launch_weighted(
        const ValueT *ct,
        const IndexT *indices,
        const OffsetT *indptr,
        const ValueT *vector,
        ValueT *dweight,
        OffsetT *task_begin,
        OffsetT *task_end,
        ValueT *task_value,
        int32_t *device_status,
        int m,
        int64_t nnz,
        int64_t task_capacity_attr,
        cudaStream_t stream)
    {
        if (m < 0 || nnz < 0 || task_capacity_attr < 0 ||
            task_capacity_attr > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
            device_status == nullptr ||
            (task_capacity_attr > 0 &&
             (task_begin == nullptr || task_end == nullptr || task_value == nullptr)) ||
            (m > 0 && nnz > 0 &&
             (ct == nullptr || indices == nullptr || indptr == nullptr ||
              vector == nullptr || dweight == nullptr)))
        {
            return cudaErrorInvalidValue;
        }
        int task_capacity = static_cast<int>(task_capacity_attr);

        // dweight arrives zeroed from the caller (aliased dweight_in).
        cudaError_t status_code =
            cudaMemsetAsync(device_status, 0, static_cast<size_t>(2) * sizeof(int32_t), stream);
        if (status_code != cudaSuccess || m == 0 || nnz == 0)
        {
            return status_code;
        }

        int extract_blocks = 1 + (m - 1) / kBlockSize;
        HybridSddmvExtractKernel<ValueT, IndexT, OffsetT>
            <<<extract_blocks, kBlockSize, 0, stream>>>(
                ct, indices, indptr, vector, dweight,
                task_begin, task_end, task_value, device_status, m, task_capacity);
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
                    ct, indices, task_begin, task_end, task_value, device_status,
                    task_capacity, dweight);
            status_code = cudaGetLastError();
            if (status_code != cudaSuccess)
            {
                return status_code;
            }
        }
        return cudaSuccess;
    }

#define INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, INDEX_T, OFFSET_T)              \
    template cudaError_t launch_weighted<INDEX_T, OFFSET_T, VALUE_T>(        \
        const VALUE_T *, const INDEX_T *, const OFFSET_T *, const VALUE_T *, \
        VALUE_T *, OFFSET_T *, OFFSET_T *, VALUE_T *, int32_t *, int, int64_t, int64_t, cudaStream_t);
#define INSTANTIATE_HYBRID_INDICES(VALUE_T)                \
    INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, int32_t, int32_t) \
    INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, int32_t, int64_t) \
    INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, int64_t, int32_t) \
    INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, int64_t, int64_t)

    INSTANTIATE_HYBRID_INDICES(__half)
    INSTANTIATE_HYBRID_INDICES(__nv_bfloat16)
    INSTANTIATE_HYBRID_INDICES(float)
    INSTANTIATE_HYBRID_INDICES(double)
#undef INSTANTIATE_HYBRID_INDICES
#undef INSTANTIATE_HYBRID_WEIGHTED

} // namespace hybrid_sddmv

#ifndef BE_SDDMV_STANDALONE
#include "cuda_common.h"
#include "brainevent/common.h"
#include "brainevent/dtypes.h"
#include "../_common/op_traits.cuh"

#ifndef BE_HYBRID_SDDMV_CHECK
#define BE_HYBRID_SDDMV_CHECK(condition, message)  \
    do                                             \
    {                                              \
        BE_CHECK(condition) << (message);          \
    } while (0)
#endif

#define DEFINE_HYBRID_SDDMV_WEIGHTED(SUFFIX, VALUE_T)                                                       \
    void hybrid_sddmv_weighted##SUFFIX(                                                                     \
        const BE::Tensor ct,                                                                                \
        const BE::Tensor indices,                                                                           \
        const BE::Tensor indptr,                                                                            \
        const BE::Tensor vector,                                                                            \
        const BE::Tensor dweight_in,                                                                        \
        const BE::Tensor task_begin_scratch,                                                                \
        const BE::Tensor task_end_scratch,                                                                  \
        const BE::Tensor task_value_scratch,                                                                \
        const BE::Tensor status_scratch,                                                                    \
        BE::Tensor dweight,                                                                                 \
        BE::Tensor task_begin_output,                                                                       \
        BE::Tensor task_end_output,                                                                         \
        BE::Tensor task_value_output,                                                                       \
        BE::Tensor status_output,                                                                           \
        int64_t task_capacity,                                                                              \
        int64_t stream)                                                                                     \
    {                                                                                                       \
        constexpr BE::DType value_dtype = op_traits::ValueTraits<VALUE_T>::dtype;                           \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            ct.dtype() == value_dtype && vector.dtype() == value_dtype &&                                   \
                dweight_in.dtype() == value_dtype && dweight.dtype() == value_dtype,                        \
            "hybrid_sddmv_weighted" #SUFFIX " expects matching ct, vector and dweight dtype");              \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            op_traits::is_signed_index(indices.dtype()) && op_traits::is_signed_index(indptr.dtype()),      \
            "hybrid_sddmv_weighted" #SUFFIX " expects indices/indptr=s32/s64");                             \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            task_begin_scratch.dtype() == indptr.dtype() && task_end_scratch.dtype() == indptr.dtype() &&   \
                task_begin_output.dtype() == indptr.dtype() && task_end_output.dtype() == indptr.dtype() && \
                task_value_scratch.dtype() == value_dtype && task_value_output.dtype() == value_dtype &&    \
                status_scratch.dtype() == BE::DType::Int32 && status_output.dtype() == BE::DType::Int32,    \
            "hybrid_sddmv_weighted" #SUFFIX                                                                 \
            " expects task ranges in the indptr dtype, task values in the ct dtype and status=s32");        \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            ct.ndim() == 1 && indices.ndim() == 1 && indptr.ndim() == 1 && vector.ndim() == 1 &&            \
                dweight_in.ndim() == 1 && dweight.ndim() == 1 &&                                            \
                task_begin_scratch.ndim() == 1 && task_end_scratch.ndim() == 1 &&                           \
                task_value_scratch.ndim() == 1 && status_scratch.ndim() == 1 &&                             \
                task_begin_output.ndim() == 1 && task_end_output.ndim() == 1 &&                             \
                task_value_output.ndim() == 1 && status_output.ndim() == 1,                                 \
            "hybrid_sddmv_weighted" #SUFFIX " expects rank-1 buffers");                                     \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            dweight_in.numel() == indices.numel() && dweight.numel() == indices.numel(),                    \
            "hybrid_sddmv_weighted" #SUFFIX " expects dweight_in, dweight and indices lengths to match");   \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            indptr.numel() == vector.numel() + 1,                                                           \
            "hybrid_sddmv_weighted" #SUFFIX " expects indptr length to be vector length plus one");         \
        const size_t capacity = static_cast<size_t>(task_capacity);                                         \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            task_capacity >= 0 &&                                                                           \
                task_begin_scratch.numel() == capacity && task_end_scratch.numel() == capacity &&           \
                task_value_scratch.numel() == capacity && task_begin_output.numel() == capacity &&          \
                task_end_output.numel() == capacity && task_value_output.numel() == capacity,               \
            "hybrid_sddmv_weighted" #SUFFIX " expects task scratch lengths to match task_capacity");        \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            status_scratch.numel() == 2 && status_output.numel() == 2,                                      \
            "hybrid_sddmv_weighted" #SUFFIX " expects status scratch length 2");                            \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            indptr.numel() - 1 <= static_cast<size_t>(std::numeric_limits<int>::max()),                     \
            "hybrid_sddmv_weighted" #SUFFIX " expects at most INT32_MAX rows");                             \
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);                                  \
        cudaError_t launch_status = cudaSuccess;                                                            \
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {                   \
            launch_status = hybrid_sddmv::launch_weighted<IndexT, OffsetT, VALUE_T>(                        \
                ct.data_ptr<const VALUE_T>(),                                                               \
                indices.data_ptr<const IndexT>(),                                                           \
                indptr.data_ptr<const OffsetT>(),                                                           \
                vector.data_ptr<const VALUE_T>(),                                                           \
                dweight.data_ptr<VALUE_T>(),                                                                \
                task_begin_output.data_ptr<OffsetT>(),                                                      \
                task_end_output.data_ptr<OffsetT>(),                                                        \
                task_value_output.data_ptr<VALUE_T>(),                                                      \
                status_output.data_ptr<int32_t>(),                                                          \
                static_cast<int>(indptr.numel()) - 1,                                                       \
                static_cast<int64_t>(indices.numel()),                                                      \
                task_capacity,                                                                              \
                cuda_stream);                                                                               \
        });                                                                                                 \
        BE_HYBRID_SDDMV_CHECK(launch_status == cudaSuccess,                                                 \
                              "hybrid_sddmv_weighted" #SUFFIX " launch failed");                            \
    }

// @BE hybrid_sddmv_weighted_f16
DEFINE_HYBRID_SDDMV_WEIGHTED(_f16, __half)

// @BE hybrid_sddmv_weighted_bf16
DEFINE_HYBRID_SDDMV_WEIGHTED(_bf16, __nv_bfloat16)

// @BE hybrid_sddmv_weighted_f32
DEFINE_HYBRID_SDDMV_WEIGHTED(_f32, float)

// @BE hybrid_sddmv_weighted_f64
DEFINE_HYBRID_SDDMV_WEIGHTED(_f64, double)

#endif // BE_SDDMV_STANDALONE
