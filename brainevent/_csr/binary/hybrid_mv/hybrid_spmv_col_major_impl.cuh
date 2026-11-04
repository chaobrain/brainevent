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
 * Heterogeneous column-major SpMV over a cached structural mirror of W(M, N).
 * weights stay in their original CSR order; the caller supplies mirror
 * (indptr, indices, perm), where perm[j] maps a mirror edge to its weight slot.
 *
 *   output[indices[j]] += weights[perm[j]] * scale
 *   scale = vector[col] for weighted, or 1 for active binary events.
 *
 * Short columns (nnz <= 128) scatter inside extraction. Longer columns enqueue
 * chunks of at most 4096 mirror edges for the block kernel. Both scatter paths
 * map only weight reads; task_begin/end always describe the mirror structure.
 * Weighted task_end stores the column id to recover its end and event value.
 *
 * FFI and workspace contracts are documented in README.md. The caller owns the
 * mirror and workspace; no weight gather or host synchronization is performed.
 */

#ifndef DEVING_HYBRID_SPMV_COL_MAJOR_IMPL_CUH
#define DEVING_HYBRID_SPMV_COL_MAJOR_IMPL_CUH

#include "cuda_common.h"
#include "brainevent/common.h"
#include "brainevent/dtypes.h"
#include "xla/ffi/api/ffi.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>

namespace ffi = xla::ffi;

#ifndef BE_SPMV_COL_MAJOR_CHECK
// Host-side check: throws BE::CheckError, which the generated FFI wrapper
// converts into an xla::ffi::Error (a Python exception) instead of terminating the process.
#define BE_SPMV_COL_MAJOR_CHECK(condition, message) \
    do                                                \
    {                                                 \
        BE_CHECK(condition) << (message);             \
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
    constexpr int64_t kInt32Max = 2147483647LL;

    static_assert(kBlockSize > 0, "BE_HYBRID_BLOCK_SIZE must be positive");
    static_assert(kBlockSize % 32 == 0, "BE_HYBRID_BLOCK_SIZE must be a multiple of 32");
    static_assert(kBlockSize <= 1024, "BE_HYBRID_BLOCK_SIZE must not exceed 1024");
    static_assert(kFixedScatterBlocks > 0, "BE_HYBRID_FIXED_SCATTER_BLOCKS must be positive");
    static_assert(kTprThreshold >= 0, "BE_HYBRID_TPR_THRESHOLD must be non-negative");
    static_assert(kTaskNnz > 0, "BE_HYBRID_TASK_NNZ must be positive");

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

    // dtype of the event buffer: the binary event type, or the weight dtype for
    // weighted kernels, which read the event value as a scale.
    template <typename WeightT, typename EventT, bool Weighted>
    struct HybridEventDType
    {
        static constexpr BE::DType value = HybridEventTraits<EventT>::dtype;
    };

    template <typename WeightT, typename EventT>
    struct HybridEventDType<WeightT, EventT, true>
    {
        static constexpr BE::DType value = HybridWeightTraits<WeightT>::dtype;
    };

    template <typename OffsetT>
    __device__ __forceinline__ int32_t TaskChunksForColumnNnz(
        OffsetT nnz,
        int32_t *__restrict__ status)
    {
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            return 0;
        }
        OffsetT chunks =
            1 + (nnz - 1) / static_cast<OffsetT>(kTaskNnz);
        if (static_cast<int64_t>(chunks) > kInt32Max)
        {
            atomicExch(&status[kStatusOverflowIndex], 1);
            return 0;
        }
        return static_cast<int32_t>(chunks);
    }

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRange(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        WeightT *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        typename HybridWeightTraits<WeightT>::AccT scale)
    {
        using WeightTraits = HybridWeightTraits<WeightT>;
        using AccT = typename WeightTraits::AccT;
        for (int64_t j = static_cast<int64_t>(begin) + threadIdx.x;
             j < static_cast<int64_t>(end);
             j += blockDim.x)
        {
            IndexT idx = __ldg(&indices[j]);
            // j is a mirror edge slot; indices[j] remains the output row.
            OffsetT weight_pos = __ldg(&perm[j]);
            AccT w = WeightTraits::read(__ldg(&weights[weight_pos]));
            if constexpr (Weighted)
            {
                w *= scale;
            }
            WeightTraits::atomic_add(&output[idx], w);
        }
    }

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeThread(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        WeightT *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        typename HybridWeightTraits<WeightT>::AccT scale)
    {
        using WeightTraits = HybridWeightTraits<WeightT>;
        using AccT = typename WeightTraits::AccT;
        for (OffsetT j = begin; j < end; ++j)
        {
            IndexT idx = __ldg(&indices[j]);
            // j is a mirror edge slot; indices[j] remains the output row.
            OffsetT weight_pos = __ldg(&perm[j]);
            AccT w = WeightTraits::read(__ldg(&weights[weight_pos]));
            if constexpr (Weighted)
            {
                w *= scale;
            }
            WeightTraits::atomic_add(&output[idx], w);
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    __global__ void HybridSpmvColMajorExtractKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        const OffsetT *__restrict__ indptr,
        const EventT *__restrict__ vector,
        WeightT *__restrict__ output,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        int32_t *__restrict__ status,
        int n_columns,
        int task_capacity)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        typename HybridWeightTraits<WeightT>::AccT scale = 1;
        if constexpr (Weighted)
        {
            // Weighted events (buffer in the weight dtype): x != 0 is active
            // and scales every weight of the column.
            if (col >= n_columns)
            {
                return;
            }
            scale = HybridWeightTraits<WeightT>::read(__ldg(&vector[col]));
            if (scale == 0)
            {
                return;
            }
        }
        else
        {
            if (col >= n_columns || !HybridEventTraits<EventT>::active(__ldg(&vector[col])))
            {
                return;
            }
        }

        OffsetT col_begin = __ldg(&indptr[col]);
        OffsetT col_end = __ldg(&indptr[col + 1]);
        OffsetT nnz = col_end - col_begin;
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            ScatterRangeThread<Weighted>(weights, indices, perm, output, col_begin, col_end, scale);
            return;
        }

        int32_t n_chunks = TaskChunksForColumnNnz(nnz, status);
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
            OffsetT begin = col_begin + static_cast<OffsetT>(chunk) * static_cast<OffsetT>(kTaskNnz);
            OffsetT remaining = col_end - begin;
            OffsetT end = remaining > static_cast<OffsetT>(kTaskNnz)
                              ? begin + static_cast<OffsetT>(kTaskNnz) : col_end;
            int32_t task = base + chunk;
            task_begin[task] = begin;
            if constexpr (Weighted)
            {
                // Weighted tasks record the column instead of the end offset so
                // the block kernel can re-read its event value.
                task_end[task] = static_cast<OffsetT>(col);
            }
            else
            {
                task_end[task] = end;
            }
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    __global__ void HybridSpmvColMajorBlockKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        const OffsetT *__restrict__ task_begin,
        const OffsetT *__restrict__ task_end,
        const int32_t *__restrict__ status,
        int task_capacity,
        WeightT *__restrict__ output,
        const OffsetT *__restrict__ indptr,
        const EventT *__restrict__ vector)
    {
        int active = __ldg(&status[kStatusCountIndex]);
        if (active > task_capacity)
        {
            active = task_capacity;
        }
        for (int task = blockIdx.x; task < active; task += gridDim.x)
        {
            OffsetT begin = __ldg(&task_begin[task]);
            OffsetT end;
            typename HybridWeightTraits<WeightT>::AccT scale = 1;
            if constexpr (Weighted)
            {
                // task_end holds the column of a weighted task (see the extract kernel).
                int col = static_cast<int>(__ldg(&task_end[task]));
                OffsetT col_end = __ldg(&indptr[col + 1]);
                OffsetT remaining = col_end - begin;
                end = remaining > static_cast<OffsetT>(kTaskNnz)
                          ? begin + static_cast<OffsetT>(kTaskNnz) : col_end;
                scale = HybridWeightTraits<WeightT>::read(__ldg(&vector[col]));
            }
            else
            {
                end = __ldg(&task_end[task]);
            }
            ScatterRange<Weighted>(weights, indices, perm, output, begin, end, scale);
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

    static bool IsSignedIndexDType(BE::DType dtype)
    {
        return dtype == BE::DType::Int32 || dtype == BE::DType::Int64;
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    static ffi::Error LaunchHybridSpmvColMajor(
        const WeightT *weights,
        const IndexT *indices,
        const OffsetT *indptr,
        const OffsetT *perm,
        const EventT *vector,
        WeightT *output,
        OffsetT *task_begin,
        OffsetT *task_end,
        int32_t *device_status,
        int n_columns,
        int n_rows,
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

        cudaError_t status_code = cudaSuccess;
        if (n_rows > 0)
        {
            status_code = cudaMemsetAsync(output, 0, static_cast<size_t>(n_rows) * sizeof(WeightT), stream);
            if (status_code != cudaSuccess)
            {
                return InternalCudaError("cudaMemsetAsync(output) failed", status_code);
            }
        }

        size_t status_bytes = static_cast<size_t>(2) * sizeof(int32_t);
        if (device_status == nullptr)
        {
            return InternalMessage("device_status scratch buffer is null");
        }
        if (task_capacity > 0 && (task_begin == nullptr || task_end == nullptr))
        {
            return InternalMessage("hybrid task scratch buffer is null");
        }

        status_code = cudaMemsetAsync(device_status, 0, status_bytes, stream);
        if (status_code != cudaSuccess)
        {
            return InternalCudaError("cudaMemsetAsync(device_status) failed",
                                     status_code);
        }

        if (n_columns == 0 || nnz == 0)
        {
            return ffi::Error::Success();
        }

        int extract_blocks = 1 + (n_columns - 1) / kBlockSize;
        HybridSpmvColMajorExtractKernel<WeightT, EventT, IndexT, OffsetT, Weighted>
            <<<extract_blocks, kBlockSize, 0, stream>>>(
                weights, indices, perm, indptr, vector, output,
                task_begin, task_end, device_status, n_columns, task_capacity);
        status_code = cudaGetLastError();
        if (status_code != cudaSuccess)
        {
            return InternalCudaError(
                "HybridSpmvColMajorExtractKernel launch failed",
                status_code);
        }

        if (task_capacity > 0)
        {
            int scatter_blocks = task_capacity < kFixedScatterBlocks
                                     ? task_capacity
                                     : kFixedScatterBlocks;
            HybridSpmvColMajorBlockKernel<WeightT, EventT, IndexT, OffsetT, Weighted>
                <<<scatter_blocks, kBlockSize, 0, stream>>>(
                    weights, indices, perm, task_begin, task_end, device_status,
                    task_capacity, output, indptr, vector);
            status_code = cudaGetLastError();
            if (status_code != cudaSuccess)
            {
                return InternalCudaError(
                    "HybridSpmvColMajorBlockKernel launch failed",
                    status_code);
            }
        }
        return ffi::Error::Success();
    }

    template <typename WeightT, typename EventT, bool Weighted>
    void HybridSpmvColMajorHetero(
        const char *name,
        const BE::Tensor weights,
        const BE::Tensor indices,
        const BE::Tensor indptr,
        const BE::Tensor perm,
        const BE::Tensor vector,
        const BE::Tensor task_begin_scratch,
        const BE::Tensor task_end_scratch,
        const BE::Tensor status_scratch,
        BE::Tensor output,
        BE::Tensor task_begin_output,
        BE::Tensor task_end_output,
        BE::Tensor status_output,
        int64_t task_capacity,
        int64_t stream)
    {
        BE_SPMV_COL_MAJOR_CHECK(
            weights.dtype() == HybridWeightTraits<WeightT>::dtype,
            (std::string(name) + " expects matching weights dtype").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            IsSignedIndexDType(indices.dtype()),
            (std::string(name) + " expects indices=s32/s64").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            IsSignedIndexDType(indptr.dtype()) && perm.dtype() == indptr.dtype(),
            (std::string(name) + " expects indptr=s32/s64 and perm dtype to match indptr").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            (vector.dtype() == HybridEventDType<WeightT, EventT, Weighted>::value),
            (std::string(name) + " expects matching vector dtype").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            task_begin_scratch.dtype() == indptr.dtype() &&
                task_end_scratch.dtype() == indptr.dtype() &&
                status_scratch.dtype() == BE::DType::Int32 &&
                task_begin_output.dtype() == indptr.dtype() &&
                task_end_output.dtype() == indptr.dtype() &&
                status_output.dtype() == BE::DType::Int32,
            (std::string(name) + " expects status=s32 and task scratch dtype to match indptr").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            output.dtype() == HybridWeightTraits<WeightT>::dtype,
            (std::string(name) + " expects matching output dtype").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            weights.ndim() == 1 && indices.ndim() == 1 &&
                indptr.ndim() == 1 && perm.ndim() == 1 && vector.ndim() == 1 &&
                task_begin_scratch.ndim() == 1 && task_end_scratch.ndim() == 1 &&
                status_scratch.ndim() == 1 && output.ndim() == 1 &&
                task_begin_output.ndim() == 1 && task_end_output.ndim() == 1 &&
                status_output.ndim() == 1,
            (std::string(name) + " expects rank-1 buffers").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            weights.numel() == indices.numel() && perm.numel() == indices.numel(),
            (std::string(name) + " expects weights/perm/indices lengths to match").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            indptr.numel() == vector.numel() + 1,
            (std::string(name) + " expects indptr length to be vector length plus one").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            task_capacity >= 0,
            (std::string(name) + " expects non-negative task_capacity").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            task_begin_scratch.numel() == static_cast<size_t>(task_capacity) &&
                task_end_scratch.numel() == static_cast<size_t>(task_capacity) &&
                task_begin_output.numel() == static_cast<size_t>(task_capacity) &&
                task_end_output.numel() == static_cast<size_t>(task_capacity),
            (std::string(name) + " expects task scratch lengths to match task_capacity").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            status_scratch.numel() == 2 && status_output.numel() == 2,
            (std::string(name) + " expects status scratch length 2").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            vector.numel() <= static_cast<size_t>(kInt32Max) &&
                output.numel() <= static_cast<size_t>(kInt32Max),
            (std::string(name) + " expects row/column counts within int32 range").c_str());
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        bool ok = true;
        std::string launch_error;
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
            ffi::Error error = LaunchHybridSpmvColMajor<WeightT, EventT, IndexT, OffsetT, Weighted>(
                weights.data_ptr<const WeightT>(),
                indices.data_ptr<const IndexT>(),
                indptr.data_ptr<const OffsetT>(),
                perm.data_ptr<const OffsetT>(),
                vector.data_ptr<const EventT>(),
                output.data_ptr<WeightT>(),
                task_begin_output.data_ptr<OffsetT>(),
                task_end_output.data_ptr<OffsetT>(),
                status_output.data_ptr<int32_t>(),
                static_cast<int>(vector.numel()),
                static_cast<int>(output.numel()),
                static_cast<int64_t>(indices.numel()),
                task_capacity,
                cuda_stream);
            ok = error.success();
            if (!ok)
                launch_error = std::string(error.message());
        });
        BE_SPMV_COL_MAJOR_CHECK(ok, (std::string(name) + " launch failed: " + launch_error).c_str());
    }

} // namespace

#endif // DEVING_HYBRID_SPMV_COL_MAJOR_IMPL_CUH
