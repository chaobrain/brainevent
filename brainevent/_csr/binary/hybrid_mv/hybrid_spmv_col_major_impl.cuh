// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

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
#include "brainevent/hybrid_mv_schedule.cuh"
#include "brainevent/op_traits.cuh"
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

namespace
{
    using namespace hybrid_mv;

    // Weights are read through op_traits::ValueTraits<WeightT>; every edge adds
    // into the accumulator HybridOutT (float for f16/bf16, through
    // op_traits::AccTraits), so the caller rounds each finished sum once.
    template <typename WeightT>
    using HybridOutT = op_traits::AccType<WeightT>;

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRange(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        HybridOutT<WeightT> *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        HybridOutT<WeightT> scale)
    {
        using WeightTraits = op_traits::ValueTraits<WeightT>;
        using AccT = HybridOutT<WeightT>;
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
            op_traits::AccTraits<WeightT>::atomic_add(&output[idx], w);
        }
    }

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeThread(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        HybridOutT<WeightT> *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        HybridOutT<WeightT> scale)
    {
        using WeightTraits = op_traits::ValueTraits<WeightT>;
        using AccT = HybridOutT<WeightT>;
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
            op_traits::AccTraits<WeightT>::atomic_add(&output[idx], w);
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    __global__ void HybridSpmvColMajorExtractKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ perm,
        const OffsetT *__restrict__ indptr,
        const EventT *__restrict__ vector,
        HybridOutT<WeightT> *__restrict__ output,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        int32_t *__restrict__ status,
        int n_columns,
        int task_capacity)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        HybridOutT<WeightT> scale = 1;
        if constexpr (Weighted)
        {
            // Weighted events (buffer in the weight dtype): x != 0 is active
            // and scales every weight of the column.
            if (col >= n_columns)
            {
                return;
            }
            scale = op_traits::ValueTraits<WeightT>::read(__ldg(&vector[col]));
            if (scale == 0)
            {
                return;
            }
        }
        else
        {
            if (col >= n_columns || !op_traits::EventTraits<EventT>::active(__ldg(&vector[col])))
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

        int32_t n_chunks = TaskChunks(nnz, status);
        if (n_chunks <= 0)
        {
            // Chunk count beyond int32: keep every edge on this thread.
            ScatterRangeThread<Weighted>(weights, indices, perm, output, col_begin, col_end, scale);
            return;
        }

        int32_t base = atomicAdd(&status[kStatusCountIndex], n_chunks);
        if (base < 0 || base > task_capacity || n_chunks > task_capacity - base)
        {
            // A stale or undersized capacity never drops edges: the chunks that
            // fit are queued, the rest are scattered by this thread below.
            atomicExch(&status[kStatusOverflowIndex], 1);
        }

        for (int32_t chunk = 0; chunk < n_chunks; ++chunk)
        {
            OffsetT begin = col_begin + static_cast<OffsetT>(chunk) * static_cast<OffsetT>(kTaskNnz);
            OffsetT remaining = col_end - begin;
            OffsetT end = remaining > static_cast<OffsetT>(kTaskNnz)
                              ? begin + static_cast<OffsetT>(kTaskNnz) : col_end;
            if (base < 0 || chunk >= task_capacity - base)
            {
                ScatterRangeThread<Weighted>(weights, indices, perm, output, begin, end, scale);
                continue;
            }
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
        HybridOutT<WeightT> *__restrict__ output,
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
            HybridOutT<WeightT> scale = 1;
            if constexpr (Weighted)
            {
                // task_end holds the column of a weighted task (see the extract kernel).
                int col = static_cast<int>(__ldg(&task_end[task]));
                OffsetT col_end = __ldg(&indptr[col + 1]);
                OffsetT remaining = col_end - begin;
                end = remaining > static_cast<OffsetT>(kTaskNnz)
                          ? begin + static_cast<OffsetT>(kTaskNnz) : col_end;
                scale = op_traits::ValueTraits<WeightT>::read(__ldg(&vector[col]));
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

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    static ffi::Error LaunchHybridSpmvColMajor(
        const WeightT *weights,
        const IndexT *indices,
        const OffsetT *indptr,
        const OffsetT *perm,
        const EventT *vector,
        HybridOutT<WeightT> *output,
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
            status_code = cudaMemsetAsync(output, 0, static_cast<size_t>(n_rows) * sizeof(HybridOutT<WeightT>), stream);
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
            weights.dtype() == op_traits::ValueTraits<WeightT>::dtype,
            (std::string(name) + " expects matching weights dtype").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            op_traits::is_signed_index(indices.dtype()),
            (std::string(name) + " expects indices=s32/s64").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            op_traits::is_signed_index(indptr.dtype()) && perm.dtype() == indptr.dtype(),
            (std::string(name) + " expects indptr=s32/s64 and perm dtype to match indptr").c_str());
        BE_SPMV_COL_MAJOR_CHECK(
            (vector.dtype() == op_traits::event_dtype<WeightT, EventT, Weighted>()),
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
            output.dtype() == op_traits::AccTraits<WeightT>::dtype,
            (std::string(name) + " expects a float32 (f16/bf16/f32) or float64 accumulator output").c_str());
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
                output.data_ptr<HybridOutT<WeightT>>(),
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
