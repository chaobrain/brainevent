// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

/*
 * hybrid_spmv_impl.cuh -- Shared binary/weighted hybrid CSR scatter implementation
 * =============================================================================
 *
 * Experimental CUDA backend for event-driven CSR transposed
 * matrix-vector multiplication:
 *
 *   output[indices[j]] += weights[j]
 *     for active row where vector[row] != 0
 *     for j in indptr[row]:indptr[row + 1]
 *
 * Strategy:
 *   1. Extract active rows. Rows with nnz <= 128 are scattered immediately by
 *      one thread per row.
 *   2. Heavy active rows reserve contiguous block tasks once with atomicAdd.
 *      Each task stores begin/end offsets for up to BE_HYBRID_TASK_NNZ nonzeros.
 *   3. Launch a fixed number of BE_HYBRID_BLOCK_SIZE-thread scatter blocks. Each block reads
 *      the active task count on device and processes tasks with a grid-stride
 *      loop, avoiding D2H status reads and host synchronization.
 *
 * Integration contract:
 *   This backend is not a plain four-buffer CSR call. The caller must provide
 *   matrix-structure metadata and reusable scratch buffers before entering the
 *   FFI handler:
 *
 *     weights        : floating[nnz] for hetero, floating[1] for homo
 *     indices        : s32/s64[nnz]
 *     indptr         : s32/s64[m + 1]
 *     vector         : bool/float32[m] for binary; weight dtype[m] for weighted
 *     task_begin/end : same dtype as indptr, scratch/output task ranges
 *     status         : s32[2] scratch/output {active_task_count, overflow_flag}
 *     output         : f32[k] for f16/bf16/f32 weights, f64[k] for f64; every
 *                      edge adds in this accumulator dtype and the caller
 *                      rounds the finished sum once to the weight dtype
 *
 *   The output length is the CSR column count supplied by the Python/JAX caller.
 *   Vector activity is read inside the extract kernel each launch.
 *
 * Required preprocessing before launch:
 *   Compute task_capacity as the worst-case sum across all static CSR rows:
 *     sum(row_nnz > BE_HYBRID_TPR_THRESHOLD ?
 *         ceil(row_nnz / BE_HYBRID_TASK_NNZ) : 0)
 *
 *   Then allocate task_begin/task_end with task_capacity entries each using the
 *   same dtype as indptr and status with two int32 entries. task_capacity is a
 *   worst-case capacity for all heavy rows; the fixed-block scatter kernel reads
 *   status[0] on device. A smaller (stale) capacity stays correct: chunks without
 *   a queue slot are scattered by the extracting thread and status[1] is set.
 *
 *   The hetero path reads weights[j]. The homo path reads weights[0] for every
 *   nonzero and otherwise reuses the same metadata and scratch contract.
 *
 * Weighted (``_wv``) entry points:
 *   The event buffer has the weight dtype; an active row (value != 0)
 *   scatters weight * value. Weighted heavy-row tasks store the row id in
 *   task_end instead of the end offset; the block kernel recomputes the end
 *   from indptr and re-reads the row value.
 */

#ifndef DEVING_HYBRID_SPMV_IMPL_CUH
#define DEVING_HYBRID_SPMV_IMPL_CUH

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

#ifndef BE_CSRMV_WAT_HYBRID_CHECK
// Host-side check: throws BE::CheckError, which the generated FFI wrapper
// converts into an xla::ffi::Error (a Python exception) instead of terminating the process.
#define BE_CSRMV_WAT_HYBRID_CHECK(condition, message) \
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
        HybridOutT<WeightT> *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        HybridOutT<WeightT> scale)
    {
        using WeightTraits = op_traits::ValueTraits<WeightT>;
        using AccT = HybridOutT<WeightT>;
        for (OffsetT j = begin + static_cast<OffsetT>(threadIdx.x);
             j < end;
             j += static_cast<OffsetT>(blockDim.x))
        {
            IndexT idx = __ldg(&indices[j]);
            AccT w = WeightTraits::read(__ldg(&weights[j]));
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
            AccT w = WeightTraits::read(__ldg(&weights[j]));
            if constexpr (Weighted)
            {
                w *= scale;
            }
            op_traits::AccTraits<WeightT>::atomic_add(&output[idx], w);
        }
    }

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeHomo(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        HybridOutT<WeightT> *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        HybridOutT<WeightT> scale)
    {
        using WeightTraits = op_traits::ValueTraits<WeightT>;
        using AccT = HybridOutT<WeightT>;
        AccT w = WeightTraits::read(__ldg(&weights[0]));
        if constexpr (Weighted)
        {
            w *= scale;
        }
        for (OffsetT j = begin + static_cast<OffsetT>(threadIdx.x);
             j < end;
             j += static_cast<OffsetT>(blockDim.x))
        {
            IndexT idx = __ldg(&indices[j]);
            op_traits::AccTraits<WeightT>::atomic_add(&output[idx], w);
        }
    }

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeThreadHomo(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        HybridOutT<WeightT> *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        HybridOutT<WeightT> scale)
    {
        using WeightTraits = op_traits::ValueTraits<WeightT>;
        using AccT = HybridOutT<WeightT>;
        AccT w = WeightTraits::read(__ldg(&weights[0]));
        if constexpr (Weighted)
        {
            w *= scale;
        }
        for (OffsetT j = begin; j < end; ++j)
        {
            IndexT idx = __ldg(&indices[j]);
            op_traits::AccTraits<WeightT>::atomic_add(&output[idx], w);
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    __global__ void BinaryCsrmvWatHybridExtractKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        const EventT *__restrict__ vector,
        HybridOutT<WeightT> *__restrict__ output,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        int32_t *__restrict__ status,
        int m,
        int task_capacity)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        HybridOutT<WeightT> scale = 1;
        if constexpr (Weighted)
        {
            // Weighted events (buffer in the weight dtype): x != 0 is active
            // and scales every weight of the row.
            if (row >= m)
            {
                return;
            }
            scale = op_traits::ValueTraits<WeightT>::read(__ldg(&vector[row]));
            if (scale == 0)
            {
                return;
            }
        }
        else
        {
            if (row >= m || !op_traits::EventTraits<EventT>::active(__ldg(&vector[row])))
            {
                return;
            }
        }

        OffsetT row_begin = __ldg(&indptr[row]);
        OffsetT row_end = __ldg(&indptr[row + 1]);
        OffsetT nnz = row_end - row_begin;
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            ScatterRangeThread<Weighted>(weights, indices, output, row_begin, row_end, scale);
            return;
        }

        int32_t n_chunks = TaskChunks(nnz, status);
        if (n_chunks <= 0)
        {
            // Chunk count beyond int32: keep every edge on this thread.
            ScatterRangeThread<Weighted>(weights, indices, output, row_begin, row_end, scale);
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
            OffsetT begin = row_begin + static_cast<OffsetT>(chunk) * static_cast<OffsetT>(kTaskNnz);
            OffsetT end = begin + static_cast<OffsetT>(kTaskNnz);
            if (end > row_end)
            {
                end = row_end;
            }
            if (base < 0 || chunk >= task_capacity - base)
            {
                ScatterRangeThread<Weighted>(weights, indices, output, begin, end, scale);
                continue;
            }
            int32_t task = base + chunk;
            task_begin[task] = begin;
            if constexpr (Weighted)
            {
                // Weighted tasks record the row instead of the end offset so
                // the block kernel can re-read the event value of the row.
                task_end[task] = static_cast<OffsetT>(row);
            }
            else
            {
                task_end[task] = end;
            }
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    __global__ void BinaryCsrmvWatHybridBlockKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
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
                // task_end holds the row of a weighted task (see the extract kernel).
                int row = static_cast<int>(__ldg(&task_end[task]));
                OffsetT row_end = __ldg(&indptr[row + 1]);
                end = begin + static_cast<OffsetT>(kTaskNnz);
                if (end > row_end)
                {
                    end = row_end;
                }
                scale = op_traits::ValueTraits<WeightT>::read(__ldg(&vector[row]));
            }
            else
            {
                end = __ldg(&task_end[task]);
            }
            ScatterRange<Weighted>(weights, indices, output, begin, end, scale);
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    __global__ void BinaryCsrmvWatHybridHomoExtractKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        const EventT *__restrict__ vector,
        HybridOutT<WeightT> *__restrict__ output,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        int32_t *__restrict__ status,
        int m,
        int task_capacity)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        HybridOutT<WeightT> scale = 1;
        if constexpr (Weighted)
        {
            // Weighted events (buffer in the weight dtype): x != 0 is active
            // and scales every weight of the row.
            if (row >= m)
            {
                return;
            }
            scale = op_traits::ValueTraits<WeightT>::read(__ldg(&vector[row]));
            if (scale == 0)
            {
                return;
            }
        }
        else
        {
            if (row >= m || !op_traits::EventTraits<EventT>::active(__ldg(&vector[row])))
            {
                return;
            }
        }

        OffsetT row_begin = __ldg(&indptr[row]);
        OffsetT row_end = __ldg(&indptr[row + 1]);
        OffsetT nnz = row_end - row_begin;
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            ScatterRangeThreadHomo<Weighted>(weights, indices, output, row_begin, row_end, scale);
            return;
        }

        int32_t n_chunks = TaskChunks(nnz, status);
        if (n_chunks <= 0)
        {
            // Chunk count beyond int32: keep every edge on this thread.
            ScatterRangeThreadHomo<Weighted>(weights, indices, output, row_begin, row_end, scale);
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
            OffsetT begin = row_begin + static_cast<OffsetT>(chunk) * static_cast<OffsetT>(kTaskNnz);
            OffsetT end = begin + static_cast<OffsetT>(kTaskNnz);
            if (end > row_end)
            {
                end = row_end;
            }
            if (base < 0 || chunk >= task_capacity - base)
            {
                ScatterRangeThreadHomo<Weighted>(weights, indices, output, begin, end, scale);
                continue;
            }
            int32_t task = base + chunk;
            task_begin[task] = begin;
            if constexpr (Weighted)
            {
                // Weighted tasks record the row instead of the end offset so
                // the block kernel can re-read the event value of the row.
                task_end[task] = static_cast<OffsetT>(row);
            }
            else
            {
                task_end[task] = end;
            }
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    __global__ void BinaryCsrmvWatHybridHomoBlockKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
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
                // task_end holds the row of a weighted task (see the extract kernel).
                int row = static_cast<int>(__ldg(&task_end[task]));
                OffsetT row_end = __ldg(&indptr[row + 1]);
                end = begin + static_cast<OffsetT>(kTaskNnz);
                if (end > row_end)
                {
                    end = row_end;
                }
                scale = op_traits::ValueTraits<WeightT>::read(__ldg(&vector[row]));
            }
            else
            {
                end = __ldg(&task_end[task]);
            }
            ScatterRangeHomo<Weighted>(weights, indices, output, begin, end, scale);
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
    static ffi::Error LaunchBinaryCsrmvWatHybrid(
        const WeightT *weights,
        const IndexT *indices,
        const OffsetT *indptr,
        const EventT *vector,
        HybridOutT<WeightT> *output,
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
            cudaMemsetAsync(output, 0, static_cast<size_t>(k) * sizeof(HybridOutT<WeightT>), stream);
        if (status_code != cudaSuccess)
        {
            return InternalCudaError("cudaMemsetAsync(output) failed", status_code);
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

        // Status is an output of every call, including empty structures.
        if (m <= 0 || nnz == 0)
        {
            return ffi::Error::Success();
        }

        int extract_blocks = (m + kBlockSize - 1) / kBlockSize;
        BinaryCsrmvWatHybridExtractKernel<WeightT, EventT, IndexT, OffsetT, Weighted>
            <<<extract_blocks, kBlockSize, 0, stream>>>(
                weights, indices, indptr, vector, output,
                task_begin, task_end, device_status, m, task_capacity);
        status_code = cudaGetLastError();
        if (status_code != cudaSuccess)
        {
            return InternalCudaError(
                "BinaryCsrmvWatHybridExtractKernel launch failed",
                status_code);
        }

        if (task_capacity > 0)
        {
            int scatter_blocks = task_capacity < kFixedScatterBlocks
                                     ? task_capacity
                                     : kFixedScatterBlocks;
            BinaryCsrmvWatHybridBlockKernel<WeightT, EventT, IndexT, OffsetT, Weighted>
                <<<scatter_blocks, kBlockSize, 0, stream>>>(
                    weights, indices, task_begin, task_end, device_status,
                    task_capacity, output, indptr, vector);
            status_code = cudaGetLastError();
            if (status_code != cudaSuccess)
            {
                return InternalCudaError(
                    "BinaryCsrmvWatHybridBlockKernel launch failed",
                    status_code);
            }
        }
        return ffi::Error::Success();
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    static ffi::Error LaunchBinaryCsrmvWatHybridHomo(
        const WeightT *weights,
        const IndexT *indices,
        const OffsetT *indptr,
        const EventT *vector,
        HybridOutT<WeightT> *output,
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
            cudaMemsetAsync(output, 0, static_cast<size_t>(k) * sizeof(HybridOutT<WeightT>), stream);
        if (status_code != cudaSuccess)
        {
            return InternalCudaError("cudaMemsetAsync(output) failed", status_code);
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

        // Status is an output of every call, including empty structures.
        if (m <= 0 || nnz == 0)
        {
            return ffi::Error::Success();
        }

        int extract_blocks = (m + kBlockSize - 1) / kBlockSize;
        BinaryCsrmvWatHybridHomoExtractKernel<WeightT, EventT, IndexT, OffsetT, Weighted>
            <<<extract_blocks, kBlockSize, 0, stream>>>(
                weights, indices, indptr, vector, output,
                task_begin, task_end, device_status, m, task_capacity);
        status_code = cudaGetLastError();
        if (status_code != cudaSuccess)
        {
            return InternalCudaError(
                "BinaryCsrmvWatHybridHomoExtractKernel launch failed",
                status_code);
        }

        if (task_capacity > 0)
        {
            int scatter_blocks = task_capacity < kFixedScatterBlocks
                                     ? task_capacity
                                     : kFixedScatterBlocks;
            BinaryCsrmvWatHybridHomoBlockKernel<WeightT, EventT, IndexT, OffsetT, Weighted>
                <<<scatter_blocks, kBlockSize, 0, stream>>>(
                    weights, indices, task_begin, task_end, device_status,
                    task_capacity, output, indptr, vector);
            status_code = cudaGetLastError();
            if (status_code != cudaSuccess)
            {
                return InternalCudaError(
                    "BinaryCsrmvWatHybridHomoBlockKernel launch failed",
                    status_code);
            }
        }
        return ffi::Error::Success();
    }

    template <typename WeightT, typename EventT, bool Weighted>
    void HybridSpmvHetero(
        const char *name,
        const BE::Tensor weights,
        const BE::Tensor indices,
        const BE::Tensor indptr,
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
        BE_CSRMV_WAT_HYBRID_CHECK(
            weights.dtype() == op_traits::ValueTraits<WeightT>::dtype,
            (std::string(name) + " expects matching weights dtype").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            op_traits::is_signed_index(indices.dtype()),
            (std::string(name) + " expects indices=s32/s64").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            op_traits::is_signed_index(indptr.dtype()),
            (std::string(name) + " expects indptr=s32/s64").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            (vector.dtype() == op_traits::event_dtype<WeightT, EventT, Weighted>()),
            (std::string(name) + " expects matching vector dtype").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            task_begin_scratch.dtype() == indptr.dtype() &&
                task_end_scratch.dtype() == indptr.dtype() &&
                status_scratch.dtype() == BE::DType::Int32 &&
                task_begin_output.dtype() == indptr.dtype() &&
                task_end_output.dtype() == indptr.dtype() &&
                status_output.dtype() == BE::DType::Int32,
            (std::string(name) + " expects status=s32 and task scratch dtype to match indptr").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            output.dtype() == op_traits::AccTraits<WeightT>::dtype,
            (std::string(name) + " expects a float32 (f16/bf16/f32) or float64 accumulator output").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            weights.ndim() == 1 && indices.ndim() == 1 &&
                indptr.ndim() == 1 && vector.ndim() == 1 &&
                task_begin_scratch.ndim() == 1 && task_end_scratch.ndim() == 1 &&
                status_scratch.ndim() == 1 && output.ndim() == 1 &&
                task_begin_output.ndim() == 1 && task_end_output.ndim() == 1 &&
                status_output.ndim() == 1,
            (std::string(name) + " expects rank-1 buffers").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            weights.numel() == indices.numel(),
            (std::string(name) + " expects weights and indices lengths to match").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            indptr.numel() == vector.numel() + 1,
            (std::string(name) + " expects indptr length to be vector length plus one").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            task_capacity >= 0,
            (std::string(name) + " expects non-negative task_capacity").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            task_begin_scratch.numel() == static_cast<size_t>(task_capacity) &&
                task_end_scratch.numel() == static_cast<size_t>(task_capacity) &&
                task_begin_output.numel() == static_cast<size_t>(task_capacity) &&
                task_end_output.numel() == static_cast<size_t>(task_capacity),
            (std::string(name) + " expects task scratch lengths to match task_capacity").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            status_scratch.numel() == 2 && status_output.numel() == 2,
            (std::string(name) + " expects status scratch length 2").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            vector.numel() <= static_cast<size_t>(kInt32Max) &&
                output.numel() <= static_cast<size_t>(kInt32Max),
            (std::string(name) + " expects row/column counts within int32 range").c_str());
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        bool ok = true;
        std::string launch_error;
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
            ffi::Error error = LaunchBinaryCsrmvWatHybrid<WeightT, EventT, IndexT, OffsetT, Weighted>(
                weights.data_ptr<const WeightT>(),
                indices.data_ptr<const IndexT>(),
                indptr.data_ptr<const OffsetT>(),
                vector.data_ptr<const EventT>(),
                output.data_ptr<HybridOutT<WeightT>>(),
                task_begin_output.data_ptr<OffsetT>(),
                task_end_output.data_ptr<OffsetT>(),
                status_output.data_ptr<int32_t>(),
                static_cast<int>(indptr.numel()) - 1,
                static_cast<int>(output.numel()),
                static_cast<int64_t>(indices.numel()),
                task_capacity,
                cuda_stream);
            ok = error.success();
            if (!ok)
                launch_error = std::string(error.message());
        });
        BE_CSRMV_WAT_HYBRID_CHECK(ok, (std::string(name) + " launch failed: " + launch_error).c_str());
    }

    template <typename WeightT, typename EventT, bool Weighted>
    void HybridSpmvHomo(
        const char *name,
        const BE::Tensor weights,
        const BE::Tensor indices,
        const BE::Tensor indptr,
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
        BE_CSRMV_WAT_HYBRID_CHECK(
            weights.dtype() == op_traits::ValueTraits<WeightT>::dtype && weights.numel() == 1,
            (std::string(name) + " expects matching weights dtype and length 1").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            op_traits::is_signed_index(indices.dtype()),
            (std::string(name) + " expects indices=s32/s64").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            op_traits::is_signed_index(indptr.dtype()),
            (std::string(name) + " expects indptr=s32/s64").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            (vector.dtype() == op_traits::event_dtype<WeightT, EventT, Weighted>()),
            (std::string(name) + " expects matching vector dtype").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            task_begin_scratch.dtype() == indptr.dtype() &&
                task_end_scratch.dtype() == indptr.dtype() &&
                status_scratch.dtype() == BE::DType::Int32 &&
                task_begin_output.dtype() == indptr.dtype() &&
                task_end_output.dtype() == indptr.dtype() &&
                status_output.dtype() == BE::DType::Int32,
            (std::string(name) + " expects status=s32 and task scratch dtype to match indptr").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            output.dtype() == op_traits::AccTraits<WeightT>::dtype,
            (std::string(name) + " expects a float32 (f16/bf16/f32) or float64 accumulator output").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            weights.ndim() == 1 && indices.ndim() == 1 &&
                indptr.ndim() == 1 && vector.ndim() == 1 &&
                task_begin_scratch.ndim() == 1 && task_end_scratch.ndim() == 1 &&
                status_scratch.ndim() == 1 && output.ndim() == 1 &&
                task_begin_output.ndim() == 1 && task_end_output.ndim() == 1 &&
                status_output.ndim() == 1,
            (std::string(name) + " expects rank-1 buffers").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            indptr.numel() == vector.numel() + 1,
            (std::string(name) + " expects indptr length to be vector length plus one").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            task_capacity >= 0,
            (std::string(name) + " expects non-negative task_capacity").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            task_begin_scratch.numel() == static_cast<size_t>(task_capacity) &&
                task_end_scratch.numel() == static_cast<size_t>(task_capacity) &&
                task_begin_output.numel() == static_cast<size_t>(task_capacity) &&
                task_end_output.numel() == static_cast<size_t>(task_capacity),
            (std::string(name) + " expects task scratch lengths to match task_capacity").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            status_scratch.numel() == 2 && status_output.numel() == 2,
            (std::string(name) + " expects status scratch length 2").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            vector.numel() <= static_cast<size_t>(kInt32Max) &&
                output.numel() <= static_cast<size_t>(kInt32Max),
            (std::string(name) + " expects row/column counts within int32 range").c_str());
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        bool ok = true;
        std::string launch_error;
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
            ffi::Error error = LaunchBinaryCsrmvWatHybridHomo<WeightT, EventT, IndexT, OffsetT, Weighted>(
                weights.data_ptr<const WeightT>(),
                indices.data_ptr<const IndexT>(),
                indptr.data_ptr<const OffsetT>(),
                vector.data_ptr<const EventT>(),
                output.data_ptr<HybridOutT<WeightT>>(),
                task_begin_output.data_ptr<OffsetT>(),
                task_end_output.data_ptr<OffsetT>(),
                status_output.data_ptr<int32_t>(),
                static_cast<int>(indptr.numel()) - 1,
                static_cast<int>(output.numel()),
                static_cast<int64_t>(indices.numel()),
                task_capacity,
                cuda_stream);
            ok = error.success();
            if (!ok)
                launch_error = std::string(error.message());
        });
        BE_CSRMV_WAT_HYBRID_CHECK(ok, (std::string(name) + " launch failed: " + launch_error).c_str());
    }

} // namespace

#endif // DEVING_HYBRID_SPMV_IMPL_CUH
