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
 *   status[0] on device.
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

#ifndef BE_HYBRID_BLOCK_SIZE
#define BE_HYBRID_BLOCK_SIZE 512
#endif

#ifndef BE_HYBRID_FIXED_SCATTER_BLOCKS
#define BE_HYBRID_FIXED_SCATTER_BLOCKS 1024
#endif

#ifndef BE_HYBRID_TPR_THRESHOLD
#define BE_HYBRID_TPR_THRESHOLD 128
#endif

#ifndef BE_HYBRID_TASK_NNZ
#define BE_HYBRID_TASK_NNZ 512
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
    // weighted (``_wv``) kernels, which read the event value as a scale.
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

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRange(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        WeightT *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        typename HybridWeightTraits<WeightT>::AccT scale)
    {
        using WeightTraits = HybridWeightTraits<WeightT>;
        using AccT = typename WeightTraits::AccT;
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
            WeightTraits::atomic_add(&output[idx], w);
        }
    }

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeThread(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
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
            AccT w = WeightTraits::read(__ldg(&weights[j]));
            if constexpr (Weighted)
            {
                w *= scale;
            }
            WeightTraits::atomic_add(&output[idx], w);
        }
    }

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeHomo(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        WeightT *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        typename HybridWeightTraits<WeightT>::AccT scale)
    {
        using WeightTraits = HybridWeightTraits<WeightT>;
        using AccT = typename WeightTraits::AccT;
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
            WeightTraits::atomic_add(&output[idx], w);
        }
    }

    template <bool Weighted, typename WeightT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void ScatterRangeThreadHomo(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        WeightT *__restrict__ output,
        OffsetT begin,
        OffsetT end,
        typename HybridWeightTraits<WeightT>::AccT scale)
    {
        using WeightTraits = HybridWeightTraits<WeightT>;
        using AccT = typename WeightTraits::AccT;
        AccT w = WeightTraits::read(__ldg(&weights[0]));
        if constexpr (Weighted)
        {
            w *= scale;
        }
        for (OffsetT j = begin; j < end; ++j)
        {
            IndexT idx = __ldg(&indices[j]);
            WeightTraits::atomic_add(&output[idx], w);
        }
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    __global__ void BinaryCsrmvWatHybridExtractKernel(
        const WeightT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        const EventT *__restrict__ vector,
        WeightT *__restrict__ output,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        int32_t *__restrict__ status,
        int m,
        int task_capacity)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        typename HybridWeightTraits<WeightT>::AccT scale = 1;
        if constexpr (Weighted)
        {
            // Weighted events (buffer in the weight dtype): x != 0 is active
            // and scales every weight of the row.
            if (row >= m)
            {
                return;
            }
            scale = HybridWeightTraits<WeightT>::read(__ldg(&vector[row]));
            if (scale == 0)
            {
                return;
            }
        }
        else
        {
            if (row >= m || !HybridEventTraits<EventT>::active(__ldg(&vector[row])))
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
                // task_end holds the row of a weighted task (see the extract kernel).
                int row = static_cast<int>(__ldg(&task_end[task]));
                OffsetT row_end = __ldg(&indptr[row + 1]);
                end = begin + static_cast<OffsetT>(kTaskNnz);
                if (end > row_end)
                {
                    end = row_end;
                }
                scale = HybridWeightTraits<WeightT>::read(__ldg(&vector[row]));
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
        WeightT *__restrict__ output,
        OffsetT *__restrict__ task_begin,
        OffsetT *__restrict__ task_end,
        int32_t *__restrict__ status,
        int m,
        int task_capacity)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        typename HybridWeightTraits<WeightT>::AccT scale = 1;
        if constexpr (Weighted)
        {
            // Weighted events (buffer in the weight dtype): x != 0 is active
            // and scales every weight of the row.
            if (row >= m)
            {
                return;
            }
            scale = HybridWeightTraits<WeightT>::read(__ldg(&vector[row]));
            if (scale == 0)
            {
                return;
            }
        }
        else
        {
            if (row >= m || !HybridEventTraits<EventT>::active(__ldg(&vector[row])))
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
                // task_end holds the row of a weighted task (see the extract kernel).
                int row = static_cast<int>(__ldg(&task_end[task]));
                OffsetT row_end = __ldg(&indptr[row + 1]);
                end = begin + static_cast<OffsetT>(kTaskNnz);
                if (end > row_end)
                {
                    end = row_end;
                }
                scale = HybridWeightTraits<WeightT>::read(__ldg(&vector[row]));
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

    static bool IsSignedIndexDType(BE::DType dtype)
    {
        return dtype == BE::DType::Int32 || dtype == BE::DType::Int64;
    }

    template <typename WeightT, typename EventT, typename IndexT, typename OffsetT, bool Weighted>
    static ffi::Error LaunchBinaryCsrmvWatHybrid(
        const WeightT *weights,
        const IndexT *indices,
        const OffsetT *indptr,
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
            weights.dtype() == HybridWeightTraits<WeightT>::dtype,
            (std::string(name) + " expects matching weights dtype").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            IsSignedIndexDType(indices.dtype()),
            (std::string(name) + " expects indices=s32/s64").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            IsSignedIndexDType(indptr.dtype()),
            (std::string(name) + " expects indptr=s32/s64").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            (vector.dtype() == HybridEventDType<WeightT, EventT, Weighted>::value),
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
            output.dtype() == HybridWeightTraits<WeightT>::dtype,
            (std::string(name) + " expects matching output dtype").c_str());
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
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        bool ok = true;
        std::string launch_error;
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
            ffi::Error error = LaunchBinaryCsrmvWatHybrid<WeightT, EventT, IndexT, OffsetT, Weighted>(
                weights.data_ptr<const WeightT>(),
                indices.data_ptr<const IndexT>(),
                indptr.data_ptr<const OffsetT>(),
                vector.data_ptr<const EventT>(),
                output.data_ptr<WeightT>(),
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
            weights.dtype() == HybridWeightTraits<WeightT>::dtype && weights.numel() == 1,
            (std::string(name) + " expects matching weights dtype and length 1").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            IsSignedIndexDType(indices.dtype()),
            (std::string(name) + " expects indices=s32/s64").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            IsSignedIndexDType(indptr.dtype()),
            (std::string(name) + " expects indptr=s32/s64").c_str());
        BE_CSRMV_WAT_HYBRID_CHECK(
            (vector.dtype() == HybridEventDType<WeightT, EventT, Weighted>::value),
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
            output.dtype() == HybridWeightTraits<WeightT>::dtype,
            (std::string(name) + " expects matching output dtype").c_str());
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
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        bool ok = true;
        std::string launch_error;
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
            ffi::Error error = LaunchBinaryCsrmvWatHybridHomo<WeightT, EventT, IndexT, OffsetT, Weighted>(
                weights.data_ptr<const WeightT>(),
                indices.data_ptr<const IndexT>(),
                indptr.data_ptr<const OffsetT>(),
                vector.data_ptr<const EventT>(),
                output.data_ptr<WeightT>(),
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
