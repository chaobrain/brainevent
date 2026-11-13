// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef DEVING_HYBRID_SDDMV_IMPL_CUH
#define DEVING_HYBRID_SDDMV_IMPL_CUH

// Shared binary/weighted hybrid SDDMV schedule (included by hybrid_sddmv_binary.cu
// and hybrid_sddmv_weighted.cu, which define the public launchers and FFI targets):
//   0. Clear status only; dweight arrives zeroed (aliased dweight_in).
//   1. Extract active rows. Light rows (nnz <= BE_HYBRID_TPR_THRESHOLD) are
//      gathered immediately by one thread per row.
//   2. Heavy active rows reserve contiguous block tasks once with atomicAdd.
//      Each task stores begin/end offsets for up to BE_HYBRID_TASK_NNZ slots;
//      weighted tasks also store the event value (the vw standard).
//   3. A fixed number of blocks read the active task count on device and gather
//      the tasks with a grid-stride loop (no D2H status read or host sync).
// Binary:   dweight[j] = ct[indices[j]]
// Weighted: dweight[j] = vector[row] * ct[indices[j]] (half types multiply in
//           float32 and round on storage)

#include "hybrid_sddmv.cuh"
#include "brainevent/cuda_traits.cuh"
#include "brainevent/hybrid_mv_schedule.cuh"

#include <limits>

namespace
{
    using namespace hybrid_mv;

    // dweight[j] = ct[indices[j]] (binary) or value * ct[indices[j]] (weighted)
    // for j = first, first + stride, ... < end. A light row is walked by one
    // thread (stride 1); a task by a block.
    template <bool Weighted, typename ValueT, typename IndexT, typename OffsetT>
    __device__ __forceinline__ void Gather(
        const ValueT *__restrict__ ct,
        const IndexT *__restrict__ indices,
        ValueT *__restrict__ dweight,
        ValueT value,
        OffsetT first,
        OffsetT end,
        OffsetT stride)
    {
        if constexpr (Weighted)
        {
            using Traits = cuda_traits::ValueTraits<ValueT>;
            const auto scale = Traits::read(value);
            for (OffsetT j = first; j < end; j += stride)
            {
                dweight[j] = Traits::write(scale * Traits::read(__ldg(&ct[__ldg(&indices[j])])));
            }
        }
        else
        {
            for (OffsetT j = first; j < end; j += stride)
            {
                dweight[j] = __ldg(&ct[__ldg(&indices[j])]);
            }
        }
    }

    template <bool Weighted, typename ValueT, typename EventT, typename IndexT, typename OffsetT>
    __global__ void HybridSddmvExtractKernel(
        const ValueT *__restrict__ ct,
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        const EventT *__restrict__ vector,
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
        const EventT event = __ldg(&vector[row]);
        if (!cuda_traits::EventTraits<EventT>::active(event))
        {
            return;
        }
        ValueT value{};
        if constexpr (Weighted)
        {
            value = event;
        }

        OffsetT row_begin = __ldg(&indptr[row]);
        OffsetT row_end = __ldg(&indptr[row + 1]);
        OffsetT nnz = row_end - row_begin;
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            Gather<Weighted>(ct, indices, dweight, value, row_begin, row_end, OffsetT(1));
            return;
        }

        int32_t n_chunks = TaskChunks(nnz, status);
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
            if constexpr (Weighted)
            {
                task_value[task] = value;
            }
        }
    }

    template <bool Weighted, typename ValueT, typename IndexT, typename OffsetT>
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
            ValueT value{};
            if constexpr (Weighted)
            {
                value = __ldg(&task_value[task]);
            }
            Gather<Weighted>(ct, indices, dweight, value,
                             __ldg(&task_begin[task]) + static_cast<OffsetT>(threadIdx.x),
                             __ldg(&task_end[task]), static_cast<OffsetT>(blockDim.x));
        }
    }

    // Shared launcher behind hybrid_sddmv::launch_binary/launch_weighted (see
    // hybrid_sddmv.cuh for the contract). task_value is used only when Weighted.
    template <bool Weighted, typename ValueT, typename EventT, typename IndexT, typename OffsetT>
    cudaError_t launch_sddmv(
        const ValueT *ct,
        const IndexT *indices,
        const OffsetT *indptr,
        const EventT *vector,
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
             (task_begin == nullptr || task_end == nullptr || (Weighted && task_value == nullptr))) ||
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
        HybridSddmvExtractKernel<Weighted, ValueT, EventT, IndexT, OffsetT>
            <<<extract_blocks, kBlockSize, 0, stream>>>(
                ct, indices, indptr, vector, dweight,
                task_begin, task_end, task_value, device_status, m, task_capacity);
        status_code = cudaGetLastError();
        if (status_code != cudaSuccess || task_capacity == 0)
        {
            return status_code;
        }

        int scatter_blocks = task_capacity < kFixedScatterBlocks ? task_capacity : kFixedScatterBlocks;
        HybridSddmvBlockKernel<Weighted, ValueT, IndexT, OffsetT>
            <<<scatter_blocks, kBlockSize, 0, stream>>>(
                ct, indices, task_begin, task_end, task_value, device_status,
                task_capacity, dweight);
        return cudaGetLastError();
    }

} // namespace

#endif // DEVING_HYBRID_SDDMV_IMPL_CUH
