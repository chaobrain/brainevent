// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef BRAINEVENT_HYBRID_DELAY_VALIDATION_CUH
#define BRAINEVENT_HYBRID_DELAY_VALIDATION_CUH

#include "../_common/cuda_traits.cuh"
#include <initializer_list>
#include <limits>

namespace region_delay::detail
{
    // Only the standalone C++ entry uses these checks. The FFI entry has
    // already passed the shared JAX shape, dtype and prepared-value checks.
    template <typename ValueT, typename RingT>
    cudaError_t ValidateLaunchArguments(
        int64_t rows, int64_t segments, int64_t cols, int64_t slots, int64_t capacity,
        const void *head, const void *output, const void *status,
        std::initializer_list<const void *> workspace,
        std::initializer_list<const void *> inputs)
    {
        constexpr int64_t max32 = std::numeric_limits<int32_t>::max();
        if (rows < 0 || rows > max32 || segments <= 0 || segments > max32 ||
            cols < 0 || cols > max32 || slots <= 0 || slots > max32 ||
            capacity < 0 || capacity > max32 || status == nullptr || head == nullptr ||
            (cols > 0 && output == nullptr))
            return cudaErrorInvalidValue;
        const int64_t ring_size = slots * cols;
        if (ring_size > std::numeric_limits<int64_t>::max() / sizeof(ValueT) ||
            ring_size > std::numeric_limits<RingT>::max())
            return cudaErrorInvalidValue;
        if (capacity > 0)
            for (const void *pointer : workspace)
                if (pointer == nullptr)
                    return cudaErrorInvalidValue;
        if (rows > 0)
            for (const void *pointer : inputs)
                if (pointer == nullptr)
                    return cudaErrorInvalidValue;
        return cudaSuccess;
    }

    // Reserve only within capacity, so summing many valid row counts cannot
    // wrap the int32 status counter before the capacity error is reported.
    __device__ inline void CountTasks(int32_t *status, int32_t count, int capacity)
    {
        int32_t old = atomicAdd(status, 0);
        for (;;)
        {
            if (count > capacity - old)
            {
                atomicExch(status + 1, 1);
                return;
            }
            const int32_t observed = atomicCAS(status, old, old + count);
            if (observed == old)
                return;
            old = observed;
        }
    }

    template <typename EventT, typename EdgeT, int Threshold, int TaskNnz>
    __global__ void ValidatePrepared(
        const EdgeT *indptr, const EventT *events, const EdgeT *counts,
        const int32_t *row_tasks, const int32_t *head, int32_t *status,
        int64_t rows, int64_t segments, int64_t slots, int capacity)
    {
        const int32_t current_head = __ldg(head);
        if (current_head < 0 || current_head >= slots)
        {
            atomicExch(status + 1, 2);
            return;
        }
        const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (row >= rows || !cuda_traits::EventTraits<EventT>::active(events[row]))
            return;
        const int64_t begin = __ldg(indptr + row);
        const int64_t end = __ldg(indptr + row + 1);
        if (begin == end)
            return;
        if (begin < 0 || end < begin)
        {
            atomicExch(status + 1, 3);
            return;
        }
        const int64_t length = end - begin;
        int64_t remaining = length;
        int64_t expected_tasks = 0;
        for (int64_t segment = 0; segment < segments; ++segment)
        {
            const int64_t count = __ldg(counts + row * segments + segment);
            if (count < 0 || count > remaining)
            {
                atomicExch(status + 1, 3);
                return;
            }
            remaining -= count;
            expected_tasks += count / TaskNnz + (count % TaskNnz != 0);
        }
        if (remaining != 0 ||
            (length > Threshold && expected_tasks != __ldg(row_tasks + row)))
        {
            atomicExch(status + 1, 3);
            return;
        }
        if (length > Threshold)
            CountTasks(status, static_cast<int32_t>(expected_tasks), capacity);
    }

    template <typename EventT, typename EdgeT, int Threshold, int TaskNnz, int BlockSize>
    cudaError_t PrepareNative(
        const EdgeT *indptr, const EventT *events, const EdgeT *counts,
        const int32_t *row_tasks, const int32_t *head, int32_t *status,
        int64_t rows, int64_t segments, int64_t slots, int capacity, cudaStream_t stream)
    {
        cudaError_t error = cudaMemsetAsync(status, 0, 2 * sizeof(int32_t), stream);
        if (error != cudaSuccess)
            return error;
        // Even an empty graph validates its head without dereferencing CSR.
        const int blocks = rows == 0 ? 1 : static_cast<int>((rows + BlockSize - 1) / BlockSize);
        ValidatePrepared<EventT, EdgeT, Threshold, TaskNnz><<<blocks, BlockSize, 0, stream>>>(
            indptr, events, counts, row_tasks, head, status, rows, segments, slots, capacity);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;
        // Extract allocates descriptors afresh. Preserve the validation error
        // so a failed native preflight cannot enter the delivery core.
        return cudaMemsetAsync(status, 0, sizeof(int32_t), stream);
    }
}

#endif // BRAINEVENT_HYBRID_DELAY_VALIDATION_CUH
