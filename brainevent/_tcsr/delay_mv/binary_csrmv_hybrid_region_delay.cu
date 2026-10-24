// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Fixed-delay CSR hybrid: one extract thread per row; whole-row classification;
// one atomic task reservation per active heavy row; tasks never cross segments.
// Callers supply delay-sorted CSR and matching task metadata. Existing ring
// contents are retained. Float events use != 0, including negative events.

#include "cuda_common.h"
#include "brainevent/common.h"
#include <cstdint>
#include <limits>

#ifndef BE_DELAY_TPR_THRESHOLD
#define BE_DELAY_TPR_THRESHOLD 128
#endif
#ifndef BE_DELAY_TASK_NNZ
#define BE_DELAY_TASK_NNZ 4096
#endif
#ifndef BE_DELAY_BLOCK_SIZE
#define BE_DELAY_BLOCK_SIZE 256
#endif
#ifndef BE_DELAY_SCATTER_BLOCKS
#define BE_DELAY_SCATTER_BLOCKS 2048
#endif

namespace region_delay
{

    constexpr int kThreshold = BE_DELAY_TPR_THRESHOLD;
    constexpr int kTaskNnz = BE_DELAY_TASK_NNZ;
    constexpr int kBlockSize = BE_DELAY_BLOCK_SIZE;
    constexpr int kScatterBlocks = BE_DELAY_SCATTER_BLOCKS;
    static_assert(kThreshold >= 0 && kTaskNnz > 0 && kScatterBlocks > 0);
    static_assert(kBlockSize >= 32 && kBlockSize <= 1024 && kBlockSize % 32 == 0);

    template <bool Homo, typename IndexT>
    __device__ __forceinline__ void Scatter(
        const float *weights, const IndexT *indices, float *output,
        int64_t begin, int64_t end, int64_t lane, int64_t stride)
    {
        float shared_weight = 0.0f;
        if constexpr (Homo)
            shared_weight = __ldg(weights);
        // Relative iteration and wide intermediates protect INT32_MAX tails.
        for (int64_t local = lane; local < end - begin; local += stride)
        {
            const int64_t edge = begin + local;
            const IndexT column = __ldg(indices + edge);
            const float value = Homo ? shared_weight : __ldg(weights + edge);
            atomicAdd(output + column, value);
        }
    }

    template <bool Homo, typename EventT, typename IndexT, typename EdgeT, typename RingT>
    __global__ void Extract(
        const float *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const EdgeT *__restrict__ indptr,
        const EventT *__restrict__ events,
        const EdgeT *__restrict__ delay_count,
        const RingT *__restrict__ delay_off,
        const int32_t *__restrict__ row_tasks,
        const int32_t *__restrict__ head,
        float *__restrict__ buffer,
        EdgeT *__restrict__ task_begin,
        EdgeT *__restrict__ task_end,
        RingT *__restrict__ task_out_off,
        int32_t *__restrict__ status,
        int64_t rows, int64_t segments, int64_t cols, int64_t slots, int capacity)
    {
        const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (row >= rows)
            return;
        const int32_t current_head = __ldg(head);
        if (current_head < 0 || current_head >= slots)
        {
            atomicExch(status + 1, 2);
            return;
        }
        if (__ldg(events + row) == static_cast<EventT>(0))
            return;
        int64_t edge = __ldg(indptr + row);
        const int64_t row_end = __ldg(indptr + row + 1);
        if (edge == row_end)
            return;
        const bool short_row = row_end - edge <= kThreshold;
        int64_t task = 0;
        int64_t task_limit = 0;
        if (!short_row)
        {
            const int32_t count = __ldg(row_tasks + row);
            if (count <= 0)
            {
                atomicExch(status + 1, 1);
                return;
            }
            const int32_t first = atomicAdd(status, count);
            if (first < 0 || first > capacity || count > capacity - first)
            {
                atomicExch(status + 1, 1);
                return;
            }
            task = first;
            task_limit = task + count;
        }
        float shared_weight = 0.0f;
        if constexpr (Homo)
        {
            if (short_row)
                shared_weight = __ldg(weights);
        }
        const int64_t ring_size = slots * cols;
        const int64_t head_base = static_cast<int64_t>(current_head) * cols;
        for (int64_t segment = 0; segment < segments; ++segment)
        {
            const int64_t count = __ldg(delay_count + row * segments + segment);
            if (count == 0)
                continue;
            if (count < 0 || count > row_end - edge)
            {
                atomicExch(status + 1, 3);
                return;
            }
            const int64_t segment_end = edge + count;
            int64_t out_off = head_base + static_cast<int64_t>(__ldg(delay_off + segment));
            if (out_off >= ring_size)
                out_off -= ring_size;
            if (short_row)
            {
                for (; edge < segment_end; ++edge)
                {
                    const IndexT column = __ldg(indices + edge);
                    const float value = Homo ? shared_weight : __ldg(weights + edge);
                    atomicAdd(buffer + out_off + column, value);
                }
            }
            else
            {
                while (edge < segment_end)
                {
                    if (task >= task_limit)
                    {
                        atomicExch(status + 1, 3);
                        return;
                    }
                    const int64_t remaining = segment_end - edge;
                    const int64_t end = edge + (remaining < kTaskNnz ? remaining : kTaskNnz);
                    task_begin[task] = static_cast<EdgeT>(edge);
                    task_end[task] = static_cast<EdgeT>(end);
                    task_out_off[task] = static_cast<RingT>(out_off);
                    edge = end;
                    ++task;
                }
            }
        }
        if (edge != row_end || (!short_row && task != task_limit))
            atomicExch(status + 1, 3);
    }

    template <bool Homo, typename IndexT, typename EdgeT, typename RingT>
    __global__ void BlockScatter(
        const float *__restrict__ weights, const IndexT *__restrict__ indices,
        const EdgeT *__restrict__ task_begin, const EdgeT *__restrict__ task_end,
        const RingT *__restrict__ task_out_off, const int32_t *__restrict__ status,
        int capacity, float *__restrict__ buffer)
    {
        // Extract has completed on this stream. Never read unwritten descriptors
        // after an overflow or inconsistent reservation.
        if (__ldg(status + 1) != 0)
            return;
        const int active = __ldg(status);
        if (active < 0 || active > capacity)
            return;
        for (int64_t task = blockIdx.x; task < active; task += gridDim.x)
        {
            const int64_t begin = __ldg(task_begin + task);
            const int64_t end = __ldg(task_end + task);
            const int64_t out_off = __ldg(task_out_off + task);
            Scatter<Homo>(weights, indices, buffer + out_off, begin, end, threadIdx.x, blockDim.x);
        }
    }

    __global__ void Consume(float *buffer, const int32_t *head, float *current,
                            int64_t cols, int64_t slots)
    {
        const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (column >= cols)
            return;
        const int32_t slot = __ldg(head);
        if (slot < 0 || slot >= slots)
        {
            current[column] = nanf("");
            return;
        }
        const int64_t index = static_cast<int64_t>(slot) * cols + column;
        current[column] = buffer[index];
        buffer[index] = 0.0f;
    }

    static bool IsIndex(BE::DType dtype)
    {
        return dtype == BE::DType::Int32 || dtype == BE::DType::Int64;
    }

    // Native entry point. All pointers are device pointers on the given stream.
    template <bool Homo, typename EventT, typename IndexT, typename EdgeT, typename RingT>
    void Launch(
        const float *weights, const IndexT *indices, const EdgeT *indptr,
        const EventT *events, const EdgeT *counts, const RingT *offsets,
        const int32_t *row_tasks, const int32_t *head,
        float *output, EdgeT *begin, EdgeT *end, RingT *out_off, int32_t *status,
        int64_t rows, int64_t segments, int64_t cols, int64_t slots,
        int capacity, cudaStream_t stream)
    {
        BE_CUDA_CHECK(cudaMemsetAsync(status, 0, 2 * sizeof(int32_t), stream));
        if (rows == 0)
            return;
        const int blocks = static_cast<int>((rows + kBlockSize - 1) / kBlockSize);
        Extract<Homo, EventT, IndexT, EdgeT, RingT><<<blocks, kBlockSize, 0, stream>>>(
            weights, indices, indptr, events, counts, offsets, row_tasks, head,
            output, begin, end, out_off, status, rows, segments, cols, slots, capacity);
        BE_CUDA_CHECK(cudaGetLastError());
        if (capacity > 0)
        {
            const int scatter_blocks = capacity < kScatterBlocks ? capacity : kScatterBlocks;
            BlockScatter<Homo, IndexT, EdgeT, RingT><<<scatter_blocks, kBlockSize, 0, stream>>>(
                weights, indices, begin, end, out_off, status, capacity, output);
            BE_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Keep tensor extraction at the FFI boundary, outside the native contract.
    template <bool Homo, typename EventT, typename IndexT, typename EdgeT, typename RingT>
    static void LaunchTensor(
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,
        const BE::Tensor events, const BE::Tensor counts, const BE::Tensor offsets,
        const BE::Tensor row_tasks, const BE::Tensor head,
        BE::Tensor output, BE::Tensor begin, BE::Tensor end, BE::Tensor out_off,
        BE::Tensor status, cudaStream_t stream)
    {
        Launch<Homo, EventT, IndexT, EdgeT, RingT>(
            weights.data_ptr<const float>(), indices.data_ptr<const IndexT>(),
            indptr.data_ptr<const EdgeT>(), events.data_ptr<const EventT>(),
            counts.data_ptr<const EdgeT>(), offsets.data_ptr<const RingT>(),
            row_tasks.data_ptr<const int32_t>(), head.data_ptr<const int32_t>(),
            output.data_ptr<float>(), begin.data_ptr<EdgeT>(), end.data_ptr<EdgeT>(),
            out_off.data_ptr<RingT>(), status.data_ptr<int32_t>(), events.numel(),
            offsets.numel(), output.shape(1), output.shape(0),
            static_cast<int>(begin.numel()), stream);
    }

} // namespace region_delay

// @BE region_delay_f32
void region_delay_f32(
    const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,
    const BE::Tensor events, const BE::Tensor counts, const BE::Tensor offsets,
    const BE::Tensor row_tasks, const BE::Tensor head, const BE::Tensor buffer,
    const BE::Tensor begin_scratch, const BE::Tensor end_scratch,
    const BE::Tensor off_scratch, const BE::Tensor status_scratch,
    BE::Tensor output, BE::Tensor begin, BE::Tensor end, BE::Tensor out_off,
    BE::Tensor status, int64_t stream)
{
    using namespace region_delay;
    BE_CHECK(weights.ndim() == 1 && weights.dtype() == BE::DType::Float32);
    BE_CHECK(indices.ndim() == 1 && IsIndex(indices.dtype()));
    BE_CHECK(indptr.ndim() == 1 && IsIndex(indptr.dtype()));
    BE_CHECK(events.ndim() == 1 && (events.dtype() == BE::DType::Bool || events.dtype() == BE::DType::Float32));
    const int64_t rows = events.numel();
    BE_CHECK(rows <= std::numeric_limits<int32_t>::max() && indptr.numel() == rows + 1);
    BE_CHECK(weights.numel() == 1 || weights.numel() == indices.numel());
    BE_CHECK(counts.ndim() == 2 && counts.shape(0) == rows && counts.dtype() == indptr.dtype());
    BE_CHECK(offsets.ndim() == 1 && offsets.numel() > 0 && IsIndex(offsets.dtype()));
    BE_CHECK(counts.shape(1) == offsets.numel());
    BE_CHECK(row_tasks.ndim() == 1 && row_tasks.numel() == rows && row_tasks.dtype() == BE::DType::Int32);
    BE_CHECK(head.ndim() == 1 && head.numel() == 1 && head.dtype() == BE::DType::Int32);
    BE_CHECK(buffer.ndim() == 2 && buffer.dtype() == BE::DType::Float32);
    BE_CHECK(output.ndim() == 2 && output.dtype() == buffer.dtype());
    BE_CHECK(output.shape(0) == buffer.shape(0) && output.shape(1) == buffer.shape(1));
    BE_CHECK(output.shape(0) > 0 && output.shape(0) <= std::numeric_limits<int32_t>::max());
    BE_CHECK(output.shape(1) <= std::numeric_limits<int32_t>::max());
    BE_CHECK(output.numel() <= std::numeric_limits<int64_t>::max() / sizeof(float));
    BE_CHECK(offsets.dtype() == BE::DType::Int64 || output.numel() <= std::numeric_limits<int32_t>::max());
    BE_CHECK(buffer.data_ptr() == output.data_ptr()) << "ring requires input/output aliasing";
    const int64_t capacity = begin.numel();
    BE_CHECK(capacity <= std::numeric_limits<int32_t>::max());
    const BE::Tensor scratch[4] = {begin_scratch, end_scratch, off_scratch, status_scratch};
    const BE::Tensor result[4] = {begin, end, out_off, status};
    for (int i = 0; i < 4; ++i)
    {
        const int64_t size = i == 3 ? 2 : capacity;
        const BE::DType dtype = i < 2 ? indptr.dtype() : i == 2 ? offsets.dtype()
                                                                : BE::DType::Int32;
        BE_CHECK(scratch[i].ndim() == 1 && result[i].ndim() == 1);
        BE_CHECK(scratch[i].numel() == size && result[i].numel() == size);
        BE_CHECK(scratch[i].dtype() == dtype && result[i].dtype() == dtype);
        BE_CHECK(scratch[i].data_ptr() == result[i].data_ptr()) << "workspace requires aliasing";
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, EdgeT, {
        BE_DISPATCH_SIGNED_INDEX(offsets.dtype(), RingT, {
            if (events.dtype() == BE::DType::Bool)
            {
                if (weights.numel() == 1)
                    LaunchTensor<true, int8_t, IndexT, EdgeT, RingT>(weights, indices, indptr, events, counts, offsets, row_tasks, head, output, begin, end, out_off, status, cuda_stream);
                else
                    LaunchTensor<false, int8_t, IndexT, EdgeT, RingT>(weights, indices, indptr, events, counts, offsets, row_tasks, head, output, begin, end, out_off, status, cuda_stream);
            }
            else
            {
                if (weights.numel() == 1)
                    LaunchTensor<true, float, IndexT, EdgeT, RingT>(weights, indices, indptr, events, counts, offsets, row_tasks, head, output, begin, end, out_off, status, cuda_stream);
                else
                    LaunchTensor<false, float, IndexT, EdgeT, RingT>(weights, indices, indptr, events, counts, offsets, row_tasks, head, output, begin, end, out_off, status, cuda_stream);
            }
        });
    });
}

// @BE consume_region_delay_f32
void consume_region_delay_f32(
    const BE::Tensor buffer, const BE::Tensor head,
    BE::Tensor current, BE::Tensor output, int64_t stream)
{
    BE_CHECK(buffer.ndim() == 2 && buffer.dtype() == BE::DType::Float32);
    BE_CHECK(output.ndim() == 2 && output.dtype() == BE::DType::Float32);
    BE_CHECK(output.shape(0) == buffer.shape(0) && output.shape(1) == buffer.shape(1));
    BE_CHECK(output.data_ptr() == buffer.data_ptr()) << "ring requires input/output aliasing";
    BE_CHECK(head.ndim() == 1 && head.numel() == 1 && head.dtype() == BE::DType::Int32);
    BE_CHECK(current.ndim() == 1 && current.numel() == buffer.shape(1) && current.dtype() == BE::DType::Float32);
    BE_CHECK(buffer.shape(0) > 0 && buffer.shape(0) <= std::numeric_limits<int32_t>::max());
    BE_CHECK(buffer.shape(1) <= std::numeric_limits<int32_t>::max());
    if (buffer.shape(1) == 0)
        return;
    const int blocks = static_cast<int>((buffer.shape(1) + 255) / 256);
    region_delay::Consume<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        output.data_ptr<float>(), head.data_ptr<const int32_t>(), current.data_ptr<float>(),
        buffer.shape(1), buffer.shape(0));
    BE_CUDA_CHECK(cudaGetLastError());
}
