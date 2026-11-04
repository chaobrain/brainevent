// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Weighted region-delay hybrid: each heavy task carries the row event in
// task_value; segment ends and ring offsets remain explicit.
#include "hybrid_delay_spmv.cuh"
#include "../_common/cuda_traits.cuh"
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

namespace
{
    constexpr int kThreshold = BE_DELAY_TPR_THRESHOLD;
    constexpr int kTaskNnz = BE_DELAY_TASK_NNZ;
    constexpr int kBlockSize = BE_DELAY_BLOCK_SIZE;
    constexpr int kScatterBlocks = BE_DELAY_SCATTER_BLOCKS;
    static_assert(kThreshold >= 0 && kTaskNnz > 0 && kScatterBlocks > 0);
    static_assert(kBlockSize >= 32 && kBlockSize <= 1024 && kBlockSize % 32 == 0);

    using cuda_traits::EventTraits;
    using cuda_traits::ValueTraits;

    template <bool Homo, typename ValueT, typename IndexT>
    __device__ __forceinline__ void Scatter(
        const ValueT *weights, const IndexT *indices, ValueT *output,
        int64_t begin, int64_t end, int64_t lane, int64_t stride, ValueT event_value)
    {
        using Traits = ValueTraits<ValueT>;
        const auto scale = Traits::read(event_value);
        typename Traits::AccT shared_weight = 0;
        if constexpr (Homo)
            shared_weight = Traits::read(weights[0]);
        // Relative iteration and wide intermediates protect INT32_MAX tails.
        for (int64_t local = lane; local < end - begin; local += stride)
        {
            const int64_t edge = begin + local;
            const IndexT column = __ldg(indices + edge);
            const typename Traits::AccT value = scale * (Homo ? shared_weight : Traits::read(weights[edge]));
            Traits::atomic_add(output + column, value);
        }
    }

    template <bool Homo, typename ValueT, typename IndexT, typename EdgeT, typename RingT>
    __global__ void Extract(
        const ValueT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const EdgeT *__restrict__ indptr,
        const ValueT *__restrict__ events,
        const EdgeT *__restrict__ delay_count,
        const RingT *__restrict__ delay_off,
        const int32_t *__restrict__ row_tasks,
        const int32_t *__restrict__ head,
        ValueT *__restrict__ buffer,
        EdgeT *__restrict__ task_begin,
        EdgeT *__restrict__ task_end,
        RingT *__restrict__ task_out_off,
        ValueT *__restrict__ task_value,
        int32_t *__restrict__ status,
        int64_t rows, int64_t segments, int64_t cols, int64_t slots, int capacity)
    {
        const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const int32_t current_head = __ldg(head);
        if (current_head < 0 || current_head >= slots)
        {
            atomicExch(status + 1, 2);
            return;
        }
        if (row >= rows)
            return;
        const ValueT event_value = __ldg(events + row);
        if (!EventTraits<ValueT>::active(event_value))
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
        using Traits = ValueTraits<ValueT>;
        const auto scale = Traits::read(event_value);
        typename Traits::AccT shared_weight = 0;
        if constexpr (Homo)
        {
            if (short_row)
                shared_weight = Traits::read(weights[0]);
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
                    const typename Traits::AccT value = scale * (Homo ? shared_weight : Traits::read(weights[edge]));
                    Traits::atomic_add(buffer + out_off + column, value);
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
                    task_value[task] = event_value;
                    edge = end;
                    ++task;
                }
            }
        }
        if (edge != row_end || (!short_row && task != task_limit))
            atomicExch(status + 1, 3);
    }

    template <bool Homo, typename ValueT, typename IndexT, typename EdgeT, typename RingT>
    __global__ void BlockScatter(
        const ValueT *__restrict__ weights, const IndexT *__restrict__ indices,
        const EdgeT *__restrict__ task_begin, const EdgeT *__restrict__ task_end,
        const RingT *__restrict__ task_out_off, const ValueT *__restrict__ task_value,
        const int32_t *__restrict__ status,
        int capacity, ValueT *__restrict__ buffer)
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
            Scatter<Homo>(weights, indices, buffer + out_off, begin, end, threadIdx.x, blockDim.x, __ldg(task_value + task));
        }
    }

} // namespace

namespace region_delay
{
    template <bool Homo, typename ValueT, typename IndexT, typename EdgeT, typename RingT>
    cudaError_t launch_weighted(
        const ValueT *weights, const IndexT *indices, const EdgeT *indptr,
        const ValueT *events, const EdgeT *counts, const RingT *offsets,
        const int32_t *row_tasks, const int32_t *head,
        ValueT *output, EdgeT *begin, EdgeT *end, RingT *out_off, ValueT *task_value, int32_t *status,
        int64_t rows, int64_t segments, int64_t cols, int64_t slots,
        int64_t task_capacity, cudaStream_t stream)
    {
        constexpr int64_t max32 = std::numeric_limits<int32_t>::max();
        if (rows < 0 || rows > max32 || segments <= 0 || segments > max32 ||
            cols < 0 || cols > max32 || slots <= 0 || slots > max32 ||
            task_capacity < 0 || task_capacity > max32 || status == nullptr || head == nullptr ||
            (cols > 0 && output == nullptr))
            return cudaErrorInvalidValue;
        const int64_t ring_size = slots * cols;
        if (ring_size > std::numeric_limits<int64_t>::max() / sizeof(ValueT) ||
            ring_size > std::numeric_limits<RingT>::max() ||
            (task_capacity > 0 && (begin == nullptr || end == nullptr || out_off == nullptr || task_value == nullptr)) ||
            (rows > 0 && (indptr == nullptr || events == nullptr || counts == nullptr ||
                          offsets == nullptr || row_tasks == nullptr)))
            return cudaErrorInvalidValue;
        const int capacity = static_cast<int>(task_capacity);
        cudaError_t error = cudaMemsetAsync(status, 0, 2 * sizeof(int32_t), stream);
        if (error != cudaSuccess)
            return error;
        // An empty graph still checks the device head; no ring slot is cleared.
        const int blocks = rows == 0 ? 1 : static_cast<int>((rows + kBlockSize - 1) / kBlockSize);
        Extract<Homo, ValueT, IndexT, EdgeT, RingT><<<blocks, kBlockSize, 0, stream>>>(
            weights, indices, indptr, events, counts, offsets, row_tasks, head,
            output, begin, end, out_off, task_value, status, rows, segments, cols, slots, capacity);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;
        if (capacity > 0)
        {
            const int scatter_blocks = capacity < kScatterBlocks ? capacity : kScatterBlocks;
            BlockScatter<Homo, ValueT, IndexT, EdgeT, RingT><<<scatter_blocks, kBlockSize, 0, stream>>>(
                weights, indices, begin, end, out_off, task_value, status, capacity, output);
            error = cudaGetLastError();
        }
        return error;
    }

#define INSTANTIATE_ONE(H, V, I, E, R)                                        \
    template cudaError_t launch_weighted<H, V, I, E, R>(                      \
        const V *, const I *, const E *, const V *, const E *, const R *,     \
        const int32_t *, const int32_t *, V *, E *, E *, R *, V *, int32_t *, \
        int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t);
#define INSTANTIATE_EVENTS(H, V, I, E, R) INSTANTIATE_ONE(H, V, I, E, R)
#define INSTANTIATE_RINGS(H, V, I, E)       \
    INSTANTIATE_EVENTS(H, V, I, E, int32_t) \
    INSTANTIATE_EVENTS(H, V, I, E, int64_t)
#define INSTANTIATE_INDICES(H, V)             \
    INSTANTIATE_RINGS(H, V, int32_t, int32_t) \
    INSTANTIATE_RINGS(H, V, int32_t, int64_t) \
    INSTANTIATE_RINGS(H, V, int64_t, int32_t) \
    INSTANTIATE_RINGS(H, V, int64_t, int64_t)
#define INSTANTIATE_VALUES(H)             \
    INSTANTIATE_INDICES(H, __half)        \
    INSTANTIATE_INDICES(H, __nv_bfloat16) \
    INSTANTIATE_INDICES(H, float)         \
    INSTANTIATE_INDICES(H, double)
    INSTANTIATE_VALUES(false)
    INSTANTIATE_VALUES(true)
#undef INSTANTIATE_VALUES
#undef INSTANTIATE_INDICES
#undef INSTANTIATE_RINGS
#undef INSTANTIATE_EVENTS
#undef INSTANTIATE_ONE
} // namespace region_delay

#ifndef BE_REGION_DELAY_STANDALONE
#include "cuda_common.h"
#include "brainevent/common.h"
#include "_common/op_traits.cuh"

namespace region_delay
{
    using op_traits::EventTraits;
    using op_traits::ValueTraits;

    // Keep tensor extraction at the FFI boundary, outside the native contract.
    template <bool Homo, typename ValueT, typename IndexT, typename EdgeT, typename RingT>
    static void LaunchTensor(
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,
        const BE::Tensor events, const BE::Tensor counts, const BE::Tensor offsets,
        const BE::Tensor row_tasks, const BE::Tensor head,
        BE::Tensor output, BE::Tensor begin, BE::Tensor end, BE::Tensor out_off,
        BE::Tensor task_value, BE::Tensor status, cudaStream_t stream)
    {
        BE_CUDA_CHECK((launch_weighted<Homo, ValueT, IndexT, EdgeT, RingT>(
            weights.data_ptr<const ValueT>(), indices.data_ptr<const IndexT>(),
            indptr.data_ptr<const EdgeT>(), events.data_ptr<const ValueT>(),
            counts.data_ptr<const EdgeT>(), offsets.data_ptr<const RingT>(),
            row_tasks.data_ptr<const int32_t>(), head.data_ptr<const int32_t>(),
            output.data_ptr<ValueT>(), begin.data_ptr<EdgeT>(), end.data_ptr<EdgeT>(),
            out_off.data_ptr<RingT>(), task_value.data_ptr<ValueT>(), status.data_ptr<int32_t>(), events.numel(),
            offsets.numel(), output.shape(1), output.shape(0),
            begin.numel(), stream)));
    }

    template <typename ValueT>
    void DeliverImpl(
        const char *name,
        const BE::Tensor &weights, const BE::Tensor &indices, const BE::Tensor &indptr,
        const BE::Tensor &events, const BE::Tensor &counts, const BE::Tensor &offsets,
        const BE::Tensor &row_tasks, const BE::Tensor &head, const BE::Tensor &buffer,
        const BE::Tensor &begin_scratch, const BE::Tensor &end_scratch,
        const BE::Tensor &off_scratch, const BE::Tensor &task_value_scratch, const BE::Tensor &status_scratch,
        BE::Tensor &output, BE::Tensor &begin, BE::Tensor &end, BE::Tensor &out_off,
        BE::Tensor &task_value, BE::Tensor &status, int64_t stream)
    {
        constexpr BE::DType value_dtype = ValueTraits<ValueT>::dtype;
        BE_CHECK(weights.ndim() == 1 && weights.dtype() == value_dtype) << name << ": weights dtype";
        BE_CHECK(indices.ndim() == 1 && op_traits::is_signed_index(indices.dtype())) << name << ": indices";
        BE_CHECK(indptr.ndim() == 1 && op_traits::is_signed_index(indptr.dtype())) << name << ": indptr";
        BE_CHECK(events.ndim() == 1 && events.dtype() == value_dtype) << name << ": events dtype";
        const int64_t rows = events.numel();
        BE_CHECK(rows <= std::numeric_limits<int32_t>::max() && indptr.numel() == rows + 1);
        BE_CHECK(weights.numel() == 1 || weights.numel() == indices.numel());
        BE_CHECK(counts.ndim() == 2 && counts.shape(0) == rows && counts.dtype() == indptr.dtype());
        BE_CHECK(offsets.ndim() == 1 && offsets.numel() > 0 && op_traits::is_signed_index(offsets.dtype()));
        BE_CHECK(counts.shape(1) == offsets.numel());
        BE_CHECK(row_tasks.ndim() == 1 && row_tasks.numel() == rows && row_tasks.dtype() == BE::DType::Int32);
        BE_CHECK(head.ndim() == 1 && head.numel() == 1 && head.dtype() == BE::DType::Int32);
        BE_CHECK(buffer.ndim() == 2 && buffer.dtype() == value_dtype) << name << ": ring dtype";
        BE_CHECK(output.ndim() == 2 && output.dtype() == buffer.dtype());
        BE_CHECK(output.shape(0) == buffer.shape(0) && output.shape(1) == buffer.shape(1));
        BE_CHECK(output.shape(0) > 0 && output.shape(0) <= std::numeric_limits<int32_t>::max());
        BE_CHECK(output.shape(1) <= std::numeric_limits<int32_t>::max());
        BE_CHECK(output.numel() <= std::numeric_limits<int64_t>::max() / sizeof(ValueT));
        BE_CHECK(offsets.dtype() == BE::DType::Int64 || output.numel() <= std::numeric_limits<int32_t>::max());
        BE_CHECK(buffer.data_ptr() == output.data_ptr()) << "ring requires input/output aliasing";
        const int64_t capacity = begin.numel();
        BE_CHECK(capacity <= std::numeric_limits<int32_t>::max());
        const BE::Tensor scratch[5] = {begin_scratch, end_scratch, off_scratch, task_value_scratch, status_scratch};
        const BE::Tensor result[5] = {begin, end, out_off, task_value, status};
        for (int i = 0; i < 5; ++i)
        {
            const int64_t size = i == 4 ? 2 : capacity;
            const BE::DType dtype = i < 2 ? indptr.dtype() : i == 2 ? offsets.dtype()
                                                         : i == 3   ? value_dtype
                                                                    : BE::DType::Int32;
            BE_CHECK(scratch[i].ndim() == 1 && result[i].ndim() == 1);
            BE_CHECK(scratch[i].numel() == size && result[i].numel() == size);
            BE_CHECK(scratch[i].dtype() == dtype && result[i].dtype() == dtype);
            BE_CHECK(scratch[i].data_ptr() == result[i].data_ptr()) << "workspace requires aliasing";
        }
        const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, EdgeT, {
            BE_DISPATCH_SIGNED_INDEX(offsets.dtype(), RingT, {
                if (weights.numel() == 1)
                    LaunchTensor<true, ValueT, IndexT, EdgeT, RingT>(weights, indices, indptr, events, counts, offsets, row_tasks, head, output, begin, end, out_off, task_value, status, cuda_stream);
                else
                    LaunchTensor<false, ValueT, IndexT, EdgeT, RingT>(weights, indices, indptr, events, counts, offsets, row_tasks, head, output, begin, end, out_off, task_value, status, cuda_stream);
            });
        });
    }

} // namespace region_delay

// Weighted events, weights, ring and task_value share the suffix dtype.
#define DEFINE_REGION_DELAY_WEIGHTED(SUFFIX, VALUE_T)                                                       \
    void region_delay##SUFFIX(                                                                              \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,                        \
        const BE::Tensor events, const BE::Tensor counts, const BE::Tensor offsets,                         \
        const BE::Tensor row_tasks, const BE::Tensor head, const BE::Tensor buffer,                         \
        const BE::Tensor begin_scratch, const BE::Tensor end_scratch,                                       \
        const BE::Tensor off_scratch, const BE::Tensor task_value_scratch, const BE::Tensor status_scratch, \
        BE::Tensor output, BE::Tensor begin, BE::Tensor end, BE::Tensor out_off,                            \
        BE::Tensor task_value, BE::Tensor status, int64_t stream)                                           \
    {                                                                                                       \
        region_delay::DeliverImpl<VALUE_T>(                                                                 \
            "region_delay" #SUFFIX, weights, indices, indptr, events, counts, offsets, row_tasks,           \
            head, buffer, begin_scratch, end_scratch, off_scratch, task_value_scratch, status_scratch,      \
            output, begin, end, out_off, task_value, status, stream);                                       \
    }

// @BE region_delay_f16_wv
DEFINE_REGION_DELAY_WEIGHTED(_f16_wv, __half)
// @BE region_delay_bf16_wv
DEFINE_REGION_DELAY_WEIGHTED(_bf16_wv, __nv_bfloat16)
// @BE region_delay_f32_wv
DEFINE_REGION_DELAY_WEIGHTED(_f32_wv, float)
// @BE region_delay_f64_wv
DEFINE_REGION_DELAY_WEIGHTED(_f64_wv, double)
#undef DEFINE_REGION_DELAY_WEIGHTED

#endif // BE_REGION_DELAY_STANDALONE
