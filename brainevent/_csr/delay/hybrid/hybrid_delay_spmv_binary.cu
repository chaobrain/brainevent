// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Binary region-delay hybrid: nonzero events gate delayed weight delivery.
#include "hybrid_delay_spmv.cuh"
#include "hybrid_delay_validation.cuh"
#include "../_common/cuda_traits.cuh"

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
        int64_t begin, int64_t end, int64_t lane, int64_t stride)
    {
        using Traits = ValueTraits<ValueT>;
        typename Traits::AccT shared_weight = 0;
        if constexpr (Homo)
            shared_weight = Traits::read(weights[0]);
        // Relative iteration and wide intermediates protect INT32_MAX tails.
        for (int64_t local = lane; local < end - begin; local += stride)
        {
            const int64_t edge = begin + local;
            const IndexT column = __ldg(indices + edge);
            const typename Traits::AccT value = Homo ? shared_weight : Traits::read(weights[edge]);
            Traits::atomic_add(output + column, value);
        }
    }

    template <bool Homo, typename ValueT, typename EventT, typename IndexT, typename EdgeT, typename RingT>
    __global__ void Extract(
        const ValueT *__restrict__ weights,
        const IndexT *__restrict__ indices,
        const EdgeT *__restrict__ indptr,
        const EventT *__restrict__ events,
        const EdgeT *__restrict__ delay_count,
        const RingT *__restrict__ delay_off,
        const int32_t *__restrict__ row_tasks,
        const int32_t *__restrict__ head,
        ValueT *__restrict__ buffer,
        EdgeT *__restrict__ task_begin,
        EdgeT *__restrict__ task_end,
        RingT *__restrict__ task_out_off,
        int32_t *__restrict__ status,
        int64_t rows, int64_t segments, int64_t cols, int64_t slots)
    {
        // The native entry may reject its preflight. FFI reaches this core
        // only with validated inputs; this gate consumes status, not inputs.
        if (__ldg(status + 1) != 0)
            return;
        const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (row >= rows)
            return;
        if (!EventTraits<EventT>::active(events[row]))
            return;
        const int32_t current_head = __ldg(head);
        int64_t edge = __ldg(indptr + row);
        const int64_t row_end = __ldg(indptr + row + 1);
        if (edge == row_end)
            return;
        const bool short_row = row_end - edge <= kThreshold;
        int64_t task = 0;
        if (!short_row)
            task = atomicAdd(status, __ldg(row_tasks + row));
        using Traits = ValueTraits<ValueT>;
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
            const int64_t segment_end = edge + count;
            int64_t out_off = head_base + static_cast<int64_t>(__ldg(delay_off + segment));
            if (out_off >= ring_size)
                out_off -= ring_size;
            if (short_row)
            {
                for (; edge < segment_end; ++edge)
                {
                    const IndexT column = __ldg(indices + edge);
                    const typename Traits::AccT value = Homo ? shared_weight : Traits::read(weights[edge]);
                    Traits::atomic_add(buffer + out_off + column, value);
                }
            }
            else
            {
                while (edge < segment_end)
                {
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
    }

    template <bool Homo, typename ValueT, typename IndexT, typename EdgeT, typename RingT>
    __global__ void BlockScatter(
        const ValueT *__restrict__ weights, const IndexT *__restrict__ indices,
        const EdgeT *__restrict__ task_begin, const EdgeT *__restrict__ task_end,
        const RingT *__restrict__ task_out_off, const int32_t *__restrict__ status,
        ValueT *__restrict__ buffer)
    {
        // Extract completed on this stream. A rejected preflight leaves the
        // count at zero; otherwise every counted descriptor was written.
        const int active = __ldg(status);
        for (int64_t task = blockIdx.x; task < active; task += gridDim.x)
        {
            const int64_t begin = __ldg(task_begin + task);
            const int64_t end = __ldg(task_end + task);
            const int64_t out_off = __ldg(task_out_off + task);
            Scatter<Homo>(weights, indices, buffer + out_off, begin, end, threadIdx.x, blockDim.x);
        }
    }

} // namespace

namespace region_delay
{
    template <bool Homo, typename ValueT, typename EventT, typename IndexT, typename EdgeT, typename RingT>
    static cudaError_t LaunchPrepared(
        const ValueT *weights, const IndexT *indices, const EdgeT *indptr,
        const EventT *events, const EdgeT *counts, const RingT *offsets,
        const int32_t *row_tasks, const int32_t *head,
        ValueT *output, EdgeT *begin, EdgeT *end, RingT *out_off, int32_t *status,
        int64_t rows, int64_t segments, int64_t cols, int64_t slots,
        int64_t task_capacity, cudaStream_t stream)
    {
        const int capacity = static_cast<int>(task_capacity);
        const int blocks = rows == 0 ? 1 : static_cast<int>((rows + kBlockSize - 1) / kBlockSize);
        Extract<Homo, ValueT, EventT, IndexT, EdgeT, RingT><<<blocks, kBlockSize, 0, stream>>>(
            weights, indices, indptr, events, counts, offsets, row_tasks, head,
            output, begin, end, out_off, status, rows, segments, cols, slots);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;
        if (capacity > 0)
        {
            const int scatter_blocks = capacity < kScatterBlocks ? capacity : kScatterBlocks;
            BlockScatter<Homo, ValueT, IndexT, EdgeT, RingT><<<scatter_blocks, kBlockSize, 0, stream>>>(
                weights, indices, begin, end, out_off, status, output);
            error = cudaGetLastError();
        }
        return error;
    }

    template <bool Homo, typename ValueT, typename EventT, typename IndexT, typename EdgeT, typename RingT>
    cudaError_t launch_binary(
        const ValueT *weights, const IndexT *indices, const EdgeT *indptr,
        const EventT *events, const EdgeT *counts, const RingT *offsets,
        const int32_t *row_tasks, const int32_t *head,
        ValueT *output, EdgeT *begin, EdgeT *end, RingT *out_off, int32_t *status,
        int64_t rows, int64_t segments, int64_t cols, int64_t slots,
        int64_t task_capacity, cudaStream_t stream)
    {
        cudaError_t error = detail::ValidateLaunchArguments<ValueT, RingT>(
            rows, segments, cols, slots, task_capacity, head, output, status,
            {begin, end, out_off}, {indptr, events, counts, offsets, row_tasks});
        if (error != cudaSuccess)
            return error;
        error = detail::PrepareNative<EventT, EdgeT, kThreshold, kTaskNnz, kBlockSize>(
            indptr, events, counts, row_tasks, head, status,
            rows, segments, slots, static_cast<int>(task_capacity), stream);
        if (error != cudaSuccess)
            return error;
        return LaunchPrepared<Homo, ValueT, EventT, IndexT, EdgeT, RingT>(
            weights, indices, indptr, events, counts, offsets, row_tasks, head,
            output, begin, end, out_off, status, rows, segments, cols, slots, task_capacity, stream);
    }

#define INSTANTIATE_ONE(H, V, X, I, E, R)                                 \
    template cudaError_t launch_binary<H, V, X, I, E, R>(                 \
        const V *, const I *, const E *, const X *, const E *, const R *, \
        const int32_t *, const int32_t *, V *, E *, E *, R *, int32_t *,  \
        int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t);
#define INSTANTIATE_EVENTS(H, V, I, E, R)  \
    INSTANTIATE_ONE(H, V, int8_t, I, E, R) \
    INSTANTIATE_ONE(H, V, float, I, E, R)
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
    template <bool Homo, typename ValueT, typename EventT, typename IndexT, typename EdgeT, typename RingT>
    static void LaunchTensor(
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,
        const BE::Tensor events, const BE::Tensor counts, const BE::Tensor offsets,
        const BE::Tensor row_tasks, const BE::Tensor head,
        BE::Tensor output, BE::Tensor begin, BE::Tensor end, BE::Tensor out_off,
        BE::Tensor status, cudaStream_t stream)
    {
        BE_CUDA_CHECK(cudaMemsetAsync(status.data_ptr<int32_t>(), 0, 2 * sizeof(int32_t), stream));
        BE_CUDA_CHECK((LaunchPrepared<Homo, ValueT, EventT, IndexT, EdgeT, RingT>(
            weights.data_ptr<const ValueT>(), indices.data_ptr<const IndexT>(),
            indptr.data_ptr<const EdgeT>(), events.data_ptr<const EventT>(),
            counts.data_ptr<const EdgeT>(), offsets.data_ptr<const RingT>(),
            row_tasks.data_ptr<const int32_t>(), head.data_ptr<const int32_t>(),
            output.data_ptr<ValueT>(), begin.data_ptr<EdgeT>(), end.data_ptr<EdgeT>(),
            out_off.data_ptr<RingT>(), status.data_ptr<int32_t>(), events.numel(),
            offsets.numel(), output.shape(1), output.shape(0),
            begin.numel(), stream)));
    }

    template <typename ValueT, typename EventT>
    void DeliverImpl(
        const BE::Tensor &weights, const BE::Tensor &indices, const BE::Tensor &indptr,
        const BE::Tensor &events, const BE::Tensor &counts, const BE::Tensor &offsets,
        const BE::Tensor &row_tasks, const BE::Tensor &head, const BE::Tensor &buffer,
        const BE::Tensor &begin_scratch, const BE::Tensor &end_scratch,
        const BE::Tensor &off_scratch, const BE::Tensor &status_scratch,
        BE::Tensor &output, BE::Tensor &begin, BE::Tensor &end, BE::Tensor &out_off,
        BE::Tensor &status, int64_t stream)
    {
        // The shared JAX primitive validates operands; FFI checks runtime aliasing.
        BE_CHECK(buffer.data_ptr() == output.data_ptr()) << "ring requires input/output aliasing";
        const BE::Tensor scratch[4] = {begin_scratch, end_scratch, off_scratch, status_scratch};
        const BE::Tensor result[4] = {begin, end, out_off, status};
        for (int i = 0; i < 4; ++i)
        {
            BE_CHECK(scratch[i].data_ptr() == result[i].data_ptr()) << "workspace requires aliasing";
        }
        const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, EdgeT, {
            BE_DISPATCH_SIGNED_INDEX(offsets.dtype(), RingT, {
                if (weights.numel() == 1)
                    LaunchTensor<true, ValueT, EventT, IndexT, EdgeT, RingT>(weights, indices, indptr, events, counts, offsets, row_tasks, head, output, begin, end, out_off, status, cuda_stream);
                else
                    LaunchTensor<false, ValueT, EventT, IndexT, EdgeT, RingT>(weights, indices, indptr, events, counts, offsets, row_tasks, head, output, begin, end, out_off, status, cuda_stream);
            });
        });
    }

} // namespace region_delay

// region_delay_{f16,bf16,f32,f64}_{bool,float}: weights and ring share the value dtype;
// events are bool or float32 (active iff != 0). Index dtypes are dispatched at run time.
#define DEFINE_REGION_DELAY(SUFFIX, VALUE_T, EVENT_T)                                             \
    void region_delay##SUFFIX(                                                                    \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,              \
        const BE::Tensor events, const BE::Tensor counts, const BE::Tensor offsets,               \
        const BE::Tensor row_tasks, const BE::Tensor head, const BE::Tensor buffer,               \
        const BE::Tensor begin_scratch, const BE::Tensor end_scratch,                             \
        const BE::Tensor off_scratch, const BE::Tensor status_scratch,                            \
        BE::Tensor output, BE::Tensor begin, BE::Tensor end, BE::Tensor out_off,                  \
        BE::Tensor status, int64_t stream)                                                        \
    {                                                                                             \
        region_delay::DeliverImpl<VALUE_T, EVENT_T>(                                              \
            weights, indices, indptr, events, counts, offsets, row_tasks,                        \
            head, buffer, begin_scratch, end_scratch, off_scratch, status_scratch, output, begin, \
            end, out_off, status, stream);                                                        \
    }

// @BE region_delay_f16_bool
DEFINE_REGION_DELAY(_f16_bool, __half, int8_t)
// @BE region_delay_f16_float
DEFINE_REGION_DELAY(_f16_float, __half, float)
// @BE region_delay_bf16_bool
DEFINE_REGION_DELAY(_bf16_bool, __nv_bfloat16, int8_t)
// @BE region_delay_bf16_float
DEFINE_REGION_DELAY(_bf16_float, __nv_bfloat16, float)
// @BE region_delay_f32_bool
DEFINE_REGION_DELAY(_f32_bool, float, int8_t)
// @BE region_delay_f32_float
DEFINE_REGION_DELAY(_f32_float, float, float)
// @BE region_delay_f64_bool
DEFINE_REGION_DELAY(_f64_bool, double, int8_t)
// @BE region_delay_f64_float
DEFINE_REGION_DELAY(_f64_float, double, float)
#endif // BE_REGION_DELAY_STANDALONE
