// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef DEVING_BATCHED_V3_HYBRID_SDDMM_IMPL_CUH
#define DEVING_BATCHED_V3_HYBRID_SDDMM_IMPL_CUH

// CUDA implementation shared by hybrid_sddmm_binary.cu (binary) and hybrid_sddmm_weighted.cu (weighted).
// Everything is a template; each operator instantiates its own launch_impl.
//
// Schedule: per batch group, compact_kernel ballots each batch's active rows into a short-row
// list and splits long rows into <= kTask-slot tasks; short_kernel lets G lanes cooperate on a
// short row; task_kernel gives each task a warp. Contributions are atomically added in AccT:
// float/double straight into dweight, half/bfloat16 into a float scratch behind the workspace
// that finalize_kernel adds to dweight with a single rounding.
//
// QueueT selects the semantics: void is binary (dweight[j] += ct[b, col]), ValueT is weighted
// (dweight[j] += events[b, row] * ct[b, col]). Weighted is the vw standard, no row lookup:
// short_kernel re-reads events[b, row] from the short list's row; long-row tasks carry their
// event value in the task value queue Scratch::value, written by compact_kernel.
#include "hybrid_sddmm.cuh"
#include "../_common/cuda_traits.cuh"

#include <type_traits>

namespace batched_v3::detail
{
    template <class QueueT>
    constexpr bool kWeighted = !std::is_void<QueueT>::value;

    template <class QueueT>
    constexpr int64_t queue_bytes()
    {
        if constexpr (kWeighted<QueueT>)
            return sizeof(QueueT);
        else
            return 0;
    }

    // Workspace views, see workspace_bytes in hybrid_sddmm_plan.cuh. value is empty for binary (QueueT=void).
    template <class QueueT>
    struct Scratch
    {
        int32_t *status;
        unsigned long long *counts;
        int64_t *begin;
        QueueT *value;
        int32_t *meta;
    };
    template <class QueueT>
    inline Scratch<QueueT> split(void *buffer, const Plan &p)
    {
        auto *status = static_cast<int32_t *>(buffer);
        auto *counts = reinterpret_cast<unsigned long long *>(status + 2LL * p.batches);
        auto *begin = reinterpret_cast<int64_t *>(counts + groups(p));
        // The queue follows begin, so it stays 8-byte aligned for double at any capacity; its padded
        // region keeps meta 4-byte aligned.
        char *value = reinterpret_cast<char *>(begin + p.capacity);
        return {status, counts, begin, reinterpret_cast<QueueT *>(value),
                reinterpret_cast<int32_t *>(value + queue_region_bytes(p.capacity, queue_bytes<QueueT>()))};
    }

    // scale * term for weighted contributions; binary adds the term unscaled.
    template <class QueueT, class AccT>
    __device__ __forceinline__ AccT contribution(AccT scale, AccT term)
    {
        if constexpr (kWeighted<QueueT>)
            return scale * term;
        else
            return term;
    }

    template <class T>
    __device__ void prefix(T *offsets, T *base, T *counter)
    {
        const int lane = threadIdx.x & 31;
        T local = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            local += offsets[lane * 4 + i];
        T inclusive = local;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1)
        {
            const T x = __shfl_up_sync(kFull, inclusive, d);
            if (lane >= d)
                inclusive += x;
        }
        T running = inclusive - local;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const T n = offsets[lane * 4 + i];
            offsets[lane * 4 + i] = running;
            running += n;
        }
        if (lane == 31)
            *base = inclusive ? atomicAdd(counter, inclusive) : 0;
    }

    // Weighted long-row tasks also store the row's event value: the owning lane re-reads
    // events[b, r] (just loaded, so cached; __ldg keeps it a distinct load) instead of keeping
    // values[16] live across the prefix.
    // Bounds: without a per-SM block minimum ptxas hoists those reloads to 60-64 registers; the
    // minimum restores 48 (Mixed) and 40, the binary counts, for both operators. Binary keeps its
    // register counts and occupancy but its double-event Mixed instance spills 4 bytes; only the
    // plain (256) bound avoids that, and it cannot differ per instance of one kernel (measured
    // alternatives: a shared __device__ body, __maxnreg__, or other minimums all cost more).
    template <bool Mixed, class EventT, class QueueT, class OffsetT>
    __global__ void __launch_bounds__(256, Mixed ? 5 : 6) compact_kernel(
        const EventT *__restrict__ events, const OffsetT *__restrict__ ptr,
        const uint32_t *__restrict__ long_bits, int32_t *__restrict__ selected,
        Scratch<QueueT> w, int rows, int batch_begin, int group_id, int64_t capacity)
    {
        using Value = std::conditional_t<kWeighted<QueueT>, QueueT, int>;
        __shared__ int short_offsets[kSegments], short_base;
        __shared__ unsigned long long task_offsets[kSegments], task_base;
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int local_b = blockIdx.y, b = batch_begin + local_b;
        const int64_t row_base = static_cast<int64_t>(blockIdx.x) * kRows;
        unsigned short_masks[16], long_masks[16];
        EventT values[16];
        uint32_t words[16];

        {
#pragma unroll
            for (int i = 0; i < 16; ++i)
            {
                const int64_t r = row_base + (i * 8 + warp) * 32 + lane;
                values[i] = r < rows ? events[static_cast<int64_t>(b) * rows + r] : EventT(0);
                if constexpr (Mixed)
                    words[i] = r < rows ? long_bits[r / 32] : 0;
            }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            const bool active = cuda_traits::EventTraits<EventT>::active(values[i]);
            bool is_long = false;
            if constexpr (Mixed)
            {
                const uint32_t word = words[i];
                is_long = active && ((word >> lane) & 1U);
                long_masks[i] = __ballot_sync(kFull, is_long);
            }
            short_masks[i] = __ballot_sync(kFull, active && !is_long);
            if (lane == 0)
                short_offsets[i * 8 + warp] = __popc(short_masks[i]);
        }
        if constexpr (Mixed)
        {
#pragma unroll
            for (int i = 0; i < 16; ++i)
            {
                unsigned n = 0;
                if (long_masks[i])
                {
                    const int64_t r = row_base + (i * 8 + warp) * 32 + lane;
                    const int64_t len = (long_masks[i] >> lane & 1U) ? static_cast<int64_t>(ptr[r + 1]) - ptr[r] : 0;
                    n = static_cast<unsigned>((len + kTask - 1) / kTask);
#pragma unroll
                    for (int d = 16; d; d >>= 1)
                        n += __shfl_xor_sync(kFull, n, d);
                }
                if (lane == 0)
                    task_offsets[i * 8 + warp] = n;
            }
        }
        __syncthreads();
        if (warp == 0)
        {
            prefix(short_offsets, &short_base, w.status + 2LL * b);
            if constexpr (Mixed)
                prefix(task_offsets, &task_base, w.counts + group_id);
        }
        __syncthreads();
        int32_t *list = selected + static_cast<int64_t>(Mixed ? local_b : b) * rows;
        const unsigned lower = (1U << lane) - 1U;
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            const int s = i * 8 + warp;
            const int64_t r = row_base + s * 32 + lane;
            if (short_masks[i] >> lane & 1U)
                list[short_base + short_offsets[s] + __popc(short_masks[i] & lower)] = static_cast<int32_t>(r);
            if constexpr (Mixed)
            {
                if (!long_masks[i])
                    continue;
                int64_t begin = 0, len = 0;
                Value value = Value(0);
                if (long_masks[i] >> lane & 1U)
                {
                    begin = ptr[r];
                    len = static_cast<int64_t>(ptr[r + 1]) - begin;
                    if constexpr (kWeighted<QueueT>)
                        value = __ldg(events + static_cast<int64_t>(b) * rows + r);
                }
                const unsigned n = static_cast<unsigned>((len + kTask - 1) / kTask);
                unsigned inclusive = n;
#pragma unroll
                for (int d = 1; d < 32; d <<= 1)
                {
                    const unsigned up = __shfl_up_sync(kFull, inclusive, d);
                    if (lane >= d)
                        inclusive += up;
                }
                const unsigned long long slot = task_base + task_offsets[s] + inclusive - n;
                if (slot > static_cast<unsigned long long>(capacity) || n > static_cast<unsigned long long>(capacity) - slot)
                {
                    if (n)
                        atomicExch(w.status + 2LL * b + 1, 1);
                    continue;
                }
                for (unsigned t = 0; t < n; ++t)
                {
                    const int64_t off = static_cast<int64_t>(t) * kTask;
                    w.begin[slot + t] = begin + off;
                    if constexpr (kWeighted<QueueT>)
                        w.value[slot + t] = value;
                    w.meta[slot + t] = (local_b << 12) | static_cast<int>(min(static_cast<int64_t>(kTask), len - off));
                }
            }
        }
    }

    // G lanes per active short row; weighted reads the row's event value once per row. Weighted
    // events share QueueT; binary passes them as an unused const void *, so binary instances do
    // not multiply per event type.
    template <int G, class ValueT, class QueueT, class IndexT, class OffsetT>
    __global__ void __launch_bounds__(256) short_kernel(
        const QueueT *__restrict__ events, const ValueT *__restrict__ ct, const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ ptr, const int32_t *__restrict__ selected, const int32_t *__restrict__ status,
        typename cuda_traits::ValueTraits<ValueT>::AccT *__restrict__ dw, int rows, int cols, int batch_begin,
        bool local_list)
    {
        using Traits = cuda_traits::ValueTraits<ValueT>;
        using AccT = typename Traits::AccT;
        const int local_b = blockIdx.y, b = batch_begin + local_b, lane = threadIdx.x % G;
        const int count = min(status[2LL * b], rows);
        const int32_t *list = selected + static_cast<int64_t>(local_list ? local_b : b) * rows;
        const ValueT *x = ct + static_cast<int64_t>(b) * cols;
        for (int64_t item = static_cast<int64_t>(blockIdx.x) * (256 / G) + threadIdx.x / G; item < count; item += static_cast<int64_t>(gridDim.x) * (256 / G))
        {
            const int row = list[item];
            AccT scale = AccT(1);
            if constexpr (kWeighted<QueueT>)
                scale = Traits::read(events[static_cast<int64_t>(b) * rows + row]);
            for (int64_t j = static_cast<int64_t>(ptr[row]) + lane; j < static_cast<int64_t>(ptr[row + 1]); j += G)
                atomicAdd(dw + j, contribution<QueueT>(scale, Traits::read(x[__ldg(indices + j)])));
        }
    }
    // One warp per long-row task; weighted takes the event value from the task value queue.
    template <class ValueT, class QueueT, class IndexT>
    __global__ void __launch_bounds__(256) task_kernel(
        const ValueT *__restrict__ ct, const IndexT *__restrict__ indices,
        typename cuda_traits::ValueTraits<ValueT>::AccT *__restrict__ dw,
        Scratch<QueueT> w, int cols, int begin_batch, int group_id, int64_t capacity)
    {
        using Traits = cuda_traits::ValueTraits<ValueT>;
        using AccT = typename Traits::AccT;
        const auto count = min(w.counts[group_id], static_cast<unsigned long long>(capacity));
        const int lane = threadIdx.x & 31;
        for (unsigned long long task = (static_cast<unsigned long long>(blockIdx.x) * 256 + threadIdx.x) / 32;
             task < count; task += static_cast<unsigned long long>(gridDim.x) * 8)
        {
            const int64_t begin = w.begin[task];
            const int meta = w.meta[task];
            AccT scale = AccT(1);
            if constexpr (kWeighted<QueueT>)
                scale = Traits::read(w.value[task]);
            const ValueT *x = ct + static_cast<int64_t>(begin_batch + (meta >> 12)) * cols;
            for (int64_t j = begin + lane; j < begin + (meta & 4095); j += 32)
                atomicAdd(dw + j, contribution<QueueT>(scale, Traits::read(x[__ldg(indices + j)])));
        }
    }
    // dweight[j] += acc[j] once, rounding to the storage type a single time. Slots whose sum is
    // zero are left untouched, so inactive rows keep their exact bits (NaN payloads, -0.0);
    // a NaN sum compares != 0 and is still written.
    template <class ValueT>
    __global__ void __launch_bounds__(256) finalize_kernel(
        ValueT *__restrict__ dw, const typename cuda_traits::ValueTraits<ValueT>::AccT *__restrict__ acc, int64_t nnz)
    {
        using Traits = cuda_traits::ValueTraits<ValueT>;
        for (int64_t j = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x; j < nnz;
             j += static_cast<int64_t>(gridDim.x) * 256)
        {
            const auto sum = acc[j];
            if (sum != 0)
                dw[j] = Traits::write(Traits::read(dw[j]) + sum);
        }
    }

    // Shared host schedule. workspace must hold workspace_bytes(p, queue_bytes<QueueT>()) +
    // accumulator_bytes<ValueT>(nnz) bytes.
    template <class ValueT, class EventT, class QueueT, class IndexT, class OffsetT>
    cudaError_t launch_impl(const EventT *events, const ValueT *ct, const IndexT *indices, const OffsetT *ptr,
                            ValueT *dweight, const uint32_t *mask, int32_t *selected, void *workspace, int rows,
                            int cols, int64_t nnz, const Plan &p, cudaStream_t stream)
    {
        using AccT = typename cuda_traits::ValueTraits<ValueT>::AccT;
        constexpr bool kDirect = std::is_same<AccT, ValueT>::value;
        static_assert(kDirect == (sizeof(ValueT) != 2), "accumulator_bytes in hybrid_sddmm_plan.cuh assumes half types sum in float");
        static_assert(!kWeighted<QueueT> || std::is_same<EventT, QueueT>::value, "weighted events must be the queue type");
        const Scratch<QueueT> w = split<QueueT>(workspace, p);
        AccT *dw = kDirect ? reinterpret_cast<AccT *>(dweight)
                           : reinterpret_cast<AccT *>(static_cast<char *>(workspace) + workspace_bytes(p, queue_bytes<QueueT>()));
        cudaError_t error = cudaMemsetAsync(workspace, 0, p.batches * 8LL + groups(p) * 8LL, stream);
        if (error != cudaSuccess || !nnz)
            return error;
        if constexpr (!kDirect)
            if ((error = cudaMemsetAsync(dw, 0, accumulator_bytes<ValueT>(nnz), stream)) != cudaSuccess)
                return error;
        const int blocks = static_cast<int>((static_cast<int64_t>(rows) + kRows - 1) / kRows);
        if (!p.long_count)
        {
            compact_kernel<false, EventT, QueueT, OffsetT><<<dim3(blocks, p.batches), 256, 0, stream>>>(events, ptr, mask, selected, w, rows, 0, 0, 0);
            if ((error = cudaGetLastError()) != cudaSuccess)
                return error;
        }
        for (int begin = 0, g = 0; begin < p.batches; begin += p.chunk, ++g)
        {
            const int size = std::min(p.chunk, p.batches - begin);
            if (p.long_count)
            {
                compact_kernel<true, EventT, QueueT, OffsetT><<<dim3(blocks, size), 256, 0, stream>>>(events, ptr, mask, selected, w, rows, begin, g, p.capacity);
                if ((error = cudaGetLastError()) != cudaSuccess)
                    return error;
            }
            if (p.short_count)
            {
                const int grid = std::max(1, p.sms * 6 / size);
                const auto *short_events = reinterpret_cast<const QueueT *>(events);  // EventT == QueueT if weighted
#define SHORT(G) short_kernel<G, ValueT, QueueT, IndexT, OffsetT><<<dim3(grid, size), 256, 0, stream>>>(short_events, ct, indices, ptr, selected, w.status, dw, rows, cols, begin, p.long_count != 0)
                switch (p.group)
                {
                case 4:
                    SHORT(4);
                    break;
                case 8:
                    SHORT(8);
                    break;
                case 16:
                    SHORT(16);
                    break;
                default:
                    SHORT(32);
                    break;
                }
#undef SHORT
                if ((error = cudaGetLastError()) != cudaSuccess)
                    return error;
            }
            if (p.long_count)
            {
                task_kernel<ValueT, QueueT, IndexT><<<p.sms * 6, 256, 0, stream>>>(ct, indices, dw, w, cols, begin, g, p.capacity);
                if ((error = cudaGetLastError()) != cudaSuccess)
                    return error;
            }
        }
        if constexpr (!kDirect)
        {
            const int64_t needed = (nnz + 255) / 256;
            const int grid = static_cast<int>(std::min<int64_t>(needed, static_cast<int64_t>(p.sms) * 32));
            finalize_kernel<ValueT><<<grid, 256, 0, stream>>>(dweight, dw, nnz);
            if ((error = cudaGetLastError()) != cudaSuccess)
                return error;
        }
        return cudaSuccess;
    }
} // namespace batched_v3::detail

#endif // DEVING_BATCHED_V3_HYBRID_SDDMM_IMPL_CUH
