// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef HYBRID_BATCHEDV3_SPMM_IMPL_CUH
#define HYBRID_BATCHEDV3_SPMM_IMPL_CUH
// hybrid (v3): neuron-major push SpMM for BN spikes into a BN output, shared by
// hybrid_batchedv3_spmm_binary.cu (binary) and hybrid_batchedv3_spmm_weighted.cu (weighted). Everything is a template.
// Per batch group of C batches (C x N x value bytes <= L2/4, the output being RED-updated at once):
//   K1 classify_extract_kernel: fused_extract's ballot extraction (4096 rows x 1 batch per block);
//      a static long-row bitmask (one word per 32 rows, built once) splits active rows: short rows
//      go to the batch's compact list, long rows become <= kTaskEdges-edge tasks (only they read
//      row_ptr); one atomicAdd per block for each list;
//   K2 scatter_short_kernel<G> (batched_mv2::scatter_kernel) over the short lists;
//   K3 task_kernel: one warp per task.
// Every work item runs at most kIterations strided iterations (short: G lanes, task: 32 lanes).
// kZeroOutput (launch_group_zeroed): K1 also zeroes its group's output slice right before K2/K3
// accumulate into it, so the zeros are still in L2 (no separate memset; no precondition on output).
//
// QueueT selects the semantics: void is binary (output[b, col] += w), ValueT is weighted
// (output[b, col] += events[b, row] * w). Weighted follows the vw standard, no row lookup: K2
// re-reads events[b, row] from the short list's row; K3 tasks carry their event value in the task
// value queue written by K1. Types: values/output ValueT in {half, bfloat16, float, double}
// (atomics through cuda_traits::ValueTraits, accumulating in AccT); binary events int8 (bool) or
// float, active iff != 0; indices IndexT and row_ptr OffsetT in {int32, int64}.
#include "hybrid_batchedv3_spmm.cuh"

#include <type_traits>

namespace hybrid
{
  template <class QueueT>
  constexpr bool kWeighted = !std::is_void<QueueT>::value;

  template <class QueueT>
  constexpr size_t queue_bytes()
  {
    if constexpr (kWeighted<QueueT>)
      return sizeof(QueueT);
    else
      return 0;
  }

  // Workspace views, see workspace_bytes in hybrid_batchedv3_spmm_plan.cuh; task_value is empty for binary (QueueT=void).
  template <class QueueT>
  struct Workspace
  {
    int32_t *status;
    unsigned long long *counters;
    int64_t *task_begin;
    QueueT *task_value;
    int32_t *task_meta;
  };

  template <class QueueT>
  inline Workspace<QueueT> split_workspace(void *workspace, const Plan &plan, int batch)
  {
    auto *bytes = static_cast<char *>(workspace);
    auto *counters = reinterpret_cast<unsigned long long *>(bytes + counters_offset(batch));
    auto *task_begin = reinterpret_cast<int64_t *>(counters + group_count(batch, plan.chunk));
    char *task_value = reinterpret_cast<char *>(task_begin + plan.task_capacity);
    return {reinterpret_cast<int32_t *>(workspace), counters, task_begin, reinterpret_cast<QueueT *>(task_value),
            reinterpret_cast<int32_t *>(task_value + queue_region_bytes(plan.task_capacity, queue_bytes<QueueT>()))};
  }

  // Binary adds the weight term; weighted scales it by the event value.
  template <class QueueT, class AccT>
  __device__ __forceinline__ AccT contribution(AccT scale, AccT term)
  {
    if constexpr (kWeighted<QueueT>)
      return scale * term;
    else
      return term;
  }

  // kZeroOutput: block (x, y) first zeroes output[batch_begin + y, x*4096 : (x+1)*4096); the grid's x
  // then spans max(row chunks, column chunks). Weighted (QueueT = ValueT = EventT): each long-row task
  // also stores its event value in task_value; the owning lane re-reads spikes[row] (just loaded, so
  // cached; __ldg keeps it a distinct load) instead of keeping the 16 events live across the scans.
  template <bool kZeroOutput, typename ValueT, typename EventT, typename QueueT, typename OffsetT>
  __global__ void __launch_bounds__(kExtractBlockSize) classify_extract_kernel(
      ValueT *__restrict__ output_bn,
      int cols,
      const EventT *__restrict__ spike_bn,
      const OffsetT *__restrict__ row_ptr,
      const uint32_t *__restrict__ long_mask,
      int32_t *__restrict__ short_rows,
      int32_t *__restrict__ status,
      unsigned long long *__restrict__ task_counter,
      int64_t *__restrict__ task_begin,
      QueueT *__restrict__ task_value,
      int32_t *__restrict__ task_meta,
      int rows,
      int batch_begin,
      int64_t task_capacity)
  {
    constexpr int kWarps = kWarpsPerBlock;
    constexpr int kSegments = kSegmentsPerChunk;
    __shared__ int short_offsets[kSegments];
    __shared__ long long task_offsets[kSegments];
    __shared__ int short_base;
    __shared__ unsigned long long task_base;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int b_local = blockIdx.y;
    const int b = batch_begin + b_local;
    const int64_t row_base = static_cast<int64_t>(blockIdx.x) * kRowsPerChunk;
    const EventT *spikes = spike_bn + static_cast<size_t>(b) * rows;
    if constexpr (kZeroOutput)
    {
      ValueT *slice = output_bn + static_cast<size_t>(b) * cols;
      const ValueT zero = cuda_traits::ValueTraits<ValueT>::write(0);
      const int64_t col_end = min(row_base + kRowsPerChunk, static_cast<int64_t>(cols));
      for (int64_t col = row_base + threadIdx.x; col < col_end; col += kExtractBlockSize)
        slice[col] = zero;
    }
    unsigned short_masks[kSegmentsPerWarp], long_masks[kSegmentsPerWarp];
    {
      // Issue all 16 event and bitmask loads before the first ballot.
      EventT event[kSegmentsPerWarp];
      uint32_t word[kSegmentsPerWarp];
#pragma unroll
      for (int iteration = 0; iteration < kSegmentsPerWarp; ++iteration)
      {
        const int64_t row = row_base + (iteration * kWarps + warp) * 32 + lane;
        const bool in_range = row < rows;
        event[iteration] = in_range ? spikes[row] : EventT(0);
        word[iteration] = in_range ? long_mask[row >> 5] : 0U;
      }
#pragma unroll
      for (int iteration = 0; iteration < kSegmentsPerWarp; ++iteration)
      {
        const bool active = cuda_traits::EventTraits<EventT>::active(event[iteration]);
        const bool is_long = active && (word[iteration] >> lane & 1U);
        short_masks[iteration] = __ballot_sync(0xffffffffU, active && !is_long); // empty rows: no-op events
        long_masks[iteration] = __ballot_sync(0xffffffffU, is_long);
      }
    }
#pragma unroll
    for (int iteration = 0; iteration < kSegmentsPerWarp; ++iteration)
    {
      const int segment = iteration * kWarps + warp;
      unsigned segment_tasks = 0;
      if (long_masks[iteration]) // warp-uniform and rare: only active long rows read row_ptr
      {
        const int64_t row = row_base + segment * 32 + lane;
        const int64_t len = long_masks[iteration] >> lane & 1U ? static_cast<int64_t>(row_ptr[row + 1]) - row_ptr[row] : 0;
        segment_tasks = __reduce_add_sync(0xffffffffU, static_cast<unsigned>((len + kTaskEdges - 1) / kTaskEdges));
      }
      if (lane == 0)
      {
        short_offsets[segment] = __popc(short_masks[iteration]);
        task_offsets[segment] = segment_tasks;
      }
    }
    __syncthreads();
    if (warp == 0) // exclusive scans of the 128 segment counts: 4 segments per lane, then a warp scan
    {
      constexpr int kPerLane = kSegments / 32;
      int short_local = 0;
      long long task_local = 0;
#pragma unroll
      for (int k = 0; k < kPerLane; ++k)
      {
        short_local += short_offsets[lane * kPerLane + k];
        task_local += task_offsets[lane * kPerLane + k];
      }
      int short_inclusive = short_local;
      long long task_inclusive = task_local;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1)
      {
        const int short_up = __shfl_up_sync(0xffffffffU, short_inclusive, d);
        const long long task_up = __shfl_up_sync(0xffffffffU, task_inclusive, d);
        if (lane >= d)
        {
          short_inclusive += short_up;
          task_inclusive += task_up;
        }
      }
      int short_running = short_inclusive - short_local;
      long long task_running = task_inclusive - task_local;
#pragma unroll
      for (int k = 0; k < kPerLane; ++k)
      {
        const int s = short_offsets[lane * kPerLane + k];
        const long long t = task_offsets[lane * kPerLane + k];
        short_offsets[lane * kPerLane + k] = short_running;
        task_offsets[lane * kPerLane + k] = task_running;
        short_running += s;
        task_running += t;
      }
      if (lane == 31)
      {
        short_base = short_inclusive ? atomicAdd(&status[static_cast<size_t>(b) * 2 + kStatusCount], short_inclusive) : 0;
        task_base = task_inclusive ? atomicAdd(task_counter, static_cast<unsigned long long>(task_inclusive)) : 0;
      }
    }
    __syncthreads();
    int32_t *selected = short_rows + static_cast<size_t>(b) * rows;
    const unsigned lower_lanes = (1U << lane) - 1U;
#pragma unroll
    for (int iteration = 0; iteration < kSegmentsPerWarp; ++iteration)
    {
      const int segment = iteration * kWarps + warp;
      const int64_t row = row_base + segment * 32 + lane;
      const unsigned mask = short_masks[iteration];
      if (mask & (1U << lane))
        selected[short_base + short_offsets[segment] + __popc(mask & lower_lanes)] = static_cast<int32_t>(row);
      if (long_masks[iteration] == 0) // warp-uniform
        continue;
      int64_t begin = 0, len = 0;
      std::conditional_t<kWeighted<QueueT>, QueueT, int> value{};
      if (long_masks[iteration] & (1U << lane))
      {
        begin = row_ptr[row];
        len = static_cast<int64_t>(row_ptr[row + 1]) - begin;
        if constexpr (kWeighted<QueueT>)
          value = __ldg(spikes + row);
      }
      const int tasks = static_cast<int>((len + kTaskEdges - 1) / kTaskEdges);
      int inclusive = tasks;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1)
      {
        const int up = __shfl_up_sync(0xffffffffU, inclusive, d);
        if (lane >= d)
          inclusive += up;
      }
      const unsigned long long slot = task_base + task_offsets[segment] + (inclusive - tasks);
      if (slot + tasks > static_cast<unsigned long long>(task_capacity))
      {
        // Only reachable with a stale or foreign plan: report instead of dropping silently.
        if (tasks)
          atomicExch(&status[static_cast<size_t>(b) * 2 + kStatusOverflow], 1);
        continue;
      }
      for (int k = 0; k < tasks; ++k)
      {
        const int64_t offset = static_cast<int64_t>(k) * kTaskEdges;
        const int64_t edges = len - offset < kTaskEdges ? len - offset : kTaskEdges;
        task_begin[slot + k] = begin + offset;
        if constexpr (kWeighted<QueueT>)
          task_value[slot + k] = value;
        task_meta[slot + k] = (b_local << kMetaLenBits) | static_cast<int>(edges);
      }
    }
  }

  // Only weight reads are permuted: entry, indices and task ranges stay in mirror order.
  // Existing row-major launchers specialize Permuted=false and do not access perm.
  template <bool Permuted, typename ValueT, typename OffsetT>
  __device__ __forceinline__ typename cuda_traits::ValueTraits<ValueT>::AccT read_edge_weight(
      const ValueT *values, const OffsetT *perm, int64_t entry)
  {
    if constexpr (Permuted)
      return cuda_traits::ValueTraits<ValueT>::read(values[perm[entry]]);
    else
      return cuda_traits::ValueTraits<ValueT>::read(values[entry]);
  }

  // K3: one warp per long-row task; weighted takes the event value from the task value queue.
  template <bool Homo, bool Permuted, typename ValueT, typename QueueT, typename IndexT, typename OffsetT>
  __global__ void __launch_bounds__(kBlockSize) task_kernel(
      const ValueT *__restrict__ values,
      const IndexT *__restrict__ indices,
      const OffsetT *__restrict__ perm,
      const unsigned long long *__restrict__ task_counter,
      const int64_t *__restrict__ task_begin,
      const QueueT *__restrict__ task_value,
      const int32_t *__restrict__ task_meta,
      ValueT *__restrict__ output_group,
      int cols,
      int64_t task_capacity)
  {
    using Traits = cuda_traits::ValueTraits<ValueT>;
    typename Traits::AccT shared_weight{};
    if constexpr (Homo)
      shared_weight = Traits::read(values[0]);
    const unsigned long long count = min(*task_counter, static_cast<unsigned long long>(task_capacity));
    const int lane = threadIdx.x & 31;
    const unsigned long long warps = static_cast<unsigned long long>(gridDim.x) * (kBlockSize / 32);
    for (unsigned long long task = (blockIdx.x * static_cast<unsigned long long>(kBlockSize) + threadIdx.x) >> 5;
         task < count; task += warps)
    {
      const int64_t begin = task_begin[task];
      const int meta = task_meta[task];
      typename Traits::AccT scale{1};
      if constexpr (kWeighted<QueueT>)
        scale = Traits::read(task_value[task]);
      ValueT *output = output_group + static_cast<size_t>(meta >> kMetaLenBits) * cols;
      const int64_t end = begin + (meta & ((1 << kMetaLenBits) - 1));
      for (int64_t entry = begin + lane; entry < end; entry += 32)
        Traits::atomic_add(&output[indices[entry]],
                           contribution<QueueT>(scale, Homo ? shared_weight : read_edge_weight<Permuted>(values, perm, entry)));
    }
  }


  // K2 (batched_mv2::scatter_kernel): kGroup lanes scatter one active short row of batch b. Weighted
  // re-reads the row's event value once per row; binary passes events as an unused const void *.
  template <int kGroup, bool Homo, bool Permuted, typename ValueT, typename QueueT, typename IndexT, typename OffsetT>
  __global__ void __launch_bounds__(kShortBlockSize) scatter_short_kernel(
      const QueueT *__restrict__ events,
      const ValueT *__restrict__ values,
      const IndexT *__restrict__ indices,
      const OffsetT *__restrict__ row_ptr,
      const OffsetT *__restrict__ perm,
      const int32_t *__restrict__ active_rows,
      const int32_t *__restrict__ status,
      ValueT *__restrict__ output_bn,
      int rows,
      int cols,
      int batch_begin)
  {
    using Traits = cuda_traits::ValueTraits<ValueT>;
    typename Traits::AccT shared_weight{};
    if constexpr (Homo)
      shared_weight = Traits::read(values[0]);
    constexpr int kGroups = kShortBlockSize / kGroup;
    const int b = batch_begin + static_cast<int>(blockIdx.y);
    const int count = min(status[static_cast<size_t>(b) * 2 + kStatusCount], rows);
    const int32_t *selected = active_rows + static_cast<size_t>(b) * rows;
    ValueT *output = output_bn + static_cast<size_t>(b) * cols;
    const int lane = threadIdx.x % kGroup;
    for (int index = blockIdx.x * kGroups + threadIdx.x / kGroup; index < count;
         index += gridDim.x * kGroups)
    {
      const int row = selected[index];
      typename Traits::AccT scale{1};
      if constexpr (kWeighted<QueueT>)
        scale = Traits::read(events[static_cast<size_t>(b) * rows + row]);
      const int64_t end = row_ptr[row + 1];
      for (int64_t entry = static_cast<int64_t>(row_ptr[row]) + lane; entry < end; entry += kGroup)
        Traits::atomic_add(&output[indices[entry]],
                           contribution<QueueT>(scale, Homo ? shared_weight : read_edge_weight<Permuted>(values, perm, entry)));
    }
  }

  template <int kGroup, bool Homo, bool Permuted, typename ValueT, typename QueueT, typename IndexT, typename OffsetT>
  cudaError_t scatter_short(const QueueT *events, const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                            const OffsetT *perm,
                            const int32_t *short_rows, const int32_t *status, ValueT *output_bn,
                            int rows, int cols, int begin, int size, int sms, cudaStream_t stream)
  {
    const int64_t groups = kShortBlockSize / kGroup;
    const int64_t blocks = std::max<int64_t>(
        1, std::min<int64_t>(static_cast<int64_t>(sms) * kBlocksPerSm / size, (rows + groups - 1) / groups));
    scatter_short_kernel<kGroup, Homo, Permuted, ValueT, QueueT, IndexT, OffsetT>
        <<<dim3(static_cast<unsigned>(blocks), size), kShortBlockSize, 0, stream>>>(
            events, values, indices, row_ptr, perm, short_rows, status, output_bn, rows, cols, begin);
    return cudaGetLastError();
  }

  // Shared schedule. Weighted: QueueT = EventT = ValueT, workspace sized with sizeof(ValueT).
  template <bool kZeroOutput, bool Homo, typename ValueT, typename EventT, typename QueueT, typename IndexT, typename OffsetT,
            bool Permuted = false>
  cudaError_t launch_impl(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                          const uint32_t *long_mask, const EventT *spike_bn, ValueT *output_bn,
                          int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                          const Plan &plan, cudaStream_t stream, const OffsetT *perm = nullptr)
  {
    static_assert(!Homo || !Permuted, "permuted weights are heterogeneous only");
    static_assert(!kWeighted<QueueT> || (std::is_same<EventT, QueueT>::value && std::is_same<ValueT, QueueT>::value),
                  "weighted events and values must be the queue type");
    if (rows < 0 || cols < 0 || batch < 0)
      return cudaErrorInvalidValue;
    if (batch == 0 || cols == 0)
      return cudaSuccess;
    if (!output_bn)
      return cudaErrorInvalidValue;
    if (rows == 0 || plan.nnz == 0)
      return kZeroOutput ? cudaMemsetAsync(output_bn, 0, static_cast<size_t>(batch) * cols * sizeof(ValueT), stream)
                         : cudaSuccess;
    if (!values || !indices || !row_ptr || (Permuted && !perm) || !long_mask || !spike_bn || !short_rows || !workspace ||
        plan.chunk <= 0 || plan.chunk > kMaxGroupBatches ||
        plan.task_capacity < 0 || (plan.group != 4 && plan.group != 8 && plan.group != 16 && plan.group != 32))
      return cudaErrorInvalidValue;
    Device info;
    cudaError_t result = device_info(&info);
    if (result != cudaSuccess)
      return result;
    const Workspace<QueueT> parts = split_workspace<QueueT>(workspace, plan, batch);
    const auto *short_events = reinterpret_cast<const QueueT *>(spike_bn); // EventT == QueueT if weighted
    if ((result = cudaMemsetAsync(workspace, 0, counters_offset(batch) + group_count(batch, plan.chunk) * sizeof(unsigned long long),
                                  stream)) != cudaSuccess)
      return result;
    const int extract_blocks = kZeroOutput ? std::max(row_chunks(rows), row_chunks(cols)) : row_chunks(rows);
    for (int g = 0, begin = 0; begin < batch; ++g, begin += plan.chunk)
    {
      const int size = std::min(plan.chunk, batch - begin);
      classify_extract_kernel<kZeroOutput, ValueT, EventT, QueueT, OffsetT><<<dim3(extract_blocks, size), kExtractBlockSize, 0, stream>>>(
          output_bn, cols, spike_bn, row_ptr, long_mask, short_rows, parts.status, parts.counters + g,
          parts.task_begin, parts.task_value, parts.task_meta, rows, begin, plan.task_capacity);
      if ((result = cudaGetLastError()) != cudaSuccess)
        return result;
      switch (plan.group)
      {
      case 4:
        result = scatter_short<4, Homo, Permuted, ValueT, QueueT, IndexT, OffsetT>(short_events, values, indices, row_ptr, perm, short_rows, parts.status, output_bn, rows, cols, begin, size, info.sms, stream);
        break;
      case 8:
        result = scatter_short<8, Homo, Permuted, ValueT, QueueT, IndexT, OffsetT>(short_events, values, indices, row_ptr, perm, short_rows, parts.status, output_bn, rows, cols, begin, size, info.sms, stream);
        break;
      case 16:
        result = scatter_short<16, Homo, Permuted, ValueT, QueueT, IndexT, OffsetT>(short_events, values, indices, row_ptr, perm, short_rows, parts.status, output_bn, rows, cols, begin, size, info.sms, stream);
        break;
      default:
        result = scatter_short<32, Homo, Permuted, ValueT, QueueT, IndexT, OffsetT>(short_events, values, indices, row_ptr, perm, short_rows, parts.status, output_bn, rows, cols, begin, size, info.sms, stream);
        break;
      }
      if (result != cudaSuccess)
        return result;
      if (plan.long_tasks == 0)
        continue;
      task_kernel<Homo, Permuted, ValueT, QueueT, IndexT, OffsetT><<<info.sms * kBlocksPerSm, kBlockSize, 0, stream>>>(
          values, indices, perm, parts.counters + g, parts.task_begin, parts.task_value, parts.task_meta,
          output_bn + static_cast<size_t>(begin) * cols, cols, plan.task_capacity);
      if ((result = cudaGetLastError()) != cudaSuccess)
        return result;
    }
    return cudaSuccess;
  }
} // namespace hybrid

#endif
