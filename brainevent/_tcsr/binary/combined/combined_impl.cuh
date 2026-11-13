// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef COMBINED_IMPL_CUH
#define COMBINED_IMPL_CUH
// Stack of the S2 winners: ordered single-pass extraction (fused_extract),
// Adaptive column tiles with at least 188 scatter CTAs, and absolute
// edge offsets without row_ptr (short_chain_abs).
//
// Self-contained: the pieces used from bn_tile (extraction geometry), tile_common
// (tile helpers), fused_extract (ordered extraction) and short_chain (absolute-offset
// scatter) form the typed implementation below; short_chain's relative-offset
// (row_ptr) branch is dropped because combined never uses it.
//
// Architecture: no sm_80-only intrinsic is used directly. The look-back warp sum goes
// through cuda_traits::warp_reduce_add_u32 (__reduce_add_sync on sm_80+, shuffles
// below). The Python dispatcher requires compute capability >= 6.0 (native double
// atomicAdd for f64 tiles) before it selects these kernels.
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "combined_kernels.cuh"
#include <limits>

namespace combined
{
  // ---- Extraction geometry and status layout (from bn_tile) ----
  constexpr int kExtractBlockSize = 256;
  constexpr int kWarpsPerBlock = kExtractBlockSize / 32;
  constexpr int kSegmentsPerWarp = 16;
  constexpr int kSegmentsPerChunk = kWarpsPerBlock * kSegmentsPerWarp;
  constexpr int kRowsPerChunk = 32 * kSegmentsPerChunk;
  constexpr int kStatusCount = 0;

  __host__ __device__ constexpr int row_chunks(int rows)
  {
    return rows <= 0 ? 0 : static_cast<int>((static_cast<int64_t>(rows) + kRowsPerChunk - 1) / kRowsPerChunk);
  }

  // ---- Tile helpers (from op1020 tile_common) ----
  constexpr int kPipelineDepth = 4;

  __host__ __device__ constexpr int tile_count(int cols, int width)
  {
    return static_cast<int>((static_cast<int64_t>(cols) + width - 1) / width);
  }

  struct DeviceLaunchResources
  {
    int max_threads = 0;
    int shared_bytes = 0;
    int optin_shared_bytes = 0;
    std::unordered_set<const void *> reserved_kernels;
  };

  inline cudaError_t prepare_scatter_kernel(const void *kernel, int threads, int shared_bytes)
  {
    // Kernel signatures can coincide across specializations. Cache the actual
    // function address on each device, and publish only successful reservations.
    static std::mutex mutex;
    static std::unordered_map<int, DeviceLaunchResources> devices;
    int device = 0;
    cudaError_t result = cudaGetDevice(&device);
    if (result != cudaSuccess)
      return result;
    const std::lock_guard<std::mutex> lock(mutex);
    auto entry = devices.find(device);
    if (entry == devices.end())
    {
      DeviceLaunchResources resources;
      result = cudaDeviceGetAttribute(&resources.max_threads, cudaDevAttrMaxThreadsPerBlock, device);
      if (result != cudaSuccess)
        return result;
      result = cudaDeviceGetAttribute(&resources.shared_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device);
      if (result != cudaSuccess)
        return result;
      result = cudaDeviceGetAttribute(&resources.optin_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      if (result != cudaSuccess)
        return result;
      entry = devices.emplace(device, resources).first;
    }
    auto &resources = entry->second;
    if (threads > resources.max_threads)
      return cudaErrorInvalidConfiguration;
    if (shared_bytes <= 48 * 1024)
      return shared_bytes <= resources.shared_bytes ? cudaSuccess : cudaErrorInvalidConfiguration;
    // Keep the native 96 KiB opt-in ceiling even when this invocation uses less.
    if (resources.optin_shared_bytes < kTileSharedBytes)
      return cudaErrorInvalidConfiguration;
    if (resources.reserved_kernels.count(kernel))
      return cudaSuccess;
    result = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kTileSharedBytes);
    if (result == cudaSuccess)
      resources.reserved_kernels.insert(kernel);
    return result;
  }

  // ---- Ordered extraction (from fused_extract) ----
  // One pass over the events that keeps the active-row list in ascending row order.
  // Blocks take chunk ids from a per-batch ticket (so every waited-on chunk is already
  // running) and find their output base by decoupled look-back over published chunk counts.
  // Workspace: status[2B] | tickets[B] | pad | flags[B * chunks] (uint64), zeroed per call.
  constexpr unsigned long long kFlagAggregate = 1ULL << 32;
  constexpr unsigned long long kFlagInclusive = 2ULL << 32;

  __host__ __device__ constexpr size_t ordered_flags_offset(int batch)
  {
    return (static_cast<size_t>(batch) * 3 + 1) / 2 * 2; // int32 slots, 8-byte aligned
  }

  __host__ __device__ constexpr size_t ordered_workspace_bytes(int rows, int batch)
  {
    return rows <= 0 || batch <= 0
               ? 0
               : ordered_flags_offset(batch) * sizeof(int32_t) +
                     static_cast<size_t>(batch) * row_chunks(rows) * sizeof(unsigned long long);
  }

  template <class EventT>
  __global__ void __launch_bounds__(kExtractBlockSize) ordered_extract_kernel(
      const EventT *__restrict__ spike_bn,
      int32_t *__restrict__ active_rows,
      int32_t *__restrict__ status,
      int32_t *__restrict__ tickets,
      unsigned long long *__restrict__ flags,
      int rows,
      int chunks)
  {
    constexpr int kWarps = kWarpsPerBlock;
    constexpr int kSegments = kSegmentsPerChunk;
    __shared__ int segment_offsets[kSegments];
    __shared__ int chunk_id;
    __shared__ int chunk_base;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int batch_col = blockIdx.y;
    if (threadIdx.x == 0)
      chunk_id = atomicAdd(&tickets[batch_col], 1);
    __syncthreads();
    const int chunk = chunk_id;
    const int64_t row_base = static_cast<int64_t>(chunk) * kRowsPerChunk;
    const EventT *spikes = spike_bn + static_cast<size_t>(batch_col) * rows;
    unsigned masks[kSegmentsPerWarp];
#pragma unroll
    for (int iteration = 0; iteration < kSegmentsPerWarp; ++iteration)
    {
      const int segment = iteration * kWarps + warp;
      const int64_t row = row_base + segment * 32 + lane;
      const unsigned mask = __ballot_sync(0xffffffffU, row < rows && cuda_traits::EventTraits<EventT>::active(spikes[row]));
      masks[iteration] = mask;
      if (lane == 0)
        segment_offsets[segment] = __popc(mask);
    }
    __syncthreads();
    if (warp == 0)
    {
      // Warp scan of the 128 segment counts (4 per lane).
      constexpr unsigned kFull = 0xffffffffU;
      constexpr int kPerLane = kSegments / 32;
      static_assert(kSegments % 32 == 0, "segments must fill the scanning warp");
      int counts[kPerLane];
      int sum = 0;
#pragma unroll
      for (int item = 0; item < kPerLane; ++item)
      {
        counts[item] = segment_offsets[lane * kPerLane + item];
        sum += counts[item];
      }
      int inclusive = sum;
#pragma unroll
      for (int distance = 1; distance < 32; distance <<= 1)
      {
        const int neighbor = __shfl_up_sync(kFull, inclusive, distance);
        if (lane >= distance)
          inclusive += neighbor;
      }
      int running = inclusive - sum;
#pragma unroll
      for (int item = 0; item < kPerLane; ++item)
      {
        segment_offsets[lane * kPerLane + item] = running;
        running += counts[item];
      }
      const int total = __shfl_sync(kFull, inclusive, 31);

      // Decoupled look-back, 32 predecessors per step: lane i reads chunk
      // (window - i); stop at the nearest inclusive prefix once every closer
      // predecessor has published at least its aggregate.
      unsigned long long *batch_flags = flags + static_cast<size_t>(batch_col) * chunks;
      int prefix = 0;
      if (chunk > 0)
      {
        if (lane == 0)
          atomicExch(&batch_flags[chunk], kFlagAggregate | static_cast<unsigned>(total));
        for (int window = chunk - 1;;)
        {
          const int previous = window - lane;
          const unsigned long long value =
              previous >= 0 ? atomicAdd(&batch_flags[previous], 0ULL) : kFlagInclusive;
          const unsigned long long flag = value & ~0xffffffffULL;
          const unsigned pending = __ballot_sync(kFull, flag == 0);
          const unsigned closed = __ballot_sync(kFull, flag == kFlagInclusive);
          const int stop = closed ? __ffs(closed) - 1 : 31;
          const unsigned needed = stop == 31 ? kFull : (1U << (stop + 1)) - 1U;
          if (pending & needed)
            continue; // some closer predecessor is still counting
          prefix += static_cast<int>(cuda_traits::warp_reduce_add_u32(
              lane <= stop ? static_cast<unsigned>(value & 0xffffffffULL) : 0U));
          if (closed)
            break;
          window -= 32;
        }
      }
      if (lane == 0)
      {
        atomicExch(&batch_flags[chunk], kFlagInclusive | static_cast<unsigned>(prefix + total));
        if (chunk == chunks - 1)
          status[static_cast<size_t>(batch_col) * 2 + kStatusCount] = prefix + total;
        chunk_base = prefix;
      }
    }
    __syncthreads();
    int32_t *selected = active_rows + static_cast<size_t>(batch_col) * rows;
    const unsigned lower_lanes = (1U << lane) - 1U;
#pragma unroll
    for (int iteration = 0; iteration < kSegmentsPerWarp; ++iteration)
    {
      const unsigned mask = masks[iteration];
      if ((mask & (1U << lane)) == 0)
        continue;
      const int segment = iteration * kWarps + warp;
      const int slot = chunk_base + segment_offsets[segment] + __popc(mask & lower_lanes);
      selected[slot] = static_cast<int32_t>(row_base + segment * 32 + lane);
    }
  }

  // memset + one ordered extraction kernel; leaves each batch's active count in status[2b].
  template <class EventT>
  cudaError_t extract_ordered(const EventT *spike_bn, int32_t *active_rows,
                              void *workspace, int rows, int batch, cudaStream_t stream)
  {
    auto *status = static_cast<int32_t *>(workspace);
    cudaError_t result = cudaMemsetAsync(workspace, 0, ordered_workspace_bytes(rows, batch), stream);
    if (result != cudaSuccess)
      return result;
    const int chunks = row_chunks(rows);
    auto *flags = reinterpret_cast<unsigned long long *>(status + ordered_flags_offset(batch));
    ordered_extract_kernel<EventT><<<dim3(chunks, batch), kExtractBlockSize, 0, stream>>>(
        spike_bn, active_rows, status, status + static_cast<size_t>(batch) * 2, flags, rows, chunks);
    return cudaGetLastError();
  }

  // ---- Absolute-offset tile scatter (from short_chain, kAbsolute = true) ----
  // Each (tile, batch) block zeroes a shared tile, adds the active rows' entries of that
  // tile and writes the tile back, overwriting the output. tile_offsets hold absolute edge
  // indices (OffsetT), so the scatter never reads row_ptr; every load of the stage loop
  // depends only on the active row index.
  template <bool Permuted, class ValueT, class OffsetT>
  __device__ __forceinline__ typename cuda_traits::ValueTraits<ValueT>::AccT read_edge_weight(
      const ValueT *values, const OffsetT *perm, int64_t entry)
  {
    if constexpr (Permuted)
      return cuda_traits::ValueTraits<ValueT>::read(values[perm[entry]]);
    else
      return cuda_traits::ValueTraits<ValueT>::read(values[entry]);
  }

  template <bool Homo, bool Permuted, bool Weighted, typename ValueT, typename EventT,
            typename OffsetT, int kBlockSize>
  __global__ void __launch_bounds__(kBlockSize) scatter_kernel(
      const ValueT *__restrict__ values,
      const uint16_t *__restrict__ local_targets,
      const OffsetT *__restrict__ tile_offsets,
      const OffsetT *__restrict__ perm,
      const EventT *__restrict__ events,
      const int32_t *__restrict__ active_rows,
      const int32_t *__restrict__ status,
      ValueT *__restrict__ output_bn,
      int rows,
      int cols,
      int tile_width,
      int tile_count)
  {
    const int tile = blockIdx.x;
    // Padding CTAs must return before any memory access or synchronization.
    if (tile >= tile_count) return;
    constexpr int kGroupSize = kBlockSize / 64;
    constexpr int kGroups = kBlockSize / kGroupSize;
    constexpr int kDepth = kPipelineDepth;
    using Traits = cuda_traits::ValueTraits<ValueT>;
    using AccT = typename Traits::AccT;
    AccT shared_weight{};
    if constexpr (Homo) shared_weight = Traits::read(values[0]);
    extern __shared__ __align__(16) unsigned char tile_memory[];
    AccT *tile_output = reinterpret_cast<AccT *>(tile_memory);
    const int batch_col = blockIdx.y;
    const int64_t tile_begin = static_cast<int64_t>(tile) * tile_width;
    const int tile_elements = static_cast<int>(min(static_cast<int64_t>(tile_width), static_cast<int64_t>(cols) - tile_begin));
    for (int local = threadIdx.x; local < tile_width; local += kBlockSize)
      tile_output[local] = 0.0F;
    __syncthreads();

    const int active_count = min(status[static_cast<size_t>(batch_col) * 2 + kStatusCount], rows);
    const int32_t *selected = active_rows + static_cast<size_t>(batch_col) * rows;
    const int group = threadIdx.x / kGroupSize;
    const int lane = threadIdx.x & (kGroupSize - 1);
    const size_t boundaries = static_cast<size_t>(tile_count) + 1;
    for (int64_t base = group; base < active_count; base += kGroups * kDepth)
    {
      // Stage loop: every load here depends only on the active row index.
      int64_t begins[kDepth];
      int64_t ends[kDepth];
      AccT scales[kDepth]; // Eliminated for binary specializations.
#pragma unroll
      for (int stage = 0; stage < kDepth; ++stage)
      {
        const int64_t index = base + stage * kGroups;
        begins[stage] = 0;
        ends[stage] = 0;
        if (index < active_count)
        {
          const int row = selected[index];
          const size_t offset = static_cast<size_t>(row) * boundaries + tile;
          begins[stage] = static_cast<int64_t>(tile_offsets[offset]);
          ends[stage] = static_cast<int64_t>(tile_offsets[offset + 1]);
          if constexpr (Weighted)
            scales[stage] = Traits::read(events[static_cast<size_t>(batch_col) * rows + row]);
        }
      }
      // Direct weights share the targets' load level; permuted weights add one indirection.
#pragma unroll
      for (int stage = 0; stage < kDepth; ++stage)
      {
        for (int64_t entry = begins[stage] + lane; entry < ends[stage]; entry += kGroupSize)
        {
          AccT weight = Homo ? shared_weight : read_edge_weight<Permuted>(values, perm, entry);
          if constexpr (Weighted) weight *= scales[stage];
          atomicAdd(&tile_output[local_targets[entry]], weight);
        }
      }
    }
    __syncthreads();
    ValueT *output = output_bn + static_cast<size_t>(batch_col) * cols;
    for (int local = threadIdx.x; local < tile_elements; local += kBlockSize)
      output[tile_begin + local] = Traits::write(tile_output[local]);
  }

  template <int kBlockSize, bool Homo, bool Permuted, bool Weighted,
            class ValueT, class EventT, class OffsetT>
  cudaError_t launch_scatter(const ValueT *values, const uint16_t *local_targets,
                            const OffsetT *tile_offsets, const OffsetT *perm, const EventT *events,
                            ValueT *output, const int32_t *active_rows, const void *workspace,
                            const Plan &plan, const TileSchedule &schedule, cudaStream_t stream)
  {
    auto kernel = scatter_kernel<Homo, Permuted, Weighted, ValueT, EventT, OffsetT, kBlockSize>;
    const cudaError_t result = prepare_scatter_kernel(
        reinterpret_cast<const void *>(kernel), kBlockSize, schedule.shared_bytes);
    if (result != cudaSuccess) return result;
    kernel<<<dim3(schedule.grid_tiles, plan.batch), kBlockSize, schedule.shared_bytes, stream>>>(
        values, local_targets, tile_offsets, perm, events, active_rows,
        static_cast<const int32_t *>(workspace), output, plan.rows, plan.cols,
        schedule.tile_width, schedule.tile_count);
    return cudaGetLastError();
  }

  template <bool Homo, class ValueT, class EventT, class OffsetT, bool Permuted = false, bool Weighted = false>
  cudaError_t launch_impl(const ValueT *values, const uint16_t *local_targets,
                     const OffsetT *tile_offsets, const EventT *events, ValueT *output,
                     int32_t *active_rows, void *workspace, const Plan &plan, cudaStream_t stream,
                     const OffsetT *perm = nullptr)
  {
    static_assert(std::is_same_v<OffsetT, int32_t> || std::is_same_v<OffsetT, int64_t>);
    static_assert(!Homo || !Permuted, "permuted weights are heterogeneous only");
    static_assert(!Weighted || std::is_same_v<ValueT, EventT>, "weighted events must match values");
    const int rows = plan.rows, cols = plan.cols, batch = plan.batch;
    Plan checked;
    if (make_plan(rows, cols, batch, plan.nnz, plan.value_type, &checked) != cudaSuccess ||
        plan.value_type != cuda_traits::ValueTraits<ValueT>::type || plan.tile_width != checked.tile_width ||
        (sizeof(OffsetT) == 4 && plan.nnz > INT32_MAX))
      return cudaErrorInvalidValue;
    if (batch == 0 || cols == 0)
      return cudaSuccess;
    if (!output)
      return cudaErrorInvalidValue;
    if (rows == 0 || plan.nnz == 0)
      return cudaMemsetAsync(output, 0, static_cast<size_t>(batch) * cols * sizeof(ValueT), stream);
    if (!values || !local_targets || !tile_offsets || (Permuted && !perm) || !events || !active_rows || !workspace)
      return cudaErrorInvalidValue;
    cudaError_t result = extract_ordered(events, active_rows, workspace, rows, batch, stream);
    if (result != cudaSuccess)
      return result;
    const auto schedule = make_tile_schedule(cols, batch, sizeof(typename cuda_traits::ValueTraits<ValueT>::AccT));
#define COMBINED_SCATTER_ARGS values, local_targets, tile_offsets, perm, events, output, active_rows, workspace, checked, schedule, stream
    if (schedule.block_size == 256)
      return launch_scatter<256, Homo, Permuted, Weighted>(COMBINED_SCATTER_ARGS);
    if (schedule.block_size == 512)
      return launch_scatter<512, Homo, Permuted, Weighted>(COMBINED_SCATTER_ARGS);
    return launch_scatter<1024, Homo, Permuted, Weighted>(COMBINED_SCATTER_ARGS);
#undef COMBINED_SCATTER_ARGS
  }
} // namespace combined
#endif
