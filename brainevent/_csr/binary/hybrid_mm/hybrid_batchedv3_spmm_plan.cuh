// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef HYBRID_BATCHEDV3_SPMM_PLAN_CUH
#define HYBRID_BATCHEDV3_SPMM_PLAN_CUH
// Geometry constants, host planning and scratch sizing shared by the binary and weighted operators.
// Everything is inline so hybrid_batchedv3_spmm_binary.cu and hybrid_batchedv3_spmm_weighted.cu link into one program.
// Included by hybrid_batchedv3_spmm.cuh after Plan/Device; plain C++ compilers can use it.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hybrid
{
  // From bn_tile: K1 extraction geometry and the per-batch status layout.
  constexpr int kExtractBlockSize = 256;
  constexpr int kWarpsPerBlock = kExtractBlockSize / 32;
  constexpr int kSegmentsPerWarp = 16;
  constexpr int kSegmentsPerChunk = kWarpsPerBlock * kSegmentsPerWarp;
  constexpr int kRowsPerChunk = 32 * kSegmentsPerChunk;
  constexpr int kStatusCount = 0;
  constexpr int kStatusOverflow = 1;

  __host__ __device__ constexpr int row_chunks(int rows)
  {
    return rows <= 0 ? 0 : static_cast<int>((static_cast<int64_t>(rows) + kRowsPerChunk - 1) / kRowsPerChunk);
  }

  // From batched_mv2: short-row scatter geometry, lane groups and L2 batch chunks.
  constexpr int kShortBlockSize = 256;
  constexpr int kBlocksPerSm = 6; // 1536 threads per SM on sm_120
#ifndef BATCHED_MV2_L2_DIVISOR
#define BATCHED_MV2_L2_DIVISOR 4 // chunk sweep, GPU 1: /4 and /8 beat /2 by 20-35%
#endif
  constexpr int kL2Divisor = BATCHED_MV2_L2_DIVISOR;

  // Lanes per event: the smallest power of two >= mean row length / 4, in [4, 32].
  constexpr int group_size(int64_t nnz, int rows)
  {
    const int64_t quarter = rows > 0 ? (nnz / rows + 3) / 4 : 1;
    return quarter <= 4 ? 4 : quarter <= 8 ? 8
                          : quarter <= 16  ? 16
                                           : 32;
  }

  // Batches per chunk so that one chunk's output (cols x value_bytes each) uses at most L2 / kL2Divisor.
  constexpr int chunk_batches(int cols, int batch, int l2_bytes, int value_bytes)
  {
    const int64_t per_batch = static_cast<int64_t>(cols) * value_bytes;
    const int64_t fit = per_batch > 0 ? (static_cast<int64_t>(l2_bytes) / kL2Divisor) / per_batch : batch;
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(fit, batch)));
  }

  // SM count and L2 size of the current device, cached per device ordinal (one cache per program).
  inline cudaError_t device_info(Device *info)
  {
    constexpr int kMaxDevices = 64;
    static Device cached[kMaxDevices];
    int device = 0;
    cudaError_t result = cudaGetDevice(&device);
    if (result != cudaSuccess)
      return result;
    Device local = device < kMaxDevices ? cached[device] : Device{};
    if (local.sms == 0)
    {
      if ((result = cudaDeviceGetAttribute(&local.sms, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess ||
          (result = cudaDeviceGetAttribute(&local.l2_bytes, cudaDevAttrL2CacheSize, device)) != cudaSuccess)
        return result;
      if (device < kMaxDevices)
        cached[device] = local; // benign race: every writer stores the same values
    }
    *info = local;
    return cudaSuccess;
  }

  constexpr int kBlockSize = 256;
#ifndef HYBRID_ITERATIONS
#define HYBRID_ITERATIONS 16 // per-work-item iteration cap K (K sweep: see dev/md/README.md)
#endif
  constexpr int kIterations = HYBRID_ITERATIONS;
  constexpr int kTaskEdges = 32 * kIterations;
  constexpr int kMetaLenBits = 12; // task length <= kTaskEdges < 2^kMetaLenBits (K <= 127)
  static_assert(kTaskEdges < (1 << kMetaLenBits), "task length must fit the meta field");

  // Plan from a host copy of row_ptr. Returns cudaErrorInvalidValue for invalid sizes.
  template <typename OffsetT>
  inline cudaError_t make_plan(const OffsetT *row_ptr, int rows, int cols, int batch, int l2_bytes,
                               int value_bytes, Plan *plan)
  {
    if (!row_ptr || !plan || rows < 0 || cols < 0 || batch < 0 || l2_bytes <= 0 || value_bytes <= 0 ||
        row_ptr[0] != 0 || ((rows == 0 || cols == 0) && row_ptr[rows] != 0))
      return cudaErrorInvalidValue;
    if (rows == 0 || cols == 0 || batch == 0)
    {
      *plan = Plan{};
      plan->nnz = row_ptr[rows];
      return cudaSuccess;
    }
    int64_t short_rows = 0, short_edges = 0;
    for (int r = 0; r < rows; ++r)
    {
      const int64_t len = row_ptr[r + 1] - row_ptr[r];
      if (len < 0)
        return cudaErrorInvalidValue;
      if (len > 0 && len <= kTaskEdges)
      {
        ++short_rows;
        short_edges += len;
      }
    }
    Plan p;
    p.nnz = row_ptr[rows];
    p.group = short_rows ? group_size(short_edges, static_cast<int>(short_rows)) : 32;
    p.long_threshold = static_cast<int64_t>(kIterations) * p.group;
    for (int r = 0; r < rows; ++r)
    {
      const int64_t len = row_ptr[r + 1] - row_ptr[r];
      if (len > p.long_threshold)
        p.long_tasks += (len + kTaskEdges - 1) / kTaskEdges;
    }
    p.chunk = std::min(chunk_batches(cols, batch, l2_bytes, value_bytes), kMaxGroupBatches);
    p.task_capacity = p.chunk * p.long_tasks;
    *plan = p;
    return cudaSuccess;
  }

  // Bit r % 32 of word r / 32 is set when row r is longer than plan.long_threshold (static per matrix).
  template <typename OffsetT>
  inline std::vector<uint32_t> long_row_mask(const OffsetT *row_ptr, int rows, const Plan &plan)
  {
    std::vector<uint32_t> mask((static_cast<size_t>(rows) + 31) / 32, 0U);
    for (int r = 0; r < rows; ++r)
      if (static_cast<int64_t>(row_ptr[r + 1]) - row_ptr[r] > plan.long_threshold)
        mask[r / 32] |= 1U << (r % 32);
    return mask;
  }

  // Workspace: status int32[2B] (batched_mv2 layout) | pad | counters uint64[groups] |
  // task_begin int64[cap] | task_value[cap] (padded to 4 bytes) | task_meta int32[cap]. task_value is
  // the weighted task value queue (value_bytes = sizeof(ValueT)); binary has none (value_bytes = 0).
  // It follows task_begin, so it is 8-byte aligned for double at any capacity, and its padding keeps
  // task_meta 4-byte aligned after an odd count of half values.
  __host__ __device__ constexpr size_t counters_offset(int batch)
  {
    return (static_cast<size_t>(batch) * 2 * sizeof(int32_t) + 7) / 8 * 8;
  }

  __host__ __device__ constexpr int group_count(int batch, int chunk)
  {
    return (batch + chunk - 1) / chunk;
  }

  __host__ __device__ constexpr size_t queue_region_bytes(int64_t task_capacity, size_t value_bytes)
  {
    return (static_cast<size_t>(task_capacity) * value_bytes + 3) / 4 * 4;
  }

  inline size_t workspace_bytes(const Plan &plan, int batch, size_t value_bytes = 0)
  {
    return counters_offset(batch) + group_count(batch, plan.chunk) * sizeof(unsigned long long) +
           static_cast<size_t>(plan.task_capacity) * (sizeof(int64_t) + sizeof(int32_t)) +
           queue_region_bytes(plan.task_capacity, value_bytes);
  }
} // namespace hybrid

#endif
