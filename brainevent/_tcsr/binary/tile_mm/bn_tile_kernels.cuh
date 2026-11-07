#ifndef BN_TILE_KERNELS_CUH_
#define BN_TILE_KERNELS_CUH_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace bn_tile
{

  constexpr int kTileSize = 8192;
  constexpr int kSharedBytes = 32 * 1024;
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

  __host__ __device__ constexpr size_t active_rows_bytes(int rows, int batch)
  {
    return rows <= 0 || batch <= 0
               ? 0
               : static_cast<size_t>(rows) * batch * sizeof(int32_t);
  }

  __host__ __device__ constexpr size_t workspace_bytes(int rows, int batch)
  {
    return rows <= 0 || batch <= 0
               ? 0
               : static_cast<size_t>(batch) * (2 + row_chunks(rows)) * sizeof(int32_t);
  }

  __global__ void count_active_rows_kernel(
      const int8_t *__restrict__ spike_bn,
      int32_t *__restrict__ chunk_counts,
      int rows,
      int chunks)
  {
    __shared__ int warp_counts[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int batch_col = blockIdx.y;
    const int64_t row_base = static_cast<int64_t>(blockIdx.x) * kRowsPerChunk;
    int count = 0;
#pragma unroll
    for (int iteration = 0; iteration < kSegmentsPerWarp; ++iteration)
    {
      const int segment = iteration * kWarpsPerBlock + warp;
      const int64_t row = row_base + segment * 32 + lane;
      const bool active = row < rows &&
                          spike_bn[static_cast<size_t>(batch_col) * rows + row] != 0;
      const unsigned mask = __ballot_sync(__activemask(), active);
      if (lane == 0)
        count += __popc(mask);
    }
    if (lane == 0)
      warp_counts[warp] = count;
    __syncthreads();
    if (threadIdx.x == 0)
    {
      int total = 0;
#pragma unroll
      for (int index = 0; index < kWarpsPerBlock; ++index)
      {
        total += warp_counts[index];
      }
      chunk_counts[static_cast<size_t>(batch_col) * chunks + blockIdx.x] = total;
    }
  }

  __global__ void scan_active_row_counts_kernel(
      int32_t *__restrict__ chunk_offsets,
      int32_t *__restrict__ status,
      int chunks)
  {
    if (threadIdx.x != 0)
      return;
    const int batch_col = blockIdx.x;
    int running = 0;
    int32_t *offsets = chunk_offsets + static_cast<size_t>(batch_col) * chunks;
    for (int chunk = 0; chunk < chunks; ++chunk)
    {
      const int count = offsets[chunk];
      offsets[chunk] = running;
      running += count;
    }
    status[static_cast<size_t>(batch_col) * 2 + kStatusCount] = running;
  }

  __global__ void write_active_rows_kernel(
      const int8_t *__restrict__ spike_bn,
      int32_t *__restrict__ active_rows,
      const int32_t *__restrict__ chunk_offsets,
      int rows,
      int chunks)
  {
    __shared__ int segment_offsets[kSegmentsPerChunk];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int batch_col = blockIdx.y;
    const int64_t row_base = static_cast<int64_t>(blockIdx.x) * kRowsPerChunk;
    unsigned masks[kSegmentsPerWarp];
#pragma unroll
    for (int iteration = 0; iteration < kSegmentsPerWarp; ++iteration)
    {
      const int segment = iteration * kWarpsPerBlock + warp;
      const int64_t row = row_base + segment * 32 + lane;
      const bool active = row < rows &&
                          spike_bn[static_cast<size_t>(batch_col) * rows + row] != 0;
      const unsigned mask = __ballot_sync(__activemask(), active);
      masks[iteration] = mask;
      if (lane == 0)
        segment_offsets[segment] = __popc(mask);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      int running = 0;
#pragma unroll
      for (int segment = 0; segment < kSegmentsPerChunk; ++segment)
      {
        const int count = segment_offsets[segment];
        segment_offsets[segment] = running;
        running += count;
      }
    }
    __syncthreads();
    const int chunk_base = chunk_offsets[static_cast<size_t>(batch_col) * chunks + blockIdx.x];
    const unsigned lower_lanes = lane == 0 ? 0U : (1U << lane) - 1U;
#pragma unroll
    for (int iteration = 0; iteration < kSegmentsPerWarp; ++iteration)
    {
      const unsigned mask = masks[iteration];
      if ((mask & (1U << lane)) == 0)
        continue;
      const int segment = iteration * kWarpsPerBlock + warp;
      const int slot = chunk_base + segment_offsets[segment] +
                       __popc(mask & lower_lanes);
      const int64_t row = row_base + segment * 32 + lane;
      active_rows[static_cast<size_t>(batch_col) * rows + slot] =
          static_cast<int32_t>(row);
    }
  }

  template <typename ValueT, bool Homogeneous>
  __global__ void scatter_kernel(
      const ValueT *__restrict__ values,
      const uint16_t *__restrict__ local_targets,
      const int64_t *__restrict__ row_ptr,
      const int32_t *__restrict__ tile_offsets,
      const int32_t *__restrict__ active_rows,
      const int32_t *__restrict__ status,
      ValueT *__restrict__ output_bn,
      int rows,
      int cols,
      int metadata_tile_count)
  {
    constexpr int kBlockSize = 512;
    constexpr int kGroupSize = 16;
    constexpr int kPipelineDepth = 4;
    constexpr int kGroups = kBlockSize / kGroupSize;
    constexpr int kComputeTileSize = kSharedBytes / sizeof(ValueT);
    constexpr int kSubtilesPerMetadata = kTileSize / kComputeTileSize;
    static_assert(
        kTileSize % kComputeTileSize == 0,
        "compute tile must divide the metadata tile");
    static_assert(
        kSubtilesPerMetadata == 1 || kSubtilesPerMetadata == 2,
        "TileMM supports only float32 and float64 values");
    extern __shared__ unsigned char tile_output_bytes[];
    ValueT *tile_output = reinterpret_cast<ValueT *>(tile_output_bytes);
    const int batch_col = blockIdx.y;
    const int compute_tile = blockIdx.x;
    const int metadata_tile = compute_tile / kSubtilesPerMetadata;
    const int metadata_subtile = compute_tile % kSubtilesPerMetadata;
    const int local_tile_begin = metadata_subtile * kComputeTileSize;
    const int tile_begin = compute_tile * kComputeTileSize;
    const int tile_elements = min(kComputeTileSize, cols - tile_begin);
    for (int local = threadIdx.x; local < kComputeTileSize;
         local += kBlockSize)
    {
      tile_output[local] = ValueT(0);
    }
    __syncthreads();
    const int active_count = min(
        status[static_cast<size_t>(batch_col) * 2 + kStatusCount], rows);
    const int32_t *selected = active_rows + static_cast<size_t>(batch_col) * rows;
    const int group = threadIdx.x / kGroupSize;
    const int lane = threadIdx.x & (kGroupSize - 1);
    const int boundaries = metadata_tile_count + 1;
    ValueT homogeneous_value = ValueT(0);
    if (Homogeneous)
    {
      homogeneous_value = values[0];
    }
    for (int base = group; base < active_count;
         base += kGroups * kPipelineDepth)
    {
      int descriptor_rows[kPipelineDepth];
      int begins[kPipelineDepth];
      int ends[kPipelineDepth];
#pragma unroll
      for (int stage = 0; stage < kPipelineDepth; ++stage)
      {
        const int index = base + stage * kGroups;
        if (index < active_count)
        {
          const int row = selected[index];
          const size_t offset =
              static_cast<size_t>(row) * boundaries + metadata_tile;
          descriptor_rows[stage] = row;
          int begin = tile_offsets[offset];
          int end = tile_offsets[offset + 1];
          if (kSubtilesPerMetadata == 2)
          {
            int midpoint = begin;
            if (lane == 0)
            {
              int low = begin;
              int high = end;
              const int64_t row_begin = row_ptr[row];
              while (low < high)
              {
                const int middle = low + (high - low) / 2;
                if (local_targets[row_begin + middle] < kComputeTileSize)
                  low = middle + 1;
                else
                  high = middle;
              }
              midpoint = low;
            }
            midpoint = __shfl_sync(
                __activemask(), midpoint, 0, kGroupSize);
            if (metadata_subtile == 0)
              end = midpoint;
            else
              begin = midpoint;
          }
          begins[stage] = begin;
          ends[stage] = end;
        }
        else
        {
          descriptor_rows[stage] = -1;
          begins[stage] = 0;
          ends[stage] = 0;
        }
      }
#pragma unroll
      for (int stage = 0; stage < kPipelineDepth; ++stage)
      {
        const int row = descriptor_rows[stage];
        if (row < 0)
          continue;
        const int64_t row_begin = row_ptr[row];
        for (int offset = begins[stage] + lane; offset < ends[stage];
             offset += kGroupSize)
        {
          const int64_t entry = row_begin + offset;
          const int local =
              static_cast<int>(local_targets[entry]) - local_tile_begin;
          const ValueT value = Homogeneous ? homogeneous_value : values[entry];
          atomicAdd(&tile_output[local], value);
        }
      }
    }
    __syncthreads();
    ValueT *output = output_bn + static_cast<size_t>(batch_col) * cols;
    for (int local = threadIdx.x; local < tile_elements; local += kBlockSize)
    {
      output[tile_begin + local] = tile_output[local];
    }
  }

  template <typename ValueT, bool Homogeneous>
  inline cudaError_t launch_impl(
      const ValueT *values,
      const uint16_t *local_targets,
      const int64_t *row_ptr,
      const int32_t *tile_offsets,
      const int8_t *spike_bn,
      ValueT *output_bn,
      int32_t *active_rows,
      void *workspace,
      int rows,
      int cols,
      int batch,
      cudaStream_t stream)
  {
    if (!values || !local_targets || !row_ptr || !tile_offsets || !spike_bn ||
        !output_bn || !active_rows || !workspace || rows <= 0 || cols <= 0 ||
        batch <= 0 || batch > 65535)
    {
      return cudaErrorInvalidValue;
    }
    auto *status = static_cast<int32_t *>(workspace);
    cudaError_t result = cudaMemsetAsync(
        status, 0, static_cast<size_t>(batch) * 2 * sizeof(int32_t), stream);
    if (result != cudaSuccess)
      return result;
    const int chunks = row_chunks(rows);
    auto *offsets = status + static_cast<size_t>(batch) * 2;
    const dim3 extract_grid(chunks, batch);
    count_active_rows_kernel<<<extract_grid, kExtractBlockSize, 0, stream>>>(
        spike_bn, offsets, rows, chunks);
    result = cudaGetLastError();
    if (result != cudaSuccess)
      return result;
    scan_active_row_counts_kernel<<<batch, 1, 0, stream>>>(
        offsets, status, chunks);
    result = cudaGetLastError();
    if (result != cudaSuccess)
      return result;
    write_active_rows_kernel<<<extract_grid, kExtractBlockSize, 0, stream>>>(
        spike_bn, active_rows, offsets, rows, chunks);
    result = cudaGetLastError();
    if (result != cudaSuccess)
      return result;
    result = cudaFuncSetAttribute(
        scatter_kernel<ValueT, Homogeneous>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes);
    if (result != cudaSuccess)
      return result;
    constexpr int kComputeTileSize = kSharedBytes / sizeof(ValueT);
    const int metadata_tile_count = (cols + kTileSize - 1) / kTileSize;
    const int compute_tile_count =
        (cols + kComputeTileSize - 1) / kComputeTileSize;
    scatter_kernel<ValueT, Homogeneous>
        <<<dim3(compute_tile_count, batch), 512, kSharedBytes, stream>>>(
            values, local_targets, row_ptr, tile_offsets, active_rows, status,
            output_bn, rows, cols, metadata_tile_count);
    return cudaGetLastError();
  }

  template <typename ValueT>
  inline cudaError_t launch(
      const ValueT *values,
      const uint16_t *local_targets,
      const int64_t *row_ptr,
      const int32_t *tile_offsets,
      const int8_t *spike_bn,
      ValueT *output_bn,
      int32_t *active_rows,
      void *workspace,
      int rows,
      int cols,
      int batch,
      cudaStream_t stream)
  {
    return launch_impl<ValueT, false>(
        values, local_targets, row_ptr, tile_offsets, spike_bn, output_bn,
        active_rows, workspace, rows, cols, batch, stream);
  }

  template <typename ValueT>
  inline cudaError_t launch_homo(
      const ValueT *values,
      const uint16_t *local_targets,
      const int64_t *row_ptr,
      const int32_t *tile_offsets,
      const int8_t *spike_bn,
      ValueT *output_bn,
      int32_t *active_rows,
      void *workspace,
      int rows,
      int cols,
      int batch,
      cudaStream_t stream)
  {
    return launch_impl<ValueT, true>(
        values, local_targets, row_ptr, tile_offsets, spike_bn, output_bn,
        active_rows, workspace, rows, cols, batch, stream);
  }

} // namespace bn_tile

#endif
