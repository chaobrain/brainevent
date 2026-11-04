// Legacy binary-only implementation for the standalone GPU-native benchmark.
// This header preserves the bool-event hybrid scheduling strategy while removing
// the XLA FFI and BrainEvent tensor wrappers. The current binary/weighted FFI
// operators share hybrid_spmv_impl.cuh instead of this benchmark header.

#ifndef HYBRID_SPMV_KERNELS_CUH_
#define HYBRID_SPMV_KERNELS_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace gpu_generated_benchmark
{

  constexpr int kHybridBlockSize = 256;
  constexpr int kHybridFixedScatterBlocks = 2048;
  constexpr int kHybridTprThreshold = 128;
  constexpr int kHybridTaskNnz = 4096;
  constexpr int kHybridStatusCountIndex = 0;
  constexpr int kHybridStatusOverflowIndex = 1;
  constexpr int64_t kHybridMaxTaskCount = 2147483647LL;

  template <typename ValueT, typename OffsetT>
  __device__ __forceinline__ void hybrid_scatter_range(
      const ValueT *__restrict__ weights, const int32_t *__restrict__ indices,
      ValueT *__restrict__ output, OffsetT begin, OffsetT end)
  {
    for (OffsetT entry = begin + static_cast<OffsetT>(threadIdx.x); entry < end;
         entry += static_cast<OffsetT>(blockDim.x))
    {
      atomicAdd(&output[indices[entry]], weights[entry]);
    }
  }

  template <typename ValueT, typename OffsetT>
  __device__ __forceinline__ void hybrid_scatter_range_thread(
      const ValueT *__restrict__ weights, const int32_t *__restrict__ indices,
      ValueT *__restrict__ output, OffsetT begin, OffsetT end)
  {
    for (OffsetT entry = begin; entry < end; ++entry)
    {
      atomicAdd(&output[indices[entry]], weights[entry]);
    }
  }

  template <typename ValueT, typename OffsetT>
  __global__ void hybrid_extract_kernel(
      const ValueT *__restrict__ weights, const int32_t *__restrict__ indices,
      const OffsetT *__restrict__ indptr, const int8_t *__restrict__ events,
      ValueT *__restrict__ output, OffsetT *__restrict__ task_begin,
      OffsetT *__restrict__ task_end, int32_t *__restrict__ status, int rows,
      int task_capacity)
  {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || events[row] == 0)
      return;

    const OffsetT begin = indptr[row];
    const OffsetT end = indptr[row + 1];
    const OffsetT row_nnz = end - begin;
    if (row_nnz <= kHybridTprThreshold)
    {
      hybrid_scatter_range_thread<ValueT, OffsetT>(weights, indices, output,
                                                   begin, end);
      return;
    }

    const int64_t chunks64 =
        (static_cast<int64_t>(row_nnz) + kHybridTaskNnz - 1) / kHybridTaskNnz;
    if (chunks64 > kHybridMaxTaskCount)
    {
      atomicExch(&status[kHybridStatusOverflowIndex], 1);
      return;
    }
    const int32_t chunks = static_cast<int32_t>(chunks64);
    const int32_t base = atomicAdd(&status[kHybridStatusCountIndex], chunks);
    if (base < 0 || base > task_capacity || chunks > task_capacity - base)
    {
      atomicExch(&status[kHybridStatusOverflowIndex], 1);
      return;
    }
    for (int32_t chunk = 0; chunk < chunks; ++chunk)
    {
      const int32_t task = base + chunk;
      const OffsetT task_begin_offset =
          begin + static_cast<OffsetT>(chunk) * kHybridTaskNnz;
      task_begin[task] = task_begin_offset;
      const OffsetT task_end_offset = task_begin_offset + kHybridTaskNnz;
      task_end[task] = task_end_offset < end ? task_end_offset : end;
    }
  }

  template <typename ValueT, typename OffsetT>
  __global__ void hybrid_block_kernel(
      const ValueT *__restrict__ weights, const int32_t *__restrict__ indices,
      const OffsetT *__restrict__ task_begin,
      const OffsetT *__restrict__ task_end,
      const int32_t *__restrict__ status, int task_capacity,
      ValueT *__restrict__ output)
  {
    int active = status[kHybridStatusCountIndex];
    if (active > task_capacity)
      active = task_capacity;
    for (int task = blockIdx.x; task < active; task += gridDim.x)
    {
      hybrid_scatter_range<ValueT, OffsetT>(weights, indices, output,
                                            task_begin[task], task_end[task]);
    }
  }

  template <typename ValueT, typename OffsetT>
  inline cudaError_t launch_hybrid(
      const ValueT *weights, const int32_t *indices, const OffsetT *indptr,
      const int8_t *events, ValueT *output, OffsetT *task_begin,
      OffsetT *task_end, int32_t *status, int rows, int task_capacity,
      cudaStream_t stream)
  {
    if (rows <= 0)
      return cudaSuccess;
    const int extract_blocks = (rows + kHybridBlockSize - 1) / kHybridBlockSize;
    hybrid_extract_kernel<ValueT, OffsetT>
        <<<extract_blocks, kHybridBlockSize, 0, stream>>>(
            weights, indices, indptr, events, output, task_begin, task_end,
            status, rows, task_capacity);
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess || task_capacity == 0)
      return result;

    const int scatter_blocks = task_capacity < kHybridFixedScatterBlocks
                                   ? task_capacity
                                   : kHybridFixedScatterBlocks;
    hybrid_block_kernel<ValueT, OffsetT>
        <<<scatter_blocks, kHybridBlockSize, 0, stream>>>(
            weights, indices, task_begin, task_end, status, task_capacity,
            output);
    return cudaGetLastError();
  }

} // namespace gpu_generated_benchmark

#endif // HYBRID_SPMV_KERNELS_CUH_
