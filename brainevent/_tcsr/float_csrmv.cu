// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

/*
 * Canonical TCSR float matrix-vector kernels.
 *
 * x @ W uses one warp per canonical sparse row and scatters directly.
 * W @ x uses one block per (4096-row chunk, input compute tile). A 512-thread
 * block caches one dense tile and uses 32 groups of 16 threads to reduce rows.
 */

#include <cstddef>
#include <cstdint>

#include "cuda_common.h"
#include "brainevent/common.h"

namespace {

constexpr int kMetadataTileSize = 8192;
constexpr int kSharedBytes = 32 * 1024;
constexpr int kBlockSize = 512;
constexpr int kGroupSize = 16;
constexpr int kGroupsPerBlock = kBlockSize / kGroupSize;
constexpr int kRowsPerChunk = 4096;
constexpr int kWarpBlockSize = 256;

template <typename ValueT>
__device__ __forceinline__ ValueT group_reduce_sum(ValueT value) {
#pragma unroll
  for (int offset = kGroupSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(__activemask(), value, offset, kGroupSize);
  }
  return value;
}

template <typename ValueT, bool Homogeneous>
__global__ void xw_wpr_kernel(
    const ValueT* __restrict__ values,
    const int32_t* __restrict__ indices,
    const int64_t* __restrict__ row_ptr,
    const ValueT* __restrict__ x,
    ValueT* __restrict__ y,
    int rows) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x / 32) + warp;
  if (row >= rows) return;

  const ValueT dense_value = x[row];
  const ValueT homogeneous_value = Homogeneous ? values[0] : ValueT(0);
  const int64_t begin = row_ptr[row];
  const int64_t end = row_ptr[row + 1];
  for (int64_t entry = begin + lane; entry < end; entry += 32) {
    const ValueT weight = Homogeneous ? homogeneous_value : values[entry];
    atomicAdd(&y[indices[entry]], weight * dense_value);
  }
}

template <typename ValueT, bool Homogeneous>
__global__ void wx_tile_kernel(
    const ValueT* __restrict__ values,
    const int64_t* __restrict__ row_ptr,
    const uint16_t* __restrict__ local_targets,
    const int32_t* __restrict__ tile_offsets,
    const ValueT* __restrict__ x,
    ValueT* __restrict__ y,
    int rows,
    int cols,
    int metadata_tile_count) {
  constexpr int kComputeTileSize = kSharedBytes / sizeof(ValueT);
  constexpr int kSubtilesPerMetadata = kMetadataTileSize / kComputeTileSize;
  static_assert(kSubtilesPerMetadata == 1 || kSubtilesPerMetadata == 2);

  __shared__ ValueT x_tile[kComputeTileSize];
  const int row_chunk = blockIdx.x;
  const int compute_tile = blockIdx.y;
  const int metadata_tile = compute_tile / kSubtilesPerMetadata;
  const int metadata_subtile = compute_tile % kSubtilesPerMetadata;
  const int local_tile_begin = metadata_subtile * kComputeTileSize;
  const int dense_tile_begin = compute_tile * kComputeTileSize;
  for (int local = threadIdx.x; local < kComputeTileSize; local += kBlockSize) {
    const int col = dense_tile_begin + local;
    x_tile[local] = col < cols ? x[col] : ValueT(0);
  }
  __syncthreads();

  const int group = threadIdx.x / kGroupSize;
  const int lane = threadIdx.x & (kGroupSize - 1);
  const int row_begin = row_chunk * kRowsPerChunk;
  const int row_end = min(rows, row_begin + kRowsPerChunk);
  const int boundaries = metadata_tile_count + 1;
  const ValueT homogeneous_value = Homogeneous ? values[0] : ValueT(0);

  for (int row = row_begin + group; row < row_end; row += kGroupsPerBlock) {
    const size_t offset = static_cast<size_t>(row) * boundaries + metadata_tile;
    int begin = tile_offsets[offset];
    int end = tile_offsets[offset + 1];
    if (kSubtilesPerMetadata == 2) {
      int midpoint = begin;
      if (lane == 0) {
        int low = begin;
        int high = end;
        const int64_t sparse_begin = row_ptr[row];
        while (low < high) {
          const int middle = low + (high - low) / 2;
          if (local_targets[sparse_begin + middle] < kComputeTileSize) {
            low = middle + 1;
          } else {
            high = middle;
          }
        }
        midpoint = low;
      }
      midpoint = __shfl_sync(__activemask(), midpoint, 0, kGroupSize);
      if (metadata_subtile == 0) {
        end = midpoint;
      } else {
        begin = midpoint;
      }
    }

    ValueT sum = ValueT(0);
    const int64_t sparse_begin = row_ptr[row];
    for (int64_t relative = static_cast<int64_t>(begin) + lane;
         relative < static_cast<int64_t>(end); relative += kGroupSize) {
      const int64_t entry = sparse_begin + relative;
      const int local = static_cast<int>(local_targets[entry]) - local_tile_begin;
      const ValueT weight = Homogeneous ? homogeneous_value : values[entry];
      sum += weight * x_tile[local];
    }
    sum = group_reduce_sum(sum);
    if (lane == 0) atomicAdd(&y[row], sum);
  }
}

template <typename ValueT, BE::DType DType, bool Homogeneous>
void launch_xw(
    const BE::Tensor values,
    const BE::Tensor indices,
    const BE::Tensor row_ptr,
    const BE::Tensor local_targets,
    const BE::Tensor tile_offsets,
    const BE::Tensor x,
    BE::Tensor y,
    int64_t stream) {
  const char* operation = Homogeneous ? "csrmv_xw_wpr_homo" : "csrmv_xw_wpr";
  BE_CHECK(values.dtype() == DType && x.dtype() == DType && y.dtype() == DType)
      << operation << " requires matching value, x, and y dtypes";
  BE_CHECK(indices.dtype() == BE::DType::Int32)
      << operation << " requires int32 indices";
  BE_CHECK(row_ptr.dtype() == BE::DType::Int64)
      << operation << " requires int64 row_ptr";
  BE_CHECK(local_targets.dtype() == BE::DType::UInt16)
      << operation << " requires uint16 local_targets";
  BE_CHECK(tile_offsets.dtype() == BE::DType::Int32)
      << operation << " requires int32 tile_offsets";
  BE_CHECK(values.ndim() == 1 && indices.ndim() == 1 && row_ptr.ndim() == 1 &&
           local_targets.ndim() == 1 && tile_offsets.ndim() == 2 &&
           x.ndim() == 1 && y.ndim() == 1)
      << operation << " received an invalid tensor rank";
  const int rows = static_cast<int>(row_ptr.size(0)) - 1;
  const int cols = static_cast<int>(y.size(0));
  const int metadata_tiles =
      (cols + kMetadataTileSize - 1) / kMetadataTileSize;
  const size_t nnz = indices.numel();
  BE_CHECK(rows >= 0 && x.numel() == static_cast<size_t>(rows))
      << operation << " x length must equal the number of rows";
  BE_CHECK(local_targets.numel() == nnz)
      << operation << " local_targets length must equal nnz";
  BE_CHECK(tile_offsets.size(0) == rows &&
           tile_offsets.size(1) == metadata_tiles + 1)
      << operation << " tile_offsets shape mismatch";
  BE_CHECK(row_ptr.numel() == static_cast<size_t>(rows + 1))
      << operation << " row_ptr length mismatch";
  BE_CHECK((Homogeneous && values.numel() == 1) ||
           (!Homogeneous && values.numel() == nnz))
      << operation << " values length mismatch";

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  BE_CUDA_CHECK(cudaMemsetAsync(
      y.data_ptr<ValueT>(), 0, y.numel() * sizeof(ValueT), cuda_stream));
  if (rows == 0 || nnz == 0 || y.numel() == 0) return;
  const int warps_per_block = kWarpBlockSize / 32;
  const int blocks = (rows + warps_per_block - 1) / warps_per_block;
  xw_wpr_kernel<ValueT, Homogeneous><<<blocks, kWarpBlockSize, 0, cuda_stream>>>(
      values.data_ptr<const ValueT>(), indices.data_ptr<const int32_t>(),
      row_ptr.data_ptr<const int64_t>(), x.data_ptr<const ValueT>(),
      y.data_ptr<ValueT>(), rows);
  BE_CUDA_CHECK(cudaGetLastError());
}

template <typename ValueT, BE::DType DType, bool Homogeneous>
void launch_wx(
    const BE::Tensor values,
    const BE::Tensor row_ptr,
    const BE::Tensor local_targets,
    const BE::Tensor tile_offsets,
    const BE::Tensor x,
    BE::Tensor y,
    int64_t stream) {
  const char* operation = Homogeneous ? "csrmv_wx_tile_homo" : "csrmv_wx_tile";
  BE_CHECK(values.dtype() == DType && x.dtype() == DType && y.dtype() == DType)
      << operation << " requires matching value, x, and y dtypes";
  BE_CHECK(row_ptr.dtype() == BE::DType::Int64)
      << operation << " requires int64 row_ptr";
  BE_CHECK(local_targets.dtype() == BE::DType::UInt16)
      << operation << " requires uint16 local_targets";
  BE_CHECK(tile_offsets.dtype() == BE::DType::Int32)
      << operation << " requires int32 tile_offsets";
  BE_CHECK(values.ndim() == 1 && row_ptr.ndim() == 1 &&
           local_targets.ndim() == 1 && tile_offsets.ndim() == 2 &&
           x.ndim() == 1 && y.ndim() == 1)
      << operation << " received an invalid tensor rank";
  const int rows = static_cast<int>(row_ptr.size(0)) - 1;
  const int cols = static_cast<int>(x.size(0));
  const int metadata_tiles = (cols + kMetadataTileSize - 1) / kMetadataTileSize;
  const size_t nnz = local_targets.numel();
  BE_CHECK(rows >= 0 && y.numel() == static_cast<size_t>(rows))
      << operation << " y length must equal the number of rows";
  BE_CHECK(tile_offsets.size(0) == rows &&
           tile_offsets.size(1) == metadata_tiles + 1)
      << operation << " tile_offsets shape mismatch";
  BE_CHECK((Homogeneous && values.numel() == 1) ||
           (!Homogeneous && values.numel() == nnz))
      << operation << " values length mismatch";

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  BE_CUDA_CHECK(cudaMemsetAsync(
      y.data_ptr<ValueT>(), 0, y.numel() * sizeof(ValueT), cuda_stream));
  if (rows == 0 || cols == 0 || nnz == 0) return;
  constexpr int compute_tile_size = kSharedBytes / sizeof(ValueT);
  const int compute_tiles = (cols + compute_tile_size - 1) / compute_tile_size;
  const int row_chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
  wx_tile_kernel<ValueT, Homogeneous>
      <<<dim3(row_chunks, compute_tiles), kBlockSize, 0, cuda_stream>>>(
          values.data_ptr<const ValueT>(), row_ptr.data_ptr<const int64_t>(),
          local_targets.data_ptr<const uint16_t>(),
          tile_offsets.data_ptr<const int32_t>(), x.data_ptr<const ValueT>(),
          y.data_ptr<ValueT>(), rows, cols, metadata_tiles);
  BE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

#define DEFINE_CSRMV_HANDLERS(SUFFIX, VALUE_T, DTYPE)                         \
  void csrmv_xw_wpr##SUFFIX(                                                  \
      const BE::Tensor values, const BE::Tensor indices,                      \
      const BE::Tensor row_ptr, const BE::Tensor local_targets,               \
      const BE::Tensor tile_offsets, const BE::Tensor x, BE::Tensor y,        \
      int64_t stream) {                                                        \
    launch_xw<VALUE_T, DTYPE, false>(                                         \
        values, indices, row_ptr, local_targets, tile_offsets, x, y, stream); \
  }                                                                            \
  void csrmv_xw_wpr_homo##SUFFIX(                                             \
      const BE::Tensor values, const BE::Tensor indices,                      \
      const BE::Tensor row_ptr, const BE::Tensor local_targets,               \
      const BE::Tensor tile_offsets, const BE::Tensor x, BE::Tensor y,        \
      int64_t stream) {                                                        \
    launch_xw<VALUE_T, DTYPE, true>(                                          \
        values, indices, row_ptr, local_targets, tile_offsets, x, y, stream); \
  }                                                                            \
  void csrmv_wx_tile##SUFFIX(                                                  \
      const BE::Tensor values, const BE::Tensor row_ptr,                      \
      const BE::Tensor local_targets, const BE::Tensor tile_offsets,          \
      const BE::Tensor x, BE::Tensor y, int64_t stream) {                     \
    launch_wx<VALUE_T, DTYPE, false>(                                         \
        values, row_ptr, local_targets, tile_offsets, x, y, stream);          \
  }                                                                            \
  void csrmv_wx_tile_homo##SUFFIX(                                             \
      const BE::Tensor values, const BE::Tensor row_ptr,                      \
      const BE::Tensor local_targets, const BE::Tensor tile_offsets,          \
      const BE::Tensor x, BE::Tensor y, int64_t stream) {                     \
    launch_wx<VALUE_T, DTYPE, true>(                                          \
        values, row_ptr, local_targets, tile_offsets, x, y, stream);          \
  }

// @BE csrmv_xw_wpr_f32 arg arg arg arg arg arg ret stream
// @BE csrmv_xw_wpr_homo_f32 arg arg arg arg arg arg ret stream
// @BE csrmv_wx_tile_f32 arg arg arg arg arg ret stream
// @BE csrmv_wx_tile_homo_f32 arg arg arg arg arg ret stream
DEFINE_CSRMV_HANDLERS(_f32, float, BE::DType::Float32)

// @BE csrmv_xw_wpr_f64 arg arg arg arg arg arg ret stream
// @BE csrmv_xw_wpr_homo_f64 arg arg arg arg arg arg ret stream
// @BE csrmv_wx_tile_f64 arg arg arg arg arg ret stream
// @BE csrmv_wx_tile_homo_f64 arg arg arg arg arg ret stream
DEFINE_CSRMV_HANDLERS(_f64, double, BE::DType::Float64)

#undef DEFINE_CSRMV_HANDLERS
