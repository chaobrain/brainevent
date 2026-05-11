// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

/*
 * compact.cu -- Stream Compaction and Fused Bit-Pack + Compaction CUDA Kernels
 * =============================================================================
 *
 * This module provides five sets of CUDA kernels:
 *
 * 1. compact_1d: Stream compaction for 1D binary arrays.
 *    Extracts indices of all non-zero elements using __ballot_sync + atomicAdd.
 *    - Input: spikes (bool/uint8), shape (n,)
 *    - Output: active_ids (int32, shape (n,)), n_active (int32, shape (1,))
 *
 * 2. fused_bitpack_compact_2d: Fused bit-pack + row-level compaction for 2D binary arrays.
 *    Each block processes one row: __ballot_sync packs 32 columns into one uint32,
 *    and rows with any active column are collected via atomicAdd.
 *    - Input: B (bool/uint8), shape (n_pre, n_batch)
 *    - Output: packed (uint32, shape (n_pre, n_batch_packed)),
 *              active_ids (int32, shape (n_pre,)),
 *              n_active (int32, shape (1,))
 *
 * 3. pair_stream_encode_2d: Compact COO-like streaming encoding for 2D binary
 *    arrays. Each active element emits one zero-based (row, col) pair into a
 *    static-capacity output buffer, along with a valid pair count. Valid pairs
 *    are compacted, but parallel emission does not guarantee row-major order.
 *    - Input: B (bool/uint8), shape (n_src, n_batch)
 *    - Output: pair_stream (int32, shape (n_src * n_batch, 2)),
 *              n_pairs (int32, shape (1,))
 *
 * 4. row_sparse_encode_2d: Fixed-width FCN spike encoding for 2D binary arrays.
 *    Each warp processes one row and emits 1-based active batch-column ids in
 *    ascending order, compacted to the front of a fixed-width row and padded
 *    with zeros.
 *    - Input: B (bool/uint8), shape (n_src, n_batch)
 *    - Output: spike_indices (int32, shape (n_src, row_size))
 *
 * 5. dense-to-CSR encode (binary, values omitted): two warp-per-row passes for
 *    row NNZ counting and CSR column-index filling.
 *    - Input: B (bool/uint8), shape (n_src, n_batch)
 *    - Output: row_counts (int32, shape (n_src,)) and CSR indices
 *      (int32, shape (n_src * n_batch,))
 */

#include <cuda_runtime.h>
#include "brainevent/common.h"

// ============================================================================
// 1D Stream Compaction
// ============================================================================

// Each thread reads one element of spikes[].
// __ballot_sync collects 32 active bits per warp.
// __popc(ballot & lane_mask) gives each thread its local prefix sum.
// Warp lane 0 atomicAdds to n_active to reserve global positions.
// Active threads write their global index to active_ids[base + local_pos].

__global__ void _compact_1d_bool_kern(
    const uint8_t* __restrict__ spikes,
    int32_t*       __restrict__ active_ids,
    int32_t*       __restrict__ n_active,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    // Each thread checks one element
    bool is_active = (tid < n) && (spikes[tid] != 0);

    // Collect active bits across the warp
    unsigned ballot = __ballot_sync(0xffffffff, is_active);
    int warp_count = __popc(ballot);

    // Lane 0 reserves space in the global output
    __shared__ int s_base[8];  // max 8 warps per block (256/32)
    if (lane == 0 && warp_count > 0) {
        s_base[warp_id] = atomicAdd(n_active, warp_count);
    }
    __syncthreads();

    // Each active thread writes its index
    if (is_active) {
        unsigned mask_below = (1u << lane) - 1u;
        int local_pos = __popc(ballot & mask_below);
        active_ids[s_base[warp_id] + local_pos] = tid;
    }
}

__global__ void _compact_1d_float_kern(
    const float* __restrict__ spikes,
    int32_t*     __restrict__ active_ids,
    int32_t*     __restrict__ n_active,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    bool is_active = (tid < n) && (spikes[tid] != 0.0f);

    unsigned ballot = __ballot_sync(0xffffffff, is_active);
    int warp_count = __popc(ballot);

    __shared__ int s_base[8];
    if (lane == 0 && warp_count > 0) {
        s_base[warp_id] = atomicAdd(n_active, warp_count);
    }
    __syncthreads();

    if (is_active) {
        unsigned mask_below = (1u << lane) - 1u;
        int local_pos = __popc(ballot & mask_below);
        active_ids[s_base[warp_id] + local_pos] = tid;
    }
}

// ---- FFI entry points for 1D compaction ----

// @BE compact_1d_bool
void compact_1d_bool(
    const BE::Tensor spikes,
    BE::Tensor active_ids,
    BE::Tensor n_active,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n = static_cast<int>(spikes.numel());
    if (n == 0) return;

    // Zero n_active before kernel launch
    cudaMemsetAsync(n_active.data_ptr(), 0, sizeof(int32_t), s);

    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    int32_t* d_ids = static_cast<int32_t*>(active_ids.data_ptr());
    int32_t* d_cnt = static_cast<int32_t*>(n_active.data_ptr());

    int bsz = 256;
    int n_blocks = (n + bsz - 1) / bsz;
    _compact_1d_bool_kern<<<n_blocks, bsz, 0, s>>>(d_spk, d_ids, d_cnt, n);
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE compact_1d_float
void compact_1d_float(
    const BE::Tensor spikes,
    BE::Tensor active_ids,
    BE::Tensor n_active,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n = static_cast<int>(spikes.numel());
    if (n == 0) return;

    cudaMemsetAsync(n_active.data_ptr(), 0, sizeof(int32_t), s);

    const float* d_spk = static_cast<const float*>(spikes.data_ptr());
    int32_t* d_ids = static_cast<int32_t*>(active_ids.data_ptr());
    int32_t* d_cnt = static_cast<int32_t*>(n_active.data_ptr());

    int bsz = 256;
    int n_blocks = (n + bsz - 1) / bsz;
    _compact_1d_float_kern<<<n_blocks, bsz, 0, s>>>(d_spk, d_ids, d_cnt, n);
    BE_CHECK_KERNEL_LAUNCH();
}

// ============================================================================
// 2D Row-Level Compaction Only (no bitpack)
// ============================================================================

// Lightweight: one thread per row, serial check of n_batch columns.
// Much lighter than the fused kernel when bitpack is not needed.

__global__ void _compact_2d_only_bool_kern(
    const uint8_t* __restrict__ B,
    int32_t*       __restrict__ active_ids,
    int32_t*       __restrict__ n_active,
    int n_pre, int n_batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_pre) return;

    const uint8_t* row_ptr = B + (size_t)row * n_batch;
    bool row_active = false;
    for (int j = 0; j < n_batch; j++) {
        if (row_ptr[j] != 0) {
            row_active = true;
            break;
        }
    }
    if (row_active) {
        int pos = atomicAdd(n_active, 1);
        active_ids[pos] = row;
    }
}

__global__ void _compact_2d_only_float_kern(
    const float* __restrict__ B,
    int32_t*     __restrict__ active_ids,
    int32_t*     __restrict__ n_active,
    int n_pre, int n_batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_pre) return;

    const float* row_ptr = B + (size_t)row * n_batch;
    bool row_active = false;
    for (int j = 0; j < n_batch; j++) {
        if (row_ptr[j] != 0.0f) {
            row_active = true;
            break;
        }
    }
    if (row_active) {
        int pos = atomicAdd(n_active, 1);
        active_ids[pos] = row;
    }
}

// ---- FFI entry points for 2D compaction only ----

// @BE compact_2d_only_bool
void compact_2d_only_bool(
    const BE::Tensor B,
    BE::Tensor active_ids,
    BE::Tensor n_active,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = static_cast<int>(B.size(0));
    int n_batch = static_cast<int>(B.size(1));

    if (n_pre == 0 || n_batch == 0) return;

    cudaMemsetAsync(n_active.data_ptr(), 0, sizeof(int32_t), s);

    const uint8_t* d_B = static_cast<const uint8_t*>(B.data_ptr());
    int32_t* d_ids = static_cast<int32_t*>(active_ids.data_ptr());
    int32_t* d_cnt = static_cast<int32_t*>(n_active.data_ptr());

    int bsz = 256;
    int n_blocks = (n_pre + bsz - 1) / bsz;
    _compact_2d_only_bool_kern<<<n_blocks, bsz, 0, s>>>(
        d_B, d_ids, d_cnt, n_pre, n_batch);
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE compact_2d_only_float
void compact_2d_only_float(
    const BE::Tensor B,
    BE::Tensor active_ids,
    BE::Tensor n_active,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = static_cast<int>(B.size(0));
    int n_batch = static_cast<int>(B.size(1));

    if (n_pre == 0 || n_batch == 0) return;

    cudaMemsetAsync(n_active.data_ptr(), 0, sizeof(int32_t), s);

    const float* d_B = static_cast<const float*>(B.data_ptr());
    int32_t* d_ids = static_cast<int32_t*>(active_ids.data_ptr());
    int32_t* d_cnt = static_cast<int32_t*>(n_active.data_ptr());

    int bsz = 256;
    int n_blocks = (n_pre + bsz - 1) / bsz;
    _compact_2d_only_float_kern<<<n_blocks, bsz, 0, s>>>(
        d_B, d_ids, d_cnt, n_pre, n_batch);
    BE_CHECK_KERNEL_LAUNCH();
}

// ============================================================================
// 2D Fused Bit-Pack + Row-Level Compaction
// ============================================================================

// Each block processes one row (blockIdx.x = row_index).
// Within a block, warps process consecutive 32-column words:
//   warp w handles columns [w*32 .. w*32+31].
// Each lane reads one column value; __ballot_sync packs 32 values into uint32.
// Lane 0 writes the packed word to B_packed[row, w].
// If any packed word in the row is nonzero, the row is marked active.
// threadIdx.x == 0 uses atomicAdd(n_active, 1) to append to active_ids.

__global__ void _fused_bitpack_compact_2d_bool_kern(
    const uint8_t* __restrict__ B,
    uint32_t*      __restrict__ B_packed,
    int32_t*       __restrict__ active_ids,
    int32_t*       __restrict__ n_active,
    int n_pre, int n_batch, int n_batch_packed
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;

    __shared__ uint32_t s_row_or;
    if (threadIdx.x == 0) s_row_or = 0;
    __syncthreads();

    const uint8_t* row_ptr = B + (size_t)row * n_batch;
    uint32_t* packed_row = B_packed + (size_t)row * n_batch_packed;

    // Each warp processes one packed word
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;

    for (int w = warp_id; w < n_batch_packed; w += warps_per_block) {
        int col = w * 32 + lane;
        bool val = (col < n_batch) ? (row_ptr[col] != 0) : false;

        unsigned packed_word = __ballot_sync(0xffffffff, val);

        // Lane 0 writes the packed word
        if (lane == 0) {
            packed_row[w] = packed_word;
            if (packed_word != 0) {
                atomicOr(&s_row_or, 1u);
            }
        }
    }
    __syncthreads();

    // Thread 0 checks if the row has any active columns
    if (threadIdx.x == 0 && s_row_or != 0) {
        int pos = atomicAdd(n_active, 1);
        active_ids[pos] = row;
    }
}

__global__ void _fused_bitpack_compact_2d_float_kern(
    const float* __restrict__ B,
    uint32_t*    __restrict__ B_packed,
    int32_t*     __restrict__ active_ids,
    int32_t*     __restrict__ n_active,
    int n_pre, int n_batch, int n_batch_packed
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;

    __shared__ uint32_t s_row_or;
    if (threadIdx.x == 0) s_row_or = 0;
    __syncthreads();

    const float* row_ptr = B + (size_t)row * n_batch;
    uint32_t* packed_row = B_packed + (size_t)row * n_batch_packed;

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;

    for (int w = warp_id; w < n_batch_packed; w += warps_per_block) {
        int col = w * 32 + lane;
        bool val = (col < n_batch) ? (row_ptr[col] != 0.0f) : false;

        unsigned packed_word = __ballot_sync(0xffffffff, val);

        if (lane == 0) {
            packed_row[w] = packed_word;
            if (packed_word != 0) {
                atomicOr(&s_row_or, 1u);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_row_or != 0) {
        int pos = atomicAdd(n_active, 1);
        active_ids[pos] = row;
    }
}

// ---- FFI entry points for 2D fused bitpack + compaction ----

// @BE fused_bitpack_compact_2d_bool
void fused_bitpack_compact_2d_bool(
    const BE::Tensor B,
    BE::Tensor B_packed,
    BE::Tensor active_ids,
    BE::Tensor n_active,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = static_cast<int>(B.size(0));
    int n_batch = static_cast<int>(B.size(1));
    int n_batch_packed = static_cast<int>(B_packed.size(1));

    if (n_pre == 0 || n_batch == 0) return;

    // Zero n_active and packed output before kernel launch
    cudaMemsetAsync(n_active.data_ptr(), 0, sizeof(int32_t), s);
    cudaMemsetAsync(B_packed.data_ptr(), 0,
                    (size_t)n_pre * n_batch_packed * sizeof(uint32_t), s);

    const uint8_t* d_B = static_cast<const uint8_t*>(B.data_ptr());
    uint32_t* d_packed = static_cast<uint32_t*>(B_packed.data_ptr());
    int32_t* d_ids = static_cast<int32_t*>(active_ids.data_ptr());
    int32_t* d_cnt = static_cast<int32_t*>(n_active.data_ptr());

    // One block per row; 256 threads = 8 warps, each warp handles one word
    int bsz = 256;
    _fused_bitpack_compact_2d_bool_kern<<<n_pre, bsz, 0, s>>>(
        d_B, d_packed, d_ids, d_cnt, n_pre, n_batch, n_batch_packed);
    BE_CHECK_KERNEL_LAUNCH();
}

// @BE fused_bitpack_compact_2d_float
void fused_bitpack_compact_2d_float(
    const BE::Tensor B,
    BE::Tensor B_packed,
    BE::Tensor active_ids,
    BE::Tensor n_active,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = static_cast<int>(B.size(0));
    int n_batch = static_cast<int>(B.size(1));
    int n_batch_packed = static_cast<int>(B_packed.size(1));

    if (n_pre == 0 || n_batch == 0) return;

    cudaMemsetAsync(n_active.data_ptr(), 0, sizeof(int32_t), s);
    cudaMemsetAsync(B_packed.data_ptr(), 0,
                    (size_t)n_pre * n_batch_packed * sizeof(uint32_t), s);

    const float* d_B = static_cast<const float*>(B.data_ptr());
    uint32_t* d_packed = static_cast<uint32_t*>(B_packed.data_ptr());
    int32_t* d_ids = static_cast<int32_t*>(active_ids.data_ptr());
    int32_t* d_cnt = static_cast<int32_t*>(n_active.data_ptr());

    int bsz = 256;
    _fused_bitpack_compact_2d_float_kern<<<n_pre, bsz, 0, s>>>(
        d_B, d_packed, d_ids, d_cnt, n_pre, n_batch, n_batch_packed);
    BE_CHECK_KERNEL_LAUNCH();
}

// ============================================================================
// 2D Row-Sparse Encoding
// ============================================================================

namespace {

constexpr int kRowSparseWarpSize = 32;
constexpr int kRowSparseWarpsPerBlock = 8;
constexpr int kRowSparseBlockSize = kRowSparseWarpSize * kRowSparseWarpsPerBlock;
constexpr int kRowSparseMaxGridX = 4096;

template <typename SpikeT>
struct RowSparseActivity;

template <>
struct RowSparseActivity<uint8_t> {
    __device__ static bool is_active(uint8_t value) {
        return value != 0;
    }
};

template <>
struct RowSparseActivity<float> {
    __device__ static bool is_active(float value) {
        return value != 0.0f;
    }
};

template <typename SpikeT>
__global__ void _pair_stream_encode_2d_kern(
    const SpikeT* __restrict__ B,
    int32_t* __restrict__ pair_stream,
    int32_t* __restrict__ n_pairs,
    int n_src,
    int n_batch
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kRowSparseWarpSize - 1);
    const int warps_per_block = blockDim.x >> 5;

    for (int base = static_cast<int>(blockIdx.x) * warps_per_block;
         base < n_src;
         base += static_cast<int>(gridDim.x) * warps_per_block) {
        const int row = base + warp_id;
        if (row >= n_src) {
            continue;
        }

        const SpikeT* row_ptr = B + static_cast<size_t>(row) * n_batch;
        for (int batch_base = 0; batch_base < n_batch; batch_base += kRowSparseWarpSize) {
            const int col = batch_base + lane;
            const bool active = (col < n_batch)
                && RowSparseActivity<SpikeT>::is_active(__ldg(&row_ptr[col]));
            const unsigned mask = __ballot_sync(0xffffffffu, active);
            const int active_count = __popc(mask);

            int base_out = 0;
            if (lane == 0 && active_count > 0) {
                base_out = atomicAdd(n_pairs, active_count);
            }
            base_out = __shfl_sync(0xffffffffu, base_out, 0);

            if (active) {
                const unsigned mask_below = (lane == 0) ? 0u : ((1u << lane) - 1u);
                const int local_pos = __popc(mask & mask_below);
                const size_t out_idx = static_cast<size_t>(base_out + local_pos) * 2;
                pair_stream[out_idx] = row;
                pair_stream[out_idx + 1] = col;
            }
        }
    }
}

template <typename SpikeT>
void pair_stream_encode_2d(
    const BE::Tensor B,
    BE::Tensor pair_stream,
    BE::Tensor n_pairs,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const int n_src = static_cast<int>(B.size(0));
    const int n_batch = static_cast<int>(B.size(1));

    BE_CUDA_CHECK(cudaMemsetAsync(n_pairs.data_ptr(), 0, sizeof(int32_t), s));

    if (n_src == 0 || n_batch == 0) {
        return;
    }

    const SpikeT* d_B = static_cast<const SpikeT*>(B.data_ptr());
    int32_t* d_pair_stream = static_cast<int32_t*>(pair_stream.data_ptr());
    int32_t* d_n_pairs = static_cast<int32_t*>(n_pairs.data_ptr());

    const int raw_grid_x = (n_src + kRowSparseWarpsPerBlock - 1) / kRowSparseWarpsPerBlock;
    const int grid_x = (raw_grid_x < kRowSparseMaxGridX) ? raw_grid_x : kRowSparseMaxGridX;
    _pair_stream_encode_2d_kern<SpikeT><<<grid_x, kRowSparseBlockSize, 0, s>>>(
        d_B, d_pair_stream, d_n_pairs, n_src, n_batch);
    BE_CHECK_KERNEL_LAUNCH();
}

template <typename SpikeT>
__global__ void _row_sparse_encode_2d_kern(
    const SpikeT* __restrict__ B,
    int32_t* __restrict__ spike_indices,
    int n_src,
    int n_batch,
    int row_size
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kRowSparseWarpSize - 1);
    const int warps_per_block = blockDim.x >> 5;

    for (int base = static_cast<int>(blockIdx.x) * warps_per_block;
         base < n_src;
         base += static_cast<int>(gridDim.x) * warps_per_block) {
        const int row = base + warp_id;
        if (row >= n_src) {
            continue;
        }

        const SpikeT* row_ptr = B + static_cast<size_t>(row) * n_batch;
        int32_t* out_row = spike_indices + static_cast<size_t>(row) * row_size;
        int nnz = 0;

        for (int batch_base = 0; batch_base < n_batch; batch_base += kRowSparseWarpSize) {
            const int col = batch_base + lane;
            const bool active = (col < n_batch)
                && RowSparseActivity<SpikeT>::is_active(__ldg(&row_ptr[col]));
            const unsigned mask = __ballot_sync(0xffffffffu, active);
            const int local_pos = __popc(mask & ((1u << lane) - 1u));
            if (active && (nnz + local_pos) < row_size) {
                out_row[nnz + local_pos] = col + 1;
            }
            nnz += __popc(mask);
        }
    }
}

template <typename SpikeT>
void row_sparse_encode_2d(
    const BE::Tensor B,
    BE::Tensor spike_indices,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const int n_src = static_cast<int>(B.size(0));
    const int n_batch = static_cast<int>(B.size(1));
    const int row_size = static_cast<int>(spike_indices.size(1));

    BE_CUDA_CHECK(cudaMemsetAsync(
        spike_indices.data_ptr(), 0,
        static_cast<size_t>(n_src) * row_size * sizeof(int32_t), s));

    if (n_src == 0 || n_batch == 0 || row_size == 0) {
        return;
    }

    const SpikeT* d_B = static_cast<const SpikeT*>(B.data_ptr());
    int32_t* d_spike_indices = static_cast<int32_t*>(spike_indices.data_ptr());

    const int raw_grid_x = (n_src + kRowSparseWarpsPerBlock - 1) / kRowSparseWarpsPerBlock;
    const int grid_x = (raw_grid_x < kRowSparseMaxGridX) ? raw_grid_x : kRowSparseMaxGridX;
    _row_sparse_encode_2d_kern<SpikeT><<<grid_x, kRowSparseBlockSize, 0, s>>>(
        d_B, d_spike_indices, n_src, n_batch, row_size);
    BE_CHECK_KERNEL_LAUNCH();
}

template <typename SpikeT>
__global__ void _csr_row_count_2d_kern(
    const SpikeT* __restrict__ B,
    int32_t* __restrict__ row_counts,
    int n_src,
    int n_batch
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kRowSparseWarpSize - 1);
    const int warps_per_block = blockDim.x >> 5;

    for (int base = static_cast<int>(blockIdx.x) * warps_per_block;
         base < n_src;
         base += static_cast<int>(gridDim.x) * warps_per_block) {
        const int row = base + warp_id;
        if (row >= n_src) {
            continue;
        }

        const SpikeT* row_ptr = B + static_cast<size_t>(row) * n_batch;
        int nnz = 0;

        for (int batch_base = 0; batch_base < n_batch; batch_base += kRowSparseWarpSize) {
            const int col = batch_base + lane;
            const bool active = (col < n_batch)
                && RowSparseActivity<SpikeT>::is_active(__ldg(&row_ptr[col]));
            const unsigned mask = __ballot_sync(0xffffffffu, active);
            if (lane == 0) {
                nnz += __popc(mask);
            }
        }

        if (lane == 0) {
            row_counts[row] = nnz;
        }
    }
}

template <typename SpikeT>
void csr_row_count_2d(
    const BE::Tensor B,
    BE::Tensor row_counts,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const int n_src = static_cast<int>(B.size(0));
    const int n_batch = static_cast<int>(B.size(1));

    BE_CUDA_CHECK(cudaMemsetAsync(
        row_counts.data_ptr(), 0,
        static_cast<size_t>(n_src) * sizeof(int32_t), s));

    if (n_src == 0 || n_batch == 0) {
        return;
    }

    const SpikeT* d_B = static_cast<const SpikeT*>(B.data_ptr());
    int32_t* d_row_counts = static_cast<int32_t*>(row_counts.data_ptr());

    const int raw_grid_x = (n_src + kRowSparseWarpsPerBlock - 1) / kRowSparseWarpsPerBlock;
    const int grid_x = (raw_grid_x < kRowSparseMaxGridX) ? raw_grid_x : kRowSparseMaxGridX;
    _csr_row_count_2d_kern<SpikeT><<<grid_x, kRowSparseBlockSize, 0, s>>>(
        d_B, d_row_counts, n_src, n_batch);
    BE_CHECK_KERNEL_LAUNCH();
}

template <typename SpikeT>
__global__ void _csr_fill_2d_kern(
    const SpikeT* __restrict__ B,
    const int32_t* __restrict__ indptr,
    int32_t* __restrict__ indices,
    int n_src,
    int n_batch
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kRowSparseWarpSize - 1);
    const int warps_per_block = blockDim.x >> 5;

    for (int block_row = static_cast<int>(blockIdx.x) * warps_per_block;
         block_row < n_src;
         block_row += static_cast<int>(gridDim.x) * warps_per_block) {
        const int row = block_row + warp_id;
        if (row >= n_src) {
            continue;
        }

        const SpikeT* row_ptr = B + static_cast<size_t>(row) * n_batch;
        const int32_t base = __ldg(&indptr[row]);
        int nnz = 0;

        for (int batch_base = 0; batch_base < n_batch; batch_base += kRowSparseWarpSize) {
            const int col = batch_base + lane;
            const bool active = (col < n_batch)
                && RowSparseActivity<SpikeT>::is_active(__ldg(&row_ptr[col]));
            const unsigned mask = __ballot_sync(0xffffffffu, active);
            const int local_pos = __popc(mask & ((1u << lane) - 1u));
            if (active) {
                indices[static_cast<size_t>(base + nnz + local_pos)] = col;
            }
            nnz += __popc(mask);
        }
    }
}

template <typename SpikeT>
void csr_fill_2d(
    const BE::Tensor B,
    const BE::Tensor indptr,
    BE::Tensor indices,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const int n_src = static_cast<int>(B.size(0));
    const int n_batch = static_cast<int>(B.size(1));

    BE_CUDA_CHECK(cudaMemsetAsync(
        indices.data_ptr(), 0,
        static_cast<size_t>(n_src) * n_batch * sizeof(int32_t), s));

    if (n_src == 0 || n_batch == 0) {
        return;
    }

    const SpikeT* d_B = static_cast<const SpikeT*>(B.data_ptr());
    const int32_t* d_indptr = static_cast<const int32_t*>(indptr.data_ptr());
    int32_t* d_indices = static_cast<int32_t*>(indices.data_ptr());

    const int raw_grid_x = (n_src + kRowSparseWarpsPerBlock - 1) / kRowSparseWarpsPerBlock;
    const int grid_x = (raw_grid_x < kRowSparseMaxGridX) ? raw_grid_x : kRowSparseMaxGridX;
    _csr_fill_2d_kern<SpikeT><<<grid_x, kRowSparseBlockSize, 0, s>>>(
        d_B, d_indptr, d_indices, n_src, n_batch);
    BE_CHECK_KERNEL_LAUNCH();
}

}  // namespace

// ---- FFI entry points for 2D row-sparse encoding ----

// @BE row_sparse_encode_2d_bool
void row_sparse_encode_2d_bool(
    const BE::Tensor B,
    BE::Tensor spike_indices,
    int64_t stream
) {
    row_sparse_encode_2d<uint8_t>(
        B, spike_indices, stream);
}

// @BE row_sparse_encode_2d_float
void row_sparse_encode_2d_float(
    const BE::Tensor B,
    BE::Tensor spike_indices,
    int64_t stream
) {
    row_sparse_encode_2d<float>(
        B, spike_indices, stream);
}

// ---- FFI entry points for 2D dense-to-CSR encoding ----

// @BE csr_row_count_2d_bool
void csr_row_count_2d_bool(
    const BE::Tensor B,
    BE::Tensor row_counts,
    int64_t stream
) {
    csr_row_count_2d<uint8_t>(
        B, row_counts, stream);
}

// @BE csr_row_count_2d_float
void csr_row_count_2d_float(
    const BE::Tensor B,
    BE::Tensor row_counts,
    int64_t stream
) {
    csr_row_count_2d<float>(
        B, row_counts, stream);
}

// @BE csr_fill_2d_bool
void csr_fill_2d_bool(
    const BE::Tensor B,
    const BE::Tensor indptr,
    BE::Tensor indices,
    int64_t stream
) {
    csr_fill_2d<uint8_t>(
        B, indptr, indices, stream);
}

// @BE csr_fill_2d_float
void csr_fill_2d_float(
    const BE::Tensor B,
    const BE::Tensor indptr,
    BE::Tensor indices,
    int64_t stream
) {
    csr_fill_2d<float>(
        B, indptr, indices, stream);
}

// @BE pair_stream_encode_2d_bool
void pair_stream_encode_2d_bool(
    const BE::Tensor B,
    BE::Tensor pair_stream,
    BE::Tensor n_pairs,
    int64_t stream
) {
    pair_stream_encode_2d<uint8_t>(
        B, pair_stream, n_pairs, stream);
}

// @BE pair_stream_encode_2d_float
void pair_stream_encode_2d_float(
    const BE::Tensor B,
    BE::Tensor pair_stream,
    BE::Tensor n_pairs,
    int64_t stream
) {
    pair_stream_encode_2d<float>(
        B, pair_stream, n_pairs, stream);
}
