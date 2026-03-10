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
 * This module provides two sets of CUDA kernels:
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
