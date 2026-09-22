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
// =============================================================================

/*
 * csr_to_csc.cu -- Column-block CSR to CSC construction kernels
 * =============================================================================
 *
 * This file provides CUDA-side primitives for streaming a CSR matrix into CSC
 * column blocks.  The caller owns CPU/GPU orchestration: run a global count,
 * prefix-sum counts into CSC starts, initialize each block's next_pos, run fill,
 * copy the block back to host, then append it to the final CSC arrays.
 *
 * Column order inside a block is grouped by column through atomic positions, but
 * row order inside each column is intentionally not stable.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

namespace
{

    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

    template <typename OffsetT>
    __device__ __forceinline__ OffsetT atomic_add_offset(OffsetT *addr, OffsetT val);

    template <>
    __device__ __forceinline__ int32_t atomic_add_offset<int32_t>(
        int32_t *addr,
        int32_t val)
    {
        return atomicAdd(addr, val);
    }

    template <>
    __device__ __forceinline__ int64_t atomic_add_offset<int64_t>(
        int64_t *addr,
        int64_t val)
    {
        return static_cast<int64_t>(
            atomicAdd(
                reinterpret_cast<unsigned long long int *>(addr),
                static_cast<unsigned long long int>(val)));
    }

    template <typename IndexT, typename OffsetT>
    __global__ void _csr_to_csc_count_warp_kern(
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        OffsetT *__restrict__ counts,
        int64_t n_rows)
    {
        int warp_in_block = threadIdx.x / kWarpSize;
        int lane = threadIdx.x & (kWarpSize - 1);
        int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
        if (row >= n_rows)
            return;

        OffsetT start = indptr[row];
        OffsetT end = indptr[row + 1];
        for (OffsetT j = start + static_cast<OffsetT>(lane); j < end; j += kWarpSize)
        {
            IndexT col = indices[j];
            atomic_add_offset(counts + col, static_cast<OffsetT>(1));
        }
    }

    template <typename IndexT, typename OffsetT>
    __global__ void _csr_to_csc_fill_block_thread_kern(
        const IndexT *__restrict__ indices,
        const OffsetT *__restrict__ indptr,
        OffsetT *__restrict__ next_pos,
        IndexT *__restrict__ local_rows,
        OffsetT *__restrict__ local_perm,
        int64_t n_rows,
        IndexT col_start,
        IndexT col_end)
    {
        int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (row >= n_rows)
            return;

        OffsetT start = indptr[row];
        OffsetT end = indptr[row + 1];
        for (OffsetT j = start; j < end; ++j)
        {
            IndexT col = indices[j];
            if (col >= col_start && col < col_end)
            {
                IndexT local_col = col - col_start;
                OffsetT dst = atomic_add_offset(next_pos + local_col, static_cast<OffsetT>(1));
                local_rows[dst] = static_cast<IndexT>(row);
                local_perm[dst] = j;
            }
        }
    }

    inline int64_t csr_num_rows(const BE::Tensor &indptr)
    {
        return indptr.shape(0) - 1;
    }

    inline int launch_blocks(int64_t n_rows)
    {
        return static_cast<int>((n_rows + kThreadsPerBlock - 1) / kThreadsPerBlock);
    }

    inline int launch_warp_row_blocks(int64_t n_rows)
    {
        return static_cast<int>((n_rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    }

} // namespace

// @BE csr_to_csc_count arg arg ret stream
void csr_to_csc_count(
    const BE::Tensor indices,
    const BE::Tensor indptr,
    BE::Tensor counts,
    int64_t stream)
{
    int64_t n_rows = csr_num_rows(indptr);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
        cudaMemsetAsync(
            counts.data_ptr(),
            0,
            static_cast<size_t>(counts.numel()) * sizeof(OffsetT),
            s);

        if (n_rows <= 0)
            return;

        _csr_to_csc_count_warp_kern<IndexT, OffsetT>
            <<<launch_warp_row_blocks(n_rows), kThreadsPerBlock, 0, s>>>(
                static_cast<const IndexT *>(indices.data_ptr()),
                static_cast<const OffsetT *>(indptr.data_ptr()),
                static_cast<OffsetT *>(counts.data_ptr()),
                n_rows);
    });
}

// @BE csr_to_csc_fill_block arg arg arg ret ret ret attr.col_start:int64 attr.col_end:int64 stream
void csr_to_csc_fill_block(
    const BE::Tensor indices,
    const BE::Tensor indptr,
    const BE::Tensor initial_pos,
    BE::Tensor scratch_pos,
    BE::Tensor local_rows,
    BE::Tensor local_perm,
    int64_t col_start,
    int64_t col_end,
    int64_t stream)
{
    if (col_end <= col_start)
        return;

    int64_t n_rows = csr_num_rows(indptr);
    if (n_rows <= 0)
        return;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
        cudaMemcpyAsync(
            scratch_pos.data_ptr(),
            initial_pos.data_ptr(),
            static_cast<size_t>(scratch_pos.numel()) * sizeof(OffsetT),
            cudaMemcpyDeviceToDevice,
            s);

        _csr_to_csc_fill_block_thread_kern<IndexT, OffsetT>
            <<<launch_blocks(n_rows), kThreadsPerBlock, 0, s>>>(
                static_cast<const IndexT *>(indices.data_ptr()),
                static_cast<const OffsetT *>(indptr.data_ptr()),
                static_cast<OffsetT *>(scratch_pos.data_ptr()),
                static_cast<IndexT *>(local_rows.data_ptr()),
                static_cast<OffsetT *>(local_perm.data_ptr()),
                n_rows,
                static_cast<IndexT>(col_start),
                static_cast<IndexT>(col_end));
    });
}
