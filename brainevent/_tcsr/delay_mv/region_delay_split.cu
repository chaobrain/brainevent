// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Split a delay-sorted CSR into K per-delay CSR sub-matrices for cuSPARSE.
// Inputs are the RegionDelay arrays consumed by the hybrid delay kernel.
// Outputs are packed in segment-major order: sub-matrix k owns entries
// [sub_base[k], sub_base[k+1]) and row pointers sub_indptr[k, :], which start
// at zero. Homogeneous weights are expanded so cuSPARSE always uses alpha = 1.
// Runs once at initialization; Launch takes raw pointers for native reuse.

#include "cuda_common.h"
#include "brainevent/common.h"
#include <cub/block/block_scan.cuh>
#include <cstdint>
#include <limits>

namespace region_delay_split
{

    constexpr int kBlockSize = 256;

    // One block per segment: exclusive scan of delay_count[:, k] over rows.
    template <typename EdgeT, typename SubT>
    __global__ void ColumnScan(const EdgeT *__restrict__ counts, SubT *__restrict__ sub_indptr,
                               int64_t rows, int64_t segments)
    {
        using Scan = cub::BlockScan<int64_t, kBlockSize>;
        __shared__ typename Scan::TempStorage temp;
        const int64_t segment = blockIdx.x;
        SubT *out = sub_indptr + segment * (rows + 1);
        int64_t running = 0;
        for (int64_t first = 0; first < rows; first += kBlockSize)
        {
            const int64_t row = first + threadIdx.x;
            const int64_t value = row < rows ? static_cast<int64_t>(counts[row * segments + segment]) : 0;
            int64_t prefix = 0;
            int64_t total = 0;
            Scan(temp).ExclusiveSum(value, prefix, total);
            if (row < rows)
                out[row] = static_cast<SubT>(running + prefix);
            running += total;
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[rows] = static_cast<SubT>(running);
    }

    template <typename SubT>
    __global__ void BaseScan(const SubT *__restrict__ sub_indptr, SubT *__restrict__ sub_base,
                             int64_t rows, int64_t segments)
    {
        int64_t running = 0;
        sub_base[0] = 0;
        for (int64_t segment = 0; segment < segments; ++segment)
        {
            running += static_cast<int64_t>(sub_indptr[segment * (rows + 1) + rows]);
            sub_base[segment + 1] = static_cast<SubT>(running);
        }
    }

    // One thread per row walks its delay segments in order and copies each one
    // to the matching sub-matrix row.
    template <bool Homo, typename IndexT, typename EdgeT, typename SubT>
    __global__ void Gather(
        const float *__restrict__ weights, const IndexT *__restrict__ indices,
        const EdgeT *__restrict__ indptr, const EdgeT *__restrict__ counts,
        const SubT *__restrict__ sub_indptr, const SubT *__restrict__ sub_base,
        SubT *__restrict__ sub_indices, float *__restrict__ sub_values,
        int64_t rows, int64_t segments)
    {
        const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (row >= rows)
            return;
        const float shared_weight = Homo ? __ldg(weights) : 0.0f;
        int64_t edge = __ldg(indptr + row);
        for (int64_t segment = 0; segment < segments; ++segment)
        {
            const int64_t count = __ldg(counts + row * segments + segment);
            if (count == 0)
                continue;
            const int64_t dst = static_cast<int64_t>(sub_base[segment]) +
                                static_cast<int64_t>(sub_indptr[segment * (rows + 1) + row]);
            for (int64_t i = 0; i < count; ++i)
            {
                sub_indices[dst + i] = static_cast<SubT>(__ldg(indices + edge + i));
                sub_values[dst + i] = Homo ? shared_weight : __ldg(weights + edge + i);
            }
            edge += count;
        }
    }

    template <bool Homo, typename IndexT, typename EdgeT, typename SubT>
    void Launch(
        const float *weights, const IndexT *indices, const EdgeT *indptr, const EdgeT *counts,
        SubT *sub_indptr, SubT *sub_base, SubT *sub_indices, float *sub_values,
        int64_t rows, int64_t segments, cudaStream_t stream)
    {
        ColumnScan<EdgeT, SubT><<<static_cast<unsigned>(segments), kBlockSize, 0, stream>>>(
            counts, sub_indptr, rows, segments);
        BE_CUDA_CHECK(cudaGetLastError());
        BaseScan<SubT><<<1, 1, 0, stream>>>(sub_indptr, sub_base, rows, segments);
        BE_CUDA_CHECK(cudaGetLastError());
        if (rows == 0)
            return;
        const unsigned blocks = static_cast<unsigned>((rows + kBlockSize - 1) / kBlockSize);
        Gather<Homo, IndexT, EdgeT, SubT><<<blocks, kBlockSize, 0, stream>>>(
            weights, indices, indptr, counts, sub_indptr, sub_base, sub_indices, sub_values,
            rows, segments);
        BE_CUDA_CHECK(cudaGetLastError());
    }

    static bool IsIndex(BE::DType dtype)
    {
        return dtype == BE::DType::Int32 || dtype == BE::DType::Int64;
    }

} // namespace region_delay_split

// @BE region_delay_split_f32
void region_delay_split_f32(
    const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,
    const BE::Tensor counts, BE::Tensor sub_indptr, BE::Tensor sub_base,
    BE::Tensor sub_indices, BE::Tensor sub_values, int64_t stream)
{
    using namespace region_delay_split;
    BE_CHECK(weights.ndim() == 1 && weights.dtype() == BE::DType::Float32);
    BE_CHECK(indices.ndim() == 1 && IsIndex(indices.dtype()));
    BE_CHECK(indptr.ndim() == 1 && indptr.numel() > 0 && IsIndex(indptr.dtype()));
    const int64_t rows = indptr.numel() - 1;
    const int64_t nnz = indices.numel();
    BE_CHECK(weights.numel() == 1 || weights.numel() == nnz);
    BE_CHECK(counts.ndim() == 2 && counts.shape(0) == rows && counts.dtype() == indptr.dtype());
    const int64_t segments = counts.shape(1);
    BE_CHECK(segments > 0 && segments <= std::numeric_limits<int32_t>::max());
    BE_CHECK(IsIndex(sub_indices.dtype()));
    BE_CHECK(sub_indices.dtype() != BE::DType::Int32 || nnz <= std::numeric_limits<int32_t>::max());
    BE_CHECK(sub_indptr.ndim() == 2 && sub_indptr.shape(0) == segments && sub_indptr.shape(1) == rows + 1);
    BE_CHECK(sub_base.ndim() == 1 && sub_base.numel() == segments + 1);
    BE_CHECK(sub_indices.ndim() == 1 && sub_indices.numel() == nnz);
    BE_CHECK(sub_values.ndim() == 1 && sub_values.numel() == nnz && sub_values.dtype() == BE::DType::Float32);
    BE_CHECK(sub_indptr.dtype() == sub_indices.dtype() && sub_base.dtype() == sub_indices.dtype());
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, EdgeT, {
        BE_DISPATCH_SIGNED_INDEX(sub_indices.dtype(), SubT, {
            if (weights.numel() == 1)
                Launch<true, IndexT, EdgeT, SubT>(
                    weights.data_ptr<const float>(), indices.data_ptr<const IndexT>(),
                    indptr.data_ptr<const EdgeT>(), counts.data_ptr<const EdgeT>(),
                    sub_indptr.data_ptr<SubT>(), sub_base.data_ptr<SubT>(),
                    sub_indices.data_ptr<SubT>(), sub_values.data_ptr<float>(),
                    rows, segments, cuda_stream);
            else
                Launch<false, IndexT, EdgeT, SubT>(
                    weights.data_ptr<const float>(), indices.data_ptr<const IndexT>(),
                    indptr.data_ptr<const EdgeT>(), counts.data_ptr<const EdgeT>(),
                    sub_indptr.data_ptr<SubT>(), sub_base.data_ptr<SubT>(),
                    sub_indices.data_ptr<SubT>(), sub_values.data_ptr<float>(),
                    rows, segments, cuda_stream);
        });
    });
}
