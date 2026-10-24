// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Accumulate all K delay sub-matrices in one step with cuSPARSE.
// Sub-matrices come from region_delay_split.cu. StepPlan owns the K loop:
// each nonempty segment runs SpMV into scratch, then RingAdd uses the same offset
// as the hybrid Extract kernel: head * N + delay_off[k], wrapped by L * N.
// The scratch vector keeps the head on device (no host sync) and gives
// cuSPARSE a 16-byte aligned output. Existing ring contents are retained.

#include "cuda_common.h"
#include "brainevent/common.h"
#include <cusparse.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#define BE_CUSPARSE_CHECK(expr)                                              \
    do                                                                       \
    {                                                                        \
        const cusparseStatus_t be_sparse_status = (expr);                    \
        BE_CHECK(be_sparse_status == CUSPARSE_STATUS_SUCCESS)                \
            << #expr << ": " << cusparseGetErrorString(be_sparse_status);    \
    } while (0)

namespace region_delay_cusparse
{

    constexpr int kBlockSize = 256;

    template <typename SubT>
    constexpr cusparseIndexType_t kIndexType =
        sizeof(SubT) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;

    template <typename RingT>
    __global__ void RingAdd(float *__restrict__ buffer, const float *__restrict__ scratch,
                            const RingT *__restrict__ delay_off, const int32_t *__restrict__ head,
                            int64_t segment, int64_t cols, int64_t slots)
    {
        const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (column >= cols)
            return;
        const int32_t current_head = __ldg(head);
        if (current_head < 0 || current_head >= slots)
            return;
        const int64_t ring_size = slots * cols;
        int64_t out_off = static_cast<int64_t>(current_head) * cols +
                          static_cast<int64_t>(__ldg(delay_off + segment));
        if (out_off >= ring_size)
            out_off -= ring_size;
        buffer[out_off + column] += scratch[column];
    }

    // One handle per device; callers hold the returned lock while enqueueing.
    inline cusparseHandle_t Handle(std::unique_lock<std::mutex> &lock)
    {
        static std::mutex mutex;
        static std::unordered_map<int, cusparseHandle_t> handles;
        int device = 0;
        BE_CUDA_CHECK(cudaGetDevice(&device));
        lock = std::unique_lock<std::mutex>(mutex);
        cusparseHandle_t &handle = handles[device];
        if (handle == nullptr)
            BE_CUSPARSE_CHECK(cusparseCreate(&handle));
        return handle;
    }

    // Prepare one nonempty CSR segment once; reuse its descriptors/workspace.
    // The handle, matrix arrays and scratch must outlive this plan.
    template <typename SubT>
    class SegmentPlan
    {
        cusparseHandle_t handle_;
        cudaStream_t stream_;
        cusparseSpMatDescr_t mat_ = nullptr;
        cusparseDnVecDescr_t x_ = nullptr;
        cusparseDnVecDescr_t y_ = nullptr;
        void *workspace_ = nullptr;

        void Release() noexcept
        {
            if (workspace_) cudaFreeAsync(workspace_, stream_);
            if (y_) cusparseDestroyDnVec(y_);
            if (x_) cusparseDestroyDnVec(x_);
            if (mat_) cusparseDestroySpMat(mat_);
        }

    public:
        SegmentPlan(cusparseHandle_t handle, const SubT *row_ptr,
                    const SubT *col_ind, const float *values, const float *x,
                    float *scratch, int64_t rows, int64_t cols, int64_t nnz,
                    cudaStream_t stream)
            : handle_(handle), stream_(stream)
        {
            try
            {
                BE_CUSPARSE_CHECK(cusparseCreateCsr(
                    &mat_, rows, cols, nnz, const_cast<SubT *>(row_ptr),
                    const_cast<SubT *>(col_ind), const_cast<float *>(values),
                    kIndexType<SubT>, kIndexType<SubT>, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
                BE_CUSPARSE_CHECK(cusparseCreateDnVec(&x_, rows, const_cast<float *>(x), CUDA_R_32F));
                BE_CUSPARSE_CHECK(cusparseCreateDnVec(&y_, cols, scratch, CUDA_R_32F));
                const float alpha = 1.0f, beta = 0.0f;
                size_t size = 0;
                BE_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                    handle_, CUSPARSE_OPERATION_TRANSPOSE, &alpha, mat_, x_, &beta, y_,
                    CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &size));
                if (size) BE_CUDA_CHECK(cudaMallocAsync(&workspace_, size, stream_));
            }
            catch (...)
            {
                Release();
                throw;
            }
        }

        SegmentPlan(const SegmentPlan &) = delete;
        SegmentPlan &operator=(const SegmentPlan &) = delete;
        ~SegmentPlan() { Release(); }

        template <typename RingT>
        void Project(const float *x, const RingT *delay_off, const int32_t *head,
                     float *buffer, float *scratch, int64_t cols, int64_t slots,
                     int64_t segment)
        {
            BE_CUSPARSE_CHECK(cusparseDnVecSetValues(x_, const_cast<float *>(x)));
            const float alpha = 1.0f, beta = 0.0f;
            BE_CUSPARSE_CHECK(cusparseSpMV(
                handle_, CUSPARSE_OPERATION_TRANSPOSE, &alpha, mat_, x_, &beta, y_,
                CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, workspace_));
            const unsigned blocks = static_cast<unsigned>((cols + kBlockSize - 1) / kBlockSize);
            RingAdd<RingT><<<blocks, kBlockSize, 0, stream_>>>(
                buffer, scratch, delay_off, head, segment, cols, slots);
            BE_CUDA_CHECK(cudaGetLastError());
        }
    };

    // Prepare all K sub-matrices from the four split outputs. Only construction
    // reads the small prefix array back to the host; native benchmarks retain
    // this plan outside their timing loop. Device arrays, scratch, stream and
    // handle must outlive the plan. Calls use the handle on the supplied stream.
    template <typename SubT>
    class StepPlan
    {
        std::vector<std::unique_ptr<SegmentPlan<SubT>>> plans_;
        float *scratch_;
        int64_t cols_, slots_, active_segments_ = 0;

    public:
        StepPlan(cusparseHandle_t handle, const SubT *sub_indptr,
                 const SubT *sub_base, const SubT *sub_indices, const float *sub_values,
                 const float *x, float *scratch, int64_t rows, int64_t cols,
                 int64_t segments, int64_t slots, int64_t total_nnz, cudaStream_t stream)
            : scratch_(scratch), cols_(cols), slots_(slots)
        {
            BE_CHECK(rows >= 0 && rows <= std::numeric_limits<int32_t>::max());
            BE_CHECK(cols >= 0 && cols <= std::numeric_limits<int32_t>::max());
            BE_CHECK(segments > 0 && segments <= std::numeric_limits<int32_t>::max());
            BE_CHECK(slots > 0 && slots <= std::numeric_limits<int32_t>::max());
            BE_CHECK(total_nnz >= 0);
            BE_CUSPARSE_CHECK(cusparseSetStream(handle, stream));
            std::vector<SubT> bases(segments + 1);
            BE_CUDA_CHECK(cudaMemcpyAsync(bases.data(), sub_base,
                bases.size() * sizeof(SubT), cudaMemcpyDeviceToHost, stream));
            BE_CUDA_CHECK(cudaStreamSynchronize(stream));
            BE_CHECK(bases.front() == 0 && bases.back() == total_nnz)
                << "split boundaries must span the packed arrays";
            plans_.resize(segments);
            for (int64_t k = 0; k < segments; ++k)
            {
                const int64_t base = bases[k], end = bases[k + 1];
                BE_CHECK(base >= 0 && end >= base && end <= total_nnz)
                    << "split boundaries must be nondecreasing";
                const int64_t nnz = end - base;
                BE_CHECK(nnz <= rows * cols) << "delay segment nnz exceeds rows * cols";
                if (nnz == 0) continue;
                plans_[k] = std::make_unique<SegmentPlan<SubT>>(
                    handle, sub_indptr + k * (rows + 1), sub_indices + base,
                    sub_values + base, x, scratch, rows, cols, nnz, stream);
                ++active_segments_;
            }
        }

        StepPlan(const StepPlan &) = delete;
        StepPlan &operator=(const StepPlan &) = delete;
        int64_t active_segments() const { return active_segments_; }

        // One call is one step: reuse x/head for every delay sub-matrix.
        // Consume, clear and advance head only after this method returns.
        template <typename RingT>
        void Project(const float *x, const RingT *delay_off, const int32_t *head, float *buffer)
        {
            for (size_t k = 0; k < plans_.size(); ++k)
                if (plans_[k])
                    plans_[k]->Project(x, delay_off, head, buffer, scratch_, cols_, slots_, k);
        }
    };

    // Whole-step FFI convenience entry; native benchmarks reuse StepPlan.
    // This correctness wrapper creates a temporary plan and is not timed.
    template <typename SubT, typename RingT>
    void Launch(
        const SubT *sub_indptr, const SubT *sub_base, const SubT *sub_indices,
        const float *sub_values, const float *x,
        const RingT *delay_off, const int32_t *head, float *buffer, float *scratch,
        int64_t rows, int64_t cols, int64_t segments, int64_t slots, int64_t total_nnz,
        cudaStream_t stream)
    {
        std::unique_lock<std::mutex> lock;
        const cusparseHandle_t handle = Handle(lock);
        BE_CUSPARSE_CHECK(cusparseSetStream(handle, stream));
        StepPlan<SubT> plan(handle, sub_indptr, sub_base, sub_indices, sub_values,
                           x, scratch, rows, cols, segments, slots, total_nnz, stream);
        plan.Project(x, delay_off, head, buffer);
    }

    static bool IsIndex(BE::DType dtype)
    {
        return dtype == BE::DType::Int32 || dtype == BE::DType::Int64;
    }

} // namespace region_delay_cusparse

// @BE region_delay_cusparse_f32
void region_delay_cusparse_f32(
    const BE::Tensor sub_indptr, const BE::Tensor sub_base,
    const BE::Tensor sub_indices, const BE::Tensor sub_values,
    const BE::Tensor x, const BE::Tensor delay_off, const BE::Tensor head,
    const BE::Tensor buffer, const BE::Tensor scratch,
    BE::Tensor output, BE::Tensor scratch_out,
    int64_t stream)
{
    using namespace region_delay_cusparse;
    BE_CHECK(sub_indptr.ndim() == 2 && IsIndex(sub_indptr.dtype()));
    BE_CHECK(sub_indices.ndim() == 1 && sub_indices.dtype() == sub_indptr.dtype());
    BE_CHECK(sub_values.ndim() == 1 && sub_values.numel() == sub_indices.numel());
    BE_CHECK(sub_values.dtype() == BE::DType::Float32);
    const int64_t segments = sub_indptr.shape(0);
    const int64_t rows = sub_indptr.shape(1) - 1;
    BE_CHECK(sub_base.ndim() == 1 && sub_base.numel() == segments + 1);
    BE_CHECK(sub_base.dtype() == sub_indptr.dtype());
    BE_CHECK(x.ndim() == 1 && x.numel() == rows && x.dtype() == BE::DType::Float32);
    BE_CHECK(delay_off.ndim() == 1 && delay_off.numel() == segments && IsIndex(delay_off.dtype()));
    BE_CHECK(head.ndim() == 1 && head.numel() == 1 && head.dtype() == BE::DType::Int32);
    BE_CHECK(buffer.ndim() == 2 && buffer.dtype() == BE::DType::Float32);
    BE_CHECK(output.ndim() == 2 && output.dtype() == buffer.dtype());
    BE_CHECK(output.shape(0) == buffer.shape(0) && output.shape(1) == buffer.shape(1));
    BE_CHECK(buffer.data_ptr() == output.data_ptr()) << "ring requires input/output aliasing";
    const int64_t slots = buffer.shape(0);
    const int64_t cols = buffer.shape(1);
    BE_CHECK(slots > 0 && slots <= std::numeric_limits<int32_t>::max());
    BE_CHECK(cols <= std::numeric_limits<int32_t>::max());
    BE_CHECK(scratch.ndim() == 1 && scratch.numel() == cols && scratch.dtype() == BE::DType::Float32);
    BE_CHECK(scratch_out.ndim() == 1 && scratch_out.numel() == cols && scratch_out.dtype() == BE::DType::Float32);
    BE_CHECK(scratch.data_ptr() == scratch_out.data_ptr()) << "workspace requires aliasing";
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    BE_DISPATCH_SIGNED_INDEX(sub_indptr.dtype(), SubT, {
        BE_DISPATCH_SIGNED_INDEX(delay_off.dtype(), RingT, {
            Launch<SubT, RingT>(
                sub_indptr.data_ptr<const SubT>(), sub_base.data_ptr<const SubT>(),
                sub_indices.data_ptr<const SubT>(), sub_values.data_ptr<const float>(),
                x.data_ptr<const float>(), delay_off.data_ptr<const RingT>(),
                head.data_ptr<const int32_t>(), output.data_ptr<float>(),
                scratch_out.data_ptr<float>(), rows, cols, segments, slots,
                sub_indices.numel(), cuda_stream);
        });
    });
}
