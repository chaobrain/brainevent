// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#include "brainevent/common.h"
#include "cuda_common.h"
#include <cusparse.h>
#include <limits>
#include <type_traits>
#include <unordered_map>

#define BE_CUSPARSE_CHECK(expression)                                 \
    do                                                                \
    {                                                                 \
        const cusparseStatus_t status = (expression);                 \
        BE_CHECK(status == CUSPARSE_STATUS_SUCCESS)                   \
            << #expression << ": " << cusparseGetErrorString(status); \
    } while (0)

namespace
{
    // One handle per (thread, device). A handle carries the stream it was last
    // bound to, so a handle shared by XLA's execution threads would need a lock
    // held across the whole call, serializing concurrent streams. Handles are
    // created once per thread and deliberately never destroyed: thread exit can
    // follow CUDA context teardown, and XLA's execution threads are long-lived.
    cusparseHandle_t handle_for_device()
    {
        thread_local std::unordered_map<int, cusparseHandle_t> handles;
        int device = 0;
        BE_CUDA_CHECK(cudaGetDevice(&device));
        cusparseHandle_t &handle = handles[device];
        if (handle == nullptr)
            BE_CUSPARSE_CHECK(cusparseCreate(&handle));
        return handle;
    }

    struct SDDMMPlan
    {
        cusparseDnMatDescr_t left = nullptr;
        cusparseDnMatDescr_t right = nullptr;
        cusparseSpMatDescr_t sampled = nullptr;
        void *workspace = nullptr;
        cudaStream_t stream;

        explicit SDDMMPlan(cudaStream_t stream) : stream(stream) {}
        SDDMMPlan(const SDDMMPlan &) = delete;
        SDDMMPlan &operator=(const SDDMMPlan &) = delete;

        ~SDDMMPlan()
        {
            if (workspace)
                cudaFreeAsync(workspace, stream);
            if (sampled)
                cusparseDestroySpMat(sampled);
            if (right)
                cusparseDestroyDnMat(right);
            if (left)
                cusparseDestroyDnMat(left);
        }
    };

    template <typename ValueT>
    void sampled_product(const BE::Tensor &left, const BE::Tensor &right,
                         const BE::Tensor &indices, const BE::Tensor &indptr,
                         const BE::Tensor &output_in, BE::Tensor &output,
                         int64_t stream)
    {
        constexpr bool is_double = std::is_same_v<ValueT, double>;
        constexpr BE::DType value_dtype = is_double ? BE::DType::Float64 : BE::DType::Float32;
        constexpr cudaDataType cuda_dtype = is_double ? CUDA_R_64F : CUDA_R_32F;
        BE_CHECK(left.dtype() == value_dtype && right.dtype() == value_dtype &&
                 output_in.dtype() == value_dtype && output.dtype() == value_dtype)
            << "cusparse SDDMM: invalid value dtype";
        BE_CHECK(indices.dtype() == indptr.dtype() &&
                 (indptr.dtype() == BE::DType::Int32 || indptr.dtype() == BE::DType::Int64))
            << "cusparse SDDMM: invalid CSR index dtype";
        BE_CHECK(left.ndim() == 2 && right.ndim() == 2 && indices.ndim() == 1 &&
                 indptr.ndim() == 1 && output_in.ndim() == 1 && output.ndim() == 1)
            << "cusparse SDDMM: invalid rank";
        const int64_t rows = left.size(0), cols = right.size(0), batch = left.size(1);
        const int64_t nnz = indices.numel();
        BE_CHECK(rows > 0 && cols > 0 && batch > 0 && right.size(1) == batch &&
                 indptr.numel() == rows + 1 && output_in.numel() == nnz && output.numel() == nnz)
            << "cusparse SDDMM: invalid shape";
        BE_CHECK(cols > std::numeric_limits<int64_t>::max() / rows || nnz <= rows * cols)
            << "cusparse SDDMM: nnz exceeds rows * cols";
        BE_CHECK(output_in.data_ptr<const ValueT>() == output.data_ptr<ValueT>())
            << "cusparse SDDMM: FFI requires output alias";

        const cusparseHandle_t handle = handle_for_device();
        SDDMMPlan plan(reinterpret_cast<cudaStream_t>(stream));
        BE_CUSPARSE_CHECK(cusparseSetStream(handle, plan.stream));
        BE_CUSPARSE_CHECK(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
        BE_CUSPARSE_CHECK(cusparseCreateDnMat(
            &plan.left, rows, batch, batch, const_cast<ValueT *>(left.data_ptr<const ValueT>()),
            cuda_dtype, CUSPARSE_ORDER_ROW));
        BE_CUSPARSE_CHECK(cusparseCreateDnMat(
            &plan.right, cols, batch, batch, const_cast<ValueT *>(right.data_ptr<const ValueT>()),
            cuda_dtype, CUSPARSE_ORDER_ROW));
        BE_DISPATCH_SIGNED_INDEX(indptr.dtype(), OffsetT, {
            constexpr cusparseIndexType_t offset_type =
                sizeof(OffsetT) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
            BE_CUSPARSE_CHECK(cusparseCreateCsr(
                &plan.sampled, rows, cols, nnz,
                const_cast<OffsetT *>(indptr.data_ptr<const OffsetT>()),
                const_cast<OffsetT *>(indices.data_ptr<const OffsetT>()), output.data_ptr<ValueT>(),
                offset_type, offset_type, CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));
        });

        const ValueT alpha = ValueT(1), beta = ValueT(0);
        constexpr cusparseOperation_t left_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cusparseOperation_t right_op = CUSPARSE_OPERATION_TRANSPOSE;
        constexpr cusparseSDDMMAlg_t algorithm = CUSPARSE_SDDMM_ALG_DEFAULT;
        size_t workspace_bytes = 0;
        BE_CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
            handle, left_op, right_op, &alpha, plan.left, plan.right, &beta,
            plan.sampled, cuda_dtype, algorithm, &workspace_bytes));
        // The kernel FFI exposes no XLA scratch allocator, so the workspace
        // comes from the device's stream-ordered pool on the call's stream; it
        // is freed on the same stream, after the SDDMM that uses it. It is not
        // part of XLA's memory accounting, but is usually zero or a few bytes.
        if (workspace_bytes)
            BE_CUDA_CHECK(cudaMallocAsync(&plan.workspace, workspace_bytes, plan.stream));
        // cusparseSDDMM_preprocess is optional: it only pays off when its
        // analysis is reused across calls, and every call here builds fresh
        // descriptors, so it would add a pass without speeding anything up.
        BE_CUSPARSE_CHECK(cusparseSDDMM(
            handle, left_op, right_op, &alpha, plan.left, plan.right, &beta,
            plan.sampled, cuda_dtype, algorithm, plan.workspace));
    }
}

// @BE cusparse_sddmm_f32
void cusparse_sddmm_f32(const BE::Tensor left, const BE::Tensor right,
                        const BE::Tensor indices, const BE::Tensor indptr,
                        const BE::Tensor output_in, BE::Tensor output, int64_t stream)
{
    sampled_product<float>(left, right, indices, indptr, output_in, output, stream);
}

// @BE cusparse_sddmm_f64
void cusparse_sddmm_f64(const BE::Tensor left, const BE::Tensor right,
                        const BE::Tensor indices, const BE::Tensor indptr,
                        const BE::Tensor output_in, BE::Tensor output, int64_t stream)
{
    sampled_product<double>(left, right, indices, indptr, output_in, output, stream);
}