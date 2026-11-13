// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Weighted hybrid SpMV: active rows scatter weights * event; events share the weight dtype.
// Shared kernels, launchers and FFI validation live in hybrid_spmv_impl.cuh.
#include "hybrid_spmv_impl.cuh"

#define DEFINE_HYBRID_SPMV_WEIGHTED(SUFFIX, WEIGHT_T, MODE) \
    void binary_csrmv_wat_hybrid##SUFFIX( \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr, \
        const BE::Tensor vector, const BE::Tensor task_begin_scratch, \
        const BE::Tensor task_end_scratch, const BE::Tensor status_scratch, \
        BE::Tensor output, BE::Tensor task_begin_output, BE::Tensor task_end_output, \
        BE::Tensor status_output, int64_t task_capacity, int64_t stream) \
    { \
        HybridSpmv##MODE<WEIGHT_T, WEIGHT_T, true>( \
            "binary_csrmv_wat_hybrid" #SUFFIX, weights, indices, indptr, vector, \
            task_begin_scratch, task_end_scratch, status_scratch, \
            output, task_begin_output, task_end_output, status_output, task_capacity, stream); \
    }

// @BE binary_csrmv_wat_hybrid_hetero_f32_wv
DEFINE_HYBRID_SPMV_WEIGHTED(_hetero_f32_wv, float, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_f64_wv
DEFINE_HYBRID_SPMV_WEIGHTED(_hetero_f64_wv, double, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_f16_wv
DEFINE_HYBRID_SPMV_WEIGHTED(_hetero_f16_wv, __half, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_bf16_wv
DEFINE_HYBRID_SPMV_WEIGHTED(_hetero_bf16_wv, __nv_bfloat16, Hetero)
// @BE binary_csrmv_wat_hybrid_homo_f32_wv
DEFINE_HYBRID_SPMV_WEIGHTED(_homo_f32_wv, float, Homo)
// @BE binary_csrmv_wat_hybrid_homo_f64_wv
DEFINE_HYBRID_SPMV_WEIGHTED(_homo_f64_wv, double, Homo)
// @BE binary_csrmv_wat_hybrid_homo_f16_wv
DEFINE_HYBRID_SPMV_WEIGHTED(_homo_f16_wv, __half, Homo)
// @BE binary_csrmv_wat_hybrid_homo_bf16_wv
DEFINE_HYBRID_SPMV_WEIGHTED(_homo_bf16_wv, __nv_bfloat16, Homo)

#undef DEFINE_HYBRID_SPMV_WEIGHTED
