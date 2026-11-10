// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Column-major hetero binary SpMV; weights are read through perm.
#include "hybrid_spmv_col_major_impl.cuh"

#define DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY(SUFFIX, WEIGHT_T, EVENT_T) \
    void hybrid_spmv_col_major_binary_hetero##SUFFIX( \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr, \
        const BE::Tensor perm, const BE::Tensor vector, \
        const BE::Tensor task_begin_scratch, const BE::Tensor task_end_scratch, \
        const BE::Tensor status_scratch, BE::Tensor output, \
        BE::Tensor task_begin_output, BE::Tensor task_end_output, \
        BE::Tensor status_output, int64_t task_capacity, int64_t stream) \
    { \
        HybridSpmvColMajorHetero<WEIGHT_T, EVENT_T, false>( \
            "hybrid_spmv_col_major_binary_hetero" #SUFFIX, weights, indices, indptr, perm, vector, \
            task_begin_scratch, task_end_scratch, status_scratch, \
            output, task_begin_output, task_end_output, status_output, task_capacity, stream); \
    }

// @BE hybrid_spmv_col_major_binary_hetero_f16_bool
DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY(_f16_bool, __half, int8_t)
// @BE hybrid_spmv_col_major_binary_hetero_f16_float
DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY(_f16_float, __half, float)
// @BE hybrid_spmv_col_major_binary_hetero_bf16_bool
DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY(_bf16_bool, __nv_bfloat16, int8_t)
// @BE hybrid_spmv_col_major_binary_hetero_bf16_float
DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY(_bf16_float, __nv_bfloat16, float)
// @BE hybrid_spmv_col_major_binary_hetero_f32_bool
DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY(_f32_bool, float, int8_t)
// @BE hybrid_spmv_col_major_binary_hetero_f32_float
DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY(_f32_float, float, float)
// @BE hybrid_spmv_col_major_binary_hetero_f64_bool
DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY(_f64_bool, double, int8_t)
// @BE hybrid_spmv_col_major_binary_hetero_f64_float
DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY(_f64_float, double, float)

#undef DEFINE_HYBRID_SPMV_COL_MAJOR_BINARY
