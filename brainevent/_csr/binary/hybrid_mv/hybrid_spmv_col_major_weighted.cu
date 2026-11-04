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

// Column-major hetero weighted SpMV; weights are read through perm.
#include "hybrid_spmv_col_major_impl.cuh"

#define DEFINE_HYBRID_SPMV_COL_MAJOR_WEIGHTED(SUFFIX, WEIGHT_T) \
    void hybrid_spmv_col_major_weighted_hetero##SUFFIX( \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr, \
        const BE::Tensor perm, const BE::Tensor vector, \
        const BE::Tensor task_begin_scratch, const BE::Tensor task_end_scratch, \
        const BE::Tensor status_scratch, BE::Tensor output, \
        BE::Tensor task_begin_output, BE::Tensor task_end_output, \
        BE::Tensor status_output, int64_t task_capacity, int64_t stream) \
    { \
        HybridSpmvColMajorHetero<WEIGHT_T, WEIGHT_T, true>( \
            "hybrid_spmv_col_major_weighted_hetero" #SUFFIX, weights, indices, indptr, perm, vector, \
            task_begin_scratch, task_end_scratch, status_scratch, \
            output, task_begin_output, task_end_output, status_output, task_capacity, stream); \
    }

// @BE hybrid_spmv_col_major_weighted_hetero_f16
DEFINE_HYBRID_SPMV_COL_MAJOR_WEIGHTED(_f16, __half)
// @BE hybrid_spmv_col_major_weighted_hetero_bf16
DEFINE_HYBRID_SPMV_COL_MAJOR_WEIGHTED(_bf16, __nv_bfloat16)
// @BE hybrid_spmv_col_major_weighted_hetero_f32
DEFINE_HYBRID_SPMV_COL_MAJOR_WEIGHTED(_f32, float)
// @BE hybrid_spmv_col_major_weighted_hetero_f64
DEFINE_HYBRID_SPMV_COL_MAJOR_WEIGHTED(_f64, double)

#undef DEFINE_HYBRID_SPMV_COL_MAJOR_WEIGHTED
