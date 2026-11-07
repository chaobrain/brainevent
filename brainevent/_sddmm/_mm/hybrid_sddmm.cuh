#ifndef DEVING_BATCHED_V3_HYBRID_SDDMM_CUH
#define DEVING_BATCHED_V3_HYBRID_SDDMM_CUH

// Public API. Link hybrid_sddmm_binary.cu (launch) and/or hybrid_sddmm_weighted.cu (launch_weighted);
// define BE_SDDMM_STANDALONE when compiling them to omit BrainEvent/FFI bindings.
// Planning and sizing are inline (hybrid_sddmm_plan.cuh), so both objects can share one program.
// This header also supports ordinary C++ callers.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

namespace batched_v3
{
    constexpr int kBlock = 256, kRows = 4096, kSegments = 128, kTask = 512;
    constexpr unsigned kFull = 0xffffffffU;
    struct Plan
    {
        int group = 32, chunk = 1, batches = 1, sms = 1;
        int64_t short_count = 0, long_count = 0, capacity = 0;
    };
} // namespace batched_v3

// make_plan, long_mask, groups, list_entries, workspace_bytes, accumulator_bytes.
#include "hybrid_sddmm_plan.cuh"

namespace batched_v3
{
    // Binary accumulation: dweight[j] += sum_b ct[b, indices[j]] for each
    // active events[b, row] != 0. Negative and NaN events are active.
    // ValueT: __half, __nv_bfloat16, float, double; EventT: int8_t (bool/int8), float.
    // ValueT=double additionally supports EventT=double. The f64 Bool and Int8
    // FFI entries share the same native int8_t specialization (activity != 0).
    // IndexT and OffsetT independently support int32_t/int64_t.
    // All pointers below refer to device memory. dweight has nnz initialized
    // values; it is never cleared, and inactive rows retain their exact bits.
    // Half types accumulate in float32 scratch and round once when added to
    // dweight; float/double contributions are added directly to dweight.
    // selected holds list_entries(rows, p) int32 entries; workspace must be
    // CUDA-allocation aligned and hold workspace_bytes(p) +
    // accumulator_bytes<ValueT>(nnz) bytes. Its first 2*p.batches int32 words
    // are {selected_count, overflow_flag} pairs, reset on every call. Read them
    // after stream completion; overflow means capacity was insufficient and
    // the result is incomplete. mask has ceil(rows/32) uint32 entries.
    // No allocation or host synchronization occurs. The caller owns buffers,
    // validates CSR/column bounds and dimensions, and orders reuse by stream.
    // Use a valid plan from make_plan; returns CUDA memset/launch status.
    template <class ValueT, class EventT, class IndexT, class OffsetT>
    cudaError_t launch(const EventT *events, const ValueT *ct, const IndexT *indices, const OffsetT *ptr,
                       ValueT *dweight, const uint32_t *mask, int32_t *selected, void *workspace,
                       int rows, int cols, int64_t nnz, const Plan &p, cudaStream_t stream = nullptr);

    // Weighted accumulation (hybrid_sddmm_weighted.cu): dweight[j] += sum_b events[b, row] *
    // ct[b, indices[j]] over events[b, row] != 0; events, ct and dweight share ValueT
    // (__half, __nv_bfloat16, float or double). Half products and sums use float32;
    // plan, mask, selected and the dweight contract are those of launch.
    // The workspace adds a task value queue (the vw standard: each long-row task stores its
    // event value, short rows re-read events[b, row]), so it must hold
    // workspace_bytes(p, sizeof(ValueT)) + accumulator_bytes<ValueT>(nnz) bytes.
    template <class ValueT, class IndexT, class OffsetT>
    cudaError_t launch_weighted(const ValueT *events, const ValueT *ct, const IndexT *indices, const OffsetT *ptr,
                                ValueT *dweight, const uint32_t *mask, int32_t *selected, void *workspace,
                                int rows, int cols, int64_t nnz, const Plan &p, cudaStream_t stream = nullptr);
} // namespace batched_v3

#endif // DEVING_BATCHED_V3_HYBRID_SDDMM_CUH
