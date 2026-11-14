// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef BRAINEVENT_HYBRID_MV_SCHEDULE_CUH
#define BRAINEVENT_HYBRID_MV_SCHEDULE_CUH
// Scheduler constants of the hybrid MV kernels (CSR scatter, column-major
// scatter and the sampled SDDMV gathers). Python's MVSchedule.compile_flags()
// passes the four BE_HYBRID_* values per module; the defaults below match
// MVSchedule's defaults. Light rows (nnz <= kTprThreshold) are handled by one
// thread; heavy rows are split into tasks of at most kTaskNnz edges.
#include <cstdint>

#ifndef BE_HYBRID_BLOCK_SIZE
#define BE_HYBRID_BLOCK_SIZE 512
#endif

#ifndef BE_HYBRID_FIXED_SCATTER_BLOCKS
#define BE_HYBRID_FIXED_SCATTER_BLOCKS 1024
#endif

#ifndef BE_HYBRID_TPR_THRESHOLD
#define BE_HYBRID_TPR_THRESHOLD 128
#endif

#ifndef BE_HYBRID_TASK_NNZ
#define BE_HYBRID_TASK_NNZ 512
#endif

namespace hybrid_mv
{
    constexpr int kBlockSize = BE_HYBRID_BLOCK_SIZE;
    constexpr int kFixedScatterBlocks = BE_HYBRID_FIXED_SCATTER_BLOCKS;
    constexpr int kTprThreshold = BE_HYBRID_TPR_THRESHOLD;
    constexpr int kTaskNnz = BE_HYBRID_TASK_NNZ;
    // Layout of the int32[2] status buffer.
    constexpr int kStatusCountIndex = 0;
    constexpr int kStatusOverflowIndex = 1;
    constexpr int64_t kInt32Max = 2147483647LL;

    static_assert(kBlockSize > 0, "BE_HYBRID_BLOCK_SIZE must be positive");
    static_assert(kBlockSize % 32 == 0, "BE_HYBRID_BLOCK_SIZE must be a multiple of 32");
    static_assert(kBlockSize <= 1024, "BE_HYBRID_BLOCK_SIZE must not exceed 1024");
    static_assert(kFixedScatterBlocks > 0, "BE_HYBRID_FIXED_SCATTER_BLOCKS must be positive");
    static_assert(kTprThreshold >= 0, "BE_HYBRID_TPR_THRESHOLD must be non-negative");
    static_assert(kTaskNnz > 0, "BE_HYBRID_TASK_NNZ must be positive");

#ifdef __CUDACC__
    // Number of kTaskNnz-edge tasks for a row (or column) of nnz edges: zero for
    // a light row. A count beyond int32 sets the overflow flag and returns zero.
    template <typename OffsetT>
    __device__ __forceinline__ int32_t TaskChunks(OffsetT nnz, int32_t *__restrict__ status)
    {
        if (nnz <= static_cast<OffsetT>(kTprThreshold))
        {
            return 0;
        }
        OffsetT chunks = 1 + (nnz - 1) / static_cast<OffsetT>(kTaskNnz);
        if (static_cast<int64_t>(chunks) > kInt32Max)
        {
            atomicExch(&status[kStatusOverflowIndex], 1);
            return 0;
        }
        return static_cast<int32_t>(chunks);
    }
#endif
} // namespace hybrid_mv

#endif // BRAINEVENT_HYBRID_MV_SCHEDULE_CUH
