// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef DEVING_BATCHED_V3_HYBRID_SDDMM_PLAN_CUH
#define DEVING_BATCHED_V3_HYBRID_SDDMM_PLAN_CUH

// Host planning and scratch sizing, shared by the binary and weighted operators.
// Everything here is inline so both operator objects can be linked into one program.
// Included by hybrid_sddmm.cuh after Plan; plain C++ compilers can use it.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace batched_v3
{
    inline int64_t multiply(int64_t a, int64_t b)
    {
        if (a < 0 || b < 0 || (b && a > INT64_MAX / b))
            throw std::overflow_error("workspace size overflow");
        return a * b;
    }
    inline int lane_group(int64_t edges, int64_t rows)
    {
        const int64_t q = rows ? (edges / rows + 3) / 4 : 1;
        return q <= 4 ? 4 : q <= 8 ? 8
                        : q <= 16  ? 16
                                   : 32;
    }

    // Host CSR pointers (int64) begin at zero and are nondecreasing; rows/cols
    // are positive and batch is in [1, 65535]. l2 is bytes, sms is the device SM
    // count. chunk=0 selects a chunk from L2; otherwise it is clamped to batch.
    // Invalid dimensions/CSR throw std::invalid_argument; size overflow throws
    // std::overflow_error. Reuse the plan only with the same CSR and dimensions.
    inline Plan make_plan(const int64_t *ptr, int rows, int cols, int batch, int64_t l2, int sms, int chunk = 0)
    {
        if (rows < 1 || cols < 1 || batch < 1 || batch > 65535 || l2 < 1 || sms < 1 || chunk < 0 || chunk > 65535)
            throw std::invalid_argument("invalid plan dimensions");
        Plan p;
        p.batches = batch;
        p.sms = sms;
        p.chunk = std::min(batch, chunk ? chunk : static_cast<int>(std::min<int64_t>(65535, std::max<int64_t>(1, l2 / 16 / cols))));
        int64_t small = 0, edges = 0, tasks = 0;
        if (ptr[0] != 0)
            throw std::invalid_argument("CSR must start at zero");
        for (int r = 0; r < rows; ++r)
        {
            const int64_t n = ptr[r + 1] - ptr[r];
            if (n < 0)
                throw std::invalid_argument("decreasing CSR pointers");
            if (n && n <= kTask)
            {
                ++small;
                edges += n;
            }
        }
        p.group = small ? lane_group(edges, small) : 32;
        for (int r = 0; r < rows; ++r)
        {
            const int64_t n = ptr[r + 1] - ptr[r];
            if (n > 16 * p.group)
            {
                ++p.long_count;
                tasks += (n / kTask) + (n % kTask != 0);
            }
            else if (n)
                ++p.short_count;
        }
        p.capacity = multiply(tasks, p.chunk);
        return p;
    }
    // Bit r % 32 of word r / 32 is set for rows longer than 16 * group.
    inline std::vector<uint32_t> long_mask(const int64_t *ptr, int rows, const Plan &p)
    {
        std::vector<uint32_t> mask((static_cast<int64_t>(rows) + 31) / 32, 0);
        for (int r = 0; r < rows; ++r)
            if (ptr[r + 1] - ptr[r] > 16 * p.group)
                mask[r / 32] |= 1U << (r % 32);
        return mask;
    }
    inline int groups(const Plan &p) { return (p.batches + p.chunk - 1) / p.chunk; }
    // int32 entries of the selected (short-row) list.
    inline int64_t list_entries(int rows, const Plan &p) { return multiply(rows, p.long_count ? p.chunk : p.batches); }
    // Workspace layout: status int32[2B] | counts uint64[G] | begin int64[C] | value[C] | meta int32[C],
    // B batches, G groups, C task capacity. value is the weighted task value queue
    // (value_bytes = sizeof(ValueT)); the binary operator has none (value_bytes = 0). Its region is
    // padded to 4 bytes so meta stays aligned after an odd count of half values.
    inline int64_t queue_region_bytes(int64_t capacity, int64_t value_bytes)
    {
        return (multiply(capacity, value_bytes) + 3) / 4 * 4;
    }
    inline int64_t workspace_bytes(const Plan &p, int64_t value_bytes = 0)
    {
        const int64_t header = static_cast<int64_t>(p.batches) * 8 + groups(p) * 8LL;
        const int64_t queue = queue_region_bytes(p.capacity, value_bytes);
        int64_t payload = multiply(p.capacity, 12);
        if (queue > INT64_MAX - payload)
            throw std::overflow_error("workspace size overflow");
        payload += queue;
        if (payload > INT64_MAX - header)
            throw std::overflow_error("workspace size overflow");
        return header + payload;
    }
    // Float accumulator placed after the workspace: half/bfloat16 values sum in float32,
    // float/double add straight into dweight (hybrid_sddmm_impl.cuh checks this against AccT).
    template <class ValueT>
    inline int64_t accumulator_bytes(int64_t nnz)
    {
        return sizeof(ValueT) == 2 ? multiply(nnz, static_cast<int64_t>(sizeof(float))) : 0;
    }
} // namespace batched_v3

#endif // DEVING_BATCHED_V3_HYBRID_SDDMM_PLAN_CUH
