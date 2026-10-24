#pragma once
#include <cstdint>

// CPU input arrays are contiguous; events are logical [rows, steps], column-major.
struct DelayBenchmarkInput
{
    int64_t rows, cols, steps, segments, slots, nnz, weight_count, capacity;
    int32_t index_bits, edge_bits, ring_bits;
    const float *weights;
    const void *indices;
    const void *indptr;
    const void *counts;
    const void *offsets;
    const int32_t *row_tasks;
    const float *events;
};

struct DelayBenchmarkStats
{
    double warmup_total_ms, timed_total_ms, timed_mean_ms;
    double timed_std_ms, timed_min_ms, timed_max_ms;
    int64_t duration_steps, segment_count, spmv_calls_per_run;
};

extern "C"
{
    int create_delay_benchmark(const DelayBenchmarkInput *, void **, char *, int);
    void destroy_delay_benchmark(void *);
    int benchmark_delay_hybrid(void *, int, int, DelayBenchmarkStats *, float *, float *, char *, int);
    int benchmark_delay_cusparse(void *, int, int, DelayBenchmarkStats *, float *, float *, char *, int);
}
