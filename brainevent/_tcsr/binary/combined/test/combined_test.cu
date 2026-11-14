// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Standalone correctness test for combined::launch (no JAX): random CSR matrices are converted to
// the combined metadata on the host, run on the device and compared with a float64 CPU reference.
// Build: nvcc -std=c++17 -arch=native combined_test.cu -o combined_test (or run combined_test.py)
// Exit status 0 when every case passes; failures are printed.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../combined_kernels.cuh"

#define CHECK_CUDA(call)                                                                        \
    do                                                                                          \
    {                                                                                           \
        const cudaError_t error_ = (call);                                                      \
        if (error_ != cudaSuccess)                                                              \
        {                                                                                       \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error_)); \
            std::exit(2);                                                                       \
        }                                                                                       \
    } while (0)

namespace
{
    struct Case
    {
        const char *name;
        int rows;
        int cols;
        int batch;
        double density; // expected stored entries per (row, column)
        double rate;    // fraction of active events
        int long_every; // every long_every-th row gets long_length extra entries (0: none)
        int long_length;
    };

    struct Csr
    {
        std::vector<int64_t> indptr;
        std::vector<int32_t> indices; // sorted within each row (the combined metadata requires tile order)
        std::vector<float> values;
    };

    Csr random_csr(const Case &c, std::mt19937_64 &rng)
    {
        Csr m;
        m.indptr.assign(static_cast<size_t>(c.rows) + 1, 0);
        std::uniform_int_distribution<int32_t> column(0, c.cols - 1);
        std::uniform_real_distribution<float> value(-1.0f, 1.0f);
        std::poisson_distribution<int> length(c.density * c.cols);
        for (int r = 0; r < c.rows; ++r)
        {
            int n = r % 7 == 3 ? 0 : length(rng); // some empty rows
            if (c.long_every > 0 && r % c.long_every == 0)
                n += c.long_length;
            std::vector<int32_t> row(n);
            for (auto &x : row)
                x = column(rng); // repeated columns are allowed
            std::sort(row.begin(), row.end());
            for (int32_t x : row)
            {
                m.indices.push_back(x);
                m.values.push_back(value(rng));
            }
            m.indptr[r + 1] = static_cast<int64_t>(m.indices.size());
        }
        return m;
    }

    // Metadata uses the actual width selected for this shape and batch.
    template <typename OffsetT>
    void metadata(const Csr &m, const combined::Plan &plan, std::vector<uint16_t> &local, std::vector<OffsetT> &offsets)
    {
        const int rows = plan.rows, width = plan.tile_width;
        const int tiles = (plan.cols + width - 1) / width;
        local.resize(m.indices.size());
        offsets.assign(static_cast<size_t>(rows) * (tiles + 1), 0);
        for (int r = 0; r < rows; ++r)
        {
            int64_t e = m.indptr[r];
            for (int t = 0; t <= tiles; ++t)
            {
                const int64_t bound = static_cast<int64_t>(t) * width;
                while (e < m.indptr[r + 1] && m.indices[e] < bound)
                    ++e;
                offsets[static_cast<size_t>(r) * (tiles + 1) + t] = static_cast<OffsetT>(t == tiles ? m.indptr[r + 1] : e);
            }
        }
        for (size_t e = 0; e < m.indices.size(); ++e)
            local[e] = static_cast<uint16_t>(m.indices[e] % width);
    }

    template <typename T>
    T *upload(const std::vector<T> &host)
    {
        T *device = nullptr;
        const size_t bytes = std::max<size_t>(host.size(), 1) * sizeof(T);
        CHECK_CUDA(cudaMalloc(&device, bytes));
        if (!host.empty())
            CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
        return device;
    }

    template <typename OffsetT>
    bool run_case(const Case &c, uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        const Csr m = random_csr(c, rng);
        combined::Plan plan;
        CHECK_CUDA(combined::make_plan(c.rows, c.cols, c.batch, m.indices.size(), combined::ValueType::f32, &plan));
        std::vector<uint16_t> local;
        std::vector<OffsetT> offsets;
        metadata(m, plan, local, offsets);
        std::vector<int8_t> events(static_cast<size_t>(c.batch) * c.rows);
        std::bernoulli_distribution active(c.rate);
        std::uniform_int_distribution<int> magnitude(1, 127);
        for (auto &x : events)
            x = active(rng) ? static_cast<int8_t>(rng() % 2 ? magnitude(rng) : -magnitude(rng)) : 0; // != 0 is active

        // float64 reference: Y[b, j] = sum over active rows i of batch b of W[i, j].
        std::vector<double> expected(static_cast<size_t>(c.batch) * c.cols, 0.0);
        std::vector<double> magnitude_sum(expected.size(), 0.0);
        for (int b = 0; b < c.batch; ++b)
            for (int r = 0; r < c.rows; ++r)
                if (events[static_cast<size_t>(b) * c.rows + r] != 0)
                    for (int64_t e = m.indptr[r]; e < m.indptr[r + 1]; ++e)
                    {
                        expected[static_cast<size_t>(b) * c.cols + m.indices[e]] += m.values[e];
                        magnitude_sum[static_cast<size_t>(b) * c.cols + m.indices[e]] += std::fabs(m.values[e]);
                    }

        float *values = upload(m.values);
        uint16_t *targets = upload(local);
        OffsetT *tile_offsets = upload(offsets);
        int8_t *spikes = upload(events);
        float *output = nullptr;
        int32_t *active_rows = nullptr;
        void *workspace = nullptr;
        const size_t out_bytes = static_cast<size_t>(c.batch) * c.cols * sizeof(float);
        CHECK_CUDA(cudaMalloc(&output, out_bytes));
        CHECK_CUDA(cudaMemset(output, 0x7f, out_bytes)); // garbage: the output must be fully overwritten
        CHECK_CUDA(cudaMalloc(&active_rows, static_cast<size_t>(c.batch) * c.rows * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&workspace, combined::workspace_bytes(c.rows, c.batch)));

        bool ok = true;
        for (int repeat = 0; repeat < 2 && ok; ++repeat) // a second call reuses the workspace
        {
            CHECK_CUDA((combined::launch<float, int8_t, OffsetT>(values, targets, tile_offsets, spikes, output, active_rows,
                                                  workspace, plan, nullptr)));
            CHECK_CUDA(cudaDeviceSynchronize());
            std::vector<float> actual(expected.size());
            CHECK_CUDA(cudaMemcpy(actual.data(), output, out_bytes, cudaMemcpyDeviceToHost));
            size_t bad = 0;
            for (size_t k = 0; k < actual.size(); ++k)
            {
                // Float shared-memory atomics in any order: error bound relative to the summed magnitudes.
                const double bound = 1e-5 * (1.0 + magnitude_sum[k]) * 8;
                if (!(std::fabs(actual[k] - expected[k]) <= bound))
                {
                    if (bad < 3)
                        std::fprintf(stderr, "  %s[%s] repeat %d: Y[%zu, %zu] = %.9g, expected %.9g\n", c.name,
                                     sizeof(OffsetT) == 4 ? "i32" : "i64", repeat, k / c.cols, k % c.cols,
                                     actual[k], expected[k]);
                    ++bad;
                }
            }
            ok = bad == 0;
        }
        std::printf("%-28s %-3s rows=%-6d cols=%-6d batch=%-3d nnz=%-9zu %s\n", c.name,
                    sizeof(OffsetT) == 4 ? "i32" : "i64", c.rows, c.cols, c.batch, m.indices.size(),
                    ok ? "ok" : "FAILED");
        CHECK_CUDA(cudaFree(values));
        CHECK_CUDA(cudaFree(targets));
        CHECK_CUDA(cudaFree(tile_offsets));
        CHECK_CUDA(cudaFree(spikes));
        CHECK_CUDA(cudaFree(output));
        CHECK_CUDA(cudaFree(active_rows));
        CHECK_CUDA(cudaFree(workspace));
        return ok;
    }

    // Invalid launches are rejected without touching the device.
    bool invalid_arguments_rejected()
    {
        int dummy = 0;
        combined::Plan plan{1, 1, 1, 1, combined::ValueType::f32, 24576};
        const bool ok = combined::launch<float, int8_t, int32_t>(nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, plan, nullptr) == cudaErrorInvalidValue &&
            combined::make_plan(4, 4, 65536, 1, combined::ValueType::f32, &plan) == cudaErrorInvalidValue;
        std::printf("%-28s %s\n", "invalid_arguments", ok ? "ok" : "FAILED");
        return ok;
    }
} // namespace

int main()
{
    const Case cases[] = {
        // name                       rows   cols   batch density rate  long  length
        {"single_tile",               700,   1000,  3,    0.01,   0.5,  0,    0},
        {"tile_boundary_exact",       300,   24576, 2,    0.002,  0.6,  0,    0},
        {"two_tiles_and_tail",        2000,  60000, 5,    0.0005, 0.4,  0,    0},
        {"row_chunks_and_long_rows",  10000, 30000, 4,    0.0002, 0.3,  97,   20000},
        {"batch_one_all_active",      5000,  50001, 1,    0.0003, 1.0,  0,    0},
        {"many_batches",              1500,  26000, 33,   0.001,  0.2,  0,    0},
        {"silent",                    3000,  40000, 4,    0.001,  0.0,  0,    0},
        {"hot_columns_repeat",        64,    100,   7,    0.5,    0.9,  0,    0},
        {"padding_ctas",              17,    7,     2,    0.5,    1.0,  0,    0},
        {"scatter_512_threads",       73,    400003,2,    0.0001, 0.5,  0,    0},
        {"scatter_1024_threads",      73,    1200001,2,   0.00003,0.5,  0,    0},
    };
    int device = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::printf("device %d: %s (sm_%d%d)\n", device, prop.name, prop.major, prop.minor);
    bool ok = invalid_arguments_rejected();
    uint64_t seed = 20261103;
    for (const Case &c : cases)
    {
        ok = run_case<int32_t>(c, seed) && ok;
        ok = run_case<int64_t>(c, seed) && ok;
        ++seed;
    }
    std::printf(ok ? "ALL PASSED\n" : "SOME FAILED\n");
    return ok ? 0 : 1;
}
