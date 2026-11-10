// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Compile with g++, then link the parent and column-major standalone CUDA objects.
#include "../combined_col_major.cuh"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

void check(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t error) { check(error == cudaSuccess, cudaGetErrorString(error)); }

template <class T>
struct Buffer
{
    T *data = nullptr;
    size_t count;
    explicit Buffer(size_t size) : count(size)
    {
        if (count) cuda_check(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void upload(const std::vector<T> &values)
    {
        check(values.size() == count, "upload size mismatch");
        if (count) cuda_check(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const
    {
        std::vector<T> result(count);
        if (count) cuda_check(cudaMemcpy(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
};

template <class V, class E, class O>
void run_case(cudaStream_t stream)
{
    constexpr int n = 4101, batch = 3;
    const auto type = cuda_traits::ValueTraits<V>::type;
    const int m = combined_col_major::tile_width(type) + 3;
    combined_col_major::Plan plan;
    cuda_check(combined_col_major::make_plan(n, m, batch, 0, type, &plan));
    const int width = plan.tile_width, tiles = (m + width - 1) / width;
    // Independent original CSR coordinates; duplicate coordinates retain distinct slots.
    const std::vector<int> rows{0, 1, 1, width - 1, width, width, width, width, width + 2};
    const std::vector<int> cols{4096, 0, 4100, 4095, 1, 4095, 4095, 4096, 4100};
    const size_t nnz = rows.size();
    std::vector<O> perm(nnz);
    std::iota(perm.begin(), perm.end(), O(0));
    std::stable_sort(perm.begin(), perm.end(), [&](O a, O b) {
        return cols[a] != cols[b] ? cols[a] < cols[b] : rows[a] < rows[b];
    });
    std::vector<O> ptr(n + 1, 0), offsets(static_cast<size_t>(n) * (tiles + 1));
    std::vector<int> indices(nnz);
    std::vector<uint16_t> local(nnz);
    for (size_t j = 0; j < nnz; ++j)
    {
        ++ptr[cols[j] + 1];
        indices[j] = rows[perm[j]];
        local[j] = static_cast<uint16_t>(indices[j] % width);
    }
    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
    for (int col = 0; col < n; ++col)
        for (int tile = 0; tile <= tiles; ++tile)
            offsets[static_cast<size_t>(col) * (tiles + 1) + tile] = static_cast<O>(
                std::lower_bound(indices.begin() + ptr[col], indices.begin() + ptr[col + 1], tile * width) - indices.begin());

    cuda_check(combined_col_major::make_plan(n, m, batch, nnz, type, &plan));
    Buffer<V> weights(nnz), direct_weights(nnz), output(static_cast<size_t>(batch) * m), homo(1);
    Buffer<E> events(static_cast<size_t>(batch) * n);
    Buffer<O> d_offsets(offsets.size()), d_perm(nnz);
    Buffer<uint16_t> d_local(nnz);
    Buffer<int32_t> active(static_cast<size_t>(batch) * n);
    Buffer<uint32_t> scratch((combined_col_major::workspace_bytes(n, batch) + 3) / 4);
    d_offsets.upload(offsets); d_perm.upload(perm); d_local.upload(local);
    auto launch = [&](const O *mapping, const combined_col_major::Plan &p) {
        return combined_col_major::launch(weights.data, d_local.data, d_offsets.data, mapping,
            events.data, output.data, active.data, scratch.data, p, stream);
    };
    check(launch(nullptr, plan) == cudaErrorInvalidValue, "accepted null perm");
    auto bad = plan;
    bad.tile_width = 8192;
    check(launch(d_perm.data, bad) == cudaErrorInvalidValue, "accepted mismatched tile width");
    bad = plan;
    bad.value_type = type == combined::ValueType::f32 ? combined::ValueType::f64 : combined::ValueType::f32;
    check(launch(d_perm.data, bad) == cudaErrorInvalidValue, "accepted mismatched value dtype");
    if (sizeof(O) == 4)
    {
        bad = plan;
        bad.nnz = static_cast<int64_t>(INT32_MAX) + 1;
        check(launch(d_perm.data, bad) == cudaErrorInvalidValue, "accepted int32 absolute offset overflow");
    }
    for (int step = 0; step < 3; ++step)
    {
        std::vector<V> host_weights(nnz), gathered(nnz);
        for (size_t j = 0; j < nnz; ++j)
            host_weights[j] = V((static_cast<int>(j) - 3 + step) * 0.25f);
        for (size_t j = 0; j < nnz; ++j) gathered[j] = host_weights[perm[j]];
        std::vector<E> host_events(events.count, E(0));
        if (step != 1)
            for (int b = 0; b < batch; ++b)
                for (int col : cols)
                    host_events[static_cast<size_t>(b) * n + col] = E((b + col) % 3 - 1);
        weights.upload(host_weights); direct_weights.upload(gathered); events.upload(host_events);
        cuda_check(cudaMemsetAsync(output.data, 0x7f, output.count * sizeof(V), stream));
        cuda_check(launch(d_perm.data, plan));
        cuda_check(cudaStreamSynchronize(stream));
        const auto actual = output.download();
        std::vector<double> expected(output.count, 0);
        for (int b = 0; b < batch; ++b)
            for (size_t j = 0; j < nnz; ++j)
                if (host_events[static_cast<size_t>(b) * n + cols[j]] != E(0))
                    expected[static_cast<size_t>(b) * m + rows[j]] += static_cast<double>(host_weights[j]);
        for (size_t i = 0; i < actual.size(); ++i)
            check(static_cast<double>(actual[i]) == expected[i], "native permuted output mismatch");
        // Direct and permuted entry points coexist in one linked program.
        cuda_check(combined::launch(direct_weights.data, d_local.data, d_offsets.data, events.data,
            output.data, active.data, scratch.data, plan, stream));
        cuda_check(cudaStreamSynchronize(stream));
        check(output.download() == actual, "direct/perm mismatch");
    }
    homo.upload({V(2)});
    events.upload(std::vector<E>(events.count, E(1)));
    cuda_check(combined::launch_homo(homo.data, d_local.data, d_offsets.data, events.data,
        output.data, active.data, scratch.data, plan, stream));
    cuda_check(cudaStreamSynchronize(stream));
    std::vector<int> counts(m, 0);
    for (int row : rows) counts[row] += 2;
    auto actual = output.download();
    for (int b = 0; b < batch; ++b)
        for (int r = 0; r < m; ++r)
            check(static_cast<double>(actual[static_cast<size_t>(b) * m + r]) == counts[r], "homo regression");

    // Empty matrices overwrite output without needing metadata/input/scratch pointers.
    cuda_check(combined_col_major::make_plan(n, m, batch, 0, type, &plan));
    cuda_check((combined_col_major::launch<V, E, O>(nullptr, nullptr, nullptr, nullptr, nullptr,
        output.data, nullptr, nullptr, plan, stream)));
    cuda_check(cudaStreamSynchronize(stream));
    for (V value : output.download()) check(static_cast<double>(value) == 0, "empty did not clear output");
    cuda_check(combined_col_major::make_plan(n, m, 0, 0, type, &plan));
    cuda_check((combined_col_major::launch<V, E, O>(nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, plan, stream)));
}

template <class V>
void run_type(cudaStream_t stream)
{
    run_case<V, int8_t, int32_t>(stream);
    run_case<V, int8_t, int64_t>(stream);
    run_case<V, float, int32_t>(stream);
    run_case<V, float, int64_t>(stream);
}

int main()
{
    cudaStream_t stream = nullptr;
    try
    {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        run_type<__half>(stream);
        run_type<__nv_bfloat16>(stream);
        run_type<float>(stream);
        run_type<double>(stream);
        cuda_check(cudaStreamDestroy(stream));
        std::puts("combined col-major native types, mapping, reuse, direct/homo and empty cases passed");
        return 0;
    }
    catch (const std::exception &error)
    {
        if (stream) cudaStreamDestroy(stream);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
