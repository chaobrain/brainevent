// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Ordinary C++ caller: all weighted instances link with the binary/column objects.
#include "../combined_kernels.cuh"
#include "../col_major/combined_col_major.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void cuda_check(cudaError_t result) { check(result == cudaSuccess, cudaGetErrorString(result)); }
template<class T> struct Buffer {
    T *data = nullptr;
    size_t size;
    explicit Buffer(size_t n) : size(n) {
        if (n) cuda_check(cudaMalloc(reinterpret_cast<void **>(&data), n * sizeof(T)));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void upload(const std::vector<T> &values) {
        check(values.size() == size, "upload length");
        if (size) cuda_check(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(size);
        if (size) cuda_check(cudaMemcpy(values.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

template<class V, class O> void run_case(cudaStream_t stream) {
    using Traits = cuda_traits::ValueTraits<V>;
    const int rows = 8201, cols = combined::tile_width(Traits::type) + 3, batch = 3;
    combined::Plan plan;
    cuda_check(combined::make_plan(rows, cols, batch, 0, Traits::type, &plan));
    const int width = plan.tile_width;
    const int tiles = (cols + width - 1) / width;
    // Original edge coordinates are independent of the sorted traversal metadata.
    std::vector<int> source, target;
    for (int i = 0; i < rows; ++i) {
        source.push_back(i);
        target.push_back((i * 37) % cols);
    }
    for (int r : {0, 4095, 4096, 8191, 8192})
        for (int c : {0, width - 1, width, width + 2, width + 2}) {
            source.push_back(r); target.push_back(c);
        }
    const size_t nnz = source.size();
    std::vector<O> perm(nnz), ptr(rows + 1, 0), offsets(static_cast<size_t>(rows) * (tiles + 1));
    std::iota(perm.begin(), perm.end(), O(0));
    std::stable_sort(perm.begin(), perm.end(), [&](O a, O b) {
        return source[a] == source[b] ? target[a] < target[b] : source[a] < source[b];
    });
    std::vector<int> indices(nnz);
    std::vector<uint16_t> local(nnz);
    for (size_t j = 0; j < nnz; ++j) {
        ++ptr[source[j] + 1]; indices[j] = target[perm[j]]; local[j] = indices[j] % width;
    }
    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t <= tiles; ++t)
            offsets[static_cast<size_t>(r) * (tiles + 1) + t] = static_cast<O>(
                std::lower_bound(indices.begin() + ptr[r], indices.begin() + ptr[r + 1], t * width) - indices.begin());
    cuda_check(combined::make_plan(rows, cols, batch, nnz, Traits::type, &plan));
    Buffer<V> weights(nnz), sorted(nnz), shared(1), events(static_cast<size_t>(batch) * rows), output(static_cast<size_t>(batch) * cols);
    Buffer<O> d_perm(nnz), d_offsets(offsets.size());
    Buffer<uint16_t> d_local(nnz);
    Buffer<int32_t> active(events.size);
    Buffer<uint32_t> scratch((combined::workspace_bytes(rows, batch) + 3) / 4);
    d_perm.upload(perm); d_offsets.upload(offsets); d_local.upload(local);
    auto permuted = [&](const O *mapping, const combined::Plan &p) {
        return combined_col_major::launch_weighted(weights.data, d_local.data, d_offsets.data, mapping,
            events.data, output.data, active.data, scratch.data, p, stream);
    };
    check(permuted(nullptr, plan) == cudaErrorInvalidValue, "null perm accepted");
    auto bad = plan; bad.tile_width = 8192;
    check(permuted(d_perm.data, bad) == cudaErrorInvalidValue, "bad tile accepted");
    bad = plan; bad.value_type = Traits::type == combined::ValueType::f64 ? combined::ValueType::f32 : combined::ValueType::f64;
    check(permuted(d_perm.data, bad) == cudaErrorInvalidValue, "bad dtype accepted");
    if (sizeof(O) == 4) {
        bad = plan; bad.nnz = int64_t(INT32_MAX) + 1;
        check(permuted(d_perm.data, bad) == cudaErrorInvalidValue, "int32 overflow accepted");
    }
    for (int step = 0; step < 3; ++step) {
        std::vector<V> host_weights(nnz), gathered(nnz), host_events(events.size);
        for (size_t j = 0; j < nnz; ++j) host_weights[j] = V((static_cast<int>(j % 7) - 3 + step) * .25f);
        for (size_t j = 0; j < nnz; ++j) gathered[j] = host_weights[perm[j]];
        for (size_t i = 0; i < events.size; ++i) host_events[i] = V(step == 1 ? 0.f : (static_cast<int>(i % 7) - 3) * .5f);
        weights.upload(host_weights); sorted.upload(gathered); events.upload(host_events); shared.upload({V(1.5f)});
        for (int mode = 0; mode < 5; ++mode) {
            cuda_check(cudaMemsetAsync(output.data, 0x7f, output.size * sizeof(V), stream));
            if (mode == 0) cuda_check(permuted(d_perm.data, plan));
            else if (mode == 1) cuda_check(combined::launch_weighted(sorted.data, d_local.data, d_offsets.data,
                events.data, output.data, active.data, scratch.data, plan, stream));
            else if (mode == 2) cuda_check(combined::launch_weighted_homo(shared.data, d_local.data, d_offsets.data,
                events.data, output.data, active.data, scratch.data, plan, stream));
            else if (mode == 3) cuda_check(combined::launch_weighted_permuted(weights.data, d_local.data, d_offsets.data,
                d_perm.data, events.data, output.data, active.data, scratch.data, plan, stream));
            else {
                // Coexistence with the original binary specialization, whose float events are indicators.
                Buffer<float> binary_events(events.size);
                std::vector<float> binary_host(events.size);
                for (size_t i = 0; i < events.size; ++i) binary_host[i] = static_cast<float>(host_events[i]);
                binary_events.upload(binary_host);
                cuda_check(combined_col_major::launch(weights.data, d_local.data, d_offsets.data, d_perm.data,
                    binary_events.data, output.data, active.data, scratch.data, plan, stream));
                cuda_check(cudaStreamSynchronize(stream));
            }
            cuda_check(cudaStreamSynchronize(stream));
            std::vector<double> expected(output.size, 0);
            for (int b = 0; b < batch; ++b)
                for (size_t j = 0; j < nnz; ++j) {
                    double event = static_cast<double>(host_events[static_cast<size_t>(b) * rows + source[j]]);
                    if (event != 0) expected[static_cast<size_t>(b) * cols + target[j]] +=
                        (mode == 4 ? 1 : event) * (mode == 2 ? 1.5 : static_cast<double>(host_weights[j]));
                }
            auto actual = output.download();
            for (size_t i = 0; i < actual.size(); ++i)
                check(static_cast<double>(actual[i]) == static_cast<double>(V(expected[i])), "weighted result mismatch");
        }
    }
    cuda_check(combined::make_plan(rows, cols, batch, 0, Traits::type, &plan));
    cuda_check((combined_col_major::launch_weighted<V, O>(nullptr, nullptr, nullptr, nullptr, nullptr,
        output.data, nullptr, nullptr, plan, stream)));
    cuda_check(cudaStreamSynchronize(stream));
    for (V value : output.download()) check(static_cast<double>(value) == 0, "empty output not cleared");
    cuda_check(combined::make_plan(rows, cols, 0, 0, Traits::type, &plan));
    cuda_check((combined::launch_weighted<V, O>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, plan, stream)));
}

template<class V> void run_type(cudaStream_t stream) { run_case<V, int32_t>(stream); run_case<V, int64_t>(stream); }
int main() {
    cudaStream_t stream = nullptr;
    try {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        run_type<__half>(stream); run_type<__nv_bfloat16>(stream); run_type<float>(stream); run_type<double>(stream);
        cuda_check(cudaStreamDestroy(stream));
        std::puts("ALL PASSED: weighted direct, homo, perm, column, binary coexistence, types, reuse, chunks and empty");
        return 0;
    } catch (const std::exception &error) {
        if (stream) cudaStreamDestroy(stream);
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
