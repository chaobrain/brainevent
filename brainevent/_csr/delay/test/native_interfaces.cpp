// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Compile with an ordinary C++ compiler; kernels are linked from standalone CUs.
#include "../hybrid/hybrid_delay_spmv.cuh"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

static void check(cudaError_t code) {
    if (code != cudaSuccess) {
        std::cerr << cudaGetErrorString(code) << '\n';
        std::abort();
    }
}

template<class T> struct Device {
    T *ptr = nullptr;
    size_t size;
    explicit Device(const std::vector<T>& host) : size(host.size()) {
        if (size) {
            check(cudaMalloc(&ptr, size * sizeof(T)));
            set(host);
        }
    }
    explicit Device(size_t n) : Device(std::vector<T>(n)) {}
    ~Device() { if (ptr) cudaFree(ptr); }
    void set(const std::vector<T>& host) {
        assert(host.size() == size);
        if (size) check(cudaMemcpy(ptr, host.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> host(size);
        if (size) check(cudaMemcpy(host.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

template<bool Homo, class V, class I, class E, class R>
void exercise() {
    constexpr int rows = 3, segments = 2, cols = 5, slots = 3, nnz = 4228, capacity = 4;
    std::vector<E> pointers{0, 2, 131, nnz}, counts{1, 1, 5, 124, 1, 4096};
    std::vector<I> columns(nnz);
    std::vector<V> weights(Homo ? 1 : nnz);
    for (int e = 0; e < nnz; ++e) columns[e] = e % cols;
    for (size_t e = 0; e < weights.size(); ++e) weights[e] = V((e % 4 + 1) / 8.0);
    std::vector<V> events{V(-2), V(0.5), V(1.25)};
    Device<V> w(weights), x(events), ring(std::vector<V>(cols * slots, V(1))), value(capacity);
    Device<I> idx(columns);
    Device<E> ptr(pointers), count(counts), begin(capacity), end(capacity);
    Device<R> offset(std::vector<R>{0, 2 * cols}), out_off(capacity);
    Device<int32_t> tasks(std::vector<int32_t>{0, 2, 2}), head(std::vector<int32_t>{2}), status(2);
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    check(region_delay::launch_weighted<Homo>(w.ptr, idx.ptr, ptr.ptr, x.ptr, count.ptr, offset.ptr,
        tasks.ptr, head.ptr, ring.ptr, begin.ptr, end.ptr, out_off.ptr, value.ptr, status.ptr,
        rows, segments, cols, slots, capacity, stream));
    check(cudaStreamSynchronize(stream));
    assert((status.get() == std::vector<int32_t>{capacity, 0}));
    std::vector<V> expected(cols * slots, V(1));
    for (int row = 0; row < rows; ++row) {
        int edge = static_cast<int>(pointers[row]);
        for (int k = 0; k < segments; ++k) {
            const int finish = edge + static_cast<int>(counts[row * segments + k]);
            const int slot = (2 + 2 * k) % slots;
            for (; edge < finish; ++edge)
                expected[slot * cols + columns[edge]] += weights[Homo ? 0 : edge] * events[row];
        }
    }
    assert(ring.get() == expected);
    auto starts = begin.get();
    auto values = value.get();
    for (int task = 0; task < capacity; ++task) {
        const int row = static_cast<int>(std::upper_bound(pointers.begin(), pointers.end(), starts[task]) - pointers.begin()) - 1;
        assert(values[task] == events[row]);
    }

    // Empty active set must reset status and ignore stale task values.
    x.set(std::vector<V>(rows, V(0)));
    check(region_delay::launch_weighted<Homo>(w.ptr, idx.ptr, ptr.ptr, x.ptr, count.ptr, offset.ptr,
        tasks.ptr, head.ptr, ring.ptr, begin.ptr, end.ptr, out_off.ptr, value.ptr, status.ptr,
        rows, segments, cols, slots, capacity, stream));
    check(cudaStreamSynchronize(stream));
    assert((status.get() == std::vector<int32_t>{0, 0}));
    assert(ring.get() == expected);

    // Insufficient capacity reports a device status, without reading tasks.
    x.set(std::vector<V>{V(0), V(1), V(0)});
    check(region_delay::launch_weighted<Homo, V, I, E, R>(w.ptr, idx.ptr, ptr.ptr, x.ptr, count.ptr, offset.ptr,
        tasks.ptr, head.ptr, ring.ptr, nullptr, nullptr, nullptr, nullptr, status.ptr,
        rows, segments, cols, slots, 0, stream));
    check(cudaStreamSynchronize(stream));
    assert(status.get()[1] == 1);
    assert(ring.get() == expected);

    // Validate every active row before even the valid short row can write.
    // Both long rows need two tasks, so capacity three fits either row alone
    // but cannot accommodate their combined active set.
    auto reject_without_partial_writes = [&](auto launch) {
        for (int fault = 0; fault < 3; ++fault) {
            ring.set(expected);
            auto bad_counts = counts;
            std::vector<int32_t> row_tasks{0, 2, 2};
            if (fault == 0) ++bad_counts[4];
            if (fault == 1) row_tasks[2] = 1;
            count.set(bad_counts);
            tasks.set(row_tasks);
            check(launch(fault == 2 ? capacity - 1 : capacity));
            check(cudaStreamSynchronize(stream));
            assert(status.get()[1] == (fault == 2 ? 1 : 3));
            assert(ring.get() == expected);
        }
        count.set(counts);
        tasks.set(std::vector<int32_t>{0, 2, 2});
        check(launch(capacity));
        check(cudaStreamSynchronize(stream));
        assert(status.get()[1] == 0);
        assert(ring.get() != expected);
        ring.set(expected);
    };
    x.set(events);
    reject_without_partial_writes([&](int available) {
        return region_delay::launch_weighted<Homo>(
            w.ptr, idx.ptr, ptr.ptr, x.ptr, count.ptr, offset.ptr, tasks.ptr, head.ptr,
            ring.ptr, begin.ptr, end.ptr, out_off.ptr, value.ptr, status.ptr,
            rows, segments, cols, slots, available, stream);
    });
#ifndef REGION_DELAY_TEST_WEIGHTED_ONLY
    Device<float> active_rows(std::vector<float>(rows, 1));
    reject_without_partial_writes([&](int available) {
        return region_delay::launch_binary<Homo>(
            w.ptr, idx.ptr, ptr.ptr, active_rows.ptr, count.ptr, offset.ptr, tasks.ptr, head.ptr,
            ring.ptr, begin.ptr, end.ptr, out_off.ptr, status.ptr,
            rows, segments, cols, slots, available, stream);
    });
#endif

    auto invalid = region_delay::launch_weighted<Homo, V, I, E, R>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, status.ptr, -1, 1, 0, 1, 0, stream);
    assert(invalid == cudaErrorInvalidValue);
    head.set(std::vector<int32_t>{0});
    check(region_delay::launch_weighted<Homo, V, I, E, R>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, head.ptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, status.ptr, 0, 1, 0, 1, 0, stream));
    check(cudaStreamSynchronize(stream));
    assert((status.get() == std::vector<int32_t>{0, 0}));

    // Empty graphs preserve every ring slot and still validate head on device.
    // This includes zero columns, where no ring allocation is required.
    for (int empty_cols : {0, cols}) {
        for (int empty_head : {-1, 0, slots}) {
            head.set(std::vector<int32_t>{empty_head});
            check(region_delay::launch_weighted<Homo, V, I, E, R>(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, head.ptr,
                empty_cols ? ring.ptr : nullptr, nullptr, nullptr, nullptr, nullptr,
                status.ptr, 0, segments, empty_cols, slots, 0, stream));
            check(cudaStreamSynchronize(stream));
            assert((status.get() == std::vector<int32_t>{0, empty_head == 0 ? 0 : 2}));
            assert(ring.get() == expected);
#ifndef REGION_DELAY_TEST_WEIGHTED_ONLY
            check(region_delay::launch_binary<Homo, V, float, I, E, R>(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, head.ptr,
                empty_cols ? ring.ptr : nullptr, nullptr, nullptr, nullptr,
                status.ptr, 0, segments, empty_cols, slots, 0, stream));
            check(cudaStreamSynchronize(stream));
            assert((status.get() == std::vector<int32_t>{0, empty_head == 0 ? 0 : 2}));
            assert(ring.get() == expected);
#endif
        }
    }

#ifndef REGION_DELAY_TEST_WEIGHTED_ONLY
    // Binary is supplied only by the binary CU.
    head.set(std::vector<int32_t>{2});
    Device<float> activity(std::vector<float>{-2, 0, 0});
    check(region_delay::launch_binary<Homo>(w.ptr, idx.ptr, ptr.ptr, activity.ptr, count.ptr, offset.ptr,
        tasks.ptr, head.ptr, ring.ptr, begin.ptr, end.ptr, out_off.ptr, status.ptr,
        rows, segments, cols, slots, capacity, stream));
    check(cudaStreamSynchronize(stream));
    expected[2 * cols + columns[0]] += weights[0];
    expected[cols + columns[1]] += weights[Homo ? 0 : 1];
    assert(ring.get() == expected);
#endif
    check(cudaStreamDestroy(stream));
}

int main() {
    exercise<false, double, int32_t, int64_t, int32_t>();
    exercise<true, double, int64_t, int32_t, int64_t>();
    exercise<false, float, int64_t, int64_t, int64_t>();
    std::cout << "region delay native interfaces passed\n";
}
