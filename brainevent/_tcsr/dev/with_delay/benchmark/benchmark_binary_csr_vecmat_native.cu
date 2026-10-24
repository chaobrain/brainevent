// Native delay benchmark. One event column is one step for both operators.
#include "delay_benchmark_api.h"
#include "../operator/binary_csrmv_hybrid_region_delay.cu"
#include "../operator/region_delay_split.cu"
#include "../operator/region_delay_cusparse.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace delay_benchmark
{

    template <typename T>
    class DeviceArray
    {
        T *pointer_ = nullptr;

    public:
        DeviceArray() = default;
        DeviceArray(const DeviceArray &) = delete;
        DeviceArray &operator=(const DeviceArray &) = delete;
        ~DeviceArray()
        {
            if (pointer_)
                cudaFree(pointer_);
        }
        T *get() const { return pointer_; }
        void allocate(int64_t count)
        {
            BE_CHECK(!pointer_ && count >= 0);
            BE_CHECK(static_cast<uint64_t>(count) <= SIZE_MAX / sizeof(T));
            BE_CUDA_CHECK(cudaMalloc(&pointer_, std::max<int64_t>(count, 1) * sizeof(T)));
        }
        void upload(const T *source, int64_t count, cudaStream_t stream)
        {
            allocate(count);
            if (count)
                BE_CUDA_CHECK(cudaMemcpyAsync(pointer_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream));
        }
        void download(T *target, int64_t count) const
        {
            if (count)
                BE_CUDA_CHECK(cudaMemcpy(target, pointer_, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    };

    struct Stream
    {
        cudaStream_t value = nullptr;
        Stream() { BE_CUDA_CHECK(cudaStreamCreate(&value)); }
        ~Stream()
        {
            if (value)
                cudaStreamDestroy(value);
        }
    };

    struct Timer
    {
        cudaEvent_t start = nullptr, stop = nullptr;
        Timer()
        {
            BE_CUDA_CHECK(cudaEventCreate(&start));
            try
            {
                BE_CUDA_CHECK(cudaEventCreate(&stop));
            }
            catch (...)
            {
                cudaEventDestroy(start);
                throw;
            }
        }
        Timer(const Timer &) = delete;
        Timer &operator=(const Timer &) = delete;
        ~Timer()
        {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
        }
    };

    struct SparseBase
    {
        int64_t active_segments = 0;
        virtual ~SparseBase() = default;
        virtual void project(const float *events) = 0;
    };

    template <typename SubT, typename RingT>
    class SplitProjection final : public SparseBase
    {
        cudaStream_t stream_;
        cusparseHandle_t handle_ = nullptr;
        DeviceArray<SubT> ptr_, base_, indices_;
        DeviceArray<float> values_, scratch_;
        const RingT *offsets_;
        const int32_t *head_;
        float *ring_;
        std::unique_ptr<region_delay_cusparse::StepPlan<SubT>> plan_;

    public:
        template <typename IndexT, typename EdgeT>
        SplitProjection(const DelayBenchmarkInput &in, const float *weights,
                        const IndexT *indices, const EdgeT *ptr, const EdgeT *counts,
                        const RingT *offsets, const float *events, const int32_t *head,
                        float *ring, cudaStream_t stream)
            : stream_(stream), offsets_(offsets), head_(head), ring_(ring)
        {
            ptr_.allocate(in.segments * (in.rows + 1));
            base_.allocate(in.segments + 1);
            indices_.allocate(in.nnz);
            values_.allocate(in.nnz);
            scratch_.allocate(in.cols);
            if (in.weight_count == 1)
                region_delay_split::Launch<true, IndexT, EdgeT, SubT>(weights, indices, ptr, counts,
                                                                      ptr_.get(), base_.get(), indices_.get(), values_.get(), in.rows, in.segments, stream);
            else
                region_delay_split::Launch<false, IndexT, EdgeT, SubT>(weights, indices, ptr, counts,
                                                                       ptr_.get(), base_.get(), indices_.get(), values_.get(), in.rows, in.segments, stream);
            try
            {
                BE_CUSPARSE_CHECK(cusparseCreate(&handle_));
                // Pass the complete split result to the operator; it owns all K segments.
                plan_ = std::make_unique<region_delay_cusparse::StepPlan<SubT>>(
                    handle_, ptr_.get(), base_.get(), indices_.get(), values_.get(),
                    events, scratch_.get(), in.rows, in.cols, in.segments, in.slots, in.nnz, stream);
                active_segments = plan_->active_segments();
                BE_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            catch (...)
            {
                plan_.reset();
                cudaStreamSynchronize(stream);
                if (handle_)
                    cusparseDestroy(handle_);
                throw;
            }
        }
        ~SplitProjection() override
        {
            plan_.reset();
            cudaStreamSynchronize(stream_);
            if (handle_)
                cusparseDestroy(handle_);
        }
        void project(const float *events) override
        {
            plan_->Project(events, offsets_, head_, ring_);
        }
    };

    // Run after stop[t]: preserve any hybrid error before the next launch resets it.
    __global__ void Advance(int32_t *head, int64_t slots, const int32_t *status,
                            int32_t *all_errors, bool hybrid)
    {
        if (hybrid)
            *all_errors |= status[1];
        *head = (*head + int64_t(1)) % slots;
    }

    struct CaseBase
    {
        virtual ~CaseBase() = default;
        virtual void hybrid(int, int, DelayBenchmarkStats *, float *, float *) = 0;
        virtual void cusparse(int, int, DelayBenchmarkStats *, float *, float *) = 0;
    };

    template <typename IndexT, typename EdgeT, typename RingT>
    class Case final : public CaseBase
    {
        Stream stream_; // Declared first so it outlives all allocations/plans.
        DelayBenchmarkInput in_;
        DeviceArray<float> weights_, events_, ring_, current_;
        DeviceArray<IndexT> indices_;
        DeviceArray<EdgeT> ptr_, counts_, begin_, end_;
        DeviceArray<RingT> offsets_, out_off_;
        DeviceArray<int32_t> row_tasks_, head_, status_, errors_;
        std::vector<std::unique_ptr<Timer>> timers_;
        std::unique_ptr<SparseBase> sparse_;

        template <bool Hybrid>
        void run(int warmup, int runs, DelayBenchmarkStats *stats, float *output, float *final_ring)
        {
            BE_CHECK(warmup >= 0 && runs > 0 && stats);
            DeviceArray<float> trace;
            if (output)
                trace.allocate(in_.steps * in_.cols);
            std::vector<double> samples;
            double warmup_ms = 0;
            const auto stream = stream_.value;
            for (int64_t repeat = 0; repeat < int64_t(warmup) + runs; ++repeat)
            {
                if (in_.slots * in_.cols)
                    BE_CUDA_CHECK(cudaMemsetAsync(ring_.get(), 0, in_.slots * in_.cols * sizeof(float), stream));
                BE_CUDA_CHECK(cudaMemsetAsync(head_.get(), 0, sizeof(int32_t), stream));
                BE_CUDA_CHECK(cudaMemsetAsync(errors_.get(), 0, sizeof(int32_t), stream));
                for (int64_t t = 0; t < in_.steps; ++t)
                {
                    const float *x = events_.get() + t * in_.rows;
                    BE_CUDA_CHECK(cudaEventRecord(timers_[t]->start, stream));
                    if constexpr (Hybrid)
                    {
                        if (in_.weight_count == 1)
                            region_delay::Launch<true, float, IndexT, EdgeT, RingT>(
                                weights_.get(), indices_.get(), ptr_.get(), x, counts_.get(), offsets_.get(),
                                row_tasks_.get(), head_.get(), ring_.get(), begin_.get(), end_.get(),
                                out_off_.get(), status_.get(), in_.rows, in_.segments, in_.cols, in_.slots,
                                static_cast<int>(in_.capacity), stream);
                        else
                            region_delay::Launch<false, float, IndexT, EdgeT, RingT>(
                                weights_.get(), indices_.get(), ptr_.get(), x, counts_.get(), offsets_.get(),
                                row_tasks_.get(), head_.get(), ring_.get(), begin_.get(), end_.get(),
                                out_off_.get(), status_.get(), in_.rows, in_.segments, in_.cols, in_.slots,
                                static_cast<int>(in_.capacity), stream);
                    }
                    else
                    {
                        sparse_->project(x); // All K segments use x and the same head.
                    }
                    BE_CUDA_CHECK(cudaEventRecord(timers_[t]->stop, stream));
                    if (in_.cols)
                    {
                        float *target = output ? trace.get() + t * in_.cols : current_.get();
                        region_delay::Consume<<<static_cast<unsigned>((in_.cols + 255) / 256), 256, 0, stream>>>(
                            ring_.get(), head_.get(), target, in_.cols, in_.slots);
                        BE_CUDA_CHECK(cudaGetLastError());
                    }
                    Advance<<<1, 1, 0, stream>>>(head_.get(), in_.slots, status_.get(), errors_.get(), Hybrid);
                    BE_CUDA_CHECK(cudaGetLastError());
                }
                // One synchronization per run, never per step or delay segment.
                BE_CUDA_CHECK(cudaStreamSynchronize(stream));
                int32_t error = 0;
                errors_.download(&error, 1);
                BE_CHECK(error == 0) << "hybrid projection reported status " << error;
                double total_ms = 0;
                for (const auto &timer : timers_)
                {
                    float elapsed = 0;
                    BE_CUDA_CHECK(cudaEventElapsedTime(&elapsed, timer->start, timer->stop));
                    total_ms += elapsed;
                }
                if (repeat < warmup)
                    warmup_ms += total_ms;
                else
                    samples.push_back(total_ms);
            }
            if (output)
                trace.download(output, in_.steps * in_.cols);
            if (final_ring)
                ring_.download(final_ring, in_.slots * in_.cols);
            const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
            const double mean = total / runs;
            double variance = 0;
            for (double sample : samples)
                variance += (sample - mean) * (sample - mean);
            *stats = {warmup_ms, total, mean, std::sqrt(variance / runs),
                      *std::min_element(samples.begin(), samples.end()),
                      *std::max_element(samples.begin(), samples.end()), in_.steps, in_.segments,
                      Hybrid ? 0 : in_.steps * sparse_->active_segments};
        }

    public:
        explicit Case(const DelayBenchmarkInput &in) : in_(in)
        {
            const auto stream = stream_.value;
            weights_.upload(in.weights, in.weight_count, stream);
            indices_.upload(static_cast<const IndexT *>(in.indices), in.nnz, stream);
            ptr_.upload(static_cast<const EdgeT *>(in.indptr), in.rows + 1, stream);
            counts_.upload(static_cast<const EdgeT *>(in.counts), in.rows * in.segments, stream);
            offsets_.upload(static_cast<const RingT *>(in.offsets), in.segments, stream);
            row_tasks_.upload(in.row_tasks, in.rows, stream);
            events_.upload(in.events, in.rows * in.steps, stream);
            ring_.allocate(in.slots * in.cols);
            current_.allocate(in.cols);
            begin_.allocate(in.capacity);
            end_.allocate(in.capacity);
            out_off_.allocate(in.capacity);
            head_.allocate(1);
            status_.allocate(2);
            errors_.allocate(1);
            for (int64_t t = 0; t < in.steps; ++t)
                timers_.push_back(std::make_unique<Timer>());
            BE_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        ~Case() override { cudaStreamSynchronize(stream_.value); }
        
        void hybrid(int warmup, int runs, DelayBenchmarkStats *stats, float *output, float *ring) override
        {
            run<true>(warmup, runs, stats, output, ring);
        }

        void cusparse(int warmup, int runs, DelayBenchmarkStats *stats, float *output, float *ring) override
        {
            if (!sparse_)
            {
                if (in_.nnz <= INT32_MAX)
                    sparse_ = std::make_unique<SplitProjection<int32_t, RingT>>(
                        in_, weights_.get(), indices_.get(), ptr_.get(), counts_.get(), offsets_.get(),
                        events_.get(), head_.get(), ring_.get(), stream_.value);
                else
                    sparse_ = std::make_unique<SplitProjection<int64_t, RingT>>(
                        in_, weights_.get(), indices_.get(), ptr_.get(), counts_.get(), offsets_.get(),
                        events_.get(), head_.get(), ring_.get(), stream_.value);
            }
            run<false>(warmup, runs, stats, output, ring);
        }
    };

    template <typename Function>
    int report(Function &&function, char *error, int capacity) noexcept
    {
        try
        {
            function();
            return 0;
        }
        catch (const std::exception &exception)
        {
            if (error && capacity > 0)
                std::snprintf(error, capacity, "%s", exception.what());
            return 1;
        }
    }

    void validate(const DelayBenchmarkInput &in)
    {
        BE_CHECK(in.rows >= 0 && in.rows <= INT32_MAX && in.cols >= 0 && in.cols <= INT32_MAX);
        BE_CHECK(in.steps > 0 && in.steps <= INT32_MAX && in.segments > 0 && in.segments <= INT32_MAX);
        BE_CHECK(in.slots > 0 && in.slots <= INT32_MAX && in.nnz >= 0);
        BE_CHECK(in.weight_count == 1 || in.weight_count == in.nnz);
        BE_CHECK(in.capacity >= 0 && in.capacity <= INT32_MAX);
        BE_CHECK(in.index_bits == 32 || in.index_bits == 64);
        BE_CHECK(in.edge_bits == 32 || in.edge_bits == 64);
        BE_CHECK(in.ring_bits == 32 || in.ring_bits == 64);
        BE_CHECK(in.edge_bits == 64 || in.nnz <= INT32_MAX);
        BE_CHECK(in.ring_bits == 64 || in.slots * in.cols <= INT32_MAX);
    }

} // namespace delay_benchmark

extern "C" int create_delay_benchmark(const DelayBenchmarkInput *input, void **result, char *error, int capacity)
{
    if (result)
        *result = nullptr;
    return delay_benchmark::report([&]
                                   {
        BE_CHECK(input && result);
        delay_benchmark::validate(*input);
        const auto index_dtype = input->index_bits == 32 ? BE::DType::Int32 : BE::DType::Int64;
        const auto edge_dtype = input->edge_bits == 32 ? BE::DType::Int32 : BE::DType::Int64;
        const auto ring_dtype = input->ring_bits == 32 ? BE::DType::Int32 : BE::DType::Int64;
        BE_DISPATCH_SIGNED_INDEX_PAIR(index_dtype, edge_dtype, IndexT, EdgeT, {
            BE_DISPATCH_SIGNED_INDEX(ring_dtype, RingT, {
                *result = new delay_benchmark::Case<IndexT, EdgeT, RingT>(*input);
            });
        }); }, error, capacity);
}

extern "C" void destroy_delay_benchmark(void *instance)
{
    delete static_cast<delay_benchmark::CaseBase *>(instance);
}

extern "C" int benchmark_delay_hybrid(void *instance, int warmup, int runs,
                                      DelayBenchmarkStats *stats, float *output, float *ring, char *error, int capacity)
{
    return delay_benchmark::report([&]
                                   {
        BE_CHECK(instance);
        static_cast<delay_benchmark::CaseBase*>(instance)->hybrid(warmup, runs, stats, output, ring); }, error, capacity);
}

extern "C" int benchmark_delay_cusparse(void *instance, int warmup, int runs,
                                        DelayBenchmarkStats *stats, float *output, float *ring, char *error, int capacity)
{
    return delay_benchmark::report([&]
                                   {
        BE_CHECK(instance);
        static_cast<delay_benchmark::CaseBase*>(instance)->cusparse(warmup, runs, stats, output, ring); }, error, capacity);
}
