#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "brainevent/common.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

__global__ void BoolToFloatKernel(const unsigned char* events, float* out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = events[idx] != 0 ? 1.0f : 0.0f;
    }
}

const char* CublasStatusName(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

void CheckCublas(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(
            stderr,
            "[be] cuBLAS error at %s:%d: %s (%s)\n",
            file,
            line,
            CublasStatusName(status),
            expr);
        std::fflush(stderr);
        std::abort();
    }
}

#define BE_CUBLAS_CHECK(expr) CheckCublas((expr), #expr, __FILE__, __LINE__)

#define REQUIRE_CUBLAS_DENSE(cond, message)                                      \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(                                                        \
                stderr,                                                          \
                "[be] cublas dense check failed at %s:%d: %s\n",                \
                __FILE__,                                                        \
                __LINE__,                                                        \
                (message));                                                      \
            std::fflush(stderr);                                                 \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

cublasHandle_t SharedCublasHandle() {
    static cublasHandle_t handle = nullptr;
    static std::once_flag once;
    static cublasStatus_t init_status = CUBLAS_STATUS_SUCCESS;
    std::call_once(once, []() {
        init_status = cublasCreate(&handle);
        if (init_status == CUBLAS_STATUS_SUCCESS) {
            init_status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        }
    });
    BE_CUBLAS_CHECK(init_status);
    return handle;
}

float* ConvertBoolEvents(const BE::Tensor events, cudaStream_t stream) {
    const int64_t n = events.numel();
    float* events_f = nullptr;
    BE_CUDA_CHECK(cudaMallocAsync(&events_f, static_cast<size_t>(n) * sizeof(float), stream));

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    BoolToFloatKernel<<<blocks, threads, 0, stream>>>(
        static_cast<const unsigned char*>(events.data_ptr()),
        events_f,
        n);
    BE_CHECK_KERNEL_LAUNCH();
    return events_f;
}

void LaunchBinaryDenseMvCublas(
    const BE::Tensor weights,
    const BE::Tensor spikes,
    BE::Tensor output,
    int64_t stream,
    bool transpose) {
    REQUIRE_CUBLAS_DENSE(weights.ndim() == 2, "weights must be 2D");
    REQUIRE_CUBLAS_DENSE(spikes.ndim() == 1, "spikes must be 1D");
    REQUIRE_CUBLAS_DENSE(output.ndim() == 1, "output must be 1D");
    REQUIRE_CUBLAS_DENSE(weights.dtype() == BE::DType::Float32, "weights must be float32");
    REQUIRE_CUBLAS_DENSE(spikes.dtype() == BE::DType::Bool, "spikes must be bool");
    REQUIRE_CUBLAS_DENSE(output.dtype() == BE::DType::Float32, "output must be float32");

    const int64_t m = weights.shape(0);
    const int64_t k = weights.shape(1);
    const int64_t x_len = transpose ? m : k;
    const int64_t y_len = transpose ? k : m;
    REQUIRE_CUBLAS_DENSE(spikes.shape(0) == x_len, "spikes shape mismatch");
    REQUIRE_CUBLAS_DENSE(output.shape(0) == y_len, "output shape mismatch");
    REQUIRE_CUBLAS_DENSE(m <= INT_MAX && k <= INT_MAX, "cuBLAS MV dimensions exceed int range");

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    float* spikes_f = ConvertBoolEvents(spikes, cuda_stream);

    cublasHandle_t handle = SharedCublasHandle();
    BE_CUBLAS_CHECK(cublasSetStream(handle, cuda_stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int rows = static_cast<int>(k);
    const int cols = static_cast<int>(m);
    const int lda = rows;
    const cublasOperation_t op = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;

    BE_CUBLAS_CHECK(cublasSgemv(
        handle,
        op,
        rows,
        cols,
        &alpha,
        static_cast<const float*>(weights.data_ptr()),
        lda,
        spikes_f,
        1,
        &beta,
        static_cast<float*>(output.data_ptr()),
        1));

    BE_CUDA_CHECK(cudaFreeAsync(spikes_f, cuda_stream));
}

void LaunchBinaryDenseMmCublas(
    const BE::Tensor weights,
    const BE::Tensor events,
    BE::Tensor output,
    int64_t stream,
    bool transpose) {
    REQUIRE_CUBLAS_DENSE(weights.ndim() == 2, "weights must be 2D");
    REQUIRE_CUBLAS_DENSE(events.ndim() == 2, "events must be 2D");
    REQUIRE_CUBLAS_DENSE(output.ndim() == 2, "output must be 2D");
    REQUIRE_CUBLAS_DENSE(weights.dtype() == BE::DType::Float32, "weights must be float32");
    REQUIRE_CUBLAS_DENSE(events.dtype() == BE::DType::Bool, "events must be bool");
    REQUIRE_CUBLAS_DENSE(output.dtype() == BE::DType::Float32, "output must be float32");

    const int64_t k = transpose ? weights.shape(0) : weights.shape(1);
    const int64_t m = transpose ? weights.shape(1) : weights.shape(0);
    const int64_t n = events.shape(1);
    REQUIRE_CUBLAS_DENSE(events.shape(0) == k, "events shape mismatch");
    REQUIRE_CUBLAS_DENSE(output.shape(0) == m, "output dim 0 mismatch");
    REQUIRE_CUBLAS_DENSE(output.shape(1) == n, "output dim 1 mismatch");
    REQUIRE_CUBLAS_DENSE(m <= INT_MAX && k <= INT_MAX && n <= INT_MAX, "cuBLAS MM dimensions exceed int range");

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    float* events_f = ConvertBoolEvents(events, cuda_stream);

    cublasHandle_t handle = SharedCublasHandle();
    BE_CUBLAS_CHECK(cublasSetStream(handle, cuda_stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int c_rows = static_cast<int>(n);
    const int c_cols = static_cast<int>(m);
    const int shared = static_cast<int>(k);
    const int lda = static_cast<int>(n);
    const int ldb = transpose ? static_cast<int>(m) : static_cast<int>(k);
    const int ldc = static_cast<int>(n);
    const cublasOperation_t op_a = CUBLAS_OP_N;
    const cublasOperation_t op_b = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    BE_CUBLAS_CHECK(cublasSgemm(
        handle,
        op_a,
        op_b,
        c_rows,
        c_cols,
        shared,
        &alpha,
        events_f,
        lda,
        static_cast<const float*>(weights.data_ptr()),
        ldb,
        &beta,
        static_cast<float*>(output.data_ptr()),
        ldc));

    BE_CUDA_CHECK(cudaFreeAsync(events_f, cuda_stream));
}

}  // namespace

// @BE binary_densemv_cublas_nt_f32_bool
void binary_densemv_cublas_nt_f32_bool(
    const BE::Tensor weights,
    const BE::Tensor spikes,
    BE::Tensor output,
    int64_t stream) {
    LaunchBinaryDenseMvCublas(weights, spikes, output, stream, false);
}

// @BE binary_densemv_cublas_t_f32_bool
void binary_densemv_cublas_t_f32_bool(
    const BE::Tensor weights,
    const BE::Tensor spikes,
    BE::Tensor output,
    int64_t stream) {
    LaunchBinaryDenseMvCublas(weights, spikes, output, stream, true);
}

// @BE binary_densemm_cublas_nt_f32_bool
void binary_densemm_cublas_nt_f32_bool(
    const BE::Tensor weights,
    const BE::Tensor events,
    BE::Tensor output,
    int64_t stream) {
    LaunchBinaryDenseMmCublas(weights, events, output, stream, false);
}

// @BE binary_densemm_cublas_t_f32_bool
void binary_densemm_cublas_t_f32_bool(
    const BE::Tensor weights,
    const BE::Tensor events,
    BE::Tensor output,
    int64_t stream) {
    LaunchBinaryDenseMmCublas(weights, events, output, stream, true);
}
