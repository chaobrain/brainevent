// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#pragma once
/// @file check.h
/// @brief BE assertion and CUDA error-checking macros.
///
/// On the host these macros *throw* (``BE::CheckError`` / ``BE::CudaError``,
/// both ``std::runtime_error``) so a generated FFI wrapper can catch
/// ``std::exception`` and return an ``xla::ffi::Error`` - keeping the Python
/// interpreter alive - instead of calling ``abort()`` and killing the process
/// with ``SIGABRT``.  In device code (``__CUDA_ARCH__``) exceptions are
/// unavailable, so the failing path prints a diagnostic and ``__trap()``s.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>

#ifndef __CUDA_ARCH__
#include <stdexcept>
#include <string>

namespace BE {

/// Thrown by ``BE_CHECK`` when a host-side invariant fails.
class CheckError : public std::runtime_error {
public:
    explicit CheckError(const std::string& message) : std::runtime_error(message) {}
};

/// Thrown by ``BE_CUDA_CHECK`` when a CUDA runtime call fails on the host.
class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace BE
#endif  // __CUDA_ARCH__

// ---------------------------------------------------------------------------
// BE_CHECK(cond) << "message";
//
// Throw (host) or trap (device) with a diagnostic if @p cond is false.  The
// streaming operator builds the message lazily.  ``BE_CHECK`` is intended for
// host code; it is exercised on the happy path (not during stack unwinding),
// so throwing from the message stream's destructor is safe.
// ---------------------------------------------------------------------------

namespace BE {
namespace internal {

class CheckFailMessageStream {
public:
    CheckFailMessageStream(const char* file, int line, const char* cond)
        : file_(file), line_(line), cond_(cond), active_(true) {}

    // Inactive sentinel (condition was true).
    explicit CheckFailMessageStream(std::nullptr_t)
        : file_(nullptr), line_(0), cond_(nullptr), active_(false) {}

    // Throws BE::CheckError on the host so the failure can be converted to an
    // xla::ffi::Error by the FFI wrapper.  ``noexcept(false)`` is required
    // because the throw happens here, at end of the full expression.
    ~CheckFailMessageStream() noexcept(false) {
        if (!active_) return;
#ifdef __CUDA_ARCH__
        printf("[be] CHECK FAILED at %s:%d: %s %s\n", file_, line_, cond_, buf_);
        __trap();
#else
        char msg[640];
        if (buf_[0] != '\0') {
            snprintf(msg, sizeof(msg), "[be] CHECK FAILED at %s:%d: %s - %s",
                     file_, line_, cond_, buf_);
        } else {
            snprintf(msg, sizeof(msg), "[be] CHECK FAILED at %s:%d: %s",
                     file_, line_, cond_);
        }
        throw ::BE::CheckError(msg);
#endif
    }

    // Prevent copies so the destructor fires exactly once.
    CheckFailMessageStream(const CheckFailMessageStream&) = delete;
    CheckFailMessageStream& operator=(const CheckFailMessageStream&) = delete;
    CheckFailMessageStream(CheckFailMessageStream&& o) noexcept
        : file_(o.file_), line_(o.line_), cond_(o.cond_),
          active_(o.active_), pos_(o.pos_) {
        for (int i = 0; i < pos_; ++i) buf_[i] = o.buf_[i];
        buf_[pos_] = '\0';
        o.active_ = false;  // moved-from instance must not fire
    }

    // Specialisations for common types to avoid pulling in <sstream>.
    // (No catch-all template overload: streaming an unsupported type is a
    // compile error here rather than a link error against an undefined template.)
    CheckFailMessageStream& operator<<(const char* s) {
        if (active_ && s) append(s);
        return *this;
    }
    CheckFailMessageStream& operator<<(int v) {
        if (active_) { char tmp[32]; snprintf(tmp, sizeof(tmp), "%d", v); append(tmp); }
        return *this;
    }
    CheckFailMessageStream& operator<<(int64_t v) {
        if (active_) { char tmp[32]; snprintf(tmp, sizeof(tmp), "%" PRId64, v); append(tmp); }
        return *this;
    }
    CheckFailMessageStream& operator<<(size_t v) {
        if (active_) { char tmp[32]; snprintf(tmp, sizeof(tmp), "%zu", v); append(tmp); }
        return *this;
    }
    CheckFailMessageStream& operator<<(double v) {
        if (active_) { char tmp[32]; snprintf(tmp, sizeof(tmp), "%g", v); append(tmp); }
        return *this;
    }

private:
    void append(const char* s) {
        while (*s && pos_ < (int)sizeof(buf_) - 1) buf_[pos_++] = *s++;
        buf_[pos_] = '\0';
    }

    const char* file_;
    int line_;
    const char* cond_;
    bool active_;
    char buf_[512] = {};
    int pos_ = 0;
};

// Consumes a CheckFailMessageStream and yields void so that BE_CHECK's ternary
// has two ``void`` branches.  Without this, ``cond ? Stream(nullptr) :
// Stream(...)`` would force the conditional operator to materialise a
// (deleted-copy) temporary of the non-copyable stream.  ``operator&`` has lower
// precedence than ``operator<<``, so ``Voidify() & stream << x`` binds as
// ``Voidify() & (stream << x)``.  The ``const&`` parameter binds to both the
// lvalue returned by ``operator<<`` (message present) and the stream temporary
// itself (no message); the temporary's throwing destructor still fires at the
// end of the full expression.
class Voidify {
public:
    void operator&(const CheckFailMessageStream&) const {}
};

}  // namespace internal
}  // namespace BE

/// General-purpose assertion with streaming message.
///     BE_CHECK(x.ndim() == 1) << "expected 1D, got " << x.ndim();
///
/// Expands so the ternary has two ``void`` operands (see ``Voidify``); the
/// failing branch builds a temporary ``CheckFailMessageStream`` whose
/// destructor throws ``BE::CheckError`` (host) or traps (device) at the end of
/// the full expression.
#define BE_CHECK(cond)                                                   \
    (cond) ? (void)0                                                     \
           : ::BE::internal::Voidify() &                                 \
                 ::BE::internal::CheckFailMessageStream(                 \
                     __FILE__, __LINE__, #cond)

/// CUDA runtime API error check.
///     BE_CUDA_CHECK(cudaGetLastError());
///
/// On the host a failing call throws BE::CudaError (an xla::ffi::Error after
/// the wrapper catch); in device code it traps.  ``cudaGetErrorString`` /
/// ``cudaGetErrorName`` are host-only, so the device path reports the numeric
/// error code only.
#ifdef __CUDA_ARCH__
#define BE_CUDA_CHECK(expr)                                              \
    do {                                                                  \
        cudaError_t __be_err = (expr);                                    \
        if (__be_err != cudaSuccess) {                                    \
            printf("[be] CUDA error at %s:%d: code %d\n",                 \
                   __FILE__, __LINE__, (int)__be_err);                    \
            __trap();                                                     \
        }                                                                 \
    } while (0)
#else
#define BE_CUDA_CHECK(expr)                                              \
    do {                                                                  \
        cudaError_t __be_err = (expr);                                    \
        if (__be_err != cudaSuccess) {                                    \
            char __be_msg[256];                                           \
            snprintf(__be_msg, sizeof(__be_msg),                          \
                     "[be] CUDA error at %s:%d: %s (%s)",                 \
                     __FILE__, __LINE__,                                   \
                     cudaGetErrorString(__be_err),                        \
                     cudaGetErrorName(__be_err));                         \
            throw ::BE::CudaError(__be_msg);                              \
        }                                                                 \
    } while (0)
#endif

/// Check after a kernel launch.
///
/// NOTE: ``cudaGetLastError()`` only surfaces *synchronous* launch errors
/// (invalid grid/block dims, excessive shared memory).  Asynchronous faults
/// (illegal address, out-of-bounds access) are reported at the next
/// synchronization point and may therefore be misattributed to a later,
/// unrelated operation.  Use ``BE_CHECK_KERNEL_LAUNCH_SYNC()`` while debugging
/// to pin a fault to its launch site.
#define BE_CHECK_KERNEL_LAUNCH() BE_CUDA_CHECK(cudaGetLastError())

/// Like ``BE_CHECK_KERNEL_LAUNCH()`` but also synchronizes the device first so
/// asynchronous kernel faults are caught here rather than at a later sync.
/// This forces a full device synchronization and is intended for debug builds.
#define BE_CHECK_KERNEL_LAUNCH_SYNC()                                    \
    do {                                                                  \
        BE_CUDA_CHECK(cudaGetLastError());                                \
        BE_CUDA_CHECK(cudaDeviceSynchronize());                           \
    } while (0)
