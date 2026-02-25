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
/// @brief JKB assertion and CUDA error-checking macros.

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// JKB_CHECK(cond) << "message";
//
// Abort with a diagnostic if @p cond is false.  The streaming operator
// allows building the message lazily.
// ---------------------------------------------------------------------------

namespace JKB {
namespace internal {

class CheckFailMessageStream {
public:
    CheckFailMessageStream(const char* file, int line, const char* cond)
        : file_(file), line_(line), cond_(cond), active_(true) {}

    // Inactive sentinel (condition was true).
    explicit CheckFailMessageStream(std::nullptr_t)
        : file_(nullptr), line_(0), cond_(nullptr), active_(false) {}

    ~CheckFailMessageStream() {
        if (active_) {
            fprintf(stderr, "[jkb] CHECK FAILED at %s:%d: %s", file_, line_, cond_);
            if (buf_[0] != '\0') {
                fprintf(stderr, " — %s", buf_);
            }
            fprintf(stderr, "\n");
            fflush(stderr);
            abort();
        }
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

    template <typename T>
    CheckFailMessageStream& operator<<(const T& val);

    // Specialisations for common types to avoid pulling in <sstream>.
    CheckFailMessageStream& operator<<(const char* s) {
        if (active_ && s) append(s);
        return *this;
    }
    CheckFailMessageStream& operator<<(int v) {
        if (active_) { char tmp[32]; snprintf(tmp, sizeof(tmp), "%d", v); append(tmp); }
        return *this;
    }
    CheckFailMessageStream& operator<<(int64_t v) {
        if (active_) { char tmp[32]; snprintf(tmp, sizeof(tmp), "%ld", (long)v); append(tmp); }
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

}  // namespace internal
}  // namespace JKB

/// General-purpose assertion with streaming message.
///     JKB_CHECK(x.ndim() == 1) << "expected 1D, got " << x.ndim();
#define JKB_CHECK(cond)                                                   \
    (cond) ? ::JKB::internal::CheckFailMessageStream(nullptr)             \
           : ::JKB::internal::CheckFailMessageStream(                     \
                 __FILE__, __LINE__, #cond)

/// CUDA runtime API error check.
///     JKB_CUDA_CHECK(cudaGetLastError());
#define JKB_CUDA_CHECK(expr)                                              \
    do {                                                                  \
        cudaError_t __jkb_err = (expr);                                   \
        if (__jkb_err != cudaSuccess) {                                   \
            fprintf(stderr,                                               \
                    "[jkb] CUDA error at %s:%d: %s (%s)\n",              \
                    __FILE__, __LINE__,                                    \
                    cudaGetErrorString(__jkb_err),                         \
                    cudaGetErrorName(__jkb_err));                          \
            fflush(stderr);                                               \
            abort();                                                      \
        }                                                                 \
    } while (0)

/// Check after kernel launch (catches configuration errors).
#define JKB_CHECK_KERNEL_LAUNCH() JKB_CUDA_CHECK(cudaGetLastError())
