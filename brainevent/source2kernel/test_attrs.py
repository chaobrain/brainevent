# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Comprehensive tests for scalar attribute passing via XLA FFI.

Covers all supported attr dtypes:
  bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64,
  float32, float64
  float16 / bfloat16 (raw uint16 bits — no native XLA FFI scalar decoding)

Note: complex64/complex128 are NOT supported as XLA FFI scalar attrs.
JAX's MLIR backend (mlir.ir_attribute) cannot encode numpy complex scalars.
Complex-valued math is tested via separate float32/float64 re/im attrs.

For every dtype both forms are tested:
  - Bare form ``"attr.name"``        — type auto-inferred from C++ signature
  - Explicit form ``"attr.name:T"``  — type specified in the arg_spec token
"""

import jax
import jax as _jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytest as _pytest

requires_gpu = _pytest.mark.skipif(
    not (bool(_jax.devices("gpu")) if True else False),
    reason="No GPU detected via jax.devices('gpu')",
)

# float64, uint64, int64, complex128 require x64 mode.
jax.config.update("jax_enable_x64", True)

pytestmark = requires_gpu

from brainevent._error import KernelError
import brainevent.source2kernel as jkb

# ---------------------------------------------------------------------------
# Shared CUDA source
# ---------------------------------------------------------------------------

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

// ── float32 ───────────────────────────────────────────────────────────────
__global__ void _scale_f32(const float* x, float* out, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * scale;
}
void scale_f32(BE::Tensor x, BE::Tensor out, float scale, int64_t stream) {
    int n = (int)x.numel();
    _scale_f32<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(out.data_ptr()), n, scale);
}

// ── float64 ───────────────────────────────────────────────────────────────
__global__ void _scale_f64(const double* x, double* out, int n, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * scale;
}
void scale_f64(BE::Tensor x, BE::Tensor out, double scale, int64_t stream) {
    int n = (int)x.numel();
    _scale_f64<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const double*>(x.data_ptr()),
        static_cast<double*>(out.data_ptr()), n, scale);
}

// ── int8 ──────────────────────────────────────────────────────────────────
__global__ void _add_i8(const int8_t* x, int8_t* out, int n, int8_t offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + offset;
}
void add_i8(BE::Tensor x, BE::Tensor out, int8_t offset, int64_t stream) {
    int n = (int)x.numel();
    _add_i8<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const int8_t*>(x.data_ptr()),
        static_cast<int8_t*>(out.data_ptr()), n, offset);
}

// ── uint8 ─────────────────────────────────────────────────────────────────
__global__ void _add_u8(const uint8_t* x, uint8_t* out, int n,
                         uint8_t offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + offset;
}
void add_u8(BE::Tensor x, BE::Tensor out, uint8_t offset, int64_t stream) {
    int n = (int)x.numel();
    _add_u8<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const uint8_t*>(x.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()), n, offset);
}

// ── int16 ─────────────────────────────────────────────────────────────────
__global__ void _add_i16(const int16_t* x, int16_t* out, int n,
                          int16_t offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + offset;
}
void add_i16(BE::Tensor x, BE::Tensor out, int16_t offset,
             int64_t stream) {
    int n = (int)x.numel();
    _add_i16<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const int16_t*>(x.data_ptr()),
        static_cast<int16_t*>(out.data_ptr()), n, offset);
}

// ── uint16 ────────────────────────────────────────────────────────────────
__global__ void _add_u16(const uint16_t* x, uint16_t* out, int n,
                          uint16_t offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + offset;
}
void add_u16(BE::Tensor x, BE::Tensor out, uint16_t offset,
             int64_t stream) {
    int n = (int)x.numel();
    _add_u16<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const uint16_t*>(x.data_ptr()),
        static_cast<uint16_t*>(out.data_ptr()), n, offset);
}

// ── int32 ─────────────────────────────────────────────────────────────────
__global__ void _add_i32(const int* x, int* out, int n, int32_t offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + offset;
}
void add_i32(BE::Tensor x, BE::Tensor out, int32_t offset, int64_t stream) {
    int n = (int)x.numel();
    _add_i32<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const int*>(x.data_ptr()),
        static_cast<int*>(out.data_ptr()), n, offset);
}

// ── uint32 ────────────────────────────────────────────────────────────────
__global__ void _add_u32(const uint32_t* x, uint32_t* out, int n,
                          uint32_t offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + offset;
}
void add_u32(BE::Tensor x, BE::Tensor out, uint32_t offset,
             int64_t stream) {
    int n = (int)x.numel();
    _add_u32<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const uint32_t*>(x.data_ptr()),
        static_cast<uint32_t*>(out.data_ptr()), n, offset);
}

// ── int64 ─────────────────────────────────────────────────────────────────
__global__ void _add_i64(const long long* x, long long* out, int n,
                          int64_t offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + offset;
}
void add_i64(BE::Tensor x, BE::Tensor out, int64_t offset,
             int64_t stream) {
    int n = (int)x.numel();
    _add_i64<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const long long*>(x.data_ptr()),
        static_cast<long long*>(out.data_ptr()), n, offset);
}

// ── uint64 ────────────────────────────────────────────────────────────────
__global__ void _add_u64(const unsigned long long* x,
                          unsigned long long* out, int n, uint64_t offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + offset;
}
void add_u64(BE::Tensor x, BE::Tensor out, uint64_t offset,
             int64_t stream) {
    int n = (int)x.numel();
    _add_u64<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const unsigned long long*>(x.data_ptr()),
        static_cast<unsigned long long*>(out.data_ptr()), n, offset);
}

// ── bool ──────────────────────────────────────────────────────────────────
__global__ void _maybe_negate(const float* x, float* out, int n, bool negate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = negate ? -x[i] : x[i];
}
void maybe_negate(BE::Tensor x, BE::Tensor out, bool negate,
                  int64_t stream) {
    int n = (int)x.numel();
    _maybe_negate<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(out.data_ptr()), n, negate);
}

// ── complex64 — separate float32 re/im attrs ─────────────────────────────
// JAX FFI cannot encode numpy.complex64 scalars as XLA attrs.
// Pass real and imaginary parts as two float32 attrs; kernel computes |s|^2*x.
__global__ void _scale_c64(const float* x, float* out, int n,
                             float re, float im) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * (re * re + im * im);
}
void scale_c64(BE::Tensor x, BE::Tensor out,
               float re, float im, int64_t stream) {
    int n = (int)x.numel();
    _scale_c64<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(out.data_ptr()),
        n, re, im);
}

// ── complex128 — separate float64 re/im attrs ────────────────────────────
// JAX FFI cannot encode numpy.complex128 scalars as XLA attrs.
// Pass real and imaginary parts as two float64 attrs; kernel computes |s|^2*x.
__global__ void _scale_c128(const double* x, double* out, int n,
                              double re, double im) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * (re * re + im * im);
}
void scale_c128(BE::Tensor x, BE::Tensor out,
                double re, double im, int64_t stream) {
    int n = (int)x.numel();
    _scale_c128<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const double*>(x.data_ptr()),
        static_cast<double*>(out.data_ptr()),
        n, re, im);
}

// ── multiple attrs (float32 + int32) ─────────────────────────────────────
__global__ void _scale_add(const float* x, float* out, int n,
                            float scale, int32_t offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * scale + (float)offset;
}
void scale_add(BE::Tensor x, BE::Tensor out,
               float scale, int32_t offset, int64_t stream) {
    int n = (int)x.numel();
    _scale_add<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(out.data_ptr()), n, scale, offset);
}
"""

# arg_spec token lists — used by both module fixtures
_FUNCTIONS_BARE = {
    "scale_f32": ["arg", "ret", "attr.scale", "stream"],
    "scale_f64": ["arg", "ret", "attr.scale", "stream"],
    "add_i8": ["arg", "ret", "attr.offset", "stream"],
    "add_u8": ["arg", "ret", "attr.offset", "stream"],
    "add_i16": ["arg", "ret", "attr.offset", "stream"],
    "add_u16": ["arg", "ret", "attr.offset", "stream"],
    "add_i32": ["arg", "ret", "attr.offset", "stream"],
    "add_u32": ["arg", "ret", "attr.offset", "stream"],
    "add_i64": ["arg", "ret", "attr.offset", "stream"],
    "add_u64": ["arg", "ret", "attr.offset", "stream"],
    "maybe_negate": ["arg", "ret", "attr.negate", "stream"],
    "scale_c64": ["arg", "ret", "attr.re", "attr.im", "stream"],
    "scale_c128": ["arg", "ret", "attr.re", "attr.im", "stream"],
    "scale_add": ["arg", "ret", "attr.scale", "attr.offset", "stream"],
}

_FUNCTIONS_EXPLICIT = {
    "scale_f32": ["arg", "ret", "attr.scale:float32", "stream"],
    "scale_f64": ["arg", "ret", "attr.scale:float64", "stream"],
    "add_i8": ["arg", "ret", "attr.offset:int8", "stream"],
    "add_u8": ["arg", "ret", "attr.offset:uint8", "stream"],
    "add_i16": ["arg", "ret", "attr.offset:int16", "stream"],
    "add_u16": ["arg", "ret", "attr.offset:uint16", "stream"],
    "add_i32": ["arg", "ret", "attr.offset:int32", "stream"],
    "add_u32": ["arg", "ret", "attr.offset:uint32", "stream"],
    "add_i64": ["arg", "ret", "attr.offset:int64", "stream"],
    "add_u64": ["arg", "ret", "attr.offset:uint64", "stream"],
    "maybe_negate": ["arg", "ret", "attr.negate:bool", "stream"],
    "scale_c64": ["arg", "ret", "attr.re:float32", "attr.im:float32", "stream"],
    "scale_c128": ["arg", "ret", "attr.re:float64", "attr.im:float64", "stream"],
    "scale_add": ["arg", "ret", "attr.scale:float32", "attr.offset:int32",
                  "stream"],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bare_mod():
    """Compiled with bare ``attr.name`` tokens — types auto-inferred."""
    import brainevent.source2kernel as jkb
    return jkb.load_cuda_inline(
        name="test_attrs_bare",
        cuda_sources=CUDA_SRC,
        functions=_FUNCTIONS_BARE,
        force_rebuild=True,
    )


@pytest.fixture(scope="module")
def explicit_mod():
    """Compiled with explicit ``attr.name:type`` tokens."""
    import brainevent.source2kernel as jkb
    return jkb.load_cuda_inline(
        name="test_attrs_explicit",
        cuda_sources=CUDA_SRC,
        functions=_FUNCTIONS_EXPLICIT,
        force_rebuild=True,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

N = 128


def _call(prefix, func, out_dtype, *args, out_shape=None, **kwargs):
    shape = out_shape or args[0].shape
    return jax.ffi.ffi_call(
        f"{prefix}.{func}",
        jax.ShapeDtypeStruct(shape, out_dtype),
    )(*args, **kwargs)


# ===========================================================================
# float32
# ===========================================================================

class TestFloat32:
    def test_bare(self, bare_mod):
        x = jnp.arange(N, dtype=jnp.float32)
        out = _call("test_attrs_bare", "scale_f32", jnp.float32, x,
                    scale=np.float32(2.5))
        np.testing.assert_allclose(np.asarray(out),
                                   np.arange(N, dtype=np.float32) * 2.5,
                                   rtol=1e-5)

    def test_explicit(self, explicit_mod):
        x = jnp.arange(N, dtype=jnp.float32)
        out = _call("test_attrs_explicit", "scale_f32", jnp.float32, x,
                    scale=np.float32(2.5))
        np.testing.assert_allclose(np.asarray(out),
                                   np.arange(N, dtype=np.float32) * 2.5,
                                   rtol=1e-5)

    def test_value_sweep(self, bare_mod):
        x = jnp.ones(N, dtype=jnp.float32)
        for v in [0.0, -1.0, 0.5, 100.0, 1e-6]:
            out = _call("test_attrs_bare", "scale_f32", jnp.float32, x,
                        scale=np.float32(v))
            np.testing.assert_allclose(np.asarray(out),
                                       np.full(N, v, dtype=np.float32),
                                       rtol=1e-5, err_msg=f"scale={v}")


# ===========================================================================
# float64
# ===========================================================================

class TestFloat64:
    def test_bare(self, bare_mod):
        x = jnp.arange(N, dtype=jnp.float64)
        out = _call("test_attrs_bare", "scale_f64", jnp.float64, x,
                    scale=np.float64(1.23456789))
        np.testing.assert_allclose(np.asarray(out),
                                   np.arange(N) * 1.23456789, rtol=1e-12)

    def test_explicit(self, explicit_mod):
        x = jnp.arange(N, dtype=jnp.float64)
        out = _call("test_attrs_explicit", "scale_f64", jnp.float64, x,
                    scale=np.float64(1.23456789))
        np.testing.assert_allclose(np.asarray(out),
                                   np.arange(N) * 1.23456789, rtol=1e-12)

    def test_precision(self, bare_mod):
        """f64 preserves more precision than f32 would."""
        val = 1.0 + 1e-10
        x = jnp.ones(N, dtype=jnp.float64)
        out = _call("test_attrs_bare", "scale_f64", jnp.float64, x,
                    scale=np.float64(val))
        np.testing.assert_allclose(np.asarray(out),
                                   np.full(N, val), rtol=1e-12)


# ===========================================================================
# int8
# ===========================================================================

class TestInt8:
    def test_bare(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.int8)
        out = _call("test_attrs_bare", "add_i8", jnp.int8, x,
                    offset=np.int8(42))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 42, dtype=np.int8))

    def test_explicit(self, explicit_mod):
        x = jnp.zeros(N, dtype=jnp.int8)
        out = _call("test_attrs_explicit", "add_i8", jnp.int8, x,
                    offset=np.int8(42))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 42, dtype=np.int8))

    def test_value_sweep(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.int8)
        for v in [0, 1, -1, 127, -128]:
            out = _call("test_attrs_bare", "add_i8", jnp.int8, x,
                        offset=np.int8(v))
            np.testing.assert_array_equal(np.asarray(out),
                                          np.full(N, v, dtype=np.int8),
                                          err_msg=f"offset={v}")


# ===========================================================================
# uint8
# ===========================================================================

class TestUint8:
    def test_bare(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.uint8)
        out = _call("test_attrs_bare", "add_u8", jnp.uint8, x,
                    offset=np.uint8(200))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 200, dtype=np.uint8))

    def test_explicit(self, explicit_mod):
        x = jnp.zeros(N, dtype=jnp.uint8)
        out = _call("test_attrs_explicit", "add_u8", jnp.uint8, x,
                    offset=np.uint8(200))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 200, dtype=np.uint8))

    def test_value_sweep(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.uint8)
        for v in [0, 1, 127, 255]:
            out = _call("test_attrs_bare", "add_u8", jnp.uint8, x,
                        offset=np.uint8(v))
            np.testing.assert_array_equal(np.asarray(out),
                                          np.full(N, v, dtype=np.uint8),
                                          err_msg=f"offset={v}")


# ===========================================================================
# int16
# ===========================================================================

class TestInt16:
    def test_bare(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.int16)
        out = _call("test_attrs_bare", "add_i16", jnp.int16, x,
                    offset=np.int16(1000))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 1000, dtype=np.int16))

    def test_explicit(self, explicit_mod):
        x = jnp.zeros(N, dtype=jnp.int16)
        out = _call("test_attrs_explicit", "add_i16", jnp.int16, x,
                    offset=np.int16(1000))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 1000, dtype=np.int16))

    def test_negative(self, bare_mod):
        x = jnp.full(N, 2000, dtype=jnp.int16)
        out = _call("test_attrs_bare", "add_i16", jnp.int16, x,
                    offset=np.int16(-500))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 1500, dtype=np.int16))


# ===========================================================================
# uint16
# ===========================================================================

class TestUint16:
    def test_bare(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.uint16)
        out = _call("test_attrs_bare", "add_u16", jnp.uint16, x,
                    offset=np.uint16(60000))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 60000, dtype=np.uint16))

    def test_explicit(self, explicit_mod):
        x = jnp.zeros(N, dtype=jnp.uint16)
        out = _call("test_attrs_explicit", "add_u16", jnp.uint16, x,
                    offset=np.uint16(60000))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 60000, dtype=np.uint16))

    def test_value_sweep(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.uint16)
        for v in [0, 1, 256, 65535]:
            out = _call("test_attrs_bare", "add_u16", jnp.uint16, x,
                        offset=np.uint16(v))
            np.testing.assert_array_equal(np.asarray(out),
                                          np.full(N, v, dtype=np.uint16),
                                          err_msg=f"offset={v}")


# ===========================================================================
# int32
# ===========================================================================

class TestInt32:
    def test_bare(self, bare_mod):
        x = jnp.arange(N, dtype=jnp.int32)
        out = _call("test_attrs_bare", "add_i32", jnp.int32, x,
                    offset=np.int32(7))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.arange(N, dtype=np.int32) + 7)

    def test_explicit(self, explicit_mod):
        x = jnp.arange(N, dtype=jnp.int32)
        out = _call("test_attrs_explicit", "add_i32", jnp.int32, x,
                    offset=np.int32(7))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.arange(N, dtype=np.int32) + 7)

    def test_value_sweep(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.int32)
        for v in [0, 1, -1, 2 ** 16, -(2 ** 16)]:
            out = _call("test_attrs_bare", "add_i32", jnp.int32, x,
                        offset=np.int32(v))
            np.testing.assert_array_equal(np.asarray(out),
                                          np.full(N, v, dtype=np.int32),
                                          err_msg=f"offset={v}")


# ===========================================================================
# uint32
# ===========================================================================

class TestUint32:
    def test_bare(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.uint32)
        out = _call("test_attrs_bare", "add_u32", jnp.uint32, x,
                    offset=np.uint32(2 ** 31))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 2 ** 31, dtype=np.uint32))

    def test_explicit(self, explicit_mod):
        x = jnp.zeros(N, dtype=jnp.uint32)
        out = _call("test_attrs_explicit", "add_u32", jnp.uint32, x,
                    offset=np.uint32(2 ** 31))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 2 ** 31, dtype=np.uint32))

    def test_value_sweep(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.uint32)
        for v in [0, 1, 2 ** 16, 2 ** 32 - 1]:
            out = _call("test_attrs_bare", "add_u32", jnp.uint32, x,
                        offset=np.uint32(v))
            np.testing.assert_array_equal(np.asarray(out),
                                          np.full(N, v, dtype=np.uint32),
                                          err_msg=f"offset={v}")


# ===========================================================================
# int64
# ===========================================================================

class TestInt64:
    def test_bare(self, bare_mod):
        """int64_t offset must not be confused with int64_t stream."""
        x = jnp.arange(N, dtype=jnp.int64)
        out = _call("test_attrs_bare", "add_i64", jnp.int64, x,
                    offset=np.int64(1000))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.arange(N, dtype=np.int64) + 1000)

    def test_explicit(self, explicit_mod):
        x = jnp.arange(N, dtype=jnp.int64)
        out = _call("test_attrs_explicit", "add_i64", jnp.int64, x,
                    offset=np.int64(1000))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.arange(N, dtype=np.int64) + 1000)

    def test_large_value(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.int64)
        big = np.int64(2 ** 40)
        out = _call("test_attrs_bare", "add_i64", jnp.int64, x,
                    offset=big)
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, big, dtype=np.int64))


# ===========================================================================
# uint64
# ===========================================================================

class TestUint64:
    def test_bare(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.uint64)
        out = _call("test_attrs_bare", "add_u64", jnp.uint64, x,
                    offset=np.uint64(2 ** 40))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 2 ** 40, dtype=np.uint64))

    def test_explicit(self, explicit_mod):
        x = jnp.zeros(N, dtype=jnp.uint64)
        out = _call("test_attrs_explicit", "add_u64", jnp.uint64, x,
                    offset=np.uint64(2 ** 40))
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, 2 ** 40, dtype=np.uint64))

    def test_max_value(self, bare_mod):
        x = jnp.zeros(N, dtype=jnp.uint64)
        big = np.uint64(2 ** 63)
        out = _call("test_attrs_bare", "add_u64", jnp.uint64, x,
                    offset=big)
        np.testing.assert_array_equal(np.asarray(out),
                                      np.full(N, big, dtype=np.uint64))


# ===========================================================================
# bool
# ===========================================================================

class TestBool:
    def test_bare_false(self, bare_mod):
        x = jnp.arange(N, dtype=jnp.float32) + 1.0
        out = _call("test_attrs_bare", "maybe_negate", jnp.float32, x,
                    negate=False)
        np.testing.assert_allclose(np.asarray(out), np.asarray(x), rtol=1e-6)

    def test_bare_true(self, bare_mod):
        x = jnp.arange(N, dtype=jnp.float32) + 1.0
        out = _call("test_attrs_bare", "maybe_negate", jnp.float32, x,
                    negate=True)
        np.testing.assert_allclose(np.asarray(out), -np.asarray(x), rtol=1e-6)

    def test_explicit_false(self, explicit_mod):
        x = jnp.arange(N, dtype=jnp.float32) + 1.0
        out = _call("test_attrs_explicit", "maybe_negate", jnp.float32, x,
                    negate=False)
        np.testing.assert_allclose(np.asarray(out), np.asarray(x), rtol=1e-6)

    def test_explicit_true(self, explicit_mod):
        x = jnp.arange(N, dtype=jnp.float32) + 1.0
        out = _call("test_attrs_explicit", "maybe_negate", jnp.float32, x,
                    negate=True)
        np.testing.assert_allclose(np.asarray(out), -np.asarray(x), rtol=1e-6)

    def test_toggle(self, bare_mod):
        x = jnp.ones(N, dtype=jnp.float32) * 5.0
        out_f = _call("test_attrs_bare", "maybe_negate", jnp.float32, x,
                      negate=False)
        out_t = _call("test_attrs_bare", "maybe_negate", jnp.float32, x,
                      negate=True)
        np.testing.assert_allclose(np.asarray(out_f),
                                   np.full(N, 5.0), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(out_t),
                                   np.full(N, -5.0), rtol=1e-6)


# ===========================================================================
# complex64 — passed as separate float32 re/im attrs
# (JAX FFI cannot encode numpy.complex64 as an XLA scalar attribute)
# ===========================================================================

class TestComplex64:
    def test_bare(self, bare_mod):
        """re/im passed as separate float32 attrs — |3+4j|^2 = 25."""
        x = jnp.ones(N, dtype=jnp.float32)
        out = _call("test_attrs_bare", "scale_c64", jnp.float32, x,
                    re=np.float32(3.0), im=np.float32(4.0))
        np.testing.assert_allclose(np.asarray(out),
                                   np.full(N, 25.0, dtype=np.float32),
                                   rtol=1e-5)

    def test_explicit(self, explicit_mod):
        x = jnp.ones(N, dtype=jnp.float32)
        out = _call("test_attrs_explicit", "scale_c64", jnp.float32, x,
                    re=np.float32(3.0), im=np.float32(4.0))
        np.testing.assert_allclose(np.asarray(out),
                                   np.full(N, 25.0, dtype=np.float32),
                                   rtol=1e-5)

    def test_pure_real(self, bare_mod):
        x = jnp.ones(N, dtype=jnp.float32)
        out = _call("test_attrs_bare", "scale_c64", jnp.float32, x,
                    re=np.float32(2.0), im=np.float32(0.0))  # |2+0j|^2 = 4
        np.testing.assert_allclose(np.asarray(out),
                                   np.full(N, 4.0, dtype=np.float32),
                                   rtol=1e-5)

    def test_value_sweep(self, bare_mod):
        x = jnp.ones(N, dtype=jnp.float32)
        for re, im in [(1, 0), (0, 1), (3, 4), (-2, 0)]:
            expected = float(re ** 2 + im ** 2)
            out = _call("test_attrs_bare", "scale_c64", jnp.float32, x,
                        re=np.float32(re), im=np.float32(im))
            np.testing.assert_allclose(np.asarray(out),
                                       np.full(N, expected, dtype=np.float32),
                                       rtol=1e-5,
                                       err_msg=f"re={re}, im={im}")


# ===========================================================================
# complex128 — passed as separate float64 re/im attrs
# (JAX FFI cannot encode numpy.complex128 as an XLA scalar attribute)
# ===========================================================================

class TestComplex128:
    def test_bare(self, bare_mod):
        x = jnp.ones(N, dtype=jnp.float64)
        out = _call("test_attrs_bare", "scale_c128", jnp.float64, x,
                    re=np.float64(3.0), im=np.float64(4.0))  # |3+4j|^2 = 25
        np.testing.assert_allclose(np.asarray(out),
                                   np.full(N, 25.0), rtol=1e-12)

    def test_explicit(self, explicit_mod):
        x = jnp.ones(N, dtype=jnp.float64)
        out = _call("test_attrs_explicit", "scale_c128", jnp.float64, x,
                    re=np.float64(3.0), im=np.float64(4.0))
        np.testing.assert_allclose(np.asarray(out),
                                   np.full(N, 25.0), rtol=1e-12)

    def test_precision(self, bare_mod):
        """float64 re/im preserve more precision than float32 would."""
        re_val, im_val = 1.0 + 1e-12, 1e-12
        x = jnp.ones(N, dtype=jnp.float64)
        expected = re_val ** 2 + im_val ** 2
        out = _call("test_attrs_bare", "scale_c128", jnp.float64, x,
                    re=np.float64(re_val), im=np.float64(im_val))
        np.testing.assert_allclose(np.asarray(out),
                                   np.full(N, expected), rtol=1e-12)


# ===========================================================================
# float16 / bfloat16 (raw uint16 bit representation)
# ===========================================================================

class TestFloat16Bits:
    """float16 attrs pass raw uint16 bits; the kernel interprets them.

    XLA FFI has no scalar AttrDecoding for F16/BF16, so both map to uint16_t
    in the XLA FFI binding.  The C++ function receives uint16_t and must
    reinterpret to __half / __nv_bfloat16 internally.
    """

    def test_explicit_float16_token_compiles(self):
        """attr.x:float16 compiles without error (uint16_t binding)."""
        import brainevent.source2kernel as jkb
        src = r"""
        #include <cuda_runtime.h>
        #include "brainevent/common.h"
        void f16_bits(BE::Tensor x, BE::Tensor out,
                      uint16_t bits, int64_t stream) {
            // kernel would reinterpret bits as __half; here just pass-through
            int n = (int)x.numel();
            (void)bits;
            (void)stream;
        }
        """
        jkb.load_cuda_inline(
            name="test_f16_bits_compile",
            cuda_sources=src,
            functions={"f16_bits": ["arg", "ret", "attr.bits:float16", "stream"]},
            force_rebuild=True,
        )

    def test_explicit_bfloat16_token_compiles(self):
        """attr.x:bfloat16 compiles without error (uint16_t binding)."""
        import brainevent.source2kernel as jkb
        src = r"""
        #include <cuda_runtime.h>
        #include "brainevent/common.h"
        void bf16_bits(BE::Tensor x, BE::Tensor out,
                       uint16_t bits, int64_t stream) {
            (void)bits;
            (void)stream;
        }
        """
        jkb.load_cuda_inline(
            name="test_bf16_bits_compile",
            cuda_sources=src,
            functions={"bf16_bits": ["arg", "ret", "attr.bits:bfloat16", "stream"]},
            force_rebuild=True,
        )


# ===========================================================================
# Multiple attributes
# ===========================================================================

class TestMultiAttr:
    def test_bare_two_attrs(self, bare_mod):
        x = jnp.ones(N, dtype=jnp.float32)
        out = _call("test_attrs_bare", "scale_add", jnp.float32, x,
                    scale=np.float32(3.0), offset=np.int32(10))
        np.testing.assert_allclose(np.asarray(out),
                                   np.full(N, 3.0 + 10.0, dtype=np.float32),
                                   rtol=1e-5)

    def test_explicit_two_attrs(self, explicit_mod):
        x = jnp.ones(N, dtype=jnp.float32)
        out = _call("test_attrs_explicit", "scale_add", jnp.float32, x,
                    scale=np.float32(3.0), offset=np.int32(10))
        np.testing.assert_allclose(np.asarray(out),
                                   np.full(N, 3.0 + 10.0, dtype=np.float32),
                                   rtol=1e-5)

    def test_combinations(self, bare_mod):
        x = jnp.ones(N, dtype=jnp.float32) * 2.0
        for scale, offset in [(1.0, 0), (2.0, 5), (0.5, -3), (-1.0, 100)]:
            out = _call("test_attrs_bare", "scale_add", jnp.float32, x,
                        scale=np.float32(scale), offset=np.int32(offset))
            expected = float(2.0 * scale + offset)
            np.testing.assert_allclose(
                np.asarray(out),
                np.full(N, expected, dtype=np.float32),
                rtol=1e-5, err_msg=f"scale={scale}, offset={offset}")


# ===========================================================================
# Error cases
# ===========================================================================

class TestAttrErrors:
    def test_bare_unresolvable_type(self):
        """Pointer-typed param → BEError with helpful message."""
        src = r"""
        #include <cuda_runtime.h>
        #include "brainevent/common.h"
        void bad(BE::Tensor x, BE::Tensor out,
                 const float* ptr, int64_t stream) { (void)ptr; }
        """
        with pytest.raises(KernelError, match="Cannot map C\\+\\+ type"):
            jkb.load_cuda_inline(
                name="test_attrs_err_ptr",
                cuda_sources=src,
                functions={"bad": ["arg", "ret", "attr.ptr", "stream"]},
                force_rebuild=True,
            )

    def test_bare_param_not_found(self):
        """Bare attr name that doesn't exist in C++ signature → BEError."""
        src = r"""
        #include <cuda_runtime.h>
        #include "brainevent/common.h"
        void my(BE::Tensor x, BE::Tensor out, float scale,
                int64_t stream) { (void)scale; }
        """
        with pytest.raises(KernelError, match="not found in signature"):
            jkb.load_cuda_inline(
                name="test_attrs_err_name",
                cuda_sources=src,
                functions={"my": ["arg", "ret", "attr.typo", "stream"]},
                force_rebuild=True,
            )

    def test_explicit_invalid_type_string(self):
        """Unsupported type string in explicit token → BEError."""
        src = r"""
        #include <cuda_runtime.h>
        #include "brainevent/common.h"
        void my(BE::Tensor x, BE::Tensor out, float s, int64_t stream) {}
        """
        with pytest.raises(KernelError, match="Invalid arg_spec token"):
            jkb.load_cuda_inline(
                name="test_attrs_err_type",
                cuda_sources=src,
                functions={"my": ["arg", "ret", "attr.s:complex32", "stream"]},
                force_rebuild=True,
            )
