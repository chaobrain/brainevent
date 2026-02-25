# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Test CPU/C++ compilation and FFI registration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent.source2kernel as jkb

ADD_ONE_SRC = r"""
#include "brainevent/common.h"

void add_one_cpu(const BE::Tensor x, BE::Tensor y) {
    int n = x.numel();
    const float* in_ptr = static_cast<const float*>(x.data_ptr());
    float* out_ptr = static_cast<float*>(y.data_ptr());
    for (int i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i] + 1.0f;
    }
}
"""

SCALE_SRC = r"""
#include "brainevent/common.h"

void scale_cpu(const BE::Tensor x, BE::Tensor y) {
    int n = x.numel();
    const float* in_ptr = static_cast<const float*>(x.data_ptr());
    float* out_ptr = static_cast<float*>(y.data_ptr());
    for (int i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i] * 2.0f;
    }
}
"""

MULTI_OUT_SRC = r"""
#include "brainevent/common.h"

void split_cpu(const BE::Tensor x,
               BE::Tensor lo, BE::Tensor hi) {
    int n = x.numel();
    int half = n / 2;
    const float* src = static_cast<const float*>(x.data_ptr());
    float* lo_ptr = static_cast<float*>(lo.data_ptr());
    float* hi_ptr = static_cast<float*>(hi.data_ptr());
    for (int i = 0; i < half; ++i) lo_ptr[i] = src[i];
    for (int i = 0; i < n - half; ++i) hi_ptr[i] = src[half + i];
}
"""


@pytest.fixture(scope="module")
def cpu_add_one_module():
    return jkb.load_cpp_inline(
        name="test_cpu_add_one",
        cpp_sources=ADD_ONE_SRC,
        functions=["add_one_cpu"],
        force_rebuild=True,
    )


def test_add_one_cpu(cpu_add_one_module):
    """CPU kernel: add 1 to each element."""

    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), cpu)

    result = jax.ffi.ffi_call(
        "test_cpu_add_one.add_one_cpu",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )(x)

    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(result), expected)


def test_add_one_cpu_jit(cpu_add_one_module):
    """CPU kernel works under jax.jit."""

    cpu = jax.devices("cpu")[0]

    @jax.jit
    def add_one(x):
        return jax.ffi.ffi_call(
            "test_cpu_add_one.add_one_cpu",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    x = jax.device_put(jnp.arange(256, dtype=jnp.float32), cpu)
    result = add_one(x)
    expected = np.arange(256, dtype=np.float32) + 1.0
    np.testing.assert_allclose(np.asarray(result), expected)


def test_explicit_dict_form():
    """CPU kernel with explicit dict-form functions."""

    mod = jkb.load_cpp_inline(
        name="test_cpu_scale",
        cpp_sources=SCALE_SRC,
        functions={"scale_cpu": ["arg", "ret"]},
        force_rebuild=True,
    )

    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), cpu)
    result = jax.ffi.ffi_call(
        "test_cpu_scale.scale_cpu",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )(x)

    expected = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(result), expected)


def test_multi_output_cpu():
    """CPU kernel with two output buffers."""
    mod = jkb.load_cpp_inline(
        name="test_cpu_split",
        cpp_sources=MULTI_OUT_SRC,
        functions=["split_cpu"],
        force_rebuild=True,
    )

    cpu = jax.devices("cpu")[0]
    n = 256
    x = jax.device_put(jnp.arange(n, dtype=jnp.float32), cpu)

    lo, hi = jax.ffi.ffi_call(
        "test_cpu_split.split_cpu",
        (
            jax.ShapeDtypeStruct((n // 2,), jnp.float32),
            jax.ShapeDtypeStruct((n // 2,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(x)

    np.testing.assert_allclose(
        np.asarray(lo), np.arange(n // 2, dtype=np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(hi), np.arange(n // 2, n, dtype=np.float32)
    )
