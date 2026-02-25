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

"""Test compilation caching."""

import time

from brainevent._test_util import requires_gpu

pytestmark = requires_gpu

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

__global__ void scale_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] * 2.0f;
}

void scale2x(BE::Tensor x, BE::Tensor out, int64_t stream) {
    int n = x.numel();
    scale_kernel<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(out.data_ptr()), n);
}
"""


def test_cache_hit_is_faster():
    """Second load_cuda_inline with same source should be much faster (cache hit)."""
    import brainevent.kernix as jkb

    # Clear any previous cache for this name
    jkb.clear_cache("test_cache_speed")

    # First call: compiles
    t0 = time.monotonic()
    jkb.load_cuda_inline(
        name="test_cache_speed",
        cuda_sources=CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        target_prefix="cache_test_a",
    )
    t_first = time.monotonic() - t0

    # Second call: should hit cache (much faster)
    t0 = time.monotonic()
    jkb.load_cuda_inline(
        name="test_cache_speed",
        cuda_sources=CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        target_prefix="cache_test_b",
    )
    t_second = time.monotonic() - t0

    # Cache hit should be at least 5x faster
    assert t_second < t_first * 0.5, (
        f"Cache not effective: first={t_first:.2f}s, second={t_second:.2f}s"
    )


def test_force_rebuild():
    """force_rebuild=True should recompile even with cache."""
    import brainevent.kernix as jkb

    mod = jkb.load_cuda_inline(
        name="test_force_rb",
        cuda_sources=CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        force_rebuild=True,
        target_prefix="force_rb_a",
    )
    assert "scale2x" in mod.function_names

    # Force rebuild again — should succeed without error
    mod2 = jkb.load_cuda_inline(
        name="test_force_rb",
        cuda_sources=CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        force_rebuild=True,
        target_prefix="force_rb_b",
    )
    assert "scale2x" in mod2.function_names


def test_clear_cache():
    """clear_cache removes cached entries."""
    import brainevent.kernix as jkb

    jkb.load_cuda_inline(
        name="test_clear",
        cuda_sources=CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        target_prefix="clear_test",
    )

    removed = jkb.clear_cache("test_clear")
    assert removed >= 1
