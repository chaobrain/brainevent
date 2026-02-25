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

"""Test multiple output buffers."""

import pytest
import numpy as np

from brainevent._test_util import requires_gpu

pytestmark = requires_gpu


CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

__global__ void split_kernel(const float* x, float* lo, float* hi,
                             int n, int split) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < split) lo[idx] = x[idx];
    if (idx < n - split) hi[idx] = x[split + idx];
}

void min_max(BE::Tensor x, BE::Tensor out_min,
             BE::Tensor out_max, int64_t stream) {
    // Simple test: copy first half to out_min, second half to out_max
    int n = x.numel();
    int half = n / 2;
    split_kernel<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(out_min.data_ptr()),
        static_cast<float*>(out_max.data_ptr()),
        n, half);
}
"""


@pytest.fixture(scope="module")
def multi_out_module():
    import brainevent
    return brainevent.load_cuda_inline(
        name="test_multi_out",
        cuda_sources=CUDA_SRC,
        functions={
            "min_max": ["arg", "ret", "ret", "stream"],
        },
        force_rebuild=True,
    )


def test_two_outputs(multi_out_module):
    """Function with two output buffers."""
    import jax
    import jax.numpy as jnp

    n = 256
    x = jnp.arange(n, dtype=jnp.float32)

    lo, hi = jax.ffi.ffi_call(
        "test_multi_out.min_max",
        (
            jax.ShapeDtypeStruct((n // 2,), jnp.float32),
            jax.ShapeDtypeStruct((n // 2,), jnp.float32),
        ),
    )(x)

    np.testing.assert_allclose(
        np.asarray(lo), np.arange(n // 2, dtype=np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(hi), np.arange(n // 2, n, dtype=np.float32)
    )
