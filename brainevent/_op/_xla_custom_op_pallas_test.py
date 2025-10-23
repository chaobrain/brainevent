# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import unittest

import brainstate as bst
import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl

import brainevent


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason="GPU not available")
class TestNumbaCPU(unittest.TestCase):
    def test1(self):
        def gpu_kernel(x_info, **kwargs):
            def add_vectors_kernel(x_ref, y_ref, o_ref):
                x, y = x_ref[...], y_ref[...]
                o_ref[...] = x + y

            return pl.pallas_call(
                add_vectors_kernel,
                out_shape=[jax.ShapeDtypeStruct(x_info.shape, x_info.dtype)],
                interpret=jax.default_backend() == 'cpu',
            )

        prim = brainevent.XLACustomKernel(
            'add',
            gpu_kernel=brainevent.PallasKernelGenerator(gpu_kernel),
        )

        a = bst.random.rand(64)
        b = bst.random.rand(64)
        x_info = jax.ShapeDtypeStruct(a.shape, a.dtype)
        r1 = prim(a, b, outs=[jax.ShapeDtypeStruct((64,), jax.numpy.float32)], x_info=x_info)
        r2 = gpu_kernel(x_info)(a, b)

        assert jnp.allclose(r1[0], r2[0])
