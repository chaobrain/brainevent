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

import importlib.util
import unittest

import brainstate
import jax
import pytest

import brainevent

numba_installed = importlib.util.find_spec('numba') is not None

if numba_installed:
    pass


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaCPU(unittest.TestCase):
    def test1(self):
        def cpu_kernel(**kwargs):
            @brainevent.numba_kernel
            def add_kernel_numba(x, y, out):
                out[...] = x + y

            return add_kernel_numba

        def gpu_kernel(**kwargs):
            def add_vectors_kernel(x_ref, y_ref, o_ref):
                x, y = x_ref[...], y_ref[...]
                o_ref[...] = x + y

            return brainevent.pallas_kernel(add_vectors_kernel, outs=kwargs['outs'])

        prim = brainevent.XLACustomKernel('add')
        prim.def_cpu_kernel(cpu_kernel)
        prim.def_gpu_kernel(pallas=gpu_kernel)

        a = brainstate.random.rand(64)
        b = brainstate.random.rand(64)
        x_info = jax.ShapeDtypeStruct(a.shape, a.dtype)
        r1 = prim(a, b, outs=[jax.ShapeDtypeStruct((64,), jax.numpy.float32)])
