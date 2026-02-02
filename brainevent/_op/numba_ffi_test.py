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
import os
import unittest

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import pytest

import brainstate
from brainevent._op.numba_ffi import numba_kernel

numba_installed = importlib.util.find_spec('numba') is not None


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernel1(unittest.TestCase):
    def test1(self):
        import numba

        @numba.njit
        def add_kernel_numba(x, y, out):
            out[...] = x + y

        kernel = numba_kernel(add_kernel_numba, outs=jax.ShapeDtypeStruct((64,), jax.numpy.float32))

        a = brainstate.random.rand(64)
        b = brainstate.random.rand(64)
        r1 = kernel(a, b)[0]
        print(r1)
        r2 = a + b
        print(jax.numpy.allclose(r1, r2))
