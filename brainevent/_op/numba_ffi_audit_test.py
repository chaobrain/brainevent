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
"""Regression tests for the 2026-06-13 ``_op`` audit (CPU / Numba bridge).

Covers: C1 (kernel exceptions must propagate, not silently report success),
C2 (fp16/bf16/complex buffer views must be byte-accurate), M7 (metadata struct
shape), and H1 (FFI-target registration must be cached, not leaked per call).
"""

import importlib.util
import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
import numpy as np
import pytest

numba_installed = importlib.util.find_spec('numba') is not None
cpu_platform = jax.default_backend() == 'cpu'
if not cpu_platform or not numba_installed:
    pytest.skip(
        allow_module_level=True,
        reason='Numba CPU FFI audit tests only run on CPU platform with Numba installed',
    )

import numba

from brainevent._op import numba_ffi as _m
from brainevent._op.numba_ffi import (
    _numpy_from_buffer,
    XLA_FFI_Metadata,
    numba_kernel,
)


# --- C2: byte-accurate buffer views for dtypes ctypes cannot represent ---------

class TestBufferViewDtypes:
    """``_numpy_from_buffer`` must reconstruct every fixed-width dtype exactly."""

    def _roundtrip(self, arr):
        view = _numpy_from_buffer(arr.ctypes.data, arr.shape, arr.dtype)
        np.testing.assert_array_equal(view, arr)

    def test_float16(self):
        self._roundtrip(np.arange(6, dtype=np.float16).reshape(2, 3))

    def test_bfloat16(self):
        import ml_dtypes
        bf16 = np.dtype(ml_dtypes.bfloat16)
        self._roundtrip(np.arange(6, dtype=bf16).reshape(2, 3))

    def test_complex64(self):
        arr = (np.arange(4) + 1j * np.arange(4)).astype(np.complex64)
        self._roundtrip(arr)

    def test_complex128(self):
        arr = (np.arange(4) + 1j * np.arange(4)).astype(np.complex128)
        self._roundtrip(arr)

    def test_float32_still_correct(self):
        self._roundtrip(np.arange(12, dtype=np.float32).reshape(3, 4))


# --- M7: metadata struct must expose state_type_id -----------------------------

def test_metadata_struct_has_state_type_id():
    names = {name for name, _ in XLA_FFI_Metadata._fields_}
    assert 'state_type_id' in names


# --- C1: kernel exceptions must propagate, not be reported as success ----------

class TestErrorPropagation:
    def test_raising_kernel_surfaces_exception(self):
        @numba.njit
        def boom(x, out):
            raise ValueError('intentional kernel failure')

        kernel = numba_kernel(boom, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        with pytest.raises(Exception):
            jax.block_until_ready(kernel(jnp.arange(4, dtype=jnp.float32)))


# --- H1: registration must be cached, not leaked once per call -----------------

class TestRegistrationCaching:
    def test_eager_calls_do_not_leak_targets(self):
        @numba.njit
        def add1(x, out):
            for i in range(out.size):
                out[i] = x[i] + 1.0

        kernel = numba_kernel(add1, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        x = jnp.arange(4, dtype=jnp.float32)
        jax.block_until_ready(kernel(x))  # warm up / first registration
        before = len(_m._NUMBA_CPU_FFI_HANDLES)
        for _ in range(8):
            jax.block_until_ready(kernel(x))
        after = len(_m._NUMBA_CPU_FFI_HANDLES)
        assert after == before, f'leaked {after - before} FFI targets across 8 eager calls'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
