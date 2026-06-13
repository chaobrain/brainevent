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

"""End-to-end test for host-side check propagation (H3).

A failing ``BE_CHECK`` / ``BE_CUDA_CHECK`` in host code must throw
``BE::CheckError`` / ``BE::CudaError``, which the auto-generated FFI wrapper
catches and converts to an ``xla::ffi::Error``.  JAX then raises a normal
Python exception.  The old behaviour called ``abort()``, killing the whole
interpreter with ``SIGABRT``.

The proof that the process survives is implicit: if the kernel aborted, this
pytest process would die and the test could never report ``assertRaises``
success.
"""

import platform

import jax
import jax.numpy as jnp
import pytest

import brainevent

if platform.platform().startswith('Windows'):
    pytest.skip(reason="Windows is not supported yet.", allow_module_level=True)

# A CPU kernel that fails a host-side invariant.  ``x.numel()`` is always
# non-negative, so ``x.numel() < 0`` is always false and the check fires at
# runtime (the compiler cannot prove it away from the source alone).
CHECK_FAIL_SRC = r"""
#include "brainevent/common.h"

void check_fail_cpu(const BE::Tensor x, BE::Tensor y) {
    BE_CHECK(x.numel() < 0) << "intentional failure: numel=" << x.numel();
    float* out_ptr = static_cast<float*>(y.data_ptr());
    out_ptr[0] = 1.0f;  // unreachable
}
"""


def test_host_check_failure_raises_not_aborts():
    """A failing BE_CHECK surfaces as a Python exception (no SIGABRT)."""
    mod = brainevent.load_cpp_inline(
        name="test_cpu_check_fail",
        cpp_sources=CHECK_FAIL_SRC,
        functions=["check_fail_cpu"],
        force_rebuild=True,
    )

    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), cpu)

    with pytest.raises(Exception) as excinfo:
        result = jax.ffi.ffi_call(
            "test_cpu_check_fail.check_fail_cpu",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)
        jax.block_until_ready(result)

    # The diagnostic from check.h propagates through the FFI error.
    message = str(excinfo.value)
    assert "CHECK FAILED" in message or "intentional failure" in message, message


def test_process_survives_after_check_failure():
    """After a check failure, the interpreter is still usable (not aborted)."""
    mod = brainevent.load_cpp_inline(
        name="test_cpu_check_fail2",
        cpp_sources=CHECK_FAIL_SRC.replace("check_fail_cpu", "check_fail_cpu2"),
        functions=["check_fail_cpu2"],
        force_rebuild=True,
    )
    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), cpu)

    with pytest.raises(Exception):
        jax.block_until_ready(
            jax.ffi.ffi_call(
                "test_cpu_check_fail2.check_fail_cpu2",
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                vmap_method="broadcast_all",
            )(x)
        )

    # Still alive: a normal JAX computation completes after the failure.
    assert float(jnp.sum(jnp.arange(5.0))) == 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
