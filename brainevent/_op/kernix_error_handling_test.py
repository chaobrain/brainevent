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

"""Test error handling and validation."""

import pytest
from brainevent._test_util import requires_gpu

pytestmark = requires_gpu

import brainevent


def test_compilation_error():
    """Invalid CUDA source raises CompilationError."""

    with pytest.raises(brainevent.KernelCompilationError, match="compilation failed"):
        brainevent.load_cuda_inline(
            name="test_bad_src",
            cuda_sources="THIS IS NOT VALID CUDA !!!",
            functions={"nonexistent": ["ret", "stream"]},
            force_rebuild=True,
            auto_register=False,
        )


def test_invalid_arg_spec_token():
    """Invalid arg_spec token raises BEError."""
    from brainevent._error import KernelError

    with pytest.raises(KernelError, match="Invalid arg_spec token"):
        brainevent.load_cuda_inline(
            name="test_bad_spec",
            cuda_sources="void f() {}",
            functions={"f": ["arg", "INVALID_TOKEN"]},
            auto_register=False,
        )


def test_missing_ret_in_arg_spec():
    """arg_spec without 'ret' raises BEError."""
    from brainevent._error import KernelError

    with pytest.raises(KernelError, match="at least one 'ret'"):
        brainevent.load_cuda_inline(
            name="test_no_ret",
            cuda_sources="void f() {}",
            functions={"f": ["arg", "stream"]},
            auto_register=False,
        )


def test_duplicate_registration_same_module_is_idempotent():
    """Re-registering the identical module under one name is a no-op (M5).

    Registration is idempotent for an equivalent module (same shared-library
    path, function, platform) so repeated eager registration does not raise or
    clobber the live keep-alive.
    """
    CUDA_SRC = r"""
    #include <cuda_runtime.h>
    #include "brainevent/common.h"
    void noop(BE::Tensor out, int64_t stream) {}
    """

    mod = brainevent.load_cuda_inline(
        name="test_dup_reg",
        cuda_sources=CUDA_SRC,
        functions={"noop": ["ret", "stream"]},
        auto_register=False,
        force_rebuild=True,
    )

    brainevent.register_ffi_target("test_dup.noop", mod, "noop")
    # Idempotent: the identical module re-registered is a no-op, not an error.
    brainevent.register_ffi_target("test_dup.noop", mod, "noop")
    assert "test_dup.noop" in brainevent.list_registered_targets()


def test_duplicate_registration_different_module_raises():
    """A *different* module under an already-used name is refused (M5).

    ``jax.ffi.register_ffi_target`` would silently overwrite the live target,
    dropping a still-referenced module; the wrapper raises instead.
    """
    from brainevent._error import KernelRegistrationError

    CUDA_SRC = r"""
    #include <cuda_runtime.h>
    #include "brainevent/common.h"
    void noop(BE::Tensor out, int64_t stream) {}
    """

    mod = brainevent.load_cuda_inline(
        name="test_dup_reg_a",
        cuda_sources=CUDA_SRC,
        functions={"noop": ["ret", "stream"]},
        auto_register=False,
        force_rebuild=True,
    )
    # A second, distinct shared library (different name -> different .so path).
    mod2 = brainevent.load_cuda_inline(
        name="test_dup_reg_b",
        cuda_sources=CUDA_SRC,
        functions={"noop": ["ret", "stream"]},
        auto_register=False,
        force_rebuild=True,
    )

    brainevent.register_ffi_target("test_dup_conflict.noop", mod, "noop")

    with pytest.raises(KernelRegistrationError, match="already registered to a different"):
        brainevent.register_ffi_target("test_dup_conflict.noop", mod2, "noop")


def test_diagnostics_runs():
    """print_diagnostics() doesn't crash."""
    # Just ensure it runs without error
    brainevent.print_diagnostics()
