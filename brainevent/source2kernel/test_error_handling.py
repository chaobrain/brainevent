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

import jax as _jax
import pytest
import pytest as _pytest

requires_gpu = _pytest.mark.skipif(
    not (bool(_jax.devices("gpu")) if True else False),
    reason="No GPU detected via jax.devices('gpu')",
)

pytestmark = requires_gpu

import brainevent.source2kernel as jkb


def test_compilation_error():
    """Invalid CUDA source raises CompilationError."""
    import brainevent.source2kernel as jkb

    with pytest.raises(jkb.CompilationError, match="compilation failed"):
        jkb.load_cuda_inline(
            name="test_bad_src",
            cuda_sources="THIS IS NOT VALID CUDA !!!",
            functions={"nonexistent": ["ret", "stream"]},
            force_rebuild=True,
            auto_register=False,
        )


def test_invalid_arg_spec_token():
    """Invalid arg_spec token raises BEError."""

    with pytest.raises(jkb.KernelError, match="Invalid arg_spec token"):
        jkb.load_cuda_inline(
            name="test_bad_spec",
            cuda_sources="void f() {}",
            functions={"f": ["arg", "INVALID_TOKEN"]},
            auto_register=False,
        )


def test_missing_ret_in_arg_spec():
    """arg_spec without 'ret' raises BEError."""

    with pytest.raises(jkb.KernelError, match="at least one 'ret'"):
        jkb.load_cuda_inline(
            name="test_no_ret",
            cuda_sources="void f() {}",
            functions={"f": ["arg", "stream"]},
            auto_register=False,
        )


def test_duplicate_registration():
    """Registering the same target name twice raises RegistrationError."""

    CUDA_SRC = r"""
    #include <cuda_runtime.h>
    #include "brainevent/common.h"
    void noop(BE::Tensor out, int64_t stream) {}
    """

    mod = jkb.load_cuda_inline(
        name="test_dup_reg",
        cuda_sources=CUDA_SRC,
        functions={"noop": ["ret", "stream"]},
        auto_register=False,
        force_rebuild=True,
    )

    jkb.register_ffi_target("test_dup.noop", mod, "noop")

    with pytest.raises(jkb.KernelRegistrationError, match="already registered"):
        jkb.register_ffi_target("test_dup.noop", mod, "noop")


def test_diagnostics_runs():
    """print_diagnostics() doesn't crash."""
    # Just ensure it runs without error
    jkb.print_diagnostics()
