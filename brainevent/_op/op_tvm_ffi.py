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

# -*- coding: utf-8 -*-

from typing import Sequence

# Try to import TVM FFI - will fail gracefully if not available
try:
    import jax_tvm_ffi
    import tvm_ffi.cpp

    HAS_TVM_FFI = True
except ImportError:
    HAS_TVM_FFI = False

__all__ = [
    'register_cuda_kernels',
]


def register_cuda_kernels(
    source_code: str,
    module: str,
    functions: Sequence[str],
):
    """Compile CUDA kernels and register with JAX FFI."""

    if not isinstance(source_code, str):
        return ValueError("source_code must be a string")
    if not isinstance(module, str):
        return ValueError("module must be a string")
    if not isinstance(functions, Sequence) or not all(isinstance(f, str) for f in functions):
        return ValueError("functions must be a sequence of strings")

    if not HAS_TVM_FFI:
        return False

    try:
        # Compile CUDA module
        _cuda_module = tvm_ffi.cpp.load_inline(
            name=module,
            cuda_sources=source_code,
            functions=functions,
        )

        # Register each kernel with JAX FFI
        for name in functions:
            jax_tvm_ffi.register_ffi_target(
                f"{module}.{name}",
                getattr(_cuda_module, name),
                ["args", "rets", "ctx.stream"],
                platform="gpu",
            )

        return True
    except Exception as e:
        print(f"Failed to compile/register CUDA kernels: {e}")
        return False
