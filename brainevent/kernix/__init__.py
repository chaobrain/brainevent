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

"""brainevent.kernix: runtime C++/CUDA compilation bridge for JAX FFI."""


from ._pipeline import (
    clear_cache,
    get_cache_dir,
    load_cpp_file,
    load_cpp_inline,
    load_cuda_dir,
    load_cuda_file,
    load_cuda_inline,
    print_diagnostics,
    set_cache_dir,
)
from ._compiler import (
    CompilerBackend,
    CUDABackend,
    CPPBackend,
    HIPBackend,
)
from ._codegen import normalize_tokens
from ._toolchain import so_ext
from ._runtime import (
    CompiledModule,
    list_registered_targets,
    register_ffi_target,
)

__all__ = [
    # Core API — CUDA
    "load_cuda_inline",
    "load_cuda_file",
    "load_cuda_dir",
    # Core API — CPU / C++
    "load_cpp_inline",
    "load_cpp_file",
    "register_ffi_target",
    "CompiledModule",
    # Compiler backends (extensibility)
    "CompilerBackend",
    "CUDABackend",
    "CPPBackend",
    "HIPBackend",
    # Utilities
    "list_registered_targets",
    "normalize_tokens",
    "so_ext",
    "clear_cache",
    "set_cache_dir",
    "get_cache_dir",
    "print_diagnostics",
]
