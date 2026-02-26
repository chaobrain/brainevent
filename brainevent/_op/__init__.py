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

from ._codegen import normalize_tokens
from ._compiler import CompilerBackend, CUDABackend, CPPBackend, HIPBackend
from ._pipeline import (
    load_cuda_inline,
    load_cuda_file,
    load_cuda_dir,
    load_cpp_inline,
    load_cpp_file,
    set_cache_dir,
    get_cache_dir,
    clear_cache,
    print_diagnostics,
)
from ._runtime import CompiledModule, register_ffi_target, list_registered_targets
from ._toolchain import so_ext
from .benchmark import BenchmarkConfig, BenchmarkRecord, BenchmarkResult, benchmark_function
from .main import XLACustomKernel, KernelEntry
from .numba_cuda_ffi import numba_cuda_kernel, numba_cuda_callable
from .numba_ffi import numba_kernel
from .util import defjvp, general_batching_rule, jaxinfo_to_warpinfo, jaxtype_to_warptype

__all__ = [
    'XLACustomKernel', 'KernelEntry',
    'BenchmarkConfig', 'BenchmarkRecord', 'BenchmarkResult', 'benchmark_function',
    'numba_kernel', 'numba_cuda_kernel', 'numba_cuda_callable',
    'defjvp', 'general_batching_rule',
    'jaxinfo_to_warpinfo', 'jaxtype_to_warptype',
    # kernix CUDA/C++ compilation API
    'load_cuda_inline', 'load_cuda_file', 'load_cuda_dir',
    'load_cpp_inline', 'load_cpp_file',
    'set_cache_dir', 'get_cache_dir', 'clear_cache', 'print_diagnostics',
    'CompiledModule', 'register_ffi_target', 'list_registered_targets',
    'normalize_tokens', 'so_ext',
    'CompilerBackend', 'CUDABackend', 'CPPBackend', 'HIPBackend',
]
