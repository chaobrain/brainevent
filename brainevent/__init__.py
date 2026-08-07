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


from . import _deprecation
from . import config
from ._csr import (
    CSR, CSC,
    binary_csrmv, binary_csrmv_p,
    binary_csrmv_indexed, binary_csrmv_indexed_p,
    binary_csrmm, binary_csrmm_p,
    binary_csrmm_indexed, binary_csrmm_indexed_p,
    csrmv, csrmv_p,
    csrmm, csrmm_p,
    csrmv_dt2t, cscmv_dt2t, csrmv_dt2t_p,
    csrmm_dt2t, cscmm_dt2t, csrmm_dt2t_p,
    update_csr_on_binary_pre, update_csr_on_binary_pre_p,
    update_csr_on_binary_post, update_csr_on_binary_post_p,
    update_csc_on_binary_pre, update_csc_on_binary_post,
    csr_slice_rows, csr_slice_rows_p,
    HybridConfig, get_hybrid_config, init_csr_config,
)
from ._data import (
    DataRepresentation,
    JITCMatrix,
)
from ._dense import (
    Dense,
    binary_densemv, binary_densemv_p,
    binary_densemm, binary_densemm_p,
    update_dense_on_binary_pre, update_dense_on_binary_pre_p,
    update_dense_on_binary_post, update_dense_on_binary_post_p,
)
from ._error import (
    BrainEventError,
    MathError,
    UnsupportedOperationError,
    KernelError,
    KernelNotAvailableError,
    KernelCompilationError,
    KernelFallbackExhaustedError,
    KernelExecutionError,
    KernelToolchainError,
    CompilationError,
    KernelRegistrationError,
    BenchmarkDataFnNotProvidedError,
    CUDANotInstalledError,
    NvccNotFoundError,
    HostCompilerNotFoundError,
    HeaderNotFoundError,
    GpuArchDetectionError,
    HostCompilerIncompatibleError,
    UnsupportedArchError,
    KernelLoadError,
)
from ._event import (
    EventRepresentation,
    BinaryArray,
    BitPackedBinary,
    bitpack,
    CompactBinary,
)
from ._fcn import (
    FixedNumConn, FixedNumPerPost, FixedNumPerPre,
    binary_fcnmv, binary_fcnmv_p,
    binary_fcnmm, binary_fcnmm_p,
    fcnmv, fcnmm, fcnmv_dt2t, fcnmm_dt2t,
    update_fixed_post_conn_on_binary_pre,
    update_fixed_pre_conn_on_binary_post,
    fcn_plasticity_row_p,
)
from ._jit_normal import (
    JITCNormalR, JITCNormalC,
    binary_jitnmv, binary_jitnmv_p,
    binary_jitnmm, binary_jitnmm_p,
    jitn, jitn_p,
    jitnmv, jitnmv_p,
    jitnmm, jitnmm_p,
    jitnmv_dt2t,
)
from ._jit_scalar import (
    JITCScalarMatrix, JITCScalarR, JITCScalarC,
    binary_jitsmv, binary_jitsmv_p,
    binary_jitsmm, binary_jitsmm_p,
    jits, jits_p,
    jitsmv, jitsmv_p,
    jitsmm, jitsmm_p,
    jitsmv_dt2t,
)
from ._jit_uniform import (
    JITCUniformR, JITCUniformC,
    binary_jitumv, binary_jitumv_p,
    binary_jitumm, binary_jitumm_p,
    jitu, jitu_p,
    jitumv, jitumv_p,
    jitumm, jitumm_p,
    jitumv_dt2t,
)
from ._misc import (
    csr_to_coo_index, coo_to_csc_index, csr_to_csc_index, csc_to_csr_index, coo2csr,
)
from ._op import (
    XLACustomKernel, KernelEntry,
    BenchmarkConfig, BenchmarkRecord, BenchmarkResult, benchmark_function,
    numba_kernel, numba_cuda_kernel, numba_cuda_callable,
    defjvp, general_batching_rule,
    jaxtype_to_warptype, jaxinfo_to_warpinfo,
    load_cuda_inline,
    load_cuda_file,
    load_cuda_dir,
    load_cpp_inline,
    load_cpp_file,
    set_cache_dir,
    get_cache_dir,
    clear_cache,
    print_diagnostics,
    CompiledModule,
    register_ffi_target,
    list_registered_targets,
    normalize_tokens,
    CompilerBackend,
    CUDABackend,
    CPPBackend,
    HIPBackend,
)
from ._pallas_random import (
    PallasLFSR88RNG, PallasLFSR113RNG, PallasLFSR128RNG,
    PallasLFSRRNG, get_pallas_lfsr_rng_class,
)
from ._registry import (
    get_registry, get_primitives_by_tags, get_all_primitive_names,
)
from ._version import __version__, __version_info__

__all__ = [

    # --- representing events --- #
    'EventRepresentation',
    'BinaryArray',
    'BitPackedBinary',
    'bitpack',
    'CompactBinary',

    # --- representing sparse data --- #
    'DataRepresentation',

    # --- CSR --- #
    'CSR', 'CSC',
    'binary_csrmv', 'binary_csrmv_p',
    'binary_csrmv_indexed', 'binary_csrmv_indexed_p',
    'binary_csrmm', 'binary_csrmm_p',
    'binary_csrmm_indexed', 'binary_csrmm_indexed_p',
    'csrmv', 'csrmv_p',
    'csrmm', 'csrmm_p',
    'csrmv_dt2t', 'cscmv_dt2t', 'csrmv_dt2t_p',
    'csrmm_dt2t', 'cscmm_dt2t', 'csrmm_dt2t_p',
    'HybridConfig', 'get_hybrid_config', 'init_csr_config',
    'update_csr_on_binary_pre', 'update_csr_on_binary_pre_p',
    'update_csr_on_binary_post', 'update_csr_on_binary_post_p',
    'update_csc_on_binary_pre', 'update_csc_on_binary_post',
    'csr_slice_rows', 'csr_slice_rows_p',

    # --- dense matrix --- #
    'Dense',
    'binary_densemv', 'binary_densemv_p',
    'binary_densemm', 'binary_densemm_p',
    'update_dense_on_binary_pre', 'update_dense_on_binary_pre_p',
    'update_dense_on_binary_post', 'update_dense_on_binary_post_p',

    # --- Just-In-Time Connectivity matrix --- #
    'JITCMatrix',
    'JITCScalarMatrix', 'JITCScalarR', 'JITCScalarC',
    'binary_jitsmv', 'binary_jitsmv_p',
    'binary_jitsmm', 'binary_jitsmm_p',
    'jits', 'jits_p',
    'jitsmv', 'jitsmv_p',
    'jitsmm', 'jitsmm_p',
    'jitsmv_dt2t',
    'JITCNormalR', 'JITCNormalC',
    'binary_jitnmv', 'binary_jitnmv_p',
    'binary_jitnmm', 'binary_jitnmm_p',
    'jitn', 'jitn_p',
    'jitnmv', 'jitnmv_p',
    'jitnmm', 'jitnmm_p',
    'jitnmv_dt2t',
    'JITCUniformR', 'JITCUniformC',
    'binary_jitumv', 'binary_jitumv_p',
    'binary_jitumm', 'binary_jitumm_p',
    'jitu', 'jitu_p',
    'jitumv', 'jitumv_p',
    'jitumm', 'jitumm_p',
    'jitumv_dt2t',

    # --- Fixed number connectivity --- #
    'FixedNumConn', 'FixedNumPerPost', 'FixedNumPerPre',
    'binary_fcnmv', 'binary_fcnmv_p',
    'binary_fcnmm', 'binary_fcnmm_p',
    'fcnmv',
    'fcnmm',
    'fcnmv_dt2t',
    'fcnmm_dt2t',
    'update_fixed_post_conn_on_binary_pre',
    'update_fixed_pre_conn_on_binary_post',
    'fcn_plasticity_row_p',

    # --- operator customization routines --- #
    'XLACustomKernel', 'KernelEntry',
    'BenchmarkConfig', 'BenchmarkRecord', 'BenchmarkResult', 'benchmark_function',
    'numba_kernel', 'numba_cuda_kernel', 'numba_cuda_callable',
    'defjvp', 'general_batching_rule',
    'jaxtype_to_warptype', 'jaxinfo_to_warpinfo',

    # --- CUDA/C++ compilation API --- #
    'load_cuda_inline', 'load_cuda_file', 'load_cuda_dir',
    'load_cpp_inline', 'load_cpp_file',
    'set_cache_dir', 'get_cache_dir', 'clear_cache', 'print_diagnostics',
    'CompiledModule', 'register_ffi_target', 'list_registered_targets',
    'normalize_tokens',
    'CompilerBackend', 'CUDABackend', 'CPPBackend', 'HIPBackend',

    # --- Pallas kernel --- #
    'PallasLFSR88RNG', 'PallasLFSR113RNG', 'PallasLFSR128RNG',
    'PallasLFSRRNG', 'get_pallas_lfsr_rng_class',

    # --- errors --- #
    'BrainEventError',
    'MathError',
    'UnsupportedOperationError',
    'KernelError',
    'KernelNotAvailableError',
    'KernelCompilationError',
    'KernelFallbackExhaustedError',
    'KernelExecutionError',
    'KernelToolchainError',
    'CompilationError',
    'KernelRegistrationError',
    'BenchmarkDataFnNotProvidedError',
    'CUDANotInstalledError',
    'NvccNotFoundError',
    'HostCompilerNotFoundError',
    'HeaderNotFoundError',
    'GpuArchDetectionError',
    'HostCompilerIncompatibleError',
    'UnsupportedArchError',
    'KernelLoadError',

    # --- utilities --- #
    'csr_to_coo_index', 'coo_to_csc_index', 'csr_to_csc_index', 'csc_to_csr_index', 'coo2csr',

    # --- config & registry --- #
    'config', 'get_registry', 'get_primitives_by_tags', 'get_all_primitive_names',

]


# ---------------------------------------------------------------------------
# Backward-compatibility shim for public names retired between v0.0.7 and 0.1.0.
# The tables and resolution logic live in ``brainevent._deprecation``; this module
# only installs the PEP 562 hooks that route attribute access through them.
# ---------------------------------------------------------------------------


def __getattr__(name):
    """Resolve retired v0.0.7 public names (PEP 562 module-level hook)."""
    return _deprecation.resolve(name, globals())


def __dir__():
    return _deprecation.public_dir(globals())
