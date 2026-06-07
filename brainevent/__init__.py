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


from . import config
from ._csr import (
    CSR, CSC,
    binary_csrmv, binary_csrmv_p,
    binary_csrmv_indexed, binary_csrmv_indexed_p,
    binary_csrmm, binary_csrmm_p,
    binary_csrmm_indexed, binary_csrmm_indexed_p,
    csrmv, csrmv_p,
    csrmm, csrmm_p,
    csrmv_yw2y, cscmv_yw2y, csrmv_yw2y_p,
    update_csr_on_binary_pre, update_csr_on_binary_pre_p,
    update_csr_on_binary_post, update_csr_on_binary_post_p,
    update_csc_on_binary_pre, update_csc_on_binary_post,
    csr_slice_rows, csr_slice_rows_p,
)
from ._data import (
    DataRepresentation,
    JITCMatrix,
)
from ._dense import (
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
    fcnmv, fcnmm, fcnmv_yw2y,
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
)
from ._jit_scalar import (
    JITCScalarMatrix, JITCScalarR, JITCScalarC,
    binary_jitsmv, binary_jitsmv_p,
    binary_jitsmm, binary_jitsmm_p,
    jits, jits_p,
    jitsmv, jitsmv_p,
    jitsmm, jitsmm_p,
)
from ._jit_uniform import (
    JITCUniformR, JITCUniformC,
    binary_jitumv, binary_jitumv_p,
    binary_jitumm, binary_jitumm_p,
    jitu, jitu_p,
    jitumv, jitumv_p,
    jitumm, jitumm_p,
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
    'csrmv_yw2y', 'cscmv_yw2y', 'csrmv_yw2y_p',
    'update_csr_on_binary_pre', 'update_csr_on_binary_pre_p',
    'update_csr_on_binary_post', 'update_csr_on_binary_post_p',
    'update_csc_on_binary_pre', 'update_csc_on_binary_post',
    'csr_slice_rows', 'csr_slice_rows_p',

    # --- dense matrix --- #
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
    'JITCNormalR', 'JITCNormalC',
    'binary_jitnmv', 'binary_jitnmv_p',
    'binary_jitnmm', 'binary_jitnmm_p',
    'jitn', 'jitn_p',
    'jitnmv', 'jitnmv_p',
    'jitnmm', 'jitnmm_p',
    'JITCUniformR', 'JITCUniformC',
    'binary_jitumv', 'binary_jitumv_p',
    'binary_jitumm', 'binary_jitumm_p',
    'jitu', 'jitu_p',
    'jitumv', 'jitumv_p',
    'jitumm', 'jitumm_p',

    # --- Fixed number connectivity --- #
    'FixedNumConn', 'FixedNumPerPost', 'FixedNumPerPre',
    'binary_fcnmv', 'binary_fcnmv_p',
    'binary_fcnmm', 'binary_fcnmm_p',
    'fcnmv',
    'fcnmm',
    'fcnmv_yw2y',
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
#
# Retired names stay *resolvable* so v0.0.7 code keeps importing:
#   * renamed names return their replacement and emit a ``DeprecationWarning``;
#   * names whose underlying functionality was removed raise an
#     ``AttributeError`` stating the concrete migration path.
# These names are intentionally NOT part of ``__all__`` -- they are hidden,
# deprecated aliases surfaced only on explicit access (PEP 562).
# ---------------------------------------------------------------------------

#: old public name -> (replacement object, replacement display name)
_DEPRECATED_RENAMES = {
    'EventArray': (BinaryArray, 'BinaryArray'),
    'csr_on_pre': (update_csr_on_binary_pre, 'update_csr_on_binary_pre'),
    'csr2csc_on_post': (update_csr_on_binary_post, 'update_csr_on_binary_post'),
    'dense_on_pre': (update_dense_on_binary_pre, 'update_dense_on_binary_pre'),
    'dense_on_post': (update_dense_on_binary_post, 'update_dense_on_binary_post'),
    'JITCHomoR': (JITCScalarR, 'JITCScalarR'),
    'JITCHomoC': (JITCScalarC, 'JITCScalarC'),
    'FixedPostNumConn': (FixedNumPerPre, 'FixedNumPerPre'),
    'FixedPreNumConn': (FixedNumPerPost, 'FixedNumPerPost'),
}

_COO_MIGRATION = (
    'The COO sparse format was removed in brainevent 0.1.0. Use CSR / CSC '
    'instead (brainevent.CSR / brainevent.CSC); convert indices with '
    'brainevent.coo2csr or the *_index helpers (csr_to_coo_index, '
    'coo_to_csc_index, csr_to_csc_index, csc_to_csr_index).'
)
_FCN_PACK_MIGRATION = (
    'The explicit bitpack_/compact_ FCN kernels were removed in brainevent '
    '0.1.0; they were unified into fcnmv / fcnmm, which dispatch on the input '
    'event type. Wrap spikes with brainevent.BitPackedBinary or '
    'brainevent.CompactBinary and call brainevent.fcnmv / brainevent.fcnmm.'
)
_LAYOUT_MIGRATION = (
    'The fixed-number-connection layout abstraction was removed. Use '
    'FixedNumPerPost / FixedNumPerPre directly (favorable/unfavorable dispatch '
    'is now internal).'
)

#: old public name -> migration message (functionality removed, no drop-in)
_DEPRECATED_REMOVED = {}
_DEPRECATED_REMOVED.update({
    name: _COO_MIGRATION for name in (
        'COO',
        'binary_coomv', 'binary_coomv_p',
        'binary_coomm', 'binary_coomm_p',
        'coomv', 'coomv_p',
        'coomm', 'coomm_p',
        'update_coo_on_binary_pre', 'update_coo_on_binary_post',
        'update_coo_on_binary_pre_p', 'update_coo_on_binary_post_p',
    )
})
_DEPRECATED_REMOVED.update({
    name: _FCN_PACK_MIGRATION for name in (
        'bitpack_binary_fcnmv', 'bitpack_binary_fcnmv_p',
        'bitpack_binary_fcnmm', 'bitpack_binary_fcnmm_p',
        'compact_binary_fcnmv', 'compact_binary_fcnmv_p',
        'compact_binary_fcnmm', 'compact_binary_fcnmm_p',
    )
})
_DEPRECATED_REMOVED.update({
    'EllLayout': _LAYOUT_MIGRATION,
    'CscLayout': _LAYOUT_MIGRATION,
})


def __getattr__(name):
    """Resolve retired v0.0.7 public names (PEP 562 module-level hook)."""
    import warnings
    if name in _DEPRECATED_RENAMES:
        replacement, new_name = _DEPRECATED_RENAMES[name]
        warnings.warn(
            f'brainevent.{name} is deprecated and will be removed in a future '
            f'release; use brainevent.{new_name} instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return replacement
    if name in _DEPRECATED_REMOVED:
        raise AttributeError(
            f'brainevent.{name} was removed in 0.1.0. {_DEPRECATED_REMOVED[name]}'
        )
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(
        set(globals())
        | set(_DEPRECATED_RENAMES)
        | set(_DEPRECATED_REMOVED)
    )


