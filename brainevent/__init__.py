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


__version__ = "0.0.6"
__version_info__ = tuple(map(int, __version__.split(".")))

from . import config
from ._coo import (
    COO,
    binary_coomv, binary_coomv_p, binary_coomm, binary_coomm_p,
    coomv, coomv_p, coomm, coomm_p,
    update_coo_on_binary_pre, update_coo_on_binary_post,
    update_coo_on_binary_pre_p, update_coo_on_binary_post_p,
)
from ._csr import (
    CSR, CSC,
    binary_csrmv, binary_csrmv_p, binary_csrmm, binary_csrmm_p,
    csrmv, csrmv_p, csrmm, csrmm_p, csrmv_yw2y, csrmv_yw2y_p,
    update_csr_on_binary_pre, update_csr_on_binary_pre_p,
    update_csr_on_binary_post, update_csr_on_binary_post_p,
    spfloat_csrmv, spfloat_csrmv_p, spfloat_csrmm, spfloat_csrmm_p,
    csr_slice_rows, csr_slice_rows_p,
)
from ._dense import (
    binary_densemv, binary_densemv_p,
    binary_densemm, binary_densemm_p,
    indexed_binary_densemv, indexed_binary_densemv_p,
    indexed_binary_densemm, indexed_binary_densemm_p,
    update_dense_on_binary_pre, update_dense_on_binary_pre_p,
    update_dense_on_binary_post, update_dense_on_binary_post_p,
    spfloat_densemv, spfloat_densemv_p,
    spfloat_densemm, spfloat_densemm_p,
)
from ._error import (
    MathError,
    KernelNotAvailableError,
    KernelCompilationError,
    KernelFallbackExhaustedError,
    KernelExecutionError,
)
from ._event import (
    EventRepresentation,
    IndexedEventRepresentation,
    BinaryArray,
    IndexedBinary1d,
    IndexedBinary2d,
    SparseFloat,
    IndexedSpFloat1d,
    IndexedSpFloat2d,
    binary_array_index,

)
from ._fcn import (
    FixedNumConn, FixedPreNumConn, FixedPostNumConn,
    binary_fcnmv, binary_fcnmv_p, binary_fcnmm, binary_fcnmm_p,
    fcnmv, fcnmv_p, fcnmm, fcnmm_p,
    spfloat_fcnmv, spfloat_fcnmv_p, spfloat_fcnmm, spfloat_fcnmm_p,
)
from ._jit_normal import (
    JITCNormalR, JITCNormalC,
    binary_jitnmv, binary_jitnmv_p, binary_jitnmm, binary_jitnmm_p,
    jitn, jitn_p, jitnmv, jitnmv_p, jitnmm, jitnmm_p,
)
from ._jit_scalar import (
    JITScalarMatrix, JITCScalarR, JITCScalarC,
    binary_jitsmv, binary_jitsmv_p, binary_jitsmm, binary_jitsmm_p,
    jits, jits_p, jitsmv, jitsmv_p, jitsmm, jitsmm_p,
)
from ._jit_uniform import (
    JITCUniformR, JITCUniformC,
    binary_jitumv, binary_jitumv_p, binary_jitumm, binary_jitumm_p,
    jitu, jitu_p, jitumv, jitumv_p, jitumm, jitumm_p,
)
from ._jitc_matrix import JITCMatrix
from ._misc import (
    csr_to_coo_index, coo_to_csc_index, csr_to_csc_index,
)
from ._op import (
    XLACustomKernel, KernelEntry,
    BenchmarkResult, BenchmarkReport, benchmark_function,
    numba_kernel, numba_cuda_kernel, numba_cuda_callable,
    register_tvm_cuda_kernels, defjvp, general_batching_rule,
    jaxtype_to_warptype, jaxinfo_to_warpinfo,
)
from ._pallas_random import (
    PallasLFSR88RNG, PallasLFSR113RNG, PallasLFSR128RNG,
    PallasLFSRRNG, get_pallas_lfsr_rng_class,
)
from ._registry import (
    get_registry, get_primitives_by_tags, get_all_primitive_names,
)

__all__ = [
    # --- data representing events --- #
    'EventRepresentation',
    'IndexedEventRepresentation',
    'BinaryArray',
    'IndexedBinary1d',
    'IndexedBinary2d',
    'SparseFloat',
    'IndexedSpFloat1d',
    'IndexedSpFloat2d',
    'binary_array_index',

    # --- COO --- #
    'COO',
    'binary_coomv', 'binary_coomv_p',
    'binary_coomm', 'binary_coomm_p',
    'coomv', 'coomv_p',
    'coomm', 'coomm_p',
    'update_coo_on_binary_pre', 'update_coo_on_binary_post',
    'update_coo_on_binary_pre_p', 'update_coo_on_binary_post_p',

    # --- CSR --- #
    'CSR', 'CSC',
    'binary_csrmv', 'binary_csrmv_p',
    'binary_csrmm', 'binary_csrmm_p',
    'csrmv', 'csrmv_p',
    'csrmm', 'csrmm_p',
    'csrmv_yw2y', 'csrmv_yw2y_p',
    'update_csr_on_binary_pre', 'update_csr_on_binary_pre_p',
    'update_csr_on_binary_post', 'update_csr_on_binary_post_p',
    'spfloat_csrmv', 'spfloat_csrmv_p',
    'spfloat_csrmm', 'spfloat_csrmm_p',

    # --- dense matrix --- #
    'binary_densemv', 'binary_densemv_p',
    'binary_densemm', 'binary_densemm_p',
    'indexed_binary_densemv', 'indexed_binary_densemv_p',
    'indexed_binary_densemm', 'indexed_binary_densemm_p',
    'update_dense_on_binary_pre', 'update_dense_on_binary_pre_p',
    'update_dense_on_binary_post', 'update_dense_on_binary_post_p',
    'spfloat_densemv', 'spfloat_densemv_p',
    'spfloat_densemm', 'spfloat_densemm_p',

    # --- Just-In-Time Connectivity matrix --- #
    'JITCMatrix',
    'JITScalarMatrix', 'JITCScalarR', 'JITCScalarC',
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
    'FixedNumConn', 'FixedPreNumConn', 'FixedPostNumConn',
    'binary_fcnmv', 'binary_fcnmv_p',
    'binary_fcnmm', 'binary_fcnmm_p',
    'fcnmv', 'fcnmv_p',
    'fcnmm', 'fcnmm_p',
    'spfloat_fcnmv', 'spfloat_fcnmv_p',
    'spfloat_fcnmm', 'spfloat_fcnmm_p',

    # --- operator customization routines --- #
    'XLACustomKernel', 'KernelEntry',
    'BenchmarkResult', 'BenchmarkReport', 'benchmark_function',
    'numba_kernel', 'numba_cuda_kernel', 'numba_cuda_callable',
    'register_tvm_cuda_kernels', 'defjvp', 'general_batching_rule',
    'jaxtype_to_warptype', 'jaxinfo_to_warpinfo',

    # --- Pallas kernel --- #
    'PallasLFSR88RNG', 'PallasLFSR113RNG', 'PallasLFSR128RNG',
    'PallasLFSRRNG', 'get_pallas_lfsr_rng_class',

    # --- errors --- #
    'MathError',
    'KernelNotAvailableError',
    'KernelCompilationError',
    'KernelFallbackExhaustedError',
    'KernelExecutionError',

    # --- utilities --- #
    'csr_to_coo_index', 'coo_to_csc_index', 'csr_to_csc_index',

    # --- config & registry --- #
    'config', 'get_registry', 'get_primitives_by_tags', 'get_all_primitive_names',
]


def __getattr__(name):
    import warnings
    if name == 'EventArray':
        # warnings.warn(f'EventArray is deprecated, use {BinaryArray.__name__} instead')
        return BinaryArray
    if name == 'csr_on_pre':
        # warnings.warn(f'csr_on_pre is deprecated, use {update_csr_on_binary_pre.__name__} instead')
        return update_csr_on_binary_pre
    if name == 'csr2csc_on_post':
        # warnings.warn(f'csr2csc_on_post is deprecated, use {update_csr_on_binary_post.__name__} instead')
        return update_csr_on_binary_post
    if name == 'dense_on_pre':
        # warnings.warn(f'dense_on_pre is deprecated, use {update_dense_on_binary_pre.__name__} instead')
        return update_dense_on_binary_pre
    if name == 'dense_on_post':
        # warnings.warn(f'dense_on_post is deprecated, use {update_dense_on_binary_post.__name__} instead')
        return update_dense_on_binary_post
    raise AttributeError(name)
