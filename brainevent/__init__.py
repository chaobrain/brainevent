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

from ._config import (
    load_user_defaults, save_user_defaults, get_user_default,
    set_user_default, clear_user_defaults, get_config_path,
    invalidate_cache,
)
from ._coo import (
    COO,
    binary_coomv, binary_coomv_p, binary_coomm, binary_coomm_p,
    coomv, coomv_p, coomm, coomm_p,
    plast_coo_on_binary_pre, plast_coo_on_binary_post,
    plast_coo_on_binary_pre_p, plast_coo_on_binary_post_p,
)
from ._csr import (
    CSR, CSC,
    binary_csrmv, binary_csrmv_p, binary_csrmm, binary_csrmm_p,
    csrmv, csrmv_p, csrmm, csrmm_p, csrmv_yw2y, csrmv_yw2y_p,
    plast_csr_on_binary_pre, plast_csr_on_binary_pre_p,
    plast_csr2csc_on_binary_post, plast_csr2csc_on_binary_post_p,
    spfloat_csrmv, spfloat_csrmv_p, spfloat_csrmm, spfloat_csrmm_p,
    csr_solve,
)
from ._dense import (
    dbmv, dbmv_p, bdvm, bdvm_p,
    dbmm, dbmm_p, bdmm, bdmm_p,
    indexed_bdvm, indexed_bdvm_p, indexed_dbmv, indexed_dbmm, indexed_bdmm, indexed_bdmm_p,
    plast_dense_on_binary_pre, plast_dense_on_binary_pre_p,
    plast_dense_on_binary_post, plast_dense_on_binary_post_p,
    dsfmv, dsfmv_p, sfdvm, sfdvm_p,
    dsfmm, dsfmm_p, sfdmm, sfdmm_p,
)
from ._error import (
    MathError,
    KernelNotAvailableError,
    KernelCompilationError,
    KernelFallbackExhaustedError,
    KernelExecutionError,
)
from ._event import (
    EventRepresentation, BinaryArray,
    SparseFloat,
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
    register_cuda_kernels, defjvp, general_batching_rule,
    jaxtype_to_warptype, jaxinfo_to_warpinfo,
)
from ._pallas_random import (
    PallasLFSR88RNG, PallasLFSR113RNG, PallasLFSR128RNG,
)
from ._registry import (
    get_registry, get_primitives_by_tags, get_all_primitive_names,
)

__all__ = [
    # --- data representing events --- #
    'EventRepresentation',
    'BinaryArray',
    'SparseFloat',
    'binary_array_index',

    # --- COO --- #
    'COO',
    'binary_coomv', 'binary_coomv_p', 'binary_coomm', 'binary_coomm_p',
    'coomv', 'coomv_p', 'coomm', 'coomm_p',
    'plast_coo_on_binary_pre', 'plast_coo_on_binary_post',
    'plast_coo_on_binary_pre_p', 'plast_coo_on_binary_post_p',

    # --- CSR --- #
    'CSR', 'CSC',
    'binary_csrmv', 'binary_csrmv_p', 'binary_csrmm', 'binary_csrmm_p',
    'csrmv', 'csrmv_p', 'csrmm', 'csrmm_p', 'csrmv_yw2y', 'csrmv_yw2y_p',
    'plast_csr_on_binary_pre', 'plast_csr_on_binary_pre_p',
    'plast_csr2csc_on_binary_post', 'plast_csr2csc_on_binary_post_p',
    'spfloat_csrmv', 'spfloat_csrmv_p', 'spfloat_csrmm', 'spfloat_csrmm_p',
    'csr_solve',

    # --- dense matrix --- #
    'dbmv', 'dbmv_p', 'bdvm', 'bdvm_p',
    'dbmm', 'dbmm_p', 'bdmm', 'bdmm_p',
    'indexed_bdvm', 'indexed_bdvm_p', 'indexed_dbmv', 'indexed_dbmm', 'indexed_bdmm', 'indexed_bdmm_p',
    'plast_dense_on_binary_pre', 'plast_dense_on_binary_pre_p',
    'plast_dense_on_binary_post', 'plast_dense_on_binary_post_p',
    'dsfmv', 'dsfmv_p', 'sfdvm', 'sfdvm_p',
    'dsfmm', 'dsfmm_p', 'sfdmm', 'sfdmm_p',

    # --- Just-In-Time Connectivity matrix --- #
    'JITCMatrix',
    'JITScalarMatrix', 'JITCScalarR', 'JITCScalarC',
    'binary_jitsmv', 'binary_jitsmv_p', 'binary_jitsmm', 'binary_jitsmm_p',
    'jits', 'jits_p', 'jitsmv', 'jitsmv_p', 'jitsmm', 'jitsmm_p',
    'JITCNormalR', 'JITCNormalC',
    'binary_jitnmv', 'binary_jitnmv_p', 'binary_jitnmm', 'binary_jitnmm_p',
    'jitn', 'jitn_p', 'jitnmv', 'jitnmv_p', 'jitnmm', 'jitnmm_p',
    'JITCUniformR', 'JITCUniformC',
    'binary_jitumv', 'binary_jitumv_p', 'binary_jitumm', 'binary_jitumm_p',
    'jitu', 'jitu_p', 'jitumv', 'jitumv_p', 'jitumm', 'jitumm_p',

    # --- Fixed number connectivity --- #
    'FixedNumConn', 'FixedPreNumConn', 'FixedPostNumConn',
    'binary_fcnmv', 'binary_fcnmv_p', 'binary_fcnmm', 'binary_fcnmm_p',
    'fcnmv', 'fcnmv_p', 'fcnmm', 'fcnmm_p',
    'spfloat_fcnmv', 'spfloat_fcnmv_p', 'spfloat_fcnmm', 'spfloat_fcnmm_p',

    # --- operator customization routines --- #
    'XLACustomKernel', 'KernelEntry',
    'BenchmarkResult', 'BenchmarkReport', 'benchmark_function',
    'numba_kernel', 'numba_cuda_kernel', 'numba_cuda_callable',
    'register_cuda_kernels', 'defjvp', 'general_batching_rule',
    'jaxtype_to_warptype', 'jaxinfo_to_warpinfo',

    # --- Pallas kernel --- #
    'PallasLFSR88RNG', 'PallasLFSR113RNG', 'PallasLFSR128RNG',

    # --- errors --- #
    'MathError',
    'KernelNotAvailableError',
    'KernelCompilationError',
    'KernelFallbackExhaustedError',
    'KernelExecutionError',

    # --- utilities --- #
    'csr_to_coo_index', 'coo_to_csc_index', 'csr_to_csc_index',

    # --- config & registry --- #
    'load_user_defaults', 'save_user_defaults', 'get_user_default',
    'set_user_default', 'clear_user_defaults', 'get_config_path',
    'invalidate_cache',
    'get_registry', 'get_primitives_by_tags', 'get_all_primitive_names',
]


def __getattr__(name):
    import warnings
    if name == 'csr_on_pre':
        warnings.warn(
            f'csr_on_pre is deprecated, use {plast_csr_on_binary_pre.__name__} instead',
        )
        return plast_csr_on_binary_pre
    if name == 'csr2csc_on_post':
        warnings.warn(
            f'csr2csc_on_post is deprecated, use {plast_csr2csc_on_binary_post.__name__} instead',
        )
        return plast_csr2csc_on_binary_post
    if name == 'dense_on_pre':
        return plast_dense_on_binary_pre
    if name == 'dense_on_post':
        return plast_dense_on_binary_post
    raise AttributeError(name)
