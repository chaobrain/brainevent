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

from ._coo import (
    COO,
    plast_coo_on_binary_pre,
    plast_coo_on_binary_post,
)
from ._csr import (
    CSR,
    CSC,
    plast_csr_on_binary_pre,
    plast_csr2csc_on_binary_post,
    binary_csrmv_p,
)
from ._dense import (
    plast_dense_on_binary_pre,
    plast_dense_on_binary_post,
)
from ._error import (
    MathError,
    KernelNotAvailableError,
    KernelCompilationError,
    KernelFallbackExhaustedError,
)
from ._event import (
    BaseArray,
    BinaryArray,
    EventArray,
    IndexedBinary,
    SparseFloat,
    IndexedSparseFloat,
)
from ._fcn import (
    FixedPostNumConn,
    FixedPreNumConn,
)
from ._jitc_homo import (
    JITCHomoR,
    JITCHomoC,
)
from ._jitc_normal import (
    JITCNormalR,
    JITCNormalC,
)
from ._jitc_uniform import (
    JITCUniformR,
    JITCUniformC,
)
from ._misc import (
    csr_to_coo_index,
    coo_to_csc_index,
    csr_to_csc_index,
)
from ._op import (
    XLACustomKernel,
    numba_kernel,
    numba_cuda_kernel,
    defjvp,
    general_batching_rule,
    jaxtype_to_warptype,
    jaxinfo_to_warpinfo
)
from ._pallas_random import (
    LFSR88RNG,
    LFSR113RNG,
    LFSR128RNG,
)

__all__ = [
    # --- data representing events --- #
    'BaseArray',
    'EventArray',
    'BinaryArray',
    'IndexedBinary',
    'SparseFloat',
    'IndexedSparseFloat',

    # --- COO --- #
    'COO',
    'plast_coo_on_binary_pre',
    'plast_coo_on_binary_post',

    # CSR
    'CSR',
    'CSC',
    'plast_csr_on_binary_pre',
    'plast_csr2csc_on_binary_post',

    # Just-In-Time Connectivity matrix
    'JITCHomoR',  # row-oriented JITC matrix with homogeneous weight
    'JITCHomoC',  # column-oriented JITC matrix with homogeneous weight
    'JITCNormalR',  # row-oriented JITC matrix with normal weight
    'JITCNormalC',  # column-oriented JITC matrix with normal weight
    'JITCUniformR',  # row-oriented JITC matrix with uniform weight
    'JITCUniformC',  # column-oriented JITC matrix with uniform weight

    # --- Fixed number connectivity --- #
    'FixedPreNumConn',
    'FixedPostNumConn',

    # --- dense matrix ----- #
    'plast_dense_on_binary_pre',
    'plast_dense_on_binary_post',

    # --- operator customization routines --- #

    # 1. Custom kernel
    'XLACustomKernel',

    # 2. utilities
    'defjvp',
    'general_batching_rule',

    # 3. Numba kernel
    'numba_kernel',
    'numba_cuda_kernel',

    # 4. Warp kernel
    'jaxtype_to_warptype',
    'jaxinfo_to_warpinfo',

    # 5. Pallas kernel
    'LFSR88RNG',
    'LFSR113RNG',
    'LFSR128RNG',

    # --- others --- #

    'MathError',
    'csr_to_coo_index',
    'coo_to_csc_index',
    'csr_to_csc_index',

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
