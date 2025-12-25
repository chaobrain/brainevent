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


__version__ = "0.0.5"
__version_info__ = tuple(map(int, __version__.split(".")))

from ._block_csr import BlockCSR
from ._block_ell import BlockELL
from ._config import config
from ._coo import COO, coo_on_pre, coo_on_post
from ._csr import CSR, CSC, csr_on_pre, csr2csc_on_post
from ._dense import dense_on_pre, dense_on_post
from ._error import MathError
from ._event import BaseArray, BinaryArray, EventArray, BinaryArrayIndex, MaskedFloat, MaskedFloatIndex
from ._fixed_conn_num import FixedPostNumConn, FixedPreNumConn
from ._jitc_homo import JITCHomoR, JITCHomoC
from ._jitc_normal import JITCNormalR, JITCNormalC
from ._jitc_uniform import JITCUniformR, JITCUniformC
from ._misc import csr_to_coo_index, coo_to_csc_index, csr_to_csc_index
from ._pallas_random import LFSR88RNG, LFSR113RNG, LFSR128RNG
from ._primitives import (
    ALL_PRIMITIVES,
    get_all_primitive_names,
    get_primitives_by_category,
    get_primitive_info
)
from ._op import (
    XLACustomKernel,
    GPUKernelChoice,
    numba_kernel,
    pallas_kernel,
    defjvp,
    general_batching_rule,
    warp_kernel,
    jaxtype_to_warptype,
    jaxinfo_to_warpinfo
)

__all__ = [
    # --- global configuration --- #
    'config',

    # --- data representing events --- #
    'BaseArray',
    'EventArray',
    'BinaryArray',
    'BinaryArrayIndex',
    'MaskedFloat',
    'MaskedFloatIndex',

    # --- data interoperable with events --- #
    'COO',
    'CSR',
    'CSC',

    # Just-In-Time Connectivity matrix
    'JITCHomoR',  # row-oriented JITC matrix with homogeneous weight
    'JITCHomoC',  # column-oriented JITC matrix with homogeneous weight
    'JITCNormalR',  # row-oriented JITC matrix with normal weight
    'JITCNormalC',  # column-oriented JITC matrix with normal weight
    'JITCUniformR',  # row-oriented JITC matrix with uniform weight
    'JITCUniformC',  # column-oriented JITC matrix with uniform weight

    # --- block data --- #
    'BlockCSR',
    'BlockELL',
    'FixedPreNumConn',
    'FixedPostNumConn',

    # --- operator customization routines --- #

    # 1. Custom kernel
    'XLACustomKernel',
    'GPUKernelChoice',

    # 2. utilities
    'defjvp',
    'general_batching_rule',

    # 3. Numba kernel
    'numba_kernel',

    # 4. Warp kernel
    'warp_kernel',
    'jaxtype_to_warptype',
    'jaxinfo_to_warpinfo',

    # 5. Pallas kernel
    'pallas_kernel',
    'LFSR88RNG',
    'LFSR113RNG',
    'LFSR128RNG',

    # --- others --- #

    'MathError',
    'csr_to_coo_index',
    'coo_to_csc_index',
    'csr_to_csc_index',
    'csr_on_pre',
    'csr2csc_on_post',
    'coo_on_pre',
    'coo_on_post',
    'dense_on_pre',
    'dense_on_post',

    # --- primitives --- #
    'ALL_PRIMITIVES',
    'get_all_primitive_names',
    'get_primitives_by_category',
    'get_primitive_info',

]
