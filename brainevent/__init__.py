# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

__version__ = "0.0.1"

from ._block_csr import BlockCSR
from ._block_ell import BlockELL
from ._config import config
from ._coo import COO
from ._csr import CSR, CSC
from ._event import EventArray
from ._fixed_conn_num import FixedPostNumConn, FixedPreNumConn
from ._jitc_homo import JITCHomoR, JITCHomoC
from ._jitc_normal import JITCNormalR, JITCNormalC
from ._jitc_uniform import JITCUniformR, JITCUniformC
from ._pallas_random import LFSR88RNG, LFSR113RNG, LFSR128RNG
from ._xla_custom_op import XLACustomKernel, GPUKernelChoice
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_kernel
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_util import defjvp, general_batching_rule
from ._xla_custom_op_warp import WarpKernelGenerator, dtype_to_warp_type, warp_kernel

__all__ = [
    # --- global configuration --- #
    'config',

    # --- data representing events --- #
    'EventArray',

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
    'NumbaKernelGenerator',
    'numba_kernel',

    # 4. Warp kernel
    'WarpKernelGenerator',
    'dtype_to_warp_type',
    'warp_kernel',

    # 5. Pallas kernel
    'PallasKernelGenerator',
    'LFSR88RNG',
    'LFSR113RNG',
    'LFSR128RNG',

]
