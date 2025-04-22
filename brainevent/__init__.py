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
from ._coo import COO
from ._csr import CSR, CSC
from ._csrlb import CSR_LB, CSC_LB
from ._event import EventArray
from ._fixed_conn_num import FixedPostNumConn, FixedPreNumConn
from ._jitc import JITC_CSR, JITC_CSC
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_jvp import defjvp
from ._xla_custom_op_numba import NumbaKernelGenerator, set_numba_environ
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_warp import WarpKernelGenerator, dtype_to_warp_type

__all__ = [

    # --- data representing events --- #
    'EventArray',

    # --- data interoperable with events --- #
    'COO',
    'CSR',
    'CSC',
    'CSR_LB',
    'CSC_LB',
    'JITC_CSR',
    'JITC_CSC',
    'BlockCSR',
    'BlockELL',
    'FixedPreNumConn',
    'FixedPostNumConn',

    # --- operator customization routines --- #

    # 1. Custom kernel
    'XLACustomKernel',

    # 2. define JVP
    'defjvp',

    # 3. Numba kernel
    'NumbaKernelGenerator', 'set_numba_environ',

    # 4. Warp kernel
    'WarpKernelGenerator',
    'dtype_to_warp_type',

    # 5. Pallas kernel
    'PallasKernelGenerator',

]

