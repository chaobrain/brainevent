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

# -*- coding: utf-8 -*-


from typing import Callable, Union, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainunit.sparse._csr import _csr_to_coo
from jax.interpreters import ad

from ._coo_float_impl import _coo_matvec, _coo_matmat
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator


def _event_coo_matvec(
    data: Union[jax.Array, u.Quantity],
    row: jax.Array,
    col: jax.Array,
    v: jax.Array,
    *,
    shape: Sequence[int],
    transpose: bool = False,
    float_as_event: bool = True
) -> Union[jax.Array, u.Quantity]:
    ...

def _event_coo_matmat(
data: Union[jax.Array, u.Quantity],
    row: jax.Array,
    col: jax.Array,
    B: jax.Array,
    *,
    shape: Sequence[int],
    transpose: bool = False,
    float_as_event: bool = True
) -> Union[jax.Array, u.Quantity]:
    ...