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

Kernel = Callable

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
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = event_coomv_p_call(
        data, row, col, v,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))

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
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = event_coomm_p_call(
        data, row, col, B,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))

def event_coomv_cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    ...

def event_coomv_gpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    ...

def event_coomv_jvp_v(
    v_dot,
    data,
    row,
    col,
    v,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    ...

def event_coomv_jvp_weights(
    data_dot,
    data,
    row,
    col,
    v,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    ...

def event_coomv_transpose_rule(
    ct,
    data,
    row,
    col,
    events,
    _,
    *,
    shape,
    float_as_event,
    transpose,
    **kwargs
):
    ...

def event_coomv_batching(
    args,
    axes,
    **kwargs
):
    ...

def event_coomv_p_call(
    weights,
    row,
    col,
    v,
    *,
    shape: Sequence[int],
    transpose: bool,
    float_as_event: bool,
):
    ...

event_coomv_p = XLACustomKernel(
    'event_coomv',
    cpu_kernel=NumbaKernelGenerator(event_coomv_cpu_kernel_generator, input_output_aliases={4:0}),
    gpu_kernel=WarpKernelGenerator(
        event_coomv_gpu_kernel_generator,
        # TODO: check if dim param is correct
        dim=lambda row_info, vector_info, transpose, **kwargs: (
            vector_info.shape[0] if transpose else row_info.shape[0] - 1
        ),
        input_output_aliases={4:0}
    ),
)
event_coomv_p.defjvp(event_coomv_jvp_weights, None, None, event_coomv_jvp_v)
event_coomv_p.def_transpose_rule(event_coomv_transpose_rule)
event_coomv_p.def_batching_rule(event_coomv_batching)

def event_coomm_cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    ...


def event_coomm_gpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    ...

def event_coomm_jvp_left(
    data_dot,
    data,
    row,
    col,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    ...

def event_coomm_jvp_right(
    B_dot,
    data,
    row,
    col,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    ...

def event_coomm_transpose_rule(
    ct,
    data,
    row,
    col,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    ...

def event_coomm_batching(
    args,
    axes,
    **kwargs
):
    ...

def event_coomm_p_call(
    weights,
    row,
    col,
    B,
    *,
    shape: Sequence[int],
    transpose: bool,
    float_as_event: bool,
):
    ...

event_coomm_p = XLACustomKernel(
    'event_coomm',
    cpu_kernel=NumbaKernelGenerator(
        event_coomm_cpu_kernel_generator,
        input_output_aliases={4: 0}
    ),
    gpu_kernel=WarpKernelGenerator(
        event_coomm_gpu_kernel_generator,
        # TODO: check if dim param is correct
        dim=lambda matrix_info, row_info, transpose, **kwargs: (
            tuple(reversed(matrix_info.shape))
            if transpose else
            [matrix_info.shape[1], row_info.shape[0] - 1]
        ),
        input_output_aliases={4: 0}
    )
)
event_coomm_p.defjvp(event_coomm_jvp_left, None, None, event_coomm_jvp_right)
event_coomm_p.def_transpose_rule(event_coomm_transpose_rule)
event_coomm_p.def_batching_rule(event_coomm_batching)