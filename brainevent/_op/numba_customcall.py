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

import ctypes
import importlib.util

from jax.interpreters import mlir

from brainevent._compatible_import import register_custom_call, custom_call
from brainevent._typing import KernelGenerator

numba_installed = importlib.util.find_spec('numba') is not None


def _shape_to_layout(shape):
    return tuple(range(len(shape) - 1, -1, -1))


def numba_cpu_custom_call_rule(
    kernel_generator: KernelGenerator,
    debug: bool,
    ctx,
    *ins,
    **kwargs
):
    from .op_numba import NumbaKernel
    if not numba_installed:
        raise ImportError('Numba is required to compile the CPU kernel for the custom operator.')

    from numba import types, carray, cfunc  # pylint: disable=import-error

    kernel = kernel_generator(**kwargs)
    assert isinstance(kernel, NumbaKernel), f'The kernel should be of type NumbaKernel, but got {type(kernel)}'

    # output information
    outs = ctx.avals_out
    output_shapes = tuple([out.shape for out in outs])
    output_dtypes = tuple([out.dtype for out in outs])
    output_layouts = tuple([_shape_to_layout(out.shape) for out in outs])
    result_types = [mlir.aval_to_ir_type(out) for out in outs]

    # input information
    avals_in = ctx.avals_in
    input_layouts = [_shape_to_layout(a.shape) for a in avals_in]
    input_dtypes = tuple(inp.dtype for inp in avals_in)
    input_shapes = tuple(inp.shape for inp in avals_in)

    # compiling function
    code_scope = dict(
        func_to_call=kernel.kernel,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        carray=carray
    )
    args_in = [f'in{i} = carray(input_ptrs[{i}], input_shapes[{i}], dtype=input_dtypes[{i}])'
               for i in range(len(input_shapes))]
    if len(output_shapes) > 1:
        args_out = [f'out{i} = carray(output_ptrs[{i}], output_shapes[{i}], dtype=output_dtypes[{i}])'
                    for i in range(len(output_shapes))]
        sig = types.void(types.CPointer(types.voidptr), types.CPointer(types.voidptr))
    else:
        args_out = [f'out0 = carray(output_ptrs, output_shapes[0], dtype=output_dtypes[0])']
        sig = types.void(types.voidptr, types.CPointer(types.voidptr))
    args_call = [f'in{i}' for i in range(len(input_shapes))] + [f'out{i}' for i in range(len(output_shapes))]
    code_string = '''
def numba_cpu_custom_call_target(output_ptrs, input_ptrs):
    {args_in}
    {args_out}
    func_to_call({args_call})
      '''.format(args_in="\n    ".join(args_in),
                 args_out="\n    ".join(args_out),
                 args_call=", ".join(args_call))
    if debug:
        print(code_string)
    exec(compile(code_string.strip(), '', 'exec'), code_scope)
    new_f = code_scope['numba_cpu_custom_call_target']

    # register
    xla_c_rule = cfunc(sig)(new_f)
    target_name = f'brainevent_numba_call_{str(xla_c_rule.address)}'

    PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    #                                         [void* pointer,
    #                                          const char *name,
    #                                          PyCapsule_Destructor destructor]
    PyCapsule_New.argtypes = (
        ctypes.c_void_p,
        ctypes.c_char_p,
        PyCapsule_Destructor
    )
    PyCapsule_New.restype = ctypes.py_object
    capsule = PyCapsule_New(
        xla_c_rule.address,
        b"xla._CUSTOM_CALL_TARGET",
        PyCapsule_Destructor(0)
    )

    register_custom_call(target_name, capsule, "cpu")

    # call
    return custom_call(
        call_target_name=target_name,
        operands=ins,
        operand_layouts=list(input_layouts),
        result_layouts=list(output_layouts),
        result_types=list(result_types),
        has_side_effect=False,
        operand_output_aliases=kernel.input_output_aliases,
    ).results
