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

import ctypes
import dataclasses
import functools
import importlib.util
from typing import Callable, Dict, Optional, NamedTuple, Union

from jax.interpreters import mlir

from ._compatible_import import register_custom_call, Primitive, custom_call
from ._config import config

__all__ = [
    'NumbaKernelGenerator',
    'numba_kernel',
]

numba_installed = importlib.util.find_spec('numba') is not None


def numba_jit_fn(fn: Callable):
    """
    Apply standard Numba JIT compilation to a function.

    Parameters
    ----------
    fn : Callable
        The function to be JIT compiled.

    Returns
    -------
    Callable
        The compiled function with applied JIT optimizations.
    """
    import numba
    setting = config.get_numba_setting()
    setting.pop('parallel', False)
    return numba.njit(fn, **setting)


def numba_pjit_fn(fn: Callable):
    """
    Apply parallel Numba JIT compilation to a function.

    This uses the current parallel setting to determine whether
    to enable parallel execution.

    Parameters
    ----------
    fn : Callable
        The function to be JIT compiled with parallel support.

    Returns
    -------
    Callable
        The compiled function with applied JIT optimizations and
        parallel execution if enabled.
    """
    import numba
    setting = config.get_numba_setting()
    setting.pop('parallel', False)
    return numba.njit(fn, **setting)


class NumbaKernel(NamedTuple):
    """
    A named tuple representing a compiled Numba kernel with optional input-output aliasing information.

    Attributes:
        kernel: Callable
            The compiled Numba function that performs the actual computation.
        input_output_aliases: Optional[Dict[int, int]]
            A dictionary mapping output indices to input indices, indicating which
            output buffers can reuse the same memory as input buffers.
            This enables in-place operations to avoid unnecessary memory allocations.
            The keys are output indices and the values are the corresponding input indices.
            If None, no aliasing is performed.
    """
    kernel: Callable
    input_output_aliases: Optional[Dict[int, int]]


def numba_kernel(
    fn: Callable = None,
    input_output_aliases: Dict[int, int] = None,
    parallel: bool = False,
    **kwargs
) -> Union[NumbaKernel, Callable[..., NumbaKernel]]:
    """
    Creates a NumbaKernel by compiling the provided function with Numba.

    This function can be used as a decorator or called directly to compile a Python
    function into an optimized NumbaKernel. It supports specifying input-output aliases
    for in-place operations and parallel execution.

    Parameters
    ----------
    fn : Callable, optional
        The function to be compiled with Numba. If None, returns a partial function
        that can be used as a decorator.
    input_output_aliases : Dict[int, int], optional
        A dictionary mapping output indices to input indices, indicating which
        output buffers can reuse the same memory as input buffers. Enables in-place
        operations to avoid unnecessary memory allocations.
    parallel : bool, default=False
        Whether to enable parallel execution of the Numba kernel. If True, the function
        is compiled with parallel optimizations using `numba_environ.pjit_fn`.
    **kwargs
        Additional keyword arguments to pass to the Numba compiler.

    Returns
    -------
    Union[NumbaKernel, Callable[..., NumbaKernel]]
        If `fn` is provided, returns a NumbaKernel instance containing the compiled function.
        If `fn` is None, returns a partial function that can be used as a decorator.

    Raises
    ------
    ImportError
        If Numba is not installed but is required to compile the kernel.

    Examples
    --------
    # Direct function call
    >>> kernel = numba_kernel(my_function)

    # As a decorator
    >>> @numba_kernel(parallel=True)
    ... def my_function(x, y, out):
    ...     # function implementation
    ...     pass
    """
    if fn is None:
        return functools.partial(
            numba_kernel,
            input_output_aliases=input_output_aliases,
            parallel=parallel,
            **kwargs
        )
    else:
        if not numba_installed:
            raise ImportError('Numba is required to compile the CPU kernel for the custom operator.')

        if parallel:
            return NumbaKernel(
                kernel=numba_pjit_fn(fn),
                input_output_aliases=input_output_aliases,
            )
        else:
            return NumbaKernel(
                kernel=numba_jit_fn(fn),
                input_output_aliases=input_output_aliases,
            )


@dataclasses.dataclass(frozen=True)
class NumbaKernelGenerator:
    """
    A dataclass representing a generator for Numba kernels to be used in custom JAX operations.

    This class provides a wrapper around functions that generate NumbaKernel instances
    for executing optimized code on CPU. It's designed to be used with the custom operation
    system to provide Numba-accelerated implementations of operations.

    Attributes
    ----------
    generator : Callable[..., NumbaKernel]
        A function that generates a NumbaKernel instance when called with configuration parameters.
        This function defines the computation to be performed on the CPU backend.

    Examples
    --------
    >>> def kernel_gen(**kwargs) -> NumbaKernel:
    ...     # Define and return a NumbaKernel
    ...     return numba_kernel(my_function, **kwargs)
    ...
    >>> generator = NumbaKernelGenerator(generator=kernel_gen)
    >>> kernel = generator.generate_kernel(parallel=True)
    """
    __module__ = 'brainevent'

    generator: Callable[..., NumbaKernel]

    def generate_kernel(self, **kwargs):
        """
        Generate a NumbaKernel by calling the generator function.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the generator function to configure
            the kernel generation process.

        Returns
        -------
        NumbaKernel
            A compiled Numba kernel ready for execution.
        """
        return self.generator(**kwargs)

    def __call__(self, *args, **kwargs):
        """
        Make the NumbaKernelGenerator instance callable.

        This method allows using the generator instance directly as a function.

        Parameters
        ----------
        *args
            Positional arguments (ignored, maintained for compatibility).
        **kwargs
            Keyword arguments passed to the generator function.

        Returns
        -------
        NumbaKernel
            A compiled Numba kernel ready for execution.
        """
        return self.generator(**kwargs)


def _shape_to_layout(shape):
    return tuple(range(len(shape) - 1, -1, -1))


def _numba_mlir_cpu_translation_rule(
    kernel_generator: NumbaKernelGenerator,
    debug: bool,
    ctx,
    *ins,
    **kwargs
):
    if not numba_installed:
        raise ImportError('Numba is required to compile the CPU kernel for the custom operator.')

    from numba import types, carray, cfunc  # pylint: disable=import-error

    kernel = kernel_generator.generate_kernel(**kwargs)
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


def register_numba_cpu_translation(
    primitive: Primitive,
    cpu_kernel: NumbaKernelGenerator,
    debug: bool = False
):
    """
    Register the Numba CPU translation rule for the custom operator.

    Args:
        primitive: Primitive. The custom operator.
        cpu_kernel: Callable. The function defines the computation on CPU backend.
            It can be a function to generate the Numba jitted kernel.
        debug: bool. Whether to print the generated code.
    """
    rule = functools.partial(_numba_mlir_cpu_translation_rule, cpu_kernel, debug)
    mlir.register_lowering(primitive, rule, platform='cpu')
