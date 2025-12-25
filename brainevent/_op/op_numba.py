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

# -*- coding: utf-8 -*-

import functools
import importlib.util
from typing import Callable, Dict, Optional, NamedTuple, Union

from jax.interpreters import mlir

from brainevent._compatible_import import Primitive
from brainevent._config import config
from brainevent._typing import KernelGenerator
from .numba_customcall import numba_cpu_custom_call_rule
from .numba_ffi import numba_cpu_ffi_rule

__all__ = [
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
    vmap_method: str = "sequential"


def numba_kernel(
    fn: Callable = None,
    input_output_aliases: Dict[int, int] = None,
    parallel: bool = False,
    **kwargs
) -> Union[NumbaKernel, Callable[[Callable], NumbaKernel]]:
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


def register_numba_cpu_translation(
    primitive: Primitive,
    cpu_kernel: KernelGenerator,
    debug: bool = False,
    version: str = 'custom_call',
):
    """
    Register the Numba CPU translation rule for the custom operator.

    Args:
        primitive: Primitive. The custom operator.
        cpu_kernel: Callable. The function defines the computation on CPU backend.
            It can be a function to generate the Numba jitted kernel.
        debug: bool. Whether to print the generated code.
        version: str. The lowering version, can be 'ffi' or 'custom_call'.
    """
    if version == 'ffi':
        rule = numba_cpu_ffi_rule(cpu_kernel)
    elif version == 'custom_call':
        rule = functools.partial(numba_cpu_custom_call_rule, cpu_kernel, debug)
    else:
        raise ValueError(f'Unsupported Numba CPU lowering version: {version}')
    mlir.register_lowering(primitive, rule, platform='cpu')
