# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
from typing import Callable, NamedTuple, Dict, Union, Sequence

import jax
from jax.interpreters import mlir

from ._compatible_import import Primitive, pallas as pl
from ._typing import KernelGenerator, Kernel

__all__ = [
    'pallas_kernel',
]


class PallasKernel(NamedTuple):
    kernel: Kernel


def pallas_kernel(
    fn: Callable = None,
    input_output_aliases: Dict[int, int] = None,
    tile: Sequence[int] = None,
    outs: Sequence[jax.ShapeDtypeStruct] = None,
) -> Union[PallasKernel, Callable[[Callable], PallasKernel]]:
    if fn is None:
        raise NotImplementedError

    assert isinstance(tile, (tuple, list)), 'grid must be a tuple or list of integers'
    assert outs is not None, 'outs must be specified'

    @functools.wraps(fn)
    def kernel(*args):
        fn_call = pl.pallas_call(
            fn,
            grid=tuple(tile),
            input_output_aliases=input_output_aliases,
            out_shape=outs,
        )
        return fn_call(*args)

    return PallasKernel(kernel=kernel)


def register_pallas_gpu_translation(
    primitive: Primitive,
    kernel_generator: KernelGenerator,
):
    """
    Registers a JAX Pallas translation rule for a given primitive on the GPU platform.

    This function sets up the mechanism for JAX to lower a custom high-level
    primitive (`primitive`) to a Pallas kernel specifically designed for GPU
    execution. It uses the provided `kernel_generator` to dynamically create
    the Pallas kernel based on the operation's parameters and then registers
    this kernel with JAX's MLIR lowering infrastructure for the 'cuda' platform.

    Args:
        primitive: The JAX `Primitive` object representing the custom operation
            for which the Pallas kernel translation is being registered.
        kernel_generator: A `PallasKernelGenerator` instance containing the logic
            to generate the Pallas kernel function and determine its block dimension.
            This generator encapsulates the GPU-specific computation details.

    Side Effects:
        Registers a lowering rule with JAX's MLIR system for the specified
        `primitive` on the 'cuda' platform. When JAX encounters this primitive
        during compilation for GPU, it will use the registered rule to generate
        the corresponding Pallas kernel code.
    """

    def kernel_fn(*args, **kwargs):
        """
        Inner function that generates and executes the Pallas kernel.

        This function is created dynamically and serves as the entry point
        for the Pallas kernel execution during the lowering process. It first
        determines the appropriate block dimension using the `kernel_generator`,
        then generates the actual Pallas kernel function, and finally calls
        the generated kernel with the input arguments.

        Args:
            *args: Positional arguments passed to the original primitive. These
                   will be forwarded to the generated Pallas kernel.
            **kwargs: Keyword arguments passed to the original primitive. These
                      are used by the `kernel_generator` to potentially determine
                      the block dimension and configure the kernel generation.

        Returns:
            The result(s) of executing the generated Pallas kernel.
        """
        # Generate the specific Pallas kernel function using the determined
        # block dimension and other relevant kwargs.
        kernel = kernel_generator(**kwargs)
        # Execute the generated Pallas kernel with the input arguments.
        return kernel(*args)

    # Lower the `kernel_fn` into MLIR. `lower_fun` converts the Python function
    # `kernel_fn` (which includes the Pallas kernel generation and invocation)
    # into an MLIR representation suitable for further compilation.
    # `multiple_results=True` indicates the kernel might return multiple outputs.
    lower = mlir.lower_fun(kernel_fn, multiple_results=True)

    # Register the lowered MLIR function (`lower`) as the translation rule for
    # the given `primitive` specifically when targeting the 'cuda' (GPU) platform.
    mlir.register_lowering(primitive, lower, platform='cuda')


def register_pallas_tpu_translation(
    primitive: Primitive,
    kernel_generator: KernelGenerator,
):
    """
    Registers a JAX Pallas translation rule for a given primitive on the TPU platform.

    This function sets up the mechanism for JAX to lower a custom high-level
    primitive (`primitive`) to a Pallas kernel specifically designed for TPU
    execution. It uses the provided `kernel_generator` to dynamically create
    the Pallas kernel based on the operation's parameters and then registers
    this kernel with JAX's MLIR lowering infrastructure for the 'tpu' platform.

    Args:
        primitive: The JAX `Primitive` object representing the custom operation
            for which the Pallas kernel translation is being registered.
        kernel_generator: A `PallasKernelGenerator` instance containing the logic
            to generate the Pallas kernel function and determine its block dimension.
            This generator encapsulates the TPU-specific computation details.

    Side Effects:
        Registers a lowering rule with JAX's MLIR system for the specified
        `primitive` on the 'tpu' platform. When JAX encounters this primitive
        during compilation for TPU, it will use the registered rule to generate
        the corresponding Pallas kernel code.
    """

    def kernel_fn(*args, **kwargs):
        """
        Inner function that generates and executes the Pallas kernel for TPU.

        This function is created dynamically and serves as the entry point
        for the Pallas kernel execution during the lowering process for TPU.
        It first determines the appropriate block dimension using the
        `kernel_generator`, then generates the actual Pallas kernel function,
        and finally calls the generated kernel with the input arguments.

        Args:
            *args: Positional arguments passed to the original primitive. These
                   will be forwarded to the generated Pallas kernel.
            **kwargs: Keyword arguments passed to the original primitive. These
                      are used by the `kernel_generator` to potentially determine
                      the block dimension and configure the kernel generation.

        Returns:
            The result(s) of executing the generated Pallas kernel.
        """
        # Generate the specific Pallas kernel function using the determined
        # block dimension and other relevant kwargs.
        kernel = kernel_generator(**kwargs)
        # Execute the generated Pallas kernel with the input arguments.
        return kernel(*args)

    # Lower the `kernel_fn` into MLIR. `lower_fun` converts the Python function
    # `kernel_fn` (which includes the Pallas kernel generation and invocation)
    # into an MLIR representation suitable for further compilation.
    # `multiple_results=True` indicates the kernel might return multiple outputs.
    lower = mlir.lower_fun(kernel_fn, multiple_results=True)

    # Register the lowered MLIR function (`lower`) as the translation rule for
    # the given `primitive` specifically when targeting the 'tpu' platform.
    mlir.register_lowering(primitive, lower, platform='tpu')
