# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from brainevent._compatible_import import Primitive, pallas as pl
from brainevent._typing import KernelGenerator, Kernel

__all__ = [
    'pallas_kernel',
]


class PallasKernel(NamedTuple):
    """
    A named tuple that encapsulates a Pallas kernel and its configuration parameters.

    PallasKernel serves as a container for a compiled Pallas kernel function along with
    the configuration that was used to create it. This allows for introspection of the
    kernel's properties after creation and facilitates kernel reuse.

    Attributes:
        kernel: Callable
            The compiled Pallas kernel function that can be called with input tensors
            to perform accelerator-specific computations. This is the actual executable
            that will run on the hardware accelerator (GPU/TPU).

        input_output_aliases: Dict[int, int], optional
            A dictionary mapping input buffer indices to output buffer indices,
            indicating where memory can be reused between inputs and outputs.
            This optimization allows the kernel to avoid unnecessary memory allocation.
            Default is None.

        tile: Sequence[int], optional
            The execution grid dimensions used for this kernel, defining how many
            parallel instances of the kernel will be launched. For example, [128, 128]
            would create a 2D grid of 128×128 kernel instances.
            Default is None.

        outs: Sequence[jax.ShapeDtypeStruct], optional
            Shape and dtype specifications for each output tensor produced by the kernel.
            These structures define the expected outputs and are used by JAX for type
            checking and shape inference.
            Default is None.
    """
    kernel: Kernel
    input_output_aliases: Dict[int, int] = None
    tile: Sequence[int] = None
    outs: Sequence[jax.ShapeDtypeStruct] = None

    def __call__(self, *args):
        """
        Makes the PallasKernel instance callable like a regular function.

        This method allows a PallasKernel instance to be called directly with arguments,
        delegating to the compiled kernel function stored in the 'kernel' attribute.
        It enables users to treat a PallasKernel object as if it were the function itself.

        Args:
            *args: Variable length argument list to be passed directly to the
                   underlying compiled kernel function.

        Returns:
            The result(s) of executing the compiled Pallas kernel with the given arguments.

        Example:
            >>> kernel = pallas_kernel(my_function, tile=[128], outs=[...])
            >>> result = kernel(input_data)  # calls __call__ implicitly
        """
        return self.kernel(*args)


def pallas_kernel(
    fn: Callable = None,
    input_output_aliases: Dict[int, int] = None,
    tile: Sequence[int] = (),
    outs: Sequence[jax.ShapeDtypeStruct] = None,
) -> Union[PallasKernel, Callable[[Callable], PallasKernel]]:
    """
    Wraps a function to create a Pallas kernel for accelerator execution.

    This decorator transforms a Python function into a Pallas kernel that can be executed
    on accelerator hardware (GPU/TPU). It handles the configuration of the execution grid,
    memory aliasing, and output specifications required for Pallas kernels.

    The function can be used either as a direct decorator or with parameters:

    ```python
    # Direct decorator usage
    @pallas_kernel(tile=[128, 128], outs=[jax.ShapeDtypeStruct(...)])
    def my_kernel(state):
        # kernel implementation
        pass

    # Function call usage
    kernel = pallas_kernel(my_kernel_fn, tile=[128, 128], outs=[...])
    ```

    Args:
        fn: The function to be wrapped as a Pallas kernel. When used as a decorator
            with arguments, this will be None.
        input_output_aliases: Dictionary mapping input indices to output indices for
            buffer reuse. Indicates that the input at key should be reused as the
            output at value.
        tile: Sequence of integers specifying the execution grid dimensions.
            Defines how many kernel instances will be launched in parallel.
        outs: Sequence of jax.ShapeDtypeStruct objects describing the shape and dtype
            of each output tensor.

    Returns:
        If fn is provided, returns a PallasKernel instance wrapping the function.
        If fn is None (when used as a decorator with arguments), returns a decorator
        function that will wrap the decorated function.

    Raises:
        AssertionError: If tile is not a tuple or list, or if outs is not specified.
    """
    if fn is None:
        return lambda f: pallas_kernel(
            f,
            input_output_aliases=input_output_aliases,
            tile=tile,
            outs=outs,
        )

    assert isinstance(tile, (tuple, list)), 'grid must be a tuple or list of integers'
    assert outs is not None, 'outs must be specified'
    if input_output_aliases is None:
        input_output_aliases = {}

    @functools.wraps(fn)
    def kernel(*args):
        return pl.pallas_call(fn, grid=tuple(tile), input_output_aliases=input_output_aliases, out_shape=outs)(*args)

    return PallasKernel(
        kernel=kernel,
        input_output_aliases=input_output_aliases,
        tile=tile,
        outs=outs,
    )


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
        kernel_generator: A `KernelGenerator` instance containing the logic
            to generate the Pallas kernel function based on operation parameters.
            This generator encapsulates the GPU-specific computation details.

    Side Effects:
        Registers a lowering rule with JAX's MLIR system for the specified
        `primitive` on the 'cuda' platform. When JAX encounters this primitive
        during compilation for GPU, it will use the registered rule to generate
        the corresponding Pallas kernel code.

    Example:
        >>> primitive = create_primitive("custom_op")
        >>> kernel_gen = MyPallasKernelGenerator(...)
        >>> register_pallas_gpu_translation(primitive, kernel_gen)
    """

    def kernel_fn(*args, **kwargs):
        """
        Inner function that generates and executes the Pallas kernel.

        This function is created dynamically and serves as the entry point
        for the Pallas kernel execution during the lowering process. It first
        generates the actual Pallas kernel function using the kernel_generator,
        and then calls the generated kernel with the input arguments.

        Args:
            *args: Positional arguments passed to the original primitive. These
                   will be forwarded to the generated Pallas kernel.
            **kwargs: Keyword arguments passed to the original primitive. These
                      are used by the `kernel_generator` to configure the kernel
                      generation.

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
        kernel_generator: A `KernelGenerator` instance containing the logic
            to generate the Pallas kernel function based on operation parameters.
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
        It first generates the actual Pallas kernel function using the
        kernel_generator, and then calls the generated kernel with the input arguments.

        Args:
            *args: Positional arguments passed to the original primitive. These
                   will be forwarded to the generated Pallas kernel.
            **kwargs: Keyword arguments passed to the original primitive. These
                      are used by the `kernel_generator` to configure the kernel
                      generation.

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
