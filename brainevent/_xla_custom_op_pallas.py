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


import dataclasses
from typing import Callable, Union, Dict

from jax.interpreters import mlir

from ._compatible_import import Primitive

__all__ = [
    'PallasKernelGenerator',
]


@dataclasses.dataclass(frozen=True)
class PallasKernelGenerator:
    """
    Represents a configuration for generating JAX Pallas kernels.

    This class encapsulates the necessary components to define and generate
    a JAX Pallas kernel for custom operations on GPU or TPU backends.
    It stores the kernel generation logic, block dimension specification,
    and optional input/output aliasing information.

    Attributes:
        generator: A callable that, when invoked with keyword arguments (like `block_dim`),
            returns the actual Pallas kernel function. This function defines the
            computation logic to be executed on the target hardware (GPU/TPU).
            See the `JAX Pallas documentation <https://docs.jax.dev/en/latest/pallas/quickstart.html>`_
            for details on writing Pallas kernels.
        block_dim: Specifies the block dimension for the Pallas kernel. It can be:
            - An integer: A fixed block dimension.
            - A callable: A function that takes keyword arguments (potentially derived
              from the operation's parameters or input shapes) and returns the
              calculated block dimension as an integer.
            - None: Indicates that the block dimension might be determined later or
              is not applicable.
        input_output_aliases: An optional dictionary or callable defining aliases
            between input and output buffers. This can enable optimizations by allowing
            in-place operations.
            - A dictionary mapping input buffer indices (int) to output buffer indices (int).
            - A callable that takes keyword arguments and returns such a dictionary.
            - None: No specific aliasing is defined.
    """
    __module__ = 'brainevent'
    generator: Callable[..., Callable]
    block_dim: Union[int, Callable[..., int]] = None
    input_output_aliases: Union[Dict[int, int], Callable[..., Dict[int, int]], None] = None

    def generate_kernel(self, **kwargs) -> Callable:
        """
        Generates the Pallas kernel function by invoking the stored generator.

        This method calls the `generator` callable provided during initialization,
        passing any provided keyword arguments (`kwargs`) to it. The `generator`
        is expected to use these arguments (e.g., `block_dim`) to configure and
        return the final Pallas kernel function.

        Args:
            **kwargs: Arbitrary keyword arguments that will be forwarded to the
                `self.generator` callable. This typically includes `block_dim`.

        Returns:
            The generated Pallas kernel function, ready to be compiled or used.
        """
        return self.generator(**kwargs)

    def get_block_dim(self, **kwargs) -> Union[int, None]:
        """
        Determines and returns the block dimension for the kernel.

        This method resolves the `block_dim` attribute based on its type:
        - If `block_dim` is a callable, it calls the function with `kwargs` and
          returns the result.
        - If `block_dim` is an integer, it returns the integer directly.
        - If `block_dim` is None, it returns None.

        Args:
            **kwargs: Arbitrary keyword arguments that are passed to the `block_dim`
                callable if `self.block_dim` is a function.

        Returns:
            The calculated block dimension as an integer, or None if `self.block_dim`
            is None.

        Raises:
            ValueError: If `self.block_dim` is neither callable, an integer, nor None.
        """
        if callable(self.block_dim):
            return self.block_dim(**kwargs)
        elif isinstance(self.block_dim, int):
            return self.block_dim
        elif self.block_dim is None:
            return None
        else:
            # Raise an error for unexpected types of block_dim
            raise ValueError(f"Invalid block_dim: {self.block_dim}")


def register_pallas_gpu_translation(
    primitive: Primitive,
    kernel_generator: PallasKernelGenerator,
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
        # Determine the block dimension, potentially based on kwargs.
        block_dim = kernel_generator.get_block_dim(**kwargs)
        # Generate the specific Pallas kernel function using the determined
        # block dimension and other relevant kwargs.
        kernel = kernel_generator.generate_kernel(block_dim=block_dim, **kwargs)
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
    kernel_generator: PallasKernelGenerator,
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
        # Determine the block dimension, potentially based on kwargs.
        block_dim = kernel_generator.get_block_dim(**kwargs)
        # Generate the specific Pallas kernel function using the determined
        # block dimension and other relevant kwargs.
        kernel = kernel_generator.generate_kernel(block_dim=block_dim, **kwargs)
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

