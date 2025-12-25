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
from typing import Callable, Union, Optional, Dict

from jax.interpreters import xla, mlir, batching, ad

from brainevent._compatible_import import Primitive
from brainevent._config import config
from brainevent._typing import KernelGenerator
from .op_numba import register_numba_cpu_translation
from .op_pallas import (
    register_pallas_gpu_translation,
    register_pallas_tpu_translation,
)
from .op_warp import register_warp_gpu_translation
from .util import general_batching_rule, defjvp, OutType, abstract_arguments

__all__ = [
    'XLACustomKernel',
    'GPUKernelChoice',
]


class GPUKernelChoice:
    """A class to dynamically select between different GPU kernel implementations.

    This class provides a mechanism to choose between Warp and Pallas kernel
    implementations for GPU execution. It allows specifying a default kernel type
    and dynamically selecting the appropriate kernel at runtime based on
    configuration settings.

    Attributes:
        default (str): The default kernel backend to use ('warp' or 'pallas').
        warp_kernel (Optional[KernelGenerator]): The Warp kernel implementation.
        pallas_kernel (Optional[KernelGenerator]): The Pallas kernel implementation.
        _all_kernels (dict): Dictionary mapping backend names to kernel implementations.
    """

    def __init__(
        self,
        default: str,
        warp_kernel: Optional[KernelGenerator] = None,
        pallas_kernel: Optional[KernelGenerator] = None,
    ):
        """Initialize a GPU kernel choice with Warp and/or Pallas implementations.

        Args:
            default (str): The default kernel type to use. Must be either 'warp' or 'pallas',
                and the corresponding kernel must be provided.
            warp_kernel (Optional[KernelGenerator]): The Warp kernel implementation.
                Defaults to None.
            pallas_kernel (Optional[KernelGenerator]): The Pallas kernel implementation.
                Defaults to None.

        Raises:
            ValueError: If neither warp_kernel nor pallas_kernel is provided.
            AssertionError: If default is not 'warp' or 'pallas', or if the specified
                default doesn't have a corresponding kernel implementation.
        """
        self.default = default
        assert default in ['warp', 'pallas'], (
            "default must be either 'warp' or 'pallas'."
        )
        self.warp_kernel = warp_kernel
        self.pallas_kernel = pallas_kernel
        if warp_kernel is None and pallas_kernel is None:
            raise ValueError(
                "At least one of warp_kernel or pallas_kernel must be provided."
            )
        self._all_kernels = {}
        if warp_kernel is not None:
            self._all_kernels['warp'] = warp_kernel
        if pallas_kernel is not None:
            self._all_kernels['pallas'] = pallas_kernel
        assert default in self._all_kernels, (
            f"default must be one of {list(self._all_kernels.keys())}."
        )

    def __call__(self, *args, **kwargs) -> Dict:
        """Select and return the appropriate kernel implementation based on configuration.

        This method allows the GPUKernelChoice instance to be called like a function.
        It selects the appropriate kernel implementation based on the current
        configuration settings.

        Args:
            *args: Variable positional arguments passed to the kernel implementation.
            **kwargs: Variable keyword arguments passed to the kernel implementation.
        """
        if config.gpu_kernel_backend == 'default':
            backend = self.default
        elif config.gpu_kernel_backend in self._all_kernels:
            backend = config.gpu_kernel_backend
        else:
            backend = self.default
        return {backend: self._all_kernels[backend]}


class XLACustomKernel:
    """Creates and manages a custom JAX primitive for XLA custom calls.

    This class provides a high-level interface to define custom operations
    that can be executed efficiently on different backends (CPU, GPU, TPU)
    via XLA custom calls. It handles the registration of the JAX primitive,
    its abstract evaluation rule, backend-specific kernel implementations
    (using Numba for CPU, Pallas or Warp for GPU/TPU), and JAX transformation
    rules like batching, JVP (forward-mode AD), and transpose (reverse-mode AD).

    The core idea is to define the computation logic once for each relevant
    backend using specialized kernel generators (:class:`KernelGenerator`,
    :class:`KernelGenerator`, :class:`KernelGenerator`) and then use this class
    to bind everything together into a callable JAX operation.

    Attributes:
        primitive (jax.core.Primitive): The underlying JAX primitive created.
        name (str): The name assigned to the primitive.

    Args:
        name (str): The unique name for the custom JAX primitive.
        cpu_kernel (Optional[KernelGenerator]): An instance of
            `KernelGenerator` defining the computation for the CPU backend.
            Defaults to None.
        gpu_kernel (Optional[Union[KernelGenerator, KernelGenerator]]):
            An instance of `KernelGenerator` or `KernelGenerator`
            defining the computation for the GPU backend. Defaults to None.
        tpu_kernel (Optional[KernelGenerator]): An instance of
            `KernelGenerator` defining the computation for the TPU backend.
            Defaults to None.
        batching_translation (Optional[Callable]): A function defining a custom
            batching rule for the primitive. If None, a general batching rule
            is usually registered by default. See `jax.interpreters.batching`.
            Defaults to None.
        jvp_translation (Optional[Callable]): A function defining a custom JVP
            (Jacobian-Vector Product) rule for forward-mode automatic
            differentiation. See `jax.interpreters.ad.primitive_jvps`.
            Defaults to None.
        transpose_translation (Optional[Callable]): A function defining a custom
            transpose rule for reverse-mode automatic differentiation (used with
            `jax.linear_transpose`). See `jax.interpreters.ad.primitive_transposes`.
            Defaults to None.

    """

    __module__ = 'brainevent'

    def __init__(
        self,
        name: str,
        batching_translation: Callable = None,
        jvp_translation: Callable = None,
        transpose_translation: Callable = None,
    ):
        # primitive
        self.primitive = Primitive(name)
        self.primitive.multiple_results = True

        # abstract evaluation
        self.primitive.def_impl(functools.partial(xla.apply_primitive, self.primitive))
        self.primitive.def_abstract_eval(self._abstract_eval)

        # batching rule
        if batching_translation is not None:
            batching.primitive_batchers[self.primitive] = batching_translation

        # jvp rule
        if jvp_translation is not None:
            ad.primitive_jvps[self.primitive] = jvp_translation

        # transpose rule
        if transpose_translation is not None:
            ad.primitive_transposes[self.primitive] = transpose_translation

        # batching rule
        self.register_general_batching()

        # multiple gpu kernels
        self._gpu_kernel_choice = None

    def _abstract_eval(
        self,
        *ins,
        outs: OutType,
        **kwargs
    ):
        """
        Abstract evaluation rule for the JAX primitive.

        This method defines how JAX should determine the shape and dtype of the
        primitive's output(s) based on the shapes and dtypes of the inputs,
        without performing the actual computation. In this specific implementation,
        the output shapes and dtypes are explicitly provided via the `outs`
        parameter during the `primitive.bind` call and are simply returned here.

        Args:
            *ins: Abstract values (e.g., `jax.core.ShapedArray`) corresponding
                  to the input operands. Not directly used in this implementation
                  as output shapes are pre-determined.
            outs: A sequence of `jax.core.ShapedArray` objects specifying the
                  expected shape and dtype of each output. This is passed as a
                  parameter to the primitive binding.
            **kwargs: Additional keyword arguments passed during primitive binding.
                      Not used in this abstract evaluation rule.

        Returns:
            A tuple containing the `jax.core.ShapedArray` objects passed in `outs`,
            representing the abstract value of the primitive's output(s).
        """
        return tuple(outs)

    def call(self, *ins, outs: OutType, **kwargs):
        """
        Public interface to call the custom operator.

        This method serves as a user-friendly alias for the `__call__` method,
        allowing the custom operator to be invoked similarly to a standard function.

        Args:
            *ins: Variable number of input arrays (operands) for the kernel.
            outs: A single `ShapeDtype` object or a sequence of them, specifying
                  the shape and dtype of the expected output(s).
            **kwargs: Additional keyword arguments passed to the primitive binding.

        Returns:
            The result(s) of the custom operator execution, structured according
            to the `outs` specification.
        """
        return self.__call__(*ins, outs=outs, **kwargs, )

    def bind(self, *ins, outs: OutType, **kwargs):
        """
        Bind the primitive with the given inputs and parameters.

        This method is another way to invoke the custom operator, often used
        internally or when explicitly working with JAX primitives. It forwards
        the call to the `__call__` method.

        Args:
            *ins: Variable number of input arrays (operands) for the kernel.
            outs: A single `ShapeDtype` object or a sequence of them, specifying
                  the shape and dtype of the expected output(s).
            **kwargs: Additional keyword arguments passed to the primitive binding.

        Returns:
            The result(s) of the custom operator execution, structured according
            to the `outs` specification.
        """
        return self.__call__(*ins, outs=outs, **kwargs, )

    def __call__(self, *ins, outs: OutType, **kwargs):
        """
        Core method to bind and execute the custom JAX primitive.

        This method handles the actual binding of the JAX primitive defined by
        this kernel. It processes the output specifications, binds the primitive
        with the inputs and keyword arguments, and returns the results.

        Args:
            *ins: Variable number of input arrays (operands) for the kernel.
            outs: A single `ShapeDtype` object or a sequence of them, specifying
                  the shape and dtype of the expected output(s). These are
                  transformed into `jax.core.ShapedArray` internally.
            **kwargs: Additional keyword arguments passed directly to the
                      `primitive.bind` call.

        Returns:
            The result(s) of the primitive binding, potentially a single array or
            a tuple/tree of arrays, matching the structure provided in `outs`.

        Raises:
            AssertionError: If the number of results returned by `primitive.bind`
                            does not match the number of expected outputs defined
                            by `outs`.
        """
        self.ready_to_call()

        outs, tree_def = abstract_arguments(outs)
        r = self.primitive.bind(
            *ins,
            **kwargs,
            outs=tuple(outs),
        )
        assert len(r) == len(outs), 'The number of outputs does not match the expected.'
        return tree_def.unflatten(r)

    def ready_to_call(self):
        if self._gpu_kernel_choice is not None:
            self.def_gpu_kernel(**self._gpu_kernel_choice())

    def def_cpu_kernel(self, numba: Callable):
        """
        Defines and registers the CPU kernel implementation using Numba.

        This method associates a Numba kernel generator with the primitive for CPU execution.
        The kernel generator should be a callable that produces a Numba-jitted function
        that implements the operation's computation logic on CPU.

        Args:
            numba: A callable function that generates the Numba-jitted implementation.
                This generator typically configures and returns a function optimized
                for CPU execution using Numba's JIT compilation.

        Raises:
            TypeError: If the provided `numba` parameter is not callable.

        Examples::

            def my_numba_kernel_generator():
                @numba.njit
                def kernel_impl(inputs, outputs):
                    # Implementation logic
                    pass
                return kernel_impl

            custom_op = XLACustomKernel("my_custom_op")
            custom_op.def_cpu_kernel(my_numba_kernel_generator)
        """
        if not callable(numba):
            raise TypeError(
                'The `numba` parameter must be a callable that generates '
                'the Numba-jitted kernel function.'
            )
        register_numba_cpu_translation(self.primitive, numba)

    def def_gpu_kernel(
        self,
        warp: Callable = None,
        pallas: Callable = None,
        default: str = None
    ):
        """
        Defines and registers the GPU kernel implementation using Warp and/or Pallas.

        This method associates GPU kernel implementations with the primitive. It supports
        either a single implementation (Warp or Pallas) or both with a specified default.
        When both implementations are provided, the selection is determined by the
        `config.gpu_kernel_backend` setting at runtime.

        Args:
            warp: A callable that generates the NVIDIA Warp kernel implementation.
                Defaults to None.
            pallas: A callable that generates the JAX Pallas kernel implementation.
                Defaults to None.
            default: The default kernel to use when both implementations are provided.
                Must be either 'warp' or 'pallas'. Required when both warp and pallas
                are provided. Defaults to None.

        Raises:
            AssertionError: If invalid combinations of arguments are provided:
                - When warp is None, pallas must be provided
                - When both warp and pallas are provided, default must be specified
                - When default is provided, it must be either 'warp' or 'pallas'
        """
        # Validate default if provided
        if default is not None:
            assert isinstance(default, str), (
                f'The `default` should be a string, but got {type(default)}'
            )

        # Case 1: Only Pallas implementation
        if warp is None:
            assert pallas is not None, 'The `pallas` should be provided when `warp` is not provided.'
            register_pallas_gpu_translation(self.primitive, pallas)
        # Cases 2 & 3: Warp only or both implementations
        else:
            if pallas is None:
                # Case 2: Only Warp implementation
                register_warp_gpu_translation(self.primitive, warp)
            else:
                # Case 3: Both implementations provided with a default
                assert default is not None, (
                    'The `default` should be provided when multiple kernel implementations are provided.'
                )
                assert default in ['warp', 'pallas'], (
                    'The `default` should be either `warp` or `pallas`.'
                )
                self._gpu_kernel_choice = GPUKernelChoice(
                    default=default,
                    warp_kernel=warp,
                    pallas_kernel=pallas,
                )

    def def_tpu_kernel(self, pallas: Callable):
        """
        Defines and registers the TPU kernel implementation using JAX Pallas.

        This method associates a Pallas kernel generator with the primitive for TPU execution.
        TPU implementations must use the Pallas API, which provides optimized patterns for
        XLA's MLIR-based TPU compiler.

        Args:
            pallas: A callable function that generates the Pallas implementation.
                This generator should produce a TPU-optimized kernel using the
                JAX Pallas API that implements the operation's computation logic.

        Raises:
            TypeError: If the provided `pallas` parameter is not callable.

        Examples::

            def my_pallas_tpu_kernel():
                def kernel_impl(inputs, outputs):
                    # TPU-specific implementation logic using Pallas
                    pass
                return kernel_impl

            custom_op = XLACustomKernel("my_custom_op")
            custom_op.def_tpu_kernel(my_pallas_tpu_kernel)
        """
        if not callable(pallas):
            raise TypeError(
                'The `pallas` parameter must be a callable that generates '
                'the Pallas TPU kernel implementation.'
            )
        register_pallas_tpu_translation(self.primitive, pallas)

    def def_batching_rule(self, fun: Callable):
        """
        Defines a custom batching rule for the JAX primitive.

        This rule specifies how the primitive should behave when applied to
        batched inputs (inputs with a leading batch dimension).

        Args:
            fun: A callable that implements the batching logic. It typically
                 takes batched arguments and batch dimensions as input and returns
                 batched outputs and output batch dimensions. See JAX documentation
                 for `batching.primitive_batchers`.
        """
        batching.primitive_batchers[self.primitive] = fun

    def def_jvp_rule(self, fun: Callable):
        """
        Defines a custom JVP (Jacobian-vector product) rule for the primitive.

        This rule is used for forward-mode automatic differentiation (AD). It
        specifies how to compute the directional derivative of the primitive's
        output with respect to its inputs.

        Args:
            fun: A callable that implements the JVP logic. See JAX documentation
                 for `ad.primitive_jvps`.
        """
        ad.primitive_jvps[self.primitive] = fun

    def def_jvp_rule2(self, *jvp_rules):
        """
        Defines the JVP (Jacobian-vector product) rules for the primitive.

        This is a convenience method similar to `jax.interpreters.ad.defjvp`,
        but specifically adapted to handle primitives that may have multiple
        output values. It registers the JVP rules necessary for forward-mode
        automatic differentiation.

        Args:
            *jvp_rules: A sequence of callables, each defining the JVP rule for
                        a corresponding input primal. See the implementation of
                        `brainevent._xla_custom_op_util.defjvp` and JAX AD
                        documentation for details.
        """
        defjvp(self.primitive, *jvp_rules)

    def def_transpose_rule(self, fun: Callable):
        """
        Defines a custom transpose rule for the primitive.

        This rule is used for reverse-mode automatic differentiation (AD),
        specifically within the context of `jax.linear_transpose`. It defines
        how to propagate gradients backward through the primitive.

        Args:
            fun: A callable that implements the transpose logic. See JAX
                 documentation for `ad.primitive_transposes`.
        """
        ad.primitive_transposes[self.primitive] = fun

    def def_xla_translation(self, platform: str, fun: Callable):
        """
        Defines a backend-specific XLA translation rule for the primitive.

        This allows customizing how the primitive is compiled to an XLA HLO
        computation for a specific platform (e.g., 'cpu', 'gpu', 'tpu').

        Args:
            platform: A string identifying the target platform (e.g., 'cpu', 'gpu').
            fun: A callable that takes a `mlir.LoweringContext` and the operands
                 as `mlir.Value`s, and returns the `mlir.Value`s representing the
                 results of the lowered operation. See JAX XLA integration
                 documentation.
        """
        xla.backend_specific_translations[platform][self.primitive] = fun

    def def_mlir_lowering(self, platform: str, fun: Callable):
        """
        Defines a backend-specific MLIR lowering rule for the primitive.

        This provides a way to directly specify how the primitive is lowered to
        MLIR for a given platform, offering finer-grained control than XLA
        translation rules.

        Args:
            platform: A string identifying the target platform (e.g., 'cpu', 'gpu', 'tpu').
            fun: A callable responsible for the MLIR lowering. See JAX MLIR
                 lowering documentation (`jax.interpreters.mlir.register_lowering`).
        """
        mlir.register_lowering(self.primitive, fun, platform)

    def register_general_batching(self):
        """
        Registers a predefined general-purpose batching rule for the primitive.

        This method applies a common batching pattern suitable for many custom
        operators, likely handling element-wise operations or operations where
        batching involves mapping the kernel over the batch dimension. It uses
        the `general_batching_rule` function internally.
        """
        prim = self.primitive
        batching.primitive_batchers[prim] = functools.partial(general_batching_rule, prim)
