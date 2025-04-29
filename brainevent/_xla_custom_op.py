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
from typing import Callable, Sequence, Tuple, Protocol, Union, Optional

import jax
import numpy as np
from jax.interpreters import xla, mlir, batching, ad

from ._compatible_import import Primitive
from ._xla_custom_op_numba import (
    NumbaKernelGenerator,
    register_numba_cpu_translatione
)
from ._xla_custom_op_pallas import (
    PallasKernelGenerator,
    register_pallas_gpu_translation,
    register_pallas_tpu_translation
)
from ._xla_custom_op_util import (
    general_batching_rule,
    defjvp,
)
from ._xla_custom_op_warp import (
    WarpKernelGenerator,
    register_warp_gpu_translation
)

__all__ = [
    'XLACustomKernel',
]


class ShapeDtype(Protocol):
    """A protocol defining objects that have `shape` and `dtype` attributes.

    This protocol is used for type hinting to indicate that an object is expected
    to provide information about its tensor shape (as a tuple of integers) and
    its data type (as a NumPy dtype). It's commonly used in JAX and related
    libraries to specify the expected structure of abstract arrays or outputs
    without requiring a specific concrete class like `jax.core.ShapedArray`.

    Examples:

    .. code-block:: python

        >>> import numpy as np
        >>> from typing import Tuple
        >>>
        >>> class MyTensorSpec:
        ...     def __init__(self, shape: Tuple[int, ...], dtype: np.dtype):
        ...         self._shape = shape
        ...         self._dtype = dtype
        ...
        ...     @property
        ...     def shape(self) -> Tuple[int, ...]:
        ...         return self._shape
        ...
        ...     @property
        ...     def dtype(self) -> np.dtype:
        ...         return self._dtype
        >>>
        >>> def process_spec(spec: ShapeDtype):
        ...     print(f"Shape: {spec.shape}, Dtype: {spec.dtype}")
        >>>
        >>> spec = MyTensorSpec(shape=(10, 20), dtype=np.float32)
        >>> process_spec(spec)
        Shape: (10, 20), Dtype: float32
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the tensor as a tuple of integers."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """The data type of the tensor elements (e.g., np.float32)."""
        ...


class XLACustomKernel:
    """Creates and manages a custom JAX primitive for XLA custom calls.

    This class provides a high-level interface to define custom operations
    that can be executed efficiently on different backends (CPU, GPU, TPU)
    via XLA custom calls. It handles the registration of the JAX primitive,
    its abstract evaluation rule, backend-specific kernel implementations
    (using Numba for CPU, Pallas or Warp for GPU/TPU), and JAX transformation
    rules like batching, JVP (forward-mode AD), and transpose (reverse-mode AD).

    The core idea is to define the computation logic once for each relevant
    backend using specialized kernel generators (`NumbaKernelGenerator`,
    `PallasKernelGenerator`, `WarpKernelGenerator`) and then use this class
    to bind everything together into a callable JAX operation.

    Attributes:
        primitive (jax.core.Primitive): The underlying JAX primitive created.
        name (str): The name assigned to the primitive.

    Args:
        name (str): The unique name for the custom JAX primitive.
        cpu_kernel (Optional[NumbaKernelGenerator]): An instance of
            `NumbaKernelGenerator` defining the computation for the CPU backend.
            Defaults to None.
        gpu_kernel (Optional[Union[PallasKernelGenerator, WarpKernelGenerator]]):
            An instance of `PallasKernelGenerator` or `WarpKernelGenerator`
            defining the computation for the GPU backend. Defaults to None.
        tpu_kernel (Optional[PallasKernelGenerator]): An instance of
            `PallasKernelGenerator` defining the computation for the TPU backend.
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

    Examples:

    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import brainevent
        >>>
        >>> # --- Define Kernel Generators (Conceptual) ---
        >>> class MyAddCPUImpl(brainevent.NumbaKernelGenerator):
        ...     # ... (Implementation details for Numba kernel) ...
        ...     def generate_kernel(self, ctx, *args, **kwargs):
        ...         def _kernel(a, b, out):
        ...             # Simplified Numba kernel logic
        ...             for i in range(a.shape[0]): out[i] = a[i] + b[i]
        ...         return _kernel
        ...     # ... (get_layouts, get_grid_size etc.) ...
        >>>
        >>> class MyAddGPUImpl(brainevent.PallasKernelGenerator):
        ...     # ... (Implementation details for Pallas kernel) ...
        ...     def generate_kernel(self, ctx, *args, **kwargs):
        ...         import jax.experimental.pallas as pl
        ...         def _kernel(a_ref, b_ref, out_ref):
        ...             # Simplified Pallas kernel logic
        ...             idx = pl.program_id(axis=0)
        ...             out_ref[idx] = a_ref[idx] + b_ref[idx]
        ...         return _kernel
        ...     # ... (get_grid_spec, get_input_output_aliases etc.) ...
        >>>
        >>> # --- Create the XLACustomKernel ---
        >>> my_add_op = brainevent.XLACustomKernel(
        ...     name='my_custom_add',
        ...     cpu_kernel=MyAddCPUImpl(),
        ...     gpu_kernel=MyAddGPUImpl()
        ... )
        >>>
        >>> # --- Define Output Specification ---
        >>> # Helper class or object with shape and dtype attributes
        >>> class OutputSpec:
        ...     def __init__(self, shape, dtype):
        ...         self.shape = shape
        ...         self.dtype = dtype
        >>>
        >>> # --- Call the Custom Operation ---
        >>> @jax.jit
        ... def use_custom_op(x, y):
        ...     # Specify the expected output shape and dtype
        ...     out_spec = OutputSpec(shape=x.shape, dtype=x.dtype)
        ...     # Call the kernel like a function
        ...     return my_add_op(x, y, outs=out_spec)
        >>>
        >>> a = jnp.array([1.0, 2.0, 3.0])
        >>> b = jnp.array([4.0, 5.0, 6.0])
        >>> result = use_custom_op(a, b)
        >>> print(result)
        [5. 7. 9.] # Output depends on backend and kernel implementation
    """

    __module__ = 'brainevent'

    def __init__(
        self,
        name: str,
        cpu_kernel: Optional[NumbaKernelGenerator] = None,
        gpu_kernel: Optional[Union[PallasKernelGenerator, WarpKernelGenerator]] = None,
        tpu_kernel: Optional[PallasKernelGenerator] = None,
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

        # cpu kernel
        if cpu_kernel is not None:
            self.def_cpu_kernel(cpu_kernel)

        # gpu kernel
        if gpu_kernel is not None:
            self.def_gpu_kernel(gpu_kernel)

        # tpu kernel
        if tpu_kernel is not None:
            self.def_tpu_kernel(tpu_kernel)

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

    def _abstract_eval(
        self,
        *ins,
        outs: Sequence[jax.core.ShapedArray],
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

    def call(
        self,
        *ins,
        outs: Union[ShapeDtype, Sequence[ShapeDtype]],
        **kwargs,
    ):
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

    def bind(
        self,
        *ins,
        outs: Union[ShapeDtype, Sequence[ShapeDtype]],
        **kwargs,
    ):
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

    def __call__(
        self,
        *ins,
        outs: Union[ShapeDtype, Sequence[ShapeDtype]],
        **kwargs,
    ):
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
        outs = jax.tree.map(_transform_to_shapedarray, outs)
        outs, tree_def = jax.tree.flatten(outs)
        r = self.primitive.bind(
            *ins,
            **kwargs,
            outs=tuple(outs),
        )
        assert len(r) == len(outs), 'The number of outputs does not match the expected.'
        return tree_def.unflatten(r)

    def def_cpu_kernel(
        self,
        kernel_generator: NumbaKernelGenerator
    ):
        """
        Defines and registers the CPU kernel implementation using Numba.

        This method associates a Numba-based kernel generator with the primitive
        for execution on CPU backends. It performs a type check on the provided
        generator.

        Args:
            kernel_generator: An instance of `NumbaKernelGenerator` responsible
                              for generating the Numba jitted kernel function.

        Raises:
            TypeError: If `kernel_generator` is not an instance of
                       `NumbaKernelGenerator`.
        """
        if not isinstance(kernel_generator, NumbaKernelGenerator):
            raise TypeError('The `kernel_generator` should be an instance of `NumbaKernel`.')
        register_numba_cpu_translatione(self.primitive, kernel_generator)

    def def_gpu_kernel(
        self,
        kernel_generator: Union[PallasKernelGenerator, WarpKernelGenerator]
    ):
        """
        Defines and registers the GPU kernel implementation using JAX Pallas or Warp.

        This method associates a Pallas or Warp kernel generator with the primitive
        for execution on GPU backends. It checks the type of the generator and calls
        the appropriate registration function.

        Args:
            kernel_generator: An instance of `PallasKernelGenerator` or
                              `WarpKernelGenerator` responsible for generating the
                              GPU kernel function.

        Raises:
            TypeError: If `kernel_generator` is not an instance of
                       `PallasKernelGenerator` or `WarpKernelGenerator`.
        """

        if isinstance(kernel_generator, PallasKernelGenerator):
            register_pallas_gpu_translation(self.primitive, kernel_generator)

        elif isinstance(kernel_generator, WarpKernelGenerator):
            register_warp_gpu_translation(self.primitive, kernel_generator)

        else:
            raise TypeError('The `kernel_generator` should be an instance of `PallasKernel` or `WarpKernel`.')

    def def_tpu_kernel(
        self,
        kernel_generator: PallasKernelGenerator
    ):
        """
        Defines and registers the TPU kernel implementation using JAX Pallas.

        This method associates a Pallas kernel generator with the primitive
        for execution on TPU backends.

        Args:
            kernel_generator: An instance of `PallasKernelGenerator` responsible
                              for generating the TPU kernel function.
        """
        register_pallas_tpu_translation(self.primitive, kernel_generator)

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

    def defjvp(self, *jvp_rules):
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


def _transform_to_shapedarray(a):
    return jax.core.ShapedArray(a.shape, a.dtype)
