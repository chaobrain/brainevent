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
from typing import Callable, Optional

from jax.interpreters import xla, batching, ad, mlir

from brainevent._compatible_import import Primitive
from brainevent._typing import KernelGenerator
from .util import general_batching_rule, defjvp, OutType, abstract_arguments

__all__ = [
    'XLACustomKernel',
]


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
        primitive: The underlying JAX primitive created.
        name: The name assigned to the primitive.

    Args:
        name: The unique name for the custom JAX primitive.

    """

    __module__ = 'brainevent'

    def __init__(self, name: str):
        # primitive
        self.name = name
        self.primitive = Primitive(name)
        self.primitive.multiple_results = True

        # abstract evaluation
        self.primitive.def_impl(functools.partial(xla.apply_primitive, self.primitive))
        self.primitive.def_abstract_eval(self._abstract_eval)

        # batching rule
        self.register_general_batching()

    def _abstract_eval(self, *ins, outs: OutType, **kwargs):
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

    def __call__(self, *ins, outs: OutType, backend: Optional[str] = None, **kwargs):
        # if default backend is not the given "backend",
        # call register_translation to register the kernel again.

        outs, tree_def = abstract_arguments(outs)
        r = self.primitive.bind(*ins, **kwargs, outs=tuple(outs))
        assert len(r) == len(outs), 'The number of outputs does not match the expected.'
        return tree_def.unflatten(r)

    def def_kernel(self, backend: str, platform: str, kg: KernelGenerator, is_default: bool = False):
        assert isinstance(backend, str), f'The `backend` should be a string, but got {type(backend)}.'
        assert isinstance(platform, str), f'The `platform` should be a string, but got {type(platform)}.'
        assert callable(kg), f'The `kg` should be a callable, but got {type(kg)}.'
        if is_default:
            register_translation(platform, self.primitive, kg=kg)

    def def_numba_kernel(self, kg: KernelGenerator, is_default: bool = False):
        self.def_kernel(backend='numba', platform='cpu', kg=kg, is_default=is_default)

    def def_warp_kernel(self, kg: KernelGenerator, is_default: bool = False):
        self.def_kernel(backend='warp', platform='gpu', kg=kg, is_default=is_default)

    def def_triton_kernel(self, kg: KernelGenerator, is_default: bool = False):
        self.def_kernel(backend='triton', platform='gpu', kg=kg, is_default=is_default)

    def def_pallas_kernel(self, platform: str, kg: KernelGenerator, is_default: bool = False):
        assert platform in ['gpu', 'tpu'], f'The `platform` should be either `gpu` or `tpu`, but got {platform}.'
        self.def_kernel(backend='pallas', platform=platform, kg=kg, is_default=is_default)

    def def_tvmffi_kernel(self, platform: str, kg: KernelGenerator, is_default: bool = False):
        assert platform in ['cpu', 'gpu'], f'The `platform` should be either `cpu` or `gpu`, but got {platform}.'
        self.def_kernel(backend='tvmffi', platform=platform, kg=kg, is_default=is_default)

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


def register_translation(platform: str, primitive: Primitive, kg: KernelGenerator):
    def kernel_fn(*args, **kwargs):
        kernel = kg(**kwargs)
        return kernel(*args)

    assert isinstance(platform, str), f"platform must be a string. But got {platform}."
    lower = mlir.lower_fun(kernel_fn, multiple_results=True)
    mlir.register_lowering(primitive, lower, platform=platform)
