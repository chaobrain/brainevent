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

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...


class XLACustomKernel:
    """
    Creating a XLA custom call kernel.

    This class defines a domain-specific interface for defining custom XLA kernels.
    For a kernel customization, there are two basic concepts need to be familiar with:

    1. ``operands``: The input arguments of the kernel. It should be arrays.

    Args:
        cpu_kernel: Callable. The function defines the computation on CPU backend.
            It can be a function to generate the Numba jitted kernel.
        gpu_kernel: Callable. The function defines the computation on GPU backend.
            It can be a function to generate the JAX Pallas kernel.
        batching_translation: Callable. The batching translation rule of JAX.
        jvp_translation: Callable. The JVP translation rule of JAX.
        transpose_translation: Callable. The transpose translation rule of JAX.
        name: str. The primitive name.
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
        register_general_batching(self.primitive)

    def _abstract_eval(
        self,
        *ins,
        outs: Sequence[jax.core.ShapedArray],
        **kwargs
    ):
        return tuple(outs)

    def call(
        self,
        *ins,
        outs: Union[ShapeDtype, Sequence[ShapeDtype]],
        **kwargs,
    ):
        """
        Call the custom operator.
        """
        return self.__call__(*ins, outs=outs, **kwargs, )

    def bind(
        self,
        *ins,
        outs: Union[ShapeDtype, Sequence[ShapeDtype]],
        **kwargs,
    ):
        """
        Call the custom operator.
        """
        return self.__call__(*ins, outs=outs, **kwargs, )

    def __call__(
        self,
        *ins,
        outs: Union[ShapeDtype, Sequence[ShapeDtype]],
        **kwargs,
    ):
        """
        Call the custom operator.
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
        Define the CPU kernel using Numba.

        Args:
            kernel_generator: NumbaKernelGenerator. The function to generate the Numba jitted kernel.
        """
        if not isinstance(kernel_generator, NumbaKernelGenerator):
            raise TypeError('The `kernel_generator` should be an instance of `NumbaKernel`.')
        register_numba_cpu_translatione(self.primitive, kernel_generator)

    def def_gpu_kernel(
        self,
        kernel_generator: Union[PallasKernelGenerator, WarpKernelGenerator]
    ):
        """
        Define the GPU kernel using the JAX Pallas or Warp.

        Args:
            kernel_generator: Union[PallasKernelGenerator, WarpKernelGenerator]. The function to generate the JAX Pallas kernel.
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
        Define the TPU kernel using the JAX Pallas.

        Args:
            kernel_generator: PallasKernelGenerator. The function to generate the JAX Pallas kernel.
        """
        register_pallas_tpu_translation(self.primitive, kernel_generator)

    def def_batching_rule(self, fun: Callable):
        """Define the batching rule.

        Args:
            fun: The batching rule.
        """
        batching.primitive_batchers[self.primitive] = fun

    def def_jvp_rule(self, fun: Callable):
        """Define the JVP rule.

        Args:
            fun: The JVP rule.
        """
        ad.primitive_jvps[self.primitive] = fun

    def defjvp(self, *jvp_rules):
        """
        Define the JVP rule. Similar to ``jax.interpreters.ad.defjvp``,
        but supports the Primitive with multiple results.

        Args:
            jvp_rules: The JVP rules.
        """
        defjvp(self.primitive, *jvp_rules)

    def def_transpose_rule(self, fun: Callable):
        """Define the transpose rule.

        Args:
            fun: The transpose rule.
        """
        ad.primitive_transposes[self.primitive] = fun

    def def_xla_translation(self, platform: str, fun: Callable):
        """Define the XLA translation rule.

        Args:
            platform: str. The computing platform.
            fun: The XLA translation rule.
        """
        xla.backend_specific_translations[platform][self.primitive] = fun

    def def_mlir_lowering(self, platform: str, fun: Callable):
        """
        Define the MLIR lowering rule.

        Args:
            platform: str. The computing platform.
            fun: The lowering rule.
        """
        mlir.register_lowering(self.primitive, fun, platform)

    def register_general_batching(self):
        """
        Register the general batching rule.
        """
        register_general_batching(self.primitive)


def _transform_to_shapedarray(a):
    return jax.core.ShapedArray(a.shape, a.dtype)


def register_general_batching(prim):
    batching.primitive_batchers[prim] = functools.partial(general_batching_rule, prim)
