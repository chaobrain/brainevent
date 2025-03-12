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

import jax
from jax.interpreters import mlir

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Primitive
else:
    from jax.extend.core import Primitive

__all__ = [
    'PallasKernelGenerator',
]


@dataclasses.dataclass(frozen=True)
class PallasKernelGenerator:
    """
    The JAX Pallas kernel generator.

    Args:
        generator: Callable. The function defines the computation on GPU/TPU backend using JAX Pallas.
            See the `JAX Pallas documentation <https://jax.readthedocs.io/en/latest/pallas/quickstart.html>`_
            for more details .
        block_dim: Union[int, Callable[..., int]. The block dimension of the JAX Pallas kernel.
    """
    __module__ = 'brainevent'
    generator: Callable[..., Callable]
    block_dim: Union[int, Callable[..., int]] = None
    input_output_aliases: Union[Dict[int, int], Callable[..., Dict[int, int]], None] = None

    def generate_kernel(self, **kwargs):
        return self.generator(**kwargs)

    def get_block_dim(self, **kwargs):
        if callable(self.block_dim):
            return self.block_dim(**kwargs)
        elif isinstance(self.block_dim, int):
            return self.block_dim
        elif self.block_dim is None:
            return None
        else:
            raise ValueError(f"Invalid block_dim: {self.block_dim}")


def register_pallas_gpu_translation(
    primitive: Primitive,
    kernel_generator: PallasKernelGenerator,
):
    """
    Register the JAX Pallas GPU translation rule for the custom operator.

    Args:
        primitive: Primitive. The custom operator.
        kernel_generator: Callable. The function defines the computation on GPU backend.
            It can be a function to generate the JAX Pallas kernel.
    """
    lower = mlir.lower_fun(
        lambda *args, **kwargs: kernel_generator.generate_kernel(
            block_dim=kernel_generator.get_block_dim(**kwargs),
            **kwargs
        )(*args),
        multiple_results=True
    )
    mlir.register_lowering(primitive, lower, platform='cuda')


def register_pallas_tpu_translation(
    primitive: Primitive,
    kernel_generator: PallasKernelGenerator,
):
    """
    Register the JAX Pallas GPU translation rule for the custom operator.

    Args:
        primitive: Primitive. The custom operator.
        kernel_generator: Callable. The function defines the computation on GPU backend.
            It can be a function to generate the JAX Pallas kernel.
    """
    lower = mlir.lower_fun(
        lambda *args, **kwargs: kernel_generator.generate_kernel(
            block_dim=kernel_generator.get_block_dim(**kwargs),
            **kwargs
        )(*args),
        multiple_results=True
    )
    mlir.register_lowering(primitive, lower, platform='tpu')
