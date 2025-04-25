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


from typing import Union

import brainunit as u
import jax

from ._jitc_base import JITCMatrix
from ._typing import MatrixShape

__all__ = [
    'JITRUniform',
    'JITCUniform',
]


@jax.tree_util.register_pytree_node_class
class JITRUniform(JITCMatrix):
    """
    """
    data: Union[jax.Array, u.Quantity]
    shape: MatrixShape
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        data: Union[jax.typing.ArrayLike, u.Quantity],
        seed: Union[int, jax.Array],
        *,
        shape: MatrixShape
    ):
        super().__init__((), shape=shape)
        self.data = u.math.asarray(data)
        self.seed = seed

    def with_data(self, data: Union[jax.typing.ArrayLike, u.Quantity]) -> 'JITCHomoR':
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return JITCHomoR(data, self.seed, shape=self.shape)


@jax.tree_util.register_pytree_node_class
class JITCUniform(JITCMatrix):
    """
    """
    data: Union[jax.Array, u.Quantity]
    shape: MatrixShape
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        data: Union[jax.typing.ArrayLike, u.Quantity],
        seed: Union[int, jax.Array],
        *,
        shape: MatrixShape
    ):
        super().__init__((), shape=shape)
        self.data = u.math.asarray(data)
        self.seed = seed

    def with_data(self, data: Union[jax.typing.ArrayLike, u.Quantity]) -> 'JITCHomoR':
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return JITCHomoR(data, self.seed, shape=self.shape)
