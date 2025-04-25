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


from typing import Union, Sequence

import brainunit as u
import jax

from ._jitc_base import JITCMatrix
from ._typing import MatrixShape

__all__ = [
    'JITCNormalR',
    'JITCNormalC',
]


@jax.tree_util.register_pytree_node_class
class JITCNormalR(JITCMatrix):
    """
    """
    data: Union[jax.Array, u.Quantity]
    shape: MatrixShape
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        data: Union[jax.typing.ArrayLike, u.Quantity, Sequence[Union[jax.typing.ArrayLike, u.Quantity]]],
        seed: Union[int, jax.Array],
        *,
        shape: MatrixShape
    ):
        super().__init__((), shape=shape)
        data = u.math.asarray(data)
        self.loc = u.math.asarray(data[0])
        self.scale = u.math.asarray(data[1])
        self.seed = seed

    def with_data(self, data: Union[jax.typing.ArrayLike, u.Quantity]) -> 'JITCNormalC':
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return JITCNormalC(data, self.seed, shape=self.shape)


@jax.tree_util.register_pytree_node_class
class JITCNormalC(JITCMatrix):
    """
    Just-in-time Connectivity (JITC) Matrix with normally distributed connection weights in CSC format.

    This class implements a sparse matrix with normally distributed weights using a
    column-oriented format (Compressed Sparse Column). The matrix is generated on demand
    rather than stored in full, making it memory-efficient for large neural network
    connectivity patterns.

    The normal distribution of weights is defined by location (mean) and scale (standard deviation)
    parameters, and a random seed ensures reproducibility.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from brainevent import JITCNormalC

    >>> # Create a 5x5 matrix with normally distributed weights
    >>> shape = (5, 5)
    >>> seed = 1234
    >>> mean = 0.0
    >>> std = 1.0
    >>> data = [mean, std]
    >>> matrix = JITCNormalC(data, seed, shape=shape)

    >>> # Perform matrix-vector multiplication
    >>> vec = jnp.ones(shape[0])
    >>> result = matrix @ vec
    >>> print(result)

    >>> # Perform matrix-matrix multiplication
    >>> mat = jnp.ones((shape[1], shape[0]))
    >>> result = matrix @ mat
    >>> print(result)


    Attributes:
        data (Union[jax.Array, u.Quantity]): The data values of the matrix, including
            the mean and scale of the distribution.
        shape (MatrixShape): The shape of the matrix as a tuple (rows, columns).
        loc (Union[jax.Array, u.Quantity]): Mean of the normal distribution.
        scale (Union[jax.Array, u.Quantity]): Standard deviation of the normal distribution.
        seed (Union[int, jax.Array]): Random seed for reproducible weight generation.
        dtype: The data type of the matrix elements (property derived from data).

    See Also:
        JITCNormalR: Row-oriented equivalent of this matrix class.
    """
    data: Union[jax.Array, u.Quantity]
    shape: MatrixShape
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        data: Union[jax.typing.ArrayLike, u.Quantity, Sequence[Union[jax.typing.ArrayLike, u.Quantity]]],
        seed: Union[int, jax.Array],
        *,
        shape: MatrixShape
    ):
        super().__init__((), shape=shape)
        data = u.math.asarray(data)
        self.loc = u.math.asarray(data[0])
        self.scale = u.math.asarray(data[1])
        self.seed = seed

    def with_data(self, data: Union[jax.typing.ArrayLike, u.Quantity]) -> 'JITCNormalR':
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return JITCNormalR(data, self.seed, shape=self.shape)
