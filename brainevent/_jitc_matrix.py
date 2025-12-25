# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import operator
from typing import Union

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp

__all__ = ['JITCMatrix']


class JITCMatrix(u.sparse.SparseMatrix):
    """
    Just-in-time Connectivity (JITC) matrix.

    A base class for just-in-time connectivity matrices that inherits from
    the SparseMatrix class in the ``brainunit`` library. This class serves as
    an abstraction for sparse matrices that are generated or computed on demand
    rather than stored in full.

    JITC matrices are particularly useful in neural network simulations where
    connectivity patterns might be large but follow specific patterns that
    can be efficiently computed rather than explicitly stored in memory.

    Attributes:
        Inherits all attributes from ``brainunit.sparse.SparseMatrix``

    Note:
        This is a base class and should be subclassed for specific
        implementations of JITC matrices.
    """
    __module__ = 'brainevent'

    def _unitary_op(self, op):
        """
        Apply a unitary operation to the matrix.

        This is an internal method that should be implemented by subclasses
        to handle unitary operations like absolute value, negation, etc.

        Args:
            op (callable): A function from the operator module to apply to the matrix

        Raises:
            NotImplementedError: This is a base method that must be implemented by subclasses
        """
        raise NotImplementedError("unitary operation not implemented.")

    def __abs__(self):
        """
        Implement the absolute value operation for the matrix.
        """
        return self._unitary_op(operator.abs)

    def __neg__(self):
        """
        Implement the negation operation for the matrix.
        """
        return self._unitary_op(operator.neg)

    def __pos__(self):
        """
        Implement the unary plus operation for the matrix.
        """
        return self._unitary_op(operator.pos)

    def _binary_op(self, other, op):
        """
        Apply a binary operation between this matrix and another value.

        This is an internal method that should be implemented by subclasses
        to handle binary operations like addition, subtraction, etc.

        Args:
            other (Union[jax.typing.ArrayLike, u.Quantity]): The other operand
            op (callable): A function from the operator module to apply

        Raises:
            NotImplementedError: This is a base method that must be implemented by subclasses
        """
        raise NotImplementedError("binary operation not implemented.")

    def __mul__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        """
        Implement multiplication with another value.

        Args:
            other (Union[jax.typing.ArrayLike, u.Quantity]): The value to multiply by
        """
        return self._binary_op(other, operator.mul)

    def __div__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        """
        Implement division by another value (Python 2 compatibility).

        Args:
            other (Union[jax.typing.ArrayLike, u.Quantity]): The value to divide by
        """
        return self._binary_op(other, operator.truediv)

    def __truediv__(self, other):
        """
        Implement true division by another value.

        Args:
            other (Union[jax.typing.ArrayLike, u.Quantity]): The value to divide by
        """
        return self.__div__(other)

    def __add__(self, other):
        """
        Implement addition with another value.

        Args:
            other: The value to add
        """
        return self._binary_op(other, operator.add)

    def __sub__(self, other):
        """
        Implement subtraction with another value.

        Args:
            other: The value to subtract
        """
        return self._binary_op(other, operator.sub)

    def __mod__(self, other):
        """
        Implement modulo operation with another value.

        Args:
            other: The value to use for modulo
        """
        return self._binary_op(other, operator.mod)

    def _binary_rop(self, other, op):
        """
        Apply a binary operation with the matrix as the right operand.

        This is an internal method that should be implemented by subclasses
        to handle reflected binary operations (right-side operations).

        Args:
            other (Union[jax.typing.ArrayLike, u.Quantity]): The left operand
            op (callable): A function from the operator module to apply

        Raises:
            NotImplementedError: This is a base method that must be implemented by subclasses
        """
        raise NotImplementedError("binary operation not implemented.")

    def __rmul__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        """
        Implement right multiplication (other * self).

        Args:
            other (Union[jax.typing.ArrayLike, u.Quantity]): The value multiplying this matrix
        """
        return self._binary_rop(other, operator.mul)

    def __rdiv__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        """
        Implement right division (other / self) (Python 2 compatibility).

        Args:
            other (Union[jax.typing.ArrayLike, u.Quantity]): The value being divided
        """
        return self._binary_rop(other, operator.truediv)

    def __rtruediv__(self, other):
        """
        Implement right true division (other / self).

        Args:
            other: The value being divided
        """
        return self.__rdiv__(other)

    def __radd__(self, other):
        """
        Implement right addition (other + self).

        Args:
            other: The value being added to this matrix
        """
        return self._binary_rop(other, operator.add)

    def __rsub__(self, other):
        """
        Implement right subtraction (other - self).

        Args:
            other: The value from which this matrix is subtracted
        """
        return self._binary_rop(other, operator.sub)

    def __rmod__(self, other):
        """
        Implement right modulo (other % self).

        Args:
            other: The value to use as the left operand in the modulo operation
        """
        return self._binary_rop(other, operator.mod)


def _initialize_seed(seed=None):
    """Initialize a random seed for JAX operations.

    This function ensures a consistent format for random seeds used in JAX operations.
    If no seed is provided, it generates a random integer between 0 and 10^8 at compile time,
    ensuring reproducibility within compiled functions.

    Parameters
    ----------
    seed : int or array-like, optional
        The random seed to use. If None, a random seed is generated.

    Returns
    -------
    jax.Array
        A JAX array containing the seed value(s) with int32 dtype, ensuring it's
        in a format compatible with JAX random operations.

    Notes
    -----
    The function uses `jax.ensure_compile_time_eval()` to guarantee that random
    seed generation happens during compilation rather than during execution when
    no seed is provided, which helps maintain consistency across multiple calls
    to a JIT-compiled function.
    """
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), (1,))
    return jnp.asarray(jnp.atleast_1d(seed), dtype=jnp.int32)


def _initialize_conn_length(conn_prob: float):
    """
    Convert connection probability to connection length parameter for sparse matrix generation.

    This function transforms a connection probability (proportion of non-zero entries)
    into a connection length parameter used by the sparse sampling algorithms.
    The connection length is approximately the inverse of the connection probability,
    scaled by a factor of 2 to ensure adequate sparsity in the generated matrices.

    The function ensures the calculation happens at compile time when used in JIT-compiled
    functions by using JAX's compile_time_eval context.

    Parameters
    ----------
    conn_prob : float
        The connection probability (between 0 and 1) representing the fraction
        of non-zero entries in the randomly generated matrix.

    Returns
    -------
    jax.Array
        A JAX array containing the connection length value as an int32,
        which is approximately 2/conn_prob.

    Notes
    -----
    The connection length parameter is used in the kernels to determine the
    average distance between sampled connections when generating sparse matrices.
    Larger values result in sparser matrices (fewer connections).
    """
    with jax.ensure_compile_time_eval():
        clen = jnp.ceil(2 / conn_prob)
        clen = jnp.asarray(clen, dtype=jnp.int32)
    return clen
