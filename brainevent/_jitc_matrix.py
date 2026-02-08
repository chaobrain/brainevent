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
        Apply a unary operation to the matrix.

        This is an internal method that should be implemented by subclasses
        to handle unary operations like absolute value, negation, etc.

        Args:
            op (callable): A function from the operator module to apply to the matrix

        Raises:
            NotImplementedError: This is a base method that must be implemented by subclasses
        """
        raise NotImplementedError("unitary operation not implemented.")

    def apply(self, fn):
        """
        Apply a function to matrix value parameters while keeping structure.

        Parameters
        ----------
        fn : callable
            Unary callable applied by subclasses to their value parameters.

        Returns
        -------
        JITCMatrix
            A new matrix-like object with transformed values.
        """
        return self._unitary_op(fn)

    def __abs__(self):
        return self.apply(operator.abs)

    def __neg__(self):
        return self.apply(operator.neg)

    def __pos__(self):
        return self.apply(operator.pos)

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

    def apply2(self, other, fn, *, reverse: bool = False):
        """
        Apply a binary function with consistent sparse-matrix semantics.

        Parameters
        ----------
        other : Any
            Right-hand operand for normal operations, or left-hand operand when
            ``reverse=True``.
        fn : callable
            Binary function from ``operator`` or a compatible callable.
        reverse : bool, optional
            If False, compute ``fn(self, other)`` via ``_binary_op``.
            If True, compute ``fn(other, self)`` via ``_binary_rop``.
            Defaults to False.

        Returns
        -------
        JITCMatrix or Any
            Result of the operation.
        """
        if reverse:
            return self._binary_rop(other, fn)
        return self._binary_op(other, fn)

    def __mul__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        return self.apply2(other, operator.mul)

    def __truediv__(self, other):
        return self.apply2(other, operator.truediv)

    def __add__(self, other):
        return self.apply2(other, operator.add)

    def __sub__(self, other):
        return self.apply2(other, operator.sub)

    def __mod__(self, other):
        return self.apply2(other, operator.mod)

    def __rmul__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        return self.apply2(other, operator.mul, reverse=True)

    def __rtruediv__(self, other):
        return self.apply2(other, operator.truediv, reverse=True)

    def __radd__(self, other):
        return self.apply2(other, operator.add, reverse=True)

    def __rsub__(self, other):
        return self.apply2(other, operator.sub, reverse=True)

    def __rmod__(self, other):
        return self.apply2(other, operator.mod, reverse=True)


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
