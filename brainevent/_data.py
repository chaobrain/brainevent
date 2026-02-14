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

__all__ = [
    'DataRepresentation',
    'JITCMatrix',
]


class DataRepresentation(u.sparse.SparseMatrix):
    pass


class JITCMatrix(DataRepresentation):
    """
    Just-in-time Connectivity (JITC) matrix.

    A base class for just-in-time connectivity matrices that inherits from
    the SparseMatrix class in the ``brainunit`` library. This class serves as
    an abstraction for sparse matrices that are generated or computed on demand
    rather than stored in full.

    JITC matrices are particularly useful in neural network simulations where
    connectivity patterns might be large but follow specific patterns that
    can be efficiently computed rather than explicitly stored in memory.

    Notes
    -----
    This is a base class and should be subclassed for specific
    implementations of JITC matrices. All attributes from
    :class:`brainunit.sparse.SparseMatrix` are inherited.
    """
    __module__ = 'brainevent'

    def _unitary_op(self, op):
        """
        Apply a unary operation to the matrix.

        This is an internal method that should be implemented by subclasses
        to handle unary operations like absolute value, negation, etc.

        Parameters
        ----------
        op : callable
            Function from ``operator`` or compatible callable to apply.

        Raises
        ------
        NotImplementedError
            Raised because this base method must be implemented by subclasses.
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
        """
        Return the element-wise absolute value of the matrix.

        Computes ``abs(weight)`` for each value parameter of the matrix while
        preserving the sparse connectivity structure (probability, seed, shape,
        and memory layout order).

        Returns
        -------
        JITCMatrix
            A new JITC matrix whose value parameters are the absolute values
            of the original.

        See Also
        --------
        apply : General unary function application.
        __neg__ : Element-wise negation.
        __pos__ : Element-wise positive (identity).

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((-1.5, 0.1, 42), shape=(10, 10))
            >>> abs_mat = abs(mat)
            >>> float(abs_mat.weight)
            1.5
        """
        return self.apply(operator.abs)

    def __neg__(self):
        """
        Return the element-wise negation of the matrix.

        Computes ``-weight`` for each value parameter of the matrix while
        preserving the sparse connectivity structure.

        Returns
        -------
        JITCMatrix
            A new JITC matrix whose value parameters are negated.

        See Also
        --------
        apply : General unary function application.
        __abs__ : Element-wise absolute value.
        __pos__ : Element-wise positive (identity).

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> neg_mat = -mat
            >>> float(neg_mat.weight)
            -1.5
        """
        return self.apply(operator.neg)

    def __pos__(self):
        """
        Return the element-wise positive of the matrix (identity operation).

        Computes ``+weight`` for each value parameter, which returns an
        equivalent matrix. The sparse connectivity structure is preserved.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with the same value parameters.

        See Also
        --------
        apply : General unary function application.
        __abs__ : Element-wise absolute value.
        __neg__ : Element-wise negation.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> pos_mat = +mat
            >>> float(pos_mat.weight)
            1.5
        """
        return self.apply(operator.pos)

    def _binary_op(self, other, op):
        """
        Apply a binary operation between this matrix and another value.

        This is an internal method that should be implemented by subclasses
        to handle binary operations like addition, subtraction, etc.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand operand.
        op : callable
            Function from ``operator`` or compatible callable to apply.

        Raises
        ------
        NotImplementedError
            Raised because this base method must be implemented by subclasses.
        """
        raise NotImplementedError("binary operation not implemented.")

    def _binary_rop(self, other, op):
        """
        Apply a binary operation with the matrix as the right operand.

        This is an internal method that should be implemented by subclasses
        to handle reflected binary operations (right-side operations).

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand operand.
        op : callable
            Function from ``operator`` or compatible callable to apply.

        Raises
        ------
        NotImplementedError
            Raised because this base method must be implemented by subclasses.
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
        """
        Multiply the matrix element-wise by a scalar or array.

        Computes ``self * other`` by applying ``operator.mul`` to the value
        parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand multiplicand. Typically a scalar value; support for
            non-scalar operands depends on the subclass implementation.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with scaled value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __rmul__ : Reflected multiplication (``other * self``).
        __truediv__ : Element-wise division.
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> scaled = mat * 2.0
            >>> float(scaled.weight)
            3.0
        """
        return self.apply2(other, operator.mul)

    def __truediv__(self, other):
        """
        Divide the matrix element-wise by a scalar or array.

        Computes ``self / other`` by applying ``operator.truediv`` to the
        value parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand divisor. Typically a scalar value; support for
            non-scalar operands depends on the subclass implementation.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with divided value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __rtruediv__ : Reflected division (``other / self``).
        __mul__ : Element-wise multiplication.
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((3.0, 0.1, 42), shape=(10, 10))
            >>> divided = mat / 2.0
            >>> float(divided.weight)
            1.5
        """
        return self.apply2(other, operator.truediv)

    def __add__(self, other):
        """
        Add a scalar or array element-wise to the matrix.

        Computes ``self + other`` by applying ``operator.add`` to the value
        parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand addend. Typically a scalar value or another JITC matrix
            with the same connectivity structure.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with summed value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape,
            or if two JITC matrices have incompatible seeds, shapes, or
            probabilities.

        See Also
        --------
        __radd__ : Reflected addition (``other + self``).
        __sub__ : Element-wise subtraction.
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> result = mat + 0.5
            >>> float(result.weight)
            2.0
        """
        return self.apply2(other, operator.add)

    def __sub__(self, other):
        """
        Subtract a scalar or array element-wise from the matrix.

        Computes ``self - other`` by applying ``operator.sub`` to the value
        parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand subtrahend. Typically a scalar value or another JITC
            matrix with the same connectivity structure.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with subtracted value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape,
            or if two JITC matrices have incompatible seeds, shapes, or
            probabilities.

        See Also
        --------
        __rsub__ : Reflected subtraction (``other - self``).
        __add__ : Element-wise addition.
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> result = mat - 0.5
            >>> float(result.weight)
            1.0
        """
        return self.apply2(other, operator.sub)

    def __mod__(self, other):
        """
        Compute the element-wise modulo of the matrix by a scalar or array.

        Computes ``self % other`` by applying ``operator.mod`` to the value
        parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand modulus. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with the modulo-reduced value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __rmod__ : Reflected modulo (``other % self``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((5.0, 0.1, 42), shape=(10, 10))
            >>> result = mat % 3.0
            >>> float(result.weight)
            2.0
        """
        return self.apply2(other, operator.mod)

    def __rmul__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        """
        Reflected multiplication: multiply a scalar or array by the matrix.

        Computes ``other * self`` by applying ``operator.mul`` with the
        operands in reflected order. This is invoked when the left operand
        does not support multiplication with a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand multiplicand. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with scaled value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __mul__ : Forward multiplication (``self * other``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> scaled = 2.0 * mat
            >>> float(scaled.weight)
            3.0
        """
        return self.apply2(other, operator.mul, reverse=True)

    def __rtruediv__(self, other):
        """
        Reflected division: divide a scalar or array by the matrix.

        Computes ``other / self`` by applying ``operator.truediv`` with the
        operands in reflected order. This is invoked when the left operand
        does not support division by a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand dividend. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix where each value parameter ``w`` is replaced
            by ``other / w``.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __truediv__ : Forward division (``self / other``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((2.0, 0.1, 42), shape=(10, 10))
            >>> result = 6.0 / mat
            >>> float(result.weight)
            3.0
        """
        return self.apply2(other, operator.truediv, reverse=True)

    def __radd__(self, other):
        """
        Reflected addition: add the matrix to a scalar or array.

        Computes ``other + self`` by applying ``operator.add`` with the
        operands in reflected order. This is invoked when the left operand
        does not support addition with a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand addend. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with summed value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __add__ : Forward addition (``self + other``).
        apply2 : General binary function application.

        Notes
        -----
        For commutative operands (e.g., plain scalars), ``other + self``
        produces the same result as ``self + other``.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> result = 0.5 + mat
            >>> float(result.weight)
            2.0
        """
        return self.apply2(other, operator.add, reverse=True)

    def __rsub__(self, other):
        """
        Reflected subtraction: subtract the matrix from a scalar or array.

        Computes ``other - self`` by applying ``operator.sub`` with the
        operands in reflected order. This is invoked when the left operand
        does not support subtraction of a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand minuend. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix where each value parameter ``w`` is replaced
            by ``other - w``.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __sub__ : Forward subtraction (``self - other``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> result = 3.0 - mat
            >>> float(result.weight)
            1.5
        """
        return self.apply2(other, operator.sub, reverse=True)

    def __rmod__(self, other):
        """
        Reflected modulo: compute a scalar or array modulo the matrix.

        Computes ``other % self`` by applying ``operator.mod`` with the
        operands in reflected order. This is invoked when the left operand
        does not support modulo with a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand dividend for the modulo operation.

        Returns
        -------
        JITCMatrix
            A new JITC matrix where each value parameter ``w`` is replaced
            by ``other % w``.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __mod__ : Forward modulo (``self % other``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((3.0, 0.1, 42), shape=(10, 10))
            >>> result = 7.0 % mat
            >>> float(result.weight)
            1.0
        """
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
