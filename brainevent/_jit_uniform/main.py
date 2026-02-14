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


from typing import Union, Tuple, Optional

import brainunit as u
import jax
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import JITCMatrix
from brainevent._event.binary import BinaryArray
from brainevent._typing import MatrixShape, WeightScalar, Prob, Seed
from .binary import binary_jitumv, binary_jitumm
from .float import jitu, jitumv, jitumm

__all__ = [
    'JITCUniformR',
    'JITCUniformC',
]


class JITUniformMatrix(JITCMatrix):
    """
    Base class for Just-In-Time Connectivity Uniform Distribution matrices.

    This abstract class serves as the foundation for sparse matrix representations
    that use uniformly distributed weights with stochastic connectivity patterns.
    It stores lower and upper bounds for the uniform distribution, along with
    connectivity probability and a random seed that determines the sparse structure.

    Designed for efficient representation of neural connectivity matrices where
    connections follow a uniform distribution but are sparsely distributed.

    Parameters
    ----------
    low : WeightScalar or Tuple[WeightScalar, WeightScalar, Prob, Seed]
        Either the lower bound of the uniform distribution,
        or a tuple containing (low, high, prob, seed).
    high : WeightScalar, optional
        Upper bound of the uniform distribution.
    prob : Prob, optional
        Connection probability determining matrix sparsity.
    seed : Seed, optional
        Random seed for reproducible sparse structure generation.
    shape : MatrixShape
        The shape of the matrix as a tuple (rows, columns).
    corder : bool, optional
        Memory layout order flag, by default False.
    backend : str, optional
        Computation backend override.

    Attributes
    ----------
    wlow : Union[jax.Array, u.Quantity]
        The lower bound of the uniform distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    whigh : Union[jax.Array, u.Quantity]
        The upper bound of the uniform distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    prob : Union[float, jax.Array]
        Connection probability determining the sparsity of the matrix.
        Values range from 0 (no connections) to 1 (fully connected).
    seed : Union[int, jax.Array]
        Random seed controlling the specific pattern of connections.
        Using the same seed produces identical connectivity patterns.
    shape : MatrixShape
        Tuple specifying the dimensions of the matrix as (rows, columns).
    corder : bool
        Flag indicating the memory layout order of the matrix.
        False (default) for Fortran-order (column-major), True for C-order (row-major).

    Raises
    ------
    ValueError
        If ``prob`` is not a finite scalar in [0, 1], or if ``wlow > whigh``
        element-wise.

    See Also
    --------
    JITCUniformR : Row-oriented concrete subclass.
    JITCUniformC : Column-oriented concrete subclass.

    Notes
    -----
    The mathematical model for this matrix is:

        ``W[i, j] = Uniform(w_low, w_high) * Bernoulli(prob)``

    That is, each entry ``W[i, j]`` is independently set to a value drawn from the
    continuous uniform distribution on ``[w_low, w_high]`` with probability ``prob``,
    and set to zero with probability ``1 - prob``. More precisely:

        ``W[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Uniform(w_low, w_high)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables. The connectivity pattern ``B`` and uniform
    variates ``U`` are determined by the ``seed`` parameter, so using the same seed
    always produces the same matrix.

    The matrix is never materialized in memory; instead, weights and connectivity
    are generated on-the-fly during matrix operations using a PRNG seeded by
    ``seed``.
    """
    __module__ = 'brainevent'

    wlow: Union[jax.Array, u.Quantity]
    whigh: Union[jax.Array, u.Quantity]
    prob: Union[float, jax.Array]
    seed: Union[int, jax.Array]
    shape: MatrixShape
    corder: bool

    def __init__(
        self,
        low,
        high=None,
        prob=None,
        seed=None,
        *,
        shape: MatrixShape,
        corder: bool = False,
        backend: Optional[str] = None,
    ):
        """
        Initialize a uniform distribution sparse matrix.

        Parameters
        ----------
        low : WeightScalar or Tuple[WeightScalar, WeightScalar, Prob, Seed]
            Either the lower bound of the uniform distribution,
            or a tuple containing (low, high, prob, seed).
        high : WeightScalar, optional
            Upper bound of the uniform distribution.
            If None, ``low`` is treated as a tuple of (low, high, prob, seed).
        prob : Prob, optional
            Connection probability determining matrix sparsity.
        seed : Seed, optional
            Random seed for reproducible sparse structure generation.
        shape : MatrixShape
            The shape of the matrix as a tuple (rows, columns).
        corder : bool, optional
            Memory layout order flag, by default False.
            - False: Fortran-order (column-major)
            - True: C-order (row-major)

        Notes
        -----
        The constructor extracts the components from the data tuple and sets them
        as instance attributes. The weight parameters are promoted to have compatible
        dtypes and are verified to have matching dimensions before being converted
        to JAX arrays, preserving any attached units.
        """
        if high is None and prob is None and seed is None:
            data = low
        else:
            data = (low, high, prob, seed)
        low, high, self.prob, self.seed = data
        if not isinstance(self.prob, Tracer):
            prob = np.asarray(self.prob)
            if prob.size != 1:
                raise ValueError(f"prob must be a scalar, but got shape {prob.shape}.")
            prob = float(prob.item())
            if not np.isfinite(prob):
                raise ValueError(f"prob must be finite, but got {prob}.")
            if not (0. <= prob <= 1.):
                raise ValueError(f"prob must be in [0, 1], but got {prob}.")

        low, high = u.math.promote_dtypes(low, high)
        u.fail_for_dimension_mismatch(low, high, "wlow and whigh must have the same dimension.")
        low_m = u.get_mantissa(low)
        high_m = u.get_mantissa(high)
        if not (isinstance(low_m, Tracer) or isinstance(high_m, Tracer)):
            low_arr = np.asarray(low_m)
            high_arr = np.asarray(high_m)
            if np.any(low_arr > high_arr):
                raise ValueError("wlow must be <= whigh element-wise.")
        self.wlow = u.math.asarray(low)
        self.whigh = u.math.asarray(high)
        self.corder = corder
        self.backend = backend
        super().__init__(data, shape=shape)

    def __repr__(self):
        """
        Return a string representation of the uniform distribution matrix.

        Returns
        -------
        str
            A string showing the class name, shape, lower bound, upper bound,
            probability, seed, and corder flag of the matrix instance.

        Examples
        --------
        >>> matrix = JITUniformMatrix((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>> repr(matrix)
        'JITUniformMatrix(shape=(10, 10), wlow=0.1, whigh=0.5, prob=0.2, seed=42, corder=False)'
        """
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"wlow={self.wlow}, "
            f"whigh={self.whigh}, "
            f"prob={self.prob}, "
            f"seed={self.seed}, "
            f"corder={self.corder},"
            f"backend={self.backend},"
            f")"
        )

    @property
    def dtype(self):
        """
        Get the data type of the matrix elements.

        Returns
        -------
        dtype
            The data type of the lower bound values in the matrix.

        Notes
        -----
        This property inherits the dtype directly from the wlow attribute,
        ensuring consistent data typing throughout operations involving this matrix.
        """
        return self.wlow.dtype

    @property
    def data(self) -> Tuple[WeightScalar, WeightScalar, Prob, Seed]:
        """
        Returns the core data components of the homogeneous matrix.

        This property provides access to the three fundamental components that define
        the sparse matrix: weight values, connection probabilities, and the random seed.
        It's used by the tree_flatten method to make the class compatible with JAX
        transformations.

        Returns
        -------
        Tuple[Weight, Weight, Prob, Seed]
            A tuple containing:
            - loc:
            - scale:
            - prob: Connection probability for the sparse structure
            - seed: Random seed used for generating the sparse connectivity pattern
        """
        return self.wlow, self.whigh, self.prob, self.seed

    def with_data(self, low: WeightScalar, high: WeightScalar):
        """
        Create a new matrix instance with updated weight data but preserving other properties.

        This method returns a new instance of the same class with the provided lower and
        upper bound values, while keeping the same probability, seed, shape, and other
        configuration parameters. It's useful for updating weight bounds without changing
        the connectivity pattern.

        Parameters
        ----------
        low : WeightScalar
            New lower bound value for the uniform distribution. Must have the same shape
            and units as the original lower bound.
        high : WeightScalar
            New upper bound value for the uniform distribution. Must have the same shape
            and units as the original upper bound.

        Returns
        -------
        JITUniformMatrix
            A new matrix instance of the same type as the original, with updated
            lower and upper bounds but identical connectivity structure.

        Raises
        ------
        AssertionError
            If the shapes of the provided bounds don't match the shapes of the original bounds,
            or if the units of the provided bounds don't match the units of the original bounds.

        Examples
        --------
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCUniformR
        >>>
        >>> # Create original matrix
        >>> original = JITCUniformR((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>>
        >>> # Create new matrix with updated bounds
        >>> updated = original.with_data(0.2, 0.8)
        >>> print(updated.wlow, updated.whigh)  # 0.2 0.8
        >>>
        >>> # With units
        >>> original_units = JITCUniformR((0.1 * u.mV, 0.5 * u.mV, 0.2, 42), shape=(10, 10))
        >>> updated_units = original_units.with_data(0.2 * u.mV, 0.8 * u.mV)
        """
        low = u.math.asarray(low)
        high = u.math.asarray(high)
        assert low.shape == self.wlow.shape
        assert high.shape == self.whigh.shape
        assert u.get_unit(low) == u.get_unit(self.wlow)
        assert u.get_unit(high) == u.get_unit(self.whigh)
        return type(self)(
            (low, high, self.prob, self.seed),
            shape=self.shape,
            corder=self.corder,
            backend=self.backend,
        )

    def tree_flatten(self):
        """
        Flatten the matrix into a list of leaves and auxiliary data for JAX pytree.

        Returns
        -------
        tuple
            A pair of (children, aux_data) where children is a tuple of
            (wlow, whigh, prob, seed) and aux_data is a dict containing
            shape, corder, and backend.

        Notes
        -----
        This method is used by JAX's pytree system to serialize the matrix
        for transformations such as ``jax.jit``, ``jax.grad``, and ``jax.vmap``.
        """
        aux = {'shape': self.shape, 'corder': self.corder, 'backend': self.backend}
        return (self.wlow, self.whigh, self.prob, self.seed), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a matrix from its flattened pytree representation.

        Parameters
        ----------
        aux_data : dict
            Auxiliary data containing shape, corder, and backend.
        children : tuple
            A tuple of (wlow, whigh, prob, seed) leaf values.

        Returns
        -------
        JITUniformMatrix
            A reconstructed matrix instance.

        Notes
        -----
        This classmethod is used by JAX's pytree system to deserialize the
        matrix after transformations. It bypasses ``__init__`` by using
        ``object.__new__`` and directly setting attributes.
        """
        obj = object.__new__(cls)
        obj.wlow, obj.whigh, obj.prob, obj.seed = children
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj

    def _check(self, other, op):
        """
        Validate compatibility of two matrices for binary operations.

        Parameters
        ----------
        other : JITUniformMatrix
            The other matrix to check compatibility with.
        op : str
            Name of the binary operation being performed, used in error messages.

        Raises
        ------
        NotImplementedError
            If the two matrices have different seeds, tracing seeds,
            or different corder values.
        """
        if not (isinstance(other.seed, Tracer) and isinstance(self.seed, Tracer)):
            if self.seed != other.seed:
                raise NotImplementedError(
                    f"binary operation {op} between two {self.__class__.__name__} "
                    f"objects with different seeds "
                    f"is not implemented currently."
                )
        else:
            raise NotImplementedError(
                f"binary operation {op} between two {self.__class__.__name__} "
                f"objects with tracing seeds "
                f"is not implemented currently."
            )
        if self.corder != other.corder:
            raise NotImplementedError(
                f"binary operation {op} between two {self.__class__.__name__} "
                f"objects with different corder "
                f"is not implemented currently."
            )


@jax.tree_util.register_pytree_node_class
class JITCUniformR(JITUniformMatrix):
    """
    Just-In-Time Connectivity matrix with Row-oriented representation for uniform weight distributions.

    This class implements a row-oriented sparse matrix optimized for JAX-based transformations,
    following the Compressed Sparse Row (CSR) format conceptually. Instead of storing all non-zero
    elements explicitly, it uses a uniform distribution with lower and upper bounds (wlow, whigh)
    to generate weights for connections, along with probability and seed information to
    determine the sparse structure.

    The class is designed for efficient neural network connectivity patterns where weights
    follow a uniform distribution but connectivity is sparse and stochastic. The actual sparse
    structure and uniform weight values are generated just-in-time during operations.

    Attributes
    ----------
    wlow : Union[jax.Array, u.Quantity]
        The lower bound of the uniform distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    whigh : Union[jax.Array, u.Quantity]
        The upper bound of the uniform distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    prob : Union[float, jax.Array]
        Connection probability determining the sparsity of the matrix.
        Values range from 0 (no connections) to 1 (fully connected).
    seed : Union[int, jax.Array]
        Random seed controlling the specific pattern of connections.
        Using the same seed produces identical connectivity patterns.
    shape : MatrixShape
        Tuple specifying the dimensions of the matrix as (rows, columns).
    corder : bool
        Flag indicating the memory layout order of the matrix.
        False (default) for Fortran-order (column-major), True for C-order (row-major).
    dtype
        The data type of the matrix elements (property inherited from parent).

    Examples
    --------

    .. code-block:: python

        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCUniformR

        # Create a uniform matrix with bounds [0.1, 0.5], probability 0.2, and seed 42
        >>> uniform_matrix = JITCUniformR((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>> uniform_matrix
        JITCUniformR(shape=(10, 10), wlow=0.1, whigh=0.5, prob=0.2, seed=42, corder=False)

        # Create a uniform matrix with units
        >>> uniform_matrix_mv = JITCUniformR((0.1 * u.mV, 0.5 * u.mV, 0.2, 42), shape=(10, 10))

        # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = uniform_matrix @ vec
        >>> # Each element in result is a weighted sum using uniformly distributed weights

        # Apply scalar operation (scales both lower and upper bounds)
        >>> scaled = uniform_matrix * 2.0
        >>> print(scaled.wlow, scaled.whigh)  # 0.2 1.0

        # Convert to dense representation
        >>> dense_matrix = uniform_matrix.todense()
        >>> # dense_matrix has shape (10, 10) with ~20% non-zero elements
        >>> # each non-zero element is uniformly distributed between 0.1 and 0.5

        # Transpose operation returns a JITCUniformC instance
        >>> col_matrix = uniform_matrix.transpose()
        >>> isinstance(col_matrix, JITCUniformC)  # True

        # Update bounds while preserving connectivity pattern
        >>> updated = uniform_matrix.with_data(0.2, 0.8)
        >>> print(updated.wlow, updated.whigh)  # 0.2 0.8

        # Use with JAX transformations
        >>> @jax.jit
        ... def matrix_vector_product(mat, vec):
        ...     return mat @ vec
        >>> result_jit = matrix_vector_product(uniform_matrix, vec)

    Notes
    -----
    The mathematical model for ``JITCUniformR`` is:

        ``W[i, j] = Uniform(w_low, w_high) * Bernoulli(prob)``

    Each entry ``W[i, j]`` is independently drawn from the continuous uniform
    distribution on ``[w_low, w_high]`` with probability ``prob``, and zero
    otherwise. More precisely, the entry is computed as:

        ``W[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Uniform(w_low, w_high)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables, both determined by ``seed``.

    The row-oriented representation means that the random number generator state is
    seeded per-row (or per-column, depending on ``corder``), making row-based
    operations (``W @ v``) the natural direction.

    Key properties:

    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with uniformly distributed weights
    - Optimized for matrix-vector operations common in neural simulations
    - The actual matrix elements are never explicitly stored, only generated during operations
    - Using the same seed always produces the same random connectivity pattern and weights
    """
    __module__ = 'brainevent'

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the sparse uniform matrix to a dense array.

        Generates a full dense representation of the sparse matrix by
        sampling ``Uniform(w_low, w_high)`` values for all connections
        determined by the probability and seed. The resulting dense matrix
        preserves all the numerical properties of the sparse representation.

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A dense matrix with the same shape as the sparse matrix. The data type
            will match the weight's data type, and if the weight has units (is a
            ``u.Quantity``), the returned array will have the same units.

        Raises
        ------
        None
            This method does not raise exceptions under normal use.

        See Also
        --------
        JITCUniformC.todense : Column-oriented variant.
        jitu : Standalone function to materialize JIT uniform matrices.

        Notes
        -----
        The dense matrix is generated according to:

            ``dense[i, j] = Uniform(w_low, w_high) * Bernoulli(prob)``

        for each ``(i, j)`` pair, where the random draws are determined by ``seed``.

        Examples
        --------

        .. code-block:: python

            >>> import jax
            >>> from brainevent import JITCUniformR
            >>>
            >>> mat = JITCUniformR((0.1, 0.5, 0.2, 42), shape=(4, 6))
            >>> dense = mat.todense()
            >>> dense.shape
            (4, 6)
        """
        return jitu(
            self.wlow,
            self.whigh,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
            backend=self.backend,
        )

    def transpose(self, axes=None) -> 'JITCUniformC':
        """
        Transpose the row-oriented matrix into a column-oriented matrix.

        Returns a column-oriented matrix (``JITCUniformC``) with rows and columns
        swapped, preserving the same weight bounds, probability, and seed values.
        The transpose operation effectively converts between row-oriented and
        column-oriented sparse matrix formats.

        Parameters
        ----------
        axes : None
            Not supported. This parameter exists for compatibility with the NumPy API
            but only ``None`` is accepted.

        Returns
        -------
        JITCUniformC
            A new column-oriented uniform matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``, since partial axis transposition is not supported.

        See Also
        --------
        JITCUniformC.transpose : The inverse operation.

        Notes
        -----
        The transpose satisfies ``W.T[j, i] = W[i, j]``. Since both the
        connectivity pattern and the uniform weights are deterministic functions of
        ``seed``, the transposed matrix produces identical results to materializing
        ``W`` and transposing the dense array.

        The ``corder`` flag is flipped during transposition to maintain consistency
        with the underlying PRNG state ordering.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCUniformR, JITCUniformC
            >>>
            >>> row_matrix = JITCUniformR((0.1, 0.5, 0.2, 42), shape=(30, 5))
            >>> row_matrix.shape
            (30, 5)
            >>> col_matrix = row_matrix.transpose()
            >>> col_matrix.shape
            (5, 30)
            >>> isinstance(col_matrix, JITCUniformC)
            True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCUniformC(
            (self.wlow, self.whigh, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder,
            backend=self.backend,
        )

    def _new_mat(self, wlow, whigh, prob=None, seed=None):
        """
        Create a new ``JITCUniformR`` with the given weight bounds, reusing other attributes.

        Parameters
        ----------
        wlow : WeightScalar
            New lower bound for the uniform distribution.
        whigh : WeightScalar
            New upper bound for the uniform distribution.
        prob : Prob, optional
            New connection probability. If None, the current probability is reused.
        seed : Seed, optional
            New random seed. If None, the current seed is reused.

        Returns
        -------
        JITCUniformR
            A new row-oriented matrix with the specified weight bounds.
        """
        return JITCUniformR(
            (
                wlow,
                whigh,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder,
            backend=self.backend,
        )

    def _unitary_op(self, op) -> 'JITCUniformR':
        """
        Apply a unary operation to both weight bounds.

        Parameters
        ----------
        op : callable
            A unary function to apply element-wise to ``wlow`` and ``whigh``.

        Returns
        -------
        JITCUniformR
            A new matrix with the operation applied to both bounds.
        """
        return self._new_mat(op(self.wlow), op(self.whigh))

    def _binary_op(self, other, op) -> 'JITCUniformR':
        """
        Apply a binary operation between the weight bounds and a scalar operand.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The right-hand operand. Must be a scalar (size 1) or another sparse matrix.
        op : callable
            A binary function (e.g., ``operator.mul``) to apply.

        Returns
        -------
        JITCUniformR
            A new matrix with the operation applied to both bounds.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has a non-scalar shape.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.wlow, other), op(self.whigh, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCUniformR':
        """
        Apply a reflected binary operation with this matrix as the right operand.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The left-hand operand. Must be a scalar (size 1) or another sparse matrix.
        op : callable
            A binary function (e.g., ``operator.mul``) to apply.

        Returns
        -------
        JITCUniformR
            A new matrix with the operation applied to both bounds.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has a non-scalar shape.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.wlow), op(other, self.whigh))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
        """
        Compute matrix multiplication ``self @ other``.

        Dispatches to event-driven (binary) or float kernels depending on whether
        ``other`` is a ``BinaryArray``. Supports both matrix-vector and
        matrix-matrix products.

        Parameters
        ----------
        other : jax.Array, u.Quantity, or BinaryArray
            The right-hand operand. Can be a 1-D vector (matrix-vector product)
            or a 2-D matrix (matrix-matrix product).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The result of the multiplication. If either operand carries physical
            units, the result will be a ``u.Quantity`` with the appropriate
            combined unit.

        Raises
        ------
        NotImplementedError
            If ``other`` is another sparse matrix or has more than 2 dimensions.

        Notes
        -----
        For a matrix of shape ``(m, n)``:

        - 1-D ``other`` of length ``n`` produces a result of length ``m``.
        - 2-D ``other`` of shape ``(n, k)`` produces a result of shape ``(m, k)``.
        """
        # csr @ other
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                # JIT matrix @ events
                return binary_jitumv(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                # JIT matrix @ events
                return binary_jitumm(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            loc, other = u.math.promote_dtypes(self.wlow, other)
            scale, other = u.math.promote_dtypes(self.whigh, other)
            if other.ndim == 1:
                # JIT matrix @ vector
                return jitumv(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                # JIT matrix @ matrix
                return jitumm(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other) -> Union[jax.Array, u.Quantity]:
        """
        Compute matrix multiplication ``other @ self``.

        This is implemented by transposing the operation:
        ``other @ self == (self.T @ other.T).T`` for matrices, or
        ``other @ self == self.T @ other`` for vectors.

        Dispatches to event-driven (binary) or float kernels depending on whether
        ``other`` is a ``BinaryArray``. Supports both vector-matrix and
        matrix-matrix products.

        Parameters
        ----------
        other : jax.Array, u.Quantity, or BinaryArray
            The left-hand operand. Can be a 1-D vector (vector-matrix product)
            or a 2-D matrix (matrix-matrix product).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The result of the multiplication. If either operand carries physical
            units, the result will be a ``u.Quantity`` with the appropriate
            combined unit.

        Raises
        ------
        NotImplementedError
            If ``other`` is another sparse matrix or has more than 2 dimensions.

        Notes
        -----
        For a matrix of shape ``(m, n)``:

        - 1-D ``other`` of length ``m`` produces a result of length ``n``.
        - 2-D ``other`` of shape ``(k, m)`` produces a result of shape ``(k, n)``.
        """
        # other @ csr
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                #
                # vector @ JIT matrix
                # ==
                # JIT matrix.T @ vector
                #
                return binary_jitumv(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=True,
                    corder=not self.corder,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                #
                # matrix @ JIT matrix
                # ==
                # (JIT matrix.T @ matrix.T).T
                #
                r = binary_jitumm(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other.T,
                    self.seed,
                    shape=self.shape,
                    transpose=True,
                    corder=not self.corder,
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            loc, other = u.math.promote_dtypes(self.wlow, other)
            scale, other = u.math.promote_dtypes(self.whigh, other)
            if other.ndim == 1:
                #
                # vector @ JIT matrix
                # ==
                # JIT matrix.T @ vector
                #
                return jitumv(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=True,
                    corder=not self.corder,  # This is import to generate the same matrix as ``.todense()``
                    backend=self.backend,
                )
            elif other.ndim == 2:
                #
                # matrix @ JIT matrix
                # ==
                # (JIT matrix.T @ matrix.T).T
                #
                r = jitumm(
                    loc,
                    scale,
                    self.prob,
                    other.T,
                    self.seed,
                    shape=self.shape,
                    transpose=True,
                    corder=not self.corder,  # This is import to generate the same matrix as ``.todense()``
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")


@jax.tree_util.register_pytree_node_class
class JITCUniformC(JITUniformMatrix):
    """
    Just-In-Time Connectivity matrix with Column-oriented representation for uniform weight distributions.

    This class implements a column-oriented sparse matrix optimized for JAX-based transformations,
    following the Compressed Sparse Column (CSC) format conceptually. Instead of storing all non-zero
    elements explicitly, it uses a uniform distribution with lower and upper bounds (wlow, whigh)
    to generate weights for connections, along with probability and seed information to
    determine the sparse structure.

    The class is designed for efficient neural network connectivity patterns where weights
    follow a uniform distribution but connectivity is sparse and stochastic. The column-oriented
    structure makes column-based operations more efficient than row-based ones, making this class
    the transpose-oriented counterpart to JITCUniformR.

    Attributes
    ----------
    wlow : Union[jax.Array, u.Quantity]
        The lower bound of the uniform distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    whigh : Union[jax.Array, u.Quantity]
        The upper bound of the uniform distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    prob : Union[float, jax.Array]
        Connection probability determining the sparsity of the matrix.
        Values range from 0 (no connections) to 1 (fully connected).
    seed : Union[int, jax.Array]
        Random seed controlling the specific pattern of connections.
        Using the same seed produces identical connectivity patterns.
    shape : MatrixShape
        Tuple specifying the dimensions of the matrix as (rows, columns).
    corder : bool
        Flag indicating the memory layout order of the matrix.
        False (default) for Fortran-order (column-major), True for C-order (row-major).
    dtype
        The data type of the matrix elements (property inherited from parent).

    Examples
    --------

    .. code-block:: python

        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCUniformC

        # Create a uniform matrix with bounds [0.1, 0.5], probability 0.2, and seed 42
        >>> uniform_matrix = JITCUniformC((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>> uniform_matrix
        JITCUniformC(shape=(10, 10), wlow=0.1, whigh=0.5, prob=0.2, seed=42, corder=False)

        # Create a uniform matrix with units
        >>> uniform_matrix_mv = JITCUniformC((0.1 * u.mV, 0.5 * u.mV, 0.2, 42), shape=(10, 10))

        # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = uniform_matrix @ vec
        >>> # Each element in result is a weighted sum using uniformly distributed weights

        # Apply scalar operation (scales both lower and upper bounds)
        >>> scaled = uniform_matrix * 2.0
        >>> print(scaled.wlow, scaled.whigh)  # 0.2 1.0

        # Convert to dense representation
        >>> dense_matrix = uniform_matrix.todense()
        >>> # dense_matrix has shape (10, 10) with ~20% non-zero elements
        >>> # each non-zero element is uniformly distributed between 0.1 and 0.5

        # Transpose operation returns a JITCUniformR instance
        >>> row_matrix = uniform_matrix.transpose()
        >>> isinstance(row_matrix, JITCUniformR)  # True

        # Update bounds while preserving connectivity pattern
        >>> updated = uniform_matrix.with_data(0.2, 0.8)
        >>> print(updated.wlow, updated.whigh)  # 0.2 0.8

        # Use with JAX transformations
        >>> @jax.jit
        ... def matrix_vector_product(mat, vec):
        ...     return mat @ vec
        >>> result_jit = matrix_vector_product(uniform_matrix, vec)

        # Matrix-matrix multiplication
        >>> mat = jax.numpy.ones((10, 5))
        >>> result_mat = uniform_matrix @ mat
        >>> result_mat.shape  # (10, 5)

        # Right matrix multiplication
        >>> mat = jax.numpy.ones((5, 10))
        >>> result_rmat = mat @ uniform_matrix
        >>> result_rmat.shape  # (5, 10)

    Notes
    -----
    The mathematical model for ``JITCUniformC`` is:

        ``W[i, j] = Uniform(w_low, w_high) * Bernoulli(prob)``

    Each entry ``W[i, j]`` is independently drawn from the continuous uniform
    distribution on ``[w_low, w_high]`` with probability ``prob``, and zero
    otherwise. More precisely:

        ``W[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Uniform(w_low, w_high)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables, both determined by ``seed``.

    The column-oriented representation is the transpose dual of ``JITCUniformR``.
    Internally, operations on ``JITCUniformC`` are delegated to the transposed
    ``JITCUniformR`` form: ``JITCUniformC @ v == JITCUniformR.T @ v``.

    Key properties:

    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with uniformly distributed weights
    - The column-oriented structure makes column-slicing operations more efficient
    - Optimized for matrix-vector operations common in neural simulations
    - The actual matrix elements are never explicitly stored, only generated during operations
    - Using the same seed always produces the same random connectivity pattern and weights
    """
    __module__ = 'brainevent'

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the sparse column-oriented uniform matrix to a dense array.

        Generates a full dense representation of the sparse matrix by
        sampling ``Uniform(w_low, w_high)`` values for all connections
        determined by the probability and seed.

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A dense matrix with the same shape as the sparse matrix. The data type
            will match the weight's data type, and if the weight has units (is a
            ``u.Quantity``), the returned array will have the same units.

        Raises
        ------
        None
            This method does not raise exceptions under normal use.

        See Also
        --------
        JITCUniformR.todense : Row-oriented variant.
        jitu : Standalone function to materialize JIT uniform matrices.

        Notes
        -----
        The dense matrix is generated according to:

            ``dense[i, j] = Uniform(w_low, w_high) * Bernoulli(prob)``

        for each ``(i, j)`` pair, where the random draws are determined by ``seed``.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCUniformC
            >>>
            >>> mat = JITCUniformC((0.1, 0.5, 0.2, 42), shape=(3, 10))
            >>> dense = mat.todense()
            >>> dense.shape
            (3, 10)
        """
        return jitu(
            self.wlow,
            self.whigh,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
            backend=self.backend,
        )

    def transpose(self, axes=None) -> 'JITCUniformR':
        """
        Transpose the column-oriented matrix into a row-oriented matrix.

        Returns a row-oriented matrix (``JITCUniformR``) with rows and columns
        swapped, preserving the same weight bounds, probability, and seed values.

        Parameters
        ----------
        axes : None
            Not supported. This parameter exists for compatibility with the NumPy API
            but only ``None`` is accepted.

        Returns
        -------
        JITCUniformR
            A new row-oriented uniform matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``, since partial axis transposition is not supported.

        See Also
        --------
        JITCUniformR.transpose : The inverse operation.

        Notes
        -----
        The transpose satisfies ``W.T[j, i] = W[i, j]``. The ``corder`` flag is
        flipped during transposition to maintain consistency with the underlying
        PRNG state ordering.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCUniformC, JITCUniformR
            >>>
            >>> col_matrix = JITCUniformC((0.1, 0.5, 0.2, 42), shape=(3, 5))
            >>> col_matrix.shape
            (3, 5)
            >>> row_matrix = col_matrix.transpose()
            >>> row_matrix.shape
            (5, 3)
            >>> isinstance(row_matrix, JITCUniformR)
            True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCUniformR(
            (self.wlow, self.whigh, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder,
            backend=self.backend,
        )

    def _new_mat(self, wlow, whigh, prob=None, seed=None):
        """
        Create a new ``JITCUniformC`` with the given weight bounds, reusing other attributes.

        Parameters
        ----------
        wlow : WeightScalar
            New lower bound for the uniform distribution.
        whigh : WeightScalar
            New upper bound for the uniform distribution.
        prob : Prob, optional
            New connection probability. If None, the current probability is reused.
        seed : Seed, optional
            New random seed. If None, the current seed is reused.

        Returns
        -------
        JITCUniformC
            A new column-oriented matrix with the specified weight bounds.
        """
        return JITCUniformC(
            (
                wlow,
                whigh,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder,
            backend=self.backend,
        )

    def _unitary_op(self, op) -> 'JITCUniformC':
        """
        Apply a unary operation to both weight bounds.

        Parameters
        ----------
        op : callable
            A unary function to apply element-wise to ``wlow`` and ``whigh``.

        Returns
        -------
        JITCUniformC
            A new matrix with the operation applied to both bounds.
        """
        return self._new_mat(op(self.wlow), op(self.whigh))

    def _binary_op(self, other, op) -> 'JITCUniformC':
        """
        Apply a binary operation between the weight bounds and a scalar operand.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The right-hand operand. Must be a scalar (size 1) or another sparse matrix.
        op : callable
            A binary function (e.g., ``operator.mul``) to apply.

        Returns
        -------
        JITCUniformC
            A new matrix with the operation applied to both bounds.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has a non-scalar shape.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.wlow, other), op(self.whigh, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCUniformC':
        """
        Apply a reflected binary operation with this matrix as the right operand.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The left-hand operand. Must be a scalar (size 1) or another sparse matrix.
        op : callable
            A binary function (e.g., ``operator.mul``) to apply.

        Returns
        -------
        JITCUniformC
            A new matrix with the operation applied to both bounds.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has a non-scalar shape.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.wlow), op(other, self.whigh))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
        """
        Compute matrix multiplication ``self @ other``.

        Internally delegates to the underlying ``JITCUniformR`` representation
        by using a transposed view: ``JITCUniformC @ other == JITCUniformR.T @ other``.
        Dispatches to event-driven (binary) or float kernels depending on whether
        ``other`` is a ``BinaryArray``.

        Parameters
        ----------
        other : jax.Array, u.Quantity, or BinaryArray
            The right-hand operand. Can be a 1-D vector (matrix-vector product)
            or a 2-D matrix (matrix-matrix product).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The result of the multiplication. If either operand carries physical
            units, the result will be a ``u.Quantity`` with the appropriate
            combined unit.

        Raises
        ------
        NotImplementedError
            If ``other`` is another sparse matrix or has more than 2 dimensions.

        Notes
        -----
        For a matrix of shape ``(m, n)``:

        - 1-D ``other`` of length ``n`` produces a result of length ``m``.
        - 2-D ``other`` of shape ``(n, k)`` produces a result of shape ``(m, k)``.
        """
        # csr @ other
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                # JITC_R matrix.T @ vector
                # ==
                # vector @ JITC_R matrix
                return binary_jitumv(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=True,
                    corder=self.corder,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                # JITC_R matrix.T @ matrix
                # ==
                # (matrix.T @ JITC_R matrix).T
                return binary_jitumm(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=True,
                    corder=self.corder,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            loc, other = u.math.promote_dtypes(self.wlow, other)
            scale, other = u.math.promote_dtypes(self.whigh, other)
            if other.ndim == 1:
                # JITC_R matrix.T @ vector
                # ==
                # vector @ JITC_R matrix
                return jitumv(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=True,
                    corder=self.corder,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                # JITC_R matrix.T @ matrix
                # ==
                # (matrix.T @ JITC_R matrix).T
                return jitumm(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=True,
                    corder=self.corder,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other) -> Union[jax.Array, u.Quantity]:
        """
        Compute matrix multiplication ``other @ self``.

        Internally delegates to the underlying ``JITCUniformR`` representation:
        ``other @ JITCUniformC == other @ JITCUniformR.T == JITCUniformR @ other``
        for vectors, or ``(JITCUniformR @ other.T).T`` for matrices.
        Dispatches to event-driven (binary) or float kernels depending on whether
        ``other`` is a ``BinaryArray``.

        Parameters
        ----------
        other : jax.Array, u.Quantity, or BinaryArray
            The left-hand operand. Can be a 1-D vector (vector-matrix product)
            or a 2-D matrix (matrix-matrix product).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The result of the multiplication. If either operand carries physical
            units, the result will be a ``u.Quantity`` with the appropriate
            combined unit.

        Raises
        ------
        NotImplementedError
            If ``other`` is another sparse matrix or has more than 2 dimensions.

        Notes
        -----
        For a matrix of shape ``(m, n)``:

        - 1-D ``other`` of length ``m`` produces a result of length ``n``.
        - 2-D ``other`` of shape ``(k, m)`` produces a result of shape ``(k, n)``.
        """
        # other @ csr
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                #
                # vector @ JITC_R matrix.T
                # ==
                # JITC_R matrix @ vector
                #
                return binary_jitumv(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=False,
                    corder=not self.corder,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                #
                # matrix @ JITC_R matrix.T
                # ==
                # (JITC_R matrix @ matrix.T).T
                #
                r = binary_jitumm(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other.T,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=False,
                    corder=not self.corder,
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            loc, other = u.math.promote_dtypes(self.wlow, other)
            scale, other = u.math.promote_dtypes(self.whigh, other)
            if other.ndim == 1:
                #
                # vector @ JITC_R matrix.T
                # ==
                # JITC_R matrix @ vector
                #
                return jitumv(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=False,
                    corder=not self.corder,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                #
                # matrix @ JITC_R matrix.T
                # ==
                # (JITC_R matrix @ matrix.T).T
                #
                r = jitumm(
                    loc,
                    scale,
                    self.prob,
                    other.T,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=False,
                    corder=not self.corder,
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
