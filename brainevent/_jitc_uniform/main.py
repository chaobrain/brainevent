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


from typing import Union, Tuple

import brainunit as u
import jax

from brainevent._compatible_import import JAXSparse, Tracer
from brainevent._event.binary import EventArray
from brainevent._jitc_matrix import JITCMatrix
from brainevent._typing import MatrixShape, WeightScalar, Prob, Seed
from .binary import (
    binary_jitc_uniform_matvec,
    binary_jitc_uniform_matmat,
)
from .float import (
    float_jitc_uniform_matrix,
    float_jitc_uniform_matvec,
    float_jitc_uniform_matmat,
)

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
        data: Tuple[WeightScalar, WeightScalar, Prob, Seed],
        *,
        shape: MatrixShape,
        corder: bool = False,
    ):
        """
        Initialize a uniform distribution sparse matrix.

        Parameters
        ----------
        data : Tuple[WeightScalar, WeightScalar, Prob, Seed]
            A tuple containing four elements:
            - low: Lower bound of the uniform distribution
            - high: Upper bound of the uniform distribution
            - prob: Connection probability determining matrix sparsity
            - seed: Random seed for reproducible sparse structure generation
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
        low, high, self.prob, self.seed = data
        low, high = u.math.promote_dtypes(low, high)
        u.fail_for_dimension_mismatch(low, high, "loc and scale must have the same dimension.")
        self.wlow = u.math.asarray(low)
        self.whigh = u.math.asarray(high)
        self.corder = corder
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
            f"corder={self.corder})"
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
            corder=self.corder
        )

    def tree_flatten(self):
        """
        Flattens the JITHomo object for JAX transformation compatibility.

        This method is part of JAX's pytree protocol that enables JAX transformations
        on custom classes. It separates the object into arrays that should be traced
        through JAX transformations (children) and auxiliary static data.

        Returns:
            tuple: A tuple with two elements:
                - A tuple of JAX-traceable arrays (only self.data in this case)
                - A dictionary of auxiliary data (shape, indices, and indptr)
        """
        return (self.wlow, self.whigh, self.prob, self.seed), {"shape": self.shape, 'corder': self.corder}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a JITHomo object from flattened data.

        This method is part of JAX's pytree protocol that enables JAX transformations
        on custom classes. It rebuilds the JITHomo object from the flattened representation
        produced by tree_flatten.

        Args:
            aux_data (dict): Dictionary containing auxiliary static data (shape, indices, indptr)
            children (tuple): Tuple of JAX arrays that were transformed (contains only data)

        Returns:
            JITCUniformR: Reconstructed JITHomo object

        Raises:
            ValueError: If the aux_data dictionary doesn't contain the expected keys
        """
        obj = object.__new__(cls)
        obj.wlow, obj.whigh, obj.prob, obj.seed = children
        if aux_data.keys() != {'shape', 'corder'}:
            raise ValueError(
                "aux_data must contain 'shape', 'corder' keys. "
                f"But got: {aux_data.keys()}"
            )
        obj.__dict__.update(**aux_data)
        return obj

    def _check(self, other, op):
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
        Converts the sparse homogeneous matrix to dense format.

        This method generates a full dense representation of the sparse matrix by
        using the homogeneous weight value for all connections determined by the
        probability and seed. The resulting dense matrix preserves all the numerical
        properties of the sparse representation.

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A dense matrix with the same shape as the sparse matrix. The data type
            will match the weight's data type, and if the weight has units (is a
            u.Quantity), the returned array will have the same units.

        Examples
        --------
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCHomoR
        >>>
        >>> # Create a sparse homogeneous matrix
        >>> sparse_matrix = JITCHomoR((1.5 * u.mV, 0.5, 42), shape=(10, 4))
        >>>
        >>> # Convert to dense format
        >>> dense_matrix = sparse_matrix.todense()
        >>> print(dense_matrix.shape)  # (10, 4)
        """
        return float_jitc_uniform_matrix(
            self.wlow,
            self.whigh,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
        )

    def transpose(self, axes=None) -> 'JITCUniformC':
        """
        Transposes the row-oriented matrix into a column-oriented matrix.

        This method returns a column-oriented matrix (JITCHomoC) with rows and columns
        swapped, preserving the same weight, probability, and seed values.
        The transpose operation effectively converts between row-oriented and
        column-oriented sparse matrix formats.

        Parameters
        ----------
        axes : None
            Not supported. This parameter exists for compatibility with the NumPy API
            but only None is accepted.

        Returns
        -------
        JITCUniformC
            A new column-oriented homogeneous matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If axes is not None, since partial axis transposition is not supported.

        Examples
        --------
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCHomoR
        >>>
        >>> # Create a row-oriented matrix
        >>> row_matrix = JITCHomoR((1.5, 0.5, 42), shape=(30, 5))
        >>> print(row_matrix.shape)  # (30, 5)
        >>>
        >>> # Transpose to column-oriented matrix
        >>> col_matrix = row_matrix.transpose()
        >>> print(col_matrix.shape)  # (5, 30)
        >>> isinstance(col_matrix, JITCUniformC)  # True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCUniformC(
            (self.wlow, self.whigh, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder
        )

    def _new_mat(self, wlow, whigh, prob=None, seed=None):
        return JITCUniformR(
            (
                wlow,
                whigh,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder
        )

    def _unitary_op(self, op) -> 'JITCUniformR':
        return self._new_mat(op(self.wlow), op(self.whigh))

    def _binary_op(self, other, op) -> 'JITCUniformR':
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.wlow, other), op(self.whigh, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCUniformR':
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.wlow), op(other, self.whigh))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
        # csr @ other
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                # JIT matrix @ events
                return binary_jitc_uniform_matvec(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                )
            elif other.ndim == 2:
                # JIT matrix @ events
                return binary_jitc_uniform_matmat(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            loc, other = u.math.promote_dtypes(self.wlow, other)
            scale, other = u.math.promote_dtypes(self.whigh, other)
            if other.ndim == 1:
                # JIT matrix @ vector
                return float_jitc_uniform_matvec(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                )
            elif other.ndim == 2:
                # JIT matrix @ matrix
                return float_jitc_uniform_matmat(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other) -> Union[jax.Array, u.Quantity]:
        # other @ csr
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                #
                # vector @ JIT matrix
                # ==
                # JIT matrix.T @ vector
                #
                return binary_jitc_uniform_matvec(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=True,
                    corder=not self.corder,
                )
            elif other.ndim == 2:
                #
                # matrix @ JIT matrix
                # ==
                # (JIT matrix.T @ matrix.T).T
                #
                r = binary_jitc_uniform_matmat(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other.T,
                    self.seed,
                    shape=self.shape,
                    transpose=True,
                    corder=not self.corder,
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
                return float_jitc_uniform_matvec(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=True,
                    corder=not self.corder,  # This is import to generate the same matrix as ``.todense()``
                )
            elif other.ndim == 2:
                #
                # matrix @ JIT matrix
                # ==
                # (JIT matrix.T @ matrix.T).T
                #
                r = float_jitc_uniform_matmat(
                    loc,
                    scale,
                    self.prob,
                    other.T,
                    self.seed,
                    shape=self.shape,
                    transpose=True,
                    corder=not self.corder,  # This is import to generate the same matrix as ``.todense()``
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
        Converts the sparse column-oriented homogeneous matrix to dense format.

        This method generates a full dense representation of the sparse matrix by
        using the homogeneous weight value for all connections determined by the
        probability and seed. Since this is a column-oriented matrix (JITCHomoC),
        the transpose flag is set to True to ensure proper conversion.

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A dense matrix with the same shape as the sparse matrix. The data type
            will match the weight's data type, and if the weight has units (is a
            u.Quantity), the returned array will have the same units.

        Examples
        --------
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCHomoC
        >>>
        >>> # Create a sparse column-oriented homogeneous matrix
        >>> sparse_matrix = JITCHomoC((1.5 * u.mV, 0.5, 42), shape=(3, 10))
        >>>
        >>> # Convert to dense format
        >>> dense_matrix = sparse_matrix.todense()
        >>> print(dense_matrix.shape)  # (3, 10)
        """
        return float_jitc_uniform_matrix(
            self.wlow,
            self.whigh,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
        )

    def transpose(self, axes=None) -> 'JITCUniformR':
        """
        Transposes the column-oriented matrix into a row-oriented matrix.

        This method returns a row-oriented matrix (JITCHomoR) with rows and columns
        swapped, preserving the same weight, probability, and seed values.
        The transpose operation effectively converts between column-oriented and
        row-oriented sparse matrix formats.

        Parameters
        ----------
        axes : None
            Not supported. This parameter exists for compatibility with the NumPy API
            but only None is accepted.

        Returns
        -------
        JITCUniformR
            A new row-oriented homogeneous matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If axes is not None, since partial axis transposition is not supported.

        Examples
        --------
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCHomoC
        >>>
        >>> # Create a column-oriented matrix
        >>> col_matrix = JITCHomoC((1.5, 0.5, 42), shape=(3, 5))
        >>> print(col_matrix.shape)  # (3, 5)
        >>>
        >>> # Transpose to row-oriented matrix
        >>> row_matrix = col_matrix.transpose()
        >>> print(row_matrix.shape)  # (5, 3)
        >>> isinstance(row_matrix, JITCUniformR)  # True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCUniformR(
            (self.wlow, self.whigh, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder
        )

    def _new_mat(self, wlow, whigh, prob=None, seed=None):
        return JITCUniformC(
            (
                wlow,
                whigh,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder
        )

    def _unitary_op(self, op) -> 'JITCUniformC':
        return self._new_mat(op(self.wlow), op(self.whigh))

    def _binary_op(self, other, op) -> 'JITCUniformC':
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.wlow, other), op(self.whigh, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCUniformC':
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.wlow), op(other, self.whigh))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
        # csr @ other
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                # JITC_R matrix.T @ vector
                # ==
                # vector @ JITC_R matrix
                return binary_jitc_uniform_matvec(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=True,
                    corder=self.corder,
                )
            elif other.ndim == 2:
                # JITC_R matrix.T @ matrix
                # ==
                # (matrix.T @ JITC_R matrix).T
                return binary_jitc_uniform_matmat(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=True,
                    corder=self.corder,
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
                return float_jitc_uniform_matvec(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=True,
                    corder=self.corder,
                )
            elif other.ndim == 2:
                # JITC_R matrix.T @ matrix
                # ==
                # (matrix.T @ JITC_R matrix).T
                return float_jitc_uniform_matmat(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=True,
                    corder=self.corder,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other) -> Union[jax.Array, u.Quantity]:
        # other @ csr
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                #
                # vector @ JITC_R matrix.T
                # ==
                # JITC_R matrix @ vector
                #
                return binary_jitc_uniform_matvec(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=False,
                    corder=not self.corder,
                )
            elif other.ndim == 2:
                #
                # matrix @ JITC_R matrix.T
                # ==
                # (JITC_R matrix @ matrix.T).T
                #
                r = binary_jitc_uniform_matmat(
                    self.wlow,
                    self.whigh,
                    self.prob,
                    other.T,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=False,
                    corder=not self.corder,
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
                return float_jitc_uniform_matvec(
                    loc,
                    scale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=False,
                    corder=not self.corder,
                )
            elif other.ndim == 2:
                #
                # matrix @ JITC_R matrix.T
                # ==
                # (JITC_R matrix @ matrix.T).T
                #
                r = float_jitc_uniform_matmat(
                    loc,
                    scale,
                    self.prob,
                    other.T,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=False,
                    corder=not self.corder,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
