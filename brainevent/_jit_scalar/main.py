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
import numpy as np

from brainevent._compatible_import import JAXSparse, Tracer
from brainevent._event.binary import BinaryArray
from brainevent._jitc_matrix import JITCMatrix
from brainevent._typing import MatrixShape, WeightScalar, Prob, Seed
from .binary import (
    binary_jitsmv,
    binary_jitsmm,
)
from .float import (
    jits,
    jitsmv,
    jitsmm,
)

__all__ = [
    'JITCScalarR',
    'JITCScalarC',
]


class JITScalarMatrix(JITCMatrix):
    """
    Base class for Just-In-Time Connectivity Homogeneous matrices.

    This abstract class serves as the foundation for sparse matrix representations
    that use homogeneous weights with stochastic connectivity patterns. It stores
    a single weight value applied to all non-zero elements, along with connectivity
    probability and a random seed that determines the sparse structure.

    Designed for efficient representation of neural connectivity matrices where all
    connections have the same strength (weight) but are sparsely distributed.

    Attributes
    ----------
    weight : Union[jax.Array, u.Quantity]
        The homogeneous weight value applied to all non-zero elements in the matrix.
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

    weight: Union[jax.Array, u.Quantity]
    prob: Union[float, jax.Array]
    seed: Union[int, jax.Array]
    shape: MatrixShape
    corder: bool

    def __init__(
        self,
        data: Tuple[WeightScalar, Prob, Seed],
        *,
        shape: MatrixShape,
        corder: bool = False,
    ):
        """
        Initialize a homogeneous sparse just-in-time connectivity matrix.

        Parameters
        ----------
        data : Tuple[WeightScalar, Prob, Seed]
            A tuple containing three elements:
            - weight: Homogeneous weight value for all non-zero elements
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
        as instance attributes. The weight is converted to a JAX array if it's not
        already one, preserving any attached units.
        """
        weight, self.prob, self.seed = data
        if not isinstance(self.prob, Tracer):
            prob = np.asarray(self.prob)
            if prob.size != 1:
                raise ValueError(f"prob must be a scalar, but got shape {prob.shape}.")
            prob = float(prob.item())
            if not np.isfinite(prob):
                raise ValueError(f"prob must be finite, but got {prob}.")
            if not (0. <= prob <= 1.):
                raise ValueError(f"prob must be in [0, 1], but got {prob}.")
        self.weight = u.math.asarray(weight)
        self.corder = corder
        super().__init__(data, shape=shape)

    def __repr__(self):
        """
        Return a string representation of the homogeneous matrix.

        Returns
        -------
        str
            A string showing the class name, shape, weight value, probability,
            seed, and corder flag of the matrix instance.

        Examples
        --------
        >>> matrix = JITScalarMatrix((0.5, 0.1, 42), shape=(10, 10))
        >>> repr(matrix)
        'JITHomoMatrix(shape=(10, 10), weight=0.5, prob=0.1, seed=42, corder=False)'
        """
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"weight={self.weight}, "
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
            The data type of the weight values in the matrix.

        Notes
        -----
        This property inherits the dtype directly from the weight attribute,
        ensuring consistent data typing throughout operations involving this matrix.
        """
        return self.weight.dtype

    @property
    def data(self) -> Tuple[WeightScalar, Prob, Seed]:
        """
        Returns the core data components of the homogeneous matrix.

        This property provides access to the three fundamental components that define
        the sparse matrix: weight values, connection probabilities, and the random seed.
        It's used by the tree_flatten method to make the class compatible with JAX
        transformations.

        Returns
        -------
        Tuple[Weight, Prob, Seed]
            A tuple containing:
            - weight: The homogeneous weight value for non-zero elements
            - prob: Connection probability for the sparse structure
            - seed: Random seed used for generating the sparse connectivity pattern
        """
        return self.weight, self.prob, self.seed

    def with_data(self, weight: WeightScalar):
        """
        Create a new matrix instance with updated weight data but preserving other properties.

        This method returns a new instance of the same class with the provided weight value,
        while keeping the same probability, seed, shape, and other configuration parameters.
        It's useful for updating weights without changing the connectivity pattern.

        Parameters
        ----------
        weight : Weight
            The new weight value to use. Must have the same shape and unit as the current weight.

        Raises
        ------
        AssertionError
            If the provided weight has a different shape or unit than the current weight.
        """
        weight = u.math.asarray(weight)
        assert weight.shape == self.weight.shape
        assert u.get_unit(weight) == u.get_unit(self.weight)
        return type(self)(
            (weight, self.prob, self.seed),
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
        return (self.weight, self.prob, self.seed), {"shape": self.shape, 'corder': self.corder}

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
            JITCScalarR: Reconstructed JITHomo object

        Raises:
            ValueError: If the aux_data dictionary doesn't contain the expected keys
        """
        obj = object.__new__(cls)
        obj.weight, obj.prob, obj.seed = children
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
                    f"binary operation {op} between two {self.__class__.__name__} objects with different seeds "
                    f"is not implemented currently."
                )
        else:
            raise NotImplementedError(
                f"binary operation {op} between two {self.__class__.__name__} objects with tracing seeds "
                f"is not implemented currently."
            )
        if self.corder != other.corder:
            raise NotImplementedError(
                f"binary operation {op} between two {self.__class__.__name__} objects with different corder "
                f"is not implemented currently."
            )
        if self.prob != other.prob:
            raise NotImplementedError(
                f"binary operation {op} between two {self.__class__.__name__} objects "
                f"with different prob is not supported."
            )
        if self.shape != other.shape:
            raise NotImplementedError(
                f"binary operation {op} between two {self.__class__.__name__} objects "
                f"with different shapes is not supported."
            )


@jax.tree_util.register_pytree_node_class
class JITCScalarR(JITScalarMatrix):
    """
    Just-In-Time Connectivity Homogeneous matrix with Row-oriented representation.

    This class represents a row-oriented homogeneous sparse matrix optimized for JAX-based
    transformations. It follows the Compressed Sparse Row (CSR) format conceptually, storing
    a uniform weight value for all non-zero elements in the matrix, along with probability
    and seed information to determine the sparse structure.

    The class is designed for efficient neural network connectivity patterns where weights
    are homogeneous (identical) but connectivity is sparse and stochastically determined.
    The row-oriented structure makes row-based operations more efficient than column-based ones.

    Attributes
    ----------
    weight : Union[jax.Array, u.Quantity]
        The homogeneous value used for all non-zero elements in the matrix.
        Can be a plain JAX array or a quantity with units.
    prob : Union[float, jax.Array]
        Probability for each potential connection. Controls the sparsity level
        with 0.0 meaning no connections and 1.0 meaning all possible connections.
    seed : Union[int, jax.Array]
        Random seed used for initialization of the sparse structure.
        Using the same seed produces identical connectivity patterns.
    shape : MatrixShape
        The shape of the matrix as a tuple (rows, cols).
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
        >>> from brainevent import JITCScalarR

        # Create a homogeneous matrix with value 1.5, probability 0.1, and seed 42
        >>> homo_matrix = JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
        >>> homo_matrix
        JITCHomoR(shape=(10, 10), weight=1.5, prob=0.1, seed=42, corder=False)

        # Create a matrix with units
        >>> weighted_matrix = JITCScalarR((1.5 * u.mV, 0.1, 42), shape=(10, 10))
        >>> weighted_matrix
        JITCHomoR(shape=(10, 10), weight=1.5 mV, prob=0.1, seed=42, corder=False)

        # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = homo_matrix @ vec
        >>> result.shape  # (10,)

        # Apply scalar operations
        >>> scaled = homo_matrix * 2.0
        >>> scaled.weight  # 3.0

        # Arithmetic operations maintain the sparse structure
        >>> neg_matrix = -homo_matrix
        >>> neg_matrix.weight  # -1.5

        # Convert to dense representation
        >>> dense_matrix = homo_matrix.todense()
        >>> dense_matrix.shape  # (10, 10)

        # Transpose operation returns a column-oriented matrix
        >>> col_matrix = homo_matrix.transpose()
        >>> isinstance(col_matrix, JITCScalarC)  # True
        >>> col_matrix.shape  # (10, 10)

    Notes
    -----
    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with uniform weights
    - Optimized for matrix-vector operations common in neural simulations
    - The matrix is implicitly constructed based on the probability and seed;
      the actual sparse structure is materialized only when needed
    - When used with units (e.g., u.mV), units are preserved through operations
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
        >>> from brainevent import JITCScalarR
        >>>
        >>> # Create a sparse homogeneous matrix
        >>> sparse_matrix = JITCScalarR((1.5 * u.mV, 0.5, 42), shape=(10, 4))
        >>>
        >>> # Convert to dense format
        >>> dense_matrix = sparse_matrix.todense()
        >>> print(dense_matrix.shape)  # (10, 4)
        """
        return jits(
            self.weight,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
        )

    def transpose(self, axes=None) -> 'JITCScalarC':
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
        JITCScalarC
            A new column-oriented homogeneous matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If axes is not None, since partial axis transposition is not supported.

        Examples
        --------
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCScalarR
        >>>
        >>> # Create a row-oriented matrix
        >>> row_matrix = JITCScalarR((1.5, 0.5, 42), shape=(30, 5))
        >>> print(row_matrix.shape)  # (30, 5)
        >>>
        >>> # Transpose to column-oriented matrix
        >>> col_matrix = row_matrix.transpose()
        >>> print(col_matrix.shape)  # (5, 30)
        >>> isinstance(col_matrix, JITCScalarC)  # True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCScalarC(
            (self.weight, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder
        )

    def _new_mat(self, weight, prob=None, seed=None):
        return JITCScalarR(
            (
                weight,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder
        )

    def _unitary_op(self, op) -> 'JITCScalarR':
        return self._new_mat(op(self.weight), self.prob, self.seed)

    def _binary_op(self, other, op) -> 'JITCScalarR':
        if isinstance(other, JITCScalarR):
            self._check(other, op)
            return self._new_mat(op(self.weight, other.weight))

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.weight, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCScalarR':
        if isinstance(other, JITCScalarR):
            self._check(other, op)
            return self._new_mat(op(other.weight, self.weight))

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.weight))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
        # csr @ other
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        weight = self.weight

        if isinstance(other, BinaryArray):
            other = other.data
            if other.ndim == 1:
                # JIT matrix @ events
                return binary_jitsmv(weight, self.prob, other, self.seed, shape=self.shape,
                                     transpose=False, corder=self.corder, )
            elif other.ndim == 2:
                # JIT matrix @ events
                return binary_jitsmm(weight, self.prob, other, self.seed, shape=self.shape,
                                     transpose=False, corder=self.corder, )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            weight, other = u.math.promote_dtypes(self.weight, other)
            if other.ndim == 1:
                # JIT matrix @ vector
                return jitsmv(
                    weight,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                )
            elif other.ndim == 2:
                # JIT matrix @ matrix
                return jitsmm(
                    weight,
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
        weight = self.weight

        if isinstance(other, BinaryArray):
            other = other.data
            if other.ndim == 1:
                #
                # vector @ JIT matrix
                # ==
                # JIT matrix.T @ vector
                #
                return binary_jitsmv(
                    weight,
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
                r = binary_jitsmm(
                    weight,
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
            weight, other = u.math.promote_dtypes(self.weight, other)
            if other.ndim == 1:
                #
                # vector @ JIT matrix
                # ==
                # JIT matrix.T @ vector
                #
                return jitsmv(
                    weight,
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
                r = jitsmm(
                    weight,
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
class JITCScalarC(JITScalarMatrix):
    """
    Just-In-Time Connectivity Homogeneous matrix with Column-oriented representation.

    This class represents a column-oriented homogeneous sparse matrix optimized for JAX-based
    transformations. It follows the Compressed Sparse Column (CSC) format conceptually, storing
    a uniform weight value for all non-zero elements in the matrix, along with probability
    and seed information to determine the sparse structure.

    The column-oriented structure makes column-based operations more efficient than row-based
    ones, making this class the transpose-oriented counterpart to JITCHomoR.

    Attributes
    ----------
    weight : Union[jax.Array, u.Quantity]
        The homogeneous value used for all non-zero elements in the matrix.
        Can be a plain JAX array or a quantity with units.
    prob : Union[float, jax.Array]
        Probability for each potential connection. Controls the sparsity level
        with 0.0 meaning no connections and 1.0 meaning all possible connections.
    seed : Union[int, jax.Array]
        Random seed used for initialization of the sparse structure.
        Using the same seed produces identical connectivity patterns.
    shape : MatrixShape
        The shape of the matrix as a tuple (rows, cols).
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
        >>> from brainevent import JITCScalarC

        # Create a homogeneous matrix with value 1.5, probability 0.1, and seed 42
        >>> homo_matrix = JITCScalarC((1.5, 0.1, 42), shape=(10, 10))
        >>> homo_matrix
        JITCHomoC(shape=(10, 10), weight=1.5, prob=0.1, seed=42, corder=False)

        # Create a matrix with units
        >>> weighted_matrix = JITCScalarC((1.5 * u.mV, 0.1, 42), shape=(10, 10))
        >>> weighted_matrix
        JITCHomoC(shape=(10, 10), weight=1.5 mV, prob=0.1, seed=42, corder=False)

        # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = homo_matrix @ vec
        >>> result.shape  # (10,)

        # Apply scalar operations
        >>> scaled = homo_matrix * 2.0
        >>> scaled.weight  # 3.0

        # Arithmetic operations maintain the sparse structure
        >>> neg_matrix = -homo_matrix
        >>> neg_matrix.weight  # -1.5

        # Convert to dense representation
        >>> dense_matrix = homo_matrix.todense()
        >>> dense_matrix.shape  # (10, 10)

        # Transpose operation returns a row-oriented matrix
        >>> row_matrix = homo_matrix.transpose()
        >>> isinstance(row_matrix, JITCScalarR)  # True
        >>> row_matrix.shape  # (10, 10)

    Notes
    -----
    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - More efficient than JITCHomoR for column-based operations
    - Well-suited for neural network connectivity matrices with uniform weights
    - Optimized for matrix-vector operations common in neural simulations
    - The matrix is implicitly constructed based on the probability and seed;
      the actual sparse structure is materialized only when needed
    - When used with units (e.g., u.mV), units are preserved through operations
    """
    __module__ = 'brainevent'

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Converts the sparse column-oriented homogeneous matrix to dense format.

        This method generates a full dense representation of the sparse matrix by
        using the homogeneous weight value for all connections determined by the
        probability and seed. The generated dense matrix always has ``self.shape``.

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
        >>> from brainevent import JITCScalarC
        >>>
        >>> # Create a sparse column-oriented homogeneous matrix
        >>> sparse_matrix = JITCScalarC((1.5 * u.mV, 0.5, 42), shape=(3, 10))
        >>>
        >>> # Convert to dense format
        >>> dense_matrix = sparse_matrix.todense()
        >>> print(dense_matrix.shape)  # (3, 10)
        """
        return jits(
            self.weight,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
        )

    def transpose(self, axes=None) -> 'JITCScalarR':
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
        JITCScalarR
            A new row-oriented homogeneous matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If axes is not None, since partial axis transposition is not supported.

        Examples
        --------
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCScalarC
        >>>
        >>> # Create a column-oriented matrix
        >>> col_matrix = JITCScalarC((1.5, 0.5, 42), shape=(3, 5))
        >>> print(col_matrix.shape)  # (3, 5)
        >>>
        >>> # Transpose to row-oriented matrix
        >>> row_matrix = col_matrix.transpose()
        >>> print(row_matrix.shape)  # (5, 3)
        >>> isinstance(row_matrix, JITCScalarR)  # True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCScalarR(
            (self.weight, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder
        )

    def _new_mat(self, weight, prob=None, seed=None):
        return JITCScalarC(
            (
                weight,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder
        )

    def _unitary_op(self, op) -> 'JITCScalarC':
        return self._new_mat(op(self.weight))

    def _binary_op(self, other, op) -> 'JITCScalarC':
        if isinstance(other, JITCScalarC):
            self._check(other, op)
            return self._new_mat(op(self.weight, other.weight))

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.weight, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCScalarC':
        if isinstance(other, JITCScalarC):
            self._check(other, op)
            return self._new_mat(op(other.weight, self.weight))

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.weight))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
        # csr @ other
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        weight = self.weight

        if isinstance(other, BinaryArray):
            other = other.data
            if other.ndim == 1:
                # JITC_R matrix.T @ vector
                # ==
                # vector @ JITC_R matrix
                return binary_jitsmv(
                    weight,
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
                return binary_jitsmm(
                    weight,
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
            weight, other = u.math.promote_dtypes(self.weight, other)
            if other.ndim == 1:
                # JITC_R matrix.T @ vector
                # ==
                # vector @ JITC_R matrix
                return jitsmv(
                    weight,
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
                return jitsmm(
                    weight,
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
        weight = self.weight

        if isinstance(other, BinaryArray):
            other = other.data
            if other.ndim == 1:
                #
                # vector @ JITC_R matrix.T
                # ==
                # JITC_R matrix @ vector
                #
                return binary_jitsmv(
                    weight,
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
                r = binary_jitsmm(
                    weight,
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
            weight, other = u.math.promote_dtypes(self.weight, other)
            if other.ndim == 1:
                #
                # vector @ JITC_R matrix.T
                # ==
                # JITC_R matrix @ vector
                #
                return jitsmv(
                    weight,
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
                r = jitsmm(
                    weight,
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
