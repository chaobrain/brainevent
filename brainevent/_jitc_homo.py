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


import numbers
from typing import Union, Tuple

import brainunit as u
import jax
import numpy as np

from ._compatible_import import JAXSparse, Tracer
from ._event import EventArray
from ._jitc_base import JITCMatrix
from ._jitc_float_homo_impl import jitc_homo_matrix, jitc_homo_matvec, jitc_homo_matmat
from ._typing import MatrixShape

__all__ = [
    'JITCHomoR',
    'JITCHomoC',
]

Weight = Union[numbers.Number, np.ndarray, jax.Array, u.Quantity]
Prob = Union[float, np.ndarray, jax.Array]
Seed = Union[int, np.ndarray, jax.Array]


class JITHomoMatrix(JITCMatrix):
    weight: Union[jax.Array, u.Quantity]
    prob: Union[float, jax.Array]
    seed: Union[int, jax.Array]
    shape: MatrixShape
    corder: bool

    def __init__(
        self,
        data: Tuple[Weight, Prob, Seed],
        *,
        shape: MatrixShape,
        corder: bool = False,
    ):
        weight, self.prob, self.seed = data
        self.weight = u.math.asarray(weight)
        self.corder = corder
        super().__init__(data, shape=shape)

    def __repr__(self):
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
        return self.weight.dtype

    @property
    def data(self) -> Tuple[Weight, Prob, Seed]:
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

    def with_data(self, weight: Weight):
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
            JITCHomoR: Reconstructed JITHomo object

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


@jax.tree_util.register_pytree_node_class
class JITCHomoR(JITHomoMatrix):
    """
    Just-In-Time Connectivity Homogeneous matrix with Row-oriented representation.

    This class represents a row-oriented homogeneous sparse matrix optimized for JAX-based
    transformations. It follows the Compressed Sparse Row (CSR) format, storing a uniform value
    for all non-zero elements in the matrix, along with probability and seed information to
    determine the sparse structure.

    The class is designed for efficient neural network connectivity patterns where weights
    are homogeneous but connectivity is sparse and stochastic.

    Attributes
    ----------
    weight (Union[jax.Array, u.Quantity]): The homogeneous value used for all non-zero elements
    prob (Union[float, jax.Array]): Probability for each potential connection
    seed (Union[int, jax.Array]): Random seed used for initialization of the sparse structure
    shape (MatrixShape): The shape of the matrix as a tuple (rows, cols)
    dtype: The data type of the matrix elements (property inherited from parent)

    Examples
    --------

    >>> import jax
    >>> import brainunit as u
    >>> from brainevent import JITCHomoR

    # Create a homogeneous matrix with value 1.5, probability 0.1, and seed 42
    >>> homo_matrix = JITCHomoR((1.5, 0.1, 42), shape=(10, 10))
    >>> homo_matrix
    JITCHomoR(shape=(10, 10), dtype=float32, weight=1.5, prob=0.1, seed=42)

    >>> # Perform matrix-vector multiplication
    >>> vec = jax.numpy.ones(10)
    >>> result = homo_matrix @ vec

    >>> # Apply scalar operation
    >>> scaled = homo_matrix * 2.0
    >>>
    >>> # Convert to dense representation
    >>> dense_matrix = homo_matrix.todense()
    >>>
    >>> # Transpose operation returns a JITCHomo instance
    >>> col_matrix = homo_matrix.transpose()

    Notes
    -----
    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with uniform weights
    - Optimized for matrix-vector operations common in neural simulations
    """

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
        return jitc_homo_matrix(
            self.weight,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
        )

    def transpose(self, axes=None) -> 'JITCHomoC':
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
        JITCHomoC
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
        >>> isinstance(col_matrix, JITCHomoC)  # True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCHomoC(
            (self.weight, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder
        )

    def _new_mat(self, weight, prob=None, seed=None):
        return JITCHomoR(
            (
                weight,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder
        )

    def _unitary_op(self, op) -> 'JITCHomoR':
        return self._new_mat(op(self.weight), self.prob, self.seed)

    def _binary_op(self, other, op) -> 'JITCHomoR':
        if isinstance(other, JITCHomoR):
            self._check(other, op)
            return self._new_mat(op(self.weight, other.weight))

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.weight, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCHomoR':
        if isinstance(other, JITCHomoR):
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

        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                # JIT matrix @ events
                return jitc_homo_matvec(
                    weight,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                )
            elif other.ndim == 2:
                # JIT matrix @ events
                return jitc_homo_matmat(
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

        else:
            other = u.math.asarray(other)
            weight, other = u.math.promote_dtypes(self.weight, other)
            if other.ndim == 1:
                # JIT matrix @ vector
                return jitc_homo_matvec(
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
                return jitc_homo_matmat(
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

        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                #
                # vector @ JIT matrix
                # ==
                # JIT matrix.T @ vector
                #
                return jitc_homo_matvec(
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
                r = jitc_homo_matmat(
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
                return jitc_homo_matvec(
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
                r = jitc_homo_matmat(
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
class JITCHomoC(JITHomoMatrix):
    """
    Just-In-Time Connectivity Homogeneous matrix with Column-oriented representation.

    This class represents a column-oriented homogeneous sparse matrix optimized for JAX-based
    transformations. It follows the Compressed Sparse Column (CSC) format, storing a uniform value
    for all non-zero elements in the matrix, along with probability and seed information to
    determine the sparse structure.

    The column-oriented structure makes column-based operations more efficient than row-based ones,
    making this class the transpose-oriented counterpart to JITRHomo.

    Attributes:
        weight (Union[jax.Array, u.Quantity]): The homogeneous value used for all non-zero elements
        prob (Union[float, jax.Array]): Probability for each potential connection
        seed (Union[int, jax.Array]): Random seed used for initialization of the sparse structure
        shape (MatrixShape): The shape of the matrix as a tuple (rows, cols)
        dtype: The data type of the matrix elements (property inherited from parent)

    Examples:
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCHomoC
        >>>
        >>> # Create a homogeneous matrix with value 1.5, probability 0.1, and seed 42
        >>> homo_matrix = JITCHomoC((1.5, 0.1, 42), shape=(10, 10))
        >>>
        >>> # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = homo_matrix @ vec
        >>>
        >>> # Apply scalar operation
        >>> scaled = homo_matrix * 2.0
        >>>
        >>> # Transpose to get a row-oriented matrix
        >>> row_matrix = homo_matrix.transpose()
        >>>
        >>> # Convert to dense representation
        >>> dense_matrix = homo_matrix.todense()


    Notes:
        - Registered as a JAX pytree node for compatibility with JAX transformations (jit, grad, vmap)
        - More efficient than JITRHomo for column slicing operations
        - Compatible with all standard mathematical operations
        - Well-suited for neural network connectivity matrices with uniform weights
        - Optimized for neural simulations with sparse connectivity patterns
    """

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
        return jitc_homo_matrix(
            self.weight,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
        )

    def transpose(self, axes=None) -> 'JITCHomoR':
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
        JITCHomoR
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
        >>> isinstance(row_matrix, JITCHomoR)  # True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCHomoR(
            (self.weight, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder
        )

    def _new_mat(self, weight, prob=None, seed=None):
        return JITCHomoC(
            (
                weight,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder
        )

    def _unitary_op(self, op) -> 'JITCHomoC':
        return self._new_mat(op(self.weight))

    def _binary_op(self, other, op) -> 'JITCHomoC':
        if isinstance(other, JITCHomoC):
            self._check(other, op)
            return self._new_mat(op(self.weight, other.weight))

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.weight, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCHomoC':
        if isinstance(other, JITCHomoC):
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

        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                return _event_csr_matvec(
                    weight,
                    self.prob,
                    self.seed,
                    other,
                    shape=self.shape
                )
            elif other.ndim == 2:
                return _event_csr_matmat(
                    weight,
                    self.prob,
                    self.seed,
                    other,
                    shape=self.shape
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
                return jitc_homo_matvec(
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
                return jitc_homo_matmat(
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

        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                return _event_csr_matvec(
                    weight,
                    self.prob,
                    self.seed,
                    other,
                    shape=self.shape,
                    transpose=True
                )
            elif other.ndim == 2:
                other = other.T
                r = _event_csr_matmat(
                    weight,
                    self.prob,
                    self.seed,
                    other,
                    shape=self.shape,
                    transpose=True
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
                return jitc_homo_matvec(
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
                r = jitc_homo_matmat(
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
