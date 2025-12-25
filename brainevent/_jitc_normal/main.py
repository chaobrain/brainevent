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
    binary_jitc_normal_matvec,
    binary_jitc_normal_matmat,
)
from .float import (
    float_jitc_normal_matrix,
    float_jitc_normal_matvec,
    float_jitc_normal_matmat,
)

__all__ = [
    'JITCNormalR',
    'JITCNormalC',
]


class JITNormalMatrix(JITCMatrix):
    """
    Base class for Just-In-Time Connectivity Normal Distribution matrices.

    This abstract class serves as the foundation for sparse matrix representations
    that use normally distributed weights with stochastic connectivity patterns.
    It stores location (mean) and scale (standard deviation) parameters for the
    normal distribution, along with connectivity probability and a random seed
    that determines the sparse structure.

    Designed for efficient representation of neural connectivity matrices where
    connections follow a normal distribution but are sparsely distributed.

    Attributes
    ----------
    wloc : Union[jax.Array, u.Quantity]
        The location (mean) parameter of the normal distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    wscale : Union[jax.Array, u.Quantity]
        The scale (standard deviation) parameter of the normal distribution for non-zero elements.
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

    wloc: Union[jax.Array, u.Quantity]
    wscale: Union[jax.Array, u.Quantity]
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
        Initialize a normal distribution sparse matrix.

        Parameters
        ----------
        data : Tuple[WeightScalar, WeightScalar, Prob, Seed]
            A tuple containing four elements:
            - loc: Location (mean) parameter of the normal distribution
            - scale: Scale (standard deviation) parameter of the normal distribution
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
        loc, scale, self.prob, self.seed = data
        loc, scale = u.math.promote_dtypes(loc, scale)
        u.fail_for_dimension_mismatch(loc, scale, "loc and scale must have the same dimension.")
        self.wloc = u.math.asarray(loc)
        self.wscale = u.math.asarray(scale)
        self.corder = corder
        super().__init__(data, shape=shape)

    def __repr__(self):
        """
        Return a string representation of the normal distribution matrix.

        Returns
        -------
        str
            A string showing the class name, shape, location (mean), scale (standard deviation),
            probability, seed, and corder flag of the matrix instance.

        Examples
        --------
        >>> matrix = JITNormalMatrix((0.5, 0.1, 0.2, 42), shape=(10, 10))
        >>> repr(matrix)
        'JITNormalMatrix(shape=(10, 10), wloc=0.5, wscale=0.1, prob=0.2, seed=42, corder=False)'
        """
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"wloc={self.wloc}, "
            f"wscale={self.wscale}, "
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
            The data type of the location (mean) values in the matrix.

        Notes
        -----
        This property inherits the dtype directly from the wloc attribute,
        ensuring consistent data typing throughout operations involving this matrix.
        """
        return self.wloc.dtype

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
        return self.wloc, self.wscale, self.prob, self.seed

    def with_data(self, loc: WeightScalar, scale: WeightScalar):
        """
        Create a new matrix instance with updated weight data but preserving other properties.

        This method returns a new instance of the same class with the provided weight value,
        while keeping the same probability, seed, shape, and other configuration parameters.
        It's useful for updating weights without changing the connectivity pattern.

        """
        loc = u.math.asarray(loc)
        scale = u.math.asarray(scale)
        assert loc.shape == self.wloc.shape
        assert scale.shape == self.wscale.shape
        assert u.get_unit(loc) == u.get_unit(self.wloc)
        assert u.get_unit(scale) == u.get_unit(self.wscale)
        return type(self)(
            (loc, scale, self.prob, self.seed),
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
        return (self.wloc, self.wscale, self.prob, self.seed), {"shape": self.shape, 'corder': self.corder}

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
            JITCNormalR: Reconstructed JITHomo object

        Raises:
            ValueError: If the aux_data dictionary doesn't contain the expected keys
        """
        obj = object.__new__(cls)
        obj.wloc, obj.wscale, obj.prob, obj.seed = children
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
class JITCNormalR(JITNormalMatrix):
    """
    Just-In-Time Connectivity Normal distribution matrix with Row-oriented representation.

    This class represents a row-oriented sparse matrix optimized for JAX-based
    transformations where non-zero elements follow a normal distribution. It follows
    the Compressed Sparse Row (CSR) format conceptually, storing location (mean) and
    scale (standard deviation) parameters for the normal distribution, along with
    probability and seed information to determine the sparse structure.

    The class is designed for efficient neural network connectivity patterns where weights
    follow a normal distribution but connectivity is sparse and stochastic.

    Attributes
    ----------
    wloc : Union[jax.Array, u.Quantity]
        The location (mean) parameter of the normal distribution for non-zero elements.
    wscale : Union[jax.Array, u.Quantity]
        The scale (standard deviation) parameter of the normal distribution for non-zero elements.
    prob : Union[float, jax.Array]
        Probability for each potential connection.
    seed : Union[int, jax.Array]
        Random seed used for initialization of the sparse structure.
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
        >>> from brainevent import JITCNormalR

        # Create a normal distribution matrix with mean 1.5, std 0.2, probability 0.1, and seed 42
        >>> normal_matrix = JITCNormalR((1.5, 0.2, 0.1, 42), shape=(10, 10))
        >>> normal_matrix
        JITCNormalR(shape=(10, 10), wloc=1.5, wscale=0.2, prob=0.1, seed=42, corder=False)

        >>> # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = normal_matrix @ vec

        >>> # Apply scalar operation
        >>> scaled = normal_matrix * 2.0

        >>> # Convert to dense representation
        >>> dense_matrix = normal_matrix.todense()

        >>> # Transpose operation returns a JITCNormalC instance
        >>> col_matrix = normal_matrix.transpose()

    Notes
    -----
    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with normally distributed weights
    - Optimized for matrix-vector operations common in neural simulations
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
        return float_jitc_normal_matrix(
            self.wloc,
            self.wscale,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
        )

    def transpose(self, axes=None) -> 'JITCNormalC':
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
        JITCNormalC
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
        >>> isinstance(col_matrix, JITCNormalC)  # True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCNormalC(
            (self.wloc, self.wscale, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder
        )

    def _new_mat(self, loc, scale, prob=None, seed=None):
        return JITCNormalR(
            (
                loc,
                scale,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder
        )

    def _unitary_op(self, op) -> 'JITCNormalR':
        return self._new_mat(op(self.wloc), self.wscale)

    def _binary_op(self, other, op) -> 'JITCNormalR':
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.wloc, other), self.wscale)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCNormalR':
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.wloc), self.wscale)
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
                return binary_jitc_normal_matvec(
                    self.wloc,
                    self.wscale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
                    corder=self.corder,
                )
            elif other.ndim == 2:
                # JIT matrix @ events
                return binary_jitc_normal_matmat(
                    self.wloc,
                    self.wscale,
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
            loc, other = u.math.promote_dtypes(self.wloc, other)
            scale, other = u.math.promote_dtypes(self.wscale, other)
            if other.ndim == 1:
                # JIT matrix @ vector
                return float_jitc_normal_matvec(
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
                return float_jitc_normal_matmat(
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
                return binary_jitc_normal_matvec(
                    self.wloc,
                    self.wscale,
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
                r = binary_jitc_normal_matmat(
                    self.wloc,
                    self.wscale,
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
            loc, other = u.math.promote_dtypes(self.wloc, other)
            scale, other = u.math.promote_dtypes(self.wscale, other)
            if other.ndim == 1:
                #
                # vector @ JIT matrix
                # ==
                # JIT matrix.T @ vector
                #
                return float_jitc_normal_matvec(
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
                r = float_jitc_normal_matmat(
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
class JITCNormalC(JITNormalMatrix):
    """
    Just-In-Time Connectivity Normal distribution matrix with Column-oriented representation.

    This class represents a column-oriented sparse matrix optimized for JAX-based
    transformations where non-zero elements follow a normal distribution. It follows
    the Compressed Sparse Column (CSC) format conceptually, storing location (mean) and
    scale (standard deviation) parameters for the normal distribution, along with
    probability and seed information to determine the sparse structure.

    The column-oriented structure makes column-based operations more efficient than row-based
    ones, making this class the transpose-oriented counterpart to JITCNormalR.

    Attributes
    ----------
    wloc : Union[jax.Array, u.Quantity]
        The location (mean) parameter of the normal distribution for non-zero elements.
    wscale : Union[jax.Array, u.Quantity]
        The scale (standard deviation) parameter of the normal distribution for non-zero elements.
    prob : Union[float, jax.Array]
        Probability for each potential connection.
    seed : Union[int, jax.Array]
        Random seed used for initialization of the sparse structure.
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
        >>> from brainevent import JITCNormalC

        # Create a normal distribution matrix with mean 1.5, std 0.2, probability 0.1, and seed 42
        >>> normal_matrix = JITCNormalC((1.5, 0.2, 0.1, 42), shape=(10, 10))
        >>> normal_matrix
        JITCNormalC(shape=(10, 10), wloc=1.5, wscale=0.2, prob=0.1, seed=42, corder=False)

        >>> # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = normal_matrix @ vec

        >>> # Apply scalar operation
        >>> scaled = normal_matrix * 2.0

        >>> # Convert to dense representation
        >>> dense_matrix = normal_matrix.todense()

        >>> # Transpose operation returns a JITCNormalR instance
        >>> row_matrix = normal_matrix.transpose()

    Notes
    -----
    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - More efficient than JITCNormalR for column-based operations
    - Well-suited for neural network connectivity matrices with normally distributed weights
    - Optimized for matrix-vector operations common in neural simulations
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
        return float_jitc_normal_matrix(
            self.wloc,
            self.wscale,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
        )

    def transpose(self, axes=None) -> 'JITCNormalR':
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
        JITCNormalR
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
        >>> isinstance(row_matrix, JITCNormalR)  # True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCNormalR(
            (self.wloc, self.wscale, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder
        )

    def _new_mat(self, loc, scale, prob=None, seed=None):
        return JITCNormalC(
            (
                loc,
                scale,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder
        )

    def _unitary_op(self, op) -> 'JITCNormalC':
        return self._new_mat(op(self.wloc), self.wscale)

    def _binary_op(self, other, op) -> 'JITCNormalC':
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.wloc, other), self.wscale)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCNormalC':
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.wloc), self.wscale)
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
                return binary_jitc_normal_matvec(
                    self.wloc,
                    self.wscale,
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
                return binary_jitc_normal_matmat(
                    self.wloc,
                    self.wscale,
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
            loc, other = u.math.promote_dtypes(self.wloc, other)
            scale, other = u.math.promote_dtypes(self.wscale, other)
            if other.ndim == 1:
                # JITC_R matrix.T @ vector
                # ==
                # vector @ JITC_R matrix
                return float_jitc_normal_matvec(
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
                return float_jitc_normal_matmat(
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
                return binary_jitc_normal_matvec(
                    self.wloc,
                    self.wscale,
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
                r = binary_jitc_normal_matmat(
                    self.wloc,
                    self.wscale,
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
            loc, other = u.math.promote_dtypes(self.wloc, other)
            scale, other = u.math.promote_dtypes(self.wscale, other)
            if other.ndim == 1:
                #
                # vector @ JITC_R matrix.T
                # ==
                # JITC_R matrix @ vector
                #
                return float_jitc_normal_matvec(
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
                r = float_jitc_normal_matmat(
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
