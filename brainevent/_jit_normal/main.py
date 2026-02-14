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

from brainevent._compatible_import import Tracer
from brainevent._data import JITCMatrix
from brainevent._event.binary import BinaryArray
from brainevent._typing import MatrixShape, WeightScalar, Prob, Seed
from .binary import (
    binary_jitnmv,
    binary_jitnmm,
)
from .float import (
    jitn,
    jitnmv,
    jitnmm,
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

    Parameters
    ----------
    loc : WeightScalar or Tuple[WeightScalar, WeightScalar, Prob, Seed]
        Either the location (mean) parameter, or a tuple ``(loc, scale, prob, seed)``.
    scale : WeightScalar, optional
        Scale (standard deviation) parameter.
    prob : Prob, optional
        Connection probability determining matrix sparsity.
    seed : Seed, optional
        Random seed for reproducible sparse structure generation.
    shape : MatrixShape
        The shape of the matrix as a tuple ``(rows, columns)``.
    corder : bool, optional
        Memory layout order flag, by default False.
    backend : str or None, optional
        The computation backend to use.

    Returns
    -------
    JITNormalMatrix
        A new normal-weight JIT connectivity matrix instance.

    Raises
    ------
    brainunit.DimensionMismatchError
        If ``loc`` and ``scale`` do not have the same physical dimension.

    See Also
    --------
    JITCNormalR : Row-oriented concrete subclass.
    JITCNormalC : Column-oriented concrete subclass.
    JITCMatrix : Parent class for all JIT connectivity matrices.

    Notes
    -----
    The matrix ``W`` is defined by a location (mean) ``mu``, a scale
    (standard deviation) ``sigma``, a connection probability ``p``, and a
    deterministic pseudo-random seed ``s``. Each element is given by:

    ``W[i, j] = Normal(mu, sigma) * Bernoulli(p)``

    Equivalently, each non-zero entry is an independent draw from
    ``N(mu, sigma^2)``, and the set of non-zero positions is determined by
    the Bernoulli mask with probability ``p``, whose realization is fully
    determined by the seed. This means:

    - The same ``(loc, scale, prob, seed, shape)`` always produces the
      identical matrix.
    - The expected value of each element is ``E[W[i,j]] = mu * p``.
    - The variance of each element is
      ``Var[W[i,j]] = p * (sigma^2 + mu^2) - (p * mu)^2``.
    - The matrix is never materialized in memory; it is regenerated
      on-the-fly during each operation.

    The connection length parameter ``clen = 2 / p`` controls the average
    stride between successive non-zero entries in the sampling loop.

    Examples
    --------

    .. code-block:: python

        >>> from brainevent import JITCNormalR
        >>> mat = JITCNormalR((1.0, 0.1, 0.2, 42), shape=(100, 50))
        >>> mat.wloc    # 1.0
        >>> mat.wscale  # 0.1
        >>> mat.prob    # 0.2
        >>> mat.seed    # 42

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
    backend: Optional[str]

    def __init__(
        self,
        loc,
        scale=None,
        prob=None,
        seed=None,
        *,
        shape: MatrixShape,
        corder: bool = False,
        backend: Optional[str] = None,
    ):
        """
        Initialize a normal distribution sparse matrix.

        Parameters
        ----------
        loc : WeightScalar or Tuple[WeightScalar, WeightScalar, Prob, Seed]
            Either the location (mean) parameter of the normal distribution,
            or a tuple containing (loc, scale, prob, seed).
        scale : WeightScalar, optional
            Scale (standard deviation) parameter of the normal distribution.
            If None, ``loc`` is treated as a tuple of (loc, scale, prob, seed).
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
        if scale is None and prob is None and seed is None:
            data = loc
        else:
            data = (loc, scale, prob, seed)
        loc, scale, self.prob, self.seed = data
        loc, scale = u.math.promote_dtypes(loc, scale)
        u.fail_for_dimension_mismatch(loc, scale, "loc and scale must have the same dimension.")
        self.wloc = u.math.asarray(loc)
        self.wscale = u.math.asarray(scale)
        self.corder = corder
        self.backend = backend
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

        Parameters
        ----------
        loc : WeightScalar
            The new location (mean) parameter for the normal distribution.
            Must have the same shape and unit as the current ``wloc``.
        scale : WeightScalar
            The new scale (standard deviation) parameter for the normal distribution.
            Must have the same shape and unit as the current ``wscale``.

        Returns
        -------
        JITNormalMatrix
            A new instance of the same class with updated weight parameters.

        Raises
        ------
        AssertionError
            If the shape or unit of the new parameters does not match the originals.

        See Also
        --------
        data : Property returning the current (wloc, wscale, prob, seed) tuple.
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
            corder=self.corder,
            backend=self.backend,
        )

    def tree_flatten(self):
        """
        Flatten the matrix into leaves and auxiliary data for JAX pytree compatibility.

        Returns
        -------
        tuple
            A pair ``(children, aux_data)`` where ``children`` is a tuple of
            ``(wloc, wscale, prob, seed)`` JAX-traceable arrays, and ``aux_data``
            is a dict of static metadata (``shape``, ``corder``, ``backend``).

        See Also
        --------
        tree_unflatten : Reconstruct the matrix from flattened data.
        """
        aux = {'shape': self.shape, 'corder': self.corder, 'backend': self.backend}
        return (self.wloc, self.wscale, self.prob, self.seed), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a matrix instance from flattened pytree data.

        Parameters
        ----------
        aux_data : dict
            Dictionary containing static metadata: ``shape``, ``corder``, ``backend``.
        children : tuple
            Tuple of JAX-traceable arrays ``(wloc, wscale, prob, seed)``.

        Returns
        -------
        JITNormalMatrix
            A reconstructed instance of the matrix class.

        See Also
        --------
        tree_flatten : Flatten the matrix for JAX pytree serialization.
        """
        obj = object.__new__(cls)
        obj.wloc, obj.wscale, obj.prob, obj.seed = children
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj

    def _check(self, other, op):
        """
        Validate compatibility of two JITNormalMatrix instances for binary operations.

        Parameters
        ----------
        other : JITNormalMatrix
            The other matrix to check compatibility with.
        op : str
            Name of the operation being performed, used in error messages.

        Raises
        ------
        NotImplementedError
            If the two matrices have different seeds, if either seed is being
            traced by JAX, or if the matrices have different ``corder`` flags.
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
    The mathematical model for this matrix is:

    ``W[i, j] = Normal(mu, sigma) * Bernoulli(p)``

    where ``mu`` is ``wloc``, ``sigma`` is ``wscale``, ``p`` is ``prob``,
    and the Bernoulli and Normal draws are both determined by the seed.
    For a matrix-vector product ``y = W @ x``:

    ``y[i] = sum_{j in C(i)} N_ij * x[j]``

    where ``C(i)`` is the deterministic random connection set for row ``i``
    and ``N_ij ~ Normal(mu, sigma)`` is the weight for that connection.

    Key properties:

    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with normally distributed weights
    - Optimized for matrix-vector operations common in neural simulations
    - The matrix is implicitly constructed based on the probability and seed;
      the actual sparse structure is materialized only when needed
    """
    __module__ = 'brainevent'

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the sparse normal-weight matrix to dense format.

        Generates a full dense representation where each non-zero entry is
        drawn from ``Normal(wloc, wscale)`` at positions determined by the
        probability and seed.

        Parameters
        ----------
        None

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A dense matrix with the same shape as the sparse matrix. The data type
            will match the weight's data type, and if the weight has units (is a
            ``u.Quantity``), the returned array will have the same units.

        Raises
        ------
        None

        See Also
        --------
        jitn : The underlying function that materializes the matrix.

        Notes
        -----
        The dense matrix is generated element-wise as:

        ``dense[i, j] = Normal(mu, sigma)  if  hash(seed, i, j) < p  else  0``

        where ``mu = wloc``, ``sigma = wscale``, and ``p = prob``.

        Examples
        --------

        .. code-block:: python

            >>> import brainunit as u
            >>> from brainevent import JITCNormalR
            >>> sparse_matrix = JITCNormalR((1.5, 0.2, 0.5, 42), shape=(10, 4))
            >>> dense_matrix = sparse_matrix.todense()
            >>> dense_matrix.shape  # (10, 4)
        """
        return jitn(
            self.wloc,
            self.wscale,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
            backend=self.backend,
        )

    def transpose(self, axes=None) -> 'JITCNormalC':
        """
        Transpose the row-oriented matrix into a column-oriented matrix.

        Returns a column-oriented matrix (``JITCNormalC``) with rows and columns
        swapped, preserving the same weight parameters, probability, and seed.

        Parameters
        ----------
        axes : None
            Not supported. This parameter exists for compatibility with the NumPy API
            but only None is accepted.

        Returns
        -------
        JITCNormalC
            A new column-oriented normal-weight matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If axes is not None, since partial axis transposition is not supported.

        See Also
        --------
        JITCNormalC.transpose : The inverse operation.

        Notes
        -----
        The transpose swaps the ``shape`` and inverts the ``corder`` flag so that
        the same PRNG sequence is used, ensuring ``mat.transpose().todense()``
        equals ``mat.todense().T``.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCNormalR
            >>> row_matrix = JITCNormalR((1.5, 0.2, 0.5, 42), shape=(30, 5))
            >>> col_matrix = row_matrix.transpose()
            >>> col_matrix.shape  # (5, 30)
            >>> isinstance(col_matrix, JITCNormalC)  # True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCNormalC(
            (self.wloc, self.wscale, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder,
            backend=self.backend,
        )

    def _new_mat(self, loc, scale, prob=None, seed=None):
        """
        Create a new ``JITCNormalR`` with the given parameters, inheriting shape and layout.

        Parameters
        ----------
        loc : WeightScalar
            The location (mean) parameter for the new matrix.
        scale : WeightScalar
            The scale (standard deviation) parameter for the new matrix.
        prob : Prob, optional
            Connection probability. Defaults to ``self.prob`` if None.
        seed : Seed, optional
            Random seed. Defaults to ``self.seed`` if None.

        Returns
        -------
        JITCNormalR
            A new row-oriented normal distribution matrix.
        """
        return JITCNormalR(
            (
                loc,
                scale,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder,
            backend=self.backend,
        )

    def _unitary_op(self, op) -> 'JITCNormalR':
        """
        Apply a unary operation to the location parameter of the matrix.

        The operation is applied only to ``wloc``; ``wscale`` is preserved unchanged.

        Parameters
        ----------
        op : callable
            A unary function to apply to the location parameter.

        Returns
        -------
        JITCNormalR
            A new matrix with the operation applied to ``wloc``.
        """
        return self._new_mat(op(self.wloc), self.wscale)

    def _binary_op(self, other, op) -> 'JITCNormalR':
        """
        Apply a binary operation between this matrix's location parameter and a scalar.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The right-hand operand. Must be a scalar (size 1).
        op : callable
            A binary function (e.g., ``operator.mul``) to apply.

        Returns
        -------
        JITCNormalR
            A new matrix with the operation applied to ``wloc``.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has size greater than 1.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.wloc, other), self.wscale)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCNormalR':
        """
        Apply a reflected binary operation with this matrix's location parameter.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The left-hand operand. Must be a scalar (size 1).
        op : callable
            A binary function (e.g., ``operator.mul``) to apply as ``op(other, wloc)``.

        Returns
        -------
        JITCNormalR
            A new matrix with the reflected operation applied to ``wloc``.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has size greater than 1.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.wloc), self.wscale)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
        """
        Compute the matrix product ``self @ other``.

        Dispatches to the appropriate kernel depending on the type and dimensionality
        of ``other``:

        - If ``other`` is a ``BinaryArray`` (event-driven), uses the binary event
          kernel (``binary_jitnmv`` for 1-D, ``binary_jitnmm`` for 2-D).
        - Otherwise, uses the float kernel (``jitnmv`` for 1-D, ``jitnmm`` for 2-D).

        Parameters
        ----------
        other : jax.Array, u.Quantity, or BinaryArray
            The right-hand operand. Must be 1-D (vector) or 2-D (matrix).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The result of the matrix-vector or matrix-matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has dimensionality other than 1 or 2.

        See Also
        --------
        __rmatmul__ : Compute ``other @ self``.
        binary_jitnmv : Event-driven matrix-vector multiplication kernel.
        jitnmv : Float matrix-vector multiplication kernel.

        Notes
        -----
        Data types of the weight parameters and the operand are promoted to
        compatible types before the multiplication is dispatched.
        """
        # csr @ other
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                # JIT matrix @ events
                return binary_jitnmv(
                    self.wloc,
                    self.wscale,
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
                return binary_jitnmm(
                    self.wloc,
                    self.wscale,
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
            loc, other = u.math.promote_dtypes(self.wloc, other)
            scale, other = u.math.promote_dtypes(self.wscale, other)
            if other.ndim == 1:
                # JIT matrix @ vector
                return jitnmv(
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
                return jitnmm(
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
        Compute the matrix product ``other @ self``.

        This is implemented by transposing the operation:

        - For 1-D ``other``: ``other @ M`` is equivalent to ``M.T @ other``.
        - For 2-D ``other``: ``other @ M`` is equivalent to ``(M.T @ other.T).T``.

        Parameters
        ----------
        other : jax.Array, u.Quantity, or BinaryArray
            The left-hand operand. Must be 1-D (vector) or 2-D (matrix).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The result of the matrix-vector or matrix-matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has dimensionality other than 1 or 2.

        See Also
        --------
        __matmul__ : Compute ``self @ other``.

        Notes
        -----
        The ``corder`` flag is inverted when performing the transposed operation
        to ensure the generated matrix is consistent with ``.todense()``.
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
                return binary_jitnmv(
                    self.wloc,
                    self.wscale,
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
                r = binary_jitnmm(
                    self.wloc,
                    self.wscale,
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
            loc, other = u.math.promote_dtypes(self.wloc, other)
            scale, other = u.math.promote_dtypes(self.wscale, other)
            if other.ndim == 1:
                #
                # vector @ JIT matrix
                # ==
                # JIT matrix.T @ vector
                #
                return jitnmv(
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
                r = jitnmm(
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
    The mathematical model is the same as ``JITCNormalR``:

    ``W[i, j] = Normal(mu, sigma) * Bernoulli(p)``

    where ``mu = wloc``, ``sigma = wscale``, and ``p = prob``. The
    column-oriented representation means that ``JITCNormalC`` is
    conceptually the transpose of a ``JITCNormalR`` matrix with swapped
    dimensions.

    Key properties:

    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - More efficient than ``JITCNormalR`` for column-based operations
    - Well-suited for neural network connectivity matrices with normally distributed weights
    - The matrix is implicitly constructed based on the probability and seed;
      the actual sparse structure is materialized only when needed
    """
    __module__ = 'brainevent'

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the sparse column-oriented normal-weight matrix to dense format.

        Generates a full dense representation where each non-zero entry is
        drawn from ``Normal(wloc, wscale)`` at positions determined by the
        probability and seed. The generated dense matrix always has ``self.shape``.

        Parameters
        ----------
        None

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A dense matrix with the same shape as the sparse matrix. The data type
            will match the weight's data type, and if the weight has units (is a
            ``u.Quantity``), the returned array will have the same units.

        Raises
        ------
        None

        See Also
        --------
        jitn : The underlying function that materializes the matrix.

        Notes
        -----
        The dense matrix is generated element-wise as:

        ``dense[i, j] = Normal(mu, sigma)  if  hash(seed, i, j) < p  else  0``

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCNormalC
            >>> sparse_matrix = JITCNormalC((1.5, 0.2, 0.5, 42), shape=(3, 10))
            >>> dense_matrix = sparse_matrix.todense()
            >>> dense_matrix.shape  # (3, 10)
        """
        return jitn(
            self.wloc,
            self.wscale,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
            backend=self.backend,
        )

    def transpose(self, axes=None) -> 'JITCNormalR':
        """
        Transposes the column-oriented matrix into a row-oriented matrix.

        This method returns a row-oriented matrix (``JITCNormalR``) with rows and
        columns swapped, preserving the same weight parameters (``wloc``,
        ``wscale``), probability, and seed values. The transpose operation
        effectively converts between column-oriented and row-oriented sparse
        matrix formats.

        Parameters
        ----------
        axes : None
            Not supported. This parameter exists for compatibility with the NumPy API
            but only None is accepted.

        Returns
        -------
        JITCNormalR
            A new row-oriented normal distribution matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If axes is not None, since partial axis transposition is not supported.

        See Also
        --------
        JITCNormalR : Row-oriented counterpart.

        Notes
        -----
        The transpose preserves the mathematical identity:

        ``JITCNormalC(shape=(m, n)).transpose().todense() == JITCNormalC(shape=(m, n)).todense().T``

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCNormalC
            >>>
            >>> # Create a column-oriented matrix
            >>> col_matrix = JITCNormalC((1.5, 0.2, 0.5, 42), shape=(3, 5))
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
            corder=not self.corder,
            backend=self.backend,
        )

    def _new_mat(self, loc, scale, prob=None, seed=None):
        """
        Create a new ``JITCNormalC`` with the given parameters, inheriting shape and layout.

        Parameters
        ----------
        loc : WeightScalar
            The location (mean) parameter for the new matrix.
        scale : WeightScalar
            The scale (standard deviation) parameter for the new matrix.
        prob : Prob, optional
            Connection probability. Defaults to ``self.prob`` if None.
        seed : Seed, optional
            Random seed. Defaults to ``self.seed`` if None.

        Returns
        -------
        JITCNormalC
            A new column-oriented normal distribution matrix.
        """
        return JITCNormalC(
            (
                loc,
                scale,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder,
            backend=self.backend,
        )

    def _unitary_op(self, op) -> 'JITCNormalC':
        """
        Apply a unary operation to the location parameter of the matrix.

        The operation is applied only to ``wloc``; ``wscale`` is preserved unchanged.

        Parameters
        ----------
        op : callable
            A unary function to apply to the location parameter.

        Returns
        -------
        JITCNormalC
            A new matrix with the operation applied to ``wloc``.
        """
        return self._new_mat(op(self.wloc), self.wscale)

    def _binary_op(self, other, op) -> 'JITCNormalC':
        """
        Apply a binary operation between this matrix's location parameter and a scalar.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The right-hand operand. Must be a scalar (size 1).
        op : callable
            A binary function (e.g., ``operator.mul``) to apply.

        Returns
        -------
        JITCNormalC
            A new matrix with the operation applied to ``wloc``.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has size greater than 1.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(self.wloc, other), self.wscale)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCNormalC':
        """
        Apply a reflected binary operation with this matrix's location parameter.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The left-hand operand. Must be a scalar (size 1).
        op : callable
            A binary function (e.g., ``operator.mul``) to apply as ``op(other, wloc)``.

        Returns
        -------
        JITCNormalC
            A new matrix with the reflected operation applied to ``wloc``.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has size greater than 1.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return self._new_mat(op(other, self.wloc), self.wscale)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
        """
        Compute the matrix product ``self @ other``.

        Since this is a column-oriented matrix (conceptually ``M_R.T``), the operation
        ``M_C @ other`` is implemented as ``M_R.T @ other``, which dispatches to
        the row-oriented kernel with the ``transpose=True`` flag and reversed shape.

        Parameters
        ----------
        other : jax.Array, u.Quantity, or BinaryArray
            The right-hand operand. Must be 1-D (vector) or 2-D (matrix).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The result of the matrix-vector or matrix-matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has dimensionality other than 1 or 2.

        See Also
        --------
        __rmatmul__ : Compute ``other @ self``.
        JITCNormalR.__matmul__ : Row-oriented forward multiplication.
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
                return binary_jitnmv(
                    self.wloc,
                    self.wscale,
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
                return binary_jitnmm(
                    self.wloc,
                    self.wscale,
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
            loc, other = u.math.promote_dtypes(self.wloc, other)
            scale, other = u.math.promote_dtypes(self.wscale, other)
            if other.ndim == 1:
                # JITC_R matrix.T @ vector
                # ==
                # vector @ JITC_R matrix
                return jitnmv(
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
                return jitnmm(
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
        Compute the matrix product ``other @ self``.

        Since this is a column-oriented matrix (conceptually ``M_R.T``), the operation
        ``other @ M_C`` is implemented as ``other @ M_R.T``, which is equivalent to
        ``M_R @ other`` (for 1-D) or ``(M_R @ other.T).T`` (for 2-D).

        Parameters
        ----------
        other : jax.Array, u.Quantity, or BinaryArray
            The left-hand operand. Must be 1-D (vector) or 2-D (matrix).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The result of the matrix-vector or matrix-matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has dimensionality other than 1 or 2.

        See Also
        --------
        __matmul__ : Compute ``self @ other``.
        JITCNormalR.__rmatmul__ : Row-oriented reflected multiplication.

        Notes
        -----
        The ``corder`` flag is inverted when performing the transposed operation
        to ensure the generated matrix is consistent with ``.todense()``.
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
                return binary_jitnmv(
                    self.wloc,
                    self.wscale,
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
                r = binary_jitnmm(
                    self.wloc,
                    self.wscale,
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
            loc, other = u.math.promote_dtypes(self.wloc, other)
            scale, other = u.math.promote_dtypes(self.wscale, other)
            if other.ndim == 1:
                #
                # vector @ JITC_R matrix.T
                # ==
                # JITC_R matrix @ vector
                #
                return jitnmv(
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
                r = jitnmm(
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
