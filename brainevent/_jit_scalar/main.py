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


from typing import Union, Tuple, Optional, Dict

import brainunit as u
import jax
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import JITCMatrix
from brainevent._event.binary import BinaryArray
from brainevent._typing import MatrixShape, WeightScalar, Prob, Seed
from .binary import binary_jitsmv, binary_jitsmm
from .csr import MatrixMode, jits_to_csr
from .float import jits, jitsmv, jitsmm
from .dt2t import jitsmv_dt2t

__all__ = [
    'JITCScalarR',
    'JITCScalarC',
]


class JITCScalarMatrix(JITCMatrix):
    """
    Base class for Just-In-Time Connectivity Scalar Distribution matrices.

    This abstract class serves as the foundation for sparse matrix representations
    that use scalarly distributed weights with stochastic connectivity patterns.
    It stores lower and upper bounds for the scalar distribution, along with
    connectivity probability and a random seed that determines the sparse structure.

    Designed for efficient representation of neural connectivity matrices where
    connections follow a scalar distribution but are sparsely distributed.

    Parameters
    ----------
    low : WeightScalar or Tuple[WeightScalar, WeightScalar, Prob, Seed]
        Either the lower bound of the scalar distribution,
        or a tuple containing (low, high, prob, seed).
    high : WeightScalar, optional
        Upper bound of the scalar distribution.
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
        The lower bound of the scalar distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    whigh : Union[jax.Array, u.Quantity]
        The upper bound of the scalar distribution for non-zero elements.
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
    JITCScalarR : Row-oriented concrete subclass.
    JITCScalarC : Column-oriented concrete subclass.

    Notes
    -----
    The mathematical model for this matrix is:

        ``W[i, j] = Scalar(w0, w1) * Bernoulli(prob)``

    That is, each entry ``W[i, j]`` is independently set to a value drawn from the
    continuous scalar distribution on ``[w0, w1]`` with probability ``prob``,
    and set to zero with probability ``1 - prob``. More precisely:

        ``W[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Scalar(w0, w1)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables. The connectivity pattern ``B`` and scalar
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
        buffers: Optional[Dict] = None,
    ):
        """
        Initialize a scalar distribution sparse matrix.

        Parameters
        ----------
        low : WeightScalar or Tuple[WeightScalar, WeightScalar, Prob, Seed]
            Either the lower bound of the scalar distribution,
            or a tuple containing (low, high, prob, seed).
        high : WeightScalar, optional
            Upper bound of the scalar distribution.
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
        super().__init__(data, shape=shape, buffers=buffers)

    def __repr__(self):
        """
        Return a string representation of the scalar distribution matrix.

        Returns
        -------
        str
            A string showing the class name, shape, lower bound, upper bound,
            probability, seed, and corder flag of the matrix instance.

        Examples
        --------
        >>> matrix = JITCScalarMatrix((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>> repr(matrix)
        'JITScalarMatrix(shape=(10, 10), wlow=0.1, whigh=0.5, prob=0.2, seed=42, corder=False)'
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
    def data(self) -> Tuple[WeightScalar, WeightScalar]:
        """
        Return the trainable weights of the matrix.

        Only the trainable value parameters ``(wlow, whigh)`` are exposed here.
        The structural parameters ``prob`` and ``seed`` are non-trainable and are
        therefore excluded. This property mirrors :meth:`with_data`, which accepts
        exactly the tuple returned here, so ``mat.with_data(mat.data)``
        round-trips.

        Returns
        -------
        Tuple[WeightScalar, WeightScalar]
            The ``(wlow, whigh)`` pair: the lower and upper bounds of the scalar
            distribution.

        See Also
        --------
        with_data : Rebuild the matrix from the tuple returned here.
        """
        return self.wlow, self.whigh

    def with_data(self, data: Tuple[WeightScalar, WeightScalar]):
        """
        Create a new matrix instance with updated bounds, preserving all other structure.

        Accepts exactly the tuple returned by :attr:`data`, i.e. the
        ``(low, high)`` pair, while keeping the same ``prob``, ``seed``,
        ``shape``, ``corder``, ``backend``, and buffers. It is useful for updating
        weight bounds without changing the connectivity pattern.

        Parameters
        ----------
        data : Tuple[WeightScalar, WeightScalar]
            The new ``(low, high)`` bounds of the scalar distribution. Each must
            have the same shape and unit as the corresponding current bound
            (``wlow`` and ``whigh``).

        Returns
        -------
        JITCScalarMatrix
            A new matrix instance of the same type as the original, with updated
            lower and upper bounds but identical connectivity structure.

        Raises
        ------
        AssertionError
            If the shapes of the provided bounds don't match the shapes of the original bounds,
            or if the units of the provided bounds don't match the units of the original bounds.

        See Also
        --------
        data : Property returning the tuple accepted here.

        Examples
        --------
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCScalarR
        >>>
        >>> # Create original matrix
        >>> original = JITCScalarR((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>>
        >>> # Create new matrix with updated bounds
        >>> updated = original.with_data((0.2, 0.8))
        >>> print(updated.wlow, updated.whigh)  # 0.2 0.8
        >>>
        >>> # With units
        >>> original_units = JITCScalarR((0.1 * u.mV, 0.5 * u.mV, 0.2, 42), shape=(10, 10))
        >>> updated_units = original_units.with_data((0.2 * u.mV, 0.8 * u.mV))
        """
        low, high = data
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
            buffers=self.buffers,
        )

    def tocsr(
        self,
        *,
        matrix_mode: MatrixMode = "mv",
        chunk_size: Optional[int] = None,
        target_chunks: int = 4,
    ):
        """
        Convert the sparse scalar matrix to Compressed Sparse Row (CSR) format.

        Generates the non-zero structure ``(data, indices, indptr)`` directly
        from the connectivity parameters using dedicated CPU/CUDA operators,
        without ever materializing the dense matrix. The resulting
        :class:`~brainevent.CSR` reproduces exactly the same matrix as
        :meth:`todense` for the active compute backend.

        Returns
        -------
        CSR
            A :class:`~brainevent.CSR` matrix with the same shape and values as
            :meth:`todense`. The data type matches the weight bounds, and
            physical units (``brainunit.Quantity``) are preserved on the stored
            values.

        See Also
        --------
        todense : Materialize the matrix as a dense array.
        JITCScalarR.transpose : Switch between row- and column-oriented forms.

        Notes
        -----
        Generation uses a count pass followed by a fill pass; the number of
        stored elements is read back between the two passes, so ``tocsr`` is an
        eager-only conversion and cannot be traced under ``jax.jit``. Because
        the connectivity is reproduced from the seed rather than read from a
        dense buffer, peak memory is ``O(nnz)`` rather than ``O(rows * cols)``.

        Examples
        --------
        .. code-block:: python

            >>> from brainevent import JITCScalarR
            >>> mat = JITCScalarR((0.1, 0.5, 0.2, 42), shape=(10, 10))
            >>> csr = mat.tocsr()
            >>> csr.shape
            (10, 10)
        """
        return jits_to_csr(
            self.wlow,
            self.whigh,
            self.prob,
            self.seed,
            shape=self.shape,
            corder=self.corder,
            backend=self.backend,
            matrix_mode=matrix_mode,
            chunk_size=chunk_size,
            target_chunks=target_chunks,
        )

    def dt2t(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Generate per-synapse ``sampled_weight * y[row]`` using the matrix parameters.

        ``w_dim_arr`` is required by the :class:`DataRepresentation` protocol and is not used.
        JITC scalar connectivity and weights are generated from this matrix's own
        metadata, including ``wlow``, ``whigh``, ``prob``, ``seed``, ``shape``,
        ``corder``, and ``backend``.
        """
        return jitsmv_dt2t(
            self.wlow,
            self.whigh,
            self.prob,
            y_dim_arr,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
            backend=self.backend,
        )

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Generate per-synapse ``sampled_weight * y[col]`` using the matrix parameters.

        ``w_dim_arr`` is required by the :class:`DataRepresentation` protocol and is not used.
        JITC scalar connectivity and weights are generated from this matrix's own
        metadata, including ``wlow``, ``whigh``, ``prob``, ``seed``, ``shape``,
        ``corder``, and ``backend``.
        """
        return jitsmv_dt2t(
            self.wlow,
            self.whigh,
            self.prob,
            y_dim_arr,
            self.seed,
            shape=self.shape,
            transpose=True,
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
        return (self.wlow, self.whigh, self.prob, self.seed), (aux, self.buffers)

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
        JITCScalarMatrix
            A reconstructed matrix instance.

        Notes
        -----
        This classmethod is used by JAX's pytree system to deserialize the
        matrix after transformations. It bypasses ``__init__`` by using
        ``object.__new__`` and directly setting attributes.
        """
        obj = object.__new__(cls)
        obj.wlow, obj.whigh, obj.prob, obj.seed = children
        aux_data, buffer = aux_data
        obj._buffer_registry = set(buffer.keys())
        for k, v in aux_data.items():
            setattr(obj, k, v)
        for k, v in buffer.items():
            setattr(obj, k, v)
        return obj

    def _check(self, other, op):
        """
        Validate compatibility of two matrices for binary operations.

        Parameters
        ----------
        other : JITCScalarMatrix
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
class JITCScalarR(JITCScalarMatrix):
    """
    Just-In-Time Connectivity matrix with Row-oriented representation for scalar weight distributions.

    This class implements a row-oriented sparse matrix optimized for JAX-based transformations,
    following the Compressed Sparse Row (CSR) format conceptually. Instead of storing all non-zero
    elements explicitly, it uses a scalar distribution with lower and upper bounds (wlow, whigh)
    to generate weights for connections, along with probability and seed information to
    determine the sparse structure.

    The class is designed for efficient neural network connectivity patterns where weights
    follow a scalar distribution but connectivity is sparse and stochastic. The actual sparse
    structure and scalar weight values are generated just-in-time during operations.

    Attributes
    ----------
    wlow : Union[jax.Array, u.Quantity]
        The lower bound of the scalar distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    whigh : Union[jax.Array, u.Quantity]
        The upper bound of the scalar distribution for non-zero elements.
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
        >>> from brainevent import JITCScalarR

        # Create a scalar matrix with bounds [0.1, 0.5], probability 0.2, and seed 42
        >>> scalar_matrix = JITCScalarR((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>> scalar_matrix
        JITCScalarR(shape=(10, 10), wlow=0.1, whigh=0.5, prob=0.2, seed=42, corder=False)

        # Create a scalar matrix with units
        >>> scalar_matrix_mv = JITCScalarR((0.1 * u.mV, 0.5 * u.mV, 0.2, 42), shape=(10, 10))

        # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = scalar_matrix @ vec
        >>> # Each element in result is a weighted sum using scalarly distributed weights

        # Apply scalar operation (scales both lower and upper bounds)
        >>> scaled = scalar_matrix * 2.0
        >>> print(scaled.wlow, scaled.whigh)  # 0.2 1.0

        # Convert to dense representation
        >>> dense_matrix = scalar_matrix.todense()
        >>> # dense_matrix has shape (10, 10) with ~20% non-zero elements
        >>> # each non-zero element is scalarly distributed between 0.1 and 0.5

        # Transpose operation returns a JITCScalarC instance
        >>> col_matrix = scalar_matrix.transpose()
        >>> isinstance(col_matrix, JITCScalarC)  # True

        # Update bounds while preserving connectivity pattern
        >>> updated = scalar_matrix.with_data((0.2, 0.8))
        >>> print(updated.wlow, updated.whigh)  # 0.2 0.8

        # Use with JAX transformations
        >>> @jax.jit
        ... def matrix_vector_product(mat, vec):
        ...     return mat @ vec
        >>> result_jit = matrix_vector_product(scalar_matrix, vec)

    Notes
    -----
    The mathematical model for ``JITCScalarR`` is:

        ``W[i, j] = Scalar(w0, w1) * Bernoulli(prob)``

    Each entry ``W[i, j]`` is independently drawn from the continuous scalar
    distribution on ``[w0, w1]`` with probability ``prob``, and zero
    otherwise. More precisely, the entry is computed as:

        ``W[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Scalar(w0, w1)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables, both determined by ``seed``.

    The row-oriented representation means that the random number generator state is
    seeded per-row (or per-column, depending on ``corder``), making row-based
    operations (``W @ v``) the natural direction.

    Key properties:

    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with scalarly distributed weights
    - Optimized for matrix-vector operations common in neural simulations
    - The actual matrix elements are never explicitly stored, only generated during operations
    - Using the same seed always produces the same random connectivity pattern and weights
    """
    __module__ = 'brainevent'

    def todense(
        self,
        *,
        matrix_mode: MatrixMode = "mv",
        chunk_size: Optional[int] = None,
        target_chunks: int = 4,
    ) -> Union[jax.Array, u.Quantity]:
        """
        Convert the sparse scalar matrix to a dense array.

        Generates a full dense representation of the sparse matrix by
        sampling ``Scalar(w0, w1)`` values for all connections
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
        JITCScalarC.todense : Column-oriented variant.
        jits : Standalone function to materialize JIT scalar matrices.

        Notes
        -----
        The dense matrix is generated according to:

            ``dense[i, j] = Scalar(w0, w1) * Bernoulli(prob)``

        for each ``(i, j)`` pair, where the random draws are determined by ``seed``.

        Examples
        --------

        .. code-block:: python

            >>> import jax
            >>> from brainevent import JITCScalarR
            >>>
            >>> mat = JITCScalarR((0.1, 0.5, 0.2, 42), shape=(4, 6))
            >>> dense = mat.todense()
            >>> dense.shape
            (4, 6)
        """
        return jits(
            self.wlow,
            self.whigh,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            corder=self.corder,
            matrix_mode=matrix_mode,
            chunk_size=chunk_size,
            target_chunks=target_chunks,
            backend=self.backend,
        )

    def transpose(self, axes=None) -> 'JITCScalarC':
        """
        Transpose the row-oriented matrix into a column-oriented matrix.

        Returns a column-oriented matrix (``JITCScalarC``) with rows and columns
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
        JITCScalarC
            A new column-oriented scalar matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``, since partial axis transposition is not supported.

        See Also
        --------
        JITCScalarC.transpose : The inverse operation.

        Notes
        -----
        The transpose satisfies ``W.T[j, i] = W[i, j]``. Since both the
        connectivity pattern and the scalar weights are deterministic functions of
        ``seed``, the transposed matrix produces identical results to materializing
        ``W`` and transposing the dense array.

        The ``corder`` flag is flipped during transposition to maintain consistency
        with the underlying PRNG state ordering.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCScalarR, JITCScalarC
            >>>
            >>> row_matrix = JITCScalarR((0.1, 0.5, 0.2, 42), shape=(30, 5))
            >>> row_matrix.shape
            (30, 5)
            >>> col_matrix = row_matrix.transpose()
            >>> col_matrix.shape
            (5, 30)
            >>> isinstance(col_matrix, JITCScalarC)
            True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCScalarC(
            (self.wlow, self.whigh, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder,
            backend=self.backend,
            buffers=self.buffers,
        )

    def _new_mat(self, wlow, whigh, prob=None, seed=None):
        """
        Create a new ``JITCScalarR`` with the given weight bounds, reusing other attributes.

        Parameters
        ----------
        wlow : WeightScalar
            New lower bound for the scalar distribution.
        whigh : WeightScalar
            New upper bound for the scalar distribution.
        prob : Prob, optional
            New connection probability. If None, the current probability is reused.
        seed : Seed, optional
            New random seed. If None, the current seed is reused.

        Returns
        -------
        JITCScalarR
            A new row-oriented matrix with the specified weight bounds.
        """
        return JITCScalarR(
            (
                wlow,
                whigh,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder,
            backend=self.backend,
            buffers=self.buffers,
        )

    def _unitary_op(self, op) -> 'JITCScalarR':
        """
        Apply a unary operation to both weight bounds.

        Parameters
        ----------
        op : callable
            A unary function to apply element-wise to ``wlow`` and ``whigh``.

        Returns
        -------
        JITCScalarR
            A new matrix with the operation applied to both bounds.
        """
        return self._new_mat(op(self.wlow), op(self.whigh))

    def _binary_op(self, other, op) -> 'JITCScalarR':
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
        JITCScalarR
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

    def _binary_rop(self, other, op) -> 'JITCScalarR':
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
        JITCScalarR
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
                return binary_jitsmv(
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
                return binary_jitsmm(
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
                return jitsmv(
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
                return jitsmm(
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
                return binary_jitsmv(
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
                r = binary_jitsmm(
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
                return jitsmv(
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
                r = jitsmm(
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
class JITCScalarC(JITCScalarMatrix):
    """
    Just-In-Time Connectivity matrix with Column-oriented representation for scalar weight distributions.

    This class implements a column-oriented sparse matrix optimized for JAX-based transformations,
    following the Compressed Sparse Column (CSC) format conceptually. Instead of storing all non-zero
    elements explicitly, it uses a scalar distribution with lower and upper bounds (wlow, whigh)
    to generate weights for connections, along with probability and seed information to
    determine the sparse structure.

    The class is designed for efficient neural network connectivity patterns where weights
    follow a scalar distribution but connectivity is sparse and stochastic. The column-oriented
    structure makes column-based operations more efficient than row-based ones, making this class
    the transpose-oriented counterpart to JITCScalarR.

    Attributes
    ----------
    wlow : Union[jax.Array, u.Quantity]
        The lower bound of the scalar distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    whigh : Union[jax.Array, u.Quantity]
        The upper bound of the scalar distribution for non-zero elements.
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
        >>> from brainevent import JITCScalarC

        # Create a scalar matrix with bounds [0.1, 0.5], probability 0.2, and seed 42
        >>> scalar_matrix = JITCScalarC((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>> scalar_matrix
        JITCScalarC(shape=(10, 10), wlow=0.1, whigh=0.5, prob=0.2, seed=42, corder=False)

        # Create a scalar matrix with units
        >>> scalar_matrix_mv = JITCScalarC((0.1 * u.mV, 0.5 * u.mV, 0.2, 42), shape=(10, 10))

        # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = scalar_matrix @ vec
        >>> # Each element in result is a weighted sum using scalarly distributed weights

        # Apply scalar operation (scales both lower and upper bounds)
        >>> scaled = scalar_matrix * 2.0
        >>> print(scaled.wlow, scaled.whigh)  # 0.2 1.0

        # Convert to dense representation
        >>> dense_matrix = scalar_matrix.todense()
        >>> # dense_matrix has shape (10, 10) with ~20% non-zero elements
        >>> # each non-zero element is scalarly distributed between 0.1 and 0.5

        # Transpose operation returns a JITCScalarR instance
        >>> row_matrix = scalar_matrix.transpose()
        >>> isinstance(row_matrix, JITCScalarR)  # True

        # Update bounds while preserving connectivity pattern
        >>> updated = scalar_matrix.with_data((0.2, 0.8))
        >>> print(updated.wlow, updated.whigh)  # 0.2 0.8

        # Use with JAX transformations
        >>> @jax.jit
        ... def matrix_vector_product(mat, vec):
        ...     return mat @ vec
        >>> result_jit = matrix_vector_product(scalar_matrix, vec)

        # Matrix-matrix multiplication
        >>> mat = jax.numpy.ones((10, 5))
        >>> result_mat = scalar_matrix @ mat
        >>> result_mat.shape  # (10, 5)

        # Right matrix multiplication
        >>> mat = jax.numpy.ones((5, 10))
        >>> result_rmat = mat @ scalar_matrix
        >>> result_rmat.shape  # (5, 10)

    Notes
    -----
    The mathematical model for ``JITCScalarC`` is:

        ``W[i, j] = Scalar(w0, w1) * Bernoulli(prob)``

    Each entry ``W[i, j]`` is independently drawn from the continuous scalar
    distribution on ``[w0, w1]`` with probability ``prob``, and zero
    otherwise. More precisely:

        ``W[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Scalar(w0, w1)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables, both determined by ``seed``.

    The column-oriented representation is the transpose dual of ``JITCScalarR``.
    Internally, operations on ``JITCScalarC`` are delegated to the transposed
    ``JITCScalarR`` form: ``JITCScalarC @ v == JITCScalarR.T @ v``.

    Key properties:

    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with scalarly distributed weights
    - The column-oriented structure makes column-slicing operations more efficient
    - Optimized for matrix-vector operations common in neural simulations
    - The actual matrix elements are never explicitly stored, only generated during operations
    - Using the same seed always produces the same random connectivity pattern and weights
    """
    __module__ = 'brainevent'

    def todense(
        self,
        *,
        matrix_mode: MatrixMode = "mv",
        chunk_size: Optional[int] = None,
        target_chunks: int = 4,
    ) -> Union[jax.Array, u.Quantity]:
        """
        Convert the sparse column-oriented scalar matrix to a dense array.

        Generates a full dense representation of the sparse matrix by
        sampling ``Scalar(w0, w1)`` values for all connections
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
        JITCScalarR.todense : Row-oriented variant.
        jits : Standalone function to materialize JIT scalar matrices.

        Notes
        -----
        The dense matrix is generated according to:

            ``dense[i, j] = Scalar(w0, w1) * Bernoulli(prob)``

        for each ``(i, j)`` pair, where the random draws are determined by ``seed``.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCScalarC
            >>>
            >>> mat = JITCScalarC((0.1, 0.5, 0.2, 42), shape=(3, 10))
            >>> dense = mat.todense()
            >>> dense.shape
            (3, 10)
        """
        return jits(
            self.wlow,
            self.whigh,
            self.prob,
            self.seed,
            shape=self.shape[::-1],
            transpose=True,
            corder=not self.corder,
            matrix_mode=matrix_mode,
            chunk_size=chunk_size,
            target_chunks=target_chunks,
            backend=self.backend,
        )

    def tocsr(
        self,
        *,
        matrix_mode: MatrixMode = "mv",
        chunk_size: Optional[int] = None,
        target_chunks: int = 4,
    ):
        csr = jits_to_csr(
            self.wlow,
            self.whigh,
            self.prob,
            self.seed,
            shape=self.shape[::-1],
            corder=not self.corder,
            backend=self.backend,
            matrix_mode=matrix_mode,
            chunk_size=chunk_size,
            target_chunks=target_chunks,
        )
        return csr.transpose().tocsr()

    def transpose(self, axes=None) -> 'JITCScalarR':
        """
        Transpose the column-oriented matrix into a row-oriented matrix.

        Returns a row-oriented matrix (``JITCScalarR``) with rows and columns
        swapped, preserving the same weight bounds, probability, and seed values.

        Parameters
        ----------
        axes : None
            Not supported. This parameter exists for compatibility with the NumPy API
            but only ``None`` is accepted.

        Returns
        -------
        JITCScalarR
            A new row-oriented scalar matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``, since partial axis transposition is not supported.

        See Also
        --------
        JITCScalarR.transpose : The inverse operation.

        Notes
        -----
        The transpose satisfies ``W.T[j, i] = W[i, j]``. The ``corder`` flag is
        flipped during transposition to maintain consistency with the underlying
        PRNG state ordering.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCScalarC, JITCScalarR
            >>>
            >>> col_matrix = JITCScalarC((0.1, 0.5, 0.2, 42), shape=(3, 5))
            >>> col_matrix.shape
            (3, 5)
            >>> row_matrix = col_matrix.transpose()
            >>> row_matrix.shape
            (5, 3)
            >>> isinstance(row_matrix, JITCScalarR)
            True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCScalarR(
            (self.wlow, self.whigh, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            corder=not self.corder,
            backend=self.backend,
            buffers=self.buffers,
        )

    def _new_mat(self, wlow, whigh, prob=None, seed=None):
        """
        Create a new ``JITCScalarC`` with the given weight bounds, reusing other attributes.

        Parameters
        ----------
        wlow : WeightScalar
            New lower bound for the scalar distribution.
        whigh : WeightScalar
            New upper bound for the scalar distribution.
        prob : Prob, optional
            New connection probability. If None, the current probability is reused.
        seed : Seed, optional
            New random seed. If None, the current seed is reused.

        Returns
        -------
        JITCScalarC
            A new column-oriented matrix with the specified weight bounds.
        """
        return JITCScalarC(
            (
                wlow,
                whigh,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            corder=self.corder,
            backend=self.backend,
            buffers=self.buffers,
        )

    def _unitary_op(self, op) -> 'JITCScalarC':
        """
        Apply a unary operation to both weight bounds.

        Parameters
        ----------
        op : callable
            A unary function to apply element-wise to ``wlow`` and ``whigh``.

        Returns
        -------
        JITCScalarC
            A new matrix with the operation applied to both bounds.
        """
        return self._new_mat(op(self.wlow), op(self.whigh))

    def _binary_op(self, other, op) -> 'JITCScalarC':
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
        JITCScalarC
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

    def _binary_rop(self, other, op) -> 'JITCScalarC':
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
        JITCScalarC
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

        Internally delegates to the underlying ``JITCScalarR`` representation
        by using a transposed view: ``JITCScalarC @ other == JITCScalarR.T @ other``.
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
                return binary_jitsmv(
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
                return binary_jitsmm(
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
                return jitsmv(
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
                return jitsmm(
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

        Internally delegates to the underlying ``JITCScalarR`` representation:
        ``other @ JITCScalarC == other @ JITCScalarR.T == JITCScalarR @ other``
        for vectors, or ``(JITCScalarR @ other.T).T`` for matrices.
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
                return binary_jitsmv(
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
                r = binary_jitsmm(
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
                return jitsmv(
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
                r = jitsmm(
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
