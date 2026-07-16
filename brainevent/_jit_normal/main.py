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
import warnings

import brainunit as u
import jax
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import JITCMatrix
from brainevent._event.binary import BinaryArray
from brainevent._typing import MatrixShape, WeightScalar, Prob, Seed
from .binary import binary_jitnmv, binary_jitnmm
from .csr import MatrixMode, jitn_to_csr
from .float import jitn, jitnmv, jitnmm
from .dt2t import jitnmv_dt2t

__all__ = [
    'JITCNormalR',
    'JITCNormalC',
]


def _warn_corder_deprecated(corder: Optional[bool]) -> None:
    if corder is None:
        return
    warnings.warn(
        "corder is deprecated and ignored by JITCNormalMatrix; use transpose/T for orientation.",
        FutureWarning,
        stacklevel=3,
    )


class JITCNormalMatrix(JITCMatrix):
    """
    Base class for Just-In-Time Connectivity Normal Distribution matrices.

    This abstract class serves as the foundation for sparse matrix representations
    that use normally distributed weights with stochastic connectivity patterns.
    It stores location (mean) and scale (standard deviation) parameters for the
    normal distribution, along with connectivity probability and a random seed
    that determines the sparse structure.

    Designed for efficient representation of neural connectivity matrices where
    connection weights follow a normal distribution but connections are sparsely distributed.

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
        The shape of the matrix as a tuple (rows, columns).
    corder : bool, optional
        Deprecated compatibility argument. Ignored; use ``transpose``/``T`` for orientation.
    backend : str, optional
        Computation backend override.

    Attributes
    ----------
    wloc : Union[jax.Array, u.Quantity]
        The lower bound of the normal distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    wscale : Union[jax.Array, u.Quantity]
        The upper bound of the normal distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    prob : Union[float, jax.Array]
        Connection probability determining the sparsity of the matrix.
        Values range from 0 (no connections) to 1 (fully connected).
    seed : Union[int, jax.Array]
        Random seed controlling the specific pattern of connections.
        Using the same seed produces identical connectivity patterns.
    shape : MatrixShape
        Tuple specifying the dimensions of the matrix as (rows, columns).

    Raises
    ------
    ValueError
        If ``prob`` is not a finite scalar in [0, 1], or if ``wscale`` is not
        positive.

    See Also
    --------
    JITCNormalR : Row-oriented concrete subclass.
    JITCNormalC : Column-oriented concrete subclass.

    Notes
    -----
    The mathematical model for this matrix is:

        ``W[i, j] = Normal(w_loc, w_scale) * Bernoulli(prob)``

    That is, each entry ``W[i, j]`` is independently set to a value drawn from the
    continuous normal distribution on ``[w_loc, w_scale]`` with probability ``prob``,
    and set to zero with probability ``1 - prob``. More precisely:

        ``W[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Normal(w_loc, w_scale)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables. The connectivity pattern ``B`` and normal
    variates ``U`` are determined by the ``seed`` parameter, so using the same seed
    always produces the same matrix.

    The matrix is never materialized in memory; instead, weights and connectivity
    are generated on-the-fly during matrix operations using a PRNG seeded by
    ``seed``.
    """
    __module__ = 'brainevent'

    wloc: Union[jax.Array, u.Quantity]
    wscale: Union[jax.Array, u.Quantity]
    prob: Union[float, jax.Array]
    seed: Union[int, jax.Array]
    shape: MatrixShape

    def __init__(
        self,
        loc,
        scale=None,
        prob=None,
        seed=None,
        *,
        shape: MatrixShape,
        corder: Optional[bool] = None,
        backend: Optional[str] = None,
        buffers: Optional[Dict] = None,
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
        _warn_corder_deprecated(corder)
        if scale is None and prob is None and seed is None:
            data = loc
        else:
            data = (loc, scale, prob, seed)
        loc, scale, self.prob, self.seed = data
        if not isinstance(self.prob, Tracer):
            prob = np.asarray(self.prob)
            if prob.size != 1:
                raise ValueError(f"prob must be a scalar, but got shape {prob.shape}.")
            prob = float(prob.item())
            if not np.isfinite(prob):
                raise ValueError(f"prob must be finite, but got {prob}.")
            if not (0. <= prob <= 1.):
                raise ValueError(f"prob must be in [0, 1], but got {prob}.")

        loc, scale = u.math.promote_dtypes(loc, scale)
        u.fail_for_dimension_mismatch(loc, scale, "loc and scale must have the same dimension.")
        loc_m = u.get_mantissa(loc)
        scale_m = u.get_mantissa(scale)
        if not (isinstance(loc_m, Tracer) or isinstance(scale_m, Tracer)):
            scale_arr = np.asarray(scale_m)
            if np.any(scale_arr <= 0):
                raise ValueError("wscale must be positive.")
        self.wloc = u.math.asarray(loc)
        self.wscale = u.math.asarray(scale)
        self.backend = backend
        super().__init__(data, shape=shape, buffers=buffers)

    def __repr__(self):
        """
        Return a string representation of the normal distribution matrix.

        Returns
        -------
        str
            A string showing the class name, shape, lower bound, upper bound,
            probability, seed, and backend of the matrix instance.

        Examples
        --------
        >>> matrix = JITCNormalMatrix((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>> repr(matrix)
        'JITNormalMatrix(shape=(10, 10), wloc=0.1, wscale=0.5, prob=0.2, seed=42, backend=None,)'
        """
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"wloc={self.wloc}, "
            f"wscale={self.wscale}, "
            f"prob={self.prob}, "
            f"seed={self.seed}, "
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
        This property inherits the dtype directly from the wloc attribute,
        ensuring consistent data typing throughout operations involving this matrix.
        """
        return self.wloc.dtype

    @property
    def data(self) -> Tuple[WeightScalar, WeightScalar]:
        """
        Return the trainable weights of the matrix.

        Only the trainable value parameters ``(wloc, wscale)`` are exposed here.
        The structural parameters ``prob`` and ``seed`` are non-trainable and are
        therefore excluded. This property mirrors :meth:`with_data`, which accepts
        exactly the tuple returned here, so ``mat.with_data(mat.data)``
        round-trips.

        Returns
        -------
        Tuple[WeightScalar, WeightScalar]
            The ``(wloc, wscale)`` pair: the mean and standard deviation of the normal
            distribution.

        See Also
        --------
        with_data : Rebuild the matrix from the tuple returned here.
        """
        return self.wloc, self.wscale

    def with_data(self, data: Tuple[WeightScalar, WeightScalar]):
        """
        Create a new matrix instance with updated parameters, preserving all other structure.

        Accepts exactly the tuple returned by :attr:`data`, i.e. the
        ``(loc, scale)`` pair, while keeping the same ``prob``, ``seed``,
        ``shape``, ``backend``, and buffers.

        Parameters
        ----------
        data : Tuple[WeightScalar, WeightScalar]
            The new ``(loc, scale)`` parameters of the normal distribution. Each must
            have the same shape and unit as the corresponding current parameter
            (``wloc`` and ``wscale``).

        Returns
        -------
        JITCNormalMatrix
            A new matrix instance of the same type as the original, with updated
            location and scale but identical connectivity structure.

        Raises
        ------
        AssertionError
            If the shapes of the provided parameters don't match the shapes of the original,
            or if the units don't match.
        """
        loc, scale = data
        loc = u.math.asarray(loc)
        scale = u.math.asarray(scale)
        assert loc.shape == self.wloc.shape
        assert scale.shape == self.wscale.shape
        assert u.get_unit(loc) == u.get_unit(self.wloc)
        assert u.get_unit(scale) == u.get_unit(self.wscale)
        return type(self)(
            (loc, scale, self.prob, self.seed),
            shape=self.shape,
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
        Convert the sparse normal matrix to Compressed Sparse Row (CSR) format.

        Generates the non-zero structure ``(data, indices, indptr)`` directly
        from the connectivity parameters using dedicated CPU/CUDA operators,
        without ever materializing the dense matrix. The resulting
        :class:`~brainevent.CSR` reproduces exactly the same matrix as
        :meth:`todense` for the active compute backend.

        Returns
        -------
        CSR
            A :class:`~brainevent.CSR` matrix with the same shape and values as
            :meth:`todense`. The data type matches the weight parameters, and
            physical units (``brainunit.Quantity``) are preserved on the stored
            values.

        See Also
        --------
        todense : Materialize the matrix as a dense array.
        JITCNormalR.transpose : Switch between row- and column-oriented forms.

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

            >>> from brainevent import JITCNormalR
            >>> mat = JITCNormalR((0.1, 0.5, 0.2, 42), shape=(10, 10))
            >>> csr = mat.tocsr()
            >>> csr.shape
            (10, 10)
        """
        return jitn_to_csr(
            self.wloc,
            self.wscale,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
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
        JITC normal connectivity and weights are generated from this matrix's own
        metadata, including ``wloc``, ``wscale``, ``prob``, ``seed``, ``shape``,
        and ``backend``.
        """
        return jitnmv_dt2t(
            self.wloc,
            self.wscale,
            self.prob,
            y_dim_arr,
            self.seed,
            shape=self.shape,
            transpose=False,
            backend=self.backend,
        )

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Generate per-synapse ``sampled_weight * y[col]`` using the matrix parameters.

        ``w_dim_arr`` is required by the :class:`DataRepresentation` protocol and is not used.
        JITC normal connectivity and weights are generated from this matrix's own
        metadata, including ``wloc``, ``wscale``, ``prob``, ``seed``, ``shape``,
        and ``backend``.
        """
        return jitnmv_dt2t(
            self.wloc,
            self.wscale,
            self.prob,
            y_dim_arr,
            self.seed,
            shape=self.shape,
            transpose=True,
            backend=self.backend,
        )

    def tree_flatten(self):
        """
        Flatten the matrix into a list of leaves and auxiliary data for JAX pytree.

        Returns
        -------
        tuple
            A pair of (children, aux_data) where children is a tuple of
            (wloc, wscale, prob, seed) and aux_data is a dict containing
            shape and backend.

        Notes
        -----
        This method is used by JAX's pytree system to serialize the matrix
        for transformations such as ``jax.jit``, ``jax.grad``, and ``jax.vmap``.
        """
        aux = {'shape': self.shape, 'backend': self.backend}
        return (self.wloc, self.wscale, self.prob, self.seed), (aux, self.buffers)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a matrix from its flattened pytree representation.

        Parameters
        ----------
        aux_data : dict
            Auxiliary data containing shape and backend.
        children : tuple
            A tuple of (wloc, wscale, prob, seed) leaf values.

        Returns
        -------
        JITCNormalMatrix
            A reconstructed matrix instance.

        Notes
        -----
        This classmethod is used by JAX's pytree system to deserialize the
        matrix after transformations. It bypasses ``__init__`` by using
        ``object.__new__`` and directly setting attributes.
        """
        obj = object.__new__(cls)
        obj.wloc, obj.wscale, obj.prob, obj.seed = children
        aux_data, buffer = aux_data
        aux_data = dict(aux_data)
        aux_data.pop('corder', None)
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
        other : JITCNormalMatrix
            The other matrix to check compatibility with.
        op : str
            Name of the binary operation being performed, used in error messages.

        Raises
        ------
        NotImplementedError
            If the two matrices have different seeds, tracing seeds,
            or incompatible traced seeds.
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


@jax.tree_util.register_pytree_node_class
class JITCNormalR(JITCNormalMatrix):
    """
    Just-In-Time Connectivity matrix with Row-oriented representation for normal weight distributions.

    This class implements a row-oriented sparse matrix optimized for JAX-based transformations,
    following the Compressed Sparse Row (CSR) format conceptually. Instead of storing all non-zero
    elements explicitly, it uses a normal distribution with mean and standard deviation (wloc, wscale)
    to generate weights for connections, along with probability and seed information to
    determine the sparse structure.

    The class is designed for efficient neural network connectivity patterns where weights
    follow a normal distribution but connectivity is sparse and stochastic. The actual sparse
    structure and normal weight values are generated just-in-time during operations.

    Attributes
    ----------
    wloc : Union[jax.Array, u.Quantity]
        The lower bound of the normal distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    wscale : Union[jax.Array, u.Quantity]
        The upper bound of the normal distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    prob : Union[float, jax.Array]
        Connection probability determining the sparsity of the matrix.
        Values range from 0 (no connections) to 1 (fully connected).
    seed : Union[int, jax.Array]
        Random seed controlling the specific pattern of connections.
        Using the same seed produces identical connectivity patterns.
    shape : MatrixShape
        Tuple specifying the dimensions of the matrix as (rows, columns).
    dtype
        The data type of the matrix elements (property inherited from parent).

    Examples
    --------

    .. code-block:: python

        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCNormalR

        # Create a normal matrix with bounds [0.1, 0.5], probability 0.2, and seed 42
        >>> normal_matrix = JITCNormalR((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>> normal_matrix
        JITCNormalR(shape=(10, 10), wloc=0.1, wscale=0.5, prob=0.2, seed=42, backend=None,)

        # Create a normal matrix with units
        >>> normal_matrix_mv = JITCNormalR((0.1 * u.mV, 0.5 * u.mV, 0.2, 42), shape=(10, 10))

        # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = normal_matrix @ vec
        >>> # Each element in result is a weighted sum using normally distributed weights

        # Apply scalar operation (scales both normal parameters)
        >>> scaled = normal_matrix * 2.0
        >>> print(scaled.wloc, scaled.wscale)  # 0.2 1.0

        # Convert to dense representation
        >>> dense_matrix = normal_matrix.todense()
        >>> # dense_matrix has shape (10, 10) with ~20% non-zero elements
        >>> # each non-zero element is normally distributed between 0.1 and 0.5

        # Transpose operation returns a JITCNormalC instance
        >>> col_matrix = normal_matrix.transpose()
        >>> isinstance(col_matrix, JITCNormalC)  # True

        # Update bounds while preserving connectivity pattern
        >>> updated = normal_matrix.with_data((0.2, 0.8))
        >>> print(updated.wloc, updated.wscale)  # 0.2 0.8

        # Use with JAX transformations
        >>> @jax.jit
        ... def matrix_vector_product(mat, vec):
        ...     return mat @ vec
        >>> result_jit = matrix_vector_product(normal_matrix, vec)

    Notes
    -----
    The mathematical model for ``JITCNormalR`` is:

        ``W[i, j] = Normal(w_loc, w_scale) * Bernoulli(prob)``

    Each entry ``W[i, j]`` is independently drawn from the continuous normal
    distribution on ``[w_loc, w_scale]`` with probability ``prob``, and zero
    otherwise. More precisely, the entry is computed as:

        ``W[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Normal(w_loc, w_scale)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables, both determined by ``seed``.

    The row-oriented representation means that the random number generator state is
    generated from a canonical row-stream; ``transpose`` selects whether operations
    consume the matrix in its original or transposed orientation.

    Key properties:

    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with normally distributed weights
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
        Convert the sparse normal matrix to a dense array.

        Generates a full dense representation of the sparse matrix by
        sampling ``Normal(w_loc, w_scale)`` values for all connections
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
        JITCNormalC.todense : Column-oriented variant.
        jitn : Standalone function to materialize JIT normal matrices.

        Notes
        -----
        The dense matrix is generated according to:

            ``dense[i, j] = Normal(w_loc, w_scale) * Bernoulli(prob)``

        for each ``(i, j)`` pair, where the random draws are determined by ``seed``.

        Examples
        --------

        .. code-block:: python

            >>> import jax
            >>> from brainevent import JITCNormalR
            >>>
            >>> mat = JITCNormalR((0.1, 0.5, 0.2, 42), shape=(4, 6))
            >>> dense = mat.todense()
            >>> dense.shape
            (4, 6)
        """
        return jitn(
            self.wloc,
            self.wscale,
            self.prob,
            self.seed,
            shape=self.shape,
            transpose=False,
            matrix_mode=matrix_mode,
            chunk_size=chunk_size,
            target_chunks=target_chunks,
            backend=self.backend,
        )

    def transpose(self, axes=None) -> 'JITCNormalC':
        """
        Transpose the row-oriented matrix into a column-oriented matrix.

        Returns a column-oriented matrix (``JITCNormalC``) with rows and columns
        swapped, preserving the same weight parameters, probability, and seed values.
        The transpose operation effectively converts between row-oriented and
        column-oriented sparse matrix formats.

        Parameters
        ----------
        axes : None
            Not supported. This parameter exists for compatibility with the NumPy API
            but only ``None`` is accepted.

        Returns
        -------
        JITCNormalC
            A new column-oriented normal matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``, since partial axis transposition is not supported.

        See Also
        --------
        JITCNormalC.transpose : The inverse operation.

        Notes
        -----
        The transpose satisfies ``W.T[j, i] = W[i, j]``. Since both the
        connectivity pattern and the normal weights are deterministic functions of
        ``seed``, the transposed matrix produces identical results to materializing
        ``W`` and transposing the dense array.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCNormalR, JITCNormalC
            >>>
            >>> row_matrix = JITCNormalR((0.1, 0.5, 0.2, 42), shape=(30, 5))
            >>> row_matrix.shape
            (30, 5)
            >>> col_matrix = row_matrix.transpose()
            >>> col_matrix.shape
            (5, 30)
            >>> isinstance(col_matrix, JITCNormalC)
            True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCNormalC(
            (self.wloc, self.wscale, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            backend=self.backend,
            buffers=self.buffers,
        )

    def _new_mat(self, wloc, wscale, prob=None, seed=None):
        """
        Create a new ``JITCNormalR`` with the given weight parameters, reusing other attributes.

        Parameters
        ----------
        wloc : WeightScalar
            New lower bound for the normal distribution.
        wscale : WeightScalar
            New upper bound for the normal distribution.
        prob : Prob, optional
            New connection probability. If None, the current probability is reused.
        seed : Seed, optional
            New random seed. If None, the current seed is reused.

        Returns
        -------
        JITCNormalR
            A new row-oriented matrix with the specified weight parameters.
        """
        return JITCNormalR(
            (
                wloc,
                wscale,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            backend=self.backend,
            buffers=self.buffers,
        )

    def _unitary_op(self, op) -> 'JITCNormalR':
        """
        Apply a unary operation to both weight parameters.

        Parameters
        ----------
        op : callable
            A unary function to apply element-wise to ``wloc`` and ``wscale``.

        Returns
        -------
        JITCNormalR
            A new matrix with the operation applied to both bounds.
        """
        return self._new_mat(op(self.wloc), op(self.wscale))

    def _binary_op(self, other, op) -> 'JITCNormalR':
        """
        Apply a binary operation between the weight parameters and a scalar operand.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The right-hand operand. Must be a scalar (size 1) or another sparse matrix.
        op : callable
            A binary function (e.g., ``operator.mul``) to apply.

        Returns
        -------
        JITCNormalR
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
            return self._new_mat(op(self.wloc, other), op(self.wscale, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCNormalR':
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
        JITCNormalR
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
            return self._new_mat(op(other, self.wloc), op(other, self.wscale))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
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
                return binary_jitnmv(
                    self.wloc,
                    self.wscale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=False,
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
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
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
                return binary_jitnmv(
                    self.wloc,
                    self.wscale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape,
                    transpose=True,
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
                    transpose=True,  # This is import to generate the same matrix as ``.todense()``
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
                    transpose=True,  # This is import to generate the same matrix as ``.todense()``
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")


@jax.tree_util.register_pytree_node_class
class JITCNormalC(JITCNormalMatrix):
    """
    Just-In-Time Connectivity matrix with Column-oriented representation for normal weight distributions.

    This class implements a column-oriented sparse matrix optimized for JAX-based transformations,
    following the Compressed Sparse Column (CSC) format conceptually. Instead of storing all non-zero
    elements explicitly, it uses a normal distribution with mean and standard deviation (wloc, wscale)
    to generate weights for connections, along with probability and seed information to
    determine the sparse structure.

    The class is designed for efficient neural network connectivity patterns where weights
    follow a normal distribution but connectivity is sparse and stochastic. The column-oriented
    structure makes column-based operations more efficient than row-based ones, making this class
    the transpose-oriented counterpart to JITCNormalR.

    Attributes
    ----------
    wloc : Union[jax.Array, u.Quantity]
        The lower bound of the normal distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    wscale : Union[jax.Array, u.Quantity]
        The upper bound of the normal distribution for non-zero elements.
        Can be a plain JAX array or a quantity with units.
    prob : Union[float, jax.Array]
        Connection probability determining the sparsity of the matrix.
        Values range from 0 (no connections) to 1 (fully connected).
    seed : Union[int, jax.Array]
        Random seed controlling the specific pattern of connections.
        Using the same seed produces identical connectivity patterns.
    shape : MatrixShape
        Tuple specifying the dimensions of the matrix as (rows, columns).
    dtype
        The data type of the matrix elements (property inherited from parent).

    Examples
    --------

    .. code-block:: python

        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCNormalC

        # Create a normal matrix with bounds [0.1, 0.5], probability 0.2, and seed 42
        >>> normal_matrix = JITCNormalC((0.1, 0.5, 0.2, 42), shape=(10, 10))
        >>> normal_matrix
        JITCNormalC(shape=(10, 10), wloc=0.1, wscale=0.5, prob=0.2, seed=42, backend=None,)

        # Create a normal matrix with units
        >>> normal_matrix_mv = JITCNormalC((0.1 * u.mV, 0.5 * u.mV, 0.2, 42), shape=(10, 10))

        # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = normal_matrix @ vec
        >>> # Each element in result is a weighted sum using normally distributed weights

        # Apply scalar operation (scales both normal parameters)
        >>> scaled = normal_matrix * 2.0
        >>> print(scaled.wloc, scaled.wscale)  # 0.2 1.0

        # Convert to dense representation
        >>> dense_matrix = normal_matrix.todense()
        >>> # dense_matrix has shape (10, 10) with ~20% non-zero elements
        >>> # each non-zero element is normally distributed between 0.1 and 0.5

        # Transpose operation returns a JITCNormalR instance
        >>> row_matrix = normal_matrix.transpose()
        >>> isinstance(row_matrix, JITCNormalR)  # True

        # Update bounds while preserving connectivity pattern
        >>> updated = normal_matrix.with_data((0.2, 0.8))
        >>> print(updated.wloc, updated.wscale)  # 0.2 0.8

        # Use with JAX transformations
        >>> @jax.jit
        ... def matrix_vector_product(mat, vec):
        ...     return mat @ vec
        >>> result_jit = matrix_vector_product(normal_matrix, vec)

        # Matrix-matrix multiplication
        >>> mat = jax.numpy.ones((10, 5))
        >>> result_mat = normal_matrix @ mat
        >>> result_mat.shape  # (10, 5)

        # Right matrix multiplication
        >>> mat = jax.numpy.ones((5, 10))
        >>> result_rmat = mat @ normal_matrix
        >>> result_rmat.shape  # (5, 10)

    Notes
    -----
    The mathematical model for ``JITCNormalC`` is:

        ``W[i, j] = Normal(w_loc, w_scale) * Bernoulli(prob)``

    Each entry ``W[i, j]`` is independently drawn from the continuous normal
    distribution on ``[w_loc, w_scale]`` with probability ``prob``, and zero
    otherwise. More precisely:

        ``W[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Normal(w_loc, w_scale)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables, both determined by ``seed``.

    The column-oriented representation is the transpose dual of ``JITCNormalR``.
    Internally, operations on ``JITCNormalC`` are delegated to the transposed
    ``JITCNormalR`` form: ``JITCNormalC @ v == JITCNormalR.T @ v``.

    Key properties:

    - JAX PyTree compatible for use with JAX transformations (jit, grad, vmap)
    - More memory-efficient than dense matrices for sparse connectivity patterns
    - Well-suited for neural network connectivity matrices with normally distributed weights
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
        Convert the sparse column-oriented normal matrix to a dense array.

        Generates a full dense representation of the sparse matrix by
        sampling ``Normal(w_loc, w_scale)`` values for all connections
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
        JITCNormalR.todense : Row-oriented variant.
        jitn : Standalone function to materialize JIT normal matrices.

        Notes
        -----
        The dense matrix is generated according to:

            ``dense[i, j] = Normal(w_loc, w_scale) * Bernoulli(prob)``

        for each ``(i, j)`` pair, where the random draws are determined by ``seed``.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCNormalC
            >>>
            >>> mat = JITCNormalC((0.1, 0.5, 0.2, 42), shape=(3, 10))
            >>> dense = mat.todense()
            >>> dense.shape
            (3, 10)
        """
        return jitn(
            self.wloc,
            self.wscale,
            self.prob,
            self.seed,
            shape=self.shape[::-1],
            transpose=True,
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
        return jitn_to_csr(
            self.wloc,
            self.wscale,
            self.prob,
            self.seed,
            shape=self.shape[::-1],
            transpose=True,
            backend=self.backend,
            matrix_mode=matrix_mode,
            chunk_size=chunk_size,
            target_chunks=target_chunks,
        )

    def transpose(self, axes=None) -> 'JITCNormalR':
        """
        Transpose the column-oriented matrix into a row-oriented matrix.

        Returns a row-oriented matrix (``JITCNormalR``) with rows and columns
        swapped, preserving the same weight parameters, probability, and seed values.

        Parameters
        ----------
        axes : None
            Not supported. This parameter exists for compatibility with the NumPy API
            but only ``None`` is accepted.

        Returns
        -------
        JITCNormalR
            A new row-oriented normal matrix with transposed dimensions.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``, since partial axis transposition is not supported.

        See Also
        --------
        JITCNormalR.transpose : The inverse operation.

        Notes
        -----
        The transpose satisfies ``W.T[j, i] = W[i, j]`` while preserving the
        canonical row-stream generated by ``seed``.

        Examples
        --------

        .. code-block:: python

            >>> from brainevent import JITCNormalC, JITCNormalR
            >>>
            >>> col_matrix = JITCNormalC((0.1, 0.5, 0.2, 42), shape=(3, 5))
            >>> col_matrix.shape
            (3, 5)
            >>> row_matrix = col_matrix.transpose()
            >>> row_matrix.shape
            (5, 3)
            >>> isinstance(row_matrix, JITCNormalR)
            True
        """
        assert axes is None, "transpose does not support axes argument."
        return JITCNormalR(
            (self.wloc, self.wscale, self.prob, self.seed),
            shape=(self.shape[1], self.shape[0]),
            backend=self.backend,
            buffers=self.buffers,
        )

    def _new_mat(self, wloc, wscale, prob=None, seed=None):
        """
        Create a new ``JITCNormalC`` with the given weight parameters, reusing other attributes.

        Parameters
        ----------
        wloc : WeightScalar
            New lower bound for the normal distribution.
        wscale : WeightScalar
            New upper bound for the normal distribution.
        prob : Prob, optional
            New connection probability. If None, the current probability is reused.
        seed : Seed, optional
            New random seed. If None, the current seed is reused.

        Returns
        -------
        JITCNormalC
            A new column-oriented matrix with the specified weight parameters.
        """
        return JITCNormalC(
            (
                wloc,
                wscale,
                self.prob if prob is None else prob,
                self.seed if seed is None else seed
            ),
            shape=self.shape,
            backend=self.backend,
            buffers=self.buffers,
        )

    def _unitary_op(self, op) -> 'JITCNormalC':
        """
        Apply a unary operation to both weight parameters.

        Parameters
        ----------
        op : callable
            A unary function to apply element-wise to ``wloc`` and ``wscale``.

        Returns
        -------
        JITCNormalC
            A new matrix with the operation applied to both bounds.
        """
        return self._new_mat(op(self.wloc), op(self.wscale))

    def _binary_op(self, other, op) -> 'JITCNormalC':
        """
        Apply a binary operation between the weight parameters and a scalar operand.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            The right-hand operand. Must be a scalar (size 1) or another sparse matrix.
        op : callable
            A binary function (e.g., ``operator.mul``) to apply.

        Returns
        -------
        JITCNormalC
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
            return self._new_mat(op(self.wloc, other), op(self.wscale, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCNormalC':
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
        JITCNormalC
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
            return self._new_mat(op(other, self.wloc), op(other, self.wscale))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        """
        Compute matrix multiplication ``self @ other``.

        Internally delegates to the underlying ``JITCNormalR`` representation
        by using a transposed view: ``JITCNormalC @ other == JITCNormalR.T @ other``.
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
                return binary_jitnmv(
                    self.wloc,
                    self.wscale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=True,
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
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        """
        Compute matrix multiplication ``other @ self``.

        Internally delegates to the underlying ``JITCNormalR`` representation:
        ``other @ JITCNormalC == other @ JITCNormalR.T == JITCNormalR @ other``
        for vectors, or ``(JITCNormalR @ other.T).T`` for matrices.
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
                return binary_jitnmv(
                    self.wloc,
                    self.wscale,
                    self.prob,
                    other,
                    self.seed,
                    shape=self.shape[::-1],
                    transpose=False,
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
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
