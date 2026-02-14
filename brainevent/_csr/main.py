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


import operator
from typing import Optional, Union, Sequence, Dict

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._data import DataRepresentation
from brainevent._event import BinaryArray, SparseFloat
from brainevent._misc import _csr_to_coo, _csr_todense
from brainevent._typing import Data, Indptr, Index, MatrixShape
from .binary import binary_csrmv, binary_csrmm
from .diag_add import csr_diag_position_v2, csr_diag_add_v2
from .float import csrmv, csrmm
from .slice import csr_slice_rows
from .sparse_float import spfloat_csrmv, spfloat_csrmm
from .spsolve import csr_solve
from .yw2y import csrmv_yw2y

__all__ = [
    'CSR',
    'CSC',
]


class CompressedSparseData(DataRepresentation):
    """
    Abstract base class for compressed sparse matrix formats.

    ``CompressedSparseData`` provides the common interface shared by :class:`CSR` and
    :class:`CSC`. It inherits from ``brainunit.sparse.SparseMatrix`` and
    adds arithmetic operators, JAX pytree support, and helper methods for
    event-driven neural simulation.

    Subclasses must implement :meth:`apply`, :meth:`_binary_op`,
    :meth:`_binary_rop`, :meth:`todense`, :meth:`tocoo`, :meth:`with_data`,
    :meth:`fromdense`, :meth:`yw_to_w`, and :meth:`yw_to_w_transposed`.

    Parameters
    ----------
    data : array_like or Sequence
        Non-zero values, or a length-3 sequence ``(data, indices, indptr)``.
    indices : array_like, optional
        Secondary-axis indices for each stored element.
    indptr : array_like, optional
        Primary-axis pointers into ``data`` and ``indices``.
    shape : tuple[int, int]
        Matrix shape ``(num_rows, num_columns)``.

    Attributes
    ----------
    data : Data
        Array of stored (non-zero) values.
    indices : Index
        Secondary-axis index array.
    indptr : Indptr
        Primary-axis pointer array.
    shape : tuple[int, int]
        Shape of the full matrix.

    See Also
    --------
    CSR : Compressed Sparse Row implementation.
    CSC : Compressed Sparse Column implementation.
    """

    data: Data
    indices: Index
    indptr: Indptr
    shape: MatrixShape

    def __init__(
        self,
        data,
        indices=None,
        indptr=None,
        *,
        shape: MatrixShape,
        backend: Optional[str] = None,
        buffers: Optional[Dict] = None,
    ):
        if indices is None and indptr is None:
            # Tuple syntax: CSR((data, indices, indptr), shape=...)
            args = data
        else:
            # Positional syntax: CSR(data, indices, indptr, shape=...)
            args = (data, indices, indptr)

        assert len(args) == 3, "Expected three arguments: data, indices, indptr."
        self.data, self.indices, self.indptr = map(u.math.asarray, args)
        self.backend = backend
        super().__init__(args, shape=shape, buffers=buffers)

    @property
    def nse(self):
        """
        Number of stored elements in the sparse matrix.

        This counts all explicitly stored entries, including any stored
        zeros.  It equals ``self.indices.size``.

        Returns
        -------
        int
            The number of stored elements.
        """
        return self.indices.size

    @property
    def dtype(self):
        """
        Data type of the stored values.

        Returns
        -------
        numpy.dtype
            The dtype of ``self.data``.
        """
        return self.data.dtype

    def tree_flatten(self):
        """
        Flatten this sparse matrix into JAX pytree leaves and auxiliary data.

        This method is part of the JAX pytree protocol.  The ``data`` array
        is the only leaf (traced by JAX); ``indices``, ``indptr``, ``shape``,
        and ``diag_positions`` are stored as static auxiliary data.

        Returns
        -------
        children : tuple
            A 1-tuple ``(data,)`` containing the traceable leaf array.
        aux_data : dict
            Dictionary with keys ``'indices'``, ``'indptr'``, ``'shape'``,
            and ``'diag_positions'``.

        See Also
        --------
        tree_unflatten : Reconstruct a sparse matrix from flattened data.
        """
        aux = {
            'indices': self.indices,
            'indptr': self.indptr,
            'shape': self.shape,
            'backend': self.backend,
        }
        aux.update(self._flatten_buffers())
        return (self.data,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a sparse matrix from JAX pytree leaves and auxiliary data.

        This is the inverse of :meth:`tree_flatten` and is called by JAX
        when unflattening a pytree (e.g., after ``jax.jit`` compilation).

        Parameters
        ----------
        aux_data : dict
            Auxiliary data produced by :meth:`tree_flatten`, containing
            ``'indices'``, ``'indptr'``, ``'shape'``, and
            ``'diag_positions'``.
        children : tuple
            A 1-tuple ``(data,)`` containing the leaf array.

        Returns
        -------
        CompressedSparseData
            A new instance of the sparse matrix class with restored
            attributes.

        See Also
        --------
        tree_flatten : Flatten a sparse matrix for JAX pytree handling.
        """
        obj = object.__new__(cls)
        obj.data, = children
        registry = aux_data.pop('_buffer_registry', frozenset())
        obj._buffer_registry = set(registry)
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj

    def apply(self, fn):
        """
        Apply a function to the data and return a new sparse matrix with the same structure.

        Unlike :meth:`with_data`, which requires the new data to have the same
        shape, dtype, and unit, ``apply`` allows transformations that change
        dtype or unit.

        Parameters
        ----------
        fn : callable
            A function to apply to ``self.data``.

        Returns
        -------
        CSR or CSC
            A new sparse matrix with ``fn(self.data)`` and the same structure.
        """
        raise NotImplementedError

    def __abs__(self):
        """
        Return a sparse matrix with element-wise absolute values.

        Computes ``abs(x)`` for every stored element *x* while preserving the
        sparsity structure (indices, indptr, and shape).

        Returns
        -------
        CSR or CSC
            A new sparse matrix whose data values are the absolute values of
            the original stored elements.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([-1.0, 2.0, -3.0]), indices, indptr, shape=(3, 3))
            >>> result = abs(csr)
        """
        return self.apply(operator.abs)

    def __neg__(self):
        """
        Return a sparse matrix with element-wise negation.

        Computes ``-x`` for every stored element *x* while preserving the
        sparsity structure.

        Returns
        -------
        CSR or CSC
            A new sparse matrix whose data values are the negation of the
            original stored elements.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 3.0]), indices, indptr, shape=(3, 3))
            >>> result = -csr
        """
        return self.apply(operator.neg)

    def __pos__(self):
        """
        Return a sparse matrix with the unary positive operator applied.

        Computes ``+x`` for every stored element *x*.  For numeric types this
        is typically the identity, but the operator may trigger type promotion
        for certain array-like objects.

        Returns
        -------
        CSR or CSC
            A new sparse matrix whose data values are ``+self.data``.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 3.0]), indices, indptr, shape=(3, 3))
            >>> result = +csr
        """
        return self.apply(operator.pos)

    def _binary_op(self, other, op):
        raise NotImplementedError

    def apply2(self, other, fn, *, reverse: bool = False):
        """
        Apply a binary function while preserving sparse structure semantics.

        Parameters
        ----------
        other : Any
            Right-hand operand for normal operations, or left-hand operand when
            ``reverse=True``.
        fn : callable
            Binary function from ``operator`` or a compatible callable.
        reverse : bool, optional
            If False, compute ``fn(self, other)`` semantics using ``_binary_op``.
            If True, compute ``fn(other, self)`` semantics using ``_binary_rop``.
            Defaults to False.

        Returns
        -------
        CSR or CSC or Data
            Result of the operation.
        """
        if reverse:
            return self._binary_rop(other, fn)
        return self._binary_op(other, fn)

    def __mul__(self, other: Data):
        """
        Element-wise multiplication: ``self * other``.

        When ``other`` is a scalar, each stored element is multiplied by that
        scalar.  When ``other`` is a dense matrix of the same shape, only the
        values at the stored positions are multiplied.  When ``other`` is a
        sparse matrix with identical structure (same ``indices`` and
        ``indptr`` objects), the data arrays are multiplied directly.

        Parameters
        ----------
        other : Data
            Scalar, dense array, or structurally identical sparse matrix.

        Returns
        -------
        CSR or CSC
            A new sparse matrix containing the element-wise product.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix with different structure, or a
            dense array whose shape is incompatible.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 3.0]), indices, indptr, shape=(3, 3))
            >>> result = csr * 2.0
        """
        return self.apply2(other, operator.mul)

    def __truediv__(self, other):
        """
        Element-wise true division: ``self / other``.

        Divides every stored element by ``other``.  Semantics for scalar,
        dense, and structurally identical sparse operands match those of
        :meth:`__mul__`.

        Parameters
        ----------
        other : Data
            Scalar, dense array, or structurally identical sparse matrix.

        Returns
        -------
        CSR or CSC
            A new sparse matrix containing the element-wise quotient.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix with different structure, or a
            dense array whose shape is incompatible.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([2.0, 4.0, 6.0]), indices, indptr, shape=(3, 3))
            >>> result = csr / 2.0
        """
        return self.apply2(other, operator.truediv)

    def __add__(self, other):
        """
        Element-wise addition: ``self + other``.

        For addition and subtraction the sparse matrix is first converted to a
        dense matrix via :meth:`todense`, so the result is always a dense
        array.

        Parameters
        ----------
        other : array_like
            Dense array with a shape broadcastable to ``self.shape``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Dense result of the addition.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0]), indices, indptr, shape=(3, 3))
            >>> dense = jnp.ones((3, 3))
            >>> result = csr + dense
        """
        return self.apply2(other, operator.add)

    def __sub__(self, other):
        """
        Element-wise subtraction: ``self - other``.

        The sparse matrix is first converted to a dense matrix via
        :meth:`todense`, so the result is always a dense array.

        Parameters
        ----------
        other : array_like
            Dense array with a shape broadcastable to ``self.shape``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Dense result of the subtraction.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0]), indices, indptr, shape=(3, 3))
            >>> dense = jnp.ones((3, 3))
            >>> result = csr - dense
        """
        return self.apply2(other, operator.sub)

    def _binary_rop(self, other, op):
        raise NotImplementedError

    def __rmul__(self, other: Data):
        """
        Reflected element-wise multiplication: ``other * self``.

        Called when the left operand does not support multiplication with this
        sparse type.  Semantics are identical to :meth:`__mul__` because
        multiplication is commutative for scalars and element-wise operations.

        Parameters
        ----------
        other : Data
            Scalar, dense array, or structurally identical sparse matrix.

        Returns
        -------
        CSR or CSC
            A new sparse matrix containing the element-wise product.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix with different structure, or a
            dense array whose shape is incompatible.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 3.0]), indices, indptr, shape=(3, 3))
            >>> result = 2.0 * csr
        """
        return self.apply2(other, operator.mul, reverse=True)

    def __rtruediv__(self, other):
        """
        Reflected element-wise true division: ``other / self``.

        Computes ``other / x`` for every stored element *x*.  Note that this
        is **not** equivalent to ``self / other``; the operand order matters.

        Parameters
        ----------
        other : Data
            Scalar, dense array, or structurally identical sparse matrix.

        Returns
        -------
        CSR or CSC
            A new sparse matrix containing the element-wise quotient.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix with different structure, or a
            dense array whose shape is incompatible.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 4.0]), indices, indptr, shape=(3, 3))
            >>> result = 8.0 / csr  # stored values become [8.0, 4.0, 2.0]
        """
        return self.apply2(other, operator.truediv, reverse=True)

    def __radd__(self, other):
        """
        Reflected element-wise addition: ``other + self``.

        The sparse matrix is first converted to dense, so the result is always
        a dense array.  Because addition is commutative the result equals
        ``self + other``.

        Parameters
        ----------
        other : array_like
            Dense array with a shape broadcastable to ``self.shape``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Dense result of the addition.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0]), indices, indptr, shape=(3, 3))
            >>> dense = jnp.ones((3, 3))
            >>> result = dense + csr
        """
        return self.apply2(other, operator.add, reverse=True)

    def __rsub__(self, other):
        """
        Reflected element-wise subtraction: ``other - self``.

        The sparse matrix is first converted to dense, so the result is always
        a dense array.  Note that ``other - self`` is **not** equivalent to
        ``self - other``.

        Parameters
        ----------
        other : array_like
            Dense array with a shape broadcastable to ``self.shape``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Dense result of the subtraction.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0]), indices, indptr, shape=(3, 3))
            >>> dense = jnp.ones((3, 3))
            >>> result = dense - csr
        """
        return self.apply2(other, operator.sub, reverse=True)

    def yw_to_w(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity]
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute a sparse matrix-vector product mapping y-w space to w space.

        This method is used in event-driven neural simulations to efficiently
        compute the effect of synaptic connections.  Given a per-target array
        ``y_dim_arr`` and a per-synapse weight array ``w_dim_arr``, it
        performs a specialised sparse product that accumulates contributions
        along the connectivity defined by this matrix.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the weight / synapse dimension.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Result of the sparse transformation, preserving physical units
            when present.
        """
        raise NotImplementedError

    def yw_to_w_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity]
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute the transposed sparse matrix-vector product mapping y-w space to w space.

        This is the adjoint of :meth:`yw_to_w`.  It is useful for
        back-propagation or adjoint computations in event-driven neural
        simulations.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the weight / synapse dimension.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Result of the transposed sparse transformation, preserving
            physical units when present.

        See Also
        --------
        yw_to_w : The forward (non-transposed) variant.
        """
        raise NotImplementedError

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=jnp.int32):
        """
        Create a compressed sparse matrix from a dense matrix.

        Parameters
        ----------
        mat : array_like
            The dense matrix to convert.
        nse : int, optional
            Number of stored (non-zero) elements.  If ``None`` it is
            inferred from ``mat``.
        index_dtype : dtype, optional
            Data type for index arrays.  Defaults to ``jnp.int32``.

        Returns
        -------
        CSR or CSC
            A new sparse matrix in the appropriate compressed format.
        """
        raise NotImplementedError

    def with_data(self, data: Data):
        """
        Create a new sparse matrix with the same structure but different data.

        Unlike :meth:`apply`, the new ``data`` must have the same shape,
        dtype, and unit as the original.

        Parameters
        ----------
        data : Data
            Replacement data array.

        Returns
        -------
        CSR or CSC
            New sparse matrix sharing indices and indptr with this instance.

        Raises
        ------
        AssertionError
            If the shape, dtype, or unit of ``data`` does not match the
            original.
        """
        raise NotImplementedError

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the sparse matrix to a dense array.

        Returns
        -------
        jax.Array or brainunit.Quantity
            A dense ``(num_rows, num_columns)`` matrix equivalent to this
            sparse matrix.
        """
        raise NotImplementedError

    def tocoo(self):
        """
        Convert the sparse matrix to COO (Coordinate) format.

        Returns
        -------
        COO
            A :class:`~brainevent.COO` matrix containing the same data.

        See Also
        --------
        brainevent.COO : The Coordinate sparse matrix class.
        """
        raise NotImplementedError

    def diag_add(self, other):
        """
        Add values to the matrix diagonal and return a new sparse matrix.

        This method adds the provided diagonal value to the diagonal elements of the
        sparse matrix represented in Compressed Sparse Row (CSR) format. If the diagonal
        positions have not been computed yet, it will first calculate them.

        Parameters
        ----------
        other : array-like
            The diagonal value to be added to the sparse matrix. It should be compatible
            with the data type of the matrix's non-zero elements.

        Raises
        ------
        AssertionError
            If `other` is an instance of `JAXSparse`, as this operation does not support
            `JAXSparse` objects.

        Notes
        -----
        - The diagonal positions are computed only once and cached in the `diag_positions`
          attribute of the matrix instance.
        - This method relies on `csr_diag_position_v2` to find diagonal positions and
          `csr_diag_add_v2` to perform the actual addition.
        """
        if not hasattr(self, 'diag_positions'):
            self.register_buffer(
                'diag_positions',
                csr_diag_position_v2(self.indptr, self.indices, shape=self.shape)
            )
        assert not isinstance(other, u.sparse.SparseMatrix), "diag_add does not support JAXSparse objects."
        return self.with_data(csr_diag_add_v2(self.data, self.diag_positions, other))

    def solve(self, b: Union[jax.Array, u.Quantity]) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system ``A x = b`` where ``A`` is this sparse matrix.

        Parameters
        ----------
        b : jax.Array or brainunit.Quantity
            Right-hand side vector of the linear system.  Its first dimension
            must match ``self.shape[0]``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Solution vector *x* satisfying ``A x = b``.

        Raises
        ------
        AssertionError
            If the first dimension of ``b`` does not match the number of rows
            in the matrix.
        """
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class CSR(CompressedSparseData):
    """
    Event-driven and Unit-aware Compressed Sparse Row (CSR) matrix.

    This class represents a sparse matrix in CSR format, which is efficient for
    row-wise operations and matrix-vector multiplications. It is compatible with
    JAX's tree utilities and supports unit-aware computations.

    The class supports arithmetic with scalars and dense arrays, plus sparse-dense
    matrix multiplication via ``@``. Sparse-sparse operations are limited.

    Attributes
    ----------
    data : Data
        Array of the non-zero values in the matrix.
    indices : jax.Array
        Array of column indices for the non-zero values.
    indptr : jax.Array
        Array of row pointers indicating where each row starts in the data and indices arrays.
    shape : tuple[int, int]
        The shape of the matrix as (rows, columns).
    nse : int
        Number of stored elements (non-zero entries).
    dtype : dtype
        Data type of the matrix values.

    Notes
    -----
    In CSR format a matrix of shape ``(m, n)`` is stored as three arrays:

    * ``indptr`` of length ``m + 1`` -- the *i*-th row occupies entries
      ``indptr[i]`` to ``indptr[i+1]`` in the ``data`` and ``indices``
      arrays.
    * ``indices`` -- column indices of the stored elements.
    * ``data`` -- the corresponding non-zero values.

    The ``@`` operator dispatches to optimised kernels depending on the
    right-hand operand type:

    * :class:`~brainevent.BinaryArray` -- event-driven binary CSR MV/MM.
    * :class:`~brainevent.SparseFloat` -- sparse-float CSR MV/MM.
    * Dense ``jax.Array`` / ``brainunit.Quantity`` -- standard float CSR
      MV/MM with automatic dtype promotion.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import brainevent

        data    = jnp.array([1.0, 2.0, 3.0])
        indices = jnp.array([0, 2, 1])
        indptr  = jnp.array([0, 1, 2, 3])
        csr     = brainevent.CSR((data, indices, indptr), shape=(3, 3))

        # Sparse-dense matrix-vector product
        x = jnp.ones(3)
        y = csr @ x

    See Also
    --------
    CSC : Compressed Sparse Column format.
    brainevent.COO : Coordinate sparse format.
    """
    __module__ = 'brainevent'

    @classmethod
    def fromdense(
        cls,
        mat,
        *,
        nse: Optional[int] = None,
        index_dtype=jnp.int32,
        backend: Optional[str] = None,
    ) -> 'CSR':
        """
        Create a CSR matrix from a dense matrix.

        This method converts a dense matrix to a Compressed Sparse Row (CSR) format.

        Parameters
        ----------
        mat : array_like
            The dense matrix to be converted to CSR format.
        nse : int, optional
            The number of non-zero elements in the matrix. If None, it will be
            calculated from the input matrix.
        index_dtype : dtype, optional
            The data type to be used for index arrays (default is jnp.int32).

        Returns
        -------
        CSR
            A new CSR matrix object created from the input dense matrix.

        Examples
        --------
        .. code-block:: python

            import jax.numpy as jnp
            import brainevent

            dense = jnp.array([[1.0, 0.0], [0.0, 2.0]])
            csr = brainevent.CSR.fromdense(dense)
        """
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csr = u.sparse.csr_fromdense(mat, nse=nse, index_dtype=index_dtype)
        return CSR(csr.data, csr.indices, csr.indptr, shape=csr.shape, backend=backend)

    def with_data(self, data: Data) -> 'CSR':
        """
        Create a new CSR matrix with updated data while keeping the same structure.

        This method creates a new CSR matrix instance with the provided data,
        maintaining the original indices, indptr, and shape.

        Parameters
        ----------
        data : Data
            The new data array to replace the existing data in the CSR matrix.
            It must have the same shape, dtype, and unit as the original data.

        Returns
        -------
        CSR
            A new CSR matrix instance with updated data and the same structure as the original.

        Raises
        ------
        AssertionError
            If the shape, dtype, or unit of the new data doesn't match the original data.

        Examples
        --------
        .. code-block:: python

            new_data = jnp.array([10.0, 20.0, 30.0])
            new_csr = csr.with_data(new_data)
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return CSR(
            (data, self.indices, self.indptr),
            shape=self.shape,
            buffers=self._flatten_buffers(),
            backend=self.backend,
        )

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the CSR matrix to a dense matrix.

        This method transforms the compressed sparse row (CSR) representation
        into a full dense matrix.

        Returns
        -------
        jax.Array or brainunit.Quantity
            A dense matrix of shape ``self.shape`` containing all the values
            (including zeros) of the sparse matrix.

        Examples
        --------
        .. code-block:: python

            dense = csr.todense()
        """
        return _csr_todense(self.data, self.indices, self.indptr, shape=self.shape)

    def tocoo(self):
        """
        Convert the CSR matrix to COO (Coordinate) format.

        This method transforms the Compressed Sparse Row (CSR) matrix into a COO matrix,
        which stores sparse data as a collection of (row, column, value) triplets.

        Returns
        -------
        COO
            A COO matrix containing the same data as the original CSR matrix.

        See Also
        --------
        brainevent.COO : The Coordinate sparse matrix class.

        Examples
        --------
        .. code-block:: python

            csr_matrix = CSR((data, indices, indptr), shape=(3, 4))
            coo_matrix = csr_matrix.tocoo()
        """
        from brainevent import COO
        pre_ids, post_ids = _csr_to_coo(self.indices, self.indptr)
        return COO((self.data, pre_ids, post_ids), shape=self.shape, backend=self.backend)

    def transpose(self, axes=None) -> 'CSC':
        """
        Transpose the CSR matrix.

        This method returns the transpose of the CSR matrix as a CSC matrix.
        Because the transpose of a CSR matrix is a CSC matrix with the same
        underlying arrays, this operation is essentially free (no data is
        copied or rearranged).

        Parameters
        ----------
        axes : None
            This parameter is not used and must be None. Included for compatibility
            with numpy's transpose function signature.

        Returns
        -------
        CSC
            The transpose of the CSR matrix as a CSC (Compressed Sparse Column) matrix.

        Raises
        ------
        AssertionError
            If axes is not None, as this implementation doesn't support custom axis ordering.

        Examples
        --------
        .. code-block:: python

            csc = csr.transpose()
            # or equivalently:
            csc = csr.T
        """
        assert axes is None, "transpose does not support axes argument."
        return CSC(
            self.data, self.indices, self.indptr,
            shape=self.shape[::-1],
            buffers=self._flatten_buffers(),
            backend=self.backend
        )

    def apply(self, fn) -> 'CSR':
        """
        Apply a unary function to the stored data values.

        Creates a new :class:`CSR` matrix with ``fn(self.data)`` while
        preserving the sparsity structure (indices, indptr, shape, and cached
        diagonal positions).

        Parameters
        ----------
        fn : callable
            A function that accepts a single array argument and returns an
            array of the same shape.  The dtype and unit may differ from the
            input.

        Returns
        -------
        CSR
            A new CSR matrix with transformed data.

        Examples
        --------
        .. code-block:: python

            squared = csr.apply(lambda x: x ** 2)
        """
        return CSR(
            fn(self.data), self.indices, self.indptr,
            shape=self.shape,
            buffers=self._flatten_buffers(),
            backend=self.backend,
        )

    def __getitem__(self, index):
        """Extract rows from the CSR matrix as a dense array.

        Parameters
        ----------
        index : int, tuple, list, or array
            Row index or indices to extract.

        Returns
        -------
        jax.Array or brainunit.Quantity
            For a single integer index, a 1-D dense vector of length
            ``n_cols``. For multiple indices, a 2-D dense matrix of shape
            ``(len(index), n_cols)``.
        """
        if isinstance(index, (int, np.integer)):
            row_indices = jnp.array(index, dtype=jnp.int32)
        elif isinstance(index, (tuple, list)):
            row_indices = jnp.asarray(index, dtype=jnp.int32)
        elif isinstance(index, (jnp.ndarray, np.ndarray)):
            row_indices = jnp.asarray(index, dtype=jnp.int32)
        else:
            raise IndexError(f"Unsupported index type: {type(index)}")
        return csr_slice_rows(
            self.data, self.indices, self.indptr, row_indices, shape=self.shape,
            backend=self.backend
        )

    def _binary_op(self, other, op) -> 'CSR':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(dense, other)

        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    op(self.data, other.data,
                       self.indices,
                       self.indptr),
                    shape=self.shape,
                    buffers=self._flatten_buffers(),
                    backend=self.backend,
                )
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR(
                op(self.data, other), self.indices, self.indptr,
                shape=self.shape,
                buffers=self._flatten_buffers(),
                backend=self.backend,
            )

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                op(self.data, other),
                self.indices,
                self.indptr,
                shape=self.shape,
                buffers=self._flatten_buffers(),
                backend=self.backend,
            )

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'CSR':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(other, dense)

        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    op(other.data, self.data),
                    self.indices,
                    self.indptr,
                    shape=self.shape,
                    buffers=self._flatten_buffers(),
                    backend=self.backend,
                )
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR(
                op(other, self.data),
                self.indices,
                self.indptr,
                shape=self.shape,
                buffers=self._flatten_buffers(),
                backend=self.backend,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                op(other, self.data),
                self.indices,
                self.indptr,
                shape=self.shape,
                buffers=self._flatten_buffers(),
                backend=self.backend,
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        """
        Sparse-dense matrix multiplication: ``self @ other``.

        Dispatches to an optimised kernel based on the type and
        dimensionality of ``other``:

        * 1-D array -- sparse matrix-vector product (MV).
        * 2-D array -- sparse matrix-matrix product (MM).
        * :class:`~brainevent.BinaryArray` -- event-driven binary kernel.
        * :class:`~brainevent.SparseFloat` -- sparse-float kernel.
        * Dense ``jax.Array`` / ``brainunit.Quantity`` -- standard float
          kernel with automatic dtype promotion.

        Parameters
        ----------
        other : jax.Array, brainunit.Quantity, BinaryArray, or SparseFloat
            The right-hand operand.  Must be 1-D or 2-D.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The result of the matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix (sparse-sparse matmul is not
            supported) or if ``other`` has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            x = jnp.ones(n)
            y = csr @ x          # matrix-vector

            X = jnp.ones((n, k))
            Y = csr @ X          # matrix-matrix
        """
        # csr @ other
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                return binary_csrmv(self.data, self.indices, self.indptr, other,
                                    shape=self.shape, backend=self.backend)
            elif other.ndim == 2:
                return binary_csrmm(self.data, self.indices, self.indptr, other,
                                    shape=self.shape, backend=self.backend)
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        elif isinstance(other, SparseFloat):
            other = other.value
            if other.ndim == 1:
                return spfloat_csrmv(self.data, self.indices, self.indptr, other,
                                     shape=self.shape, backend=self.backend)
            elif other.ndim == 2:
                return spfloat_csrmm(self.data, self.indices, self.indptr, other,
                                     shape=self.shape, backend=self.backend)
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=False,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                return csrmm(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=False,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        """
        Reflected sparse-dense matrix multiplication: ``other @ self``.

        Computes the product with ``self`` on the right by using the
        transposed CSR kernel.  Dispatch logic mirrors :meth:`__matmul__`.

        Parameters
        ----------
        other : jax.Array, brainunit.Quantity, BinaryArray, or SparseFloat
            The left-hand operand.  Must be 1-D or 2-D.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The result of the matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            x = jnp.ones(m)
            y = x @ csr          # vector-matrix

            X = jnp.ones((k, m))
            Y = X @ csr          # matrix-matrix
        """
        # other @ csr
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                return binary_csrmv(self.data, self.indices, self.indptr, other,
                                    shape=self.shape, transpose=True, backend=self.backend)
            elif other.ndim == 2:
                other = other.T
                r = binary_csrmm(self.data, self.indices, self.indptr, other,
                                 shape=self.shape, transpose=True, backend=self.backend)
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        elif isinstance(other, SparseFloat):
            other = other.value
            if other.ndim == 1:
                return spfloat_csrmv(
                    self.data, self.indices, self.indptr, other,
                    shape=self.shape,
                    transpose=True,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                other = other.T
                r = spfloat_csrmm(
                    self.data, self.indices, self.indptr, other,
                    shape=self.shape,
                    transpose=True,
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=True,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                other = other.T
                r = csrmm(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=True,
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def solve(self, b: Union[jax.Array, u.Quantity], tol=1e-6, reorder=1) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system ``A x = b`` where ``A`` is this CSR matrix.

        Uses a sparse direct solver via the underlying ``csr_solve`` routine.

        Parameters
        ----------
        b : jax.Array or brainunit.Quantity
            Right-hand side vector.  Its first dimension must equal
            ``self.shape[0]``.
        tol : float, optional
            Tolerance for singularity detection.  Defaults to ``1e-6``.
        reorder : int, optional
            Fill-reducing reordering scheme: ``0`` for no reordering,
            ``1`` for symrcm, ``2`` for symamd, ``3`` for csrmetisnd.
            Defaults to ``1``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Solution vector *x* satisfying ``A x = b``.

        Raises
        ------
        AssertionError
            If ``b.shape[0] != self.shape[0]``.

        Examples
        --------
        .. code-block:: python

            x = csr.solve(b)
        """
        assert self.shape[0] == b.shape[0], ("The number of rows in the matrix must match "
                                             "the size of the right-hand side vector b.")
        return csr_solve(self.data, self.indices, self.indptr, b)

    def yw_to_w(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute a sparse transformation from y-w space to w space.

        Performs a specialised sparse matrix-vector product optimised for
        event-driven neural simulations, accumulating contributions from the
        target (post-synaptic) dimension ``y_dim_arr`` weighted by the
        per-synapse values ``w_dim_arr`` according to the connectivity
        defined by this CSR matrix.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Accumulated result, preserving physical units when present.

        See Also
        --------
        yw_to_w_transposed : The transposed (adjoint) variant.

        Notes
        -----
        Internally calls ``csrmv_yw2y`` with ``transpose=False``.
        """
        return csrmv_yw2y(y_dim_arr, w_dim_arr, self.indices, self.indptr,
                          shape=self.shape, transpose=False, backend=self.backend)

    def yw_to_w_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute the transposed sparse transformation from y-w space to w space.

        This is the adjoint of :meth:`yw_to_w`, useful for back-propagation
        or adjoint computations in event-driven neural simulations.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Accumulated result of the transposed operation, preserving
            physical units when present.

        See Also
        --------
        yw_to_w : The forward (non-transposed) variant.

        Notes
        -----
        Internally calls ``csrmv_yw2y`` with ``transpose=True``.
        """
        return csrmv_yw2y(y_dim_arr, w_dim_arr, self.indices, self.indptr,
                          shape=self.shape, transpose=True, backend=self.backend)


@jax.tree_util.register_pytree_node_class
class CSC(CompressedSparseData):
    """
    Event-driven and Unit-aware Compressed Sparse Column (CSC) matrix.

    This class represents a sparse matrix in CSC format, which is efficient for
    column-wise operations. It is compatible with JAX's tree utilities and
    supports unit-aware computations.

    The class supports arithmetic with scalars and dense arrays, plus sparse-dense
    matrix multiplication via ``@``. Sparse-sparse operations are limited.

    Attributes
    ----------
    data : Data
        Array of the non-zero values in the matrix.
    indices : jax.Array
        Array of row indices for the non-zero values.
    indptr : jax.Array
        Array of column pointers indicating where each column starts in the data and indices arrays.
    shape : tuple[int, int]
        The shape of the matrix as (rows, columns).
    nse : int
        Number of stored elements (non-zero entries).
    dtype : dtype
        Data type of the matrix values.

    Notes
    -----
    In CSC format a matrix of shape ``(m, n)`` is stored as three arrays:

    * ``indptr`` of length ``n + 1`` -- the *j*-th column occupies entries
      ``indptr[j]`` to ``indptr[j+1]`` in the ``data`` and ``indices``
      arrays.
    * ``indices`` -- row indices of the stored elements.
    * ``data`` -- the corresponding non-zero values.

    Internally, CSC operations are implemented by treating the underlying
    arrays as a CSR matrix with transposed shape and applying the appropriate
    transpose flags to the CSR kernels.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import brainevent

        data    = jnp.array([1.0, 2.0, 3.0])
        indices = jnp.array([0, 2, 1])
        indptr  = jnp.array([0, 1, 2, 3])
        csc     = brainevent.CSC((data, indices, indptr), shape=(3, 3))

        # Sparse-dense matrix-vector product
        x = jnp.ones(3)
        y = csc @ x

    See Also
    --------
    CSR : Compressed Sparse Row format.
    brainevent.COO : Coordinate sparse format.
    """
    __module__ = 'brainevent'

    @classmethod
    def fromdense(
        cls,
        mat,
        *,
        nse: int = None,
        index_dtype=jnp.int32,
        backend: Optional[str] = None,
    ) -> 'CSC':
        """
        Create a CSC (Compressed Sparse Column) matrix from a dense matrix.

        This method converts a dense matrix to CSC format, which is an efficient
        storage format for sparse matrices.

        Parameters
        ----------
        mat : array_like
            The dense matrix to be converted to CSC format.
        nse : int, optional
            The number of non-zero elements in the matrix. If None, it will be
            calculated from the input matrix.
        index_dtype : dtype, optional
            The data type to be used for index arrays (default is jnp.int32).

        Returns
        -------
        CSC
            A new CSC matrix instance created from the input dense matrix.

        Examples
        --------
        .. code-block:: python

            import jax.numpy as jnp
            import brainevent

            dense = jnp.array([[1.0, 0.0], [0.0, 2.0]])
            csc = brainevent.CSC.fromdense(dense)
        """
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csc = u.sparse.csr_fromdense(mat.T, nse=nse, index_dtype=index_dtype).T
        return CSC((csc.data, csc.indices, csc.indptr), shape=csc.shape, backend=backend)

    def with_data(self, data: Data) -> 'CSC':
        """
        Create a new CSC matrix with updated data while keeping the same structure.

        This method creates a new CSC matrix instance with the provided data,
        maintaining the original indices, indptr, and shape.

        Parameters
        ----------
        data : Data
            The new data array to replace the existing data in the CSC matrix.
            It must have the same shape, dtype, and unit as the original data.

        Returns
        -------
        CSC
            A new CSC matrix instance with updated data and the same structure as the original.

        Raises
        ------
        AssertionError
            If the shape, dtype, or unit of the new data doesn't match the original data.

        Examples
        --------
        .. code-block:: python

            new_data = jnp.array([10.0, 20.0, 30.0])
            new_csc = csc.with_data(new_data)
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return CSC((data, self.indices, self.indptr),
                   shape=self.shape,
                   buffers=self._flatten_buffers(),
                   backend=self.backend)

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the CSC matrix to a dense matrix.

        Transposes the underlying CSR-style storage, converts to dense, and
        transposes back.

        Returns
        -------
        jax.Array or brainunit.Quantity
            A dense matrix of shape ``self.shape`` containing all the values
            (including zeros) of the sparse matrix.

        Examples
        --------
        .. code-block:: python

            dense = csc.todense()
        """
        return self.T.todense().T

    def tocoo(self):
        """
        Convert the CSC matrix to COO (Coordinate) format.

        This method transforms the Compressed Sparse Column (CSC) matrix into a COO matrix,
        which stores sparse data as a collection of (row, column, value) triplets.

        Returns
        -------
        COO
            A COO matrix containing the same data as the original CSC matrix.

        See Also
        --------
        brainevent.COO : The Coordinate sparse matrix class.

        Examples
        --------
        .. code-block:: python

            csc_matrix = CSC((data, indices, indptr), shape=(3, 4))
            coo_matrix = csc_matrix.tocoo()
        """
        from brainevent import COO
        post_ids, pre_ids = _csr_to_coo(self.indices, self.indptr)
        return COO((self.data, pre_ids, post_ids), shape=self.shape, backend=self.backend)

    def transpose(self, axes=None) -> 'CSR':
        """
        Transpose the CSC matrix.

        Returns the transpose as a :class:`CSR` matrix.  Because the
        transpose of a CSC matrix is a CSR matrix with the same underlying
        arrays, this operation is essentially free.

        Parameters
        ----------
        axes : None
            Must be ``None``.  Included for API compatibility with NumPy.

        Returns
        -------
        CSR
            The transpose of the CSC matrix as a CSR (Compressed Sparse Row) matrix.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``.

        Examples
        --------
        .. code-block:: python

            csr = csc.transpose()
            # or equivalently:
            csr = csc.T
        """
        assert axes is None
        return CSR((self.data, self.indices, self.indptr),
                   shape=self.shape[::-1],
                   buffers=self._flatten_buffers(),
                   backend=self.backend)

    def apply(self, fn) -> 'CSC':
        """
        Apply a unary function to the stored data values.

        Creates a new :class:`CSC` matrix with ``fn(self.data)`` while
        preserving the sparsity structure (indices, indptr, shape, and cached
        diagonal positions).

        Parameters
        ----------
        fn : callable
            A function that accepts a single array argument and returns an
            array of the same shape.  The dtype and unit may differ from the
            input.

        Returns
        -------
        CSC
            A new CSC matrix with transformed data.

        Examples
        --------
        .. code-block:: python

            squared = csc.apply(lambda x: x ** 2)
        """
        return CSC((fn(self.data), self.indices, self.indptr),
                   shape=self.shape, buffers=self._flatten_buffers(), backend=self.backend)

    def __getitem__(self, index):
        """Extract columns from the CSC matrix as a dense array.

        Parameters
        ----------
        index : int, tuple, list, or array
            Column index or indices to extract.

        Returns
        -------
        jax.Array or brainunit.Quantity
            For a single integer index, a 1-D dense vector of length
            ``n_rows``. For multiple indices, a 2-D dense matrix of shape
            ``(n_rows, len(index))``.
        """
        # CSC stores columns as "rows" of the transposed CSR.
        # csr_slice_rows on the transposed shape gives us (num_selected, n_rows).
        # We transpose the result to get (n_rows, num_selected).
        transposed_shape = self.shape[::-1]
        if isinstance(index, (int, np.integer)):
            col_indices = jnp.array(index, dtype=jnp.int32)
        elif isinstance(index, (tuple, list)):
            col_indices = jnp.asarray(index, dtype=jnp.int32)
        elif isinstance(index, (jnp.ndarray, np.ndarray)):
            col_indices = jnp.asarray(index, dtype=jnp.int32)
        else:
            raise IndexError(f"Unsupported index type: {type(index)}")
        result = csr_slice_rows(
            self.data, self.indices, self.indptr, col_indices,
            shape=transposed_shape, backend=self.backend
        )
        return result if col_indices.ndim == 0 else result.T

    def _binary_op(self, other, op) -> 'CSC':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(dense, other)
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(self.data, other.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape,
                    buffers=self._flatten_buffers(),
                    backend=self.backend,
                )
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape,
                buffers=self._flatten_buffers(),
                backend=self.backend,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape,
                buffers=self._flatten_buffers(),
                backend=self.backend,
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'CSC':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(other, dense)
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(other.data, self.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape,
                    buffers=self._flatten_buffers(),
                    backend=self.backend,
                )
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape,
                buffers=self._flatten_buffers(),
                backend=self.backend,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape,
                buffers=self._flatten_buffers(),
                backend=self.backend,
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        """
        Sparse-dense matrix multiplication: ``self @ other``.

        Dispatches to an optimised kernel based on the type and
        dimensionality of ``other``.  Internally the CSC storage is treated
        as a transposed CSR matrix, so the CSR kernels are called with
        ``shape=self.shape[::-1]`` and ``transpose=True``.

        Parameters
        ----------
        other : jax.Array, brainunit.Quantity, BinaryArray, or SparseFloat
            The right-hand operand.  Must be 1-D or 2-D.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The result of the matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            x = jnp.ones(n)
            y = csc @ x          # matrix-vector

            X = jnp.ones((n, k))
            Y = csc @ X          # matrix-matrix
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                return binary_csrmv(
                    data, self.indices, self.indptr, other,
                    shape=self.shape[::-1],
                    transpose=True,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                return binary_csrmm(
                    data, self.indices, self.indptr, other,
                    shape=self.shape[::-1],
                    transpose=True,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        elif isinstance(other, SparseFloat):
            other = other.value
            if other.ndim == 1:
                return spfloat_csrmv(
                    data, self.indices, self.indptr, other,
                    shape=self.shape[::-1],
                    transpose=True,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                return spfloat_csrmm(
                    data, self.indices, self.indptr, other,
                    shape=self.shape[::-1],
                    transpose=True,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:

            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=True,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                return csrmm(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=True,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        """
        Reflected sparse-dense matrix multiplication: ``other @ self``.

        Computes the product with ``self`` on the right.  Internally the CSC
        storage is treated as a transposed CSR matrix with
        ``transpose=False``.

        Parameters
        ----------
        other : jax.Array, brainunit.Quantity, BinaryArray, or SparseFloat
            The left-hand operand.  Must be 1-D or 2-D.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The result of the matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            x = jnp.ones(m)
            y = x @ csc          # vector-matrix

            X = jnp.ones((k, m))
            Y = X @ csc          # matrix-matrix
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                return binary_csrmv(data, self.indices, self.indptr, other,
                                    shape=self.shape[::-1], transpose=False, backend=self.backend)
            elif other.ndim == 2:
                return binary_csrmm(data, self.indices, self.indptr, other.T,
                                    shape=self.shape[::-1], transpose=False, backend=self.backend).T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        elif isinstance(other, SparseFloat):
            other = other.value
            if other.ndim == 1:
                return spfloat_csrmv(data, self.indices, self.indptr, other,
                                     shape=self.shape[::-1], transpose=False, backend=self.backend)
            elif other.ndim == 2:
                return spfloat_csrmm(data, self.indices, self.indptr, other.T,
                                     shape=self.shape[::-1], transpose=False, backend=self.backend).T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=False,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                other = other.T
                r = csrmm(
                    data,
                    self.indices,
                    self.indptr, other,
                    shape=self.shape[::-1],
                    transpose=False,
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def solve(self, b: Union[jax.Array, u.Quantity], tol=1e-6, reorder=1) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system ``A x = b`` where ``A`` is this CSC matrix.

        Delegates to the CSR solver by transposing the matrix.

        Parameters
        ----------
        b : jax.Array or brainunit.Quantity
            Right-hand side vector.  Its first dimension must equal
            ``self.shape[0]``.
        tol : float, optional
            Tolerance for singularity detection.  Defaults to ``1e-6``.
        reorder : int, optional
            Fill-reducing reordering scheme: ``0`` for no reordering,
            ``1`` for symrcm, ``2`` for symamd, ``3`` for csrmetisnd.
            Defaults to ``1``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Solution vector *x* satisfying ``A x = b``.

        Raises
        ------
        AssertionError
            If ``b.shape[0] != self.shape[0]``.

        Examples
        --------
        .. code-block:: python

            x = csc.solve(b)
        """
        assert self.shape[0] == b.shape[0], ("The number of rows in the matrix must match "
                                             "the size of the right-hand side vector b.")
        return self.T.solve(b, tol=tol, reorder=reorder)

    def yw_to_w(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute a sparse transformation from y-w space to w space.

        Performs a specialised sparse matrix-vector product optimised for
        event-driven neural simulations.  The CSC storage is treated as a
        transposed CSR matrix, so this method calls the CSR kernel with
        ``shape=self.shape[::-1]`` and ``transpose=True``.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Accumulated result, preserving physical units when present.

        See Also
        --------
        yw_to_w_transposed : The transposed (adjoint) variant.

        Notes
        -----
        Internally calls ``csrmv_yw2y`` with ``transpose=True`` and reversed
        shape to account for the column-oriented storage format.
        """
        return csrmv_yw2y(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape[::-1], transpose=True,
                          backend=self.backend)

    def yw_to_w_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute the transposed sparse transformation from y-w space to w space.

        This is the adjoint of :meth:`yw_to_w`, useful for back-propagation
        or adjoint computations in event-driven neural simulations.  Uses
        ``transpose=False`` with the reversed shape to compute the
        appropriate transposed operation for CSC storage.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Accumulated result of the transposed operation, preserving
            physical units when present.

        See Also
        --------
        yw_to_w : The forward (non-transposed) variant.

        Notes
        -----
        Internally calls ``csrmv_yw2y`` with ``transpose=False`` and reversed
        shape.
        """
        return csrmv_yw2y(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape[::-1], transpose=False,
                          backend=self.backend)
