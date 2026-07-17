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
from dataclasses import dataclass
from typing import Optional, Union, Sequence, Dict, cast

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._data import DataRepresentation
from brainevent._event import BinaryArray
from brainevent._misc import (
    _as_indptr,
    _as_int32_indices,
    _check_compressed_structure,
    _csr_to_coo,
    _csr_todense,
    csr_to_csc_index,
    csc_to_csr_index,
    normalize_row_index,
    build_sub_csr,
)
from brainevent._typing import ArrayData, Data, Indptr, Index, MatrixShape
from .binary import binary_csrmv, binary_csrmm
from .binary_indexed import binary_csrmv_indexed, binary_csrmm_indexed
from .diag_add import csr_diag_position, csr_diag_add
from .float import csrmv, csrmm
from .plasticity_binary import update_csr_on_binary_pre, update_csr_on_binary_post
from .slice import csr_slice_rows
from .spsolve import csr_solve
from .dt2t import csrmv_dt2t

__all__ = [
    'CSR',
    'CSC',
]





def _binary_task_capacity_from_indptr(indptr) -> int:
    indptr_np = np.asarray(jax.device_get(indptr), dtype=np.int64)
    if indptr_np.ndim != 1:
        raise ValueError(f"indptr must be one-dimensional, got shape={indptr_np.shape}.")
    if indptr_np.size == 0:
        raise ValueError("indptr must contain at least one element.")
    row_lengths = np.diff(indptr_np)
    if np.any(row_lengths < 0):
        raise ValueError("CSR row lengths must be non-negative.")
    
    _BINARY_TASK_TPR_THRESHOLD = 128

    _BINARY_TASK_NNZ = 4096

    chunks = np.where(
        row_lengths > _BINARY_TASK_TPR_THRESHOLD,
        (row_lengths + _BINARY_TASK_NNZ - 1) // _BINARY_TASK_NNZ,
        0,
    )
    task_capacity = int(chunks.sum())
    if task_capacity > np.iinfo(np.int32).max:
        raise ValueError("binary task capacity exceeds int32 range.")
    return task_capacity


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _BinaryTaskWorkspace:
    task_capacity: int
    task_begin: jax.Array
    task_end: jax.Array
    status: jax.Array

    def tree_flatten(self):
        return (self.task_begin, self.task_end, self.status), {"task_capacity": self.task_capacity}

    @classmethod
    def tree_unflatten(cls, aux, children):
        task_begin, task_end, status = children
        return cls(
            task_capacity=int(aux["task_capacity"]),
            task_begin=task_begin,
            task_end=task_end,
            status=status,
        )


def _make_binary_task_workspace(indptr) -> _BinaryTaskWorkspace:
    task_capacity = _binary_task_capacity_from_indptr(indptr)
    task_dtype = jnp.dtype(indptr.dtype)
    return _BinaryTaskWorkspace(
        task_capacity=task_capacity,
        task_begin=jnp.empty((task_capacity,), dtype=task_dtype),
        task_end=jnp.empty((task_capacity,), dtype=task_dtype),
        status=jnp.empty((2,), dtype=jnp.int32),
    )




def _binary_workspace_helpers(buffer_name: str):
    def workspace_map_from(buffers):
        return dict(buffers.get(buffer_name, {}) or {})

    def buffers_with_workspace(buffers, workspace_map):
        buffers = dict(buffers)
        if workspace_map:
            buffers[buffer_name] = workspace_map
        else:
            buffers.pop(buffer_name, None)
        return buffers

    def copy_matrix_with_buffers(matrix, buffers):
        return type(matrix)(
            (matrix.data, matrix.indices, matrix.indptr),
            shape=matrix.shape,
            buffers=buffers,
            backend=matrix.backend,
            indptr_dtype=matrix.indptr.dtype,
        )

    def split(buffers):
        buffers = dict(buffers)
        workspace_map = workspace_map_from(buffers)
        buffers.pop(buffer_name, None)
        workspace_items = tuple(sorted(workspace_map.items()))
        workspace_keys = tuple(key for key, _ in workspace_items)
        workspace_leaves = tuple(workspace for _, workspace in workspace_items)
        return buffers, workspace_keys, workspace_leaves

    def restore(static_buffers, workspace_keys, workspace_leaves):
        workspace_map = {
            key: workspace for key, workspace in zip(workspace_keys, workspace_leaves)
        }
        return buffers_with_workspace(static_buffers, workspace_map)

    def get(matrix, key: str) -> _BinaryTaskWorkspace:
        workspace_map = workspace_map_from(matrix.buffers)
        try:
            return workspace_map[key]
        except KeyError as exc:
            raise ValueError(f"binary task workspace {key!r} is not prepared") from exc

    def with_workspace(matrix, key: str, workspace: _BinaryTaskWorkspace):
        workspace_map = workspace_map_from(matrix.buffers)
        workspace_map[key] = workspace
        buffers = buffers_with_workspace(matrix.buffers, workspace_map)
        return copy_matrix_with_buffers(matrix, buffers)

    def move_key(buffers, src: str, dst: str):
        workspace_map = workspace_map_from(buffers)
        workspace = workspace_map.pop(src, None)
        if workspace is not None:
            workspace_map[dst] = workspace
        return buffers_with_workspace(buffers, workspace_map)

    def ensure(matrix, key: str, indptr):
        workspace_map = workspace_map_from(matrix.buffers)
        if key in workspace_map:
            return matrix
        workspace_map[key] = _make_binary_task_workspace(indptr)
        if buffer_name in matrix.buffers:
            matrix.set_buffer(buffer_name, workspace_map)
        else:
            matrix.register_buffer(buffer_name, workspace_map)
        return matrix

    def ensure_and_get(matrix, key: str, indptr):
        matrix = ensure(matrix, key, indptr)
        return matrix, get(matrix, key)

    return split, restore, get, with_workspace, move_key, ensure, ensure_and_get

_BINARY_WORKSPACE_BUFFER = "binary_workspace"

(
    _split_binary_workspace_buffers,
    _restore_binary_workspace_buffers,
    _binary_workspace,
    _with_binary_workspace,
    _move_binary_workspace_key,
    _ensure_binary_workspace,
    _ensure_binary_workspace_and_get,
) = _binary_workspace_helpers(_BINARY_WORKSPACE_BUFFER)


def _use_cuda_indexed_binary_route(backend: Optional[str]) -> bool:
    return backend == "cuda_raw"


class CompressedSparseData(DataRepresentation):
    """
    Abstract base class for compressed sparse matrix formats.

    ``CompressedSparseData`` provides the common interface shared by :class:`CSR` and
    :class:`CSC`. It inherits from ``brainunit.sparse.SparseMatrix`` and
    adds arithmetic operators, JAX pytree support, and helper methods for
    event-driven neural simulation.

    Subclasses must implement :meth:`apply`, :meth:`_binary_op`,
    :meth:`_binary_rop`, :meth:`todense`, :meth:`with_data`,
    :meth:`fromdense`, :meth:`dt2t`, and :meth:`dt2t_transposed`. Both
    concrete subclasses (:class:`CSR`, :class:`CSC`) also provide the common
    conversion contract :meth:`tocsr`, :meth:`tocsc`, and :meth:`tocoo`.

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

    data: ArrayData
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
        indptr_dtype="auto",
        check_structure: bool = True,
    ):
        if indices is None and indptr is None:
            # Tuple syntax: CSR((data, indices, indptr), shape=...)
            args = data
        else:
            # Positional syntax: CSR(data, indices, indptr, shape=...)
            args = (data, indices, indptr)

        assert len(args) == 3, "Expected three arguments: data, indices, indptr."
        data_arg, indices_arg, indptr_arg = args
        if check_structure:
            fmt = type(self).__name__.lower()
            secondary_dim = shape[1] if fmt == "csr" else shape[0]
            nse = np.asarray(jax.device_get(indices_arg)).size
            context = f"{type(self).__name__} structure"
            self.data = u.math.asarray(data_arg)
            self.indices = _as_int32_indices(indices_arg, secondary_dim, context, output_is_numpy=False)
            self.indptr = _as_indptr(indptr_arg, nse, indptr_dtype, context, output_is_numpy=False)
            _check_compressed_structure(self.indices, self.indptr, shape, fmt)
        else:
            self.data = u.math.asarray(data_arg)
            self.indices = u.math.asarray(indices_arg)
            self.indptr = u.math.asarray(indptr_arg)
        self.backend = backend
        super().__init__((self.data, self.indices, self.indptr), shape=shape, buffers=buffers)

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
        is always a leaf. Hidden binary task workspace arrays, when present
        in ``buffers``, are also leaves; structural arrays and ordinary
        buffers are stored as auxiliary data.

        Returns
        -------
        children : tuple
            Leaf arrays containing ``data`` followed by any hidden binary
            workspace arrays.
        aux_data : tuple
            Auxiliary metadata and non-workspace buffers.

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
        static_buffers, workspace_keys, workspace_leaves = _split_binary_workspace_buffers(self.buffers)
        aux["_binary_workspace_keys"] = workspace_keys
        return (self.data, *workspace_leaves), (aux, static_buffers)

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
        aux, static_buffers = aux_data
        aux = dict(aux)
        data, *workspace_leaves = children
        workspace_keys = aux.pop("_binary_workspace_keys", ())
        buffers = _restore_binary_workspace_buffers(static_buffers, workspace_keys, workspace_leaves)
        return cls(
            data,
            aux["indices"],
            aux["indptr"],
            shape=aux["shape"],
            backend=aux["backend"],
            buffers=buffers,
            indptr_dtype=aux["indptr"].dtype,
            check_structure=False,
        )

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

    def dt2t(
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

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity]
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute the transposed sparse matrix-vector product mapping y-w space to w space.

        This is the adjoint of :meth:`dt2t`.  It is useful for
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
        dt2t : The forward (non-transposed) variant.
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

    def with_data(self, data: ArrayData):
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

    def diag_add(self, other):
        """
        Add values to the matrix diagonal and return a new sparse matrix.

        Computes ``A + diag(other)`` exactly. Diagonal entries that are missing
        from the current sparsity pattern are **inserted**, so in general the
        returned matrix has a different ``indices``/``indptr`` (and a larger
        ``nse``) than ``self``: its pattern is the union of ``self``'s pattern
        and the full main diagonal ``{(i, i) : 0 <= i < min(shape)}``.

        Parameters
        ----------
        other : array-like
            The diagonal values to add, with one entry per diagonal element
            (i.e. length ``min(self.shape)``). It should share the dtype and be
            unit-compatible with the matrix's non-zero elements.

        Returns
        -------
        CSR or CSC
            A new sparse matrix of the same class and shape holding
            ``A + diag(other)`` with the diagonal-augmented sparsity pattern.

        Raises
        ------
        AssertionError
            If ``other`` is an instance of ``JAXSparse``, as this operation does
            not support sparse operands.

        Notes
        -----
        - The structural plan describing where existing entries move and where
          diagonals are inserted depends only on the sparsity structure, so it is
          computed once and cached in the ``diag_positions`` buffer. The returned
          matrix already has its full diagonal present, so it carries a matching
          ``diag_positions`` plan that subsequent ``diag_add`` calls reuse without
          recomputation.
        - Structure-dependent caches (e.g. the transposed-view weight indices) are
          intentionally not propagated, since the new matrix has a different
          sparsity pattern.
        - This method relies on :func:`csr_diag_position` to plan the structural
          change and :func:`csr_diag_add` to compute the augmented values.
        """
        assert not isinstance(other, u.sparse.SparseMatrix), "diag_add does not support JAXSparse objects."
        if not hasattr(self, 'diag_positions'):
            self.register_buffer(
                'diag_positions',
                csr_diag_position(self.indptr, self.indices, shape=self.shape)
            )
        new_data = csr_diag_add(self.data, self.diag_positions, other)
        new_indptr, new_indices, _, diag_dest = self.diag_positions
        # Materialise the augmented structure as concrete device arrays even when
        # this method runs inside ``jax.jit``. Converting Numba's NumPy output to
        # JAX arrays mid-trace would otherwise produce constant tracers that leak
        # through the static (aux) part of the returned matrix's pytree.
        with jax.ensure_compile_time_eval():
            new_indptr = jnp.asarray(new_indptr)
            new_indices = jnp.asarray(new_indices)
            diag_dest = jnp.asarray(diag_dest)
            # The result already contains the full diagonal, so its own plan is an
            # identity relocation of every stored element plus the same diagonal
            # destinations -- reusable by a later diag_add without touching Numba.
            identity = jnp.arange(new_indices.shape[0], dtype=new_indices.dtype)
        result_plan = (new_indptr, new_indices, identity, diag_dest)
        return type(self)(
            (new_data, new_indices, new_indptr),
            shape=self.shape,
            backend=self.backend,
            buffers={'diag_positions': result_plan},
        )

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
    """
    __module__ = 'brainevent'

    @classmethod
    def fromdense(
        cls,
        mat,
        *,
        nse: Optional[int] = None,
        index_dtype=jnp.int32,
        indptr_dtype="auto",
        backend: Optional[str] = None,
        precompute_weight_indices: bool = False,
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
        backend : str or None, optional
            Compute backend to attach to the matrix. Default ``None``.
        precompute_weight_indices : bool, optional
            If ``True``, eagerly build and cache the column-major (CSC-like)
            weight indices used by the *unfavorable* ``CSR @ event`` direction
            (see :meth:`build_weight_indices`). If ``False`` (default), the
            indices are built lazily on first use. Default ``False``.

        Returns
        -------
        CSR
            A new CSR matrix object created from the input dense matrix.

        See Also
        --------
        build_weight_indices : Eagerly build the cached weight indices.
        CSR._weight_indices : Lazily build/return the cached weight indices.

        Examples
        --------
        .. code-block:: python

            import jax.numpy as jnp
            import brainevent

            dense = jnp.array([[1.0, 0.0], [0.0, 2.0]])
            csr = brainevent.CSR.fromdense(dense)
        """
        if np.dtype(index_dtype) != np.dtype(np.int32):
            raise ValueError(
                "CSR.fromdense only supports int32 indices. Use indptr_dtype='auto' "
                "or indptr_dtype=jnp.int64 to control row-pointer precision."
            )
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csr = u.sparse.csr_fromdense(mat, nse=nse, index_dtype=index_dtype)
        out = CSR(csr.data, csr.indices, csr.indptr, shape=csr.shape, backend=backend,
                  indptr_dtype=indptr_dtype)
        if precompute_weight_indices:
            out = out.build_weight_indices()
        return out

    def with_data(self, data: ArrayData) -> 'CSR':
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
            buffers=self.buffers,
            backend=self.backend,
            indptr_dtype=self.indptr.dtype,
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

    def tocsr(self) -> 'CSR':
        """Return this matrix in CSR format (a no-op that returns ``self``).

        Provided for a uniform conversion interface across data
        representations; ``CSR`` is already row-compressed.

        Returns
        -------
        CSR
            ``self``, unchanged.

        See Also
        --------
        tocsc : Re-encode the same logical matrix column-major.
        tocoo : Convert to coordinate format.
        """
        return self

    def tocsc(self) -> 'CSC':
        """Re-encode the same logical matrix in CSC format.

        Unlike :meth:`transpose` (which reinterprets the arrays as ``W.T`` with
        swapped shape), ``tocsc`` returns a :class:`CSC` describing the *same*
        matrix ``W`` with the *same* ``shape`` -- the entries are resorted into
        column-major order.

        Returns
        -------
        CSC
            The same logical matrix in CSC format, ``shape`` unchanged.

        See Also
        --------
        tocsr : Identity conversion.
        transpose : Logical transpose (swaps ``shape``).
        """
        csc_indptr, csc_indices, perm = csr_to_csc_index(self.indptr, self.indices, shape=self.shape)
        csc_data = self.data if self.data.size == 1 else self.data[perm]
        return CSC((csc_data, csc_indices, csc_indptr), shape=self.shape, backend=self.backend)

    def tocoo(self) -> u.sparse.COO:
        """Convert to coordinate (COO) format.

        Returns
        -------
        brainunit.sparse.COO
            The same logical matrix in COO format, ``shape`` unchanged. A
            homogeneous (size-1) value is broadcast to one entry per stored
            element.

        See Also
        --------
        tocsr : Identity conversion.
        tocsc : Re-encode the same logical matrix column-major.
        """
        rows, cols = _csr_to_coo(self.indices, self.indptr)
        data = self.data if self.data.size == rows.size else u.math.broadcast_to(self.data, rows.shape)
        return u.sparse.COO((data, rows, cols), shape=self.shape, rows_sorted=True)

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
        # The CSC-like view of ``W`` is, array-for-array, the CSR-like view of
        # ``W.T``: the cached weight indices transfer for free across transpose.
        # Re-key the ``'csc'`` buffer to ``'csr'`` so the resulting CSC finds it.
        buffers = dict(self.buffers)
        alt = buffers.pop('csc', None)
        if alt is not None:
            buffers['csr'] = alt
        buffers = _move_binary_workspace_key(buffers, "csc", "csr")
        return CSC(
            self.data, self.indices, self.indptr,
            shape=self.shape[::-1],
            buffers=buffers,
            backend=self.backend,
            indptr_dtype=self.indptr.dtype,
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
            buffers=self.buffers,
            backend=self.backend,
            indptr_dtype=self.indptr.dtype,
        )

    def _weight_indices(self):
        """Return the cached column-major (CSC-like) weight indices, building lazily.

        The *unfavorable* direction ``CSR @ event`` is evaluated by traversing
        the matrix column-by-column (a CSC-like view) and scattering events.
        That traversal needs the matrix structure re-expressed in column-major
        order together with a permutation ``perm`` mapping each column-major
        slot back to the canonical CSR ``data`` order, so that structural slot
        ``j`` reads ``data[perm[j]]``.

        The triple depends only on the sparse *structure* (``indices``,
        ``indptr``, ``shape``) and not on the stored values, so it survives
        :meth:`apply` / :meth:`with_data` and is cached in the ``'csc'`` buffer.
        It is computed on first access and reused thereafter.

        Returns
        -------
        csc_indptr : jax.Array
            Column pointer array of the CSC-like view. Length ``shape[1] + 1``.
        csc_indices : jax.Array
            Row index array of the CSC-like view. Shape ``(nse,)``.
        perm : jax.Array
            Permutation mapping column-major slot ``j`` to the canonical CSR
            ``data`` index ``perm[j]``. Shape ``(nse,)``.

        See Also
        --------
        build_weight_indices : Eagerly build and cache the same triple.
        brainevent.csr_to_csc_index : Underlying index conversion.
        """
        cached = self.buffers.get('csc')
        if cached is not None:
            return cached
        with jax.ensure_compile_time_eval():
            csc = csr_to_csc_index(self.indptr, self.indices, shape=self.shape)
        self.register_buffer('csc', csc)
        return csc

    def build_weight_indices(self) -> 'CSR':
        """Return a copy of this CSR with the weight indices eagerly cached.

        Builds the column-major (CSC-like) structure and permutation used by the
        ``CSR @ event`` direction (see :meth:`_weight_indices`) and stores it in
        the ``'csc'`` buffer of the returned matrix. The underlying ``data``,
        ``indices``, and ``indptr`` arrays are shared (not copied).

        Returns
        -------
        CSR
            A new CSR matrix sharing this matrix's arrays, with the ``'csc'``
            weight-index buffer populated.

        See Also
        --------
        CSR._weight_indices : Lazy builder/accessor for the same triple.
        CSR.fromdense : Accepts ``precompute_weight_indices=True`` to call this.
        """
        with jax.ensure_compile_time_eval():
            csc = csr_to_csc_index(self.indptr, self.indices, shape=self.shape)
        buffers = dict(self.buffers)
        buffers['csc'] = csc
        return CSR(
            self.data, self.indices, self.indptr,
            shape=self.shape,
            buffers=buffers,
            backend=self.backend,
            indptr_dtype=self.indptr.dtype,
        )

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None) -> 'CSR':
        """Apply a presynaptic-spike-triggered STDP update, returning a new CSR.

        Convenience wrapper around :func:`brainevent.update_csr_on_binary_pre`
        that keeps the sparsity structure (and therefore the cached weight
        indices) intact.  For each firing presynaptic neuron ``i`` every stored
        synapse is updated ``W[i, j] <- clip(W[i, j] + post_trace[j], w_min, w_max)``.

        Parameters
        ----------
        pre_spike : jax.Array
            Binary/boolean presynaptic spikes, shape ``(shape[0],)``.
        post_trace : jax.Array or Quantity
            Postsynaptic eligibility trace, shape ``(shape[1],)``.
        w_min, w_max : optional
            Clipping bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        CSR
            A new CSR matrix with updated data and identical structure.

        See Also
        --------
        update_on_post : Postsynaptic-spike-triggered counterpart.
        brainevent.update_csr_on_binary_pre : Underlying module function.
        """
        new_w = update_csr_on_binary_pre(
            self.data, self.indices, self.indptr, pre_spike, post_trace,
            w_min, w_max, shape=self.shape, backend=self.backend,
        )
        return self.with_data(new_w)

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None) -> 'CSR':
        """Apply a postsynaptic-spike-triggered STDP update, returning a new CSR.

        Convenience wrapper around :func:`brainevent.update_csr_on_binary_post`.
        Iterating by postsynaptic spike is the *unfavorable* direction for CSR,
        so this reuses the cached column-major weight indices
        (:meth:`_weight_indices`) to scatter the updates back into canonical
        order.  For each firing postsynaptic neuron ``j`` every stored synapse is
        updated ``W[i, j] <- clip(W[i, j] + pre_trace[i], w_min, w_max)``.

        Parameters
        ----------
        pre_trace : jax.Array or Quantity
            Presynaptic eligibility trace, shape ``(shape[0],)``.
        post_spike : jax.Array
            Binary/boolean postsynaptic spikes, shape ``(shape[1],)``.
        w_min, w_max : optional
            Clipping bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        CSR
            A new CSR matrix with updated data and identical structure.

        See Also
        --------
        update_on_pre : Presynaptic-spike-triggered counterpart.
        brainevent.update_csr_on_binary_post : Underlying module function.
        """
        csc_indptr, csc_indices, perm = self._weight_indices()
        new_w = update_csr_on_binary_post(
            self.data, csc_indices, csc_indptr, perm, pre_trace, post_spike,
            w_min, w_max, shape=self.shape, backend=self.backend,
        )
        return self.with_data(new_w)

    def __getitem__(self, index):
        """Extract rows of the matrix ``W`` as a dense array (NumPy semantics).

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0. Negative indices wrap; Python slices are
            supported. Concrete out-of-bounds indices raise ``IndexError``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            ``(n_cols,)`` for a single int, otherwise ``(len(rows), n_cols)``.
        """
        rows = normalize_row_index(index, self.shape[0])
        return csr_slice_rows(
            self.data, self.indices, self.indptr, rows,
            shape=self.shape, backend=self.backend,
        )

    def slice_rows(self, index) -> 'CSR':
        """Return ``W[rows, :]`` as a new :class:`CSR` (outside ``jax.jit``).

        The output non-zero count is data-dependent, so ``index`` must be
        concrete. Accepts the same selectors as :meth:`__getitem__`; a single
        int yields a ``1 x n_cols`` matrix.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0.

        Returns
        -------
        CSR
            Sparse sub-matrix of shape ``(len(rows), n_cols)``.
        """
        rows = jnp.atleast_1d(normalize_row_index(index, self.shape[0]))
        new_data, new_indices, new_indptr, shape = build_sub_csr(
            self.data, self.indices, self.indptr, rows, self.shape[1],
        )
        return CSR((new_data, new_indices, new_indptr), shape=shape, backend=self.backend)

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
                    buffers=self.buffers,
                    backend=self.backend,
                    indptr_dtype=self.indptr.dtype,
                )
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR(
                op(self.data, other), self.indices, self.indptr,
                shape=self.shape,
                buffers=self.buffers,
                backend=self.backend,
                indptr_dtype=self.indptr.dtype,
            )

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                op(self.data, other),
                self.indices,
                self.indptr,
                shape=self.shape,
                buffers=self.buffers,
                backend=self.backend,
                indptr_dtype=self.indptr.dtype,
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
                    buffers=self.buffers,
                    backend=self.backend,
                    indptr_dtype=self.indptr.dtype,
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
                buffers=self.buffers,
                backend=self.backend,
                indptr_dtype=self.indptr.dtype,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                op(other, self.data),
                self.indices,
                self.indptr,
                shape=self.shape,
                buffers=self.buffers,
                backend=self.backend,
                indptr_dtype=self.indptr.dtype,
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
        * Dense ``jax.Array`` / ``brainunit.Quantity`` -- standard float
          kernel with automatic dtype promotion.

        Parameters
        ----------
        other : jax.Array, brainunit.Quantity, or BinaryArray
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
                if not _use_cuda_indexed_binary_route(self.backend):
                    matrix, workspace = _ensure_binary_workspace_and_get(self, "csr", self.indptr)
                    return binary_csrmv(
                        matrix.data, matrix.indices, matrix.indptr, other,
                        shape=matrix.shape, transpose=False, backend=matrix.backend, workspace=workspace,
                    )
                # Explicit CUDA raw uses the mirror route so the transpose
                # hybrid kernels handle the unfavorable direction.
                csc_indptr, csc_indices, perm = self._weight_indices()
                matrix, workspace = _ensure_binary_workspace_and_get(self, "csc", csc_indptr)
                return binary_csrmv_indexed(
                    matrix.data, csc_indices, csc_indptr, perm, other,
                    shape=matrix.shape[::-1], transpose=True, backend=matrix.backend, workspace=workspace,
                )
            elif other.ndim == 2:
                if not _use_cuda_indexed_binary_route(self.backend):
                    matrix, workspace = _ensure_binary_workspace_and_get(self, "csr", self.indptr)
                    return binary_csrmm(
                        matrix.data, matrix.indices, matrix.indptr, other,
                        shape=matrix.shape, transpose=False, backend=matrix.backend, workspace=workspace,
                    )
                # Explicit CUDA raw uses the mirror route so the transpose
                # hybrid kernels handle the unfavorable direction.
                csc_indptr, csc_indices, perm = self._weight_indices()
                matrix, workspace = _ensure_binary_workspace_and_get(self, "csc", csc_indptr)
                return binary_csrmm_indexed(
                    matrix.data, csc_indices, csc_indptr, perm, other,
                    shape=matrix.shape[::-1], transpose=True, backend=matrix.backend, workspace=workspace,
                )
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
        other : jax.Array, brainunit.Quantity, or BinaryArray
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
                matrix, workspace = _ensure_binary_workspace_and_get(self, "csr", self.indptr)
                return binary_csrmv(matrix.data, matrix.indices, matrix.indptr, other,
                                    shape=matrix.shape, transpose=True, backend=matrix.backend, workspace=workspace)
            elif other.ndim == 2:
                other = other.T
                matrix, workspace = _ensure_binary_workspace_and_get(self, "csr", self.indptr)
                r = binary_csrmm(matrix.data, matrix.indices, matrix.indptr, other,
                                 shape=matrix.shape, transpose=True, backend=matrix.backend, workspace=workspace)
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

    def dt2t(
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
        dt2t_transposed : The transposed (adjoint) variant.

        Notes
        -----
        Internally calls ``csrmv_dt2t`` with ``transpose=False``.
        """
        return csrmv_dt2t(y_dim_arr, w_dim_arr, self.indices, self.indptr,
                          shape=self.shape, transpose=False, backend=self.backend)

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute the transposed sparse transformation from y-w space to w space.

        This is the adjoint of :meth:`dt2t`, useful for back-propagation
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
        dt2t : The forward (non-transposed) variant.

        Notes
        -----
        Internally calls ``csrmv_dt2t`` with ``transpose=True``.
        """
        return csrmv_dt2t(y_dim_arr, w_dim_arr, self.indices, self.indptr,
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
    """
    __module__ = 'brainevent'

    @classmethod
    def fromdense(
        cls,
        mat,
        *,
        nse: Optional[int] = None,
        index_dtype=jnp.int32,
        indptr_dtype="auto",
        backend: Optional[str] = None,
        precompute_weight_indices: bool = False,
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
        backend : str or None, optional
            Compute backend to attach to the matrix. Default ``None``.
        precompute_weight_indices : bool, optional
            If ``True``, eagerly build and cache the row-major (CSR-like) weight
            indices used by the *unfavorable* ``event @ CSC`` direction (see
            :meth:`build_weight_indices`). If ``False`` (default), the indices
            are built lazily on first use. Default ``False``.

        Returns
        -------
        CSC
            A new CSC matrix instance created from the input dense matrix.

        See Also
        --------
        build_weight_indices : Eagerly build the cached weight indices.
        CSC._weight_indices : Lazily build/return the cached weight indices.

        Examples
        --------
        .. code-block:: python

            import jax.numpy as jnp
            import brainevent

            dense = jnp.array([[1.0, 0.0], [0.0, 2.0]])
            csc = brainevent.CSC.fromdense(dense)
        """
        if np.dtype(index_dtype) != np.dtype(np.int32):
            raise ValueError(
                "CSC.fromdense only supports int32 indices. Use indptr_dtype='auto' "
                "or indptr_dtype=jnp.int64 to control column-pointer precision."
            )
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csc = u.sparse.csr_fromdense(mat.T, nse=nse, index_dtype=index_dtype).T
        out = CSC((csc.data, csc.indices, csc.indptr), shape=csc.shape, backend=backend,
                  indptr_dtype=indptr_dtype)
        if precompute_weight_indices:
            out = out.build_weight_indices()
        return out

    def with_data(self, data: ArrayData) -> 'CSC':
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
                   buffers=self.buffers,
                   backend=self.backend,
                   indptr_dtype=self.indptr.dtype)

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

    def tocsc(self) -> 'CSC':
        """Return this matrix in CSC format (a no-op that returns ``self``).

        Provided for a uniform conversion interface across data
        representations; ``CSC`` is already column-compressed.

        Returns
        -------
        CSC
            ``self``, unchanged.

        See Also
        --------
        tocsr : Re-encode the same logical matrix row-major.
        tocoo : Convert to coordinate format.
        """
        return self

    def tocsr(self) -> 'CSR':
        """Re-encode the same logical matrix in CSR format.

        Unlike :meth:`transpose` (which reinterprets the arrays as ``W.T`` with
        swapped shape), ``tocsr`` returns a :class:`CSR` describing the *same*
        matrix ``W`` with the *same* ``shape`` -- the entries are resorted into
        row-major order.

        Returns
        -------
        CSR
            The same logical matrix in CSR format, ``shape`` unchanged.

        See Also
        --------
        tocsc : Identity conversion.
        transpose : Logical transpose (swaps ``shape``).
        """
        csr_indptr, csr_indices, perm = csc_to_csr_index(self.indptr, self.indices, shape=self.shape)
        csr_data = self.data if self.data.size == 1 else self.data[perm]
        return CSR((csr_data, csr_indices, csr_indptr), shape=self.shape, backend=self.backend)

    def tocoo(self) -> u.sparse.COO:
        """Convert to coordinate (COO) format.

        Returns
        -------
        brainunit.sparse.COO
            The same logical matrix in COO format, ``shape`` unchanged. A
            homogeneous (size-1) value is broadcast to one entry per stored
            element.

        See Also
        --------
        tocsr : Re-encode the same logical matrix row-major.
        tocsc : Identity conversion.
        """
        cols, rows = _csr_to_coo(self.indices, self.indptr)
        data = self.data if self.data.size == rows.size else u.math.broadcast_to(self.data, rows.shape)
        return u.sparse.COO((data, rows, cols), shape=self.shape, cols_sorted=True)

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
        # The CSR-like view of this CSC is, array-for-array, the CSC-like view
        # of its transpose: re-key the ``'csr'`` buffer to ``'csc'`` so the
        # resulting CSR reuses the cached weight indices for free.
        buffers = dict(self.buffers)
        alt = buffers.pop('csr', None)
        if alt is not None:
            buffers['csc'] = alt
        buffers = _move_binary_workspace_key(buffers, "csr", "csc")
        return CSR((self.data, self.indices, self.indptr),
                   shape=self.shape[::-1],
                   buffers=buffers,
                   backend=self.backend,
                   indptr_dtype=self.indptr.dtype)

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
                   shape=self.shape, buffers=self.buffers, backend=self.backend,
                   indptr_dtype=self.indptr.dtype)

    def _weight_indices(self):
        """Return the cached row-major (CSR-like) weight indices, building lazily.

        The *unfavorable* direction ``event @ CSC`` is evaluated by traversing
        the matrix row-by-row (a CSR-like view) and scattering events. That
        traversal needs the matrix structure re-expressed in row-major order
        together with a permutation ``perm`` mapping each row-major slot back to
        the canonical CSC ``data`` order, so that structural slot ``j`` reads
        ``data[perm[j]]``.

        The triple depends only on the sparse *structure* (``indices``,
        ``indptr``, ``shape``) and not on the stored values, so it survives
        :meth:`apply` / :meth:`with_data` and is cached in the ``'csr'`` buffer.
        It is computed on first access and reused thereafter.

        Returns
        -------
        csr_indptr : jax.Array
            Row pointer array of the CSR-like view. Length ``shape[0] + 1``.
        csr_indices : jax.Array
            Column index array of the CSR-like view. Shape ``(nse,)``.
        perm : jax.Array
            Permutation mapping row-major slot ``j`` to the canonical CSC
            ``data`` index ``perm[j]``. Shape ``(nse,)``.

        See Also
        --------
        build_weight_indices : Eagerly build and cache the same triple.
        brainevent.csc_to_csr_index : Underlying index conversion.
        """
        cached = self.buffers.get('csr')
        if cached is not None:
            return cached
        with jax.ensure_compile_time_eval():
            csr = csc_to_csr_index(self.indptr, self.indices, shape=self.shape)
        self.register_buffer('csr', csr)
        return csr

    def build_weight_indices(self) -> 'CSC':
        """Return a copy of this CSC with the weight indices eagerly cached.

        Builds the row-major (CSR-like) structure and permutation used by the
        ``event @ CSC`` direction (see :meth:`_weight_indices`) and stores it in
        the ``'csr'`` buffer of the returned matrix. The underlying ``data``,
        ``indices``, and ``indptr`` arrays are shared (not copied).

        Returns
        -------
        CSC
            A new CSC matrix sharing this matrix's arrays, with the ``'csr'``
            weight-index buffer populated.

        See Also
        --------
        CSC._weight_indices : Lazy builder/accessor for the same triple.
        CSC.fromdense : Accepts ``precompute_weight_indices=True`` to call this.
        """
        with jax.ensure_compile_time_eval():
            csr = csc_to_csr_index(self.indptr, self.indices, shape=self.shape)
        buffers = dict(self.buffers)
        buffers['csr'] = csr
        return CSC(
            (self.data, self.indices, self.indptr),
            shape=self.shape,
            buffers=buffers,
            backend=self.backend,
            indptr_dtype=self.indptr.dtype,
        )

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None) -> 'CSC':
        """Apply a presynaptic-spike-triggered STDP update, returning a new CSC.

        Iterating by presynaptic spike is the *unfavorable* direction for CSC,
        so this reuses the cached row-major weight indices
        (:meth:`_weight_indices`) and routes through
        :func:`brainevent.update_csr_on_binary_post`, scattering updates back
        into canonical CSC order.  For each firing presynaptic neuron ``i`` every
        stored synapse is updated
        ``W[i, j] <- clip(W[i, j] + post_trace[j], w_min, w_max)``.

        Parameters
        ----------
        pre_spike : jax.Array
            Binary/boolean presynaptic spikes, shape ``(shape[0],)``.
        post_trace : jax.Array or Quantity
            Postsynaptic eligibility trace, shape ``(shape[1],)``.
        w_min, w_max : optional
            Clipping bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        CSC
            A new CSC matrix with updated data and identical structure.

        See Also
        --------
        update_on_post : Postsynaptic-spike-triggered counterpart.
        brainevent.update_csc_on_binary_pre : Equivalent module function.
        """
        csr_indptr, csr_indices, perm = self._weight_indices()
        new_w = update_csr_on_binary_post(
            self.data, csr_indices, csr_indptr, perm, post_trace, pre_spike,
            w_min, w_max, shape=self.shape[::-1], backend=self.backend,
        )
        return self.with_data(new_w)

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None) -> 'CSC':
        """Apply a postsynaptic-spike-triggered STDP update, returning a new CSC.

        Iterating by postsynaptic spike is the *favorable* direction for CSC, so
        this streams directly over the stored arrays (no permutation) via
        :func:`brainevent.update_csr_on_binary_pre` on the transposed shape.  For
        each firing postsynaptic neuron ``j`` every stored synapse is updated
        ``W[i, j] <- clip(W[i, j] + pre_trace[i], w_min, w_max)``.

        Parameters
        ----------
        pre_trace : jax.Array or Quantity
            Presynaptic eligibility trace, shape ``(shape[0],)``.
        post_spike : jax.Array
            Binary/boolean postsynaptic spikes, shape ``(shape[1],)``.
        w_min, w_max : optional
            Clipping bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        CSC
            A new CSC matrix with updated data and identical structure.

        See Also
        --------
        update_on_pre : Presynaptic-spike-triggered counterpart.
        brainevent.update_csc_on_binary_post : Equivalent module function.
        """
        new_w = update_csr_on_binary_pre(
            self.data, self.indices, self.indptr, post_spike, pre_trace,
            w_min, w_max, shape=self.shape[::-1], backend=self.backend,
        )
        return self.with_data(new_w)

    def __getitem__(self, index):
        """Extract rows of the matrix ``W`` as a dense array (NumPy semantics).

        Row slicing is the *unfavorable* direction for CSC, so it reuses the
        cached row-major (CSR-of-``W``) view from :meth:`_weight_indices` and the
        shared ``csr_slice_rows`` kernel.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0. Negative indices wrap; Python slices are
            supported. Concrete out-of-bounds indices raise ``IndexError``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            ``(n_cols,)`` for a single int, otherwise ``(len(rows), n_cols)``.
        """
        rows = normalize_row_index(index, self.shape[0])
        csr_indptr, csr_indices, perm = self._weight_indices()
        weights = self.data if self.data.size == 1 else self.data.reshape(-1)[perm]
        return csr_slice_rows(
            weights, csr_indices, csr_indptr, rows,
            shape=self.shape, backend=self.backend,
        )
    def slice_rows(self, index) -> 'CSC':
        """Return ``W[rows, :]`` as a new :class:`CSC` (outside ``jax.jit``).

        Builds the CSR arrays of ``W[rows, :]`` through the cached CSR-of-``W``
        view, then converts to CSC. The output non-zero count is data-dependent,
        so ``index`` must be concrete.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0.

        Returns
        -------
        CSC
            Sparse sub-matrix of shape ``(len(rows), n_cols)``.
        """
        rows = jnp.atleast_1d(normalize_row_index(index, self.shape[0]))
        csr_indptr, csr_indices, perm = self._weight_indices()
        data = cast(jax.Array, self.data)
        weights = data if data.size == 1 else data.reshape(-1)[perm]
        sub_data, sub_indices, sub_indptr, shape = build_sub_csr(
            weights, csr_indices, csr_indptr, rows, self.shape[1],
        )
        with jax.ensure_compile_time_eval():
            csc_indptr, csc_indices, cperm = csr_to_csc_index(
                sub_indptr, sub_indices, shape=shape,
            )
        csc_data = sub_data if sub_data.size == 1 else sub_data[cperm]
        return CSC((csc_data, csc_indices, csc_indptr), shape=shape, backend=self.backend)

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
                    buffers=self.buffers,
                    backend=self.backend,
                    indptr_dtype=self.indptr.dtype,
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
                buffers=self.buffers,
                backend=self.backend,
                indptr_dtype=self.indptr.dtype,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape,
                buffers=self.buffers,
                backend=self.backend,
                indptr_dtype=self.indptr.dtype,
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
                    buffers=self.buffers,
                    backend=self.backend,
                    indptr_dtype=self.indptr.dtype,
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
                buffers=self.buffers,
                backend=self.backend,
                indptr_dtype=self.indptr.dtype,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape,
                buffers=self.buffers,
                backend=self.backend,
                indptr_dtype=self.indptr.dtype,
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
        other : jax.Array, brainunit.Quantity, or BinaryArray
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
                matrix, workspace = _ensure_binary_workspace_and_get(self, "csc", self.indptr)
                return binary_csrmv(
                    matrix.data, matrix.indices, matrix.indptr, other,
                    shape=matrix.shape[::-1],
                    transpose=True,
                    backend=matrix.backend,
                    workspace=workspace,
                )
            elif other.ndim == 2:
                matrix, workspace = _ensure_binary_workspace_and_get(self, "csc", self.indptr)
                return binary_csrmm(
                    matrix.data, matrix.indices, matrix.indptr, other,
                    shape=matrix.shape[::-1],
                    transpose=True,
                    backend=matrix.backend,
                    workspace=workspace,
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
        other : jax.Array, brainunit.Quantity, or BinaryArray
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
                if not _use_cuda_indexed_binary_route(self.backend):
                    matrix, workspace = _ensure_binary_workspace_and_get(self, "csc", self.indptr)
                    return binary_csrmv(
                        matrix.data, matrix.indices, matrix.indptr, other,
                        shape=matrix.shape[::-1], transpose=False, backend=matrix.backend, workspace=workspace,
                    )
                # Explicit CUDA raw uses the mirror route so the transpose
                # hybrid kernels handle the unfavorable direction.
                csr_indptr, csr_indices, perm = self._weight_indices()
                matrix, workspace = _ensure_binary_workspace_and_get(self, "csr", csr_indptr)
                return binary_csrmv_indexed(
                    matrix.data, csr_indices, csr_indptr, perm, other,
                    shape=matrix.shape, transpose=True, backend=matrix.backend, workspace=workspace,
                )
            elif other.ndim == 2:
                if not _use_cuda_indexed_binary_route(self.backend):
                    matrix, workspace = _ensure_binary_workspace_and_get(self, "csc", self.indptr)
                    r = binary_csrmm(
                        matrix.data, matrix.indices, matrix.indptr, other.T,
                        shape=matrix.shape[::-1], transpose=False, backend=matrix.backend, workspace=workspace,
                    )
                    return r.T
                # Explicit CUDA raw uses the mirror route so the transpose
                # hybrid kernels handle the unfavorable direction.
                csr_indptr, csr_indices, perm = self._weight_indices()
                matrix, workspace = _ensure_binary_workspace_and_get(self, "csr", csr_indptr)
                r = binary_csrmm_indexed(
                    matrix.data, csr_indices, csr_indptr, perm, other.T,
                    shape=matrix.shape, transpose=True, backend=matrix.backend, workspace=workspace,
                )
                return r.T
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

    def dt2t(
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
        dt2t_transposed : The transposed (adjoint) variant.

        Notes
        -----
        Internally calls ``csrmv_dt2t`` with ``transpose=True`` and reversed
        shape to account for the column-oriented storage format.
        """
        return csrmv_dt2t(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape[::-1], transpose=True,
                          backend=self.backend)

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute the transposed sparse transformation from y-w space to w space.

        This is the adjoint of :meth:`dt2t`, useful for back-propagation
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
        dt2t : The forward (non-transposed) variant.

        Notes
        -----
        Internally calls ``csrmv_dt2t`` with ``transpose=False`` and reversed
        shape.
        """
        return csrmv_dt2t(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape[::-1], transpose=False,
                          backend=self.backend)
