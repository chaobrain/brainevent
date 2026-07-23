# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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
"""Explicit dense matrix representation.

This module provides :class:`Dense`, the dense counterpart to the CSR/CSC and
fixed-connection data representations.  It keeps the full weight matrix as the
single JAX pytree leaf while sharing the same high-level data contract used by
the sparse families.
"""

import operator
from typing import Dict, Optional, Sequence, Union

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from brainevent._data import DataRepresentation
from brainevent._typing import Data, MatrixShape
from .binary import binary_densemm, binary_densemv
from .plasticity_binary import update_dense_on_binary_pre, update_dense_on_binary_post

__all__ = [
    'Dense',
]


def _event_value(obj):
    from brainevent._event import BinaryArray, BitPackedBinary
    if isinstance(obj, (BinaryArray, BitPackedBinary)):
        return obj.value
    return None


def _normalize_dense_index(index):
    if isinstance(index, list):
        return jnp.asarray(index)
    if isinstance(index, tuple):
        return tuple(jnp.asarray(item) if isinstance(item, list) else item for item in index)
    return index


@register_pytree_node_class
class Dense(DataRepresentation):
    """
    Unit-aware explicit dense matrix.

    ``Dense`` stores a full two-dimensional weight matrix and exposes the same
    representation contract as the sparse matrix families.  It is useful when a
    caller wants matrix metadata, backend selection, plasticity helpers, and
    event-driven binary matmul dispatch without converting the matrix to CSR or
    CSC format.

    The class is a JAX pytree with a single dynamic leaf, ``data``.  Static
    metadata such as ``shape``, ``backend``, and registered buffers are carried
    in the pytree auxiliary data so ``Dense`` can be passed through ``jax.jit``
    in the same style as :class:`~brainevent.CSR` and
    :class:`~brainevent.CSC`.

    Parameters
    ----------
    data : array_like or brainunit.Quantity
        Dense two-dimensional matrix data.  Units are preserved.
    shape : sequence of int, optional
        Explicit matrix shape.  When provided it must match ``data.shape``.
    backend : str or None, optional
        Backend attached to event-driven binary matmul and plasticity calls.
        Typical values are ``None``, ``'jax_raw'``, ``'cuda_raw'``, and
        ``'cublas'`` where supported.
    buffers : dict, optional
        Named auxiliary buffers to carry with the representation.

    Attributes
    ----------
    data : Data
        Full dense matrix data.
    shape : tuple[int, int]
        Matrix shape ``(rows, columns)``.
    backend : str or None
        Backend preference propagated to dense binary primitives.
    nse : int
        Number of stored entries.  For a dense matrix this is ``data.size``.
    dtype : dtype
        Data type of the matrix values.

    Notes
    -----
    ``Dense @ BinaryArray`` and ``BinaryArray @ Dense`` dispatch to the
    event-driven dense binary primitives.  Dense numeric operands fall back to
    ordinary JAX matrix multiplication with unit-aware dtype promotion.

    The per-synapse ``dt2t`` protocol is intentionally not implemented directly
    for explicit dense matrices.  Materialise a sparse representation first, for
    example ``dense.tocsr().dt2t(y, dense.tocsr().data)``, when a per-stored
    synapse output is required.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import brainevent

        weights = jnp.array([[1.0, 0.0], [2.0, 3.0]])
        dense = brainevent.Dense(weights, backend='jax_raw')
        spikes = brainevent.BinaryArray(jnp.array([True, False]))

        y = dense @ spikes

    See Also
    --------
    CSR : Compressed sparse row representation.
    CSC : Compressed sparse column representation.
    binary_densemv : Dense matrix-vector event primitive.
    binary_densemm : Dense matrix-matrix event primitive.
    """

    data: Data
    shape: MatrixShape
    __module__ = 'brainevent'

    def __init__(
        self,
        data,
        *,
        shape: Optional[Sequence[int]] = None,
        backend: Optional[str] = None,
        buffers: Optional[Dict] = None,
    ):
        data = u.math.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Dense data must be 2-D, got shape={data.shape}.")
        if shape is None:
            shape = tuple(data.shape)
        else:
            shape = tuple(shape)
        assert len(shape) == 2, f"Dense shape must be 2-D, got shape={shape}."
        assert shape == tuple(data.shape), (
            f"Dense shape {shape} does not match data shape {tuple(data.shape)}."
        )
        self.data = data
        self.backend = backend
        super().__init__((self.data,), shape=shape, buffers=buffers)

    @property
    def nse(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def tree_flatten(self):
        """Return pytree children and static metadata for JAX transformations."""
        aux = {
            'shape': self.shape,
            'backend': self.backend,
        }
        return (self.data,), (aux, self.buffers)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild a ``Dense`` instance from pytree leaves and metadata."""
        obj = object.__new__(cls)
        obj.data, = children
        aux, buffers = aux_data
        obj._buffer_registry = set(buffers.keys())
        for k, v in aux.items():
            setattr(obj, k, v)
        for k, v in buffers.items():
            setattr(obj, k, v)
        return obj

    @classmethod
    def fromdense(
        cls,
        mat,
        *,
        backend: Optional[str] = None,
        buffers: Optional[Dict] = None,
    ) -> 'Dense':
        """
        Create a ``Dense`` representation from an explicit dense matrix.

        Parameters
        ----------
        mat : array_like or brainunit.Quantity
            Two-dimensional matrix to wrap.
        backend : str or None, optional
            Backend preference attached to the resulting representation.
        buffers : dict, optional
            Named buffers to register on the resulting representation.

        Returns
        -------
        Dense
            A new dense matrix representation containing ``mat``.
        """
        return cls(mat, backend=backend, buffers=buffers)

    def with_data(self, data: Data) -> 'Dense':
        """
        Return a new ``Dense`` with replacement data and unchanged metadata.

        The replacement must preserve shape, dtype, and physical unit.  This is
        the dense analogue of ``CSR.with_data``: only values change, while
        backend and buffers are carried forward.
        """
        data = u.math.asarray(data)
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return Dense(data, shape=self.shape, backend=self.backend, buffers=self.buffers)

    def apply(self, fn) -> 'Dense':
        """Apply ``fn`` to the dense data and wrap the result as ``Dense``."""
        return Dense(fn(self.data), shape=self.shape, backend=self.backend, buffers=self.buffers)

    def _binary_operand_data(self, other):
        if isinstance(other, Dense):
            if other.shape != self.shape:
                raise ValueError(
                    f"Dense operand shape {other.shape} is not compatible with shape {self.shape}."
                )
            return other.data
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("binary operation between Dense and sparse objects.")

        other = u.math.asarray(other)
        try:
            result_shape = jnp.broadcast_shapes(self.shape, other.shape)
        except ValueError as exc:
            raise ValueError(
                f"operand shape {other.shape} cannot broadcast to Dense shape {self.shape}."
            ) from exc
        if result_shape != self.shape:
            raise ValueError(
                f"operand shape {other.shape} broadcasts to {result_shape}, expected {self.shape}."
            )
        return other

    def _binary_op(self, other, op) -> 'Dense':
        return Dense(
            op(self.data, self._binary_operand_data(other)),
            shape=self.shape,
            backend=self.backend,
            buffers=self.buffers,
        )

    def _binary_rop(self, other, op) -> 'Dense':
        return Dense(
            op(self._binary_operand_data(other), self.data),
            shape=self.shape,
            backend=self.backend,
            buffers=self.buffers,
        )

    def apply2(self, other, fn, *, reverse: bool = False):
        """Apply a binary elementwise operation while preserving metadata."""
        if reverse:
            return self._binary_rop(other, fn)
        return self._binary_op(other, fn)

    def __abs__(self):
        return self.apply(operator.abs)

    def __neg__(self):
        return self.apply(operator.neg)

    def __pos__(self):
        return self.apply(operator.pos)

    def __mul__(self, other):
        return self.apply2(other, operator.mul)

    def __truediv__(self, other):
        return self.apply2(other, operator.truediv)

    def __add__(self, other):
        return self.apply2(other, operator.add)

    def __sub__(self, other):
        return self.apply2(other, operator.sub)

    def __rmul__(self, other):
        return self.apply2(other, operator.mul, reverse=True)

    def __rtruediv__(self, other):
        return self.apply2(other, operator.truediv, reverse=True)

    def __radd__(self, other):
        return self.apply2(other, operator.add, reverse=True)

    def __rsub__(self, other):
        return self.apply2(other, operator.sub, reverse=True)

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Return the explicit dense matrix data.

        Unlike sparse representations, no materialisation is needed because the
        storage format is already dense.
        """
        return self.data

    def tocsr(
        self,
        *,
        nse: Optional[int] = None,
        index_dtype=jnp.int32,
        precompute_weight_indices: bool = False,
    ):
        """
        Convert the dense matrix to :class:`~brainevent.CSR`.

        Parameters mirror :meth:`brainevent.CSR.fromdense`; ``backend`` is
        propagated from this ``Dense`` instance.
        """
        from brainevent._csr import CSR
        return CSR.fromdense(
            self.data,
            nse=nse,
            index_dtype=index_dtype,
            backend=self.backend,
            precompute_weight_indices=precompute_weight_indices,
        )

    def tocsc(
        self,
        *,
        nse: Optional[int] = None,
        index_dtype=jnp.int32,
        precompute_weight_indices: bool = False,
    ):
        """
        Convert the dense matrix to :class:`~brainevent.CSC`.

        Parameters mirror :meth:`brainevent.CSC.fromdense`; ``backend`` is
        propagated from this ``Dense`` instance.
        """
        from brainevent._csr import CSC
        return CSC.fromdense(
            self.data,
            nse=nse,
            index_dtype=index_dtype,
            backend=self.backend,
            precompute_weight_indices=precompute_weight_indices,
        )

    def tocoo(self):
        """Convert the dense matrix to COO through the CSR conversion path."""
        return self.tocsr().tocoo()

    def transpose(self, axes=None) -> 'Dense':
        """
        Return ``self.T`` as a new ``Dense`` representation.

        Only the standard matrix transpose is supported; ``axes`` must be
        ``None`` to match the two-dimensional data contract.
        """
        assert axes is None, f"axes must be None, got {axes}."
        return Dense(self.data.T, shape=self.shape[::-1], backend=self.backend, buffers=self.buffers)

    def __getitem__(self, index):
        """Index directly into the underlying dense matrix."""
        return self.data[_normalize_dense_index(index)]

    def slice_rows(self, index) -> 'Dense':
        """Return selected rows as a new ``Dense`` matrix."""
        data = self.data[_normalize_dense_index(index)]
        if data.ndim == 1:
            data = data[None, :]
        return Dense(data, backend=self.backend, buffers=self.buffers)

    def diag_add(self, other) -> 'Dense':
        """
        Add values to the main diagonal and return a new ``Dense``.

        ``other`` must have length ``min(self.shape)``.  Units are handled by
        the underlying ``jax.Array`` / ``brainunit.Quantity`` addition.
        """
        diag_size = min(self.shape)
        other = u.math.asarray(other)
        if other.shape != (diag_size,):
            raise ValueError(
                f"diag_add operand must have shape {(diag_size,)}, got {other.shape}."
            )
        rows = jnp.arange(diag_size)
        data = self.data.at[rows, rows].add(other)
        return Dense(data, shape=self.shape, backend=self.backend, buffers=self.buffers)

    def solve(self, b: Union[jax.Array, u.Quantity], tol=1e-6, reorder=1) -> Union[jax.Array, u.Quantity]:
        """
        Solve the dense linear system ``self @ x = b``.

        The ``tol`` and ``reorder`` parameters are accepted for API parity with
        sparse solvers and are currently unused.  Units are propagated as
        ``unit(b) / unit(self.data)``.
        """
        data, data_unit = u.split_mantissa_unit(self.data)
        b, b_unit = u.split_mantissa_unit(b)
        assert self.shape[0] == self.shape[1], "Dense.solve requires a square matrix."
        assert self.shape[0] == b.shape[0], (
            "The number of rows in the matrix must match "
            "the size of the right-hand side b."
        )
        res = jnp.linalg.solve(data, b)
        return u.maybe_decimal(res * b_unit / data_unit)

    def __matmul__(self, other):
        """
        Dense matrix multiplication: ``self @ other``.

        Binary event operands use :func:`binary_densemv` or
        :func:`binary_densemm` with ``transpose=False``.  Dense numeric operands
        use ordinary JAX matrix multiplication.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        event = _event_value(other)
        if event is not None:
            if event.ndim == 1:
                return binary_densemv(self.data, event, transpose=False, backend=self.backend)
            elif event.ndim == 2:
                return binary_densemm(self.data, event, transpose=False, backend=self.backend)
            else:
                raise NotImplementedError(f"matmul with object of shape {event.shape}")

        other = u.math.asarray(other)
        data, other = u.math.promote_dtypes(self.data, other)
        if other.ndim in (1, 2):
            return data @ other
        raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        """
        Reflected dense matrix multiplication: ``other @ self``.

        Binary event operands use the transposed dense binary primitives.
        Dense numeric operands use ordinary JAX matrix multiplication.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        event = _event_value(other)
        if event is not None:
            if event.ndim == 1:
                return binary_densemv(self.data, event, transpose=True, backend=self.backend)
            elif event.ndim == 2:
                return binary_densemm(self.data, event.T, transpose=True, backend=self.backend).T
            else:
                raise NotImplementedError(f"matmul with object of shape {event.shape}")

        other = u.math.asarray(other)
        data, other = u.math.promote_dtypes(self.data, other)
        if other.ndim in (1, 2):
            return other @ data
        raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None) -> 'Dense':
        """
        Apply a pre-spike-triggered dense plasticity update.

        Returns a new ``Dense`` with updated data and the same metadata.
        """
        data = update_dense_on_binary_pre(
            self.data,
            pre_spike,
            post_trace,
            w_min=w_min,
            w_max=w_max,
            backend=self.backend,
        )
        return self.with_data(data)

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None) -> 'Dense':
        """
        Apply a post-spike-triggered dense plasticity update.

        Returns a new ``Dense`` with updated data and the same metadata.
        """
        data = update_dense_on_binary_post(
            self.data,
            pre_trace,
            post_spike,
            w_min=w_min,
            w_max=w_max,
            backend=self.backend,
        )
        return self.with_data(data)

    def dt2t(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Report that direct dense ``dt2t`` is not implemented.

        The dense representation has no compressed per-synapse storage order to
        target.  Convert to CSR/CSC first when a per-synapse ``w * y`` output is
        required.
        """
        raise NotImplementedError(
            "Dense.dt2t is not implemented for explicit dense matrices. "
            "Use a sparse representation such as Dense.tocsr().dt2t(y, w) "
            "when per-synapse dt2t output is required."
        )

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Report that direct transposed dense ``dt2t`` is not implemented.

        Convert to CSR/CSC first when a per-synapse ``w * y`` output indexed by
        columns is required.
        """
        raise NotImplementedError(
            "Dense.dt2t_transposed is not implemented for explicit dense matrices. "
            "Use a sparse representation such as Dense.tocsr().dt2t_transposed(y, w) "
            "when per-synapse dt2t output is required."
        )
