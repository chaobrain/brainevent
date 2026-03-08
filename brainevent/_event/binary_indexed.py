# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Optional

from jax.tree_util import register_pytree_node_class

from .base import IndexedEventRepresentation, is_known_type
from .binary_indexed_extraction import binary_1d_array_index_p_call, binary_2d_array_index_p_call

__all__ = [
    'IndexedBinary1d',
    'IndexedBinary2d',
]


@register_pytree_node_class
class IndexedBinary1d(IndexedEventRepresentation):
    """A 1-D binary event array with pre-computed stream compaction.

    On construction the input array is scanned to extract the positions of
    all non-zero (active) elements into ``active_ids``, together with the
    total count ``n_active``.  This pre-indexing enables downstream kernels
    to iterate only over active elements.

    Parameters
    ----------
    value : array_like
        A 1-D boolean or numeric array.  Non-zero entries are considered
        active events.
    backend : str or None, optional
        Compute backend for the index-extraction kernel (e.g. ``'numba'``,
        ``'pallas'``, ``'cuda_raw'``).  ``None`` selects the default.

    Attributes
    ----------
    active_ids : jax.Array
        Int32 array of shape ``(n,)`` where ``n`` is the length of the input.
        The first ``n_active[0]`` entries contain the indices of non-zero
        elements.  Trailing entries are undefined.
    n_active : jax.Array
        Int32 array of shape ``(1,)`` holding the count of active elements.
    length : int
        Original vector length ``n`` (Python int, static).

    See Also
    --------
    IndexedBinary2d : 2-D variant with bit-pack and row compaction.
    BinaryArray : Simpler wrapper without pre-computed indices.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import IndexedBinary1d
        >>> spikes = IndexedBinary1d(jnp.array([True, False, True, False]))
        >>> spikes.n_active
        Array([2], dtype=int32)
        >>> spikes.active_ids[:spikes.n_active[0]]
        Array([0, 2], dtype=int32)
    """
    __module__ = 'brainevent'

    def __init__(self, value, backend: Optional[str] = None):
        super().__init__(value)
        self._active_ids, self._n_active = binary_1d_array_index_p_call(self._value, backend=backend)

    @property
    def active_ids(self):
        """Indices of non-zero elements in the array.

        Returns
        -------
        jax.Array
            An int32 array of shape ``(n,)`` containing the positions of
            active elements in the first ``n_active[0]`` entries.
        """
        return self._active_ids

    @property
    def n_active(self):
        """Number of active (non-zero) elements.

        Returns
        -------
        jax.Array
            An int32 array of shape ``(1,)`` holding the count of non-zero
            entries.
        """
        return self._n_active

    @property
    def length(self):
        """Original vector length.

        Returns
        -------
        int
            The length of the original 1-D input array.
        """
        return self._value.shape[0]

    # Keep backward-compatible aliases
    @property
    def spike_indices(self):
        """Alias for ``active_ids`` (deprecated)."""
        return self._active_ids

    @property
    def spike_count(self):
        """Alias for ``n_active`` (deprecated)."""
        return self._n_active

    def __matmul__(self, oc):
        """Matrix multiplication is not supported for ``IndexedBinary1d``.

        Raises
        ------
        ValueError
            Always raised -- use ``BinaryArray`` for ``@`` operations.
        """
        raise ValueError

    def __rmatmul__(self, oc):
        """Reverse matrix multiplication is not supported for ``IndexedBinary1d``.

        Raises
        ------
        ValueError
            Always raised -- use ``BinaryArray`` for ``@`` operations.
        """
        raise ValueError

    def __imatmul__(self, oc):
        """In-place matrix multiplication (not truly in-place under JAX).

        Raises
        ------
        ValueError
            Delegates to ``__matmul__`` which always raises.
        """
        if is_known_type(oc):
            return self.with_value(self.__matmul__(oc))
        return self.with_value(oc.__rmatmul__(self))

    def tree_flatten(self):
        """Flatten for JAX PyTree serialisation.

        Returns
        -------
        children : tuple
            ``(value,)`` -- the dynamic array leaf.
        aux_data : dict
            Contains ``_active_ids`` and ``_n_active`` as static metadata.
        """
        aux = {
            '_active_ids': self._active_ids,
            '_n_active': self._n_active,
        }
        return (self._value,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """Reconstruct from PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata including active_ids and n_active.
        flat_contents : tuple
            The dynamic array leaf.

        Returns
        -------
        IndexedBinary1d
            Reconstructed instance.
        """
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj


@register_pytree_node_class
class IndexedBinary2d(IndexedEventRepresentation):
    """A 2-D binary event array with bit-pack and row-level compaction.

    On construction the input array is processed to produce:

    1. A **bit-packed** representation where every 32 batch columns are
       compressed into one uint32 word.  This is used by gather-mode
       kernels so that a warp of 32 threads can read 32 batch columns
       in a single memory transaction.
    2. A **stream-compacted** list of active row indices (rows with at
       least one non-zero column).  This is used by scatter-mode kernels
       to skip entirely-zero rows.

    Both representations are computed in a single fused GPU kernel.

    Parameters
    ----------
    value : array_like
        A 2-D boolean or numeric array of shape ``(n_pre, n_batch)``.
    backend : str or None, optional
        Compute backend for the extraction kernel (e.g. ``'numba'``,
        ``'cuda_raw'``).  ``None`` selects the default.

    Attributes
    ----------
    packed : jax.Array
        Uint32 array of shape ``(n_pre, n_batch_packed)`` where
        ``n_batch_packed = ceil(n_batch / 32)``.  Bit ``b`` of
        ``packed[i, w]`` equals ``value[i, w*32 + b]``.
    active_ids : jax.Array
        Int32 array of shape ``(n_pre,)`` listing row indices that
        have at least one non-zero batch column.  Only the first
        ``n_active[0]`` entries are valid.
    n_active : jax.Array
        Int32 array of shape ``(1,)`` with the count of active rows.

    See Also
    --------
    IndexedBinary1d : 1-D variant.
    BinaryArray : Simpler wrapper without pre-computed indices.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import IndexedBinary2d
        >>> B = jnp.array([[True, False, True],
        ...                [False, False, False],
        ...                [True, True, False]])
        >>> ib = IndexedBinary2d(B)
        >>> ib.n_active
        Array([2], dtype=int32)
        >>> ib.active_ids[:ib.n_active[0]]
        Array([0, 2], dtype=int32)
        >>> ib.packed.shape
        (3, 1)
    """
    __module__ = 'brainevent'

    def __init__(
        self,
        value,
        *,
        backend: Optional[str] = None,
    ):
        super().__init__(value)
        self._packed, self._active_ids, self._n_active = binary_2d_array_index_p_call(
            self._value, backend=backend
        )

    @property
    def packed(self):
        """Bit-packed representation of the binary matrix.

        Returns
        -------
        jax.Array
            Uint32 array of shape ``(n_pre, ceil(n_batch / 32))``.
        """
        return self._packed

    @property
    def active_ids(self):
        """Row indices with at least one non-zero batch column.

        Returns
        -------
        jax.Array
            Int32 array of shape ``(n_pre,)``.  Only the first
            ``n_active[0]`` entries are valid.
        """
        return self._active_ids

    @property
    def n_active(self):
        """Number of active rows.

        Returns
        -------
        jax.Array
            Int32 array of shape ``(1,)``.
        """
        return self._n_active

    # Keep backward-compatible aliases
    @property
    def spike_indices(self):
        """Alias for ``active_ids`` (deprecated)."""
        return self._active_ids

    @property
    def spike_count(self):
        """Alias for ``n_active`` (deprecated)."""
        return self._n_active

    def __matmul__(self, oc):
        """Matrix multiplication is not supported for ``IndexedBinary2d``.

        Raises
        ------
        ValueError
            Always raised.
        """
        raise ValueError

    def __rmatmul__(self, oc):
        """Reverse matrix multiplication is not supported for ``IndexedBinary2d``.

        Raises
        ------
        ValueError
            Always raised.
        """
        raise ValueError

    def __imatmul__(self, oc):
        """In-place matrix multiplication (not truly in-place under JAX).

        Raises
        ------
        ValueError
            Delegates to ``__matmul__`` which always raises.
        """
        if is_known_type(oc):
            return self.with_value(self.__matmul__(oc))
        return self.with_value(oc.__rmatmul__(self))

    def tree_flatten(self):
        """Flatten for JAX PyTree serialisation.

        Returns
        -------
        children : tuple
            ``(value,)`` -- the dynamic array leaf.
        aux_data : dict
            Contains ``_packed``, ``_active_ids``, and ``_n_active``.
        """
        aux = {
            '_packed': self._packed,
            '_active_ids': self._active_ids,
            '_n_active': self._n_active,
        }
        return (self._value,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """Reconstruct from PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata including packed, active_ids, and n_active.
        flat_contents : tuple
            The dynamic array leaf.

        Returns
        -------
        IndexedBinary2d
            Reconstructed instance.
        """
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj
