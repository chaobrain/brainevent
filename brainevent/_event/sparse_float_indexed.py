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

from typing import Optional

from jax.tree_util import register_pytree_node_class

from .base import IndexedEventRepresentation, is_known_type
from .binary_indexed_extraction import binary_1d_array_index_p_call, binary_2d_array_index_p_call

__all__ = [
    'IndexedSpFloat1d',
    'IndexedSpFloat2d',
]


@register_pytree_node_class
class IndexedSpFloat1d(IndexedEventRepresentation):
    """A 1-D sparse float event array with pre-computed indices of non-zero elements.

    On construction the input array is scanned to extract the positions of
    all non-zero elements.  This is analogous to ``IndexedBinary1d`` but
    for arrays whose non-zero entries carry floating-point magnitudes.

    Parameters
    ----------
    value : array_like
        A 1-D numeric array.  Non-zero entries are considered active.
    backend : str or None, optional
        Compute backend for the index-extraction kernel (e.g. ``'numba'``,
        ``'pallas'``).  ``None`` selects the default.

    Notes
    -----
    The extracted ``spike_indices`` array has the same length as ``value``.
    Only the first ``spike_count[0]`` entries are meaningful; the rest are
    zero-filled.

    See Also
    --------
    IndexedBinary1d : Binary variant of indexed event arrays.
    SparseFloat : Non-indexed sparse float wrapper.
    """
    __module__ = 'brainevent'

    def __init__(self, value, backend: Optional[str] = None):
        super().__init__(value)
        self._spike_indices, self._spike_count = binary_1d_array_index_p_call(self._value, backend=backend)

    @property
    def spike_indices(self):
        """Indices of non-zero elements in the array.

        Returns
        -------
        jax.Array
            An int32 array of shape ``(k,)`` containing the positions of
            non-zero elements.
        """
        return self._spike_indices

    @property
    def spike_count(self):
        """Number of non-zero elements.

        Returns
        -------
        jax.Array
            An int32 array of shape ``(1,)`` holding the count.
        """
        return self._spike_count

    def __matmul__(self, oc):
        """Matrix multiplication is not supported for ``IndexedSpFloat1d``.

        Raises
        ------
        ValueError
            Always raised.
        """
        raise ValueError

    def __rmatmul__(self, oc):
        """Reverse matrix multiplication is not supported for ``IndexedSpFloat1d``.

        Raises
        ------
        ValueError
            Always raised.
        """
        raise ValueError

    def __imatmul__(self, oc):
        """In-place matrix multiplication (not truly in-place under JAX).

        Parameters
        ----------
        oc : array_like
            Right operand.

        Returns
        -------
        IndexedSpFloat1d
            A new instance wrapping the result.

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
            ``(value,)`` — the dynamic array leaf.
        aux_data : dict
            Contains ``_spike_indices`` and ``_spike_count``.
        """
        aux = {
            '_spike_indices': self._spike_indices,
            '_spike_count': self._spike_count,
        }
        return (self._value,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """Reconstruct from PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata including spike indices and count.
        flat_contents : tuple
            The dynamic array leaf.

        Returns
        -------
        IndexedSpFloat1d
            Reconstructed instance.
        """
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj


@register_pytree_node_class
class IndexedSpFloat2d(IndexedEventRepresentation):
    """A 2-D sparse float event array with pre-computed indices of non-zero elements.

    Extends ``IndexedEventRepresentation`` for 2-D sparse float arrays,
    extracting per-row indices of non-zero elements on construction.

    Parameters
    ----------
    value : array_like
        A 2-D numeric array of shape ``(batch, k)``.
    row_indices : None, optional
        Reserved for future use.

    Notes
    -----
    The 2-D index extraction is not yet implemented and will raise
    ``NotImplementedError`` during construction.

    See Also
    --------
    IndexedSpFloat1d : 1-D variant.
    IndexedBinary2d : Binary variant of 2-D indexed event arrays.
    """
    __module__ = 'brainevent'

    def __init__(
        self,
        value,
        *,
        row_indices=None,
    ):
        super().__init__(value)
        self._spike_indices, self._spike_count = binary_2d_array_index_p_call(self._value)

    @property
    def spike_indices(self):
        """Indices of non-zero elements per row.

        Returns
        -------
        jax.Array
            Index array for the active elements.
        """
        return self._spike_indices

    @property
    def spike_count(self):
        """Number of non-zero elements per row.

        Returns
        -------
        jax.Array
            Count array for non-zero entries.
        """
        return self._spike_count

    def __matmul__(self, oc):
        """Matrix multiplication is not supported for ``IndexedSpFloat2d``.

        Raises
        ------
        ValueError
            Always raised.
        """
        raise ValueError

    def __rmatmul__(self, oc):
        """Reverse matrix multiplication is not supported for ``IndexedSpFloat2d``.

        Raises
        ------
        ValueError
            Always raised.
        """
        raise ValueError

    def __imatmul__(self, oc):
        """In-place matrix multiplication (not truly in-place under JAX).

        Parameters
        ----------
        oc : array_like
            Right operand.

        Returns
        -------
        IndexedSpFloat2d
            A new instance wrapping the result.

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
            ``(value,)`` — the dynamic array leaf.
        aux_data : dict
            Contains ``_spike_indices`` and ``_spike_count``.
        """
        aux = {
            '_spike_indices': self._spike_indices,
            '_spike_count': self._spike_count,
        }
        return (self._value,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """Reconstruct from PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata including spike indices and count.
        flat_contents : tuple
            The dynamic array leaf.

        Returns
        -------
        IndexedSpFloat2d
            Reconstructed instance.
        """
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj
