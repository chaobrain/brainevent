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

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .base import EventRepresentation

__all__ = [
    'BitPackedBinary',
]


def _bitpack_along_axis(arr, axis):
    """Pack a boolean array into uint32 words along the given axis.

    Each uint32 word stores 32 binary values.  Bit ``b`` of word ``w``
    corresponds to element ``w * 32 + b`` along the packed axis.

    Parameters
    ----------
    arr : jax.Array
        Input array (any dtype; non-zero values are treated as ``True``).
    axis : int
        Axis along which to pack (already normalised to non-negative).

    Returns
    -------
    jax.Array
        Packed uint32 array.  The packed axis has length
        ``ceil(arr.shape[axis] / 32)``.
    """
    arr = jnp.asarray(arr, dtype=jnp.bool_)
    n = arr.shape[axis]
    n_packed = (n + 31) // 32

    # Pad to a multiple of 32 along *axis*
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, n_packed * 32 - n)
    padded = jnp.pad(arr, pad_width).astype(jnp.uint32)

    # Insert a group-of-32 dimension right after *axis*
    new_shape = list(padded.shape)
    new_shape[axis] = n_packed
    new_shape.insert(axis + 1, 32)
    reshaped = padded.reshape(new_shape)

    # Bit-shift each element by its position within the group
    shifts_shape = [1] * reshaped.ndim
    shifts_shape[axis + 1] = 32
    shifts = jnp.arange(32, dtype=jnp.uint32).reshape(shifts_shape)

    # For binary (0/1) values with non-overlapping bit positions, sum == OR
    packed = jnp.sum(reshaped << shifts, axis=axis + 1, dtype=jnp.uint32)
    return packed


@register_pytree_node_class
class BitPackedBinary(EventRepresentation):
    """Bit-packed binary event representation.

    ``BitPackedBinary`` stores binary spike data as uint32 words where each
    word encodes 32 consecutive spikes.  Bit ``b`` of word ``w`` corresponds
    to element ``w * 32 + b`` along the packed axis.  This 8x memory
    reduction (vs bool) improves GPU cache utilisation, especially for
    gather-mode FCN sparse kernels.

    Instances are typically created via :meth:`BinaryArray.bitpack` rather
    than direct construction.

    Parameters
    ----------
    arr : jax.Array
        The original binary spike array (bool or 0/1).
    axis : int, optional
        Axis along which to pack.  Default is ``-1`` (last axis).

    Notes
    -----
    The class stores both the original spike array (``value``) and the
    packed uint32 representation (``packed``).  The original array is
    used for autodiff (gradient propagation), while the packed array
    is used for efficient CUDA kernel computation.

    The class is registered as a JAX PyTree node, so it is compatible with
    ``jax.jit``, ``jax.grad``, ``jax.vmap``, and other transformations.

    See Also
    --------
    BinaryArray : Unpacked binary event representation.
    BinaryArray.bitpack : Creates a ``BitPackedBinary`` from a ``BinaryArray``.
    """
    __slots__ = ('_value', '_packed', '_original_shape', '_pack_axis')
    __module__ = 'brainevent'

    def __init__(self, arr, *, axis=-1):
        # _value stores the original spike array (for gradient propagation)
        super().__init__(arr)
        if arr.ndim not in (1, 2):
            raise ValueError(f"bitpack() only supports 1-D and 2-D arrays, got {arr.ndim}-D.")
        self._original_shape = tuple(arr.shape)
        norm_axis = axis % len(self._original_shape)
        self._packed = _bitpack_along_axis(self._value, norm_axis)
        self._pack_axis = norm_axis

    @property
    def packed(self):
        """The packed uint32 array.

        Returns
        -------
        jax.Array
            Packed uint32 data where each word stores 32 binary values.
        """
        return self._packed

    @property
    def original_shape(self):
        """Shape of the original (unpacked) boolean array.

        Returns
        -------
        tuple[int, ...]
            The shape before bit-packing.
        """
        return self._original_shape

    @property
    def pack_axis(self):
        """Axis along which the packing was performed.

        Returns
        -------
        int
            Non-negative axis index.
        """
        return self._pack_axis

    @property
    def shape(self):
        """Logical shape (original unpacked shape).

        Returns
        -------
        tuple[int, ...]
            The shape of the original boolean array, not the packed
            uint32 array.  This makes ``BitPackedBinary`` shape-compatible
            with the original ``BinaryArray``.
        """
        return self._original_shape

    @property
    def ndim(self):
        """Number of dimensions of the original array.

        Returns
        -------
        int
            Same as ``len(self.original_shape)``.
        """
        return len(self._original_shape)

    def __matmul__(self, oc):
        """Compute ``self @ oc``.

        Returns ``NotImplemented`` to allow the right operand (e.g. a
        ``FixedPostNumConn``) to handle the operation via ``__rmatmul__``.
        """
        return NotImplemented

    def __rmatmul__(self, oc):
        """Compute ``oc @ self``.

        Returns ``NotImplemented`` to allow the left operand (e.g. a
        ``FixedPostNumConn``) to handle the operation via ``__matmul__``.
        """
        return NotImplemented

    def dot(self, oc):
        pass

    def tree_flatten(self):
        """Flatten this instance for JAX PyTree serialisation.

        Returns
        -------
        children : tuple
            ``(value, packed)`` — the original spike array and the
            packed uint32 array.  Both are dynamic JAX-traced leaves.
        aux_data : dict
            Contains ``original_shape`` and ``pack_axis``.
        """
        aux = {
            'original_shape': self._original_shape,
            'pack_axis': self._pack_axis,
        }
        return (self._value, self._packed), aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """Reconstruct a ``BitPackedBinary`` from its PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata produced by ``tree_flatten``.
        flat_contents : tuple
            Dynamic leaves — the original spike array and packed uint32 array.

        Returns
        -------
        BitPackedBinary
            A new instance wrapping both arrays.
        """
        value, packed = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        obj._packed = packed
        obj._original_shape = aux_data['original_shape']
        obj._pack_axis = aux_data['pack_axis']
        return obj
