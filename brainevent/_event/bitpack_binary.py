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

from brainevent._dense import binary_densemm, binary_densemv
from brainevent._error import MathError
from .base import EventRepresentation, extract_raw_value, is_known_type

__all__ = [
    'BitPackedBinary',
    'bitpack',
]


def bitpack(arr, axis):
    """Pack a boolean array into uint32 words along the given axis.

    Each uint32 word stores 32 binary values.  Bit ``b`` of word ``w``
    corresponds to element ``w * 32 + b`` along the packed axis.

    Parameters
    ----------
    arr : jax.Array
        Input array (any dtype; non-zero values are treated as ``True``).
    axis : int
        Axis along which to pack.

    Returns
    -------
    jax.Array
        Packed uint32 array.  The packed axis has length
        ``ceil(arr.shape[axis] / 32)``.
    """
    arr = jnp.asarray(arr, dtype=jnp.bool_)
    axis = axis % arr.ndim
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
    word encodes 32 consecutive spikes.  Packing is performed along **every**
    axis, so the number of packed arrays equals the number of dimensions.

    Instances are typically created via :meth:`BinaryArray.bitpack` rather
    than direct construction.

    Parameters
    ----------
    arr : jax.Array
        The original binary spike array (bool or 0/1).

    Notes
    -----
    The class stores both the original spike array (``value``) and one
    packed uint32 representation per axis (``packed``).  The original array
    is used for autodiff (gradient propagation), while the packed arrays
    are used for efficient CUDA kernel computation.

    The class is registered as a JAX PyTree node, so it is compatible with
    ``jax.jit``, ``jax.grad``, ``jax.vmap``, and other transformations.

    See Also
    --------
    BinaryArray : Unpacked binary event representation.
    BinaryArray.bitpack : Creates a ``BitPackedBinary`` from a ``BinaryArray``.
    """
    __slots__ = ('_value', '_packed', '_original_shape')
    __module__ = 'brainevent'

    def __init__(self, arr):
        super().__init__(arr)
        self._original_shape = tuple(self._value.shape)
        self._packed = tuple(
            bitpack(self._value, axis) for axis in range(self._value.ndim)
        )

    @property
    def packed(self):
        """Tuple of packed uint32 arrays, one per axis.

        Returns
        -------
        tuple[jax.Array, ...]
            ``packed[i]`` is the uint32 array obtained by packing along
            axis ``i``.  Its shape matches the original shape except that
            dimension ``i`` is ``ceil(original_shape[i] / 32)``.
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
    def shape(self):
        """Logical shape (original unpacked shape).

        Returns
        -------
        tuple[int, ...]
            The shape of the original boolean array, not the packed
            uint32 arrays.  This makes ``BitPackedBinary`` shape-compatible
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
        """Compute ``self @ oc`` using event-driven binary multiplication.

        Parameters
        ----------
        oc : array_like
            Right operand, must be a 2-D dense weight matrix of shape
            ``(k, n)`` where ``k == self.shape[-1]``.

        Returns
        -------
        jax.Array
            For a 1-D ``self`` of shape ``(k,)``: a vector of shape ``(n,)``.
            For a 2-D ``self`` of shape ``(m, k)``: a matrix of shape ``(m, n)``.

        Raises
        ------
        MathError
            If ``self`` has more than 2 dimensions or is a scalar.
        """
        if is_known_type(oc):
            oc = extract_raw_value(oc)
            if self.ndim not in (1, 2):
                raise MathError(
                    f"Matrix multiplication is only supported "
                    f"for 1D and 2D arrays. Got {self.ndim}D array."
                )
            if self.ndim == 0:
                raise MathError("Matrix multiplication is not supported for scalar arrays.")
            assert oc.ndim == 2, (
                f"Right operand must be a 2D array in "
                f"matrix multiplication. Got {oc.ndim}D array."
            )
            assert self.shape[-1] == oc.shape[0], (
                f"Incompatible dimensions for matrix multiplication: "
                f"{self.shape[-1]} and {oc.shape[0]}."
            )
            if self.ndim == 1:
                return binary_densemv(oc, self.value, transpose=True)
            else:
                return binary_densemm(oc, self.value.T, transpose=True).T
        else:
            return oc.__rmatmul__(self)

    def __rmatmul__(self, oc):
        """Compute ``oc @ self`` using event-driven binary multiplication.

        Parameters
        ----------
        oc : array_like
            Left operand, must be a 2-D dense weight matrix of shape
            ``(m, k)`` where ``k == self.shape[0]``.

        Returns
        -------
        jax.Array
            For a 1-D ``self`` of shape ``(k,)``: a vector of shape ``(m,)``.
            For a 2-D ``self`` of shape ``(k, n)``: a matrix of shape ``(m, n)``.

        Raises
        ------
        MathError
            If ``self`` has more than 2 dimensions or is a scalar.
        """
        if is_known_type(oc):
            oc = extract_raw_value(oc)
            if self.ndim not in (1, 2):
                raise MathError(
                    f"Matrix multiplication is only supported "
                    f"for 1D and 2D arrays. Got {self.ndim}D array."
                )
            if self.ndim == 0:
                raise MathError("Matrix multiplication is not supported for scalar arrays.")
            assert oc.ndim == 2, (
                f"Left operand must be a 2D array in "
                f"matrix multiplication. Got {oc.ndim}D array."
            )
            assert oc.shape[-1] == self.shape[0], (
                f"Incompatible dimensions for matrix multiplication: "
                f"{oc.shape[-1]} and {self.shape[0]}."
            )
            if self.ndim == 1:
                return binary_densemv(oc, self.value, transpose=False)
            else:
                return binary_densemm(oc, self.value, transpose=False)
        else:
            return oc.__matmul__(self)

    @property
    def T(self):
        """Transpose of the bit-packed binary array.

        Returns
        -------
        BitPackedBinary
            A new transposed instance.
        """
        return self.transpose()

    def transpose(self, *axes):
        """Return a transposed ``BitPackedBinary``.

        Parameters
        ----------
        *axes : int, optional
            Axis permutation.  If omitted, reverses the axis order
            (standard transpose).

        Returns
        -------
        BitPackedBinary
            A new instance with permuted axes.  Both ``value`` and
            all ``packed`` arrays are transposed accordingly.
        """
        if not axes:
            perm = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        else:
            perm = tuple(axes)

        obj = object.__new__(BitPackedBinary)
        obj._value = jnp.transpose(self._value, perm)
        obj._original_shape = tuple(self._original_shape[i] for i in perm)
        # new packed[i] corresponds to packing along new axis i,
        # which was old axis perm[i]
        obj._packed = tuple(
            jnp.transpose(self._packed[perm[i]], perm)
            for i in range(self.ndim)
        )
        return obj

    def dot(self, oc):
        return self.__matmul__(oc)

    def tree_flatten(self):
        """Flatten this instance for JAX PyTree serialisation.

        Returns
        -------
        children : tuple
            ``(value, packed[0], packed[1], ...)`` — the original spike
            array followed by packed uint32 arrays for each axis.
        aux_data : dict
            Contains ``original_shape``.
        """
        children = (self._value,) + self._packed
        aux = {
            'original_shape': self._original_shape,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """Reconstruct a ``BitPackedBinary`` from its PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata produced by ``tree_flatten``.
        flat_contents : tuple
            Dynamic leaves — the original spike array followed by
            packed uint32 arrays for each axis.

        Returns
        -------
        BitPackedBinary
            A new instance wrapping all arrays.
        """
        obj = object.__new__(cls)
        obj._value = flat_contents[0]
        obj._packed = tuple(flat_contents[1:])
        obj._original_shape = aux_data['original_shape']
        return obj
