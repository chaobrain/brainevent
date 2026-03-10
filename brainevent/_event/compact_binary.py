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

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .bitpack_binary import bitpack
from .compact import (
    _compact_1d_jax,
    binary_2d_array_index_p_call,
)

__all__ = [
    'CompactBinary',
]

@register_pytree_node_class
class CompactBinary:
    """Binary event representation with bitpack and stream compaction.

    Combines two compression strategies for binary (0/1) spike data:

    - **Bitpack**: Packs 32 binary values into each uint32 word.
    - **Compaction**: Extracts indices of active (non-zero) elements into
      a contiguous list, enabling scatter kernels to skip inactive rows.

    For 1D input ``(n,)``:
        - Bitpack along axis 0: ``packed`` shape ``(ceil(n/32),)``
        - Compaction: indices of non-zero elements

    For 2D input ``(n, batch_size)``:
        - Bitpack along axis 1 (batch): ``packed`` shape ``(n, ceil(batch_size/32))``
        - Compaction along axis 0 (feature): indices of rows active in ANY batch

    Instances are typically created via :meth:`from_array` rather than
    direct construction.

    Parameters
    ----------
    packed : jax.Array
        Bit-packed uint32 data.
    active_ids : jax.Array
        Int32 array of active element indices, shape ``(n_orig,)``.
    n_active : jax.Array
        Int32 scalar (shape ``(1,)``) count of active elements.
    value : jax.Array
        Original dense binary array (for autodiff).
    n_orig : int
        Original feature dimension size.
    batch_size : int or None
        Batch dimension size, or None for 1D input.
    bit_width : int
        Bit width for packing (32).

    See Also
    --------
    BitPackedBinary : Bit-packed only (no compaction).
    """
    __slots__ = ('_packed', '_active_ids', '_n_active', '_value',
                 '_n_orig', '_batch_size', '_bit_width')
    __array_priority__ = 100
    __module__ = 'brainevent'

    def __init__(self, packed, active_ids, n_active, value,
                 n_orig, batch_size=None, bit_width=32):
        self._packed = packed
        self._active_ids = active_ids
        self._n_active = n_active
        self._value = value
        self._n_orig = n_orig
        self._batch_size = batch_size
        self._bit_width = bit_width

    @classmethod
    def from_array(cls, x, bit_width=32):
        """Create a ``CompactBinary`` from a raw binary array.

        Parameters
        ----------
        x : jax.Array
            Binary array of shape ``(n,)`` or ``(n, batch_size)``.
            Non-zero values are treated as 1.
        bit_width : int, optional
            Bit width for packing.  Must be 32.

        Returns
        -------
        CompactBinary
            New instance with bitpack and compaction data.

        Raises
        ------
        ValueError
            If ``x`` is not 1D or 2D, or ``bit_width`` is not 32.
        """
        if bit_width != 32:
            raise ValueError(f"Only bit_width=32 is supported, got {bit_width}.")
        x = jnp.asarray(x)
        if x.ndim == 1:
            packed = bitpack(x, axis=0)
            # Use JAX prefix-sum for 1D (cheaper under vmap than CUDA
            # primitive, which would launch sequential kernels per batch).
            active_ids, n_active = _compact_1d_jax(x)
            return cls(packed, active_ids, n_active, x,
                       n_orig=x.shape[0], batch_size=None, bit_width=32)
        elif x.ndim == 2:
            # Fused CUDA kernel: bitpack + compaction in one launch
            packed, active_ids, n_active = binary_2d_array_index_p_call(x)
            return cls(packed, active_ids, n_active, x,
                       n_orig=x.shape[0], batch_size=x.shape[1], bit_width=32)
        else:
            raise ValueError(
                f"CompactBinary only supports 1D and 2D arrays, got {x.ndim}D."
            )

    @classmethod
    def from_array_light(cls, x, bit_width=32):
        """Create a ``CompactBinary`` with deferred compaction.

        For 1D input, skips computing ``active_ids`` / ``n_active``
        (uses zeros). This is faster under ``jax.vmap`` because the
        MV→MM batching rule recomputes compaction for the merged matrix.

        For 2D input, identical to :meth:`from_array`.

        Parameters
        ----------
        x : jax.Array
            Binary array of shape ``(n,)`` or ``(n, batch_size)``.
        bit_width : int, optional
            Must be 32.

        Returns
        -------
        CompactBinary
        """
        if bit_width != 32:
            raise ValueError(f"Only bit_width=32 is supported, got {bit_width}.")
        x = jnp.asarray(x)
        if x.ndim == 1:
            packed = bitpack(x, axis=0)
            n = x.shape[0]
            # Skip compaction — use zeros. The MV→MM batching rule
            # recomputes compaction for the merged 2D matrix.
            active_ids = jnp.zeros(n, dtype=jnp.int32)
            n_active = jnp.zeros(1, dtype=jnp.int32)
            return cls(packed, active_ids, n_active, x,
                       n_orig=n, batch_size=None, bit_width=32)
        elif x.ndim == 2:
            packed, active_ids, n_active = binary_2d_array_index_p_call(x)
            return cls(packed, active_ids, n_active, x,
                       n_orig=x.shape[0], batch_size=x.shape[1], bit_width=32)
        else:
            raise ValueError(
                f"CompactBinary only supports 1D and 2D arrays, got {x.ndim}D."
            )

    @classmethod
    def from_packed(cls, packed, active_ids, n_active, value,
                    n_orig, batch_size=None, bit_width=32):
        """Construct from pre-computed bitpack and compaction data.

        Parameters
        ----------
        packed : jax.Array
            Pre-computed bit-packed uint32 array.
        active_ids : jax.Array
            Pre-computed active indices, int32, shape ``(n_orig,)``.
        n_active : jax.Array
            Pre-computed active count, int32, shape ``(1,)``.
        value : jax.Array
            Original dense binary array.
        n_orig : int
            Original feature dimension.
        batch_size : int or None
            Batch size, or None for 1D.
        bit_width : int
            Must be 32.

        Returns
        -------
        CompactBinary
        """
        return cls(packed, active_ids, n_active, value,
                   n_orig=n_orig, batch_size=batch_size, bit_width=bit_width)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def packed(self):
        """Bit-packed uint32 data.

        Returns
        -------
        jax.Array
            Shape ``(n_words,)`` for 1D or ``(n_orig, batch_words)`` for 2D.
        """
        return self._packed

    @property
    def active_ids(self):
        """Indices of active elements.

        Returns
        -------
        jax.Array
            Shape ``(n_orig,)``, int32.  Only ``[:n_active]`` entries valid.
        """
        return self._active_ids

    @property
    def n_active(self):
        """Number of active elements.

        Returns
        -------
        jax.Array
            Shape ``(1,)``, int32.
        """
        return self._n_active

    @property
    def value(self):
        """Original dense binary array (for autodiff).

        Returns
        -------
        jax.Array
        """
        return self._value

    @property
    def n_orig(self):
        """Original feature dimension size.

        Returns
        -------
        int
        """
        return self._n_orig

    @property
    def batch_size(self):
        """Batch dimension size, or None for 1D.

        Returns
        -------
        int or None
        """
        return self._batch_size

    @property
    def bit_width(self):
        """Bit width for packing (32).

        Returns
        -------
        int
        """
        return self._bit_width

    @property
    def shape(self):
        """Logical shape of the original array.

        Returns
        -------
        tuple[int, ...]
        """
        if self._batch_size is None:
            return (self._n_orig,)
        return (self._n_orig, self._batch_size)

    @property
    def ndim(self):
        """Number of dimensions of the original array.

        Returns
        -------
        int
        """
        return 1 if self._batch_size is None else 2

    @property
    def dtype(self):
        """Dtype of the original array.

        Returns
        -------
        jnp.dtype
        """
        return self._value.dtype

    @property
    def size(self):
        """Total number of elements in the original array.

        Returns
        -------
        int
        """
        if self._batch_size is None:
            return self._n_orig
        return self._n_orig * self._batch_size

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def to_dense(self):
        """Reconstruct the original dense binary array.

        Returns
        -------
        jax.Array
            The original dense array stored during construction.
        """
        return self._value

    def __repr__(self):
        return (
            f"CompactBinary(shape={self.shape}, "
            f"n_active=<dynamic>, bit_width={self._bit_width})"
        )

    # ------------------------------------------------------------------
    # JAX PyTree protocol
    # ------------------------------------------------------------------

    def tree_flatten(self):
        children = (self._packed, self._active_ids, self._n_active, self._value)
        aux = {
            'n_orig': self._n_orig,
            'batch_size': self._batch_size,
            'bit_width': self._bit_width,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        packed, active_ids, n_active, value = flat_contents
        return cls(
            packed, active_ids, n_active, value,
            n_orig=aux_data['n_orig'],
            batch_size=aux_data['batch_size'],
            bit_width=aux_data['bit_width'],
        )
