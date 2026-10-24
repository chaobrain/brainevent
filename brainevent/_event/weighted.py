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
    'WeightedArray',
]


@register_pytree_node_class
class WeightedArray(EventRepresentation):
    """Array wrapper for value-weighted (count or graded) events.

    ``WeightedArray`` represents a mostly-zero array whose non-zero entries
    carry a value, for example the number of spikes a neuron emitted in one
    step. Matrix multiplication with ``@`` is exact (``x @ W``) but only
    visits rows/columns where the event is non-zero.

    Parameters
    ----------
    value : array_like
        Event values. Integer (``int8``, ``uint8``, ``int16``, ``int32``) or
        floating-point dtypes; signed values are allowed. Boolean arrays are
        accepted and behave like :class:`BinaryArray`.

    Notes
    -----
    Given an event vector ``x`` of shape ``(k,)`` and a weight matrix ``W``
    of shape ``(k, n)``, ``x @ W`` computes

        y[j] = sum_{i : x[i] != 0} x[i] * W[i, j]

    Unlike :class:`BinaryArray`, the value is not binarised, and the
    derivatives are exact:

    - ``d y / d W[i, j] = x[i]``;
    - ``d y / d x[i] = W[i, :]`` (integer events get ``float0`` cotangents).

    ``WeightedArray`` is not a subclass of :class:`BinaryArray`.

    See Also
    --------
    BinaryArray : Binary events (the value is not used).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainevent as be
        >>> counts = be.WeightedArray(jnp.array([0, 2, 1], dtype=jnp.int8))
        >>> W = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
        >>> counts @ W
        Array([11., 14.], dtype=float32)
    """
    __module__ = 'brainevent'

    def __init__(self, value):
        super().__init__(value)
        dtype = jnp.dtype(self.dtype)
        if not (jnp.issubdtype(dtype, jnp.integer)
                or jnp.issubdtype(dtype, jnp.floating)
                or dtype == jnp.bool_):
            raise TypeError(
                f'WeightedArray values must be boolean, integer or floating-point, got {dtype}.'
            )

    @property
    def T(self):
        """Transpose of the underlying array.

        Returns
        -------
        jax.Array
            The transposed raw array (not wrapped in ``WeightedArray``).
        """
        return self.value.T

    def transpose(self, *axes):
        """Return the underlying array with axes permuted.

        Parameters
        ----------
        *axes : int, optional
            Axis permutation. If omitted, reverses the axis order.

        Returns
        -------
        jax.Array
            The transposed raw array (not wrapped in ``WeightedArray``).
        """
        return self.value.transpose(*axes)

    def _check_matmul(self, oc, left: bool):
        if self.ndim == 0:
            raise MathError("Matrix multiplication is not supported for scalar arrays.")
        if self.ndim not in (1, 2):
            raise MathError(
                f"Matrix multiplication is only supported for 1D and 2D arrays. "
                f"Got {self.ndim}D array."
            )
        side = 'Right' if left else 'Left'
        assert oc.ndim == 2, (f"{side} operand must be a 2D array in "
                              f"matrix multiplication. Got {oc.ndim}D array.")
        inner_self = self.shape[-1] if left else self.shape[0]
        inner_oc = oc.shape[0] if left else oc.shape[-1]
        assert inner_self == inner_oc, (
            f"Incompatible dimensions for matrix multiplication: {inner_self} and {inner_oc}."
        )

    def __matmul__(self, oc):
        """Compute ``self @ oc`` with value-weighted event-driven multiplication.

        Parameters
        ----------
        oc : array_like or sparse matrix
            Right operand: a 2-D dense matrix of shape ``(k, n)`` with
            ``k == self.shape[-1]``, or a brainevent matrix.

        Returns
        -------
        jax.Array
            ``self.value @ oc`` of shape ``(n,)`` or ``(m, n)``.

        Raises
        ------
        MathError
            If ``self`` is a scalar or has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent as be
            >>> x = be.WeightedArray(jnp.array([[1., 0.], [0., -2.]]))
            >>> x @ jnp.array([[1., 2.], [3., 4.]])
            Array([[ 1.,  2.],
                   [-6., -8.]], dtype=float32)
        """
        if is_known_type(oc):
            oc = extract_raw_value(oc)
            self._check_matmul(oc, left=True)
            if self.ndim == 1:
                return binary_densemv(oc, self.value, transpose=True, weighted=True)
            return binary_densemm(oc, self.value.T, transpose=True, weighted=True).T
        return oc.__rmatmul__(self)

    def __rmatmul__(self, oc):
        """Compute ``oc @ self`` with value-weighted event-driven multiplication.

        Parameters
        ----------
        oc : array_like or sparse matrix
            Left operand: a 2-D dense matrix of shape ``(m, k)`` with
            ``k == self.shape[0]``, or a brainevent matrix.

        Returns
        -------
        jax.Array
            ``oc @ self.value`` of shape ``(m,)`` or ``(m, n)``.

        Raises
        ------
        MathError
            If ``self`` is a scalar or has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent as be
            >>> W = jnp.array([[1., 2., 3.], [4., 5., 6.]])
            >>> W @ be.WeightedArray(jnp.array([0., 2., 1.]))
            Array([ 7., 16.], dtype=float32)
        """
        if is_known_type(oc):
            oc = extract_raw_value(oc)
            self._check_matmul(oc, left=False)
            if self.ndim == 1:
                return binary_densemv(oc, self.value, transpose=False, weighted=True)
            return binary_densemm(oc, self.value, transpose=False, weighted=True)
        return oc.__matmul__(self)

    def tree_flatten(self):
        """Flatten this instance for JAX PyTree serialisation.

        Returns
        -------
        children : tuple
            ``(value,)``.
        aux_data : dict
            Empty dictionary (no static metadata).
        """
        return (self._value,), dict()

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """Reconstruct a ``WeightedArray`` from its PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata produced by ``tree_flatten``.
        flat_contents : tuple
            Dynamic leaves, the underlying array.

        Returns
        -------
        WeightedArray
            A new instance wrapping the given array.
        """
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        return obj
