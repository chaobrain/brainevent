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

from jax.tree_util import register_pytree_node_class

from brainevent._dense import binary_densemm, binary_densemv
from brainevent._error import MathError
from .base import EventRepresentation, extract_raw_value, is_known_type

__all__ = [
    'BinaryArray',
]


@register_pytree_node_class
class BinaryArray(EventRepresentation):
    """Array wrapper for binary (0/1) event vectors and matrices.

    ``BinaryArray`` represents a boolean or 0/1 array and provides
    event-driven matrix multiplication via the ``@`` operator.  When a
    ``BinaryArray`` is multiplied with a dense weight matrix, only the
    rows/columns corresponding to non-zero (active) elements are
    accumulated, which is mathematically equivalent to standard matrix
    multiplication but can exploit sparsity for efficiency.

    Parameters
    ----------
    value : array_like
        The input binary array data.  Typically a boolean JAX array, but
        any array whose non-zero pattern encodes binary events is accepted.

    Notes
    -----
    Given a binary spike vector ``s`` of shape ``(k,)`` and a weight matrix
    ``W`` of shape ``(k, n)``, the forward multiplication ``s @ W`` computes:

        y[j] = sum_{i : s[i] != 0} W[i, j]

    This is equivalent to ``s.astype(float) @ W`` but the implementation
    only iterates over the non-zero entries of ``s``.

    The class is registered as a JAX PyTree node, so it is compatible with
    ``jax.jit``, ``jax.grad``, ``jax.vmap``, and other transformations.

    See Also
    --------
    SparseFloat : Similar wrapper for sparse floating-point event arrays.
    binary_densemv : Underlying primitive for binary vector-matrix multiply.
    binary_densemm : Underlying primitive for binary matrix-matrix multiply.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainevent as be
        >>> spikes = be.BinaryArray(jnp.array([True, False, True]))
        >>> W = jnp.ones((3, 4))
        >>> spikes @ W  # sums rows 0 and 2 of W
        Array([2., 2., 2., 2.], dtype=float32)
    """
    __module__ = 'brainevent'

    def __init__(self, value):
        super().__init__(value)

    @property
    def T(self):
        """Transpose of the underlying array.

        Returns
        -------
        jax.Array
            The transposed raw array (not wrapped in ``BinaryArray``).
        """
        return self.value.T

    def transpose(self, *axes):
        """Return the underlying array with axes permuted.

        Parameters
        ----------
        *axes : int, optional
            Axis permutation.  If omitted, reverses the axis order.

        Returns
        -------
        jax.Array
            The transposed raw array (not wrapped in ``BinaryArray``).
        """
        return self.value.transpose(*axes)

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
        AssertionError
            If ``oc`` is not 2-D or if inner dimensions do not match.

        Notes
        -----
        For 1-D ``self``, the computation is:

            y[j] = sum_{i : self[i] != 0} oc[i, j]

        For 2-D ``self`` of shape ``(m, k)``, the computation is:

            Y[r, j] = sum_{i : self[r, i] != 0} oc[i, j]

        The 2-D case is implemented as
        ``binary_densemm(oc, self.T, transpose=True).T``.

        See Also
        --------
        binary_densemv : Underlying vector-matrix multiply primitive.
        binary_densemm : Underlying matrix-matrix multiply primitive.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent as be
            >>> s = be.BinaryArray(jnp.array([True, False, True]))
            >>> W = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
            >>> s @ W
            Array([6., 8.], dtype=float32)
        """
        if is_known_type(oc):
            oc = extract_raw_value(oc)
            # Check dimensions for both operands
            if self.ndim not in (1, 2):
                raise MathError(
                    f"Matrix multiplication is only supported "
                    f"for 1D and 2D arrays. Got {self.ndim}D array."
                )

            if self.ndim == 0:
                raise MathError("Matrix multiplication is not supported for scalar arrays.")

            assert oc.ndim == 2, (f"Right operand must be a 2D array in "
                                  f"matrix multiplication. Got {oc.ndim}D array.")
            assert self.shape[-1] == oc.shape[0], (f"Incompatible dimensions for matrix multiplication: "
                                                   f"{self.shape[-1]} and {oc.shape[0]}.")

            # Perform the appropriate multiplication based on dimensions
            if self.ndim == 1:
                return binary_densemv(oc, self.value, transpose=True)
            else:  # self.ndim == 2
                # self[m,k] @ oc[k,n]: use weights=oc[k,n], spikes=self.value.T[k,m]
                # gives oc.T @ self.value.T = [n,m], then .T = [m,n]
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
        AssertionError
            If ``oc`` is not 2-D or if inner dimensions do not match.

        Notes
        -----
        For 1-D ``self`` of shape ``(k,)``, the computation is:

            y[i] = sum_{j : self[j] != 0} oc[i, j]

        This is equivalent to ``oc @ self.astype(float)`` but only accumulates
        columns of ``oc`` where ``self`` is active.

        See Also
        --------
        binary_densemv : Underlying vector-matrix multiply primitive.
        binary_densemm : Underlying matrix-matrix multiply primitive.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent as be
            >>> W = jnp.array([[1., 2., 3.], [4., 5., 6.]])
            >>> s = be.BinaryArray(jnp.array([True, False, True]))
            >>> W @ s
            Array([ 4., 10.], dtype=float32)
        """
        if is_known_type(oc):
            oc = extract_raw_value(oc)
            # Check dimensions for both operands
            if self.ndim not in (1, 2):
                raise MathError(f"Matrix multiplication is only supported "
                                f"for 1D and 2D arrays. Got {self.ndim}D array.")

            if self.ndim == 0:
                raise MathError("Matrix multiplication is not supported for scalar arrays.")

            assert oc.ndim == 2, (f"Left operand must be a 2D array in "
                                  f"matrix multiplication. Got {oc.ndim}D array.")
            assert oc.shape[-1] == self.shape[0], (f"Incompatible dimensions for matrix "
                                                   f"multiplication: {oc.shape[-1]} and {self.shape[0]}.")

            # Perform the appropriate multiplication based on dimensions
            if self.ndim == 1:
                return binary_densemv(oc, self.value, transpose=False)
            else:
                return binary_densemm(oc, self.value, transpose=False)
        else:
            return oc.__matmul__(self)

    def tree_flatten(self):
        """Flatten this instance for JAX PyTree serialisation.

        Returns
        -------
        children : tuple
            A single-element tuple ``(value,)`` containing the dynamic
            array leaf.
        aux_data : dict
            Empty dictionary (no static metadata).
        """
        aux = dict()
        return (self._value,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """Reconstruct a ``BinaryArray`` from its PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata produced by ``tree_flatten``.
        flat_contents : tuple
            Dynamic leaves — the underlying array.

        Returns
        -------
        BinaryArray
            A new instance wrapping the given array.
        """
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj
