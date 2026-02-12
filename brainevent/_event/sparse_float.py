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


from jax.tree_util import register_pytree_node_class

from brainevent._dense import spfloat_densemv, spfloat_densemm
from brainevent._error import MathError
from .base import EventRepresentation
from .base import extract_raw_value, is_known_type

__all__ = [
    'SparseFloat',
]


@register_pytree_node_class
class SparseFloat(EventRepresentation):
    """Array wrapper for sparse floating-point event vectors and matrices.

    ``SparseFloat`` extends ``EventRepresentation`` to represent arrays where
    most elements are zero and only a few carry floating-point values.  The
    ``@`` operator skips zero entries during multiplication, which is
    mathematically equivalent to dense matrix multiplication but exploits
    the sparsity pattern for efficiency.

    Unlike ``BinaryArray``, where non-zero entries are implicitly 1, the
    non-zero entries in a ``SparseFloat`` carry arbitrary floating-point
    magnitudes that are multiplied with the corresponding weight elements.

    Parameters
    ----------
    value : array_like
        The underlying sparse float array data.  Zero entries are treated as
        inactive (skipped during computation).

    Notes
    -----
    Given a sparse float vector ``x`` of shape ``(k,)`` and a dense weight
    matrix ``W`` of shape ``(k, n)``, the forward multiplication ``x @ W``
    computes:

        y[j] = sum_{i : x[i] != 0} x[i] * W[i, j]

    This is identical to the dense ``x @ W`` result but the implementation
    iterates only over non-zero entries of ``x``.

    The class is registered as a JAX PyTree node, so instances are compatible
    with ``jax.jit``, ``jax.grad``, ``jax.vmap``, and other transformations.

    See Also
    --------
    BinaryArray : Similar wrapper for binary (0/1) event arrays.
    spfloat_densemv : Unified primitive for sparse-float vector-matrix multiplication.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainevent as be
        >>> x = be.SparseFloat(jnp.array([0.0, 2.5, 0.0, 1.0]))
        >>> W = jnp.ones((4, 3))
        >>> x @ W  # only indices 1 and 3 contribute
        Array([3.5, 3.5, 3.5], dtype=float32)
    """
    __module__ = 'brainevent'

    def __init__(self, value):
        super().__init__(value)

    def __matmul__(self, oc):
        """Compute ``self @ oc`` using sparse-float event-driven multiplication.

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

            y[j] = sum_{i : self[i] != 0} self[i] * oc[i, j]

        For 2-D ``self``, the computation is:

            Y[r, j] = sum_{i : self[r, i] != 0} self[r, i] * oc[i, j]

        See Also
        --------
        spfloat_densemv : Sparse-float vector @ dense matrix primitive.
        spfloat_densemm : Unified sparse-float matrix multiplication primitive.
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

            assert oc.ndim == 2, (
                f"Right operand must be a 2D array in "
                f"matrix multiplication. Got {oc.ndim}D array."
            )
            assert self.shape[-1] == oc.shape[0], (
                f"Incompatible dimensions for matrix multiplication: "
                f"{self.shape[-1]} and {oc.shape[0]}."
            )

            # Perform the appropriate multiplication based on dimensions
            if self.ndim == 1:
                return spfloat_densemv(oc, self.value, transpose=True)
            else:  # self.ndim == 2
                return spfloat_densemm(oc, self.value, transpose=True)
        else:
            return oc.__rmatmul__(self)

    def __rmatmul__(self, oc):
        """Compute ``oc @ self`` using sparse-float event-driven multiplication.

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
        For 1-D ``self`` of shape ``(k,)``:

            y[i] = sum_{j : self[j] != 0} oc[i, j] * self[j]

        See Also
        --------
        spfloat_densemv : Dense matrix @ sparse-float vector primitive.
        spfloat_densemm : Unified sparse-float matrix multiplication primitive.
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

            assert oc.ndim == 2, (
                f"Left operand must be a 2D array in "
                f"matrix multiplication. Got {oc.ndim}D array."
            )
            assert oc.shape[-1] == self.shape[0], (
                f"Incompatible dimensions for matrix "
                f"multiplication: {oc.shape[-1]} and {self.shape[0]}."
            )

            # Perform the appropriate multiplication based on dimensions
            if self.ndim == 1:
                return spfloat_densemv(oc, self.value, transpose=False)
            else:
                return spfloat_densemm(oc, self.value, transpose=False)
        else:
            return oc.__matmul__(self)

    def __imatmul__(self, oc):
        """Compute in-place ``self @= oc``.

        Parameters
        ----------
        oc : array_like
            Right operand for the matrix multiplication.

        Returns
        -------
        SparseFloat
            A new ``SparseFloat`` wrapping the multiplication result.

        Notes
        -----
        JAX arrays are immutable, so this does not perform a true in-place
        update.  Instead it returns a new ``SparseFloat`` whose value is the
        result of ``self @ oc``.
        """
        if is_known_type(oc):
            return self.with_value(self.__matmul__(oc))
        return self.with_value(oc.__rmatmul__(self))

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
        """Reconstruct a ``SparseFloat`` from its PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata produced by ``tree_flatten``.
        flat_contents : tuple
            Dynamic leaves — the underlying array.

        Returns
        -------
        SparseFloat
            A new instance wrapping the given array.
        """
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj
