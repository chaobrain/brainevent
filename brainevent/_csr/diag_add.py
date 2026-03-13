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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters.partial_eval import DynamicJaxprTracer

from brainevent._misc import MatrixShape

__all__ = [
    'csr_diag_position',
    'csr_diag_add',
]


def _is_tracer(x):
    return isinstance(x, (jax.ShapeDtypeStruct, jax.core.ShapedArray, DynamicJaxprTracer, jax.core.Tracer))


def csr_diag_position(indptr, indices, *, shape: MatrixShape):
    """Find the positions of diagonal elements in a CSR sparse matrix (v2).

    Searches through each row of the CSR matrix to locate elements that lie on
    the main diagonal (i.e., where the column index equals the row index). Returns
    both the CSR data-array positions and the corresponding diagonal indices,
    allowing selective updates to diagonal elements.

    Unlike :func:`csr_diag_position`, this version returns a pair of arrays and
    gracefully handles matrices where not every diagonal element is present in
    the sparsity pattern.

    Parameters
    ----------
    indptr : ndarray or jax.Array
        Row pointer array of the CSR format, with shape ``(n_rows + 1,)`` and
        integer dtype. Must not be a JAX tracer (i.e., cannot be used inside
        ``jax.jit``).
    indices : ndarray or jax.Array
        Column indices array of the CSR format, with shape ``(nse,)`` and integer
        dtype. Must not be a JAX tracer.
    shape : tuple of int
        Shape of the sparse matrix as ``(n_rows, n_cols)``. Both dimensions must
        be positive integers.

    Returns
    -------
    csr_positions : ndarray
        1-D ``int32`` array of positions in the CSR data array where diagonal
        elements were found.
    diag_positions : ndarray or None
        1-D ``int32`` array of the corresponding diagonal indices. ``None`` if
        the lengths of ``csr_positions`` and ``diag_positions`` differ (should
        not normally occur).

    Raises
    ------
    AssertionError
        If ``shape`` is not a length-2 tuple/list of positive integers, or if
        ``indptr`` is not 1-D, or if ``indices`` is not 1-D.
    ValueError
        If ``indptr`` or ``indices`` is a JAX tracer, since this function
        requires concrete values to iterate over the sparsity structure.

    See Also
    --------
    csr_diag_position : Simpler version that returns a single array with ``-1``
        for missing diagonals.
    csr_diag_add_v2 : Add values to diagonal elements using positions from this
        function.

    Notes
    -----
    This function uses Numba JIT compilation internally for efficient iteration
    over the sparse structure. The result is cached by Numba.

    The number of diagonal elements searched is ``min(n_rows, n_cols)``.  For
    each row ``i`` in ``range(min(n_rows, n_cols))``, the algorithm scans the
    column indices ``indices[indptr[i] : indptr[i+1]]`` looking for an entry
    where ``indices[j] == i``.  If found, ``j`` is recorded as the CSR
    position and ``i`` as the diagonal position.

    Formally, the returned arrays satisfy:

    ``A[diag_positions[k], diag_positions[k]] == data[csr_positions[k]]``

    for all ``k`` in ``range(len(csr_positions))``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._csr.diag_add import csr_diag_position
        >>> indptr = np.array([0, 2, 4, 6], dtype=np.int32)
        >>> indices = np.array([0, 1, 1, 2, 0, 2], dtype=np.int32)
        >>> csr_pos, diag_pos = csr_diag_position(indptr, indices, shape=(3, 3))
    """
    assert isinstance(shape, (tuple, list)), "shape must be a tuple or list"
    assert indptr.ndim == 1, "indptr must be a 1D array"
    assert indices.ndim == 1, "indices must be a 1D array"
    assert len(shape) == 2, "shape must be a tuple or list of length 2"
    assert all(isinstance(s, int) and s > 0 for s in shape), "shape must be a tuple or list of non-negative integers"
    if _is_tracer(indptr):
        raise ValueError('Cannot trace indptr when finding diagonal position')
    if _is_tracer(indices):
        raise ValueError('Cannot trace indices when finding diagonal position')
    n_size = min(shape)

    import numba

    @numba.njit(cache=True)
    def _find_diag_position(indptr_, indices_, n):
        csr_positions = []
        diag_positions = []
        for i in range(n):
            start = indptr_[i]
            end = indptr_[i + 1]
            for j in range(start, end):
                if indices_[j] == i:
                    csr_positions.append(j)
                    diag_positions.append(i)
                    break
        return np.asarray(csr_positions, dtype=np.int32), np.asarray(diag_positions, dtype=np.int32)

    csr_pos, diag_pos = _find_diag_position(np.asarray(indptr), np.asarray(indices), n_size)

    return (csr_pos, diag_pos) if len(csr_pos) == len(diag_pos) else (csr_pos, None)


def csr_diag_add(csr_value, positions, diag_value):
    """Add values to the diagonal elements of a CSR sparse matrix (v2).

    Uses the position mapping returned by :func:`csr_diag_position_v2` to
    efficiently add ``diag_value`` to the diagonal entries of ``csr_value``. This
    is equivalent to the operation ``A = A + diag(diag_value)`` restricted to
    the existing sparsity pattern.

    Parameters
    ----------
    csr_value : jax.Array or Quantity
        Values of the non-zero elements in the CSR matrix, with shape ``(nse,)``.
        May carry physical units via ``brainunit.Quantity``.
    positions : tuple
        A 2-tuple ``(csr_positions, diag_positions)`` as returned by
        :func:`csr_diag_position_v2`. ``csr_positions`` is a 1-D integer array of
        indices into ``csr_value``. ``diag_positions`` is either a 1-D integer
        array of the same length or ``None``.
    diag_value : jax.Array or Quantity
        Values to add to the diagonal, with shape ``(n_diag,)``. Must have the
        same units and dtype as ``csr_value``.

    Returns
    -------
    jax.Array or Quantity
        Updated CSR value array with the same shape and units as ``csr_value``.

    Raises
    ------
    AssertionError
        If ``csr_value`` or ``diag_value`` is not 1-D, if their dtypes differ,
        if the position arrays are not 1-D integer arrays, or if the units of
        ``csr_value`` and ``diag_value`` are incompatible.

    See Also
    --------
    csr_diag_position_v2 : Compute the position mapping required by this
        function.
    csr_diag_add : Alternative implementation using :func:`csr_diag_position`.

    Notes
    -----
    This function performs the equivalent of the dense operation:

    ``A <- A + diag(d)``

    restricted to the existing sparsity pattern, where ``A`` is the sparse
    matrix represented by ``csr_value`` and ``d`` is ``diag_value``.

    When ``diag_positions`` is ``None``, ``diag_value`` is indexed directly by
    ``csr_positions`` (i.e., a one-to-one mapping is assumed):

    ``csr_value[csr_positions[k]] += diag_value[k]``

    When ``diag_positions`` is provided, the update is:

    ``csr_value[csr_positions[k]] += diag_value[diag_positions[k]]``

    for all ``k`` in ``range(len(csr_positions))``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from brainevent._csr.diag_add import csr_diag_position, csr_diag_add
        >>> indptr = np.array([0, 2, 4, 6], dtype=np.int32)
        >>> indices = np.array([0, 1, 1, 2, 0, 2], dtype=np.int32)
        >>> positions = csr_diag_position(indptr, indices, shape=(3, 3))
        >>> csr_value = jnp.ones(6, dtype=jnp.float32)
        >>> diag_value = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        >>> updated = csr_diag_add(csr_value, positions, diag_value)
    """
    assert u.fail_for_dimension_mismatch(csr_value, diag_value)
    assert csr_value.ndim == 1, "csr_value must be a 1D array"
    assert diag_value.ndim == 1, "diag_value must be a 1D array"
    assert csr_value.dtype == diag_value.dtype, "csr_value and diag_value must have the same dtype"
    csr_pos, diag_pos = positions
    assert csr_pos.ndim == 1, "csr_pos must be a 1D array"
    assert jnp.issubdtype(csr_pos.dtype, jnp.integer), "diag_position must be an integer array"
    if diag_pos is not None:
        assert diag_pos.ndim == 1, "diag_pos must be a 1D array"
        assert jnp.issubdtype(diag_pos.dtype, jnp.integer), "diag_position must be an integer array"

    diag_value = u.Quantity(diag_value).to(u.get_unit(csr_value)).mantissa
    csr_value, csr_unit = u.split_mantissa_unit(csr_value)
    if diag_pos is None:
        csr_value = csr_value.at[csr_pos].add(diag_value)
    else:
        csr_value = csr_value.at[csr_pos].add(diag_value[diag_pos])
    return u.maybe_decimal(csr_value * csr_unit)
