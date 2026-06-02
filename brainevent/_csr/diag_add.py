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
    """Plan the structural change required to add a full diagonal to a CSR matrix.

    Computing ``A + diag(d)`` for a sparse ``A`` is *not* a pure value update:
    whenever a diagonal entry ``A[i, i]`` is absent from the sparsity pattern it
    must be **inserted**, which changes ``indices`` and ``indptr``.  This function
    pre-computes, from the concrete sparsity structure alone, everything needed to
    materialise the augmented matrix whose pattern is the union of ``A``'s pattern
    and the full main diagonal ``{(i, i) : 0 <= i < min(shape)}``.

    The returned *plan* is value-independent, so it can be cached once per
    structure and reused for every subsequent diagonal update inside ``jax.jit``
    (see :func:`csr_diag_add`).

    Parameters
    ----------
    indptr : ndarray or jax.Array
        Row pointer array of the CSR format, with shape ``(n_rows + 1,)`` and
        integer dtype. Must not be a JAX tracer (i.e., cannot be used inside
        ``jax.jit``). For a CSC matrix pass the column pointer array; the diagonal
        of a matrix and its transpose coincide, so the same routine applies.
    indices : ndarray or jax.Array
        Column indices array of the CSR format, with shape ``(nse,)`` and integer
        dtype. Must not be a JAX tracer. (Row indices for a CSC matrix.)
    shape : tuple of int
        Shape of the sparse matrix as ``(n_rows, n_cols)``. Both dimensions must
        be positive integers.

    Returns
    -------
    new_indptr : ndarray
        1-D ``int32`` row-pointer array of the diagonal-augmented matrix, with
        shape ``(n_rows + 1,)``.
    new_indices : ndarray
        1-D ``int32`` column-index array of the augmented matrix, with the
        inserted diagonal entries placed in sorted (ascending) position within
        each row.
    old_to_new : ndarray
        1-D ``int32`` array of shape ``(nse,)`` mapping every position in the
        original ``data`` array to its position in the augmented ``data`` array.
    diag_dest : ndarray
        1-D ``int32`` array of shape ``(min(shape),)`` whose ``i``-th entry is the
        position, in the augmented ``data`` array, of the diagonal element
        ``(i, i)``. Both pre-existing and freshly inserted diagonals are covered.

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
    csr_diag_add : Apply the plan returned here to obtain the augmented values.

    Notes
    -----
    This function uses Numba JIT compilation internally for efficient iteration
    over the sparse structure; the compiled kernel is cached by Numba.

    For every row ``i`` with ``i < min(shape)`` the kernel checks whether the
    column index ``i`` already appears in ``indices[indptr[i] : indptr[i + 1]]``.
    If it does, the row length is unchanged and ``diag_dest[i]`` records the new
    position of that entry. If it does not, a single entry with column ``i`` is
    inserted in sorted position, the row grows by one, and ``diag_dest[i]``
    records the inserted position. Rows with ``i >= min(shape)`` are copied
    verbatim.

    The plan satisfies, for the augmented ``data`` array ``nd``:

    ``nd[old_to_new[p]] == data[p]``  for all original positions ``p``, and the
    diagonal element ``(i, i)`` lives at ``nd[diag_dest[i]]``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._csr.diag_add import csr_diag_position
        >>> # Row 1 is missing its diagonal (1, 1):
        >>> indptr = np.array([0, 1, 2, 4], dtype=np.int32)
        >>> indices = np.array([0, 2, 0, 2], dtype=np.int32)
        >>> new_indptr, new_indices, old_to_new, diag_dest = csr_diag_position(
        ...     indptr, indices, shape=(3, 3))
        >>> new_indptr
        array([0, 1, 3, 5], dtype=int32)
        >>> new_indices
        array([0, 1, 2, 0, 2], dtype=int32)
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
    n_diag = min(shape)

    import numba

    @numba.njit(cache=True)
    def _build_diag_augmented_structure(indptr_, indices_, n_diag_):
        n_outer = indptr_.shape[0] - 1
        old_nse = indices_.shape[0]

        # First pass: compute the length of every augmented row so we can size
        # the output arrays exactly (JAX needs a static, value-independent shape).
        new_indptr_ = np.zeros(n_outer + 1, dtype=np.int32)
        for i in range(n_outer):
            start = indptr_[i]
            end = indptr_[i + 1]
            extra = 0
            if i < n_diag_:
                has_diag = False
                for j in range(start, end):
                    if indices_[j] == i:
                        has_diag = True
                        break
                if not has_diag:
                    extra = 1
            new_indptr_[i + 1] = new_indptr_[i] + (end - start) + extra

        new_nse = new_indptr_[n_outer]
        new_indices_ = np.empty(new_nse, dtype=np.int32)
        old_to_new_ = np.empty(old_nse, dtype=np.int32)
        diag_dest_ = np.full(n_diag_, -1, dtype=np.int32)

        # Second pass: emit each row, inserting the diagonal in sorted position
        # where it is missing, and record both mappings.
        for i in range(n_outer):
            start = indptr_[i]
            end = indptr_[i + 1]
            dst = new_indptr_[i]
            if i >= n_diag_:
                for j in range(start, end):
                    new_indices_[dst] = indices_[j]
                    old_to_new_[j] = dst
                    dst += 1
                continue

            has_diag = False
            for j in range(start, end):
                if indices_[j] == i:
                    has_diag = True
                    break

            if has_diag:
                for j in range(start, end):
                    col = indices_[j]
                    new_indices_[dst] = col
                    old_to_new_[j] = dst
                    if col == i:
                        diag_dest_[i] = dst
                    dst += 1
            else:
                inserted = False
                for j in range(start, end):
                    col = indices_[j]
                    if (not inserted) and (col > i):
                        new_indices_[dst] = i
                        diag_dest_[i] = dst
                        dst += 1
                        inserted = True
                    new_indices_[dst] = col
                    old_to_new_[j] = dst
                    dst += 1
                if not inserted:
                    new_indices_[dst] = i
                    diag_dest_[i] = dst
                    dst += 1

        return new_indptr_, new_indices_, old_to_new_, diag_dest_

    return _build_diag_augmented_structure(np.asarray(indptr), np.asarray(indices), n_diag)


def csr_diag_add(csr_value, positions, diag_value):
    """Add ``diag(diag_value)`` to a CSR matrix, inserting missing diagonals.

    Applies the structural *plan* produced by :func:`csr_diag_position` to compute
    the value array of ``A + diag(diag_value)``.  Existing entries are relocated
    to their augmented positions, every diagonal value is added on top of the
    (possibly newly inserted) diagonal entry, and the result has the augmented
    sparsity pattern.  This is mathematically exact: unlike a plain value update,
    diagonal entries absent from ``A`` are *not* dropped but materialised.

    Parameters
    ----------
    csr_value : jax.Array or Quantity
        Values of the non-zero elements in the original CSR matrix, with shape
        ``(nse,)``. May carry physical units via ``brainunit.Quantity``.
    positions : tuple
        The 4-tuple ``(new_indptr, new_indices, old_to_new, diag_dest)`` returned
        by :func:`csr_diag_position`.
    diag_value : jax.Array or Quantity
        Values to add to the diagonal, with shape ``(min(shape),)``. Must carry
        units compatible with ``csr_value`` and share its dtype.

    Returns
    -------
    jax.Array or Quantity
        Value array of the diagonal-augmented matrix, with shape
        ``(new_indices.size,)`` and the same units as ``csr_value``. Pair it with
        ``new_indices`` / ``new_indptr`` from the plan to build the new matrix.

    Raises
    ------
    AssertionError
        If ``csr_value`` or ``diag_value`` is not 1-D, if their dtypes differ, if
        the plan index arrays are not 1-D integer arrays, if ``diag_value`` does
        not have one entry per diagonal, if ``csr_value`` does not match the
        original ``nse``, or if the units are incompatible.

    See Also
    --------
    csr_diag_position : Compute the structural plan consumed by this function.

    Notes
    -----
    The computation is

    ``new_data = scatter(csr_value -> old_to_new); new_data[diag_dest] += diag_value``

    where ``new_data`` is zero-initialised. Because each original position maps to
    a distinct augmented position, the scatter-set is unambiguous; the subsequent
    scatter-add lands the diagonal contribution on both relocated existing
    diagonals and freshly inserted ones.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from brainevent._csr.diag_add import csr_diag_position, csr_diag_add
        >>> indptr = np.array([0, 1, 2, 4], dtype=np.int32)
        >>> indices = np.array([0, 2, 0, 2], dtype=np.int32)
        >>> positions = csr_diag_position(indptr, indices, shape=(3, 3))
        >>> csr_value = jnp.ones(4, dtype=jnp.float32)
        >>> diag_value = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        >>> new_data = csr_diag_add(csr_value, positions, diag_value)
        >>> new_data
        Array([1.1, 0.2, 1. , 1. , 1.3], dtype=float32)
    """
    assert u.fail_for_dimension_mismatch(csr_value, diag_value)
    assert csr_value.ndim == 1, "csr_value must be a 1D array"
    assert diag_value.ndim == 1, "diag_value must be a 1D array"
    assert csr_value.dtype == diag_value.dtype, "csr_value and diag_value must have the same dtype"
    _, new_indices, old_to_new, diag_dest = positions
    assert old_to_new.ndim == 1, "old_to_new must be a 1D array"
    assert diag_dest.ndim == 1, "diag_dest must be a 1D array"
    assert jnp.issubdtype(old_to_new.dtype, jnp.integer), "old_to_new must be an integer array"
    assert jnp.issubdtype(diag_dest.dtype, jnp.integer), "diag_dest must be an integer array"
    assert csr_value.shape[0] == old_to_new.shape[0], "csr_value length must match the original number of stored elements"
    assert diag_value.shape[0] == diag_dest.shape[0], "diag_value must have one entry per diagonal (min(shape))"

    diag_value = u.Quantity(diag_value).to(u.get_unit(csr_value)).mantissa
    csr_value, csr_unit = u.split_mantissa_unit(csr_value)
    new_data = jnp.zeros(new_indices.shape[0], dtype=csr_value.dtype)
    new_data = new_data.at[old_to_new].set(csr_value)
    new_data = new_data.at[diag_dest].add(diag_value)
    return u.maybe_decimal(new_data * csr_unit)
