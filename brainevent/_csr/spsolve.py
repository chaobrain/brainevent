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
from jax.experimental.sparse.linalg import spsolve

from brainevent._misc import _csr_to_coo
from .float import csrmv

__all__ = [
    'csr_solve'
]


def csr_solve(data, indices, indptr, b, tol=1e-6, reorder=1):
    """Sparse direct solver using QR factorization.

    Solve the linear system ``M @ x = b`` where ``M`` is a sparse matrix
    in CSR format.  Currently the CUDA GPU backend is implemented; the
    CPU backend falls back to ``scipy.sparse.linalg.spsolve``.

    Parameters
    ----------
    data : jax.Array or brainunit.Quantity
        Non-zero entries of the CSR matrix.
    indices : jax.Array
        Column indices of the CSR matrix.
    indptr : jax.Array
        Row pointer array of the CSR matrix.
    b : jax.Array or brainunit.Quantity
        Right-hand side vector.
    tol : float, optional
        Tolerance to decide if the matrix is singular.  Default is ``1e-6``.
    reorder : int, optional
        Reordering scheme to reduce fill-in.  ``0`` for no reordering,
        ``1`` for symrcm, ``2`` for symamd, ``3`` for csrmetisnd.
        Default is ``1`` (symrcm).

    Returns
    -------
    jax.Array or brainunit.Quantity
        Solution vector ``x`` with the size of ``b`` and the promoted dtype of
        ``data`` and ``b``.

    Raises
    ------
    ValueError
        If the matrix stores no entries (``nse == 0``) and ``b`` is not
        empty: the system is singular.

    Notes
    -----
    The system solved is:

        ``M @ x = b``

    where ``M`` is the sparse matrix represented by ``(data, indices,
    indptr)`` in CSR format.  Repeated ``(row, col)`` entries add, and the
    column indices need not be sorted within a row: the matrix is made
    canonical (sorted, repeats summed) before the solve, which a canonical
    matrix passes unchanged (a stored ``-0.0`` becomes ``+0.0``).  ``data``
    of shape ``(1,)`` is a weight shared by every stored entry.

    A singular matrix with stored entries (e.g. all-zero ``data``) follows
    jax's ``spsolve``: the CPU backend returns ``nan`` and the GPU backend
    raises ``Singular matrix in linear solve``.

    The solve is differentiable in forward and reverse mode with respect to
    ``data`` and ``b`` (implicit differentiation through
    :func:`jax.lax.custom_linear_solve`).  Neither the CPU nor the GPU
    implementation supports batching with ``vmap``.
    """
    data, data_unit = u.split_mantissa_unit(data)
    b, b_unit = u.split_mantissa_unit(b)
    # ``spsolve`` needs one dtype for the matrix and the right-hand side.
    dtype = jnp.result_type(data, b)
    data, b = jnp.asarray(data, dtype), jnp.asarray(b, dtype)
    if data.size == 1 and indices.size != 1:
        data = jnp.broadcast_to(data.reshape(()), indices.shape)
    n = indptr.shape[0] - 1
    if n == 0:
        return u.maybe_decimal(b * b_unit / data_unit)  # the empty system
    if indices.shape[0] == 0:
        # cuSOLVER fails deep inside csrlsvqr on an empty matrix; SciPy returns nan.
        raise ValueError(f'csr_solve: the {n} x {n} matrix stores no entries (nse = 0), so it is singular.')
    rows, cols = _csr_to_coo(indices, indptr)

    def matvec(x):
        return csrmv(data, indices, indptr, x, shape=(n, n))

    def solve(_, rhs):
        return spsolve(*_canonical_csr(data, rows, cols, n, indptr.dtype), rhs, tol=tol, reorder=reorder)

    def transpose_solve(_, rhs):
        return spsolve(*_canonical_csr(data, cols, rows, n, indptr.dtype), rhs, tol=tol, reorder=reorder)

    res = jax.lax.custom_linear_solve(matvec, b, solve, transpose_solve)
    return u.maybe_decimal(res * b_unit / data_unit)


def _canonical_csr(data, rows, cols, n, indptr_dtype):
    """Canonical CSR arrays of the COO entries ``(rows, cols, data)`` of an ``n x n`` matrix.

    jax's ``spsolve`` assumes column indices sorted within each row and no
    repeated ``(row, col)`` entries: cuSOLVER keeps one of the repeats, and
    the SciPy callback on CPU sorts its read-only buffers in place (#225).
    The entries are sorted by ``(row, col)`` and repeats are summed into one
    slot. The number of distinct entries is data-dependent, so the arrays keep
    the static size ``min(nse, n * n)``; slots past ``indptr[-1]`` are zero
    and outside the matrix, which both backends ignore. A canonical input
    comes back unchanged, except that a stored ``-0.0`` becomes ``+0.0``
    (each slot is ``0 + value``); that does not change the solve.

    Parameters
    ----------
    data : jax.Array
        Values, shape ``(nse,)``.
    rows, cols : jax.Array
        Row and column index of every entry, shape ``(nse,)``.
    n : int
        Matrix size.
    indptr_dtype : dtype
        Dtype of the returned row pointer.

    Returns
    -------
    tuple of jax.Array
        ``(data, indices, indptr)`` of the canonical matrix.
    """
    nse = data.shape[0]
    size = min(nse, n * n)
    rows, cols, data = jax.lax.sort((rows, cols, data), num_keys=2)
    first = jnp.ones((nse,), dtype=bool)
    if nse > 1:
        first = first.at[1:].set((rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1]))
    slot = jnp.cumsum(first) - 1
    # 0 + (-0.0) is +0.0: the only bits of a canonical input that change.
    data_c = jnp.zeros((size,), data.dtype).at[slot].add(data)
    cols_c = jnp.zeros((size,), cols.dtype).at[slot].set(cols)
    counts = jnp.zeros((n,), indptr_dtype).at[rows].add(first.astype(indptr_dtype))
    indptr_c = jnp.concatenate([jnp.zeros((1,), indptr_dtype), jnp.cumsum(counts, dtype=indptr_dtype)])
    return data_c, cols_c, indptr_c
