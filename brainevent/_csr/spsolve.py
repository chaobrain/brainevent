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
from jax.experimental.sparse.linalg import spsolve

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
        Solution vector ``x`` with the same dtype and size as ``b``.

    Notes
    -----
    The system solved is:

        ``M @ x = b``

    where ``M`` is the sparse matrix represented by ``(data, indices,
    indptr)`` in CSR format.  Neither the CPU nor the GPU implementation
    supports batching with ``vmap``.
    """
    data, data_unit = u.split_mantissa_unit(data)
    b, b_unit = u.split_mantissa_unit(b)
    res = spsolve(data, indices, indptr, b, tol=tol, reorder=reorder)
    return u.maybe_decimal(res * b_unit / data_unit)
