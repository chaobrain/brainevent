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
from jax.experimental.sparse.linalg import spsolve as raw_spsolve

__all__ = [
    'csr_solve'
]


def csr_solve(data, indices, indptr, b, tol=1e-6, reorder=1):
    """
    A sparse direct solver using QR factorization.

    Compute

    $$
    M x = b
    $$

    where $M$ is the CSR sparse matrix, and b is the vector.

    Accepts a sparse matrix in CSR format `data, indices, indptr` arrays.
    Currently only the CUDA GPU backend is implemented, the CPU backend will fall
    back to `scipy.sparse.linalg.spsolve`. Neither the CPU nor the GPU
    implementation support batching with `vmap`.

    Args:
      data : An array containing the non-zero entries of the CSR matrix.
      indices : The column indices of the CSR matrix.
      indptr : The row pointer array of the CSR matrix.
      b : The right hand side of the linear system.
      tol : Tolerance to decide if singular or not. Defaults to 1e-6.
      reorder : The reordering scheme to use to reduce fill-in. No reordering if
        ``reorder=0``. Otherwise, symrcm, symamd, or csrmetisnd (``reorder=1,2,3``),
        respectively. Defaults to symrcm.

    Returns:
      An array with the same dtype and size as b representing the solution to
      the sparse linear system.
    """
    data, data_unit = u.split_mantissa_unit(data)
    b, b_unit = u.split_mantissa_unit(b)
    res = raw_spsolve(data, indices, indptr, b, tol=tol, reorder=reorder)
    return u.maybe_decimal(res * b_unit / data_unit)
