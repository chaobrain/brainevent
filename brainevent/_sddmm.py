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

import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax.experimental.sparse import BCOO

from brainevent._misc import namescope

__all__ = [
    'sddmm_indices',
    'sddmm_coo_indices',
    'sddmm_bcoo',
]


@namescope
def sddmm_indices(
    A: jax.Array,
    B: jax.Array,
    indices: jax.Array,
) -> BCOO:
    """Sampled Dense-Dense Matrix Multiplication using JAX BCOO.

    Computes ``S = (A @ B) * M`` where ``M`` is the sparsity pattern
    (mask), and the result is sampled only at the non-zero positions
    of ``M``.

    Parameters
    ----------
    A : jax.Array
        Dense matrix of shape ``(m, k)``.
    B : jax.Array
        Dense matrix of shape ``(k, n)``.
    indices : jax.Array
        2-D index array of shape ``(nse, 2)`` specifying the
        ``(row, col)`` positions of non-zero entries in the mask.

    Returns
    -------
    BCOO
        BCOO sparse matrix with the sampled results at non-zero positions.

    Notes
    -----
    The SDDMM operation computes the element-wise product of a dense
    matrix product and a sparse mask:

        ``S[i, j] = (A @ B)[i, j]``  for ``(i, j)`` in the sparsity pattern

    This avoids forming the full ``(m, n)`` product, only computing
    entries at the ``nse`` positions specified by ``indices``.
    """
    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[1] == B.shape[0]
    assert indices.ndim == 2
    assert indices.shape[1] == 2
    # bcoo_dot_general_sampled computes the dot product and samples at sparse positions
    # dimension_numbers specify the contraction:
    # ((1,), (0,)) means contract A's dim 1 with B's dim 0 (i.e., A @ B)
    # ((), ()) means no batch dimensions
    result_data = sparse.bcoo_dot_general_sampled(
        A, B, indices, dimension_numbers=(((1,), (0,)), ((), ()))
    )
    return BCOO((result_data, indices), shape=(A.shape[0], B.shape[1]))


@namescope
def sddmm_coo_indices(
    A: jax.Array,
    B: jax.Array,
    pre_idx: jax.Array,
    post_idx: jax.Array,
) -> BCOO:
    """Sampled Dense-Dense Matrix Multiplication with COO-style indices.

    Computes ``S = (A @ B) * M`` where ``M`` is the sparsity pattern
    defined by separate row and column index arrays.

    Parameters
    ----------
    A : jax.Array
        Dense matrix of shape ``(m, k)``.
    B : jax.Array
        Dense matrix of shape ``(k, n)``.
    pre_idx : jax.Array
        1-D array of row indices for the non-zero positions.
    post_idx : jax.Array
        1-D array of column indices for the non-zero positions.
        Must have the same shape as ``pre_idx``.

    Returns
    -------
    BCOO
        BCOO sparse matrix with the sampled results at non-zero positions.

    See Also
    --------
    sddmm_indices : Variant accepting a single ``(nse, 2)`` index array.
    sddmm_bcoo : Variant accepting a BCOO sparsity pattern directly.
    """
    assert pre_idx.ndim == 1
    assert post_idx.ndim == 1
    assert A.shape[1] == B.shape[0]
    assert pre_idx.shape == post_idx.shape
    indices = jnp.stack([pre_idx, post_idx], axis=1)
    return sddmm_indices(A, B, indices)


@namescope
def sddmm_bcoo(
    A: jax.Array,
    B: jax.Array,
    sparsity_pattern: BCOO,
) -> BCOO:
    """Sampled Dense-Dense Matrix Multiplication with a BCOO sparsity pattern.

    Computes ``S = (A @ B) * M`` where ``M`` is a BCOO sparse matrix
    defining the sparsity pattern.

    Parameters
    ----------
    A : jax.Array
        Dense matrix of shape ``(m, k)``.
    B : jax.Array
        Dense matrix of shape ``(k, n)``.
    sparsity_pattern : BCOO
        BCOO sparse matrix of shape ``(m, n)`` defining the positions at
        which to sample the dense product.

    Returns
    -------
    BCOO
        BCOO sparse matrix with the sampled results at non-zero positions.

    See Also
    --------
    sddmm_indices : Variant accepting a raw index array.
    sddmm_coo_indices : Variant accepting separate row/col arrays.
    """
    return sddmm_indices(A, B, sparsity_pattern.indices)
