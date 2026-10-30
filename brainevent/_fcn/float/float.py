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

from typing import Union, Tuple

import brainunit as u
import jax
import jax.numpy as jnp

from brainevent._misc import _promote_float_operands, check_fixed_conn_num_shape, namescope

__all__ = [
    'fcnmv',
    'fcnmm',
]


@namescope(static_argnames=['shape', 'transpose'])
def fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse matrix--vector product with fixed connection number.

    Computes ``y = W @ v`` (or ``y = W^T @ v`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``v`` is a dense floating-point vector.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    vector : jax.Array or u.Quantity
        Dense vector to multiply with.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool
        If ``False``, compute ``W @ v`` (fixed post-synaptic connections,
        gather mode).  If ``True``, compute ``W^T @ v`` (fixed
        pre-synaptic connections, scatter mode).

    Returns
    -------
    jax.Array or u.Quantity
        Result vector.  Shape is ``(num_pre,)`` when ``transpose=False``
        or ``(num_post,)`` when ``transpose=True``.

    See Also
    --------
    fcnmm : Float sparse matrix--matrix product with fixed connection number.
    binary_fcnmv : Event-driven (binary) variant.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    When ``transpose=False`` (gather mode), the matrix-vector product computes:

        ``y[i] = sum_{k=0}^{n_conn-1} weights[i, k] * v[indices[i, k]]``

    For homogeneous weights (``weights`` has shape ``(1,)``):

        ``y[i] = w * sum_{k=0}^{n_conn-1} v[indices[i, k]]``

    When ``transpose=True`` (scatter mode), the computation distributes each
    row's contributions to the target columns:

        ``y[indices[i, k]] += weights[i, k] * v[i]``    for all ``i, k``

    The computational cost is ``O(num_pre * n_conn)`` regardless of the number
    of post-synaptic neurons, making this efficient for sparse connectivity.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.float import fcnmv
        >>>
        >>> weights = jnp.array([[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float32)
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> vector = jnp.array([1.0, 2.0, 3.0])
        >>> y = fcnmv(weights, indices, vector, shape=(2, 3), transpose=False)
        >>> print(y)
        [2.5 9. ]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(
        weights, indices, vector, shape, transpose, require_scalar_weight=True,
    )
    vector, v_unit = u.split_mantissa_unit(vector)
    weights, w_unit = u.split_mantissa_unit(weights)
    weights, vector = _promote_float_operands(weights, vector)
    if transpose:
        if weights.ndim == 0:
            r = weights * jnp.zeros(n_post, vector.dtype).at[indices].add(
                jnp.broadcast_to(vector[:, None], indices.shape)
            )
        else:
            r = jnp.zeros(n_post, weights.dtype).at[indices].add(vector[:, None] * weights)
    else:
        if weights.ndim == 0:
            r = jax.vmap(lambda ind: weights * jnp.sum(vector[ind]))(indices)
        else:
            r = jax.vmap(lambda w, ind: jnp.sum(w * vector[ind]))(weights, indices)
    return u.maybe_decimal(r * v_unit * w_unit)


#: Upper bound on the elements of the ``(n_pre, n_conn, chunk)`` update
#: temporary of the scatter (``transpose=True``) path of :func:`fcnmm`, which
#: is processed in chunks of output columns. ``vmap`` multiplies it by the batch
#: size. 2**24 keeps single-call speed (measured on an RTX 3090) while bounding
#: the batched temporary.
_FCNMM_CHUNK_ELEMS = 1 << 24


def _fcnmm_scatter(weights, indices, matrix, n_post):
    """Compute ``Y[indices[i, k], :] += w[i, k] * M[i, :]`` in chunks of columns.

    A scatter-add needs its updates materialised, an ``(n_pre, n_conn, n)``
    temporary. Columns of ``M`` are processed in chunks so that it stays below
    ``_FCNMM_CHUNK_ELEMS`` elements (one step when it already fits; at least one
    column per step).
    """
    n_pre, n_conn = indices.shape
    n = matrix.shape[1]
    homo = weights.ndim == 0
    dtype = matrix.dtype if homo else jnp.result_type(weights, matrix)
    matrix = matrix.astype(dtype)

    def scatter(cols):
        upd = cols[:, None, :] if homo else weights.astype(dtype)[:, :, None] * cols[:, None, :]
        if homo:
            upd = jnp.broadcast_to(upd, (n_pre, n_conn, cols.shape[1]))
        return jnp.zeros((n_post, cols.shape[1]), dtype).at[indices].add(upd)

    cn = max(1, min(n, _FCNMM_CHUNK_ELEMS // max(1, n_pre * n_conn)))
    n_chunks = -(-n // cn)
    if n_chunks == 1:
        out = scatter(matrix)
    else:
        mp = jnp.pad(matrix, ((0, 0), (0, n_chunks * cn - n)))

        def step(c, acc):
            cols = jax.lax.dynamic_slice_in_dim(mp, c * cn, cn, axis=1)
            return jax.lax.dynamic_update_slice_in_dim(acc, scatter(cols), c * cn, axis=1)

        out = jax.lax.fori_loop(0, n_chunks, step, jnp.zeros((n_post, n_chunks * cn), dtype))[:, :n]
    return weights * out if homo else out


@namescope(static_argnames=['shape', 'transpose'])
def fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse matrix--matrix product with fixed connection number.

    Computes ``Y = W @ M`` (or ``Y = W^T @ M`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``M`` is a dense floating-point matrix.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    matrix : jax.Array or u.Quantity
        Dense matrix to multiply with, of shape ``(k, n)`` where ``k``
        matches the appropriate sparse-matrix dimension.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool
        If ``False``, compute ``W @ M`` (fixed post-synaptic connections,
        gather mode).  If ``True``, compute ``W^T @ M`` (fixed
        pre-synaptic connections, scatter mode).

    Returns
    -------
    jax.Array or u.Quantity
        Result matrix of shape ``(num_pre, n)`` when ``transpose=False``
        or ``(num_post, n)`` when ``transpose=True``.

    See Also
    --------
    fcnmv : Float sparse matrix--vector product with fixed connection number.
    binary_fcnmm : Event-driven (binary) variant.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    When ``transpose=False`` (gather mode), each output element is:

        ``Y[i, j] = sum_{k=0}^{n_conn-1} weights[i, k] * M[indices[i, k], j]``

    For homogeneous weights (``weights`` has shape ``(1,)``):

        ``Y[i, j] = w * sum_{k=0}^{n_conn-1} M[indices[i, k], j]``

    When ``transpose=True`` (scatter mode), the computation distributes
    contributions to target rows:

        ``Y[indices[i, k], j] += weights[i, k] * M[i, j]``    for all ``i, k, j``

    The computational cost is ``O(num_pre * n_conn * n)`` where ``n`` is the
    number of columns in ``M``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.float import fcnmm
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)  # homogeneous
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> matrix = jnp.array([[1.0, 0.5],
        ...                     [2.0, 1.0],
        ...                     [3.0, 1.5]])
        >>> y = fcnmm(weights, indices, matrix, shape=(2, 3), transpose=False)
        >>> print(y)
        [[3.  1.5]
         [5.  2.5]]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(
        weights, indices, matrix, shape, transpose, require_scalar_weight=True,
    )
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    weights, matrix = _promote_float_operands(weights, matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    if transpose:
        # Scatter mode: Y[n_post, n], Y[indices[i, k], :] += weights[i, k] * matrix[i, :]
        r = _fcnmm_scatter(weights, indices, matrix, n_post)
    else:
        # Gather mode: Y[n_pre, n], Y[i, :] = sum_k weights[i, k] * matrix[indices[i, k], :]
        # (XLA fuses the gather into the reduction; no (n_pre, n_conn, n) buffer.)
        if weights.ndim == 0:
            r = jax.vmap(lambda ind: weights * jnp.sum(matrix[ind], axis=0))(indices)
        else:
            r = jax.vmap(lambda w, ind: jnp.sum(w[:, None] * matrix[ind], axis=0))(weights, indices)
    return u.maybe_decimal(r * m_unit * w_unit)
