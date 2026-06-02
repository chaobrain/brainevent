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

from brainevent._misc import check_fixed_conn_num_shape, namescope

__all__ = [
    'fcnmv',
    'fcnmm',
    'fcnmv_yw2y',
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
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    n = matrix.shape[1]
    if transpose:
        # Scatter mode: Y[n_post, n]
        # Y[indices[i, k], :] += weights[i, k] * matrix[i, :]
        if weights.ndim == 0:
            # homo: scatter matrix rows first, then scale by scalar weight
            # indices: (n_pre, n_conn), matrix: (n_pre, n)
            # expand indices to (n_pre, n_conn, 1) for 2D scatter
            idx_exp = jnp.broadcast_to(indices[:, :, None], (n_pre, indices.shape[1], n))
            m_exp = jnp.broadcast_to(matrix[:, None, :], (n_pre, indices.shape[1], n))
            r = weights * jnp.zeros((n_post, n), matrix.dtype).at[idx_exp, jnp.arange(n)].add(m_exp)
        else:
            # hetero: weights (n_pre, n_conn), matrix (n_pre, n)
            idx_exp = jnp.broadcast_to(indices[:, :, None], (n_pre, indices.shape[1], n))
            vals = weights[:, :, None] * matrix[:, None, :]
            r = jnp.zeros((n_post, n), weights.dtype).at[idx_exp, jnp.arange(n)].add(vals)
    else:
        # Gather mode: Y[n_pre, n]
        # Y[i, :] = sum_k weights[i, k] * matrix[indices[i, k], :]
        if weights.ndim == 0:
            r = jax.vmap(lambda ind: weights * jnp.sum(matrix[ind], axis=0))(indices)
        else:
            r = jax.vmap(lambda w, ind: jnp.sum(w[:, None] * matrix[ind], axis=0))(weights, indices)
    return u.maybe_decimal(r * m_unit * w_unit)


@namescope(static_argnames=['shape', 'transpose'])
def fcnmv_yw2y(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    y: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Union[jax.Array, u.Quantity]:
    """
    Per-synapse element-wise product of a vector and fixed-connection weights.

    For each stored connection slot ``(i, k)`` of the fixed-connection-number
    (ELL) structure, computes ``out[i, k] = weights[i, k] * y[i]`` when
    ``transpose=False`` or ``out[i, k] = weights[i, k] * y[indices[i, k]]`` when
    ``transpose=True``.  The result has the same shape as ``indices`` (one value
    per structural non-zero).

    This is the fixed-connection-number analog of :func:`brainevent.csrmv_yw2y`
    and follows the **same** ``transpose`` convention as that operator: the flag
    selects whether ``y`` is indexed by the leading (row) axis or by the stored
    ``indices`` (column) axis.  Note this is the opposite sense of the
    ``transpose`` flag in :func:`fcnmv` / :func:`fcnmm`.

    Parameters
    ----------
    weights : jax.Array or brainunit.Quantity
        Per-synapse weights.  Either a scalar / size-1 array (homogeneous) or a
        ``(rows, n_conn)`` array matching ``indices``.  Must be floating dtype.
    indices : jax.Array
        Integer ELL index array of shape ``(rows, n_conn)``.
    y : jax.Array or brainunit.Quantity
        Dense 1-D vector.  Sized ``shape[0]`` when ``transpose=False`` or
        ``shape[1]`` when ``transpose=True``.
    shape : tuple of int
        Two-element ``(rows_dim, cols_dim)`` logical shape of the ELL operand.
    transpose : bool
        If ``False``, index ``y`` by the leading axis (broadcast).  If ``True``,
        index ``y`` by ``indices`` (gather).

    Returns
    -------
    out : jax.Array or brainunit.Quantity
        Per-synapse result of shape ``indices.shape``.  Output unit is
        ``unit(weights) * unit(y)``.

    Raises
    ------
    ValueError
        If ``indices`` is not 2-D, ``shape`` is not length-2, ``weights`` is
        neither size-1 nor shaped like ``indices``, ``weights`` is not floating,
        or ``y`` is not 1-D with the length required by ``transpose``.

    See Also
    --------
    csrmv_yw2y : CSR equivalent of this operator.
    fcnmv : Fixed-connection sparse matrix--vector product.

    Notes
    -----
    Unlike :func:`csrmv_yw2y` (which drops ``y``'s physical unit), this operator
    keeps both units, consistent with the :func:`fcnmv` sibling and with the
    mathematics of a product.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import fcnmv_yw2y
        >>> weights = jnp.array([[0.5, 1.0], [1.5, 2.0]])
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> y = jnp.array([10.0, 20.0])
        >>> fcnmv_yw2y(weights, indices, y, shape=(2, 3), transpose=False)
        Array([[ 5., 10.],
               [30., 40.]], dtype=float32)
    """
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got {indices.ndim}D.")
    if len(shape) != 2:
        raise ValueError(f"shape must be length-2, got {shape!r}.")

    weights, w_unit = u.split_mantissa_unit(weights)
    y, y_unit = u.split_mantissa_unit(y)

    if not jnp.issubdtype(weights.dtype, jnp.floating):
        raise ValueError(f"weights must be a floating-point type, got {weights.dtype}.")
    if jnp.size(weights) != 1 and weights.shape != indices.shape:
        raise ValueError(
            f"weights must be size-1 or match indices shape {indices.shape}, "
            f"got {weights.shape}."
        )
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got {y.ndim}D.")
    expected_y = shape[1] if transpose else shape[0]
    if y.shape[0] != expected_y:
        raise ValueError(
            f"y length {y.shape[0]} does not match expected {expected_y} for "
            f"transpose={transpose} and shape={tuple(shape)}."
        )

    yv = y[indices] if transpose else y[:, None]
    res = jnp.broadcast_to(weights * yv, indices.shape)
    return u.maybe_decimal(res * w_unit * y_unit)
