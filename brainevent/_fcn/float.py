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
]

#import oping

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
            ct_vector = fcnmm_p_call(
                weights,
                indices,
                ct,
                shape=shape,
                transpose=not transpose,
                backend=kwargs['backend'],
            )[0]

        return weights, indices, ct_vector
    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_weight = ad.Zero(weights)

        elif homo:
            ct_weight = fcnmm_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                matrix,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )[0]
            ct_weight = jnp.sum(ct * ct_weight).reshape(*weight_info.shape)

        else:
            if transpose:
                # inputs: [k, n] @ [k, n_conn]
                # ct: [m, n]
                ct_weight = jax.vmap(lambda mat, ind: ct[ind] @ mat)(matrix, indices)
            else:
                # inputs: [m, n] @ [m, n_conn]
                # ct: [k, n]
                ct_weight = jax.vmap(lambda c, ind: (matrix[ind] @ c))(ct, indices)
        return ct_weight, indices, matrix


def _batching_base_fn(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = fcnmm_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _fcnmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 1):
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 2):
        return _batching_base_fn(args, axis=2, **kwargs)

    else:
        return general_batching_rule(fcnmm_p, args, axes, **kwargs)


def _fcnmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indices = jnp.asarray(np.random.randint(0, n_post, (n_pre, n_conn), dtype=np.int32))
            if homo:
                weights = jnp.ones(1, dtype=dtype)
            else:
                weights = jnp.ones((n_pre, n_conn), dtype=dtype)
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, B),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def fcnmm_p_call(
    weights: jax.Array,
    indices: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for sparse matrix--matrix product with fixed
    connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba or Pallas) without performing any
    physical-unit bookkeeping.  It is typically called from :func:`fcnmm`
    or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    matrix : jax.Array
        Dense matrix to multiply with, of shape ``(k, n)``.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, or
        ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result matrix.

    Notes
    -----
    The ``transpose=True`` path uses scatter-based accumulation via
    ``atomic_add`` on GPU backends, while ``transpose=False`` uses a
    gather-based reduction.

    See Also
    --------
    fcnmm : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return fcnmm_p(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=[out],
        backend=backend,
    )


fcnmm_p = XLACustomKernel(
    'fixed_num_mm',
    doc="""
Low-level XLA custom-kernel primitive for ``fcnmm``.

This ``XLACustomKernel`` instance dispatches the fixed-connection matrix-matrix
multiplication operation with floating-point weights to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. Unlike the binary variant, this operation
uses full floating-point weights and processes all entries (not just spikes).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``fcnmm_p.available_backends(platform)``,
and the default backend can be configured with ``fcnmm_p.set_default(platform, backend)``.

See Also
--------
fcnmm : High-level user-facing function wrapper.
"""
)
fcnmm_p.def_numba_kernel(_fcnmm_numba_kernel)
fcnmm_p.def_warp_kernel(_fcnmm_warp_kernel)
fcnmm_p.def_pallas_kernel('gpu', _fcnmm_pallas_kernel)
fcnmm_p.def_cuda_raw_kernel(_fcnmm_cuda_kernel)
fcnmm_p.def_kernel('jax_raw', 'cpu', _fcnmm_jax_kernel)
fcnmm_p.def_kernel('jax_raw', 'gpu', _fcnmm_jax_kernel)
fcnmm_p.def_kernel('jax_raw', 'tpu', _fcnmm_jax_kernel)
fcnmm_p.def_jvp_rule2(_fcnmm_jvp_weights, None, _fcnmm_jvp_matrix)
fcnmm_p.def_transpose_rule(_fcnmm_transpose_rule)
fcnmm_p.def_batching_rule(_fcnmm_batching)
fcnmm_p.def_call(fcnmm_p_call)
fcnmm_p.def_tags('fcn', 'float')
fcnmm_p.def_benchmark_data(_fcnmm_benchmark_data)
