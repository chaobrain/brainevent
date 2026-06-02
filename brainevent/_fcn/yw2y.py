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

from typing import Tuple, Union

import brainunit as u
import jax
import jax.numpy as jnp

from brainevent._misc import namescope

__all__ = [
    'fcnmv_yw2y',
]


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
    ``transpose`` flag in :func:`brainevent.fcnmv` / :func:`brainevent.fcnmm`.

    Unlike :func:`brainevent.csrmv_yw2y`, which returns a flat ``(nse,)`` vector
    because CSR stores its non-zeros in a 1-D array, this operator returns a 2-D
    array shaped like ``indices`` because the ELL layout stores non-zeros in a
    ``(rows, n_conn)`` grid.

    The operator is a plain element-wise / gather-multiply, so it is fully
    differentiable through JAX autodiff with respect to both ``weights`` and
    ``y``; no custom differentiation rule is required.  It is also ``jit``- and
    ``vmap``-compatible.

    Parameters
    ----------
    weights : jax.Array or brainunit.Quantity
        Per-synapse weights.  Either a scalar / size-1 array (homogeneous) or a
        ``(rows, n_conn)`` array matching ``indices`` (heterogeneous).  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer ELL index array of shape ``(rows, n_conn)``.
    y : jax.Array or brainunit.Quantity
        Dense 1-D vector.  Sized ``shape[0]`` when ``transpose=False`` or
        ``shape[1]`` when ``transpose=True``.
    shape : tuple of int
        Two-element ``(rows_dim, cols_dim)`` logical shape of the ELL operand.
    transpose : bool
        If ``False``, index ``y`` by the leading axis (broadcast over the
        connection axis).  If ``True``, index ``y`` by ``indices`` (gather).

    Returns
    -------
    out : jax.Array or brainunit.Quantity
        Per-synapse result of shape ``indices.shape``.  The output unit is
        ``unit(weights) * unit(y)``.

    Raises
    ------
    ValueError
        If ``indices`` is not 2-D.
    ValueError
        If ``shape`` is not length-2.
    ValueError
        If ``weights`` is not a floating-point dtype.
    ValueError
        If ``weights`` is neither size-1 nor shaped like ``indices``.
    ValueError
        If ``y`` is not 1-D.
    ValueError
        If ``y`` does not have the length required by ``transpose`` and
        ``shape``.

    See Also
    --------
    csrmv_yw2y : CSR equivalent of this operator.
    fcnmv : Fixed-connection sparse matrix--vector product.
    fcnmm : Fixed-connection sparse matrix--matrix product.

    Notes
    -----
    Unlike :func:`brainevent.csrmv_yw2y` (which drops ``y``'s physical unit),
    this operator keeps both units, consistent with the :func:`brainevent.fcnmv`
    sibling and with the mathematics of a product.

    Mixed floating dtypes between ``weights`` and ``y`` are promoted following
    JAX's standard promotion rules (e.g. ``float32 * float64 -> float64``); no
    equal-dtype constraint is imposed.

    Empty structures are supported: an ``indices`` array with ``rows == 0`` or
    ``n_conn == 0`` returns an array of the same (empty) shape.

    This operation is memory-bandwidth-bound and maps to a fused XLA
    broadcast-multiply (``transpose=False``) or gather-multiply
    (``transpose=True``); it is therefore implemented in pure JAX rather than as
    a custom kernel.

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

        >>> # transpose=True gathers y by the stored column indices
        >>> y_post = jnp.array([1.0, 2.0, 3.0])
        >>> fcnmv_yw2y(weights, indices, y_post, shape=(2, 3), transpose=True)
        Array([[0.5, 2. ],
               [3. , 6. ]], dtype=float32)
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
