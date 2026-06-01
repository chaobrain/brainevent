# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Event-driven STDP weight updates for matrices stored in CSC order.

These helpers mirror :func:`brainevent.update_csr_on_binary_pre` /
:func:`brainevent.update_csr_on_binary_post` but operate on ``weight`` arrays
stored in *column-major* (CSC) order.  They reuse the existing CSR plasticity
primitives by exploiting the structural symmetry between the two layouts:

* ``update_csc_on_binary_post`` is the *favorable* direction for CSC -- it
  iterates over postsynaptic spikes, which index the CSC primary (column) axis.
  The CSC arrays are, array-for-array, the CSR arrays of ``W.T``, so the update
  reduces to :func:`update_csr_on_binary_pre` on the transposed shape, with no
  permutation required.
* ``update_csc_on_binary_pre`` is the *unfavorable* direction -- it iterates
  over presynaptic spikes (the secondary/row axis of CSC).  It builds the
  row-major (CSR-like) view of the structure together with a permutation
  ``perm`` (via :func:`brainevent.csc_to_csr_index`) and routes through
  :func:`update_csr_on_binary_post`, which scatters the per-synapse updates back
  into the canonical CSC ``weight`` order through ``perm``.
"""

import numbers
from typing import Optional, Union

import brainunit as u
import jax
import numpy as np

from brainevent._misc import csc_to_csr_index
from brainevent._typing import MatrixShape
from .plasticity_binary import update_csr_on_binary_pre, update_csr_on_binary_post

__all__ = [
    'update_csc_on_binary_pre',
    'update_csc_on_binary_post',
]


def update_csc_on_binary_pre(
    weight: Union[u.Quantity, jax.Array, numbers.Number],
    indices: Union[np.ndarray, jax.Array],
    indptr: Union[np.ndarray, jax.Array],
    pre_spike: jax.Array,
    post_trace: Union[u.Quantity, jax.Array],
    w_min: Optional[Union[u.Quantity, jax.Array, numbers.Number]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array, numbers.Number]] = None,
    *,
    shape: MatrixShape,
    backend: Optional[str] = None,
):
    """Update CSC synaptic weights triggered by presynaptic binary spike events.

    Implements the presynaptic component of additive spike-timing-dependent
    plasticity (STDP) for a weight matrix ``W`` of shape ``(n_pre, n_post)``
    stored in Compressed Sparse Column (CSC) order.  For each presynaptic neuron
    ``i`` that fires (``pre_spike[i]`` is ``True`` or nonzero), every stored
    synapse ``(i, j)`` is updated:

    ``W[i, j] <- clip(W[i, j] + post_trace[j], w_min, w_max)``

    This is the *unfavorable* direction for CSC (presynaptic spikes index the
    row axis, not the stored column axis).  The function builds the row-major
    (CSR-like) view of the structure and a permutation ``perm`` mapping each
    row-major slot back to the canonical CSC ``weight`` order, then delegates to
    :func:`brainevent.update_csr_on_binary_post`, which scatters the per-synapse
    updates back through ``perm``.

    Parameters
    ----------
    weight : jax.Array, Quantity, or number
        Sparse synaptic weight array in CSC data order, with shape ``(nse,)``.
        May carry physical units via ``brainunit.Quantity``.
    indices : ndarray or jax.Array
        Row index array of the CSC format, with shape ``(nse,)`` and integer
        dtype.
    indptr : ndarray or jax.Array
        Column pointer array of the CSC format, with shape ``(n_post + 1,)`` and
        integer dtype.
    pre_spike : jax.Array
        Binary or boolean presynaptic spike array, with shape ``(n_pre,)``.
        Boolean ``True`` or any nonzero float indicates a spike.
    post_trace : jax.Array or Quantity
        Postsynaptic eligibility trace, with shape ``(n_post,)``.  Must be
        unit-compatible with ``weight``.
    w_min, w_max : jax.Array, Quantity, number, or None, optional
        Lower/upper bounds for weight clipping (same units as ``weight``).  If
        ``None``, the corresponding bound is not applied.
    shape : tuple of int
        Full matrix shape ``(n_pre, n_post)``.
    backend : str or None, optional
        Compute backend forwarded to the underlying primitive.

    Returns
    -------
    jax.Array or Quantity
        Updated weight array with the same shape ``(nse,)`` and units as the
        input ``weight``, in canonical CSC order.

    See Also
    --------
    update_csc_on_binary_post : Postsynaptic-spike-triggered CSC weight update.
    brainevent.update_csr_on_binary_post : The CSR primitive this reuses.
    brainevent.csc_to_csr_index : Builds the CSR-like view and ``perm``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainevent
        >>> W = jnp.array([[0.5, 0.0, 0.8],
        ...                [0.0, 0.3, 0.2]], dtype=jnp.float32)
        >>> csc = brainevent.CSC.fromdense(W)
        >>> pre_spike = jnp.array([True, False])
        >>> post_trace = jnp.array([0.1, 0.2, 0.05], dtype=jnp.float32)
        >>> new_w = brainevent.update_csc_on_binary_pre(
        ...     csc.data, csc.indices, csc.indptr, pre_spike, post_trace,
        ...     shape=csc.shape,
        ... )
    """
    with jax.ensure_compile_time_eval():
        csr_indptr, csr_indices, perm = csc_to_csr_index(indptr, indices, shape=shape)
    return update_csr_on_binary_post(
        weight,
        csr_indices,
        csr_indptr,
        perm,
        post_trace,
        pre_spike,
        w_min,
        w_max,
        shape=shape[::-1],
        backend=backend,
    )


def update_csc_on_binary_post(
    weight: Union[u.Quantity, jax.Array, numbers.Number],
    indices: Union[np.ndarray, jax.Array],
    indptr: Union[np.ndarray, jax.Array],
    pre_trace: Union[u.Quantity, jax.Array],
    post_spike: jax.Array,
    w_min: Optional[Union[u.Quantity, jax.Array, numbers.Number]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array, numbers.Number]] = None,
    *,
    shape: MatrixShape,
    backend: Optional[str] = None,
):
    """Update CSC synaptic weights triggered by postsynaptic binary spike events.

    Implements the postsynaptic component of additive spike-timing-dependent
    plasticity (STDP) for a weight matrix ``W`` of shape ``(n_pre, n_post)``
    stored in Compressed Sparse Column (CSC) order.  For each postsynaptic
    neuron ``j`` that fires (``post_spike[j]`` is ``True`` or nonzero), every
    stored synapse ``(i, j)`` is updated:

    ``W[i, j] <- clip(W[i, j] + pre_trace[i], w_min, w_max)``

    This is the *favorable* direction for CSC: postsynaptic spikes index the
    stored column axis, so the update streams directly over the CSC arrays with
    no permutation.  Because the CSC arrays of ``W`` are the CSR arrays of
    ``W.T``, the operation reduces to :func:`brainevent.update_csr_on_binary_pre`
    on the transposed shape.

    Parameters
    ----------
    weight : jax.Array, Quantity, or number
        Sparse synaptic weight array in CSC data order, with shape ``(nse,)``.
        May carry physical units via ``brainunit.Quantity``.
    indices : ndarray or jax.Array
        Row index array of the CSC format, with shape ``(nse,)`` and integer
        dtype.
    indptr : ndarray or jax.Array
        Column pointer array of the CSC format, with shape ``(n_post + 1,)`` and
        integer dtype.
    pre_trace : jax.Array or Quantity
        Presynaptic eligibility trace, with shape ``(n_pre,)``.  Must be
        unit-compatible with ``weight``.
    post_spike : jax.Array
        Binary or boolean postsynaptic spike array, with shape ``(n_post,)``.
        Boolean ``True`` or any nonzero float indicates a spike.
    w_min, w_max : jax.Array, Quantity, number, or None, optional
        Lower/upper bounds for weight clipping (same units as ``weight``).  If
        ``None``, the corresponding bound is not applied.
    shape : tuple of int
        Full matrix shape ``(n_pre, n_post)``.
    backend : str or None, optional
        Compute backend forwarded to the underlying primitive.

    Returns
    -------
    jax.Array or Quantity
        Updated weight array with the same shape ``(nse,)`` and units as the
        input ``weight``, in canonical CSC order.

    See Also
    --------
    update_csc_on_binary_pre : Presynaptic-spike-triggered CSC weight update.
    brainevent.update_csr_on_binary_pre : The CSR primitive this reuses.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainevent
        >>> W = jnp.array([[0.5, 0.0, 0.8],
        ...                [0.0, 0.3, 0.2]], dtype=jnp.float32)
        >>> csc = brainevent.CSC.fromdense(W)
        >>> pre_trace = jnp.array([0.1, -0.05], dtype=jnp.float32)
        >>> post_spike = jnp.array([True, False, True])
        >>> new_w = brainevent.update_csc_on_binary_post(
        ...     csc.data, csc.indices, csc.indptr, pre_trace, post_spike,
        ...     shape=csc.shape,
        ... )
    """
    return update_csr_on_binary_pre(
        weight,
        indices,
        indptr,
        post_spike,
        pre_trace,
        w_min,
        w_max,
        shape=shape[::-1],
        backend=backend,
    )
