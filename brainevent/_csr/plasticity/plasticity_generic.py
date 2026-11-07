# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Generic event-triggered per-edge plasticity rules on CSR connectivity.

:func:`csr_apply_on_pre` and :func:`csr_apply_on_post` apply a user-supplied elementwise
``rule`` to an arbitrary per-edge state (a pytree of ``(nse,)`` arrays) for every
edge whose presynaptic row (``csr_apply_on_pre``) or postsynaptic column
(``csr_apply_on_post``) is active, and leave the other edges untouched.  The
implementation is pure JAX (gather, ``jax.vmap(rule)``, masked select), so it is
differentiable through ``rule``, the edge state, the per-neuron values, and the
rule parameters, and composes with ``jit`` and ``vmap``.
"""

from typing import Any, Callable, Optional, Tuple

import brainunit as u
import jax
import jax.numpy as jnp

from brainevent._op.util import event_active
from brainevent._typing import MatrixShape

__all__ = [
    'csr_apply_on_pre',
    'csr_apply_on_post',
]

PlasticityRule = Callable[[Any, Any, Any, Any], Any]


def _leading_dim_check(tree, size: int, what: str) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        shape = jnp.shape(leaf)
        if len(shape) == 0 or shape[0] != size:
            raise ValueError(
                f"Every leaf of {what} must have leading dimension {size}, got a leaf of shape {shape}."
            )


def _check_spike(spike, size: int, what: str) -> None:
    if jnp.ndim(spike) != 1 or jnp.shape(spike)[0] != size:
        raise ValueError(f"{what} must have shape ({size},), got {jnp.shape(spike)}.")


def _is_quantity(x) -> bool:
    return isinstance(x, u.Quantity)


def _select(active, new, old):
    """Return ``new`` on active edges and ``old`` elsewhere, in ``old``'s unit and dtype."""
    old_m, unit = u.split_mantissa_unit(old)
    new_m = u.Quantity(new).to(unit).mantissa
    old_m = jnp.asarray(old_m)
    mask = jnp.reshape(active, active.shape + (1,) * (old_m.ndim - 1))
    out = jnp.where(mask, new_m, old_m).astype(old_m.dtype)
    return u.maybe_decimal(out * unit)


def _apply_rule(
    rule: PlasticityRule,
    edge_state,
    active: jax.Array,
    pre_e,
    post_e,
    params,
):
    new_state = jax.vmap(rule, in_axes=(0, 0, 0, None))(edge_state, pre_e, post_e, params)
    in_tree = jax.tree_util.tree_structure(edge_state, is_leaf=_is_quantity)
    out_tree = jax.tree_util.tree_structure(new_state, is_leaf=_is_quantity)
    if in_tree != out_tree:
        raise TypeError(
            "The plasticity rule must return a pytree with the same structure as edge_state; "
            f"got {out_tree}, expected {in_tree}."
        )
    return jax.tree_util.tree_map(
        lambda n, o: _select(active, n, o), new_state, edge_state, is_leaf=_is_quantity,
    )


def _prepare(
    edge_state,
    indices,
    indptr,
    spike,
    pre_vals,
    post_vals,
    shape: MatrixShape,
    spike_dim: int,
    spike_name: str,
) -> Tuple[jax.Array, jax.Array]:
    n_pre, n_post = shape
    nse = jnp.shape(indices)[0]
    if jnp.shape(indptr)[0] != n_pre + 1:
        raise ValueError(f"indptr must have shape ({n_pre + 1},), got {jnp.shape(indptr)}.")
    _check_spike(spike, shape[spike_dim], spike_name)
    _leading_dim_check(edge_state, nse, 'edge_state')
    _leading_dim_check(pre_vals, n_pre, 'pre_vals')
    _leading_dim_check(post_vals, n_post, 'post_vals')
    indptr = jnp.asarray(indptr)
    row_ids = jnp.repeat(
        jnp.arange(n_pre, dtype=indptr.dtype),
        jnp.diff(indptr),
        total_repeat_length=nse,
    )
    return row_ids, jnp.asarray(indices)


def csr_apply_on_pre(
    rule: PlasticityRule,
    edge_state,
    indices,
    indptr,
    pre_spike: jax.Array,
    *,
    pre_vals=None,
    post_vals=None,
    params=None,
    shape: MatrixShape,
):
    """Apply a per-edge plasticity rule to the edges of active presynaptic neurons.

    For every stored edge ``e = (i, j)`` of a CSR matrix with
    ``shape = (n_pre, n_post)``, evaluate

    ``new_e = rule(edge_state[e], pre_vals[i], post_vals[j], params)``

    and return ``new_e`` where presynaptic neuron ``i`` is active and the
    unchanged ``edge_state[e]`` elsewhere.  A numeric spike is active iff it is
    non-zero (negative values and NaN are active); a boolean spike iff ``True``.

    Parameters
    ----------
    rule : callable
        ``rule(edge_e, pre_e, post_e, params) -> new_edge_e``.  ``edge_e``,
        ``pre_e`` and ``post_e`` are the pytrees ``edge_state``, ``pre_vals``
        and ``post_vals`` indexed at one edge (leading dimension removed);
        ``params`` is passed unchanged.  It must return a pytree with the
        structure of ``edge_state`` and is applied to every edge with
        :func:`jax.vmap`.
    edge_state : pytree
        Per-edge state in CSR data order; every leaf has leading dimension
        ``nse`` (trailing dimensions are allowed), e.g.
        ``{'w': w, 'elig': elig}``.
    indices : jax.Array
        CSR column indices, shape ``(nse,)``.
    indptr : jax.Array
        CSR row pointers, shape ``(n_pre + 1,)``.
    pre_spike : jax.Array
        Presynaptic spikes (bool or numeric), shape ``(n_pre,)``.
    pre_vals : pytree or None, optional
        Per-presynaptic-neuron values (leading dimension ``n_pre``), gathered
        to edges by row.  ``None`` passes ``None`` to ``rule``.
    post_vals : pytree or None, optional
        Per-postsynaptic-neuron values (leading dimension ``n_post``),
        gathered to edges by column.  ``None`` passes ``None`` to ``rule``.
    params : pytree or None, optional
        Rule parameters, broadcast to every edge (differentiable).
    shape : tuple of int
        Matrix shape ``(n_pre, n_post)``.

    Returns
    -------
    pytree
        The new edge state, with the structure of ``edge_state`` and leaves of
        the same shape and dtype.

    Raises
    ------
    TypeError
        If ``rule`` returns a pytree whose structure differs from
        ``edge_state``.
    ValueError
        If a leaf of ``edge_state``, ``pre_vals`` or ``post_vals`` has the
        wrong leading dimension, or ``pre_spike`` / ``indptr`` do not match
        ``shape``.

    See Also
    --------
    csr_apply_on_post : Postsynaptic-spike-triggered counterpart.
    update_csr_on_binary_pre : Fast event-driven kernel for the additive rule
        ``w += post_trace``.
    CSR.apply_on_pre : Method form on a :class:`~brainevent.CSR` matrix.

    Notes
    -----
    The rule is evaluated on every edge and the result is masked, so the cost
    is ``O(nse)`` per call independent of the number of spikes; for the plain
    additive rule prefer :func:`update_csr_on_binary_pre`.

    Gradients flow through ``rule``, ``edge_state``, ``pre_vals``,
    ``post_vals`` and ``params``; spikes and structure are non-differentiable.
    As with any :func:`jax.numpy.where`, inactive edges send a zero cotangent
    into ``rule``; if the rule's derivative is not finite there (e.g.
    ``log(0)``) the gradient becomes ``nan``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainevent
        >>> csr = brainevent.CSR.fromdense(jnp.array([[1., 0.5], [0., 2.]]))
        >>> def rule(edge, pre, post, p):
        ...     w = edge['w'] + p['lr'] * (p['w_max'] - edge['w']) * post
        ...     return {'w': w, 'elig': edge['elig'] * 0.9 + post}
        >>> state = {'w': csr.data, 'elig': jnp.zeros_like(csr.data)}
        >>> new = brainevent.csr_apply_on_pre(
        ...     rule, state, csr.indices, csr.indptr, jnp.array([True, False]),
        ...     post_vals=jnp.array([1., 0.5]), params={'lr': 0.1, 'w_max': 3.},
        ...     shape=csr.shape,
        ... )
        >>> new['w']
        Array([1.2  , 0.625, 2.   ], dtype=float32)
    """
    row_ids, indices = _prepare(
        edge_state, indices, indptr, pre_spike, pre_vals, post_vals, shape, 0, 'pre_spike',
    )
    active = event_active(jnp.asarray(pre_spike))[row_ids]
    pre_e = jax.tree_util.tree_map(lambda v: v[row_ids], pre_vals)
    post_e = jax.tree_util.tree_map(lambda v: v[indices], post_vals)
    return _apply_rule(rule, edge_state, active, pre_e, post_e, params)


def csr_apply_on_post(
    rule: PlasticityRule,
    edge_state,
    indices,
    indptr,
    post_spike: jax.Array,
    *,
    pre_vals=None,
    post_vals=None,
    params=None,
    shape: MatrixShape,
):
    """Apply a per-edge plasticity rule to the edges of active postsynaptic neurons.

    For every stored edge ``e = (i, j)`` of a CSR matrix with
    ``shape = (n_pre, n_post)``, evaluate

    ``new_e = rule(edge_state[e], pre_vals[i], post_vals[j], params)``

    and return ``new_e`` where postsynaptic neuron ``j`` is active and the
    unchanged ``edge_state[e]`` elsewhere.  The edge state stays in CSR data
    order: the post id of every edge is ``indices[e]``, so no CSC permutation
    is needed.

    Parameters
    ----------
    rule : callable
        ``rule(edge_e, pre_e, post_e, params) -> new_edge_e``; see
        :func:`csr_apply_on_pre`.
    edge_state : pytree
        Per-edge state in CSR data order; every leaf has leading dimension
        ``nse``.
    indices : jax.Array
        CSR column indices, shape ``(nse,)``.
    indptr : jax.Array
        CSR row pointers, shape ``(n_pre + 1,)``.
    post_spike : jax.Array
        Postsynaptic spikes (bool or numeric), shape ``(n_post,)``.
    pre_vals : pytree or None, optional
        Per-presynaptic-neuron values (leading dimension ``n_pre``).
    post_vals : pytree or None, optional
        Per-postsynaptic-neuron values (leading dimension ``n_post``).
    params : pytree or None, optional
        Rule parameters, broadcast to every edge (differentiable).
    shape : tuple of int
        Matrix shape ``(n_pre, n_post)``.

    Returns
    -------
    pytree
        The new edge state, with the structure of ``edge_state`` and leaves of
        the same shape and dtype.

    Raises
    ------
    TypeError
        If ``rule`` returns a pytree whose structure differs from
        ``edge_state``.
    ValueError
        If a leaf has the wrong leading dimension, or ``post_spike`` /
        ``indptr`` do not match ``shape``.

    See Also
    --------
    csr_apply_on_pre : Presynaptic-spike-triggered counterpart.
    update_csr_on_binary_post : Fast event-driven kernel for the additive rule
        ``w += pre_trace``.
    CSR.apply_on_post : Method form on a :class:`~brainevent.CSR` matrix.

    Notes
    -----
    Cost and gradient behaviour are as in :func:`csr_apply_on_pre`.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainevent
        >>> csr = brainevent.CSR.fromdense(jnp.array([[1., 0.5], [0., 2.]]))
        >>> def depress(w, pre, post, lr):
        ...     return w - lr * w * pre
        >>> brainevent.csr_apply_on_post(
        ...     depress, csr.data, csr.indices, csr.indptr, jnp.array([0., 1.]),
        ...     pre_vals=jnp.array([1., 0.5]), params=0.1, shape=csr.shape,
        ... )
        Array([1.  , 0.45, 1.9 ], dtype=float32)
    """
    row_ids, indices = _prepare(
        edge_state, indices, indptr, post_spike, pre_vals, post_vals, shape, 1, 'post_spike',
    )
    active = event_active(jnp.asarray(post_spike))[indices]
    pre_e = jax.tree_util.tree_map(lambda v: v[row_ids], pre_vals)
    post_e = jax.tree_util.tree_map(lambda v: v[indices], post_vals)
    return _apply_rule(rule, edge_state, active, pre_e, post_e, params)
