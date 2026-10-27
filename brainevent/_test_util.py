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


import functools
import hashlib
import random
from contextlib import contextmanager
from typing import Union

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent


def _has_gpu() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


requires_gpu = pytest.mark.skipif(
    not _has_gpu(),
    reason="No GPU detected via jax.devices('gpu')",
)


@contextmanager
def jax_x64_enabled():
    """Enable JAX 64-bit mode for the duration of the block, then restore it.

    Yields
    ------
    None
        The block runs with ``jax_enable_x64`` set to ``True``.

    Notes
    -----
    ``jax_enable_x64`` is global process state, so tests that need int64 ``indptr``
    or float64 data must restore the previous value even on failure -- otherwise
    they leak x64 into every later test in the session.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._test_util import jax_x64_enabled
        >>> with jax_x64_enabled():
        ...     jnp.array([0, 1], dtype=jnp.int64).dtype
        dtype('int64')
    """
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def with_jax_x64(fn):
    """Run a test function inside :func:`jax_x64_enabled`.

    Parameters
    ----------
    fn : callable
        Test function that needs int64 or float64 arrays.

    Returns
    -------
    callable
        ``fn`` wrapped so x64 is on during the call and restored afterwards.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._test_util import with_jax_x64
        >>> @with_jax_x64
        ... def int64_dtype():
        ...     return jnp.array([0, 1], dtype=jnp.int64).dtype
        >>> int64_dtype()
        dtype('int64')
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with jax_x64_enabled():
            return fn(*args, **kwargs)

    return wrapper


def generate_fixed_conn_num_indices(
    n_pre: int,
    n_post: int,
    n_conn: int,
    replace: Union[bool, str] = 'rand',
    rng=brainstate.random.DEFAULT
):
    """Generate random post-synaptic index arrays for fixed-number connectivity.

    Parameters
    ----------
    n_pre : int
        Number of pre-synaptic neurons.
    n_post : int
        Number of post-synaptic neurons.
    n_conn : int
        Number of connections per pre-synaptic neuron.
    replace : bool or str, optional
        Whether to sample with replacement.  ``'rand'`` (default) randomly
        chooses with 50 % probability.
    rng : brainstate.random.RandomState, optional
        Random number generator instance.

    Returns
    -------
    jax.Array
        Int array of shape ``(n_pre, n_conn)`` with post-synaptic indices.
    """
    if isinstance(replace, str) and replace == 'rand':
        replace = random.random() < 0.5

    if replace:
        indices = rng.randint(0, n_post, (n_pre, n_conn))
    else:
        indices = brainstate.transform.for_loop(
            lambda *args: rng.choice(n_post, size=n_conn, replace=False),
            length=n_pre
        )
    return jnp.asarray(indices)


@brainstate.transform.jit(static_argnums=(3,), )
def vector_fcn(x, weights, indices, shape):
    """Reference implementation of fixed-connectivity vector multiply (transpose).

    Computes ``FCN.T @ x`` using an explicit loop for testing.

    Parameters
    ----------
    x : array_like
        Input vector of shape ``(n_pre,)``.
    weights : array_like
        Connection weights — scalar for homogeneous, or ``(n_pre,)``/``(n_pre, n_conn)``.
    indices : array_like
        Post-synaptic index array of shape ``(n_pre, n_conn)``.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.

    Returns
    -------
    jax.Array
        Output vector of shape ``(n_post,)``.
    """
    x = x.value if isinstance(x, brainevent.EventRepresentation) else x
    weights = weights.value if isinstance(weights, brainevent.EventRepresentation) else weights
    indices = indices.value if isinstance(indices, brainevent.EventRepresentation) else indices

    homo_w = jnp.size(weights) == 1
    post = jnp.zeros((shape[1],))

    def loop_fn(i_pre, post):
        post_ids = indices[i_pre]
        post = post.at[post_ids].add(weights * x[i_pre] if homo_w else weights[i_pre] * x[i_pre])
        return post

    return jax.lax.fori_loop(
        0, x.shape[0], loop_fn, post
    )


@brainstate.transform.jit(static_argnums=(3,))
def matrix_fcn(xs, weights, indices, shape):
    """Reference implementation of fixed-connectivity matrix multiply (transpose).

    Computes ``FCN.T @ xs`` using an explicit loop for testing.

    Parameters
    ----------
    xs : array_like
        Input matrix of shape ``(batch, n_pre)``.
    weights : array_like
        Connection weights.
    indices : array_like
        Post-synaptic index array of shape ``(n_pre, n_conn)``.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.

    Returns
    -------
    jax.Array
        Output matrix of shape ``(batch, n_post)``.
    """
    xs = xs.value if isinstance(xs, brainevent.EventRepresentation) else xs
    weights = weights.value if isinstance(weights, brainevent.EventRepresentation) else weights
    indices = indices.value if isinstance(indices, brainevent.EventRepresentation) else indices

    homo_w = jnp.size(weights) == 1
    post = jnp.zeros((xs.shape[0], shape[1]))

    def loop_fn(i_pre, post):
        post_ids = indices[i_pre]
        x = jax.lax.dynamic_slice(xs, (0, i_pre), (xs.shape[0], 1))
        post = post.at[:, post_ids].add(
            weights * x
            if homo_w else
            (weights[i_pre] * x)
        )
        return post

    return jax.lax.fori_loop(
        0, xs.shape[1], loop_fn, post
    )


@brainstate.transform.jit(static_argnums=(3,))
def fcn_vector(x, weights, indices, shape):
    """Reference implementation of fixed-connectivity vector multiply (forward).

    Computes ``FCN @ x`` using an explicit loop for testing.

    Parameters
    ----------
    x : array_like
        Input vector of shape ``(n_post,)``.
    weights : array_like
        Connection weights.
    indices : array_like
        Post-synaptic index array of shape ``(n_pre, n_conn)``.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.

    Returns
    -------
    jax.Array
        Output vector of shape ``(n_pre,)``.
    """
    x = x.value if isinstance(x, brainevent.EventRepresentation) else x
    weights = weights.value if isinstance(weights, brainevent.EventRepresentation) else weights
    indices = indices.value if isinstance(indices, brainevent.EventRepresentation) else indices

    homo_w = jnp.size(weights) == 1
    out = jnp.zeros([shape[0]])

    def loop_fn(i_pre, post):
        post_ids = indices[i_pre]
        ws = weights if homo_w else weights[i_pre]
        post = post.at[i_pre].add(jnp.sum(x[post_ids] * ws))
        return post

    return jax.lax.fori_loop(
        0, shape[0], loop_fn, out
    )


@brainstate.transform.jit(static_argnums=(3,))
def fcn_matrix(xs, weights, indices, shape):
    """Reference implementation of fixed-connectivity matrix multiply (forward).

    Computes ``FCN @ xs`` using an explicit loop for testing.

    Parameters
    ----------
    xs : array_like
        Input matrix of shape ``(n_post, n_cols)``.
    weights : array_like
        Connection weights.
    indices : array_like
        Post-synaptic index array of shape ``(n_pre, n_conn)``.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.

    Returns
    -------
    jax.Array
        Output matrix of shape ``(n_pre, n_cols)``.
    """
    xs = xs.value if isinstance(xs, brainevent.EventRepresentation) else xs
    weights = weights.value if isinstance(weights, brainevent.EventRepresentation) else weights
    indices = indices.value if isinstance(indices, brainevent.EventRepresentation) else indices

    # CSR @ matrix
    homo_w = jnp.size(weights) == 1
    out = jnp.zeros([shape[0], xs.shape[1]])

    def loop_fn(i_pre, post):
        post_ids = indices[i_pre]
        ws = weights if homo_w else jnp.expand_dims(weights[i_pre], axis=1)
        post = post.at[i_pre].add(jnp.sum(xs[post_ids] * ws, axis=0))
        return post

    return jax.lax.fori_loop(
        0, shape[0], loop_fn, out
    )


def allclose(x, y, rtol=1e-6, atol=1e-6):
    """Check whether two arrays are element-wise equal within tolerances.

    Parameters
    ----------
    x, y : array_like
        Arrays to compare.  ``BinaryArray`` instances are automatically
        unwrapped.
    rtol : float, optional
        Relative tolerance.  Default is ``1e-6``.
    atol : float, optional
        Absolute tolerance.  Default is ``1e-6``.

    Returns
    -------
    bool
        ``True`` if all elements satisfy the tolerance check.
    """
    x = x.value if isinstance(x, brainevent.BinaryArray) else x
    y = y.value if isinstance(y, brainevent.BinaryArray) else y
    return jnp.allclose(x, y, rtol=rtol, atol=atol)


def gen_events(shape, prob=0.5, asbool=True):
    """Generate a random ``BinaryArray`` for testing.

    Parameters
    ----------
    shape : tuple of int
        Shape of the event array.
    prob : float, optional
        Probability that each element is active.  Default is 0.5.
    asbool : bool, optional
        If ``True`` (default), the underlying array is boolean.
        If ``False``, it is cast to float.

    Returns
    -------
    BinaryArray
        A random binary event array.
    """
    events = brainstate.random.random(shape) < prob
    if not asbool:
        events = jnp.asarray(events, dtype=float)
    return brainevent.BinaryArray(events)


def ones_like(x):
    """Create a pytree with the same structure as *x* filled with ones.

    Parameters
    ----------
    x : pytree
        Reference pytree structure.

    Returns
    -------
    pytree
        Pytree of the same structure with ``jnp.ones_like`` applied to
        every leaf.
    """
    return jax.tree.map(jnp.ones_like, x)


def _tree_vdot(a, b):
    leaves = jax.tree_util.tree_map(
        lambda u, v: jnp.vdot(jnp.ravel(u).astype(jnp.float64 if u.dtype == jnp.float64 else jnp.float32),
                              jnp.ravel(v).astype(jnp.float64 if u.dtype == jnp.float64 else jnp.float32)),
        a, b,
    )
    return sum(jax.tree_util.tree_leaves(leaves))


def check_jitc_weight_grads(op, params, *, atol=1e-4):
    """Check the weight JVPs and gradients of a JITC operator.

    ``op(*params)`` must be linear in the scalar weight ``params`` (as every
    JITC product is), so its derivative along parameter ``k`` is ``op`` at the
    unit weights ``e_k``. Verifies the JVP and gradient values against that,
    and that each tangent and gradient keeps the dtype of the output and the
    parameter (the rules must not compute in a promoted dtype).

    Parameters
    ----------
    op : callable
        Maps the scalar weight parameters to the operator output.
    params : sequence of jax.Array
        The weight parameters, as 0-d floating arrays.
    atol : float, optional
        Absolute tolerance of the value comparisons.
    """
    params = tuple(params)
    out = op(*params)
    ct = jax.random.uniform(jax.random.PRNGKey(0), out.shape, out.dtype)
    grads = jax.grad(lambda *p: jnp.sum(op(*p) * ct), argnums=tuple(range(len(params))))(*params)
    for k, (p, g) in enumerate(zip(params, grads)):
        unit = tuple(jnp.ones_like(q) if i == k else jnp.zeros_like(q) for i, q in enumerate(params))
        basis = op(*unit)
        _, tangent = jax.jvp(op, params, unit)
        assert tangent.dtype == out.dtype, (k, tangent.dtype, out.dtype)
        assert jnp.allclose(tangent, basis, atol=atol), (k, tangent, basis)
        assert g.dtype == p.dtype, (k, g.dtype, p.dtype)
        assert jnp.allclose(g, jnp.sum(basis * ct), atol=atol), (k, g, jnp.sum(basis * ct))


def check_event_op_grads(
    op,
    dense_op,
    params,
    x,
    *,
    mode: str = 'binary',
    rtol: float = 1e-4,
    atol: float = 1e-4,
    check_vmap: bool = True,
    seed: int = 0,
):
    """Check an event operation's forward pass and derivatives against a dense reference.

    The event operation ``op(params, x)`` wraps the raw event array ``x`` in
    an event representation internally (for example
    ``BinaryArray(x) @ CSR(params)``). The reference ``dense_op(params, z)``
    is the same product written with dense arrays, evaluated at
    ``z = factor(x)``, where ``factor`` is the activity mask for
    ``mode='binary'`` and the identity for ``mode='weighted'``.

    Parameters
    ----------
    op : callable
        ``op(params, x) -> y``, the operation under test.
    dense_op : callable
        ``dense_op(params, z) -> y``, linear in ``z``.
    params : pytree of jax.Array
        Differentiable weight parameters.
    x : jax.Array
        Raw event array. Integer arrays are allowed (weighted mode); their
        event gradient is not checked. Should contain values other than 0 and 1
        (for example ``2.0`` and ``-1.0``) to exercise #198.
    mode : {'binary', 'weighted'}, optional
        Event semantics.
    rtol, atol : float, optional
        Tolerances.
    check_vmap : bool, optional
        Also check the parameter gradient under ``jax.vmap`` over a stacked
        pair of event arrays.
    seed : int, optional
        Seed for the random cotangent and tangents.

    Notes
    -----
    Checked properties:

    1. Forward: ``op(params, x) == dense_op(params, factor(x))``.
    2. Reverse mode w.r.t. ``params`` equals the dense reference gradient
       (the #198 property: binary weight gradients use the mask).
    3. Reverse mode w.r.t. ``x`` equals ``d dense_op / d z``. For binary
       events this is the straight-through estimator; for weighted events it
       is exact.
    4. Forward and reverse mode agree:
       ``<ct, J t> == <J^T ct, t>`` for random tangents ``t``.
    5. Optionally, the parameter gradient under ``jax.vmap``.

    ``dense_op`` runs under ``jax.default_matmul_precision('highest')``, so
    the reference is not degraded by TF32 on GPUs.
    """
    from brainevent._op.util import event_factor

    def ref_op(p, v):
        # Full float32 precision: GPUs default float32 matmuls to TF32
        # (~1e-3 relative error), which would swamp the comparison.
        with jax.default_matmul_precision('highest'):
            return dense_op(p, v)

    if mode == 'binary':
        z = event_factor(x, dtype=x.dtype)
    elif mode == 'weighted':
        z = x
    else:
        raise ValueError(f'Unknown mode {mode!r}.')

    x_is_float = jnp.issubdtype(x.dtype, jnp.floating)
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    y = op(params, x)
    y_ref = ref_op(params, z if x_is_float else z.astype(y.dtype))
    assert y.shape == y_ref.shape, (y.shape, y_ref.shape)
    assert jnp.allclose(y, y_ref, rtol=rtol, atol=atol), 'forward mismatch'

    ct = jax.random.normal(keys[0], y.shape, dtype=y.dtype)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    tp = jax.tree_util.tree_unflatten(
        treedef,
        [jax.random.normal(k, l.shape, dtype=l.dtype)
         for k, l in zip(jax.random.split(keys[1], len(leaves)), leaves)],
    )

    if x_is_float:
        _, vjp = jax.vjp(op, params, x)
        g_p, g_x = vjp(ct)
        _, vjp_ref = jax.vjp(ref_op, params, z)
        g_p_ref, g_x_ref = vjp_ref(ct)
        assert jnp.allclose(g_x, g_x_ref, rtol=rtol, atol=atol), \
            f'event gradient mismatch:\n{g_x}\nvs\n{g_x_ref}'
    else:
        # Integer events are not differentiable (float0 cotangents); only the
        # weight gradient is checked. The reference uses the float values.
        _, vjp = jax.vjp(lambda p: op(p, x), params)
        g_p, = vjp(ct)
        _, vjp_ref = jax.vjp(lambda p: ref_op(p, z.astype(y.dtype)), params)
        g_p_ref, = vjp_ref(ct)

    for a, b in zip(jax.tree_util.tree_leaves(g_p), jax.tree_util.tree_leaves(g_p_ref)):
        assert jnp.allclose(a, b, rtol=rtol, atol=atol), f'weight gradient mismatch:\n{a}\nvs\n{b}'

    # Forward mode w.r.t. params only for binary events: the event tangent of a
    # binary op is straight-through, which has no forward-mode counterpart of
    # the mask, so the dot-product identity is checked per argument.
    _, jp = jax.jvp(lambda p: op(p, x), (params,), (tp,))
    lhs = jnp.vdot(ct, jp)
    rhs = _tree_vdot(g_p, tp)
    assert jnp.allclose(lhs, rhs, rtol=rtol, atol=atol * max(1, y.size)), \
        f'weight JVP/VJP mismatch: {lhs} vs {rhs}'
    if x_is_float:
        tx = jax.random.normal(keys[2], x.shape, dtype=x.dtype)
        _, jx = jax.jvp(lambda v: op(params, v), (x,), (tx,))
        lhs = jnp.vdot(ct, jx)
        rhs = jnp.vdot(g_x, tx)
        assert jnp.allclose(lhs, rhs, rtol=rtol, atol=atol * max(1, y.size)), \
            f'event JVP/VJP mismatch: {lhs} vs {rhs}'

    if check_vmap:
        xs = jnp.stack([x, jnp.flip(x, axis=0)])
        zs = jax.vmap(lambda v: event_factor(v, dtype=v.dtype) if mode == 'binary' else v)(xs)
        g = jax.vmap(lambda v: jax.grad(lambda p: jnp.vdot(ct, op(p, v)))(params))(xs)
        g_ref = jax.vmap(lambda v: jax.grad(lambda p: jnp.vdot(ct, ref_op(p, v.astype(y.dtype))))(params))(zs)
        for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(g_ref)):
            assert jnp.allclose(a, b, rtol=rtol, atol=atol), f'vmapped weight gradient mismatch:\n{a}\nvs\n{b}'


def check_plasticity_grads(
    op,
    ref,
    weight,
    trace,
    w_min,
    w_max,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    seed: int = 0,
):
    """Check a plasticity update's derivatives against a pure-JAX reference.

    Parameters
    ----------
    op : callable
        ``op(weight, trace, w_min, w_max) -> new_weight``, the update under
        test (spikes and structure closed over).
    ref : callable
        Pure-JAX reference with the same signature.
    weight, trace : jax.Array
        Primal weight (per-synapse) and trace arrays.
    w_min, w_max : jax.Array
        Scalar clip bounds (differentiated as well).
    rtol, atol : float, optional
        Tolerances.
    seed : int, optional
        Seed for the random cotangent and tangents.

    Notes
    -----
    Checked properties: forward values; reverse mode w.r.t. all four
    arguments; forward mode; the dot-product identity
    ``<ct, J t> == <J^T ct, t>``; ``jit(grad)``; and ``vmap`` of the gradient
    over a stack of ``(weight, trace)`` pairs.
    """
    args = (weight, trace, w_min, w_max)
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)

    y = op(*args)
    y_ref = ref(*args)
    assert y.shape == y_ref.shape, (y.shape, y_ref.shape)
    assert jnp.allclose(y, y_ref, rtol=rtol, atol=atol), f'forward mismatch:\n{y}\nvs\n{y_ref}'

    ct = jax.random.normal(keys[0], y.shape, dtype=y.dtype)
    tangents = tuple(
        jax.random.normal(k, jnp.shape(a), dtype=jnp.asarray(a).dtype)
        for k, a in zip(jax.random.split(keys[1], 4), args)
    )

    _, vjp = jax.vjp(op, *args)
    g = vjp(ct)
    _, vjp_ref = jax.vjp(ref, *args)
    g_ref = vjp_ref(ct)
    for name, a, b in zip(('weight', 'trace', 'w_min', 'w_max'), g, g_ref):
        assert jnp.allclose(a, b, rtol=rtol, atol=atol), f'{name} gradient mismatch:\n{a}\nvs\n{b}'

    _, jt = jax.jvp(op, args, tangents)
    _, jt_ref = jax.jvp(ref, args, tangents)
    assert jnp.allclose(jt, jt_ref, rtol=rtol, atol=atol), f'JVP mismatch:\n{jt}\nvs\n{jt_ref}'

    lhs = jnp.vdot(ct, jt)
    rhs = _tree_vdot(g, tangents)
    assert jnp.allclose(lhs, rhs, rtol=rtol, atol=atol * max(1, y.size)), f'JVP/VJP mismatch: {lhs} vs {rhs}'

    def loss(f):
        return lambda w, t: jnp.vdot(ct, f(w, t, w_min, w_max))

    g_jit = jax.jit(jax.grad(loss(op), argnums=(0, 1)))(weight, trace)
    for a, b in zip(g_jit, g_ref[:2]):
        assert jnp.allclose(a, b, rtol=rtol, atol=atol), 'jit(grad) mismatch'

    ws = jnp.stack([weight, weight + 0.25 * jax.random.normal(keys[2], weight.shape, dtype=weight.dtype)])
    ts = jnp.stack([trace, -trace])
    gv = jax.vmap(jax.grad(loss(op), argnums=(0, 1)))(ws, ts)
    gv_ref = jax.vmap(jax.grad(loss(ref), argnums=(0, 1)))(ws, ts)
    for a, b in zip(gv, gv_ref):
        assert jnp.allclose(a, b, rtol=rtol, atol=atol), f'vmap(grad) mismatch:\n{a}\nvs\n{b}'


# Mask digests (:func:`jitc_mask_digest`) of the row-class JITC matrices
# ``shape=(37, 129), seed=123`` drawn by brainevent 0.3.0, keyed by
# ``(prob, corder)``. The three families share one walk, hence one mask.
JITC_GOLDEN_0_3_0 = {
    (0.1, False): '89f2ebf82eea0cfc',
    (0.25, False): '3e1d1fa632214caf',
    (0.1, True): '43ce994bddcf6b25',
    (0.25, True): '88612f59e30adab3',
}


def jitc_mask_digest(dense) -> str:
    """Short digest of the non-zero pattern of a dense matrix."""
    mask = np.asarray(dense) != 0
    return hashlib.sha256(np.packbits(mask).tobytes() + str(mask.shape).encode()).hexdigest()[:16]


def check_jitc_ops_match_dense(mat, *, seed=0, rtol=1e-4, atol=1e-4):
    """Check every product, ``.T`` and ``tocsr`` of a JITC matrix against its ``todense``.

    Covers float and binary-event operands on both sides, as vectors and
    matrices, so each walk (gather and scatter, mv and mm) is exercised.
    """
    dense = np.asarray(mat.todense(), np.float64)
    m, n = dense.shape
    rng = np.random.default_rng(seed)
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    v_r, v_l = rng.normal(size=n), rng.normal(size=m)
    b_r, b_l = rng.normal(size=(n, 3)), rng.normal(size=(3, m))
    e_r, e_l = rng.random(n) < 0.3, rng.random(m) < 0.3
    E_r, E_l = rng.random((n, 3)) < 0.3, rng.random((3, m)) < 0.3
    pairs = {
        'M @ v': (mat @ f32(v_r), dense @ v_r),
        'v @ M': (f32(v_l) @ mat, v_l @ dense),
        'M @ B': (mat @ f32(b_r), dense @ b_r),
        'B @ M': (f32(b_l) @ mat, b_l @ dense),
        'M @ events': (mat @ brainevent.BinaryArray(jnp.asarray(e_r)), dense @ e_r),
        'events @ M': (brainevent.BinaryArray(jnp.asarray(e_l)) @ mat, e_l @ dense),
        'M @ event matrix': (mat @ brainevent.BinaryArray(jnp.asarray(E_r)), dense @ E_r),
        'event matrix @ M': (brainevent.BinaryArray(jnp.asarray(E_l)) @ mat, E_l @ dense),
        'M.T': (mat.T.todense(), dense.T),
        'tocsr': (mat.tocsr().todense(), dense),
    }
    for name, (got, want) in pairs.items():
        np.testing.assert_allclose(np.asarray(got, np.float64), want, rtol=rtol, atol=atol, err_msg=name)


def check_jitc_grads_match_dense(make, params, *, seed=0, rtol=1e-4, atol=1e-4):
    """Check weight and operand gradients of ``v @ M`` and ``M @ x`` against ``todense``.

    Parameters
    ----------
    make : callable
        Maps the weight parameters to a JITC matrix.
    params : sequence of float
        The weight parameters.
    """
    params = tuple(jnp.asarray(p, jnp.float32) for p in params)
    m, n = make(*params).shape
    rng = np.random.default_rng(seed)
    operands = {
        'left': jnp.asarray(rng.normal(size=m), jnp.float32),
        'right': jnp.asarray(rng.normal(size=n), jnp.float32),
    }
    for side, a in operands.items():
        def loss(p, a, dense):
            mat = make(*p)
            mat = mat.todense() if dense else mat
            return jnp.sum(jnp.sin(a @ mat if side == 'left' else mat @ a))

        got = jax.grad(lambda p, a: loss(p, a, False), argnums=(0, 1))(params, a)
        want = jax.grad(lambda p, a: loss(p, a, True), argnums=(0, 1))(params, a)
        for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
            np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=rtol, atol=atol, err_msg=side)


def jitc_kernel_orientations(fn, *args):
    """``(primitive, transpose, corder)`` of every JITC kernel call in ``fn(*args)``."""
    found = []

    def visit(jaxpr):
        for eqn in jaxpr.eqns:
            if 'corder' in eqn.params and 'transpose' in eqn.params:
                found.append((eqn.primitive.name, eqn.params['transpose'], eqn.params['corder']))
            for param in eqn.params.values():
                inner = getattr(param, 'jaxpr', param)
                if hasattr(inner, 'eqns'):
                    visit(inner)

    visit(jax.make_jaxpr(fn)(*args).jaxpr)
    return found
