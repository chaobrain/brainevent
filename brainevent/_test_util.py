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


import random
from typing import Union

import brainstate
import jax
import jax.numpy as jnp

import brainevent


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
