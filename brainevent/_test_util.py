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


import brainstate
import jax
import jax.numpy as jnp

import brainevent


def generate_fixed_conn_num_indices(
    n_pre: int,
    n_post: int,
    n_conn: int,
    replace: bool = True,
    rng=brainstate.random.DEFAULT
):
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
    x = x.data if isinstance(x, brainevent.BaseArray) else x
    weights = weights.data if isinstance(weights, brainevent.BaseArray) else weights
    indices = indices.data if isinstance(indices, brainevent.BaseArray) else indices

    homo_w = jnp.size(weights) == 1
    post = jnp.zeros((shape[1],))

    def loop_fn(i_pre, post):
        post_ids = indices[i_pre]
        post = post.at[post_ids].add(weights * x[i_pre] if homo_w else weights[i_pre] * x[i_pre])
        return post

    return jax.lax.fori_loop(
        0, x.shape[0], loop_fn, post
    )

    # for i_pre in range(x.shape[0]):
    #     post_ids = indices[i_pre]
    #     post = post.at[post_ids].add(weights * x[i_pre] if homo_w else weights[i_pre] * x[i_pre])
    # return post


@brainstate.transform.jit(static_argnums=(3,))
def matrix_fcn(xs, weights, indices, shape):
    xs = xs.data if isinstance(xs, brainevent.BaseArray) else xs
    weights = weights.data if isinstance(weights, brainevent.BaseArray) else weights
    indices = indices.data if isinstance(indices, brainevent.BaseArray) else indices

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

    # for i_pre in range(xs.shape[1]):
    #     post_ids = indices[i_pre]
    #     post = post.at[:, post_ids].add(
    #         weights * xs[:, i_pre: i_pre + 1]
    #         if homo_w else
    #         (weights[i_pre] * xs[:, i_pre: i_pre + 1])
    #     )
    # return post


@brainstate.transform.jit(static_argnums=(3,))
def fcn_vector(x, weights, indices, shape):
    x = x.data if isinstance(x, brainevent.BaseArray) else x
    weights = weights.data if isinstance(weights, brainevent.BaseArray) else weights
    indices = indices.data if isinstance(indices, brainevent.BaseArray) else indices

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

    # for i in range(shape[0]):
    #     post_ids = indices[i]
    #     ws = weights if homo_w else weights[i]
    #     out = out.at[i].set(jnp.sum(x[post_ids] * ws))
    # return out


@brainstate.transform.jit(static_argnums=(3,))
def fcn_matrix(xs, weights, indices, shape):
    xs = xs.data if isinstance(xs, brainevent.BaseArray) else xs
    weights = weights.data if isinstance(weights, brainevent.BaseArray) else weights
    indices = indices.data if isinstance(indices, brainevent.BaseArray) else indices

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

    # for i in range(shape[0]):
    #     post_ids = indices[i]
    #     ws = weights if homo_w else jnp.expand_dims(weights[i], axis=1)
    #     out = out.at[i].set(jnp.sum(xs[post_ids] * ws, axis=0))
    # return out


def allclose(x, y, rtol=1e-4, atol=1e-4):
    x = x.data if isinstance(x, brainevent.EventArray) else x
    y = y.data if isinstance(y, brainevent.EventArray) else y
    return jnp.allclose(x, y, rtol=rtol, atol=atol)


def gen_events(shape, prob=0.5, asbool=True):
    events = brainstate.random.random(shape) < prob
    if not asbool:
        events = jnp.asarray(events, dtype=float)
    return brainevent.EventArray(events)


def ones_like(x):
    return jax.tree.map(jnp.ones_like, x)
