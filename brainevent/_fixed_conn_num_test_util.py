# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


import brainstate as bst
import jax.numpy as jnp
import numpy as np


def generate_data(n_pre: int, n_post: int, n_conn: int, replace: bool = True, rng=bst.random.DEFAULT):
    if replace:
        indices = rng.randint(0, n_post, (n_pre, n_conn))
    else:
        indices = bst.compile.for_loop(
            lambda *args: rng.choice(n_post, n_conn, replace=False),
            length=n_pre
        )
    return indices


def vector_csr(x, weights, indices, shape):
    homo_w = jnp.size(weights) == 1
    post = jnp.zeros((shape[1],))
    for i_pre in range(x.shape[0]):
        post_ids = indices[i_pre]
        post = post.at[post_ids].add(weights * x[i_pre] if homo_w else weights[i_pre] * x[i_pre])
    return post


def matrix_csr(xs, weights, indices, shape):
    homo_w = jnp.size(weights) == 1
    post = jnp.zeros((xs.shape[0], shape[1]))
    for i_pre in range(xs.shape[1]):
        post_ids = indices[i_pre]
        post = post.at[:, post_ids].add(
            weights * xs[:, i_pre: i_pre + 1]
            if homo_w else
            (weights[i_pre] * xs[:, i_pre: i_pre + 1])
        )
    return post


def csr_vector(x, weights, indices, shape):
    homo_w = jnp.size(weights) == 1
    out = jnp.zeros([shape[0]])
    for i in range(shape[0]):
        post_ids = indices[i]
        ws = weights if homo_w else weights[i]
        out = out.at[i].set(jnp.sum(x[post_ids] * ws))
    return out


def csr_matrix(xs, weights, indices, shape):
    # CSR @ matrix
    homo_w = jnp.size(weights) == 1
    out = jnp.zeros([shape[0], xs.shape[1]])
    for i in range(shape[0]):
        post_ids = indices[i]
        ws = weights if homo_w else jnp.expand_dims(weights[i], axis=1)
        out = out.at[i].set(jnp.sum(xs[post_ids] * ws, axis=0))
    return out
