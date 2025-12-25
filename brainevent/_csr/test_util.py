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
import jax.numpy as jnp
import numpy as np


def get_csr(n_pre, n_post, prob, replace=True):
    n_conn = int(n_post * prob)
    indptr = np.arange(n_pre + 1) * n_conn
    if replace:
        indices = brainstate.random.randint(0, n_post, (n_pre * n_conn,))
    else:
        indices = brainstate.transform.for_loop(
            lambda *args: brainstate.random.choice(n_post, n_conn, replace=False),
            length=n_pre
        ).flatten()
    return indptr, indices


def vector_csr(x, w, indices, indptr, shape):
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((shape[1],))
    for i_pre in range(x.shape[0]):
        ids = indices[indptr[i_pre]: indptr[i_pre + 1]]
        inc = w * x[i_pre] if homo_w else w[indptr[i_pre]: indptr[i_pre + 1]] * x[i_pre]
        ids, inc = jnp.broadcast_arrays(ids, inc)
        post = post.at[ids].add(inc)
    return post


def matrix_csr(xs, w, indices, indptr, shape):
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((xs.shape[0], shape[1]))
    for i_pre in range(xs.shape[1]):
        ids = indices[indptr[i_pre]: indptr[i_pre + 1]]
        post = post.at[:, ids].add(
            w * xs[:, i_pre: i_pre + 1]
            if homo_w else
            (w[indptr[i_pre]: indptr[i_pre + 1]] * xs[:, i_pre: i_pre + 1])
        )
    return post


def csr_vector(x, w, indices, indptr, shape):
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0]])
    for i in range(shape[0]):
        ids = indices[indptr[i]: indptr[i + 1]]
        ws = w if homo_w else w[indptr[i]: indptr[i + 1]]
        out = out.at[i].set(jnp.sum(x[ids] * ws))
    return out


def csr_matrix(xs, w, indices, indptr, shape):
    # CSR @ matrix
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0], xs.shape[1]])
    for i in range(shape[0]):
        ids = indices[indptr[i]: indptr[i + 1]]
        ws = w if homo_w else jnp.expand_dims(w[indptr[i]: indptr[i + 1]], axis=1)
        out = out.at[i].set(jnp.sum(xs[ids] * ws, axis=0))
    return out
