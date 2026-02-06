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


def _get_n_conn(indptr):
    """Extract n_conn from indptr as a concrete integer."""
    if hasattr(indptr, '__array__'):
        arr = np.asarray(indptr)
        return int(arr[1] - arr[0])
    return int(indptr[1] - indptr[0])


def vector_csr(x, w, indices, indptr, shape, n_conn=None):
    if n_conn is None:
        n_conn = _get_n_conn(indptr)
    return _vector_csr_impl(x, w, indices, indptr, shape, n_conn)


@brainstate.transform.jit(static_argnums=(4, 5), static_argnames=['shape', 'n_conn'])
def _vector_csr_impl(x, w, indices, indptr, shape, n_conn):
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((shape[1],))

    def body_fn(i_pre, post):
        start = indptr[i_pre]
        ids = jax.lax.dynamic_slice(indices, (start,), (n_conn,))
        if homo_w:
            inc = w * x[i_pre]
        else:
            ws = jax.lax.dynamic_slice(w, (start,), (n_conn,))
            inc = ws * x[i_pre]
        ids, inc = jnp.broadcast_arrays(ids, inc)
        return post.at[ids].add(inc)

    return jax.lax.fori_loop(0, x.shape[0], body_fn, post)


def matrix_csr(xs, w, indices, indptr, shape, n_conn=None):
    if n_conn is None:
        n_conn = _get_n_conn(indptr)
    return _matrix_csr_impl(xs, w, indices, indptr, shape, n_conn)


@brainstate.transform.jit(static_argnums=(4, 5), static_argnames=['shape', 'n_conn'])
def _matrix_csr_impl(xs, w, indices, indptr, shape, n_conn):
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((xs.shape[0], shape[1]))

    def body_fn(i_pre, post):
        start = indptr[i_pre]
        ids = jax.lax.dynamic_slice(indices, (start,), (n_conn,))
        xs_col = jax.lax.dynamic_slice(xs, (0, i_pre), (xs.shape[0], 1))
        if homo_w:
            inc = w * xs_col
        else:
            ws = jax.lax.dynamic_slice(w, (start,), (n_conn,))
            inc = ws * xs_col
        return post.at[:, ids].add(inc)

    return jax.lax.fori_loop(0, xs.shape[1], body_fn, post)


def csr_vector(x, w, indices, indptr, shape, n_conn=None):
    if n_conn is None:
        n_conn = _get_n_conn(indptr)
    return _csr_vector_impl(x, w, indices, indptr, shape, n_conn)


@brainstate.transform.jit(static_argnums=(4, 5), static_argnames=['shape', 'n_conn'])
def _csr_vector_impl(x, w, indices, indptr, shape, n_conn):
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0]])

    def body_fn(i, out):
        start = indptr[i]
        ids = jax.lax.dynamic_slice(indices, (start,), (n_conn,))
        ws = w if homo_w else jax.lax.dynamic_slice(w, (start,), (n_conn,))
        return out.at[i].set(jnp.sum(x[ids] * ws))

    return jax.lax.fori_loop(0, shape[0], body_fn, out)


def csr_matrix(xs, w, indices, indptr, shape, n_conn=None):
    if n_conn is None:
        n_conn = _get_n_conn(indptr)
    return _csr_matrix_impl(xs, w, indices, indptr, shape, n_conn)


@brainstate.transform.jit(static_argnums=(4, 5), static_argnames=['shape', 'n_conn'])
def _csr_matrix_impl(xs, w, indices, indptr, shape, n_conn):
    # CSR @ matrix
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0], xs.shape[1]])

    def body_fn(i, out):
        start = indptr[i]
        ids = jax.lax.dynamic_slice(indices, (start,), (n_conn,))
        ws = w if homo_w else jnp.expand_dims(jax.lax.dynamic_slice(w, (start,), (n_conn,)), axis=1)
        return out.at[i].set(jnp.sum(xs[ids] * ws, axis=0))

    return jax.lax.fori_loop(0, shape[0], body_fn, out)
