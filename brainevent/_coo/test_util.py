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


import brainstate
import jax.lax
import jax.numpy as jnp
import numpy as np


def _get_coo(n_pre, n_post, prob, replace=True):
    n_conn = int(n_post * prob)
    rows = np.repeat(np.arange(n_pre), n_conn)

    if replace:
        cols = np.random.randint(0, n_post, size=(n_pre * n_conn,))
    else:
        cols = brainstate.transform.for_loop(
            lambda *args: brainstate.random.choice(n_post, n_conn, replace=False),
            length=n_pre
        ).flatten()

    return rows, cols


def _coo_matvec_impl(data, row, col, v, *, spinfo, transpose):
    v = jnp.asarray(v)
    if transpose:
        row, col = col, row
    out_shape = spinfo[1] if transpose else spinfo[0]
    dv = data * v[col]
    return jnp.zeros(out_shape, dv.dtype).at[row].add(dv)


@brainstate.transform.jit(static_argnames=['shape'], static_argnums=4)
def vector_coo(x, w, row, col, shape):
    homo_w = jnp.size(w) == 1
    if homo_w:
        data = jnp.ones(row.shape) * w
        return _coo_matvec_impl(data, row, col, x, spinfo=shape, transpose=True)
    else:
        return _coo_matvec_impl(w, row, col, x, spinfo=shape, transpose=True)


@brainstate.transform.jit(static_argnames=['shape'], static_argnums=4)
def coo_vector(x, w, row, col, shape):
    homo_w = jnp.size(w) == 1
    if homo_w:
        data = jnp.ones(row.shape) * w
        return _coo_matvec_impl(data, row, col, x, spinfo=shape, transpose=False)
    else:
        return _coo_matvec_impl(w, row, col, x, spinfo=shape, transpose=False)


@brainstate.transform.jit(static_argnames=['shape'], static_argnums=4)
def matrix_coo(xs, w, row, col, shape):
    homo_w = jnp.size(w) == 1
    data = jnp.ones(row.shape) * w if homo_w else w
    row = jnp.asarray(row)
    col = jnp.asarray(col)

    output = jnp.zeros((xs.shape[0], shape[1]), dtype=xs.dtype)
    if data.shape[0] == 0:
        return output
    def body(i, o):
        r = row[i]
        c = col[i]
        return o.at[:, c].add(xs[:, r] * data[i])

    output = jax.lax.fori_loop(0, data.shape[0], body, output)
    return output


@brainstate.transform.jit(static_argnames=['shape'], static_argnums=4)
def coo_matrix(xs, w, row, col, shape):
    homo_w = jnp.size(w) == 1
    data = jnp.ones(row.shape) * w if homo_w else w
    row = jnp.asarray(row)
    col = jnp.asarray(col)

    output = jnp.zeros((shape[0], xs.shape[1]), dtype=xs.dtype)
    if data.shape[0] == 0:
        return output
    def body(i, o):
        r = row[i]
        c = col[i]
        return o.at[r].add(data[i] * xs[c])

    output = jax.lax.fori_loop(0, data.shape[0], body, output)
    return output
