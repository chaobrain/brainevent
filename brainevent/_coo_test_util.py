# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


import jax.numpy as jnp
import numpy as np
import brainstate as bst

def _get_coo(n_pre, n_post, prob, replace=True):
    n_conn = int(n_post * prob)
    rows = np.repeat(np.arange(n_pre), n_conn)

    if replace:
        cols = np.random.randint(0, n_post, size=(n_pre * n_conn,))
    else:
        cols = bst.compile.for_loop(
            lambda *args: bst.random.choice(n_post, n_conn, replace=False),
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

def vector_coo(x, w, row, col, shape):
    homo_w = jnp.size(w) == 1
    if homo_w:
        data = jnp.ones(row.shape) * w
        return _coo_matvec_impl(data, row, col, x, spinfo=shape, transpose=True)
    else:
        return _coo_matvec_impl(w, row, col, x, spinfo=shape, transpose=True)

def coo_vector(x, w, row, col, shape):
    homo_w = jnp.size(w) == 1
    if homo_w:
        data = jnp.ones(row.shape) * w
        return _coo_matvec_impl(data, row, col, x, spinfo=shape, transpose=False)
    else:
        return _coo_matvec_impl(w, row, col, x, spinfo=shape, transpose=False)

def matrix_coo(xs, w, row, col, shape):
    homo_w = jnp.size(w) == 1
    data = jnp.ones(row.shape) * w if homo_w else w
    output = jnp.zeros((xs.shape[0], shape[1]), dtype=xs.dtype)

    for i in range(len(data)):
        r = row[i]
        c = col[i]
        output = output.at[:, c].add(xs[:, r] * data[i])

    return output


def coo_matrix(xs, w, row, col, shape):
    homo_w = jnp.size(w) == 1
    data = jnp.ones(row.shape) * w if homo_w else w
    output = jnp.zeros((shape[0], xs.shape[1]), dtype=xs.dtype)

    for i in range(len(data)):
        r = row[i]
        c = col[i]
        output = output.at[r].add(data[i] * xs[c])

    return output
