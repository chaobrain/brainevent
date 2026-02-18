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

# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp

from brainevent import numba_kernel, binary_coomv_p


def _coomv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        if v[row[i]]:
                            posts[col[i]] += w
            else:
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        if v[row[i]] > 0.:
                            posts[col[i]] += w
        else:
            # transpose=True, heterogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        if v[row[i]]:
                            posts[col[i]] += weights[i]
            else:
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        if v[row[i]] > 0.:
                            posts[col[i]] += weights[i]
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        if v[col[i]]:
                            posts[row[i]] += w
            else:
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        if v[col[i]] > 0.:
                            posts[row[i]] += w
        else:
            # transpose=False, heterogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        if v[col[i]]:
                            posts[row[i]] += weights[i]
            else:
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        if v[col[i]] > 0.:
                            posts[row[i]] += weights[i]

    def kernel(weights, row, col, v):
        return numba_kernel(mv, outs=kwargs['outs'])(weights, row, col, v)

    return kernel


def _coomv_jax_kernel(
    transpose: bool,
    shape,
    **kwargs,
):
    from jax.experimental.sparse import COO

    def kernel(weights, row, col, vector):
        vector = jnp.asarray(vector, dtype=weights.dtype)
        weights_dense = (jnp.full(row.shape, weights[0], dtype=weights.dtype) if jnp.size(weights) == 1 else weights)
        jax_coo = COO((weights_dense, row, col), shape=shape)
        return [(jax_coo.T @ vector) if transpose else (jax_coo @ vector)]

    return kernel


PLATFORM = jax.devices()[0].platform
binary_coomv_p.def_kernel('jax', 'cpu', _coomv_jax_kernel)
binary_coomv_p.def_numba_kernel(_coomv_numba_kernel)
result = binary_coomv_p.benchmark(platform='cpu', n_warmup=10, n_runs=10, n_batch_per_run=100, rtol=1e-2, atol=1e-2)
print(result)
