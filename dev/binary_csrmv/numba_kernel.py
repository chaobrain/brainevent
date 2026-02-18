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

from brainevent import numba_kernel, binary_csrmv_p
from brainevent.config import get_numba_parallel


def _csrmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # A.T @ v  —  scatter: one thread per row i of A, writes to posts[indices[j]]
        # Cannot parallelize over i due to write-after-write race on posts[indices[j]].
        if weight_info.size == 1:
            # homogeneous weight
            if vector_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w
            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i] > 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w
        else:
            # heterogeneous weights
            if vector_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]
            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i] > 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

    else:
        # A @ v  —  gather: one thread per output row i, reads from v[indices[j]]
        # Safe to parallelize over rows.
        if weight_info.size == 1:
            # homogeneous weight
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += w
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] > 0.:
                                r += w
                        posts[i] = r
        else:
            # heterogeneous weights
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += weights[j]
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] > 0.:
                                r += weights[j]
                        posts[i] = r

    def kernel(weights, indices, indptr, vector):
        return numba_kernel(mv, outs=kwargs['outs'])(weights, indices, indptr, vector)

    return kernel


def _csrmv_jax_kernel(
    transpose: bool,
    shape,
    **kwargs,
):
    from jax.experimental.sparse import CSR

    def kernel(weights, indices, indptr, vector):
        vector = jnp.asarray(vector, dtype=jnp.float32)
        weights_dense = (
            jnp.full(indices.shape, weights[0], dtype=jnp.float32)
            if jnp.size(weights) == 1
            else jnp.asarray(weights, dtype=jnp.float32)
        )
        jax_csr = CSR((weights_dense, indices, indptr), shape=shape)
        return [(jax_csr.T @ vector) if transpose else (jax_csr @ vector)]

    return kernel


PLATFORM = jax.devices()[0].platform
binary_csrmv_p.def_kernel('jax', PLATFORM, _csrmv_jax_kernel)
binary_csrmv_p.def_numba_kernel(_csrmv_numba_kernel)
result = binary_csrmv_p.benchmark(platform=PLATFORM, n_warmup=1, n_runs=3, n_batch_per_run=10, verbose=True)
result.print(group_by='label', highlight_best=True)
