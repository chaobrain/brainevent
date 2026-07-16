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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._jit_normal.csr import (
    jitn_to_csr,
    jitn_csr_count_p,
    jitn_csr_count_p_call,
    jitn_csr_fill_p,
    jitn_csr_fill_p_call,
)
from brainevent._test_util import allclose, requires_gpu_only_kernel

# The CSR count/fill primitives currently have only a GPU cuda_raw backend.
# Mark the whole module slow so the default pytest run skips it; CI can run
# it explicitly when GPU coverage is available.
pytestmark = pytest.mark.slow


class Test_Normal_To_CSR:
    @pytest.mark.parametrize('corder', [True, False])
    @requires_gpu_only_kernel(jitn_csr_count_p, jitn_csr_fill_p)
    def test_to_csr_roundtrip(self, corder):
        # Converting directly to CSR reproduces the dense matrix; this also
        # exercises seed normalization plus count/fill PRNG alignment (including
        # the per-connection normal weight draw) end-to-end.
        shape = (20, 30)
        mat = brainevent.JITCNormalR((1.5, 0.2, 0.2, 42), shape=shape, corder=corder)

        csr = jitn_to_csr(
            mat.wloc, mat.wscale, mat.prob, mat.seed,
            shape=mat.shape, backend=mat.backend,
        )
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        assert allclose(csr.todense(), mat.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('corder', [True, False])
    @requires_gpu_only_kernel(jitn_csr_count_p)
    def test_count_matches_row_nnz(self, corder):
        # The count primitive returns, per row, the number of non-zeros of the
        # dense matrix it mirrors. The dense ``jitn`` normalizes the seed, so the
        # raw count primitive must be fed the normalized seed to agree.
        shape = (20, 30)
        mat = brainevent.JITCNormalR((1.5, 0.2, 0.2, 42), shape=shape, corder=corder)

        clen = _initialize_conn_length(mat.prob)
        seed = _initialize_seed(mat.seed)
        w0 = jnp.atleast_1d(jnp.asarray(mat.wloc))
        w1 = jnp.atleast_1d(jnp.asarray(mat.wscale))
        chunk_counts = jitn_csr_count_p_call(
            w0, w1, clen, seed, shape=shape, backend=mat.backend,
        )[0]
        row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32)
        expected = (np.asarray(mat.todense()) != 0).sum(axis=1)
        assert np.array_equal(np.asarray(row_counts), expected)

    @pytest.mark.parametrize('corder', [True, False])
    @requires_gpu_only_kernel(jitn_csr_count_p, jitn_csr_fill_p)
    def test_fill_given_indptr(self, corder):
        # Given the count-derived indptr, the fill primitive writes a structure
        # whose densification matches the dense matrix (normalized seed).
        shape = (20, 30)
        mat = brainevent.JITCNormalR((1.5, 0.2, 0.2, 42), shape=shape, corder=corder)

        clen = _initialize_conn_length(mat.prob)
        seed = _initialize_seed(mat.seed)
        w0 = jnp.atleast_1d(jnp.asarray(mat.wloc))
        w1 = jnp.atleast_1d(jnp.asarray(mat.wscale))
        chunk_counts = jitn_csr_count_p_call(
            w0, w1, clen, seed, shape=shape, backend=mat.backend,
        )[0]
        row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32)
        indptr = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
        )
        nnz = int(indptr[-1])
        chunk_offsets = (
            indptr[:-1, None]
            + jnp.cumsum(chunk_counts, axis=1, dtype=jnp.int32)
            - chunk_counts
        )
        indices, data = jitn_csr_fill_p_call(
            w0, w1, clen, seed, chunk_offsets, nnz, shape=shape, backend=mat.backend,
        )
        csr = brainevent.CSR((data, indices, indptr), shape=shape)
        assert allclose(csr.todense(), mat.todense())

    def test_to_csr_prob_zero_empty(self):
        shape = (20, 30)
        csr = jitn_to_csr(1.5, 0.2, 0.0, 42, shape=shape, corder=True, backend=None)
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        assert np.asarray(csr.indices).shape == (0,)
        assert np.asarray(csr.data).shape == (0,)
        assert np.all(np.asarray(csr.indptr) == 0)

    @requires_gpu_only_kernel(jitn_csr_count_p, jitn_csr_fill_p)
    def test_to_csr_units(self):
        import brainunit as u

        shape = (20, 30)
        mat = brainevent.JITCNormalR((1.5 * u.mV, 0.2 * u.mV, 0.2, 42), shape=shape)

        csr = jitn_to_csr(
            mat.wloc, mat.wscale, mat.prob, mat.seed,
            shape=mat.shape, backend=mat.backend,
        )
        dense = mat.todense()
        assert u.get_unit(csr.data) == u.get_unit(dense)
        assert u.math.allclose(csr.todense(), dense)
