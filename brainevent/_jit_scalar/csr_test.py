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
from brainevent._data import _initialize_conn_length
from brainevent._jit_scalar.csr import (
    jits_to_csr,
    jits_csr_count_p_call,
    jits_csr_fill_p_call,
)
from brainevent._test_util import allclose

# The CSR count/fill primitives only have a native ``numba`` backend, which compiles per
# test and dominates wall-clock. Mark the whole module ``slow`` so the default ``pytest``
# run skips it; CI runs it via ``pytest -m ""``.
pytestmark = pytest.mark.slow


class Test_Scalar_To_CSR:
    @pytest.mark.parametrize('corder', [True, False])
    def test_to_csr_roundtrip(self, corder):
        # Converting directly to CSR reproduces the dense matrix; this also
        # exercises count/fill PRNG alignment end-to-end.
        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5, 0.2, 123), shape=shape, corder=corder)

        csr = jits_to_csr(
            mat.weight, mat.prob, mat.seed,
            shape=mat.shape, corder=mat.corder, backend=mat.backend,
        )
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        assert allclose(csr.todense(), mat.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('corder', [True, False])
    def test_count_matches_row_nnz(self, corder):
        # The count primitive returns, per row, the number of non-zeros of the
        # dense matrix it mirrors.
        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5, 0.2, 123), shape=shape, corder=corder)

        clen = _initialize_conn_length(mat.prob)
        w = jnp.atleast_1d(jnp.asarray(mat.weight))
        row_counts = jits_csr_count_p_call(
            w, w, clen, mat.seed, shape=shape, corder=corder, backend=mat.backend,
        )[0]
        expected = (np.asarray(mat.todense()) != 0).sum(axis=1)
        assert np.array_equal(np.asarray(row_counts), expected)

    @pytest.mark.parametrize('corder', [True, False])
    def test_fill_given_indptr(self, corder):
        # Given the count-derived indptr, the fill primitive writes a structure
        # whose densification matches the dense matrix.
        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5, 0.2, 123), shape=shape, corder=corder)

        clen = _initialize_conn_length(mat.prob)
        w = jnp.atleast_1d(jnp.asarray(mat.weight))
        row_counts = jits_csr_count_p_call(
            w, w, clen, mat.seed, shape=shape, corder=corder, backend=mat.backend,
        )[0]
        indptr = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
        )
        nnz = int(indptr[-1])
        indices, data = jits_csr_fill_p_call(
            w, w, clen, mat.seed, indptr, nnz, shape=shape, corder=corder, backend=mat.backend,
        )
        csr = brainevent.CSR((data, indices, indptr), shape=shape)
        assert allclose(csr.todense(), mat.todense())

    def test_to_csr_prob_zero_empty(self):
        shape = (20, 30)
        csr = jits_to_csr(1.5, 0.0, 123, shape=shape, corder=True, backend=None)
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        assert np.asarray(csr.indices).shape == (0,)
        assert np.asarray(csr.data).shape == (0,)
        assert np.all(np.asarray(csr.indptr) == 0)

    def test_to_csr_units(self):
        import brainunit as u

        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5 * u.mV, 0.2, 123), shape=shape)

        csr = jits_to_csr(
            mat.weight, mat.prob, mat.seed,
            shape=mat.shape, corder=mat.corder, backend=mat.backend,
        )
        dense = mat.todense()
        assert u.get_unit(csr.data) == u.get_unit(dense)
        assert u.math.allclose(csr.todense(), dense)
