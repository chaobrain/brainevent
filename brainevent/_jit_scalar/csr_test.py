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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._data import _initialize_conn_length
from brainevent._jit_scalar.csr import (
    jits_to_csr,
    jits_csr_count_p,
    jits_csr_count_p_call,
    jits_csr_fill_p_call,
)
from brainevent._jit_scalar.float import jits
from brainevent._test_util import allclose

# The light-RNG CSR count/fill primitives have both CUDA and ``numba`` backends
# (the numba path draws the same matrix as CUDA). Mark the whole module ``slow`` so
# the default ``pytest`` run skips it (numba JIT compilation is slow); CI runs it via
# ``pytest -m ""``.
pytestmark = pytest.mark.slow

platform = jax.default_backend()
CSR_IMPLEMENTATIONS = tuple(jits_csr_count_p.available_backends(platform))
MATRIX_MODES = ['mv', 'mm']

# The count/fill primitives are only registered on CUDA; skip cleanly elsewhere.
requires_csr_backend = pytest.mark.skipif(
    not CSR_IMPLEMENTATIONS,
    reason=f'No jits_csr_count/fill implementation on platform={platform}',
)


def _reference_dense(weight, prob, seed, shape, corder, matrix_mode, backend):
    """The (non-transposed) dense matrix that ``jits_to_csr`` materializes."""
    return jits(
        weight, prob, seed,
        shape=shape, transpose=False, corder=corder,
        matrix_mode=matrix_mode, backend=backend,
    )


def _counts_to_offsets(count_out, corder, n_rows):
    """Mirror ``jits_to_csr``'s orchestration: turn the count-pass output into the
    ``indptr`` and the fill-pass ``offsets`` (per-(row, chunk) for ``corder=True``,
    ``indptr`` for ``corder=False``) plus the per-row counts."""
    if corder:
        chunk_counts = count_out.astype(jnp.int32)           # (n_rows, n_chunks)
        row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32)
    else:
        row_counts = count_out.astype(jnp.int32)             # (n_rows,)
    indptr = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
    )
    if corder:
        offsets = indptr[:-1, None] + jnp.cumsum(chunk_counts, axis=1, dtype=jnp.int32) - chunk_counts
    else:
        offsets = indptr
    return indptr, offsets, row_counts


@requires_csr_backend
class Test_Scalar_To_CSR:
    @pytest.mark.parametrize('implementation', CSR_IMPLEMENTATIONS)
    @pytest.mark.parametrize('matrix_mode', MATRIX_MODES)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (64, 33)])
    def test_to_csr_roundtrip(self, implementation, matrix_mode, corder, shape):
        # Converting directly to CSR reproduces the mode-specific dense matrix; this
        # exercises count/fill PRNG alignment end-to-end.
        csr = jits_to_csr(
            1.5, 0.2, 123,
            shape=shape, corder=corder, matrix_mode=matrix_mode, backend=implementation,
        )
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        dense = _reference_dense(1.5, 0.2, 123, shape, corder, matrix_mode, implementation)
        assert allclose(csr.todense(), dense)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('implementation', CSR_IMPLEMENTATIONS)
    @pytest.mark.parametrize('matrix_mode', MATRIX_MODES)
    @pytest.mark.parametrize('corder', [True, False])
    def test_to_csr_agrees_with_matrix_view(self, implementation, matrix_mode, corder):
        # jits_to_csr(..., matrix_mode=mode) reproduces exactly the matrix that the
        # matrix class exposes via ``mat.mv`` / ``mat.mm``.
        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5, 0.2, 123), shape=shape, corder=corder, backend=implementation)
        csr = jits_to_csr(
            mat.weight, mat.prob, mat.seed,
            shape=shape, corder=corder, matrix_mode=matrix_mode, backend=implementation,
        )
        assert allclose(csr.todense(), getattr(mat, matrix_mode).todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    def test_mv_and_mm_differ(self):
        # The mv (32-lane) and mm (4-thread AW-T4) kernels draw *different* matrices,
        # which is exactly why materialization must pick a mode.
        shape = (64, 48)
        impl = CSR_IMPLEMENTATIONS[0]
        csr_mv = jits_to_csr(1.5, 0.2, 123, shape=shape, corder=True, matrix_mode='mv', backend=impl)
        csr_mm = jits_to_csr(1.5, 0.2, 123, shape=shape, corder=True, matrix_mode='mm', backend=impl)
        assert not np.allclose(np.asarray(csr_mv.todense()), np.asarray(csr_mm.todense()))

    @pytest.mark.parametrize('implementation', CSR_IMPLEMENTATIONS)
    @pytest.mark.parametrize('matrix_mode', MATRIX_MODES)
    @pytest.mark.parametrize('corder', [True, False])
    def test_count_matches_row_nnz(self, implementation, matrix_mode, corder):
        # The count primitive, summed to per-row totals, equals the number of
        # non-zeros of the dense matrix it mirrors.
        shape = (20, 30)
        clen = _initialize_conn_length(0.2)
        w = jnp.atleast_1d(jnp.asarray(1.5, dtype=jnp.float32))
        seed = jnp.asarray([123], dtype=jnp.int32)
        count_out = jits_csr_count_p_call(
            w, clen, seed,
            shape=shape, corder=corder, matrix_mode=matrix_mode, backend=implementation,
        )[0]
        _, _, row_counts = _counts_to_offsets(count_out, corder, shape[0])
        dense = _reference_dense(1.5, 0.2, 123, shape, corder, matrix_mode, implementation)
        expected = (np.asarray(dense) != 0).sum(axis=1)
        assert np.array_equal(np.asarray(row_counts), expected)

    @pytest.mark.parametrize('implementation', CSR_IMPLEMENTATIONS)
    @pytest.mark.parametrize('matrix_mode', MATRIX_MODES)
    @pytest.mark.parametrize('corder', [True, False])
    def test_fill_given_offsets(self, implementation, matrix_mode, corder):
        # Given the count-derived offsets, the fill primitive writes a structure
        # whose densification matches the dense matrix.
        shape = (20, 30)
        clen = _initialize_conn_length(0.2)
        w = jnp.atleast_1d(jnp.asarray(1.5, dtype=jnp.float32))
        seed = jnp.asarray([123], dtype=jnp.int32)
        count_out = jits_csr_count_p_call(
            w, clen, seed,
            shape=shape, corder=corder, matrix_mode=matrix_mode, backend=implementation,
        )[0]
        indptr, offsets, _ = _counts_to_offsets(count_out, corder, shape[0])
        nnz = int(indptr[-1])
        indices, data = jits_csr_fill_p_call(
            w, clen, seed, offsets, nnz,
            shape=shape, corder=corder, matrix_mode=matrix_mode, backend=implementation,
        )
        csr = brainevent.CSR((data, indices, indptr), shape=shape)
        dense = _reference_dense(1.5, 0.2, 123, shape, corder, matrix_mode, implementation)
        assert allclose(csr.todense(), dense)
        jax.block_until_ready((indices, data))

    @pytest.mark.parametrize('matrix_mode', MATRIX_MODES)
    def test_to_csr_prob_zero_empty(self, matrix_mode):
        shape = (20, 30)
        csr = jits_to_csr(1.5, 0.0, 123, shape=shape, corder=True, matrix_mode=matrix_mode, backend=None)
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        assert np.asarray(csr.indices).shape == (0,)
        assert np.asarray(csr.data).shape == (0,)
        assert np.all(np.asarray(csr.indptr) == 0)

    def test_to_csr_requires_matrix_mode(self):
        # mv and mm draw different matrices, so the CSR is only defined once a mode
        # is chosen; the bare call must raise.
        with pytest.raises(TypeError):
            jits_to_csr(1.5, 0.2, 123, shape=(20, 30), corder=True)

    @requires_csr_backend
    @pytest.mark.parametrize('matrix_mode', MATRIX_MODES)
    def test_to_csr_units(self, matrix_mode):
        shape = (20, 30)
        weight = 1.5 * u.mV
        csr = jits_to_csr(
            weight, 0.2, 123,
            shape=shape, corder=True, matrix_mode=matrix_mode, backend=None,
        )
        dense = _reference_dense(weight, 0.2, 123, shape, True, matrix_mode, None)
        assert u.get_unit(csr.data) == u.get_unit(dense)
        assert u.math.allclose(csr.todense(), dense)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @requires_csr_backend
    @pytest.mark.parametrize('matrix_mode', MATRIX_MODES)
    @pytest.mark.parametrize('corder', [True, False])
    def test_to_csr_structure_valid(self, matrix_mode, corder):
        # The materialized CSR is a canonical, column-sorted structure.
        shape = (20, 30)
        csr = jits_to_csr(1.5, 0.2, 123, shape=shape, corder=corder, matrix_mode=matrix_mode, backend=None)
        indptr = np.asarray(csr.indptr)
        indices = np.asarray(csr.indices)
        assert indptr.shape == (shape[0] + 1,)
        assert indptr[0] == 0
        assert indptr[-1] == indices.shape[0]
        assert np.all(np.diff(indptr) >= 0)
        assert np.all((indices >= 0) & (indices < shape[1]))
        for r in range(shape[0]):
            seg = indices[indptr[r]:indptr[r + 1]]
            assert np.all(np.diff(seg) > 0)      # strictly increasing: sorted, no dups
        assert allclose(csr.data, jnp.full_like(csr.data, 1.5))
