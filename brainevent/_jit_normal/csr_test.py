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
from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._jit_normal.csr import (
    jitn_to_csr,
    jitn_csr_count_p,
    jitn_csr_count_p_call,
    jitn_csr_fill_p_call,
)
from brainevent._jit_normal._test_util import dense_normal_reference
from brainevent._jit_normal.float import jitn
from brainevent._test_util import allclose

# The light-RNG CSR count/fill primitives compile native kernels. Mark the
# whole module slow so the default pytest run skips it; CI runs it via
# ``pytest -m ""``.
pytestmark = pytest.mark.slow

platform = jax.default_backend()
CPU_DEVICE = jax.devices('cpu')[0]
CSR_IMPLEMENTATIONS = tuple(jitn_csr_count_p.available_backends(platform))
CPU_CSR_IMPLEMENTATIONS = tuple(jitn_csr_count_p.available_backends('cpu'))

requires_csr_backend = pytest.mark.skipif(
    not CSR_IMPLEMENTATIONS,
    reason=f'No jitn_csr_count/fill implementation on platform={platform}',
)
requires_cpu_csr_backend = pytest.mark.skipif(
    'numba' not in CPU_CSR_IMPLEMENTATIONS,
    reason='No jitn_csr_count/fill numba backend registered on CPU',
)


def _reference_dense(w_loc, w_scale, prob, seed, shape, corder, backend):
    """The non-transposed dense matrix that ``jitn_to_csr`` materializes."""
    return jitn(
        w_loc, w_scale, prob, seed,
        shape=shape, transpose=False, corder=corder,
        backend=backend,
    )


def _counts_to_offsets(count_out, corder, n_rows):
    """Mirror ``jitn_to_csr`` orchestration."""
    if corder:
        chunk_counts = count_out.astype(jnp.int32)
        row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32)
    else:
        row_counts = count_out.astype(jnp.int32)
    indptr = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
    )
    if corder:
        offsets = indptr[:-1, None] + jnp.cumsum(chunk_counts, axis=1, dtype=jnp.int32) - chunk_counts
    else:
        offsets = indptr
    return indptr, offsets, row_counts


@requires_csr_backend
class Test_Normal_To_CSR:
    @requires_cpu_csr_backend
    @pytest.mark.parametrize('corder', [True, False])
    def test_to_csr_numba_matches_light_rng_reference(self, corder):
        shape = (13, 17)
        w_loc = jnp.asarray(1.5, dtype=jnp.float32)
        w_scale = jnp.asarray(0.15, dtype=jnp.float32)
        with jax.default_device(CPU_DEVICE):
            csr = jitn_to_csr(
                w_loc, w_scale, 0.2, 123,
                shape=shape, corder=corder, backend='numba',
            )
        expected = dense_normal_reference(
            w_loc, w_scale, 0.2, 123,
            shape=shape, corder=corder,
        )
        assert np.allclose(np.asarray(csr.todense()), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('implementation', CSR_IMPLEMENTATIONS)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (64, 33)])
    def test_to_csr_roundtrip(self, implementation, corder, shape):
        csr = jitn_to_csr(
            1.5, 0.2, 0.2, 42,
            shape=shape, corder=corder, backend=implementation,
        )
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        dense = _reference_dense(1.5, 0.2, 0.2, 42, shape, corder, implementation)
        assert allclose(csr.todense(), dense)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('implementation', CSR_IMPLEMENTATIONS)
    @pytest.mark.parametrize('corder', [True, False])
    def test_to_csr_agrees_with_matrix_view(self, implementation, corder):
        shape = (20, 30)
        mat = brainevent.JITCNormalR((1.5, 0.2, 0.2, 42), shape=shape, corder=corder, backend=implementation)
        csr = jitn_to_csr(
            mat.wloc, mat.wscale, mat.prob, mat.seed,
            shape=shape, corder=corder, backend=implementation,
        )
        assert allclose(csr.todense(), mat.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('implementation', CSR_IMPLEMENTATIONS)
    @pytest.mark.parametrize('corder', [True, False])
    def test_count_matches_row_nnz(self, implementation, corder):
        shape = (20, 30)
        clen = _initialize_conn_length(0.2)
        seed = _initialize_seed(42)
        w_loc = jnp.atleast_1d(jnp.asarray(1.5, dtype=jnp.float32))
        w_scale = jnp.atleast_1d(jnp.asarray(0.2, dtype=jnp.float32))
        count_out = jitn_csr_count_p_call(
            w_loc, w_scale, clen, seed,
            shape=shape, corder=corder, backend=implementation,
        )[0]
        _, _, row_counts = _counts_to_offsets(count_out, corder, shape[0])
        dense = _reference_dense(1.5, 0.2, 0.2, 42, shape, corder, implementation)
        expected = (np.asarray(dense) != 0).sum(axis=1)
        assert np.array_equal(np.asarray(row_counts), expected)

    @pytest.mark.parametrize('implementation', CSR_IMPLEMENTATIONS)
    @pytest.mark.parametrize('corder', [True, False])
    def test_fill_given_offsets(self, implementation, corder):
        shape = (20, 30)
        clen = _initialize_conn_length(0.2)
        seed = _initialize_seed(42)
        w_loc = jnp.atleast_1d(jnp.asarray(1.5, dtype=jnp.float32))
        w_scale = jnp.atleast_1d(jnp.asarray(0.2, dtype=jnp.float32))
        count_out = jitn_csr_count_p_call(
            w_loc, w_scale, clen, seed,
            shape=shape, corder=corder, backend=implementation,
        )[0]
        indptr, offsets, _ = _counts_to_offsets(count_out, corder, shape[0])
        nnz = int(indptr[-1])
        indices, data = jitn_csr_fill_p_call(
            w_loc, w_scale, clen, seed, offsets, nnz,
            shape=shape, corder=corder, backend=implementation,
        )
        csr = brainevent.CSR((data, indices, indptr), shape=shape)
        dense = _reference_dense(1.5, 0.2, 0.2, 42, shape, corder, implementation)
        assert allclose(csr.todense(), dense)
        jax.block_until_ready((indices, data))

    def test_to_csr_prob_zero_empty(self):
        shape = (20, 30)
        csr = jitn_to_csr(1.5, 0.2, 0.0, 42, shape=shape, corder=True,
                          backend=None)
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        assert np.asarray(csr.indices).shape == (0,)
        assert np.asarray(csr.data).shape == (0,)
        assert np.all(np.asarray(csr.indptr) == 0)

    def test_to_csr_units(self):
        shape = (20, 30)
        w_loc = 1.5 * u.mV
        w_scale = 0.2 * u.mV
        csr = jitn_to_csr(
            w_loc, w_scale, 0.2, 42,
            shape=shape, corder=True, backend=None,
        )
        dense = _reference_dense(w_loc, w_scale, 0.2, 42, shape, True, None)
        assert u.get_unit(csr.data) == u.get_unit(dense)
        assert u.math.allclose(csr.todense(), dense)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('corder', [True, False])
    def test_to_csr_structure_valid(self, corder):
        shape = (20, 30)
        csr = jitn_to_csr(1.5, 0.2, 0.2, 42, shape=shape, corder=corder,
                          backend=None)
        indptr = np.asarray(csr.indptr)
        indices = np.asarray(csr.indices)
        assert indptr.shape == (shape[0] + 1,)
        assert indptr[0] == 0
        assert indptr[-1] == indices.shape[0]
        assert np.all(np.diff(indptr) >= 0)
        assert np.all((indices >= 0) & (indices < shape[1]))
        for r in range(shape[0]):
            seg = indices[indptr[r]:indptr[r + 1]]
            assert np.all(np.diff(seg) > 0)


# ---- Public interface: back to the v0.1.2 parameter list, exactly ----
# ``matrix_mode`` never ships, and ``chunk_size`` / ``target_chunks`` are gone
# too: the chunk width is a fixed property of the connectivity, derived from the
# walked dimension, not something a caller may vary.

def test_to_csr_signature_matches_0_1_2():
    import inspect
    assert tuple(inspect.signature(jitn_to_csr).parameters) == tuple(
        p.strip() for p in 'w_loc, w_scale'.split(',')
    ) + ('prob', 'seed', 'shape', 'corder', 'backend')
