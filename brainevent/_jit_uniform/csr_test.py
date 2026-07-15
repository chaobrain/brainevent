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
from brainevent._jit_uniform import csr as jitu_csr
from brainevent._jit_uniform.csr import (
    jitu_to_csr,
    jitu_csr_count_p_call,
    jitu_csr_fill_p_call,
)
from brainevent._test_util import allclose, requires_gpu


def test_jitu_to_csr_rejects_invalid_matrix_mode_before_empty_return():
    with pytest.raises(ValueError, match="matrix_mode"):
        jitu_to_csr(0.1, 0.5, 0.0, 42, shape=(2, 3), corder=True, matrix_mode="bad")


def test_jitu_to_csr_default_matrix_mode_is_mv_for_empty_matrix():
    with pytest.warns(UserWarning, match="corder.*ignored"):
        implicit = jitu_to_csr(0.1, 0.5, 0.0, 42, shape=(2, 3), corder=True)
    with pytest.warns(UserWarning, match="corder.*ignored"):
        explicit = jitu_to_csr(
            0.1,
            0.5,
            0.0,
            42,
            shape=(2, 3),
            corder=True,
            matrix_mode="mv",
        )

    assert isinstance(implicit, brainevent.CSR)
    assert implicit.shape == explicit.shape == (2, 3)
    assert np.array_equal(np.asarray(implicit.indptr), np.asarray(explicit.indptr))
    assert np.array_equal(np.asarray(implicit.indices), np.asarray(explicit.indices))
    assert np.array_equal(np.asarray(implicit.data), np.asarray(explicit.data))


def test_jitu_to_csr_warns_that_corder_is_ignored():
    with pytest.warns(UserWarning, match="corder.*ignored"):
        csr = jitu_to_csr(0.1, 0.5, 0.0, 42, shape=(2, 3), corder=False)
    assert isinstance(csr, brainevent.CSR)


@pytest.mark.parametrize(
    ("matrix_mode", "count_symbol", "fill_symbol"),
    [
        ("mv", "jit_uniform_csr.count_chunks_f32", "jit_uniform_csr.fill_f32"),
        (
            "mm",
            "jit_uniform_csr.count_chunks_mm_aw_t4_f32",
            "jit_uniform_csr.fill_mm_aw_t4_f32",
        ),
    ],
)
def test_cuda_generators_select_light_matrix_mode_symbols(
    monkeypatch,
    matrix_mode,
    count_symbol,
    fill_symbol,
):
    calls = []

    def fake_load_cuda_file(path, name):
        calls.append(("load", str(path), name))

    def fake_ffi_call(name, outs):
        calls.append(("ffi", name, outs))

        def invoke(*args, **kwargs):
            calls.append(("invoke", name, kwargs))
            return tuple(jnp.zeros(out.shape, out.dtype) for out in outs)

        return invoke

    monkeypatch.setattr(jitu_csr, "load_cuda_file", fake_load_cuda_file)
    monkeypatch.setattr(jitu_csr.jax.ffi, "ffi_call", fake_ffi_call)

    shape = (5, 7)
    w_info = jax.ShapeDtypeStruct((1,), jnp.float32)
    i_info = jax.ShapeDtypeStruct((1,), jnp.int32)
    chunk_offsets_info = jax.ShapeDtypeStruct((5, 2), jnp.int32)

    count_kernel = jitu_csr._jitu_csr_count_cuda_kernel(
        True,
        shape,
        chunk_size=4,
        target_chunks=4,
        matrix_mode=matrix_mode,
        outs=[jax.ShapeDtypeStruct((5, 2), jnp.int32)],
        w0_info=w_info,
        w1_info=w_info,
        clen_info=i_info,
        seed_info=i_info,
    )
    count_kernel(
        jnp.asarray([0.1], dtype=jnp.float32),
        jnp.asarray([0.5], dtype=jnp.float32),
        jnp.asarray([4], dtype=jnp.int32),
        jnp.asarray([42], dtype=jnp.int32),
    )

    fill_kernel = jitu_csr._jitu_csr_fill_cuda_kernel(
        True,
        shape,
        chunk_size=4,
        target_chunks=4,
        matrix_mode=matrix_mode,
        outs=[
            jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.float32),
        ],
        w0_info=w_info,
        w1_info=w_info,
        clen_info=i_info,
        seed_info=i_info,
        chunk_offsets_info=chunk_offsets_info,
    )
    fill_kernel(
        jnp.asarray([0.1], dtype=jnp.float32),
        jnp.asarray([0.5], dtype=jnp.float32),
        jnp.asarray([4], dtype=jnp.int32),
        jnp.asarray([42], dtype=jnp.int32),
        jnp.zeros((5, 2), dtype=jnp.int32),
    )

    ffi_names = [entry[1] for entry in calls if entry[0] == "ffi"]
    assert ffi_names == [count_symbol, fill_symbol]


class Test_Uniform_To_CSR:
    # These tests compile native kernels and are intentionally kept out of the
    # default test run.
    pytestmark = pytest.mark.slow

    @requires_gpu
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('matrix_mode', ['mv', 'mm'])
    def test_to_csr_roundtrip(self, corder, matrix_mode):
        shape = (20, 30)
        mat = brainevent.JITCUniformR((0.1, 0.5, 0.2, 42), shape=shape, corder=corder)

        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr = jitu_to_csr(
                mat.wlow, mat.whigh, mat.prob, mat.seed,
                shape=mat.shape, corder=mat.corder, backend="cuda_raw", matrix_mode=matrix_mode,
            )
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        assert np.asarray(csr.indptr).shape == (shape[0] + 1,)
        assert np.asarray(csr.indptr)[-1] == np.asarray(csr.indices).shape[0]
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @requires_gpu
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('matrix_mode', ['mv', 'mm'])
    def test_count_matches_row_nnz(self, corder, matrix_mode):
        shape = (20, 30)
        mat = brainevent.JITCUniformR((0.1, 0.5, 0.2, 42), shape=shape, corder=corder)

        clen = _initialize_conn_length(mat.prob)
        w0 = jnp.atleast_1d(jnp.asarray(mat.wlow))
        w1 = jnp.atleast_1d(jnp.asarray(mat.whigh))
        with pytest.warns(UserWarning, match="corder.*ignored"):
            chunk_counts = jitu_csr_count_p_call(
                w0, w1, clen, mat.seed, shape=shape, corder=corder,
                backend="cuda_raw", matrix_mode=matrix_mode,
            )[0]
        row_counts = np.asarray(chunk_counts).sum(axis=1)
        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr = jitu_to_csr(
                mat.wlow, mat.whigh, mat.prob, mat.seed,
                shape=shape, corder=corder, backend="cuda_raw", matrix_mode=matrix_mode,
            )
        assert np.array_equal(row_counts, np.diff(np.asarray(csr.indptr)))

    @requires_gpu
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('matrix_mode', ['mv', 'mm'])
    def test_fill_given_indptr(self, corder, matrix_mode):
        shape = (20, 30)
        mat = brainevent.JITCUniformR((0.1, 0.5, 0.2, 42), shape=shape, corder=corder)

        clen = _initialize_conn_length(mat.prob)
        w0 = jnp.atleast_1d(jnp.asarray(mat.wlow))
        w1 = jnp.atleast_1d(jnp.asarray(mat.whigh))
        with pytest.warns(UserWarning, match="corder.*ignored"):
            chunk_counts = jitu_csr_count_p_call(
                w0, w1, clen, mat.seed, shape=shape, corder=corder,
                backend="cuda_raw", matrix_mode=matrix_mode,
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
        with pytest.warns(UserWarning, match="corder.*ignored"):
            indices, data = jitu_csr_fill_p_call(
                w0, w1, clen, mat.seed, chunk_offsets, nnz, shape=shape, corder=corder,
                backend="cuda_raw", matrix_mode=matrix_mode,
            )
        csr = brainevent.CSR((data, indices, indptr), shape=shape)
        with pytest.warns(UserWarning, match="corder.*ignored"):
            expected = jitu_to_csr(
                mat.wlow, mat.whigh, mat.prob, mat.seed,
                shape=shape, corder=corder, backend="cuda_raw", matrix_mode=matrix_mode,
            )
        assert allclose(csr.todense(), expected.todense())

    def test_to_csr_prob_zero_empty(self):
        shape = (20, 30)
        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr = jitu_to_csr(0.1, 0.5, 0.0, 42, shape=shape, corder=True, backend=None)
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        assert np.asarray(csr.indices).shape == (0,)
        assert np.asarray(csr.data).shape == (0,)
        assert np.all(np.asarray(csr.indptr) == 0)

    def test_to_csr_units(self):
        import brainunit as u

        shape = (20, 30)
        mat = brainevent.JITCUniformR((0.1 * u.mV, 0.5 * u.mV, 0.0, 42), shape=shape)

        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr = jitu_to_csr(
                mat.wlow, mat.whigh, mat.prob, mat.seed,
                shape=mat.shape, corder=mat.corder, backend=mat.backend,
            )
        assert u.get_unit(csr.data) == u.get_unit(u.mV)
        assert np.asarray(u.get_mantissa(csr.data)).shape == (0,)
