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

import inspect

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._data import _initialize_conn_length
from brainevent._jit_scalar.csr import (
    jits_csr_count_p_call,
    jits_csr_fill_p_call,
    jits_to_csr,
)
from brainevent._jit_scalar.dt2t import (
    jitsmv_dt2t,
    jitsmv_dt2t_p,
    jitsmv_dt2t_p_call,
)
from brainevent._test_util import allclose

# ``jitsmv_dt2t`` composes over the light-RNG CSR kernels, which have both CUDA and
# ``numba`` backends. Mark the module ``slow`` so the default ``pytest`` run skips it
# (numba JIT compilation is slow); CI runs it via ``pytest -m ""``.
pytestmark = pytest.mark.slow

platform = jax.default_backend()
JITS_dt2t_IMPLEMENTATIONS = tuple(jitsmv_dt2t_p.available_backends(platform))

requires_dt2t_backend = pytest.mark.skipif(
    not JITS_dt2t_IMPLEMENTATIONS,
    reason=f'No jitsmv_dt2t implementation on platform={platform}',
)


def _csr_yw_reference(csr, y, transpose):
    """``weight * y[col]`` (transpose) or ``weight * y[row]`` (non-transpose), in the
    materialized CSR's flat order."""
    row_ids = jnp.repeat(
        jnp.arange(csr.shape[0], dtype=csr.indptr.dtype),
        jnp.diff(csr.indptr),
        total_repeat_length=csr.data.shape[0],
    )
    return csr.data * (y[csr.indices] if transpose else y[row_ids])


# --------------------------------------------------------------------------- #
#  Public ``jitsmv_dt2t`` — always materializes the mv matrix
# --------------------------------------------------------------------------- #

@requires_dt2t_backend
@pytest.mark.parametrize('implementation', JITS_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (64, 33)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitsmv_dt2t_matches_csr_reference(implementation, shape, corder, transpose):
    y_size = shape[1] if transpose else shape[0]
    y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)

    out = jitsmv_dt2t(
        1.5, 0.2, y, 123,
        shape=shape, transpose=transpose, corder=corder, backend=implementation,
    )
    # dt2t materializes the mv matrix, so the reference is the mv CSR.
    csr = jits_to_csr(1.5, 0.2, 123, shape=shape, corder=corder, backend=implementation)
    expected = _csr_yw_reference(csr, y, transpose)

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


@requires_dt2t_backend
@pytest.mark.parametrize('implementation', JITS_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype,tol', [
    (jnp.float32, 1e-4),
    (jnp.float16, 1e-2),
    (jnp.bfloat16, 5e-2),
])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitsmv_dt2t_dtypes(implementation, dtype, tol, transpose):
    # Each weight dtype (the light kernels have _f32/_f16/_bf16/_f64 suffixes) flows
    # through to the generated values. (float64 needs jax_enable_x64 and is skipped.)
    shape = (20, 30)
    y_size = shape[1] if transpose else shape[0]
    y = jnp.linspace(-1.0, 2.0, y_size, dtype=dtype)
    w = jnp.asarray(1.5, dtype=dtype)

    out = jitsmv_dt2t(
        w, 0.2, y, 123,
        shape=shape, transpose=transpose, corder=True, backend=implementation,
    )
    assert out.dtype == dtype
    csr = jits_to_csr(w, 0.2, 123, shape=shape, corder=True, backend=implementation)
    expected = _csr_yw_reference(csr, y, transpose)
    assert allclose(out, expected, rtol=tol, atol=tol)
    jax.block_until_ready((out, expected))


def test_jitsmv_dt2t_prob_zero_empty():
    out = jitsmv_dt2t(
        1.5, 0.0, jnp.ones(20, dtype=jnp.float32), 123,
        shape=(20, 30), corder=True,
    )
    assert np.asarray(out).shape == (0,)


@requires_dt2t_backend
@pytest.mark.parametrize('implementation', JITS_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitsmv_dt2t_is_repeatable(implementation, corder, transpose):
    # Materialization is deterministic (canonical column-sorted CSR), so repeated
    # calls are byte-for-byte identical for both corder values.
    shape = (20, 30)
    y_size = shape[1] if transpose else shape[0]
    y = jnp.linspace(0.2, 1.7, y_size, dtype=jnp.float32)

    out1 = jitsmv_dt2t(1.5, 0.2, y, 123, shape=shape, transpose=transpose, corder=corder, backend=implementation)
    out2 = jitsmv_dt2t(1.5, 0.2, y, 123, shape=shape, transpose=transpose, corder=corder, backend=implementation)
    assert np.array_equal(np.asarray(out1), np.asarray(out2))


@requires_dt2t_backend
def test_jitsmv_dt2t_units_are_weight_times_y():
    out = jitsmv_dt2t(
        1.5 * u.siemens, 0.2, jnp.ones(20, dtype=jnp.float32) * u.mV, 123,
        shape=(20, 30), corder=True,
    )
    assert u.get_unit(out) == u.mA


def test_jitsmv_dt2t_exports_from_package():
    assert brainevent.jitsmv_dt2t is jitsmv_dt2t


# --------------------------------------------------------------------------- #
#  Fused fill primitive (corder=True / mv-notrans structure), CUDA + numba
# --------------------------------------------------------------------------- #

@requires_dt2t_backend
@pytest.mark.parametrize('implementation', JITS_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jitsmv_dt2t_fused_fill_generates_y_times_weight(implementation, transpose):
    # The fused primitive replays the mv-notrans (corder=True) walk and writes
    # ``w * y[row]`` (notrans) or ``w * y[col]`` (trans) in that walk's flat order,
    # which matches the corder=True fill order entry-for-entry.
    shape = (20, 30)
    y_size = shape[1] if transpose else shape[0]
    y = jnp.linspace(0.2, 1.7, y_size, dtype=jnp.float32)
    w = jnp.asarray([1.5], dtype=jnp.float32)
    clen = _initialize_conn_length(0.2)
    seed = jnp.asarray([123], dtype=jnp.int32)

    chunk_counts = jits_csr_count_p_call(
        w, clen, seed, shape=shape, corder=True, backend=implementation,
    )[0]
    row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32)
    indptr = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)])
    cc = chunk_counts.astype(jnp.int32)
    chunk_offsets = indptr[:-1, None] + jnp.cumsum(cc, axis=1, dtype=jnp.int32) - cc
    nnz = int(indptr[-1])

    # Raw (unsorted) fill in the same walk order the fused kernel uses.
    indices, weights = jits_csr_fill_p_call(
        w, clen, seed, chunk_offsets, nnz, shape=shape, corder=True, backend=implementation,
    )
    out = jitsmv_dt2t_p_call(
        w, clen, y, seed, chunk_offsets, nnz,
        shape=shape, transpose=transpose, backend=implementation,
    )[0]
    row_ids = jnp.repeat(jnp.arange(shape[0], dtype=jnp.int32), jnp.diff(indptr), total_repeat_length=nnz)
    expected = weights * (y[indices] if transpose else y[row_ids])

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


# --------------------------------------------------------------------------- #
#  Matrix-class ``dt2t`` / ``dt2t_transposed`` integration
# --------------------------------------------------------------------------- #

def test_jits_matrix_dt2t_signatures_align_contracts():
    base_sig = inspect.signature(brainevent.DataRepresentation.dt2t)
    base_sig_t = inspect.signature(brainevent.DataRepresentation.dt2t_transposed)
    for cls in (brainevent.JITCScalarR, brainevent.JITCScalarC):
        sig = inspect.signature(cls.dt2t)
        assert list(sig.parameters) == list(base_sig.parameters)
        assert sig.parameters['y_dim_arr'].annotation == base_sig.parameters['y_dim_arr'].annotation
        assert sig.parameters['w_dim_arr'].annotation == base_sig.parameters['w_dim_arr'].annotation
        assert sig.parameters['w_dim_arr'].default is inspect._empty
        assert sig.return_annotation == base_sig.return_annotation

        sig_t = inspect.signature(cls.dt2t_transposed)
        assert list(sig_t.parameters) == list(base_sig_t.parameters)
        assert sig_t.parameters['y_dim_arr'].annotation == base_sig_t.parameters['y_dim_arr'].annotation
        assert sig_t.parameters['w_dim_arr'].annotation == base_sig_t.parameters['w_dim_arr'].annotation
        assert sig_t.parameters['w_dim_arr'].default is inspect._empty
        assert sig_t.return_annotation == base_sig_t.return_annotation


@requires_dt2t_backend
@pytest.mark.parametrize('implementation', JITS_dt2t_IMPLEMENTATIONS)
def test_jits_matrix_dt2t_requires_w_dim_arr(implementation):
    mat = brainevent.JITCScalarR((1.5, 0.2, 123), shape=(20, 30), backend=implementation)
    y_pre = jnp.linspace(-1.0, 2.0, 20, dtype=jnp.float32)
    y_post = jnp.linspace(-1.0, 2.0, 30, dtype=jnp.float32)
    with pytest.raises(TypeError):
        mat.dt2t(y_pre)
    with pytest.raises(TypeError):
        mat.dt2t_transposed(y_post)


@requires_dt2t_backend
@pytest.mark.parametrize('implementation', JITS_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jits_matrix_dt2t_uses_init_parameters(implementation, cls, corder, transpose):
    shape = (20, 30)
    y_size = shape[1] if transpose else shape[0]
    y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)
    mat = cls((1.5, 0.2, 123), shape=shape, corder=corder, backend=implementation)

    w_dim_arr = jnp.empty(0, dtype=jnp.float32)
    out = mat.dt2t_transposed(y, w_dim_arr) if transpose else mat.dt2t(y, w_dim_arr)
    expected = jitsmv_dt2t(
        1.5, 0.2, y, 123,
        shape=shape, transpose=transpose, corder=corder, backend=implementation,
    )
    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


# ---- Public interface: back to the v0.1.2 parameter list ----

def test_dt2t_signature_matches_0_1_2():
    import inspect
    params = tuple(inspect.signature(jitsmv_dt2t).parameters)
    assert params == tuple(p.strip() for p in 'weight'.split(',')) + (
        'prob', 'y', 'seed', 'shape', 'transpose', 'corder', 'backend',
    )
