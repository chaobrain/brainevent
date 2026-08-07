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


import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

from brainevent._csr.dt2t import (
    csrmv_dt2t, cscmv_dt2t, csrmv_dt2t_p,
    csrmm_dt2t, cscmm_dt2t, csrmm_dt2t_p,
)
import brainevent._csr.dt2t as dt2t_mod
from brainevent._csr._test_util import (
    get_csr,
    cuda_kwargs, int64_structure, recording_ffi_call, requires_gpu_backend, shape_of,
)
from brainevent._test_util import jax_x64_enabled

platform = jax.default_backend()
CSRMV_dt2t_IMPLEMENTATIONS = tuple(csrmv_dt2t_p.available_backends(platform))
CSRMM_dt2t_IMPLEMENTATIONS = tuple(csrmm_dt2t_p.available_backends(platform))


def _row_ids_from_indptr(indptr):
    indptr = jnp.asarray(indptr)
    counts = jnp.diff(indptr)
    return jnp.repeat(jnp.arange(counts.shape[0], dtype=indptr.dtype), counts)


@pytest.mark.skipif(
    not CSRMV_dt2t_IMPLEMENTATIONS,
    reason=f'No csrmv_dt2t implementation on platform={platform}',
)
class TestCSRMVdt2t:
    @pytest.mark.parametrize('implementation', CSRMV_dt2t_IMPLEMENTATIONS)
    @pytest.mark.parametrize('shape', [(100, 200), (200, 400)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr(self, implementation, shape, transpose):
        m, n = shape
        indptr, indices = get_csr(m, n, 0.5)

        data = braintools.init.Normal(0.0, 1.0)(indices.shape)
        y = brainstate.random.rand(n) if transpose else brainstate.random.rand(m)

        result = csrmv_dt2t(y, data, indices, indptr, shape=(m, n), transpose=transpose, backend=implementation)

        if transpose:
            expected = data * y[indices]
        else:
            row_ids = _row_ids_from_indptr(indptr)
            expected = data * y[row_ids]

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((data, y, indptr, indices, result, expected))


@pytest.mark.skipif(
    not CSRMV_dt2t_IMPLEMENTATIONS,
    reason=f'No csrmv_dt2t implementation on platform={platform}',
)
class TestCSCMVdt2t:
    @pytest.mark.parametrize('implementation', CSRMV_dt2t_IMPLEMENTATIONS)
    @pytest.mark.parametrize('shape', [(100, 200), (200, 400)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csc(self, implementation, shape, transpose):
        m, k = shape
        # CSC structure of an (m, k) matrix: ``indptr`` partitions the nse
        # entries over the k columns, and ``indices`` holds the row ids in
        # ``[0, m)``.  ``get_csr`` produces exactly this layout when its
        # ``(n_pre, n_post)`` roles are swapped to ``(k, m)``.
        indptr, indices = get_csr(k, m, 0.5)

        data = braintools.init.Normal(0.0, 1.0)(indices.shape)
        y = brainstate.random.rand(k) if transpose else brainstate.random.rand(m)

        result = cscmv_dt2t(y, data, indices, indptr, shape=(m, k), transpose=transpose, backend=implementation)

        if transpose:
            # index by CSC column id (derived from indptr)
            col_ids = _row_ids_from_indptr(indptr)
            expected = data * y[col_ids]
        else:
            # index by CSC row id (stored directly in indices)
            expected = data * y[indices]

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((data, y, indptr, indices, result, expected))

    @pytest.mark.parametrize('implementation', CSRMV_dt2t_IMPLEMENTATIONS)
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csc_matches_transposed_csr(self, implementation, transpose):
        # cscmv_dt2t(shape=(m, k), transpose=t) must equal
        # csrmv_dt2t(shape=(k, m), transpose=not t) on the same arrays.
        m, k = 120, 80
        indptr, indices = get_csr(k, m, 0.5)

        data = braintools.init.Normal(0.0, 1.0)(indices.shape)
        y = brainstate.random.rand(k) if transpose else brainstate.random.rand(m)

        csc_result = cscmv_dt2t(y, data, indices, indptr, shape=(m, k), transpose=transpose, backend=implementation)
        csr_result = csrmv_dt2t(y, data, indices, indptr, shape=(k, m), transpose=not transpose, backend=implementation)

        assert jnp.allclose(csc_result, csr_result, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((data, y, indptr, indices, csc_result, csr_result))


def test_mm_module_and_toplevel_are_same_objects():
    # The dedicated module is the single source of truth; the top-level
    # exports must be the very same objects.
    import brainevent
    assert brainevent.csrmm_dt2t is csrmm_dt2t
    assert brainevent.cscmm_dt2t is cscmm_dt2t
    assert brainevent.csrmm_dt2t_p is csrmm_dt2t_p


@pytest.mark.skipif(
    not CSRMM_dt2t_IMPLEMENTATIONS,
    reason=f'No csrmm_dt2t implementation on platform={platform}',
)
class TestCSRMMdt2t:
    @pytest.mark.parametrize('implementation', CSRMM_dt2t_IMPLEMENTATIONS)
    @pytest.mark.parametrize('shape', [(100, 200), (200, 400)])
    @pytest.mark.parametrize('n_batch', [1, 8])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr(self, implementation, shape, n_batch, transpose):
        m, k = shape
        indptr, indices = get_csr(m, k, 0.5)

        data = braintools.init.Normal(0.0, 1.0)((n_batch, indices.shape[0]))
        y = brainstate.random.rand(n_batch, k if transpose else m)

        result = csrmm_dt2t(y, data, indices, indptr, shape=(m, k), transpose=transpose, backend=implementation)

        if transpose:
            expected = data * y[:, indices]
        else:
            row_ids = _row_ids_from_indptr(indptr)
            expected = data * y[:, row_ids]

        assert result.shape == (n_batch, indices.shape[0])
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((data, y, indptr, indices, result, expected))

    @pytest.mark.parametrize('implementation', CSRMM_dt2t_IMPLEMENTATIONS)
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_matches_stacked_mv(self, implementation, transpose):
        # The mm variant must equal the mv variant applied per batch row.
        m, k, n_batch = 100, 200, 4
        indptr, indices = get_csr(m, k, 0.5)

        data = braintools.init.Normal(0.0, 1.0)((n_batch, indices.shape[0]))
        y = brainstate.random.rand(n_batch, k if transpose else m)

        mm_result = csrmm_dt2t(y, data, indices, indptr, shape=(m, k), transpose=transpose, backend=implementation)
        mv_result = jnp.stack([
            csrmv_dt2t(y[b], data[b], indices, indptr, shape=(m, k), transpose=transpose, backend=implementation)
            for b in range(n_batch)
        ])

        assert jnp.allclose(mm_result, mv_result, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((data, y, indptr, indices, mm_result, mv_result))

    @pytest.mark.parametrize('implementation', CSRMM_dt2t_IMPLEMENTATIONS)
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_jvp(self, implementation, transpose):
        m, k, n_batch = 100, 200, 4
        indptr, indices = get_csr(m, k, 0.5)

        data = braintools.init.Normal(0.0, 1.0)((n_batch, indices.shape[0]))
        y = brainstate.random.rand(n_batch, k if transpose else m)
        d_data = jnp.ones_like(data)
        d_y = jnp.ones_like(y)

        f = lambda y_, w_: csrmm_dt2t(
            y_, w_, indices, indptr, shape=(m, k), transpose=transpose, backend=implementation
        )
        out, out_dot = jax.jvp(f, (y, data), (d_y, d_data))

        if transpose:
            yv, dyv = y[:, indices], d_y[:, indices]
        else:
            row_ids = _row_ids_from_indptr(indptr)
            yv, dyv = y[:, row_ids], d_y[:, row_ids]
        expected = data * yv
        expected_dot = d_data * yv + data * dyv

        assert jnp.allclose(out, expected, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(out_dot, expected_dot, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((data, y, indptr, indices, out, out_dot))

    @pytest.mark.parametrize('implementation', CSRMM_dt2t_IMPLEMENTATIONS)
    def test_csr_units(self, implementation):
        # Like ``csrmv_dt2t``, the wrapper keeps the weight unit and drops
        # the unit of ``y``.
        import brainunit as u
        m, k, n_batch = 100, 200, 4
        indptr, indices = get_csr(m, k, 0.5)

        data = braintools.init.Normal(0.0, 1.0)((n_batch, indices.shape[0]))
        y = brainstate.random.rand(n_batch, m)

        result = csrmm_dt2t(
            y * u.mV, data * u.siemens, indices, indptr,
            shape=(m, k), transpose=False, backend=implementation,
        )

        row_ids = _row_ids_from_indptr(indptr)
        expected = (data * y[:, row_ids]) * u.siemens
        assert u.math.allclose(result, expected, rtol=1e-3, atol=1e-3 * u.siemens)


@pytest.mark.skipif(
    not CSRMM_dt2t_IMPLEMENTATIONS,
    reason=f'No csrmm_dt2t implementation on platform={platform}',
)
class TestCSCMMdt2t:
    @pytest.mark.parametrize('implementation', CSRMM_dt2t_IMPLEMENTATIONS)
    @pytest.mark.parametrize('shape', [(100, 200), (200, 400)])
    @pytest.mark.parametrize('n_batch', [1, 8])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csc(self, implementation, shape, n_batch, transpose):
        m, k = shape
        # CSC structure of an (m, k) matrix: ``indptr`` partitions the nse
        # entries over the k columns, and ``indices`` holds the row ids in
        # ``[0, m)``.  ``get_csr`` produces exactly this layout when its
        # ``(n_pre, n_post)`` roles are swapped to ``(k, m)``.
        indptr, indices = get_csr(k, m, 0.5)

        data = braintools.init.Normal(0.0, 1.0)((n_batch, indices.shape[0]))
        y = brainstate.random.rand(n_batch, k if transpose else m)

        result = cscmm_dt2t(y, data, indices, indptr, shape=(m, k), transpose=transpose, backend=implementation)

        if transpose:
            # index by CSC column id (derived from indptr)
            col_ids = _row_ids_from_indptr(indptr)
            expected = data * y[:, col_ids]
        else:
            # index by CSC row id (stored directly in indices)
            expected = data * y[:, indices]

        assert result.shape == (n_batch, indices.shape[0])
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((data, y, indptr, indices, result, expected))

    @pytest.mark.parametrize('implementation', CSRMM_dt2t_IMPLEMENTATIONS)
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csc_matches_transposed_csr(self, implementation, transpose):
        # cscmm_dt2t(shape=(m, k), transpose=t) must equal
        # csrmm_dt2t(shape=(k, m), transpose=not t) on the same arrays.
        m, k, n_batch = 120, 80, 4
        indptr, indices = get_csr(k, m, 0.5)

        data = braintools.init.Normal(0.0, 1.0)((n_batch, indices.shape[0]))
        y = brainstate.random.rand(n_batch, k if transpose else m)

        csc_result = cscmm_dt2t(y, data, indices, indptr, shape=(m, k), transpose=transpose, backend=implementation)
        csr_result = csrmm_dt2t(y, data, indices, indptr, shape=(k, m), transpose=not transpose, backend=implementation)

        assert jnp.allclose(csc_result, csr_result, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((data, y, indptr, indices, csc_result, csr_result))


# ---------------------------------------------------------------------------
# int64 ``indptr`` policy on the CUDA path.
#
# ``indices`` stay int32 (the CUDA ABI is int32-only for coordinates) while
# ``indptr`` may widen to int64. The generator tests run without a real GPU by
# stubbing ``load_cuda_file``/``ffi_call``; the ``accepts`` test needs one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'factory,args,kwargs',
    [
        (
            dt2t_mod._csrmv_dt2t_cuda_kernel,
            (False, shape_of(jnp.float32)),
            {'outs': [shape_of(jnp.float32)]},
        ),
    ],
)
def test_cuda_kernel_generators_reject_int64_indices_before_loading_cuda(factory, args, kwargs):
    call_kwargs = cuda_kwargs()
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match="indices with dtype int32"):
        factory(*args, **call_kwargs)


def test_dt2t_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(dt2t_mod, "load_cuda_file", lambda path, name: load_calls.append((path, name)))
    monkeypatch.setattr(dt2t_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    with jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)

        dt2t_kernel = dt2t_mod._csrmv_dt2t_cuda_kernel(
            False,
            shape_of(jnp.float32, (2,)),
            **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
        )
        dt2t_kernel(jnp.array([1.0]), jnp.array([2.0, 3.0]), indices, indptr)

    assert [name for _, name in load_calls] == ['csrmv_dt2t']
    assert [call[0] for call in ffi_calls] == ['csrmv_dt2t.csrmv_dt2t_nt_auto_f32']


@requires_gpu_backend
def test_dt2t_cuda_accepts_int64_indptr():
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    y = jnp.array([1.0, 2.0], dtype=jnp.float32)

    got = csrmv_dt2t(y, weights, indices, indptr64, shape=(2, 3), backend='cuda_raw')
    expected = csrmv_dt2t(y, weights, indices, indptr32, shape=(2, 3), backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)
