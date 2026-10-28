# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from contextlib import contextmanager

import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

import brainevent._csr.float as float_mod
from brainevent._csr.float import csrmv, csrmv_p, csrmm, csrmm_p
from brainevent._csr._test_util import (
    get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix,
    cuda_kwargs, int64_structure, recording_ffi_call, requires_gpu_backend, shape_of,
)
from brainevent._test_util import jax_x64_enabled, with_jax_x64

platform = jax.default_backend()
CSRMV_IMPLEMENTATIONS = tuple(csrmv_p.available_backends(platform))
CSRMM_IMPLEMENTATIONS = tuple(csrmm_p.available_backends(platform))


@contextmanager
def _jax_x64_enabled():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def _make_data(homo_w, shape):
    if homo_w:
        return jnp.asarray(1.5, dtype=jnp.float32)
    return braintools.init.Normal(0.0, 1.0)(shape)


def _vector_csr_api(x, data, indices, indptr, shape, implementation):
    return csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=True,
        backend=implementation,
    )


def _csr_vector_api(x, data, indices, indptr, shape, implementation):
    return csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
    )


def _matrix_csr_api(x, data, indices, indptr, shape, implementation):
    return csrmm(
        data,
        indices,
        indptr,
        x.T,
        shape=shape,
        transpose=True,
        backend=implementation,
    ).T


def _csr_matrix_api(x, data, indices, indptr, shape, implementation):
    return csrmm(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
    )


def _row_ids_from_indptr(indptr):
    indptr = jnp.asarray(indptr)
    counts = jnp.diff(indptr)
    return jnp.repeat(jnp.arange(counts.shape[0], dtype=indptr.dtype), counts)


def test_csrmv_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        vector = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

        got = csrmv(weights, indices, indptr, vector, shape=(2, 3), backend='jax_raw')

        assert jnp.allclose(got, jnp.array([7.0, 18.0], dtype=jnp.float32))


def test_csrmv_rejects_int64_indices_with_int32_indptr():
    with _jax_x64_enabled():
        weights = jnp.ones(2, dtype=jnp.float32)
        indices = jnp.array([0, 1], dtype=jnp.int64)
        indptr = jnp.array([0, 2], dtype=jnp.int32)
        vector = jnp.ones(2, dtype=jnp.float32)

        with pytest.raises(AssertionError, match="Indices must be int32"):
            csrmv(weights, indices, indptr, vector, shape=(1, 2), backend='jax_raw')


def test_csrmm_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        matrix = jnp.array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]], dtype=jnp.float32)

        got = csrmm(weights, indices, indptr, matrix, shape=(2, 3), backend='jax_raw')
        expected = jnp.array([[7.0, 5.5], [18.0, 14.5]], dtype=jnp.float32)

        assert jnp.allclose(got, expected)


def test_csrmm_rejects_unsigned_structure_dtype():
    weights = jnp.ones(2, dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.uint32)
    indptr = jnp.array([0, 2], dtype=jnp.uint32)
    matrix = jnp.ones((2, 1), dtype=jnp.float32)

    with pytest.raises(AssertionError, match="Indices must be int32"):
        csrmm(weights, indices, indptr, matrix, shape=(1, 2), backend='jax_raw')


@pytest.mark.skipif(
    not CSRMV_IMPLEMENTATIONS,
    reason=f'No csrmv implementation on platform={platform}',
)
class TestFloatCSRMV:
    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, implementation, homo_w):
        m, n = 20, 40
        x = brainstate.random.rand(m)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
        y_ref = vector_csr(x, data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, implementation, homo_w):
        m, n = 20, 40
        v = brainstate.random.rand(n)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = _csr_vector_api(v, data, indices, indptr, (m, n), implementation)
        y_ref = csr_vector(v, data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((v, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr_vmap_vector(self, implementation, homo_w):
        n_batch, m, n = 10, 20, 40
        xs = brainstate.random.rand(n_batch, m)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = brainstate.transform.vmap2(
            lambda x: _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
        )(xs)
        y_ref = brainstate.transform.vmap2(lambda x: vector_csr(x, data, indices, indptr, (m, n)))(xs)
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)

        indptr, indices = get_csr(n_in, n_out, 0.2, replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x_arg, w_arg):
            if transpose:
                return _vector_csr_api(x_arg, w_arg, indices, indptr, shape, implementation).sum()
            return _csr_vector_api(x_arg, w_arg, indices, indptr, shape, implementation).sum()

        def f_ref(x_arg, w_arg):
            if transpose:
                return vector_csr(x_arg, w_arg, indices, indptr, shape=shape).sum()
            return csr_vector(x_arg, w_arg, indices, indptr, shape=shape).sum()

        r = jax.grad(f_api, argnums=(0, 1))(x, w)
        r_ref = jax.grad(f_ref, argnums=(0, 1))(x, w)
        assert jnp.allclose(r[0], r_ref[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(r[1], r_ref[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, r, r_ref))

    @pytest.mark.parametrize('implementation', CSRMV_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in if transpose else n_out)

        indptr, indices = get_csr(n_in, n_out, 0.1, replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x_arg, w_arg):
            if transpose:
                return _vector_csr_api(x_arg, w_arg, indices, indptr, shape, implementation)
            return _csr_vector_api(x_arg, w_arg, indices, indptr, shape, implementation)

        def f_ref(x_arg, w_arg):
            if transpose:
                return vector_csr(x_arg, w_arg, indices, indptr, shape=shape)
            return csr_vector(x_arg, w_arg, indices, indptr, shape=shape)

        o1, r1 = jax.jvp(f_api, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        o2, r2 = jax.jvp(f_ref, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, o1, r1, o2, r2))


@pytest.mark.skipif(
    not CSRMM_IMPLEMENTATIONS,
    reason=f'No csrmm implementation on platform={platform}',
)
class TestFloatCSRMM:
    @pytest.mark.parametrize('implementation', CSRMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, implementation, homo_w):
        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = _matrix_csr_api(x, data, indices, indptr, (m, n), implementation)
        y_ref = matrix_csr(x, data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, implementation, homo_w):
        m, n, k = 20, 40, 10
        x = brainstate.random.rand(n, k)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_data(homo_w, indices.shape)

        y = _csr_matrix_api(x, data, indices, indptr, (m, n), implementation)
        y_ref = csr_matrix(x, data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('implementation', CSRMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, implementation, homo_w, replace, transpose):
        m, n, k = 20, 30, 8
        shape = (m, n)
        x = brainstate.random.rand(m, k) if transpose else brainstate.random.rand(n, k)

        indptr, indices = get_csr(m, n, 0.2, replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x_arg, w_arg):
            return csrmm(
                w_arg,
                indices,
                indptr,
                x_arg,
                shape=shape,
                transpose=transpose,
                backend=implementation,
            ).sum()

        def f_ref(x_arg, w_arg):
            if transpose:
                return matrix_csr(x_arg.T, w_arg, indices, indptr, shape).T.sum()
            return csr_matrix(x_arg, w_arg, indices, indptr, shape).sum()

        r = jax.grad(f_api, argnums=(0, 1))(x, w)
        r_ref = jax.grad(f_ref, argnums=(0, 1))(x, w)
        assert jnp.allclose(r[0], r_ref[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(r[1], r_ref[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, r, r_ref))

    @pytest.mark.parametrize('implementation', CSRMM_IMPLEMENTATIONS)
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, implementation, homo_w, replace, transpose):
        m, n, k = 20, 30, 8
        shape = (m, n)
        x = brainstate.random.rand(m, k) if transpose else brainstate.random.rand(n, k)

        indptr, indices = get_csr(m, n, 0.1, replace=replace)
        w = _make_data(homo_w, indices.shape)

        def f_api(x_arg, w_arg):
            return csrmm(
                w_arg,
                indices,
                indptr,
                x_arg,
                shape=shape,
                transpose=transpose,
                backend=implementation,
            )

        def f_ref(x_arg, w_arg):
            if transpose:
                return matrix_csr(x_arg.T, w_arg, indices, indptr, shape).T
            return csr_matrix(x_arg, w_arg, indices, indptr, shape)

        o1, r1 = jax.jvp(f_api, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        o2, r2 = jax.jvp(f_ref, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, o1, r1, o2, r2))


# ---------------------------------------------------------------------------
# int64 ``indptr`` policy on the CUDA path.
#
# ``indices`` stay int32 (the CUDA ABI is int32-only for coordinates) while
# ``indptr`` may widen to int64. The generator tests run without a real GPU by
# stubbing ``load_cuda_file``/``ffi_call``; the ``accepts`` tests need one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'factory,args,kwargs',
    [
        (
            float_mod._csrmv_cuda_kernel,
            (shape_of(jnp.float32), False),
            {'outs': [shape_of(jnp.float32)]},
        ),
        (
            float_mod._csrmm_cuda_kernel,
            (shape_of(jnp.float32), False),
            {'outs': [shape_of(jnp.float32, (1, 1))]},
        ),
    ],
)
def test_cuda_kernel_generators_reject_int64_indices_before_loading_cuda(factory, args, kwargs):
    call_kwargs = cuda_kwargs()
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match="indices with dtype int32"):
        factory(*args, **call_kwargs)


def test_float_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(float_mod, "load_cuda_file", lambda path, name: load_calls.append((path, name)))
    monkeypatch.setattr(float_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    with jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)

        mv_kernel = float_mod._csrmv_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            False,
            **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
        )
        mv_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([1.0, 3.0], dtype=jnp.float64),
        )

        mm_kernel = float_mod._csrmm_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            True,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': [shape_of(jnp.float32, (2, 1))],
            },
        )
        mm_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([[1.0], [3.0]], dtype=jnp.float32),
        )

    assert [name for _, name in load_calls] == ['csr_float_csrmv', 'csr_float_csrmm']
    assert [call[0] for call in ffi_calls] == [
        'csr_float_csrmv.csrmv_nt_auto_f32',
        'csr_float_csrmm.csrmm_t_warp_homo_f32',
    ]


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@with_jax_x64
def test_float_csrmv_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    vector = jnp.array([1.0, 2.0], dtype=jnp.float32) if transpose else jnp.array([1.0, 2.0, 3.0])

    got = csrmv(data, indices, indptr64, vector, shape=(2, 3), transpose=transpose, backend='cuda_raw')
    expected = csrmv(data, indices, indptr32, vector, shape=(2, 3), transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@with_jax_x64
def test_float_csrmm_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    matrix = (
        jnp.array([[1.0, 0.5], [2.0, 1.5]], dtype=jnp.float32)
        if transpose else
        jnp.array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]], dtype=jnp.float32)
    )

    got = csrmm(data, indices, indptr64, matrix, shape=(2, 3), transpose=transpose, backend='cuda_raw')
    expected = csrmm(data, indices, indptr32, matrix, shape=(2, 3), transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype', ['int32', 'bool', 'bfloat16', 'float16'])
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_mixed_operand_dtype_is_promoted(backend, dtype, homo, transpose):
    # Regression: the kernels read the vector / B with the weight dtype, so a
    # non-float32 operand crashed numba (half precision) or was misread on CUDA.
    import numpy as np
    rng = np.random.default_rng(0)
    dense = (rng.random((6, 5)) < 0.5) * (rng.random((6, 5)) + 0.5)
    if homo:
        dense = (dense != 0) * 1.5
    rows, cols = np.nonzero(dense)
    indptr = jnp.asarray(np.searchsorted(rows, np.arange(7)), jnp.int32)
    indices = jnp.asarray(cols, jnp.int32)
    data = jnp.asarray([1.5] if homo else dense[rows, cols], jnp.float32)
    op = dense.T if transpose else dense
    n_in = op.shape[1]
    v = jnp.asarray(rng.random(n_in) * 3, jnp.float32).astype(dtype)
    B = jnp.asarray(rng.random((n_in, 4)) * 3, jnp.float32).astype(dtype)
    tol = 5e-2 if dtype in ('bfloat16', 'float16') else 1e-5
    for fn, x, impls in ((csrmv, v, CSRMV_IMPLEMENTATIONS), (csrmm, B, CSRMM_IMPLEMENTATIONS)):
        if backend not in impls:
            continue
        out = fn(data, indices, indptr, x, shape=(6, 5), transpose=transpose, backend=backend)
        assert out.dtype == jnp.float32
        assert np.allclose(np.asarray(out), op @ np.asarray(x, np.float64), rtol=tol, atol=tol)


@pytest.mark.parametrize('undefined', ['B', 'data'])
def test_csrmm_transpose_rule_symbolic_zero_cotangent(undefined):
    # A symbolic-zero cotangent must map to symbolic zeros, not be fed to a kernel.
    from jax.interpreters import ad
    data = jnp.asarray([1.0, 2.0, 3.0], jnp.float32)
    indices = jnp.asarray([0, 2, 1], jnp.int32)
    indptr = jnp.asarray([0, 2, 3], jnp.int32)
    B = jnp.ones((3, 4), jnp.float32)
    ct = [ad.Zero(jax.core.ShapedArray((2, 4), jnp.float32))]
    args = [data, indices, indptr, B]
    position = 3 if undefined == 'B' else 0
    args[position] = ad.UndefinedPrimal(jax.core.ShapedArray(args[position].shape, jnp.float32))
    out = float_mod._csrmm_transpose_rule(ct, *args, shape=(2, 3), transpose=False, backend=None)
    assert type(out[position]) is ad.Zero
    assert out[position].aval.shape == args[position].aval.shape


# ---------------------------------------------------------------------------
# Repeated (row, col) entries add, also when they outnumber the matrix
# elements (#223). cuSPARSE rejects a CSR descriptor with ``nnz > rows * cols``,
# so the heterogeneous ``cuda_raw`` kernels must not hand such a matrix to it.
# ---------------------------------------------------------------------------

def _repeated_entry_csr(m, k, nse, homo, seed=0):
    """Row-sorted CSR of ``nse`` entries drawn with replacement, and its dense sum."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.integers(0, m, nse))
    cols = rng.integers(0, k, nse)
    w = np.full(nse, 1.5) if homo else rng.standard_normal(nse)
    dense = np.zeros((m, k))
    np.add.at(dense, (rows, cols), w)
    data = jnp.asarray([1.5] if homo else w, jnp.float32)
    indptr = jnp.asarray(np.searchsorted(rows, np.arange(m + 1)), jnp.int32)
    return data, jnp.asarray(cols, jnp.int32), indptr, dense


# (m, k, nse): 2 x 3 storing 8 (the issue's case), 1 x 1 storing 5, a random
# matrix storing 3x its elements, and one whose 20 entries (15 distinct) fit
# in its 30 elements, which cuSPARSE takes and adds.
_REPEATED_CASES = [(2, 3, 8), (1, 1, 5), (7, 5, 105), (6, 5, 20)]


@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('m,k,nse', _REPEATED_CASES)
def test_repeated_entries_add(backend, homo, transpose, m, k, nse):
    import numpy as np
    data, indices, indptr, dense = _repeated_entry_csr(m, k, nse, homo)
    op = dense.T if transpose else dense
    rng = np.random.default_rng(1)
    v = rng.standard_normal(op.shape[1]).astype(np.float32)
    B = rng.standard_normal((op.shape[1], 4)).astype(np.float32)
    for fn, x, impls in ((csrmv, v, CSRMV_IMPLEMENTATIONS), (csrmm, B, CSRMM_IMPLEMENTATIONS)):
        if backend not in impls:
            continue
        out = fn(data, indices, indptr, jnp.asarray(x), shape=(m, k), transpose=transpose, backend=backend)
        assert np.allclose(np.asarray(out), op @ x, rtol=1e-4, atol=1e-4), (fn.__name__, backend)


@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('batch', [None, 3])
def test_repeated_entries_beyond_matrix_size_gradients(backend, transpose, batch):
    # Reverse-mode gradients w.r.t. the weights and the operand, against the
    # same product written densely (repeated entries summed by a scatter-add).
    import numpy as np
    m, k, nse = 4, 3, 30
    data, indices, indptr, _ = _repeated_entry_csr(m, k, nse, homo=False)
    rows = jnp.asarray(np.repeat(np.arange(m), np.diff(np.asarray(indptr))))
    n_in = m if transpose else k
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal((n_in,) if batch is None else (n_in, batch)), jnp.float32)
    fn = csrmv if batch is None else csrmm
    if backend not in (CSRMV_IMPLEMENTATIONS if batch is None else CSRMM_IMPLEMENTATIONS):
        pytest.skip(f'{backend} has no {fn.__name__}')

    def sparse(w, x):
        return fn(w, indices, indptr, x, shape=(m, k), transpose=transpose, backend=backend)

    def dense(w, x):
        mat = jnp.zeros((m, k), w.dtype).at[rows, indices].add(w)
        return (mat.T if transpose else mat) @ x

    out, vjp = jax.vjp(sparse, data, x)
    ref, vjp_ref = jax.vjp(dense, data, x)
    ct = jnp.asarray(rng.standard_normal(out.shape), jnp.float32)
    assert np.allclose(np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-4)
    for got, want in zip(vjp(ct), vjp_ref(ct)):
        assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('generator,jax_kernel,primitive,vector_shape', [
    ('_csrmv_cuda_kernel', '_csrmv_jax_kernel', 'csr_matvec_p', (2,)),
    ('_csrmm_cuda_kernel', '_csrmm_jax_kernel', 'csr_matmat_p', (2, 4)),
])
@pytest.mark.parametrize('nse,uses_cusparse', [(6, True), (7, False)])
def test_cuda_hetero_kernel_skips_cusparse_beyond_matrix_size(
    monkeypatch, generator, jax_kernel, primitive, vector_shape, nse, uses_cusparse,
):
    # Static routing, checked without a GPU: a (2, 3) matrix storing at most
    # 6 entries goes to cuSPARSE; storing more takes the exact pure-JAX kernel.
    sentinel = object()
    monkeypatch.setattr(float_mod, jax_kernel, lambda *a, **kw: sentinel)
    bound = []

    class _Recorder:
        def bind(self, *args, **kwargs):
            bound.append(kwargs)
            return jnp.zeros(1)

    monkeypatch.setattr(float_mod, primitive, _Recorder())
    kernel = getattr(float_mod, generator)(
        shape_of(jnp.float32, (nse,)),
        True,
        outs=[shape_of(jnp.float32, (3,) + vector_shape[1:])],
        shape=(2, 3),
        indices_info=shape_of(jnp.int32, (nse,)),
        indptr_info=shape_of(jnp.int32, (3,)),
        vector_info=shape_of(jnp.float32, vector_shape),
    )
    if uses_cusparse:
        assert kernel is not sentinel
        kernel(jnp.ones(nse), jnp.zeros(nse, jnp.int32), jnp.asarray([0, nse, nse], jnp.int32),
               jnp.ones(vector_shape))
        assert bound == [{'shape': (2, 3), 'transpose': True}]
    else:
        assert kernel is sentinel
