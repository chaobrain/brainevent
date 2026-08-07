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
# -*- coding: utf-8 -*-


from contextlib import contextmanager

import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

from brainevent._csr.binary import (
    _binary_csrmv_benchmark_data,
    _binary_csrmm_benchmark_data,
    _binary_csrmm_cusparse_kernel,
    _binary_csrmv_cusparse_kernel,
    _csrmv_batching,
    binary_csrmv,
    binary_csrmv_p,
    binary_csrmv_p_call,
    binary_csrmm,
    binary_csrmm_p,
    binary_csrmm_p_call,
)
import brainevent._csr.binary as binary_mod
from brainevent._csr.main import _make_binary_task_workspace
from brainevent._csr._test_util import (
    get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix,
    cuda_kwargs, int64_structure, recording_ffi_call, requires_gpu_backend,
    shape_of, strip_hybrid_suffix,
)
from brainevent._test_util import jax_x64_enabled

# The backend-sweeping tests below loop over all native backends (incl. ``numba``), which
# compile per test and dominate wall-clock, so each carries ``@pytest.mark.slow`` and the
# default ``pytest`` run skips it; CI runs them via ``pytest -m ""``. The marker is per-test
# rather than module-wide so that cheap tests here (e.g. the backend-name assertions at the
# bottom) stay in the default run.

platform = jax.default_backend()
CSRMV_IMPLEMENTATIONS = tuple(binary_csrmv_p.available_backends(platform))
CSRMM_IMPLEMENTATIONS = tuple(binary_csrmm_p.available_backends(platform))


@contextmanager
def _jax_x64_enabled():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def _require_implementations(implementations, op_name: str):
    if not implementations:
        pytest.skip(f'No {op_name} implementation on platform={platform}')


@pytest.mark.slow
def test_binary_csrmv_jax_raw_accepts_workspace_and_preserves_result():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    vector = jnp.array([True, False])
    workspace = _make_binary_task_workspace(indptr)

    got = binary_csrmv(
        weights,
        indices,
        indptr,
        vector,
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        workspace=workspace,
    )

    assert got.shape == (1,)
    assert got.dtype == weights.dtype
    assert jnp.allclose(got, jnp.array([1.0], dtype=jnp.float32))


@pytest.mark.slow
def test_binary_csrmv_p_call_always_returns_task_outputs():
    weights = jnp.array([1.0], dtype=jnp.float32)
    indices = jnp.array([0], dtype=jnp.int32)
    indptr = jnp.array([0, 1], dtype=jnp.int32)
    vector = jnp.array([True])
    workspace = _make_binary_task_workspace(indptr)

    out, task_begin, task_end, status = binary_csrmv_p_call(
        weights,
        indices,
        indptr,
        vector,
        shape=(1, 1),
        transpose=False,
        backend="jax_raw",
        workspace=workspace,
    )

    assert out.shape == (1,)
    assert task_begin.shape == workspace.task_begin.shape
    assert task_end.shape == workspace.task_end.shape
    assert status.shape == workspace.status.shape


@pytest.mark.slow
def test_binary_csrmm_jax_raw_accepts_workspace_and_preserves_result():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    matrix = jnp.array([[True, False], [False, True]])
    workspace = _make_binary_task_workspace(indptr)

    got = binary_csrmm(
        weights,
        indices,
        indptr,
        matrix,
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        workspace=workspace,
    )

    assert got.shape == (1, 2)
    assert got.dtype == weights.dtype
    assert jnp.allclose(got, jnp.array([[1.0, 2.0]], dtype=jnp.float32))


@pytest.mark.slow
def test_binary_csrmm_p_call_always_returns_task_outputs():
    weights = jnp.array([1.0], dtype=jnp.float32)
    indices = jnp.array([0], dtype=jnp.int32)
    indptr = jnp.array([0, 1], dtype=jnp.int32)
    matrix = jnp.array([[True]])
    workspace = _make_binary_task_workspace(indptr)

    out, task_begin, task_end, status = binary_csrmm_p_call(
        weights,
        indices,
        indptr,
        matrix,
        shape=(1, 1),
        transpose=False,
        backend="jax_raw",
        workspace=workspace,
    )

    assert out.shape == (1, 1)
    assert task_begin.shape == workspace.task_begin.shape
    assert task_end.shape == workspace.task_end.shape
    assert status.shape == workspace.status.shape


@pytest.mark.slow
def test_binary_csrmv_benchmark_data_includes_workspace_for_p_call():
    config = next(_binary_csrmv_benchmark_data(platform='cpu'))

    out, task_begin, task_end, status = binary_csrmv_p_call(
        *config.args,
        **config.kernel_kwargs,
        backend='jax_raw',
    )

    workspace = config.args[-1]
    assert out.shape == (config.kernel_kwargs['shape'][0],)
    assert task_begin.shape == workspace.task_begin.shape
    assert task_end.shape == workspace.task_end.shape
    assert status.shape == workspace.status.shape


@pytest.mark.slow
def test_binary_csrmm_benchmark_data_includes_workspace_for_p_call():
    config = _binary_csrmm_benchmark_data(platform='cpu')[0]

    out, task_begin, task_end, status = binary_csrmm_p_call(
        *config.args,
        **config.kernel_kwargs,
        backend='jax_raw',
    )

    workspace = config.args[-1]
    assert out.shape == (config.kernel_kwargs['shape'][0], config.args[-2].shape[1])
    assert task_begin.shape == workspace.task_begin.shape
    assert task_end.shape == workspace.task_end.shape
    assert status.shape == workspace.status.shape


@pytest.mark.slow
def test_binary_csrmv_batching_vector_axis_0_uses_csrmm_fast_path(monkeypatch):
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    vector = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
    workspace = _make_binary_task_workspace(indptr)
    math_out = jnp.ones((1, 2), dtype=jnp.float32)
    calls = []

    def fake_csrmm_p_call(w, idx, ptr, b, workspace, *, shape, transpose, backend):
        calls.append((w, idx, ptr, b, shape, transpose, backend, workspace))
        return (math_out, workspace.task_begin, workspace.task_end, workspace.status)

    monkeypatch.setattr(
        "brainevent._csr.binary.binary_csrmm_p_call",
        fake_csrmm_p_call,
    )

    out, out_axes = _csrmv_batching(
        (weights, indices, indptr, vector, workspace.task_begin, workspace.task_end, workspace.status),
        (None, None, None, 0, None, None, None),
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
    )

    assert calls
    assert out == (math_out, workspace.task_begin, workspace.task_end, workspace.status)
    assert out_axes == (1, None, None, None)


@pytest.mark.slow
def test_binary_csrmv_batching_vector_axis_1_uses_csrmm_fast_path(monkeypatch):
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    vector = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
    workspace = _make_binary_task_workspace(indptr)
    math_out = jnp.ones((1, 2), dtype=jnp.float32)
    calls = []

    def fake_csrmm_p_call(w, idx, ptr, b, workspace, *, shape, transpose, backend):
        calls.append((w, idx, ptr, b, shape, transpose, backend, workspace))
        return (math_out, workspace.task_begin, workspace.task_end, workspace.status)

    monkeypatch.setattr(
        "brainevent._csr.binary.binary_csrmm_p_call",
        fake_csrmm_p_call,
    )

    out, out_axes = _csrmv_batching(
        (weights, indices, indptr, vector, workspace.task_begin, workspace.task_end, workspace.status),
        (None, None, None, 1, None, None, None),
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
    )

    assert calls
    assert calls[0][3] is vector
    assert out == (math_out, workspace.task_begin, workspace.task_end, workspace.status)
    assert out_axes == (1, None, None, None)


@pytest.mark.slow
def test_binary_csrmv_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        events = jnp.array([True, False, True])
        workspace = _make_binary_task_workspace(indptr)

        got = binary_csrmv(
            weights,
            indices,
            indptr,
            events,
            shape=(2, 3),
            backend='jax_raw',
            workspace=workspace,
        )

        assert jnp.allclose(got, jnp.array([3.0, 4.0], dtype=jnp.float32))


@pytest.mark.slow
def test_binary_csrmv_rejects_int64_indices_with_int32_indptr():
    with _jax_x64_enabled():
        weights = jnp.ones(2, dtype=jnp.float32)
        indices = jnp.array([0, 1], dtype=jnp.int64)
        indptr = jnp.array([0, 2], dtype=jnp.int32)
        events = jnp.ones(2, dtype=bool)
        workspace = _make_binary_task_workspace(indptr)

        with pytest.raises(AssertionError, match="Indices must be int32"):
            binary_csrmv(
                weights,
                indices,
                indptr,
                events,
                shape=(1, 2),
                backend='jax_raw',
                workspace=workspace,
            )


@pytest.mark.slow
def test_binary_csrmm_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        events = jnp.array([[True, False], [False, True], [True, True]])
        workspace = _make_binary_task_workspace(indptr)

        got = binary_csrmm(weights, indices, indptr, events, shape=(2, 3), backend='jax_raw', workspace=workspace)
        expected = jnp.array([[3.0, 2.0], [4.0, 7.0]], dtype=jnp.float32)

        assert jnp.allclose(got, expected)


@pytest.mark.slow
def test_binary_csrmm_rejects_unsigned_structure_dtype():
    weights = jnp.ones(2, dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.uint32)
    indptr = jnp.array([0, 2], dtype=jnp.uint32)
    events = jnp.ones((2, 1), dtype=bool)
    workspace = _make_binary_task_workspace(indptr)

    with pytest.raises(AssertionError, match="Indices must be int32"):
        binary_csrmm(weights, indices, indptr, events, shape=(1, 2), backend='jax_raw', workspace=workspace)


@pytest.mark.slow
def test_binary_csrmv_jax_csr_kernel_homo_bool_casts_indices_to_indptr_dtype():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        workspace = _make_binary_task_workspace(indptr)
        kernel = _binary_csrmv_cusparse_kernel(
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct((3,), jnp.bool_),
            (2, 3),
            False,
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            outs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        )

        got = kernel(
            jnp.array([2.0], dtype=jnp.float32),
            jnp.array([0, 2, 1, 2], dtype=jnp.int32),
            indptr,
            jnp.array([True, False, True]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )[0]

        assert jnp.allclose(got, jnp.array([4.0, 2.0], dtype=jnp.float32))


@pytest.mark.slow
def test_binary_csrmv_jax_csr_kernel_hetero_float_transpose():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        workspace = _make_binary_task_workspace(indptr)
        kernel = _binary_csrmv_cusparse_kernel(
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.float32),
            (2, 3),
            True,
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            outs=[jax.ShapeDtypeStruct((3,), jnp.float32)],
        )

        got = kernel(
            jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
            jnp.array([0, 2, 1, 2], dtype=jnp.int32),
            indptr,
            jnp.array([1.0, -1.0], dtype=jnp.float32),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )[0]

        assert jnp.allclose(got, jnp.array([1.0, 0.0, 2.0], dtype=jnp.float32))


@pytest.mark.slow
def test_binary_csrmm_jax_csr_kernel_homo_bool_casts_indices_to_indptr_dtype():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        workspace = _make_binary_task_workspace(indptr)
        kernel = _binary_csrmm_cusparse_kernel(
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct((3, 2), jnp.bool_),
            (2, 3),
            False,
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            outs=[jax.ShapeDtypeStruct((2, 2), jnp.float32)],
        )

        got = kernel(
            jnp.array([2.0], dtype=jnp.float32),
            jnp.array([0, 2, 1, 2], dtype=jnp.int32),
            indptr,
            jnp.array([[True, False], [False, True], [True, True]]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )[0]

        expected = jnp.array([[4.0, 2.0], [2.0, 4.0]], dtype=jnp.float32)
        assert jnp.allclose(got, expected)


@pytest.mark.slow
def test_binary_csrmm_jax_csr_kernel_hetero_float_transpose():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        workspace = _make_binary_task_workspace(indptr)
        kernel = _binary_csrmm_cusparse_kernel(
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((2, 2), jnp.float32),
            (2, 3),
            True,
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            outs=[jax.ShapeDtypeStruct((3, 2), jnp.float32)],
        )

        got = kernel(
            jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
            jnp.array([0, 2, 1, 2], dtype=jnp.int32),
            indptr,
            jnp.array([[1.0, -1.0], [-1.0, 1.0]], dtype=jnp.float32),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )[0]

        expected = jnp.array([[1.0, 0.0], [0.0, 3.0], [2.0, 4.0]], dtype=jnp.float32)
        assert jnp.allclose(got, expected)


def _vector_csr_api(x, data, indices, indptr, shape, implementation):
    workspace = _make_binary_task_workspace(indptr)
    return binary_csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=True,
        backend=implementation,
        workspace=workspace,
    )


def _csr_vector_api(x, data, indices, indptr, shape, implementation):
    workspace = _make_binary_task_workspace(indptr)
    return binary_csrmv(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
        workspace=workspace,
    )


def _matrix_csr_api(x, data, indices, indptr, shape, implementation):
    # x @ csr: csrmm expects input as [shape[0], cols] for transpose=True.
    workspace = _make_binary_task_workspace(indptr)
    return binary_csrmm(
        data,
        indices,
        indptr,
        x.T,
        shape=shape,
        transpose=True,
        backend=implementation,
        workspace=workspace,
    ).T


def _csr_matrix_api(x, data, indices, indptr, shape, implementation):
    workspace = _make_binary_task_workspace(indptr)
    return binary_csrmm(
        data,
        indices,
        indptr,
        x,
        shape=shape,
        transpose=False,
        backend=implementation,
        workspace=workspace,
    )


@pytest.mark.slow
class TestVectorCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        m, n = 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = vector_csr(x, data, indices, indptr, (m, n))

        for implementation in CSRMV_IMPLEMENTATIONS:
            y = _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((x, indptr, indices, y2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr_vmap_vector(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        with jax.checking_leaks():
            n_batch, m, n = 10, 20, 40
            xs = brainstate.random.rand(n_batch, m) < 0.1
            indptr, indices = get_csr(m, n, 0.1)

            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            y2 = brainstate.transform.vmap2(lambda x: vector_csr(x, data, indices, indptr, (m, n)))(xs)

            for implementation in CSRMV_IMPLEMENTATIONS:
                y = brainstate.transform.vmap2(
                    lambda x: _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
                )(xs)
                assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        m, n = 20, 40
        v = brainstate.random.rand(n) < 0.1
        indptr, indices = get_csr(m, n, 0.2)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = csr_vector(v, data, indices, indptr, (m, n))

        for implementation in CSRMV_IMPLEMENTATIONS:
            y = _csr_vector_api(v, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2, rtol=1e-5, atol=1e-5)

        jax.block_until_ready((v, indptr, indices, y2))

    def _test_vjp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)
        x = (x < 0.6).astype(float)

        indptr, indices = get_csr(n_in, n_out, 0.2, replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, shape, implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, shape, implementation)
            return r.sum()

        r = jax.grad(f_api, argnums=(0, 1))(x, w)

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, w)
        assert jnp.allclose(r[0], r2[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(r[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, homo_w, replace, transpose):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._test_vjp(
                implementation=implementation,
                homo_w=homo_w,
                replace=replace,
                transpose=transpose,
            )

    def _test_jvp(self, implementation, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in if transpose else n_out)
        x = (x < 0.6).astype(float)

        indptr, indices = get_csr(n_in, n_out, 0.1, replace=replace)

        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, shape, implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, shape, implementation)
            return r

        o1, r1 = jax.jvp(f_api, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r

        o2, r2 = jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, o1, r1, o2, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, homo_w, replace, transpose):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._test_jvp(
                implementation=implementation,
                homo_w=homo_w,
                replace=replace,
                transpose=transpose,
            )


@pytest.mark.slow
class TestBatchingVectorCSR:
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        implementation = self._implementation
        if transpose:
            y1 = _vector_csr_api(x, data, indices, indptr, (m, n), implementation)
            y2 = vector_csr(x, data, indices, indptr, (m, n))
        else:
            y1 = _csr_vector_api(x, data, indices, indptr, (m, n), implementation)
            y2 = csr_vector(x, data, indices, indptr, (m, n))
        return jnp.allclose(y1, y2)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda x: self._run(x, data, indices, indptr, m, n))(xs)
            assert jnp.all(res)

        jax.block_until_ready((xs, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda data: self._run(x, data, indices, indptr, m, n))(data)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda ind: self._run(x, data, ind, indptr, m, n))(indices)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, (m, n), implementation)
            return r.sum()

        r1 = jax.grad(f_api, argnums=(0, 1))(x, data)

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_vector(x, w, indices, indptr, shape=(m, n))
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, data)

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_vjp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda x: self._run_vjp(x, data, indices, indptr, m, n))(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_vjp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_vjp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_vjp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_vjp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _vector_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_vector_api(x, w, indices, indptr, (m, n), implementation)
            return r

        r1 = jax.jvp(f_api, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_vector(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_jvp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda x: self._run_jvp(x, data, indices, indptr, m, n))(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_jvp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_jvp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_jvp(self, homo_w):
        _require_implementations(CSRMV_IMPLEMENTATIONS, 'binary_csrmv')

        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMV_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_jvp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))


@pytest.mark.slow
class TestMatrixCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = matrix_csr(x, data, indices, indptr, (m, n))

        for implementation in CSRMM_IMPLEMENTATIONS:
            y = _matrix_csr_api(x, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, y2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        m, n, k = 20, 40, 10
        matrix = brainstate.random.rand(n, k) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        y2 = csr_matrix(matrix, data, indices, indptr, (m, n))

        for implementation in CSRMM_IMPLEMENTATIONS:
            y = _csr_matrix_api(matrix, data, indices, indptr, (m, n), implementation)
            assert jnp.allclose(y, y2)

        jax.block_until_ready((matrix, indptr, indices, y2))


@pytest.mark.slow
class TestBatchingMatrixCSR:
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        implementation = self._implementation
        if transpose:
            y1 = _matrix_csr_api(x, data, indices, indptr, (m, n), implementation)
            y2 = matrix_csr(x, data, indices, indptr, (m, n))
        else:
            y1 = _csr_matrix_api(x, data, indices, indptr, (m, n), implementation)
            y2 = csr_matrix(x, data, indices, indptr, (m, n))
        return jnp.allclose(y1, y2)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_matrix(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        xs = brainstate.random.rand(b, k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda x: self._run(x, data, indices, indptr, m, n))(xs)
            assert jnp.all(res)

        jax.block_until_ready((xs, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda data: self._run(x, data, indices, indptr, m, n))(data)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            res = brainstate.transform.vmap2(lambda ind: self._run(x, data, ind, indptr, m, n))(indices)
            assert jnp.all(res)

        jax.block_until_ready((x, indptr, indices))

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _matrix_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_matrix_api(x, w, indices, indptr, (m, n), implementation)
            return r.sum()

        r1 = jax.grad(f_api, argnums=(0, 1))(x, data)

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_matrix(x, w, indices, indptr, shape=(m, n))
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, data)

        return r1, r2

    @pytest.mark.parametrize('transpose', [True, False])
    def test_vmap_matrix_vjp(self, transpose):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        if transpose:
            xs = brainstate.random.rand(b, n, m) < 0.1
        else:
            xs = brainstate.random.rand(b, n, k) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(
                lambda x: self._run_vjp(x, data, indices, indptr, m, n, transpose=transpose)
            )(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_vjp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_vjp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_vjp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_vjp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        implementation = self._implementation

        def f_api(x, w):
            if transpose:
                r = _matrix_csr_api(x, w, indices, indptr, (m, n), implementation)
            else:
                r = _csr_matrix_api(x, w, indices, indptr, (m, n), implementation)
            return r

        r1 = jax.jvp(f_api, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_matrix(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_jvp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        xs = brainstate.random.rand(b, k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda x: self._run_jvp(x, data, indices, indptr, m, n))(xs)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_jvp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda data: self._run_jvp(x, data, indices, indptr, m, n))(data)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_jvp(self, homo_w):
        _require_implementations(CSRMM_IMPLEMENTATIONS, 'binary_csrmm')

        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        for implementation in CSRMM_IMPLEMENTATIONS:
            self._implementation = implementation
            r1, r2 = brainstate.transform.vmap2(lambda ind: self._run_jvp(x, data, ind, indptr, m, n))(indices)
            assert jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, r1, r2))


# ---------------------------------------------------------------------------
# Backend registration. Cheap metadata assertions -- no kernel compilation, so
# deliberately not marked ``slow``.
# ---------------------------------------------------------------------------


def test_binary_csrmv_gpu_cusparse_backend_names():
    backends = binary_csrmv_p.available_backends('gpu')

    assert 'cusparse' in backends
    # Legacy names were removed / renamed.
    assert 'BCOO_cusparse' not in backends
    assert 'JAX_cusparse' not in backends


def test_binary_csrmm_gpu_cusparse_backend_names():
    backends = binary_csrmm_p.available_backends('gpu')

    assert 'cusparse' in backends
    # Legacy names were removed / renamed.
    assert 'BCOO_cusparse' not in backends
    assert 'JAX_cusparse' not in backends


# ---------------------------------------------------------------------------
# int64 ``indptr`` policy on the CUDA path.
#
# ``indices`` stay int32 (the CUDA ABI is int32-only for coordinates) while
# ``indptr`` may widen to int64. The generator tests run without a real GPU by
# stubbing ``load_cuda_file``/``ffi_call``; the ``accepts`` tests need one.
# Cheap stub-driven checks, so deliberately not marked ``slow``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'factory,args,kwargs',
    [
        (
            binary_mod._binary_csrmv_cuda_kernel,
            (shape_of(jnp.float32), shape_of(jnp.bool_), False),
            {'outs': [shape_of(jnp.float32)]},
        ),
        (
            binary_mod._binary_csrmm_cuda_kernel,
            (shape_of(jnp.float32), shape_of(jnp.bool_, (2, 1)), False),
            {'outs': [shape_of(jnp.float32, (1, 1))]},
        ),
    ],
)
def test_cuda_kernel_generators_reject_int64_indices_before_loading_cuda(factory, args, kwargs):
    call_kwargs = cuda_kwargs()
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match="indices with dtype int32"):
        factory(*args, **call_kwargs)


def test_binary_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(binary_mod, "load_cuda_file", lambda path, name, **kwargs: load_calls.append((path, name, kwargs)))
    monkeypatch.setattr(binary_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    with jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)
        workspace = _make_binary_task_workspace(indptr)
        task_kwargs = {
            'task_begin_info': shape_of(workspace.task_begin.dtype, workspace.task_begin.shape),
            'task_end_info': shape_of(workspace.task_end.dtype, workspace.task_end.shape),
            'status_info': shape_of(workspace.status.dtype, workspace.status.shape),
            'task_capacity': workspace.task_capacity,
        }
        mv_task_outs = (
            shape_of(jnp.float32),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )
        mm_nt_task_outs = (
            shape_of(jnp.float32, (1, 1)),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )
        mm_t_task_outs = (
            shape_of(jnp.float32, (2, 1)),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )

        mv_kernel = binary_mod._binary_csrmv_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (2,)),
            False,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mv_task_outs,
                **task_kwargs,
            },
        )
        mv_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([True, False]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mv_t_kernel = binary_mod._binary_csrmv_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (1,)),
            True,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mv_task_outs,
                **task_kwargs,
            },
        )
        mv_t_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([True]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_nt_kernel = binary_mod._binary_csrmm_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (2, 1)),
            False,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_nt_task_outs,
                **task_kwargs,
            },
        )
        mm_nt_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([[True], [False]]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_t_kernel = binary_mod._binary_csrmm_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.float32, (2, 1)),
            True,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_t_task_outs,
                **task_kwargs,
            },
        )
        mm_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([[1.0], [0.0]], dtype=jnp.float32),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

    assert [strip_hybrid_suffix(name) for _, name, _ in load_calls] == [
        'csr_binary_csrmv',
        'csr_binary_csrmv_hybrid',
        'csr_binary_csrmm',
        'csr_binary_csrmm_hybrid',
    ]
    assert [strip_hybrid_suffix(call[0]) for call in ffi_calls] == [
        'csr_binary_csrmv.binary_csrmv_nt_auto_homo_f32_bool',
        'csr_binary_csrmv_hybrid.binary_csrmv_wat_hybrid_homo_f32_bool',
        'csr_binary_csrmm.binary_csrmm_nt_auto_hetero_f32_bool',
        'csr_binary_csrmm_hybrid.binary_csrmm_sraw_hybrid_hetero_f32_float',
    ]


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmv_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    vector = jnp.array([True, False], dtype=jnp.bool_) if transpose else jnp.array([True, False, True])
    workspace64 = _make_binary_task_workspace(indptr64)
    workspace32 = _make_binary_task_workspace(indptr32)

    got = binary_csrmv(data, indices, indptr64, vector, shape=(2, 3), transpose=transpose,
                       backend='cuda_raw', workspace=workspace64)
    expected = binary_csrmv(data, indices, indptr32, vector, shape=(2, 3), transpose=transpose,
                            backend='jax_raw', workspace=workspace32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmm_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    matrix = (
        jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        if transpose else
        jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    )
    workspace64 = _make_binary_task_workspace(indptr64)
    workspace32 = _make_binary_task_workspace(indptr32)

    got = binary_csrmm(data, indices, indptr64, matrix, shape=(2, 3), transpose=transpose,
                       backend='cuda_raw', workspace=workspace64)
    expected = binary_csrmm(data, indices, indptr32, matrix, shape=(2, 3), transpose=transpose,
                            backend='jax_raw', workspace=workspace32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)
