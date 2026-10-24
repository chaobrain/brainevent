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
import numpy as np
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

    def fake_csrmm_p_call(w, idx, ptr, b, workspace, *, shape, transpose, backend, weighted=False):
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

    def fake_csrmm_p_call(w, idx, ptr, b, workspace, *, shape, transpose, backend, weighted=False):
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

        # Negative events are active (``x != 0``).
        assert jnp.allclose(got, jnp.array([1.0, 3.0, 6.0], dtype=jnp.float32))


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

        # Negative events are active (``x != 0``), so every entry of B fires.
        expected = jnp.array([[1.0, 1.0], [3.0, 3.0], [6.0, 6.0]], dtype=jnp.float32)
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


@requires_gpu_backend
@pytest.mark.parametrize("event_dtype", [jnp.int8, jnp.uint8, jnp.float64])
def test_binary_csrmm_cuda_normalizes_numeric_events_to_bool_abi(event_dtype):
    """Route non-native numeric MM events through the boolean CUDA ABI."""
    with jax.enable_x64():
        data = jnp.array([2.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        matrix = jnp.array([[1, 0], [0, 1]], dtype=event_dtype)
        workspace = _make_binary_task_workspace(indptr)

        got = binary_csrmm(
            data,
            indices,
            indptr,
            matrix,
            shape=(2, 3),
            transpose=True,
            backend="cuda_raw",
            workspace=workspace,
        )
        expected = binary_csrmm(
            data,
            indices,
            indptr,
            matrix,
            shape=(2, 3),
            transpose=True,
            backend="jax_raw",
            workspace=workspace,
        )

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# The ``nt_block`` strategy inside ``binary_csrmm_nt_auto_*`` is only selected when
# ``avg_nnz > 512``; its block reduction stages one accumulator per (strip, lane)
# pair, i.e. 8 * 32 elements of shared memory. The launch used to request only
# ``8 * sizeof(ACC_T)`` bytes -- a 32x under-request that overran the block's
# dynamic shared-memory window. For float32 the overrun happened to stay inside the
# driver's allocation granularity, but float64 (2048 bytes needed, 64 requested)
# faulted with CUDA_ERROR_ILLEGAL_ADDRESS.
_NT_BLOCK_NNZ_PER_ROW = 1024  # > 512 so nt_auto picks the block kernel


def _dense_csr_operands(rng, m, k, n, dtype, homo):
    """Build a CSR/dense pair whose average row length selects the block kernel."""
    indptr = np.arange(0, (m + 1) * _NT_BLOCK_NNZ_PER_ROW,
                       _NT_BLOCK_NNZ_PER_ROW, dtype=np.int32)
    indices = np.concatenate([
        np.sort(rng.choice(k, _NT_BLOCK_NNZ_PER_ROW, replace=False)).astype(np.int32)
        for _ in range(m)
    ])
    nnz = int(indptr[-1])
    weights = rng.standard_normal(1 if homo else nnz).astype(dtype)
    matrix = rng.random((k, n)) < 0.5
    return weights, indices, indptr, matrix


@requires_gpu_backend
@pytest.mark.slow
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmm_nt_block_shared_memory_is_large_enough(homo):
    """float64 ``nt_auto`` must not fault when it routes to the block kernel."""
    with jax_x64_enabled():
        rng = np.random.default_rng(0)
        m, k, n = 32, 2048, 64
        weights, indices, indptr, matrix = _dense_csr_operands(
            rng, m, k, n, np.float64, homo
        )
        workspace = _make_binary_task_workspace(jnp.asarray(indptr))
        kwargs = dict(shape=(m, k), transpose=False, workspace=workspace)
        args = (jnp.asarray(weights), jnp.asarray(indices),
                jnp.asarray(indptr), jnp.asarray(matrix))

        got = binary_csrmm(*args, backend='cuda_raw', **kwargs)
        expected = binary_csrmm(*args, backend='jax_raw', **kwargs)

        assert jnp.allclose(got, expected, rtol=1e-10, atol=1e-10)


# ``binary_csrmv_nt_auto`` picks one of three row-mapping strategies from the
# average row length: one thread per row below 16, one warp per row up to 512,
# one block per row above that. The warp tier used to be missing, so rows of
# moderate length fell to the thread kernel and read ``indices``/``weights``
# uncoalesced -- measurably slower than a warp-per-row mapping. These row
# lengths straddle both thresholds so every branch is exercised.
_CSRMV_TIER_NNZ_PER_ROW = [4, 15, 16, 64, 511, 512, 1024]


@requires_gpu_backend
@pytest.mark.slow
@pytest.mark.parametrize('nnz_per_row', _CSRMV_TIER_NNZ_PER_ROW)
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmv_matches_reference_across_dispatch_tiers(nnz_per_row, homo):
    """Every ``nt_auto`` row-mapping strategy must agree with the reference."""
    with jax_x64_enabled():
        rng = np.random.default_rng(nnz_per_row)
        m, k = 256, 2048
        indptr = np.arange(0, (m + 1) * nnz_per_row, nnz_per_row, dtype=np.int32)
        indices = np.concatenate([
            np.sort(rng.choice(k, nnz_per_row, replace=False)).astype(np.int32)
            for _ in range(m)
        ])
        nnz = int(indptr[-1])
        weights = rng.standard_normal(1 if homo else nnz).astype(np.float64)
        vector = rng.random(k) < 0.5

        workspace = _make_binary_task_workspace(jnp.asarray(indptr))
        kwargs = dict(shape=(m, k), transpose=False, workspace=workspace)
        args = (jnp.asarray(weights), jnp.asarray(indices),
                jnp.asarray(indptr), jnp.asarray(vector))

        got = binary_csrmv(*args, backend='cuda_raw', **kwargs)
        expected = binary_csrmv(*args, backend='jax_raw', **kwargs)

        assert jnp.allclose(got, expected, rtol=1e-10, atol=1e-10)


@requires_gpu_backend
@pytest.mark.slow
def test_binary_csrmv_warp_tier_handles_ragged_and_empty_rows():
    """The warp kernel must tolerate empty rows and a row count off the tiling.

    One warp handles one row and the grid is strided, so a row count that is
    not a multiple of the warps-per-block, mixed with empty and over-long rows,
    is the case where an off-by-one in the row loop or a divergent lane in the
    shuffle reduction would show up.
    """
    with jax_x64_enabled():
        rng = np.random.default_rng(7)
        k = 1024
        # average stays in the warp tier while individual rows vary wildly
        row_lengths = np.array([0, 1, 33, 0, 200, 64, 0, 7, 512, 90, 0, 31, 63, 65, 128],
                               dtype=np.int32)
        m = row_lengths.size
        assert 16 <= row_lengths.sum() // m < 512, 'must land in the warp tier'
        indptr = np.concatenate([[0], np.cumsum(row_lengths)]).astype(np.int32)
        indices = np.concatenate(
            [np.sort(rng.choice(k, int(n), replace=False)).astype(np.int32)
             for n in row_lengths]
        ).astype(np.int32)
        weights = rng.standard_normal(int(indptr[-1])).astype(np.float64)
        vector = rng.random(k) < 0.5

        workspace = _make_binary_task_workspace(jnp.asarray(indptr))
        kwargs = dict(shape=(m, k), transpose=False, workspace=workspace)
        args = (jnp.asarray(weights), jnp.asarray(indices),
                jnp.asarray(indptr), jnp.asarray(vector))

        got = binary_csrmv(*args, backend='cuda_raw', **kwargs)
        expected = binary_csrmv(*args, backend='jax_raw', **kwargs)

        assert jnp.allclose(got, expected, rtol=1e-10, atol=1e-10)
        # empty rows must be written, not left uninitialised
        assert jnp.all(jnp.asarray(got)[row_lengths == 0] == 0.0)


def test_binary_csrmv_cuda_source_keeps_three_dispatch_tiers():
    """``nt_auto`` must keep the warp tier between the thread and block tiers.

    Source-level so it runs without a GPU. A two-tier dispatcher still computes
    the right answer, so the numeric tests above cannot catch its removal.
    """
    from pathlib import Path

    cu = Path(binary_mod.__file__).with_name('binary_csrmv.cu').read_text()
    for kind in ('homo', 'hetero'):
        assert f'DEFINE_CSRMV_NT_WARP_{kind.upper()}(' in cu
        assert f'_csrmv_nt_warp_{kind}_kern' in cu
    # the three-way ladder itself
    assert cu.count('if (avg_nnz < 16)') == 2
    assert cu.count('} else if (avg_nnz < 512) {') == 2
    assert 'avg_nnz <= 512' not in cu, 'two-tier dispatch came back'


# ---------------------------------------------------------------------------
# Event activity contract (``x != 0``), CUDA event ABI (#199, #204) and binary
# weight gradients (#198). See docs/specs/2026-10-24-event-activity-and-abi.md.
# ---------------------------------------------------------------------------

from brainevent._csr._test_util import (  # noqa: E402
    EVENT_DTYPE_NAMES, event_dtype_context, event_mask, event_values,
    random_csr_numpy, run_python_isolated,
)
from brainevent._test_util import check_event_op_grads, requires_gpu  # noqa: E402

_ACT_M, _ACT_K, _ACT_N = 13, 9, 3


def _hp_matmul(a, b):
    """Full-precision float32 matmul (GPU defaults to TF32, ~1e-3 error)."""
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _activity_operands(homo, seed=0):
    rng = np.random.default_rng(seed)
    data, indices, indptr, dense = random_csr_numpy(rng, _ACT_M, _ACT_K)
    if homo:
        data = np.array([1.5], dtype=np.float32)
        dense = (dense != 0) * 1.5
    return jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr), dense


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmv_event_is_active_iff_nonzero(backend, dtype_name, homo, transpose):
    """Negative and fractional events are active and weights are never scaled by them."""
    with event_dtype_context(dtype_name):
        data, indices, indptr, dense = _activity_operands(homo)
        x = event_values(dtype_name, _ACT_M if transpose else _ACT_K)
        got = binary_csrmv(
            data, indices, indptr, x, shape=(_ACT_M, _ACT_K), transpose=transpose,
            backend=backend, workspace=_make_binary_task_workspace(indptr),
        )
        mask = event_mask(x)
        expected = mask @ dense if transpose else dense @ mask
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmm_event_is_active_iff_nonzero(backend, dtype_name, homo, transpose):
    """Matrix events follow the same ``x != 0`` rule on every backend."""
    with event_dtype_context(dtype_name):
        data, indices, indptr, dense = _activity_operands(homo, seed=1)
        B = event_values(dtype_name, (_ACT_M if transpose else _ACT_K, _ACT_N), seed=1)
        got = binary_csrmm(
            data, indices, indptr, B, shape=(_ACT_M, _ACT_K), transpose=transpose,
            backend=backend, workspace=_make_binary_task_workspace(indptr),
        )
        mask = event_mask(B)
        expected = dense.T @ mask if transpose else dense @ mask
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
def test_binary_csrmv_nan_event_is_active(backend):
    data, indices, indptr, dense = _activity_operands(homo=False)
    x = jnp.zeros(_ACT_M, dtype=jnp.float32).at[jnp.array([0, 5])].set(jnp.nan)
    got = binary_csrmv(
        data, indices, indptr, x, shape=(_ACT_M, _ACT_K), transpose=True,
        backend=backend, workspace=_make_binary_task_workspace(indptr),
    )
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), event_mask(x) @ dense, rtol=1e-5, atol=1e-5)


def _dense_from_data(data, indices, indptr, shape):
    """Differentiable dense form of a CSR matrix (homogeneous if ``data.size == 1``)."""
    rows = np.repeat(np.arange(shape[0]), np.diff(np.asarray(indptr)))
    cols = np.asarray(indices)
    vals = jnp.broadcast_to(data, (cols.size,)) if data.size == 1 else data
    return jnp.zeros(shape, dtype=data.dtype).at[rows, cols].add(vals)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmv_gradients_use_event_mask(backend, homo, transpose):
    """#198: weight cotangents use the activity mask, event cotangents are straight-through."""
    data, indices, indptr, _ = _activity_operands(homo, seed=2)
    shape = (_ACT_M, _ACT_K)
    x = event_values('float32', _ACT_M if transpose else _ACT_K, seed=2)

    def op(d, v):
        return binary_csrmv(d, indices, indptr, v, shape=shape, transpose=transpose,
                            backend=backend, workspace=_make_binary_task_workspace(indptr))

    def dense_op(d, z):
        W = _dense_from_data(d, indices, indptr, shape)
        return _hp_matmul(z, W) if transpose else _hp_matmul(W, z)

    check_event_op_grads(op, dense_op, data, x, mode='binary')


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmm_gradients_use_event_mask(backend, homo, transpose):
    """#198 for the matrix product (the hetero branch goes through SDDMM)."""
    data, indices, indptr, _ = _activity_operands(homo, seed=3)
    shape = (_ACT_M, _ACT_K)
    B = event_values('float32', (_ACT_M if transpose else _ACT_K, _ACT_N), seed=3)

    def op(d, v):
        return binary_csrmm(d, indices, indptr, v, shape=shape, transpose=transpose,
                            backend=backend, workspace=_make_binary_task_workspace(indptr))

    def dense_op(d, z):
        W = _dense_from_data(d, indices, indptr, shape)
        return _hp_matmul(W.T, z) if transpose else _hp_matmul(W, z)

    check_event_op_grads(op, dense_op, data, B, mode='binary')


def test_issue_198_binary_csr_weight_gradient_regression():
    """Exact #198 repro: a count of 2 contributes 1 to the weight gradient, not 2."""
    import brainevent as be

    W = jnp.array([[1., 10.], [100., 0.], [0., 1000.]])
    csr = be.CSR.fromdense(W)
    mk = lambda d: be.CSR((d, csr.indices, csr.indptr), shape=W.shape)

    x = jnp.array([0., 2., 1.])
    g = jax.grad(lambda d: (be.BinaryArray(x) @ mk(d)).sum())(csr.data)
    np.testing.assert_allclose(np.asarray(g), [0., 0., 1., 1.])

    X = jnp.stack([x, x])
    g_mm = jax.grad(lambda d: (be.BinaryArray(X) @ mk(d)).sum())(csr.data)
    np.testing.assert_allclose(np.asarray(g_mm), [0., 0., 2., 2.])

    # Negative events are active: forward and gradient both count them once.
    xn = jnp.array([-1., 1., 0.])
    assert float((be.BinaryArray(xn) @ csr).sum()) == 111.
    g_neg = jax.grad(lambda d: (be.BinaryArray(xn) @ mk(d)).sum())(csr.data)
    np.testing.assert_allclose(np.asarray(g_neg), [1., 1., 1., 0.])


@requires_gpu
def test_issue_199_float64_events_hetero_float64_csr_default_backend():
    """#199: float64 spikes @ hetero float64 CSR on GPU must not abort and must be correct."""
    import brainevent as be

    with jax_x64_enabled():
        rng = np.random.default_rng(0)
        n_pre, n_post, p = 1000, 1000, 0.1
        W = (rng.random((n_pre, n_post)) < p) * rng.normal(size=(n_pre, n_post))
        csr = be.CSR.fromdense(jnp.asarray(W, dtype=jnp.float64))
        spikes = jnp.asarray(rng.random(n_pre) < 0.05, dtype=jnp.float64)
        spikes = spikes.at[:3].set(jnp.array([2.0, -1.0, 0.5]))
        out = jax.jit(lambda s: be.BinaryArray(s) @ csr)(spikes).block_until_ready()
        assert out.dtype == jnp.float64
        ref = (np.asarray(spikes) != 0).astype(np.float64) @ W
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10, atol=1e-10)


_BAD_HYBRID_CALL = r'''
import sys
from pathlib import Path
import jax, jax.numpy as jnp
import brainevent._csr.binary as binary_mod
from brainevent._op import load_cuda_file
from brainevent._csr.hybrid_config import (
    compile_flags_for_config, get_hybrid_config, module_suffix_for_config,
)
cfg = get_hybrid_config()
mod = 'csr_binary_csrmv_hybrid' + module_suffix_for_config(cfg)
load_cuda_file(Path(binary_mod.__file__).parent / 'binary_csrmv_hybrid.cu', name=mod,
               extra_cuda_cflags=compile_flags_for_config(cfg), allow_cuda_graph=False)
cap = 4
weights = jnp.ones(3, jnp.float32)
indices = jnp.array([0, 1, 2], jnp.int32)
indptr = jnp.array([0, 2, 3], jnp.int32)
vector = jnp.array([1, 0], jnp.int32)  # wrong dtype: the _float kernel expects float32
tb = jnp.zeros(cap, jnp.int32); te = jnp.zeros(cap, jnp.int32); st = jnp.zeros(2, jnp.int32)
outs = (jax.ShapeDtypeStruct((3,), jnp.float32), jax.ShapeDtypeStruct((cap,), jnp.int32),
        jax.ShapeDtypeStruct((cap,), jnp.int32), jax.ShapeDtypeStruct((2,), jnp.int32))
try:
    r = jax.ffi.ffi_call(mod + '.binary_csrmv_wat_hybrid_hetero_f32_float', outs,
                         input_output_aliases={4: 1, 5: 2, 6: 3})(
        weights, indices, indptr, vector, tb, te, st, task_capacity=cap)
    jax.block_until_ready(r)
except Exception as e:
    print('RAISED', type(e).__name__, e)
    sys.exit(0)
print('NO ERROR')
sys.exit(3)
'''


@requires_gpu
def test_hybrid_cuda_dtype_mismatch_raises_python_exception():
    """A failing host-side check in a hybrid kernel raises instead of aborting the process."""
    proc = run_python_isolated(_BAD_HYBRID_CALL)
    output = proc.stdout + proc.stderr
    assert proc.returncode >= 0, f'process killed by signal {-proc.returncode}:\n{output}'
    assert proc.returncode == 0, output
    assert 'expects matching vector dtype' in output, output


# ---------------------------------------------------------------------------
# Value-weighted events (``weighted=True``, WeightedArray, #201).
# See docs/specs/2026-10-24-weighted-array.md.
# ---------------------------------------------------------------------------

from brainevent._csr._test_util import WEIGHTED_EVENT_DTYPE_NAMES, heavy_row_csr_numpy  # noqa: E402


def _weighted_tol(dtype_name):
    # float16/bfloat16 events hold exact values (2, -1, 0.5) and the product is
    # computed in the weight dtype, but the event cotangent/tangent checks are
    # rounded to the event dtype.
    if dtype_name in ('float16', 'bfloat16'):
        return dict(rtol=2e-2, atol=2e-2)
    return dict(rtol=1e-4, atol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype_name', WEIGHTED_EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmv_weighted_matches_dense_and_grads(backend, dtype_name, homo, transpose):
    """Weighted mat-vec is the exact product ``x @ W`` with exact derivatives."""
    with event_dtype_context(dtype_name):
        data, indices, indptr, _ = _activity_operands(homo, seed=4)
        shape = (_ACT_M, _ACT_K)
        x = event_values(dtype_name, _ACT_M if transpose else _ACT_K, seed=4)

        def op(d, v):
            return binary_csrmv(d, indices, indptr, v, shape=shape, transpose=transpose, backend=backend,
                                workspace=_make_binary_task_workspace(indptr), weighted=True)

        def dense_op(d, z):
            # Events are cast to the weight dtype (the kernels' compute dtype).
            W = _dense_from_data(d, indices, indptr, shape)
            z = z.astype(d.dtype)
            return _hp_matmul(z, W) if transpose else _hp_matmul(W, z)

        check_event_op_grads(op, dense_op, data, x, mode='weighted',
                             check_vmap=dtype_name == 'float32', **_weighted_tol(dtype_name))


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype_name', WEIGHTED_EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmm_weighted_matches_dense_and_grads(backend, dtype_name, homo, transpose):
    """Weighted mat-mat (hetero weight cotangent through the SDDMM path)."""
    with event_dtype_context(dtype_name):
        data, indices, indptr, _ = _activity_operands(homo, seed=5)
        shape = (_ACT_M, _ACT_K)
        B = event_values(dtype_name, (_ACT_M if transpose else _ACT_K, _ACT_N), seed=5)

        def op(d, v):
            return binary_csrmm(d, indices, indptr, v, shape=shape, transpose=transpose, backend=backend,
                                workspace=_make_binary_task_workspace(indptr), weighted=True)

        def dense_op(d, z):
            W = _dense_from_data(d, indices, indptr, shape)
            z = z.astype(d.dtype)
            return _hp_matmul(W.T, z) if transpose else _hp_matmul(W, z)

        check_event_op_grads(op, dense_op, data, B, mode='weighted',
                             check_vmap=dtype_name == 'float32', **_weighted_tol(dtype_name))


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_csrmv_weighted_vmap_over_events_matches_dense(backend, homo, transpose):
    """vmap over the event axis lowers to the weighted mat-mat primitive."""
    data, indices, indptr, dense = _activity_operands(homo, seed=6)
    shape = (_ACT_M, _ACT_K)
    xs = jnp.stack([event_values('float32', _ACT_M if transpose else _ACT_K, seed=s) for s in range(3)])

    f = jax.vmap(lambda v: binary_csrmv(data, indices, indptr, v, shape=shape, transpose=transpose,
                                        backend=backend, workspace=_make_binary_task_workspace(indptr),
                                        weighted=True))
    got = np.asarray(f(xs), dtype=np.float64)
    xs64 = np.asarray(xs, dtype=np.float64)
    expected = xs64 @ dense if transpose else xs64 @ dense.T
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
def test_binary_csrmv_weighted_heavy_rows_use_row_value(backend, homo):
    """Rows longer than one hybrid task (> 4096 entries) are scaled by their own event value."""
    rng = np.random.default_rng(7)
    m, k = 6, 9000
    data, indices, indptr, dense = heavy_row_csr_numpy(rng, m, k, row_nnz=8500)
    if homo:
        dense = (dense != 0) * 1.5
        data = np.array([1.5], np.float32)
    x = jnp.array([2.0, 0.0, -1.0, 0.5, 3.0, 0.0], dtype=jnp.float32)
    indptr = jnp.asarray(indptr)
    got = binary_csrmv(jnp.asarray(data), jnp.asarray(indices), indptr, x, shape=(m, k), transpose=True,
                       backend=backend, workspace=_make_binary_task_workspace(indptr), weighted=True)
    np.testing.assert_allclose(np.asarray(got, np.float64), np.asarray(x, np.float64) @ dense,
                               rtol=1e-4, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize('backend', CSRMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('homo', [True, False])
def test_binary_csrmm_weighted_heavy_rows_use_row_value(backend, homo):
    rng = np.random.default_rng(8)
    m, k = 5, 9000
    data, indices, indptr, dense = heavy_row_csr_numpy(rng, m, k, row_nnz=8500)
    if homo:
        dense = (dense != 0) * 1.5
        data = np.array([1.5], np.float32)
    B = jnp.array([[2.0, 0.0], [0.0, -1.0], [-1.0, 0.5], [0.5, 0.0], [0.0, 3.0]], dtype=jnp.float32)
    indptr = jnp.asarray(indptr)
    got = binary_csrmm(jnp.asarray(data), jnp.asarray(indices), indptr, B, shape=(m, k), transpose=True,
                       backend=backend, workspace=_make_binary_task_workspace(indptr), weighted=True)
    np.testing.assert_allclose(np.asarray(got, np.float64), dense.T @ np.asarray(B, np.float64),
                               rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('weight_dtype', [jnp.float32, jnp.float16])
@pytest.mark.parametrize('homo', [True, False])
def test_binary_cuda_generators_select_weighted_abi(monkeypatch, weight_dtype, homo):
    """``weighted=True`` picks the ``_wv`` symbols and casts events to the weight dtype in the lowering."""
    ffi_calls = []
    monkeypatch.setattr(binary_mod, "load_cuda_file", lambda path, name, **kwargs: None)
    monkeypatch.setattr(binary_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    workspace = _make_binary_task_workspace(indptr)
    task_kwargs = {
        'task_begin_info': shape_of(workspace.task_begin.dtype, workspace.task_begin.shape),
        'task_end_info': shape_of(workspace.task_end.dtype, workspace.task_end.shape),
        'status_info': shape_of(workspace.status.dtype, workspace.status.shape),
        'task_capacity': workspace.task_capacity,
    }
    tail = (task_kwargs['task_begin_info'], task_kwargs['task_end_info'], task_kwargs['status_info'])
    n_w = 1 if homo else 2
    weights = jnp.ones(n_w, dtype=weight_dtype)
    sfx = {jnp.float32: '_f32', jnp.float16: '_f16'}[weight_dtype]
    hs = '_homo' if homo else '_hetero'
    cases = [
        (binary_mod._binary_csrmv_cuda_kernel, (2,), False, (1,),
         f'csr_binary_csrmv.binary_csrmv_nt_auto{hs}{sfx}_wv'),
        (binary_mod._binary_csrmv_cuda_kernel, (1,), True, (2,),
         f'csr_binary_csrmv_hybrid.binary_csrmv_wat_hybrid{hs}{sfx}_wv'),
        (binary_mod._binary_csrmm_cuda_kernel, (2, 3), False, (1, 3),
         f'csr_binary_csrmm.binary_csrmm_nt_auto{hs}{sfx}_wv'),
        (binary_mod._binary_csrmm_cuda_kernel, (1, 3), True, (2, 3),
         f'csr_binary_csrmm_hybrid.binary_csrmm_sraw_hybrid{hs}{sfx}_wv'),
    ]
    for factory, ev_shape, transpose, out_shape, expected_name in cases:
        ffi_calls.clear()
        events = jnp.full(ev_shape, -3, dtype=jnp.int8)
        kernel = factory(
            shape_of(weight_dtype, (n_w,)), shape_of(jnp.int8, ev_shape), transpose,
            **{**cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int32),
               'outs': (shape_of(weight_dtype, out_shape), *tail), 'weighted': True, **task_kwargs},
        )
        kernel(weights, indices, indptr, events, workspace.task_begin, workspace.task_end, workspace.status)
        assert len(ffi_calls) == 1
        name, _, _, args, _ = ffi_calls[0]
        assert strip_hybrid_suffix(name) == expected_name
        event_arg = args[3]
        assert event_arg.dtype == weight_dtype
        np.testing.assert_array_equal(np.asarray(event_arg, np.float32), np.full(ev_shape, -3.0))


def test_binary_csrmv_weighted_integer_events_have_zero_cotangent():
    """Integer weighted events are not differentiable: the event cotangent is float0."""
    data, indices, indptr, _ = _activity_operands(False, seed=9)
    x = event_values('int8', _ACT_K, seed=9)
    f = lambda d, v: binary_csrmv(d, indices, indptr, v, shape=(_ACT_M, _ACT_K), backend='jax_raw',
                                  workspace=_make_binary_task_workspace(indptr), weighted=True).sum()
    g_d, g_x = jax.grad(f, argnums=(0, 1), allow_int=True)(data, x)
    assert g_x.dtype == jax.dtypes.float0
    W = np.asarray(_dense_from_data(data, indices, indptr, (_ACT_M, _ACT_K)))
    rows = np.repeat(np.arange(_ACT_M), np.diff(np.asarray(indptr)))
    np.testing.assert_allclose(np.asarray(g_d), np.asarray(x, np.float32)[np.asarray(indices)], rtol=1e-6)
    assert W.shape == (_ACT_M, _ACT_K) and rows.size == data.size


def test_binary_csrmv_weighted_default_is_binary():
    """``weighted`` defaults to ``False``: values are binarised."""
    data, indices, indptr, dense = _activity_operands(False, seed=10)
    x = event_values('float32', _ACT_K, seed=10)
    ws = _make_binary_task_workspace(indptr)
    binary = binary_csrmv(data, indices, indptr, x, shape=(_ACT_M, _ACT_K), backend='jax_raw', workspace=ws)
    weighted = binary_csrmv(data, indices, indptr, x, shape=(_ACT_M, _ACT_K), backend='jax_raw',
                            workspace=ws, weighted=True)
    np.testing.assert_allclose(np.asarray(binary, np.float64), dense @ event_mask(x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(weighted, np.float64), dense @ np.asarray(x, np.float64),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('axis', [0, 1])
def test_binary_csrmv_batching_forwards_weighted(monkeypatch, axis):
    """The mat-vec batching rule keeps ``weighted`` when it lowers to mat-mat."""
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    vector = jnp.array([[2.0, 0.0], [0.0, -1.0]], dtype=jnp.float32)
    workspace = _make_binary_task_workspace(indptr)
    seen = []

    def fake_csrmm_p_call(w, idx, ptr, b, workspace, *, shape, transpose, backend, weighted=False):
        seen.append(weighted)
        return (jnp.ones((1, 2), dtype=jnp.float32), workspace.task_begin, workspace.task_end, workspace.status)

    monkeypatch.setattr("brainevent._csr.binary.binary_csrmm_p_call", fake_csrmm_p_call)
    axes = (None, None, None, axis, None, None, None)
    _csrmv_batching(
        (weights, indices, indptr, vector, workspace.task_begin, workspace.task_end, workspace.status),
        axes, shape=(1, 2), transpose=False, backend="jax_raw", weighted=True,
    )
    assert seen == [True]
