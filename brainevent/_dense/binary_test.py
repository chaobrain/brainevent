# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


import contextlib

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
import brainevent._dense.binary as dense_binary_mod
from brainevent._dense.binary import (
    binary_densemv, binary_densemv_p, binary_densemv_p_call,
    binary_densemm, binary_densemm_p, binary_densemm_p_call,
)
from brainevent._test_util import check_event_op_grads, jax_x64_enabled, requires_gpu

jax.config.update('jax_default_matmul_precision', 'highest')

platform = jax.default_backend()
DENSEMV_IMPLEMENTATIONS = tuple(binary_densemv_p.available_backends(platform))
DENSEMM_IMPLEMENTATIONS = tuple(binary_densemm_p.available_backends(platform))


def _capture_cuda_ffi(monkeypatch):
    captured = {}

    def fake_load_cuda_file(path, name, **kwargs):
        captured['load'] = (path.name, name, kwargs)

    def fake_ffi_call(kernel_name, out_info):
        captured['kernel_name'] = kernel_name
        captured['out_info'] = out_info

        def run(*args):
            infos = out_info if isinstance(out_info, (list, tuple)) else (out_info,)
            return tuple(jnp.zeros(info.shape, info.dtype) for info in infos)

        return run

    monkeypatch.setattr(dense_binary_mod, 'load_cuda_file', fake_load_cuda_file)
    monkeypatch.setattr(dense_binary_mod.jax.ffi, 'ffi_call', fake_ffi_call)
    return captured


@pytest.mark.parametrize(
    "transpose,weight_shape,expected_kernel",
    [
        (False, (2, 3), 'dense_binary_mv.binary_densemv_no_transpose_f32_bool'),
        (True, (3, 2), 'dense_binary_mv.binary_densemv_transpose_f32_bool'),
    ],
)
def test_densemv_cuda_raw_uses_wpr_entrypoints(monkeypatch, transpose, weight_shape, expected_kernel):
    captured = _capture_cuda_ffi(monkeypatch)
    out_shape = (weight_shape[1],) if transpose else (weight_shape[0],)
    kernel = dense_binary_mod._binary_densemv_cuda_kernel(
        jax.ShapeDtypeStruct((3,), jnp.bool_),
        transpose,
        outs=[jax.ShapeDtypeStruct(out_shape, jnp.float32)],
        weight_info=jax.ShapeDtypeStruct(weight_shape, jnp.float32),
    )

    kernel(jnp.ones(weight_shape, dtype=jnp.float32), jnp.ones((3,), dtype=jnp.bool_))

    assert captured['load'][:2] == ('binary_densemv.cu', 'dense_binary_mv')
    assert captured['kernel_name'] == expected_kernel


@pytest.mark.parametrize(
    "transpose,weight_shape,expected_kernel",
    [
        (False, (2, 3), 'dense_binary_mm.binary_densemm_no_transpose_f32_bool'),
        (True, (3, 2), 'dense_binary_mm.binary_densemm_transpose_f32_bool'),
    ],
)
def test_densemm_cuda_raw_uses_wpr_entrypoints_and_batch_major_physical_output(
    monkeypatch, transpose, weight_shape, expected_kernel
):
    captured = _capture_cuda_ffi(monkeypatch)
    logical_out = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    kernel = dense_binary_mod._binary_densemm_cuda_kernel(
        spk_info=jax.ShapeDtypeStruct((3, 4), jnp.bool_),
        weight_info=jax.ShapeDtypeStruct(weight_shape, jnp.float32),
        transpose=transpose,
        outs=[logical_out],
    )

    result = kernel(
        jnp.ones(weight_shape, dtype=jnp.float32),
        jnp.ones((3, 4), dtype=jnp.bool_),
    )[0]

    assert captured['load'][:2] == ('binary_densemm.cu', 'dense_binary_mm')
    assert captured['kernel_name'] == expected_kernel
    assert captured['out_info'][0].shape == (4, 2)
    assert result.shape == (2, 4)


def test_dense_cublas_backend_registered_on_gpu_without_becoming_default():
    assert 'cublas' in binary_densemv_p.available_backends('gpu')
    assert 'cublas' in binary_densemm_p.available_backends('gpu')
    assert binary_densemv_p.get_default('gpu') == 'cuda_raw'
    assert binary_densemm_p.get_default('gpu') == 'cuda_raw'


# ---- Forward: dense matrix @ binary vector (transpose=False) ----

@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15, 20])
@pytest.mark.parametrize("dtype", [bool, float])
def test_densemv_forward_no_transpose(implementation, m, k, dtype):
    weights = brainstate.random.randn(m, k)
    spikes = brainstate.random.randn(k) < 0.3
    if dtype == float:
        spikes = u.math.asarray(spikes, dtype=float)
    result = binary_densemv(weights, spikes, transpose=False, backend=implementation)
    expected = weights @ u.math.asarray(spikes, dtype=float)
    assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
    jax.block_until_ready((weights, spikes, result, expected))


# ---- Forward: binary vector @ dense matrix (transpose=True) ----

@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("k", [15, 20])
@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("dtype", [bool, float])
def test_densemv_forward_transpose(implementation, k, n, dtype):
    spikes = brainstate.random.randn(k) < 0.3
    if dtype == float:
        spikes = u.math.asarray(spikes, dtype=float)
    weights = brainstate.random.randn(k, n)
    result = binary_densemv(weights, spikes, transpose=True, backend=implementation)
    expected = u.math.asarray(spikes, dtype=float) @ weights
    assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
    jax.block_until_ready((spikes, weights, result, expected))


# ---- Forward: dense matrix @ binary matrix (transpose=False) ----

@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15, 20])
@pytest.mark.parametrize("n", [30])
@pytest.mark.parametrize("dtype", [bool, float])
def test_densemm_forward_no_transpose(implementation, m, k, n, dtype):
    weights = brainstate.random.randn(m, k)
    spikes = brainstate.random.randn(k, n) < 0.3
    if dtype == float:
        spikes = u.math.asarray(spikes, dtype=float)
    result = binary_densemm(weights, spikes, transpose=False, backend=implementation)
    expected = weights @ u.math.asarray(spikes, dtype=float)
    assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
    jax.block_until_ready((weights, spikes, result, expected))


# ---- Forward: weights.T @ binary matrix (transpose=True) ----

@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15, 20])
@pytest.mark.parametrize("n", [30])
def test_densemm_forward_transpose(implementation, m, k, n):
    weights = brainstate.random.randn(k, m)
    spikes = brainstate.random.randn(k, n) < 0.3
    result = binary_densemm(weights, spikes, transpose=True, backend=implementation)
    expected = weights.T @ u.math.asarray(spikes, dtype=float)
    print(jax.numpy.abs(result - expected).max())
    assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
    jax.block_until_ready((spikes, weights, result, expected))


# ---- Gradient: binary_densemv transpose=False ----

@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
def test_densemv_grad_weights_no_transpose(implementation, m, k):
    weights = brainstate.random.randn(m, k)
    spikes = u.math.asarray(brainstate.random.randn(k) < 0.3, dtype=float)

    def f(w):
        return binary_densemv(w, spikes, transpose=False, backend=implementation).sum()

    grad = jax.grad(f)(weights)
    assert grad.shape == weights.shape
    jax.block_until_ready((weights, spikes, grad))


# ---- Gradient: binary_densemv transpose=True ----

@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
def test_densemv_grad_weights_transpose(implementation, k, n):
    spikes = u.math.asarray(brainstate.random.randn(k) < 0.3, dtype=float)
    weights = brainstate.random.randn(k, n)

    def f(w):
        return binary_densemv(w, spikes, transpose=True, backend=implementation).sum()

    grad = jax.grad(f)(weights)
    assert grad.shape == weights.shape
    jax.block_until_ready((spikes, weights, grad))


# ---- Gradient: binary_densemm transpose=False ----

@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
def test_densemm_grad_weights_no_transpose(implementation, m, k, n):
    weights = brainstate.random.randn(m, k)
    spikes = u.math.asarray(brainstate.random.randn(k, n) < 0.3, dtype=float)

    def f(w):
        return binary_densemm(w, spikes, transpose=False, backend=implementation).sum()

    grad = jax.grad(f)(weights)
    assert grad.shape == weights.shape
    jax.block_until_ready((weights, spikes, grad))


# ---- Gradient: binary_densemm transpose=True ----

@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
def test_densemm_grad_weights_transpose(implementation, m, k, n):
    spikes = u.math.asarray(brainstate.random.randn(k, n) < 0.3, dtype=float)
    weights = brainstate.random.randn(k, m)

    def f(w):
        return binary_densemm(w, spikes, transpose=True, backend=implementation).sum()

    grad = jax.grad(f)(weights)
    assert grad.shape == weights.shape
    jax.block_until_ready((spikes, weights, grad))


# ---- Batching (vmap): binary_densemv transpose=False ----

@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("batch_size", [5])
def test_densemv_vmap_over_spikes_no_transpose(implementation, m, k, batch_size):
    weights = brainstate.random.randn(m, k)
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, k) < 0.3, dtype=float
    )
    batched_fn = jax.vmap(lambda s: binary_densemv(weights, s, transpose=False, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, m)
    jax.block_until_ready((weights, batched_spikes, result))


# ---- Batching (vmap): binary_densemv transpose=True ----

@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("batch_size", [5])
def test_densemv_vmap_over_spikes_transpose(implementation, k, n, batch_size):
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, k) < 0.3, dtype=float
    )
    weights = brainstate.random.randn(k, n)
    batched_fn = jax.vmap(lambda s: binary_densemv(weights, s, transpose=True, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, n)
    jax.block_until_ready((batched_spikes, weights, result))


# ---- Batching (vmap): binary_densemm transpose=False ----

@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("batch_size", [5])
def test_densemm_vmap_over_spikes_no_transpose(implementation, m, k, n, batch_size):
    weights = brainstate.random.randn(m, k)
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, k, n) < 0.3, dtype=float
    )
    batched_fn = jax.vmap(lambda s: binary_densemm(weights, s, transpose=False, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, m, n)
    jax.block_until_ready((weights, batched_spikes, result))


# ---- Batching (vmap): binary_densemm transpose=True ----

@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("batch_size", [5])
def test_densemm_vmap_over_spikes_transpose(implementation, m, k, n, batch_size):
    weights = brainstate.random.randn(k, m)
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, k, n) < 0.3, dtype=float
    )
    batched_fn = jax.vmap(lambda s: binary_densemm(weights, s, transpose=True, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, m, n)
    jax.block_until_ready((batched_spikes, weights, result))


# ---------------------------------------------------------------------------
# Activity contract: an event is active iff ``x != 0`` (bool: ``x``) on every
# backend and for every event dtype (#199, #204).
# ---------------------------------------------------------------------------

_EVENT_PATTERN = np.array([2., -1., 0., 0.5, 0., 1., -3., 0.])
EVENT_DTYPES = (
    'bool', 'int8', 'uint8', 'int16', 'int32',
    'float16', 'bfloat16', 'float32', 'float64',
)


def _x64_if_needed(dtype_name):
    return jax_x64_enabled() if dtype_name == 'float64' else contextlib.nullcontext()


def _make_events(shape, dtype_name):
    """Tile ``[2, -1, 0, 0.5, ...]`` to ``shape`` and cast to ``dtype_name``.

    Integer casts go through int64 so ``-1`` wraps (uint8 -> 255, still
    active) and ``0.5`` truncates to ``0`` (inactive).
    """
    raw = np.resize(_EVENT_PATTERN, int(np.prod(shape))).reshape(shape)
    if dtype_name == 'bool':
        arr = raw != 0
    elif dtype_name.startswith(('int', 'uint')):
        arr = raw.astype(np.int64).astype(dtype_name)
    else:
        arr = raw
    return jnp.asarray(arr, dtype=jnp.dtype(dtype_name))


def _active(events):
    return (np.asarray(events) != 0).astype(np.float64)


@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype_name", EVENT_DTYPES)
@pytest.mark.parametrize("entry", ['public', 'primitive'])
def test_densemv_event_activity_all_dtypes(implementation, transpose, dtype_name, entry):
    k, n = 8, 5
    with _x64_if_needed(dtype_name):
        rng = np.random.default_rng(0)
        w_np = rng.standard_normal((k, n) if transpose else (n, k)).astype(np.float32)
        weights = jnp.asarray(w_np)
        spikes = _make_events((k,), dtype_name)
        fn = binary_densemv if entry == 'public' else (lambda *a, **kw: binary_densemv_p_call(*a, **kw)[0])
        result = np.asarray(fn(weights, spikes, transpose=transpose, backend=implementation))
    s = _active(spikes)
    expected = s @ w_np if transpose else w_np @ s
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype_name", EVENT_DTYPES)
@pytest.mark.parametrize("entry", ['public', 'primitive'])
def test_densemm_event_activity_all_dtypes(implementation, transpose, dtype_name, entry):
    k, m, n = 8, 5, 3
    with _x64_if_needed(dtype_name):
        rng = np.random.default_rng(1)
        w_np = rng.standard_normal((k, m) if transpose else (m, k)).astype(np.float32)
        weights = jnp.asarray(w_np)
        spikes = _make_events((k, n), dtype_name)
        fn = binary_densemm if entry == 'public' else (lambda *a, **kw: binary_densemm_p_call(*a, **kw)[0])
        result = np.asarray(fn(weights, spikes, transpose=transpose, backend=implementation))
    s = _active(spikes)
    expected = w_np.T @ s if transpose else w_np @ s
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
def test_densemv_nan_event_is_active(implementation):
    weights = jnp.asarray(np.arange(6, dtype=np.float32).reshape(3, 2))
    spikes = jnp.asarray([np.nan, 0., -2.], dtype=jnp.float32)
    result = binary_densemv(weights, spikes, transpose=True, backend=implementation)
    np.testing.assert_allclose(np.asarray(result), [0. + 4., 1. + 5.])


@pytest.mark.parametrize(
    "dtype,native",
    [(jnp.bool_, True), (jnp.float32, True), (jnp.float64, False),
     (jnp.float16, False), (jnp.int32, False), (jnp.uint8, False)],
)
@pytest.mark.parametrize("op", ['mv', 'mm'])
def test_dense_cuda_raw_event_abi_selection(monkeypatch, dtype, native, op):
    """Non-native event dtypes select the ``_bool`` symbol and are converted
    to their activity mask inside the lowering (#204)."""
    with jax_x64_enabled():
        captured = {}

        def fake_ffi_call(kernel_name, out_info):
            captured['kernel_name'] = kernel_name

            def run(*args):
                captured['args'] = args
                infos = out_info if isinstance(out_info, (list, tuple)) else (out_info,)
                return tuple(jnp.zeros(info.shape, info.dtype) for info in infos)

            return run

        monkeypatch.setattr(dense_binary_mod, 'load_cuda_file', lambda *a, **kw: None)
        monkeypatch.setattr(dense_binary_mod.jax.ffi, 'ffi_call', fake_ffi_call)
        spikes = jnp.asarray(_make_events((4,) if op == 'mv' else (4, 2), 'float64'), dtype=dtype)
        if op == 'mv':
            kernel = dense_binary_mod._binary_densemv_cuda_kernel(
                jax.ShapeDtypeStruct(spikes.shape, dtype), False,
                outs=[jax.ShapeDtypeStruct((3,), jnp.float32)],
                weight_info=jax.ShapeDtypeStruct((3, 4), jnp.float32),
            )
        else:
            kernel = dense_binary_mod._binary_densemm_cuda_kernel(
                spk_info=jax.ShapeDtypeStruct(spikes.shape, dtype),
                weight_info=jax.ShapeDtypeStruct((3, 4), jnp.float32),
                transpose=False,
                outs=[jax.ShapeDtypeStruct((3, 2), jnp.float32)],
            )
        kernel(jnp.ones((3, 4), jnp.float32), spikes)
    expected_sfx = ('_bool' if dtype == jnp.bool_ else '_float') if native else '_bool'
    assert captured['kernel_name'].endswith('_f32' + expected_sfx)
    passed = captured['args'][1]
    if native:
        assert passed.dtype == jnp.dtype(dtype)
    else:
        assert passed.dtype == jnp.bool_
        np.testing.assert_array_equal(np.asarray(passed), np.asarray(spikes) != 0)


@requires_gpu
@pytest.mark.parametrize("backend", [None, 'cuda_raw'])
@pytest.mark.parametrize("w_dtype", ['float32', 'float64'])
@pytest.mark.parametrize("e_dtype", ['bool', 'float32', 'float64'])
@pytest.mark.parametrize("transpose", [False, True])
def test_issue_204_dense_x64_event_dtypes_gpu(backend, w_dtype, e_dtype, transpose):
    """#204: float64 events reached the float32 CUDA kernel and produced garbage."""
    with jax_x64_enabled():
        rng = np.random.default_rng(204)
        n_pre, n_post = 300, 200
        W = rng.standard_normal((n_pre, n_post)).astype(w_dtype)
        spk = rng.random(n_pre) < 0.1
        events = jnp.asarray(spk, dtype=jnp.dtype(e_dtype))
        if transpose:
            out = binary_densemv(jnp.asarray(W), events, transpose=True, backend=backend)
            ref = spk.astype(np.float64) @ W.astype(np.float64)
            mm = binary_densemm(jnp.asarray(W), events[:, None], transpose=True, backend=backend)[:, 0]
        else:
            Wt = np.ascontiguousarray(W.T)
            out = binary_densemv(jnp.asarray(Wt), events, transpose=False, backend=backend)
            ref = Wt.astype(np.float64) @ spk.astype(np.float64)
            mm = binary_densemm(jnp.asarray(Wt), events[:, None], transpose=False, backend=backend)[:, 0]
        assert out.dtype == jnp.dtype(w_dtype)
        for got in (out, mm):
            got = np.asarray(got, dtype=np.float64)
            rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
            assert rel < 1e-5, f'relative error {rel}'


# ---------------------------------------------------------------------------
# Gradients (#198): weight cotangents use the activity mask, event cotangents
# are straight-through.
# ---------------------------------------------------------------------------

_GRAD_X = jnp.asarray([2., -1., 0., 1., 0., 0.5], dtype=jnp.float32)


@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("transpose", [False, True])
def test_densemv_grads_use_activity_mask(implementation, transpose):
    k, n = _GRAD_X.shape[0], 4
    w = jnp.asarray(np.random.default_rng(3).standard_normal((k, n) if transpose else (n, k)), jnp.float32)
    if transpose:
        op = lambda p, x: binary_densemv(p, x, transpose=True, backend=implementation)
        dense = lambda p, z: z @ p
    else:
        op = lambda p, x: binary_densemv(p, x, transpose=False, backend=implementation)
        dense = lambda p, z: p @ z
    check_event_op_grads(op, dense, w, _GRAD_X, mode='binary')


@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("transpose", [False, True])
def test_densemm_grads_use_activity_mask(implementation, transpose):
    k, m = _GRAD_X.shape[0], 4
    x = jnp.stack([_GRAD_X, jnp.roll(_GRAD_X, 1), -_GRAD_X], axis=1)
    w = jnp.asarray(np.random.default_rng(4).standard_normal((k, m) if transpose else (m, k)), jnp.float32)
    if transpose:
        op = lambda p, v: binary_densemm(p, v, transpose=True, backend=implementation)
        dense = lambda p, z: p.T @ z
    else:
        op = lambda p, v: binary_densemm(p, v, transpose=False, backend=implementation)
        dense = lambda p, z: p @ z
    check_event_op_grads(op, dense, w, x, mode='binary')


def test_issue_198_dense_binary_array_weight_grad():
    """Exact numbers from #198 (Dense case)."""
    W = jnp.array([[1., 10.], [100., 0.], [0., 1000.]])
    x = jnp.array([0., 2., 1.])
    s = (x != 0).astype(W.dtype)

    f = lambda w: (brainevent.BinaryArray(x) @ w).sum()
    f_ref = lambda w: (s @ w).sum()
    assert float(f(W)) == pytest.approx(float(f_ref(W)))  # 1101
    g = jax.grad(f)(W)
    np.testing.assert_allclose(np.asarray(g), [[0., 0.], [1., 1.], [1., 1.]])
    np.testing.assert_allclose(np.asarray(g), np.asarray(jax.grad(f_ref)(W)))
    t = jax.jvp(f, (W,), (jnp.ones_like(W),))[1]
    assert float(t) == pytest.approx(float(g.sum()))

    X = jnp.stack([x, x])
    fm = lambda w: (brainevent.BinaryArray(X) @ w).sum()
    gm = jax.grad(fm)(W)
    np.testing.assert_allclose(np.asarray(gm), [[0., 0.], [2., 2.], [2., 2.]])


# ---------------------------------------------------------------------------
# Weighted events (#201): ``weighted=True`` multiplies by the event value.
# ---------------------------------------------------------------------------

WEIGHTED_EVENT_DTYPES = tuple(d for d in EVENT_DTYPES if d != 'bool') + ('bool',)


def _values(events):
    return np.asarray(events.astype(jnp.float32) if events.dtype == jnp.bfloat16 else events,
                      dtype=np.float64)


@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype_name", WEIGHTED_EVENT_DTYPES)
def test_densemv_weighted_all_dtypes(implementation, transpose, dtype_name):
    k, n = 8, 5
    with _x64_if_needed(dtype_name):
        rng = np.random.default_rng(10)
        w_np = rng.standard_normal((k, n) if transpose else (n, k)).astype(np.float32)
        spikes = _make_events((k,), dtype_name)
        result = binary_densemv(jnp.asarray(w_np), spikes, transpose=transpose,
                                backend=implementation, weighted=True)
        assert result.dtype == jnp.float32
    x = _values(spikes)
    expected = x @ w_np if transpose else w_np @ x
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype_name", WEIGHTED_EVENT_DTYPES)
def test_densemm_weighted_all_dtypes(implementation, transpose, dtype_name):
    k, m, n = 8, 5, 3
    with _x64_if_needed(dtype_name):
        rng = np.random.default_rng(11)
        w_np = rng.standard_normal((k, m) if transpose else (m, k)).astype(np.float32)
        spikes = _make_events((k, n), dtype_name)
        result = binary_densemm(jnp.asarray(w_np), spikes, transpose=transpose,
                                backend=implementation, weighted=True)
    x = _values(spikes)
    expected = w_np.T @ x if transpose else w_np @ x
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
def test_densemv_weighted_skips_zero_events(implementation):
    """A non-finite weight in an inactive row must not reach the output."""
    if implementation in ('jax_raw', 'cublas'):
        pytest.skip('float matmul backends propagate 0 * inf.')
    w = jnp.asarray([[1., 2.], [np.inf, np.nan], [3., 4.]], dtype=jnp.float32)
    x = jnp.asarray([2., 0., -1.], dtype=jnp.float32)
    result = binary_densemv(w, x, transpose=True, backend=implementation, weighted=True)
    np.testing.assert_allclose(np.asarray(result), [2. - 3., 4. - 4.])


def test_dense_weighted_binary_unchanged():
    """``weighted=False`` (default) still binarises; ``weighted=True`` does not."""
    w = jnp.asarray([[1., 2.], [3., 4.], [5., 6.]])
    x = jnp.asarray([2, 0, -1], dtype=jnp.int8)
    np.testing.assert_allclose(binary_densemv(w, x, transpose=True), [6., 8.])
    np.testing.assert_allclose(binary_densemv(w, x, transpose=True, weighted=True), [-3., -2.])


_WEIGHTED_GRAD_DTYPES = ('float32', 'int8', 'int32', 'float16', 'bfloat16', 'float64')


def _weighted_grad_x(shape, dtype_name):
    x = _make_events(shape, dtype_name)
    tol = 2e-2 if dtype_name in ('float16', 'bfloat16') else 1e-4
    return x, tol


@pytest.mark.parametrize("implementation", DENSEMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype_name", _WEIGHTED_GRAD_DTYPES)
def test_densemv_weighted_grads(implementation, transpose, dtype_name):
    with _x64_if_needed(dtype_name):
        x, tol = _weighted_grad_x((6,), dtype_name)
        k, m = x.shape[0], 4
        w = jnp.asarray(np.random.default_rng(12).standard_normal((k, m) if transpose else (m, k)), jnp.float32)
        if transpose:
            op = lambda p, v: binary_densemv(p, v, transpose=True, backend=implementation, weighted=True)
            dense = lambda p, z: z.astype(p.dtype) @ p
        else:
            op = lambda p, v: binary_densemv(p, v, transpose=False, backend=implementation, weighted=True)
            dense = lambda p, z: p @ z.astype(p.dtype)
        check_event_op_grads(op, dense, w, x, mode='weighted', rtol=tol, atol=tol)


@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype_name", _WEIGHTED_GRAD_DTYPES)
def test_densemm_weighted_grads(implementation, transpose, dtype_name):
    with _x64_if_needed(dtype_name):
        x, tol = _weighted_grad_x((6, 3), dtype_name)
        k, m = x.shape[0], 4
        w = jnp.asarray(np.random.default_rng(13).standard_normal((k, m) if transpose else (m, k)), jnp.float32)
        if transpose:
            op = lambda p, v: binary_densemm(p, v, transpose=True, backend=implementation, weighted=True)
            dense = lambda p, z: p.T @ z.astype(p.dtype)
        else:
            op = lambda p, v: binary_densemm(p, v, transpose=False, backend=implementation, weighted=True)
            dense = lambda p, z: p @ z.astype(p.dtype)
        check_event_op_grads(op, dense, w, x, mode='weighted', rtol=tol, atol=tol)


@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("transpose", [False, True])
def test_densemm_weighted_vmap_over_weights(implementation, transpose):
    """Batching over the weight operand keeps ``weighted``."""
    x = _make_events((6, 3), 'int8')
    ws = jnp.asarray(np.random.default_rng(14).standard_normal((2, 6, 4) if transpose else (2, 4, 6)), jnp.float32)
    f = lambda w: binary_densemm(w, x, transpose=transpose, backend=implementation, weighted=True)
    got = jax.vmap(f)(ws)
    xf = x.astype(jnp.float32)
    expected = jnp.einsum('bkm,kn->bmn' if transpose else 'bmk,kn->bmn', ws, xf)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.bool_, jnp.int8, jnp.int32, jnp.float16, jnp.float32])
@pytest.mark.parametrize("w_dtype", [jnp.float32, jnp.float16])
@pytest.mark.parametrize("op", ['mv', 'mm'])
@pytest.mark.parametrize("transpose", [False, True])
def test_dense_cuda_raw_weighted_abi(monkeypatch, dtype, w_dtype, op, transpose):
    """Weighted events select the ``_wv`` symbol, cast to the weight dtype."""
    captured = {}

    def fake_ffi_call(kernel_name, out_info):
        captured['kernel_name'] = kernel_name

        def run(*args):
            captured['args'] = args
            infos = out_info if isinstance(out_info, (list, tuple)) else (out_info,)
            return tuple(jnp.zeros(info.shape, info.dtype) for info in infos)

        return run

    monkeypatch.setattr(dense_binary_mod, 'load_cuda_file', lambda *a, **kw: None)
    monkeypatch.setattr(dense_binary_mod.jax.ffi, 'ffi_call', fake_ffi_call)
    spikes = jnp.asarray(_make_events((4,) if op == 'mv' else (4, 2), 'float32'), dtype=dtype)
    w_shape = (4, 3) if transpose else (3, 4)
    if op == 'mv':
        kernel = dense_binary_mod._binary_densemv_cuda_kernel(
            jax.ShapeDtypeStruct(spikes.shape, dtype), transpose, weighted=True,
            outs=[jax.ShapeDtypeStruct((3,), w_dtype)],
            weight_info=jax.ShapeDtypeStruct(w_shape, w_dtype),
        )
    else:
        kernel = dense_binary_mod._binary_densemm_cuda_kernel(
            spk_info=jax.ShapeDtypeStruct(spikes.shape, dtype),
            weight_info=jax.ShapeDtypeStruct(w_shape, w_dtype),
            transpose=transpose, weighted=True,
            outs=[jax.ShapeDtypeStruct((3, 2), w_dtype)],
        )
    kernel(jnp.ones(w_shape, w_dtype), spikes)
    w_sfx = '_f32' if w_dtype == jnp.float32 else '_f16'
    direction = 'transpose' if transpose else 'no_transpose'
    assert captured['kernel_name'] == f'dense_binary_{op}.binary_dense{op}_{direction}{w_sfx}_wv'
    passed = captured['args'][1]
    assert passed.dtype == jnp.dtype(w_dtype)
    np.testing.assert_array_equal(np.asarray(passed), np.asarray(spikes.astype(w_dtype)))


@pytest.mark.parametrize("op", ['mv', 'mm'])
def test_dense_cublas_weighted_uses_float_product(op):
    """The bool-only cuBLAS entry points are bypassed for weighted events."""
    fn = (dense_binary_mod._binary_densemv_cublas_kernel if op == 'mv'
          else dense_binary_mod._binary_densemm_cublas_kernel)
    x = _make_events((4,) if op == 'mv' else (4, 2), 'int8')
    kernel = fn(spk_info=jax.ShapeDtypeStruct(x.shape, x.dtype),
                weight_info=jax.ShapeDtypeStruct((4, 3), jnp.float32),
                transpose=True, weighted=True,
                outs=[jax.ShapeDtypeStruct((3,) if op == 'mv' else (3, 2), jnp.float32)])
    w = jnp.asarray(np.arange(12, dtype=np.float32).reshape(4, 3))
    got = kernel(w, x)[0]
    np.testing.assert_allclose(np.asarray(got), np.asarray(w.T @ x.astype(jnp.float32)))


@requires_gpu
@pytest.mark.parametrize("backend", [None, 'cuda_raw', 'cublas'])
@pytest.mark.parametrize("w_dtype", ['float32', 'float64', 'float16', 'bfloat16'])
@pytest.mark.parametrize("transpose", [False, True])
def test_dense_weighted_gpu_large(backend, w_dtype, transpose):
    if backend == 'cublas' and w_dtype != 'float32':
        pytest.skip('cublas supports float32 weights only')
    with jax_x64_enabled():
        rng = np.random.default_rng(201)
        k, m = 300, 200
        W = rng.standard_normal((k, m) if transpose else (m, k))
        x = rng.integers(-3, 4, size=k) * (rng.random(k) < 0.2)
        events = jnp.asarray(x, dtype=jnp.int8)
        Wj = jnp.asarray(W, dtype=jnp.dtype(w_dtype))
        Wf = np.asarray(Wj.astype(jnp.float64) if w_dtype == 'bfloat16' else Wj, dtype=np.float64)
        out = binary_densemv(Wj, events, transpose=transpose, backend=backend, weighted=True)
        mm = binary_densemm(Wj, jnp.stack([events, -events], 1), transpose=transpose,
                            backend=backend, weighted=True)
        ref = x @ Wf if transpose else Wf @ x
        ref_mm = (Wf.T if transpose else Wf) @ np.stack([x, -x], 1)
    # bf16 outputs accumulate atomically in bf16 (8-bit mantissa).
    tol = {'float16': 1e-2, 'bfloat16': 3e-2}.get(w_dtype, 1e-5)
    for got, r in ((out, ref), (mm, ref_mm)):
        got = np.asarray(jnp.asarray(got, jnp.float32) if w_dtype == 'bfloat16' else got, dtype=np.float64)
        rel = np.linalg.norm(got - r) / np.linalg.norm(r)
        assert rel < tol, f'relative error {rel}'
