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


import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import pytest

import brainevent._dense.binary as dense_binary_mod
from brainevent._dense.binary import (
    binary_densemv, binary_densemv_p,
    binary_densemm, binary_densemm_p,
)

jax.config.update('jax_default_matmul_precision', 'highest')

platform = jax.default_backend()
DENSEMV_IMPLEMENTATIONS = tuple(binary_densemv_p.available_backends(platform))
DENSEMM_IMPLEMENTATIONS = tuple(binary_densemm_p.available_backends(platform))


def _skip_cublas_if_unsupported(implementation, dtype):
    if implementation == 'cublas' and dtype is not bool:
        pytest.skip("cublas backend only supports bool spikes/events")


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
    _skip_cublas_if_unsupported(implementation, dtype)
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
    _skip_cublas_if_unsupported(implementation, dtype)
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
    _skip_cublas_if_unsupported(implementation, dtype)
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
    if implementation == 'cublas':
        pytest.skip("cublas backend only supports forward bool spikes")
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
    if implementation == 'cublas':
        pytest.skip("cublas backend only supports forward bool spikes")
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
    if implementation == 'cublas':
        pytest.skip("cublas backend only supports forward bool events")
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
    if implementation == 'cublas':
        pytest.skip("cublas backend only supports forward bool events")
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
    if implementation == 'cublas':
        pytest.skip("cublas backend only supports unbatched bool spikes")
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
    if implementation == 'cublas':
        pytest.skip("cublas backend only supports unbatched bool spikes")
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
    if implementation == 'cublas':
        pytest.skip("cublas backend only supports unbatched bool events")
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
    if implementation == 'cublas':
        pytest.skip("cublas backend only supports unbatched bool events")
    weights = brainstate.random.randn(k, m)
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, k, n) < 0.3, dtype=float
    )
    batched_fn = jax.vmap(lambda s: binary_densemm(weights, s, transpose=True, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, m, n)
    jax.block_until_ready((batched_spikes, weights, result))
