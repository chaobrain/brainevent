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
import pytest

from brainevent._dense.binary import (
    binary_densemv, binary_densemv_p,
    binary_densemm, binary_densemm_p,
)

jax.config.update('jax_default_matmul_precision', 'highest')

platform = jax.default_backend()
DENSEMV_IMPLEMENTATIONS = tuple(binary_densemv_p.available_backends(platform))
DENSEMM_IMPLEMENTATIONS = tuple(binary_densemm_p.available_backends(platform))


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


# ---- Forward: binary matrix @ dense matrix (transpose=True) ----

@pytest.mark.parametrize("implementation", DENSEMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15, 20])
@pytest.mark.parametrize("n", [30])
@pytest.mark.parametrize("dtype", [bool, float])
def test_densemm_forward_transpose(implementation, m, k, n, dtype):
    spikes = brainstate.random.randn(m, k) < 0.3
    if dtype == float:
        spikes = u.math.asarray(spikes, dtype=float)
    weights = brainstate.random.randn(k, n)
    result = binary_densemm(weights, spikes, transpose=True, backend=implementation)
    expected = u.math.asarray(spikes, dtype=float) @ weights
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
    spikes = u.math.asarray(brainstate.random.randn(m, k) < 0.3, dtype=float)
    weights = brainstate.random.randn(k, n)

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
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, m, k) < 0.3, dtype=float
    )
    weights = brainstate.random.randn(k, n)
    batched_fn = jax.vmap(lambda s: binary_densemm(weights, s, transpose=True, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, m, n)
    jax.block_until_ready((batched_spikes, weights, result))
