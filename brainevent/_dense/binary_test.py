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

from brainevent._dense.binary import dbmm, dbmm_p, bdmm, bdmm_p, dbmv, dbmv_p, bdvm, bdvm_p

platform = jax.default_backend()
DBMV_IMPLEMENTATIONS = tuple(dbmv_p.available_backends(platform))
BDVM_IMPLEMENTATIONS = tuple(bdvm_p.available_backends(platform))
DBMM_IMPLEMENTATIONS = tuple(dbmm_p.available_backends(platform))
BDMM_IMPLEMENTATIONS = tuple(bdmm_p.available_backends(platform))


# ---- Forward: dense matrix @ binary vector (dbmv) ----

@pytest.mark.parametrize("implementation", DBMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15, 20])
@pytest.mark.parametrize("dtype", [bool, float])
def test_dbmv_forward(implementation, m, k, dtype):
    weights = brainstate.random.randn(m, k)
    spikes = brainstate.random.randn(k) < 0.3
    if dtype == float:
        spikes = u.math.asarray(spikes, dtype=float)
    result = dbmv(weights, spikes, backend=implementation)
    expected = weights @ u.math.asarray(spikes, dtype=float)
    assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
    jax.block_until_ready((weights, spikes, result, expected))


# ---- Forward: binary vector @ dense matrix (bdvm) ----

@pytest.mark.parametrize("implementation", BDVM_IMPLEMENTATIONS)
@pytest.mark.parametrize("k", [15, 20])
@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("dtype", [bool, float])
def test_bdvm_forward(implementation, k, n, dtype):
    spikes = brainstate.random.randn(k) < 0.3
    if dtype == float:
        spikes = u.math.asarray(spikes, dtype=float)
    weights = brainstate.random.randn(k, n)
    result = bdvm(spikes, weights, backend=implementation)
    expected = u.math.asarray(spikes, dtype=float) @ weights
    assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
    jax.block_until_ready((spikes, weights, result, expected))


# ---- Forward: dense matrix @ binary matrix (dbmm) ----

@pytest.mark.parametrize("implementation", DBMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15, 20])
@pytest.mark.parametrize("n", [30])
@pytest.mark.parametrize("dtype", [bool, float])
def test_dbmm_forward(implementation, m, k, n, dtype):
    weights = brainstate.random.randn(m, k)
    spikes = brainstate.random.randn(k, n) < 0.3
    if dtype == float:
        spikes = u.math.asarray(spikes, dtype=float)
    result = dbmm(weights, spikes, backend=implementation)
    expected = weights @ u.math.asarray(spikes, dtype=float)
    assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
    jax.block_until_ready((weights, spikes, result, expected))


# ---- Forward: binary matrix @ dense matrix (bdmm) ----

@pytest.mark.parametrize("implementation", BDMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15, 20])
@pytest.mark.parametrize("n", [30])
@pytest.mark.parametrize("dtype", [bool, float])
def test_bdmm_forward(implementation, m, k, n, dtype):
    spikes = brainstate.random.randn(m, k) < 0.3
    if dtype == float:
        spikes = u.math.asarray(spikes, dtype=float)
    weights = brainstate.random.randn(k, n)
    result = bdmm(spikes, weights, backend=implementation)
    expected = u.math.asarray(spikes, dtype=float) @ weights
    assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
    jax.block_until_ready((spikes, weights, result, expected))


# ---- Gradient: dbmv ----

@pytest.mark.parametrize("implementation", DBMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
def test_dbmv_grad_weights(implementation, m, k):
    weights = brainstate.random.randn(m, k)
    spikes = u.math.asarray(brainstate.random.randn(k) < 0.3, dtype=float)

    def f(w):
        return dbmv(w, spikes, backend=implementation).sum()

    grad = jax.grad(f)(weights)
    assert grad.shape == weights.shape
    jax.block_until_ready((weights, spikes, grad))


# ---- Gradient: bdvm ----

@pytest.mark.parametrize("implementation", BDVM_IMPLEMENTATIONS)
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
def test_bdvm_grad_weights(implementation, k, n):
    spikes = u.math.asarray(brainstate.random.randn(k) < 0.3, dtype=float)
    weights = brainstate.random.randn(k, n)

    def f(w):
        return bdvm(spikes, w, backend=implementation).sum()

    grad = jax.grad(f)(weights)
    assert grad.shape == weights.shape
    jax.block_until_ready((spikes, weights, grad))


# ---- Gradient: dbmm ----

@pytest.mark.parametrize("implementation", DBMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
def test_dbmm_grad_weights(implementation, m, k, n):
    weights = brainstate.random.randn(m, k)
    spikes = u.math.asarray(brainstate.random.randn(k, n) < 0.3, dtype=float)

    def f(w):
        return dbmm(w, spikes, backend=implementation).sum()

    grad = jax.grad(f)(weights)
    assert grad.shape == weights.shape
    jax.block_until_ready((weights, spikes, grad))


# ---- Gradient: bdmm ----

@pytest.mark.parametrize("implementation", BDMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
def test_bdmm_grad_weights(implementation, m, k, n):
    spikes = u.math.asarray(brainstate.random.randn(m, k) < 0.3, dtype=float)
    weights = brainstate.random.randn(k, n)

    def f(w):
        return bdmm(spikes, w, backend=implementation).sum()

    grad = jax.grad(f)(weights)
    assert grad.shape == weights.shape
    jax.block_until_ready((spikes, weights, grad))


# ---- Batching (vmap): dbmv ----

@pytest.mark.parametrize("implementation", DBMV_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("batch_size", [5])
def test_dbmv_vmap_over_spikes(implementation, m, k, batch_size):
    weights = brainstate.random.randn(m, k)
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, k) < 0.3, dtype=float
    )
    batched_fn = jax.vmap(lambda s: dbmv(weights, s, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, m)
    jax.block_until_ready((weights, batched_spikes, result))


# ---- Batching (vmap): bdvm ----

@pytest.mark.parametrize("implementation", BDVM_IMPLEMENTATIONS)
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("batch_size", [5])
def test_bdvm_vmap_over_spikes(implementation, k, n, batch_size):
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, k) < 0.3, dtype=float
    )
    weights = brainstate.random.randn(k, n)
    batched_fn = jax.vmap(lambda s: bdvm(s, weights, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, n)
    jax.block_until_ready((batched_spikes, weights, result))


# ---- Batching (vmap): dbmm ----

@pytest.mark.parametrize("implementation", DBMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("batch_size", [5])
def test_dbmm_vmap_over_spikes(implementation, m, k, n, batch_size):
    weights = brainstate.random.randn(m, k)
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, k, n) < 0.3, dtype=float
    )
    batched_fn = jax.vmap(lambda s: dbmm(weights, s, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, m, n)
    jax.block_until_ready((weights, batched_spikes, result))


# ---- Batching (vmap): bdmm ----

@pytest.mark.parametrize("implementation", BDMM_IMPLEMENTATIONS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("k", [15])
@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("batch_size", [5])
def test_bdmm_vmap_over_spikes(implementation, m, k, n, batch_size):
    batched_spikes = u.math.asarray(
        brainstate.random.randn(batch_size, m, k) < 0.3, dtype=float
    )
    weights = brainstate.random.randn(k, n)
    batched_fn = jax.vmap(lambda s: bdmm(s, weights, backend=implementation))
    result = batched_fn(batched_spikes)
    assert result.shape == (batch_size, m, n)
    jax.block_until_ready((batched_spikes, weights, result))
