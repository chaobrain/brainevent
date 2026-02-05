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
import pytest

import brainevent
from brainevent._dense.binary import (
    dm_bm,
    bm_dm,
    dm_bv,
    bv_dm,
)


class TestMatrixEvent:
    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    @pytest.mark.parametrize("asbool", [True, False])
    def test_mm(self, m, k, n, asbool):
        matrix = brainstate.random.randn(m, k)
        events = brainevent.EventArray(
            brainstate.random.randn(k, n) < 0.5
        )
        if not asbool:
            events.value = u.math.asarray(events.value, dtype=float)
        out1 = matrix @ events
        out2 = matrix @ (events.data).astype(float)
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    def test_dense_mat_dot_binary_mat(self, m, k, n):
        matrix = brainstate.random.randn(m, k)
        events = u.math.asarray(brainstate.random.randn(k, n) < 0.5, dtype=float)
        out1 = dm_bm(matrix, events)
        out2 = matrix @ events
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)


class TestEventMatrix:
    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    @pytest.mark.parametrize("asbool", [True, False])
    def test_mm(self, m, k, n, asbool):
        events = brainevent.EventArray(
            brainstate.random.randn(m, k) < 0.5
        )
        if not asbool:
            events.value = u.math.asarray(events.value, dtype=float)
        matrix = brainstate.random.randn(k, n)
        out1 = events @ matrix
        out2 = events.data @ matrix
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    def test_dense_mat_dot_binary_mat(self, m, k, n):
        events = u.math.asarray(brainstate.random.randn(m, k) < 0.5, dtype=float)
        matrix = brainstate.random.randn(k, n)
        out1 = bm_dm(events, matrix)
        out2 = events @ matrix
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)


class TestMatrixEvent_mv:
    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("asbool", [True, False])
    def test_mm(self, m, k, asbool):
        matrix = brainstate.random.randn(m, k)
        events = brainevent.EventArray(
            brainstate.random.randn(k) < 0.5
        )
        if not asbool:
            events.value = u.math.asarray(events.value, dtype=float)
        out1 = matrix @ events
        out2 = matrix @ events.data
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    def test_matrix_event_mv(self, m, k):
        matrix = brainstate.random.randn(m, k)
        events = u.math.asarray(brainstate.random.randn(k) < 0.5, dtype=float)
        out1 = dm_bv(matrix, events)
        out2 = matrix @ events
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)


class TestEventMatrix_mv:
    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("asbool", [True, False])
    def test_mm(self, m, k, asbool):
        events = brainevent.EventArray(
            brainstate.random.randn(k) < 0.5
        )
        if not asbool:
            events.value = u.math.asarray(events.value, dtype=float)
        matrix = brainstate.random.randn(k, m)
        out1 = events @ matrix
        out2 = events.data @ matrix
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    def test_matrix_event_mv(self, m, k):
        events = u.math.asarray(brainstate.random.randn(m) < 0.5, dtype=float)
        matrix = brainstate.random.randn(m, k)
        out1 = bv_dm(events, matrix)
        out2 = events @ matrix
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)


class TestForwardPass:
    """Test forward pass for all 4 operations with boolean and float spikes."""

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("dtype", [bool, float])
    def test_dense_mat_dot_binary_vec(self, m, k, dtype):
        weights = brainstate.random.randn(m, k)
        spikes = brainstate.random.randn(k) < 0.3
        if dtype == float:
            spikes = u.math.asarray(spikes, dtype=float)
        result = dm_bv(weights, spikes)
        expected = weights @ u.math.asarray(spikes, dtype=float)
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("n", [20])
    @pytest.mark.parametrize("dtype", [bool, float])
    def test_binary_vec_dot_dense_mat(self, k, n, dtype):
        spikes = brainstate.random.randn(k) < 0.3
        if dtype == float:
            spikes = u.math.asarray(spikes, dtype=float)
        weights = brainstate.random.randn(k, n)
        result = bv_dm(spikes, weights)
        expected = u.math.asarray(spikes, dtype=float) @ weights
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("n", [20])
    @pytest.mark.parametrize("dtype", [bool, float])
    def test_dense_mat_dot_binary_mat(self, m, k, n, dtype):
        weights = brainstate.random.randn(m, k)
        spikes = brainstate.random.randn(k, n) < 0.3
        if dtype == float:
            spikes = u.math.asarray(spikes, dtype=float)
        result = dm_bm(weights, spikes)
        expected = weights @ u.math.asarray(spikes, dtype=float)
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("n", [20])
    @pytest.mark.parametrize("dtype", [bool, float])
    def test_binary_mat_dot_dense_mat(self, m, k, n, dtype):
        spikes = brainstate.random.randn(m, k) < 0.3
        if dtype == float:
            spikes = u.math.asarray(spikes, dtype=float)
        weights = brainstate.random.randn(k, n)
        result = bm_dm(spikes, weights)
        expected = u.math.asarray(spikes, dtype=float) @ weights
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)


class TestGradient:
    """Test gradient computation (JVP/VJP) for all 4 operations."""

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15])
    def test_dense_mat_dot_binary_vec_grad_weights(self, m, k):
        import jax
        weights = brainstate.random.randn(m, k)
        spikes = u.math.asarray(brainstate.random.randn(k) < 0.3, dtype=float)

        def f(w):
            return dm_bv(w, spikes).sum()

        grad = jax.grad(f)(weights)
        assert grad.shape == weights.shape

    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("n", [20])
    def test_binary_vec_dot_dense_mat_grad_weights(self, k, n):
        import jax
        spikes = u.math.asarray(brainstate.random.randn(k) < 0.3, dtype=float)
        weights = brainstate.random.randn(k, n)

        def f(w):
            return bv_dm(spikes, w).sum()

        grad = jax.grad(f)(weights)
        assert grad.shape == weights.shape

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("n", [20])
    def test_dense_mat_dot_binary_mat_grad_weights(self, m, k, n):
        import jax
        weights = brainstate.random.randn(m, k)
        spikes = u.math.asarray(brainstate.random.randn(k, n) < 0.3, dtype=float)

        def f(w):
            return dm_bm(w, spikes).sum()

        grad = jax.grad(f)(weights)
        assert grad.shape == weights.shape

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("n", [20])
    def test_binary_mat_dot_dense_mat_grad_weights(self, m, k, n):
        import jax
        spikes = u.math.asarray(brainstate.random.randn(m, k) < 0.3, dtype=float)
        weights = brainstate.random.randn(k, n)

        def f(w):
            return bm_dm(spikes, w).sum()

        grad = jax.grad(f)(weights)
        assert grad.shape == weights.shape


class TestBatching:
    """Test vmap batching for all 4 operations."""

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("batch_size", [5])
    def test_dense_mat_dot_binary_vec_vmap_over_spikes(self, m, k, batch_size):
        import jax
        weights = brainstate.random.randn(m, k)
        batched_spikes = u.math.asarray(
            brainstate.random.randn(batch_size, k) < 0.3, dtype=float
        )
        batched_fn = jax.vmap(dm_bv, in_axes=(None, 0))
        result = batched_fn(weights, batched_spikes)
        assert result.shape == (batch_size, m)

    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("n", [20])
    @pytest.mark.parametrize("batch_size", [5])
    def test_binary_vec_dot_dense_mat_vmap_over_spikes(self, k, n, batch_size):
        import jax
        batched_spikes = u.math.asarray(
            brainstate.random.randn(batch_size, k) < 0.3, dtype=float
        )
        weights = brainstate.random.randn(k, n)
        batched_fn = jax.vmap(bv_dm, in_axes=(0, None))
        result = batched_fn(batched_spikes, weights)
        assert result.shape == (batch_size, n)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("n", [20])
    @pytest.mark.parametrize("batch_size", [5])
    def test_dense_mat_dot_binary_mat_vmap_over_spikes(self, m, k, n, batch_size):
        import jax
        weights = brainstate.random.randn(m, k)
        batched_spikes = u.math.asarray(
            brainstate.random.randn(batch_size, k, n) < 0.3, dtype=float
        )
        batched_fn = jax.vmap(dm_bm, in_axes=(None, 0))
        result = batched_fn(weights, batched_spikes)
        assert result.shape == (batch_size, m, n)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15])
    @pytest.mark.parametrize("n", [20])
    @pytest.mark.parametrize("batch_size", [5])
    def test_binary_mat_dot_dense_mat_vmap_over_spikes(self, m, k, n, batch_size):
        import jax
        batched_spikes = u.math.asarray(
            brainstate.random.randn(batch_size, m, k) < 0.3, dtype=float
        )
        weights = brainstate.random.randn(k, n)
        batched_fn = jax.vmap(bm_dm, in_axes=(0, None))
        result = batched_fn(batched_spikes, weights)
        assert result.shape == (batch_size, m, n)
