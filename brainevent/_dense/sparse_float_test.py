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

from contextlib import contextmanager

import brainstate
import brainunit as u
import jax
import pytest

from brainevent._dense.sparse_float import (
    spfloat_densemv,
    spfloat_densemv_p,
    spfloat_densemm,
    spfloat_densemm_p,
)

jax.config.update('jax_default_matmul_precision', 'highest')

platform = jax.default_backend()
SPFLOAT_DENSEMV_IMPLEMENTATIONS = tuple(spfloat_densemv_p.available_backends(platform))
SPFLOAT_DENSEMM_IMPLEMENTATIONS = tuple(spfloat_densemm_p.available_backends(platform))


def _as_float(x):
    return u.math.asarray(x, dtype=float)


@contextmanager
def _primitive_backend(primitive, implementation):
    default_backend = primitive.get_default(platform)
    primitive.set_default(platform, implementation)
    try:
        yield
    finally:
        if default_backend is not None:
            primitive.set_default(platform, default_backend)


@pytest.mark.skipif(
    not SPFLOAT_DENSEMV_IMPLEMENTATIONS,
    reason=f'No spfloat_densemv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', SPFLOAT_DENSEMV_IMPLEMENTATIONS)
class TestDSFMV:
    @pytest.mark.parametrize('dtype', [bool, float])
    def test_forward(self, implementation, dtype):
        m, k = 12, 16
        weights = brainstate.random.randn(m, k)
        spikes = brainstate.random.randn(k) < 0.3
        if dtype is float:
            spikes = _as_float(spikes)

        result = spfloat_densemv(weights, spikes, transpose=False, backend=implementation)
        expected = weights @ _as_float(spikes)
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((weights, spikes, result, expected))

    def test_grad_weights(self, implementation):
        m, k = 12, 16
        weights = brainstate.random.randn(m, k)
        spikes = _as_float(brainstate.random.randn(k))

        def f_test(w):
            return spfloat_densemv(w, spikes, transpose=False, backend=implementation).sum()

        def f_ref(w):
            return (w @ spikes).sum()

        grad_test = jax.grad(f_test)(weights)
        grad_ref = jax.grad(f_ref)(weights)
        assert u.math.allclose(grad_test, grad_ref, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((weights, spikes, grad_test, grad_ref))

    def test_vmap_spikes(self, implementation):
        b, m, k = 5, 12, 16
        weights = brainstate.random.randn(m, k)
        spikes = _as_float(brainstate.random.randn(b, k) * (brainstate.random.randn(b, k) > 0.0))

        result = jax.vmap(lambda s: spfloat_densemv(weights, s, transpose=False, backend=implementation))(spikes)
        expected = jax.vmap(lambda s: weights @ s)(spikes)
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((weights, spikes, result, expected))


@pytest.mark.skipif(
    not SPFLOAT_DENSEMV_IMPLEMENTATIONS,
    reason=f'No spfloat_densemv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', SPFLOAT_DENSEMV_IMPLEMENTATIONS)
class TestSFDVM:
    @pytest.mark.parametrize('dtype', [bool, float])
    def test_forward(self, implementation, dtype):
        k, n = 16, 20
        spikes = brainstate.random.randn(k) < 0.3
        if dtype is float:
            spikes = _as_float(spikes)
        weights = brainstate.random.randn(k, n)

        with _primitive_backend(spfloat_densemv_p, implementation):
            result = spfloat_densemv(weights, spikes, transpose=True)
        expected = _as_float(spikes) @ weights
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((spikes, weights, result, expected))

    def test_grad_weights(self, implementation):
        k, n = 16, 20
        spikes = _as_float(brainstate.random.randn(k))
        weights = brainstate.random.randn(k, n)

        def f_ref(w):
            return (spikes @ w).sum()

        with _primitive_backend(spfloat_densemv_p, implementation):
            grad_test = jax.grad(lambda w: spfloat_densemv(w, spikes, transpose=True).sum())(weights)
        grad_ref = jax.grad(f_ref)(weights)
        assert u.math.allclose(grad_test, grad_ref, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((spikes, weights, grad_test, grad_ref))

    def test_vmap_spikes(self, implementation):
        b, k, n = 5, 16, 20
        spikes = _as_float(brainstate.random.randn(b, k) * (brainstate.random.randn(b, k) > 0.0))
        weights = brainstate.random.randn(k, n)

        with _primitive_backend(spfloat_densemv_p, implementation):
            result = jax.vmap(lambda s: spfloat_densemv(weights, s, transpose=True))(spikes)
        expected = jax.vmap(lambda s: s @ weights)(spikes)
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((spikes, weights, result, expected))


@pytest.mark.skipif(
    not SPFLOAT_DENSEMM_IMPLEMENTATIONS,
    reason=f'No spfloat_densemm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', SPFLOAT_DENSEMM_IMPLEMENTATIONS)
class TestDSFMM:
    @pytest.mark.parametrize('dtype', [bool, float])
    def test_forward(self, implementation, dtype):
        m, k, n = 12, 16, 10
        weights = brainstate.random.randn(m, k)
        spikes = brainstate.random.randn(k, n) < 0.3
        if dtype is float:
            spikes = _as_float(spikes)

        with _primitive_backend(spfloat_densemm_p, implementation):
            result = spfloat_densemm(weights, spikes, transpose=False)
        expected = weights @ _as_float(spikes)
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((weights, spikes, result, expected))

    def test_grad_weights(self, implementation):
        m, k, n = 12, 16, 10
        weights = brainstate.random.randn(m, k)
        spikes = _as_float(brainstate.random.randn(k, n))

        def f_ref(w):
            return (w @ spikes).sum()

        with _primitive_backend(spfloat_densemm_p, implementation):
            grad_test = jax.grad(lambda w: spfloat_densemm(w, spikes, transpose=False).sum())(weights)
        grad_ref = jax.grad(f_ref)(weights)
        assert u.math.allclose(grad_test, grad_ref, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((weights, spikes, grad_test, grad_ref))

    def test_vmap_spikes(self, implementation):
        b, m, k, n = 4, 12, 16, 10
        weights = brainstate.random.randn(m, k)
        spikes = _as_float(brainstate.random.randn(b, k, n) * (brainstate.random.randn(b, k, n) > 0.0))

        with _primitive_backend(spfloat_densemm_p, implementation):
            result = jax.vmap(lambda s: spfloat_densemm(weights, s, transpose=False))(spikes)
        expected = jax.vmap(lambda s: weights @ s)(spikes)
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((weights, spikes, result, expected))


@pytest.mark.skipif(
    not SPFLOAT_DENSEMM_IMPLEMENTATIONS,
    reason=f'No spfloat_densemm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', SPFLOAT_DENSEMM_IMPLEMENTATIONS)
class TestSFDMM:
    @pytest.mark.parametrize('dtype', [bool, float])
    def test_forward(self, implementation, dtype):
        m, k, n = 12, 16, 10
        spikes = brainstate.random.randn(m, k) < 0.3
        if dtype is float:
            spikes = _as_float(spikes)
        weights = brainstate.random.randn(k, n)

        with _primitive_backend(spfloat_densemm_p, implementation):
            result = spfloat_densemm(weights, spikes, transpose=True)
        expected = _as_float(spikes) @ weights
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((spikes, weights, result, expected))

    def test_grad_weights(self, implementation):
        m, k, n = 12, 16, 10
        spikes = _as_float(brainstate.random.randn(m, k))
        weights = brainstate.random.randn(k, n)

        def f_ref(w):
            return (spikes @ w).sum()

        with _primitive_backend(spfloat_densemm_p, implementation):
            grad_test = jax.grad(lambda w: spfloat_densemm(w, spikes, transpose=True).sum())(weights)
        grad_ref = jax.grad(f_ref)(weights)
        assert u.math.allclose(grad_test, grad_ref, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((spikes, weights, grad_test, grad_ref))

    def test_vmap_spikes(self, implementation):
        b, m, k, n = 4, 12, 16, 10
        spikes = _as_float(brainstate.random.randn(b, m, k) * (brainstate.random.randn(b, m, k) > 0.0))
        weights = brainstate.random.randn(k, n)

        with _primitive_backend(spfloat_densemm_p, implementation):
            result = jax.vmap(lambda s: spfloat_densemm(weights, s, transpose=True))(spikes)
        expected = jax.vmap(lambda s: s @ weights)(spikes)
        assert u.math.allclose(result, expected, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((spikes, weights, result, expected))
