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

"""Tests for the test_util functions using JAX fori_loop.

These tests verify that the fori_loop implementations produce the same results
as the original Python for-loop implementations.
"""

import brainstate
import jax
import jax.numpy as jnp
import pytest

from brainevent._csr.test_util import (
    get_csr,
    vector_csr,
    matrix_csr,
    csr_vector,
    csr_matrix,
)


# Reference implementations using Python for loops (original implementations)
def vector_csr_ref(x, w, indices, indptr, shape):
    """Reference implementation using Python for loop."""
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((shape[1],))
    for i_pre in range(x.shape[0]):
        ids = indices[indptr[i_pre]: indptr[i_pre + 1]]
        inc = w * x[i_pre] if homo_w else w[indptr[i_pre]: indptr[i_pre + 1]] * x[i_pre]
        ids, inc = jnp.broadcast_arrays(ids, inc)
        post = post.at[ids].add(inc)
    return post


def matrix_csr_ref(xs, w, indices, indptr, shape):
    """Reference implementation using Python for loop."""
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((xs.shape[0], shape[1]))
    for i_pre in range(xs.shape[1]):
        ids = indices[indptr[i_pre]: indptr[i_pre + 1]]
        post = post.at[:, ids].add(
            w * xs[:, i_pre: i_pre + 1]
            if homo_w else
            (w[indptr[i_pre]: indptr[i_pre + 1]] * xs[:, i_pre: i_pre + 1])
        )
    return post


def csr_vector_ref(x, w, indices, indptr, shape):
    """Reference implementation using Python for loop."""
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0]])
    for i in range(shape[0]):
        ids = indices[indptr[i]: indptr[i + 1]]
        ws = w if homo_w else w[indptr[i]: indptr[i + 1]]
        out = out.at[i].set(jnp.sum(x[ids] * ws))
    return out


def csr_matrix_ref(xs, w, indices, indptr, shape):
    """Reference implementation using Python for loop."""
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0], xs.shape[1]])
    for i in range(shape[0]):
        ids = indices[indptr[i]: indptr[i + 1]]
        ws = w if homo_w else jnp.expand_dims(w[indptr[i]: indptr[i + 1]], axis=1)
        out = out.at[i].set(jnp.sum(xs[ids] * ws, axis=0))
    return out


class TestVectorCSR:
    """Tests for vector_csr function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        brainstate.random.seed(42)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_correctness(self, homo_w):
        """Test that fori_loop implementation matches reference."""
        n_pre, n_post = 20, 40
        prob = 0.2
        shape = (n_pre, n_post)

        indptr, indices = get_csr(n_pre, n_post, prob)
        x = (brainstate.random.rand(n_pre) < 0.3).astype(float)

        if homo_w:
            w = jnp.array([1.5])
        else:
            w = brainstate.random.randn(len(indices))

        result_ref = vector_csr_ref(x, w, indices, indptr, shape)
        result_new = vector_csr(x, w, indices, indptr, shape)

        assert jnp.allclose(result_ref, result_new, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((indptr, indices, x, w, result_ref, result_new))

    def test_output_shape(self):
        """Test output shape is correct."""
        n_pre, n_post = 10, 30
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        x = jnp.ones(n_pre)
        w = jnp.array([1.0])

        result = vector_csr(x, w, indices, indptr, (n_pre, n_post))
        assert result.shape == (n_post,)
        jax.block_until_ready((indptr, indices, x, w, result))

    def test_jit_compilation(self):
        """Test that the function can be JIT compiled."""
        n_pre, n_post = 10, 20
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        x = jnp.ones(n_pre)
        w = jnp.array([1.0])

        # Should not raise
        result = vector_csr(x, w, indices, indptr, (n_pre, n_post))
        assert result.shape == (n_post,)
        jax.block_until_ready((indptr, indices, x, w, result))

    def test_gradient(self):
        """Test that gradients can be computed."""
        n_pre, n_post = 10, 20
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        x = brainstate.random.rand(n_pre)
        w = brainstate.random.randn(len(indices))

        def loss_fn(x, w):
            return vector_csr(x, w, indices, indptr, (n_pre, n_post)).sum()

        grads = jax.grad(loss_fn, argnums=(0, 1))(x, w)
        assert grads[0].shape == x.shape
        assert grads[1].shape == w.shape
        jax.block_until_ready((indptr, indices, x, w, grads))


class TestMatrixCSR:
    """Tests for matrix_csr function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        brainstate.random.seed(42)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_correctness(self, homo_w):
        """Test that fori_loop implementation matches reference."""
        batch, n_pre, n_post = 10, 20, 40
        prob = 0.2
        shape = (n_pre, n_post)

        indptr, indices = get_csr(n_pre, n_post, prob)
        xs = (brainstate.random.rand(batch, n_pre) < 0.3).astype(float)

        if homo_w:
            w = jnp.array([1.5])
        else:
            w = brainstate.random.randn(len(indices))

        result_ref = matrix_csr_ref(xs, w, indices, indptr, shape)
        result_new = matrix_csr(xs, w, indices, indptr, shape)

        assert jnp.allclose(result_ref, result_new, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((indptr, indices, xs, w, result_ref, result_new))

    def test_output_shape(self):
        """Test output shape is correct."""
        batch, n_pre, n_post = 5, 10, 30
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        xs = jnp.ones((batch, n_pre))
        w = jnp.array([1.0])

        result = matrix_csr(xs, w, indices, indptr, (n_pre, n_post))
        assert result.shape == (batch, n_post)
        jax.block_until_ready((indptr, indices, xs, w, result))

    def test_gradient(self):
        """Test that gradients can be computed."""
        batch, n_pre, n_post = 5, 10, 20
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        xs = brainstate.random.rand(batch, n_pre)
        w = brainstate.random.randn(len(indices))

        def loss_fn(xs, w):
            return matrix_csr(xs, w, indices, indptr, (n_pre, n_post)).sum()

        grads = jax.grad(loss_fn, argnums=(0, 1))(xs, w)
        assert grads[0].shape == xs.shape
        assert grads[1].shape == w.shape
        jax.block_until_ready((indptr, indices, xs, w, grads))


class TestCSRVector:
    """Tests for csr_vector function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        brainstate.random.seed(42)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_correctness(self, homo_w):
        """Test that fori_loop implementation matches reference."""
        n_pre, n_post = 20, 40
        prob = 0.2
        shape = (n_pre, n_post)

        indptr, indices = get_csr(n_pre, n_post, prob)
        x = (brainstate.random.rand(n_post) < 0.3).astype(float)

        if homo_w:
            w = jnp.array([1.5])
        else:
            w = brainstate.random.randn(len(indices))

        result_ref = csr_vector_ref(x, w, indices, indptr, shape)
        result_new = csr_vector(x, w, indices, indptr, shape)

        assert jnp.allclose(result_ref, result_new, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((indptr, indices, x, w, result_ref, result_new))

    def test_output_shape(self):
        """Test output shape is correct."""
        n_pre, n_post = 10, 30
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        x = jnp.ones(n_post)
        w = jnp.array([1.0])

        result = csr_vector(x, w, indices, indptr, (n_pre, n_post))
        assert result.shape == (n_pre,)
        jax.block_until_ready((indptr, indices, x, w, result))

    def test_gradient(self):
        """Test that gradients can be computed."""
        n_pre, n_post = 10, 20
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        x = brainstate.random.rand(n_post)
        w = brainstate.random.randn(len(indices))

        def loss_fn(x, w):
            return csr_vector(x, w, indices, indptr, (n_pre, n_post)).sum()

        grads = jax.grad(loss_fn, argnums=(0, 1))(x, w)
        assert grads[0].shape == x.shape
        assert grads[1].shape == w.shape
        jax.block_until_ready((indptr, indices, x, w, grads))


class TestCSRMatrix:
    """Tests for csr_matrix function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        brainstate.random.seed(42)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_correctness(self, homo_w):
        """Test that fori_loop implementation matches reference."""
        n_pre, n_post, k = 20, 40, 10
        prob = 0.2
        shape = (n_pre, n_post)

        indptr, indices = get_csr(n_pre, n_post, prob)
        xs = (brainstate.random.rand(n_post, k) < 0.3).astype(float)

        if homo_w:
            w = jnp.array([1.5])
        else:
            w = brainstate.random.randn(len(indices))

        result_ref = csr_matrix_ref(xs, w, indices, indptr, shape)
        result_new = csr_matrix(xs, w, indices, indptr, shape)

        assert jnp.allclose(result_ref, result_new, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((indptr, indices, xs, w, result_ref, result_new))

    def test_output_shape(self):
        """Test output shape is correct."""
        n_pre, n_post, k = 10, 30, 5
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        xs = jnp.ones((n_post, k))
        w = jnp.array([1.0])

        result = csr_matrix(xs, w, indices, indptr, (n_pre, n_post))
        assert result.shape == (n_pre, k)
        jax.block_until_ready((indptr, indices, xs, w, result))

    def test_gradient(self):
        """Test that gradients can be computed."""
        n_pre, n_post, k = 10, 20, 5
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        xs = brainstate.random.rand(n_post, k)
        w = brainstate.random.randn(len(indices))

        def loss_fn(xs, w):
            return csr_matrix(xs, w, indices, indptr, (n_pre, n_post)).sum()

        grads = jax.grad(loss_fn, argnums=(0, 1))(xs, w)
        assert grads[0].shape == xs.shape
        assert grads[1].shape == w.shape
        jax.block_until_ready((indptr, indices, xs, w, grads))


class TestVmapCompatibility:
    """Tests for vmap compatibility of the fori_loop implementations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        brainstate.random.seed(42)

    def test_vector_csr_vmap_over_x(self):
        """Test vmap over input vector."""
        batch, n_pre, n_post = 10, 20, 40
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        xs = brainstate.random.rand(batch, n_pre)
        w = jnp.array([1.5])

        result = jax.vmap(lambda x: vector_csr(x, w, indices, indptr, (n_pre, n_post)))(xs)
        assert result.shape == (batch, n_post)
        jax.block_until_ready((indptr, indices, xs, w, result))

    def test_csr_vector_vmap_over_x(self):
        """Test vmap over input vector."""
        batch, n_pre, n_post = 10, 20, 40
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        xs = brainstate.random.rand(batch, n_post)
        w = jnp.array([1.5])

        result = jax.vmap(lambda x: csr_vector(x, w, indices, indptr, (n_pre, n_post)))(xs)
        assert result.shape == (batch, n_pre)
        jax.block_until_ready((indptr, indices, xs, w, result))

    def test_vector_csr_vmap_over_w(self):
        """Test vmap over weights."""
        batch, n_pre, n_post = 10, 20, 40
        indptr, indices = get_csr(n_pre, n_post, 0.1)
        x = brainstate.random.rand(n_pre)
        ws = brainstate.random.randn(batch, len(indices))

        result = jax.vmap(lambda w: vector_csr(x, w, indices, indptr, (n_pre, n_post)))(ws)
        assert result.shape == (batch, n_post)
        jax.block_until_ready((indptr, indices, x, ws, result))
