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

import jax
import jax.numpy as jnp
import numpy as np

import brainevent
from brainevent._dense.indexed_binary import (
    binary_vec_dot_dense_mat,
    dense_mat_dot_binary_vec,
    binary_mat_dot_dense_mat,
)
from brainevent._dense.plasticity import dense_on_pre, dense_on_post
from brainevent._dense.sparse_float import (
    dense_mat_dot_sparse_float_vec,
    sparse_float_vec_dot_dense_mat,
    dense_mat_dot_sparse_float_mat,
    sparse_float_mat_dot_dense_mat,
)


class _FakeIndexedBinary:
    def __init__(self, value, indices, count):
        self.value = value
        self.spike_indices = indices
        self.spike_count = count


def _matrix_indices(spikes):
    batch, n_in = spikes.shape
    indices = np.zeros((batch, n_in), dtype=np.int32)
    count = np.zeros((batch,), dtype=np.int32)
    for i in range(batch):
        idx = np.nonzero(spikes[i])[0]
        count[i] = idx.size
        if idx.size:
            indices[i, : idx.size] = idx
    return indices, count


def test_indexed_binary_vector_forward_and_grads():
    rng = np.random.default_rng(0)
    n_in, n_out = 6, 4
    spikes = rng.random(n_in) < 0.5
    weights = jnp.asarray(rng.normal(size=(n_in, n_out)).astype(np.float32))
    idx = brainevent.IndexedBinary(jnp.asarray(spikes))

    out = binary_vec_dot_dense_mat(idx, weights)
    mask = jnp.asarray(spikes, dtype=weights.dtype)
    expected = (weights * mask[:, None]).sum(axis=0)
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)

    weights2 = jnp.asarray(rng.normal(size=(n_out, n_in)).astype(np.float32))
    out2 = dense_mat_dot_binary_vec(weights2, idx)
    expected2 = (weights2 * mask[None, :]).sum(axis=1)
    assert jnp.allclose(out2, expected2, rtol=1e-6, atol=1e-6)

    w_dot = jnp.asarray(rng.normal(size=(n_in, n_out)).astype(np.float32))
    _, out_dot = jax.jvp(lambda w: binary_vec_dot_dense_mat(idx, w), (weights,), (w_dot,))
    expected_dot = (w_dot * mask[:, None]).sum(axis=0)
    assert jnp.allclose(out_dot, expected_dot, rtol=1e-6, atol=1e-6)

    grad = jax.grad(lambda w: binary_vec_dot_dense_mat(idx, w).sum())(weights)
    mask = jnp.asarray(spikes, dtype=weights.dtype)
    expected_grad = mask[:, None] * jnp.ones((n_out,), dtype=weights.dtype)
    assert jnp.allclose(grad, expected_grad, rtol=1e-6, atol=1e-6)


def test_indexed_binary_vector_batching():
    rng = np.random.default_rng(1)
    n_in, n_out = 5, 3
    spikes = rng.random(n_in) < 0.4
    idx = brainevent.IndexedBinary(jnp.asarray(spikes))
    weights_batch = jnp.asarray(rng.normal(size=(3, n_in, n_out)).astype(np.float32))

    out = jax.vmap(lambda w: binary_vec_dot_dense_mat(idx, w))(weights_batch)
    mask = jnp.asarray(spikes, dtype=weights_batch.dtype)
    expected = jnp.stack([(w * mask[:, None]).sum(axis=0) for w in weights_batch])
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_indexed_binary_matrix_forward():
    rng = np.random.default_rng(2)
    batch, n_in, n_out = 3, 6, 4
    spikes = rng.random((batch, n_in)) < 0.3
    indices, count = _matrix_indices(spikes)
    fake = _FakeIndexedBinary(
        jnp.asarray(spikes),
        jnp.asarray(indices),
        jnp.asarray(count),
    )
    weights = jnp.asarray(rng.normal(size=(n_in, n_out)).astype(np.float32))

    out = binary_mat_dot_dense_mat(fake, weights)
    expected = jnp.stack([
        (weights * jnp.asarray(spikes[i], dtype=weights.dtype)[:, None]).sum(axis=0)
        for i in range(batch)
    ])
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_dense_plasticity_forward_grads_and_batching():
    rng = np.random.default_rng(3)
    n_pre, n_post = 5, 4
    weight = jnp.asarray(rng.normal(size=(n_pre, n_post)).astype(np.float32))
    pre_spike = jnp.asarray(rng.random(n_pre) < 0.5)
    post_trace = jnp.asarray(rng.normal(size=(n_post,)).astype(np.float32))

    out = dense_on_pre(weight, pre_spike, post_trace)
    expected = weight + pre_spike.astype(weight.dtype)[:, None] * post_trace[None, :]
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)

    w_dot = jnp.asarray(rng.normal(size=weight.shape).astype(np.float32))
    _, out_dot = jax.jvp(lambda w: dense_on_pre(w, pre_spike, post_trace), (weight,), (w_dot,))
    assert jnp.allclose(out_dot, w_dot, rtol=1e-6, atol=1e-6)

    grad = jax.grad(lambda w: dense_on_pre(w, pre_spike, post_trace).sum())(weight)
    assert jnp.allclose(grad, jnp.ones_like(weight), rtol=1e-6, atol=1e-6)

    weight_batch = jnp.asarray(rng.normal(size=(2, n_pre, n_post)).astype(np.float32))
    out_batch = jax.vmap(lambda w: dense_on_pre(w, pre_spike, post_trace))(weight_batch)
    expected_batch = weight_batch + pre_spike.astype(weight.dtype)[None, :, None] * post_trace[None, None, :]
    assert jnp.allclose(out_batch, expected_batch, rtol=1e-6, atol=1e-6)

    pre_trace = jnp.asarray(rng.normal(size=(n_pre,)).astype(np.float32))
    post_spike = jnp.asarray(rng.random(n_post) < 0.5)
    out_post = dense_on_post(weight, pre_trace, post_spike)
    expected_post = weight + pre_trace[:, None] * post_spike.astype(weight.dtype)[None, :]
    assert jnp.allclose(out_post, expected_post, rtol=1e-6, atol=1e-6)

    _, out_post_dot = jax.jvp(lambda w: dense_on_post(w, pre_trace, post_spike), (weight,), (w_dot,))
    assert jnp.allclose(out_post_dot, w_dot, rtol=1e-6, atol=1e-6)

    grad_post = jax.grad(lambda w: dense_on_post(w, pre_trace, post_spike).sum())(weight)
    assert jnp.allclose(grad_post, jnp.ones_like(weight), rtol=1e-6, atol=1e-6)

    out_post_batch = jax.vmap(lambda w: dense_on_post(w, pre_trace, post_spike))(weight_batch)
    expected_post_batch = weight_batch + pre_trace[None, :, None] * post_spike.astype(weight.dtype)[None, None, :]
    assert jnp.allclose(out_post_batch, expected_post_batch, rtol=1e-6, atol=1e-6)


def test_sparse_float_vector_forward_grads_and_batching():
    rng = np.random.default_rng(4)
    m, k, n = 5, 4, 3
    weights = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32))
    spikes = jnp.asarray(rng.normal(size=(k,)).astype(np.float32))
    spikes = spikes.at[jnp.asarray(rng.random(k) < 0.5)].set(0.0)

    out = dense_mat_dot_sparse_float_vec(weights, spikes)
    expected = weights @ spikes
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)

    w_dot = jnp.asarray(rng.normal(size=weights.shape).astype(np.float32))
    _, out_dot = jax.jvp(lambda w: dense_mat_dot_sparse_float_vec(w, spikes), (weights,), (w_dot,))
    assert jnp.allclose(out_dot, w_dot @ spikes, rtol=1e-6, atol=1e-6)

    spk_dot = jnp.asarray(rng.normal(size=spikes.shape).astype(np.float32))
    _, out_dot_spk = jax.jvp(lambda s: dense_mat_dot_sparse_float_vec(weights, s), (spikes,), (spk_dot,))
    assert jnp.allclose(out_dot_spk, weights @ spk_dot, rtol=1e-6, atol=1e-6)

    grad_w = jax.grad(lambda w: dense_mat_dot_sparse_float_vec(w, spikes).sum())(weights)
    expected_grad = jnp.tile(spikes, (m, 1))
    assert jnp.allclose(grad_w, expected_grad, rtol=1e-6, atol=1e-6)

    weights2 = jnp.asarray(rng.normal(size=(k, n)).astype(np.float32))
    out2 = sparse_float_vec_dot_dense_mat(spikes, weights2)
    expected2 = spikes @ weights2
    assert jnp.allclose(out2, expected2, rtol=1e-6, atol=1e-6)

    grad_w2 = jax.grad(lambda w: sparse_float_vec_dot_dense_mat(spikes, w).sum())(weights2)
    expected_grad2 = jnp.outer(spikes, jnp.ones((n,), dtype=weights2.dtype))
    assert jnp.allclose(grad_w2, expected_grad2, rtol=1e-6, atol=1e-6)

    spikes_batch = jnp.asarray(rng.normal(size=(2, k)).astype(np.float32))
    spikes_batch = spikes_batch.at[:, jnp.asarray(rng.random(k) < 0.5)].set(0.0)
    out_batch = jax.vmap(lambda s: dense_mat_dot_sparse_float_vec(weights, s))(spikes_batch)
    expected_batch = (weights @ spikes_batch.T).T
    assert jnp.allclose(out_batch, expected_batch, rtol=1e-6, atol=1e-6)

    out_batch2 = jax.vmap(lambda s: sparse_float_vec_dot_dense_mat(s, weights2))(spikes_batch)
    expected_batch2 = spikes_batch @ weights2
    assert jnp.allclose(out_batch2, expected_batch2, rtol=1e-6, atol=1e-6)


def test_sparse_float_matrix_forward_grads_and_batching():
    rng = np.random.default_rng(5)
    m, k, n = 4, 3, 5
    weights = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32))
    spikes = jnp.asarray(rng.normal(size=(k, n)).astype(np.float32))
    spikes = spikes.at[jnp.asarray(rng.random((k, n)) < 0.4)].set(0.0)

    out = dense_mat_dot_sparse_float_mat(weights, spikes)
    expected = weights @ spikes
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)

    spk_dot = jnp.asarray(rng.normal(size=spikes.shape).astype(np.float32))
    _, out_dot = jax.jvp(lambda s: dense_mat_dot_sparse_float_mat(weights, s), (spikes,), (spk_dot,))
    assert jnp.allclose(out_dot, weights @ spk_dot, rtol=1e-6, atol=1e-6)

    weights2 = jnp.asarray(rng.normal(size=(k, n)).astype(np.float32))
    spikes2 = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32))
    spikes2 = spikes2.at[jnp.asarray(rng.random((m, k)) < 0.4)].set(0.0)

    out2 = sparse_float_mat_dot_dense_mat(spikes2, weights2)
    expected2 = spikes2 @ weights2
    assert jnp.allclose(out2, expected2, rtol=1e-6, atol=1e-6)

    grad_w2 = jax.grad(lambda w: sparse_float_mat_dot_dense_mat(spikes2, w).sum())(weights2)
    expected_grad2 = jnp.sum(spikes2, axis=0)[:, None] * jnp.ones((1, n), dtype=weights2.dtype)
    assert jnp.allclose(grad_w2, expected_grad2, rtol=1e-6, atol=1e-6)

    spikes_batch = jnp.asarray(rng.normal(size=(2, m, k)).astype(np.float32))
    spikes_batch = spikes_batch.at[jnp.asarray(rng.random((2, m, k)) < 0.4)].set(0.0)
    out_batch = jax.vmap(lambda s: sparse_float_mat_dot_dense_mat(s, weights2))(spikes_batch)
    expected_batch = jax.vmap(lambda s: s @ weights2)(spikes_batch)
    assert jnp.allclose(out_batch, expected_batch, rtol=1e-6, atol=1e-6)

    spikes_3d = jnp.asarray(rng.normal(size=(k, 2, n)).astype(np.float32))
    spikes_3d = spikes_3d.at[jnp.asarray(rng.random((k, 2, n)) < 0.4)].set(0.0)
    out_batch2 = jax.vmap(lambda s: dense_mat_dot_sparse_float_mat(weights, s), in_axes=1, out_axes=1)(spikes_3d)
    expected_batch2 = jax.vmap(lambda s: weights @ s, in_axes=1, out_axes=1)(spikes_3d)
    assert jnp.allclose(out_batch2, expected_batch2, rtol=1e-6, atol=1e-6)
