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

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.slice import csr_slice_rows, csr_slice_rows_p
from brainevent._csr.main import CSR, CSC
from brainevent._csr.test_util import get_csr

platform = jax.default_backend()
SLICE_IMPLEMENTATIONS = tuple(csr_slice_rows_p.available_backends(platform))


def _make_csr_and_dense(m, n, prob=0.3):
    """Create a CSR matrix and its dense equivalent for testing."""
    indptr, indices = get_csr(m, n, prob, replace=False)
    data = jnp.asarray(np.random.randn(indices.shape[0]).astype(np.float32))
    dense = np.zeros((m, n), dtype=np.float32)
    indptr_np = np.asarray(indptr)
    indices_np = np.asarray(indices)
    data_np = np.asarray(data)
    for i in range(m):
        for j in range(indptr_np[i], indptr_np[i + 1]):
            dense[i, indices_np[j]] += data_np[j]
    return data, indices, indptr, jnp.asarray(dense)


@pytest.mark.skipif(
    not SLICE_IMPLEMENTATIONS,
    reason=f'No csr_slice_rows implementation on platform={platform}',
)
class TestCSRSliceRows:

    @pytest.mark.parametrize('implementation', SLICE_IMPLEMENTATIONS)
    def test_single_row(self, implementation):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        for row in [0, 3, m - 1]:
            row_indices = jnp.array([row], dtype=jnp.int32)
            result = csr_slice_rows(data, indices, indptr, row_indices,
                                    shape=(m, n), backend=implementation)
            expected = dense[row:row + 1]
            assert jnp.allclose(result, expected, atol=1e-5), \
                f"Row {row}: {result} != {expected}"

    @pytest.mark.parametrize('implementation', SLICE_IMPLEMENTATIONS)
    def test_multi_row(self, implementation):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        rows = [0, 3, 7]
        row_indices = jnp.array(rows, dtype=jnp.int32)
        result = csr_slice_rows(data, indices, indptr, row_indices,
                                shape=(m, n), backend=implementation)
        expected = dense[jnp.array(rows)]
        assert jnp.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize('implementation', SLICE_IMPLEMENTATIONS)
    def test_array_index(self, implementation):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.array([0, 2, 4, 6, 8], dtype=jnp.int32)
        result = csr_slice_rows(data, indices, indptr, row_indices,
                                shape=(m, n), backend=implementation)
        expected = dense[np.array([0, 2, 4, 6, 8])]
        assert jnp.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize('implementation', SLICE_IMPLEMENTATIONS)
    def test_duplicate_rows(self, implementation):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.array([2, 2, 5, 5], dtype=jnp.int32)
        result = csr_slice_rows(data, indices, indptr, row_indices,
                                shape=(m, n), backend=implementation)
        expected = dense[np.array([2, 2, 5, 5])]
        assert jnp.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize('implementation', SLICE_IMPLEMENTATIONS)
    def test_oob_row(self, implementation):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.array([999], dtype=jnp.int32)
        result = csr_slice_rows(data, indices, indptr, row_indices,
                                shape=(m, n), backend=implementation)
        expected = jnp.zeros((1, n), dtype=data.dtype)
        assert jnp.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize('implementation', SLICE_IMPLEMENTATIONS)
    def test_all_rows(self, implementation):
        m, n = 8, 12
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.arange(m, dtype=jnp.int32)
        result = csr_slice_rows(data, indices, indptr, row_indices,
                                shape=(m, n), backend=implementation)
        assert jnp.allclose(result, dense, atol=1e-5)


def _make_homo_csr_and_dense(m, n, prob=0.3):
    """Create a homogeneous-weight CSR matrix and its dense equivalent."""
    indptr, indices = get_csr(m, n, prob, replace=False)
    data = jnp.array([1.5], dtype=jnp.float32)
    dense = np.zeros((m, n), dtype=np.float32)
    indptr_np = np.asarray(indptr)
    indices_np = np.asarray(indices)
    for i in range(m):
        for j in range(indptr_np[i], indptr_np[i + 1]):
            dense[i, indices_np[j]] += 1.5
    return data, indices, indptr, jnp.asarray(dense)


@pytest.mark.skipif(
    not SLICE_IMPLEMENTATIONS,
    reason=f'No csr_slice_rows implementation on platform={platform}',
)
class TestCSRSliceRowsHomo:
    """Tests for scalar (homogeneous) data support."""

    @pytest.mark.parametrize('implementation', SLICE_IMPLEMENTATIONS)
    def test_homo_single_row(self, implementation):
        m, n = 10, 15
        data, indices, indptr, dense = _make_homo_csr_and_dense(m, n)
        for row in [0, 3, m - 1]:
            row_indices = jnp.array([row], dtype=jnp.int32)
            result = csr_slice_rows(data, indices, indptr, row_indices, shape=(m, n), backend=implementation)
            expected = dense[row:row + 1]
            assert jnp.allclose(result, expected, atol=1e-5), f"Row {row}: {result} != {expected}"
            row = jnp.array(row, dtype=jnp.int32)
            result2 = csr_slice_rows(data, indices, indptr, row, shape=(m, n), backend=implementation)
            assert jnp.allclose(result2, expected[0], atol=1e-5)

    @pytest.mark.parametrize('implementation', SLICE_IMPLEMENTATIONS)
    def test_homo_multi_row(self, implementation):
        m, n = 10, 15
        data, indices, indptr, dense = _make_homo_csr_and_dense(m, n)
        rows = [0, 3, 7]
        row_indices = jnp.array(rows, dtype=jnp.int32)
        result = csr_slice_rows(data, indices, indptr, row_indices,
                                shape=(m, n), backend=implementation)
        expected = dense[jnp.array(rows)]
        assert jnp.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize('implementation', SLICE_IMPLEMENTATIONS)
    def test_homo_all_rows(self, implementation):
        m, n = 8, 12
        data, indices, indptr, dense = _make_homo_csr_and_dense(m, n)
        row_indices = jnp.arange(m, dtype=jnp.int32)
        result = csr_slice_rows(data, indices, indptr, row_indices,
                                shape=(m, n), backend=implementation)
        assert jnp.allclose(result, dense, atol=1e-5)

    def test_homo_vjp(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_homo_csr_and_dense(m, n)
        row_indices = jnp.array([1, 3, 5], dtype=jnp.int32)

        def loss_fn(d):
            sliced = csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n))
            return jnp.sum(sliced ** 2)

        grad_data = jax.grad(loss_fn)(data)
        assert grad_data.shape == (1,)

        eps = 1e-3
        fd = (loss_fn(data + eps) - loss_fn(data - eps)) / (2 * eps)
        assert abs(float(grad_data[0]) - float(fd)) < 1e-2, \
            f"Grad mismatch: analytic={grad_data[0]}, fd={fd}"

    def test_homo_jvp(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_homo_csr_and_dense(m, n)
        row_indices = jnp.array([0, 2, 4], dtype=jnp.int32)
        data_dot = jnp.ones_like(data)

        primals, tangents = jax.jvp(
            lambda d: csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n)),
            (data,),
            (data_dot,),
        )
        expected_tangent = csr_slice_rows(data_dot, indices, indptr, row_indices, shape=(m, n))
        assert jnp.allclose(tangents, expected_tangent, atol=1e-5)


@pytest.mark.skipif(
    not SLICE_IMPLEMENTATIONS,
    reason=f'No csr_slice_rows implementation on platform={platform}',
)
class TestCSRGetitem:

    def test_single_int_index(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        csr = CSR(data, indices, indptr, shape=(m, n))
        result = csr[3]
        expected = dense[3]
        assert result.shape == (n,)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_tuple_index(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        csr = CSR(data, indices, indptr, shape=(m, n))
        result = csr[(0, 3, 7)]
        expected = dense[np.array([0, 3, 7])]
        assert result.shape == (3, n)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_list_index(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        csr = CSR(data, indices, indptr, shape=(m, n))
        result = csr[[1, 5, 9]]
        expected = dense[np.array([1, 5, 9])]
        assert result.shape == (3, n)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_array_index(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        csr = CSR(data, indices, indptr, shape=(m, n))
        idx = jnp.array([0, 2, 4], dtype=jnp.int32)
        result = csr[idx]
        expected = dense[np.array([0, 2, 4])]
        assert result.shape == (3, n)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_jit_single(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        csr = CSR(data, indices, indptr, shape=(m, n))
        result = jax.jit(lambda c: c[0])(csr)
        expected = dense[0]
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_jit_multi(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        csr = CSR(data, indices, indptr, shape=(m, n))
        idx = jnp.array([1, 3, 5], dtype=jnp.int32)
        result = jax.jit(lambda c, i: c[i])(csr, idx)
        expected = dense[np.array([1, 3, 5])]
        assert jnp.allclose(result, expected, atol=1e-5)


@pytest.mark.skipif(
    not SLICE_IMPLEMENTATIONS,
    reason=f'No csr_slice_rows implementation on platform={platform}',
)
class TestCSRSliceAD:

    def test_jvp(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.array([1, 3, 5], dtype=jnp.int32)
        data_dot = jnp.ones_like(data)

        primals, tangents = jax.jvp(
            lambda d: csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n)),
            (data,),
            (data_dot,),
        )
        # Since the operation is linear in data, tangent should equal the
        # slice applied to data_dot
        expected_tangent = csr_slice_rows(data_dot, indices, indptr, row_indices, shape=(m, n))
        assert jnp.allclose(tangents, expected_tangent, atol=1e-5)

    def test_vjp(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.array([1, 3, 5], dtype=jnp.int32)

        def loss_fn(d):
            sliced = csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n))
            return jnp.sum(sliced ** 2)

        grad_data = jax.grad(loss_fn)(data)
        assert grad_data.shape == data.shape

        # Verify numerically with finite differences
        eps = 1e-3
        for idx in [0, data.shape[0] // 2, data.shape[0] - 1]:
            data_plus = data.at[idx].add(eps)
            data_minus = data.at[idx].add(-eps)
            fd = (loss_fn(data_plus) - loss_fn(data_minus)) / (2 * eps)
            assert abs(float(grad_data[idx]) - float(fd)) < 1e-2, \
                f"Grad mismatch at {idx}: analytic={grad_data[idx]}, fd={fd}"

    def test_grad_through_sum(self):
        m, n = 10, 15
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.array([0, 5], dtype=jnp.int32)

        def loss_fn(d):
            return jnp.sum(csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n)))

        grad_data = jax.grad(loss_fn)(data)
        assert grad_data.shape == data.shape


@pytest.mark.skipif(
    not SLICE_IMPLEMENTATIONS,
    reason=f'No csr_slice_rows implementation on platform={platform}',
)
class TestCSRSliceGradAD:
    """Tests for JVP and transpose rules on the gradient primitive itself."""

    def test_grad_of_grad(self):
        """Second-order gradient: grad(grad(loss)) should work via the grad primitive's transpose rule."""
        m, n = 8, 10
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.array([1, 3], dtype=jnp.int32)

        def loss_fn(d):
            sliced = csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n))
            return jnp.sum(sliced ** 3)

        # First-order gradient
        grad_fn = jax.grad(loss_fn)
        grad_data = grad_fn(data)
        assert grad_data.shape == data.shape

        # Second-order gradient (Hessian-vector product via grad-of-grad)
        def grad_sum(d):
            return jnp.sum(jax.grad(loss_fn)(d))

        hessian_diag_sum = jax.grad(grad_sum)(data)
        assert hessian_diag_sum.shape == data.shape

        # Verify numerically with finite differences on the gradient
        eps = 1e-3
        for idx in [0, data.shape[0] // 2]:
            data_plus = data.at[idx].add(eps)
            data_minus = data.at[idx].add(-eps)
            fd = (grad_fn(data_plus)[idx] - grad_fn(data_minus)[idx]) / (2 * eps)
            assert abs(float(hessian_diag_sum[idx]) - float(fd)) < 1e-1, \
                f"Second-order grad mismatch at {idx}: analytic={hessian_diag_sum[idx]}, fd={fd}"

    def test_jvp_through_grad(self):
        """JVP through the gradient primitive: jvp(grad(loss)) should work via the grad primitive's JVP rule."""
        m, n = 8, 10
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.array([0, 2, 4], dtype=jnp.int32)

        def loss_fn(d):
            sliced = csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n))
            return jnp.sum(sliced ** 2)

        grad_fn = jax.grad(loss_fn)
        data_dot = jnp.ones_like(data)

        # JVP through the gradient function
        primals_out, tangents_out = jax.jvp(grad_fn, (data,), (data_dot,))
        assert primals_out.shape == data.shape
        assert tangents_out.shape == data.shape

        # Verify numerically: tangent should approximate directional derivative of grad
        eps = 1e-4
        fd = (grad_fn(data + eps * data_dot) - grad_fn(data - eps * data_dot)) / (2 * eps)
        assert jnp.allclose(tangents_out, fd, atol=1e-2, rtol=1e-2), \
            f"JVP through grad mismatch: max diff={jnp.max(jnp.abs(tangents_out - fd))}"

    def test_grad_linear_loss(self):
        """For a linear loss (sum of sliced rows), grad is constant, so grad-of-grad should be zero."""
        m, n = 8, 10
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        row_indices = jnp.array([1, 5], dtype=jnp.int32)

        def linear_loss(d):
            return jnp.sum(csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n)))

        # First-order gradient of a linear function is constant
        grad_data = jax.grad(linear_loss)(data)
        assert grad_data.shape == data.shape

        # Second-order gradient of a linear function is zero
        def grad_sum(d):
            return jnp.sum(jax.grad(linear_loss)(d))

        grad2 = jax.grad(grad_sum)(data)
        assert jnp.allclose(grad2, jnp.zeros_like(grad2), atol=1e-6)


@pytest.mark.skipif(
    not SLICE_IMPLEMENTATIONS,
    reason=f'No csr_slice_rows implementation on platform={platform}',
)
class TestCSRSliceBatching:
    """Tests for vmap (batching) through csr_slice_rows and its gradient."""

    def test_vmap_over_data(self):
        """vmap over data: batch of CSR matrices (same sparsity, different values)."""
        m, n = 8, 10
        _, indices, indptr, _ = _make_csr_and_dense(m, n)
        batch_size = 3
        batch_data = jnp.asarray(np.random.randn(batch_size, indices.shape[0]).astype(np.float32))
        row_indices = jnp.array([1, 3, 5], dtype=jnp.int32)

        result = jax.vmap(
            lambda d: csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n))
        )(batch_data)
        assert result.shape == (batch_size, 3, n)

        for b in range(batch_size):
            expected = csr_slice_rows(batch_data[b], indices, indptr, row_indices, shape=(m, n))
            assert jnp.allclose(result[b], expected, atol=1e-5)

    def test_vmap_over_row_indices(self):
        """vmap over row_indices: different row subsets from the same CSR matrix."""
        m, n = 8, 10
        data, indices, indptr, dense = _make_csr_and_dense(m, n)
        batch_row_indices = jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)

        result = jax.vmap(
            lambda ri: csr_slice_rows(data, indices, indptr, ri, shape=(m, n))
        )(batch_row_indices)
        assert result.shape == (3, 2, n)

        for b in range(3):
            expected = csr_slice_rows(data, indices, indptr, batch_row_indices[b], shape=(m, n))
            assert jnp.allclose(result[b], expected, atol=1e-5)

    def test_vmap_grad_over_data(self):
        """vmap(grad) over data: batched gradient computation."""
        m, n = 8, 10
        _, indices, indptr, _ = _make_csr_and_dense(m, n)
        batch_size = 3
        batch_data = jnp.asarray(np.random.randn(batch_size, indices.shape[0]).astype(np.float32))
        row_indices = jnp.array([1, 3], dtype=jnp.int32)

        def loss(d):
            return jnp.sum(csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n)) ** 2)

        batched_grad = jax.vmap(jax.grad(loss))(batch_data)
        assert batched_grad.shape == batch_data.shape

        for b in range(batch_size):
            expected_grad = jax.grad(loss)(batch_data[b])
            assert jnp.allclose(batched_grad[b], expected_grad, atol=1e-5)

    def test_vmap_jit_combined(self):
        """vmap + jit combined."""
        m, n = 8, 10
        _, indices, indptr, _ = _make_csr_and_dense(m, n)
        batch_size = 4
        batch_data = jnp.asarray(np.random.randn(batch_size, indices.shape[0]).astype(np.float32))
        row_indices = jnp.array([0, 2, 4], dtype=jnp.int32)

        @jax.jit
        @jax.vmap
        def batched_slice(d):
            return csr_slice_rows(d, indices, indptr, row_indices, shape=(m, n))

        result = batched_slice(batch_data)
        assert result.shape == (batch_size, 3, n)

        for b in range(batch_size):
            expected = csr_slice_rows(batch_data[b], indices, indptr, row_indices, shape=(m, n))
            assert jnp.allclose(result[b], expected, atol=1e-5)


@pytest.mark.skipif(
    not SLICE_IMPLEMENTATIONS,
    reason=f'No csr_slice_rows implementation on platform={platform}',
)
class TestCSCGetitem:

    def test_single_col(self):
        m, n = 10, 15
        dense = np.random.randn(m, n).astype(np.float32)
        csc = CSC.fromdense(jnp.asarray(dense))
        result = csc[3]
        expected = dense[:, 3]
        assert result.shape == (m,)
        assert jnp.allclose(result, jnp.asarray(expected), atol=1e-5)

    def test_multi_col(self):
        m, n = 10, 15
        dense = np.random.randn(m, n).astype(np.float32)
        csc = CSC.fromdense(jnp.asarray(dense))
        result = csc[(0, 3, 7)]
        expected = dense[:, [0, 3, 7]]
        assert result.shape == (m, 3)
        assert jnp.allclose(result, jnp.asarray(expected), atol=1e-5)

    def test_array_col(self):
        m, n = 10, 15
        dense = np.random.randn(m, n).astype(np.float32)
        csc = CSC.fromdense(jnp.asarray(dense))
        idx = jnp.array([1, 5, 9], dtype=jnp.int32)
        result = csc[idx]
        expected = dense[:, [1, 5, 9]]
        assert result.shape == (m, 3)
        assert jnp.allclose(result, jnp.asarray(expected), atol=1e-5)
