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

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from brainevent._csr.binary import binary_csrmm, binary_csrmv
from brainevent._csr.binary_indexed import binary_csrmm_indexed, binary_csrmv_indexed
from brainevent._csr.float import csrmm, csrmv
from brainevent._csr.plasticity_binary import update_csr_on_binary_post, update_csr_on_binary_pre
from brainevent._csr.slice import csr_slice_rows, csr_slice_rows_grad
from brainevent._csr.yw2y import csrmv_yw2y


requires_gpu = pytest.mark.skipif(
    jax.default_backend() != 'gpu',
    reason='CUDA int64 indptr tests require a GPU backend',
)


def _structure(indptr_dtype):
    weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 4], dtype=indptr_dtype)
    return weights, indices, indptr


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_float_csrmv_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    vector = jnp.array([1.0, 2.0], dtype=jnp.float32) if transpose else jnp.array([1.0, 2.0, 3.0])

    got = csrmv(data, indices, indptr64, vector, shape=(2, 3), transpose=transpose, backend='cuda_raw')
    expected = csrmv(data, indices, indptr32, vector, shape=(2, 3), transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_float_csrmm_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    matrix = (
        jnp.array([[1.0, 0.5], [2.0, 1.5]], dtype=jnp.float32)
        if transpose else
        jnp.array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]], dtype=jnp.float32)
    )

    got = csrmm(data, indices, indptr64, matrix, shape=(2, 3), transpose=transpose, backend='cuda_raw')
    expected = csrmm(data, indices, indptr32, matrix, shape=(2, 3), transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmv_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    vector = jnp.array([True, False], dtype=jnp.bool_) if transpose else jnp.array([True, False, True])

    got = binary_csrmv(data, indices, indptr64, vector, shape=(2, 3), transpose=transpose, backend='cuda_raw')
    expected = binary_csrmv(data, indices, indptr32, vector, shape=(2, 3), transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmm_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    matrix = (
        jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        if transpose else
        jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    )

    got = binary_csrmm(data, indices, indptr64, matrix, shape=(2, 3), transpose=transpose, backend='cuda_raw')
    expected = binary_csrmm(data, indices, indptr32, matrix, shape=(2, 3), transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
def test_binary_indexed_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    vector = jnp.array([True, False], dtype=jnp.bool_) if transpose else jnp.array([True, False, True])

    got = binary_csrmv_indexed(weights, indices, indptr64, perm, vector, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw')
    expected = binary_csrmv_indexed(weights, indices, indptr32, perm, vector, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
def test_binary_indexed_csrmm_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    matrix = (
        jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        if transpose else
        jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    )

    got = binary_csrmm_indexed(weights, indices, indptr64, perm, matrix, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw')
    expected = binary_csrmm_indexed(weights, indices, indptr32, perm, matrix, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_slice_cuda_accepts_int64_indptr():
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    rows = jnp.array([1, 0], dtype=jnp.int32)

    got = csr_slice_rows(
        weights, indices, indptr64, rows, shape=(2, 3), backend='cuda_raw'
    )
    expected = jnp.array([[0.0, 3.0, 4.0], [1.0, 0.0, 2.0]], dtype=jnp.float32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_slice_grad_cuda_accepts_int64_indptr():
    _, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    rows = jnp.array([1, 0], dtype=jnp.int32)
    ct = jnp.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], dtype=jnp.float32)

    got = csr_slice_rows_grad(ct, indices, indptr64, rows, shape=(2, 3), backend='cuda_raw')
    expected = jnp.array([1.0, 3.0, 20.0, 30.0], dtype=jnp.float32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_yw2y_cuda_accepts_int64_indptr():
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    y = jnp.array([1.0, 2.0], dtype=jnp.float32)

    got = csrmv_yw2y(y, weights, indices, indptr64, shape=(2, 3), backend='cuda_raw')
    expected = csrmv_yw2y(y, weights, indices, indptr32, shape=(2, 3), backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_plasticity_pre_cuda_accepts_int64_indptr():
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    pre_spike = jnp.array([True, False])
    post_trace = jnp.array([0.5, 1.5, 2.5], dtype=jnp.float32)

    got = update_csr_on_binary_pre(
        weights, indices, indptr64, pre_spike, post_trace, shape=(2, 3), backend='cuda_raw'
    )
    expected = jnp.array([1.5, 4.5, 3.0, 4.0], dtype=jnp.float32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_plasticity_post_cuda_accepts_int64_indptr():
    weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
    weight_indices = jnp.array([0, 2, 1, 3], dtype=jnp.int32)
    pre_trace = jnp.array([0.5, 1.5], dtype=jnp.float32)
    post_spike = jnp.array([False, True])

    got = update_csr_on_binary_post(
        weights, indices, indptr, weight_indices, pre_trace, post_spike, shape=(2, 2), backend='cuda_raw'
    )
    expected = jnp.array([1.0, 2.5, 3.0, 5.5], dtype=jnp.float32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_csr_cuda_sources_do_not_cast_indptr_to_int32():
    csr_dir = Path(__file__).parent
    for path in csr_dir.glob('*.cu'):
        text = path.read_text()
        assert 'static_cast<const int32_t*>(indptr.data_ptr())' not in text, path.name
        assert 'const int32_t*  __restrict__ indptr' not in text, path.name
        assert 'const int32_t*   __restrict__ indptr' not in text, path.name
