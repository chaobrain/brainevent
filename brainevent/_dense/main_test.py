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
import pytest

import brainunit as u
import brainevent as be


def test_dense_is_exported_from_top_level_and_dense_package():
    from brainevent._dense import Dense

    assert be.Dense is Dense


def test_dense_fromdense_todense_with_units():
    mat = jnp.asarray([[1., 0., 2.], [0., 3., 4.]], dtype=jnp.float32) * u.mV

    dense = be.Dense.fromdense(mat, backend='jax_raw')

    assert dense.shape == (2, 3)
    assert dense.backend == 'jax_raw'
    assert u.get_unit(dense.todense()) == u.mV
    assert u.math.allclose(dense.todense(), mat)


def test_dense_with_data_and_elementwise_ops_preserve_metadata():
    mat = jnp.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32) * u.mV
    dense = be.Dense(mat, backend='jax_raw', buffers={'tag': 'keep'})
    other = jnp.asarray([[0.5, 1., 1.5], [2., 2.5, 3.]], dtype=jnp.float32) * u.mV

    updated = dense.with_data(mat + 1. * u.mV)
    added = dense + other
    scaled = 2. * dense

    assert isinstance(updated, be.Dense)
    assert updated.shape == dense.shape
    assert updated.backend == dense.backend
    assert updated.buffers == dense.buffers
    assert u.math.allclose(updated.todense(), mat + 1. * u.mV)
    assert u.math.allclose(added.todense(), mat + other)
    assert u.math.allclose(scaled.todense(), 2. * mat)
    assert added.buffers == dense.buffers


def test_dense_rejects_incompatible_elementwise_shapes():
    dense = be.Dense(jnp.ones((2, 3), dtype=jnp.float32))

    with pytest.raises(ValueError):
        dense + jnp.ones((2, 2), dtype=jnp.float32)
    with pytest.raises(ValueError):
        dense + be.Dense(jnp.ones((3, 2), dtype=jnp.float32))
    with pytest.raises(NotImplementedError):
        dense * be.CSR.fromdense(jnp.ones((2, 3), dtype=jnp.float32))


def test_dense_transpose_getitem_slice_and_diag_add():
    mat = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    dense = be.Dense(mat, backend='cuda_raw', buffers={'tag': 'keep'})
    rows = jnp.asarray([2, 0])

    transposed = dense.transpose()
    single = dense.slice_rows(1)
    multiple = dense.slice_rows([2, 0])
    diag_added = dense.diag_add(jnp.asarray([10., 20., 30.], dtype=jnp.float32))

    assert isinstance(transposed, be.Dense)
    assert transposed.shape == (4, 3)
    assert transposed.backend == 'cuda_raw'
    assert jnp.allclose(transposed.todense(), mat.T)
    assert jnp.allclose(dense[1], mat[1])
    assert jnp.allclose(dense[[2, 0]], mat[rows])
    assert single.shape == (1, 4)
    assert single.buffers == dense.buffers
    assert jnp.allclose(single.todense(), mat[1:2])
    assert multiple.shape == (2, 4)
    assert jnp.allclose(multiple.todense(), mat[rows])
    assert jnp.allclose(
        diag_added.todense(),
        mat.at[jnp.arange(3), jnp.arange(3)].add(jnp.asarray([10., 20., 30.], dtype=jnp.float32)),
    )
    with pytest.raises(ValueError):
        dense.diag_add(jnp.ones((4,), dtype=jnp.float32))


def test_dense_binary_and_float_matmul_match_dense_reference():
    weights = jnp.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)
    dense = be.Dense(weights, backend='jax_raw')
    rhs_vec = be.BinaryArray(jnp.asarray([True, False, True]))
    lhs_vec = be.BinaryArray(jnp.asarray([True, False]))
    rhs_mat = be.BinaryArray(jnp.asarray([[True, False], [False, True], [True, True]]))
    lhs_mat = be.BinaryArray(jnp.asarray([[True, False], [False, True], [True, True]]))
    float_rhs = jnp.asarray([0.5, 1.5, -1.], dtype=jnp.float32)
    float_lhs = jnp.asarray([2., -1.], dtype=jnp.float32)

    assert jnp.allclose(dense @ rhs_vec, weights @ rhs_vec.value.astype(weights.dtype))
    assert jnp.allclose(lhs_vec @ dense, lhs_vec.value.astype(weights.dtype) @ weights)
    assert jnp.allclose(dense @ rhs_mat, weights @ rhs_mat.value.astype(weights.dtype))
    assert jnp.allclose(lhs_mat @ dense, lhs_mat.value.astype(weights.dtype) @ weights)
    assert jnp.allclose(dense @ float_rhs, weights @ float_rhs)
    assert jnp.allclose(float_lhs @ dense, float_lhs @ weights)


def test_dense_conversions_update_and_solve_match_references():
    mat = jnp.asarray([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]], dtype=jnp.float32)
    dense = be.Dense.fromdense(mat, backend='jax_raw')
    weights = jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=jnp.float32)
    plastic = be.Dense(weights, backend='jax_raw')
    pre_spike = jnp.asarray([True, False])
    post_trace = jnp.asarray([0.2, 0.4, 0.6], dtype=jnp.float32)
    pre_trace = jnp.asarray([0.3, 0.7], dtype=jnp.float32)
    post_spike = jnp.asarray([False, True, True])
    solve_mat = jnp.asarray([[3., 1.], [1., 2.]], dtype=jnp.float32)
    rhs = jnp.asarray([9., 8.], dtype=jnp.float32)

    on_pre = plastic.update_on_pre(pre_spike, post_trace, w_min=0.0, w_max=0.75)
    on_post = plastic.update_on_post(pre_trace, post_spike, w_min=0.0, w_max=0.9)

    assert jnp.allclose(dense.tocsr().todense(), mat)
    assert jnp.allclose(dense.tocsc().todense(), mat)
    assert jnp.allclose(dense.tocoo().todense(), mat)
    assert jnp.allclose(on_pre.todense(), jnp.clip(weights + jnp.outer(pre_spike.astype(weights.dtype), post_trace), 0.0, 0.75))
    assert jnp.allclose(on_post.todense(), jnp.clip(weights + jnp.outer(pre_trace, post_spike.astype(weights.dtype)), 0.0, 0.9))
    assert jnp.allclose(be.Dense(solve_mat).solve(rhs), jnp.linalg.solve(solve_mat, rhs))


def test_dense_sparse_conversion_kwargs_are_forwarded(monkeypatch):
    mat = jnp.asarray([[1., 0.], [0., 2.]], dtype=jnp.float32)
    dense = be.Dense.fromdense(mat, backend='jax_raw')
    calls = {}

    def fake_csr_fromdense(cls, mat_arg, *, nse=None, index_dtype=jnp.int32,
                           backend=None, precompute_weight_indices=False):
        calls['csr'] = (mat_arg, nse, index_dtype, backend, precompute_weight_indices)
        return 'csr-result'

    def fake_csc_fromdense(cls, mat_arg, *, nse=None, index_dtype=jnp.int32,
                           backend=None, precompute_weight_indices=False):
        calls['csc'] = (mat_arg, nse, index_dtype, backend, precompute_weight_indices)
        return 'csc-result'

    monkeypatch.setattr(be.CSR, 'fromdense', classmethod(fake_csr_fromdense))
    monkeypatch.setattr(be.CSC, 'fromdense', classmethod(fake_csc_fromdense))

    assert dense.tocsr(nse=2, index_dtype=jnp.int16, precompute_weight_indices=True) == 'csr-result'
    assert dense.tocsc(nse=3, index_dtype=jnp.int32, precompute_weight_indices=True) == 'csc-result'
    assert calls['csr'] == (dense.data, 2, jnp.int16, dense.backend, True)
    assert calls['csc'] == (dense.data, 3, jnp.int32, dense.backend, True)


def test_dense_dt2t_interfaces_report_unimplemented():
    dense = be.Dense(jnp.ones((2, 3), dtype=jnp.float32))
    y_pre = jnp.ones((2,), dtype=jnp.float32)
    y_post = jnp.ones((3,), dtype=jnp.float32)
    weights = dense.data

    with pytest.raises(NotImplementedError, match='Dense.dt2t is not implemented'):
        dense.dt2t(y_pre, weights)
    with pytest.raises(NotImplementedError, match='Dense.dt2t_transposed is not implemented'):
        dense.dt2t_transposed(y_post, weights)
    with pytest.warns(DeprecationWarning, match='yw_to_w is deprecated'):
        with pytest.raises(NotImplementedError, match='Dense.dt2t is not implemented'):
            dense.yw_to_w(y_pre, weights)
    with pytest.warns(DeprecationWarning, match='yw_to_w_transposed is deprecated'):
        with pytest.raises(NotImplementedError, match='Dense.dt2t_transposed is not implemented'):
            dense.yw_to_w_transposed(y_post, weights)


def test_dense_is_jit_compatible_pytree_for_binary_matmul_and_with_data():
    weights = jnp.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)
    dense = be.Dense(weights, backend='jax_raw')
    spikes = be.BinaryArray(jnp.asarray([True, False, True]))

    @jax.jit
    def run(conn, events):
        updated = conn.with_data(conn.data + 1.)
        return updated @ events

    got = run(dense, spikes)
    expected = (weights + 1.) @ spikes.value.astype(weights.dtype)
    assert jnp.allclose(got, expected)
