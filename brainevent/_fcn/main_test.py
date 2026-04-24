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

import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'


import functools

import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

# Use full-precision GEMM on GPU to keep dense-reference paths numerically
# consistent with sparse kernels (avoid TF32 drift on large reductions).
if jax.default_backend() == 'gpu' and jax.config.jax_default_matmul_precision is None:
    jax.config.update('jax_default_matmul_precision', 'highest')

import brainevent
import brainevent._fcn.main as fcn_main_mod
from brainevent._misc import fixed_conn_num_to_csc
from brainevent._test_util import (
    allclose,
    generate_fixed_conn_num_indices,
    vector_fcn,
    matrix_fcn,
    fcn_vector,
    fcn_matrix,
    ones_like,
)

platform = jax.default_backend()

if platform == 'cpu':
    shapes = [
        (200, 300),
        (100, 500)
    ]
else:
    shapes = [
        (2000, 3000),
        (1000, 5000)
    ]

if platform == 'cpu':
    operator_shapes = [
        (20, 40),
        (50, 30),
    ]
else:
    operator_shapes = [
        (20, 40),
        # (50, 30),
        (200, 400),
        # (500, 300),
        # (2000, 4000),
        (5000, 3000),
    ]


def _binary_mask(x, dtype):
    return jnp.asarray(jnp.asarray(x) > 0, dtype=dtype)


def _remove_event_array(x):
    if isinstance(x, brainevent.BinaryArray):
        return x.value
    return x


class Test_To_Dense:
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_todense(self, shape, homo_w):
        m, n = shape
        x = brainstate.random.rand(m)
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        csc = csr.T

        out1 = csr.todense()
        out2 = csc.todense().T
        out3 = csr.T.todense().T
        out4 = csc.T.todense()
        assert allclose(out1, out2)
        assert allclose(out1, out3)
        assert allclose(out1, out4)
        jax.block_until_ready((x, indices, out1, out2, out3, out4))


class Test_To_COO:
    def test_tocoo_round_trip(self):
        post_indices = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        post_data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        post = brainevent.FixedPostNumConn((post_data, post_indices), shape=(3, 4))
        assert allclose(post.tocoo().todense(), post.todense())

        pre_indices = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        pre_data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        pre = brainevent.FixedPreNumConn((pre_data, pre_indices), shape=(4, 3))
        assert allclose(pre.tocoo().todense(), pre.todense())
        jax.block_until_ready((post_indices, post_data, pre_indices, pre_data))


class Test_Illegal_Slots:
    def test_invalid_indices_rejected_post(self):
        idx = jnp.array([[0, -1, 2, 2], [1, 4, 3, 1], [2, 0, -3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        with pytest.raises(ValueError, match="invalid indices"):
            brainevent.FixedPostNumConn((data, idx), shape=(3, 4))
        jax.block_until_ready((idx, data))

    def test_invalid_indices_rejected_pre(self):
        idx = jnp.array([[0, -1, 2, 2], [1, 4, 3, 1], [2, 0, -3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        with pytest.raises(ValueError, match="invalid indices"):
            brainevent.FixedPreNumConn((data, idx), shape=(4, 3))
        jax.block_until_ready((idx, data))

    def test_invalid_indices_rejected_homo(self):
        idx = jnp.array([[0, -1, 2], [1, 5, 1]], dtype=jnp.int32)
        with pytest.raises(ValueError, match="invalid indices"):
            brainevent.FixedPostNumConn((jnp.array(1.5, dtype=jnp.float32), idx), shape=(2, 4))
        jax.block_until_ready((idx,))

    def test_duplicates_are_supported_post(self):
        idx = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        conn = brainevent.FixedPostNumConn((data, idx), shape=(3, 4))

        dense = conn.todense()
        x = jnp.array([1., 2., 3.], dtype=jnp.float32)
        v = jnp.array([1., 2., 3., 4.], dtype=jnp.float32)
        X = jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)
        V = jnp.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=jnp.float32)

        assert allclose(x @ conn, x @ dense)
        assert allclose(conn @ v, dense @ v)
        assert allclose(X @ conn, X @ dense)
        assert allclose(conn @ V, dense @ V)
        jax.block_until_ready((idx, data, dense, x, v, X, V))

    def test_duplicates_are_supported_pre(self):
        idx = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        conn = brainevent.FixedPreNumConn((data, idx), shape=(4, 3))

        dense = conn.todense()
        x = jnp.array([1., 2., 3., 4.], dtype=jnp.float32)
        v = jnp.array([1., 2., 3.], dtype=jnp.float32)
        X = jnp.array([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=jnp.float32)
        V = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)

        assert allclose(x @ conn, x @ dense)
        assert allclose(conn @ v, dense @ v)
        assert allclose(X @ conn, X @ dense)
        assert allclose(conn @ V, dense @ V)
        jax.block_until_ready((idx, data, dense, x, v, X, V))

    def test_homo_weight_with_duplicates(self):
        idx = jnp.array([[0, 1, 2], [1, 3, 1]], dtype=jnp.int32)
        conn = brainevent.FixedPostNumConn((jnp.array(1.5, dtype=jnp.float32), idx), shape=(2, 4))
        dense = conn.todense()
        x = jnp.array([1., 2.], dtype=jnp.float32)
        v = jnp.array([1., 2., 3., 4.], dtype=jnp.float32)

        assert allclose(x @ conn, x @ dense)
        assert allclose(conn @ v, dense @ v)
        jax.block_until_ready((idx, dense, x, v))


class Test_Dual_Layout:
    def test_default_layout_buffers_are_empty(self):
        shape = (3, 4)
        idx = generate_fixed_conn_num_indices(*shape, 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedPostNumConn((data, idx), shape=shape)

        assert conn.maintain_dual_layout is False
        assert conn.primary_layout == 'row'
        assert conn.col_weights is None
        assert conn.col_indices is None
        assert conn.col_indptr is None

    @pytest.mark.parametrize(
        ('cls', 'shape', 'expected_shape'),
        [
            (brainevent.FixedPostNumConn, (3, 4), (3, 4)),
            (brainevent.FixedPreNumConn, (4, 3), (3, 4)),
        ],
    )
    def test_dual_layout_builds_expected_csc_mirror(self, cls, shape, expected_shape):
        if cls is brainevent.FixedPostNumConn:
            idx = generate_fixed_conn_num_indices(shape[0], shape[1], 2)
        else:
            idx = generate_fixed_conn_num_indices(shape[1], shape[0], 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)

        conn = cls(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )

        exp_w, exp_i, exp_p = fixed_conn_num_to_csc(conn.data, conn.indices, shape=expected_shape)
        assert allclose(conn.col_weights, exp_w)
        assert jnp.array_equal(conn.col_indices, exp_i)
        assert jnp.array_equal(conn.col_indptr, exp_p)

    @pytest.mark.parametrize('cls, shape', [
        (brainevent.FixedPostNumConn, (3, 4)),
        (brainevent.FixedPreNumConn, (4, 3)),
    ])
    def test_primary_layout_col_is_rejected(self, cls, shape):
        if cls is brainevent.FixedPostNumConn:
            idx = generate_fixed_conn_num_indices(shape[0], shape[1], 2)
        else:
            idx = generate_fixed_conn_num_indices(shape[1], shape[0], 2)

        with pytest.raises(NotImplementedError, match='primary_layout="col"'):
            cls(
                (jnp.ones(idx.shape, dtype=jnp.float32), idx),
                shape=shape,
                primary_layout='col',
            )

    def test_fixed_post_dual_layout_binary_vector_matches_dense(self):
        shape = (3, 4)
        idx = jnp.array([[0, 1], [1, 3], [2, 0]], dtype=jnp.int32)
        data = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
        conn = brainevent.FixedPostNumConn(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )
        spikes = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.7, 1.0], dtype=jnp.float32))
        dense = conn.todense()

        assert allclose(conn @ spikes, dense @ _binary_mask(spikes.value, dense.dtype))

    def test_fixed_post_dual_layout_compact_vector_matches_dense(self):
        shape = (3, 4)
        idx = jnp.array([[0, 1], [1, 3], [2, 0]], dtype=jnp.int32)
        data = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
        conn = brainevent.FixedPostNumConn(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )
        spikes = jnp.array([1.0, 0.0, 0.7, 1.0], dtype=jnp.float32)
        compact = brainevent.CompactBinary.from_array(spikes)
        dense = conn.todense()

        assert allclose(conn @ compact, dense @ _binary_mask(spikes, dense.dtype))

    def test_fixed_pre_dual_layout_binary_vector_matches_dense(self):
        shape = (4, 3)
        idx = jnp.array([[0, 1], [2, 1], [3, 0]], dtype=jnp.int32)
        data = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
        conn = brainevent.FixedPreNumConn(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )
        spikes = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.7, 1.0], dtype=jnp.float32))
        dense = conn.todense()

        assert allclose(spikes @ conn, _binary_mask(spikes.value, dense.dtype) @ dense)

    def test_fixed_pre_dual_layout_compact_vector_matches_dense(self):
        shape = (4, 3)
        idx = jnp.array([[0, 1], [2, 1], [3, 0]], dtype=jnp.int32)
        data = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
        conn = brainevent.FixedPreNumConn(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )
        spikes = jnp.array([1.0, 0.0, 0.7, 1.0], dtype=jnp.float32)
        compact = brainevent.CompactBinary.from_array(spikes)
        dense = conn.todense()

        assert allclose(compact @ conn, _binary_mask(spikes, dense.dtype) @ dense)

    def test_fixed_post_compact_left_vector_uses_scatter_and_matches_dense(self, monkeypatch):
        shape = (3, 4)
        idx = jnp.array([[0, 1], [1, 3], [2, 0]], dtype=jnp.int32)
        data = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
        conn = brainevent.FixedPostNumConn((data, idx), shape=shape)
        spikes = jnp.array([1.0, 0.0, 0.7], dtype=jnp.float32)
        compact = brainevent.CompactBinary.from_array(spikes)
        dense = conn.todense()
        calls = []
        original = fcn_main_mod.compact_binary_fcnmv

        def _spy(weights, indices, packed, active_ids, n_active, events, **kwargs):
            calls.append(kwargs)
            return original(weights, indices, packed, active_ids, n_active, events, **kwargs)

        monkeypatch.setattr(fcn_main_mod, 'compact_binary_fcnmv', _spy)
        y = compact @ conn

        assert allclose(y, _binary_mask(spikes, dense.dtype) @ dense)
        assert len(calls) == 1
        assert calls[0]['transpose'] is True

    def test_fixed_post_dual_layout_binary_forward_injects_csc_mirror(self, monkeypatch):
        shape = (3, 4)
        idx = generate_fixed_conn_num_indices(*shape, 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedPostNumConn(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )
        spikes = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.5, 1.0], dtype=jnp.float32))
        calls = []
        original = fcn_main_mod.binary_fcnmv

        def _spy(weights, indices, events, **kwargs):
            calls.append(kwargs)
            return original(weights, indices, events, **kwargs)

        monkeypatch.setattr(fcn_main_mod, 'binary_fcnmv', _spy)
        conn @ spikes

        assert len(calls) == 1
        assert calls[0]['col_weights'] is conn.col_weights
        assert calls[0]['col_indices'] is conn.col_indices
        assert calls[0]['col_indptr'] is conn.col_indptr

    def test_fixed_pre_dual_layout_compact_forward_injects_csc_mirror(self, monkeypatch):
        shape = (4, 3)
        idx = generate_fixed_conn_num_indices(shape[1], shape[0], 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedPreNumConn(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )
        spikes = jnp.array([1.0, 0.0, 0.5, 1.0], dtype=jnp.float32)
        compact = brainevent.CompactBinary.from_array(spikes)
        calls = []
        original = fcn_main_mod.compact_binary_fcnmv

        def _spy(weights, indices, packed, active_ids, n_active, events, **kwargs):
            calls.append(kwargs)
            return original(weights, indices, packed, active_ids, n_active, events, **kwargs)

        monkeypatch.setattr(fcn_main_mod, 'compact_binary_fcnmv', _spy)
        compact @ conn

        assert len(calls) == 1
        assert calls[0]['col_weights'] is conn.col_weights
        assert calls[0]['col_indices'] is conn.col_indices
        assert calls[0]['col_indptr'] is conn.col_indptr

    def test_dual_layout_forward_reuses_prebuilt_mirror(self, monkeypatch):
        shape = (3, 4)
        idx = generate_fixed_conn_num_indices(*shape, 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedPostNumConn(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )
        spikes = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.5, 1.0], dtype=jnp.float32))

        def _fail(*args, **kwargs):
            raise AssertionError('row->col conversion helper should not run during forward dual-layout MV')

        monkeypatch.setattr(fcn_main_mod, '_build_col_major_fcn', _fail)
        conn @ spikes

    def test_with_data_rebuilds_dual_layout_mirror(self):
        shape = (3, 4)
        idx = generate_fixed_conn_num_indices(*shape, 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedPostNumConn(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )
        new_data = data + 1.0
        updated = conn.with_data(new_data)
        exp_w, exp_i, exp_p = fixed_conn_num_to_csc(new_data, idx, shape=shape)

        assert updated.maintain_dual_layout is True
        assert updated.primary_layout == 'row'
        assert allclose(updated.col_weights, exp_w)
        assert jnp.array_equal(updated.col_indices, exp_i)
        assert jnp.array_equal(updated.col_indptr, exp_p)
        assert not allclose(updated.col_weights, conn.col_weights)

    def test_transpose_preserves_config_and_rebuilds_dual_layout_mirror(self):
        shape = (3, 4)
        idx = generate_fixed_conn_num_indices(*shape, 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedPostNumConn(
            (data, idx),
            shape=shape,
            maintain_dual_layout=True,
        )
        transposed = conn.T
        exp_w, exp_i, exp_p = fixed_conn_num_to_csc(
            transposed.data,
            transposed.indices,
            shape=transposed.shape[::-1],
        )
        spikes = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.5, 1.0], dtype=jnp.float32))
        dense = transposed.todense()

        assert transposed.maintain_dual_layout is True
        assert transposed.primary_layout == 'row'
        assert allclose(transposed.col_weights, exp_w)
        assert jnp.array_equal(transposed.col_indices, exp_i)
        assert jnp.array_equal(transposed.col_indptr, exp_p)
        assert allclose(spikes @ transposed, _binary_mask(spikes.value, dense.dtype) @ dense)


class Test_Init_Outside_JIT:
    def test_fixed_post_init_rejects_first_construction_inside_jax_jit(self):
        @jax.jit
        def build():
            idx = jnp.array([[0, 1], [1, 3]], dtype=jnp.int32)
            data = jnp.array([[1., 2.], [3., 4.]], dtype=jnp.float32)
            conn = brainevent.FixedPostNumConn((data, idx), shape=(2, 4))
            return conn.nse

        with pytest.raises(RuntimeError, match='must be first constructed outside'):
            build()

    def test_fixed_pre_init_rejects_first_construction_inside_brainstate_jit(self):
        @brainstate.transform.jit
        def build():
            idx = jnp.array([[0, 1], [2, 1], [3, 0]], dtype=jnp.int32)
            data = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
            conn = brainevent.FixedPreNumConn(
                (data, idx),
                shape=(4, 3),
                maintain_dual_layout=True,
            )
            return conn.nse

        with pytest.raises(RuntimeError, match='must be first constructed outside'):
            build()


class Test_Operator_Behavior:
    def test_fixed_post_binary_array_operator_behavior(self):
        idx = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        conn = brainevent.FixedPostNumConn((data, idx), shape=(3, 4))
        dense = conn.todense()

        left_vector = brainevent.BinaryArray(jnp.array([0.2, 0.0, 1.0], dtype=jnp.float32))
        right_vector = brainevent.BinaryArray(jnp.array([0.0, 0.6, 0.0, 1.0], dtype=jnp.float32))
        left_matrix = brainevent.BinaryArray(jnp.array([[0.0, 1.0, 0.3], [1.0, 0.0, 0.0]], dtype=jnp.float32))
        right_matrix = brainevent.BinaryArray(
            jnp.array([[0.0, 1.0], [0.2, 0.0], [1.0, 0.0], [0.0, 0.4]], dtype=jnp.float32)
        )

        assert allclose(left_vector @ conn, _binary_mask(left_vector.value, dense.dtype) @ dense)
        assert allclose(conn @ right_vector, dense @ _binary_mask(right_vector.value, dense.dtype))
        assert allclose(left_matrix @ conn, _binary_mask(left_matrix.value, dense.dtype) @ dense)
        assert allclose(conn @ right_matrix, dense @ _binary_mask(right_matrix.value, dense.dtype))
        jax.block_until_ready(
            (idx, data, dense, left_vector.value, right_vector.value, left_matrix.value, right_matrix.value))

    def test_fixed_pre_binary_array_operator_behavior(self):
        idx = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        conn = brainevent.FixedPreNumConn((data, idx), shape=(4, 3))
        dense = conn.todense()

        left_vector = brainevent.BinaryArray(jnp.array([0.0, 1.0, 0.4, 1.0], dtype=jnp.float32))
        right_vector = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.7], dtype=jnp.float32))
        left_matrix = brainevent.BinaryArray(
            jnp.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.2]], dtype=jnp.float32)
        )
        right_matrix = brainevent.BinaryArray(jnp.array([[0.2, 0.0], [1.0, 1.0], [0.0, 0.8]], dtype=jnp.float32))

        assert allclose(left_vector @ conn, _binary_mask(left_vector.value, dense.dtype) @ dense)
        assert allclose(conn @ right_vector, dense @ _binary_mask(right_vector.value, dense.dtype))
        assert allclose(left_matrix @ conn, _binary_mask(left_matrix.value, dense.dtype) @ dense)
        assert allclose(conn @ right_matrix, dense @ _binary_mask(right_matrix.value, dense.dtype))
        jax.block_until_ready(
            (idx, data, dense, left_vector.value, right_vector.value, left_matrix.value, right_matrix.value))



class TestVector:
    def _generate_x(self, shape, require_float=False):
        if isinstance(shape, (tuple, list)):
            yield brainstate.random.rand(*shape)
        else:
            yield brainstate.random.rand(shape)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    def test_vector_csr(self, homo_w, shape):
        m, n = shape
        for x in self._generate_x(m):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))

            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            conn = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = jax.jit(lambda: x @ conn)()
            y2 = jax.jit(lambda: conn.T @ x)()
            y3 = _remove_event_array(x) @ conn.todense()

            y_true = vector_fcn(x, conn.data, indices, (m, n))
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y3, y_true, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, y1, y2, y3, y_true))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    def test_csr_vector(self, homo_w, shape):
        m, n = shape
        for v in self._generate_x(n):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            conn = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = jax.jit(lambda: conn @ v)()
            y2 = jax.jit(lambda: v @ conn.T)()
            y_true = fcn_vector(v, conn.data, indices, (m, n))
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((v, indices, y1, y2, y_true))

    def _test_vjp(self, homo_w, transpose, shape):
        n_in, n_out = shape

        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1))
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        conn = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w_data):
            if transpose:
                r = x @ conn.with_data(w_data)
            else:
                r = conn.with_data(w_data) @ x
            return r.sum()

        def f_ref(x, w_data):
            if transpose:
                r = vector_fcn(x, w_data, indices, shape)
            else:
                r = fcn_vector(x, w_data, indices, shape)
            return r.sum()

        for x in self._generate_x(n_in if transpose else n_out, require_float=True):
            r1 = jax.jit(lambda x_arg, w_arg: jax.grad(f_brainevent, argnums=(0, 1))(x_arg, w_arg))(x, w)
            r2 = jax.jit(lambda x_arg, w_arg: jax.grad(f_ref, argnums=(0, 1))(x_arg, w_arg))(x, w)

            assert allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, r1[0], r1[1], r2[0], r2[1]))

    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    def test_vjp(self, transpose, homo_w, shape):
        self._test_vjp(homo_w=homo_w, transpose=transpose, shape=shape)

    def _test_jvp(self, homo_w, transpose, shape):
        n_in, n_out = shape

        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1))
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        conn = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w_data):
            if transpose:
                r = x @ conn.with_data(w_data)
            else:
                r = conn.with_data(w_data) @ x
            return r

        def f_ref(x, w_data):
            if transpose:
                r = vector_fcn(x, w_data, indices, shape)
            else:
                r = fcn_vector(x, w_data, indices, shape)
            return r

        for x in self._generate_x(n_in if transpose else n_out, require_float=True):
            o1, r1 = jax.jit(
                lambda x_arg, w_arg: jax.jvp(
                    f_brainevent,
                    (x_arg, w_arg),
                    (ones_like(x_arg), ones_like(w_arg))
                )
            )(x, w)
            o2, r2 = jax.jit(
                lambda x_arg, w_arg: jax.jvp(
                    f_ref,
                    (x_arg, w_arg),
                    (ones_like(x_arg), ones_like(w_arg))
                )
            )(x, w)

            assert allclose(r1, r2, rtol=1e-3, atol=1e-3)
            assert allclose(o1, o2, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, o1, r1, o2, r2))

    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    def test_jvp(self, transpose, homo_w, shape):
        self._test_jvp(homo_w=homo_w, transpose=transpose, shape=shape)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    @pytest.mark.parametrize('batch_size', [32])
    def test_batching_weight(self, homo_w, shape, batch_size):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))

        data = (
            brainstate.random.rand(batch_size)
            if homo_w else
            braintools.init.Normal(0., 1.)((batch_size,) + indices.shape)
        )

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def f_compare_vector_conn(w, x):
            conn = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = x @ conn
            y2 = conn.T @ x
            y_true = vector_fcn(x, conn.data, indices, (m, n))
            return y1, y2, y_true

        for x in self._generate_x(m):
            y1, y2, y_true = f_compare_vector_conn(data, x)
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def f_compare_conn_vector(w, x):
            conn = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = conn @ x
            y2 = x @ conn.T
            y_true = fcn_vector(x, conn.data, indices, (m, n))
            return y1, y2, y_true

        for x in self._generate_x(n):
            y1, y2, y_true = f_compare_conn_vector(data, x)
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((indices, data, y1, y2, y_true))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('batch_axis', [0, 1])
    def test_batching_vector(self, homo_w, shape, batch_size, batch_axis):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_vector_conn(x):
            conn = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = x @ conn
            y2 = conn.T @ x
            y_true = vector_fcn(x, conn.data, indices, (m, n))
            return y1, y2, y_true

        for xs in self._generate_x([batch_size, m] if batch_axis == 0 else [m, batch_size]):
            y1, y2, y_true = f_compare_vector_conn(xs)
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_conn_vector(x):
            conn = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = conn @ x
            y2 = x @ conn.T
            y_true = fcn_vector(x, conn.data, indices, (m, n))
            return y1, y2, y_true

        for xs in self._generate_x([batch_size, n] if batch_axis == 0 else [n, batch_size]):
            y1, y2, y_true = f_compare_conn_vector(xs)
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((indices, y1, y2, y_true))


class TestMatrix:
    def _generate_x(self, shape, require_float=False):
        if isinstance(shape, (tuple, list)):
            yield brainstate.random.rand(*shape)
        else:
            yield brainstate.random.rand(shape)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    @pytest.mark.parametrize('k', [10])
    def test_matrix_csr(self, homo_w, shape, k):
        m, n = shape
        for x in self._generate_x([k, m]):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            conn = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = jax.jit(lambda: x @ conn)()
            y2 = jax.jit(lambda: (conn.T @ x.T).T)()
            y_true = matrix_fcn(x, conn.data, indices, (m, n))
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, y1, y2, y_true))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    @pytest.mark.parametrize('k', [10])
    def test_csr_matrix(self, homo_w, shape, k):
        m, n = shape
        for matrix in self._generate_x([n, k]):
            indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
            conn = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = jax.jit(lambda: conn @ matrix)()
            y2 = jax.jit(lambda: (matrix.T @ conn.T).T)()
            y_true = fcn_matrix(matrix, conn.data, indices, (m, n))
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((matrix, indices, y1, y2, y_true))

    def _test_vjp(self, homo_w, transpose, shape, k):
        n_in, n_out = shape

        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1))
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        conn = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w_data):
            if transpose:
                r = x @ conn.with_data(w_data)
            else:
                r = conn.with_data(w_data) @ x
            return r.sum()

        def f_ref(x, w_data):
            if transpose:
                r = matrix_fcn(x, w_data, indices, shape)
            else:
                r = fcn_matrix(x, w_data, indices, shape)
            return r.sum()

        for x in self._generate_x([k, n_in] if transpose else [n_out, k], require_float=True):
            r1 = jax.jit(lambda x_arg, w_arg: jax.grad(f_brainevent, argnums=(0, 1))(x_arg, w_arg))(x, w)
            r2 = jax.jit(lambda x_arg, w_arg: jax.grad(f_ref, argnums=(0, 1))(x_arg, w_arg))(x, w)

            assert allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3)
            assert allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, r1[0], r1[1], r2[0], r2[1]))

    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    @pytest.mark.parametrize('k', [10])
    def test_vjp(self, transpose, homo_w, shape, k):
        self._test_vjp(homo_w=homo_w, transpose=transpose, shape=shape, k=k)

    def _test_jvp(self, homo_w, transpose, shape, k):
        n_in, n_out = shape

        indices = generate_fixed_conn_num_indices(n_in, n_out, int(n_out * 0.1))
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        conn = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w_data):
            if transpose:
                r = x @ conn.with_data(w_data)
            else:
                r = conn.with_data(w_data) @ x
            return r

        def f_ref(x, w_data):
            if transpose:
                r = matrix_fcn(x, w_data, indices, shape)
            else:
                r = fcn_matrix(x, w_data, indices, shape)
            return r

        for x in self._generate_x((k, n_in) if transpose else (n_out, k), require_float=True):
            o1, r1 = jax.jit(
                lambda x_arg, w_arg: jax.jvp(
                    f_brainevent,
                    (x_arg, w_arg),
                    (ones_like(x_arg), ones_like(w_arg))
                )
            )(x, w)
            o2, r2 = jax.jit(
                lambda x_arg, w_arg: jax.jvp(f_ref, (x_arg, w_arg), (ones_like(x_arg), ones_like(w_arg)))
            )(x, w)

            assert allclose(r1, r2, rtol=1e-3, atol=1e-3)
            assert allclose(o1, o2, rtol=1e-3, atol=1e-3)
            jax.block_until_ready((x, indices, o1, r1, o2, r2))

    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    @pytest.mark.parametrize('k', [10])
    def test_jvp(self, transpose, homo_w, shape, k):
        self._test_jvp(homo_w=homo_w, transpose=transpose, shape=shape, k=k)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('k', [32])
    def test_batching_weight(self, homo_w, shape, batch_size, k):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))

        data = (
            brainstate.random.rand(batch_size)
            if homo_w else
            braintools.init.Normal(0., 1.)((batch_size,) + indices.shape)
        )

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def f_compare_matrix_conn(w, x):
            conn = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = x @ conn
            y2 = (conn.T @ x.T).T
            y_true = matrix_fcn(x, conn.data, indices, (m, n))
            return y1, y2, y_true

        for x in self._generate_x([k, m]):
            y1, y2, y_true = f_compare_matrix_conn(data, x)
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def f_compare_conn_vector(w, x):
            conn = brainevent.FixedPostNumConn((w, indices), shape=(m, n))
            y1 = conn @ x
            y2 = (x.T @ conn.T).T
            y_true = fcn_matrix(x, conn.data, indices, (m, n))
            return y1, y2, y_true

        for x in self._generate_x([n, k]):
            y1, y2, y_true = f_compare_conn_vector(data, x)
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((indices, data, y1, y2, y_true))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', operator_shapes)
    @pytest.mark.parametrize('batch_size', [32])
    @pytest.mark.parametrize('k', [32])
    @pytest.mark.parametrize('batch_axis', [0, 1, 2])
    def test_batching_vector(self, homo_w, shape, batch_size, k, batch_axis):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_vector_conn(x):
            conn = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = x @ conn
            y2 = (conn.T @ x.T).T
            y_true = matrix_fcn(x, conn.data, indices, (m, n))
            return y1, y2, y_true

        if batch_axis == 0:
            batch_shape = [batch_size, k, m]
        elif batch_axis == 1:
            batch_shape = [k, batch_size, m]
        else:
            batch_shape = [k, m, batch_size]
        for xs in self._generate_x(batch_shape):
            y1, y2, y_true = f_compare_vector_conn(xs)
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_conn_vector(x):
            conn = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
            y1 = conn @ x
            y2 = (x.T @ conn.T).T
            y_true = fcn_matrix(x, conn.data, indices, (m, n))
            return y1, y2, y_true

        if batch_axis == 0:
            batch_shape = [batch_size, n, k]
        elif batch_axis == 1:
            batch_shape = [n, batch_size, k]
        else:
            batch_shape = [n, k, batch_size]
        for xs in self._generate_x(batch_shape):
            y1, y2, y_true = f_compare_conn_vector(xs)
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((indices, xs, y1, y2, y_true))
