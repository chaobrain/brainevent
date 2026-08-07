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
import inspect
import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import pytest

# Use full-precision GEMM on GPU to keep dense-reference paths numerically
# consistent with sparse kernels (avoid TF32 drift on large reductions).
if jax.default_backend() == 'gpu' and jax.config.jax_default_matmul_precision is None:
    jax.config.update('jax_default_matmul_precision', 'highest')

import brainevent
import brainevent._fcn.main as fcn_main_mod
from brainevent._misc import fixed_conn_num_csc_structure
from brainevent._test_util import (
    allclose,
    generate_fixed_conn_num_indices,
    vector_fcn,
    matrix_fcn,
    fcn_vector,
    fcn_matrix,
    ones_like,
)
from brainevent import CSC, CSR, FixedNumPerPost, FixedNumPerPre, BinaryArray

# The operator tests below dispatch to the native ``numba`` backend, which compiles per test
# and dominates wall-clock, so each carries ``@pytest.mark.slow`` and the default ``pytest``
# run skips it; CI runs them via ``pytest -m ""``. The marker is per-test rather than
# module-wide so the cheap format-conversion and slicing tests further down stay in the
# default run.

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


def _skip_batching_if_gpu_too_small(shape, batch_size):
    """Skip batched matrix tests on GPUs too small to hold the dense reference.

    The batched dense-reference paths materialise several ``(batch_size, m, n)``
    float32 temporaries (the dense weight matrix, the matmul output and transpose
    fusions).  For the largest ``operator_shapes`` entry this peaks at well over
    10 GB and exceeds small (e.g. 16 GB) GPUs.  The computation is correct — it
    merely does not fit — so skip on memory-constrained GPUs rather than fail with
    a ``RESOURCE_EXHAUSTED`` error.  CPU (CI) and larger GPUs run the case normally.
    """
    device = jax.local_devices()[0]
    if device.platform != 'gpu':
        return
    try:
        limit = device.memory_stats().get('bytes_limit') or 0
    except Exception:
        return
    m, n = shape
    # Empirically the peak is ~6x the batched dense buffer for these fusions.
    est_bytes = 6 * batch_size * m * n * 4
    if limit and est_bytes > 0.8 * limit:
        pytest.skip(
            f'batched shape {shape} (batch {batch_size}) needs ~{est_bytes / 1e9:.1f} GB; '
            f'GPU memory limit is {limit / 1e9:.1f} GB'
        )


def _binary_mask(x, dtype):
    return jnp.asarray(jnp.asarray(x) > 0, dtype=dtype)


def _remove_event_array(x):
    if isinstance(x, brainevent.BinaryArray):
        return x.value
    return x


@pytest.mark.slow
class Test_To_Dense:
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_todense(self, shape, homo_w):
        m, n = shape
        x = brainstate.random.rand(m)
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))
        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
        csc = csr.T

        out1 = csr.todense()
        out2 = csc.todense().T
        out3 = csr.T.todense().T
        out4 = csc.T.todense()
        assert allclose(out1, out2)
        assert allclose(out1, out3)
        assert allclose(out1, out4)
        jax.block_until_ready((x, indices, out1, out2, out3, out4))


@pytest.mark.slow
def test_fixed_post_num_conn_tree_flatten_data_only_leaf():
    indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    data = jnp.array([1.5], dtype=jnp.float32)
    conn = brainevent.FixedNumPerPre((data, indices), shape=(2, 3))

    children, aux = conn.tree_flatten()
    aux_dict, buffers = aux

    # ``data`` is the only traced leaf; indices/shape/backend live in static aux
    # (mirrors CompressedSparseData / CSR).
    assert len(children) == 1
    assert children[0] is conn.data
    assert aux_dict['indices'] is conn.indices
    assert aux_dict['shape'] == (2, 3)
    assert isinstance(buffers, dict)


@pytest.mark.slow
class Test_Illegal_Slots:
    def test_invalid_indices_rejected_post(self):
        idx = jnp.array([[0, -1, 2, 2], [1, 4, 3, 1], [2, 0, -3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        with pytest.raises(ValueError, match="invalid indices"):
            brainevent.FixedNumPerPre((data, idx), shape=(3, 4))
        jax.block_until_ready((idx, data))

    def test_invalid_indices_rejected_pre(self):
        idx = jnp.array([[0, -1, 2, 2], [1, 4, 3, 1], [2, 0, -3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        with pytest.raises(ValueError, match="invalid indices"):
            brainevent.FixedNumPerPost((data, idx), shape=(4, 3))
        jax.block_until_ready((idx, data))

    def test_invalid_indices_rejected_homo(self):
        idx = jnp.array([[0, -1, 2], [1, 5, 1]], dtype=jnp.int32)
        with pytest.raises(ValueError, match="invalid indices"):
            brainevent.FixedNumPerPre((jnp.array(1.5, dtype=jnp.float32), idx), shape=(2, 4))
        jax.block_until_ready((idx,))

    def test_duplicates_are_supported_post(self):
        idx = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        conn = brainevent.FixedNumPerPre((data, idx), shape=(3, 4))

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
        conn = brainevent.FixedNumPerPost((data, idx), shape=(4, 3))

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
        conn = brainevent.FixedNumPerPre((jnp.array(1.5, dtype=jnp.float32), idx), shape=(2, 4))
        dense = conn.todense()
        x = jnp.array([1., 2.], dtype=jnp.float32)
        v = jnp.array([1., 2., 3., 4.], dtype=jnp.float32)

        assert allclose(x @ conn, x @ dense)
        assert allclose(conn @ v, dense @ v)
        jax.block_until_ready((idx, dense, x, v))


@pytest.mark.slow
class Test_Lazy_Csc_Layout:
    def test_default_layout_has_no_csc_mirror(self):
        shape = (3, 4)
        idx = generate_fixed_conn_num_indices(*shape, 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedNumPerPre((data, idx), shape=shape)

        assert conn.buffers.get('csc') is None

    @pytest.mark.parametrize(
        ('cls', 'shape', 'a_shape'),
        [
            (brainevent.FixedNumPerPre, (3, 4), (3, 4)),
            (brainevent.FixedNumPerPost, (4, 3), (3, 4)),
        ],
    )
    def test_weight_indices_builds_expected_structure(self, cls, shape, a_shape):
        if cls is brainevent.FixedNumPerPre:
            idx = generate_fixed_conn_num_indices(shape[0], shape[1], 2)
        else:
            idx = generate_fixed_conn_num_indices(shape[1], shape[0], 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)

        conn = cls((data, idx), shape=shape)
        csc_indptr, csc_indices, perm = conn._weight_indices()

        exp_p, exp_i, exp_perm = fixed_conn_num_csc_structure(conn.indices, shape=a_shape)
        assert jnp.array_equal(csc_indptr, exp_p)
        assert jnp.array_equal(csc_indices, exp_i)
        assert jnp.array_equal(perm, exp_perm)
        # ``perm`` gathers the flattened ELL weights into canonical CSC order.
        assert allclose(conn.data.reshape(-1)[perm], conn.data.reshape(-1)[exp_perm])

    def test_weight_indices_caches_and_reuses(self):
        shape = (3, 4)
        idx = generate_fixed_conn_num_indices(*shape, 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedNumPerPre((data, idx), shape=shape)

        first = conn._weight_indices()
        second = conn._weight_indices()
        assert first is second

    def test_fixed_post_binary_vector_matches_dense(self):
        shape = (3, 4)
        idx = jnp.array([[0, 1], [1, 3], [2, 0]], dtype=jnp.int32)
        data = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
        conn = brainevent.FixedNumPerPre((data, idx), shape=shape)
        spikes = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.7, 1.0], dtype=jnp.float32))
        dense = conn.todense()

        assert allclose(conn @ spikes, dense @ _binary_mask(spikes.value, dense.dtype))

    def test_fixed_pre_binary_vector_matches_dense(self):
        shape = (4, 3)
        idx = jnp.array([[0, 1], [2, 1], [3, 0]], dtype=jnp.int32)
        data = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
        conn = brainevent.FixedNumPerPost((data, idx), shape=shape)
        spikes = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.7, 1.0], dtype=jnp.float32))
        dense = conn.todense()

        assert allclose(spikes @ conn, _binary_mask(spikes.value, dense.dtype) @ dense)

    def test_with_data_carries_mirror_and_matches_new_values(self):
        shape = (3, 4)
        idx = generate_fixed_conn_num_indices(*shape, 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedNumPerPre((data, idx), shape=shape)
        conn._weight_indices()  # populate the structure cache

        new_data = data + 1.0
        updated = conn.with_data(new_data)
        exp_p, exp_i, exp_perm = fixed_conn_num_csc_structure(idx, shape=shape)

        # ``with_data`` is structure-preserving, so the (data-independent) CSC
        # mirror is carried through to the new matrix (parity with CSR/CSC).
        assert updated.buffers.get('csc') is not None
        csc_indptr, csc_indices, perm = updated._weight_indices()
        assert jnp.array_equal(csc_indptr, exp_p)
        assert jnp.array_equal(csc_indices, exp_i)
        assert jnp.array_equal(perm, exp_perm)
        # The unfavorable matvec reflects the *new* weights, not the old ones.
        spikes = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.7, 1.0], dtype=jnp.float32))
        assert not allclose(updated @ spikes, conn @ spikes)

    def test_transpose_binary_vector_matches_dense(self):
        shape = (3, 4)
        idx = generate_fixed_conn_num_indices(*shape, 2)
        data = braintools.init.Normal(0., 1.)(idx.shape)
        conn = brainevent.FixedNumPerPre((data, idx), shape=shape)
        transposed = conn.T
        assert isinstance(transposed, brainevent.FixedNumPerPost)
        assert transposed.shape == (4, 3)

        spikes = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.5], dtype=jnp.float32))
        dense = transposed.todense()
        assert allclose(transposed @ spikes, dense @ _binary_mask(spikes.value, dense.dtype))



@pytest.mark.slow
class Test_Init_Outside_JIT:
    def test_fixed_post_init_rejects_first_construction_inside_jax_jit(self):
        @jax.jit
        def build():
            idx = jnp.array([[0, 1], [1, 3]], dtype=jnp.int32)
            data = jnp.array([[1., 2.], [3., 4.]], dtype=jnp.float32)
            conn = brainevent.FixedNumPerPre((data, idx), shape=(2, 4))
            return conn.nse

        with pytest.raises(RuntimeError, match='must be first constructed outside'):
            build()

    def test_fixed_pre_init_rejects_first_construction_inside_brainstate_jit(self):
        @brainstate.transform.jit
        def build():
            idx = jnp.array([[0, 1], [2, 1], [3, 0]], dtype=jnp.int32)
            data = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
            conn = brainevent.FixedNumPerPost(
                (data, idx),
                shape=(4, 3),
            )
            return conn.nse

        with pytest.raises(RuntimeError, match='must be first constructed outside'):
            build()


@pytest.mark.slow
class Test_Operator_Behavior:
    def test_fixed_post_binary_array_operator_behavior(self):
        idx = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        conn = brainevent.FixedNumPerPre((data, idx), shape=(3, 4))
        dense = conn.todense()

        left_vector = brainevent.BinaryArray(jnp.array([0.2, 0.0, 1.0], dtype=jnp.float32))
        right_vector = brainevent.BinaryArray(jnp.array([0.0, 0.6, 0.0, 1.0], dtype=jnp.float32))
        left_matrix = brainevent.BinaryArray(jnp.array([[0.0, 1.0, 0.3], [1.0, 0.0, 0.0]], dtype=jnp.float32))
        right_matrix = brainevent.BinaryArray(
            jnp.array([[0.0, 1.0], [0.2, 0.0], [1.0, 0.0], [0.0, 0.4]], dtype=jnp.float32)
        )

        assert allclose(left_vector @ conn, _binary_mask(left_vector.value, dense.dtype) @ dense)
        # The event-driven W @ s gather now resolves to the CSC column-scatter
        # path on CUDA (and to the ELL gather elsewhere); both match dense.
        assert allclose(conn @ right_vector, dense @ _binary_mask(right_vector.value, dense.dtype))
        assert allclose(left_matrix @ conn, _binary_mask(left_matrix.value, dense.dtype) @ dense)
        assert allclose(conn @ right_matrix, dense @ _binary_mask(right_matrix.value, dense.dtype))
        jax.block_until_ready(
            (idx, data, dense, left_vector.value, right_vector.value, left_matrix.value, right_matrix.value))

    def test_fixed_pre_binary_array_operator_behavior(self):
        idx = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        conn = brainevent.FixedNumPerPost((data, idx), shape=(4, 3))
        dense = conn.todense()

        left_vector = brainevent.BinaryArray(jnp.array([0.0, 1.0, 0.4, 1.0], dtype=jnp.float32))
        right_vector = brainevent.BinaryArray(jnp.array([1.0, 0.0, 0.7], dtype=jnp.float32))
        left_matrix = brainevent.BinaryArray(
            jnp.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.2]], dtype=jnp.float32)
        )
        right_matrix = brainevent.BinaryArray(jnp.array([[0.2, 0.0], [1.0, 1.0], [0.0, 0.8]], dtype=jnp.float32))

        # The event-driven W @ s gather (x @ conn for fixed-pre) now resolves to
        # the CSC column-scatter path on CUDA (ELL gather elsewhere); both match.
        assert allclose(left_vector @ conn, _binary_mask(left_vector.value, dense.dtype) @ dense)
        assert allclose(conn @ right_vector, dense @ _binary_mask(right_vector.value, dense.dtype))
        assert allclose(left_matrix @ conn, _binary_mask(left_matrix.value, dense.dtype) @ dense)
        assert allclose(conn @ right_matrix, dense @ _binary_mask(right_matrix.value, dense.dtype))
        jax.block_until_ready(
            (idx, data, dense, left_vector.value, right_vector.value, left_matrix.value, right_matrix.value))




@pytest.mark.slow
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
            conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
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
            conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
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
        conn = brainevent.FixedNumPerPre((w, indices), shape=shape)

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
        conn = brainevent.FixedNumPerPre((w, indices), shape=shape)

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
            conn = brainevent.FixedNumPerPre((w, indices), shape=(m, n))
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
            conn = brainevent.FixedNumPerPre((w, indices), shape=(m, n))
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
            conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
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
            conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
            y1 = conn @ x
            y2 = x @ conn.T
            y_true = fcn_vector(x, conn.data, indices, (m, n))
            return y1, y2, y_true

        for xs in self._generate_x([batch_size, n] if batch_axis == 0 else [n, batch_size]):
            y1, y2, y_true = f_compare_conn_vector(xs)
            assert allclose(y1, y_true, rtol=1e-3, atol=1e-3)
            assert allclose(y2, y_true, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((indices, y1, y2, y_true))


@pytest.mark.slow
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
            conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
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
            conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
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
        conn = brainevent.FixedNumPerPre((w, indices), shape=shape)

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
        conn = brainevent.FixedNumPerPre((w, indices), shape=shape)

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
        _skip_batching_if_gpu_too_small(shape, batch_size)
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
            conn = brainevent.FixedNumPerPre((w, indices), shape=(m, n))
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
            conn = brainevent.FixedNumPerPre((w, indices), shape=(m, n))
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
        _skip_batching_if_gpu_too_small(shape, batch_size)
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1))

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        @jax.jit
        @functools.partial(jax.vmap, in_axes=batch_axis)
        def f_compare_vector_conn(x):
            conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
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
            conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
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


@pytest.mark.slow
class Test_dt2t:
    def test_dt2t_signature_aligns_data_representation_contract(self):
        base_sig = inspect.signature(fcn_main_mod.DataRepresentation.dt2t)
        for cls in (FixedNumPerPre, FixedNumPerPost):
            sig = inspect.signature(cls.dt2t)
            assert list(sig.parameters) == list(base_sig.parameters)
            assert sig.parameters['y_dim_arr'].annotation == base_sig.parameters['y_dim_arr'].annotation
            assert sig.parameters['w_dim_arr'].annotation == base_sig.parameters['w_dim_arr'].annotation
            assert sig.parameters['w_dim_arr'].default is inspect._empty
            assert sig.return_annotation == base_sig.return_annotation

    def test_dt2t_transposed_signature_aligns_data_representation_contract(self):
        base_sig = inspect.signature(fcn_main_mod.DataRepresentation.dt2t_transposed)
        for cls in (FixedNumPerPre, FixedNumPerPost):
            sig = inspect.signature(cls.dt2t_transposed)
            assert list(sig.parameters) == list(base_sig.parameters)
            assert sig.parameters['y_dim_arr'].annotation == base_sig.parameters['y_dim_arr'].annotation
            assert sig.parameters['w_dim_arr'].annotation == base_sig.parameters['w_dim_arr'].annotation
            assert sig.parameters['w_dim_arr'].default is inspect._empty
            assert sig.return_annotation == base_sig.return_annotation

    def test_fixed_post(self):
        m, n, k = 5, 7, 3
        indices = generate_fixed_conn_num_indices(m, n, k, replace=True)
        data = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
        conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
        y_pre = jnp.arange(1, m + 1, dtype=jnp.float32)
        y_post = jnp.arange(1, n + 1, dtype=jnp.float32)

        # dt2t: y indexed by row=pre -> broadcast
        assert allclose(conn.dt2t(y_pre, data), data * y_pre[:, None])
        # dt2t_transposed: y indexed by col=post -> gather
        assert allclose(conn.dt2t_transposed(y_post, data), data * y_post[indices])

    def test_fixed_pre(self):
        num_pre, num_post, k = 7, 5, 3
        # indices: (num_post, k) with values in [0, num_pre)
        indices = generate_fixed_conn_num_indices(num_post, num_pre, k, replace=True)
        data = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
        conn = brainevent.FixedNumPerPost((data, indices), shape=(num_pre, num_post))
        y_pre = jnp.arange(1, num_pre + 1, dtype=jnp.float32)
        y_post = jnp.arange(1, num_post + 1, dtype=jnp.float32)

        # dt2t: y indexed by row=pre -> gather (indices are pre ids)
        assert allclose(conn.dt2t(y_pre, data), data * y_pre[indices])
        # dt2t_transposed: y indexed by col=post=leading -> broadcast
        assert allclose(conn.dt2t_transposed(y_post, data), data * y_post[:, None])

    def test_w_dim_arr_is_required(self):
        m, n, k = 5, 7, 3
        indices = generate_fixed_conn_num_indices(m, n, k, replace=True)
        data = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
        conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))
        y_pre = jnp.arange(1, m + 1, dtype=jnp.float32)
        y_post = jnp.arange(1, n + 1, dtype=jnp.float32)
        with pytest.raises(TypeError):
            conn.dt2t(y_pre)
        with pytest.raises(TypeError):
            conn.dt2t_transposed(y_post)

    def test_golden_parity_csr(self):
        m, n, k = 5, 7, 3
        indices = generate_fixed_conn_num_indices(m, n, k, replace=True)
        data = jnp.arange(1, indices.size + 1, dtype=jnp.float32).reshape(indices.shape)
        conn = brainevent.FixedNumPerPre((data, indices), shape=(m, n))

        indptr = jnp.arange(m + 1, dtype=jnp.int32) * k
        csr = brainevent.CSR(
            (data.flatten(), indices.flatten().astype(jnp.int32), indptr), shape=(m, n)
        )
        y_pre = jnp.arange(1, m + 1, dtype=jnp.float32)
        y_post = jnp.arange(1, n + 1, dtype=jnp.float32)

        assert allclose(conn.dt2t(y_pre, data).flatten(),
                        csr.dt2t(y_pre, data.flatten()))
        assert allclose(conn.dt2t_transposed(y_post, data).flatten(),
                        csr.dt2t_transposed(y_post, data.flatten()))


# --------------------------------------------------------------------------- #
# Buffer model + data-only pytree leaf + jit-survives-mirror (CSR/CSC parity)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_fcn_buffers_and_build_weight_indices():
    rng = np.random.default_rng(0)
    for cls, rows, upper in ((brainevent.FixedNumPerPre, 6, 5),
                             (brainevent.FixedNumPerPost, 5, 6)):
        idx = jnp.asarray(rng.integers(0, upper, size=(rows, 3)).astype(np.int32))
        dat = jnp.asarray(rng.random((rows, 3)) + 0.5, dtype=jnp.float32)
        m = cls(dat, idx, shape=(6, 5))
        # mirror not built yet (lazy)
        assert m.buffers.get('csc') is None
        # eager builder returns a new instance with the mirror cached
        m2 = m.build_weight_indices()
        assert m2.buffers.get('csc') is not None
        assert m2.data is m.data and m2.indices is m.indices
        # precompute flag builds it at construction time
        m3 = cls(dat, idx, shape=(6, 5), precompute_weight_indices=True)
        assert m3.buffers.get('csc') is not None


@pytest.mark.slow
def test_fcn_pytree_roundtrip_data_only_leaf():
    rng = np.random.default_rng(1)
    idx = jnp.asarray(rng.integers(0, 5, size=(6, 3)).astype(np.int32))
    dat = jnp.asarray(rng.random((6, 3)) + 0.5, dtype=jnp.float32)
    m = brainevent.FixedNumPerPre(dat, idx, shape=(6, 5), precompute_weight_indices=True)
    leaves, treedef = jax.tree_util.tree_flatten(m)
    assert len(leaves) == 1  # data only
    m2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.allclose(m2.data, m.data)
    assert jnp.array_equal(m2.indices, m.indices)
    assert m2.shape == m.shape
    assert m2.buffers.get('csc') is not None  # mirror carried through aux


@pytest.mark.slow
def test_fcn_jit_plasticity_loop_with_precomputed_mirror():
    rng = np.random.default_rng(2)
    # Distinct columns per row so todense() has one slot per (i, j) and the dense
    # reference does not double-count the per-synapse delta.
    idx = jnp.asarray(
        np.stack([rng.choice(5, size=3, replace=False) for _ in range(6)]).astype(np.int32)
    )
    dat = jnp.asarray(rng.random((6, 3)) + 0.5, dtype=jnp.float32)
    m = brainevent.FixedNumPerPre(dat, idx, shape=(6, 5), precompute_weight_indices=True)
    pre_trace = jnp.asarray(rng.random(6), dtype=jnp.float32)
    post_spike = jnp.asarray(rng.random(5) > 0.5)

    @jax.jit
    def step(mat):
        return mat.update_on_post(pre_trace, post_spike)

    out = step(m)  # unfavorable plasticity inside jit; mirror precomputed
    dense = jnp.asarray(m.todense(), jnp.float32)
    ref = dense + (pre_trace[:, None] * jnp.asarray(post_spike, jnp.float32)[None, :]) * (dense != 0)
    assert jnp.allclose(jnp.asarray(out.todense(), jnp.float32), ref, atol=1e-5)


# --- merged from brainevent/_fcn/golden_parity_test.py and mm_golden_parity_test.py ---
# Golden parity pinning FixedNumPerPost/FixedNumPerPre matvec & matmat behavior
# against the dense reference ``M.todense()``. The two source files shared a
# byte-identical ``_cases`` helper; one copy is kept here.


def _cases():
    rng = np.random.default_rng(0)
    n_pre, n_post, n_conn = 6, 5, 3
    for cls in (FixedNumPerPre, FixedNumPerPost):
        rows = n_pre if cls is FixedNumPerPre else n_post
        upper = n_post if cls is FixedNumPerPre else n_pre
        for homo in (True, False):
            indices = rng.integers(0, upper, size=(rows, n_conn)).astype(np.int32)
            if homo:
                data = jnp.asarray(rng.random(1) + 0.5, dtype=jnp.float32)
            else:
                data = jnp.asarray(rng.random((rows, n_conn)) + 0.5, dtype=jnp.float32)
            yield cls, data, jnp.asarray(indices), (n_pre, n_post)


@pytest.mark.slow
def test_fcn_matvec_golden():
    rng = np.random.default_rng(7)
    for cls, data, indices, shape in _cases():
        M = cls(data, indices, shape=shape)
        dense = jnp.asarray(M.todense(), dtype=jnp.float32)
        n_pre, n_post = shape
        for ev_dtype in (jnp.bool_, jnp.float32):
            left = jnp.asarray(rng.random(n_pre) > 0.5, dtype=ev_dtype)
            right = jnp.asarray(rng.random(n_post) > 0.5, dtype=ev_dtype)

            got_l = BinaryArray(left) @ M
            ref_l = jnp.asarray(left, dtype=jnp.float32) @ dense
            assert jnp.allclose(got_l, ref_l, atol=1e-5), (cls.__name__, str(ev_dtype), 'left')

            got_r = M @ BinaryArray(right)
            ref_r = dense @ jnp.asarray(right, dtype=jnp.float32)
            assert jnp.allclose(got_r, ref_r, atol=1e-5), (cls.__name__, str(ev_dtype), 'right')


def _hetero_cases():
    rng = np.random.default_rng(11)
    n_pre, n_post, n_conn = 6, 5, 3
    for cls in (FixedNumPerPre, FixedNumPerPost):
        rows = n_pre if cls is FixedNumPerPre else n_post
        upper = n_post if cls is FixedNumPerPre else n_pre
        for homo in (True, False):
            indices = rng.integers(0, upper, size=(rows, n_conn)).astype(np.int32)
            data = (jnp.asarray(rng.random(1) + 0.5, dtype=jnp.float32) if homo
                    else jnp.asarray(rng.random((rows, n_conn)) + 0.5, dtype=jnp.float32))
            yield cls, jnp.asarray(data), jnp.asarray(indices), (n_pre, n_post)


def _distinct_indices(rng, rows, upper, n_conn):
    """Indices with distinct columns per row (so todense() has one slot per (i, j))."""
    return jnp.asarray(
        np.stack([rng.choice(upper, size=n_conn, replace=False) for _ in range(rows)]).astype(np.int32)
    )


@pytest.mark.slow
def test_fcn_unfavorable_matmat_golden():
    rng = np.random.default_rng(3)
    k = 4
    for cls, data, indices, shape in _hetero_cases():
        M = cls(data, indices, shape=shape)
        dense = jnp.asarray(M.todense(), dtype=jnp.float32)
        n_pre, n_post = shape
        for ev_dtype in (jnp.bool_, jnp.float32):
            X = jnp.asarray(rng.random((n_post, k)) > 0.5, dtype=ev_dtype)
            got = M @ BinaryArray(X)
            ref = dense @ jnp.asarray(X, dtype=jnp.float32)
            assert jnp.allclose(got, ref, atol=1e-5), (cls.__name__, 'M@X', str(ev_dtype))

            Xl = jnp.asarray(rng.random((k, n_pre)) > 0.5, dtype=ev_dtype)
            got_l = BinaryArray(Xl) @ M
            ref_l = jnp.asarray(Xl, dtype=jnp.float32) @ dense
            assert jnp.allclose(got_l, ref_l, atol=1e-5), (cls.__name__, 'X@M', str(ev_dtype))


@pytest.mark.slow
def test_fcn_matvec_grad_golden():
    rng = np.random.default_rng(5)
    for cls, data, indices, shape in _hetero_cases():
        if data.size == 1:
            continue  # gradient wrt per-synapse (heterogeneous) weights
        n_pre, n_post = shape
        ev = jnp.asarray(rng.random(n_post) > 0.5, dtype=jnp.float32)

        def f(d, _indices=indices):
            return (cls(d, _indices, shape=shape) @ BinaryArray(ev)).sum()

        def f_dense(d, _indices=indices):
            M = cls(d, _indices, shape=shape)
            return (jnp.asarray(M.todense(), dtype=jnp.float32)
                    @ jnp.asarray(ev, dtype=jnp.float32)).sum()

        g = jax.grad(f)(data)
        g_ref = jax.grad(f_dense)(data)
        assert jnp.allclose(g, g_ref, atol=1e-4), (cls.__name__, 'grad')


@pytest.mark.slow
def test_fcn_plasticity_unfavorable_golden():
    rng = np.random.default_rng(7)
    n_pre, n_post, n_conn = 6, 5, 3
    for cls in (FixedNumPerPre, FixedNumPerPost):
        rows = n_pre if cls is FixedNumPerPre else n_post
        upper = n_post if cls is FixedNumPerPre else n_pre
        indices = _distinct_indices(rng, rows, upper, n_conn)
        data = jnp.asarray(rng.random((rows, n_conn)) + 0.5, dtype=jnp.float32)
        M = cls(data, indices, shape=(n_pre, n_post))
        dense = jnp.asarray(M.todense(), dtype=jnp.float32)
        mask = (dense != 0).astype(jnp.float32)
        pre_trace = jnp.asarray(rng.random(n_pre), dtype=jnp.float32)
        post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)
        pre_spike = jnp.asarray(rng.random(n_pre) > 0.5)
        post_spike = jnp.asarray(rng.random(n_post) > 0.5)
        for w_min, w_max in ((None, None), (0.0, 1.0)):
            up = M.update_on_post(pre_trace, post_spike, w_min=w_min, w_max=w_max)
            ref = dense + (pre_trace[:, None] * jnp.asarray(post_spike, jnp.float32)[None, :]) * mask
            if w_min is not None:
                ref = jnp.clip(ref, w_min, w_max) * mask
            assert jnp.allclose(jnp.asarray(up.todense(), jnp.float32), ref, atol=1e-5), (cls.__name__, 'on_post', w_min)

            up2 = M.update_on_pre(pre_spike, post_trace, w_min=w_min, w_max=w_max)
            ref2 = dense + (jnp.asarray(pre_spike, jnp.float32)[:, None] * post_trace[None, :]) * mask
            if w_min is not None:
                ref2 = jnp.clip(ref2, w_min, w_max) * mask
            assert jnp.allclose(jnp.asarray(up2.todense(), jnp.float32), ref2, atol=1e-5), (cls.__name__, 'on_pre', w_min)


@pytest.mark.slow
def test_fcn_matmat_golden():
    rng = np.random.default_rng(7)
    for cls, data, indices, shape in _cases():
        M = cls(data, indices, shape=shape)
        dense = jnp.asarray(M.todense(), dtype=jnp.float32)
        n_pre, n_post = shape
        for ev in (jnp.bool_, jnp.float32):
            for n in (1, 4):
                right = jnp.asarray(rng.random((n_post, n)) > 0.5, dtype=ev)   # W @ M
                got_r = M @ BinaryArray(right)
                ref_r = dense @ jnp.asarray(right, dtype=jnp.float32)
                assert jnp.allclose(got_r, ref_r, atol=1e-5), (cls.__name__, str(ev), n, 'right')

                left = jnp.asarray(rng.random((n, n_pre)) > 0.5, dtype=ev)     # M @ W
                got_l = BinaryArray(left) @ M
                ref_l = jnp.asarray(left, dtype=jnp.float32) @ dense
                assert jnp.allclose(got_l, ref_l, atol=1e-5), (cls.__name__, str(ev), n, 'left')


@pytest.mark.slow
def test_fixed_num_per_post_rmatmul_mirror_path_transposes_indexed_output():
    n_pre, n_post, nbatch = 3, 5, 4
    indices = jnp.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 2],
            [1, 0],
        ],
        dtype=jnp.int32,
    )
    data = jnp.arange(1, 11, dtype=jnp.float32).reshape(n_post, 2)
    mat = FixedNumPerPost((data, indices), shape=(n_pre, n_post), backend='jax_raw')
    left = jnp.array(
        [
            [True, False, True],
            [False, True, False],
            [True, True, False],
            [False, False, True],
        ],
        dtype=jnp.bool_,
    )

    assert mat._ell_transpose(True) is False

    got = BinaryArray(left) @ mat
    ref = jnp.asarray(left, dtype=jnp.float32) @ jnp.asarray(mat.todense(), dtype=jnp.float32)

    assert got.shape == (nbatch, n_post)
    assert ref.shape == (nbatch, n_post)
    assert jnp.allclose(got, ref, atol=1e-5)


@pytest.mark.slow
def test_fcn_binary_matmat_dispatch_contract():
    dispatch_source = inspect.getsource(fcn_main_mod.FixedNumConn._dispatch)
    assert "_align_binary_matmat_output" in dispatch_source
    assert "_binary_matmat(value.T, transpose_W).T" not in dispatch_source
    assert "expected_shape = (value.shape[0], self.shape[1])" in dispatch_source
    assert "expected_shape = (self.shape[0], value.shape[1])" in dispatch_source


@pytest.mark.slow
def test_fcn_binary_matmat_output_adapter_contract():
    raw_batch_first = jnp.ones((3, 5))
    raw_neuron_first = jnp.ones((5, 3))
    assert fcn_main_mod._align_binary_matmat_output(
        raw_batch_first, (3, 5), "pre"
    ).shape == (3, 5)
    assert fcn_main_mod._align_binary_matmat_output(
        raw_neuron_first, (3, 5), "pre"
    ).shape == (3, 5)
    with pytest.raises(ValueError, match="binary matmat output shape mismatch"):
        fcn_main_mod._align_binary_matmat_output(jnp.ones((2, 4)), (3, 5), "pre")


@pytest.mark.slow
def test_fcn_matmat_unfavorable_builds_weight_indices():
    # Unfavorable matmat must build the cached CSC mirror (perm-fused path),
    # exactly as the matvec unfavorable path does. FixedNumPerPre mirrors CSR,
    # so __matmul__ (W @ M, transpose_W=False) is the *unfavorable* direction.
    # The mirror is cached in the buffer registry (self.buffers['csc']).
    rng = np.random.default_rng(5)
    cls, data, indices, shape = next(_cases())   # FixedNumPerPre, homo
    M = cls(data, indices, shape=shape)
    n_pre, n_post = shape
    assert M.buffers.get('csc') is None            # FCN caches its CSC view in buffers['csc']
    right = jnp.asarray(rng.random((n_post, 3)) > 0.5, dtype=jnp.bool_)
    _ = M @ BinaryArray(right)
    assert M.buffers.get('csc') is not None


@pytest.mark.slow
def test_fcn_matmat_units():
    rng = np.random.default_rng(10)
    cls, data, indices, shape = next(_cases())
    M = cls(data * u.mV, indices, shape=shape)
    dense = jnp.asarray(cls(data, indices, shape=shape).todense(), jnp.float32)
    n_pre, n_post = shape
    right = jnp.asarray(rng.random((n_post, 4)) > 0.5, dtype=jnp.bool_)
    got = M @ BinaryArray(right)
    assert u.get_unit(got) == u.mV
    ref = dense @ jnp.asarray(right, jnp.float32)
    assert jnp.allclose(u.get_mantissa(got), ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Format conversions: ``todense``, ``tocsr`` and ``tocsc``.
#
# All correctness checks route through the dense round-trip, which uses the
# pure-JAX ``jax.experimental.sparse`` primitives (``coo_todense`` /
# ``csr_todense``) and therefore needs no compilation-heavy backend -- so these
# are deliberately *not* marked ``slow`` and run in the default ``pytest`` lane.
# ---------------------------------------------------------------------------

# Duplicate columns within a row exercise the accumulation semantics shared by
# ``todense`` and the matmul kernels: row 0 has column ``2`` twice; row 1 has
# columns ``3`` and ``1`` twice.
DUP_INDICES = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
DUP_DATA = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)


def _make_conn(cls, homo, shape, n_conn, seed=0):
    """Build a connection of ``cls`` with random structure and values."""
    pre = shape[0] if cls is FixedNumPerPre else shape[1]
    n_post = shape[1] if cls is FixedNumPerPre else shape[0]
    # ``replace=True`` keeps the cheap ``randint`` path (no ``for_loop``); the
    # duplicate connections it may produce are themselves useful coverage.
    indices = generate_fixed_conn_num_indices(pre, n_post, n_conn, replace=True)
    if homo:
        data = jnp.asarray(1.5, dtype=jnp.float32)
    else:
        rng = np.random.default_rng(seed)
        data = jnp.asarray(rng.standard_normal(indices.shape), dtype=jnp.float32)
    return cls((data, indices), shape=shape)


CONVERSION_CLASSES = [FixedNumPerPre, FixedNumPerPost]
CONVERSION_SHAPES = [(6, 8), (10, 5)]


class TestToDense:
    @pytest.mark.parametrize('cls', CONVERSION_CLASSES)
    @pytest.mark.parametrize('shape', CONVERSION_SHAPES)
    @pytest.mark.parametrize('homo', [True, False])
    def test_todense_shape(self, cls, shape, homo):
        conn = _make_conn(cls, homo, shape, n_conn=3)
        assert conn.todense().shape == shape


class TestToCsr:
    @pytest.mark.parametrize('cls', CONVERSION_CLASSES)
    @pytest.mark.parametrize('shape', CONVERSION_SHAPES)
    @pytest.mark.parametrize('homo', [True, False])
    def test_roundtrip_matches_dense(self, cls, shape, homo):
        conn = _make_conn(cls, homo, shape, n_conn=4)
        csr = conn.tocsr()
        assert isinstance(csr, CSR)
        assert csr.shape == shape
        assert allclose(csr.todense(), conn.todense())

    @pytest.mark.parametrize('cls', CONVERSION_CLASSES)
    def test_metadata(self, cls):
        shape = (6, 8)
        conn = _make_conn(cls, homo=False, shape=shape, n_conn=3)
        csr = conn.tocsr()
        assert csr.nse == conn.indices.size
        assert csr.dtype == conn.dtype
        assert csr.indptr.shape == (shape[0] + 1,)

    @pytest.mark.parametrize('cls', CONVERSION_CLASSES)
    def test_homogeneous_weight_is_kept_compact(self, cls):
        conn = _make_conn(cls, homo=True, shape=(6, 8), n_conn=3)
        csr = conn.tocsr()
        # A single shared value is preserved rather than materialised per entry.
        assert csr.data.size == 1
        assert allclose(csr.todense(), conn.todense())

    @pytest.mark.parametrize('cls', CONVERSION_CLASSES)
    def test_num_conn_one(self, cls):
        conn = _make_conn(cls, homo=False, shape=(5, 7), n_conn=1)
        assert allclose(conn.tocsr().todense(), conn.todense())

    def test_duplicates_preserved_pre(self):
        conn = FixedNumPerPre((DUP_DATA, DUP_INDICES), shape=(3, 4))
        assert allclose(conn.tocsr().todense(), conn.todense())

    def test_duplicates_preserved_post(self):
        conn = FixedNumPerPost((DUP_DATA, DUP_INDICES), shape=(4, 3))
        assert allclose(conn.tocsr().todense(), conn.todense())

    def test_units_preserved(self):
        conn = FixedNumPerPre((DUP_DATA * u.mS, DUP_INDICES), shape=(3, 4))
        csr = conn.tocsr()
        assert u.get_unit(csr.data) == u.mS
        assert u.get_unit(csr.todense()) == u.mS
        assert allclose(u.get_mantissa(csr.todense()), u.get_mantissa(conn.todense()))

    def test_backend_propagated(self):
        conn = FixedNumPerPre((DUP_DATA, DUP_INDICES), shape=(3, 4), backend='numba')
        assert conn.tocsr().backend == 'numba'

    def test_requires_outside_jit(self):
        conn = FixedNumPerPre((DUP_DATA, DUP_INDICES), shape=(3, 4))
        with pytest.raises(RuntimeError, match='outside'):
            jax.jit(lambda m: m.tocsr())(conn)


class TestToCsc:
    @pytest.mark.parametrize('cls', CONVERSION_CLASSES)
    @pytest.mark.parametrize('shape', CONVERSION_SHAPES)
    @pytest.mark.parametrize('homo', [True, False])
    def test_roundtrip_matches_dense(self, cls, shape, homo):
        conn = _make_conn(cls, homo, shape, n_conn=4)
        csc = conn.tocsc()
        assert isinstance(csc, CSC)
        assert csc.shape == shape
        assert allclose(csc.todense(), conn.todense())

    def test_duplicates_preserved_pre(self):
        conn = FixedNumPerPre((DUP_DATA, DUP_INDICES), shape=(3, 4))
        assert allclose(conn.tocsc().todense(), conn.todense())

    def test_duplicates_preserved_post(self):
        conn = FixedNumPerPost((DUP_DATA, DUP_INDICES), shape=(4, 3))
        assert allclose(conn.tocsc().todense(), conn.todense())

    def test_units_preserved(self):
        conn = FixedNumPerPost((DUP_DATA * u.mV, DUP_INDICES), shape=(4, 3))
        csc = conn.tocsc()
        assert u.get_unit(csc.data) == u.mV
        assert u.get_unit(csc.todense()) == u.mV
        assert allclose(u.get_mantissa(csc.todense()), u.get_mantissa(conn.todense()))


class TestCrossFormatConsistency:
    @pytest.mark.parametrize('cls', CONVERSION_CLASSES)
    @pytest.mark.parametrize('homo', [True, False])
    def test_csr_csc_dense_agree(self, cls, homo):
        conn = _make_conn(cls, homo, (7, 9), n_conn=3)
        dense = conn.todense()
        assert allclose(conn.tocsr().todense(), dense)
        assert allclose(conn.tocsc().todense(), dense)
        # Transpose equivalence: CSR of W^T equals CSC of W, array-for-array.
        assert allclose(conn.tocsr().todense().T, conn.transpose().tocsr().todense())
        jax.block_until_ready((dense,))


# ---------------------------------------------------------------------------
# Row indexing and slicing: ``__getitem__`` and ``slice_rows``.
# Dense-reference comparisons only -- no native backend, so not marked ``slow``.
# ---------------------------------------------------------------------------


def _make_perpre(m, n, num_conn, homo=False):
    rng = np.random.default_rng(0)
    indices = np.stack(
        [rng.choice(n, size=num_conn, replace=False) for _ in range(m)]
    ).astype(np.int32)
    if homo:
        data = jnp.array(1.5, dtype=jnp.float32)
    else:
        data = jnp.asarray(rng.standard_normal((m, num_conn)).astype(np.float32))
    conn = FixedNumPerPre((data, jnp.asarray(indices)), shape=(m, n))
    return conn


def _make_perpost(m, n, num_conn, homo=False):
    rng = np.random.default_rng(1)
    indices = np.stack(
        [rng.choice(m, size=num_conn, replace=False) for _ in range(n)]
    ).astype(np.int32)
    if homo:
        data = jnp.array(2.0, dtype=jnp.float32)
    else:
        data = jnp.asarray(rng.standard_normal((n, num_conn)).astype(np.float32))
    conn = FixedNumPerPost((data, jnp.asarray(indices)), shape=(m, n))
    return conn


class TestFixedNumPerPreGetitem:

    @pytest.mark.parametrize('homo', [False, True])
    def test_single_row(self, homo):
        conn = _make_perpre(8, 12, 3, homo=homo)
        dense = conn.todense()
        r = conn[3]
        assert r.shape == (12,)
        assert jnp.allclose(r, dense[3], atol=1e-5)

    @pytest.mark.parametrize('homo', [False, True])
    def test_multi_and_slice(self, homo):
        conn = _make_perpre(8, 12, 3, homo=homo)
        dense = conn.todense()
        assert jnp.allclose(conn[[1, 4, 6]], np.asarray(dense)[[1, 4, 6]], atol=1e-5)
        assert jnp.allclose(conn[2:7:2], dense[np.arange(2, 7, 2)], atol=1e-5)

    def test_negative_and_oob(self):
        conn = _make_perpre(8, 12, 3)
        dense = conn.todense()
        assert jnp.allclose(conn[-1], dense[7], atol=1e-5)
        with pytest.raises(IndexError):
            _ = conn[8]


class TestFixedNumPerPreSliceRows:

    @pytest.mark.parametrize('homo', [False, True])
    def test_returns_same_type_and_matches_dense(self, homo):
        conn = _make_perpre(8, 12, 3, homo=homo)
        dense = conn.todense()
        sub = conn.slice_rows([1, 3, 5])
        assert isinstance(sub, FixedNumPerPre)
        assert sub.shape == (3, 12)
        assert sub.num_conn == 3
        assert jnp.allclose(sub.todense(), np.asarray(dense)[[1, 3, 5]], atol=1e-5)

    def test_single_int_is_one_row(self):
        conn = _make_perpre(8, 12, 3)
        dense = conn.todense()
        sub = conn.slice_rows(4)
        assert isinstance(sub, FixedNumPerPre)
        assert sub.shape == (1, 12)
        assert jnp.allclose(sub.todense(), dense[4:5], atol=1e-5)

    def test_jit_safe(self):
        conn = _make_perpre(8, 12, 3)
        dense = conn.todense()
        idx = jnp.array([0, 2, 4], dtype=jnp.int32)

        def f(c, i):
            return c.slice_rows(i).todense()
        out = jax.jit(f)(conn, idx)
        assert jnp.allclose(out, np.asarray(dense)[[0, 2, 4]], atol=1e-5)


class TestFixedNumPerPostGetitem:

    @pytest.mark.parametrize('homo', [False, True])
    def test_single_and_multi_row(self, homo):
        conn = _make_perpost(10, 7, 4, homo=homo)  # W is (10, 7)
        dense = conn.todense()
        assert conn[3].shape == (7,)
        assert jnp.allclose(conn[3], dense[3], atol=1e-5)
        assert jnp.allclose(conn[[0, 4, 8]], np.asarray(dense)[[0, 4, 8]], atol=1e-5)

    def test_slice_negative_oob(self):
        conn = _make_perpost(10, 7, 4)
        dense = conn.todense()
        assert jnp.allclose(conn[1:9:3], dense[np.arange(1, 9, 3)], atol=1e-5)
        assert jnp.allclose(conn[-1], dense[9], atol=1e-5)
        with pytest.raises(IndexError):
            _ = conn[10]


class TestFixedNumPerPostSliceRows:

    @pytest.mark.parametrize('homo', [False, True])
    def test_returns_csr_and_matches_dense(self, homo):
        conn = _make_perpost(10, 7, 4, homo=homo)
        dense = conn.todense()
        sub = conn.slice_rows([1, 3, 5])
        assert isinstance(sub, CSR)
        assert sub.shape == (3, 7)
        assert jnp.allclose(sub.todense(), np.asarray(dense)[[1, 3, 5]], atol=1e-5)

    def test_slice_and_single(self):
        conn = _make_perpost(10, 7, 4)
        dense = conn.todense()
        sub = conn.slice_rows(slice(0, 6, 2))
        assert isinstance(sub, CSR)
        assert jnp.allclose(sub.todense(), dense[np.arange(0, 6, 2)], atol=1e-5)
        one = conn.slice_rows(4)
        assert one.shape == (1, 7)
        assert jnp.allclose(one.todense(), dense[4:5], atol=1e-5)


class TestFcnSliceAD:

    def test_grad_perpre(self):
        conn = _make_perpre(8, 12, 3)
        rows = jnp.array([1, 3, 5], dtype=jnp.int32)

        def loss(data):
            c = FixedNumPerPre((data, conn.indices), shape=conn.shape)
            return jnp.sum(c[rows] ** 2)
        g = jax.grad(loss)(conn.data)
        # finite-difference check on one entry
        eps = 1e-3
        d = conn.data
        d1 = d.at[0, 0].add(eps)
        d2 = d.at[0, 0].add(-eps)
        num = (loss(d1) - loss(d2)) / (2 * eps)
        assert jnp.allclose(g[0, 0], num, atol=1e-2)

    def test_grad_perpost(self):
        conn = _make_perpost(10, 7, 4)
        rows = jnp.array([0, 4, 8], dtype=jnp.int32)

        def loss(data):
            c = FixedNumPerPost((data, conn.indices), shape=conn.shape)
            return jnp.sum(c[rows] ** 2)
        g = jax.grad(loss)(conn.data)
        eps = 1e-3
        d = conn.data
        d1 = d.at[0, 0].add(eps)
        d2 = d.at[0, 0].add(-eps)
        num = (loss(d1) - loss(d2)) / (2 * eps)
        assert jnp.allclose(g[0, 0], num, atol=1e-2)

    def test_vmap_perpre(self):
        conn = _make_perpre(8, 12, 3)
        dense = conn.todense()
        batched_rows = jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)
        out = jax.vmap(lambda r: conn[r])(batched_rows)
        assert out.shape == (3, 2, 12)
        assert jnp.allclose(out[0], np.asarray(dense)[[0, 1]], atol=1e-5)
        assert jnp.allclose(out[2], np.asarray(dense)[[4, 5]], atol=1e-5)
