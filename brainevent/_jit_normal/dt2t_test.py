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

import inspect

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._jit_normal.csr import (
    jitn_csr_count_p_call,
    jitn_csr_fill_p_call,
    jitn_to_csr,
)
from brainevent._jit_normal.dt2t import (
    jitnmv_dt2t,
    jitnmv_dt2t_p,
    jitnmv_dt2t_p_call,
)
from brainevent._test_util import allclose, requires_gpu

pytestmark = pytest.mark.slow

platform = 'cpu'
CPU_DEVICE = jax.devices('cpu')[0]
JITN_dt2t_IMPLEMENTATIONS = tuple(jitnmv_dt2t_p.available_backends(platform))
JITN_dt2t_GPU_IMPLEMENTATIONS = tuple(jitnmv_dt2t_p.available_backends('gpu'))
X64_ENABLED = bool(jax.config.read('jax_enable_x64'))

requires_dt2t_backend = pytest.mark.skipif(
    not JITN_dt2t_IMPLEMENTATIONS,
    reason=f'No jitnmv_dt2t implementation on platform={platform}',
)


def _csr_yw_reference(csr, y, transpose):
    row_ids = jnp.repeat(
        jnp.arange(csr.shape[0], dtype=csr.indptr.dtype),
        jnp.diff(csr.indptr),
        total_repeat_length=csr.data.shape[0],
    )
    return csr.data * (y[csr.indices] if transpose else y[row_ids])


@pytest.mark.skipif(
    not JITN_dt2t_IMPLEMENTATIONS,
    reason=f'No jitnmv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITN_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitnmv_dt2t_matches_csr_reference(implementation, shape, corder, transpose):
    with jax.default_device(CPU_DEVICE):
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)

        out = jitnmv_dt2t(
            1.5,
            0.2,
            0.2,
            y,
            42,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
        csr = jitn_to_csr(
            1.5, 0.2, 0.2, 42,
            shape=shape, corder=corder, backend=implementation,
        )
        expected = _csr_yw_reference(csr, y, transpose)

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


@pytest.mark.skipif(not X64_ENABLED, reason='JAX x64 is disabled.')
@pytest.mark.skipif(
    not JITN_dt2t_IMPLEMENTATIONS,
    reason=f'No jitnmv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITN_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_dt2t_float64_matches_csr_reference(implementation, corder):
    with jax.default_device(CPU_DEVICE):
        shape = (13, 17)
        y = jnp.linspace(-1.0, 2.0, shape[0], dtype=jnp.float64)

        out = jitnmv_dt2t(
            jnp.asarray(1.5, dtype=jnp.float64),
            jnp.asarray(0.2, dtype=jnp.float64),
            0.2,
            y,
            42,
            shape=shape,
            corder=corder,
            backend=implementation,
        )
        csr = jitn_to_csr(
            jnp.asarray(1.5, dtype=jnp.float64),
            jnp.asarray(0.2, dtype=jnp.float64),
            0.2,
            42,
            shape=shape,
            corder=corder,
            backend=implementation,
        )
        expected = _csr_yw_reference(csr, y, False)

    assert out.dtype == jnp.float64
    assert allclose(out, expected)


def test_jitnmv_dt2t_prob_zero_empty():
    with jax.default_device(CPU_DEVICE):
        out = jitnmv_dt2t(
            1.5,
            0.2,
            0.0,
            jnp.ones(20, dtype=jnp.float32),
            42,
            shape=(20, 30),
            corder=True,
        )

    assert np.asarray(out).shape == (0,)


@pytest.mark.skipif(
    not JITN_dt2t_IMPLEMENTATIONS,
    reason=f'No jitnmv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITN_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jitnmv_dt2t_corder_false_is_repeatable(implementation, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(0.2, 1.7, y_size, dtype=jnp.float32)

        out1 = jitnmv_dt2t(
            1.5,
            0.2,
            0.2,
            y,
            42,
            shape=shape,
            transpose=transpose,
            corder=False,
            backend=implementation,
        )
        out2 = jitnmv_dt2t(
            1.5,
            0.2,
            0.2,
            y,
            42,
            shape=shape,
            transpose=transpose,
            corder=False,
            backend=implementation,
        )

    assert np.array_equal(np.asarray(out1), np.asarray(out2))


@requires_dt2t_backend
@pytest.mark.parametrize('implementation', JITN_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jitnmv_dt2t_fill_generates_y_times_weight_directly(implementation, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(0.2, 1.7, y_size, dtype=jnp.float32)
        w0 = jnp.asarray([1.5], dtype=jnp.float32)
        w1 = jnp.asarray([0.2], dtype=jnp.float32)
        clen = _initialize_conn_length(0.2)
        seed = _initialize_seed(42)

        chunk_counts = jitn_csr_count_p_call(
            w0, w1, clen, seed,
            shape=shape, corder=True, backend=implementation,
        )[0]
        row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32)
        indptr = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)])
        cc = chunk_counts.astype(jnp.int32)
        chunk_offsets = indptr[:-1, None] + jnp.cumsum(cc, axis=1, dtype=jnp.int32) - cc
        nnz = int(indptr[-1])

        indices, weights = jitn_csr_fill_p_call(
            w0, w1, clen, seed, chunk_offsets, nnz,
            shape=shape, corder=True, backend=implementation,
        )
        out = jitnmv_dt2t_p_call(
            w0,
            w1,
            clen,
            y,
            seed,
            chunk_offsets,
            nnz,
            shape=shape,
            transpose=transpose,
            backend=implementation,
        )[0]
        row_ids = jnp.repeat(
            jnp.arange(shape[0], dtype=indptr.dtype),
            jnp.diff(indptr),
            total_repeat_length=nnz,
        )
        expected = weights * (y[indices] if transpose else y[row_ids])

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


@requires_dt2t_backend
def test_jitnmv_dt2t_units_are_weight_times_y():
    with jax.default_device(CPU_DEVICE):
        out = jitnmv_dt2t(
            1.5 * u.siemens,
            0.2 * u.siemens,
            0.2,
            jnp.ones(20, dtype=jnp.float32) * u.mV,
            42,
            shape=(20, 30),
            corder=True,
        )

    assert u.get_unit(out) == u.mA


def test_jitnmv_dt2t_exports_from_package():
    assert brainevent.jitnmv_dt2t is jitnmv_dt2t


def test_jitn_matrix_dt2t_signatures_align_contracts():
    base_sig = inspect.signature(brainevent.DataRepresentation.dt2t)
    base_sig_t = inspect.signature(brainevent.DataRepresentation.dt2t_transposed)
    for cls in (brainevent.JITCNormalR, brainevent.JITCNormalC):
        sig = inspect.signature(cls.dt2t)
        assert list(sig.parameters) == list(base_sig.parameters)
        assert sig.parameters['y_dim_arr'].annotation == base_sig.parameters['y_dim_arr'].annotation
        assert sig.parameters['w_dim_arr'].annotation == base_sig.parameters['w_dim_arr'].annotation
        assert sig.parameters['w_dim_arr'].default is inspect._empty
        assert sig.return_annotation == base_sig.return_annotation

        sig_t = inspect.signature(cls.dt2t_transposed)
        assert list(sig_t.parameters) == list(base_sig_t.parameters)
        assert sig_t.parameters['y_dim_arr'].annotation == base_sig_t.parameters['y_dim_arr'].annotation
        assert sig_t.parameters['w_dim_arr'].annotation == base_sig_t.parameters['w_dim_arr'].annotation
        assert sig_t.parameters['w_dim_arr'].default is inspect._empty
        assert sig_t.return_annotation == base_sig_t.return_annotation


@pytest.mark.skipif(
    not JITN_dt2t_IMPLEMENTATIONS,
    reason=f'No jitnmv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITN_dt2t_IMPLEMENTATIONS)
def test_jitn_matrix_dt2t_requires_w_dim_arr(implementation):
    with jax.default_device(CPU_DEVICE):
        mat = brainevent.JITCNormalR((1.5, 0.2, 0.2, 42), shape=(20, 30), backend=implementation)
        y_pre = jnp.linspace(-1.0, 2.0, 20, dtype=jnp.float32)
        y_post = jnp.linspace(-1.0, 2.0, 30, dtype=jnp.float32)
        with pytest.raises(TypeError):
            mat.dt2t(y_pre)
        with pytest.raises(TypeError):
            mat.dt2t_transposed(y_post)


@pytest.mark.parametrize('implementation', JITN_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jitn_matrix_dt2t_uses_init_parameters(implementation, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)
        mat = brainevent.JITCNormalR(
            (1.5, 0.2, 0.2, 42),
            shape=shape,
            corder=True,
            backend=implementation,
        )

        w_dim_arr = jnp.empty(0, dtype=jnp.float32)
        out = mat.dt2t_transposed(y, w_dim_arr) if transpose else mat.dt2t(y, w_dim_arr)
        expected = jitnmv_dt2t(
            1.5,
            0.2,
            0.2,
            y,
            42,
            shape=shape,
            transpose=transpose,
            corder=True,
            backend=implementation,
        )

    assert allclose(out, expected)


@pytest.mark.skipif(
    not JITN_dt2t_IMPLEMENTATIONS,
    reason=f'No jitnmv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITN_dt2t_IMPLEMENTATIONS)
def test_jitn_matrix_dt2t_uses_instance_backend_and_corder(implementation):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y = jnp.linspace(-1.0, 2.0, shape[0], dtype=jnp.float32)
        mat = brainevent.JITCNormalR(
            (1.5, 0.2, 0.2, 42),
            shape=shape,
            corder=False,
            backend=implementation,
        )

        out = mat.dt2t(y, jnp.empty(0, dtype=jnp.float32))
        expected = jitnmv_dt2t(
            1.5,
            0.2,
            0.2,
            y,
            42,
            shape=shape,
            corder=False,
            backend=implementation,
        )

    assert allclose(out, expected)


@requires_gpu
@pytest.mark.skipif(
    'cuda_raw' not in JITN_dt2t_GPU_IMPLEMENTATIONS,
    reason='No jitnmv_dt2t cuda_raw implementation registered on GPU.',
)
@pytest.mark.parametrize('shape', [(20, 30), (64, 33)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitnmv_dt2t_cuda_matches_cuda_csr_reference(shape, corder, transpose):
    with jax.default_device(jax.devices('gpu')[0]):
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)

        out = jitnmv_dt2t(
            1.5,
            0.2,
            0.2,
            y,
            42,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend='cuda_raw',
        )
        csr = jitn_to_csr(
            1.5,
            0.2,
            0.2,
            42,
            shape=shape,
            corder=corder,
            backend='cuda_raw',
        )
        expected = _csr_yw_reference(csr, y, transpose)

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


# ---- Public interface: back to the v0.1.2 parameter list ----

def test_dt2t_signature_matches_0_1_2():
    import inspect
    params = tuple(inspect.signature(jitnmv_dt2t).parameters)
    assert params == tuple(p.strip() for p in 'w_loc, w_scale'.split(',')) + (
        'prob', 'y', 'seed', 'shape', 'transpose', 'corder', 'backend',
    )
