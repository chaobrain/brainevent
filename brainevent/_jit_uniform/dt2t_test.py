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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._data import _initialize_conn_length
from brainevent._jit_uniform.csr import (
    jitu_csr_count_p_call,
    jitu_csr_fill_p_call,
    jitu_to_csr,
)
from brainevent._jit_uniform.dt2t import (
    jitumv_dt2t,
    jitumv_dt2t_p,
    jitumv_dt2t_p_call,
)
from brainevent._test_util import allclose, requires_gpu

pytestmark = pytest.mark.slow

platform = 'cpu'
CPU_DEVICE = jax.devices('cpu')[0]
JITU_dt2t_IMPLEMENTATIONS = tuple(jitumv_dt2t_p.available_backends(platform))
GPU_DEVICE = jax.devices('gpu')[0] if jax.default_backend() == 'gpu' else None
JITU_dt2t_GPU_IMPLEMENTATIONS = tuple(jitumv_dt2t_p.available_backends('gpu'))


@pytest.mark.skipif(
    not JITU_dt2t_IMPLEMENTATIONS,
    reason=f'No jitumv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitumv_dt2t_matches_csr_reference(implementation, shape, corder, transpose):
    with jax.default_device(CPU_DEVICE):
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)

        out = jitumv_dt2t(
            0.1,
            0.5,
            0.2,
            y,
            42,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
        csr = jitu_to_csr(0.1, 0.5, 0.2, 42, shape=shape, corder=corder, backend=implementation)
        expected = (
            csr.dt2t_transposed(y, csr.data)
            if transpose
            else csr.dt2t(y, csr.data)
        )

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


def test_jitumv_dt2t_prob_zero_empty():
    with jax.default_device(CPU_DEVICE):
        out = jitumv_dt2t(
            0.1,
            0.5,
            0.0,
            jnp.ones(20, dtype=jnp.float32),
            42,
            shape=(20, 30),
            corder=True,
        )

    assert np.asarray(out).shape == (0,)


def test_jitumv_dt2t_exports_from_package():
    assert brainevent.jitumv_dt2t is jitumv_dt2t


@pytest.mark.skipif(
    not JITU_dt2t_IMPLEMENTATIONS,
    reason=f'No jitumv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitumv_dt2t_fill_generates_y_times_weight_directly(implementation, corder, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(0.2, 1.7, y_size, dtype=jnp.float32)
        w0 = jnp.asarray([0.1], dtype=jnp.float32)
        w1 = jnp.asarray([0.5], dtype=jnp.float32)
        clen = _initialize_conn_length(0.2)
        seed = jnp.asarray([42], dtype=jnp.int32)

        row_counts = jitu_csr_count_p_call(
            w0, w1, clen, seed, shape=shape, corder=corder, backend=implementation,
        )[0]
        indptr = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
        )
        nnz = int(indptr[-1])

        indices, weights = jitu_csr_fill_p_call(
            w0, w1, clen, seed, indptr, nnz, shape=shape, corder=corder, backend=implementation,
        )
        out = jitumv_dt2t_p_call(
            w0,
            w1,
            clen,
            y,
            seed,
            indptr,
            nnz,
            shape=shape,
            transpose=transpose,
            corder=corder,
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


def test_jitu_matrix_dt2t_signatures_align_contracts():
    base_sig = inspect.signature(brainevent.DataRepresentation.dt2t)
    base_sig_t = inspect.signature(brainevent.DataRepresentation.dt2t_transposed)
    for cls in (brainevent.JITCUniformR, brainevent.JITCUniformC):
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
    not JITU_dt2t_IMPLEMENTATIONS,
    reason=f'No jitumv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_dt2t_IMPLEMENTATIONS)
def test_jitu_matrix_dt2t_requires_w_dim_arr(implementation):
    with jax.default_device(CPU_DEVICE):
        mat = brainevent.JITCUniformR((0.1, 0.5, 0.2, 42), shape=(20, 30), backend=implementation)
        y_pre = jnp.linspace(-1.0, 2.0, 20, dtype=jnp.float32)
        y_post = jnp.linspace(-1.0, 2.0, 30, dtype=jnp.float32)
        with pytest.raises(TypeError):
            mat.dt2t(y_pre)
        with pytest.raises(TypeError):
            mat.dt2t_transposed(y_post)


@pytest.mark.parametrize('implementation', JITU_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jitu_matrix_dt2t_uses_init_parameters(implementation, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)
        mat = brainevent.JITCUniformR(
            (0.1, 0.5, 0.2, 42),
            shape=shape,
            corder=True,
            backend=implementation,
        )

        w_dim_arr = jnp.empty(0, dtype=jnp.float32)
        out = mat.dt2t_transposed(y, w_dim_arr) if transpose else mat.dt2t(y, w_dim_arr)
        expected = jitumv_dt2t(
            0.1,
            0.5,
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
    not JITU_dt2t_IMPLEMENTATIONS,
    reason=f'No jitumv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_dt2t_IMPLEMENTATIONS)
def test_jitu_matrix_dt2t_uses_instance_backend_and_corder(implementation):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y = jnp.linspace(-1.0, 2.0, shape[0], dtype=jnp.float32)
        mat = brainevent.JITCUniformR(
            (0.1, 0.5, 0.2, 42),
            shape=shape,
            corder=False,
            backend=implementation,
        )

        out = mat.dt2t(y, jnp.empty(0, dtype=jnp.float32))
        expected = jitumv_dt2t(
            0.1,
            0.5,
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
    'cuda_raw' not in JITU_dt2t_GPU_IMPLEMENTATIONS,
    reason='No jitumv_dt2t cuda_raw implementation registered on GPU.',
)
@pytest.mark.parametrize('shape', [(20, 30), (64, 33)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitumv_dt2t_cuda_matches_cuda_csr_reference(shape, corder, transpose):
    with jax.default_device(jax.devices('gpu')[0]):
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)

        out = jitumv_dt2t(
            0.1,
            0.5,
            0.2,
            y,
            42,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend='cuda_raw',
        )
        csr = jitu_to_csr(
            0.1,
            0.5,
            0.2,
            42,
            shape=shape,
            corder=corder,
            backend='cuda_raw',
        )
        row_ids = jnp.repeat(
            jnp.arange(shape[0], dtype=csr.indptr.dtype),
            jnp.diff(csr.indptr),
            total_repeat_length=csr.data.shape[0],
        )
        expected = csr.data * (y[csr.indices] if transpose else y[row_ids])

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))
