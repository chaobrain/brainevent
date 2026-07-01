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
import pytest

import brainevent
from brainevent._data import _initialize_conn_length
from brainevent._jit_uniform.csr import (
    jitu_csr_count_p_call,
    jitu_csr_fill_p_call,
    jitu_to_csr,
)
from brainevent._jit_uniform.yw2w import (
    jitu_yw2w,
    jitu_yw2w_fill_p,
    jitu_yw2w_fill_p_call,
)
from brainevent._test_util import allclose, requires_gpu

pytestmark = pytest.mark.slow

platform = 'cpu'
CPU_DEVICE = jax.devices('cpu')[0]
JITU_YW2W_IMPLEMENTATIONS = tuple(jitu_yw2w_fill_p.available_backends(platform))
GPU_DEVICE = jax.devices('gpu')[0] if jax.default_backend() == 'gpu' else None
JITU_YW2W_GPU_IMPLEMENTATIONS = tuple(jitu_yw2w_fill_p.available_backends('gpu'))


@pytest.mark.skipif(
    not JITU_YW2W_IMPLEMENTATIONS,
    reason=f'No jitu_yw2w implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_YW2W_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitu_yw2w_matches_csr_reference(implementation, shape, corder, transpose):
    with jax.default_device(CPU_DEVICE):
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)

        out = jitu_yw2w(
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
            csr.yw_to_w_transposed(y, csr.data)
            if transpose
            else csr.yw_to_w(y, csr.data)
        )

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


def test_jitu_yw2w_prob_zero_empty():
    with jax.default_device(CPU_DEVICE):
        out = jitu_yw2w(
            0.1,
            0.5,
            0.0,
            jnp.ones(20, dtype=jnp.float32),
            42,
            shape=(20, 30),
            corder=True,
        )

    assert np.asarray(out).shape == (0,)


def test_jitu_yw2w_exports_from_package():
    assert brainevent.jitu_yw2w is jitu_yw2w
    assert brainevent.jitu_yw2w_fill_p is jitu_yw2w_fill_p


@pytest.mark.skipif(
    not JITU_YW2W_IMPLEMENTATIONS,
    reason=f'No jitu_yw2w implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_YW2W_IMPLEMENTATIONS)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitu_yw2w_fill_generates_y_times_weight_directly(implementation, corder, transpose):
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
        out = jitu_yw2w_fill_p_call(
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


@pytest.mark.skipif(
    not JITU_YW2W_IMPLEMENTATIONS,
    reason=f'No jitu_yw2w implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_YW2W_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jitu_matrix_yw_to_w_uses_init_parameters(implementation, transpose):
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

        out = mat.yw_to_w_transposed(y) if transpose else mat.yw_to_w(y)
        expected = jitu_yw2w(
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
    not JITU_YW2W_IMPLEMENTATIONS,
    reason=f'No jitu_yw2w implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_YW2W_IMPLEMENTATIONS)
def test_jitu_matrix_yw_to_w_accepts_backend_and_corder_overrides(implementation):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y = jnp.linspace(-1.0, 2.0, shape[0], dtype=jnp.float32)
        mat = brainevent.JITCUniformR(
            (0.1, 0.5, 0.2, 42),
            shape=shape,
            corder=True,
            backend=None,
        )

        out = mat.yw_to_w(y, backend=implementation, corder=False)
        expected = jitu_yw2w(
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
    'cuda_raw' not in JITU_YW2W_GPU_IMPLEMENTATIONS,
    reason='No jitu_yw2w cuda_raw implementation registered on GPU.',
)
@pytest.mark.parametrize('shape', [(20, 30), (64, 33)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitu_yw2w_cuda_matches_cuda_csr_reference(shape, corder, transpose):
    with jax.default_device(jax.devices('gpu')[0]):
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)

        out = jitu_yw2w(
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
