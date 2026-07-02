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
from typing import Union

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._data import _initialize_conn_length
from brainevent._jit_scalar.csr import (
    jits_csr_count_p_call,
    jits_csr_fill_p_call,
    jits_to_csr,
)
from brainevent._jit_scalar.yw2w import (
    jits_yw2w,
    jits_yw2w_fill_p,
    jits_yw2w_fill_p_call,
)
from brainevent._test_util import allclose, requires_gpu

pytestmark = pytest.mark.slow

platform = 'cpu'
CPU_DEVICE = jax.devices('cpu')[0]
JITS_YW2W_IMPLEMENTATIONS = tuple(jits_yw2w_fill_p.available_backends(platform))
JITS_YW2W_GPU_IMPLEMENTATIONS = tuple(jits_yw2w_fill_p.available_backends('gpu'))


def _csr_yw_reference(csr, y, transpose):
    row_ids = jnp.repeat(
        jnp.arange(csr.shape[0], dtype=csr.indptr.dtype),
        jnp.diff(csr.indptr),
        total_repeat_length=csr.data.shape[0],
    )
    return csr.data * (y[csr.indices] if transpose else y[row_ids])


@pytest.mark.skipif(
    not JITS_YW2W_IMPLEMENTATIONS,
    reason=f'No jits_yw2w implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITS_YW2W_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jits_yw2w_matches_csr_reference(implementation, shape, corder, transpose):
    with jax.default_device(CPU_DEVICE):
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)

        out = jits_yw2w(
            1.5,
            0.2,
            y,
            123,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
        csr = jits_to_csr(1.5, 0.2, 123, shape=shape, corder=corder, backend=implementation)
        expected = _csr_yw_reference(csr, y, transpose)

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


def test_jits_yw2w_prob_zero_empty():
    with jax.default_device(CPU_DEVICE):
        out = jits_yw2w(
            1.5,
            0.0,
            jnp.ones(20, dtype=jnp.float32),
            123,
            shape=(20, 30),
            corder=True,
        )

    assert np.asarray(out).shape == (0,)


@pytest.mark.skipif(
    not JITS_YW2W_IMPLEMENTATIONS,
    reason=f'No jits_yw2w implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITS_YW2W_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jits_yw2w_corder_false_is_repeatable(implementation, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(0.2, 1.7, y_size, dtype=jnp.float32)

        out1 = jits_yw2w(
            1.5,
            0.2,
            y,
            123,
            shape=shape,
            transpose=transpose,
            corder=False,
            backend=implementation,
        )
        out2 = jits_yw2w(
            1.5,
            0.2,
            y,
            123,
            shape=shape,
            transpose=transpose,
            corder=False,
            backend=implementation,
        )

    assert np.array_equal(np.asarray(out1), np.asarray(out2))


@pytest.mark.skipif(
    not JITS_YW2W_IMPLEMENTATIONS,
    reason=f'No jits_yw2w implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITS_YW2W_IMPLEMENTATIONS)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jits_yw2w_fill_generates_y_times_weight_directly(implementation, corder, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(0.2, 1.7, y_size, dtype=jnp.float32)
        w = jnp.asarray([1.5], dtype=jnp.float32)
        clen = _initialize_conn_length(0.2)
        seed = jnp.asarray([123], dtype=jnp.int32)

        row_counts = jits_csr_count_p_call(
            w, w, clen, seed, shape=shape, corder=corder, backend=implementation,
        )[0]
        indptr = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
        )
        nnz = int(indptr[-1])

        indices, weights = jits_csr_fill_p_call(
            w, w, clen, seed, indptr, nnz, shape=shape, corder=corder, backend=implementation,
        )
        out = jits_yw2w_fill_p_call(
            w,
            w,
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


def test_jits_yw2w_units_are_weight_times_y():
    with jax.default_device(CPU_DEVICE):
        out = jits_yw2w(
            1.5 * u.siemens,
            0.2,
            jnp.ones(20, dtype=jnp.float32) * u.mV,
            123,
            shape=(20, 30),
            corder=True,
        )

    assert u.get_unit(out) == u.mA


def test_jits_yw2w_exports_from_package():
    assert brainevent.jits_yw2w is jits_yw2w
    assert brainevent.jits_yw2w_fill_p is jits_yw2w_fill_p


def test_jits_matrix_yw_to_w_signatures_align_contracts():
    expected_input = Union[jax.Array, np.ndarray, u.Quantity]
    expected_output = Union[jax.Array, u.Quantity]
    for cls in (brainevent.JITCScalarR, brainevent.JITCScalarC):
        sig = inspect.signature(cls.yw_to_w)
        assert list(sig.parameters) == ['self', 'y_dim_arr', 'w_dim_arr']
        assert sig.parameters['y_dim_arr'].annotation == expected_input
        assert sig.parameters['w_dim_arr'].annotation == expected_input
        assert sig.parameters['w_dim_arr'].default is inspect._empty
        assert sig.return_annotation == expected_output

        sig_t = inspect.signature(cls.yw_to_w_transposed)
        assert list(sig_t.parameters) == ['self', 'y_dim_arr', 'w_dim_arr']
        assert sig_t.parameters['w_dim_arr'].default is inspect._empty


@pytest.mark.skipif(
    not JITS_YW2W_IMPLEMENTATIONS,
    reason=f'No jits_yw2w implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITS_YW2W_IMPLEMENTATIONS)
def test_jits_matrix_yw_to_w_requires_w_dim_arr(implementation):
    with jax.default_device(CPU_DEVICE):
        mat = brainevent.JITCScalarR((1.5, 0.2, 123), shape=(20, 30), backend=implementation)
        y_pre = jnp.linspace(-1.0, 2.0, 20, dtype=jnp.float32)
        y_post = jnp.linspace(-1.0, 2.0, 30, dtype=jnp.float32)
        with pytest.raises(TypeError):
            mat.yw_to_w(y_pre)
        with pytest.raises(TypeError):
            mat.yw_to_w_transposed(y_post)


@pytest.mark.parametrize('implementation', JITS_YW2W_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jits_matrix_yw_to_w_uses_init_parameters(implementation, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)
        mat = brainevent.JITCScalarR(
            (1.5, 0.2, 123),
            shape=shape,
            corder=True,
            backend=implementation,
        )

        w_dim_arr = jnp.empty(0, dtype=jnp.float32)
        out = mat.yw_to_w_transposed(y, w_dim_arr) if transpose else mat.yw_to_w(y, w_dim_arr)
        expected = jits_yw2w(
            1.5,
            0.2,
            y,
            123,
            shape=shape,
            transpose=transpose,
            corder=True,
            backend=implementation,
        )

    assert allclose(out, expected)


@pytest.mark.skipif(
    not JITS_YW2W_IMPLEMENTATIONS,
    reason=f'No jits_yw2w implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITS_YW2W_IMPLEMENTATIONS)
def test_jits_matrix_yw_to_w_uses_instance_backend_and_corder(implementation):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y = jnp.linspace(-1.0, 2.0, shape[0], dtype=jnp.float32)
        mat = brainevent.JITCScalarR(
            (1.5, 0.2, 123),
            shape=shape,
            corder=False,
            backend=implementation,
        )

        out = mat.yw_to_w(y, jnp.empty(0, dtype=jnp.float32))
        expected = jits_yw2w(
            1.5,
            0.2,
            y,
            123,
            shape=shape,
            corder=False,
            backend=implementation,
        )

    assert allclose(out, expected)


@requires_gpu
@pytest.mark.skipif(
    'cuda_raw' not in JITS_YW2W_GPU_IMPLEMENTATIONS,
    reason='No jits_yw2w cuda_raw implementation registered on GPU.',
)
@pytest.mark.parametrize('shape', [(20, 30), (64, 33)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jits_yw2w_cuda_matches_cuda_csr_reference(shape, corder, transpose):
    with jax.default_device(jax.devices('gpu')[0]):
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=jnp.float32)

        out = jits_yw2w(
            1.5,
            0.2,
            y,
            123,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend='cuda_raw',
        )
        csr = jits_to_csr(
            1.5,
            0.2,
            123,
            shape=shape,
            corder=corder,
            backend='cuda_raw',
        )
        expected = _csr_yw_reference(csr, y, transpose)

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))
