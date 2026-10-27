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
JITU_dt2t_GPU_IMPLEMENTATIONS = tuple(jitumv_dt2t_p.available_backends('gpu'))
X64_ENABLED = bool(jax.config.read('jax_enable_x64'))

requires_dt2t_backend = pytest.mark.skipif(
    not JITU_dt2t_IMPLEMENTATIONS,
    reason=f'No jitumv_dt2t implementation on platform={platform}',
)


def _csr_yw_reference(csr, y, transpose):
    row_ids = jnp.repeat(
        jnp.arange(csr.shape[0], dtype=csr.indptr.dtype),
        jnp.diff(csr.indptr),
        total_repeat_length=csr.data.shape[0],
    )
    return csr.data * (y[csr.indices] if transpose else y[row_ids])


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
        csr = jitu_to_csr(
            0.1, 0.5, 0.2, 42,
            shape=shape, corder=corder, backend=implementation,
        )
        expected = _csr_yw_reference(csr, y, transpose)

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


@pytest.mark.skipif(not X64_ENABLED, reason='JAX x64 is disabled.')
@pytest.mark.skipif(
    not JITU_dt2t_IMPLEMENTATIONS,
    reason=f'No jitumv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumv_dt2t_float64_matches_csr_reference(implementation, corder):
    with jax.default_device(CPU_DEVICE):
        shape = (13, 17)
        y = jnp.linspace(-1.0, 2.0, shape[0], dtype=jnp.float64)

        out = jitumv_dt2t(
            jnp.asarray(0.1, dtype=jnp.float64),
            jnp.asarray(0.5, dtype=jnp.float64),
            0.2,
            y,
            42,
            shape=shape,
            corder=corder,
            backend=implementation,
        )
        csr = jitu_to_csr(
            jnp.asarray(0.1, dtype=jnp.float64),
            jnp.asarray(0.5, dtype=jnp.float64),
            0.2,
            42,
            shape=shape,
            corder=corder,
            backend=implementation,
        )
        expected = _csr_yw_reference(csr, y, False)

    assert out.dtype == jnp.float64
    assert allclose(out, expected)


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


@pytest.mark.skipif(
    not JITU_dt2t_IMPLEMENTATIONS,
    reason=f'No jitumv_dt2t implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jitumv_dt2t_corder_false_is_repeatable(implementation, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(0.2, 1.7, y_size, dtype=jnp.float32)

        out1 = jitumv_dt2t(
            0.1,
            0.5,
            0.2,
            y,
            42,
            shape=shape,
            transpose=transpose,
            corder=False,
            backend=implementation,
        )
        out2 = jitumv_dt2t(
            0.1,
            0.5,
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
@pytest.mark.parametrize('implementation', JITU_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [False, True])
def test_jitumv_dt2t_fill_generates_y_times_weight_directly(implementation, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(0.2, 1.7, y_size, dtype=jnp.float32)
        w0 = jnp.asarray([0.1], dtype=jnp.float32)
        w1 = jnp.asarray([0.5], dtype=jnp.float32)
        clen = _initialize_conn_length(0.2)
        seed = _initialize_seed(42)

        chunk_counts = jitu_csr_count_p_call(
            w0, w1, clen, seed,
            shape=shape, corder=True, backend=implementation,
        )[0]
        row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32)
        indptr = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)])
        cc = chunk_counts.astype(jnp.int32)
        chunk_offsets = indptr[:-1, None] + jnp.cumsum(cc, axis=1, dtype=jnp.int32) - cc
        nnz = int(indptr[-1])

        indices, weights = jitu_csr_fill_p_call(
            w0, w1, clen, seed, chunk_offsets, nnz,
            shape=shape, corder=True, backend=implementation,
        )
        out = jitumv_dt2t_p_call(
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
def test_jitumv_dt2t_units_are_weight_times_y():
    with jax.default_device(CPU_DEVICE):
        out = jitumv_dt2t(
            0.1 * u.siemens,
            0.5 * u.siemens,
            0.2,
            jnp.ones(20, dtype=jnp.float32) * u.mV,
            42,
            shape=(20, 30),
            corder=True,
        )

    assert u.get_unit(out) == u.mA


def test_jitumv_dt2t_exports_from_package():
    assert brainevent.jitumv_dt2t is jitumv_dt2t


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
        expected = _csr_yw_reference(csr, y, transpose)

    assert allclose(out, expected)
    jax.block_until_ready((out, expected))


# ---- Public interface: back to the v0.1.2 parameter list ----

def test_dt2t_signature_matches_0_1_2():
    import inspect
    params = tuple(inspect.signature(jitumv_dt2t).parameters)
    assert params == tuple(p.strip() for p in 'w_low, w_high'.split(',')) + (
        'prob', 'y', 'seed', 'shape', 'transpose', 'corder', 'backend',
    )


# ---- Thinned probabilities (``2 / p`` not an integer, so ``clen[1] < 2**24``) ----

from brainevent._jit_uniform.float import jitu

_THINNED_BACKENDS = [
    pytest.param('cpu', impl, id=f'cpu-{impl}') for impl in JITU_dt2t_IMPLEMENTATIONS
] + [
    pytest.param('gpu', impl, id=f'gpu-{impl}', marks=requires_gpu) for impl in JITU_dt2t_GPU_IMPLEMENTATIONS
]


def _dense_yw_reference(dense, y, transpose):
    """``W[r, c] * y[c]`` (transpose) or ``W[r, c] * y[r]`` over the non-zeros of the
    dense ``W``, row by row with columns ascending (the canonical CSR order)."""
    dense = np.asarray(dense)
    rows, cols = np.nonzero(dense)
    return dense[rows, cols] * np.asarray(y)[cols if transpose else rows]


@pytest.mark.parametrize('device, implementation', _THINNED_BACKENDS)
@pytest.mark.parametrize('shape', [(20, 30), (150, 133)])
@pytest.mark.parametrize('prob', [0.3, 0.7])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitumv_dt2t_thinned_matches_dense_reference(device, implementation, shape, prob, corder, transpose):
    # The dense generator does not share the count/fill kernels dt2t composes over.
    # (150, 133) walks chunks longer than 32, so lanes take several positions.
    with jax.default_device(jax.devices(device)[0]):
        y = jnp.linspace(-1.0, 2.0, shape[1] if transpose else shape[0], dtype=jnp.float32)
        out = jitumv_dt2t(
            0.1, 0.5, prob, y, 42,
            shape=shape, transpose=transpose, corder=corder, backend=implementation,
        )
        dense = jitu(0.1, 0.5, prob, 42, shape=shape, corder=corder)
    assert allclose(out, _dense_yw_reference(dense, y, transpose))


@pytest.mark.parametrize('device, implementation', _THINNED_BACKENDS)
@pytest.mark.parametrize('prob', [0.3, 0.7])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitumv_dt2t_fused_fill_drops_the_same_candidates(device, implementation, prob, transpose):
    # The fused kernel, the count kernel and the fill kernel each apply the keep
    # hash; all three must drop the same walk candidates, the dense matrix's zeros.
    shape = (150, 133)
    with jax.default_device(jax.devices(device)[0]):
        y = jnp.linspace(0.2, 1.7, shape[1] if transpose else shape[0], dtype=jnp.float32)
        w0 = jnp.asarray([0.1], dtype=jnp.float32)
        w1 = jnp.asarray([0.5], dtype=jnp.float32)
        clen = _initialize_conn_length(prob)
        assert int(clen[1]) < 2 ** 24
        seed = _initialize_seed(42)

        cc = jitu_csr_count_p_call(w0, w1, clen, seed, shape=shape, corder=True, backend=implementation)[0]
        cc = cc.astype(jnp.int32)
        indptr = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(cc.sum(axis=1), dtype=jnp.int32)])
        chunk_offsets = indptr[:-1, None] + jnp.cumsum(cc, axis=1, dtype=jnp.int32) - cc
        nnz = int(indptr[-1])
        indices, weights = jitu_csr_fill_p_call(
            w0, w1, clen, seed, chunk_offsets, nnz, shape=shape, corder=True, backend=implementation,
        )
        out = jitumv_dt2t_p_call(
            w0, w1, clen, y, seed, chunk_offsets, nnz,
            shape=shape, transpose=transpose, backend=implementation,
        )[0]
        rows = np.repeat(np.arange(shape[0]), np.diff(np.asarray(indptr)))
        expected = weights * (y[indices] if transpose else y[rows])
        mask = np.asarray(jitu(0.1, 0.5, prob, 42, shape=shape, corder=True)) != 0
    assert allclose(out, expected)

    filled = np.zeros(shape, dtype=bool)
    filled[rows, np.asarray(indices)] = True
    assert nnz == mask.sum()
    assert np.array_equal(filled, mask)


@pytest.mark.parametrize('prob', [0.25, 0.3])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitumv_dt2t_reverse_mode_matches_finite_differences(transpose, corder, prob):
    from jax.test_util import check_grads
    from brainevent._test_util import jax_x64_enabled

    shape = (12, 9)
    with jax_x64_enabled():
        y = jnp.linspace(0.5, 2., shape[1] if transpose else shape[0])
        f = lambda a, b, y_: jitumv_dt2t(a, b, prob, y_, 7, shape=shape, transpose=transpose, corder=corder)
        check_grads(f, (jnp.asarray(0.5), jnp.asarray(1.5), y), order=1, modes=['rev'])


def test_jitumv_dt2t_forward_values_are_unchanged_by_the_gradient_path():
    """The structure is drawn with the parameters stopped; the value must still be the drawn weights."""
    y = jnp.linspace(0.5, 2., 12)
    f = lambda a, b: jitumv_dt2t(a, b, 0.3, y, 7, shape=(12, 9))
    params = (jnp.asarray(0.5), jnp.asarray(1.5))
    out_on_grad_path, _ = jax.vjp(f, *params)
    np.testing.assert_array_equal(np.asarray(out_on_grad_path), np.asarray(f(*params)))


def test_jitc_uniform_matrix_dt2t_differentiates_both_parameters():
    from jax.test_util import check_grads
    from brainevent._test_util import jax_x64_enabled

    with jax_x64_enabled():
        y = jnp.linspace(0.5, 2., 12)
        loss = lambda a, b: brainevent.JITCUniformR(a, b, prob=0.3, seed=7, shape=(12, 9)).dt2t(y, None)
        check_grads(loss, (jnp.asarray(0.5), jnp.asarray(1.5)), order=1, modes=['rev'])
