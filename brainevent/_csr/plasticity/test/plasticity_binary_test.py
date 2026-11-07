# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import brainevent
import brainevent as be
import brainevent._csr.plasticity.plasticity_binary as plasticity_mod
from brainevent._csr.plasticity.plasticity_binary import (
    update_csr_on_binary_pre,
    update_csr_on_binary_post,
    update_csr_on_binary_pre_p,
    update_csr_on_binary_post_p,
)
from brainevent._csr._test_util import (
    cuda_kwargs, int64_structure, recording_ffi_call, requires_gpu_backend, shape_of,
)
from brainevent._test_util import jax_x64_enabled, with_jax_x64

PLATFORM = jax.default_backend()
PRE_BACKENDS = tuple(update_csr_on_binary_pre_p.available_backends(PLATFORM))
POST_BACKENDS = tuple(update_csr_on_binary_post_p.available_backends(PLATFORM))

shapes = [
    (20, 100),
    (100, 100),
    (100, 50),
]


class Test_csr_on_pre:
    @pytest.mark.parametrize('backend', PRE_BACKENDS)
    @pytest.mark.parametrize('shape', shapes)
    def test_csr_on_pre_v1(self, backend, shape):
        n_pre, n_post = shape
        mat = brainstate.random.random((n_pre, n_post))
        mask = (mat < 0.5) & (mat != 0.)
        mat = jnp.where(mask, mat, 0.)

        pre_spike = brainstate.random.random((n_pre,)) < 0.5
        post_trace = brainstate.random.random((n_post,))

        csr = brainevent.CSR.fromdense(mat)
        csr2 = csr.with_data(
            update_csr_on_binary_pre(csr.data, csr.indices, csr.indptr, pre_spike, post_trace,
                                     shape=csr.shape, backend=backend)
        )
        dense2 = jnp.where(mask, mat + jnp.outer(pre_spike.astype(float), post_trace), 0.)

        assert jnp.allclose(csr2.todense(), dense2)

        jax.block_until_ready((mat, pre_spike, post_trace))

    @pytest.mark.parametrize('backend', PRE_BACKENDS)
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
    @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
    def test_csr_on_pre_with_unit(self, backend, shape, mat_unit, trace_unit):
        def run():
            n_pre, n_post = shape
            mat = brainstate.random.random((n_pre, n_post))
            mask = (mat < 0.5) & (mat != 0.)
            mat = jnp.where(mask, mat, 0.) * mat_unit
            pre_spike = brainstate.random.random((n_pre,)) < 0.1
            post_trace = brainstate.random.random((n_post,)) * trace_unit

            csr = brainevent.CSR.fromdense(mat)
            csr = csr.with_data(
                update_csr_on_binary_pre(csr.data, csr.indices, csr.indptr, pre_spike, post_trace,
                                         shape=csr.shape, backend=backend)
            )

            dense = mat + u.math.outer(pre_spike.astype(float), post_trace)
            dense = u.math.where(mask, dense, 0. * mat_unit)

            assert u.math.allclose(csr.todense(), dense)

            jax.block_until_ready((mat, pre_spike, post_trace))

        if mat_unit.has_same_dim(trace_unit):
            run()
        else:
            with pytest.raises(u.UnitMismatchError):
                run()

    @pytest.mark.parametrize('backend', PRE_BACKENDS)
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('w_in', [None, 0.1])
    @pytest.mark.parametrize('w_max', [None, 0.5])
    def test_csr_on_pre_v2(self, backend, shape, w_in, w_max):
        n_pre, n_post = shape
        mat = brainstate.random.random((n_pre, n_post))
        mask = (mat < 0.5) & (mat != 0.)
        mat = jnp.where(mask, mat, 0.)
        pre_spike = brainstate.random.random((n_pre,)) < 0.1
        post_trace = brainstate.random.random((n_post,))

        csr = brainevent.CSR.fromdense(mat)
        csr = csr.with_data(
            update_csr_on_binary_pre(
                csr.data, csr.indices, csr.indptr, pre_spike, post_trace,
                w_min=w_in, w_max=w_max, shape=csr.shape, backend=backend
            )
        )

        mat = mat + jnp.outer(pre_spike.astype(float), post_trace)
        mat = u.math.clip(mat, a_min=w_in, a_max=w_max)

        mat = jnp.where(mask, mat, 0.)
        assert jnp.allclose(csr.todense(), mat, atol=1e-1, rtol=1e-1)

        jax.block_until_ready((mat, pre_spike, post_trace))


def _csr_to_csc_with_weight_indices(csr_data, csr_indices, csr_indptr, shape):
    """Convert CSR format to CSC format and return weight indices mapping.

    Returns:
        csc_indices: Row indices in CSC format (presynaptic neuron indices)
        csc_indptr: Column pointers in CSC format
        weight_indices: Mapping from CSC positions to original CSR data positions
    """
    # Handle Quantity data by extracting mantissa
    if hasattr(csr_data, 'mantissa'):
        csr_data_np = np.asarray(csr_data.mantissa)
    else:
        csr_data_np = np.asarray(csr_data)

    # Create scipy CSR matrix
    scipy_csr = sp.csr_matrix(
        (csr_data_np, np.asarray(csr_indices), np.asarray(csr_indptr)),
        shape=shape
    )
    # Convert to CSC
    scipy_csc = scipy_csr.tocsc()

    # The weight_indices maps CSC position -> CSR position
    # We can compute this by tracking where each element came from
    # CSC data comes from reordering CSR data based on column sorting

    # Create a CSR matrix with data = position indices
    position_indices = np.arange(len(csr_data_np), dtype=np.int32)
    scipy_csr_indices = sp.csr_matrix(
        (position_indices, np.asarray(csr_indices), np.asarray(csr_indptr)),
        shape=shape
    )
    scipy_csc_indices = scipy_csr_indices.tocsc()
    weight_indices = scipy_csc_indices.data

    return (
        jnp.asarray(scipy_csc.indices, dtype=jnp.int32),
        jnp.asarray(scipy_csc.indptr, dtype=jnp.int32),
        jnp.asarray(weight_indices, dtype=jnp.int32)
    )


class Test_on_post:
    @pytest.mark.parametrize('backend', POST_BACKENDS)
    @pytest.mark.parametrize('shape', shapes)
    def test_csr_on_post_v1(self, backend, shape):
        n_pre, n_post = shape
        mat = brainstate.random.random((n_pre, n_post))
        mask = (mat < 0.5) & (mat != 0.)
        mat = jnp.where(mask, mat, 0.)

        post_spike = brainstate.random.random((n_post,)) < 0.5
        pre_trace = brainstate.random.random((n_pre,))

        csr = brainevent.CSR.fromdense(mat)
        # Convert CSR to CSC format for csr2csc_on_post
        csc_indices, csc_indptr, weight_indices = _csr_to_csc_with_weight_indices(
            csr.data, csr.indices, csr.indptr, csr.shape
        )

        new_weights = update_csr_on_binary_post(
            csr.data, csc_indices, csc_indptr, weight_indices,
            pre_trace, post_spike, shape=csr.shape, backend=backend
        )
        csr2 = csr.with_data(new_weights)
        dense2 = jnp.where(mask, mat + jnp.outer(pre_trace, post_spike.astype(float)), 0.)

        assert jnp.allclose(csr2.todense(), dense2)

        jax.block_until_ready((mat, post_spike, pre_trace))

    @pytest.mark.parametrize('backend', POST_BACKENDS)
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
    @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
    def test_csr_on_post_with_unit(self, backend, shape, mat_unit, trace_unit):
        def run():
            n_pre, n_post = shape
            mat = brainstate.random.random((n_pre, n_post))
            mask = (mat < 0.5) & (mat != 0.)
            mat = jnp.where(mask, mat, 0.) * mat_unit
            post_spike = brainstate.random.random((n_post,)) < 0.1
            pre_trace = brainstate.random.random((n_pre,)) * trace_unit

            csr = brainevent.CSR.fromdense(mat)
            # Convert CSR to CSC format for csr2csc_on_post
            csc_indices, csc_indptr, weight_indices = _csr_to_csc_with_weight_indices(
                csr.data, csr.indices, csr.indptr, csr.shape
            )

            new_weights = update_csr_on_binary_post(
                csr.data, csc_indices, csc_indptr, weight_indices,
                pre_trace, post_spike, shape=csr.shape, backend=backend
            )
            csr = csr.with_data(new_weights)

            dense = mat + u.math.outer(pre_trace, post_spike.astype(float))
            dense = u.math.where(mask, dense, 0. * mat_unit)

            assert u.math.allclose(csr.todense(), dense)

            jax.block_until_ready((mat, post_spike, pre_trace))

        if mat_unit.has_same_dim(trace_unit):
            run()
        else:
            with pytest.raises(u.UnitMismatchError):
                run()

    @pytest.mark.parametrize('backend', POST_BACKENDS)
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('w_in', [None, 0.1])
    @pytest.mark.parametrize('w_max', [None, 0.5])
    def test_csr_on_post_v2(self, backend, shape, w_in, w_max):
        n_pre, n_post = shape
        mat = brainstate.random.random((n_pre, n_post))
        mask = (mat < 0.5) & (mat != 0.)
        mat = jnp.where(mask, mat, 0.)
        post_spike = brainstate.random.random((n_post,)) < 0.1
        pre_trace = brainstate.random.random((n_pre,))

        csr = brainevent.CSR.fromdense(mat)
        # Convert CSR to CSC format for csr2csc_on_post
        csc_indices, csc_indptr, weight_indices = _csr_to_csc_with_weight_indices(
            csr.data, csr.indices, csr.indptr, csr.shape
        )

        new_weights = update_csr_on_binary_post(
            csr.data, csc_indices, csc_indptr, weight_indices,
            pre_trace, post_spike, w_min=w_in, w_max=w_max, shape=csr.shape, backend=backend
        )
        csr = csr.with_data(new_weights)

        mat = mat + jnp.outer(pre_trace, post_spike.astype(float))
        mat = u.math.clip(mat, a_min=w_in, a_max=w_max)

        mat = jnp.where(mask, mat, 0.)
        assert jnp.allclose(csr.todense(), mat, rtol=1e-1, atol=1e-1)

        jax.block_until_ready((mat, post_spike, pre_trace))


def test_csr_homogeneous_weight_raises():
    weight = jnp.asarray([1.5], dtype=jnp.float32)  # homogeneous over >1 synapse
    indices = jnp.array([0, 1, 0, 2], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    pre_spike = jnp.array([True, False])
    post_trace = jnp.array([0.1, 0.2, 0.05], dtype=jnp.float32)
    with pytest.raises(ValueError, match="per-synapse"):
        update_csr_on_binary_pre(weight, indices, indptr, pre_spike, post_trace, shape=(2, 3))


# ---------------------------------------------------------------------------
# CSC-layout plasticity updates (``update_csc_on_binary_pre`` /
# ``update_csc_on_binary_post``), validated against a dense reference.
# ---------------------------------------------------------------------------


def _dense(rng, n_pre, n_post, p=0.5):
    mask = rng.random((n_pre, n_post)) < p
    vals = rng.random((n_pre, n_post)) + 0.5  # in [0.5, 1.5)
    return jnp.asarray(mask * vals, dtype=jnp.float32)


def _ref_pre(W, pre_spike, post_trace, w_min, w_max):
    W = np.asarray(W)
    mask = (W != 0.0)
    active = (np.asarray(pre_spike) != 0).astype(W.dtype)        # (n_pre,)
    delta = mask * active[:, None] * np.asarray(post_trace)[None, :]
    return np.clip(W + delta, w_min, w_max)


def _ref_post(W, pre_trace, post_spike, w_min, w_max):
    W = np.asarray(W)
    mask = (W != 0.0)
    active = (np.asarray(post_spike) != 0).astype(W.dtype)       # (n_post,)
    delta = mask * np.asarray(pre_trace)[:, None] * active[None, :]
    return np.clip(W + delta, w_min, w_max)


@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
def test_update_csc_on_binary_pre_matches_dense(spike_dtype):
    rng = np.random.default_rng(0)
    n_pre, n_post = 4, 6
    w_min, w_max = 0.0, 1.2
    W = _dense(rng, n_pre, n_post)
    csc = be.CSC.fromdense(W)
    pre_spike = jnp.asarray(rng.random(n_pre) > 0.5, dtype=spike_dtype)
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)
    new_w = be.update_csc_on_binary_pre(
        csc.data, csc.indices, csc.indptr, pre_spike, post_trace,
        w_min, w_max, shape=csc.shape,
    )
    got = csc.with_data(new_w).todense()
    ref = _ref_pre(W, pre_spike, post_trace, w_min, w_max)
    assert jnp.allclose(got, jnp.asarray(ref), atol=1e-5)


@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
def test_update_csc_on_binary_post_matches_dense(spike_dtype):
    rng = np.random.default_rng(1)
    n_pre, n_post = 5, 3
    w_min, w_max = 0.0, 1.2
    W = _dense(rng, n_pre, n_post)
    csc = be.CSC.fromdense(W)
    pre_trace = jnp.asarray(rng.random(n_pre), dtype=jnp.float32)
    post_spike = jnp.asarray(rng.random(n_post) > 0.5, dtype=spike_dtype)
    new_w = be.update_csc_on_binary_post(
        csc.data, csc.indices, csc.indptr, pre_trace, post_spike,
        w_min, w_max, shape=csc.shape,
    )
    got = csc.with_data(new_w).todense()
    ref = _ref_post(W, pre_trace, post_spike, w_min, w_max)
    assert jnp.allclose(got, jnp.asarray(ref), atol=1e-5)


def test_update_csc_on_binary_pre_no_clip():
    # Without bounds the rule is a pure additive update at stored positions.
    rng = np.random.default_rng(2)
    n_pre, n_post = 3, 4
    W = _dense(rng, n_pre, n_post)
    csc = be.CSC.fromdense(W)
    pre_spike = jnp.asarray([True, False, True])
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)
    new_w = be.update_csc_on_binary_pre(
        csc.data, csc.indices, csc.indptr, pre_spike, post_trace, shape=csc.shape,
    )
    got = csc.with_data(new_w).todense()
    ref = _ref_pre(W, pre_spike, post_trace, None, None)
    assert jnp.allclose(got, jnp.asarray(ref), atol=1e-5)


# ---------------------------------------------------------------------------
# int64 ``indptr`` policy on the CUDA path.
#
# ``indices`` and ``weight_indices`` stay int32 (the CUDA ABI is int32-only for
# coordinates) while ``indptr`` may widen to int64. The generator tests run
# without a real GPU by stubbing ``load_cuda_file``/``ffi_call``; the ``accepts``
# tests need one.
# ---------------------------------------------------------------------------


def test_plasticity_pre_cuda_rejects_int64_indices_before_loading_cuda():
    with pytest.raises(TypeError, match="indices with dtype int32"):
        plasticity_mod._csr_on_pre_cuda_kernel(
            shape_of(jnp.float32),
            shape_of(jnp.bool_),
            shape_of(jnp.int64),
            outs=[shape_of(jnp.float32)],
            indptr_info=shape_of(jnp.int64),
        )


def test_plasticity_post_cuda_rejects_int64_indices_before_loading_cuda():
    with pytest.raises(TypeError, match="indices with dtype int32"):
        plasticity_mod._csr2csc_on_post_cuda_kernel(
            shape_of(jnp.float32),
            shape_of(jnp.bool_),
            shape_of(jnp.int64),
            outs=[shape_of(jnp.float32)],
            indptr_info=shape_of(jnp.int64),
            weight_indices_info=shape_of(jnp.int32),
        )


def test_plasticity_post_cuda_rejects_int64_weight_indices_before_loading_cuda():
    kwargs = {
        'outs': [shape_of(jnp.float32)],
        'indptr_info': shape_of(jnp.int64),
        'weight_indices_info': shape_of(jnp.int64),
    }

    with pytest.raises(TypeError, match="weight_indices with dtype int32"):
        plasticity_mod._csr2csc_on_post_cuda_kernel(
            shape_of(jnp.float32),
            shape_of(jnp.bool_),
            shape_of(jnp.int32),
            **kwargs,
        )


def test_plasticity_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(plasticity_mod, "load_cuda_file", lambda path, name: load_calls.append((path, name)))
    monkeypatch.setattr(plasticity_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    with jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)

        pre_kernel = plasticity_mod._csr_on_pre_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (1,)),
            shape_of(jnp.int32, (2,)),
            outs=[shape_of(jnp.float32, (2,))],
            indptr_info=shape_of(jnp.int64, (2,)),
        )
        pre_kernel(
            jnp.array([1.0, 2.0]),
            indices,
            indptr,
            jnp.array([True]),
            jnp.array([0.5, 1.5]),
        )

        post_kernel = plasticity_mod._csr2csc_on_post_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.int32, (2,)),
            outs=[shape_of(jnp.float32, (2,))],
            indptr_info=shape_of(jnp.int64, (2,)),
            weight_indices_info=shape_of(jnp.int32, (2,)),
        )
        post_kernel(
            jnp.array([1.0, 2.0]),
            indices,
            indptr,
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.array([0.5]),
            jnp.array([1.0, -1.0]),
        )

    assert [name for _, name in load_calls] == [
        'csr_plasticity_binary_pre',
        'csr_plasticity_binary_post',
    ]
    assert [call[0] for call in ffi_calls] == [
        'csr_plasticity_binary_pre.update_csr_on_pre_f32_bool',
        'csr_plasticity_binary_post.update_csr_on_post_f32_float',
    ]


@requires_gpu_backend
@with_jax_x64
def test_plasticity_pre_cuda_accepts_int64_indptr():
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    pre_spike = jnp.array([True, False])
    post_trace = jnp.array([0.5, 1.5, 2.5], dtype=jnp.float32)

    got = update_csr_on_binary_pre(
        weights, indices, indptr64, pre_spike, post_trace, shape=(2, 3), backend='cuda_raw'
    )
    expected = jnp.array([1.5, 4.5, 3.0, 4.0], dtype=jnp.float32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu_backend
@with_jax_x64
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


# ---------------------------------------------------------------------------
# Event activity contract: a spike is active iff ``spike != 0`` (negative and
# fractional values included) on every backend, including CUDA.
# ---------------------------------------------------------------------------

from brainevent._csr._test_util import (  # noqa: E402
    EVENT_DTYPE_NAMES, event_dtype_context, event_mask, event_values, random_csr_numpy,
)

_PL_PRE, _PL_POST = 12, 9


@pytest.mark.slow
@pytest.mark.parametrize('backend', PRE_BACKENDS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
def test_csr_on_pre_spike_is_active_iff_nonzero(backend, dtype_name):
    with event_dtype_context(dtype_name):
        rng = np.random.default_rng(0)
        data, indices, indptr, _ = random_csr_numpy(rng, _PL_PRE, _PL_POST)
        pre_spike = event_values(dtype_name, _PL_PRE)
        post_trace = rng.standard_normal(_PL_POST).astype(np.float32)
        got = update_csr_on_binary_pre(
            jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr), pre_spike,
            jnp.asarray(post_trace), shape=(_PL_PRE, _PL_POST), backend=backend,
        )
        rows = np.repeat(np.arange(_PL_PRE), np.diff(indptr))
        expected = data + event_mask(pre_spike)[rows] * post_trace[indices]
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', POST_BACKENDS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
def test_csr_on_post_spike_is_active_iff_nonzero(backend, dtype_name):
    with event_dtype_context(dtype_name):
        rng = np.random.default_rng(1)
        data, indices, indptr, _ = random_csr_numpy(rng, _PL_PRE, _PL_POST)
        csc_indices, csc_indptr, weight_indices = _csr_to_csc_with_weight_indices(
            data, indices, indptr, (_PL_PRE, _PL_POST)
        )
        post_spike = event_values(dtype_name, _PL_POST, seed=1)
        pre_trace = rng.standard_normal(_PL_PRE).astype(np.float32)
        got = update_csr_on_binary_post(
            jnp.asarray(data), csc_indices, csc_indptr, weight_indices,
            jnp.asarray(pre_trace), post_spike, shape=(_PL_PRE, _PL_POST), backend=backend,
        )
        rows = np.repeat(np.arange(_PL_PRE), np.diff(indptr))
        expected = data + pre_trace[rows] * event_mask(post_spike)[indices]
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Autodiff (#202): the updates are jointly linear in (weight, trace); clipping
# happens in the wrapper, so w_min / w_max gradients come from JAX.
# ---------------------------------------------------------------------------

from brainevent._test_util import check_plasticity_grads  # noqa: E402

_AD_PRE, _AD_POST = 9, 7
_W_LO, _W_HI = jnp.float32(0.2), jnp.float32(1.3)


def _ad_spikes(n, seed):
    """Spikes containing 2, -1 (both active) and 0."""
    base = np.array([2., -1., 0., 0.5, 0., 1.], dtype=np.float32)
    return jnp.asarray(np.resize(np.roll(base, seed), n))


def _ad_problem(seed=0):
    rng = np.random.default_rng(seed)
    data, indices, indptr, _ = random_csr_numpy(rng, _AD_PRE, _AD_POST)
    data = jnp.asarray(rng.uniform(0.3, 1.2, data.shape), dtype=jnp.float32)
    rows = np.repeat(np.arange(_AD_PRE), np.diff(indptr))
    return data, jnp.asarray(indices), jnp.asarray(indptr), rows, rng


def _ref_edges(weight, active_e, trace_e, w_min, w_max):
    return jnp.clip(weight + jnp.where(active_e, trace_e, 0.), w_min, w_max)


@pytest.mark.parametrize('backend', PRE_BACKENDS)
@pytest.mark.parametrize('spike_kind', ['float', 'bool'])
def test_csr_on_pre_grads_match_reference(backend, spike_kind):
    data, indices, indptr, rows, rng = _ad_problem(0)
    pre_spike = _ad_spikes(_AD_PRE, 0)
    if spike_kind == 'bool':
        pre_spike = pre_spike != 0
    post_trace = jnp.asarray(rng.uniform(-0.6, 0.6, _AD_POST), dtype=jnp.float32)
    active_e = np.asarray(pre_spike != 0)[rows]

    def op(w, t, lo, hi):
        return update_csr_on_binary_pre(w, indices, indptr, pre_spike, t, lo, hi,
                                        shape=(_AD_PRE, _AD_POST), backend=backend)

    def ref(w, t, lo, hi):
        return _ref_edges(w, active_e, t[indices], lo, hi)

    check_plasticity_grads(op, ref, data, post_trace, _W_LO, _W_HI)


@pytest.mark.parametrize('backend', POST_BACKENDS)
def test_csr_on_post_grads_match_reference(backend):
    data, indices, indptr, rows, rng = _ad_problem(1)
    csc_indices, csc_indptr, perm = _csr_to_csc_with_weight_indices(
        np.asarray(data), np.asarray(indices), np.asarray(indptr), (_AD_PRE, _AD_POST)
    )
    post_spike = _ad_spikes(_AD_POST, 1)
    pre_trace = jnp.asarray(rng.uniform(-0.6, 0.6, _AD_PRE), dtype=jnp.float32)
    active_e = np.asarray(post_spike != 0)[np.asarray(indices)]

    def op(w, t, lo, hi):
        return update_csr_on_binary_post(w, csc_indices, csc_indptr, perm, t, post_spike, lo, hi,
                                         shape=(_AD_PRE, _AD_POST), backend=backend)

    def ref(w, t, lo, hi):
        return _ref_edges(w, active_e, t[rows], lo, hi)

    check_plasticity_grads(op, ref, data, pre_trace, _W_LO, _W_HI)


def test_csc_updates_grads_match_reference():
    W = _dense(np.random.default_rng(3), _AD_PRE, _AD_POST)
    csc = brainevent.CSC.fromdense(W)
    rows = np.asarray(csc.indices)
    cols = np.repeat(np.arange(_AD_POST), np.diff(np.asarray(csc.indptr)))
    pre_spike = _ad_spikes(_AD_PRE, 2)
    post_spike = _ad_spikes(_AD_POST, 3)
    rng = np.random.default_rng(4)
    pre_trace = jnp.asarray(rng.uniform(-0.6, 0.6, _AD_PRE), dtype=jnp.float32)
    post_trace = jnp.asarray(rng.uniform(-0.6, 0.6, _AD_POST), dtype=jnp.float32)
    hi = jnp.float32(1.6)

    check_plasticity_grads(
        lambda w, t, lo, h: brainevent.update_csc_on_binary_pre(
            w, csc.indices, csc.indptr, pre_spike, t, lo, h, shape=csc.shape),
        lambda w, t, lo, h: _ref_edges(w, np.asarray(pre_spike != 0)[rows], t[cols], lo, h),
        csc.data, post_trace, _W_LO, hi,
    )
    check_plasticity_grads(
        lambda w, t, lo, h: brainevent.update_csc_on_binary_post(
            w, csc.indices, csc.indptr, t, post_spike, lo, h, shape=csc.shape),
        lambda w, t, lo, h: _ref_edges(w, np.asarray(post_spike != 0)[cols], t[rows], lo, h),
        csc.data, pre_trace, _W_LO, hi,
    )


def test_csr_class_methods_differentiable():
    data, indices, indptr, rows, rng = _ad_problem(5)
    csr = brainevent.CSR((data, indices, indptr), shape=(_AD_PRE, _AD_POST))
    pre_spike = _ad_spikes(_AD_PRE, 4)
    post_spike = _ad_spikes(_AD_POST, 5)
    pre_trace = jnp.asarray(rng.uniform(-0.6, 0.6, _AD_PRE), dtype=jnp.float32)
    post_trace = jnp.asarray(rng.uniform(-0.6, 0.6, _AD_POST), dtype=jnp.float32)

    def step(w, tr_pre, tr_post):
        m = csr.with_data(w).update_on_post(tr_pre, post_spike)
        return m.update_on_pre(pre_spike, tr_post).data

    def ref(w, tr_pre, tr_post):
        w = w + jnp.where(np.asarray(post_spike != 0)[np.asarray(indices)], tr_pre[rows], 0.)
        return w + jnp.where(np.asarray(pre_spike != 0)[rows], tr_post[indices], 0.)

    ct = jnp.asarray(rng.standard_normal(data.shape), dtype=jnp.float32)
    g = jax.grad(lambda *a: jnp.vdot(ct, step(*a)), argnums=(0, 1, 2))(data, pre_trace, post_trace)
    g_ref = jax.grad(lambda *a: jnp.vdot(ct, ref(*a)), argnums=(0, 1, 2))(data, pre_trace, post_trace)
    for a, b in zip(g, g_ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


def test_issue_202_repro():
    csr = brainevent.CSR.fromdense(jnp.array([[1., 0.5], [0., 2.], [0.3, 0.]]))
    w, idx, ptr = csr.data, csr.indices, csr.indptr
    pre_spike = jnp.array([True, False, True])

    def loss_pre(w, post_trace):
        w_new = brainevent.update_csr_on_binary_pre(w, idx, ptr, pre_spike, post_trace, 0., 5., shape=(3, 2))
        return w_new.sum()

    trace = jnp.array([0.1, 0.2])
    np.testing.assert_allclose(jax.grad(loss_pre, 0)(w, trace), np.ones(4))
    # rows 0 and 2 fire: column 0 is hit by edges (0,0), (2,0); column 1 by (0,1).
    np.testing.assert_allclose(jax.grad(loss_pre, 1)(w, trace), [2., 1.])


@pytest.mark.parametrize('prim', [update_csr_on_binary_pre_p, update_csr_on_binary_post_p])
def test_csr_plasticity_transpose_rejects_structure_and_spikes(prim):
    from jax.interpreters import ad
    data, indices, indptr, rows, rng = _ad_problem(6)
    ct = [jnp.ones_like(data)]
    kw = dict(shape=(_AD_PRE, _AD_POST), backend=None)
    rule = ad.primitive_transposes[prim.primitive]
    if prim is update_csr_on_binary_pre_p:
        args = [data, indices, indptr, jnp.ones(_AD_PRE), jnp.ones(_AD_POST)]
        spike_pos = 3
    else:
        args = [data, indices, indptr, indices, jnp.ones(_AD_PRE), jnp.ones(_AD_POST)]
        spike_pos = 5
    for pos in (1, spike_pos):
        bad = list(args)
        bad[pos] = ad.UndefinedPrimal(jax.typeof(args[pos]))
        with pytest.raises(ValueError):
            rule(ct, *bad, **kw)
    # Zero cotangents propagate as symbolic zeros.
    und = list(args)
    und[0] = ad.UndefinedPrimal(jax.typeof(data))
    trace_pos = 4
    und[trace_pos] = ad.UndefinedPrimal(jax.typeof(args[trace_pos]))
    out = rule([ad.Zero(jax.typeof(data))], *und, **kw)
    assert type(out[0]) is ad.Zero and type(out[trace_pos]) is ad.Zero


# ---------------------------------------------------------------------------
# Inactive edges are returned bit-identical, whatever the trace (NaN/Inf).
# docs/specs/2026-10-24-inactive-row-nan-isolation.md
# ---------------------------------------------------------------------------

def _nan_trace_problem(seed=31):
    rng = np.random.default_rng(seed)
    n_pre, n_post = 9, 7
    dense = rng.uniform(0.1, 1.0, (n_pre, n_post)) * (rng.random((n_pre, n_post)) < 0.6)
    csr = brainevent.CSR.fromdense(jnp.asarray(dense, jnp.float32))
    rows = np.repeat(np.arange(n_pre), np.diff(np.asarray(csr.indptr)))
    cols = np.asarray(csr.indices)
    return csr, rows, cols, (n_pre, n_post)


@pytest.mark.parametrize('backend', PRE_BACKENDS)
@pytest.mark.parametrize('spike_dtype', ['bool', 'float32'])
@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_csr_on_pre_inactive_edges_ignore_nonfinite_trace(backend, spike_dtype, bad):
    csr, rows, cols, shape = _nan_trace_problem()
    spike = np.resize([2., 0., -1., 0.], shape[0]).astype(np.float32)
    spike = jnp.asarray(spike != 0) if spike_dtype == 'bool' else jnp.asarray(spike)
    trace = jnp.full((shape[1],), bad, jnp.float32)
    new = np.asarray(update_csr_on_binary_pre(
        csr.data, csr.indices, csr.indptr, spike, trace, shape=shape, backend=backend,
    ))
    inactive = ~(np.asarray(spike) != 0)[rows]
    assert inactive.any() and (~inactive).any()
    np.testing.assert_array_equal(new[inactive], np.asarray(csr.data)[inactive])
    assert not np.isfinite(new[~inactive]).any()


@pytest.mark.parametrize('backend', POST_BACKENDS)
@pytest.mark.parametrize('spike_dtype', ['bool', 'float32'])
@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_csr_on_post_inactive_edges_ignore_nonfinite_trace(backend, spike_dtype, bad):
    csr, rows, cols, shape = _nan_trace_problem(32)
    csc_indices, csc_indptr, weight_indices = _csr_to_csc_with_weight_indices(
        csr.data, csr.indices, csr.indptr, shape,
    )
    spike = np.resize([0., 1., -2., 0.], shape[1]).astype(np.float32)
    spike = jnp.asarray(spike != 0) if spike_dtype == 'bool' else jnp.asarray(spike)
    trace = jnp.full((shape[0],), bad, jnp.float32)
    new = np.asarray(update_csr_on_binary_post(
        csr.data, csc_indices, csc_indptr, weight_indices, trace, spike, shape=shape, backend=backend,
    ))
    inactive = ~(np.asarray(spike) != 0)[cols]
    assert inactive.any() and (~inactive).any()
    np.testing.assert_array_equal(new[inactive], np.asarray(csr.data)[inactive])
    assert not np.isfinite(new[~inactive]).any()


@pytest.mark.parametrize('backend', PRE_BACKENDS)
def test_csr_on_pre_inactive_edges_are_bit_identical(backend):
    """Inactive edges are returned unchanged, including the sign of ``-0.0``."""
    csr, rows, cols, shape = _nan_trace_problem(34)
    data = jnp.where(jnp.arange(csr.nse) % 2 == 0, -0.0, csr.data).astype(jnp.float32)
    spike = jnp.asarray(np.resize([True, False], shape[0]))
    trace = jnp.full((shape[1],), jnp.nan, jnp.float32)
    new = np.asarray(update_csr_on_binary_pre(
        data, csr.indices, csr.indptr, spike, trace, shape=shape, backend=backend,
    ))
    inactive = ~np.asarray(spike)[rows]
    np.testing.assert_array_equal(np.signbit(new[inactive]), np.signbit(np.asarray(data)[inactive]))
    np.testing.assert_array_equal(new[inactive], np.asarray(data)[inactive])


@pytest.mark.parametrize('backend', POST_BACKENDS)
def test_csr_on_post_inactive_edges_are_bit_identical(backend):
    """Inactive edges are returned unchanged, including the sign of ``-0.0``."""
    csr, rows, cols, shape = _nan_trace_problem(35)
    data = jnp.where(jnp.arange(csr.nse) % 2 == 0, -0.0, csr.data).astype(jnp.float32)
    csc_indices, csc_indptr, weight_indices = _csr_to_csc_with_weight_indices(
        data, csr.indices, csr.indptr, shape,
    )
    spike = jnp.asarray(np.resize([False, True], shape[1]))
    trace = jnp.full((shape[0],), jnp.nan, jnp.float32)
    new = np.asarray(update_csr_on_binary_post(
        data, csc_indices, csc_indptr, weight_indices, trace, spike, shape=shape, backend=backend,
    ))
    inactive = ~np.asarray(spike)[cols]
    np.testing.assert_array_equal(np.signbit(new[inactive]), np.signbit(np.asarray(data)[inactive]))
    np.testing.assert_array_equal(new[inactive], np.asarray(data)[inactive])


@pytest.mark.parametrize('backend', PRE_BACKENDS)
def test_csr_on_pre_trace_cotangent_ignores_inactive_edges(backend):
    csr, rows, cols, shape = _nan_trace_problem(33)
    spike = jnp.asarray(np.resize([True, False], shape[0]))
    inactive = ~np.asarray(spike)[rows]
    ct = jnp.asarray(np.where(inactive, np.nan, 1.0).astype(np.float32))

    def f(trace):
        return update_csr_on_binary_pre(csr.data, csr.indices, csr.indptr, spike, trace,
                                        shape=shape, backend=backend)

    _, vjp = jax.vjp(f, jnp.ones(shape[1], jnp.float32))
    (ct_trace,) = vjp(ct)
    assert np.isfinite(np.asarray(ct_trace)).all()
