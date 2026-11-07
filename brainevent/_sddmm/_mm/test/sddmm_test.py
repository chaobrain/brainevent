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

"""Tests for ``brainevent/_sddmm/_mm/sddmm.py``, grouped by behavior."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from brainevent._sddmm import csr_sddmm_dweight, csr_sddmm_p, csr_sddmv_p, sddmm
from brainevent._test_util import jax_x64_enabled


# Matrix sampled products and event selection
# -------------------------------------------

@pytest.fixture(autouse=False)
def enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def operands(event_dtype, value_dtype, transpose):
    # Empty middle row and duplicate slots are deliberately retained.
    indices = jnp.array([2, 0, 1, 1], jnp.int32)
    indptr = jnp.array([0, 2, 2, 4], jnp.int64)
    events = jnp.asarray(np.resize([0, 2, 1, 3, 0], (3 if transpose else 4, 3)), event_dtype)
    ct = jnp.asarray(np.resize([1.25, -2, 4, 0.5], (4 if transpose else 3, 3)), value_dtype)
    return events, ct, indices, indptr


def reference(events, ct, indices, *, transpose, weighted):
    rows = jnp.array([0, 0, 2, 2], jnp.int32)
    unit, out = (rows, indices) if transpose else (indices, rows)
    x = events[unit]
    values = x.astype(ct.dtype) * ct[out] if weighted else ct[out]
    return jnp.sum(jnp.where(x != 0, values, 0), axis=1)


_EVENT_DTYPES = [jnp.bool_, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                 jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64,
                 jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]


_TYPE_PAIRS = [(dtype, jnp.float32) for dtype in _EVENT_DTYPES]


_TYPE_PAIRS += [(jnp.float64, dtype) for dtype in (jnp.float16, jnp.bfloat16, jnp.float64)]


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('event_dtype,value_dtype', _TYPE_PAIRS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_value_and_event_types(event_dtype, value_dtype, transpose, weighted):
    args = operands(event_dtype, value_dtype, transpose)
    result = jax.jit(lambda e, c, i, p: csr_sddmm_dweight(
        e, c, i, p, transpose=transpose, weighted=weighted))(*args)
    expected = reference(*args[:3], transpose=transpose, weighted=weighted)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(result, np.float64), np.asarray(expected, np.float64))


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_nonfinite_values_are_selected_before_batch_sum(value_dtype, transpose, weighted):
    e, c, i, p = operands(value_dtype, value_dtype, transpose)
    c = c.at[:, 0].set(jnp.nan).at[:, 1].set(jnp.inf)
    e = e.at[:, 0].set(0).at[0, 1].set(0)
    result = jax.jit(lambda x, y: csr_sddmm_dweight(
        x, y, i, p, transpose=transpose, weighted=weighted))(e, c)
    expected = reference(e, c, i, transpose=transpose, weighted=weighted)
    np.testing.assert_allclose(np.asarray(result, np.float64), np.asarray(expected, np.float64),
                               equal_nan=True)
    assert not jnp.any(jnp.isnan(result))


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
def test_jax_activity_precedes_narrowing_and_reduction(transpose):
    i, p = jnp.array([0, 1], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    e = jnp.array([[1e-200, 0], [0, 0]], jnp.float64)
    c = jnp.full((2, 2), jnp.inf, jnp.float32)
    result = csr_sddmm_dweight(e, c, i, p, transpose=transpose, weighted=True, backend='jax_raw')
    assert jnp.isnan(result[0]) and result[1] == 0


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('nonfinite', [False, True])
def test_vmap_and_shard_map_keep_each_batch_local(transpose, nonfinite):
    e, c, i, p = operands(jnp.float32, jnp.float32, transpose)
    if nonfinite:
        c = c.at[0, 0].set(jnp.inf)
    fn = lambda x, y: csr_sddmm_dweight(x, y, i, p, transpose=transpose, weighted=True)
    expected = reference(e, c, i, transpose=transpose, weighted=True)
    batched = jax.jit(jax.vmap(fn))(jnp.stack([e, e * 2]), jnp.stack([c, c * 3]))
    np.testing.assert_allclose(batched, jnp.stack([expected, expected * 6]), equal_nan=True)
    mesh = Mesh(np.array(jax.devices()[:1]), ('s',))
    sharded = jax.shard_map(fn, mesh=mesh, in_specs=(P('s'), P('s')), out_specs=P('s'))
    np.testing.assert_allclose(jax.jit(sharded)(e, c), expected, equal_nan=True)


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('rows,cols', [(0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize('batch', [0, 2])
def test_empty_patterns_and_matrix_axes(rows, cols, batch):
    i, p = jnp.empty(0, jnp.int32), jnp.zeros(rows + 1, jnp.int32)
    for transpose in (False, True):
        e = jnp.zeros((rows if transpose else cols, batch), jnp.float32)
        c = jnp.zeros((cols if transpose else rows, batch), jnp.float32)
        assert csr_sddmm_dweight(e, c, i, p, transpose=transpose).shape == (0,)


@pytest.mark.usefixtures('enable_x64')
@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
def test_coo_empty_and_out_of_bounds_slots(batch, mode):
    a, b = jnp.ones((2, batch)), jnp.full((batch, 3), 2.)
    coords = jnp.array([[0, 1], [0, 3], [2, 0], [0, 1]], jnp.int32)
    result = sddmm(a, b, coords, mode=mode)
    np.testing.assert_array_equal(result, [2 * batch, 0, 0, 2 * batch])
    assert sddmm(a, b, coords[:0], mode=mode).shape == (0,)


@pytest.mark.usefixtures('enable_x64')
def test_registry_call_and_no_event_derivative_rule():
    e, c, i, p = operands(jnp.float32, jnp.float32, True)
    assert set(csr_sddmm_p.available_backends('gpu')) == {'auto', 'cuda_raw', 'cusparse', 'jax_raw'}
    np.testing.assert_allclose(csr_sddmm_p.call(e, c, i, p, transpose=True, weighted=True),
                               reference(e, c, i, transpose=True, weighted=True))
    with pytest.raises(NotImplementedError, match='Differentiation rule'):
        jax.jvp(lambda x: csr_sddmm_dweight(e, x, i, p, transpose=True), (c,), (c,))


# Native registration, activity and portable execution
# ----------------------------------------------------

@pytest.mark.parametrize('batch', [3])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('backend', [None, 'auto', 'cuda_raw'])
def test_cuda_registration_and_lowering(batch, weighted, backend):
    primitive = csr_sddmv_p if batch == 1 else csr_sddmm_p
    assert {'auto', 'cuda_raw', 'jax_raw'} <= set(primitive.available_backends('gpu'))
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    e = jnp.asarray(np.resize([0., -2., 1.], (2, batch)), jnp.float32)
    c = jnp.arange(3 * batch, dtype=jnp.float32).reshape(3, batch)
    i, p = jnp.array([2, 0, 0], jnp.int32), jnp.array([0, 1, 3], jnp.int32)
    fn = jax.jit(lambda e, c, i, p: csr_sddmm_dweight(
        e, c, i, p, transpose=True, weighted=weighted, backend=backend))
    expected = csr_sddmm_dweight(e, c, i, p, transpose=True, weighted=weighted, backend='jax_raw')
    np.testing.assert_array_equal(fn(e, c, i, p), expected)
    target = 'hybrid_sddmv_' if batch == 1 else 'batched_v3_sddmm_'
    assert target in fn.lower(e, c, i, p).as_text()


@pytest.mark.parametrize('batch', [3])
@pytest.mark.parametrize('transpose,backend', [(True, 'jax_raw'), (False, 'jax_raw'), (False, 'auto')])
def test_portable_routes_do_not_load_cuda(monkeypatch, batch, transpose, backend):
    leaf = 'mv.sddmv' if batch == 1 else 'mm.sddmm'
    module = importlib.import_module(f'brainevent._sddmm._{leaf}')
    def fail_load(*args, **kwargs):
        raise AssertionError('portable execution must not load CUDA')
    monkeypatch.setattr(module, '_load_hybrid', fail_load)
    e = jnp.ones((2 if transpose else 3, batch), jnp.float32)
    c = jnp.full((3 if transpose else 2, batch), 2., jnp.float32)
    i, p = jnp.array([2, 0], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    fn = jax.jit(lambda e, c: csr_sddmm_dweight(
        e, c, i, p, transpose=transpose, weighted=True, backend=backend))
    np.testing.assert_array_equal(fn(e, c), [2 * batch, 2 * batch])
    ir = fn.lower(e, c).as_text()
    assert 'hybrid_sddmv_' not in ir and 'batched_v3_sddmm_' not in ir


@pytest.mark.parametrize('batch', [3])
@pytest.mark.parametrize('backend', ['auto', 'cuda_raw'])
@pytest.mark.parametrize('event_dtype,value_dtype,tiny', [
    (jnp.float64, jnp.float32, 1e-200),
    (jnp.float32, jnp.float16, 1e-10),
    (jnp.float64, jnp.bfloat16, 1e-200),
])
def test_weighted_cuda_checks_activity_after_cast(batch, backend, event_dtype, value_dtype, tiny):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    with jax_x64_enabled():
        e = jnp.zeros((3, batch), event_dtype).at[:, 0].set(
            jnp.array([tiny, -tiny, 2.], event_dtype))
        c = jnp.full((3, batch), -jnp.inf, value_dtype).at[:, 0].set(
            jnp.array([jnp.inf, jnp.nan, 3.], value_dtype))
        i, p = jnp.arange(3, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32)
        fn = jax.jit(lambda e, c: csr_sddmm_dweight(
            e, c, i, p, transpose=True, weighted=True, backend=backend))
        result = fn(e, c)
        assert result.dtype == c.dtype
        # Narrowed events and exact zeros skip nonfinite cotangents; the
        # active third row still contributes its weighted finite value.
        np.testing.assert_array_equal(np.asarray(result, np.float64), [0., 0., 6.])
        ir = fn.lower(e, c).as_text()
        target = 'hybrid_sddmv_weighted_' if batch == 1 else 'batched_v3_sddmm_weighted_'
        assert target in ir
        assert 'stablehlo.is_finite' not in ir


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('value_dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('offset_dtype', [jnp.int32, jnp.int64])
def test_mm_runtime_patterns_heavy_rows_and_repeated_calls(weighted, value_dtype, offset_dtype):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    with jax_x64_enabled():
        # 35 rows cross a mask word; 65 batches cross the adapter's batch chunk.
        rows, cols, batch, nnz = 35, 17, 65, 1539
        e = jnp.asarray(np.resize([0., 1., -2., .5], (rows, batch)), value_dtype)
        c = jnp.asarray(np.resize([.25, -.5, 1., 0.], (cols, batch)), value_dtype)
        i = jnp.arange(nnz, dtype=jnp.int32) % cols
        traces = []
        def call(e, c, i, p):
            traces.append(True)
            return csr_sddmm_dweight(e, c, i, p, transpose=True, weighted=weighted, backend='cuda_raw')
        fn = jax.jit(call)
        patterns = ([nnz] + [0] * 34, [513] * 3 + [0] * 32,
                    [0] * 31 + [513, 0, 0, 1026], [44] * 34 + [43])
        for lengths in patterns:
            p = jnp.asarray(np.r_[0, np.cumsum(lengths)], offset_dtype)
            r = np.repeat(np.arange(rows), lengths)
            ev = np.asarray(e, np.float64)[r]
            ct = np.asarray(c, np.float64)[np.asarray(i)]
            expected = np.where(ev != 0, ev * ct if weighted else ct, 0).sum(axis=1)
            expected = np.asarray(jnp.asarray(expected, value_dtype), np.float64)
            np.testing.assert_allclose(np.asarray(fn(e, c, i, p), np.float64), expected, rtol=1e-6)
            np.testing.assert_array_equal(fn(jnp.zeros_like(e), c, i, p), np.zeros(nnz))
        assert len(traces) == 1


@pytest.mark.parametrize('nnz,batch', [(0, 3), (2, 0), (513, 2)])
def test_mm_degenerate_shapes_and_single_row(nnz, batch):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    e, c = jnp.ones((1, batch), jnp.float32), jnp.full((2, batch), 2., jnp.float32)
    i, p = jnp.zeros(nnz, jnp.int32), jnp.array([0, nnz], jnp.int32)
    for weighted in (False, True):
        fn = jax.jit(lambda e, c: csr_sddmm_dweight(
            e, c, i, p, transpose=True, weighted=weighted, backend='cuda_raw'))
        np.testing.assert_array_equal(fn(e, c), np.full(nnz, 2 * batch))


def test_mm_out_of_range_batch_falls_back_under_auto():
    if jax.default_backend() != 'gpu':
        pytest.skip('GPU auto routing requires a GPU')
    e = jnp.ones((1, 65536), jnp.float32)
    i, p = jnp.array([0], jnp.int32), jnp.array([0, 1], jnp.int32)
    np.testing.assert_array_equal(csr_sddmm_dweight(e, e, i, p, transpose=True), [65536.])
    with pytest.raises(ValueError, match='65535'):
        csr_sddmm_dweight(e, e, i, p, transpose=True, backend='cuda_raw')
