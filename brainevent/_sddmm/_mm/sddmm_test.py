"""Matrix event gradients preserve each slot, batch contribution and dtype."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from brainevent._sddmm import csr_sddmm_dweight, csr_sddmm_p, sddmm


@pytest.fixture(autouse=True)
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


@pytest.mark.parametrize('transpose', [False, True])
def test_activity_precedes_narrowing_and_reduction(transpose):
    i, p = jnp.array([0, 1], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    e = jnp.array([[1e-200, 0], [0, 0]], jnp.float64)
    c = jnp.full((2, 2), jnp.inf, jnp.float32)
    result = csr_sddmm_dweight(e, c, i, p, transpose=transpose, weighted=True)
    assert jnp.isnan(result[0]) and result[1] == 0


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


@pytest.mark.parametrize('rows,cols', [(0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize('batch', [0, 2])
def test_empty_patterns_and_matrix_axes(rows, cols, batch):
    i, p = jnp.empty(0, jnp.int32), jnp.zeros(rows + 1, jnp.int32)
    for transpose in (False, True):
        e = jnp.zeros((rows if transpose else cols, batch), jnp.float32)
        c = jnp.zeros((cols if transpose else rows, batch), jnp.float32)
        assert csr_sddmm_dweight(e, c, i, p, transpose=transpose).shape == (0,)


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
def test_coo_empty_and_out_of_bounds_slots(batch, mode):
    a, b = jnp.ones((2, batch)), jnp.full((batch, 3), 2.)
    coords = jnp.array([[0, 1], [0, 3], [2, 0], [0, 1]], jnp.int32)
    result = sddmm(a, b, coords, mode=mode)
    np.testing.assert_array_equal(result, [2 * batch, 0, 0, 2 * batch])
    assert sddmm(a, b, coords[:0], mode=mode).shape == (0,)


def test_registry_call_and_no_event_derivative_rule():
    e, c, i, p = operands(jnp.float32, jnp.float32, True)
    assert set(csr_sddmm_p.available_backends('gpu')) == {'jax_raw'}
    np.testing.assert_allclose(csr_sddmm_p.call(e, c, i, p, transpose=True, weighted=True),
                               reference(e, c, i, transpose=True, weighted=True))
    with pytest.raises(NotImplementedError, match='Differentiation rule'):
        jax.jvp(lambda x: csr_sddmm_dweight(e, x, i, p, transpose=True), (c,), (c,))
