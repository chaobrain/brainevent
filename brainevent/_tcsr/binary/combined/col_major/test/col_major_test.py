# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""GPU mapping checks against independent original edge coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._tcsr.binary.combined.col_major import jax_ffi as api

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="CUDA device required")
DTYPES = [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]


def make_matrix(m=24581, n=7):
    lengths = np.array([0, 2, 65, 513, 17, 3, 129])[:n]
    cols = np.repeat(np.arange(n), lengths)
    rows = (np.arange(len(cols)) * 37 + cols * 13) % m
    rows[2:10] = np.array([0, 8191, 8192, 12287, 12288, 24575, 24576, 24576]) % m
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]  # Original CSR edge/value order.
    perm = np.lexsort((rows, cols))
    ptr = np.concatenate(([0], np.cumsum(np.bincount(cols, minlength=n))))
    return rows, cols, (ptr, rows[perm], perm)


def reference(rows, cols, weights, events, m):
    result = np.zeros((len(events), m), np.float64)
    for b, event in enumerate(events):
        active = event[cols] != 0
        with np.errstate(invalid="ignore"):
            np.add.at(result[b], rows[active], weights[active].astype(np.float64))
    return result


def direct_call(values, perm, events, plan, *, local=None, offsets=None, width=None,
                active_size=None, scratch_size=None):
    outputs = (jax.ShapeDtypeStruct((plan.batch, plan.cols), values.dtype),
               jax.ShapeDtypeStruct((max(1, plan.batch * plan.rows) if active_size is None else active_size,), jnp.int32),
               jax.ShapeDtypeStruct((api.workspace_words(plan.rows, plan.batch) if scratch_size is None else scratch_size,), jnp.uint32))
    suffix = f"{api._base.VALUE_SUFFIX[values.dtype]}_{api._base.EVENT_SUFFIX[events.dtype]}"
    target = f"{api._load()}.combined_col_major_binary_hetero_{suffix}"
    return jax.ffi.ffi_call(target, outputs)(
        values, jnp.asarray(plan.local_targets) if local is None else local,
        jnp.asarray(plan.tile_offsets) if offsets is None else offsets, perm, events,
        tile_width=plan.tile_width if width is None else width,
        value_type=api._base.VALUE_CODE[plan.value_dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.float32])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("offset_dtype", [np.int32, np.int64])
def test_dtype_matrix_jit_and_updated_weights(dtype, event_dtype, index_dtype, offset_dtype):
    rows, cols, (ptr, idx, perm) = make_matrix()
    assert not np.array_equal(perm, np.arange(len(perm)))
    plan = api.prepare(ptr.astype(offset_dtype), idx.astype(index_dtype), 24581, 3, dtype=dtype)
    device_perm = jnp.asarray(perm, offset_dtype)
    events = jnp.asarray([[0, 1, -2, 0, 0.5, np.nan, 1], [1, 0, 1, -1, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0]], event_dtype)
    call = jax.jit(lambda v, p, e: api.spmm(v, p, e, plan))
    for shift, event in [(0, events), (0, jnp.zeros_like(events)), (3, events)]:
        values = jnp.asarray(((np.arange(len(rows)) + shift) % 7 - 3) / 4, dtype)
        out = call(values, device_perm, event)
        expected = reference(rows, cols, np.asarray(values), np.asarray(event), 24581)
        np.testing.assert_array_equal(np.asarray(out).astype(np.float64), expected)
        assert out.dtype == jnp.dtype(dtype)
    # perm is an operand, not folded into a second weight array or a host-only plan.
    text = str(jax.make_jaxpr(lambda v, p, e: api.spmm(v, p, e, plan))(values, device_perm, events))
    assert "ffi_call" in text and "gather" not in text
    reverse = device_perm[::-1]
    remapped = np.empty(len(rows))
    remapped[perm] = np.asarray(values)[np.asarray(reverse)]
    np.testing.assert_array_equal(np.asarray(call(values, reverse, events)),
                                  reference(rows, cols, remapped, np.asarray(events), 24581))


@pytest.mark.parametrize("dtype", DTYPES)
def test_nonfinite_active_and_inactive_weights(dtype):
    rows = np.array([0, 1, 2, 3, 4])
    cols = np.array([2, 0, 1, 3, 0])
    perm = np.argsort(cols, kind="stable")
    ptr = np.array([0, 2, 3, 4, 5], np.int64)
    plan = api.prepare(ptr, rows[perm], 5, 3, dtype=dtype)
    values = jnp.asarray([1, np.nan, np.inf, -np.inf, 2], dtype)
    events = jnp.asarray([[0, 0, np.nan, 0], [0, -2, 0, 0], [1, 0, 0, np.inf]], jnp.float32)
    out = api.spmm(values, jnp.asarray(perm, jnp.int64), events, plan)
    np.testing.assert_array_equal(np.asarray(out).astype(np.float64),
                                  reference(rows, cols, np.asarray(values), np.asarray(events), 5))


@pytest.mark.parametrize("cols", [12287, 12288, 12289, 24575, 24576, 24577])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tail_tile_and_output_bounds(cols, dtype):
    plan = api.prepare(np.array([0, 2, 3], np.int32), np.array([0, cols - 1, cols - 1], np.int64), cols, 2, dtype=dtype)
    out = api.spmm(jnp.array([1, 2, 4], dtype), jnp.array([2, 0, 1], jnp.int32),
                   jnp.array([[1, 1], [0, 0]], jnp.bool_), plan)
    expected = np.zeros((2, cols))
    expected[0, 0], expected[0, -1] = 4, 3
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_ordered_extraction_across_chunks():
    n, m, batch = 8201, 24577, 3
    cols = np.array([0, 31, 32, 4095, 4096, 4097, 8191, 8192, 8200])
    rows = np.array([0, 1, 24576, 3, 12288, 8192, 6, 7, 8])
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    perm = np.argsort(cols, kind="stable")
    ptr = np.concatenate(([0], np.cumsum(np.bincount(cols, minlength=n)))).astype(np.int64)
    plan = api.prepare(ptr, rows[perm], m, batch)
    events = np.zeros((batch, n), np.float32)
    events[0, cols] = 1
    events[1, cols[::2]] = np.nan
    values = jnp.asarray(np.arange(len(rows)) + 1, jnp.float32)
    out, scratch_active, scratch = direct_call(values, jnp.asarray(perm, jnp.int64), jnp.asarray(events), plan)
    np.testing.assert_array_equal(np.asarray(out), reference(rows, cols, np.asarray(values), events, m))
    counts = np.asarray(scratch)[:2 * batch].reshape(batch, 2)[:, 0]
    active = np.asarray(scratch_active).reshape(batch, n)
    for b in range(batch):
        expected = np.flatnonzero(events[b] != 0)
        assert counts[b] == len(expected)
        np.testing.assert_array_equal(active[b, :len(expected)], expected)


@pytest.mark.parametrize("shape", [(5, 3, 2), (5, 0, 2), (0, 3, 2), (0, 0, 2), (5, 3, 0)])
def test_empty_structure(shape):
    m, n, batch = shape
    plan = api.prepare(np.zeros(n + 1, np.int64), np.empty(0, np.int32), m, batch)
    out = api.spmm(jnp.empty(0, jnp.float32), jnp.empty(0, jnp.int64), jnp.ones((batch, n), jnp.bool_), plan)
    np.testing.assert_array_equal(np.asarray(out), np.zeros((batch, m)))


def test_one_edge_hetero_and_int32_without_x64():
    with jax.enable_x64(False):
        plan = api.prepare(np.array([0, 0, 1], np.int32), np.array([2], np.int32), 4, 2)
        result = api.spmm(jnp.array([3], jnp.float32), jnp.array([0], jnp.int32),
                          jnp.array([[0, -2], [1, 0]], jnp.float32), plan)
        np.testing.assert_array_equal(np.asarray(result), [[0, 0, 3, 0], [0, 0, 0, 0]])


@pytest.mark.parametrize("invalid", ["perm_dtype", "perm_rank", "perm_length", "homo", "offset_length",
                                     "local_dtype", "tile_width", "active", "workspace"])
def test_ffi_rejects_invalid_metadata(invalid):
    plan = api.prepare(np.array([0, 2], np.int64), np.array([0, 1], np.int32), 2, 1)
    values, perm = jnp.array([1, 2], jnp.float32), jnp.array([1, 0], jnp.int64)
    events, kwargs = jnp.ones((1, 1), jnp.bool_), {}
    if invalid == "perm_dtype":
        perm = perm.astype(jnp.int32)
    elif invalid == "perm_rank":
        perm = perm.reshape(1, 2)
    elif invalid == "perm_length":
        perm = perm[:1]
    elif invalid == "homo":
        values = values[:1]
    elif invalid == "offset_length":
        kwargs["offsets"] = jnp.asarray(plan.tile_offsets[:-1])
    elif invalid == "local_dtype":
        kwargs["local"] = jnp.asarray(plan.local_targets, jnp.int32)
    elif invalid == "tile_width":
        kwargs["width"] = 8192
    elif invalid == "active":
        kwargs["active_size"] = 0
    else:
        kwargs["scratch_size"] = 0
    with pytest.raises(Exception, match="dtype|rank|length|plan|metadata|scratch"):
        jax.block_until_ready(direct_call(values, perm, events, plan, **kwargs))


def test_empty_structure_still_validates_perm():
    plan = api.prepare(np.array([0, 0], np.int64), np.empty(0, np.int32), 2, 1)
    values, perm = jnp.empty(0, jnp.float32), jnp.empty(0, jnp.int32)
    events = jnp.ones((1, 1), jnp.bool_)
    with pytest.raises(ValueError, match="perm dtype"):
        api.spmm(values, perm, events, plan)
    with pytest.raises(Exception, match="dtype"):
        jax.block_until_ready(direct_call(values, perm, events, plan))


def test_existing_tcsr_mirror_is_reused():
    from brainevent import CSR, TCSR

    rows, cols, _ = make_matrix(m=37)
    ptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=37))))
    weights = jnp.asarray((np.arange(len(rows)) % 7 - 3) / 4, jnp.float32)
    matrix = TCSR.fromcsr(CSR(weights, cols, ptr, shape=(37, 7), indptr_dtype=np.int64))
    assert not matrix.has_tcsc_mirror
    matrix = matrix.build_weight_indices()
    mirror = matrix.buffers['csc']
    plan = api.prepare(np.asarray(mirror[0]), np.asarray(mirror[1]), 37, 2)
    events = jnp.array([[1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 0, 1]], jnp.bool_)
    out = api.spmm(matrix.data, mirror[2], events, plan)
    np.testing.assert_array_equal(np.asarray(out), reference(rows, cols, np.asarray(weights), np.asarray(events), 37))
    assert matrix.build_weight_indices().buffers['csc'] is mirror
