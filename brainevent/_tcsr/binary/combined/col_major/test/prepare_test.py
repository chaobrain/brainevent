# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Host metadata and input contracts, without loading CUDA."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._tcsr.binary.combined.col_major import jax_ffi as api


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize("offset_dtype", [np.int32, np.int64])
def test_absolute_boundaries_and_mirror_order(dtype, offset_dtype):
    max_width = 12288 if dtype == jnp.float64 else 24576
    ptr = np.array([0, 0, 4, 7], offset_dtype)
    idx = np.array([0, max_width - 1, max_width, max_width, 1, max_width + 1, 2 * max_width], np.int64)
    plan = api.prepare(ptr, idx, 2 * max_width + 1, 3, dtype=dtype)
    width = plan.tile_width
    tiles = (plan.cols + width - 1) // width
    expected = [[int(ptr[row]) + np.count_nonzero(idx[ptr[row]:ptr[row + 1]] < tile * width)
                 for tile in range(tiles + 1)] for row in range(3)]
    np.testing.assert_array_equal(plan.tile_offsets.reshape(3, tiles + 1), expected)
    np.testing.assert_array_equal(plan.local_targets, idx % width)
    assert plan.tile_offsets.dtype == offset_dtype and plan.local_targets.dtype == np.uint16
    assert not plan.tile_offsets.flags.writeable and not plan.local_targets.flags.writeable
    ptr[:] = 0
    idx[:] = 0
    assert plan.tile_offsets[-1] == 7 and plan.local_targets[1] == (max_width - 1) % width
    assert not hasattr(plan, "permutation")  # The existing mirror perm stays a runtime input.


@pytest.mark.parametrize("invalid", ["unsorted", "negative", "range", "ptr_start", "ptr_end",
                                     "ptr_order", "ptr_dtype", "idx_dtype", "idx_rank", "batch",
                                     "output_rows", "dtype"])
def test_prepare_rejects_invalid_structure(invalid):
    ptr = np.array([0, 2, 3], np.int64)
    idx = np.array([0, 2, 1], np.int32)
    m, batch, dtype = 3, 2, jnp.float32
    if invalid == "unsorted":
        idx[:2] = [2, 0]
    elif invalid == "negative":
        idx[0] = -1
    elif invalid == "range":
        idx[-1] = 3
    elif invalid == "ptr_start":
        ptr[0] = 1
    elif invalid == "ptr_end":
        ptr[-1] = 2
    elif invalid == "ptr_order":
        ptr[1] = 4
    elif invalid == "ptr_dtype":
        ptr = ptr.astype(np.uint32)
    elif invalid == "idx_dtype":
        idx = idx.astype(np.float32)
    elif invalid == "idx_rank":
        idx = idx.reshape(1, 3)
    elif invalid == "batch":
        batch = 65536
    elif invalid == "output_rows":
        m = 0
    else:
        dtype = jnp.int32
    with pytest.raises(ValueError):
        api.prepare(ptr, idx, m, batch, dtype=dtype)


@pytest.mark.parametrize("invalid", ["perm_dtype", "perm_length", "perm_rank", "scalar", "homo",
                                     "weight_dtype", "event_dtype", "event_shape", "tile_width"])
def test_python_rejects_invalid_inputs_before_loading(monkeypatch, invalid):
    monkeypatch.setattr(api, "_load", lambda: pytest.fail("must reject before loading CUDA"))
    plan = api.prepare(np.array([0, 2], np.int64), np.array([0, 1], np.int32), 2, 1)
    values, perm = jnp.array([1, 2], jnp.float32), jnp.array([1, 0], jnp.int64)
    events = jnp.ones((1, 1), jnp.bool_)
    if invalid == "perm_dtype":
        perm = perm.astype(jnp.int32)
    elif invalid == "perm_length":
        perm = perm[:1]
    elif invalid == "perm_rank":
        perm = perm.reshape(1, 2)
    elif invalid == "scalar":
        values = values[0]
    elif invalid == "homo":
        values = values[:1]
    elif invalid == "weight_dtype":
        values = values.astype(jnp.float64)
    elif invalid == "event_dtype":
        events = events.astype(jnp.int32)
    elif invalid == "event_shape":
        events = jnp.ones((1, 2), jnp.bool_)
    else:
        plan = replace(plan, tile_width=8192)
    with pytest.raises(ValueError):
        api.spmm(values, perm, events, plan)


def test_int64_metadata_cannot_be_silently_narrowed(monkeypatch):
    plan = api.prepare(np.array([0, 1], np.int64), np.array([0], np.int32), 2, 1)
    values, perm = jnp.array([1], jnp.float32), jnp.array([0], jnp.int64)
    events = jnp.ones((1, 1), jnp.bool_)
    monkeypatch.setattr(api, "_load", lambda: pytest.fail("must reject before loading CUDA"))
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="preserve int64"):
            api.spmm(values, perm, events, plan)
