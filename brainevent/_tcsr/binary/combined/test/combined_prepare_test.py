# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Batch-dependent metadata and validation for both combined directions."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._tcsr.binary.combined import jax_ffi as api
from brainevent._tcsr.binary.combined.col_major import jax_ffi as col_api
from brainevent._tcsr.binary.combined.planner import tile_schedule


@pytest.mark.parametrize("direction", [api, col_api])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_batch_changes_metadata_and_rejects_stale_plan(direction, dtype):
    cols = 100000
    ptr = np.array([0, 5, 5, 8], np.int64)
    idx = np.array([0, 1055, 1056, 2144, cols - 1, 7, 2143, 4288], np.int32)
    plans = [direction.prepare(ptr, idx, cols, batch, dtype=dtype) for batch in (2, 4)]
    assert [p.tile_width for p in plans] == [1056, 2144]
    for plan in plans:
        width = plan.tile_width
        tiles = (cols + width - 1) // width
        expected_offsets = [[int(ptr[r]) + np.count_nonzero(idx[ptr[r]:ptr[r + 1]] < tile * width)
                             for tile in range(tiles + 1)] for r in range(3)]
        np.testing.assert_array_equal(plan.tile_offsets.reshape(3, tiles + 1), expected_offsets)
        np.testing.assert_array_equal(plan.local_targets, idx % width)
        values = jnp.ones(8, dtype)
        events = jnp.ones((plan.batch, plan.rows), jnp.bool_)
        stale = replace(plan, batch=8)
        with pytest.raises(ValueError, match="tile width"):
            if direction is api:
                direction._validate(values, jnp.ones((8, 3), jnp.bool_), stale)
            else:
                direction._validate(values, jnp.arange(8, dtype=jnp.int64), jnp.ones((8, 3), jnp.bool_), stale)
        if direction is api:
            direction._validate(values, events, plan)
        else:
            direction._validate(values, jnp.arange(8, dtype=jnp.int64), events, plan)


@pytest.mark.parametrize("direction", [api, col_api])
def test_padding_ctas_are_excluded_from_offsets(direction):
    plan = direction.prepare(np.array([0, 2], np.int32), np.array([0, 1], np.int32), 2, 1)
    schedule = tile_schedule(plan.cols, plan.batch)
    assert schedule.grid_tiles == 188 and schedule.tile_count == 2
    assert plan.tile_width == 1
    np.testing.assert_array_equal(plan.tile_offsets, [0, 1, 2])
    np.testing.assert_array_equal(plan.local_targets, [0, 0])


@pytest.mark.parametrize("direction", [api, col_api])
@pytest.mark.parametrize("cols,batch", [(0, 0), (0, 2), (100000, 0), (100000, 2)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_empty_csr_metadata(direction, cols, batch, dtype):
    plan = direction.prepare(np.zeros(4, np.int64), np.empty(0, np.int32), cols, batch, dtype=dtype)
    schedule = tile_schedule(cols, batch, 8 if dtype == jnp.float64 else 4)
    assert plan.tile_width == schedule.tile_width
    assert plan.tile_offsets.size == 3 * (schedule.tile_count + 1)
    assert not np.any(plan.tile_offsets)
    assert plan.local_targets.size == 0
