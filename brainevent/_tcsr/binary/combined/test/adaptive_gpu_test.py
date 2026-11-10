# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Exercise typed shared accumulation and permuted reads in every CTA tier."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._tcsr.binary.combined import jax_ffi as direct
from brainevent._tcsr.binary.combined.col_major import jax_ffi as mirror
from brainevent._tcsr.binary.combined.planner import tile_schedule


pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="CUDA device required")


@pytest.mark.parametrize("cols,batch,dtype,threads", [
    (100000, 2, jnp.float32, 256),
    (400003, 2, jnp.float32, 512),
    (2000003, 2, jnp.float32, 1024),
    (1200001, 2, jnp.float64, 512),
])
@pytest.mark.parametrize("mode", ["direct", "homo", "mirror"])
@pytest.mark.parametrize("weighted", [False, True])
def test_adaptive_scatter_tiers(cols, batch, dtype, threads, mode, weighted):
    schedule = tile_schedule(cols, batch, 8 if dtype == jnp.float64 else 4)
    assert schedule.block_size == threads
    width = schedule.tile_width
    ptr = np.array([0, 3, 5, 5, 8], np.int64)
    indices = np.array([cols - 1, 0, width - 1, width, width,
                        cols - 1, width + 1, 0], np.int32)
    values = jnp.asarray([1, 2, -0.5, 0.25, 1.5, -1, 2, 0.5], dtype)
    events = jnp.asarray([[1, -2, 3, 0.5], [0, 0, 1, 0]], dtype if weighted else jnp.float32)
    if mode == "homo":
        values = jnp.asarray([0.5], dtype)
    if mode == "mirror":
        # The mirror traversal retains its sorted order; only weights use perm.
        perm = np.concatenate([np.argsort(indices[ptr[r]:ptr[r + 1]], kind="stable") + ptr[r]
                               for r in range(4)]).astype(np.int64)
        plan = mirror.prepare(ptr, indices[perm], cols, batch, dtype=dtype)
        launch = mirror.spmm_weighted if weighted else mirror.spmm
        execute = jax.jit(lambda v, e: launch(v, jnp.asarray(perm), e, plan))
    else:
        plan = direct.prepare(ptr, indices, cols, batch, dtype=dtype)
        launch = direct.spmm_weighted if weighted else direct.spmm
        execute = jax.jit(lambda v, e: launch(v, e, plan))
    expected = np.zeros((batch, cols), np.float64)
    host_values = np.asarray(values)
    for b in range(batch):
        for r in range(4):
            event = float(events[b, r])
            if event != 0:
                entries = slice(ptr[r], ptr[r + 1])
                weights = np.full(ptr[r + 1] - ptr[r], host_values[0]) if mode == "homo" else host_values[entries]
                np.add.at(expected[b], indices[entries], weights * (event if weighted else 1))
    np.testing.assert_array_equal(np.asarray(execute(values, events)), expected)
    # Reuse output/scratch on another step with no active events.
    np.testing.assert_array_equal(np.asarray(execute(values, jnp.zeros_like(events))), np.zeros_like(expected))
