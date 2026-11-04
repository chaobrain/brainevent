"""Large-ring indexing without allocating the ring's storage."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.interpreters import ad

from brainevent._csr.delay.delay import _edge_locations


@pytest.mark.parametrize("slots,cols,offset_dtype,x64", [
    (65536, 32767, np.int32, False),
    (50000, 40000, np.int32, False),
    (2**31 - 1, 1, np.int32, False),
    (65536, 32769, np.int64, True),
    (2**31 - 1, 2**30 - 1, np.int64, True),
    (1, 5, np.int32, False),
])
@pytest.mark.parametrize("undefined_ring", [False, True], ids=["forward", "transpose"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_large_ring_slots_do_not_overflow(slots, cols, offset_dtype, x64,
                                         undefined_ring, compiled):
    # Python integer arithmetic is the reference, including when even the
    # slot-level head + delay exceeds INT32_MAX. Only O(nnz) arrays are created.
    delays = sorted({0, min(1, slots - 1), slots // 2, slots - 1})
    counts = np.ones((2, len(delays)), np.int32)
    counts[1, 0] = 0
    counts[1, -1] = 2
    edge_rows = [row for row in range(2) for k in range(len(delays))
                 for _ in range(int(counts[row, k]))]
    edge_delays = [delay for row in range(2) for k, delay in enumerate(delays)
                   for _ in range(int(counts[row, k]))]
    nnz = len(edge_rows)
    ring = (ad.UndefinedPrimal(jax.core.ShapedArray((slots, cols), jnp.float32))
            if undefined_ring else jax.ShapeDtypeStruct((slots, cols), jnp.float32))

    with jax.enable_x64(x64):
        args = (ring, jnp.ones(2), jnp.ones(nnz), jnp.zeros(nnz, jnp.int32),
                jnp.asarray([0, int(counts[0].sum()), nnz], jnp.int32),
                jnp.asarray(counts))
        offsets = jnp.asarray([delay * cols for delay in delays], offset_dtype)

        def locate(head, offsets):
            return _edge_locations((*args, offsets, head))[1:3]

        if compiled:
            locate = jax.jit(locate)
        for head in sorted({0, min(1, slots - 1), slots // 2, slots - 1}):
            rows, actual = locate(jnp.asarray([head], jnp.int32), offsets)
            expected = [(head + delay) % slots for delay in edge_delays]
            np.testing.assert_array_equal(rows, edge_rows)
            np.testing.assert_array_equal(actual, expected,
                                          err_msg=f"head={head}, shape={(slots, cols)}")
            assert actual.dtype == np.dtype(offset_dtype)


def test_empty_zero_column_ring_has_no_edge_locations():
    with jax.enable_x64(False):
        args = (jax.ShapeDtypeStruct((2**31 - 1, 0), jnp.float32),
                jnp.zeros(2), jnp.zeros(0), jnp.zeros(0, jnp.int32),
                jnp.zeros(3, jnp.int32), jnp.zeros((2, 1), jnp.int32),
                jnp.zeros(1, jnp.int32), jnp.asarray([2**31 - 2], jnp.int32))
        rows, edge_rows, slots, head = jax.jit(lambda: _edge_locations(args))()
        assert int(rows) == 2
        assert edge_rows.shape == slots.shape == (0,)
        np.testing.assert_array_equal(head, args[7])
