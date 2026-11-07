"""int64 indptr with nnz > 2**31: the hybrid delay projection reads edges beyond INT32_MAX correctly."""

import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import KernelCompilationError
from brainevent._csr.delay.test._reference import large_indices, large_structure
from brainevent._csr.delay.region_delay import TASK_NNZ, TPR_THRESHOLD, HybridWorkspace
from brainevent._csr.delay.test._helpers import GPU, cuda_step

pytestmark = [pytest.mark.slow, GPU]


def test_edges_beyond_int32(request) -> None:
    if not request.config.getoption("--run-large-delay-tests"):
        pytest.skip("requires --run-large-delay-tests (about 20-30 GB GPU memory)")
    cols = 1021
    indptr, tail = large_structure(cols)
    nnz = int(indptr[-1])
    indices = large_indices(tail)
    weights = jnp.ones(nnz, jnp.float32)  # ring entries are exact edge counts
    lengths = np.diff(indptr)
    # One delay segment (delay 0): rows need no sorting, so the metadata is built directly.
    delay_count = lengths.reshape(-1, 1)
    delay_off = np.zeros(1, np.int32)
    tasks = np.where(lengths > TPR_THRESHOLD, (lengths + TASK_NNZ - 1) // TASK_NNZ, 0).astype(np.int32)
    capacity = int(tasks.sum())
    work = HybridWorkspace(jnp.zeros(capacity, jnp.int64), jnp.zeros(capacity, jnp.int64),
                           jnp.zeros(capacity, jnp.int32), jnp.zeros(2, jnp.int32))
    events = np.array([0, 1, 1, -2, np.nan], np.float32)  # the leading row stays inactive
    ring, work = cuda_step(weights, indices, jnp.asarray(indptr), jnp.asarray(events),
                                      jnp.asarray(delay_count), jnp.asarray(delay_off), jnp.asarray(tasks),
                                      jnp.array([0], jnp.int32), jnp.zeros((1, cols), jnp.float32), work)
    assert np.asarray(work.status)[1] == 0
    np.testing.assert_array_equal(np.asarray(ring)[0], np.bincount(tail, minlength=cols).astype(np.float32))

    with pytest.raises(KernelCompilationError, match="int32 indptr"):
        cuda_step(weights, indices, jnp.asarray(indptr.astype(np.int32)), jnp.asarray(events),
                             jnp.asarray(delay_count.astype(np.int32)), jnp.asarray(delay_off), jnp.asarray(tasks),
                             jnp.array([0], jnp.int32), ring, work)
