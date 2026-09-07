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

"""Test staged TCSR preprocessing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR as PlainCSR
from brainevent._tcsr import preprocess
from brainevent._tcsr.preprocess_config import hybrid_task_capacity


def test_sort_csr_stably_aligns_data_and_indices() -> None:
    """Keep duplicate values in input order while sorting each CSR row."""
    source = PlainCSR(
        (
            jnp.asarray([30.0, 10.0, 11.0, 20.0], dtype=jnp.float32),
            jnp.asarray([3, 1, 1, 2], dtype=jnp.int32),
            jnp.asarray([0, 4, 4], dtype=jnp.int32),
        ),
        shape=(2, 5),
    )

    with jax.enable_x64():
        result = preprocess.sort_csr(source)

    assert isinstance(result, PlainCSR)
    np.testing.assert_array_equal(result.indices, [1, 1, 2, 3])
    np.testing.assert_array_equal(result.indptr, [0, 4, 4])
    np.testing.assert_array_equal(result.data, [10.0, 11.0, 20.0, 30.0])


def test_sort_csr_preserves_homogeneous_data() -> None:
    """Keep one shared value compact when sorting multiple entries."""
    source = PlainCSR(
        (
            jnp.asarray([2.0], dtype=jnp.float32),
            jnp.asarray([3, 1, 2], dtype=jnp.int32),
            jnp.asarray([0, 3], dtype=jnp.int32),
        ),
        shape=(1, 4),
    )

    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        result = preprocess.sort_csr(source)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)

    np.testing.assert_array_equal(result.indices, [1, 2, 3])
    np.testing.assert_array_equal(result.data, [2.0])


def test_build_tcsr_structure_constructs_cross_tile_metadata() -> None:
    """Describe exact row-local boundaries across full and partial tiles."""
    source = PlainCSR(
        (
            jnp.arange(1.0, 7.0, dtype=jnp.float32),
            jnp.asarray([3, 8191, 8192, 9000, 0, 9999], dtype=jnp.int32),
            jnp.asarray([0, 4, 4, 6], dtype=jnp.int32),
        ),
        shape=(3, 10_000),
    )

    with jax.enable_x64():
        components, local_targets, tile_offsets = preprocess.build_tcsr_structure(
            source
        )

    np.testing.assert_array_equal(components.data, [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(components.indices, [3, 8191, 8192, 9000, 0, 9999])
    assert components.indptr.dtype == jnp.int64
    np.testing.assert_array_equal(components.indptr, [0, 4, 4, 6])
    np.testing.assert_array_equal(local_targets, [3, 8191, 0, 808, 0, 1807])
    assert local_targets.dtype == jnp.uint16
    assert tile_offsets.dtype == jnp.int32
    np.testing.assert_array_equal(
        tile_offsets,
        [[0, 2, 4], [0, 0, 0], [0, 1, 2]],
    )


def test_build_tcsr_structure_accepts_rows_larger_than_uint16() -> None:
    """Represent row-relative boundaries beyond uint16 with int32 offsets."""
    source = PlainCSR(
        (
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.zeros((65_536,), dtype=jnp.int32),
            jnp.asarray([0, 65_536], dtype=jnp.int32),
        ),
        shape=(1, 1),
    )

    with jax.enable_x64():
        _, _, tile_offsets = preprocess.build_tcsr_structure(source)

    assert tile_offsets.dtype == jnp.int32
    np.testing.assert_array_equal(tile_offsets, [[0, 65_536]])


def test_build_hybrid_workspace_matches_scheduler_capacity() -> None:
    """Allocate task and status arrays with the scheduler's required layout."""
    indptr = jnp.asarray([0, 2, 2, 140], dtype=jnp.int32)

    workspace = preprocess.build_hybrid_workspace(indptr)

    assert workspace.task_capacity == hybrid_task_capacity(indptr)
    assert workspace.task_begin.shape == (workspace.task_capacity,)
    assert workspace.task_end.shape == (workspace.task_capacity,)
    assert workspace.task_begin.dtype == indptr.dtype
    assert workspace.task_end.dtype == indptr.dtype
    assert workspace.status.shape == (2,)
    assert workspace.status.dtype == jnp.int32


def test_build_tcsc_mirror_maps_mirror_slots_to_canonical_tcsr_slots() -> None:
    """Preserve the data-free mirror permutation direction."""
    indices = jnp.asarray([0, 2, 1, 2], dtype=jnp.int32)
    indptr = jnp.asarray([0, 2, 4], dtype=jnp.int32)

    with jax.enable_x64():
        mirror = preprocess.build_tcsc_mirror(indices, indptr, shape=(2, 3))

    np.testing.assert_array_equal(mirror.indices, [0, 1, 0, 1])
    np.testing.assert_array_equal(mirror.indptr, [0, 1, 2, 4])
    np.testing.assert_array_equal(mirror.permutation, [0, 2, 1, 3])
    np.testing.assert_array_equal(mirror.local_targets, [0, 1, 0, 1])
    assert mirror.local_targets.dtype == jnp.uint16
    assert mirror.tile_offsets.dtype == jnp.int32
    np.testing.assert_array_equal(mirror.tile_offsets, [[0, 1], [0, 1], [0, 2]])
