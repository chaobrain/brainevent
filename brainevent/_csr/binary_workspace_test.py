from contextlib import contextmanager

import jax
import jax.numpy as jnp

from brainevent._csr.main import (
    _BinaryTaskWorkspace,
    _binary_task_capacity_from_indptr,
    _make_binary_task_workspace,
)


@contextmanager
def _jax_x64_enabled():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def test_binary_task_capacity_ignores_short_rows():
    indptr = jnp.array([0, 10, 138, 139], dtype=jnp.int32)

    assert _binary_task_capacity_from_indptr(indptr) == 0


def test_binary_task_capacity_counts_heavy_row_chunks():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 129, 129 + 4096, 129 + 4096 + 4097], dtype=jnp.int64)

        assert _binary_task_capacity_from_indptr(indptr) == 1 + 1 + 2


def test_binary_task_capacity_rejects_non_monotonic_indptr():
    indptr = jnp.array([0, 10, 9], dtype=jnp.int32)

    try:
        _binary_task_capacity_from_indptr(indptr)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-monotonic indptr")


def test_make_binary_task_workspace_shapes_dtypes_and_pytree():
    with _jax_x64_enabled():
        indptr = jnp.array([0, 129, 129 + 4097], dtype=jnp.int64)

        workspace = _make_binary_task_workspace(indptr)
        leaves, treedef = jax.tree_util.tree_flatten(workspace)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(workspace, _BinaryTaskWorkspace)
        assert workspace.task_capacity == 3
        assert workspace.task_begin.shape == (3,)
        assert workspace.task_end.shape == (3,)
        assert workspace.status.shape == (2,)
        assert workspace.task_begin.dtype == indptr.dtype
        assert workspace.task_end.dtype == indptr.dtype
        assert workspace.status.dtype == jnp.int32
        assert len(leaves) == 3
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)
        assert restored.task_capacity == workspace.task_capacity
        assert restored.task_begin.shape == workspace.task_begin.shape
        assert restored.task_end.shape == workspace.task_end.shape
        assert restored.status.shape == workspace.status.shape
