from contextlib import contextmanager

import jax
import jax.numpy as jnp

from brainevent._csr.main import (
    CSR,
    _BinaryTaskWorkspace,
    _binary_workspace,
    _binary_task_capacity_from_indptr,
    _ensure_binary_workspace,
    _make_binary_task_workspace,
    _with_binary_workspace,
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


def test_binary_workspace_buffer_is_pytree_leaf_and_hidden_buffer():
    csr = CSR(
        (
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0, 1], dtype=jnp.int32),
        ),
        shape=(1, 1),
    )
    workspace = _make_binary_task_workspace(csr.indptr)
    csr = _with_binary_workspace(csr, "csr", workspace)

    leaves, treedef = jax.tree_util.tree_flatten(csr)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    restored_workspace = _binary_workspace(restored, "csr")

    assert "binary_workspace" in csr.buffers
    assert [getattr(leaf, "shape", None) for leaf in leaves] == [
        csr.data.shape,
        workspace.task_begin.shape,
        workspace.task_end.shape,
        workspace.status.shape,
    ]
    assert restored_workspace.task_capacity == workspace.task_capacity
    assert restored_workspace.status.shape == (2,)


def test_ensure_binary_workspace_reuses_existing_workspace():
    csr = CSR(
        (
            jnp.ones((129,), dtype=jnp.float32),
            jnp.zeros((129,), dtype=jnp.int32),
            jnp.array([0, 129], dtype=jnp.int32),
        ),
        shape=(1, 1),
    )

    prepared = _ensure_binary_workspace(csr, "csr", csr.indptr)
    prepared_again = _ensure_binary_workspace(prepared, "csr", csr.indptr)

    first = _binary_workspace(prepared, "csr")
    second = _binary_workspace(prepared_again, "csr")
    assert first.task_capacity == 1
    assert second.task_capacity == first.task_capacity
    assert second.task_begin is first.task_begin


def test_csr_transpose_rekeys_binary_workspace_csc_to_csr():
    csr = CSR(
        (
            jnp.ones((129,), dtype=jnp.float32),
            jnp.zeros((129,), dtype=jnp.int32),
            jnp.array([0, 129], dtype=jnp.int32),
        ),
        shape=(1, 1),
    )
    csc_indptr = jnp.array([0, 129], dtype=jnp.int32)
    csr = _with_binary_workspace(csr, "csc", _make_binary_task_workspace(csc_indptr))

    csc = csr.T

    assert _binary_workspace(csc, "csr").task_capacity == 1


def test_csc_transpose_rekeys_binary_workspace_csr_to_csc():
    csc = CSR(
        (
            jnp.ones((129,), dtype=jnp.float32),
            jnp.zeros((129,), dtype=jnp.int32),
            jnp.array([0, 129], dtype=jnp.int32),
        ),
        shape=(1, 1),
    ).T
    csr_indptr = jnp.array([0, 129], dtype=jnp.int32)
    csc = _with_binary_workspace(csc, "csr", _make_binary_task_workspace(csr_indptr))

    csr = csc.T

    assert _binary_workspace(csr, "csc").task_capacity == 1
