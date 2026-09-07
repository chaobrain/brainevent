# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Construct tile-ordered sparse components and traversal workspaces."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from brainevent._csr.main import CSR as PlainCSR
from brainevent._misc import csr_to_csc_index
from brainevent._typing import Data, MatrixShape

from .preprocess_config import hybrid_task_capacity

__all__ = [
    "HybridWorkspace",
    "TCSBuffers",
    "TCSCMirror",
    "TCSRComponents",
    "build_hybrid_workspace",
    "build_tcsr_structure",
    "build_tcsc_mirror",
    "sort_csr",
    "validate_tcsc_mirror",
    "workspace_from_operands",
    "workspace_operands",
]


_TILE_SIZE = 8192
_MAX_ROW_NNZ = np.iinfo(np.int32).max


@dataclass(frozen=True)
class TCSRComponents:
    """Store canonical TCSR values and compressed structure.

    Parameters
    ----------
    data : Data
        Canonical values in stable row-local target order.
    indices : jax.Array
        Int32 target indices.
    indptr : jax.Array
        Int64 row pointers.
    """

    data: Data
    indices: jax.Array
    indptr: jax.Array


def sort_csr(csr: PlainCSR) -> PlainCSR:
    """Sort CSR target indices and align their corresponding values.

    Parameters
    ----------
    csr : brainevent._csr.main.CSR
        Plain CSR matrix to sort in stable row-local target order.

    Returns
    -------
    brainevent._csr.main.CSR
        Plain CSR with sorted indices and correspondingly reordered values.

    Raises
    ------
    TypeError
        If ``csr`` is not a plain CSR matrix.
    RuntimeError
        If heterogeneous data requires a sorting permutation and JAX cannot
        preserve its int64 dtype.

    Notes
    -----
    The sorting permutation maps sorted slots to input slots and remains an
    implementation detail. Row-local sorting leaves ``indptr`` unchanged.
    """
    if not isinstance(csr, PlainCSR):
        raise TypeError("sort_csr requires a plain CSR input")

    indices_np = np.asarray(jax.device_get(csr.indices))
    indptr_np = np.asarray(jax.device_get(csr.indptr))
    nnz = int(indices_np.size)
    sorted_indices: NDArray[np.int32] = np.empty((nnz,), dtype=np.int32)
    permutation: NDArray[np.int64] | None = None
    if int(csr.data.size) != 1:
        permutation = np.empty((nnz,), dtype=np.int64)

    for row in range(int(csr.shape[0])):
        begin = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        order = np.argsort(indices_np[begin:end], kind="stable")
        sorted_indices[begin:end] = indices_np[begin:end][order]
        if permutation is not None:
            permutation[begin:end] = begin + order

    if permutation is None:
        sorted_data = jax.device_put(csr.data)
    else:
        device_permutation = jnp.asarray(permutation, dtype=jnp.int64)
        if device_permutation.dtype != jnp.int64:
            raise RuntimeError(
                "TCSR sorting requires an explicit int64 permutation; set "
                "jax_explicit_x64_dtypes='allow' or enable JAX x64."
            )
        sorted_data = jax.device_put(csr.data[device_permutation])
    return PlainCSR._from_parts(
        sorted_data,
        jnp.asarray(sorted_indices, dtype=jnp.int32),
        csr.indptr,
        shape=csr.shape,
        backend=csr.backend,
    )


def build_tcsr_structure(
    csr: PlainCSR,
) -> tuple[TCSRComponents, jax.Array, jax.Array]:
    """Build canonical TCSR arrays and tile traversal metadata.

    Parameters
    ----------
    csr : brainevent._csr.main.CSR
        Plain CSR whose target indices are already sorted within each row.

    Returns
    -------
    components : TCSRComponents
        Canonical values, int32 indices, and int64 row pointers.
    local_targets : jax.Array
        Uint16 target offsets within each 8192-column tile.
    tile_offsets : jax.Array
        Int32 row-local entry boundaries for every output tile.

    Raises
    ------
    TypeError
        If ``csr`` is not a plain CSR matrix.
    ValueError
        If a row exceeds the int32 tile-offset capacity.
    RuntimeError
        If JAX cannot preserve the required explicit int64 row pointers.

    Notes
    -----
    This function assumes sorted row-local indices and never sorts its input.
    """
    if not isinstance(csr, PlainCSR):
        raise TypeError("build_tcsr_structure requires a plain CSR input")

    rows, cols = int(csr.shape[0]), int(csr.shape[1])
    indices_np = np.asarray(jax.device_get(csr.indices))
    row_ptr = np.ascontiguousarray(jax.device_get(csr.indptr), dtype=np.int64)
    if np.any(np.diff(row_ptr) > _MAX_ROW_NNZ):
        raise ValueError(
            "TCSR tile offsets require each CSR row to fit int32"
        )

    tile_count = (cols + _TILE_SIZE - 1) // _TILE_SIZE
    tile_offsets: NDArray[np.int32] = np.empty(
        (rows, tile_count + 1), dtype=np.int32
    )
    for row in range(rows):
        begin = int(row_ptr[row])
        end = int(row_ptr[row + 1])
        row_targets = indices_np[begin:end]
        for tile in range(tile_count + 1):
            boundary = min(tile * _TILE_SIZE, cols)
            tile_offsets[row, tile] = np.searchsorted(
                row_targets,
                boundary,
                side="left",
            )

    device_indptr = jnp.asarray(row_ptr, dtype=jnp.int64)
    if device_indptr.dtype != jnp.int64:
        raise RuntimeError(
            "TCSR requires explicit int64 row pointers; set "
            "jax_explicit_x64_dtypes='allow' or enable JAX x64."
        )
    components = TCSRComponents(
        data=jax.device_put(csr.data),
        indices=jnp.asarray(csr.indices, dtype=jnp.int32),
        indptr=device_indptr,
    )
    local_targets = np.asarray(
        indices_np & (_TILE_SIZE - 1),
        dtype=np.uint16,
    )
    return (
        components,
        jnp.asarray(local_targets, dtype=jnp.uint16),
        jnp.asarray(tile_offsets, dtype=jnp.int32),
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HybridWorkspace:
    """Store task arrays for one hybrid sparse traversal.

    Parameters
    ----------
    task_capacity : int
        Maximum number of tasks stored in the workspace.
    task_begin : jax.Array
        Per-task inclusive sparse offsets.
    task_end : jax.Array
        Per-task exclusive sparse offsets.
    status : jax.Array
        Int32 runtime status array with shape ``(2,)``.
    """

    task_capacity: int
    task_begin: jax.Array
    task_end: jax.Array
    status: jax.Array

    def tree_flatten(self):
        """Flatten dynamic workspace arrays and static capacity metadata."""
        return (
            self.task_begin,
            self.task_end,
            self.status,
        ), (self.task_capacity,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore a workspace from flattened pytree state."""
        task_begin, task_end, status = children
        return cls(int(aux_data[0]), task_begin, task_end, status)

    def block_until_ready(self):
        """Block until all workspace arrays are ready and return this object."""
        jax.block_until_ready((self.task_begin, self.task_end, self.status))
        return self


def workspace_from_operands(
    task_capacity: int,
    task_begin: jax.Array,
    task_end: jax.Array,
    status: jax.Array,
) -> HybridWorkspace:
    """Restore a hybrid workspace from primitive operands.

    Parameters
    ----------
    task_capacity : int
        Static task capacity.
    task_begin : jax.Array
        Per-task inclusive offsets.
    task_end : jax.Array
        Per-task exclusive offsets.
    status : jax.Array
        Runtime status array.

    Returns
    -------
    HybridWorkspace
        Reconstructed workspace object.
    """
    return HybridWorkspace(int(task_capacity), task_begin, task_end, status)


def workspace_operands(
    workspace: HybridWorkspace | None,
    indptr: Any,
) -> tuple[int, jax.Array, jax.Array, jax.Array]:
    """Validate and flatten a workspace for a primitive call.

    Parameters
    ----------
    workspace : HybridWorkspace or None
        Workspace associated with the traversed sparse structure.
    indptr : array-like
        Sparse row pointers whose dtype determines the task offset dtype.

    Returns
    -------
    task_capacity : int
        Static task capacity.
    task_begin : jax.Array
        Per-task inclusive offsets.
    task_end : jax.Array
        Per-task exclusive offsets.
    status : jax.Array
        Runtime status array.

    Raises
    ------
    ValueError
        If workspace arrays have invalid shapes or capacity.
    TypeError
        If workspace arrays have invalid dtypes.
    """
    if workspace is None:
        raise ValueError("hybrid workspace is required")
    if not isinstance(workspace, HybridWorkspace):
        raise TypeError("workspace must be a HybridWorkspace")

    capacity = int(workspace.task_capacity)
    if capacity < 0:
        raise ValueError("task_capacity must be non-negative")
    if tuple(workspace.task_begin.shape) != (capacity,):
        raise ValueError("task_begin shape must match task_capacity")
    if tuple(workspace.task_end.shape) != (capacity,):
        raise ValueError("task_end shape must match task_capacity")
    if tuple(workspace.status.shape) != (2,):
        raise ValueError("workspace status shape must be (2,)")

    offset_dtype = jnp.dtype(indptr.dtype)
    if (
        jnp.dtype(workspace.task_begin.dtype) != offset_dtype
        or jnp.dtype(workspace.task_end.dtype) != offset_dtype
    ):
        raise TypeError("workspace task offsets must use the same dtype as indptr")
    if jnp.dtype(workspace.status.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("workspace status must use int32")

    return capacity, workspace.task_begin, workspace.task_end, workspace.status


def build_hybrid_workspace(indptr: Any) -> HybridWorkspace:
    """Build a hybrid workspace for sparse row pointers.

    Parameters
    ----------
    indptr : array-like
        Sparse row pointers used to determine task capacity and offset dtype.

    Returns
    -------
    HybridWorkspace
        Newly allocated uninitialized task arrays and status storage.
    """
    capacity = hybrid_task_capacity(indptr)
    offset_dtype = jnp.dtype(indptr.dtype)
    return HybridWorkspace(
        task_capacity=capacity,
        task_begin=jnp.empty((capacity,), dtype=offset_dtype),
        task_end=jnp.empty((capacity,), dtype=offset_dtype),
        status=jnp.empty((2,), dtype=jnp.int32),
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TCSCMirror:
    """Store the data-free TCSC mirror and its traversal buffers.

    Parameters
    ----------
    indices : jax.Array
        Int32 row indices in mirror order.
    indptr : jax.Array
        Int64 column pointers represented as CSR pointers of the transpose.
    permutation : jax.Array
        Int64 mapping from mirror slots to canonical TCSR value slots.
    local_targets : jax.Array
        Uint16 target offsets within each tile.
    tile_offsets : jax.Array
        Int32 row-local entry boundaries for every output tile.
    workspace : HybridWorkspace
        Hybrid traversal workspace for the mirror structure.
    """

    indices: jax.Array
    indptr: jax.Array
    permutation: jax.Array
    local_targets: jax.Array
    tile_offsets: jax.Array
    workspace: HybridWorkspace

    def tree_flatten(self):
        """Flatten every dynamic mirror member for the JAX PyTree protocol."""
        return (
            self.indices,
            self.indptr,
            self.permutation,
            self.local_targets,
            self.tile_offsets,
            self.workspace,
        ), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore a TCSC mirror from flattened PyTree state."""
        del aux_data
        return cls(*children)


@dataclass
class TCSBuffers:
    """Store direct traversal buffers and an optional TCSC mirror.

    Parameters
    ----------
    tcsr_local_targets : jax.Array
        Uint16 direct TCSR target offsets within each tile.
    tcsr_tile_offsets : jax.Array
        Int32 direct TCSR tile boundaries for every row.
    tcsr_workspace : HybridWorkspace
        Hybrid traversal workspace for the direct TCSR structure.
    tcsc : TCSCMirror, optional
        Lazily materialized data-free TCSC mirror.
    """

    tcsr_local_targets: jax.Array
    tcsr_tile_offsets: jax.Array
    tcsr_workspace: HybridWorkspace
    tcsc: TCSCMirror | None = None

    @property
    def has_tcsc(self) -> bool:
        """Return whether the optional TCSC mirror has been materialized."""
        return self.tcsc is not None


def validate_tcsc_mirror(
    mirror: TCSCMirror,
    *,
    shape: MatrixShape,
    nnz: int,
) -> TCSCMirror:
    """Validate a TCSC mirror against its canonical TCSR dimensions.

    Parameters
    ----------
    mirror : TCSCMirror
        Mirror structure and traversal buffers to validate.
    shape : tuple of int
        Canonical TCSR matrix shape ``(num_pre, num_post)``.
    nnz : int
        Number of canonical structural entries.

    Returns
    -------
    TCSCMirror
        The validated mirror object.

    Raises
    ------
    TypeError
        If mirror arrays use unsupported kernel dtypes.
    ValueError
        If mirror shapes, pointers, indices, permutation, or workspace are
        inconsistent with the canonical structure.
    """
    rows, cols = int(shape[0]), int(shape[1])
    nnz = int(nnz)
    if rows <= 0 or cols <= 0:
        raise ValueError("TCSC mirror shape dimensions must be positive")
    if nnz < 0:
        raise ValueError("TCSC mirror nnz must be non-negative")

    indices = np.asarray(jax.device_get(mirror.indices))
    indptr = np.asarray(jax.device_get(mirror.indptr))
    permutation = np.asarray(jax.device_get(mirror.permutation))
    local_targets = np.asarray(jax.device_get(mirror.local_targets))
    tile_offsets = np.asarray(jax.device_get(mirror.tile_offsets))

    if indices.dtype != np.dtype(np.int32):
        raise TypeError("TCSC mirror indices must use int32")
    if indptr.dtype != np.dtype(np.int64):
        raise TypeError("TCSC mirror indptr must use int64")
    if permutation.dtype != np.dtype(np.int64):
        raise TypeError("TCSC mirror permutation must use int64")
    if local_targets.dtype != np.dtype(np.uint16):
        raise TypeError("TCSC mirror local_targets must use uint16")
    if tile_offsets.dtype != np.dtype(np.int32):
        raise TypeError("TCSC mirror tile_offsets must use int32")

    if indices.shape != (nnz,):
        raise ValueError("TCSC mirror indices shape must equal nnz")
    if indptr.shape != (cols + 1,):
        raise ValueError("TCSC mirror indptr length must equal shape[1] + 1")
    if permutation.shape != (nnz,):
        raise ValueError("TCSC mirror permutation shape must equal nnz")
    if local_targets.shape != (nnz,):
        raise ValueError("TCSC mirror local_targets shape must equal nnz")
    tile_count = (rows + _TILE_SIZE - 1) // _TILE_SIZE
    if tile_offsets.shape != (cols, tile_count + 1):
        raise ValueError("TCSC mirror tile_offsets shape is inconsistent")

    if indptr[0] != 0 or indptr[-1] != nnz or np.any(np.diff(indptr) < 0):
        raise ValueError(
            "TCSC mirror indptr must be monotonic, start at zero, and end at nnz"
        )
    if np.any(indices < 0) or np.any(indices >= rows):
        raise ValueError("TCSC mirror index is outside [0, shape[0])")
    if nnz and (
        np.any(permutation < 0)
        or np.any(permutation >= nnz)
        or np.unique(permutation).size != nnz
    ):
        raise ValueError("TCSC mirror permutation must contain each slot once")
    expected_local_targets = np.asarray(
        indices & (_TILE_SIZE - 1),
        dtype=np.uint16,
    )
    if not np.array_equal(local_targets, expected_local_targets):
        raise ValueError("TCSC mirror local_targets do not match indices")

    expected_tile_offsets = np.empty_like(tile_offsets)
    for row in range(cols):
        begin = int(indptr[row])
        end = int(indptr[row + 1])
        row_targets = indices[begin:end]
        if np.any(np.diff(row_targets) < 0):
            raise ValueError("TCSC mirror indices must be sorted within each row")
        for tile in range(tile_count + 1):
            boundary = min(tile * _TILE_SIZE, rows)
            expected_tile_offsets[row, tile] = np.searchsorted(
                row_targets,
                boundary,
                side="left",
            )
    if not np.array_equal(tile_offsets, expected_tile_offsets):
        raise ValueError("TCSC mirror tile_offsets do not match indices")
    workspace_operands(mirror.workspace, mirror.indptr)
    return mirror


def build_tcsc_mirror(
    tcsr_indices: jax.Array,
    tcsr_indptr: jax.Array,
    *,
    shape: MatrixShape,
) -> TCSCMirror:
    """Build and validate a data-free TCSC mirror from TCSR structure.

    Parameters
    ----------
    tcsr_indices : jax.Array
        Canonical TCSR int32 target indices.
    tcsr_indptr : jax.Array
        Canonical TCSR row pointers.
    shape : tuple of int
        Canonical TCSR matrix shape ``(num_pre, num_post)``.

    Returns
    -------
    TCSCMirror
        Validated mirror structure, permutation, and traversal buffers.

    Raises
    ------
    RuntimeError
        If the CSR-to-CSC conversion does not produce a permutation.
    """
    with jax.enable_x64():
        tcsc_indptr, tcsc_indices, tcsc_permutation = csr_to_csc_index(
            tcsr_indptr,
            tcsr_indices,
            shape=shape,
            include_perm=True,
            method="numpy",
        )
        if tcsc_permutation is None:
            raise RuntimeError("TCSC mirror construction requires a permutation")
        mirror_csr = PlainCSR(
            (
                jnp.asarray(tcsc_permutation, dtype=jnp.int64),
                tcsc_indices,
                tcsc_indptr,
            ),
            shape=shape[::-1],
            check_structure=False,
        )
        sorted_mirror_csr = sort_csr(mirror_csr)
        components, local_targets, tile_offsets = build_tcsr_structure(
            sorted_mirror_csr
        )
        mirror = TCSCMirror(
            indices=components.indices,
            indptr=components.indptr,
            permutation=jnp.asarray(components.data, dtype=jnp.int64),
            local_targets=local_targets,
            tile_offsets=tile_offsets,
            workspace=build_hybrid_workspace(components.indptr),
        )
        return validate_tcsc_mirror(
            mirror,
            shape=shape,
            nnz=int(tcsr_indices.size),
        )
