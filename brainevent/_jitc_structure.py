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

"""Connectivity of a JITC matrix, drawn once and reused under ``jit``.

The connectivity of a JITC matrix depends only on ``(shape, prob, seed,
corder)``, never on the weights. :func:`build_jitc_structure` draws it at
trace time (under :func:`jax.ensure_compile_time_eval`), so the number of
stored entries is a static shape. The weight fill then runs inside the graph
with possibly traced weights, and :class:`JITCStructure.order` puts its
output in canonical (column-sorted) CSR order.

Both ``corder`` values fill with the deterministic row-stream
(``corder=True``) kernels: ``corder=False`` fills the transposed shape, which
draws the same matrix bit for bit. The ``corder=False`` kernels scatter with
atomics, so their order within a row is not reproducible on GPU and could not
be paired with a precomputed ``order``.
"""

from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._misc import _as_int32_cuda_offsets, _require_jax_x64_for_int64, _resolve_indptr_dtype
from brainevent._typing import MatrixShape

__all__ = [
    'JITCStructure',
    'build_jitc_structure',
    'check_jitc_structure',
    'concrete_seed',
    'fill_shape',
]

#: ``count(clen, seed, shape)`` returns the ``(n_rows, n_chunks)`` per-chunk
#: counts of the row-stream walk over ``shape``.
CountFn = Callable[[jax.Array, jax.Array, Tuple[int, int]], jax.Array]
#: ``fill(clen, seed, offsets, nnz, shape)`` returns the ``(indices, data)`` of
#: the row-stream walk over ``shape``, in fill order.
FillFn = Callable[[jax.Array, jax.Array, jax.Array, int, Tuple[int, int]], Tuple[jax.Array, jax.Array]]


class JITCStructure(NamedTuple):
    """Connectivity of a JITC matrix in canonical CSR form.

    Attributes
    ----------
    fill_offsets : jax.Array
        ``int32`` array of shape ``(n_fill_rows, n_chunks)`` with the exclusive
        write offset of every ``(row, chunk)`` of the row-stream fill over
        :func:`fill_shape`.
    indptr : jax.Array
        Row pointer of the canonical CSR, shape ``(n_rows + 1,)``.
    indices : jax.Array
        ``int32`` column indices of the canonical CSR (sorted within each row),
        shape ``(nnz,)``.
    order : jax.Array
        Position in the fill output of every canonical CSR entry, shape
        ``(nnz,)``: ``csr_data = fill_data[order]``.
    """
    fill_offsets: jax.Array
    indptr: jax.Array
    indices: jax.Array
    order: jax.Array

    @property
    def nnz(self) -> int:
        """Number of stored entries (static: the length of :attr:`order`)."""
        return int(self.order.shape[0])


def fill_shape(shape: MatrixShape, corder: bool) -> Tuple[int, int]:
    """Shape walked by the row-stream fill that draws a ``(shape, corder)`` matrix.

    Parameters
    ----------
    shape : tuple of int
        Logical ``(n_rows, n_cols)`` of the matrix.
    corder : bool
        Stream ownership of the matrix.

    Returns
    -------
    tuple of int
        ``shape`` for ``corder=True``; the transposed shape for ``corder=False``,
        whose row streams are the matrix's column streams.
    """
    n_rows, n_cols = int(shape[0]), int(shape[1])
    return (n_rows, n_cols) if corder else (n_cols, n_rows)


def concrete_seed(seed):
    """Replace a ``None`` seed by a concrete random one, drawn on the host.

    Parameters
    ----------
    seed : int, array_like or None
        Seed given by the caller.

    Returns
    -------
    int, array_like
        ``seed`` itself, or a ``(1,)`` NumPy array when ``seed`` is ``None``.

    Notes
    -----
    The structure pass and the in-graph weight fill must use the same seed, and
    any ``jnp`` operation inside a trace returns a tracer. The seed is
    therefore resolved here, before either pass, and stays a host value.
    """
    if seed is None:
        return np.random.randint(0, int(1e8), (1,))
    return seed


def _traced_connectivity_error(what: str) -> ValueError:
    return ValueError(
        f"{what} needs a concrete `prob` and `seed` to draw the connectivity, but they "
        "are traced here (the JITC matrix was passed into `jax.jit` or `jax.vmap` as an "
        "argument). Call `M = M.prepare_structure()` before `jax.jit` and pass that "
        "matrix, or close over the matrix instead of passing it as an argument."
    )


def build_jitc_structure(
    count: CountFn,
    fill: FillFn,
    prob,
    seed,
    *,
    shape: MatrixShape,
    corder: bool,
    what: str,
) -> Tuple[JITCStructure, jax.Array]:
    """Draw the connectivity of a JITC matrix as concrete arrays.

    Runs eagerly even inside a trace (``jax.ensure_compile_time_eval``), so it
    needs a concrete ``prob`` and ``seed``.

    Parameters
    ----------
    count : callable
        Count pass of the family, bound to a placeholder weight.
    fill : callable
        Fill pass of the family, bound to concrete weights (a placeholder when
        the real weights are traced).
    prob : float or array_like
        Connection probability (concrete).
    seed : int or array_like
        Random seed (concrete).
    shape : tuple of int
        Logical ``(n_rows, n_cols)`` of the matrix.
    corder : bool
        Stream ownership of the matrix.
    what : str
        Name of the calling operation, for the error message.

    Returns
    -------
    structure : JITCStructure
        The canonical CSR connectivity and the fill-to-CSR order.
    data : jax.Array
        The weights drawn by ``fill``, in canonical CSR order.

    Raises
    ------
    ValueError
        If ``prob`` or ``seed`` is traced.
    """
    if any(isinstance(x, Tracer) for x in jax.tree.leaves((prob, seed))):
        raise _traced_connectivity_error(what)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    fshape = fill_shape(shape, corder)
    with jax.ensure_compile_time_eval():
        clen = _initialize_conn_length(prob)
        seed = _initialize_seed(seed)
        chunk_counts = np.asarray(jax.device_get(count(clen, seed, fshape)), dtype=np.int64)
        fill_row_counts = chunk_counts.sum(axis=1)
        nnz = int(fill_row_counts.sum())
        fill_starts = np.concatenate([[0], np.cumsum(fill_row_counts)])[:-1]
        fill_offsets = fill_starts[:, None] + np.cumsum(chunk_counts, axis=1) - chunk_counts
        offset_dtype = _resolve_indptr_dtype(nnz, "auto")
        _require_jax_x64_for_int64(offset_dtype, "JITC structure")
        fill_offsets = _as_int32_cuda_offsets(
            jnp.asarray(fill_offsets, dtype=offset_dtype), "JITC fill offsets"
        )
        walked, walked_data = fill(clen, seed, fill_offsets, nnz, fshape)
        walked = np.asarray(jax.device_get(walked))
        fill_rows = np.repeat(np.arange(fshape[0], dtype=np.int64), fill_row_counts)
        rows, cols = (fill_rows, walked) if corder else (walked, fill_rows)
        order = np.lexsort((cols, rows))
        indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_rows))])
        structure = JITCStructure(
            fill_offsets=fill_offsets,
            indptr=jnp.asarray(indptr, dtype=offset_dtype),
            indices=jnp.asarray(cols[order], dtype=jnp.int32),
            order=jnp.asarray(order, dtype=offset_dtype),
        )
        return structure, walked_data[structure.order]


def check_jitc_structure(structure: Optional[JITCStructure], *, shape: MatrixShape, corder: bool) -> None:
    """Raise ``ValueError`` if ``structure`` cannot belong to a ``(shape, corder)`` matrix.

    Parameters
    ----------
    structure : JITCStructure or None
        Structure to check; ``None`` passes.
    shape : tuple of int
        Logical ``(n_rows, n_cols)`` of the matrix.
    corder : bool
        Stream ownership of the matrix.

    Raises
    ------
    ValueError
        If the array shapes disagree with ``shape`` / ``corder``.
    """
    if structure is None:
        return
    n_rows = int(shape[0])
    if structure.indptr.shape != (n_rows + 1,):
        raise ValueError(f"structure indptr has shape {structure.indptr.shape}, expected ({n_rows + 1},).")
    if structure.fill_offsets.ndim != 2 or structure.fill_offsets.shape[0] != fill_shape(shape, corder)[0]:
        raise ValueError(
            f"structure fill_offsets has shape {structure.fill_offsets.shape}, which does not match "
            f"shape={tuple(shape)} with corder={corder}."
        )
    if structure.indices.shape != structure.order.shape:
        raise ValueError("structure indices and order must have the same shape.")
