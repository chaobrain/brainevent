# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Select tiled event MM using the compilation platform and public CSR policy."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import numpy as np

from brainevent._csr.main import CSR, _use_cuda_indexed_binary_route
from brainevent._csr.binary.binary import binary_csrmm_p as csr_binary_csrmm_p
from brainevent._csr.binary.binary_indexed import binary_csrmm_indexed_p
from brainevent._event import as_event_operand

from brainevent._tcsr._resources import launch_supported, static_launch_supported
from brainevent._tcsr.binary.binary import binary_csrmm
from brainevent._tcsr.binary.combined.planner import tile_schedule


def dispatch_binary_mm(matrix: CSR, other: Any, *, transpose: bool, plan: Any) -> Any:
    """Execute prepared event MM on CUDA or use the inherited CSR route.

    Parameters
    ----------
    matrix : CSR
        TCSR instance whose canonical structure owns ``plan``.
    other : BinaryArray, CompactBinary, or BitPackedBinary
        Original public operand, in BN layout for ``other @ matrix`` or
        NB layout for ``matrix @ other``. Parent fallbacks retain its type,
        values and units.
    transpose : bool
        Select canonical scatter (True) or mirror traversal (False).
    plan : TileGeometry
        Matching prepared plan, including local targets, tile boundaries and
        actual tile width. The caller validates its structure and dimensions.

    Returns
    -------
    jax.Array or brainunit.Quantity
        Event product with the public CSR dtype and unit convention.

    Notes
    -----
    ``platform_dependent`` selects at lowering time, including when the JIT
    compilation platform differs from the Python default device. Tracing the
    CUDA branch resolves backend names. When every local CUDA device of a
    single process gives the same resource verdict, only that branch is
    traced. Otherwise resource capabilities are checked for the lowering
    device assignment before selecting native execution; unsupported
    capabilities select the corresponding public CSR operation.
    """
    # An explicit backend other than CUDA (such as numba) always requests the
    # ordinary CSR operation; it must not be resolved against the GPU registry
    # while platform_dependent traces its unselected CUDA branch.
    if matrix.backend is not None and matrix.backend != 'cuda_raw':
        return _parent_product(matrix, other, transpose=transpose)
    # The tile arrays are separate operands, so dispatch on a plain CSR view:
    # a TCSR's pytree lists every prepared tile width, and preparing a new
    # width would otherwise retrace every earlier batch size.
    view = CSR._from_parts(
        matrix.data, matrix.indices, matrix.indptr, shape=matrix.shape,
        backend=matrix.backend, config=matrix.config, cache=matrix._cache,
    )
    # Stage the platform choice even for eager calls. A bare
    # platform_dependent eagerly computes its zero-operand platform index on
    # the default device, which can differ from committed operand placement.
    return _dispatch_on_platform(
        view, other, plan.local_targets, plan.tile_offsets,
        transpose=transpose, tile_width=plan.tile_width,
    )


@partial(jax.jit, static_argnames=('transpose', 'tile_width'))
def _dispatch_on_platform(
    matrix: CSR, other: Any, local_targets: Any, tile_offsets: Any, *,
    transpose: bool, tile_width: int,
) -> Any:
    """Lower a cached dispatch program on the actual operand platform."""
    def inherited(operand):
        return _parent_product(matrix, operand, transpose=transpose)

    def cuda(operand):
        primitive = csr_binary_csrmm_p
        if not transpose and _use_cuda_indexed_binary_route(matrix.backend):
            primitive = binary_csrmm_indexed_p
        if primitive._resolve_backend('gpu', matrix.backend) != 'cuda_raw':
            return inherited(operand)
        event = as_event_operand(operand)
        if event is None or event[1] or event[0].ndim != 2:
            return inherited(operand)
        events = event[0].T if transpose else event[0]
        target = matrix.shape[1] if transpose else matrix.shape[0]
        schedule = tile_schedule(target, events.shape[1],
                                 8 if np.dtype(matrix.dtype) == np.dtype('float64') else 4)
        perm = None if transpose else matrix._cache.mirror[2]
        def tiled(original):
            original_events, _ = as_event_operand(original)
            internal_events = original_events.T if transpose else original_events
            output = binary_csrmm(
                matrix.data, matrix.indices, matrix.indptr, internal_events,
                local_targets=local_targets, tile_offsets=tile_offsets,
                tile_width=tile_width, perm=perm, shape=matrix.shape,
                transpose=transpose, backend='cuda_raw', config=matrix.config,
            )
            return output.T if transpose else output
        # Prefer a trace-time decision: lowering only one branch avoids
        # compiling (and planning) the CSR fallback when tiles always fit.
        decision = static_launch_supported(threads=schedule.block_size, shared_bytes=schedule.shared_bytes)
        if decision is not None:
            return tiled(operand) if decision else inherited(operand)
        return jax.lax.cond(
            launch_supported(threads=schedule.block_size, shared_bytes=schedule.shared_bytes),
            tiled, inherited, operand,
        )

    return jax.lax.platform_dependent(other, cuda=cuda, default=inherited)


def _parent_product(matrix: CSR, other: Any, *, transpose: bool) -> Any:
    """Use the original public operand in the matching CSR operator."""
    if transpose:
        return CSR.__rmatmul__(matrix, other)
    return CSR.__matmul__(matrix, other)
