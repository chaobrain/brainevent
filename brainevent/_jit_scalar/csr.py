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

# -*- coding: utf-8 -*-

"""
Materialize a scalar-weight just-in-time connectivity (JITC) matrix directly
into Compressed Sparse Row (CSR) format using the light-RNG chunk kernels.

``jits_to_csr`` reproduces exactly the matrix used by ``jits``, ``jitsmv``,
``jitsmm`` and the event kernels -- they all draw the same 32-lane matrix.
``corder`` keeps its usual meaning (it selects the notrans/trans generation,
exactly as in ``jitsmv``); the matrix class flips it on transpose.

Generation is split into two passes (``nnz`` is data dependent, XLA needs static shapes):

1. count : per-(row, chunk) counts of the row-stream walk
2. fill  : column indices + values, given the offsets derived from the counts

The connectivity depends only on ``(shape, prob, seed, corder)``; it is drawn
once at trace time (see :mod:`brainevent._jitc_structure`), and only the weight
fill runs in the compiled program, so ``to_csr`` works under ``jax.jit``.
``corder=False`` fills the transposed shape with the row-stream kernel.
"""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._jitc_structure import (
    JITCStructure, build_jitc_structure, check_jitc_structure, concrete_seed, fill_shape,
)
from brainevent._misc import _is_static_zero, _n_chunks
from brainevent._numba_random import get_numba_light_rng_funcs
from brainevent._op import XLACustomKernel, load_cuda_file, numba_kernel
from brainevent._typing import MatrixShape
from .float import _chunk_size, _walk_length, _LANE_STRIDE
from brainevent._op.util import dtype_suffix

__all__ = [
    'jits_to_csr',
    'jits_csr_count_p',
    'jits_csr_count_p_call',
    'jits_csr_fill_p',
    'jits_csr_fill_p_call',
]

_NUMBA_UNSUPPORTED_DTYPES = frozenset({
    np.dtype('float16'),
    np.dtype('bfloat16'),
})


# ──────────────────────────────────────────────────────────────────────
#  Count pass
# ──────────────────────────────────────────────────────────────────────

def _jits_csr_count_cuda_kernel(
    shape: MatrixShape,
    corder: bool,
    **kwargs,
):
    load_cuda_file(Path(__file__).parent.joinpath('csr.cu'), name='jit_scalar_csr')
    sfx = dtype_suffix(kwargs['weight_info'].dtype)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    walk = _walk_length(shape, False, corder)
    chunk_size_value = _chunk_size(walk)

    if corder:
        # notrans: chunk the mat columns; per-(row, chunk) counts.
        kernel_name = f'jit_scalar_csr.count_chunks_notrans{sfx}'

        def kernel(weight, clen, seed):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                weight, clen, seed,
                n_cols=np.int32(n_cols), chunk_size=np.int32(chunk_size_value),
            )
    else:
        # trans: seed over mat columns, walk mat rows, atomic per-row counts.
        kernel_name = f'jit_scalar_csr.count_chunks_trans{sfx}'

        def kernel(weight, clen, seed):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                weight, clen, seed,
                n_rows=np.int32(n_cols), n_cols=np.int32(n_rows),
                chunk_size=np.int32(chunk_size_value),
            )

    return kernel


def _jits_csr_count_numba_kernel(
    shape: MatrixShape,
    corder: bool,
    **kwargs,
):
    """Numba CPU count pass mirroring ``csr.cu`` (bit-identical connectivity).

    ``corder=True`` writes per-(row, chunk) counts (notrans); ``corder=False``
    writes per-output-row counts (trans). The light-RNG stride is 32,
    matching the CUDA kernels.
    """
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_keep = _rng['keep']

    stride = _LANE_STRIDE
    n_cols = int(shape[1])
    cs_val = _chunk_size(_walk_length(shape, False, corder))

    if corder:
        # notrans: walk the matrix columns (k = n_cols); count per (row, chunk).
        k = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, chunk_counts):
            m = chunk_counts.shape[0]
            n_chunks = chunk_counts.shape[1]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            thr = np.uint32(clen[1])
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        chunk_counts[row, chunk_id] = 0
                        continue
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    cnt = 0
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            if _rng_keep(thr, seed0, row, chunk_start + local_j):
                                cnt += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                    chunk_counts[row, chunk_id] = cnt
    else:
        # trans: walk source rows m = n_cols, walk dim k = n_rows (output rows).
        m_walk = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, row_counts):
            row_counts[:] = 0
            k = row_counts.shape[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            thr = np.uint32(clen[1])
            n_chunks = (k + cs_val - 1) // cs_val
            for row in range(m_walk):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            if _rng_keep(thr, seed0, row, chunk_start + local_j):
                                row_counts[chunk_start + local_j] += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(weight, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, seed)

    return kernel


def jits_csr_count_p_call(
    weight,
    clen,
    seed,
    *,
    shape: MatrixShape,
    corder: bool,
    backend: Optional[str] = None,
):
    """Count pass. Returns ``chunk_counts`` (n_rows, n_chunks) for ``corder=True``,
    or ``row_counts`` (n_rows,) for ``corder=False``."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    walk = _walk_length(shape, False, corder)
    chunk_size_value = _chunk_size(walk)
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    assert clen.shape == (2,), f"clen must hold (cl, thr), but got shape {clen.shape}."
    seed = jnp.atleast_1d(seed)
    # Counting only depends on shape/probability/seed; cast low-precision
    # scalar weights so Numba does not reject unsupported array arguments.
    count_weight = (
        weight.astype(jnp.float32)
        if np.dtype(weight.dtype) in _NUMBA_UNSUPPORTED_DTYPES
        else weight
    )

    if corder:
        n_chunks = _n_chunks(walk, chunk_size_value)
        if n_rows == 0 or n_chunks == 0:
            return (jnp.zeros((n_rows, n_chunks), dtype=jnp.int32),)
        out_info = jax.ShapeDtypeStruct((n_rows, n_chunks), jnp.int32)
    else:
        if n_rows == 0 or n_cols == 0:
            return (jnp.zeros((n_rows,), dtype=jnp.int32),)
        out_info = jax.ShapeDtypeStruct((n_rows,), jnp.int32)

    return jits_csr_count_p(
        count_weight, clen, seed,
        outs=[out_info],
        shape=(n_rows, n_cols),
        corder=corder,
        backend=backend,
        weight_info=jax.ShapeDtypeStruct(count_weight.shape, count_weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
    )


jits_csr_count_p = XLACustomKernel(
    'jits_csr_count',
    doc="Low-level CUDA primitive counting non-zeros for light scalar CSR materialization.",
)
jits_csr_count_p.def_cuda_raw_kernel(_jits_csr_count_cuda_kernel, asdefault=True)
jits_csr_count_p.def_numba_kernel(_jits_csr_count_numba_kernel)
jits_csr_count_p.def_call(jits_csr_count_p_call)
jits_csr_count_p.def_tags('jit_scalar', 'csr', 'light_rng')


# ──────────────────────────────────────────────────────────────────────
#  Fill pass
# ──────────────────────────────────────────────────────────────────────

def _jits_csr_fill_cuda_kernel(
    shape: MatrixShape,
    corder: bool,
    **kwargs,
):
    load_cuda_file(Path(__file__).parent.joinpath('csr.cu'), name='jit_scalar_csr')
    sfx = dtype_suffix(kwargs['weight_info'].dtype)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    walk = _walk_length(shape, False, corder)
    chunk_size_value = _chunk_size(walk)

    if corder:
        kernel_name = f'jit_scalar_csr.fill_notrans{sfx}'

        def kernel(weight, clen, seed, offsets):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                weight, clen, seed, offsets,
                n_cols=np.int32(n_cols), chunk_size=np.int32(chunk_size_value),
            )
    else:
        kernel_name = f'jit_scalar_csr.fill_trans{sfx}'

        def kernel(weight, clen, seed, offsets):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                weight, clen, seed, offsets,
                n_rows=np.int32(n_cols), n_cols=np.int32(n_rows),
                chunk_size=np.int32(chunk_size_value),
            )

    return kernel


def _jits_csr_fill_numba_kernel(
    shape: MatrixShape,
    corder: bool,
    **kwargs,
):
    """Numba CPU fill pass mirroring ``csr.cu``.

    ``corder=True`` (notrans): ``offsets`` are the per-(row, chunk) exclusive
    offsets; entries are written contiguously per (row, chunk). ``corder=False``
    (trans): ``offsets`` is the ``indptr`` and an int32 ``cursor`` (one per output
    row) tracks write positions. Both are re-sorted to canonical column order by
    the caller, so the within-row write order need not match the CUDA lane order.
    """
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_keep = _rng['keep']

    stride = _LANE_STRIDE
    n_cols = int(shape[1])
    cs_val = _chunk_size(_walk_length(shape, False, corder))

    if corder:
        # notrans: walk the matrix columns (k = n_cols); write per (row, chunk).
        k = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, offsets, indices, data):
            m = offsets.shape[0]
            n_chunks = offsets.shape[1]
            w = weight[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            thr = np.uint32(clen[1])
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        continue
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    pos = offsets[row, chunk_id]
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            if _rng_keep(thr, seed0, row, chunk_start + local_j):
                                indices[pos] = chunk_start + local_j
                                data[pos] = w
                                pos += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
    else:
        # trans: walk source rows m = n_cols, walk dim k = n_rows (output rows).
        m_walk = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, offsets, indices, data, cursor):
            cursor[:] = 0
            k = cursor.shape[0]
            w = weight[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            thr = np.uint32(clen[1])
            n_chunks = (k + cs_val - 1) // cs_val
            for row in range(m_walk):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            if _rng_keep(thr, seed0, row, chunk_start + local_j):
                                j = chunk_start + local_j
                                pos = offsets[j] + cursor[j]
                                cursor[j] += 1
                                indices[pos] = row
                                data[pos] = w
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(weight, clen, seed, offsets):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, seed, offsets)

    return kernel


def jits_csr_fill_p_call(
    weight,
    clen,
    seed,
    offsets,
    nnz: int,
    *,
    shape: MatrixShape,
    corder: bool,
    backend: Optional[str] = None,
):
    """Fill pass. ``offsets`` are the per-(row, chunk) exclusive offsets
    (``corder=True``, shape ``(n_rows, n_chunks)``) or the ``indptr``
    (``corder=False``, shape ``(n_rows + 1,)``). Returns ``(indices, data)``."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    walk = _walk_length(shape, False, corder)
    chunk_size_value = _chunk_size(walk)
    weight = jnp.atleast_1d(weight)
    out_dtype = weight.dtype
    kernel_weight = weight
    if np.dtype(out_dtype) in _NUMBA_UNSUPPORTED_DTYPES:
        # Numba cannot lower f16/bf16 arrays; the scalar fill value is constant,
        # so compute it through float32 and restore the public dtype.
        kernel_weight = weight.astype(jnp.float32)
    clen = jnp.atleast_1d(clen)
    assert clen.shape == (2,), f"clen must hold (cl, thr), but got shape {clen.shape}."
    seed = jnp.atleast_1d(seed)
    offsets = jnp.asarray(offsets, dtype=jnp.int32)
    nnz = int(nnz)
    if nnz == 0:
        return (jnp.zeros((0,), dtype=jnp.int32), jnp.zeros((0,), dtype=out_dtype))

    outs = [
        jax.ShapeDtypeStruct((nnz,), jnp.int32),
        jax.ShapeDtypeStruct((nnz,), kernel_weight.dtype),
    ]
    if not corder:
        # trans fill also needs an int32 write cursor (one per output row).
        outs.append(jax.ShapeDtypeStruct((n_rows,), jnp.int32))

    res = jits_csr_fill_p(
        kernel_weight, clen, seed, offsets,
        outs=outs,
        shape=(n_rows, n_cols),
        corder=corder,
        backend=backend,
        weight_info=jax.ShapeDtypeStruct(kernel_weight.shape, kernel_weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        offsets_info=jax.ShapeDtypeStruct(offsets.shape, offsets.dtype),
    )
    data = res[1].astype(out_dtype) if res[1].dtype != out_dtype else res[1]
    return res[0], data


jits_csr_fill_p = XLACustomKernel(
    'jits_csr_fill',
    doc="Low-level CUDA primitive filling CSR indices/data for light scalar CSR.",
)
jits_csr_fill_p.def_cuda_raw_kernel(_jits_csr_fill_cuda_kernel, asdefault=True)
jits_csr_fill_p.def_numba_kernel(_jits_csr_fill_numba_kernel)
jits_csr_fill_p.def_call(jits_csr_fill_p_call)
jits_csr_fill_p.def_tags('jit_scalar', 'csr', 'light_rng')


# ──────────────────────────────────────────────────────────────────────
#  High-level orchestration
# ──────────────────────────────────────────────────────────────────────

def jits_structure(prob, seed, *, shape: MatrixShape, corder: bool, backend: Optional[str] = None):
    """Draw the connectivity of a JIT scalar matrix (see :func:`~brainevent._jitc_structure.build_jitc_structure`).

    Parameters
    ----------
    prob : float or array_like
        Connection probability (concrete).
    seed : int or array_like
        Random seed (concrete).
    shape : tuple of int
        Logical ``(n_rows, n_cols)``.
    corder : bool
        Stream ownership.
    backend : str, optional
        Backend for the count and fill kernels.

    Returns
    -------
    JITCStructure
        The canonical CSR connectivity.
    """
    structure, _ = _draw_structure(np.ones((1,), np.float32), prob, seed, shape=shape, corder=corder, backend=backend)
    return structure


def _draw_structure(weight, prob, seed, *, shape, corder, backend):
    placeholder = np.zeros((1,), np.float32)
    return build_jitc_structure(
        lambda clen, seed_, s: jits_csr_count_p_call(
            placeholder, clen, seed_, shape=s, corder=True, backend=backend)[0],
        lambda clen, seed_, offsets, nnz, s: jits_csr_fill_p_call(
            weight, clen, seed_, offsets, nnz, shape=s, corder=True, backend=backend),
        prob, seed, shape=shape, corder=corder, what='jits_to_csr',
    )


def jits_to_csr(
    weight,
    prob,
    seed,
    *,
    shape: MatrixShape,
    corder: bool,
    backend: Optional[str] = None,
    structure: Optional[JITCStructure] = None,
):
    """Materialize the light-RNG JIT scalar matrix as a :class:`~brainevent.CSR`.

    ``jits_to_csr`` reproduces exactly the matrix drawn by ``jits(corder=...)``
    (and thus by ``jitsmv`` / ``jitsmm``). It works under ``jax.jit``,
    ``jax.vmap`` and ``jax.grad``: the connectivity is drawn at trace time from
    a concrete ``prob`` and ``seed`` (or taken from ``structure``), and
    ``weight`` may be traced.

    Parameters
    ----------
    weight : float, array_like or Quantity
        Scalar weight of every stored entry.
    prob : float or array_like
        Connection probability. Must be concrete unless ``structure`` is given.
    seed : int, array_like or None
        Random seed. Must be concrete unless ``structure`` is given.
    shape : tuple of int
        Logical ``(n_rows, n_cols)``.
    corder : bool
        Stream ownership.
    backend : str, optional
        Backend for the count and fill kernels.
    structure : JITCStructure, optional
        Connectivity from :func:`jits_structure` (or ``M.prepare_structure()``).

    Returns
    -------
    CSR
        The matrix, with column-sorted rows.

    Raises
    ------
    ValueError
        If ``prob`` or ``seed`` is traced and ``structure`` is not given.
    """
    from brainevent._csr import CSR

    n_rows, n_cols = int(shape[0]), int(shape[1])
    weight, unitd = u.split_mantissa_unit(weight)
    weight = jnp.atleast_1d(jnp.asarray(weight))

    if structure is None and (n_rows == 0 or n_cols == 0 or _is_static_zero(prob)):
        indptr = jnp.zeros((n_rows + 1,), dtype=jnp.int32)
        indices = jnp.zeros((0,), dtype=jnp.int32)
        data = u.maybe_decimal(jnp.zeros((0,), dtype=weight.dtype) * unitd)
        return CSR((data, indices, indptr), shape=(n_rows, n_cols))

    if structure is None:
        structure = jits_structure(prob, concrete_seed(seed), shape=shape, corder=corder,
                                   backend=backend)
    else:
        check_jitc_structure(structure, shape=shape, corder=corder)
    # Every stored entry of a scalar JITC matrix holds ``weight``; no fill needed.
    data = jnp.full((structure.nnz,), weight[0], dtype=weight.dtype)
    return CSR((u.maybe_decimal(data * unitd), structure.indices, structure.indptr), shape=(n_rows, n_cols))
