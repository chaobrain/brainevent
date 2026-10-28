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
Materialize a normal-weight just-in-time connectivity (JITC) matrix directly
into Compressed Sparse Row (CSR) format using the light-RNG chunk kernels.

``jitn_to_csr`` reproduces exactly the matrix used by ``jitn``, ``jitnmv``,
``jitnmm`` and the event kernels -- they all draw the same 32-lane matrix.
``corder`` keeps its usual meaning (it selects the notrans/trans generation,
exactly as in ``jitnmv``); the matrix class flips it on transpose.

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
from brainevent._compatible_import import Tracer
from brainevent._jitc_structure import (
    JITCStructure, build_jitc_structure, check_jitc_structure, concrete_seed, fill_shape,
)
from brainevent._misc import _is_static_zero, _n_chunks, _with_linear_param_grads
from brainevent._numba_random import get_numba_light_rng_funcs
from brainevent._op import XLACustomKernel, load_cuda_file, numba_kernel
from brainevent._typing import MatrixShape
from .float import _LANE_STRIDE, _chunk_size, _walk_length
from brainevent._op.util import dtype_suffix

__all__ = [
    'jitn_to_csr',
    'jitn_csr_count_p',
    'jitn_csr_count_p_call',
    'jitn_csr_fill_p',
    'jitn_csr_fill_p_call',
]


# ──────────────────────────────────────────────────────────────────────
#  Count pass
# ──────────────────────────────────────────────────────────────────────

def _jitn_csr_count_cuda_kernel(
    shape: MatrixShape,
    corder: bool,
    **kwargs,
):
    load_cuda_file(Path(__file__).parent.joinpath('csr.cu'), name='jit_normal_csr')
    sfx = dtype_suffix(kwargs['w_loc_info'].dtype)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    walk = _walk_length(shape, False, corder)
    chunk_size_value = _chunk_size(walk)

    if corder:
        # notrans: chunk the mat columns; per-(row, chunk) counts.
        kernel_name = f'jit_normal_csr.count_chunks_notrans{sfx}'

        def kernel(w_loc, w_scale, clen, seed):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                w_loc, w_scale, clen, seed,
                n_cols=np.int32(n_cols), chunk_size=np.int32(chunk_size_value),
            )
    else:
        # trans: seed over mat columns, walk mat rows, atomic per-row counts.
        kernel_name = f'jit_normal_csr.count_chunks_trans{sfx}'

        def kernel(w_loc, w_scale, clen, seed):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                w_loc, w_scale, clen, seed,
                n_rows=np.int32(n_cols), n_cols=np.int32(n_rows),
                chunk_size=np.int32(chunk_size_value),
            )

    return kernel


def _jitn_csr_count_numba_kernel(
    shape: MatrixShape,
    corder: bool,
    **kwargs,
):
    """Numba CPU count pass mirroring ``csr.cu``."""
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
        k = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_loc, w_scale, clen, seed, chunk_counts):
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
        m_walk = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_loc, w_scale, clen, seed, row_counts):
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

    def kernel(w_loc, w_scale, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_loc, w_scale, clen, seed)

    return kernel


def jitn_csr_count_p_call(
    w_loc,
    w_scale,
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
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)
    assert clen.shape == (2,), f"clen must hold (cl, thr), but got shape {clen.shape}."
    seed = jnp.atleast_1d(seed)

    if corder:
        n_chunks = _n_chunks(walk, chunk_size_value)
        if n_rows == 0 or n_chunks == 0:
            return (jnp.zeros((n_rows, n_chunks), dtype=jnp.int32),)
        out_info = jax.ShapeDtypeStruct((n_rows, n_chunks), jnp.int32)
    else:
        if n_rows == 0 or n_cols == 0:
            return (jnp.zeros((n_rows,), dtype=jnp.int32),)
        out_info = jax.ShapeDtypeStruct((n_rows,), jnp.int32)

    return jitn_csr_count_p(
        w_loc, w_scale, clen, seed,
        outs=[out_info],
        shape=(n_rows, n_cols),
        corder=corder,
        backend=backend,
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
    )


jitn_csr_count_p = XLACustomKernel(
    'jitn_csr_count',
    doc="Low-level primitive counting non-zeros for light normal CSR materialization.",
)
jitn_csr_count_p.def_cuda_raw_kernel(_jitn_csr_count_cuda_kernel, asdefault=True)
jitn_csr_count_p.def_numba_kernel(_jitn_csr_count_numba_kernel)
jitn_csr_count_p.def_call(jitn_csr_count_p_call)
jitn_csr_count_p.def_tags('jit_normal', 'csr', 'light_rng')


# ──────────────────────────────────────────────────────────────────────
#  Fill pass
# ──────────────────────────────────────────────────────────────────────

def _jitn_csr_fill_cuda_kernel(
    shape: MatrixShape,
    corder: bool,
    **kwargs,
):
    load_cuda_file(Path(__file__).parent.joinpath('csr.cu'), name='jit_normal_csr')
    sfx = dtype_suffix(kwargs['w_loc_info'].dtype)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    walk = _walk_length(shape, False, corder)
    chunk_size_value = _chunk_size(walk)

    if corder:
        kernel_name = f'jit_normal_csr.fill_notrans{sfx}'

        def kernel(w_loc, w_scale, clen, seed, offsets):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                w_loc, w_scale, clen, seed, offsets,
                n_cols=np.int32(n_cols), chunk_size=np.int32(chunk_size_value),
            )
    else:
        kernel_name = f'jit_normal_csr.fill_trans{sfx}'

        def kernel(w_loc, w_scale, clen, seed, offsets):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                w_loc, w_scale, clen, seed, offsets,
                n_rows=np.int32(n_cols), n_cols=np.int32(n_rows),
                chunk_size=np.int32(chunk_size_value),
            )

    return kernel


def _jitn_csr_fill_numba_kernel(
    shape: MatrixShape,
    corder: bool,
    **kwargs,
):
    """Numba CPU fill pass mirroring ``csr.cu``."""
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_keep = _rng['keep']
    _rng_normal01 = _rng['normal01']

    stride = _LANE_STRIDE
    n_cols = int(shape[1])
    cs_val = _chunk_size(_walk_length(shape, False, corder))

    if corder:
        k = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_loc, w_scale, clen, seed, offsets, indices, data):
            m = offsets.shape[0]
            n_chunks = offsets.shape[1]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
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
                                col = chunk_start + local_j
                                n01 = _rng_normal01(seed0, row, col)
                                indices[pos] = col
                                data[pos] = w_loc0 + n01 * w_scale0
                                pos += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
    else:
        m_walk = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_loc, w_scale, clen, seed, offsets, indices, data, cursor):
            cursor[:] = 0
            k = cursor.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
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
                                n01 = _rng_normal01(seed0, row, j)
                                indices[pos] = row
                                data[pos] = w_loc0 + n01 * w_scale0
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_loc, w_scale, clen, seed, offsets):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_loc, w_scale, clen, seed, offsets)

    return kernel


def jitn_csr_fill_p_call(
    w_loc,
    w_scale,
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
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)
    assert clen.shape == (2,), f"clen must hold (cl, thr), but got shape {clen.shape}."
    seed = jnp.atleast_1d(seed)
    offsets = jnp.asarray(offsets, dtype=jnp.int32)
    nnz = int(nnz)
    if nnz == 0:
        return (jnp.zeros((0,), dtype=jnp.int32), jnp.zeros((0,), dtype=w_loc.dtype))

    outs = [
        jax.ShapeDtypeStruct((nnz,), jnp.int32),
        jax.ShapeDtypeStruct((nnz,), w_loc.dtype),
    ]
    if not corder:
        # trans fill also needs an int32 write cursor (one per output row).
        outs.append(jax.ShapeDtypeStruct((n_rows,), jnp.int32))

    res = jitn_csr_fill_p(
        w_loc, w_scale, clen, seed, offsets,
        outs=outs,
        shape=(n_rows, n_cols),
        corder=corder,
        backend=backend,
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        offsets_info=jax.ShapeDtypeStruct(offsets.shape, offsets.dtype),
    )
    return res[0], res[1]


jitn_csr_fill_p = XLACustomKernel(
    'jitn_csr_fill',
    doc="Low-level primitive filling CSR indices/data for light normal CSR.",
)
jitn_csr_fill_p.def_cuda_raw_kernel(_jitn_csr_fill_cuda_kernel, asdefault=True)
jitn_csr_fill_p.def_numba_kernel(_jitn_csr_fill_numba_kernel)
jitn_csr_fill_p.def_call(jitn_csr_fill_p_call)
jitn_csr_fill_p.def_tags('jit_normal', 'csr', 'light_rng')


# ──────────────────────────────────────────────────────────────────────
#  High-level orchestration
# ──────────────────────────────────────────────────────────────────────

def jitn_structure(prob, seed, *, shape: MatrixShape, corder: bool, backend: Optional[str] = None):
    """Draw the connectivity of a JIT normal matrix (see :func:`~brainevent._jitc_structure.build_jitc_structure`).

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
    zero, one = np.zeros((1,), np.float32), np.ones((1,), np.float32)
    structure, _ = _draw_structure(zero, one, prob, seed, shape=shape, corder=corder, backend=backend)
    return structure


def _draw_structure(w_loc, w_scale, prob, seed, *, shape, corder, backend):
    zero, one = np.zeros((1,), np.float32), np.ones((1,), np.float32)
    return build_jitc_structure(
        lambda clen, seed_, s: jitn_csr_count_p_call(
            zero, one, clen, seed_, shape=s, corder=True, backend=backend)[0],
        lambda clen, seed_, offsets, nnz, s: jitn_csr_fill_p_call(
            w_loc, w_scale, clen, seed_, offsets, nnz, shape=s, corder=True, backend=backend),
        prob, seed, shape=shape, corder=corder, what='jitn_to_csr',
    )


def jitn_to_csr(
    w_loc,
    w_scale,
    prob,
    seed,
    *,
    shape: MatrixShape,
    corder: bool,
    backend: Optional[str] = None,
    structure: Optional[JITCStructure] = None,
):
    """Materialize the light-RNG JIT normal matrix as a :class:`~brainevent.CSR`.

    ``jitn_to_csr`` reproduces exactly the matrix drawn by ``jitn(corder=...)``
    (and thus by ``jitnmv`` / ``jitnmm``). It works under ``jax.jit``,
    ``jax.vmap`` and ``jax.grad``: the connectivity is drawn at trace time from
    a concrete ``prob`` and ``seed`` (or taken from ``structure``), and the
    weight parameters may be traced. ``data`` is differentiable with respect
    to ``w_loc`` and ``w_scale``.

    Parameters
    ----------
    w_loc : float, array_like or Quantity
        Mean of the normal weights.
    w_scale : float, array_like or Quantity
        Standard deviation of the normal weights, in the unit of ``w_loc``.
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
        Connectivity from :func:`jitn_structure` (or ``M.prepare_structure()``).

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
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    dtype = jnp.result_type(w_loc, w_scale)
    w_loc = jnp.atleast_1d(jnp.asarray(w_loc, dtype=dtype))
    w_scale = jnp.atleast_1d(jnp.asarray(w_scale, dtype=dtype))

    if structure is None and (n_rows == 0 or n_cols == 0 or _is_static_zero(prob)):
        indptr = jnp.zeros((n_rows + 1,), dtype=jnp.int32)
        indices = jnp.zeros((0,), dtype=jnp.int32)
        data = u.maybe_decimal(jnp.zeros((0,), dtype=dtype) * unitd)
        return CSR((data, indices, indptr), shape=(n_rows, n_cols))

    # The connectivity never depends on the weights: draw it with them stopped,
    # and attach the weight derivative below.
    stopped = jax.lax.stop_gradient((w_loc, w_scale))
    seed = concrete_seed(seed)
    data = None
    if structure is None:
        concrete = not any(isinstance(x, Tracer) for x in stopped)
        first, second = stopped if concrete else (np.zeros((1,), np.float32), np.ones((1,), np.float32))
        structure, drawn = _draw_structure(
            first, second, prob, seed, shape=shape, corder=corder, backend=backend,
        )
        if concrete:
            data = drawn
    else:
        check_jitc_structure(structure, shape=shape, corder=corder)

    clen = _initialize_conn_length(prob)
    seed_arr = _initialize_seed(seed)
    fshape = fill_shape(shape, corder)
    nnz = structure.nnz

    def draw(a, b, clen_, seed_, offsets, order):
        filled = jitn_csr_fill_p_call(
            a, b, clen_, seed_, offsets, nnz, shape=fshape, corder=True, backend=backend,
        )[1]
        return filled[order]

    operands = (clen, seed_arr, structure.fill_offsets, structure.order)
    if data is None:
        data = draw(stopped[0], stopped[1], clen, seed_arr, structure.fill_offsets, structure.order)

    def coefficients(clen_, seed_, offsets, order):
        # Each weight is ``w_loc + z * w_scale``; ``z`` is the draw with (0, 1).
        z = draw(jnp.zeros((1,), dtype), jnp.ones((1,), dtype), clen_, seed_, offsets, order)
        return jnp.ones_like(z), z

    data = _with_linear_param_grads(coefficients, data, (w_loc, w_scale), operands)
    return CSR((u.maybe_decimal(data * unitd), structure.indices, structure.indptr), shape=(n_rows, n_cols))
