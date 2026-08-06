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
Direct per-synapse ``y * w`` generation for normal-weight just-in-time
connectivity (JITC) matrices, on the light-RNG (mv) kernels.

:func:`jitnmv_dt2t` returns one value per generated structural non-zero in
canonical CSR flat order (the same order as ``jitn_to_csr(..., matrix_mode='mv')``),
namely ``sampled_weight * y[row]`` (``transpose=False``) or ``sampled_weight * y[col]``
(``transpose=True``).  It always materializes the mv matrix.

``corder`` keeps its usual meaning (it selects the notrans/trans generation).  The
``corder=True`` path can use the fused fill primitive (which replays the mv-notrans
walk); the ``corder=False`` path composes over :func:`jitn_to_csr` (whose trans
materialization has a different flat order the fused kernel can't reproduce).
"""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import _initialize_seed
from brainevent._numba_random import get_numba_light_rng_funcs
from brainevent._op import XLACustomKernel, load_cuda_file, numba_kernel
from brainevent._typing import MatrixShape
from .csr import jitn_to_csr
from .float import _MV_STRIDE, _normalize_chunk_size
from brainevent._op.util import dtype_suffix

__all__ = [
    'jitnmv_dt2t',
    'jitnmv_dt2t_p',
    'jitnmv_dt2t_p_call',
]


def jitnmv_dt2t(
    w_loc,
    w_scale,
    prob,
    y,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
):
    """Generate per-synapse ``y * w`` values for a normal JITC (mv) matrix."""
    shape = (int(shape[0]), int(shape[1]))
    n_rows, n_cols = shape

    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    y, unity = u.split_mantissa_unit(y)
    common_dtype = jnp.result_type(w_loc, w_scale, y)
    w_loc = jnp.atleast_1d(jnp.asarray(w_loc, dtype=common_dtype))
    w_scale = jnp.atleast_1d(jnp.asarray(w_scale, dtype=common_dtype))
    y = jnp.asarray(y, dtype=common_dtype)
    seed = _initialize_seed(seed)

    if y.ndim != 1:
        raise AssertionError("y must be 1D.")
    if transpose:
        assert n_cols == y.shape[0], "Shape mismatch for transpose operation."
    else:
        assert n_rows == y.shape[0], "Shape mismatch for non-transpose operation."

    if not isinstance(prob, Tracer) and float(np.asarray(prob)) == 0.0:
        data = jnp.zeros(0, dtype=common_dtype)
        return u.maybe_decimal(data * unitd * unity)

    # Materialize the canonical (column-sorted) mv CSR; dt2t is ``weight * y`` at
    # each structural non-zero, taken in that CSR's flat order.  Composing over
    # ``jitn_to_csr`` keeps both corder values and both directions consistent with
    # the materialized structure and is deterministic.
    csr = jitn_to_csr(
        w_loc, w_scale, prob, seed,
        shape=shape, corder=corder, matrix_mode='mv', backend=backend,
    )
    indptr = csr.indptr
    nnz = int(indptr[-1])
    if nnz == 0:
        return u.maybe_decimal(jnp.zeros(0, dtype=common_dtype) * unitd * unity)

    if transpose:
        gathered = y[csr.indices]                                     # weight * y[col]
    else:
        row_ids = jnp.repeat(
            jnp.arange(n_rows, dtype=jnp.int32), jnp.diff(indptr), total_repeat_length=nnz
        )
        gathered = y[row_ids]                                         # weight * y[row]
    return u.maybe_decimal(csr.data * gathered * unitd * unity)


# ---------------------------------------------------------------------- #
#  Fused fill primitive (corder=True / notrans structure), CUDA + numba
# ---------------------------------------------------------------------- #

def _jitnmv_dt2t_fill_cuda_kernel(
    shape: MatrixShape,
    transpose: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs,
):
    load_cuda_file(Path(__file__).parent.joinpath('dt2t.cu'), name='jit_normal_dt2t')
    sfx = dtype_suffix(kwargs['w_loc_info'].dtype)
    direction = 'trans' if transpose else 'notrans'
    kernel_name = f'jit_normal_dt2t.fill_{direction}{sfx}'
    n_cols = int(shape[1])
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)

    def kernel(w_loc, w_scale, clen, y, seed, chunk_offsets):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w_loc, w_scale, clen, y, seed, chunk_offsets,
            n_cols=np.int32(n_cols), chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jitnmv_dt2t_fill_numba_kernel(
    shape: MatrixShape,
    transpose: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs,
):
    """Numba CPU fused ``dt2t`` fill mirroring ``dt2t.cu``."""
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_normal01 = _rng['normal01']

    stride = _MV_STRIDE
    k = int(shape[1])
    cs_val = _normalize_chunk_size(k, chunk_size, target_chunks)

    if transpose:
        @numba.njit(fastmath=True)
        def kernel_impl(w_loc, w_scale, clen, y, seed, chunk_offsets, data):
            m = chunk_offsets.shape[0]
            n_chunks = chunk_offsets.shape[1]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        continue
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    pos = chunk_offsets[row, chunk_id]
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            n01 = _rng_normal01(seed0, row, j)
                            data[pos] = (w_loc0 + n01 * w_scale0) * y[j]
                            pos += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w_loc, w_scale, clen, y, seed, chunk_offsets, data):
            m = chunk_offsets.shape[0]
            n_chunks = chunk_offsets.shape[1]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            for row in range(m):
                yrow = y[row]
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        continue
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    pos = chunk_offsets[row, chunk_id]
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            n01 = _rng_normal01(seed0, row, j)
                            data[pos] = (w_loc0 + n01 * w_scale0) * yrow
                            pos += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_loc, w_scale, clen, y, seed, chunk_offsets):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_loc, w_scale, clen, y, seed, chunk_offsets)

    return kernel


def jitnmv_dt2t_p_call(
    w_loc,
    w_scale,
    clen,
    y,
    seed,
    chunk_offsets,
    nnz: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    """Fused ``dt2t`` fill over the mv-notrans structure (``chunk_offsets`` is the
    per-(row, chunk) exclusive-offset table). Returns ``(data,)`` of length ``nnz``."""
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)
    y = jnp.asarray(y)
    seed = jnp.atleast_1d(seed)
    chunk_offsets = jnp.asarray(chunk_offsets, dtype=jnp.int32)
    assert len(shape) == 2, f"shape must be two-dimensional, but got {shape}."
    assert y.ndim == 1, "y must be 1D."
    assert chunk_offsets.ndim == 2, "chunk_offsets must be 2D (n_rows, n_chunks)."
    assert jnp.issubdtype(w_loc.dtype, jnp.floating), "w_loc must be a floating-point type."
    assert jnp.issubdtype(w_scale.dtype, jnp.floating), "w_scale must be a floating-point type."
    assert w_loc.dtype == w_scale.dtype == y.dtype, (
        f"w_loc, w_scale, and y must share dtype, got {w_loc.dtype}, {w_scale.dtype}, {y.dtype}."
    )
    if transpose:
        assert shape[1] == y.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[0] == y.shape[0], "Shape mismatch for non-transpose operation."

    return jitnmv_dt2t_p(
        w_loc, w_scale, clen, y, seed, chunk_offsets,
        outs=[jax.ShapeDtypeStruct((int(nnz),), y.dtype)],
        shape=(int(shape[0]), int(shape[1])),
        transpose=transpose,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        y_info=jax.ShapeDtypeStruct(y.shape, y.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        chunk_offsets_info=jax.ShapeDtypeStruct(chunk_offsets.shape, chunk_offsets.dtype),
    )


jitnmv_dt2t_p = XLACustomKernel(
    'jitnmv_dt2t_fill',
    doc="""
Low-level CUDA primitive filling per-synapse ``sampled_weight * y`` values for a normal
JITC (mv) matrix, in the mv-notrans CSR flat order.
""",
)
jitnmv_dt2t_p.def_cuda_raw_kernel(_jitnmv_dt2t_fill_cuda_kernel, asdefault=True)
jitnmv_dt2t_p.def_numba_kernel(_jitnmv_dt2t_fill_numba_kernel)
jitnmv_dt2t_p.def_call(jitnmv_dt2t_p_call)
jitnmv_dt2t_p.def_tags('jit_normal', 'dt2t', 'light_rng')
