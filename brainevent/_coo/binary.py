# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


from pathlib import Path
from typing import Sequence, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, namescope
from brainevent._op import numba_kernel, XLACustomKernel, general_batching_rule, register_tvm_cuda_from_file, \
    jaxinfo_to_warpinfo
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Row, Col, MatrixShape
from .float import coomv, coomm

__all__ = [
    'binary_coomv',
    'binary_coomv_p',
    'binary_coomm',
    'binary_coomm_p',
]


@namescope(static_argnames=("shape", "transpose"))
def binary_coomv(
    data: Data,
    row: Row,
    col: Col,
    v: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """
    Perform event-driven COO sparse matrix-vector multiplication.

    Computes the product of a sparse matrix stored in COO (Coordinate) format
    and a dense vector, where the vector ``v`` is treated as containing binary
    events (spikes). Only entries corresponding to active (nonzero / ``True``)
    events contribute to the output, making this operation efficient for
    spike-based neural network simulations.

    Mathematically, with ``transpose=False`` the operation is

    ``result[i] = sum_j A[i, j] * (v[j] > 0)``

    and with ``transpose=True``:

    ``result[j] = sum_i A[i, j] * (v[i] > 0)``

    where ``A`` is the sparse matrix defined by (``data``, ``row``, ``col``).

    Parameters
    ----------
    data : jax.Array
        The non-zero weight values of the sparse matrix.  Can be either a
        scalar (shape ``(1,)``, homogeneous weights) or a 1-D array of
        length ``nnz`` (heterogeneous weights).
    row : jax.Array
        1-D array of row indices for each non-zero element, with length
        ``nnz``.
    col : jax.Array
        1-D array of column indices for each non-zero element, with length
        ``nnz``.
    v : jax.Array
        Dense input vector treated as binary events. Elements are
        interpreted as active when ``True`` (boolean dtype) or ``> 0``
        (numeric dtype).  Shape must be ``(shape[0],)`` when
        ``transpose=True`` or ``(shape[1],)`` when ``transpose=False``.
    shape : tuple of int
        The ``(m, k)`` shape of the sparse matrix.
    transpose : bool, optional
        If ``True``, multiply by the transpose of the sparse matrix,
        i.e. ``A^T @ v``.  Default is ``False``.
    backend : str or None, optional
        Compute backend to use (e.g. ``'numba'``, ``'pallas'``).
        When ``None`` the backend is chosen automatically.

    Returns
    -------
    jax.Array
        Result vector of the matrix-vector product.  Shape is
        ``(shape[0],)`` when ``transpose=False`` or ``(shape[1],)`` when
        ``transpose=True``.

    Raises
    ------
    ValueError
        If ``row`` or ``col`` are not 1-D, have mismatched lengths, ``v``
        is not 1-D, ``data`` is not scalar or 1-D, or array dimensions
        are incompatible with ``shape`` and ``transpose``.

    See Also
    --------
    binary_coomm : Event-driven COO sparse matrix-matrix multiplication.
    coomv : Standard (non-event-driven) COO sparse matrix-vector multiplication.

    Notes
    -----
    Physical units attached via ``brainunit`` are automatically split off
    before the computation and re-applied to the result.

    This function supports automatic differentiation (JVP and transpose
    rules), ``vmap`` batching, and multiple hardware backends (CPU via
    Numba, GPU via Pallas/Triton, TPU via Pallas/Mosaic).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._coo.binary import binary_coomv
        >>> data = jnp.array([0.5])          # homogeneous weight
        >>> row = jnp.array([0, 1, 2])
        >>> col = jnp.array([1, 0, 2])
        >>> v = jnp.array([True, False, True])  # binary events
        >>> binary_coomv(data, row, col, v, shape=(3, 3), transpose=False)
    """
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = binary_coomv_p_call(
        data, row, col, v,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


@namescope(static_argnames=("shape", "transpose"))
def binary_coomm(
    data: Data,
    row: Row,
    col: Col,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """
    Perform event-driven COO sparse matrix-matrix multiplication.

    Computes the product of a sparse matrix stored in COO (Coordinate) format
    and a dense matrix ``B``, where ``B`` is treated as containing binary
    events (spikes). Only entries of ``B`` that are active (nonzero /
    ``True``) contribute to the output.

    Mathematically, with ``transpose=False`` the operation is

    ``result[i, n] = sum_j A[i, j] * (B[j, n] > 0)``

    and with ``transpose=True``:

    ``result[j, n] = sum_i A[i, j] * (B[i, n] > 0)``

    where ``A`` is the sparse matrix defined by (``data``, ``row``, ``col``).

    Parameters
    ----------
    data : jax.Array
        The non-zero weight values of the sparse matrix.  Can be either a
        scalar (shape ``(1,)``, homogeneous weights) or a 1-D array of
        length ``nnz`` (heterogeneous weights).
    row : jax.Array
        1-D array of row indices for each non-zero element, with length
        ``nnz``.
    col : jax.Array
        1-D array of column indices for each non-zero element, with length
        ``nnz``.
    B : jax.Array
        Dense input matrix treated as binary events.  Shape must be
        ``(shape[0], n)`` when ``transpose=True`` or ``(shape[1], n)``
        when ``transpose=False``.
    shape : tuple of int
        The ``(m, k)`` shape of the sparse matrix.
    transpose : bool, optional
        If ``True``, multiply by the transpose of the sparse matrix,
        i.e. ``A^T @ B``.  Default is ``False``.
    backend : str or None, optional
        Compute backend to use (e.g. ``'numba'``, ``'pallas'``).
        When ``None`` the backend is chosen automatically.

    Returns
    -------
    jax.Array
        Result matrix of the multiplication.  Shape is
        ``(shape[0], n)`` when ``transpose=False`` or ``(shape[1], n)``
        when ``transpose=True``.

    Raises
    ------
    ValueError
        If ``row`` or ``col`` are not 1-D, have mismatched lengths, ``B``
        is not 2-D, ``data`` is not scalar or 1-D, or array dimensions
        are incompatible with ``shape`` and ``transpose``.

    See Also
    --------
    binary_coomv : Event-driven COO sparse matrix-vector multiplication.
    coomm : Standard (non-event-driven) COO sparse matrix-matrix multiplication.

    Notes
    -----
    Physical units attached via ``brainunit`` are automatically split off
    before the computation and re-applied to the result.

    This function supports automatic differentiation (JVP and transpose
    rules), ``vmap`` batching, and multiple hardware backends (CPU via
    Numba, GPU via Pallas/Triton, TPU via Pallas/Mosaic).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._coo.binary import binary_coomm
        >>> data = jnp.array([0.5])          # homogeneous weight
        >>> row = jnp.array([0, 1, 2])
        >>> col = jnp.array([1, 0, 2])
        >>> B = jnp.array([[True, False],
        ...                [False, True],
        ...                [True, True]])    # binary event matrix
        >>> binary_coomm(data, row, col, B, shape=(3, 3), transpose=False)
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = binary_coomm_p_call(
        data, row, col, B,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


# =============================================================================
# COO Matrix-Vector Multiplication (coomv)
# =============================================================================


def _coomv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        if v[row[i]]:
                            posts[col[i]] += w
            else:
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        if v[row[i]] > 0.:
                            posts[col[i]] += w
        else:
            # transpose=True, heterogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        if v[row[i]]:
                            posts[col[i]] += weights[i]
            else:
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        if v[row[i]] > 0.:
                            posts[col[i]] += weights[i]
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        if v[col[i]]:
                            posts[row[i]] += w
            else:
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        if v[col[i]] > 0.:
                            posts[row[i]] += w
        else:
            # transpose=False, heterogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        if v[col[i]]:
                            posts[row[i]] += weights[i]
            else:
                @numba.njit(fastmath=True)
                def mv(weights, row, col, v, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        if v[col[i]] > 0.:
                            posts[row[i]] += weights[i]

    def kernel(weights, row, col, v):
        return numba_kernel(mv, outs=kwargs['outs'])(weights, row, col, v)

    return kernel


def _coomv_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    row_warp_info = jaxinfo_to_warpinfo(row_info)
    col_warp_info = jaxinfo_to_warpinfo(col_info)
    spike_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[row[i]]:
                        warp.atomic_add(posts, col[i], w)
            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[row[i]] > 0.:
                        warp.atomic_add(posts, col[i], w)
        else:
            # transpose=True, heterogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    if v[row[i]]:
                        warp.atomic_add(posts, col[i], weights[i])
            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    if v[row[i]] > 0.:
                        warp.atomic_add(posts, col[i], weights[i])
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[col[i]]:
                        warp.atomic_add(posts, row[i], w)
            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[col[i]] > 0.:
                        warp.atomic_add(posts, row[i], w)
        else:
            # transpose=False, heterogeneous
            if vector_info.dtype == jnp.bool_:
                # bool
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    if v[col[i]]:
                        warp.atomic_add(posts, row[i], weights[i])
            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    if v[col[i]] > 0.:
                        warp.atomic_add(posts, row[i], weights[i])

    def kernel(weights, row, col, v):
        dim = row_info.shape[0]
        out_info = kwargs['outs'][0]
        fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
        return fn(weights, row, col, v, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _coomv_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    from jax.experimental.pallas.triton import atomic_add

    nnz = row_info.shape[0]
    block_dim = generate_block_dim(nnz)
    block_dim = 32 if block_dim < 32 else block_dim

    if transpose:
        if weight_info.size == 1:
            # coo.T @ v (homogeneous weights)
            #
            # coo: [m, k]
            # v: [m]
            # result: [k]
            #
            def mv(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                vector_ref,  # [m]
                _,  # [k]
                posts_ref,  # [k]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = plgpu.load(row_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                cols = plgpu.load(col_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                events = plgpu.load(
                    vector_ref.at[rows],
                    mask=mask,
                    other=False if vector_ref.dtype == jnp.bool_ else 0
                )

                if vector_ref.dtype == jnp.bool_:
                    event_mask = mask & events
                else:
                    event_mask = mask & (events > 0.)

                data = jnp.full((block_dim,), jnp.asarray(data_ref[0], dtype=posts_ref.dtype), dtype=posts_ref.dtype)
                atomic_add(posts_ref, cols, data, mask=event_mask)

        else:
            # coo.T @ v (heterogeneous weights)
            #
            # coo: [m, k]
            # v: [m]
            # result: [k]
            #
            def mv(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                vector_ref,  # [m]
                _,  # [k]
                posts_ref,  # [k]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = plgpu.load(row_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                cols = plgpu.load(col_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                weights = plgpu.load(data_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                events = plgpu.load(
                    vector_ref.at[rows],
                    mask=mask,
                    other=False if vector_ref.dtype == jnp.bool_ else 0
                )

                if vector_ref.dtype == jnp.bool_:
                    event_mask = mask & events
                else:
                    event_mask = mask & (events > 0.)

                data = jnp.asarray(weights, dtype=posts_ref.dtype)
                atomic_add(posts_ref, cols, data, mask=event_mask)

        def kernel(data, row, col, vector):
            fn = pl.pallas_call(
                mv,
                grid=(pl.cdiv(nnz, block_dim),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, vector, posts)

        return kernel

    else:
        # coo @ v (non-transpose)
        # Also needs atomic operations due to COO race conditions
        if weight_info.size == 1:
            # coo @ v (homogeneous weights)
            #
            # coo: [m, k]
            # v: [k]
            # result: [m]
            #
            def mv(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                vector_ref,  # [k]
                _,  # [m]
                posts_ref,  # [m]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = plgpu.load(row_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                cols = plgpu.load(col_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                events = plgpu.load(
                    vector_ref.at[cols],
                    mask=mask,
                    other=False if vector_ref.dtype == jnp.bool_ else 0
                )

                if vector_ref.dtype == jnp.bool_:
                    event_mask = mask & events
                else:
                    event_mask = mask & (events > 0.)

                data = jnp.full((block_dim,), jnp.asarray(data_ref[0], dtype=posts_ref.dtype), dtype=posts_ref.dtype)
                atomic_add(posts_ref, rows, data, mask=event_mask)

        else:
            # coo @ v (heterogeneous weights)
            #
            # coo: [m, k]
            # v: [k]
            # result: [m]
            #
            def mv(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                vector_ref,  # [k]
                _,  # [m]
                posts_ref,  # [m]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = plgpu.load(row_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                cols = plgpu.load(col_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                weights = plgpu.load(data_ref.at[pl.ds(i_start, block_dim)], mask=mask, other=0)
                events = plgpu.load(
                    vector_ref.at[cols],
                    mask=mask,
                    other=False if vector_ref.dtype == jnp.bool_ else 0
                )

                if vector_ref.dtype == jnp.bool_:
                    event_mask = mask & events
                else:
                    event_mask = mask & (events > 0.)

                data = jnp.asarray(weights, dtype=posts_ref.dtype)
                atomic_add(posts_ref, rows, data, mask=event_mask)

        def kernel(data, row, col, vector):
            fn = pl.pallas_call(
                mv,
                grid=(pl.cdiv(nnz, block_dim),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, vector, posts)

        return kernel


def _coomv_pallas_tpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl

    nnz = row_info.shape[0]
    block_dim = generate_block_dim(nnz)
    block_dim = 32 if block_dim < 32 else block_dim
    num_nnz_blocks = pl.cdiv(nnz, block_dim)
    out_len = kwargs['outs'][0].shape[0]

    if weight_info.size == 1:
        def mv(
            data_ref,  # [1]
            row_ref,  # [nnz]
            col_ref,  # [nnz]
            vector_ref,  # [m] or [k]
            _,  # [k] or [m]
            posts_ref,  # [k] or [m]
        ):
            i_out = pl.program_id(0)
            scalar_w = jnp.asarray(data_ref[0], dtype=posts_ref.dtype)

            def loop_fn(i_blk, acc):
                i_start = i_blk * block_dim
                elems = i_start + jnp.arange(block_dim)
                valid = elems < nnz
                safe_elems = jnp.where(valid, elems, 0)
                rows = row_ref[safe_elems]
                cols = col_ref[safe_elems]
                safe_rows = jnp.where(valid, rows, 0)
                safe_cols = jnp.where(valid, cols, 0)

                if transpose:
                    events = vector_ref[safe_rows]
                    out_idx = cols
                else:
                    events = vector_ref[safe_cols]
                    out_idx = rows

                if vector_ref.dtype == jnp.bool_:
                    active = events
                else:
                    active = events > 0.

                lane_mask = valid & (out_idx == i_out) & active
                return acc + jnp.sum(jnp.where(lane_mask, scalar_w, 0))

            acc0 = jnp.asarray(0, dtype=posts_ref.dtype)
            posts_ref[i_out] = jax.lax.fori_loop(0, num_nnz_blocks, loop_fn, acc0)

    else:
        def mv(
            data_ref,  # [nnz]
            row_ref,  # [nnz]
            col_ref,  # [nnz]
            vector_ref,  # [m] or [k]
            _,  # [k] or [m]
            posts_ref,  # [k] or [m]
        ):
            i_out = pl.program_id(0)

            def loop_fn(i_blk, acc):
                i_start = i_blk * block_dim
                elems = i_start + jnp.arange(block_dim)
                valid = elems < nnz
                safe_elems = jnp.where(valid, elems, 0)
                rows = row_ref[safe_elems]
                cols = col_ref[safe_elems]
                weights = jnp.asarray(data_ref[safe_elems], dtype=posts_ref.dtype)
                safe_rows = jnp.where(valid, rows, 0)
                safe_cols = jnp.where(valid, cols, 0)

                if transpose:
                    events = vector_ref[safe_rows]
                    out_idx = cols
                else:
                    events = vector_ref[safe_cols]
                    out_idx = rows

                if vector_ref.dtype == jnp.bool_:
                    active = events
                else:
                    active = events > 0.

                lane_mask = valid & (out_idx == i_out) & active
                return acc + jnp.sum(jnp.where(lane_mask, weights, 0))

            acc0 = jnp.asarray(0, dtype=posts_ref.dtype)
            posts_ref[i_out] = jax.lax.fori_loop(0, num_nnz_blocks, loop_fn, acc0)

    def kernel(data, row, col, vector):
        fn = pl.pallas_call(
            mv,
            grid=(out_len,),
            input_output_aliases={4: 0},
            out_shape=kwargs['outs'],
            backend='mosaic_tpu',
        )
        posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
        return fn(data, row, col, vector, posts)

    return kernel


def _binary_coomv_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for binary (event-driven) COO matrix-vector multiplication (all platforms)."""
    m, k = shape
    is_homo = (weight_info.size == 1)
    is_bool = (vector_info.dtype == jnp.bool_)
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, row, col, vector):
            v_vals = vector[row]
            events = v_vals.astype(out_dtype) if is_bool else (v_vals > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(k, dtype=out_dtype).at[col].add(w * events),)
    else:
        def kernel(weights, row, col, vector):
            v_vals = vector[col]
            events = v_vals.astype(out_dtype) if is_bool else (v_vals > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(m, dtype=out_dtype).at[row].add(w * events),)

    return kernel


def _binary_coomv_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed kernel for binary COO SpMV via jax.experimental.sparse (GPU only)."""
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    is_bool = (vector_info.dtype == jnp.bool_)
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        if is_homo:
            def kernel(weights, row, col, vector):
                events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
                ones = jnp.ones_like(row, dtype=out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((ones, idx), shape=(m, k))
                return ((mat.T @ events) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, row, col, vector):
                events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((weights.astype(out_dtype), idx), shape=(m, k))
                return (mat.T @ events,)
    else:
        if is_homo:
            def kernel(weights, row, col, vector):
                events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
                ones = jnp.ones_like(row, dtype=out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((ones, idx), shape=(m, k))
                return ((mat @ events) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, row, col, vector):
                events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((weights.astype(out_dtype), idx), shape=(m, k))
                return (mat @ events,)
    return kernel


def _coomv_tvmffi_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    """TVM FFI CUDA kernel for binary COO SpMV.

    Dispatches to one of the ``binary_coomv_atomic_{nt,t}`` kernels compiled
    from ``binary.cu`` via ``register_tvm_cuda_from_file``.

    Kernel selection:
    - Direction: ``_nt`` (transpose=False) or ``_t`` (transpose=True).
    - Weight dtype: ``_f32``, ``_f64``, ``_f16``, or ``_bf16``.
    - Spike type: ``_bool`` (int8) or ``_float`` (float32).
    - Homo vs. hetero: detected at runtime from ``data.size(0) == 1``.

    The output buffer is zero-initialized inside the CUDA entry function
    (via ``cudaMemsetAsync``) before the atomic-scatter kernel runs.
    """
    register_tvm_cuda_from_file(
        module='coo_binary_coomv',
        source=Path(__file__).parent.joinpath('binary_coomv.cu'),
    )

    out_info = kwargs['outs']
    spk_suffix = '_bool' if vector_info.dtype == jnp.bool_ else '_float'

    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    direction = '_t' if transpose else '_nt'
    kernel_name = f'coo_binary_coomv.binary_coomv_atomic{direction}{wt_sfx}{spk_suffix}'

    def kernel(weights, row, col, v):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, row, col, v)

    return kernel


def _coomv_jvp_vector(v_dot, data, row, col, v, *, shape, transpose, **kwargs):
    return [coomv(data, row, col, v_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _coomv_jvp_weights(data_dot, data, row, col, v, *, shape, transpose, **kwargs):
    return binary_coomv_p_call(data_dot, row, col, v, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _coomv_transpose_rule(ct, data, row, col, events, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(row)
    assert not ad.is_undefined_primal(col)
    ct = ct[0]

    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events)
        else:
            ct_events = coomv(
                data,
                row,
                col,
                ct,
                shape=shape,
                transpose=not transpose,
                backend=kwargs['backend']
            )
        return data, row, col, ct_events
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = binary_coomv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    row,
                    col,
                    events,
                    shape=shape,
                    transpose=transpose,
                    backend=kwargs['backend']
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:
                ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
        return ct_values, row, col, events


def _coomv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]

    else:
        return general_batching_rule(binary_coomv_p_call, args, axes, **kwargs)


def _coomv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                nnz = max(1, int(n_pre * n_post * prob))
                row = np.random.randint(0, n_pre, nnz, dtype=np.int32)
                col = np.random.randint(0, n_post, nnz, dtype=np.int32)
                weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(nnz, dtype=dtype)
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    vector = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    vector = jnp.asarray(np.random.rand(v_size), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, jnp.asarray(row), jnp.asarray(col), vector),
                        {'shape': (n_pre, n_post), 'transpose': transpose}
                    )
                )
    return configs


def binary_coomv_p_call(
    weights,
    row,
    col,
    v,
    *,
    shape: Sequence[int],
    transpose: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for event-driven COO sparse matrix-vector multiplication.

    Validates inputs, constructs output metadata, and dispatches to the
    registered ``XLACustomKernel`` (``binary_coomv_p``) which selects a
    backend-specific kernel (Numba, Pallas GPU, or Pallas TPU).

    Unlike :func:`binary_coomv`, this function does **not** handle physical
    units and returns a raw list of JAX arrays.

    Parameters
    ----------
    weights : jax.Array
        Non-zero values of the sparse matrix.  Must be a floating-point
        scalar (shape ``(1,)``) for homogeneous weights or a 1-D array
        of length ``nnz`` for heterogeneous weights.
    row : jax.Array
        1-D integer array of row indices with length ``nnz``.
    col : jax.Array
        1-D integer array of column indices with length ``nnz``.
    v : jax.Array
        Dense event vector.  Entries are treated as active when ``True``
        (boolean) or ``> 0`` (numeric).  Shape must be ``(shape[0],)``
        when ``transpose=True`` or ``(shape[1],)`` when
        ``transpose=False``.
    shape : Sequence[int]
        The ``(m, k)`` shape of the logical sparse matrix.
    transpose : bool
        If ``True``, compute ``A^T @ v`` instead of ``A @ v``.
    backend : str or None, optional
        Compute backend override (``'numba'``, ``'pallas'``).
        When ``None`` the backend is selected automatically.

    Returns
    -------
    list of jax.Array
        A single-element list containing the result vector.  Shape is
        ``(shape[0],)`` when ``transpose=False`` or ``(shape[1],)`` when
        ``transpose=True``, with dtype matching ``weights``.

    Raises
    ------
    ValueError
        If ``row`` or ``col`` are not 1-D arrays, have different lengths,
        ``v`` is not 1-D, ``weights`` is not scalar or 1-D, weights
        length is neither 1 nor ``nnz``, or ``v`` length is incompatible
        with ``shape`` and ``transpose``.
    AssertionError
        If ``weights`` dtype is not a floating-point type.

    See Also
    --------
    binary_coomv : High-level wrapper with physical unit support.

    Notes
    -----
    When ``nnz == 0`` the function short-circuits and returns a zero vector
    without dispatching to any kernel.

    The function registers JVP rules, transpose rules, and batching rules
    on the underlying ``binary_coomv_p`` primitive so that it integrates
    with JAX's autodiff and ``vmap`` transformations.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._coo.binary import binary_coomv_p_call
        >>> weights = jnp.array([1.0, 2.0, 3.0])
        >>> row = jnp.array([0, 1, 2])
        >>> col = jnp.array([2, 0, 1])
        >>> v = jnp.array([True, False, True])
        >>> result = binary_coomv_p_call(
        ...     weights, row, col, v, shape=(3, 3), transpose=False
        ... )
        >>> result[0]  # the output vector
    """
    row = jnp.asarray(row)
    col = jnp.asarray(col)
    v = jnp.asarray(v)
    if row.ndim != 1 or col.ndim != 1:
        raise ValueError(f'`row` and `col` must be 1D arrays, got row.ndim={row.ndim}, col.ndim={col.ndim}.')
    if row.shape[0] != col.shape[0]:
        raise ValueError(f'`row` and `col` must have the same length, got {row.shape[0]} and {col.shape[0]}.')
    if v.ndim != 1:
        raise ValueError(f'`v` must be a 1D array, got ndim={v.ndim}.')

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    else:
        weights = jnp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError(f'`weights` must be a scalar or 1D array, got ndim={weights.ndim}.')
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    nnz = row.shape[0]
    if weights.shape[0] not in (1, nnz):
        raise ValueError(f'`weights` length must be 1 or nnz={nnz}, got {weights.shape[0]}.')

    expected_v = shape[0] if transpose else shape[1]
    if v.shape[0] != expected_v:
        raise ValueError(f'`v` has incompatible length {v.shape[0]} for shape={tuple(shape)}, transpose={transpose}.')

    out_len = shape[1] if transpose else shape[0]
    if nnz == 0:
        return [jnp.zeros((out_len,), dtype=weights.dtype)]

    out_info = jax.ShapeDtypeStruct([out_len], weights.dtype)

    # Call the custom kernel with the provided arguments and output information
    return binary_coomv_p(
        weights,
        row,
        col,
        v,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        # Provide shape and dtype information for row indices
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        # Provide shape and dtype information for column indices
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
        # Provide shape and dtype information for non-zero values
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and dtype information for the dense vector
        vector_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
    )


binary_coomv_p = XLACustomKernel(
    'binary_coomv',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_coomv``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) COO
sparse matrix-vector multiplication operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata
provided by the high-level wrapper.

The operation computes ``result[i] = sum_j A[i, j] * (v[j] > 0)`` when
``transpose=False`` or ``result[j] = sum_i A[i, j] * (v[i] > 0)`` when
``transpose=True``, where only active (nonzero) events in ``v`` contribute
to the output. This makes the operation efficient for spike-based neural
network simulations.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_coomv_p.available_backends(platform)``,
and the default backend can be configured with ``binary_coomv_p.set_default(platform, backend)``.

See Also
--------
binary_coomv : High-level user-facing function wrapper.
"""
)
binary_coomv_p.def_numba_kernel(_coomv_numba_kernel)
binary_coomv_p.def_warp_kernel(_coomv_warp_kernel)
binary_coomv_p.def_pallas_kernel('gpu', _coomv_pallas_gpu_kernel)
binary_coomv_p.def_pallas_kernel('tpu', _coomv_pallas_tpu_kernel)
binary_coomv_p.def_tvmffi_kernel('gpu', _coomv_tvmffi_kernel)
binary_coomv_p.def_kernel('jax_raw', 'cpu', _binary_coomv_jax_kernel)
binary_coomv_p.def_kernel('jax_raw', 'gpu', _binary_coomv_jax_kernel)
binary_coomv_p.def_kernel('jax_raw', 'tpu', _binary_coomv_jax_kernel)
binary_coomv_p.def_kernel('cusparse', 'gpu', _binary_coomv_cusparse_kernel)
binary_coomv_p.def_jvp_rule2(_coomv_jvp_weights, None, None, _coomv_jvp_vector)
binary_coomv_p.def_transpose_rule(_coomv_transpose_rule)
binary_coomv_p.def_batching_rule(_coomv_batching)
binary_coomv_p.def_call(binary_coomv_p_call)
binary_coomv_p.def_tags('coo', 'binary')
binary_coomv_p.def_benchmark_data(_coomv_benchmark_data)


# =============================================================================
# COO Matrix-Matrix Multiplication (coomm)
# =============================================================================


def _coomm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            if matrix_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mm(weights, row, col, B, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        for j in range(B.shape[1]):
                            if B[row[i], j]:
                                posts[col[i], j] += w
            else:
                @numba.njit(fastmath=True)
                def mm(weights, row, col, B, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        for j in range(B.shape[1]):
                            if B[row[i], j] > 0.:
                                posts[col[i], j] += w
        else:
            # transpose=True, heterogeneous
            if matrix_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mm(weights, row, col, B, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        for j in range(B.shape[1]):
                            if B[row[i], j]:
                                posts[col[i], j] += weights[i]
            else:
                @numba.njit(fastmath=True)
                def mm(weights, row, col, B, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        for j in range(B.shape[1]):
                            if B[row[i], j] > 0.:
                                posts[col[i], j] += weights[i]
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            if matrix_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mm(weights, row, col, B, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        for j in range(B.shape[1]):
                            if B[col[i], j]:
                                posts[row[i], j] += w
            else:
                @numba.njit(fastmath=True)
                def mm(weights, row, col, B, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(row.shape[0]):
                        for j in range(B.shape[1]):
                            if B[col[i], j] > 0.:
                                posts[row[i], j] += w
        else:
            # transpose=False, heterogeneous
            if matrix_info.dtype == jnp.bool_:
                # bool
                @numba.njit(fastmath=True)
                def mm(weights, row, col, B, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        for j in range(B.shape[1]):
                            if B[col[i], j]:
                                posts[row[i], j] += weights[i]
            else:
                @numba.njit(fastmath=True)
                def mm(weights, row, col, B, posts):
                    posts[:] = 0.
                    for i in range(row.shape[0]):
                        for j in range(B.shape[1]):
                            if B[col[i], j] > 0.:
                                posts[row[i], j] += weights[i]

    def kernel(weights, row, col, B):
        return numba_kernel(mm, outs=kwargs['outs'])(weights, row, col, B)

    return kernel


def _coomm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    row_warp_info = jaxinfo_to_warpinfo(row_info)
    col_warp_info = jaxinfo_to_warpinfo(col_info)
    spike_warp_info = jaxinfo_to_warpinfo(matrix_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            if matrix_info.dtype == jnp.bool_:
                # bool
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info
                ):
                    i, j = warp.tid()
                    w = weights[0]
                    if B[row[i], j]:
                        warp.atomic_add(posts, col[i], j, w)
            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info
                ):
                    i, j = warp.tid()
                    w = weights[0]
                    if B[row[i], j] > 0.:
                        warp.atomic_add(posts, col[i], j, w)
        else:
            # transpose=True, heterogeneous
            if matrix_info.dtype == jnp.bool_:
                # bool
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info
                ):
                    i, j = warp.tid()
                    if B[row[i], j]:
                        warp.atomic_add(posts, col[i], j, weights[i])
            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info
                ):
                    i, j = warp.tid()
                    if B[row[i], j] > 0.:
                        warp.atomic_add(posts, col[i], j, weights[i])
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            if matrix_info.dtype == jnp.bool_:
                # bool
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info
                ):
                    i, j = warp.tid()
                    w = weights[0]
                    if B[col[i], j]:
                        warp.atomic_add(posts, row[i], j, w)
            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info
                ):
                    i, j = warp.tid()
                    w = weights[0]
                    if B[col[i], j] > 0.:
                        warp.atomic_add(posts, row[i], j, w)
        else:
            # transpose=False, heterogeneous
            if matrix_info.dtype == jnp.bool_:
                # bool
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info
                ):
                    i, j = warp.tid()
                    if B[col[i], j]:
                        warp.atomic_add(posts, row[i], j, weights[i])
            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info
                ):
                    i, j = warp.tid()
                    if B[col[i], j] > 0.:
                        warp.atomic_add(posts, row[i], j, weights[i])

    def kernel(weights, row, col, B):
        dim = (row_info.shape[0], matrix_info.shape[1])
        out_info = kwargs['outs'][0]
        fn = jax_kernel(mm, launch_dims=dim, num_outputs=1, in_out_argnames=['posts'])
        return fn(weights, row, col, B, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _coomm_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    nnz = row_info.shape[0]
    n = matrix_info.shape[1]
    block_dim = generate_block_dim(nnz)
    block_dim = 32 if block_dim < 32 else block_dim
    block_dim_n = generate_block_dim(n, 512)

    if transpose:
        if weight_info.size == 1:
            # coo.T @ B (homogeneous weights)
            #
            # coo: [m, k]
            # B: [m, n]
            # result: [k, n]
            #
            def mm(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                B_ref,  # [m, n]
                _,  # [k, n]
                posts_ref,  # [k, n]
            ):
                i = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                i_start = i * block_dim
                scalar_w = jnp.asarray(data_ref[0], dtype=posts_ref.dtype)

                def loop_fn(idx, _):
                    elem = i_start + idx
                    valid = elem < nnz
                    row_idx = plgpu.load(row_ref.at[elem], mask=valid, other=0)
                    col_idx = plgpu.load(col_ref.at[elem], mask=valid, other=0)
                    events = plgpu.load(
                        B_ref.at[row_idx, pl.ds(i_col_start, block_dim_n)],
                        mask=col_mask,
                        other=False if B_ref.dtype == jnp.bool_ else 0
                    )
                    if B_ref.dtype == jnp.bool_:
                        event_mask = col_mask & events & valid
                    else:
                        event_mask = col_mask & (events > 0.) & valid
                    data = jnp.full((block_dim_n,), scalar_w, dtype=posts_ref.dtype)
                    plgpu.atomic_add(posts_ref, (col_idx, pl.ds(i_col_start, block_dim_n)), data, mask=event_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        else:
            # coo.T @ B (heterogeneous weights)
            #
            # coo: [m, k]
            # B: [m, n]
            # result: [k, n]
            #
            def mm(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                B_ref,  # [m, n]
                _,  # [k, n]
                posts_ref,  # [k, n]
            ):
                i = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                i_start = i * block_dim

                def loop_fn(idx, _):
                    elem = i_start + idx
                    valid = elem < nnz
                    row_idx = plgpu.load(row_ref.at[elem], mask=valid, other=0)
                    col_idx = plgpu.load(col_ref.at[elem], mask=valid, other=0)
                    w = jnp.asarray(plgpu.load(data_ref.at[elem], mask=valid, other=0), dtype=posts_ref.dtype)
                    events = plgpu.load(
                        B_ref.at[row_idx, pl.ds(i_col_start, block_dim_n)],
                        mask=col_mask,
                        other=False if B_ref.dtype == jnp.bool_ else 0
                    )
                    if B_ref.dtype == jnp.bool_:
                        event_mask = col_mask & events & valid
                    else:
                        event_mask = col_mask & (events > 0.) & valid
                    data = jnp.full((block_dim_n,), w, dtype=posts_ref.dtype)
                    plgpu.atomic_add(posts_ref, (col_idx, pl.ds(i_col_start, block_dim_n)), data, mask=event_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        def kernel(data, row, col, B):
            fn = pl.pallas_call(
                mm,
                grid=(pl.cdiv(nnz, block_dim), pl.cdiv(n, block_dim_n)),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, B, posts)

        return kernel

    else:
        # coo @ B (non-transpose)
        if weight_info.size == 1:
            # coo @ B (homogeneous weights)
            #
            # coo: [m, k]
            # B: [k, n]
            # result: [m, n]
            #
            def mm(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                B_ref,  # [k, n]
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                i_start = i * block_dim
                scalar_w = jnp.asarray(data_ref[0], dtype=posts_ref.dtype)

                def loop_fn(idx, _):
                    elem = i_start + idx
                    valid = elem < nnz
                    row_idx = plgpu.load(row_ref.at[elem], mask=valid, other=0)
                    col_idx = plgpu.load(col_ref.at[elem], mask=valid, other=0)
                    events = plgpu.load(
                        B_ref.at[col_idx, pl.ds(i_col_start, block_dim_n)],
                        mask=col_mask,
                        other=False if B_ref.dtype == jnp.bool_ else 0
                    )
                    if B_ref.dtype == jnp.bool_:
                        event_mask = col_mask & events & valid
                    else:
                        event_mask = col_mask & (events > 0.) & valid
                    data = jnp.full((block_dim_n,), scalar_w, dtype=posts_ref.dtype)
                    plgpu.atomic_add(posts_ref, (row_idx, pl.ds(i_col_start, block_dim_n)), data, mask=event_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        else:
            # coo @ B (heterogeneous weights)
            #
            # coo: [m, k]
            # B: [k, n]
            # result: [m, n]
            #
            def mm(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                B_ref,  # [k, n]
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                i_start = i * block_dim

                def loop_fn(idx, _):
                    elem = i_start + idx
                    valid = elem < nnz
                    row_idx = plgpu.load(row_ref.at[elem], mask=valid, other=0)
                    col_idx = plgpu.load(col_ref.at[elem], mask=valid, other=0)
                    w = jnp.asarray(plgpu.load(data_ref.at[elem], mask=valid, other=0), dtype=posts_ref.dtype)
                    events = plgpu.load(
                        B_ref.at[col_idx, pl.ds(i_col_start, block_dim_n)],
                        mask=col_mask,
                        other=False if B_ref.dtype == jnp.bool_ else 0
                    )
                    if B_ref.dtype == jnp.bool_:
                        event_mask = col_mask & events & valid
                    else:
                        event_mask = col_mask & (events > 0.) & valid
                    data = jnp.full((block_dim_n,), w, dtype=posts_ref.dtype)
                    plgpu.atomic_add(posts_ref, (row_idx, pl.ds(i_col_start, block_dim_n)), data, mask=event_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        def kernel(data, row, col, B):
            fn = pl.pallas_call(
                mm,
                grid=(pl.cdiv(nnz, block_dim), pl.cdiv(n, block_dim_n)),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, B, posts)

        return kernel


def _coomm_pallas_tpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    nnz = row_info.shape[0]
    n = matrix_info.shape[1]
    block_dim = generate_block_dim(nnz)
    block_dim = 32 if block_dim < 32 else block_dim
    block_dim_n = generate_block_dim(n, 512)
    num_nnz_blocks = pl.cdiv(nnz, block_dim)
    out_rows = kwargs['outs'][0].shape[0]

    if weight_info.size == 1:
        def mm(
            data_ref,  # [1]
            row_ref,  # [nnz]
            col_ref,  # [nnz]
            B_ref,  # [m, n] or [k, n]
            _,  # [k, n] or [m, n]
            posts_ref,  # [k, n] or [m, n]
        ):
            i_out = pl.program_id(0)
            i_n = pl.program_id(1)
            i_col_start = i_n * block_dim_n
            col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
            scalar_w = jnp.asarray(data_ref[0], dtype=posts_ref.dtype)

            def loop_fn(i_blk, acc):
                i_start = i_blk * block_dim
                elems = i_start + jnp.arange(block_dim)
                valid = elems < nnz
                safe_elems = jnp.where(valid, elems, 0)
                rows = row_ref[safe_elems]
                cols = col_ref[safe_elems]
                safe_rows = jnp.where(valid, rows, 0)
                safe_cols = jnp.where(valid, cols, 0)

                if transpose:
                    events = B_ref[safe_rows, pl.ds(i_col_start, block_dim_n)]
                    out_idx = cols
                else:
                    events = B_ref[safe_cols, pl.ds(i_col_start, block_dim_n)]
                    out_idx = rows

                if B_ref.dtype == jnp.bool_:
                    active = events
                else:
                    active = events > 0.

                lane_mask = valid & (out_idx == i_out)
                active_mask = lane_mask[:, None] & active & col_mask[None, :]
                return acc + jnp.sum(jnp.where(active_mask, scalar_w, 0), axis=0)

            acc0 = jnp.zeros((block_dim_n,), dtype=posts_ref.dtype)
            acc = jax.lax.fori_loop(0, num_nnz_blocks, loop_fn, acc0)
            pltpu.store(posts_ref.at[i_out, pl.ds(i_col_start, block_dim_n)], acc, mask=col_mask)

    else:
        def mm(
            data_ref,  # [nnz]
            row_ref,  # [nnz]
            col_ref,  # [nnz]
            B_ref,  # [m, n] or [k, n]
            _,  # [k, n] or [m, n]
            posts_ref,  # [k, n] or [m, n]
        ):
            i_out = pl.program_id(0)
            i_n = pl.program_id(1)
            i_col_start = i_n * block_dim_n
            col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

            def loop_fn(i_blk, acc):
                i_start = i_blk * block_dim
                elems = i_start + jnp.arange(block_dim)
                valid = elems < nnz
                safe_elems = jnp.where(valid, elems, 0)
                rows = row_ref[safe_elems]
                cols = col_ref[safe_elems]
                weights = jnp.asarray(data_ref[safe_elems], dtype=posts_ref.dtype)
                safe_rows = jnp.where(valid, rows, 0)
                safe_cols = jnp.where(valid, cols, 0)

                if transpose:
                    events = B_ref[safe_rows, pl.ds(i_col_start, block_dim_n)]
                    out_idx = cols
                else:
                    events = B_ref[safe_cols, pl.ds(i_col_start, block_dim_n)]
                    out_idx = rows

                if B_ref.dtype == jnp.bool_:
                    active = events
                else:
                    active = events > 0.

                lane_mask = valid & (out_idx == i_out)
                active_mask = lane_mask[:, None] & active & col_mask[None, :]
                weighted = jnp.where(active_mask, weights[:, None], 0)
                return acc + jnp.sum(weighted, axis=0)

            acc0 = jnp.zeros((block_dim_n,), dtype=posts_ref.dtype)
            acc = jax.lax.fori_loop(0, num_nnz_blocks, loop_fn, acc0)
            pltpu.store(posts_ref.at[i_out, pl.ds(i_col_start, block_dim_n)], acc, mask=col_mask)

    def kernel(data, row, col, B):
        fn = pl.pallas_call(
            mm,
            grid=(out_rows, pl.cdiv(n, block_dim_n)),
            input_output_aliases={4: 0},
            out_shape=kwargs['outs'],
            backend='mosaic_tpu',
        )
        posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
        return fn(data, row, col, B, posts)

    return kernel


def _binary_coomm_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    shape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for binary (event-driven) COO matrix-matrix multiplication (all platforms)."""
    m, k = shape
    is_homo = (weight_info.size == 1)
    is_bool = (matrix_info.dtype == jnp.bool_)
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, row, col, B):
            B_rows = B[row]  # [nse, n]
            events = B_rows.astype(out_dtype) if is_bool else (B_rows > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights[:, None]
            return (jnp.zeros((k, B.shape[1]), dtype=out_dtype).at[col].add(w * events),)
    else:
        def kernel(weights, row, col, B):
            B_rows = B[col]  # [nse, n]
            events = B_rows.astype(out_dtype) if is_bool else (B_rows > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights[:, None]
            return (jnp.zeros((m, B.shape[1]), dtype=out_dtype).at[row].add(w * events),)

    return kernel


def _binary_coomm_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    shape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed kernel for binary COO SpMM via jax.experimental.sparse (GPU only)."""
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    is_bool = (matrix_info.dtype == jnp.bool_)
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        if is_homo:
            def kernel(weights, row, col, B):
                events = B.astype(out_dtype) if is_bool else (B > 0.).astype(out_dtype)
                ones = jnp.ones_like(row, dtype=out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((ones, idx), shape=(m, k))
                return ((mat.T @ events) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, row, col, B):
                events = B.astype(out_dtype) if is_bool else (B > 0.).astype(out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((weights.astype(out_dtype), idx), shape=(m, k))
                return (mat.T @ events,)
    else:
        if is_homo:
            def kernel(weights, row, col, B):
                events = B.astype(out_dtype) if is_bool else (B > 0.).astype(out_dtype)
                ones = jnp.ones_like(row, dtype=out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((ones, idx), shape=(m, k))
                return ((mat @ events) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, row, col, B):
                events = B.astype(out_dtype) if is_bool else (B > 0.).astype(out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((weights.astype(out_dtype), idx), shape=(m, k))
                return (mat @ events,)
    return kernel


def _coomm_tvmffi_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    """TVM FFI CUDA kernel for binary COO SpMM.

    Dispatches to one of the ``binary_coomm_{variant}_{nt,t}`` kernels compiled
    from ``binary.cu`` via ``register_tvm_cuda_from_file``.

    Kernel variant selection (based on n = number of output columns):
    - CT (Column-Tiled, n ≤ 64): One warp per block serially iterates over
      32 NNZ entries while all 32 threads process a 32-column output tile.
      Uses ``__ballot_sync`` to skip atomics for inactive spike tiles.
      Block=(32,), Grid=(ceil(nnz/32), ceil(n/32)).
    - WPE (Warp-Per-Entry, n > 64): Each of 8 warps in a 256-thread block
      handles a single NNZ entry and 32 consecutive output columns.
      Block=(256,), Grid=(ceil(nnz/8), ceil(n/32)).

    Direction suffix: ``_nt`` (transpose=False) or ``_t`` (transpose=True).
    Weight dtype suffix: ``_f32``, ``_f64``, ``_f16``, or ``_bf16``.
    Spike type suffix: ``_bool`` (int8) or ``_float`` (float32).
    Homo vs. hetero: detected at runtime from ``data.size(0) == 1``.

    The output buffer is zero-initialized inside the CUDA entry function
    (via ``cudaMemsetAsync``) before the atomic-scatter kernel runs.
    """
    register_tvm_cuda_from_file(
        module='coo_binary_coomm',
        source=Path(__file__).parent.joinpath('binary_coomm.cu'),
    )

    out_info = kwargs['outs']
    spk_suffix = '_bool' if matrix_info.dtype == jnp.bool_ else '_float'

    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    direction = '_t' if transpose else '_nt'

    # Dispatch heuristic: CT is better for small n (serial NNZ loop amortised
    # across many CUDA blocks in the nnz dimension); WPE is better for large n
    # (each warp independently covers one NNZ entry with no serial loop).
    n = matrix_info.shape[1]
    variant = 'ct' if n <= 64 else 'wpe'
    kernel_name = f'coo_binary_coomm.binary_coomm_{variant}{direction}{wt_sfx}{spk_suffix}'

    def kernel(weights, row, col, B):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, row, col, B)

    return kernel


def _coomm_jvp_left(data_dot, data, row, col, B, *, shape, transpose, **kwargs):
    return [coomm(data_dot, row, col, B, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _coomm_jvp_right(B_dot, data, row, col, B, *, shape, transpose, **kwargs):
    return [coomm(data, row, col, B_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _coomm_transpose_rule(ct, data, row, col, B, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(row)
    assert not ad.is_undefined_primal(col)
    ct = ct[0]

    if ad.is_undefined_primal(B):
        if type(ct) is ad.Zero:
            dB = ad.Zero(B)
        else:
            dB = coomm(data, row, col, ct, shape=shape, transpose=not transpose, backend=kwargs['backend'])
        return data, row, col, dB
    else:
        if type(ct) is ad.Zero:
            d_data = ad.Zero(data)
        else:
            d_data = sddmm_coo_indices(ct, B, row, col).data
        return d_data, row, col, B


def _coomm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = binary_coomm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, batch_size, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = binary_coomm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 2):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = binary_coomm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], n, batch_size])
        return [r], [2]

    else:
        return general_batching_rule(binary_coomm_p_call, args, axes, **kwargs)


def _coomm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                nnz = max(1, int(n_pre * n_post * prob))
                row = np.random.randint(0, n_pre, nnz, dtype=np.int32)
                col = np.random.randint(0, n_post, nnz, dtype=np.int32)
                weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(nnz, dtype=dtype)
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    B = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    B = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, jnp.asarray(row), jnp.asarray(col), B),
                        {'shape': (n_pre, n_post), 'transpose': transpose}
                    )
                )
    return configs


def binary_coomm_p_call(
    weights,
    row,
    col,
    B,
    *,
    shape: Sequence[int],
    transpose: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for event-driven COO sparse matrix-matrix multiplication.

    Validates inputs, constructs output metadata, and dispatches to the
    registered ``XLACustomKernel`` (``binary_coomm_p``) which selects a
    backend-specific kernel (Numba, Pallas GPU, or Pallas TPU).

    Unlike :func:`binary_coomm`, this function does **not** handle physical
    units and returns a raw list of JAX arrays.

    Parameters
    ----------
    weights : jax.Array
        Non-zero values of the sparse matrix.  Must be a floating-point
        scalar (shape ``(1,)``) for homogeneous weights or a 1-D array
        of length ``nnz`` for heterogeneous weights.
    row : jax.Array
        1-D integer array of row indices with length ``nnz``.
    col : jax.Array
        1-D integer array of column indices with length ``nnz``.
    B : jax.Array
        Dense event matrix of shape ``(shape[0], n)`` when
        ``transpose=True`` or ``(shape[1], n)`` when ``transpose=False``.
        Entries are treated as active when ``True`` (boolean) or ``> 0``
        (numeric).
    shape : Sequence[int]
        The ``(m, k)`` shape of the logical sparse matrix.
    transpose : bool
        If ``True``, compute ``A^T @ B`` instead of ``A @ B``.
    backend : str or None, optional
        Compute backend override (``'numba'``, ``'pallas'``).
        When ``None`` the backend is selected automatically.

    Returns
    -------
    list of jax.Array
        A single-element list containing the result matrix.  Shape is
        ``(shape[0], n)`` when ``transpose=False`` or ``(shape[1], n)``
        when ``transpose=True``, with dtype matching ``weights``.

    Raises
    ------
    ValueError
        If ``row`` or ``col`` are not 1-D arrays, have different lengths,
        ``B`` is not 2-D, ``weights`` is not scalar or 1-D, weights
        length is neither 1 nor ``nnz``, or ``B`` row count is
        incompatible with ``shape`` and ``transpose``.
    AssertionError
        If ``weights`` dtype is not a floating-point type.

    See Also
    --------
    binary_coomm : High-level wrapper with physical unit support.

    Notes
    -----
    When ``nnz == 0`` the function short-circuits and returns a zero
    matrix without dispatching to any kernel.

    The function registers JVP rules, transpose rules, and batching rules
    on the underlying ``binary_coomm_p`` primitive so that it integrates
    with JAX's autodiff and ``vmap`` transformations.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._coo.binary import binary_coomm_p_call
        >>> weights = jnp.array([1.0, 2.0, 3.0])
        >>> row = jnp.array([0, 1, 2])
        >>> col = jnp.array([2, 0, 1])
        >>> B = jnp.array([[True, False],
        ...                [False, True],
        ...                [True, True]])
        >>> result = binary_coomm_p_call(
        ...     weights, row, col, B, shape=(3, 3), transpose=False
        ... )
        >>> result[0]  # the output matrix
    """
    row = jnp.asarray(row)
    col = jnp.asarray(col)
    B = jnp.asarray(B)
    if row.ndim != 1 or col.ndim != 1:
        raise ValueError(f'`row` and `col` must be 1D arrays, got row.ndim={row.ndim}, col.ndim={col.ndim}.')
    if row.shape[0] != col.shape[0]:
        raise ValueError(f'`row` and `col` must have the same length, got {row.shape[0]} and {col.shape[0]}.')
    if B.ndim != 2:
        raise ValueError(f'`B` must be a 2D array, got ndim={B.ndim}.')

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    else:
        weights = jnp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError(f'`weights` must be a scalar or 1D array, got ndim={weights.ndim}.')
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    nnz = row.shape[0]
    if weights.shape[0] not in (1, nnz):
        raise ValueError(f'`weights` length must be 1 or nnz={nnz}, got {weights.shape[0]}.')

    expected_b_rows = shape[0] if transpose else shape[1]
    if B.shape[0] != expected_b_rows:
        raise ValueError(f'`B` has incompatible shape {B.shape} for shape={tuple(shape)}, transpose={transpose}.')

    out_rows = shape[1] if transpose else shape[0]
    if nnz == 0:
        return [jnp.zeros((out_rows, B.shape[1]), dtype=weights.dtype)]

    out_info = jax.ShapeDtypeStruct([out_rows, B.shape[1]], weights.dtype)
    # Call the custom kernel with the provided arguments and output information
    return binary_coomm_p(
        weights,
        row,
        col,
        B,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


binary_coomm_p = XLACustomKernel(
    'binary_coomm',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_coomm``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) COO
sparse matrix-matrix multiplication operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata
provided by the high-level wrapper.

The operation computes ``result[i, n] = sum_j A[i, j] * (B[j, n] > 0)`` when
``transpose=False`` or ``result[j, n] = sum_i A[i, j] * (B[i, n] > 0)`` when
``transpose=True``, where only active (nonzero) events in the dense matrix
``B`` contribute to the output. This is efficient for processing batches of
spike events in neural network simulations.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_coomm_p.available_backends(platform)``,
and the default backend can be configured with ``binary_coomm_p.set_default(platform, backend)``.

See Also
--------
binary_coomm : High-level user-facing function wrapper.
"""
)
binary_coomm_p.def_numba_kernel(_coomm_numba_kernel)
binary_coomm_p.def_warp_kernel(_coomm_warp_kernel)
binary_coomm_p.def_pallas_kernel('gpu', _coomm_pallas_gpu_kernel)
binary_coomm_p.def_pallas_kernel('tpu', _coomm_pallas_tpu_kernel)
binary_coomm_p.def_tvmffi_kernel('gpu', _coomm_tvmffi_kernel)
binary_coomm_p.def_kernel('jax_raw', 'cpu', _binary_coomm_jax_kernel)
binary_coomm_p.def_kernel('jax_raw', 'gpu', _binary_coomm_jax_kernel)
binary_coomm_p.def_kernel('jax_raw', 'tpu', _binary_coomm_jax_kernel)
binary_coomm_p.def_kernel('cusparse', 'gpu', _binary_coomm_cusparse_kernel)
binary_coomm_p.def_jvp_rule2(_coomm_jvp_left, None, None, _coomm_jvp_right)
binary_coomm_p.def_transpose_rule(_coomm_transpose_rule)
binary_coomm_p.def_batching_rule(_coomm_batching)
binary_coomm_p.def_call(binary_coomm_p_call)
binary_coomm_p.def_tags('coo', 'binary')
binary_coomm_p.def_benchmark_data(_coomm_benchmark_data)
