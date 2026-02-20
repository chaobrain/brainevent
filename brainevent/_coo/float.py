# Copyright 2024- BrainX Ecosystem Limited. All Rights Reserved.
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
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, namescope
from brainevent._op import XLACustomKernel, general_batching_rule, numba_kernel, register_tvm_cuda_from_file
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Row, Col, MatrixShape

__all__ = [
    "coomv",
    "coomv_p",
    "coomm",
    "coomm_p",
]


@namescope(static_argnames=("shape", "transpose"))
def coomv(
    data: Data,
    row: Row,
    col: Col,
    vector: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """Perform COO sparse matrix-vector multiplication.

    Computes the product of a sparse matrix stored in COO (Coordinate)
    format and a dense vector.

    With ``transpose=False`` the operation computes:

        ``y[i] = sum_{k} A[i, k] * v[k]``

    With ``transpose=True``:

        ``y[k] = sum_{i} A[i, k] * v[i]``

    where ``A`` is the sparse matrix defined by (*data*, *row*, *col*).

    Parameters
    ----------
    data : jax.Array or Quantity
        Non-zero values of the sparse matrix.  Either a scalar (shape
        ``(1,)`` for homogeneous weights) or a 1-D array of length
        ``nnz`` (heterogeneous weights).
    row : jax.Array
        1-D int array of row indices, length ``nnz``.
    col : jax.Array
        1-D int array of column indices, length ``nnz``.
    vector : jax.Array or Quantity
        Dense input vector.  Shape ``(shape[1],)`` when
        ``transpose=False``, or ``(shape[0],)`` when ``transpose=True``.
    shape : tuple of int
        Logical ``(m, k)`` shape of the sparse matrix.
    transpose : bool, optional
        If ``True``, multiply by ``A.T``.  Default is ``False``.
    backend : str or None, optional
        Compute backend (e.g. ``'numba'``, ``'pallas'``).
        ``None`` selects the default.

    Returns
    -------
    jax.Array or Quantity
        Result vector.  Shape ``(shape[0],)`` when ``transpose=False``,
        or ``(shape[1],)`` when ``transpose=True``.  Carries the product
        of the units of *data* and *vector* if applicable.

    See Also
    --------
    coomm : COO sparse matrix-matrix multiplication.
    binary_coomv : Event-driven (binary) COO matrix-vector multiplication.

    Notes
    -----
    The kernel iterates over all ``nnz`` stored elements and, for each
    triplet ``(data[s], row[s], col[s])``, accumulates
    ``data[s] * vector[col[s]]`` into ``y[row[s]]`` (forward) or
    ``data[s] * vector[row[s]]`` into ``y[col[s]]`` (transpose).

    When *data* is a scalar the same weight is used for every non-zero,
    enabling a more compact kernel.

    Physical units attached via ``brainunit`` are split before the
    computation and re-applied to the result.

    This function supports automatic differentiation (JVP and transpose
    rules), ``vmap`` batching, and multiple hardware backends.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import coomv
        >>> data = jnp.array([1.0, 2.0, 3.0])
        >>> row = jnp.array([0, 1, 2])
        >>> col = jnp.array([1, 0, 2])
        >>> v = jnp.array([1.0, 2.0, 3.0])
        >>> coomv(data, row, col, v, shape=(3, 3))
    """
    data, unitd = u.split_mantissa_unit(data)
    vector, unitv = u.split_mantissa_unit(vector)
    res = coomv_p_call(
        data,
        row,
        col,
        vector,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose"))
def coomm(
    data: Data,
    row: Row,
    col: Col,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
):
    """Perform COO sparse matrix-matrix multiplication.

    Computes the product of a sparse matrix stored in COO format and a
    dense matrix ``B``.

    With ``transpose=False``:

        ``Y[i, n] = sum_{k} A[i, k] * B[k, n]``

    With ``transpose=True``:

        ``Y[k, n] = sum_{i} A[i, k] * B[i, n]``

    Parameters
    ----------
    data : jax.Array or Quantity
        Non-zero values of the sparse matrix.  Either a scalar (shape
        ``(1,)`` for homogeneous weights) or a 1-D array of length
        ``nnz``.
    row : jax.Array
        1-D int array of row indices, length ``nnz``.
    col : jax.Array
        1-D int array of column indices, length ``nnz``.
    B : jax.Array or Quantity
        Dense right-hand matrix.  Shape ``(shape[1], n)`` when
        ``transpose=False``, or ``(shape[0], n)`` when
        ``transpose=True``.
    shape : tuple of int
        Logical ``(m, k)`` shape of the sparse matrix.
    transpose : bool, optional
        If ``True``, multiply by ``A.T``.  Default is ``False``.
    backend : str or None, optional
        Compute backend.  ``None`` selects the default.

    Returns
    -------
    jax.Array or Quantity
        Result matrix.  Shape ``(shape[0], n)`` when ``transpose=False``,
        or ``(shape[1], n)`` when ``transpose=True``.

    See Also
    --------
    coomv : COO sparse matrix-vector multiplication.
    binary_coomm : Event-driven (binary) COO matrix-matrix multiplication.

    Notes
    -----
    The kernel iterates over all ``nnz`` stored elements and, for each
    triplet ``(data[s], row[s], col[s])``, adds
    ``data[s] * B[col[s], :]`` into ``Y[row[s], :]`` (forward) or
    ``data[s] * B[row[s], :]`` into ``Y[col[s], :]`` (transpose).

    Physical units from ``brainunit`` are handled transparently.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import coomm
        >>> data = jnp.array([1.0, 2.0])
        >>> row = jnp.array([0, 1])
        >>> col = jnp.array([1, 0])
        >>> B = jnp.ones((2, 3))
        >>> coomm(data, row, col, B, shape=(2, 2))
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = coomm_p_call(data, row, col, B, shape=shape, transpose=transpose, backend=backend)[0]
    return u.maybe_decimal(res * (unitd * unitb))


# =============================================================================
# COO Matrix-Vector Multiplication (coomv)
# =============================================================================


def _coomv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    posts[col[i]] += w * v[row[i]]
        else:
            # transpose=True, heterogeneous
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    posts[col[i]] += weights[i] * v[row[i]]
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    posts[row[i]] += w * v[col[i]]
        else:
            # transpose=False, heterogeneous
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    posts[row[i]] += weights[i] * v[col[i]]

    def kernel(weights, row, col, v):
        # row_ptr is unused for numba backend; present for signature parity
        return numba_kernel(mv, outs=kwargs['outs'])(weights, row, col, v)

    return kernel


def _coomv_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    nnz = row_info.shape[0]
    block_dim = generate_block_dim(nnz)
    block_dim = 32 if block_dim < 32 else block_dim

    if transpose:
        if weight_info.size == 1:
            # coo.T @ v (homogeneous weights)
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
                vals = plgpu.load(vector_ref.at[rows], mask=mask, other=0)
                data = jnp.asarray(vals, dtype=posts_ref.dtype) * jnp.asarray(data_ref[0], dtype=posts_ref.dtype)
                plgpu.atomic_add(posts_ref, cols, data, mask=mask)

        else:
            # coo.T @ v (heterogeneous weights)
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
                vals = plgpu.load(vector_ref.at[rows], mask=mask, other=0)
                data = jnp.asarray(vals, dtype=posts_ref.dtype) * jnp.asarray(weights, dtype=posts_ref.dtype)
                plgpu.atomic_add(posts_ref, cols, data, mask=mask)

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
        if weight_info.size == 1:
            # coo @ v (homogeneous weights)
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
                vals = plgpu.load(vector_ref.at[cols], mask=mask, other=0)
                data = jnp.asarray(vals, dtype=posts_ref.dtype) * jnp.asarray(data_ref[0], dtype=posts_ref.dtype)
                plgpu.atomic_add(posts_ref, rows, data, mask=mask)

        else:
            # coo @ v (heterogeneous weights)
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
                vals = plgpu.load(vector_ref.at[cols], mask=mask, other=0)
                data = jnp.asarray(vals, dtype=posts_ref.dtype) * jnp.asarray(weights, dtype=posts_ref.dtype)
                plgpu.atomic_add(posts_ref, rows, data, mask=mask)

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
                    vals = vector_ref[safe_rows]
                    out_idx = cols
                else:
                    vals = vector_ref[safe_cols]
                    out_idx = rows

                contrib = jnp.asarray(vals, dtype=posts_ref.dtype) * scalar_w
                lane_mask = valid & (out_idx == i_out)
                return acc + jnp.sum(jnp.where(lane_mask, contrib, 0))

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
                    vals = vector_ref[safe_rows]
                    out_idx = cols
                else:
                    vals = vector_ref[safe_cols]
                    out_idx = rows

                contrib = jnp.asarray(vals, dtype=posts_ref.dtype) * weights
                lane_mask = valid & (out_idx == i_out)
                return acc + jnp.sum(jnp.where(lane_mask, contrib, 0))

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


def _coomv_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    shape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for COO sparse matrix-vector multiplication (all platforms)."""
    m, k = shape
    is_homo = (weight_info.size == 1)
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, row, col, vector):
            v_vals = vector[row].astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(k, dtype=out_dtype).at[col].add(w * v_vals),)
    else:
        def kernel(weights, row, col, vector):
            v_vals = vector[col].astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(m, dtype=out_dtype).at[row].add(w * v_vals),)

    return kernel


def _coomv_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    shape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed kernel for COO SpMV via jax.experimental.sparse (GPU only)."""
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        if is_homo:
            def kernel(weights, row, col, vector):
                ones = jnp.ones_like(row, dtype=out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((ones, idx), shape=(m, k))
                return ((mat.T @ vector.astype(out_dtype)) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, row, col, vector):
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((weights.astype(out_dtype), idx), shape=(m, k))
                return (mat.T @ vector.astype(out_dtype),)
    else:
        if is_homo:
            def kernel(weights, row, col, vector):
                ones = jnp.ones_like(row, dtype=out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((ones, idx), shape=(m, k))
                return ((mat @ vector.astype(out_dtype)) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, row, col, vector):
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((weights.astype(out_dtype), idx), shape=(m, k))
                return (mat @ vector.astype(out_dtype),)
    return kernel


def _coomv_tvmffi_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    """TVM FFI CUDA kernel for float COO SpMV.

    Dispatches to one of the ``coomv_atomic_{nt,t}`` kernels compiled from
    ``float.cu`` via ``register_tvm_cuda_from_file``.

    Both directions use a grid-stride atomic-scatter kernel (256 threads/block):

    * NT (transpose=False): ``out[row[k]] += data[k] * v[col[k]]``
    * T  (transpose=True):  ``out[col[k]] += data[k] * v[row[k]]``

    Weight dtype suffix: ``_f32``, ``_f64``, ``_f16``, or ``_bf16``.
    Homo/hetero detection is done at runtime inside the CUDA entry function
    via ``data.size(0) == 1``; no separate kernel variants are needed.

    The output buffer is zero-initialized inside the CUDA entry function via
    ``cudaMemsetAsync`` before the kernel runs.

    Notes
    -----
    v must have the same dtype as weights.  The Python caller (``coomv_p_call``)
    ensures both are the same dtype before dispatch.
    """
    register_tvm_cuda_from_file(
        module='coo_float',
        source=Path(__file__).parent.joinpath('float.cu'),
    )

    out_info = kwargs['outs']
    _dtype_sfx = {
        jnp.dtype('float16'):  '_f16',
        jnp.dtype('float32'):  '_f32',
        jnp.dtype('float64'):  '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    direction = '_t' if transpose else '_nt'
    kernel_name = f'coo_float.coomv_atomic{direction}{wt_sfx}'

    def kernel(weights, row, col, v):
        v_cast = v.astype(weight_info.dtype) if v.dtype != weight_info.dtype else v
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, row, col, v_cast)

    return kernel


def _coomv_jvp_vector(vector_dot, data, row, col, vector, *, shape, transpose, **kwargs):
    return coomv_p_call(data, row, col, vector_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _coomv_jvp_weights(data_dot, data, row, col, vector, *, shape, transpose, **kwargs):
    return coomv_p_call(data_dot, row, col, vector, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _coomv_transpose_rule(ct, data, row, col, v, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(row)
    assert not ad.is_undefined_primal(col)
    ct = ct[0]

    if ad.is_undefined_primal(v):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(v)
        else:
            ct_events = coomv(data, row, col, ct, shape=shape, transpose=not transpose, backend=kwargs['backend'])
        return data, row, col, ct_events
    else:
        v = jnp.asarray(v)
        if data.aval.shape[0] == 1:  # scalar
            ct_values = coomv_p_call(
                jnp.ones(1, dtype=data.aval.dtype),
                row, col, v,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )[0]
            ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
        else:
            ct_values = v[row] * ct[col] if transpose else v[col] * ct[row]
        return ct_values, row, col, v


def _coomv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = coomm_p_call(
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
        r = coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]

    else:
        return general_batching_rule(coomv_p_call, args, axes, **kwargs)


def _coomv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            nnz = max(1, int(n_pre * n_post * prob))
            row = np.random.randint(0, n_pre, nnz, dtype=np.int32)
            col = np.random.randint(0, n_post, nnz, dtype=np.int32)
            weights = jnp.ones(1, dtype=dtype) if homo else jnp.asarray(np.random.randn(nnz), dtype=dtype)
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, jnp.asarray(row), jnp.asarray(col), vector),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def coomv_p_call(
    weights,
    row,
    col,
    v,
    *,
    shape: Sequence[int],
    transpose: bool,
    backend: Optional[str] = None,
):
    """Low-level primitive call for COO sparse matrix-vector multiplication.

    Validates inputs, normalises *weights* to at least 1-D, and dispatches
    the ``coomv_p`` XLA custom kernel.

    Parameters
    ----------
    weights : jax.Array
        Non-zero values (unitless mantissa).  Scalar or 1-D of length
        ``nnz``.
    row : jax.Array
        1-D int array of row indices, length ``nnz``.
    col : jax.Array
        1-D int array of column indices, length ``nnz``.
    v : jax.Array
        Dense input vector.
    shape : Sequence[int]
        Logical ``(m, k)`` matrix shape.
    transpose : bool
        Whether to use the transposed matrix.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    tuple of jax.Array
        Single-element tuple containing the result vector.

    Raises
    ------
    ValueError
        If *row* or *col* are not 1-D, have different lengths, or *v*
        is not 1-D.

    See Also
    --------
    coomv : Public API with unit handling.
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

    out_info = (
        jax.ShapeDtypeStruct([out_len], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([out_len], weights.dtype)
    )

    return coomv_p(
        weights,
        row,
        col,
        v,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
        vector_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
    )


coomv_p = XLACustomKernel(
    'coomv',
    doc="""
Low-level XLA custom-kernel primitive for ``coomv``.

This ``XLACustomKernel`` instance dispatches the COO sparse matrix-vector
multiplication with floating-point weights operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata
provided by the high-level wrapper.

The operation computes ``y[i] = sum_k A[i, k] * v[k]`` when
``transpose=False`` or ``y[k] = sum_i A[i, k] * v[i]`` when
``transpose=True``, where ``A`` is a sparse matrix defined by (data, row,
col) in COO format and ``v`` is a dense vector. This is the standard
sparse matrix-vector multiplication with full floating-point arithmetic.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``coomv_p.available_backends(platform)``,
and the default backend can be configured with ``coomv_p.set_default(platform, backend)``.

See Also
--------
coomv : High-level user-facing function wrapper.
"""
)
coomv_p.def_numba_kernel(_coomv_numba_kernel)
coomv_p.def_pallas_kernel('gpu', _coomv_pallas_gpu_kernel)
coomv_p.def_pallas_kernel('tpu', _coomv_pallas_tpu_kernel)
coomv_p.def_tvmffi_kernel('gpu', _coomv_tvmffi_kernel)
coomv_p.def_kernel('jax_raw', 'cpu', _coomv_jax_kernel)
coomv_p.def_kernel('jax_raw', 'gpu', _coomv_jax_kernel)
coomv_p.def_kernel('jax_raw', 'tpu', _coomv_jax_kernel)
coomv_p.def_kernel('cusparse', 'gpu', _coomv_cusparse_kernel)
coomv_p.def_jvp_rule2(_coomv_jvp_weights, None, None, _coomv_jvp_vector)
coomv_p.def_transpose_rule(_coomv_transpose_rule)
coomv_p.def_batching_rule(_coomv_batching)
coomv_p.def_call(coomv_p_call)
coomv_p.def_tags('coo', 'float')
coomv_p.def_benchmark_data(_coomv_benchmark_data)


# =============================================================================
# COO Matrix-Matrix Multiplication (coomm)
# =============================================================================


def _coomm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    posts[col[i], :] += w * B[row[i], :]
        else:
            # transpose=True, heterogeneous
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    posts[col[i], :] += weights[i] * B[row[i], :]
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    posts[row[i], :] += w * B[col[i], :]
        else:
            # transpose=False, heterogeneous
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    posts[row[i], :] += weights[i] * B[col[i], :]

    def kernel(weights, row, col, B):
        return numba_kernel(mm, outs=kwargs['outs'])(weights, row, col, B)

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
                w = jnp.asarray(data_ref[0], dtype=posts_ref.dtype)

                def loop_fn(idx, _):
                    elem = i_start + idx
                    valid = elem < nnz
                    row_idx = plgpu.load(row_ref.at[elem], mask=valid, other=0)
                    col_idx = plgpu.load(col_ref.at[elem], mask=valid, other=0)
                    vals = plgpu.load(B_ref.at[row_idx, pl.ds(i_col_start, block_dim_n)], mask=col_mask, other=0)
                    data = jnp.asarray(vals, dtype=posts_ref.dtype) * w
                    lane_mask = col_mask & valid
                    plgpu.atomic_add(posts_ref, (col_idx, pl.ds(i_col_start, block_dim_n)), data, mask=lane_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        else:
            # coo.T @ B (heterogeneous weights)
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
                    vals = plgpu.load(B_ref.at[row_idx, pl.ds(i_col_start, block_dim_n)], mask=col_mask, other=0)
                    data = jnp.asarray(vals, dtype=posts_ref.dtype) * w
                    lane_mask = col_mask & valid
                    plgpu.atomic_add(posts_ref, (col_idx, pl.ds(i_col_start, block_dim_n)), data, mask=lane_mask)

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
                w = jnp.asarray(data_ref[0], dtype=posts_ref.dtype)

                def loop_fn(idx, _):
                    elem = i_start + idx
                    valid = elem < nnz
                    row_idx = plgpu.load(row_ref.at[elem], mask=valid, other=0)
                    col_idx = plgpu.load(col_ref.at[elem], mask=valid, other=0)
                    vals = plgpu.load(B_ref.at[col_idx, pl.ds(i_col_start, block_dim_n)], mask=col_mask, other=0)
                    data = jnp.asarray(vals, dtype=posts_ref.dtype) * w
                    lane_mask = col_mask & valid
                    plgpu.atomic_add(posts_ref, (row_idx, pl.ds(i_col_start, block_dim_n)), data, mask=lane_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        else:
            # coo @ B (heterogeneous weights)
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
                    vals = plgpu.load(B_ref.at[col_idx, pl.ds(i_col_start, block_dim_n)], mask=col_mask, other=0)
                    data = jnp.asarray(vals, dtype=posts_ref.dtype) * w
                    lane_mask = col_mask & valid
                    plgpu.atomic_add(posts_ref, (row_idx, pl.ds(i_col_start, block_dim_n)), data, mask=lane_mask)

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
                    vals = B_ref[safe_rows, pl.ds(i_col_start, block_dim_n)]
                    out_idx = cols
                else:
                    vals = B_ref[safe_cols, pl.ds(i_col_start, block_dim_n)]
                    out_idx = rows

                lane_mask = valid & (out_idx == i_out)
                masked_vals = jnp.where(lane_mask[:, None] & col_mask[None, :], vals, 0)
                return acc + jnp.sum(jnp.asarray(masked_vals, dtype=posts_ref.dtype), axis=0) * scalar_w

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
                    vals = B_ref[safe_rows, pl.ds(i_col_start, block_dim_n)]
                    out_idx = cols
                else:
                    vals = B_ref[safe_cols, pl.ds(i_col_start, block_dim_n)]
                    out_idx = rows

                lane_mask = valid & (out_idx == i_out)
                masked_vals = jnp.where(lane_mask[:, None] & col_mask[None, :], vals, 0)
                return acc + jnp.sum(jnp.asarray(masked_vals, dtype=posts_ref.dtype) * weights[:, None], axis=0)

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


def _coomm_tvmffi_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    """TVM FFI CUDA kernel for float COO SpMM.

    Dispatches to one of the ``coomm_{variant}_{nt,t}`` kernels compiled
    from ``float.cu`` via ``register_tvm_cuda_from_file``.

    Kernel variant selection (based on n = number of output columns):
    - CT (Column-Tiled, n ≤ 64): One warp per block serializes over 32 NNZ
      entries while all 32 threads cover 32 output columns in parallel.
      Block=(32,), Grid=(ceil(nnz/32), ceil(n/32)).
    - WPE (Warp-Per-Entry, n > 64): Each of 8 warps in a 256-thread block
      handles a single NNZ entry × 32 consecutive output columns.
      Block=(256,), Grid=(ceil(nnz/8), ceil(n/32)).

    Direction suffix: ``_nt`` (transpose=False) or ``_t`` (transpose=True).
    Weight dtype suffix: ``_f32``, ``_f64``, ``_f16``, or ``_bf16``.

    The output buffer is zero-initialized inside the CUDA entry function
    (via ``cudaMemsetAsync``) before the atomic-scatter kernel runs.

    Notes
    -----
    B must have the same dtype as weights.  The Python caller (``coomm_p_call``)
    ensures B is promoted to ``weights.dtype`` before dispatch when necessary.
    """
    register_tvm_cuda_from_file(
        module='coo_float',
        source=Path(__file__).parent.joinpath('float.cu'),
    )

    out_info = kwargs['outs']
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    direction = '_t' if transpose else '_nt'

    # CT is better for small n (serial NNZ loop amortized over many CUDA blocks
    # in the nnz dimension); WPE is better for large n (maximum parallelism).
    n = matrix_info.shape[1]
    variant = 'ct' if n <= 64 else 'wpe'
    kernel_name = f'coo_float.coomm_{variant}{direction}{wt_sfx}'

    def kernel(weights, row, col, B):
        # Cast B to weights.dtype so the CUDA kernel receives matching types.
        B_cast = B.astype(weight_info.dtype) if B.dtype != weight_info.dtype else B
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, row, col, B_cast)

    return kernel


def _coomm_jvp_left(data_dot, data, row, col, B, *, shape, transpose, **kwargs):
    return coomm_p_call(data_dot, row, col, B, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _coomm_jvp_right(B_dot, data, row, col, B, *, shape, transpose, **kwargs):
    return coomm_p_call(data, row, col, B_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


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
        r = coomm_p_call(
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
        r = coomm_p_call(
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
        r = coomm_p_call(
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
        return general_batching_rule(coomm_p_call, args, axes, **kwargs)


def _coomm_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    shape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for COO sparse matrix-matrix multiplication (all platforms)."""
    m, k = shape
    is_homo = (weight_info.size == 1)
    out_dtype = kwargs['outs'][0].dtype
    out_shape = kwargs['outs'][0].shape  # (out_rows, n)

    if transpose:
        def kernel(weights, row, col, B):
            B_rows = B[row].astype(out_dtype)  # [nse, n]
            w = weights[0] if is_homo else weights[:, None]
            return (jnp.zeros((k, B.shape[1]), dtype=out_dtype).at[col].add(w * B_rows),)
    else:
        def kernel(weights, row, col, B):
            B_rows = B[col].astype(out_dtype)  # [nse, n]
            w = weights[0] if is_homo else weights[:, None]
            return (jnp.zeros((m, B.shape[1]), dtype=out_dtype).at[row].add(w * B_rows),)

    return kernel


def _coomm_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    shape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed kernel for COO SpMM via jax.experimental.sparse (GPU only)."""
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        if is_homo:
            def kernel(weights, row, col, B):
                ones = jnp.ones_like(row, dtype=out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((ones, idx), shape=(m, k))
                return ((mat.T @ B.astype(out_dtype)) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, row, col, B):
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((weights.astype(out_dtype), idx), shape=(m, k))
                return (mat.T @ B.astype(out_dtype),)
    else:
        if is_homo:
            def kernel(weights, row, col, B):
                ones = jnp.ones_like(row, dtype=out_dtype)
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((ones, idx), shape=(m, k))
                return ((mat @ B.astype(out_dtype)) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, row, col, B):
                idx = jnp.stack([row, col], axis=1)
                mat = jsparse.BCOO((weights.astype(out_dtype), idx), shape=(m, k))
                return (mat @ B.astype(out_dtype),)
    return kernel


def _coomm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            nnz = max(1, int(n_pre * n_post * prob))
            row = np.random.randint(0, n_pre, nnz, dtype=np.int32)
            col = np.random.randint(0, n_post, nnz, dtype=np.int32)
            weights = jnp.ones(1, dtype=dtype) if homo else jnp.asarray(np.random.randn(nnz), dtype=dtype)
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, jnp.asarray(row), jnp.asarray(col), B),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def coomm_p_call(
    weights,
    row,
    col,
    B,
    *,
    shape: Sequence[int],
    transpose: bool,
    backend: Optional[str] = None,
):
    """Low-level primitive call for COO sparse matrix-matrix multiplication.

    Validates inputs, normalises *weights* to at least 1-D, and dispatches
    the ``coomm_p`` XLA custom kernel.

    Parameters
    ----------
    weights : jax.Array
        Non-zero values (unitless mantissa).  Scalar or 1-D of length
        ``nnz``.
    row : jax.Array
        1-D int array of row indices, length ``nnz``.
    col : jax.Array
        1-D int array of column indices, length ``nnz``.
    B : jax.Array
        Dense right-hand matrix, shape ``(k, n)``.
    shape : Sequence[int]
        Logical ``(m, k)`` matrix shape.
    transpose : bool
        Whether to use the transposed matrix.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    tuple of jax.Array
        Single-element tuple containing the result matrix.

    Raises
    ------
    ValueError
        If *row* or *col* are not 1-D, have different lengths, or *B*
        is not 2-D.

    See Also
    --------
    coomm : Public API with unit handling.
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

    out_info = (
        jax.ShapeDtypeStruct([out_rows, B.shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([out_rows, B.shape[1]], weights.dtype)
    )
    return coomm_p(
        weights,
        row,
        col,
        B,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
    )


coomm_p = XLACustomKernel(
    'coomm',
    doc="""
Low-level XLA custom-kernel primitive for ``coomm``.

This ``XLACustomKernel`` instance dispatches the COO sparse matrix-matrix
multiplication with floating-point weights operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata
provided by the high-level wrapper.

The operation computes ``Y[i, n] = sum_k A[i, k] * B[k, n]`` when
``transpose=False`` or ``Y[k, n] = sum_i A[i, k] * B[i, n]`` when
``transpose=True``, where ``A`` is a sparse matrix in COO format and ``B``
is a dense matrix. This is the standard sparse matrix-matrix multiplication
with full floating-point arithmetic.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``coomm_p.available_backends(platform)``,
and the default backend can be configured with ``coomm_p.set_default(platform, backend)``.

See Also
--------
coomm : High-level user-facing function wrapper.
"""
)
coomm_p.def_numba_kernel(_coomm_numba_kernel)
coomm_p.def_pallas_kernel('gpu', _coomm_pallas_gpu_kernel)
coomm_p.def_pallas_kernel('tpu', _coomm_pallas_tpu_kernel)
coomm_p.def_tvmffi_kernel('gpu', _coomm_tvmffi_kernel)
coomm_p.def_kernel('jax_raw', 'cpu', _coomm_jax_kernel)
coomm_p.def_kernel('jax_raw', 'gpu', _coomm_jax_kernel)
coomm_p.def_kernel('jax_raw', 'tpu', _coomm_jax_kernel)
coomm_p.def_kernel('cusparse', 'gpu', _coomm_cusparse_kernel)
coomm_p.def_jvp_rule2(_coomm_jvp_left, None, None, _coomm_jvp_right)
coomm_p.def_transpose_rule(_coomm_transpose_rule)
coomm_p.def_batching_rule(_coomm_batching)
coomm_p.def_call(coomm_p_call)
coomm_p.def_tags('coo', 'float')
coomm_p.def_benchmark_data(_coomm_benchmark_data)
