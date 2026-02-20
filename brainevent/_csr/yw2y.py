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

from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._misc import generate_block_dim, namescope
from brainevent._op import numba_kernel, XLACustomKernel, register_tvm_cuda_from_file
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._typing import Data, Indptr, Index, MatrixShape

__all__ = [
    'csrmv_yw2y',
    'csrmv_yw2y_p',
]


@namescope(static_argnames=['shape', 'transpose'])
def csrmv_yw2y(
    y: Data,
    w: Data,
    indices: Index,
    indptr: Indptr,
    *,
    shape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """
    Element-wise product of a vector and CSR weights, indexed by CSR
    structure.

    For each non-zero entry ``j`` in the CSR matrix at position
    ``(row, col)``, computes ``out[j] = w[j] * y[row]`` (non-transposed)
    or ``out[j] = w[j] * y[col]`` (transposed).  The result has the same
    shape as ``w`` and ``indices`` (i.e., one value per structural
    non-zero).

    This operation is useful for computing per-synapse quantities in
    neural network models where ``y`` is a neuron-level vector and ``w``
    contains per-synapse weights stored in CSR form.

    The function supports physical units via :mod:`brainunit`.

    Parameters
    ----------
    y : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense vector indexed by the CSR structure.  Shape
        ``(shape[0],)`` when ``transpose=False`` or ``(shape[1],)`` when
        ``transpose=True``.
    w : jax.Array, numpy.ndarray, or brainunit.Quantity
        Per-synapse weight values.  Shape ``(nse,)``, must match the
        shape of ``indices``.
    indices : jax.Array or numpy.ndarray
        Column indices of the CSR matrix.  Shape ``(nse,)`` with integer
        dtype.
    indptr : jax.Array or numpy.ndarray
        Row index pointer array.  Shape ``(shape[0] + 1,)`` with integer
        dtype.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        CSR matrix.
    transpose : bool, optional
        If ``True``, index ``y`` by column indices instead of row indices.
        Default is ``False``.
    backend : str or None, optional
        Compute backend.  Default is ``None`` (auto-select).

    Returns
    -------
    out : jax.Array or brainunit.Quantity
        Per-synapse result vector.  Shape ``(nse,)``, same as ``w``.

    See Also
    --------
    csrmv : Standard CSR matrix--vector multiplication.

    Notes
    -----
    This operation is differentiable with respect to both ``y`` and ``w``
    via custom JVP rules.  The transpose rule is not yet implemented.

    Mathematically, for each structural non-zero entry ``j`` of the CSR
    matrix at position ``(row, col)``, the output is computed as:

    ``out[j] = w[j] * y[row]``  (non-transposed, ``transpose=False``)

    ``out[j] = w[j] * y[col]``  (transposed, ``transpose=True``)

    where ``row`` is determined by the ``indptr`` array (the row to
    which the ``j``-th non-zero belongs) and ``col = indices[j]``.

    This operation is distinct from standard sparse matrix--vector
    multiplication: it produces one output element per structural
    non-zero rather than one per matrix row.  It is commonly used to
    compute per-synapse quantities in spiking neural network models.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import csrmv_yw2y
        >>> y = jnp.array([1.0, 2.0])
        >>> w = jnp.array([0.5, 0.3, 0.7, 0.1])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> csrmv_yw2y(y, w, indices, indptr, shape=(2, 3))
    """
    w, w_unit = u.split_mantissa_unit(w)
    y, _ = u.split_mantissa_unit(y)
    res = csrmv_yw2y_p_call(y, w, indices, indptr, shape=shape, transpose=transpose, backend=backend)[0]
    return u.maybe_decimal(res * w_unit)


def _csrmv_yw2y_numba_kernels(
    transpose: bool,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    if transpose:
        @numba.njit
        def mm(y, w, indices, indptr, posts):
            for i_col in range(indptr.shape[0] - 1):
                i_row_start = indptr[i_col]
                i_row_end = indptr[i_col + 1]
                for j in range(i_row_start, i_row_end):
                    posts[j] = w[j] * y[indices[j]]

        def kernel(y, w, indices, indptr):
            return numba_kernel(mm, outs=kwargs['outs'])(y, w, indices, indptr)

    else:
        @numba.njit
        def mm(y, w, indptr, posts):
            for i_row in range(indptr.shape[0] - 1):
                i_col_start = indptr[i_row]
                i_col_end = indptr[i_row + 1]
                for j in range(i_col_start, i_col_end):
                    posts[j] = w[j] * y[i_row]

        def kernel(y, w, indices, indptr):
            return numba_kernel(mm, outs=kwargs['outs'])(y, w, indptr)

    return kernel


def _csrmv_yw2y_pallas_kernels(
    shape: MatrixShape,
    transpose: bool,
    y_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import load, store

    block_dim = generate_block_dim(y_info.shape[0], 128)

    if transpose:
        def mm(
            y_ref,
            w_ref,
            indices_ref,
            indptr_ref,
            posts_ref,
        ):
            i_block = pl.program_id(0)
            num_blocks_grid = indptr_ref.shape[0] - 1

            def _body():
                i_start = indptr_ref[i_block]
                i_end = indptr_ref[i_block + 1]
                num_blocks = (i_end - i_start + block_dim - 1) // block_dim

                def loop_fn(i, _):
                    offset = i_start + i * block_dim
                    mask = (offset + jnp.arange(block_dim)) < i_end

                    w = load(w_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0.0)
                    index = load(indices_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0)

                    # Indirect Read Guard
                    valid_idx = index < y_ref.shape[0]
                    safe_index = jnp.minimum(index, y_ref.shape[0] - 1)

                    # Combine Masks
                    final_mask = mask & valid_idx

                    y = y_ref[safe_index]
                    y = jnp.where(final_mask, y, 0.0)

                    store(posts_ref.at[pl.ds(offset, block_dim)], w * y, mask=final_mask)

                jax.lax.fori_loop(0, num_blocks, loop_fn, None)

            # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
            jax.lax.cond(i_block < num_blocks_grid, _body, lambda: None)

        def kernel(y, w, indices, indptr):
            fn = pl.pallas_call(
                mm,
                grid=(shape[0],),
                out_shape=kwargs['outs'],
                backend='triton'
            )
            return fn(y, w, indices, indptr)
    else:
        def mm(
            y_ref,
            w_ref,
            indptr_ref,
            posts_ref,
        ):
            i_block = pl.program_id(0)
            num_blocks_grid = indptr_ref.shape[0] - 1

            def _body():
                i_start = indptr_ref[i_block]
                i_end = indptr_ref[i_block + 1]
                num_blocks = (i_end - i_start + block_dim - 1) // block_dim
                y_scalar = y_ref[i_block]

                def loop_fn(i, _):
                    offset = i_start + i * block_dim
                    mask = (offset + jnp.arange(block_dim)) < i_end

                    w = load(w_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0.0)
                    store(posts_ref.at[pl.ds(offset, block_dim)], w * y_scalar, mask=mask)

                jax.lax.fori_loop(0, num_blocks, loop_fn, None)

            # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
            jax.lax.cond(i_block < num_blocks_grid, _body, lambda: None)

        def kernel(y, w, indices, indptr):
            fn = pl.pallas_call(mm, grid=(shape[0],), out_shape=kwargs['outs'], backend='triton')
            return fn(y, w, indptr)

    return kernel


def _csrmv_yw2y_jvp_y(y_dot, y, w, indices, indptr, *, shape, transpose, **kwargs):
    return csrmv_yw2y_p_call(y_dot, w, indices, indptr, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _csrmv_yw2y_jvp_w(w_dot, y, w, indices, indptr, *, shape, transpose, **kwargs):
    return csrmv_yw2y_p_call(y, w_dot, indices, indptr, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _csrmv_yw2y_transpose_rule(ct, y, w, indices, indptr, *, shape, transpose, **kwargs):
    raise NotImplementedError


def _csrmv_yw2y_benchmark_data(*, platform):
    """
    Benchmark configurations for ``csrmv_yw2y``.

    Covers a range of matrix sizes and connection densities representative
    of typical spiking neural network workloads:

    * Small (1K×1K):   fast iteration; useful for overhead measurement.
    * Medium (5K×5K):  common SNN scale; balances rows vs. non-zeros.
    * Large (20K×20K): large-scale simulation benchmark.

    Each size is tested at three structural densities:
    * Low   (0.1%): avg_nnz ≈ 1–20;  favours NT_row_thread.
    * Medium (1%): avg_nnz ≈ 10–200; favours NT_row_warp.
    * High  (10%): avg_nnz ≈ 100–2000; favours NT_nz_thread.

    Both transpose=False and transpose=True are included.
    """
    dtype = jnp.float32
    configs = []
    sizes = [
        (1_000,  1_000),
        (5_000,  5_000),
        (20_000, 20_000),
    ]
    probs = [0.001, 0.01, 0.1]

    for (n_pre, n_post), prob in zip(sizes, probs):
        n_conn = max(1, int(n_post * prob))
        indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
        indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
        w = jnp.asarray(np.random.randn(n_pre * n_conn), dtype=dtype)
        indptr_jax = jnp.asarray(indptr)

        for transpose in (False, True):
            y_size = n_post if transpose else n_pre
            y = jnp.asarray(np.random.randn(y_size), dtype=dtype)
            tag = 'T' if transpose else 'NT'
            name = f"{tag},n={n_pre},nnz={n_conn}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (y, w, jnp.asarray(indices), indptr_jax),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def _csrmv_yw2y_jax_kernel(
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for CSR yw2y (out[j] = w[j] * y[row/col]) (all platforms)."""
    m, k = shape
    nse = kwargs['indices_info'].size

    if transpose:
        # col index is directly in `indices`
        def kernel(y, w, indices, indptr):
            return (w * y[indices],)
    else:
        def kernel(y, w, indices, indptr):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            return (w * y[row_ids],)
    return kernel


def _csrmv_yw2y_cuda_kernel(
    shape: MatrixShape,
    transpose: bool,
    w_info: jax.ShapeDtypeStruct,
    **kwargs,
):
    """
    TVM FFI (CUDA) kernel generator for ``csrmv_yw2y``.

    Dispatches to optimised CUDA kernels compiled from ``yw2y.cu``
    via the TVM FFI compilation pipeline.

    Non-transpose (NT) kernels are auto-selected based on ``avg_nnz``:

    * ``avg_nnz < 8``   → ``NT_row_thread`` (1 thread/row, serial)
    * ``avg_nnz < 512`` → ``NT_row_warp``   (1 warp/row, stride-32)
    * ``avg_nnz >= 512``→ ``NT_nz_thread``  (1 thread/nz, binary-search row)

    The transpose (T) variant always uses ``T_nz_thread`` (1 thread/nz,
    direct column gather), which is embarrassingly parallel and needs no
    atomic operations.

    Parameters
    ----------
    shape : tuple of int
        ``(m, k)`` logical shape of the CSR matrix.
    transpose : bool
        ``False`` → NT (gather from rows);  ``True`` → T (gather from cols).
    w_info : jax.ShapeDtypeStruct
        Shape and dtype of the weight array ``w`` (always ``(nse,)``).
    **kwargs
        Must contain ``outs`` (list of ``jax.ShapeDtypeStruct`` for output).

    Returns
    -------
    kernel : callable
        ``kernel(y, w, indices, indptr) -> (output,)``

    Notes
    -----
    This backend requires ``int32`` column indices and row pointers.  The
    Python-side ``csrmv_yw2y_p_call`` asserts this before dispatching.

    Unlike ``binary_csrmv``, the transpose variant does **not** use
    ``atomicAdd`` because each output element ``out[j]`` is owned by
    exactly one thread.
    """
    from pathlib import Path

    register_tvm_cuda_from_file(
        module='csrmv_yw2y',
        source=Path(__file__).parent.joinpath('yw2y.cu'),
    )

    out_info = kwargs['outs']

    # Weight dtype suffix
    _dtype_sfx = {
        jnp.dtype('float16'):  '_f16',
        jnp.dtype('float32'):  '_f32',
        jnp.dtype('float64'):  '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(w_info.dtype), '_f32')

    if transpose:
        kernel_name = f'csrmv_yw2y.csrmv_yw2y_t_nz_thread{wt_sfx}'
    else:
        kernel_name = f'csrmv_yw2y.csrmv_yw2y_nt_auto{wt_sfx}'

    def kernel(y, w, indices, indptr):
        return jax.ffi.ffi_call(kernel_name, out_info)(y, w, indices, indptr)

    return kernel


def csrmv_yw2y_p_call(
    y: Data,
    w: Data,
    indices: Index,
    indptr: Indptr,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for the element-wise vector--weight product
    indexed by CSR structure.

    Validates inputs and dispatches the ``csrmv_yw2y_p`` XLA custom kernel.
    For each structural non-zero ``j`` at position ``(row, col)`` in the
    CSR matrix, the output is ``out[j] = w[j] * y[row]`` (non-transposed)
    or ``out[j] = w[j] * y[col]`` (transposed).

    Parameters
    ----------
    y : jax.Array
        Dense vector.  Shape ``(shape[0],)`` when ``transpose=False`` or
        ``(shape[1],)`` when ``transpose=True``.
    w : jax.Array
        Per-synapse weight values.  Shape ``(nse,)``, must match the shape
        of ``indices``.
    indices : jax.Array
        Column indices of the CSR matrix.  Shape ``(nse,)`` with integer
        dtype.
    indptr : jax.Array
        Row index pointer array.  Shape ``(shape[0] + 1,)`` with integer
        dtype.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the CSR
        matrix.
    transpose : bool, optional
        If ``True``, index ``y`` by column indices.  Default is ``False``.
    backend : str or None, optional
        Compute backend to use.  Default is ``None`` (auto-select).

    Returns
    -------
    list of jax.Array
        A single-element list containing the per-synapse result.  Shape
        ``(nse,)``, same as ``w``.

    Raises
    ------
    AssertionError
        If ``y`` and ``w`` have different dtypes.
    AssertionError
        If ``y`` or ``w`` is not 1-D.
    AssertionError
        If ``indices`` or ``indptr`` does not have an integer dtype.
    AssertionError
        If ``w`` does not have a floating-point dtype.
    AssertionError
        If ``w`` and ``indices`` do not have the same shape.
    AssertionError
        If there is a shape mismatch between ``y`` and the sparse
        matrix ``shape`` (considering the ``transpose`` flag).

    See Also
    --------
    csrmv_yw2y : High-level wrapper with unit support.

    Notes
    -----
    The computation performed is, for each structural non-zero entry
    ``j`` at CSR position ``(row, col)``:

    ``out[j] = w[j] * y[row]``  (non-transposed)

    ``out[j] = w[j] * y[col]``  (transposed)

    where ``row`` is determined by ``indptr`` and ``col = indices[j]``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import csrmv_yw2y
        >>> y = jnp.array([1.0, 2.0])
        >>> w = jnp.array([0.5, 0.3, 0.7, 0.1])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> result = csrmv_yw2y(
        ...     y, w, indices, indptr,
        ...     shape=(2, 3), transpose=False)
    """
    assert y.dtype == w.dtype, f"y and w must have the same dtype, but got {y.dtype} and {w.dtype}."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert y.ndim == w.ndim == 1, "y and w must have the same shape."
    assert jnp.issubdtype(indices.dtype, jnp.integer), "Indices must be an integer type."
    assert jnp.issubdtype(indptr.dtype, jnp.integer), "indptr must be an integer type."
    assert jnp.issubdtype(w.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w.shape == indices.shape, f"Weights shape mismatch, expected {indices.shape}, got {w.shape}."
    if transpose:
        # [x] @ [h, w] -> [w]
        assert shape[1] == y.shape[0], "Shape mismatch for transpose operation."
    else:
        # [h, w] @ [x] -> [h]
        assert shape[0] == y.shape[0], "Shape mismatch for non-transpose operation."

    return csrmv_yw2y_p(
        y,
        w,
        indices,
        indptr,
        outs=[jax.ShapeDtypeStruct(w.shape, w.dtype)],
        shape=tuple(shape),
        transpose=transpose,
        backend=backend,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        y_info=jax.ShapeDtypeStruct(y.shape, y.dtype),
        w_info=jax.ShapeDtypeStruct(w.shape, w.dtype),
    )


csrmv_yw2y_p = XLACustomKernel(
    'csrmv_yw2y',
    doc="""
Low-level XLA custom-kernel primitive for ``csrmv_yw2y``.

This ``XLACustomKernel`` instance dispatches the CSR sparse matrix-vector multiplication with element-wise product (y * W -> y)
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

For each non-zero entry at position (row, col) in the CSR matrix, computes
out[j] = w[j] * y[row] (non-transposed) or out[j] = w[j] * y[col] (transposed),
producing one output element per structural non-zero.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``csrmv_yw2y_p.available_backends(platform)``,
and the default backend can be configured with ``csrmv_yw2y_p.set_default(platform, backend)``.

See Also
--------
csrmv_yw2y : High-level user-facing function wrapper.
"""
)
csrmv_yw2y_p.def_numba_kernel(_csrmv_yw2y_numba_kernels)
csrmv_yw2y_p.def_pallas_kernel('gpu', _csrmv_yw2y_pallas_kernels)
csrmv_yw2y_p.def_tvmffi_kernel('gpu', _csrmv_yw2y_cuda_kernel)
csrmv_yw2y_p.def_kernel('jax_raw', 'cpu', _csrmv_yw2y_jax_kernel)
csrmv_yw2y_p.def_kernel('jax_raw', 'gpu', _csrmv_yw2y_jax_kernel)
csrmv_yw2y_p.def_kernel('jax_raw', 'tpu', _csrmv_yw2y_jax_kernel)
csrmv_yw2y_p.def_jvp_rule2(_csrmv_yw2y_jvp_y, _csrmv_yw2y_jvp_w, None, None)
csrmv_yw2y_p.def_call(csrmv_yw2y_p_call)
csrmv_yw2y_p.def_tags('csr', 'float')
csrmv_yw2y_p.def_benchmark_data(_csrmv_yw2y_benchmark_data)
