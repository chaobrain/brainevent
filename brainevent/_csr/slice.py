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

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import namescope
from brainevent._op import numba_kernel, XLACustomKernel, general_batching_rule, register_tvm_cuda_from_file, jaxinfo_to_warpinfo
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._typing import MatrixShape

__all__ = [
    'csr_slice_rows',
    'csr_slice_rows_p',
]


@namescope(static_argnames=['shape'])
def csr_slice_rows(
    data,
    indices,
    indptr,
    row_indices,
    *,
    shape: MatrixShape,
    backend: Optional[str] = None,
):
    """Extract selected rows from a CSR matrix as a dense submatrix.

    For each row index ``k`` in ``row_indices``, extracts the corresponding
    row of the CSR matrix and places it in the output. The result is a dense
    matrix of shape ``(len(row_indices), shape[1])``.

    Parameters
    ----------
    data : jax.Array or brainunit.Quantity
        Non-zero values of the CSR matrix, shape ``(nnz,)`` for
        heterogeneous weights or ``(1,)`` for a single homogeneous weight.
    indices : jax.Array
        Column indices array, shape ``(nnz,)`` with integer dtype.
    indptr : jax.Array
        Row pointer array, shape ``(n_rows + 1,)`` with integer dtype.
    row_indices : jax.Array
        1-D integer array of row indices to extract.
    shape : tuple of int
        Shape of the CSR matrix as ``(n_rows, n_cols)``.
    backend : str or None, optional
        Compute backend. Default is ``None`` (auto-select).

    Returns
    -------
    jax.Array or brainunit.Quantity
        Dense matrix of shape ``(len(row_indices), shape[1])``.
    """
    data, data_unit = u.split_mantissa_unit(data)
    row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
    homo = row_indices.ndim == 0
    result = csr_slice_rows_p_call(
        data, indices, indptr, row_indices,
        shape=shape, backend=backend,
    )[0]
    result = result[0] if homo else result
    return u.maybe_decimal(result * data_unit)


def _csr_slice_rows_numba_kernel_generator(
    row_indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs,
):
    import numba

    m, n = shape
    num_selected = row_indices_info.shape[0]
    homo = kwargs['data_info'].shape[0] == 1

    if homo:
        @numba.njit(fastmath=True)
        def slice_rows(data, indices, indptr, row_indices, out):
            out[:] = 0.
            w = data[0]
            for k in range(num_selected):
                r = row_indices[k]
                if 0 <= r < m:
                    for j in range(indptr[r], indptr[r + 1]):
                        out[k, indices[j]] += w
    else:
        @numba.njit(fastmath=True)
        def slice_rows(data, indices, indptr, row_indices, out):
            out[:] = 0.
            for k in range(num_selected):
                r = row_indices[k]
                if 0 <= r < m:
                    for j in range(indptr[r], indptr[r + 1]):
                        out[k, indices[j]] += data[j]

    def kernel(data, indices, indptr, row_indices):
        return numba_kernel(slice_rows, outs=kwargs['outs'])(data, indices, indptr, row_indices)

    return kernel


def _csr_slice_rows_pallas_kernel_generator(
    row_indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs,
):
    from jax.experimental import pallas as pl

    m, n = shape
    num_selected = row_indices_info.shape[0]
    homo = kwargs['data_info'].shape[0] == 1

    if homo:
        def slice_rows_pallas(data_ref, indices_ref, indptr_ref, row_indices_ref, _, out_ref):
            k = pl.program_id(0)
            r = row_indices_ref[k]
            w = data_ref[0]

            i_start = jnp.where(r < m, indptr_ref[jnp.minimum(r, m - 1)], 0)
            i_end = jnp.where(r < m, indptr_ref[jnp.minimum(r + 1, m)], 0)
            nnz_in_row = i_end - i_start

            def body_fn(j, _):
                idx = i_start + j
                col = indices_ref[idx]
                valid = (j < nnz_in_row) & (r >= 0) & (r < m)
                out_ref[k, col] = jnp.where(valid, out_ref[k, col] + w, out_ref[k, col])

            max_nnz = jnp.where(r < m, nnz_in_row, 0)
            jax.lax.fori_loop(0, max_nnz, body_fn, None)
    else:
        def slice_rows_pallas(data_ref, indices_ref, indptr_ref, row_indices_ref, _, out_ref):
            k = pl.program_id(0)
            r = row_indices_ref[k]

            i_start = jnp.where(r < m, indptr_ref[jnp.minimum(r, m - 1)], 0)
            i_end = jnp.where(r < m, indptr_ref[jnp.minimum(r + 1, m)], 0)
            nnz_in_row = i_end - i_start

            def body_fn(j, _):
                idx = i_start + j
                col = indices_ref[idx]
                val = data_ref[idx]
                valid = (j < nnz_in_row) & (r >= 0) & (r < m)
                out_ref[k, col] = jnp.where(valid, out_ref[k, col] + val, out_ref[k, col])

            max_nnz = jnp.where(r < m, nnz_in_row, 0)
            jax.lax.fori_loop(0, max_nnz, body_fn, None)

    def kernel(data, indices, indptr, row_indices):
        out_info = kwargs['outs'][0]
        fn = pl.pallas_call(
            slice_rows_pallas,
            grid=(num_selected,),
            input_output_aliases={4: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(data, indices, indptr, row_indices, jnp.zeros(out_info.shape, dtype=out_info.dtype))

    return kernel


def _csr_slice_rows_jvp_data(dot, data, indices, indptr, row_indices, *, shape, **kwargs):
    return csr_slice_rows_p_call(dot, indices, indptr, row_indices, shape=shape, backend=kwargs['backend'])


def _csr_slice_rows_transpose_rule(ct, data, indices, indptr, row_indices, *, shape, **kwargs):
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)
    assert not ad.is_undefined_primal(row_indices)

    ct = ct[0]

    if ad.is_undefined_primal(data):
        if type(ct) is ad.Zero:
            ct_data = ad.Zero(data)
        else:
            ct_data = csr_slice_rows_grad_p_call(
                ct, indices, indptr, row_indices, shape=shape, backend=kwargs['backend'],
            )[0]
            if data.aval.shape[0] == 1:
                ct_data = jnp.sum(ct_data).reshape(1)
        return ct_data, indices, indptr, row_indices
    else:
        raise ValueError("Cannot transpose with respect to indices, indptr, or row_indices.")


def _csr_slice_rows_batching(args, axes, **kwargs):
    if axes[:3] == (None, None, None) and axes[3] is not None:
        # Only row_indices is batched: flatten to one call, then reshape output.
        data, indices, indptr, row_indices = args
        if axes[3] != 0:
            row_indices = jnp.moveaxis(row_indices, axes[3], 0)
        batch_size, num_sel = row_indices.shape
        flat_row_indices = row_indices.reshape(-1)
        result = csr_slice_rows_p_call(
            data, indices, indptr, flat_row_indices,
            shape=kwargs['shape'], backend=kwargs['backend'],
        )
        out = result[0].reshape(batch_size, num_sel, kwargs['shape'][1])
        return (out,), (0,)
    return general_batching_rule(csr_slice_rows_p, args, axes, **kwargs)


def _csr_slice_rows_benchmark_data(*, platform):
    n_pre, n_post, prob = 1000, 1000, 0.1
    n_conn = max(1, int(n_post * prob))
    indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
    indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
    data = jnp.ones(n_pre * n_conn, dtype=jnp.float32)
    row_indices = jnp.array([0, 10, 50, 100, 500], dtype=jnp.int32)
    return [
        BenchmarkConfig(
            "default",
            (data, jnp.asarray(indices), jnp.asarray(indptr), row_indices),
            {'shape': (n_pre, n_post)},
        )
    ]


def _csr_slice_rows_cuda_kernel_generator(
    **kwargs,
):
    register_tvm_cuda_from_file(
        module='csr_slice_rows',
        source=Path(__file__).parent.joinpath('slice_csr_slice_rows.cu'),
        include_dir=Path(__file__).parent.parent.joinpath('include'),
    )

    out_info = kwargs['outs']
    data_info = kwargs['data_info']
    homo = data_info.shape[0] == 1

    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(data_info.dtype), '_f32')
    mode_sfx = '_homo' if homo else '_hetero'
    kernel_name = f'csr_slice_rows.csr_slice_rows_fwd{mode_sfx}_auto{wt_sfx}'

    def kernel(data, indices, indptr, row_indices):
        return jax.ffi.ffi_call(kernel_name, out_info)(data, indices, indptr, row_indices)

    return kernel


def csr_slice_rows_p_call(
    data, indices, indptr, row_indices,
    *, shape: MatrixShape, backend: Optional[str] = None,
):
    """Low-level primitive call for CSR row slicing.

    Parameters
    ----------
    data : jax.Array
        Non-zero values, shape ``(nnz,)``.
    indices : jax.Array
        Column indices, shape ``(nnz,)``, integer dtype.
    indptr : jax.Array
        Row pointers, shape ``(n_rows + 1,)``, integer dtype.
    row_indices : jax.Array
        Row indices to extract, shape ``(num_selected,)``, integer dtype.
    shape : tuple of int
        Shape ``(n_rows, n_cols)`` of the CSR matrix.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    tuple of jax.Array
        Single-element tuple with dense matrix of shape ``(num_selected, n_cols)``.
    """
    data = jnp.atleast_1d(data)
    row_indices = jnp.atleast_1d(row_indices)
    assert data.ndim == 1, "data must be 1D"
    assert indices.ndim == 1, "indices must be 1D"
    assert indptr.ndim == 1, "indptr must be 1D"
    assert row_indices.ndim == 1, "row_indices must be 1D"
    assert jnp.issubdtype(indices.dtype, jnp.integer), "indices must be integer"
    assert jnp.issubdtype(indptr.dtype, jnp.integer), "indptr must be integer"
    assert jnp.issubdtype(row_indices.dtype, jnp.integer), "row_indices must be integer"

    num_selected = row_indices.shape[0]
    n_cols = shape[1]

    return csr_slice_rows_p(
        data, indices, indptr, row_indices,
        outs=[jax.ShapeDtypeStruct((num_selected, n_cols), data.dtype)],
        shape=tuple(shape),
        backend=backend,
        data_info=jax.ShapeDtypeStruct(data.shape, data.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        row_indices_info=jax.ShapeDtypeStruct(row_indices.shape, row_indices.dtype),
    )


csr_slice_rows_p = XLACustomKernel(
    'csr_slice_rows',
    doc="""
Low-level XLA custom-kernel primitive for ``csr_slice_rows``.

This ``XLACustomKernel`` instance dispatches the CSR row slicing operation
to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

Extracts selected rows from a CSR sparse matrix and returns a dense
submatrix of shape ``(num_selected, n_cols)``. Each selected row is
independently gathered: for row index *r*, non-zero entries in
``data[indptr[r]:indptr[r+1]]`` are scattered into the corresponding
columns of the output. Out-of-bounds row indices produce zero rows.

The operation is linear in ``data``, so forward-mode (JVP) simply
applies the same slice to the tangent. Reverse-mode (transpose) calls
the companion gradient primitive ``csr_slice_rows_grad_p`` to gather
cotangent contributions back into a vector of shape ``(nnz,)``.

When ``row_indices`` is the only batched argument (e.g. under ``vmap``),
the batching rule flattens the batch of row index arrays into a single
kernel call and reshapes the output, avoiding a sequential scan.

Available backends can be queried with ``csr_slice_rows_p.available_backends(platform)``,
and the default backend can be configured with ``csr_slice_rows_p.set_default(platform, backend)``.

See Also
--------
csr_slice_rows : High-level user-facing function wrapper.
csr_slice_rows_grad_p : Companion gradient primitive used by the transpose rule.
"""
)
def _csr_slice_rows_warp_kernel_generator(
    data_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    row_indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs,
):
    import warp
    from warp.jax_experimental import jax_kernel

    data_warp = jaxinfo_to_warpinfo(data_info)
    indices_warp = jaxinfo_to_warpinfo(indices_info)
    indptr_warp = jaxinfo_to_warpinfo(indptr_info)
    row_indices_warp = jaxinfo_to_warpinfo(row_indices_info)
    out_warp = jaxinfo_to_warpinfo(kwargs['outs'][0])

    m, n = shape
    num_selected = row_indices_info.shape[0]
    homo = data_info.shape[0] == 1

    if homo:
        @warp.kernel
        def slice_rows_warp(
            data: data_warp,
            indices: indices_warp,
            indptr: indptr_warp,
            row_indices: row_indices_warp,
            out: out_warp,
        ):
            k = warp.tid()
            r = row_indices[k]
            if 0 <= r < m:
                w = data[0]
                for j in range(indptr[r], indptr[r + 1]):
                    col = indices[j]
                    warp.atomic_add(out, k, col, w)
    else:
        @warp.kernel
        def slice_rows_warp(
            data: data_warp,
            indices: indices_warp,
            indptr: indptr_warp,
            row_indices: row_indices_warp,
            out: out_warp,
        ):
            k = warp.tid()
            r = row_indices[k]
            if 0 <= r < m:
                for j in range(indptr[r], indptr[r + 1]):
                    col = indices[j]
                    warp.atomic_add(out, k, col, data[j])

    def kernel(data, indices, indptr, row_indices):
        out_info = kwargs['outs'][0]
        zeros = jnp.zeros(out_info.shape, dtype=out_info.dtype)
        fn = jax_kernel(
            slice_rows_warp,
            launch_dims=[num_selected],
            num_outputs=1,
            in_out_argnames=['out'],
        )
        return fn(data, indices, indptr, row_indices, zeros)

    return kernel


def _csr_slice_rows_grad_warp_kernel_generator(
    ct_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    row_indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs,
):
    import warp
    from warp.jax_experimental import jax_kernel

    ct_warp = jaxinfo_to_warpinfo(ct_info)
    indices_warp = jaxinfo_to_warpinfo(indices_info)
    indptr_warp = jaxinfo_to_warpinfo(indptr_info)
    row_indices_warp = jaxinfo_to_warpinfo(row_indices_info)
    out_warp = jaxinfo_to_warpinfo(kwargs['outs'][0])

    m, n = shape
    num_selected = row_indices_info.shape[0]

    @warp.kernel
    def grad_slice_warp(
        ct: ct_warp,
        indices: indices_warp,
        indptr: indptr_warp,
        row_indices: row_indices_warp,
        ct_data: out_warp,
    ):
        k = warp.tid()
        r = row_indices[k]
        if 0 <= r < m:
            for j in range(indptr[r], indptr[r + 1]):
                col = indices[j]
                warp.atomic_add(ct_data, j, ct[k, col])

    def kernel(ct, indices, indptr, row_indices):
        out_info = kwargs['outs'][0]
        zeros = jnp.zeros(out_info.shape, dtype=out_info.dtype)
        fn = jax_kernel(
            grad_slice_warp,
            launch_dims=[num_selected],
            num_outputs=1,
            in_out_argnames=['ct_data'],
        )
        return fn(ct, indices, indptr, row_indices, zeros)

    return kernel

csr_slice_rows_p.def_numba_kernel(_csr_slice_rows_numba_kernel_generator)
csr_slice_rows_p.def_warp_kernel(_csr_slice_rows_warp_kernel_generator)
csr_slice_rows_p.def_pallas_kernel('gpu', _csr_slice_rows_pallas_kernel_generator)
csr_slice_rows_p.def_tvmffi_kernel('gpu', _csr_slice_rows_cuda_kernel_generator)
csr_slice_rows_p.def_jvp_rule2(_csr_slice_rows_jvp_data, None, None, None)
csr_slice_rows_p.def_transpose_rule(_csr_slice_rows_transpose_rule)
csr_slice_rows_p.def_batching_rule(_csr_slice_rows_batching)
csr_slice_rows_p.def_call(csr_slice_rows_p_call)
csr_slice_rows_p.def_tags('csr', 'slice')
csr_slice_rows_p.def_benchmark_data(_csr_slice_rows_benchmark_data)


@namescope(static_argnames=['shape'])
def csr_slice_rows_grad(
    ct,
    indices,
    indptr,
    row_indices,
    *,
    shape: MatrixShape,
    backend: Optional[str] = None,
):
    """Extract selected rows from a CSR matrix as a dense submatrix.

    For each row index ``k`` in ``row_indices``, extracts the corresponding
    row of the CSR matrix and places it in the output. The result is a dense
    matrix of shape ``(len(row_indices), shape[1])``.

    Parameters
    ----------
    data : jax.Array or brainunit.Quantity
        Non-zero values of the CSR matrix, shape ``(nnz,)`` for
        heterogeneous weights or ``(1,)`` for a single homogeneous weight.
    indices : jax.Array
        Column indices array, shape ``(nnz,)`` with integer dtype.
    indptr : jax.Array
        Row pointer array, shape ``(n_rows + 1,)`` with integer dtype.
    row_indices : jax.Array
        1-D integer array of row indices to extract.
    shape : tuple of int
        Shape of the CSR matrix as ``(n_rows, n_cols)``.
    backend : str or None, optional
        Compute backend. Default is ``None`` (auto-select).

    Returns
    -------
    jax.Array or brainunit.Quantity
        Dense matrix of shape ``(len(row_indices), shape[1])``.
    """
    ct, ct_unit = u.split_mantissa_unit(ct)
    result = csr_slice_rows_grad_p_call(
        ct, indices, indptr, row_indices, shape=shape, backend=backend,
    )[0]
    return u.maybe_decimal(result * ct_unit)


def _csr_slice_rows_grad_numba_kernel_generator(
    row_indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs,
):
    import numba

    m, n = shape
    num_selected = row_indices_info.shape[0]

    @numba.njit(fastmath=True)
    def grad_slice(ct, indices, indptr, row_indices, ct_data):
        ct_data[:] = 0.
        for k in range(num_selected):
            r = row_indices[k]
            if 0 <= r < m:
                for j in range(indptr[r], indptr[r + 1]):
                    ct_data[j] += ct[k, indices[j]]

    def kernel(ct, indices, indptr, row_indices):
        return numba_kernel(grad_slice, outs=kwargs['outs'])(ct, indices, indptr, row_indices)

    return kernel


def _csr_slice_rows_grad_pallas_kernel_generator(
    row_indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs,
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    m, n = shape
    num_selected = row_indices_info.shape[0]

    def grad_slice_pallas(ct_ref, indices_ref, indptr_ref, row_indices_ref, _, ct_data_ref):
        k = pl.program_id(0)
        r = row_indices_ref[k]

        i_start = jnp.where(r < m, indptr_ref[jnp.minimum(r, m - 1)], 0)
        i_end = jnp.where(r < m, indptr_ref[jnp.minimum(r + 1, m)], 0)
        nnz_in_row = i_end - i_start

        def body_fn(j, _):
            idx = i_start + j
            col = indices_ref[idx]
            val = ct_ref[k, col]
            valid = (j < nnz_in_row) & (r >= 0) & (r < m)
            atomic_add(ct_data_ref, (idx,), jnp.where(valid, val, 0.0))

        max_nnz = jnp.where(r < m, nnz_in_row, 0)
        jax.lax.fori_loop(0, max_nnz, body_fn, None)

    def kernel(ct, indices, indptr, row_indices):
        out_info = kwargs['outs'][0]
        zeros = jnp.zeros(out_info.shape, dtype=out_info.dtype)
        fn = pl.pallas_call(
            grad_slice_pallas,
            grid=(num_selected,),
            input_output_aliases={4: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(ct, indices, indptr, row_indices, zeros)

    return kernel


def _csr_slice_rows_grad_jvp_ct(dot, ct, indices, indptr, row_indices, *, shape, **kwargs):
    """JVP for ct input: the grad operation is linear in ct, so tangent is just grad applied to dot."""
    return csr_slice_rows_grad_p_call(dot, indices, indptr, row_indices, shape=shape, backend=kwargs['backend'])


def _csr_slice_rows_grad_batching(args, axes, **kwargs):
    return general_batching_rule(csr_slice_rows_grad_p, args, axes, **kwargs)


def _csr_slice_rows_grad_benchmark_data(*, platform):
    n_pre, n_post, prob = 1000, 1000, 0.1
    n_conn = max(1, int(n_post * prob))
    indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
    indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
    row_indices = jnp.array([0, 10, 50, 100, 500], dtype=jnp.int32)
    ct = jnp.ones((5, n_post), dtype=jnp.float32)
    return [
        BenchmarkConfig(
            "default",
            (ct, jnp.asarray(indices), jnp.asarray(indptr), row_indices),
            {'shape': (n_pre, n_post)},
        )
    ]


def _csr_slice_rows_grad_cuda_kernel_generator(
    **kwargs,
):
    register_tvm_cuda_from_file(
        module='csr_slice_rows',
        source=Path(__file__).parent.joinpath('slice_csr_slice_rows.cu'),
        include_dir=Path(__file__).parent.parent.joinpath('include'),
    )

    out_info = kwargs['outs']
    ct_info = kwargs['ct_info']

    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(ct_info.dtype), '_f32')
    kernel_name = f'csr_slice_rows.csr_slice_rows_grad_auto{wt_sfx}'

    def kernel(ct, indices, indptr, row_indices):
        return jax.ffi.ffi_call(kernel_name, out_info)(ct, indices, indptr, row_indices)

    return kernel


def _csr_slice_rows_grad_transpose_rule(ct, ct_input, indices, indptr, row_indices, *, shape, **kwargs):
    """Transpose of G^T is G (the forward slice operation).

    When ct_input (the cotangent input to grad) is the undefined primal,
    the transpose maps ct_ct_data (shape (nnz,)) back to ct_ct (shape (num_selected, n_cols))
    via the forward csr_slice_rows primitive.
    """
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)
    assert not ad.is_undefined_primal(row_indices)

    ct = ct[0]

    if ad.is_undefined_primal(ct_input):
        if type(ct) is ad.Zero:
            ct_ct = ad.Zero(ct_input)
        else:
            ct_ct = csr_slice_rows_p_call(
                ct, indices, indptr, row_indices,
                shape=shape, backend=kwargs['backend'],
            )[0]
        return ct_ct, indices, indptr, row_indices
    else:
        raise ValueError("Cannot transpose with respect to indices, indptr, or row_indices.")


def csr_slice_rows_grad_p_call(
    ct, indices, indptr, row_indices,
    *, shape: MatrixShape, backend: Optional[str] = None,
):
    """Low-level primitive call for CSR row slice gradient.

    Parameters
    ----------
    ct : jax.Array
        Cotangent of the output, shape ``(num_selected, n_cols)``.
    indices : jax.Array
        Column indices, shape ``(nnz,)``, integer dtype.
    indptr : jax.Array
        Row pointers, shape ``(n_rows + 1,)``, integer dtype.
    row_indices : jax.Array
        Row indices that were extracted, shape ``(num_selected,)``, integer dtype.
    shape : tuple of int
        Shape ``(n_rows, n_cols)`` of the CSR matrix.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    tuple of jax.Array
        Single-element tuple with gradient w.r.t. data, shape ``(nnz,)``.
    """
    row_indices = jnp.atleast_1d(row_indices)
    assert ct.ndim == 2, "ct must be 2D"
    assert indices.ndim == 1, "indices must be 1D"
    assert indptr.ndim == 1, "indptr must be 1D"
    assert row_indices.ndim == 1, "row_indices must be 1D"

    nnz = indices.shape[0]

    return csr_slice_rows_grad_p(
        ct, indices, indptr, row_indices,
        outs=[jax.ShapeDtypeStruct((nnz,), ct.dtype)],
        shape=tuple(shape),
        backend=backend,
        ct_info=jax.ShapeDtypeStruct(ct.shape, ct.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        row_indices_info=jax.ShapeDtypeStruct(row_indices.shape, row_indices.dtype),
    )


csr_slice_rows_grad_p = XLACustomKernel(
    'csr_slice_rows_grad',
    doc="""
Low-level XLA custom-kernel primitive for the gradient of ``csr_slice_rows``.

This ``XLACustomKernel`` instance dispatches the backward pass of the CSR
row slicing operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the calling transpose rule.

Given a cotangent matrix ``ct`` of shape ``(num_selected, n_cols)`` (the
upstream gradient with respect to the sliced output), this primitive
computes the gradient with respect to the CSR ``data`` array of shape
``(nnz,)``. For each selected row *r* (from ``row_indices``), the kernel
gathers ``ct[k, indices[j]]`` for every non-zero position *j* in
``[indptr[r], indptr[r+1])`` and accumulates into ``ct_data[j]``.

The operation is linear in ``ct``, so its JVP rule applies the same
gather to the tangent. Its transpose rule calls the forward primitive
``csr_slice_rows_p`` (transpose of the transpose is the original
operation), enabling second-order differentiation (e.g. ``grad(grad(...))``
and ``jvp(grad(...))``).

Available backends can be queried with ``csr_slice_rows_grad_p.available_backends(platform)``,
and the default backend can be configured with ``csr_slice_rows_grad_p.set_default(platform, backend)``.

See Also
--------
csr_slice_rows_p : Forward primitive whose transpose rule invokes this gradient primitive.
csr_slice_rows : High-level user-facing function wrapper.
"""
)
csr_slice_rows_grad_p.def_numba_kernel(_csr_slice_rows_grad_numba_kernel_generator)
csr_slice_rows_grad_p.def_warp_kernel(_csr_slice_rows_grad_warp_kernel_generator)
csr_slice_rows_grad_p.def_pallas_kernel('gpu', _csr_slice_rows_grad_pallas_kernel_generator)
csr_slice_rows_grad_p.def_tvmffi_kernel('gpu', _csr_slice_rows_grad_cuda_kernel_generator)
csr_slice_rows_grad_p.def_jvp_rule2(_csr_slice_rows_grad_jvp_ct, None, None, None)
csr_slice_rows_grad_p.def_transpose_rule(_csr_slice_rows_grad_transpose_rule)
csr_slice_rows_grad_p.def_batching_rule(_csr_slice_rows_grad_batching)
csr_slice_rows_grad_p.def_call(csr_slice_rows_grad_p_call)
csr_slice_rows_grad_p.def_tags('csr', 'slice', 'grad')
csr_slice_rows_grad_p.def_benchmark_data(_csr_slice_rows_grad_benchmark_data)
