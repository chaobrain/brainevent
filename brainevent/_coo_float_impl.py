# Copyright 2024- BrainPy Ecosystem Limited. All Rights Reserved.
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

import warnings
from functools import partial

import numpy as np
from jax import core, numpy as jnp
from jax.interpreters import ad, mlir
from jaxlib import gpu_sparse

__all__ = [
]


# --------------------------------------------------------------------
# cusparse_coo_matvec


def _coomv_impl(data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
    v = jnp.asarray(v)
    if transpose:
        row, col = col, row
    out_shape = shape[1] if transpose else shape[0]
    dv = data * v[col]
    return jnp.zeros(out_shape, dv.dtype).at[row].add(dv)


def _coomv_abstract_eval(data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
    assert data.shape == row.shape == col.shape
    assert data.dtype == v.dtype
    assert row.dtype == col.dtype
    assert len(shape) == 2
    assert v.ndim == 1
    assert v.shape[0] == (shape[0] if transpose else shape[1])
    out_shape = shape[1] if transpose else shape[0]
    return core.ShapedArray((out_shape,), data.dtype)


_coo_matvec_lowering = mlir.lower_fun(_coomv_impl, multiple_results=False)


def _coomv_gpu_lowering(coo_matvec_mhlo, ctx, data, row, col, v, *,
                        shape, rows_sorted, cols_sorted, transpose):
    data_aval, row_aval, _, x_aval = ctx.avals_in
    dtype = data_aval.dtype
    if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
        warnings.warn(f"cusparse_coo_matvec cusparse/hipsparse lowering not available for dtype={dtype}. "
                      "Falling back to default implementation.", UserWarning)
        return _coo_matvec_lowering(ctx, data, row, col, v,
                                    shape=shape,
                                    rows_sorted=rows_sorted,
                                    cols_sorted=cols_sorted,
                                    transpose=transpose)

    if rows_sorted:
        shape = shape
    elif cols_sorted:
        row, col = col, row
        transpose = not transpose
        shape = shape[::-1]
    else:
        warnings.warn("cusparse_coo_matvec GPU lowering requires matrices with sorted rows or sorted cols. "
                      "To sort the rows in your matrix, use e.g. mat = mat._sort_rows(). Falling "
                      "back to the default implementation.", UserWarning)
        return _coo_matvec_lowering(ctx, data, row, col, v,
                                    shape=shape,
                                    rows_sorted=rows_sorted,
                                    cols_sorted=cols_sorted,
                                    transpose=transpose)

    return [coo_matvec_mhlo(data, row, col, v,
                            shape=shape,
                            transpose=transpose,
                            index_dtype=row_aval.dtype,
                            data_dtype=dtype,
                            x_dtype=x_aval.dtype)]


def _coomv_jvp_mat(data_dot, data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
    return _coomv_cusparse_p.bind(data_dot, row, col, v,
                                  shape=shape,
                                  rows_sorted=rows_sorted,
                                  cols_sorted=cols_sorted,
                                  transpose=transpose)


def _coomv_jvp_vec(v_dot, data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
    return _coomv_cusparse_p.bind(data, row, col, v_dot,
                                  shape=shape,
                                  rows_sorted=rows_sorted,
                                  cols_sorted=cols_sorted,
                                  transpose=transpose)


def _coomv_transpose(ct, data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
    assert not ad.is_undefined_primal(row)
    assert not ad.is_undefined_primal(col)

    if ad.is_undefined_primal(v):
        return data, row, col, _coomv_cusparse_p.bind(data, row, col, ct,
                                                      shape=shape,
                                                      rows_sorted=rows_sorted,
                                                      cols_sorted=cols_sorted,
                                                      transpose=not transpose)
    else:
        return ct[row] * v[col], row, col, v


_coomv_cusparse_p = core.Primitive('cusparse_coo_matvec')
_coomv_cusparse_p.def_abstract_eval(_coomv_abstract_eval)
_coomv_cusparse_p.def_impl(_coomv_impl)
ad.defjvp(_coomv_cusparse_p, _coomv_jvp_mat, None, None, _coomv_jvp_vec)
ad.primitive_transposes[_coomv_cusparse_p] = _coomv_transpose
mlir.register_lowering(_coomv_cusparse_p, _coo_matvec_lowering)
mlir.register_lowering(_coomv_cusparse_p,
                       partial(_coomv_gpu_lowering, gpu_sparse.cuda_coo_matvec),
                       platform='cuda')
