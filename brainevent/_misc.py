# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Union, Sequence, Tuple, NamedTuple

import brainunit as u
import jax
import jax.numpy as jnp
from jax.experimental.sparse import csr_todense_p, coo_todense_p


class COOInfo(NamedTuple):
    shape: Sequence[int]
    rows_sorted: bool = False
    cols_sorted: bool = False


def _coo_todense(
    data: Union[jax.Array, u.Quantity],
    row: jax.Array,
    col: jax.Array,
    *,
    spinfo: COOInfo
) -> Union[jax.Array, u.Quantity]:
    """Convert CSR-format sparse matrix to a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      row : array of shape ``(nse,)``
      col : array of shape ``(nse,)`` and dtype ``row.dtype``
      spinfo : COOInfo object containing matrix metadata

    Returns:
      mat : array with specified shape and dtype matching ``data``
    """
    data, unit = u.split_mantissa_unit(data)
    r = coo_todense_p.bind(data, row, col, spinfo=spinfo)
    return u.maybe_decimal(r * unit)


@jax.jit
def _csr_to_coo(indices: jax.Array, indptr: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Given CSR (indices, indptr) return COO (row, col)"""
    return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices


def _csr_todense(
    data: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    indptr: jax.Array, *,
    shape: Sequence[int]
) -> Union[jax.Array, u.Quantity]:
    """
    Convert CSR-format sparse matrix to a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      shape : length-2 tuple representing the matrix shape

    Returns:
      mat : array with specified shape and dtype matching ``data``
    """
    data, unit = u.split_mantissa_unit(data)
    mat = csr_todense_p.bind(data, indices, indptr, shape=shape)
    return u.maybe_decimal(mat * unit)
