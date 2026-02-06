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


import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad
from jax.interpreters.partial_eval import DynamicJaxprTracer

from brainevent._misc import generate_block_dim
from brainevent._op import numba_kernel, jaxinfo_to_warpinfo, XLACustomKernel


def _is_tracer(x):
    return isinstance(x, (jax.ShapeDtypeStruct, jax.core.ShapedArray, DynamicJaxprTracer, jax.core.Tracer))


def csr_diag_position_v2(indptr, indices, shape: brainstate.typing.Size):
    assert isinstance(shape, (tuple, list)), "shape must be a tuple or list"
    assert indptr.ndim == 1, "indptr must be a 1D array"
    assert indices.ndim == 1, "indices must be a 1D array"
    assert len(shape) == 2, "shape must be a tuple or list of length 2"
    assert all(isinstance(s, int) and s > 0 for s in shape), "shape must be a tuple or list of non-negative integers"
    if _is_tracer(indptr):
        raise ValueError('Cannot trace indptr when finding diagonal position')
    if _is_tracer(indices):
        raise ValueError('Cannot trace indices when finding diagonal position')
    n_size = min(shape)

    import numba

    @numba.njit(cache=True)
    def _find_diag_position(indptr_, indices_, n):
        csr_positions = []
        diag_positions = []
        for i in range(n):
            start = indptr_[i]
            end = indptr_[i + 1]
            for j in range(start, end):
                if indices_[j] == i:
                    csr_positions.append(j)
                    diag_positions.append(i)
                    break
        return np.asarray(csr_positions, dtype=np.int32), np.asarray(diag_positions, dtype=np.int32)

    csr_pos, diag_pos = _find_diag_position(np.asarray(indptr), np.asarray(indices), n_size)

    return (csr_pos, diag_pos) if len(csr_pos) == len(diag_pos) else (csr_pos, None)


def csr_diag_add_v2(csr_value, positions, diag_value):
    assert u.fail_for_dimension_mismatch(csr_value, diag_value)
    assert csr_value.ndim == 1, "csr_value must be a 1D array"
    assert diag_value.ndim == 1, "diag_value must be a 1D array"
    assert csr_value.dtype == diag_value.dtype, "csr_value and diag_value must have the same dtype"
    csr_pos, diag_pos = positions
    assert csr_pos.ndim == 1, "csr_pos must be a 1D array"
    assert jnp.issubdtype(csr_pos.dtype, jnp.integer), "diag_position must be an integer array"
    if diag_pos is not None:
        assert diag_pos.ndim == 1, "diag_pos must be a 1D array"
        assert jnp.issubdtype(diag_pos.dtype, jnp.integer), "diag_position must be an integer array"

    diag_value = u.Quantity(diag_value).to(u.get_unit(csr_value)).mantissa
    csr_value, csr_unit = u.split_mantissa_unit(csr_value)
    if diag_pos is None:
        csr_value = csr_value.at[csr_pos].add(diag_value)
    else:
        csr_value = csr_value.at[csr_pos].add(diag_value[diag_pos])
    return u.maybe_decimal(csr_value * csr_unit)


def csr_diag_position(indptr, indices, shape: brainstate.typing.Size):
    """
    Find the diagonal position in a sparse matrix represented by indptr and indices.

    Parameters:
        indptr (array-like): The index pointer array.
        indices (array-like): The column indices of the non-zero elements.
        shape (brainstate.typing.Size): The shape of the sparse matrix, typically a tuple (n_rows, n_cols).

    Returns:
        ndarray: The diagonal position in the sparse matrix.
    """
    assert isinstance(shape, (tuple, list)), "shape must be a tuple or list"
    assert len(shape) == 2, "shape must be a tuple or list of length 2"
    assert all(isinstance(s, int) and s > 0 for s in shape), "shape must be a tuple or list of non-negative integers"
    n_size = min(shape)

    if _is_tracer(indptr):
        raise ValueError('Cannot trace indptr when finding diagonal position')
    if _is_tracer(indices):
        raise ValueError('Cannot trace indices when finding diagonal position')

    import numba

    @numba.njit(cache=True)
    def _find_diag_position(indptr_, indices_, n):
        results = []
        for i in range(n):
            start = indptr_[i]
            end = indptr_[i + 1]
            for j in range(start, end):
                if indices_[j] == i:
                    results.append(j)
                    break
            else:
                results.append(-1)
        return np.asarray(results, dtype=np.int32)

    return jnp.asarray(
        _find_diag_position(np.asarray(indptr), np.asarray(indices), n_size)
    )


def csr_diag_add(csr_value, diag_position, diag_value):
    """
    Add a diagonal value to a sparse matrix represented in CSR format.

    Parameters:
        csr_value (array-like): The values of the non-zero elements in the sparse matrix.
        diag_position (array-like): The diagonal position in the sparse matrix.
        diag_value (array-like): The diagonal value to be added.

    Returns:
        ndarray: The result of adding the diagonal value to the sparse matrix.
    """
    assert u.fail_for_dimension_mismatch(csr_value, diag_value)

    diag_value = u.Quantity(diag_value).to(u.get_unit(csr_value)).mantissa
    csr_value, csr_unit = u.split_mantissa_unit(csr_value)
    return u.maybe_decimal(
        csr_diag_add_call(csr_value, diag_position, diag_value)[0]
        * csr_unit
    )


def _csr_diag_add_numba_kernel_generator(
    csr_value_info: jax.ShapeDtypeStruct,
    diag_pos_info: jax.ShapeDtypeStruct,
    diag_value_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    @numba.njit(fastmath=True)
    def diag_add(csr_value, diag_position, diag_value, out):
        out[:] = csr_value[:]  # Copy input to output
        for i in range(diag_position.size):
            pos = diag_position[i]
            if pos >= 0:
                out[pos] += diag_value[i]

    def kernel(csr_value, diag_position, diag_value):
        return numba_kernel(diag_add, outs=kwargs['outs'])(csr_value, diag_position, diag_value)

    return kernel


def _csr_diag_add_warp_kernel_generator(
    csr_value_info: jax.ShapeDtypeStruct,
    diag_pos_info: jax.ShapeDtypeStruct,
    diag_value_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel
    from warp.jax_experimental import jax_kernel

    csr_value_warp_info = jaxinfo_to_warpinfo(csr_value_info)
    diag_pos_warp_info = jaxinfo_to_warpinfo(diag_pos_info)
    diag_value_warp_info = jaxinfo_to_warpinfo(diag_value_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    @warp.kernel
    def diag_add_warp(
        csr_value: csr_value_warp_info,
        diag_position: diag_pos_warp_info,
        diag_value: diag_value_warp_info,
        out: out_warp_info
    ):
        i_diag = warp.tid()
        pos = diag_position[i_diag]
        if pos >= 0:
            out[pos] += diag_value[i_diag]

    def kernel(csr_value, diag_position, diag_value):
        dim = diag_pos_info.shape[0]
        fn = jax_kernel(diag_add_warp, launch_dims=[dim], num_outputs=1, in_out_argnames=['out'])
        return fn(csr_value, diag_position, diag_value, jnp.array(csr_value))

    return kernel


def _csr_diag_add_pallas_kernel_generator(
    csr_value_info: jax.ShapeDtypeStruct,
    diag_pos_info: jax.ShapeDtypeStruct,
    diag_value_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    total = diag_pos_info.shape[0]
    block_dim = generate_block_dim(total, 512)

    def diag_add_pallas(csr_value_ref, diag_position_ref, diag_value_ref, _, out_ref):
        i_tile = pl.program_id(0)
        i_tile_start = i_tile * block_dim
        mask = (i_tile_start + jnp.arange(block_dim)) < total
        positions = diag_position_ref[pl.dslice(i_tile_start, block_dim)]
        values = diag_value_ref[pl.dslice(i_tile_start, block_dim)]
        valid_mask = mask & (positions >= 0)
        atomic_add(out_ref, positions, values, mask=valid_mask)

    def kernel(csr_value, diag_position, diag_value):
        fn = pl.pallas_call(
            diag_add_pallas,
            grid=(pl.cdiv(total, block_dim),),
            input_output_aliases={3: 0},
            out_shape=kwargs['outs']
        )
        return fn(csr_value, diag_position, diag_value, jnp.array(csr_value))

    return kernel


def _csr_diag_add_jvp_csr_value(dot, csr_value, diag_position, diag_value, **kwargs):
    return csr_diag_add_call(dot, diag_position, diag_value)


def _csr_diag_add_jvp_diag_value(dot, csr_value, diag_position, diag_value, **kwargs):
    return csr_diag_add_call(csr_value, diag_position, dot)


def _csr_diag_add_transpose_value(ct, csr_value, diag_position, diag_value, **kwargs):
    assert not ad.is_undefined_primal(diag_position)
    ct = ct[0]
    raise NotImplementedError


def csr_diag_add_call(csr_value, diag_position, diag_value, *, backend=None):
    assert csr_value.ndim == 1, "csr_value must be a 1D array"
    assert diag_position.ndim == 1, "diag_position must be a 1D array"
    assert diag_value.ndim == 1, "diag_value must be a 1D array"
    assert diag_position.shape == diag_value.shape, "diag_position must have the same shape as diag_value"
    assert jnp.issubdtype(diag_position.dtype, jnp.integer), "diag_position must be an integer array"
    assert csr_value.dtype == diag_value.dtype, "csr_value and diag_value must have the same dtype"

    return csr_diag_add_p(
        csr_value, diag_position, diag_value,
        outs=[jax.ShapeDtypeStruct(csr_value.shape, csr_value.dtype)],
        backend=backend,
        csr_value_info=jax.ShapeDtypeStruct(csr_value.shape, csr_value.dtype),
        diag_pos_info=jax.ShapeDtypeStruct(diag_position.shape, diag_position.dtype),
        diag_value_info=jax.ShapeDtypeStruct(diag_value.shape, diag_value.dtype),
    )


csr_diag_add_p = XLACustomKernel('csr_diag_add')
csr_diag_add_p.def_numba_kernel(_csr_diag_add_numba_kernel_generator)
csr_diag_add_p.def_warp_kernel(_csr_diag_add_warp_kernel_generator)
csr_diag_add_p.def_pallas_kernel('gpu', _csr_diag_add_pallas_kernel_generator)
csr_diag_add_p.def_pallas_kernel('tpu', _csr_diag_add_pallas_kernel_generator)
csr_diag_add_p.def_jvp_rule2(_csr_diag_add_jvp_csr_value, None, _csr_diag_add_jvp_diag_value)
csr_diag_add_p.def_call(csr_diag_add_call)
