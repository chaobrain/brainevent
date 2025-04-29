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


from typing import Tuple, Union

import brainunit as u
import jax
import jax.numpy as jnp
from jax.interpreters import ad

from ._compatible_import import pallas as pl
from ._fixed_conn_num_float_impl import fixed_post_num_mv_p_call
from ._fixed_conn_num_misc import check_shape, generate_block_dim
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import numba_environ, NumbaKernelGenerator
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_warp import dtype_to_warp_type

TILE_THREADS = 128


def _event_fixed_post_num_mv_cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    if transpose:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting)
                def ell_mv(weights, indices, spikes, _, posts):
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w

            elif float_as_event:
                @numba.njit(**numba_environ.setting)
                def ell_mv(weights, indices, spikes, _, posts):
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i] != 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w

            else:
                @numba.njit(**numba_environ.setting)
                def ell_mv(weights, indices, spikes, _, posts):
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        sp = spikes[i]
                        if sp != 0.:
                            wsp = w * sp
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += wsp

        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting)
                def ell_mv(weights, indices, spikes, _, posts):
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]

            elif float_as_event:
                @numba.njit(**numba_environ.setting)
                def ell_mv(weights, indices, spikes, _, posts):
                    for i in range(spikes.shape[0]):
                        if spikes[i] != 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]

            else:
                @numba.njit(**numba_environ.setting)
                def ell_mv(weights, indices, spikes, _, posts):
                    for i in range(spikes.shape[0]):
                        sp = spikes[i]
                        if sp != 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j] * sp

    else:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def ell_mv(weights, indices, spikes, _, posts):
                    w = weights[0]
                    for i in range(indices.shape[0]):  # n_pre
                        r = 0.
                        for j in range(indices.shape[1]):  # n_conn
                            index = indices[i, j]
                            if spikes[index]:
                                r += w
                        posts[i] = r

            elif float_as_event:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def ell_mv(weights, indices, spikes, _, posts):
                    w = weights[0]
                    for i in range(indices.shape[0]):  # n_pre
                        r = 0.
                        for j in range(indices.shape[1]):  # n_conn
                            index = indices[i, j]
                            if spikes[index] != 0.:
                                r += w
                        posts[i] = r


            else:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def ell_mv(weights, indices, spikes, _, posts):
                    w = weights[0]
                    for i in range(indices.shape[0]):  # n_pre
                        r = 0.
                        for j in range(indices.shape[1]):  # n_conn
                            index = indices[i, j]
                            sp = spikes[index]
                            if sp != 0.:
                                r += sp
                        posts[i] = r * w

        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def ell_mv(weights, indices, spikes, _, posts):
                    for i in range(indices.shape[0]):  # n_pre
                        r = 0.
                        for j in range(indices.shape[1]):  # n_conn
                            index = indices[i, j]
                            if spikes[index]:
                                r += weights[i, j]
                        posts[i] = r

            elif float_as_event:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def ell_mv(weights, indices, spikes, _, posts):
                    for i in range(indices.shape[0]):  # n_pre
                        r = 0.
                        for j in range(indices.shape[1]):  # n_conn
                            index = indices[i, j]
                            if spikes[index] != 0.:
                                r += weights[i, j]
                        posts[i] = r

            else:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def ell_mv(weights, indices, spikes, _, posts):
                    for i in range(indices.shape[0]):  # n_pre
                        r = 0.
                        for j in range(indices.shape[1]):  # n_conn
                            index = indices[i, j]
                            sp = spikes[index]
                            if sp != 0.:
                                r += weights[i, j] * sp
                        posts[i] = r

    return ell_mv


def _event_fixed_post_num_mv_warp_kernel_generator(
    float_as_event: bool,
    transpose: bool,
    block_dim: int,
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    vector_dtype = dtype_to_warp_type(spike_info.dtype)
    indices_dtype = dtype_to_warp_type(indices_info.dtype)

    if transpose:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                def ell_mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    w = weights[0]
                    if spikes[i]:
                        # index = warp.tile_load(indices[i])
                        # warp.tile_atomic_add(posts, w, index)

                        for j in range(0, indices.shape[1], block_dim):
                            index = warp.tile_load(indices[i], block_dim, j)
                            index_thread = warp.untile(index)
                            warp.atomic_add(posts, index_thread, w)

            elif float_as_event:
                def ell_mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    w = weights[0]
                    if spikes[i] != 0.:
                        # index = warp.tile_load(indices[i])
                        # warp.tile_atomic_add(posts, w, index)

                        for j in range(0, indices.shape[1], block_dim):
                            index = warp.tile_load(indices[i], block_dim, j)
                            index_thread = warp.untile(index)
                            warp.atomic_add(posts, index_thread, w)

            else:

                def ell_mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    w = weights[0]
                    sp = spikes[i]
                    if sp != 0.:
                        wsp = w * sp
                        # index = warp.tile_load(indices[i])
                        # warp.tile_atomic_add(posts, wsp, index)

                        for j in range(0, indices.shape[1], block_dim):
                            index = warp.tile_load(indices[i], block_dim, j)
                            index_thread = warp.untile(index)
                            warp.atomic_add(posts, index_thread, wsp)

        else:
            if spike_info.dtype == jnp.bool_:
                def ell_mv(
                    weights: warp.array2d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    if spikes[i]:
                        for j in range(0, indices.shape[1], block_dim):
                            index = warp.tile_load(indices[i], block_dim, j)
                            weight = warp.tile_load(weights[i], block_dim, j)
                            index_thread = warp.untile(index)
                            weight_thread = warp.untile(weight)
                            warp.atomic_add(posts, index_thread, weight_thread)

            elif float_as_event:
                def ell_mv(
                    weights: warp.array2d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    if spikes[i] != 0.:
                        for j in range(0, indices.shape[1], block_dim):
                            index = warp.tile_load(indices[i], block_dim, j)
                            weight = warp.tile_load(weights[i], block_dim, j)
                            index_thread = warp.untile(index)
                            weight_thread = warp.untile(weight)
                            warp.atomic_add(posts, index_thread, weight_thread)

            else:
                def ell_mv(
                    weights: warp.array2d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    sp = spikes[i]
                    if sp != 0.:
                        for j in range(0, indices.shape[1], block_dim):
                            index = warp.tile_load(indices[i], block_dim, j)
                            weight = warp.tile_load(weights[i], block_dim, j) * sp
                            index_thread = warp.untile(index)
                            weight_thread = warp.untile(weight)
                            warp.atomic_add(posts, index_thread, weight_thread)

    else:
        raise NotImplementedError
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                def ell_mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):  # n_conn
                        index = indices[i, j]
                        if spikes[index]:
                            r += w
                    posts[i] = r

            elif float_as_event:
                def ell_mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):  # n_conn
                        index = indices[i, j]
                        if spikes[index] != 0.:
                            r += w
                    posts[i] = r


            else:
                def ell_mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):  # n_conn
                        index = indices[i, j]
                        sp = spikes[index]
                        if sp != 0.:
                            r += sp
                    posts[i] = r * w

        else:
            if spike_info.dtype == jnp.bool_:
                def ell_mv(
                    weights: warp.array2d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):  # n_conn
                        index = indices[i, j]
                        if spikes[index]:
                            r += weights[i, j]
                    posts[i] = r

            elif float_as_event:
                def ell_mv(
                    weights: warp.array2d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):  # n_conn
                        index = indices[i, j]
                        if spikes[index] != 0.:
                            r += weights[i, j]
                    posts[i] = r

            else:
                def ell_mv(
                    weights: warp.array2d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):  # n_conn
                        index = indices[i, j]
                        sp = spikes[index]
                        if sp != 0.:
                            r += weights[i, j] * sp
                    posts[i] = r

    return warp.kernel(ell_mv)


def _event_fixed_post_num_mv_pallas_kernel_generator(
    block_dim: int,
    transpose: int,
    shape: Tuple[int, int],
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = jnp.size(weight_info) == 1

    if transpose:
        if homo:
            def _ell_mv_kernel_homo(
                spike_ref,  # [n_pre]
                index_ref,  # [n_pre, n_conn]
                _,
                out_ref,  # [n_post]
            ):
                i_row = pl.program_id(0)
                spike = spike_ref[i_row]

                def true_fn():
                    def loop_fn(i, _):
                        i = i * block_dim
                        mask = i + jnp.arange(block_dim) < n_conn
                        ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                        data = jnp.ones(block_dim, dtype=weight_info.dtype)
                        if spike_ref.dtype != jnp.bool_ and not float_as_event:
                            data2 = data * spike
                        else:
                            data2 = data
                        pl.atomic_add(out_ref, ind, data2, mask=mask)

                    jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

                jax.lax.cond(spike if spike_ref.dtype == jnp.bool_ else (spike != 0.), true_fn, lambda: None)

            # homogenous weights
            def kernel(weight, indices, spikes, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_homo,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={2: 0},
                )
                return [fn(spikes, indices, out) * weight]

        else:
            def _ell_mv_kernel_heter(
                spike_ref,  # [n_pre]
                index_ref,  # [n_pre, n_conn]
                weight_ref,  # [n_pre, n_conn]
                _,
                out_ref,  # [n_post]
            ):
                i_row = pl.program_id(0)
                spike = spike_ref[i_row]

                def true_fn():
                    def loop_fn(i, _):
                        i = i * block_dim
                        mask = i + jnp.arange(block_dim) < n_conn
                        ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                        weight = pl.load(weight_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                        if spike_ref.dtype != jnp.bool_ and not float_as_event:
                            weight2 = weight * spike
                        else:
                            weight2 = weight
                        pl.atomic_add(out_ref, ind, weight2, mask=mask)

                    jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

                jax.lax.cond(spike if spike_ref.dtype == jnp.bool_ else (spike != 0.), true_fn, lambda: None)

            # heterogeneous weights
            def kernel(weight, indices, spikes, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_heter,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={3: 0},
                )
                return [fn(spikes, indices, weight, out)]

    else:
        if homo:
            def _ell_mv_kernel_homo(
                spike_ref,  # [n_post]
                index_ref,  # [n_pre, n_conn]
                _,  # [n_pre]
                out_ref,  # [n_pre]
            ):
                i_row = pl.program_id(0)

                def loop_fn(i, sum_):
                    i = i * block_dim
                    mask = i + jnp.arange(block_dim) < n_conn
                    ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                    spk = spike_ref[ind]
                    if spike_ref.dtype == jnp.bool_:
                        data = jnp.where(spk, 1., 0.).sum()
                    else:
                        data = jnp.sum(spk)
                    return sum_ + data

                i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
                pl.store(out_ref, i_row, i_row_sum)

            # homogenous weights
            def kernel(weight, indices, spikes, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_homo,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={2: 0}
                )
                return [fn(spikes, indices, out) * weight]

        else:
            def _ell_mv_kernel_heter(
                spike_ref,  # [n_post]
                index_ref,  # [n_pre, n_conn]
                weight_ref,  # [n_pre, n_conn]
                _,  # [n_pre]
                out_ref,  # [n_pre]
            ):
                i_row = pl.program_id(0)

                def loop_fn(i, sum_):
                    i = i * block_dim
                    mask = i + jnp.arange(block_dim) < n_conn
                    ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                    w = pl.load(weight_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                    if spike_ref.dtype == jnp.bool_:
                        data = jnp.where(spike_ref[ind], w, 0.)
                    else:
                        data = spike_ref[ind] * w
                    return sum_ + jnp.sum(data)

                i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
                pl.store(out_ref, i_row, i_row_sum)

            # heterogeneous weights
            def kernel(weight, indices, spikes, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_heter,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={3: 0}
                )
                return [fn(spikes, indices, weight, out)]

    return kernel


def _event_fixed_post_num_mv_jvp_spikes(
    spk_dot,
    weights,
    indices,
    spikes,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return fixed_post_num_mv_p_call(
        weights,
        indices,
        spk_dot,
        shape=shape,
        transpose=transpose,
    )


def _event_fixed_post_num_mv_jvp_weights(
    w_dot,
    weights,
    indices,
    spikes,
    _,
    *,
    shape,
    float_as_event,
    transpose,
    **kwargs
):
    return event_fixed_post_num_mv_p_call(
        w_dot,
        indices,
        spikes,
        float_as_event=float_as_event,
        shape=shape,
        transpose=transpose
    )


def _event_fixed_post_num_mv_transpose_rule(
    ct,
    weights,
    indices,
    spikes,
    _,
    *,
    float_as_event,
    shape,
    transpose,
    weight_info,
    **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(spikes):
        if type(ct) is ad.Zero:
            ct_spk = ad.Zero(spikes)
        else:
            if homo:
                # homogeneous weight
                ct_spk = jax.vmap(lambda idx: jnp.sum(ct[idx] * weights))(indices)
            else:
                # heterogeneous weight
                ct_spk = jax.vmap(lambda idx, w: jnp.inner(ct[idx], w))(indices, weights)
        return weights, indices, ct_spk, _

    else:
        # ∂L/∂w = ∂L/∂y * ∂y/∂w
        if type(ct) is ad.Zero:
            ct_gmax = ad.Zero(weights)
        elif homo:
            # scalar
            ct_gmax = event_fixed_post_num_mv_p_call(
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                spikes,
                shape=shape,
                transpose=transpose,
                float_as_event=float_as_event
            )
            ct_gmax = jnp.inner(ct, ct_gmax[0]).reshape(*weight_info.shape)
        else:
            if transpose:
                ct_gmax = jax.vmap(lambda v, ind: v * ct[ind])(spikes, indices)
            else:
                ct_gmax = jax.vmap(lambda c, ind: c * spikes[ind])(ct, indices)
        return ct_gmax, indices, spikes, _


def event_fixed_post_num_mv_p_call(
    weights,
    indices,
    spikes,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    float_as_event: bool = True,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    out, weights, n_pre, n_post = check_shape(weights, indices, spikes, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    spikes, v_unit = u.split_mantissa_unit(spikes)

    TILE_SIZE = indices.shape[0] if transpose else indices.shape[1]  # for warp
    r = event_fixed_post_num_mv_p(
        weights,
        indices,
        spikes,
        jnp.zeros(out.shape, dtype=out.dtype),
        outs=out,
        shape=shape,
        transpose=transpose,
        TILE_SIZE=TILE_SIZE,
        float_as_event=float_as_event,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
    )
    return (u.maybe_decimal(r * v_unit * w_unit),)


event_fixed_post_num_mv_p = XLACustomKernel(
    'event_fixed_post_num_mv',
    cpu_kernel=NumbaKernelGenerator(
        _event_fixed_post_num_mv_cpu_kernel_generator,
        input_output_aliases={3: 0}
    ),
    # gpu_kernel=WarpKernelGenerator(
    #     _event_fixed_post_num_mv_gpu_kernel_generator,
    #     tile=lambda transpose, indices_info, spike_info, **kwargs: (
    #         spike_info.shape[0]
    #         if transpose else
    #         indices_info.shape[0]
    #     ),
    #     block_dim=TILE_THREADS,
    #     input_output_aliases={3: 0}
    # ),
    gpu_kernel=PallasKernelGenerator(
        _event_fixed_post_num_mv_pallas_kernel_generator,
        block_dim=generate_block_dim,
        input_output_aliases={3: 0}
    ),
    tpu_kernel=PallasKernelGenerator(
        _event_fixed_post_num_mv_pallas_kernel_generator,
        block_dim=generate_block_dim,
        input_output_aliases={3: 0}
    ),
)
event_fixed_post_num_mv_p.defjvp(
    _event_fixed_post_num_mv_jvp_weights,
    None,
    _event_fixed_post_num_mv_jvp_spikes,
    None,
)
event_fixed_post_num_mv_p.def_transpose_rule(
    _event_fixed_post_num_mv_transpose_rule
)
