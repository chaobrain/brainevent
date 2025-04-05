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
from ._fixed_conn_num_misc import generate_block_dim, check_shape
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_warp import WarpKernelGenerator, dtype_to_warp_type


def event_fixed_post_num_mv_numba_kernel_generator(
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


def event_fixed_post_num_mv_warp_kernel_generator(
    float_as_event: bool,
    transpose: bool,
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
                @warp.kernel
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
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += w
                            # posts[indices[i, j]] += weights

            elif float_as_event:
                @warp.kernel
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
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += w

            else:
                @warp.kernel
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
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += wsp

        else:
            if spike_info.dtype == jnp.bool_:
                @warp.kernel
                def ell_mv(
                    weights: warp.array2d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    if spikes[i]:
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += weights[i, j]

            elif float_as_event:
                @warp.kernel
                def ell_mv(
                    weights: warp.array2d(dtype=weight_dtype),
                    indices: warp.array2d(dtype=indices_dtype),
                    spikes: warp.array1d(dtype=vector_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype)
                ):
                    i = warp.tid()
                    if spikes[i] != 0.:
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += weights[i, j]

            else:
                @warp.kernel
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
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += weights[i, j] * sp

    else:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @warp.kernel
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
                @warp.kernel
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
                @warp.kernel
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
                @warp.kernel
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
                @warp.kernel
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
                @warp.kernel
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

    return ell_mv


def event_fixed_post_num_mv_pallas_kernel_generator(
    block_dim: int,
    transpose: int,
    shape: Tuple[int, int],
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    if transpose:
        n_pre, n_post = shape
    else:
        n_post, n_pre = shape
    n_conn = indices_info.shape[1]
    homo = jnp.size(weight_info) == 1

    if transpose:
        # 对于具有形状 [n_event] 的 spikes 向量，以及形状 [n_event, n_conn] 的 indices 和 weights 矩阵，
        # 这个算子的计算逻辑为：
        #
        # - 每个block处理 [block_size] 个事件，每个事件对应一个 pre-synaptic neuron
        # - 每个block处理 [block_size, block_size] 个 indices 和 weights

        if homo:
            def _ell_mv_kernel_homo(
                sp_ref,  # [block_size]
                ind_ref,  # [block_size, block_size]
                _,
                y_ref,  # [n_post]
            ):
                r_pid = pl.program_id(0)
                c_start = pl.program_id(1) * block_dim
                row_length = jnp.minimum(n_pre - r_pid * block_dim, block_dim)
                mask = jnp.arange(block_dim) + c_start < n_conn

                def body_fn(j, _):
                    if sp_ref.dtype == jnp.bool_:
                        def true_fn():
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                            pl.atomic_add(y_ref, ind, jnp.ones(block_dim, dtype=weight_info.dtype), mask=mask)

                        jax.lax.cond(sp_ref[j], true_fn, lambda: None)


                    else:
                        def true_fn(sp):
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                            if float_as_event:
                                pl.atomic_add(y_ref, ind, jnp.ones(block_dim, dtype=weight_info.dtype), mask=mask)
                            else:
                                pl.atomic_add(y_ref, ind, jnp.ones(block_dim, dtype=weight_info.dtype) * sp, mask=mask)

                        sp_ = sp_ref[j]
                        jax.lax.cond(sp_ != 0., true_fn, lambda _: None, sp_)

                jax.lax.fori_loop(0, row_length, body_fn, None)

            # homogenous weights
            kernel = pl.pallas_call(
                _ell_mv_kernel_homo,
                out_shape=[
                    jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
                ],
                in_specs=[
                    pl.BlockSpec((block_dim,), lambda i, j: i),
                    pl.BlockSpec((block_dim, block_dim), lambda i, j: (i, j)),
                    pl.BlockSpec((n_post,), lambda i, j: 0)
                ],
                grid=(
                    pl.cdiv(n_pre, block_dim),
                    pl.cdiv(n_conn, block_dim),
                ),
                input_output_aliases={2: 0},
                interpret=False
            )
            return (
                lambda weight, indices, spikes, _:
                [kernel(spikes, indices, jnp.zeros(n_post, dtype=weight.dtype))[0] * weight]
            )

        else:
            def _ell_mv_kernel_heter(
                sp_ref,  # [block_size]
                ind_ref,  # [block_size, block_size]
                w_ref,  # [block_size, block_size]
                _,
                y_ref,  # [n_post]
            ):
                r_pid = pl.program_id(0)
                c_start = pl.program_id(1) * block_dim
                row_length = jnp.minimum(n_pre - r_pid * block_dim, block_dim)
                mask = jnp.arange(block_dim) + c_start < n_conn

                def body_fn(j, _):
                    if sp_ref.dtype == jnp.bool_:
                        def true_fn():
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                            w = pl.load(w_ref, (j, pl.dslice(None)), mask=mask)
                            pl.atomic_add(y_ref, ind, w, mask=mask)

                        jax.lax.cond(sp_ref[j], true_fn, lambda: None)
                    else:
                        def true_fn(spk):
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                            w = pl.load(w_ref, (j, pl.dslice(None)), mask=mask)
                            if not float_as_event:
                                w = w * spk
                            pl.atomic_add(y_ref, ind, w, mask=mask)

                        sp_ = sp_ref[j]
                        jax.lax.cond(sp_ != 0., true_fn, lambda _: None, sp_)

                jax.lax.fori_loop(0, row_length, body_fn, None)

            # heterogeneous weights
            kernel = pl.pallas_call(
                _ell_mv_kernel_heter,
                out_shape=[
                    jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
                ],
                in_specs=[
                    pl.BlockSpec((block_dim,), lambda i, j: i),  # sp_ref
                    pl.BlockSpec((block_dim, block_dim), lambda i, j: (i, j)),  # ind_ref
                    pl.BlockSpec((block_dim, block_dim), lambda i, j: (i, j)),  # w_ref,
                    pl.BlockSpec((n_post,), lambda i, j: 0)
                ],
                grid=(
                    pl.cdiv(n_pre, block_dim),
                    pl.cdiv(n_conn, block_dim),
                ),
                input_output_aliases={3: 0},
                interpret=False
            )
            return (
                lambda weight, indices, spikes, _:
                kernel(spikes, indices, weight, jnp.zeros(n_post, dtype=weight_info.dtype))
            )

    else:
        raise NotImplementedError


def event_fixed_post_num_mv_jvp_spikes(
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


def event_fixed_post_num_mv_jvp_weights(
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


def event_fixed_post_num_mv_transpose_rule(
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
    n_conn = indices.shape[1]

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

    r = event_fixed_post_num_mv_p(
        weights,
        indices,
        spikes,
        jnp.zeros(out.shape, dtype=out.dtype),
        outs=out,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
    )
    return (u.maybe_decimal(r * v_unit * w_unit),)


event_fixed_post_num_mv_p = XLACustomKernel(
    'event_fixed_post_num_mv',
    cpu_kernel=NumbaKernelGenerator(
        event_fixed_post_num_mv_numba_kernel_generator,
        input_output_aliases={3: 0}
    ),
    gpu_kernel=WarpKernelGenerator(
        event_fixed_post_num_mv_warp_kernel_generator,
        dim=lambda transpose, indices_info, spike_info, **kwargs: (
            spike_info.shape[0]
            if transpose else
            indices_info.shape[0]
        ),
        input_output_aliases={3: 0}
    ),
    tpu_kernel=PallasKernelGenerator(
        event_fixed_post_num_mv_pallas_kernel_generator,
        block_dim=generate_block_dim,
        input_output_aliases={3: 0}
    ),
)
event_fixed_post_num_mv_p.defjvp(
    event_fixed_post_num_mv_jvp_weights,
    None,
    event_fixed_post_num_mv_jvp_spikes,
    None,
)
event_fixed_post_num_mv_p.def_transpose_rule(
    event_fixed_post_num_mv_transpose_rule
)
