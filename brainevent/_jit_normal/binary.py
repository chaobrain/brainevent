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

from typing import Optional, Sequence

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._jitc_matrix import _initialize_seed, _initialize_conn_length
from brainevent._misc import generate_block_dim, namescope
from brainevent._op import XLACustomKernel, jaxinfo_to_warpinfo, numba_kernel, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._pallas_random import PallasLFSR88RNG
from brainevent._typing import Data, MatrixShape
from .float import jitnmv_p_call, jitnmm_p_call

__all__ = [
    "binary_jitnmv",
    "binary_jitnmv_p",
    "binary_jitnmm",
    "binary_jitnmm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitnmv(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    seed = _initialize_seed(seed)
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = binary_jitnmv_p_call(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


def binary_jitnmm(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    seed = _initialize_seed(seed)
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = binary_jitnmm_p_call(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitc_mv_normal_numba_kernel_generator(
    corder: bool,
    vector_info: jax.ShapeDtypeStruct,
    **kwargs
):
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_normal` operation.
    """
    import numba

    if corder:
        # This means that the for loop is parallelized along the dimension of the output vector: ``post.shape[0]``.
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, vector, seed, posts):
                posts[:] = 0.
                # Output vector dimension = number of columns in the matrix
                n_col = posts.shape[0]

                # Input vector dimension = number of rows in the matrix
                n_row = vector.shape[0]

                # Extract scalar values from input arrays
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length (inverse of connection probability)
                seed0 = seed[0]  # Random seed

                # Initialize the random number generator with the provided seed
                # This ensures reproducibility for the same seed value
                np.random.seed(seed0)

                # Process each output element (column in the matrix)
                for i_col in range(n_col):
                    # Generate first row index randomly - this determines where to start sampling
                    i_row = np.random.randint(0, clen0)

                    # Initialize accumulator for this output element with proper dtype
                    out = np.asarray(0., dtype=posts.dtype)

                    # Process all connected entries for this column
                    while i_row < n_row:
                        w = np.random.normal(loc=w_loc0, scale=w_scale0)
                        if vector[i_row]:
                            out += w

                        # Skip ahead to next connected row (sparse sampling)
                        # The random skip ensures proper connection probability
                        # Each skip distance is randomly determined to maintain the sparse pattern
                        i_row += np.random.randint(1, clen0)

                    posts[i_col] = out
        else:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, vector, seed, posts):
                posts[:] = 0.
                # Output vector dimension = number of columns in the matrix
                n_col = posts.shape[0]

                # Input vector dimension = number of rows in the matrix
                n_row = vector.shape[0]

                # Extract scalar values from input arrays
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length (inverse of connection probability)
                seed0 = seed[0]  # Random seed

                # Initialize the random number generator with the provided seed
                # This ensures reproducibility for the same seed value
                np.random.seed(seed0)

                # Process each output element (column in the matrix)
                for i_col in range(n_col):
                    # Generate first row index randomly - this determines where to start sampling
                    i_row = np.random.randint(0, clen0)

                    # Initialize accumulator for this output element with proper dtype
                    out = np.asarray(0., dtype=posts.dtype)

                    # Process all connected entries for this column
                    while i_row < n_row:
                        w = np.random.normal(loc=w_loc0, scale=w_scale0)
                        if vector[i_row] > 0.:
                            out += w

                        # Skip ahead to next connected row (sparse sampling)
                        # The random skip ensures proper connection probability
                        # Each skip distance is randomly determined to maintain the sparse pattern
                        i_row += np.random.randint(1, clen0)

                    posts[i_col] = out

    else:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                np.random.seed(seed0)
                for i_row in range(num_row):
                    v = vector[i_row]
                    i_col = np.random.randint(0, clen0)
                    while i_col < num_col:
                        w = np.random.normal(loc=w_loc0, scale=w_scale0)
                        if v:
                            posts[i_col] += w
                        i_col += np.random.randint(1, clen0)
        else:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                np.random.seed(seed0)
                for i_row in range(num_row):
                    v = vector[i_row] > 0.
                    i_col = np.random.randint(0, clen0)
                    while i_col < num_col:
                        w = np.random.normal(loc=w_loc0, scale=w_scale0)
                        if v:
                            posts[i_col] += w
                        i_col += np.random.randint(1, clen0)

    def run(w_loc, w_scale, clen, vector, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, vector, seed)

    return run


def _jitc_mv_normal_warp_kernel_generator(
    w_loc_info: jax.ShapeDtypeStruct,
    w_scale_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    r"""
    Generate the GPU kernel for the :func:`_jitc_matvec_normal` operation.

    JITC matrix generation must be consistent with _jitn_warp_kernel_generator in float.py:

    - corder=True:  i_row = tid(), loop over i_col, seed = seed0 + i_row
    - corder=False: i_col = tid(), loop over i_row, seed = seed0 + i_col
    """
    import warp
    from warp.jax_experimental import jax_kernel

    w_loc_warp_info = jaxinfo_to_warpinfo(w_loc_info)
    w_scale_warp_info = jaxinfo_to_warpinfo(w_scale_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    v_warp_info = jaxinfo_to_warpinfo(vector_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(out_info)

    if corder:
        # Consistent with jitn corder=True: i_row=tid(), loop over i_col
        # Each thread produces one output element posts[i_row] — no atomics needed.
        if vector_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel(
                w_loc: w_loc_warp_info,
                w_scale: w_scale_warp_info,
                clen: clen_warp_info,
                vector: v_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                num_col = vector.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                r = float(0.0)
                state = warp.rand_init(seed0 + i_row * num_col)
                i_col = warp.randi(state, 0, clen0)
                while i_col < num_col:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    r = warp.where(vector[i_col], r + w, r)
                    i_col += warp.randi(state, 1, clen0)
                posts[i_row] = r

        else:
            @warp.kernel
            def kernel(
                w_loc: w_loc_warp_info,
                w_scale: w_scale_warp_info,
                clen: clen_warp_info,
                vector: v_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                num_col = vector.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                r = float(0.0)
                state = warp.rand_init(seed0 + i_row * num_col)
                i_col = warp.randi(state, 0, clen0)
                while i_col < num_col:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    if vector[i_col] > float(0.0):
                        r += w
                    i_col += warp.randi(state, 1, clen0)
                posts[i_row] = r

        def run(w_loc, w_scale, clen, vector, seed):
            dim = out_info.shape[0]
            fn = jax_kernel(kernel, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(w_loc, w_scale, clen, vector, seed)

    else:
        # Consistent with jitn corder=False: i_col=tid(), loop over i_row
        # Multiple threads may scatter into the same output row — must use atomic_add.
        if vector_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel(
                w_loc: w_loc_warp_info,
                w_scale: w_scale_warp_info,
                clen: clen_warp_info,
                vector: v_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                num_row = posts.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                v = vector[i_col]
                state = warp.rand_init(seed0 + i_col * num_row)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    if v:
                        warp.atomic_add(posts, i_row, w)
                    i_row += warp.randi(state, 1, clen0)

        else:
            @warp.kernel
            def kernel(
                w_loc: w_loc_warp_info,
                w_scale: w_scale_warp_info,
                clen: clen_warp_info,
                vector: v_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                num_row = posts.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                v = vector[i_col]
                state = warp.rand_init(seed0 + i_col * num_row)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    if v > float(0.0):
                        warp.atomic_add(posts, i_row, w)
                    i_row += warp.randi(state, 1, clen0)

        def run(w_loc, w_scale, clen, vector, seed):
            dim = vector_info.shape[0]
            fn = jax_kernel(kernel, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(w_loc, w_scale, clen, vector, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return run


def _jitc_mv_normal_pallas_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Pallas GPU kernel for binary event matvec with normal-distributed JITC matrix.

    JITC matrix generation must be consistent with _jitnmv_pallas_kernel_generator
    in float.py (same RNG seeding and iteration order):
    - corder=True:  vectorize over output rows (i_cols), loop over input (i_rows)
    - corder=False: vectorize over input rows (i_rows), loop over output (i_cols)
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    block_size = generate_block_dim(dim, maximum=128)

    if corder:
        # Matches float.py _jitnmv_pallas corder=True exactly:
        # vectorize over output, seed by i_cols, loop over i_rows.
        # Binary: accumulate w only when vector[i_row] is event (>0 or True).
        def kernel(w_loc_ref, w_scale_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_row = vector_ref.shape[0]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(data):
                i_rows, i_row_mask, rng, out = data
                v = vector_ref[i_rows]
                if vector_ref.dtype != jnp.bool_:
                    v = v > 0.
                w = rng.normal(w_loc, w_scale)
                out = jnp.where(i_row_mask & v, out + w, out)
                i_rows += rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, out

            rng = PallasLFSR88RNG(seed + i_cols * num_row)
            i_rows = rng.random_integers(0, clen)
            i_row_mask = i_rows < num_row
            out = jnp.zeros(block_size, dtype=post_ref.dtype)
            out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng, out)
            )[-1]
            post_ref[i_cols] = jnp.where(i_col_mask, out, post_ref[i_cols])

    else:
        # Matches float.py _jitnmv_pallas corder=False exactly:
        # vectorize over input, seed by i_rows, loop over i_cols.
        # Binary: only scatter w (via atomic_add) when vector element is event.
        def kernel(w_loc_ref, w_scale_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim
            v = vector_ref[i_rows]
            if vector_ref.dtype != jnp.bool_:
                v = v > 0.
            # event_mask: only active lanes where the vector element is an event
            event_mask = i_row_mask & v

            def body(data):
                i_cols, i_col_mask, rng = data
                w = rng.normal(w_loc, w_scale)
                atomic_add(post_ref, (i_cols,), w, mask=event_mask & i_col_mask)
                i_cols += rng.random_integers(1, clen)
                return i_cols, i_cols < num_col, rng

            rng = PallasLFSR88RNG(seed + i_rows * num_col)
            i_cols = rng.random_integers(0, clen)
            i_col_mask = i_cols < num_col
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng)
            )

    def run(w_loc, w_scale, clen, vector, seed):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(dim, block_size),),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_loc, w_scale, clen, vector, seed, placeholder)

    return run


def _jitc_mv_normal_jvp_v(v_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    return jitnmv_p_call(
        w_loc, w_scale, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mv_normal_jvp_wloc(w_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    return binary_jitnmv_p_call(
        w_dot, w_scale, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mv_normal_jvp_wscale(w_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    return binary_jitnmv_p_call(
        w_loc, w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mv_normal_transpose_rules(ct, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        r = jitnmv_p_call(
            w_loc,
            w_scale,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            backend=kwargs['backend'],
        )[0]
        return w_loc, w_scale, clen, r, seed
    elif ad.is_undefined_primal(w_loc):
        # M = (w_loc + w_scale * Z) * mask, forward: M @ event(v)
        # d(loss)/d(w_loc) = sum((mask^T @ ct) * vector)
        r = jitnmv_p_call(
            1., 0., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw_loc = jnp.expand_dims(jnp.sum(r * vector), axis=0)
        return dw_loc, w_scale, clen, vector, seed
    elif ad.is_undefined_primal(w_scale):
        # d(loss)/d(w_scale) = sum(((Z*mask)^T @ ct) * vector)
        r = jitnmv_p_call(
            0., 1., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw_scale = jnp.expand_dims(jnp.sum(r * vector), axis=0)
        return w_loc, dw_scale, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for binary_jitnmv not implemented "
            f"when none of vector/w_loc/w_scale is an undefined primal."
        )


def _jitc_mv_normal_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitnmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitnmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(
            binary_jitnmv_p,
            args,
            axes,
            **kwargs,
        )


def _binary_jitnmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                w_loc = jnp.ones(1, dtype=dtype)
                w_scale = jnp.ones(1, dtype=dtype) * 0.1
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    vector = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    vector = jnp.asarray(np.random.rand(v_size), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(BenchmarkConfig(name, (w_loc, w_scale, clen, vector, seed), {
                    'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
                }))
    return configs


def binary_jitnmv_p_call(
    w_loc,
    w_scale,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert w_loc.shape == (1,), f"The weight shape should be (1,), but got {w_loc.shape}."
    assert w_scale.shape == (1,), f"The weight shape should be (1,), but got {w_scale.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_loc.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_loc.dtype)
    )

    return binary_jitnmv_p(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        outs=[out_info],
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


binary_jitnmv_p = XLACustomKernel('event_jitc_mv_normal')
binary_jitnmv_p.def_numba_kernel(_jitc_mv_normal_numba_kernel_generator)
binary_jitnmv_p.def_warp_kernel(_jitc_mv_normal_warp_kernel_generator)
binary_jitnmv_p.def_pallas_kernel('gpu', _jitc_mv_normal_pallas_kernel_generator)
binary_jitnmv_p.def_jvp_rule2(_jitc_mv_normal_jvp_wloc, _jitc_mv_normal_jvp_wscale, None, _jitc_mv_normal_jvp_v, None)
binary_jitnmv_p.def_transpose_rule(_jitc_mv_normal_transpose_rules)
binary_jitnmv_p.def_batching_rule(_jitc_mv_normal_batching)
binary_jitnmv_p.def_tags('jit_normal', 'binary')
binary_jitnmv_p.def_benchmark_data(_binary_jitnmv_benchmark_data)


def _jitc_mm_normal_numba_kernel_generator(
    transpose: bool,
    corder: bool,
    B_info: jax.ShapeDtypeStruct,
    **kwargs
):
    r"""
    Generate the CPU kernel for the :func:`_jitc_matmat_normal` operation.
    """
    import numba

    if corder:
        # JIT Matrix.T @ B
        #
        # - JIT matrix: [k, m]
        # - B: [k, n]

        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                k = B.shape[0]  # Number of rows in B (rows in M)

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)

                for i_m in range(m):
                    i_k = np.random.randint(0, clen0)
                    out = np.zeros(n, dtype=posts.dtype)
                    while i_k < k:
                        w = np.random.normal(w_loc0, w_scale0)
                        for j in range(B.shape[1]):
                            if B[i_k, j]:
                                out[j] += w
                        i_k += np.random.randint(1, clen0)
                    posts[i_m] = out
        else:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                k = B.shape[0]  # Number of rows in B (rows in M)

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)

                for i_m in range(m):
                    i_k = np.random.randint(0, clen0)
                    out = np.zeros(n, dtype=posts.dtype)
                    while i_k < k:
                        w = np.random.normal(w_loc0, w_scale0)
                        for j in range(B.shape[1]):
                            if B[i_k, j] > 0.:
                                out[j] += w
                        i_k += np.random.randint(1, clen0)
                    posts[i_m] = out


    else:
        # JIT Matrix.T @ B
        #
        # - JIT matrix: [k, m]
        # - B: [k, n]

        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                k = B.shape[0]  # Number of rows in B (rows in M)

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)  # Initialize random number generator with seed

                for i_k in range(k):
                    indices = np.where(B[i_k])[0]
                    i_m = np.random.randint(0, clen0)
                    while i_m < m:
                        w = np.random.normal(w_loc0, w_scale0)
                        posts[i_m, indices] += w
                        i_m += np.random.randint(1, clen0)
        else:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                k = B.shape[0]  # Number of rows in B (rows in M)

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)  # Initialize random number generator with seed

                for i_k in range(k):
                    indices = np.where(B[i_k] > 0.)[0]
                    i_m = np.random.randint(0, clen0)
                    while i_m < m:
                        w = np.random.normal(w_loc0, w_scale0)
                        posts[i_m, indices] += w
                        i_m += np.random.randint(1, clen0)

    def run(w_loc, w_scale, clen, B, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, B, seed)

    return run


def _jitc_mm_normal_warp_kernel_generator(
    w_loc_info: jax.ShapeDtypeStruct,
    w_scale_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    r"""
    Generate the GPU kernel for the :func:`_jitc_matmat_normal` operation.

    Uses scalar loops instead of warp tile operations to avoid cooperative
    warp issues. JITC matrix generation must be consistent with
    _jitn_warp_kernel_generator in float.py:
    - corder=True:  i_m = tid(), loop over i_k, seed = seed0 + i_m
    - corder=False: i_k = tid(), loop over i_m, seed = seed0 + i_k
    """
    import warp
    from warp.jax_experimental import jax_kernel

    w_loc_warp_info = jaxinfo_to_warpinfo(w_loc_info)
    w_scale_warp_info = jaxinfo_to_warpinfo(w_scale_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    B_warp_info = jaxinfo_to_warpinfo(B_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(out_info)

    if corder:
        # Consistent with jitn corder=True: i_m=tid(), loop over i_k
        # Each thread produces one output row — no atomics needed.
        if B_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel(
                w_loc: w_loc_warp_info,
                w_scale: w_scale_warp_info,
                clen: clen_warp_info,
                B: B_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                k = B.shape[0]
                n = B.shape[1]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m * k)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    for j in range(n):
                        if B[i_k, j]:
                            posts[i_m, j] += w
                    i_k += warp.randi(state, 1, clen0)

        else:
            @warp.kernel
            def kernel(
                w_loc: w_loc_warp_info,
                w_scale: w_scale_warp_info,
                clen: clen_warp_info,
                B: B_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                k = B.shape[0]
                n = B.shape[1]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m * k)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    for j in range(n):
                        if B[i_k, j] > float(0.0):
                            posts[i_m, j] += w
                    i_k += warp.randi(state, 1, clen0)

        def run(w_loc, w_scale, clen, B, seed):
            dim = out_info.shape[0]
            fn = jax_kernel(kernel, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(w_loc, w_scale, clen, B, seed, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        # Consistent with jitn corder=False: i_k=tid(), loop over i_m
        # Multiple threads scatter into output rows — must use atomic_add.
        if B_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel(
                w_loc: w_loc_warp_info,
                w_scale: w_scale_warp_info,
                clen: clen_warp_info,
                B: B_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                m = posts.shape[0]
                n = B.shape[1]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k * m)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    for j in range(n):
                        if B[i_k, j]:
                            warp.atomic_add(posts, i_m, j, w)
                    i_m += warp.randi(state, 1, clen0)

        else:
            @warp.kernel
            def kernel(
                w_loc: w_loc_warp_info,
                w_scale: w_scale_warp_info,
                clen: clen_warp_info,
                B: B_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                m = posts.shape[0]
                n = B.shape[1]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k * m)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    for j in range(n):
                        if B[i_k, j] > float(0.0):
                            warp.atomic_add(posts, i_m, j, w)
                    i_m += warp.randi(state, 1, clen0)

        def run(w_loc, w_scale, clen, B, seed):
            dim = B_info.shape[0]
            fn = jax_kernel(kernel, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(w_loc, w_scale, clen, B, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return run


def _jitc_mm_normal_pallas_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Pallas GPU kernel for binary event matmat with normal-distributed JITC matrix.

    Matches _jitnmm_pallas_kernel_generator in float.py:
    - Grid: (row_or_k_blocks, B_cols) — each block processes one B column
    - corder=True:  vectorize over output rows, seed by i_rows, loop over k
    - corder=False: vectorize over k, seed by i_ks, loop over output rows
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    B_cols = B_info.shape[1]

    if corder:
        # Match float.py _jitnmm_pallas corder=True exactly:
        # Grid: (row_blocks, B_cols). Each block processes one B column.
        out_rows = out_info.shape[0]
        row_block = generate_block_dim(out_rows, maximum=128)
        grid = (pl.cdiv(out_rows, row_block), B_cols)

        def kernel(w_loc_ref, w_scale_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            k = B_ref.shape[0]
            w_loc0 = w_loc_ref[0]
            w_scale0 = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            col_j = pl.program_id(1)

            i_rows = i_row_block * row_block + jnp.arange(row_block)
            i_row_mask = i_rows < out_rows
            safe_rows = jnp.where(i_row_mask, i_rows, 0)

            rng = PallasLFSR88RNG(seed0 + i_rows * k)
            i_cols = rng.random_integers(0, clen0)
            i_col_mask = i_cols < k

            out = jnp.zeros(row_block, dtype=post_ref.dtype)

            def body(data):
                i_cols, i_col_mask, rng, out = data
                w = rng.normal(w_loc0, w_scale0)
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                b_vals = B_ref[safe_cols, col_j]
                # Binary thresholding: treat b_vals as events
                if B_ref.dtype == jnp.bool_:
                    events = jnp.asarray(b_vals, dtype=out.dtype)
                else:
                    events = jnp.where(b_vals > 0., 1., 0.)
                out += jnp.where(i_col_mask & i_row_mask, w * events, 0.)
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < k, rng, out

            _, _, _, out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng, out)
            )
            atomic_add(post_ref, (safe_rows, col_j), out, mask=i_row_mask)

    else:
        # Match float.py _jitnmm_pallas corder=False exactly:
        # Grid: (k_blocks, B_cols). Each block processes one B column.
        k_dim = B_info.shape[0]
        k_block = generate_block_dim(k_dim, maximum=128)
        grid = (pl.cdiv(k_dim, k_block), B_cols)

        def kernel(w_loc_ref, w_scale_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            m = post_ref.shape[0]
            w_loc0 = w_loc_ref[0]
            w_scale0 = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_k_block = pl.program_id(0)
            col_j = pl.program_id(1)

            i_ks = i_k_block * k_block + jnp.arange(k_block)
            i_k_mask = i_ks < k_dim
            safe_ks = jnp.where(i_k_mask, i_ks, 0)

            # Preload B values for this column and apply binary thresholding
            b_vals = B_ref[safe_ks, col_j]
            if B_ref.dtype == jnp.bool_:
                b_events = jnp.asarray(b_vals, dtype=post_ref.dtype)
            else:
                b_events = jnp.where(b_vals > 0., 1., 0.)
            b_events = jnp.where(i_k_mask, b_events, 0.)

            rng = PallasLFSR88RNG(seed0 + i_ks * m)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < m

            def body(data):
                i_rows, i_row_mask, rng = data
                w = rng.normal(w_loc0, w_scale0)
                vals = jnp.where(i_k_mask & i_row_mask, w * b_events, 0.)
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                atomic_add(post_ref, (safe_rows, col_j), vals,
                           mask=i_k_mask & i_row_mask)
                i_rows += rng.random_integers(1, clen0)
                return i_rows, i_rows < m, rng

            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng)
            )

    def run(w_loc, w_scale, clen, B, seed):
        fn = pl.pallas_call(
            kernel,
            grid=grid,
            input_output_aliases={5: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_loc, w_scale, clen, B, seed, placeholder)

    return run


def _jitc_mm_normal_jvp_wloc(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, **kwargs):
    return binary_jitnmm_p_call(
        w_dot, w_scale, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mm_normal_jvp_wscale(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, **kwargs):
    return binary_jitnmm_p_call(
        w_loc, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mm_normal_jvp_B(B_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, **kwargs):
    return jitnmm_p_call(
        w_loc, w_scale, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mm_normal_transpose_rules(ct, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        r = jitnmm_p_call(
            w_loc,
            w_scale,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            backend=kwargs['backend'],
        )[0]
        return w_loc, w_scale, clen, r, seed
    elif ad.is_undefined_primal(w_loc):
        # M = (w_loc + w_scale * Z) * mask, forward: M @ event(B)
        # d(loss)/d(w_loc) = sum((mask^T @ ct) * B)
        r = jitnmm_p_call(
            1., 0., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw_loc = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return dw_loc, w_scale, clen, B, seed
    elif ad.is_undefined_primal(w_scale):
        # d(loss)/d(w_scale) = sum(((Z*mask)^T @ ct) * B)
        r = jitnmm_p_call(
            0., 1., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw_scale = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return w_loc, dw_scale, clen, B, seed
    else:
        raise NotImplementedError(
            'Transpose rules for binary_jitc_matmat_normal not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[3].shape
    B = args[3].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_jitnmm_p_call(
        args[0],
        args[1],
        args[2],
        B,
        args[4],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitc_mm_normal_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[3] = jnp.transpose(args[3], (1, 0, 2))
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, None, 1, None):
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, None, 2, None):
        return _batching_axis1(args, axis=2, **kwargs)

    else:
        return general_batching_rule(binary_jitnmm_p, args, axes, **kwargs)


def _binary_jitnmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                w_loc = jnp.ones(1, dtype=dtype)
                w_scale = jnp.ones(1, dtype=dtype) * 0.1
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    B = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    B = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (w_loc, w_scale, clen, B, seed),
                        {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                    )
                )
    return configs


def binary_jitnmm_p_call(
    w_loc,
    w_scale,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert w_loc.ndim == 1, "The weight should be a 1D array."
    assert w_scale.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert w_loc.shape == (1,), "The weight should be a scalar."
    assert w_scale.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_loc.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_loc.dtype)
    )

    return binary_jitnmm_p(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        outs=[out_info],
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


binary_jitnmm_p = XLACustomKernel('binary_jitc_mm_normal')
binary_jitnmm_p.def_numba_kernel(_jitc_mm_normal_numba_kernel_generator)
binary_jitnmm_p.def_warp_kernel(_jitc_mm_normal_warp_kernel_generator)
binary_jitnmm_p.def_pallas_kernel('gpu', _jitc_mm_normal_pallas_kernel_generator)
binary_jitnmm_p.def_jvp_rule2(_jitc_mm_normal_jvp_wloc, _jitc_mm_normal_jvp_wscale, None, _jitc_mm_normal_jvp_B, None)
binary_jitnmm_p.def_transpose_rule(_jitc_mm_normal_transpose_rules)
binary_jitnmm_p.def_batching_rule(_jitc_mm_normal_batching)
binary_jitnmm_p.def_tags('jit_normal', 'binary')
binary_jitnmm_p.def_benchmark_data(_binary_jitnmm_benchmark_data)
