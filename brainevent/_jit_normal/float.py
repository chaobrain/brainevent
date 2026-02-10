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

from typing import Optional

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._compatible_import import Tracer
from brainevent._jitc_matrix import _initialize_seed, _initialize_conn_length
from brainevent._misc import generate_block_dim
from brainevent._op import XLACustomKernel, jaxinfo_to_warpinfo, numba_kernel, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._pallas_random import PallasLFSR88RNG
from brainevent._typing import Data, MatrixShape

__all__ = [
    "jitn",
    "jitn_p",
    "jitnmv",
    "jitnmv_p",
    "jitnmm",
    "jitnmm_p",
]


def _is_static_zero_prob(prob: float, *, op_name: str) -> bool:
    if isinstance(prob, Tracer):
        return False
    prob_arr = np.asarray(prob)
    if prob_arr.size != 1:
        raise ValueError(f"{op_name}: prob must be a scalar, but got shape {prob_arr.shape}.")
    prob_scalar = float(prob_arr.item())
    if not np.isfinite(prob_scalar):
        raise ValueError(f"{op_name}: prob must be finite, but got {prob_scalar}.")
    if not (0. <= prob_scalar <= 1.):
        raise ValueError(f"{op_name}: prob must be in [0, 1], but got {prob_scalar}.")
    return prob_scalar == 0.


def jitn(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    out_dtype = jnp.asarray(w_loc).dtype
    if _is_static_zero_prob(prob, op_name="jitn"):
        out_shape = shape[::-1] if transpose else shape
        return u.maybe_decimal(jnp.zeros(out_shape, dtype=out_dtype) * unitd)
    clen = _initialize_conn_length(prob)
    res = jitn_p_call(
        w_loc,
        w_scale,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd)


def jitnmv(
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
    out_dtype = jnp.asarray(w_loc).dtype
    if _is_static_zero_prob(prob, op_name="jitnmv"):
        if vector.ndim != 1:
            raise ValueError(f"jitnmv: vector must be 1D, but got {vector.ndim}D.")
        expected = shape[0] if transpose else shape[1]
        if vector.shape[0] != expected:
            raise ValueError(
                f"jitnmv: shape mismatch, got matrix shape {shape} and vector shape {vector.shape}."
            )
        out_size = shape[1] if transpose else shape[0]
        return u.maybe_decimal(jnp.zeros((out_size,), dtype=out_dtype) * unitd * unitv)
    clen = _initialize_conn_length(prob)
    res = jitnmv_p_call(
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


def jitnmm(
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
    out_dtype = jnp.asarray(w_loc).dtype
    if _is_static_zero_prob(prob, op_name="jitnmm"):
        if B.ndim != 2:
            raise ValueError(f"jitnmm: B must be 2D, but got {B.ndim}D.")
        expected = shape[0] if transpose else shape[1]
        if B.shape[0] != expected:
            raise ValueError(
                f"jitnmm: shape mismatch, got matrix shape {shape} and B shape {B.shape}."
            )
        out_shape = (shape[1], B.shape[1]) if transpose else (shape[0], B.shape[1])
        return u.maybe_decimal(jnp.zeros(out_shape, dtype=out_dtype) * unitd * unitB)
    clen = _initialize_conn_length(prob)
    res = jitnmm_p_call(
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


def _jitn_numba_kernel_generator(corder: bool = True, **kwargs):
    import numba

    if corder:
        # JIT matrix
        #
        # - JIT matrix shape = [m, n]
        #

        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]
            n = posts.shape[1]

            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = seed[0]  # Random seed for reproducible matrix generation
            clen0 = clen[0]  # Connection length parameter (controls sparsity)

            # Initialize the random number generator with the provided seed
            # This ensures the "random" matrix is reproducible for the same seed value
            np.random.seed(seed0)

            # Process each output element (each row of the matrix)
            for i_row in range(m):
                # Generate first column index randomly - this determines where to start sampling
                i_col = np.random.randint(0, clen0)

                # Process all connected entries for this row
                while i_col < n:
                    # Set the current matrix element to the weight value
                    posts[i_row, i_col] = np.random.normal(loc=w_loc0, scale=w_scale0)

                    # Skip ahead to next connected column (sparse sampling)
                    # The random skip ensures proper connection probability
                    i_col += np.random.randint(1, clen0)

    else:
        # This means that the for loop is parallelized along the dimension of the vector: ``vector.shape[0]``.

        # JIT matrix
        #
        # - JIT matrix shape = [m, n]
        #

        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]  # Number of rows in the output matrix
            n = posts.shape[1]  # Number of columns in the output matrix

            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
            seed0 = seed[0]  # Random seed for reproducible matrix generation

            # Initialize the random number generator with the provided seed
            # This ensures reproducibility for the same seed value
            np.random.seed(seed0)

            # Process each column of the matrix sequentially
            for i_col in range(n):
                # Generate first row index randomly - this determines where to start sampling in this column
                i_row = np.random.randint(0, clen0)

                # Process all connected entries for this column
                while i_row < m:
                    # Set the current matrix element to the weight value
                    posts[i_row, i_col] = np.random.normal(loc=w_loc0, scale=w_scale0)

                    # Skip ahead to next connected row (sparse sampling)
                    # The random skip ensures proper connection probability
                    i_row += np.random.randint(1, clen0)

    def run(w_loc, w_scale, clen, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, seed)

    return run


def _jitn_warp_kernel_generator(
    w_loc_info: jax.ShapeDtypeStruct,
    w_scale_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    w_loc_warp_info = jaxinfo_to_warpinfo(w_loc_info)
    w_scale_warp_info = jaxinfo_to_warpinfo(w_scale_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(out_info)

    if corder:
        # JIT matrix - JIT matrix shape = [m, n]
        @warp.kernel
        def kernel(
            w_loc: w_loc_warp_info,
            w_scale: w_scale_warp_info,
            clen: clen_warp_info,
            seed: seed_warp_info,
            posts: out_warp_info,
        ):
            n = posts.shape[1]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            clen0 = clen[0]
            seed0 = seed[0]
            i_row = warp.tid()
            state = warp.rand_init(seed0 + i_row)
            i_col = warp.randi(state, 0, clen0)
            while i_col < n:
                posts[i_row, i_col] = warp.randn(state) * w_scale0 + w_loc0
                i_col += warp.randi(state, 1, clen0)


    else:
        # JIT matrix - JIT matrix shape = [m, n]
        @warp.kernel
        def kernel(
            w_loc: w_loc_warp_info,
            w_scale: w_scale_warp_info,
            clen: clen_warp_info,
            seed: seed_warp_info,
            posts: out_warp_info,
        ):
            m = posts.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            clen0 = clen[0]
            seed0 = seed[0]
            i_col = warp.tid()
            state = warp.rand_init(seed0 + i_col)
            i_row = warp.randi(state, 0, clen0)
            while i_row < m:
                posts[i_row, i_col] = warp.randn(state) * w_scale0 + w_loc0
                i_row += warp.randi(state, 1, clen0)

    def run(w_loc, w_scale, clen, seed):
        dim = out_info.shape[0] if corder else out_info.shape[1]
        fn = jax_kernel(kernel, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
        return fn(w_loc, w_scale, clen, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return run


def _jitn_pallas_kernel_generator(
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl

    dim = out_info.shape[0] if corder else out_info.shape[1]
    block_size = generate_block_dim(dim, maximum=128)
    n_block = pl.cdiv(dim, block_size)

    if corder:
        def kernel(w_loc_ref, w_scale_ref, clen_ref, seed_ref, _, post_ref):
            m = post_ref.shape[1]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim

            def body(data):
                i_cols, i_col_mask, rng = data
                val = rng.normal(w_loc, w_scale)
                post_ref[i_rows, i_cols] = jnp.where(i_row_mask & i_col_mask, val, post_ref[i_rows, i_cols])
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < m, rng

            rng = PallasLFSR88RNG(seed0 + i_rows)
            i_cols = rng.random_integers(0, clen0)
            i_col_mask = i_cols < m
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng)
            )

    else:
        def kernel(w_loc_ref, w_scale_ref, clen_ref, seed_ref, _, post_ref):
            n = post_ref.shape[0]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(data):
                i_rows, i_row_mask, rng = data
                val = rng.normal(w_loc, w_scale)
                post_ref[i_rows, i_cols] = jnp.where(i_row_mask & i_col_mask, val, post_ref[i_rows, i_cols])
                i_rows = i_rows + rng.random_integers(1, clen0)
                return i_rows, i_rows < n, rng

            rng = PallasLFSR88RNG(seed0 + i_cols)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < n
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng)
            )

    def run(w_loc, w_scale, clen, seed):
        fn = pl.pallas_call(kernel, grid=(n_block,), input_output_aliases={4: 0}, out_shape=kwargs['outs'])
        out = kwargs['outs'][0]
        placeholder = jnp.zeros(out.shape, out.dtype)
        return fn(w_loc, w_scale, clen, seed, placeholder)

    return run


def _jitn_jvp_wlow(w_loc_dot, w_loc, w_scale, clen, seed, *, out_info, **kwargs):
    out = jnp.ones_like(out_info) * w_loc_dot
    return [out]


def _jitn_jvp_whigh(w_scale_dot, w_loc, w_scale, clen, seed, *, shape, transpose: bool, corder: bool, backend=None, **kwargs):
    return jitn_p_call(0., w_scale_dot, clen, seed, shape=shape, transpose=transpose, corder=corder, backend=backend)


def _jitn_transpose(ct, w_loc, w_scale, clen, seed, *, shape, transpose: bool, corder: bool, backend=None, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(w_loc):
        dwlow = jnp.expand_dims(ct.sum(), axis=0)
        return (dwlow, w_scale, clen, seed)

    elif ad.is_undefined_primal(w_scale):
        forward = jitn_p_call(0., 1., clen, seed, shape=shape, transpose=transpose, corder=corder, backend=backend)[0]
        dwscale = jnp.expand_dims((ct * forward).sum(), axis=0)
        return (w_loc, dwscale, clen, seed)

    else:
        raise NotImplementedError(
            'JITC matrix transpose is only implemented for the w_low and w_scale arguments.'
        )


def _jitn_batching(args, axes, **kwargs):
    return general_batching_rule(jitn_p, args, axes, **kwargs)


def _jitn_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_loc = jnp.ones(1, dtype=dtype)
            w_scale = jnp.ones(1, dtype=dtype) * 0.1
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (w_loc, w_scale, clen, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jitn_p_call(w_loc, w_scale, clen, seed, *, shape, transpose: bool, corder: bool, backend: Optional[str] = None):
    seed = _initialize_seed(seed)
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)

    out_info = (
        jax.ShapeDtypeStruct(shape[::-1], dtype=w_loc.dtype)
        if transpose else
        jax.ShapeDtypeStruct(shape, dtype=w_loc.dtype)
    )

    return jitn_p(
        w_loc,
        w_scale,
        clen,
        seed,
        outs=[out_info],
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


jitn_p = XLACustomKernel('float_jitn')
jitn_p.def_numba_kernel(_jitn_numba_kernel_generator)
jitn_p.def_warp_kernel(_jitn_warp_kernel_generator)
jitn_p.def_pallas_kernel('gpu', _jitn_pallas_kernel_generator)
jitn_p.def_pallas_kernel('tpu', _jitn_pallas_kernel_generator)
jitn_p.def_jvp_rule2(_jitn_jvp_wlow, _jitn_jvp_whigh, None, None)
jitn_p.def_transpose_rule(_jitn_transpose)
jitn_p.def_batching_rule(_jitn_batching)
jitn_p.def_tags('jit_normal', 'float')
jitn_p.def_benchmark_data(_jitn_benchmark_data)


def _jitnmv_numba_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba
    if corder:
        # This means that the for loop is parallelized along the dimension of the output vector: ``post.shape[0]``.

        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, vector, seed, posts):
            posts[:] = 0.
            # Output vector dimension = number of rows in the matrix
            # Each row in the matrix will produce one element in the output vector
            num_row = posts.shape[0]

            # Input vector dimension = number of columns in the matrix
            # The input vector must match the number of columns in our implicit matrix
            num_col = vector.shape[0]

            # Extract scalar values from input arrays for more efficient access in loops
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = seed[0]  # Random seed for reproducible matrix generation
            clen0 = clen[0]  # Connection length parameter (controls sparsity)

            # Initialize the random number generator with the provided seed
            # This ensures the "random" matrix is reproducible for the same seed value
            np.random.seed(seed0)

            # Process each output element (each row of the matrix)
            for i_row in range(num_row):
                # Randomly determine the first column where this row has a connection
                # This implements efficient sampling of a sparse pattern
                i_col = np.random.randint(0, clen0)

                # Initialize accumulator for the dot product result for this row
                # Using input vector's dtype ensures proper numerical precision
                out = np.asarray(0., dtype=vector.dtype)

                # Process all connected entries for this row by skipping through columns
                # This is the core sparse sampling algorithm - we only process columns
                # that have connections rather than checking every possible column
                while i_col < num_col:
                    # Add contribution from the current connected element
                    # For connected positions, we add the corresponding vector element
                    out += vector[i_col] * np.random.normal(loc=w_loc0, scale=w_scale0)

                    # Skip ahead to next connected column using geometric-like distribution
                    # The random skip distance models the sparse connectivity pattern
                    # where each position has approximately 1/clen0 probability of connection
                    i_col += np.random.randint(1, clen0)

                posts[i_row] = out

    else:
        # This means that the for loop is parallelized along the dimension of the vector: ``vector.shape[0]``.

        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, vector, seed, posts):
            posts[:] = 0.
            # Output vector dimension = number of rows in the matrix
            # This represents the first dimension of the matrix and the result vector's size
            num_row = posts.shape[0]

            # Input vector dimension = number of columns in the matrix
            # Each element of the input vector corresponds to a column in the matrix
            num_col = vector.shape[0]

            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
            seed0 = seed[0]  # Random seed for reproducible matrix generation

            # Initialize the random number generator with the provided seed
            # This ensures the "random" matrix is reproducible for the same seed value
            np.random.seed(seed0)

            # Process each input element (each column of the matrix)
            # This implements the matrix @ vector operation one column at a time
            for i_col in range(num_col):
                v = vector[i_col]

                # Sample the first connected row using random skipping
                # This implements the sparse sampling - each column connects to ~num_row/clen0 rows on average
                # Starting from a random position in [0,clen0) creates variability in connection patterns
                i_row = np.random.randint(0, clen0)

                # Continue sampling and accumulating while we haven't exceeded output dimension
                # This loop processes all rows this particular column connects to
                while i_row < num_row:
                    # Add this connection's contribution to the appropriate output element
                    # The output is accumulated as we process each column's contributions
                    posts[i_row] += v * np.random.normal(loc=w_loc0, scale=w_scale0)

                    # Move to the next connected row using geometric-like skipping
                    # Each next connection is approximately clen0 positions away on average
                    # This creates a sparse pattern where only ~1/clen0 of all possible connections exist
                    i_row += np.random.randint(1, clen0)

    def run(w_loc, w_scale, clen, vector, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, vector, seed)

    return run


def _jitnmv_warp_kernel_generator(
    w_loc_info: jax.ShapeDtypeStruct,
    w_scale_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    r"""
    Generate the GPU kernel for the :func:`_jitc_matvec_normal` operation.
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
            state = warp.rand_init(seed0 + i_row)
            i_col = warp.randi(state, 0, clen0)
            while i_col < num_col:
                r += vector[i_col] * (warp.randn(state) * w_scale0 + w_loc0)
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
            num_row = posts.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            clen0 = clen[0]
            seed0 = seed[0]
            i_col = warp.tid()
            v = vector[i_col]
            state = warp.rand_init(seed0 + i_col)
            i_row = warp.randi(state, 0, clen0)
            while i_row < num_row:
                warp.atomic_add(posts, i_row, v * (warp.randn(state) * w_scale0 + w_loc0))
                i_row += warp.randi(state, 1, clen0)

    def run(w_loc, w_scale, clen, vector, seed):
        dim = out_info.shape[0] if corder else vector_info.shape[0]
        fn = jax_kernel(kernel, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
        return fn(w_loc, w_scale, clen, vector, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return run


def _jitnmv_pallas_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    block_size = generate_block_dim(dim, maximum=128)

    if corder:
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
                v = jnp.where(i_row_mask, vector_ref[i_rows], 0.)
                out += v * rng.normal(w_loc, w_scale)
                i_rows += rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, out

            rng = PallasLFSR88RNG(seed + i_cols)
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
        def kernel(w_loc_ref, w_scale_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim
            vector = jnp.where(i_row_mask, vector_ref[i_rows], 0.)

            def body(data):
                i_cols, i_col_mask, rng = data
                atomic_add(post_ref, (i_cols,), vector * rng.normal(w_loc, w_scale), mask=i_row_mask & i_col_mask)
                i_cols += rng.random_integers(1, clen)
                return i_cols, i_cols < num_col, rng

            rng = PallasLFSR88RNG(seed + i_rows)
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
            out_shape=kwargs['outs']
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_loc, w_scale, clen, vector, seed, placeholder)

    return run


def _jitnmv_jvp_v(v_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, backend=None, **kwargs):
    return jitnmv_p_call(w_loc, w_scale, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=backend)


def _jitnmv_jvp_wloc(w_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, backend=None, **kwargs):
    return jitnmv_p_call(w_dot, w_scale, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=backend)


def _jitnmv_jvp_wscale(w_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, backend=None, **kwargs):
    return jitnmv_p_call(w_loc, w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=backend)


def _jitnmv_transpose_rules(ct, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, backend=None, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        # Gradient w.r.t. vector: d(loss)/d(v) = M^T @ ct
        r = jitnmv_p_call(
            w_loc,
            w_scale,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            backend=backend,
        )[0]
        return w_loc, w_scale, clen, r, seed
    elif ad.is_undefined_primal(w_loc):
        # M = (w_loc + w_scale * Z) * mask
        # d(M @ v)/d(w_loc) = mask @ v
        # d(loss)/d(w_loc) = ct^T @ (mask @ v) = (mask^T @ ct) . v
        # mask^T @ ct = jitnmv(1., 0., ...) with transposed shape
        r = jitnmv_p_call(
            1., 0., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=backend,
        )[0]
        dw_loc = jnp.expand_dims(jnp.sum(r * vector), axis=0)
        return dw_loc, w_scale, clen, vector, seed
    elif ad.is_undefined_primal(w_scale):
        # M = (w_loc + w_scale * Z) * mask
        # d(M @ v)/d(w_scale) = (Z * mask) @ v
        # d(loss)/d(w_scale) = ct^T @ ((Z * mask) @ v) = ((Z * mask)^T @ ct) . v
        # (Z * mask)^T @ ct = jitnmv(0., 1., ...) with transposed shape
        r = jitnmv_p_call(
            0., 1., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=backend,
        )[0]
        dw_scale = jnp.expand_dims(jnp.sum(r * vector), axis=0)
        return w_loc, dw_scale, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for jitnmv not implemented "
            f"when none of vector/w_loc/w_scale is an undefined primal."
        )


def _jitnmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitnmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs.get('backend'),
        )
        return r, [1]
    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitnmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs.get('backend'),
        )
        return r, [1]
    else:
        return general_batching_rule(
            jitnmv_p,
            args,
            axes,
            **kwargs,
        )


def _jitnmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_loc = jnp.ones(1, dtype=dtype)
            w_scale = jnp.ones(1, dtype=dtype) * 0.1
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w_loc, w_scale, clen, vector, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                )
            )
    return configs


def jitnmv_p_call(
    w_loc,
    w_scale,
    clen,
    vector,
    seed,
    *,
    shape,
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

    return jitnmv_p(
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


jitnmv_p = XLACustomKernel('float_jitnmv')
jitnmv_p.def_numba_kernel(_jitnmv_numba_kernel_generator)
jitnmv_p.def_warp_kernel(_jitnmv_warp_kernel_generator)
jitnmv_p.def_pallas_kernel('gpu', _jitnmv_pallas_kernel_generator)
jitnmv_p.def_pallas_kernel('tpu', _jitnmv_pallas_kernel_generator)
jitnmv_p.def_jvp_rule2(_jitnmv_jvp_wloc, _jitnmv_jvp_wscale, None, _jitnmv_jvp_v, None)
jitnmv_p.def_transpose_rule(_jitnmv_transpose_rules)
jitnmv_p.def_batching_rule(_jitnmv_batching)
jitnmv_p.def_tags('jit_normal', 'float')
jitnmv_p.def_benchmark_data(_jitnmv_benchmark_data)


def _jitnmm_numba_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba
    if corder:
        # JIT Matrix @ B
        #
        # - JIT matrix: [m, k]
        # - B: [k, n]

        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, B, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]  # Number of rows in output matrix (rows in M)
            n = posts.shape[1]  # Number of columns in output matrix (columns in B)
            k = B.shape[0]  # Number of rows in B (columns in M)

            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = seed[0]  # Random seed for reproducible matrix generation
            clen0 = clen[0]  # Connection length parameter (controls sparsity)
            np.random.seed(seed0)  # Initialize random number generator with seed for reproducibility

            for i_m in range(m):
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_k = np.random.randint(0, clen0)

                # Initialize accumulator for this output row with proper dtype
                out = np.zeros(n, dtype=B.dtype)

                # Process all connected entries for this output row
                while i_k < k:
                    # Add contribution from the current connected input row
                    out += B[i_k] * np.random.normal(w_loc0, w_scale0)

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_k += np.random.randint(1, clen0)

                # Scale accumulated sum by weight and store in output array
                posts[i_m] = out

    else:
        # JIT Matrix @ B
        #
        # - JIT matrix: [m, k]
        # - B: [k, n]

        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, B, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]  # Number of rows in output matrix (rows in M)
            k = B.shape[0]  # Number of rows in B (columns in M)

            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = seed[0]  # Random seed for reproducible matrix generation
            clen0 = clen[0]  # Connection length parameter (controls sparsity)
            np.random.seed(seed0)  # Initialize random number generator with seed

            # Process each input column sequentially
            for i_k in range(k):
                # Pre-multiply the current row by weight for efficiency
                out = B[i_k]

                # Sample the first connected output row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_m = np.random.randint(0, clen0)

                # Process all connected output rows for this input column
                while i_m < m:
                    # Add contribution to the connected output row
                    # Using += to accumulate results across all input columns
                    posts[i_m] += out * np.random.normal(w_loc0, w_scale0)

                    # Skip ahead to next connected output row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_m += np.random.randint(1, clen0)

    def run(w_loc, w_scale, clen, B, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, B, seed)

    return run


def _jitnmm_warp_kernel_generator(
    w_loc_info: jax.ShapeDtypeStruct,
    w_scale_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    TITLE_SIZE: int,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    w_loc_warp_info = jaxinfo_to_warpinfo(w_loc_info)
    w_scale_warp_info = jaxinfo_to_warpinfo(w_scale_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    B_warp_info = jaxinfo_to_warpinfo(B_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(out_info)

    if corder:
        # JIT Matrix @ B
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
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_m = warp.tid()
            state = warp.rand_init(seed0 + i_m)

            out = warp.tile_zeros(TITLE_SIZE, dtype=w_loc.dtype)
            i_k = warp.randi(state, 0, clen0)
            while i_k < k:
                w = warp.randn(state) * w_scale0 + w_loc0
                out += warp.tile_load(B[i_k], TITLE_SIZE) * w
                i_k += warp.randi(state, 1, clen0)
            warp.tile_store(posts[i_m], out)

    else:
        # JIT Matrix @ B
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
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_k = warp.tid()
            state = warp.rand_init(seed0 + i_k)

            out = warp.tile_load(B[i_k], TITLE_SIZE)
            i_m = warp.randi(state, 0, clen0)
            while i_m < m:
                w = warp.randn(state) * w_scale0 + w_loc0
                warp.tile_atomic_add(posts[i_m], out * w)
                i_m += warp.randi(state, 1, clen0)

    def run(w_loc, w_scale, clen, B, seed):
        dim = out_info.shape[0] if corder else B_info.shape[0]
        fn = jax_kernel(kernel, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
        return fn(w_loc, w_scale, clen, B, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return run


def _jitnmm_pallas_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    B_cols = B_info.shape[1]

    if corder:
        # Match jitn corder=True: vectorize over rows with identical RNG seeding.
        # Grid: (row_blocks, B_cols). Each kernel block processes one B column,
        # using vector RNG identical to jitn. Accumulates into 1D local array
        # to avoid 2D local arrays (dynamic_slice unsupported in Pallas Triton).
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
            col_j = pl.program_id(1)  # scalar B column index

            # Row indices — VECTOR, matching jitn exactly
            i_rows = i_row_block * row_block + jnp.arange(row_block)
            i_row_mask = i_rows < out_rows
            safe_rows = jnp.where(i_row_mask, i_rows, 0)

            # VECTOR RNG seeded identically to jitn
            rng = PallasLFSR88RNG(seed0 + i_rows)
            i_cols = rng.random_integers(0, clen0)  # [row_block]
            i_col_mask = i_cols < k

            out = jnp.zeros(row_block, dtype=post_ref.dtype)

            def body(data):
                i_cols, i_col_mask, rng, out = data
                w = rng.normal(w_loc0, w_scale0)  # [row_block]
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                b_vals = B_ref[safe_cols, col_j]  # [row_block] vector gather
                out += jnp.where(i_col_mask & i_row_mask, w * b_vals, 0.)
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < k, rng, out

            _, _, _, out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng, out)
            )
            atomic_add(post_ref, (safe_rows, col_j), out, mask=i_row_mask)

    else:
        # Match jitn corder=False: vectorize over k-dim columns with identical
        # RNG seeding. Grid: (k_blocks, B_cols). Each block processes one B column.
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
            col_j = pl.program_id(1)  # scalar B column index

            i_ks = i_k_block * k_block + jnp.arange(k_block)
            i_k_mask = i_ks < k_dim
            safe_ks = jnp.where(i_k_mask, i_ks, 0)

            # Preload B values for this column (1D vector gather)
            b_vals = B_ref[safe_ks, col_j]  # [k_block]

            # VECTOR RNG seeded identically to jitn corder=False
            rng = PallasLFSR88RNG(seed0 + i_ks)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < m

            def body(data):
                i_rows, i_row_mask, rng = data
                w = rng.normal(w_loc0, w_scale0)  # [k_block]
                vals = jnp.where(i_k_mask & i_row_mask, w * b_vals, 0.)
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
            out_shape=kwargs['outs']
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_loc, w_scale, clen, B, seed, placeholder)

    return run


def _jitnmm_jvp_wloc(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend=None, **kwargs):
    return jitnmm_p_call(w_dot, w_scale, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=backend)


def _jitnmm_jvp_wscale(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend=None, **kwargs):
    return jitnmm_p_call(w_loc, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=backend)


def _jitnmm_jvp_B(B_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend=None, **kwargs):
    return jitnmm_p_call(w_loc, w_scale, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=backend)


def _jitnmm_transpose_rules(ct, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend=None, **kwargs):
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
            backend=backend,
        )[0]
        return w_loc, w_scale, clen, r, seed
    elif ad.is_undefined_primal(w_loc):
        # M = (w_loc + w_scale * Z) * mask
        # d(M @ B)/d(w_loc) = mask @ B
        # d(loss)/d(w_loc) = sum((mask^T @ ct) * B)
        r = jitnmm_p_call(
            1., 0., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=backend,
        )[0]
        dw_loc = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return dw_loc, w_scale, clen, B, seed
    elif ad.is_undefined_primal(w_scale):
        # M = (w_loc + w_scale * Z) * mask
        # d(M @ B)/d(w_scale) = (Z * mask) @ B
        # d(loss)/d(w_scale) = sum(((Z*mask)^T @ ct) * B)
        r = jitnmm_p_call(
            0., 1., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=backend,
        )[0]
        dw_scale = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return w_loc, dw_scale, clen, B, seed
    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_normal not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[3].shape
    B = args[3].reshape(m, maybe_batch1 * maybe_batch2)
    r = jitnmm_p_call(
        args[0],
        args[1],
        args[2],
        B,
        args[4],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
        backend=kwargs.get('backend'),
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitnmm_batching(args, axes, **kwargs):
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
        return general_batching_rule(jitnmm_p, args, axes, **kwargs)


def _jitnmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_loc = jnp.ones(1, dtype=dtype)
            w_scale = jnp.ones(1, dtype=dtype) * 0.1
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (w_loc, w_scale, clen, B, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jitnmm_p_call(w_loc, w_scale, clen, B, seed, *, shape: MatrixShape, transpose: bool, corder: bool,
                  backend: Optional[str] = None):
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

    return jitnmm_p(
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
        TITLE_SIZE=B.shape[1],  # Assuming B is [k, n], we want to process n columns at once
        backend=backend,
    )


jitnmm_p = XLACustomKernel('float_jitnmm')
jitnmm_p.def_numba_kernel(_jitnmm_numba_kernel_generator)
jitnmm_p.def_warp_kernel(_jitnmm_warp_kernel_generator)
jitnmm_p.def_pallas_kernel('gpu', _jitnmm_pallas_kernel_generator)
jitnmm_p.def_pallas_kernel('tpu', _jitnmm_pallas_kernel_generator)
jitnmm_p.def_jvp_rule2(_jitnmm_jvp_wloc, _jitnmm_jvp_wscale, None, _jitnmm_jvp_B, None)
jitnmm_p.def_transpose_rule(_jitnmm_transpose_rules)
jitnmm_p.def_batching_rule(_jitnmm_batching)
jitnmm_p.def_tags('jit_normal', 'float')
jitnmm_p.def_benchmark_data(_jitnmm_benchmark_data)
