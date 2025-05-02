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

from typing import Optional, Sequence

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from ._compatible_import import pallas as pl
from ._config import numba_environ
from ._jitc_util import _initialize_seed, _initialize_conn_length
from ._pallas_random import LFSR88RNG
from ._typing import Kernel, Data, MatrixShape
from ._xla_custom_op import XLACustomKernel, GPUKernelChoice
from ._xla_custom_op_numba import NumbaKernelGenerator
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_util import general_batching_rule
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator

__all__ = [
    "float_jitc_normal_matrix",
    "float_jitc_normal_matvec",
    "float_jitc_normal_matmat",
]


def float_jitc_normal_matrix(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
) -> Data:
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    clen = _initialize_conn_length(prob)
    res = float_jitc_normal_matrix_p_call(
        w_loc,
        w_scale,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd)


def float_jitc_normal_matvec(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
) -> Data:
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    seed = _initialize_seed(seed)
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = float_jitc_mv_normal_p_call(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


def float_jitc_normal_matmat(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
) -> Data:
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    seed = _initialize_seed(seed)
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = float_jitc_mm_normal_p_call(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitc_normal_matrix_cpu_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    if corder:
        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(w_loc, w_scale, clen, seed, _, posts):
                m = posts.shape[1]
                n = posts.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length (inverse of connection probability)
                seed0 = seed[0]  # Random seed

                # Initialize the random number generator with the provided seed
                # This ensures reproducibility for the same seed value
                np.random.seed(seed0)

                # Process each output element (column in the matrix)
                for i_row in range(n):
                    # Generate first row index randomly - this determines where to start sampling
                    i_col = np.random.randint(0, clen0)

                    # Process all connected entries for this column
                    while i_col < m:
                        posts[i_row, i_col] = np.random.normal(loc=w_loc0, scale=w_scale0)

                        # Skip ahead to next connected row (sparse sampling)
                        # The random skip ensures proper connection probability
                        # Each skip distance is randomly determined to maintain the sparse pattern
                        i_col += np.random.randint(1, clen0)

        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(w_loc, w_scale, clen, seed, _, posts):
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

        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(w_loc, w_scale, clen, seed, _, posts):
                m = posts.shape[1]
                n = posts.shape[0]

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation

                # Initialize the random number generator with the provided seed
                # This ensures reproducibility for the same seed value
                np.random.seed(seed0)

                # Process each column of the matrix sequentially
                for i_col in range(m):
                    # Generate first row index randomly - this determines where to start sampling in this column
                    i_row = np.random.randint(0, clen0)

                    # Process all connected entries for this column
                    while i_row < n:
                        # Set the current matrix element to the weight value
                        posts[i_row, i_col] = np.random.normal(loc=w_loc0, scale=w_scale0)

                        # Skip ahead to next connected row (sparse sampling)
                        # The random skip ensures proper connection probability
                        i_row += np.random.randint(1, clen0)

        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(w_loc, w_scale, clen, seed, _, posts):
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

    return numba_environ.jit_fn(kernel)


def _jitc_normal_matrix_gpu_kernel_generator(
    w_loc_info: jax.ShapeDtypeStruct,
    w_scale_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    import warp

    w_loc_dtype = dtype_to_warp_type(w_loc_info.dtype)
    w_scale_dtype = dtype_to_warp_type(w_scale_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if corder:
        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_loc_dtype),
                posts: warp.array2d(dtype=w_loc_dtype),
            ):
                m = posts.shape[1]

                # Extract scalar values from input arrays for more efficient access
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one output element
                i_row = warp.tid()

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_row)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_col = warp.randi(state, 0, clen0)

                # Process all connected entries for this output element
                while i_col < m:
                    posts[i_row, i_col] = warp.randn(state) * w_scale0 + w_loc0

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_col += warp.randi(state, 1, clen0)


        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_loc_dtype),
                posts: warp.array2d(dtype=w_loc_dtype),
            ):
                n = posts.shape[1]  # Get number of columns in the output matrix

                # Extract scalar values from input arrays for more efficient access
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one output element (one row of the matrix)
                i_row = warp.tid()

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_row)

                # Sample the first connected column using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_col = warp.randi(state, 0, clen0)

                # Process all connected entries for this output element (row)
                while i_col < n:
                    # Add contribution from the current connected element
                    posts[i_row, i_col] = warp.randn(state) * w_scale0 + w_loc0

                    # Skip ahead to next connected column using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_col += warp.randi(state, 1, clen0)

    else:

        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_loc_dtype),
                posts: warp.array2d(dtype=w_loc_dtype),
            ):
                n = posts.shape[0]

                # Extract scalar values from input arrays for more efficient access
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one input element (column)
                i_col = warp.tid()

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_col)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_row = warp.randi(state, 0, clen0)

                # Process all connected output positions for this input element
                while i_row < n:
                    # Set the current matrix element to the weight value
                    # For this transpose=True and corder=False case, we're setting elements column by column
                    posts[i_row, i_col] = warp.randn(state) * w_scale0 + w_loc0

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_row += warp.randi(state, 1, clen0)


        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_loc_dtype),
                posts: warp.array2d(dtype=w_loc_dtype),
            ):
                m = posts.shape[0]

                # Extract scalar values from input arrays for more efficient access
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one input element (column)
                i_col = warp.tid()

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_col)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_row = warp.randi(state, 0, clen0)

                # Process all connected output positions for this input element
                while i_row < m:
                    # Set the current matrix element to the weight value
                    posts[i_row, i_col] = warp.randn(state) * w_scale0 + w_loc0

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_row += warp.randi(state, 1, clen0)

    return warp.kernel(kernel)


def _jitc_normal_matrix_pallas_kernel_generator(
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    if corder:
        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def _raw_kernel(
                w_loc_ref,
                w_scale_ref,
                clen_ref,
                seed_ref,
                _,
                post_ref,
            ):
                m = post_ref.shape[1]
                w_loc0 = w_loc_ref[0]
                w_scale0 = w_scale_ref[0]
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_row = pl.program_id(0)

                def body(data):
                    i, rng_ = data
                    post_ref[i_row, i] = rng_.normal(w_loc0, w_scale0)
                    i = i + rng_.random_integers(1, clen0)
                    return i, rng_

                rng = LFSR88RNG(seed0 + i_row)
                jax.lax.while_loop(
                    lambda data: data[0] < m,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def _raw_kernel(
                w_loc_ref,
                w_scale_ref,
                clen_ref,
                seed_ref,
                _,
                post_ref,
            ):
                n = post_ref.shape[1]  # Get number of columns in the output matrix
                w_loc0 = w_loc_ref[0]
                w_scale0 = w_scale_ref[0]
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_row = pl.program_id(0)

                def body(data):
                    i_col, rng_ = data
                    post_ref[i_row, i_col] = rng_.normal(w_loc0, w_scale0)
                    i_col = i_col + rng_.random_integers(1, clen0)
                    return i_col, rng_

                rng = LFSR88RNG(seed0 + i_row)
                jax.lax.while_loop(
                    lambda data: data[0] < n,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

    else:
        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def _raw_kernel(
                w_loc_ref,
                w_scale_ref,
                clen_ref,
                seed_ref,
                _,
                post_ref,
            ):
                n = post_ref.shape[0]
                w_loc0 = w_loc_ref[0]
                w_scale0 = w_scale_ref[0]
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_col = pl.program_id(0)

                def body(data):
                    i_row, rng_ = data
                    post_ref[i_row, i_col] = rng_.normal(w_loc0, w_scale0)
                    i_row = i_row + rng_.random_integers(0, clen0)
                    return i_row, rng_

                rng = LFSR88RNG(seed0 + i_col)
                jax.lax.while_loop(
                    lambda data: data[0] < n,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

                # rng = LFSR88(seed0 + i_col)
                # i_row = rng.random_integers(0, clen0)
                # while i_row < n:
                #     post_ref[i_row, i_col] = weight0
                #     i_row += rng.random_integers(0, clen0)


        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def _raw_kernel(
                w_loc_ref,
                w_scale_ref,
                clen_ref,
                seed_ref,
                _,
                post_ref,
            ):
                m = post_ref.shape[0]
                w_loc0 = w_loc_ref[0]
                w_scale0 = w_scale_ref[0]
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_col = pl.program_id(0)

                def body(data):
                    i_row, rng_ = data
                    post_ref[i_row, i_col] = rng_.normal(w_loc0, w_scale0)
                    i_row = i_row + rng_.random_integers(0, clen0)
                    return i_row, rng_

                rng = LFSR88RNG(seed0 + i_col)
                jax.lax.while_loop(
                    lambda data: data[0] < m,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

                # rng = LFSR88(seed0 + i_col)
                # i_row = rng.random_integers(0, clen0)
                # while i_row < m:
                #     post_ref[i_row, i_col] = weight0
                #     i_row += rng.random_integers(0, clen0)

    dim = out_info.shape[0] if corder else out_info.shape[1]

    def kernel(w_loc, w_scale, clen, seed, out):
        fn = pl.pallas_call(
            _raw_kernel,
            out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
            grid=(dim,),
            input_output_aliases={4: 0},
        )
        return [fn(w_loc, w_scale, clen, seed, out)]

    return kernel


def _jitc_normal_matrix_batching(
    args,
    axes,
    **kwargs
):
    return general_batching_rule(
        float_jitc_normal_matrix_p,
        args,
        axes,
        **kwargs,
    )


def float_jitc_normal_matrix_p_call(
    w_loc,
    w_scale,
    clen,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
):
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)

    out_info = (
        jax.ShapeDtypeStruct(shape[::-1], dtype=w_loc.dtype)
        if transpose else
        jax.ShapeDtypeStruct(shape, dtype=w_loc.dtype)
    )

    return float_jitc_normal_matrix_p(
        w_loc,
        w_scale,
        clen,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


float_jitc_normal_matrix_p = XLACustomKernel('float_jitc_normal_matrix')
float_jitc_normal_matrix_p.def_cpu_kernel(
    NumbaKernelGenerator(
        _jitc_normal_matrix_cpu_kernel_generator,
        input_output_aliases={4: 0}
    )
)
float_jitc_normal_matrix_p.def_gpu_kernel(
    GPUKernelChoice(
        default='warp',
        warp_kernel=WarpKernelGenerator(
            _jitc_normal_matrix_gpu_kernel_generator,
            dim=lambda out_info, corder, **kwargs: out_info.shape[0] if corder else out_info.shape[1],
            input_output_aliases={4: 0}
        ),
        pallas_kernel=PallasKernelGenerator(
            _jitc_normal_matrix_pallas_kernel_generator,
            block_dim=1,
            input_output_aliases={4: 0}
        ),
    )
)
float_jitc_normal_matrix_p.def_tpu_kernel(
    PallasKernelGenerator(
        _jitc_normal_matrix_pallas_kernel_generator,
        block_dim=1,
        input_output_aliases={4: 0}
    ),
)
float_jitc_normal_matrix_p.def_batching_rule(_jitc_normal_matrix_batching)


# Kernel generators for JIT connection SPMV

def _jitc_mv_normal_cpu_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_normal` operation.
    """

    if corder:
        # This means that the for loop is parallelized along the dimension of the output vector: ``post.shape[0]``.

        if transpose:
            @numba_environ.jit_fn
            def kernel(w_loc, w_scale, clen, vector, seed, _, posts):
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
                    out = np.asarray(0., dtype=vector.dtype)

                    # Process all connected entries for this column
                    while i_row < n_row:
                        out += vector[i_row] * np.random.normal(loc=w_loc0, scale=w_scale0)

                        # Skip ahead to next connected row (sparse sampling)
                        # The random skip ensures proper connection probability
                        # Each skip distance is randomly determined to maintain the sparse pattern
                        i_row += np.random.randint(1, clen0)

                    posts[i_col] = out

        else:
            @numba_environ.jit_fn
            def kernel(w_loc, w_scale, clen, vector, seed, _, posts):
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

        if transpose:
            @numba_environ.jit_fn
            def kernel(w_loc, w_scale, clen, vector, seed, _, posts):
                # Output vector dimension = number of columns in the matrix
                # This is the dimension of the result vector in the vector @ matrix operation
                num_col = posts.shape[0]

                # Input vector dimension = number of rows in the matrix
                # The vector elements are processed one by one, with each contributing to multiple output elements
                num_row = vector.shape[0]

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation

                # Initialize the random number generator with the provided seed
                # This ensures the "random" matrix is reproducible for the same seed value
                np.random.seed(seed0)

                # Process each input row (vector element) and distribute its value to connected columns
                # This implements the vector @ matrix operation one row at a time
                for i_row in range(num_row):
                    v = vector[i_row]

                    # Sample the first connected column using random skipping
                    # This implements the sparse sampling - each row connects to ~num_col/clen0 columns on average
                    # Starting from a random position in [0,clen0) creates variability in connection patterns
                    i_col = np.random.randint(0, clen0)

                    # Continue sampling and accumulating while we haven't exceeded output dimension
                    # This loop processes all columns this particular row connects to
                    while i_col < num_col:
                        # Add this connection's contribution to the appropriate output element
                        # The output is accumulated as we process each input element's contributions
                        posts[i_col] += v * np.random.normal(loc=w_loc0, scale=w_scale0)

                        # Move to the next connected column using geometric-like skipping
                        # Each next connection is approximately clen0 positions away on average
                        # This creates a sparse pattern where only ~1/clen0 of all possible connections exist
                        i_col += np.random.randint(1, clen0)

        else:
            @numba_environ.jit_fn
            def kernel(w_loc, w_scale, clen, vector, seed, _, posts):
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
    return kernel


def _jitc_mv_normal_gpu_kernel_generator(
    w_loc_info: jax.ShapeDtypeStruct,
    w_scale_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    r"""
    Generate the GPU kernel for the :func:`_jitc_matvec_normal` operation.
    """
    import warp

    w_loc_dtype = dtype_to_warp_type(w_loc_info.dtype)
    w_scale_dtype = dtype_to_warp_type(w_scale_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    v_dtype = dtype_to_warp_type(vector_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if corder:

        if transpose:
            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=w_loc_dtype),
                posts: warp.array1d(dtype=w_loc_dtype),
            ):
                # Input vector dimension (number of rows in the matrix)
                num_row = vector.shape[0]

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one output element
                i_col = warp.tid()

                # Initialize accumulator for dot product calculation
                r = float(0.0)

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_col)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_row = warp.randi(state, 0, clen0)

                # Process all connected entries for this output element
                while i_row < num_row:
                    # Add contribution from the current connected element
                    r += vector[i_row] * (warp.randn(state) * w_scale0 + w_loc0)

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_row += warp.randi(state, 1, clen0)

                # Scale accumulated sum by weight and store in output array
                posts[i_col] = r

        else:
            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=w_loc_dtype),
                posts: warp.array1d(dtype=w_loc_dtype),
            ):
                # Input vector dimension (number of columns in the matrix)
                num_col = vector.shape[0]

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one output element (one row of the matrix)
                i_row = warp.tid()

                # Initialize accumulator for dot product calculation
                r = float(0.0)

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_row)

                # Sample the first connected column using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_col = warp.randi(state, 0, clen0)

                # Process all connected entries for this output element (row)
                while i_col < num_col:
                    # Add contribution from the current connected element
                    r += vector[i_col] * (warp.randn(state) * w_scale0 + w_loc0)

                    # Skip ahead to next connected column using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_col += warp.randi(state, 1, clen0)

                # Scale accumulated sum by weight and store in output array
                posts[i_row] = r
    else:

        if transpose:
            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=w_loc_dtype),
                posts: warp.array1d(dtype=w_loc_dtype),
            ):
                # Output dimension (number of columns in the matrix)
                num_col = posts.shape[0]

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one input element (row)
                i_row = warp.tid()

                # Pre-multiply the input value by weight for efficiency
                # This avoids multiplying inside the inner loop for each connection
                v = vector[i_row]

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_row)

                # Sample the first connected column using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_col = warp.randi(state, 0, clen0)

                # Process all connected output positions for this input element
                while i_col < num_col:
                    # Atomically add contribution to the appropriate output element
                    # Using atomic operation because multiple threads may update the same output element
                    posts[i_col] += v * (warp.randn(state) * w_scale0 + w_loc0)

                    # Skip ahead to next connected column using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_col += warp.randi(state, 1, clen0)


        else:

            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=w_loc_dtype),
                posts: warp.array1d(dtype=w_loc_dtype),
            ):
                # Output dimension (number of rows in the matrix)
                num_row = posts.shape[0]

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one input element (column)
                i_col = warp.tid()

                # Pre-multiply the input value by weight for efficiency
                # This avoids multiplying inside the inner loop for each connection
                v = vector[i_col]

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_col)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_row = warp.randi(state, 0, clen0)

                # Process all connected output positions for this input element
                while i_row < num_row:
                    # Atomically add contribution to the appropriate output element
                    # Using atomic operation because multiple threads may update the same output element
                    posts[i_row] += v * (warp.randn(state) * w_scale0 + w_loc0)

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_row += warp.randi(state, 1, clen0)

    return warp.kernel(kernel)


def _jitc_mv_normal_jvp_v(
    v_dot,
    w_loc,
    w_scale,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mv_normal_p_call(
        w_loc,
        w_scale,
        clen,
        v_dot,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )


def _jitc_mv_normal_jvp_wloc(
    w_dot,
    w_loc,
    w_scale,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mv_normal_p_call(
        w_dot,
        w_scale,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )


def _jitc_mv_normal_jvp_wscale(
    w_dot,
    w_loc,
    w_scale,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mv_normal_p_call(
        w_loc,
        w_dot,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )


def _jitc_mv_normal_transpose_rules(
    ct,
    w_loc,
    w_scale,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert not ad.is_undefined_primal(w_loc)
    assert not ad.is_undefined_primal(w_scale)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        r = float_jitc_mv_normal_p_call(
            w_loc,
            w_scale,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder
        )[0]
        return w_loc, w_scale, clen, r, seed, _
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitc_mv_normal_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = float_jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, None, 1, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = float_jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )
        return r, [1]
    else:
        return general_batching_rule(
            float_jitc_mv_normal_p,
            args,
            axes,
            **kwargs,
        )


def float_jitc_mv_normal_p_call(
    w_loc,
    w_scale,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
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

    return float_jitc_mv_normal_p(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
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
    )


float_jitc_mv_normal_p = XLACustomKernel(
    'float_jitc_mv_normal',
    cpu_kernel=NumbaKernelGenerator(
        _jitc_mv_normal_cpu_kernel_generator,
        input_output_aliases={5: 0}
    ),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mv_normal_gpu_kernel_generator,
        dim=lambda out_info, vector_info, corder, **kwargs: (out_info.shape[0] if corder else vector_info.shape[0]),
        input_output_aliases={5: 0}
    )
)
float_jitc_mv_normal_p.def_jvp_rule2(
    _jitc_mv_normal_jvp_wloc,
    _jitc_mv_normal_jvp_wscale,
    None,
    _jitc_mv_normal_jvp_v,
    None,
    None
)
float_jitc_mv_normal_p.def_transpose_rule(_jitc_mv_normal_transpose_rules)
float_jitc_mv_normal_p.def_batching_rule(_jitc_mv_normal_batching)


def _jitc_mm_normal_cpu_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    r"""
    Generate the CPU kernel for the :func:`_jitc_matmat_normal` operation.
    """

    if corder:

        if transpose:
            # JIT Matrix.T @ B
            #
            # - JIT matrix: [k, m]
            # - B: [k, n]

            def kernel(w_loc, w_scale, clen, B, seed, _, posts):
                m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                k = B.shape[0]  # Number of rows in B (rows in M)

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)

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

            def kernel(w_loc, w_scale, clen, B, seed, _, posts):
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
        if transpose:
            # JIT Matrix.T @ B
            #
            # - JIT matrix: [k, m]
            # - B: [k, n]

            def kernel(w_loc, w_scale, clen, B, seed, _, posts):
                m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                k = B.shape[0]  # Number of rows in B (rows in M)

                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)  # Initialize random number generator with seed

                # Process each input row sequentially
                for i_k in range(k):
                    # Pre-multiply the current row by weight for efficiency
                    out = B[i_k]

                    # Sample the first connected output row using random skipping
                    # Start at a random position in [0, clen0) for variability in connection patterns
                    i_m = np.random.randint(0, clen0)

                    # Process all connected output rows for this input row
                    while i_m < m:
                        # Add contribution to the connected output row
                        # Using += to accumulate results across all input rows
                        posts[i_m] += out * np.random.normal(w_loc0, w_scale0)

                        # Skip ahead to next connected output row using geometric-like distribution
                        # This creates sparse connectivity with ~1/clen0 connection probability
                        i_m += np.random.randint(1, clen0)

        else:
            # JIT Matrix @ B
            #
            # - JIT matrix: [m, k]
            # - B: [k, n]

            def kernel(w_loc, w_scale, clen, B, seed, _, posts):
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

    return numba_environ.jit_fn(kernel)


def _jitc_mm_normal_gpu_kernel_generator(
    w_loc_info: jax.ShapeDtypeStruct,
    w_scale_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    TITLE_SIZE: int,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    import warp

    w_loc_dtype = dtype_to_warp_type(w_loc_info.dtype)
    w_scale_dtype = dtype_to_warp_type(w_scale_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    B_dtype = dtype_to_warp_type(B_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if corder:
        if transpose:
            # JIT Matrix.T @ B
            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_loc_dtype),
                posts: warp.array2d(dtype=w_loc_dtype),
            ):
                k = B.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]

                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m)

                out = warp.tile_zeros(TITLE_SIZE, dtype=w_loc_dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    out += warp.tile_load(B[i_k], TITLE_SIZE) * w
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out)

        else:
            # JIT Matrix @ B
            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_loc_dtype),
                posts: warp.array2d(dtype=w_loc_dtype),
            ):
                k = B.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]

                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m)

                out = warp.tile_zeros(TITLE_SIZE, dtype=w_loc_dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randn(state) * w_scale0 + w_loc0
                    out += warp.tile_load(B[i_k], TITLE_SIZE) * w
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out)

    else:
        if transpose:
            # JIT Matrix.T @ B
            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_loc_dtype),
                posts: warp.array2d(dtype=w_loc_dtype),
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


        else:
            # JIT Matrix @ B
            def kernel(
                w_loc: warp.array1d(dtype=w_loc_dtype),
                w_scale: warp.array1d(dtype=w_scale_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_loc_dtype),
                posts: warp.array2d(dtype=w_loc_dtype),
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

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mm_normal_jvp_wloc(
    w_dot,
    w_loc,
    w_scale,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mm_normal_p_call(
        w_dot,
        w_scale,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


def _jitc_mm_normal_jvp_wscale(
    w_dot,
    w_loc,
    w_scale,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mm_normal_p_call(
        w_loc,
        w_dot,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


def _jitc_mm_normal_jvp_B(
    B_dot,
    w_loc,
    w_scale,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mm_normal_p_call(
        w_loc,
        w_scale,
        clen,
        B_dot,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


def _jitc_mm_normal_transpose_rules(
    ct,
    w_loc,
    w_scale,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert not ad.is_undefined_primal(w_loc)
    assert not ad.is_undefined_primal(w_scale)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        r = float_jitc_mm_normal_p_call(
            w_loc,
            w_scale,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
        )[0]

        return w_loc, w_scale, clen, r, seed, _

    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_normal not implemented for '
            'non-undefined primals.'
        )


def _batching_axis0(args, axes, **kwargs):
    assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
    batch_size, m, n = args[3].shape
    B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
    r = float_jitc_mm_normal_p_call(
        args[0],
        args[1],
        args[2],
        B,
        args[4],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
    return [r], [1]


def _jitc_mm_normal_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, None, 0, None, None):
        return _batching_axis0(args, axes, **kwargs)

    elif tuple(axes) == (None, None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[3] = jnp.transpose(args[3], (1, 0, 2))
        return _batching_axis0(args, axes, **kwargs)

    elif tuple(axes) == (None, None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[3] = jnp.transpose(args[3], (2, 0, 1))
        return _batching_axis0(args, axes, **kwargs)

    else:
        return general_batching_rule(
            float_jitc_mm_normal_p,
            args,
            axes,
            **kwargs,
        )


def float_jitc_mm_normal_p_call(
    w_loc,
    w_scale,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
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

    return float_jitc_mm_normal_p(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
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
    )


float_jitc_mm_normal_p = XLACustomKernel(
    'float_jitc_mm_normal',
    cpu_kernel=NumbaKernelGenerator(
        _jitc_mm_normal_cpu_kernel_generator,
        input_output_aliases={5: 0}
    ),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mm_normal_gpu_kernel_generator,
        tile=lambda out_info, B_info, corder, **kwargs: (out_info.shape[0] if corder else B_info.shape[0]),
        block_dim=256,
        input_output_aliases={5: 0}
    )
)
float_jitc_mm_normal_p.def_jvp_rule2(
    _jitc_mm_normal_jvp_wloc,
    _jitc_mm_normal_jvp_wscale,
    None,
    _jitc_mm_normal_jvp_B,
    None,
    None
)
float_jitc_mm_normal_p.def_transpose_rule(_jitc_mm_normal_transpose_rules)
float_jitc_mm_normal_p.def_batching_rule(_jitc_mm_normal_batching)
