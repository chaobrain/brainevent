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
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._pallas_random import PallasLFSR88RNG
from brainevent._typing import Data, MatrixShape

__all__ = [
    "jits",
    "jits_p",
    "jitsmv",
    "jitsmv_p",
    "jitsmm",
    "jitsmm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def jits(
    weight: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
) -> Data:
    r"""Generate a homogeneous sparse random matrix on-the-fly.

    This function creates a sparse random matrix where all non-zero values are set
    to the same homogeneous weight. Instead of storing the full matrix in memory,
    this function efficiently represents it in a form that can be used with JAX
    transformations including jit(), vmap(), grad() and pmap().

    Parameters
    ----------
    weight : Data
        The value to use for all non-zero entries in the matrix. Can be a scalar,
        an Array, ndarray, or a Quantity with units.
    prob : float
        Connection probability for the matrix (between 0 and 1). Determines the
        sparsity of the generated matrix.
    seed : int
        Random seed for reproducible matrix generation.
    shape : MatrixShape
        The shape of the matrix as a tuple (num_rows, num_cols).
    transpose : bool, default=False
        If True, return the transposed random matrix.
    corder : bool, default=True
        Controls whether the parallelization order is oriented along the matrix columns:
        - True: Sampling index along collum dimension
        - False: Sampling index along row dimension

    Returns
    -------
    Data
        The generated sparse random matrix with the specified shape. If `transpose`
        is True, the matrix is transposed, and the output shape is ``shape``.
        Otherwise, the output shape is ``(shape[1], shape[0])``.

    Notes
    -----
    The matrix is generated using a probabilistic sampling approach rather than
    explicitly storing all values. This allows efficient operations with very large
    sparse matrices that would otherwise be impractical to store in memory.

    When using corder=True (default), the matrix generated with transpose=True
    will generally be different from the transpose of the matrix generated with transpose=False.
    Set corder=False if exact correspondence between these two cases is required.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>>
    >>> # Generate a 1000x500 sparse matrix with 10% connection probability
    >>> rng_seed = 42
    >>> weight = 0.01  # All connections have this value
    >>> matrix = jits(weight, prob=0.1, seed=rng_seed,
    ...                           shape=(1000, 500))
    >>>
    >>> # With units
    >>> import brainunit as u
    >>> weight_with_units = 0.01 * u.mA
    >>> matrix_with_units = jits(weight_with_units, prob=0.1,
    ...                                      seed=rng_seed, shape=(1000, 500))
    """
    weight, unitd = u.split_mantissa_unit(weight)
    clen = _initialize_conn_length(prob)
    res = jits_p_call(
        weight,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd)


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitsmv(
    weight: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
) -> Data:
    r"""
    Perform the :math:`y=M@v` or :math:`y=M.T@v` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).

        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``corder=True``, with the sacrifice of
        the speed compared with ``corder=False``.

    Parameters
    ----------
    weight: Array, ndarray, Quantity, float
        The value of the random matrix.
    prob: float
        The connection probability.
    vector: Array, ndarray, Quantity
        The vector.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    corder : bool, default=True
        Controls whether the parallelization order is oriented along the matrix columns:
        - True: Sampling index along collum dimension
        - False: Sampling index along row dimension

    Returns
    -------
    out: Array, ndarray, Quantity
        The output of :math:`y = M @ v` if ``transpose=False``,
        or the output of :math:`y = M^T @ v` if ``transpose=True``.
    """

    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = jitsmv_p_call(
        weight,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitsmm(
    weight: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
) -> Data:
    r"""
    Perform the :math:`y=M@B` or :math:`y=M.T@B` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@B`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``corder=True``, with the sacrifice of
        the speed compared with ``corder=False``.

    Parameters
    ----------
    weight: Array, ndarray, Quantity, float
        The value of the random matrix.
    prob: float
        The connection probability.
    B: Array, ndarray, Quantity
        The matrix.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    corder : bool, default=True
        Controls whether the parallelization order is oriented along the matrix columns:
        - True: Sampling index along collum dimension
        - False: Sampling index along row dimension

    Returns
    -------
    out: Array, ndarray
        The output of :math:`y = M @ B` if ``transpose=False``,
        or the output of :math:`y = M^T @ B` if ``transpose=True``.
    """

    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = jitsmm_p_call(
        weight,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitc_homo_matrix_numba_kernel(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba

    if corder:
        if transpose:
            # JIT matrix.T - JIT matrix shape = [m, n]
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, seed, posts):
                posts[:] = 0.
                m = posts.shape[1]
                n = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_row in range(n):
                    i_col = np.random.randint(0, clen0)
                    while i_col < m:
                        posts[i_row, i_col] = weight0
                        i_col += np.random.randint(1, clen0)
        else:
            # JIT matrix - JIT matrix shape = [m, n]
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                n = posts.shape[1]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_row in range(m):
                    i_col = np.random.randint(0, clen0)
                    while i_col < n:
                        posts[i_row, i_col] = weight0
                        i_col += np.random.randint(1, clen0)
    else:
        if transpose:
            # JIT matrix.T - JIT matrix shape = [m, n]
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, seed, posts):
                posts[:] = 0.
                m = posts.shape[1]
                n = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_col in range(m):
                    i_row = np.random.randint(0, clen0)
                    while i_row < n:
                        posts[i_row, i_col] = weight0
                        i_row += np.random.randint(1, clen0)
        else:
            # JIT matrix - JIT matrix shape = [m, n]
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                n = posts.shape[1]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_col in range(n):
                    i_row = np.random.randint(0, clen0)
                    while i_row < m:
                        posts[i_row, i_col] = weight0
                        i_row += np.random.randint(1, clen0)

    def kernel(weight, clen, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, seed)

    return kernel


def _jitc_homo_matrix_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if corder:
        if transpose:
            # JIT matrix.T - JIT matrix shape = [m, n]
            @warp.kernel
            def mat(
                weight: weight_warp_info,
                clen: clen_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                m = posts.shape[1]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                state = warp.rand_init(seed0 + i_row)
                i_col = warp.randi(state, 0, clen0)
                while i_col < m:
                    posts[i_row, i_col] = weight0
                    i_col += warp.randi(state, 1, clen0)
        else:
            # JIT matrix - JIT matrix shape = [m, n]
            @warp.kernel
            def mat(
                weight: weight_warp_info,
                clen: clen_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                n = posts.shape[1]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                state = warp.rand_init(seed0 + i_row)
                i_col = warp.randi(state, 0, clen0)
                while i_col < n:
                    posts[i_row, i_col] = weight0
                    i_col += warp.randi(state, 1, clen0)

        def kernel(weight, clen, seed, _):
            dim = out_info.shape[0]
            fn = jax_kernel(mat, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weight, clen, seed)
    else:
        if transpose:
            # JIT matrix.T - JIT matrix shape = [m, n]
            @warp.kernel
            def mat(
                weight: weight_warp_info,
                clen: clen_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                n = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < n:
                    posts[i_row, i_col] = weight0
                    i_row += warp.randi(state, 1, clen0)
        else:
            # JIT matrix - JIT matrix shape = [m, n]
            @warp.kernel
            def mat(
                weight: weight_warp_info,
                clen: clen_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                m = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < m:
                    posts[i_row, i_col] = weight0
                    i_row += warp.randi(state, 1, clen0)

        def kernel(weight, clen, seed, _):
            dim = out_info.shape[1]
            fn = jax_kernel(mat, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weight, clen, seed)

    return kernel


def _jitc_homo_matrix_pallas_kernel(
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl

    dim = out_info.shape[0] if corder else out_info.shape[1]
    block_size = generate_block_dim(dim, maximum=128)

    if corder:
        def pallas_kernel_fn(weight_ref, clen_ref, seed_ref, _, post_ref):
            m = post_ref.shape[1]
            weight = weight_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim

            def body(data):
                i_cols, i_col_mask, rng = data
                mask = i_row_mask & i_col_mask
                val = jnp.full(block_size, weight)
                post_ref[i_rows, i_cols] = jnp.where(mask, val, post_ref[i_rows, i_cols])
                i_cols = i_cols + rng.random_integers(1, clen0)
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
        def pallas_kernel_fn(weight_ref, clen_ref, seed_ref, _, post_ref):
            n = post_ref.shape[0]
            weight = weight_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(data):
                i_rows, i_row_mask, rng = data
                mask = i_row_mask & i_col_mask
                val = jnp.full(block_size, weight)
                post_ref[i_rows, i_cols] = jnp.where(mask, val, post_ref[i_rows, i_cols])
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

    def kernel(weight, clen, seed, _):
        fn = pl.pallas_call(
            pallas_kernel_fn,
            grid=(pl.cdiv(dim, block_size),),
            input_output_aliases={3: 0},
            out_shape=kwargs['outs']
        )
        return fn(weight, clen, seed, _)

    return kernel


def _jitc_homo_matrix_jvp_weight(weight_dot, weight, clen, seed, _, *, shape: Sequence[int], transpose: bool,
                                 corder: bool, **kwargs):
    return jits_p_call(weight_dot, clen, seed, shape=shape, transpose=transpose, corder=corder)


def _jitc_homo_matrix_transpose(
    ct, weight, clen, seed, _, *, shape: Sequence[int], transpose: bool, corder: bool, **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(weight):
        forward = jits_p_call(
            1., clen, seed, shape=shape, transpose=transpose, corder=corder
        )[0]
        dw = jnp.expand_dims((ct * forward).sum(), axis=0)
        return (dw, clen, seed, _)

    else:
        raise NotImplementedError(
            'JITC matrix transpose is only implemented for the weight arguments.'
        )


def _jitc_homo_matrix_batching(args, axes, **kwargs):
    if tuple(axes)[1:] == (None, None, None):
        # vmap on weight data
        r = jits_p_call(
            jnp.asarray([1.], dtype=args[0].dtype),
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )[0]
        weight = args[0]
        axis = axes[0]
        r = jax.vmap(lambda w: r * w, in_axes=axis, out_axes=axis)(weight)
        return [r], [axis]
    else:
        return general_batching_rule(jits_p, args, axes, **kwargs)


def _jits_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            weight = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (weight, clen, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jits_p_call(
    weight,
    clen,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)

    out_info = (
        jax.ShapeDtypeStruct(shape[::-1], dtype=weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct(shape, dtype=weight.dtype)
    )

    return jits_p(
        weight,
        clen,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


jits_p = XLACustomKernel('float_jitc_homo_matrix')
jits_p.def_numba_kernel(_jitc_homo_matrix_numba_kernel)
jits_p.def_warp_kernel(_jitc_homo_matrix_warp_kernel)
jits_p.def_pallas_kernel('gpu', _jitc_homo_matrix_pallas_kernel)
jits_p.def_pallas_kernel('tpu', _jitc_homo_matrix_pallas_kernel)
jits_p.def_jvp_rule2(_jitc_homo_matrix_jvp_weight, None, None)
jits_p.def_transpose_rule(_jitc_homo_matrix_transpose)
jits_p.def_batching_rule(_jitc_homo_matrix_batching)
jits_p.def_call(jits_p_call)
jits_p.def_tags('jit_scalar', 'float')
jits_p.def_benchmark_data(_jits_benchmark_data)


def _jitsmv_numba_kernel(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba

    if corder:
        if transpose:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_col in range(n_col):
                    i_row = np.random.randint(0, clen0)
                    out = np.float64(0.)
                    while i_row < n_row:
                        out += vector[i_row]
                        i_row += np.random.randint(1, clen0)
                    posts[i_col] = out * weight0
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                num_row = posts.shape[0]
                num_col = vector.shape[0]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_row in range(num_row):
                    i_col = np.random.randint(0, clen0)
                    out = np.float64(0.)
                    while i_col < num_col:
                        out += vector[i_col]
                        i_col += np.random.randint(1, clen0)
                    posts[i_row] = out * weight0
    else:
        if transpose:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_row in range(num_row):
                    v = vector[i_row] * weight0
                    i_col = np.random.randint(0, clen0)
                    while i_col < num_col:
                        posts[i_col] += v
                        i_col += np.random.randint(1, clen0)
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                posts[:] = 0.
                num_row = posts.shape[0]
                num_col = vector.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_col in range(num_col):
                    v = vector[i_col] * weight0
                    i_row = np.random.randint(0, clen0)
                    while i_row < num_row:
                        posts[i_row] += v
                        i_row += np.random.randint(1, clen0)

    def kernel(weight, clen, vector, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, vector, seed)

    return kernel


def _jitsmv_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    vector_warp_info = jaxinfo_to_warpinfo(vector_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if corder:
        if transpose:
            @warp.kernel
            def mv(
                weight: weight_warp_info,
                clen: clen_warp_info,
                vector: vector_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                num_row = vector.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                r = weight.dtype(0.)
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    r += vector[i_row]
                    i_row += warp.randi(state, 1, clen0)
                posts[i_col] = r * weight0
        else:
            @warp.kernel
            def mv(
                weight: weight_warp_info,
                clen: clen_warp_info,
                vector: vector_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                num_col = vector.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                r = weight.dtype(0.)
                state = warp.rand_init(seed0 + i_row)
                i_col = warp.randi(state, 0, clen0)
                while i_col < num_col:
                    r += vector[i_col]
                    i_col += warp.randi(state, 1, clen0)
                posts[i_row] = r * weight0

        def kernel(weight, clen, vector, seed, _):
            dim = out_info.shape[0]
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weight, clen, vector, seed)
    else:
        if transpose:
            @warp.kernel
            def mv(
                weight: weight_warp_info,
                clen: clen_warp_info,
                vector: vector_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                num_col = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                v = vector[i_row] * weight0
                state = warp.rand_init(seed0 + i_row)
                i_col = warp.randi(state, 0, clen0)
                while i_col < num_col:
                    posts[i_col] += v
                    i_col += warp.randi(state, 1, clen0)
        else:
            @warp.kernel
            def mv(
                weight: weight_warp_info,
                clen: clen_warp_info,
                vector: vector_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                num_row = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                v = vector[i_col] * weight0
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    posts[i_row] += v
                    i_row += warp.randi(state, 1, clen0)

        def kernel(weight, clen, vector, seed, _):
            dim = vector_info.shape[0]
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weight, clen, vector, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _jitsmv_pallas_kernel(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    tiled = True

    if tiled:
        block_size = generate_block_dim(dim, maximum=128)

        if corder:
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_row = vector_ref.shape[0]
                weight = weight_ref[0]
                clen = clen_ref[0]
                seed = seed_ref[0]
                i_col_block = pl.program_id(0)
                i_cols = i_col_block * block_size + jnp.arange(block_size)
                i_col_mask = i_cols < dim

                def body(data):
                    i_rows, i_row_mask, rng, out = data
                    v = vector_ref[i_rows]
                    v = jnp.where(i_row_mask, v, 0.0)
                    out += v * weight
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
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_col = post_ref.shape[0]
                weight = weight_ref[0]
                clen = clen_ref[0]
                seed = seed_ref[0]
                i_row_block = pl.program_id(0)
                i_rows = i_row_block * block_size + jnp.arange(block_size)
                i_row_mask = i_rows < dim
                v = vector_ref[i_rows]
                vector = jnp.where(i_row_mask, v, 0.0) * weight

                def body(data):
                    i_cols, i_col_mask, rng = data
                    atomic_add(post_ref, i_cols, vector, mask=i_row_mask & i_col_mask)
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

        def run(weights, clen, vector, seed, _):
            fn = pl.pallas_call(
                kernel,
                grid=(pl.cdiv(dim, block_size),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs']
            )
            return fn(weights, clen, vector, seed, _)

        return run

    else:
        if corder:
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_row = vector_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_col = pl.program_id(0)

                def body(data):
                    i, rng, res = data
                    res += vector_ref[i] * weight
                    i += rng.random_integers(1, clen0)
                    return i, rng, res

                rng = PallasLFSR88RNG(seed0 + i_col)
                _, _, r = jax.lax.while_loop(
                    lambda data: data[0] < num_row,
                    body,
                    (rng.random_integers(0, clen0), rng, 0.0)
                )
                post_ref[i_col] = r

        else:
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_col = post_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_row = pl.program_id(0)
                v = vector_ref[i_row] * weight

                def body(data):
                    i, rng = data
                    atomic_add(post_ref, (i,), v)
                    i += rng.random_integers(1, clen0)
                    return i, rng

                rng = PallasLFSR88RNG(seed0 + i_row)
                jax.lax.while_loop(
                    lambda data: data[0] < num_col,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

        def run(weights, clen, vector, seed, _):
            fn = pl.pallas_call(
                kernel,
                grid=(dim,),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs']
            )
            return fn(weights, clen, vector, seed, _)

        return run


def _jitsmv_jvp_v(v_dot, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    return jitsmv_p_call(weight, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder)


def _jitsmv_jvp_weights(w_dot, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    return jitsmv_p_call(w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder)


def _jitsmv_transpose_rules(
    ct,
    weight,
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

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        r = jitsmv_p_call(
            weight,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder
        )[0]
        return weight, clen, r, seed, _
    elif ad.is_undefined_primal(weight):
        row = jitsmv_p_call(
            jnp.ones((1,), dtype=ct.dtype),
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder
        )[0]
        dw = jnp.sum(row * vector, keepdims=True)
        return dw, clen, vector, seed, _
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitsmv_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitsmm_p_call(
            args[0],
            args[1],
            args[2].T,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitsmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )
        return r, [1]
    else:
        return general_batching_rule(jitsmv_p, args, axes, **kwargs)


def _jitsmv_benchmark_data(*, platform):
    import numpy as _np
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            weight = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(_np.random.randn(v_size), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (weight, clen, vector, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jitsmv_p_call(
    weight,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
):
    r"""
    Low-level implementation function for just-in-time generated sparse matrix-vector multiplication
    with homogeneous weight values.

    This function prepares inputs and calls the XLA custom kernel primitive for matrix-vector
    multiplication with a sparsely connected matrix that is generated on-the-fly during execution.
    It handles necessary type conversions and array formatting before passing to the underlying
    primitive operation.

    Parameters
    ----------
    weight : Array, float
        Scalar weight value for non-zero connections in the randomly generated matrix.
        Will be converted to at least 1D array internally.
    clen : Array, float
        Connection length parameter (approximately 2/connection_probability).
        Controls the sparsity of the generated matrix.
    vector : Array
        Input vector for multiplication. Shape must be compatible with the matrix shape.
    seed : int, Array
        Random seed for reproducible matrix generation.
    shape : Sequence[int]
        The shape of the implicit matrix as a tuple (num_rows, num_cols).
    transpose : bool, default=False
        If True, perform ``y = M^T @ vector`` instead of ``y = M @ vector``.
    corder : bool, default=True
        Controls the parallelization strategy:
        - True: Parallelize along output dimension (typically faster)
        - False: Parallelize along input dimension (ensures reproducibility between
                 transposed operations, but may be slower)

    Returns
    -------
    tuple
        A tuple containing the output array from the primitive operation.
        The output shape is determined by the matrix shape and transpose flag:
        - If ``transpose=False``: output shape is (shape[0],)
        - If ``transpose=True``: output shape is (shape[1],)

    Notes
    -----
    This function is intended as an internal implementation detail and is used by the
    higher-level `jitc_matvec_homo` function, which properly handles units and provides
    a more user-friendly interface.

    The operation is implemented as an XLA custom kernel to achieve high performance on
    both CPU and GPU. The primitive supports JAX transformations including grad, vmap, and jit.

    When using ``corder=True`` (default), the generated matrix $M$ when ``transpose=False``
    will generally be different from the implicitly generated $M^T$ when ``transpose=True``.
    Set ``corder=False`` if exact correspondence between $M$ and $M^T$ is required.
    """

    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert weight.shape == (1,), f"The weight shape should be (1,), but got {weight.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weight.dtype)
    )

    return jitsmv_p(
        weight,
        clen,
        vector,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


jitsmv_p = XLACustomKernel('float_jitsmv')
jitsmv_p.def_numba_kernel(_jitsmv_numba_kernel)
jitsmv_p.def_warp_kernel(_jitsmv_warp_kernel)
jitsmv_p.def_pallas_kernel('gpu', _jitsmv_pallas_kernel)
jitsmv_p.def_pallas_kernel('tpu', _jitsmv_pallas_kernel)
jitsmv_p.def_jvp_rule2(_jitsmv_jvp_weights, None, _jitsmv_jvp_v, None, None)
jitsmv_p.def_transpose_rule(_jitsmv_transpose_rules)
jitsmv_p.def_batching_rule(_jitsmv_batching)
jitsmv_p.def_call(jitsmv_p_call)
jitsmv_p.def_tags('jit_scalar', 'float')
jitsmv_p.def_benchmark_data(_jitsmv_benchmark_data)


def _jitsmm_numba_kernel(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba

    if corder:
        if transpose:
            # JIT Matrix.T @ B - JIT matrix: [k, m], B: [k, n]
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_m in range(m):
                    i_k = np.random.randint(0, clen0)
                    out = np.zeros(n, dtype=B.dtype)
                    while i_k < k:
                        out += B[i_k]
                        i_k += np.random.randint(1, clen0)
                    posts[i_m] = out * weight0
        else:
            # JIT Matrix @ B - JIT matrix: [m, k], B: [k, n]
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_m in range(m):
                    i_k = np.random.randint(0, clen0)
                    out = np.zeros(n, dtype=B.dtype)
                    while i_k < k:
                        out += B[i_k]
                        i_k += np.random.randint(1, clen0)
                    posts[i_m] = out * weight0
    else:
        if transpose:
            # JIT Matrix.T @ B - JIT matrix: [k, m], B: [k, n]
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_k in range(k):
                    out = B[i_k] * weight0
                    i_m = np.random.randint(0, clen0)
                    while i_m < m:
                        posts[i_m] += out
                        i_m += np.random.randint(1, clen0)
        else:
            # JIT Matrix @ B - JIT matrix: [m, k], B: [k, n]
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_k in range(k):
                    out = B[i_k] * weight0
                    i_m = np.random.randint(0, clen0)
                    while i_m < m:
                        posts[i_m] += out
                        i_m += np.random.randint(1, clen0)

    def kernel(weight, clen, B, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, B, seed)

    return kernel


def _jitsmm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    TITLE_SIZE: int,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    B_warp_info = jaxinfo_to_warpinfo(B_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if corder:
        if transpose:
            # JIT Matrix.T @ B
            @warp.kernel
            def mm(
                weight: weight_warp_info,
                clen: clen_warp_info,
                B: B_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                k = B.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m)
                out = warp.tile_zeros(TITLE_SIZE, dtype=weight.dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    out += warp.tile_load(B[i_k], TITLE_SIZE)
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out * weight0)
        else:
            # JIT Matrix @ B
            @warp.kernel
            def mm(
                weight: weight_warp_info,
                clen: clen_warp_info,
                B: B_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                k = B.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m)
                out = warp.tile_zeros(TITLE_SIZE, dtype=weight.dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    out += warp.tile_load(B[i_k], TITLE_SIZE)
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out * weight0)

        def kernel(weight, clen, B, seed, _):
            dim = out_info.shape[0]
            fn = jax_kernel(mm, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weight, clen, B, seed)
    else:
        if transpose:
            # JIT Matrix.T @ B
            @warp.kernel
            def mm(
                weight: weight_warp_info,
                clen: clen_warp_info,
                B: B_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                m = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k)
                out = warp.tile_load(B[i_k], TITLE_SIZE) * weight0
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    warp.tile_atomic_add(posts[i_m], out)
                    i_m += warp.randi(state, 1, clen0)
        else:
            # JIT Matrix @ B
            @warp.kernel
            def mm(
                weight: weight_warp_info,
                clen: clen_warp_info,
                B: B_warp_info,
                seed: seed_warp_info,
                posts: out_warp_info,
            ):
                m = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k)
                out = warp.tile_load(B[i_k], TITLE_SIZE) * weight0
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    warp.tile_atomic_add(posts[i_m], out)
                    i_m += warp.randi(state, 1, clen0)

        def kernel(weight, clen, B, seed, _):
            dim = B_info.shape[0]
            fn = jax_kernel(mm, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weight, clen, B, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _jitsmm_pallas_kernel(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    block_dim = generate_block_dim(B_info.shape[1], maximum=1024)

    if corder:
        if transpose:
            # JIT Matrix.T @ B
            # - JIT matrix: [k, m]
            # - B: [k, n]
            def kernel(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
                k = B_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_m = pl.program_id(0)
                i_n_block = pl.program_id(1)
                i_n_start = block_dim * i_n_block
                mask = i_n_start + jnp.arange(block_dim) < B_info.shape[1]

                def body(data):
                    i, rng, out = data
                    b_val = B_ref[i, pl.dslice(i_n_start, block_dim)]
                    b_val = jnp.where(mask, b_val, 0.0)
                    out += b_val * weight
                    i += rng.random_integers(1, clen0)
                    return i, rng, out

                rng = PallasLFSR88RNG(seed0 + i_m)
                out = jnp.zeros(block_dim, dtype=post_ref.dtype)
                _, _, out = jax.lax.while_loop(
                    lambda data: data[0] < k,
                    body,
                    (rng.random_integers(0, clen0), rng, out)
                )
                post_ref[i_m, pl.dslice(i_n_start, block_dim)] = jnp.where(
                    mask, out, post_ref[i_m, pl.dslice(i_n_start, block_dim)]
                )

        else:
            # JIT Matrix @ B
            # - JIT matrix: [m, k]
            # - B: [k, n]
            def kernel(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
                k = B_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_m = pl.program_id(0)
                i_n_block = pl.program_id(1)
                i_n_start = block_dim * i_n_block
                mask = i_n_start + jnp.arange(block_dim) < B_info.shape[1]

                def body(data):
                    i, rng, out = data
                    b_val = B_ref[i, pl.dslice(i_n_start, block_dim)]
                    b_val = jnp.where(mask, b_val, 0.0)
                    out += b_val * weight
                    i += rng.random_integers(1, clen0)
                    return i, rng, out

                rng = PallasLFSR88RNG(seed0 + i_m)
                out = jnp.zeros(block_dim, dtype=post_ref.dtype)
                _, _, out = jax.lax.while_loop(
                    lambda data: data[0] < k,
                    body,
                    (rng.random_integers(0, clen0), rng, out)
                )
                post_ref[i_m, pl.dslice(i_n_start, block_dim)] = jnp.where(
                    mask, out, post_ref[i_m, pl.dslice(i_n_start, block_dim)]
                )

    else:
        if transpose:
            # JIT Matrix.T @ B
            # - JIT matrix: [k, m]
            # - B: [k, n]
            def kernel(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
                m = post_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_k = pl.program_id(0)
                i_n_block = pl.program_id(1)
                i_n_start = block_dim * i_n_block
                mask = i_n_start + jnp.arange(block_dim) < B_info.shape[1]

                b_val = B_ref[i_k, pl.dslice(i_n_start, block_dim)]
                B_block = jnp.where(mask, b_val, 0.0)
                out = B_block * weight

                def body(data):
                    i, rng = data
                    atomic_add(post_ref, (i, pl.dslice(i_n_start, block_dim)), out, mask=mask)
                    i += rng.random_integers(1, clen0)
                    return i, rng

                rng = PallasLFSR88RNG(seed0 + i_k)
                jax.lax.while_loop(
                    lambda data: data[0] < m,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

        else:
            # JIT Matrix @ B
            # - JIT matrix: [m, k]
            # - B: [k, n]
            def kernel(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
                m = post_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_k = pl.program_id(0)
                i_n_block = pl.program_id(1)
                i_n_start = block_dim * i_n_block
                mask = i_n_start + jnp.arange(block_dim) < B_info.shape[1]

                b_val = B_ref[i_k, pl.dslice(i_n_start, block_dim)]
                B_block = jnp.where(mask, b_val, 0.0)
                out = B_block * weight

                def body(data):
                    i, rng = data
                    atomic_add(post_ref, (i, pl.dslice(i_n_start, block_dim)), out, mask=mask)
                    i += rng.random_integers(1, clen0)
                    return i, rng

                rng = PallasLFSR88RNG(seed0 + i_k)
                jax.lax.while_loop(
                    lambda data: data[0] < m,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

    tile = (out_info.shape[0] if corder else B_info.shape[0])
    grid = (tile, pl.cdiv(B_info.shape[1], block_dim))

    def run(weights, clen, B, seed, _):
        fn = pl.pallas_call(
            kernel,
            grid=grid,
            input_output_aliases={4: 0},
            out_shape=kwargs['outs']
        )
        return fn(weights, clen, B, seed, _)

    return run


def _jitsmm_jvp_w(w_dot, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    return jitsmm_p_call(w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder)


def _jitsmm_jvp_B(B_dot, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    return jitsmm_p_call(weight, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder)


def _jitsmm_transpose_rules(
    ct,
    weight,
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

    ct = ct[0]
    if ad.is_undefined_primal(B):
        r = jitsmm_p_call(
            weight,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
        )[0]

        return weight, clen, r, seed, _

    elif ad.is_undefined_primal(weight):
        r = jitsmm_p_call(
            jnp.ones((1,), dtype=ct.dtype),
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder
        )[0]
        dw = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return dw, clen, B, seed, _

    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_homo not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = jitsmm_p_call(
        args[0],
        args[1],
        B,
        args[3],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitsmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, 1, None, None):
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, 2, None, None):
        return _batching_axis1(args, axis=2, **kwargs)

    else:
        return general_batching_rule(jitsmm_p, args, axes, **kwargs)


def _jitsmm_benchmark_data(*, platform):
    import numpy as _np
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            weight = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(_np.random.randn(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (weight, clen, B, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jitsmm_p_call(
    weight,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert weight.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert weight.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weight.dtype)
    )

    return jitsmm_p(
        weight,
        clen,
        B,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        TITLE_SIZE=B.shape[1],  # Assuming B is [k, n], we want to process n columns at once
    )


jitsmm_p = XLACustomKernel('float_jitsmm')
jitsmm_p.def_numba_kernel(_jitsmm_numba_kernel)
jitsmm_p.def_warp_kernel(_jitsmm_warp_kernel)
jitsmm_p.def_pallas_kernel('gpu', _jitsmm_pallas_kernel)
jitsmm_p.def_pallas_kernel('tpu', _jitsmm_pallas_kernel)
jitsmm_p.def_jvp_rule2(_jitsmm_jvp_w, None, _jitsmm_jvp_B, None, None)
jitsmm_p.def_transpose_rule(_jitsmm_transpose_rules)
jitsmm_p.def_batching_rule(_jitsmm_batching)
jitsmm_p.def_call(jitsmm_p_call)
jitsmm_p.def_tags('jit_scalar', 'float')
jitsmm_p.def_benchmark_data(_jitsmm_benchmark_data)
