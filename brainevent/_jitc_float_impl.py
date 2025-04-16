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

from typing import Optional, Tuple, Sequence

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from ._typing import Kernel, Data
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator


__all__ = [
    "_jitc_matvec_homo",
    "_jitc_matvec_uniform",
    "_jitc_matvec_normal",
    "_jitc_matmat_homo",
    "_jitc_matmat_uniform",
    "_jitc_matmat_normal",
]


def _initialize_seed(seed=None):
    """Initialize random seed if not provided."""
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    return jnp.asarray(jnp.atleast_1d(seed), dtype=jnp.int32)

@warp.func
def _binomial_n1(state: warp.uint32, p: float) -> int:
    """
    Draw samples from a binomial distribution.
    """
    return 1 if warp.randf(state) < p else 0

def _jitc_matvec_homo(
    weight: Data | float,
    conn_prob: float,
    v: Data,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@v` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.

    .. warning::

        This API may change in the future.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).

        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.

    Parameters
    ----------
    weight: Array, ndarray, Quantity, float
        The value of the random matrix.
    conn_prob: float
        The connection probability.
    v: Array, ndarray, Quantity
        The vector.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    outdim_parallel: bool
        Perform the parallel random generations along the out dimension or not.
        It can be used to set the just-in-time generated :math:M^T: is the same
        as the just-in-time generated :math:`M` when ``transpose=True``.

    Returns
    -------
    out: Array, ndarray, Quantity
        The output of :math:`y = M @ v`.
    """
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    v, unitv = u.split_mantissa_unit(v)
    res = jitc_mv_homo_p_call(
        weight, conn_prob, v, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


def _jitc_matvec_uniform(
    w_low: Data | float,
    w_high: Data | float,
    conn_prob: float,
    v: Data,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@v` operation,
    where :math:`M` is just-in-time randomly generated with a uniform distribution for its value.

    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.

    .. warning::

        This API may change in the future.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is sampled from a uniform
    distrubtion within the range of `w_low` and `w_high`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).

        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.

    Parameters
    ----------
    w_low: Array, ndarray, Quantity, float
        The lower boundary of the random matrix.
    w_high: Array, ndarray, Quantity, float
        The upper boundary of the random matrix.
    conn_prob: float
        The connection probability.
    v: Array, ndarray, Quantity
        The vector.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    outdim_parallel: bool
        Perform the parallel random generations along the out dimension or not.
        It can be used to set the just-in-time generated :math:M^T: is the same
        as the just-in-time generated :math:`M` when ``transpose=True``.

    Returns
    -------
    out: Array, ndarray
        The output of :math:`y = M @ v`.
    """
    seed = _initialize_seed(seed)
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    w_low, unit_w_low = u.split_mantissa_unit(w_low)
    w_high, unit_w_high = u.split_mantissa_unit(
        w_high.in_unit(unit_w_low) if isinstance(w_high, u.Quantity) else w_high)
    v, unitv = u.split_mantissa_unit(v)
    res = jitc_mv_uniform_p_call(
        w_low, w_high, conn_prob, v, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unit_w_low * unitv)


def _jitc_matvec_normal(
    w_mu: Data | float,
    w_sigma: Data | float,
    conn_prob: float,
    v: Data,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    ...
    r"""Perform the :math:`y=M@v` operation,
    where :math:`M` is just-in-time randomly generated with a normal distribution for its value.

    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.

    .. warning::

        This API may change in the future.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is sampled from a normal distribution
    with mean `w_mu` and standard deviation `w_sigma`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).

        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.

    Parameters
    ----------
    w_mu: Array, ndarray, Quantity, float
        Mean (centre) of the distribution.
    w_sigma: Array, ndarray, Quantity, float
        Standard deviation (spread or "width") of the distribution. Must be non-negative.
    conn_prob: float
        The connection probability.
    v: Array, ndarray, Quantity
        The vector.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    outdim_parallel: bool
        Perform the parallel random generations along the out dimension or not.
        It can be used to set the just-in-time generated :math:M^T: is the same
        as the just-in-time generated :math:`M` when ``transpose=True``.

    Returns
    -------
    out: Array, ndarray, Quantity
        The output of :math:`y = M @ v`.
    """
    seed = _initialize_seed(seed)
    u.fail_for_dimension_mismatch(w_mu, w_sigma, "w_low and w_high must have the same dimension.")
    w_mu, unit_w_mu = u.split_mantissa_unit(w_mu)
    w_sigma, unit_w_sigma = u.split_mantissa_unit(
        w_sigma.in_unit(unit_w_mu) if isinstance(w_mu, u.Quantity) else w_sigma)
    v, unitv = u.split_mantissa_unit(v)
    res = jitc_mv_normal_p_call(
        w_mu, w_sigma, conn_prob, v, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unit_w_mu * unitv)


def _jitc_matmat_homo(
    weight: Data | float,
    conn_prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@B` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.
    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.
    .. warning::
        This API may change in the future.
    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@B`.
    .. note::
        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.
    Parameters
    ----------
    weight: Array, ndarray, Quantity, float
        The value of the random matrix.
    conn_prob: float
        The connection probability.
    B: Array, ndarray, Quantity
        The matrix.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    outdim_parallel: bool
        Perform the parallel random generations along the out dimension or not.
        It can be used to set the just-in-time generated :math:M^T: is the same
        as the just-in-time generated :math:`M` when ``transpose=True``.
    Returns
    -------
    out: Array, ndarray
        The output of :math:`y = M @ B`.
    """
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    B, unitB = u.split_mantissa_unit(B)
    res = jitc_mm_homo_p_call(
        weight, conn_prob, B, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitc_matmat_uniform(
    w_low: Data | float,
    w_high: Data | float,
    conn_prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@B` operation,
    where :math:`M` is just-in-time randomly generated with a uniform distribution for its value.
    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.
    .. warning::
        This API may change in the future.
    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is sampled from a uniform
    distrubtion within the range of `w_low` and `w_high`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@m`.
    .. note::
        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).            
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time   
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.
    Parameters
    ----------
    w_low: Array, ndarray, Quantity, float
        The lower boundary of the random matrix.
    w_high: Array, ndarray, Quantity, float
        The upper boundary of the random matrix.
    conn_prob: float
        The connection probability.
    B: Array, ndarray, Quantity
        The matrix.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    outdim_parallel: bool
        Perform the parallel random generations along the out dimension or not.
        It can be used to set the just-in-time generated :math:M^T: is the same
        as the just-in-time generated :math:`M` when ``transpose=True``.
    Returns
    -------
    out: Array, ndarray
        The output of :math:`y = M @ B`.
    """
    seed = _initialize_seed(seed)
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    w_low, unit_w_low = u.split_mantissa_unit(w_low)
    w_high, unit_w_high = u.split_mantissa_unit(
        w_high.in_unit(unit_w_low) if isinstance(w_high, u.Quantity) else w_high)
    B, unitB = u.split_mantissa_unit(B)
    res = jitc_mm_uniform_p_call(
        w_low, w_high, conn_prob, B, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unit_w_low * unitB)


def _jitc_matmat_normal(
    w_mu: Data | float,
    w_sigma: Data | float,
    conn_prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@m` operation,
    where :math:`M` is just-in-time randomly generated with a normal distribution for its value.
    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.
    .. warning::
        This API may change in the future.
    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is sampled from a normal distribution
    with mean `w_mu` and standard deviation `w_sigma`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@m`.
    .. note::
        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.
    Parameters
    ----------
    w_mu: float
        Mean (centre) of the distribution.
    w_sigma: float
        Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    conn_prob: float
        The connection probability.
    m: Array, ndarray
        The matrix.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    outdim_parallel: bool
        Perform the parallel random generations along the out dimension or not.
        It can be used to set the just-in-time generated :math:M^T: is the same
        as the just-in-time generated :math:`M` when ``transpose=True``.
    Returns
    -------
    out: Array, ndarray
        The output of :math:`y = M @ m`.
    """
    seed = _initialize_seed(seed)
    u.fail_for_dimension_mismatch(w_mu, w_sigma, "w_low and w_high must have the same dimension.")
    w_mu, unit_w_mu = u.split_mantissa_unit(w_mu)
    w_sigma, unit_w_sigma = u.split_mantissa_unit(
        w_sigma.in_unit(unit_w_mu) if isinstance(w_mu, u.Quantity) else w_sigma)
    B, unitB = u.split_mantissa_unit(B)
    res = jitc_mm_normal_p_call(
        w_mu, w_sigma, conn_prob, B, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unit_w_mu * unitB)


# Kernel generators for JIT connection SPMV

# jitc csrmv homo

def _jitc_mv_homo_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_homo` operation.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(weight, conn_prob, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_row):
                connections = np.random.binomial(1, conn_prob0, num_col)
                r = np.sum(v * connections)

                posts[i_row] = r * weight0
    else:
        # outdim_parallel=False
        def kernel(weight, conn_prob, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_col):
                r = v[i_col] * weight0
                connections = np.random.binomial(1, conn_prob0, num_row)

                posts += connections * r

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mv_homo_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    v_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matvec_homo` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    v_dtype = dtype_to_warp_type(v_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            weight: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            i_row = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + tid)
            i_col = int(0)

            while i_col < num_col:
                if _binomial_n1(state, conn_prob0) == 1:
                    r += v[i_col]
                i_col += 1

            posts[i_row] = r * weight0
    else:
        def kernel(
            weight: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            i_col = warp.tid()

            col_v = v[i_col] * weight0
            state = warp.rand_init(seed0 + tid)
            i_row = int(0)

            for i_row in range(num_row):
                if _binomial_n1(state, conn_prob0) == 1:
                    posts[i_row] += col_v
                i_row += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mv_homo_jvp_v(
    v_dot,
    weight,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return [
        _jitc_matvec_homo(
            weight,
            conn_prob,
            v_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def _jitc_mv_homo_jvp_weights(
    w_dot,
    weight,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mv_homo_p_call(
        w_dot,
        conn_prob,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_homo_transpose_rules(
    ct,
    weight,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    assert not ad.is_undefined_primal(weight)
    assert not ad.is_undefined_primal(conn_prob)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(v)

    r = jitc_mv_homo_p_call(
        weight,
        conn_prob,
        ct,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return weight, conn_prob, r, seed, _


def _jitc_mv_homo_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            args[2].T,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven COO matrix-vector product.")


def jitc_mv_homo_p_call(
    weight,
    conn_prob,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    weight = jnp.atleast_1d(weight)
    conn_prob = jnp.atleast_1d(conn_prob)

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weight.dtype)
    )

    return jitc_mv_homo_p(
        weight,
        conn_prob,
        v,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        conn_prob_info=jax.ShapeDtypeStruct(conn_prob.shape, conn_prob.dtype),
        v_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mv_homo_p = XLACustomKernel(
    'jitc_mv_homo',
    cpu_kernel=NumbaKernelGenerator(_jitc_mv_homo_cpu_kernel_generator, input_output_aliases={4: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mv_homo_gpu_kernel_generator,
        dim=lambda v_info, out_info, outdim_parallel, **kwargs: (
            out_info.shape[0] if outdim_parallel else
            v_info.shape[0]
        ),
        input_output_aliases={4: 0}
    )
)

jitc_mv_homo_p.defjvp(_jitc_mv_homo_jvp_weights, None, _jitc_mv_homo_jvp_v)
jitc_mv_homo_p.def_transpose_rule(_jitc_mv_homo_transpose_rules)
jitc_mv_homo_p.def_batching_rule(_jitc_mv_homo_batching)


# jitc csrmv uniform

def _jitc_mv_uniform_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_uniform` operation.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_low, w_high, conn_prob, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_row):
                connections = np.random.binomial(1, conn_prob0, num_col)
                random_weights = np.random.uniform(w_low0, w_high0, num_col)

                posts[i_row] = np.sum(v * random_weights * connections)
    else:
        # outdim_parallel=False
        def kernel(w_low, w_high, conn_prob, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_col):
                connections = np.random.binomial(1, conn_prob0, num_row)
                random_weights = np.random.uniform(w_low0, w_high0, num_row)

                posts += connections * random_weights * v[i_col]

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mv_uniform_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    v_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matvec_uniform` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    v_dtype = dtype_to_warp_type(v_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            w_low: warp.array1d(dtype=weight_dtype),
            w_high: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            i_row = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + tid)
            i_col = int(0)

            while i_col < num_col:
                if _binomial_n1(state, conn_prob0) == 1:
                    raw_v = warp.randf(state, w_low0, w_high0)
                    r += v[i_col] * raw_v
                i_col += 1

            posts[i_row] = r * weight0
    else:
        def kernel(
            w_low: warp.array1d(dtype=weight_dtype),
            w_high: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            i_col = warp.tid()

            col_v = v[i_col] * weight0
            state = warp.rand_init(seed0 + tid)
            i_row = int(0)

            for i_row in range(num_row):
                if _binomial_n1(state, conn_prob0) == 1:
                    raw_v = warp.randf(state, w_low0, w_high0)
                    posts[i_row] += col_v * raw_v
                i_row += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mv_uniform_jvp_v(
    v_dot,
    w_low,
    w_high,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return [
        _jitc_matvec_uniform(
            w_low,
            w_high,
            conn_prob,
            v_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def _jitc_mv_uniform_jvp_w_low(
    w_low_dot,
    w_low,
    w_high,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mv_uniform_p_call(
        w_low_dot,
        w_high,
        conn_prob,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_uniform_jvp_w_high(
    w_high_dot,
    w_low,
    w_high,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mv_uniform_p_call(
        w_high_dot,
        w_high,
        conn_prob,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_uniform_transpose_rules(
    ct,
    w_low,
    w_high,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    assert not ad.is_undefined_primal(w_low)
    assert not ad.is_undefined_primal(w_high)
    assert not ad.is_undefined_primal(conn_prob)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(v)

    r = jitc_mv_uniform_p_call(
        w_low,
        w_high,
        conn_prob,
        ct[0],
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_low, w_high, conn_prob, r, seed, _


def _jitc_mv_uniform_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven COO matrix-vector product.")


def jitc_mv_uniform_p_call(
    w_low,
    w_high,
    conn_prob,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    conn_prob = jnp.atleast_1d(conn_prob)

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_low.dtype)
    )

    return jitc_mv_uniform_p(
        w_low,
        w_high,
        conn_prob,
        v,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        conn_prob_info=jax.ShapeDtypeStruct(conn_prob.shape, conn_prob.dtype),
        v_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mv_uniform_p = XLACustomKernel(
    'jitc_mv_uniform',
    cpu_kernel=NumbaKernelGenerator(_jitc_mv_uniform_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mv_uniform_gpu_kernel_generator,
        dim=lambda v_info, out_info, outdim_parallel, **kwargs: (
            out_info.shape[0] if outdim_parallel else
            v_info.shape[0]
        ),
        input_output_aliases={5: 0}
    )
)

jitc_mv_uniform_p.defjvp(_jitc_mv_uniform_jvp_w_low,
                            _jitc_mv_uniform_jvp_w_high,
                            None,
                            _jitc_mv_uniform_jvp_v)
jitc_mv_uniform_p.def_transpose_rule(_jitc_mv_uniform_transpose_rules)
jitc_mv_uniform_p.def_batching_rule(_jitc_mv_uniform_batching)


# jitc csrmv normal

def _jitc_mv_normal_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_normal` operation.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_mu, w_sigma, conn_prob, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_row):
                connections = np.random.binomial(1, conn_prob0, num_col)
                random_weights = np.random.normal(w_mu0, w_sigma0, num_col)

                posts[i_row] = np.sum(v * random_weights * connections)
    else:
        # outdim_parallel=False
        def kernel(w_mu, w_sigma, conn_prob, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_col):
                connections = np.random.binomial(1, conn_prob0, num_row)
                random_weights = np.random.normal(w_mu0, w_sigma0, num_row)

                posts += connections * random_weights * v[i_col]

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mv_normal_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    v_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matvec_normal` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    v_dtype = dtype_to_warp_type(v_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            w_mu: warp.array1d(dtype=weight_dtype),
            w_sigma: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            i_row = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + tid)
            i_col = int(0)

            while i_col < num_col:
                if _binomial_n1(state, conn_prob0) == 1:
                    raw_v = w_mu0 + w_sigma0 * warp.randn(state)
                    r += v[i_col] * raw_v
                i_col += 1

            posts[i_row] = r * weight0

    else:
        def kernel(
            w_mu: warp.array1d(dtype=weight_dtype),
            w_sigma: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            i_col = warp.tid()

            col_v = v[i_col] * weight0
            state = warp.rand_init(seed0 + tid)
            i_row = int(0)

            for i_row in range(num_row):
                if _binomial_n1(state, conn_prob0) == 1:
                    raw_v = w_mu0 + w_sigma0 * warp.randn(state)
                    posts[i_row] += col_v * raw_v
                i_row += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mv_normal_jvp_v(
    v_dot,
    w_mu,
    w_sigma,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return [
        _jitc_matvec_normal(
            w_mu,
            w_sigma,
            conn_prob,
            v_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def _jitc_mv_normal_jvp_w_mu(
    w_mu_dot,
    w_mu,
    w_sigma,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mv_uniform_p_call(
        w_mu_dot,
        w_sigma,
        conn_prob,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_normal_jvp_w_sigma(
    w_sigma_dot,
    w_mu,
    w_sigma,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mv_uniform_p_call(
        w_mu,
        w_sigma_dot,
        conn_prob,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_normal_transpose_rules(
    ct,
    w_mu,
    w_sigma,
    conn_prob,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    assert not ad.is_undefined_primal(w_mu)
    assert not ad.is_undefined_primal(w_sigma)
    assert not ad.is_undefined_primal(conn_prob)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(v)

    r = jitc_mv_uniform_p_call(
        w_mu,
        w_sigma,
        conn_prob,
        ct[0],
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_mu, w_sigma, conn_prob, r, seed, _


def _jitc_mv_normal_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven COO matrix-vector product.")


def jitc_mv_normal_p_call(
    w_mu,
    w_sigma,
    conn_prob,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_mu = jnp.atleast_1d(w_mu)
    w_sigma = jnp.atleast_1d(w_sigma)
    conn_prob = jnp.atleast_1d(conn_prob)

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_mu.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_mu.dtype)
    )

    return jitc_mv_uniform_p(
        w_mu,
        w_sigma,
        conn_prob,
        v,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_mu.shape, w_mu.dtype),
        conn_prob_info=jax.ShapeDtypeStruct(conn_prob.shape, conn_prob.dtype),
        v_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mv_normal_p = XLACustomKernel(
    'jitc_mv_normal',
    cpu_kernel=NumbaKernelGenerator(_jitc_mv_normal_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mv_normal_gpu_kernel_generator,
        dim=lambda v_info, out_info, outdim_parallel, **kwargs: (
            out_info.shape[0] if outdim_parallel else
            v_info.shape[0]
        ),
        input_output_aliases={5: 0}
    )
)

jitc_mv_normal_p.defjvp(_jitc_mv_normal_jvp_w_mu,
                           _jitc_mv_normal_jvp_w_sigma,
                           None,
                           _jitc_mv_normal_jvp_v,
                           None)
jitc_mv_normal_p.def_transpose_rule(_jitc_mv_normal_transpose_rules)
jitc_mv_normal_p.def_batching_rule(_jitc_mv_normal_batching)


# Kernel generators for JIT connection SPMM

# jitc csrmm homo

def _jitc_mm_homo_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matmat_homo` operation.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(weight, conn_prob, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            weight0 = weight[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_rows):
                for i_col in range(num_cols):
                    connections = np.random.binomial(1, conn_prob0, num_cols)

                    r = np.sum(B[i_row, :] * connections)
                    posts[i_row, i_col] = r * weight0

    else:
        # outdim_parallel=False
        # TODO: more checks on this kernel (random generation method)
        def kernel(weight, conn_prob, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            weight0 = weight[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_cols):
                for i_row in range(num_rows):
                    r = B[i_row, :] * weight0
                    connections = np.random.binomial(1, conn_prob0, num_cols)

                    posts += connections[:, np.newaxis] * r

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mm_homo_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matmat_homo` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    B_dtype = dtype_to_warp_type(B_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            weight: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            # num_rows, num_cols = posts.shape
            weight0 = weight[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)

            while cursor < num_cols:
                if _binomial_n1(state, conn_prob0) == 1:
                    r += B[i_row, cursor]
                cursor += 1
            posts[i_row, i_col] = r * weight0
    else:
        # outdim_parallel=False
        def kernel(
            weight: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            weight0 = weight[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)

            while cursor < num_cols:
                # posts[cursor, :] += r
                if _binomial_n1(state, conn_prob0) == 1:
                    for j in range(posts.shape[1]):
                        posts[cursor, j] += B[i_row, j] * weight0
                cursor += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mm_homo_jvp_w(
    w_dot,
    weight,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mm_homo_p_call(
        w_dot,
        conn_prob,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


def _jitc_mm_homo_jvp_B(
    B_dot,
    weight,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return [
        _jitc_matmat_homo(
            weight,
            conn_prob,
            B_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel,
        )
    ]


def _jitc_mm_homo_transpose_rules(
    ct,
    weight,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    assert not ad.is_undefined_primal(weight)
    assert not ad.is_undefined_primal(conn_prob)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(B)

    r = jitc_mv_homo_p_call(
        weight,
        conn_prob,
        ct,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )[0]

    return weight, conn_prob, r, seed, _


def _jitc_mm_homo_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = jnp.transpose(args[2], (1, 0, 2)).reshape(m, batch_size * n)
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            B,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = args[2].reshape(m, batch_size * n)
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            B,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = args[2].reshape(m, batch_size * n)
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            B,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape, batch_size, n])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for JIT connection CSR matrix-matrix product")


def jitc_mm_homo_p_call(
    weight,
    conn_prob,
    B,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    weight = jnp.atleast_1d(weight)
    conn_prob = jnp.atleast_1d(conn_prob)

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weight.dtype)
    )

    return jitc_mm_homo_p(
        weight,
        conn_prob,
        B,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(conn_prob.shape, conn_prob.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mm_homo_p = XLACustomKernel(
    'jitc_mm_homo',
    cpu_kernel=NumbaKernelGenerator(_jitc_mm_homo_cpu_kernel_generator, input_output_aliases={4: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mm_homo_gpu_kernel_generator,
        dim=lambda out_info, **kwargs: (
            out_info.shape[0], out_info.shape[1]
        ),
        input_output_aliases={4: 0}
    )
)

jitc_mm_homo_p.defjvp(_jitc_mm_homo_jvp_w, None, None, _jitc_mm_homo_jvp_B)
jitc_mm_homo_p.def_transpose_rule(_jitc_mm_homo_transpose_rules)
jitc_mm_homo_p.def_batching_rule(_jitc_mm_homo_batching)


# jitc csrmm uniform

def _jitc_mm_uniform_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matmat_uniform` operation.
    """
    import numba

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_low, w_high, conn_prob, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_rows):
                for i_col in range(num_cols):
                    connections = np.random.binomial(1, conn_prob0, num_cols)
                    random_weights = np.random.uniform(w_low0, w_high0, num_cols)

                    posts[i_row, i_col] = np.sum(B[i_row, :] * random_weights * connections)

    else:
        # outdim_parallel=False
        # TODO: more checks on this kernel (random generation method)
        def kernel(w_low, w_high, conn_prob, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_cols):
                for i_row in range(num_rows):
                    r = B[i_row, :]

                    connections = np.random.binomial(1, conn_prob0, num_rows)
                    random_weights = np.random.uniform(w_low0, w_high0, num_rows)
                    effective_weights = connections * random_weights

                    posts += effective_weights[:, np.newaxis] * r

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mm_uniform_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matmat_uniform` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    B_dtype = dtype_to_warp_type(B_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            w_low: warp.array1d(dtype=weight_dtype),
            w_high: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)

            while cursor < num_cols:
                if _binomial_n1(state, conn_prob0) == 1:
                    raw_v = warp.randf(state, w_low0, w_high0)
                    r += B[i_row, cursor] * raw_v
                cursor += 1
            posts[i_row, i_col] = r
    else:
        # outdim_parallel=False
        def kernel(
            w_low: warp.array1d(dtype=weight_dtype),
            w_high: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)
            while cursor < num_cols:
                if _binomial_n1(state, conn_prob0) == 1:
                    for j in range(posts.shape[1]):
                        raw_v = warp.randf(state, w_low0, w_high0)
                        posts[cursor, j] += B[i_row, j] * raw_v
                cursor += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mm_uniform_jvp_w_low(
    w_low_dot,
    w_low,
    w_high,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mm_uniform_p_call(
        w_low_dot,
        w_high,
        conn_prob,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


def _jitc_mm_uniform_jvp_w_high(
    w_high_dot,
    w_low,
    w_high,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mm_uniform_p_call(
        w_low,
        w_high_dot,
        conn_prob,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


def _jitc_mm_uniform_jvp_B(
    B_dot,
    w_low,
    w_high,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return [
        _jitc_matmat_uniform(
            w_low,
            w_high,
            conn_prob,
            B_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def _jitc_mm_uniform_transpose_rules(
    ct,
    w_low,
    w_high,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    assert not ad.is_undefined_primal(w_low)
    assert not ad.is_undefined_primal(w_high)
    assert not ad.is_undefined_primal(conn_prob)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(B)

    r = jitc_mm_uniform_p_call(
        w_low,
        w_high,
        conn_prob,
        ct,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_low, w_high, conn_prob, r, seed, _


def _jitc_mm_uniform_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = args[3].reshape(m, batch_size * n)
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape, batch_size, n])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for JIT connection CSR matrix-matrix product")


def jitc_mm_uniform_p_call(
    w_low,
    w_high,
    conn_prob,
    B,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    conn_prob = jnp.atleast_1d(conn_prob)

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_low.dtype)
    )

    return jitc_mm_uniform_p(
        w_low,
        w_high,
        conn_prob,
        B,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        conn_prob_info=jax.ShapeDtypeStruct(conn_prob.shape, conn_prob.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mm_uniform_p = XLACustomKernel(
    'jitc_mm_uniform',
    cpu_kernel=NumbaKernelGenerator(_jitc_mm_uniform_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mm_uniform_gpu_kernel_generator,
        dim=lambda out_info, **kwargs: (
            out_info.shape[0], out_info.shape[1]
        ),
        input_output_aliases={5: 0}
    )
)

jitc_mm_uniform_p.defjvp(_jitc_mm_uniform_jvp_w_low,
                            _jitc_mm_uniform_jvp_w_high,
                            None,
                            _jitc_mm_uniform_jvp_B)
jitc_mm_uniform_p.def_transpose_rule(_jitc_mm_uniform_transpose_rules)
jitc_mm_uniform_p.def_batching_rule(_jitc_mm_uniform_batching)


# jitc csrmm normal

def _jitc_mm_normal_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matmat_normal` operation.
    """
    import numba

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_mu, w_sigma, conn_prob, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_rows):
                for i_col in range(num_cols):
                    connections = np.random.binomial(1, conn_prob0, num_cols)
                    random_weights = np.random.normal(w_mu0, w_sigma0, num_cols)

                    posts[i_row, i_col] = np.sum(B[i_row, :] * random_weights * connections)

    else:
        # outdim_parallel=False
        # TODO: more checks on this kernel (random generation method)
        def kernel(w_mu, w_sigma, conn_prob, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_cols):
                for i_row in range(num_rows):
                    r = B[i_row, :]

                    connections = np.random.binomial(1, conn_prob0, num_rows)
                    random_weights = np.random.normal(w_mu0, w_sigma0, num_rows)
                    effective_weights = connections * random_weights

                    posts += effective_weights[:, np.newaxis] * r

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mm_normal_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matmat_normal` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    B_dtype = dtype_to_warp_type(B_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            w_mu: warp.array1d(dtype=weight_dtype),
            w_sigma: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)

            while cursor < num_cols:
                if _binomial_n1(state, conn_prob0) == 1:
                    raw_v = w_mu0 + w_sigma0 * warp.randf(state)
                    r += B[i_row, cursor] * raw_v
                cursor += 1
            posts[i_row, i_col] = r
    else:
        # outdim_parallel=False
        def kernel(
            w_mu: warp.array1d(dtype=weight_dtype),
            w_sigma: warp.array1d(dtype=weight_dtype),
            conn_prob: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            conn_prob0 = conn_prob[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)
            while cursor < num_cols:
                if _binomial_n1(state, conn_prob0) == 1:
                    for j in range(posts.shape[1]):
                        raw_v = w_mu0 + w_sigma0 * warp.randf(state)
                        posts[cursor, j] += B[i_row, j] * raw_v
                cursor += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mm_normal_jvp_w_mu(
    w_mu_dot,
    w_mu,
    w_sigma,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    return jitc_mm_normal_p_call(
        w_mu_dot,
        w_sigma,
        conn_prob,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


def _jitc_mm_normal_jvp_w_sigma(
    w_sigma_dot,
    w_mu,
    w_sigma,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mm_normal_p_call(
        w_mu,
        w_sigma_dot,
        conn_prob,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


def _jitc_mm_normal_jvp_B(
    B_dot,
    w_mu,
    w_sigma,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return [
        _jitc_matmat_normal(
            w_mu,
            w_sigma,
            conn_prob,
            B_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def _jitc_mm_normal_transpose_rules(
    ct,
    w_mu,
    w_sigma,
    conn_prob,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    assert not ad.is_undefined_primal(w_mu)
    assert not ad.is_undefined_primal(w_sigma)
    assert not ad.is_undefined_primal(conn_prob)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(B)

    r = jitc_mm_normal_p_call(
        w_mu,
        w_sigma,
        conn_prob,
        ct,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_mu, w_sigma, conn_prob, r, seed, _


def _jitc_mm_normal_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = args[3].reshape(m, batch_size * n)
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape, batch_size, n])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for JIT connection CSR matrix-matrix product")


def jitc_mm_normal_p_call(
    w_mu,
    w_sigma,
    conn_prob,
    B,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_mu = jnp.atleast_1d(w_mu)
    w_sigma = jnp.atleast_1d(w_sigma)
    conn_prob = jnp.atleast_1d(conn_prob)

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_mu.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_mu.dtype)
    )

    return jitc_mm_uniform_p(
        w_mu,
        w_sigma,
        conn_prob,
        B,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_mu.shape, w_mu.dtype),
        conn_prob_info=jax.ShapeDtypeStruct(conn_prob.shape, conn_prob.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mm_normal_p = XLACustomKernel(
    'jitc_mm_normal',
    cpu_kernel=NumbaKernelGenerator(_jitc_mm_normal_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mm_normal_gpu_kernel_generator,
        dim=lambda out_info, **kwargs: (
            out_info.shape[0], out_info.shape[1]
        ),
        input_output_aliases={5: 0}
    )
)

jitc_mm_normal_p.defjvp(_jitc_mm_normal_jvp_w_mu,
                           _jitc_mm_normal_jvp_w_sigma,
                           None,
                           _jitc_mm_normal_jvp_B)
jitc_mm_normal_p.def_transpose_rule(_jitc_mm_normal_transpose_rules)
jitc_mm_normal_p.def_batching_rule(_jitc_mm_normal_batching)
