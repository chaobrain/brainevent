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

# -*- coding: utf-8 -*-

__all__ = [
    "_jitc_csr_matvec_homo",
    "_jitc_csr_matvec_uniform",
    "_jitc_csr_matvec_normal",
    "_jitc_csr_matmat_homo",
    "_jitc_csr_matmat_uniform",
    "_jitc_csr_matmat_normal",
]


def _jitc_csr_matvec_homo(
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
    conn_len = jnp.ceil(1.0 / conn_prob) * 2 - 1
    clen = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.asarray(seed, dtype=jnp.int32)
    seed = jnp.atleast_1d(seed)
    return _raw_jitc_csr_matvec_homo(
        weight, clen, v, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )


def _raw_jitc_csr_matvec_homo(
    weight: Data | float,
    clen: Data,
    v: Data,
    seed: Optional[Data],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    weight, unitd = u.split_mantissa_unit(weight)
    v, unitv = u.split_mantissa_unit(v)
    res = jitc_csrmv_homo_p_call(
        weight, clen, v, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )
    return u.maybe_decimal(res * unitd * unitv)


def _jitc_csr_matvec_uniform(
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
    w_low: float
        Lower boundary of the output interval.
    w_high: float
        Upper boundary of the output interval.
    conn_prob: float
        The connection probability.
    vector: Array, ndarray
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
    conn_len = jnp.ceil(1.0 / conn_prob) * 2 - 1
    clen = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.asarray(seed, dtype=jnp.int32)
    seed = jnp.atleast_1d(seed)
    return _raw_jitc_csr_matvec_uniform(
        w_low, w_high, clen, v, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )


def _raw_jitc_csr_matvec_uniform(
    w_low: Data | float,
    w_high: Data | float,
    clen: Data,
    v: Data,
    seed: Data,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    w_low, unit_w_low = u.split_mantissa_unit(w_low)
    w_high, unit_w_high = u.split_mantissa_unit(w_high.in_unit(unit_w_low))
    v, unitv = u.split_mantissa_unit(v)
    res = jitc_csrmv_uniform_p_call(
        w_low, w_high, clen, v, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )
    return u.maybe_decimal(res * unit_w_low * unitv)


def _jitc_csr_matvec_normal(
    w_mu: float,
    w_sigma: float,
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
    w_mu: float
        Mean (centre) of the distribution.
    w_sigma: float
        Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    conn_prob: float
        The connection probability.
    vector: Array, ndarray
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
    conn_len = jnp.ceil(1.0 / conn_prob) * 2 - 1
    clen = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.asarray(seed, dtype=jnp.int32)
    seed = jnp.atleast_1d(seed)
    return _raw_jitc_csr_matvec_normal(
        w_mu, w_sigma, clen, v, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )


def _raw_jitc_csr_matvec_normal(
    w_mu: Data | float,
    w_sigma: Data | float,
    clen: Data,
    v: Data,
    seed: Data,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    u.fail_for_dimension_mismatch(w_mu, w_sigma, "w_low and w_high must have the same dimension.")
    w_mu, unit_w_mu = u.split_mantissa_unit(w_mu)
    w_sigma, unit_w_sigma = u.split_mantissa_unit(w_sigma.in_unit(unit_w_mu))
    v, unitv = u.split_mantissa_unit(v)
    res = jitc_csrmv_normal_p_call(
        w_mu, w_sigma, clen, v, seed,
        shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )
    return u.maybe_decimal(res * unit_w_mu * unitv)


def _jitc_csr_matmat_homo(
    weight: float,
    conn_prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@m` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.
    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.
    .. warning::
        This API may change in the future.
    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@m`.
    .. note::
        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.
    Parameters
    ----------
    weight: float
        The value of the random matrix.
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
    ...


def _jitc_csr_matmat_uniform(
    w_low: float,
    w_high: float,
    conn_prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@m` operation,
    where :math:`M` is just-in-time randomly generated with a uniform distribution for its value.
    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.
    .. warning::
        This API may change in the future.
    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@m`.
    .. note::
        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).            
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time   
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.
    Parameters
    ----------
    w_low: float
        Lower boundary of the output interval.
    w_high: float
        Upper boundary of the output interval.
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
    ...


def _jitc_csr_matmat_normal(
    w_mu: float,
    w_sigma: float,
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
    `conn_prob`, and at each connection the value is the same scalar `weight`.
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
    ...


# Kernel generators for JIT connection SPMV

# jitc csrmv homo

def jitc_csrmv_homo_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_csr_matvec_homo` operation.
    Parameters
    ----------
    weight: float
        The value of the random matrix.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The CPU kernel.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(weight, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_row):
                r = 0.0
                i_col = np.random.randint(0, clen0 - 1)

                while i_col < num_col:
                    r += v[i_col]
                    i_col += np.random.randint(1, clen0)

                posts[i_row] = r * weight0
    else:
        # outdim_parallel=False
        def kernel(weight, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_col):
                i_row = np.random.randint(0, clen0 - 1)

                r = v[i_col] * weight0
                while i_row < num_row:
                    posts[i_row] += r
                    i_row += np.random.randint(1, clen0)

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def jitc_csrmv_homo_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    v_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_csr_matvec_homo` operation.
    Parameters
    ----------
    weight: float
        The value of the random matrix.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The GPU kernel.
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
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            tid = warp.tid()

            i_row = tid >> 5
            i_thread = tid & 31

            i_col = warp.int32(step * i_thread - 1)
            end_col = warp.min(i_col + step, num_col)

            r = float(0.0)
            state = warp.rand_init(seed0 + tid)

            inc = warp.randi(state, 1, clen0)
            i_col += inc

            while i_col < end_col:
                r += v[i_col]
                inc = warp.randi(state, 1, clen0)
                i_col += inc

            posts[i_row] += r * weight0
    else:
        def kernel(
            weight: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            tid = warp.tid()
            i_col = tid >> 5
            index = tid & 31

            col_v = v[i_col]
            i_row = warp.int32(step * index - 1)
            end = warp.min(i_row + step, num_row)

            state = warp.rand_init(seed0 + tid)

            inc = warp.randi(state, 1, clen0)
            i_row += inc

            while i_row < end:
                posts[i_row] += col_v * weight0
                inc = warp.randi(state, 1, clen0)
                i_row += inc

    kernel = warp.kernel(kernel)
    return kernel


def jitc_csrmv_homo_jvp_v(
    v_dot,
    weight,
    clen,
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
        _raw_jitc_csr_matvec_homo(
            weight,
            clen,
            v_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def jitc_csrmv_homo_jvp_weights(
    w_dot,
    weight,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_csrmv_homo_p_call(
        w_dot,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def jitc_csrmv_homo_transpose_rules(
    ct,
    weight,
    clen,
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
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(v)

    r = jitc_csrmv_homo_p_call(
        weight,
        clen,
        ct[0],
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return weight, clen, r, seed, _


def jitc_csrmv_homo_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitc_csrmm_homo_p_call(
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
        r = jitc_csrmm_homo_p_call(
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


def jitc_csrmv_homo_p_call(
    weight,
    clen,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    weight = jnp.atleast_1d(weight)

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weight.dtype)
    )

    return jitc_csrmv_homo_p(
        weight,
        clen,
        v,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        v_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_csrmv_homo_p = XLACustomKernel(
    'jitc_csrmv_homo',
    cpu_kernel=NumbaKernelGenerator(jitc_csrmv_homo_cpu_kernel_generator, input_output_aliases={4: 0}),
    gpu_kernel=WarpKernelGenerator(
        jitc_csrmv_homo_gpu_kernel_generator,
        dim=lambda v_info, out_info, outdim_parallel, **kwargs: (
            out_info.shape[0] * 32 if outdim_parallel else
            v_info.shape[0] * 32
        ),
        input_output_aliases={4: 0}
    )
)

jitc_csrmv_homo_p.defjvp(jitc_csrmv_homo_jvp_weights, None, jitc_csrmv_homo_jvp_v, None)
jitc_csrmv_homo_p.def_transpose_rule(jitc_csrmv_homo_transpose_rules)
jitc_csrmv_homo_p.def_batching_rule(jitc_csrmv_homo_batching)


# jitc csrmv uniform

def jitc_csrmv_uniform_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_csr_matvec_uniform` operation.
    Parameters
    ----------
    w_low: float
        Lower boundary of the output interval.
    w_high: float
        Upper boundary of the output interval.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The CPU kernel.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_low, w_high, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_row):
                total = 0.0
                i_col = np.random.randint(0, clen0 - 1)

                while i_col < num_col:
                    raw_v = np.random.uniform(w_low0, w_high0)
                    total += v[i_col] * raw_v
                    i_col += np.random.randint(1, clen0)

                posts[i_row] = total
    else:
        # outdim_parallel=False
        def kernel(w_low, w_high, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_col):
                col_v = v[i_col]
                i_row = np.random.randint(0, clen0 - 1)

                while i_row < num_row:
                    raw_v = np.random.uniform(w_low0, w_high0)
                    posts[i_row] += col_v * raw_v
                    i_row += np.random.randint(1, clen0)

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def jitc_csrmv_uniform_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    v_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_csr_matvec_uniform` operation.
    Parameters
    ----------
    w_low: float
        Lower boundary of the output interval.
    w_high: float
        Upper boundary of the output interval.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The GPU kernel.
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
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            tid = warp.tid()

            i_row = tid >> 5
            i_thread = tid & 31

            i_col = warp.int32(step * i_thread - 1)
            end_col = warp.min(i_col + step, num_col)

            r = float(0.0)
            state = warp.rand_init(seed0 + tid)

            inc = warp.randi(state, 1, clen0)
            i_col += inc

            while i_col < end_col:
                row_v = warp.randu(state, w_low0, w_high0)
                r += v[i_col] * row_v
                inc = warp.randi(state, 1, clen0)
                i_col += inc

            posts[i_row] += r
    else:
        def kernel(
            w_low: warp.array1d(dtype=weight_dtype),
            w_high: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            tid = warp.tid()
            i_col = tid >> 5
            index = tid & 31

            col_v = v[i_col]
            i_row = warp.int32(step * index - 1)
            end = warp.min(i_row + step, num_row)

            state = warp.rand_init(seed0 + tid)

            inc = warp.randi(state, 1, clen0)
            i_row += inc

            while i_row < end:
                row_v = warp.randu(state, w_low0, w_high0)
                posts[i_row] += col_v * row_v
                inc = warp.randi(state, 1, clen0)
                i_row += inc

    kernel = warp.kernel(kernel)
    return kernel


def jitc_csrmv_uniform_jvp_v(
    v_dot,
    w_low,
    w_high,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    return [
        _raw_jitc_csr_matvec_uniform(
            w_low,
            w_high,
            clen,
            v_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def jitc_csrmv_uniform_jvp_w_low(
    w_low_dot,
    w_low,
    w_high,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    return jitc_csrmv_uniform_p_call(
        w_low_dot,
        w_high,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def jitc_csrmv_uniform_jvp_w_high(
    w_high_dot,
    w_low,
    w_high,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    return jitc_csrmv_uniform_p_call(
        w_high_dot,
        w_high,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def jitc_csrmv_uniform_transpose_rules(
    ct,
    w_low,
    w_high,
    clen,
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
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(v)

    r = jitc_csrmv_uniform_p_call(
        w_low,
        w_high,
        clen,
        ct[0],
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_low, w_high, clen, r, seed, _


def jitc_csrmv_uniform_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitc_csrmm_uniform_p_call(
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
        r = jitc_csrmm_uniform_p_call(
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


def jitc_csrmv_uniform_p_call(
    w_low,
    w_high,
    clen,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_low.dtype)
    )

    return jitc_csrmv_uniform_p(
        w_low,
        w_high,
        clen,
        v,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        v_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_csrmv_uniform_p = XLACustomKernel(
    'jitc_csrmv_uniform',
    cpu_kernel=NumbaKernelGenerator(jitc_csrmv_uniform_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        jitc_csrmv_uniform_gpu_kernel_generator,
        dim=lambda v_info, out_info, outdim_parallel, **kwargs: (
            out_info.shape[0] * 32 if outdim_parallel else
            v_info.shape[0] * 32
        ),
        input_output_aliases={5: 0}
    )
)

jitc_csrmv_uniform_p.defjvp(jitc_csrmv_uniform_jvp_w_low, jitc_csrmv_uniform_jvp_w_high, None, jitc_csrmv_uniform_jvp_v)
jitc_csrmv_uniform_p.def_transpose_rule(jitc_csrmv_uniform_transpose_rules)
jitc_csrmv_uniform_p.def_batching_rule(jitc_csrmv_uniform_batching)


# jitc csrmv normal

def jitc_csrmv_normal_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_csr_matvec_normal` operation.
    Parameters
    ----------
    w_mu: float
        Mean (centre) of the distribution.
    w_sigma: float
        Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The CPU kernel.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_mu, w_sigma, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_row):
                total = 0.0
                i_col = np.random.randint(0, clen0 - 1)

                while i_col < num_col:
                    raw_v = np.random.normal(w_mu0, w_sigma0)
                    total += v[i_col] * raw_v
                    i_col += np.random.randint(1, clen0)

                posts[i_row] = total
    else:
        # outdim_parallel=False
        def kernel(w_mu, w_sigma, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_col):
                col_v = v[i_col]
                i_row = np.random.randint(0, clen0 - 1)

                while i_row < num_row:
                    raw_v = np.random.normal(w_mu0, w_sigma0)
                    posts[i_row] += col_v * raw_v
                    i_row += np.random.randint(1, clen0)

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def jitc_csrmv_normal_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    v_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_csr_matvec_normal` operation.
    Parameters
    ----------
    w_mu: float
        Mean (centre) of the distribution.
    w_sigma: float
        Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The GPU kernel.
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
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            tid = warp.tid()

            i_row = tid >> 5
            i_thread = tid & 31

            i_col = warp.int32(step * i_thread - 1)
            end_col = warp.min(i_col + step, num_col)

            r = float(0.0)
            state = warp.rand_init(seed0 + tid)

            inc = warp.randi(state, 1, clen0)
            i_col += inc

            while i_col < end_col:
                row_v = w_mu0 + w_sigma0 * warp.randn(state)
                r += v[i_col] * row_v
                inc = warp.randi(state, 1, clen0)
                i_col += inc

            posts[i_row] += r
    else:
        def kernel(
            w_mu: warp.array1d(dtype=weight_dtype),
            w_sigma: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            tid = warp.tid()
            i_col = tid >> 5
            index = tid & 31

            col_v = v[i_col]
            i_row = warp.int32(step * index - 1)
            end = warp.min(i_row + step, num_row)

            state = warp.rand_init(seed0 + tid)

            inc = warp.randi(state, 1, clen0)
            i_row += inc

            while i_row < end:
                row_v = w_mu0 + w_sigma0 * warp.randn(state)
                posts[i_row] += col_v * row_v
                inc = warp.randi(state, 1, clen0)
                i_row += inc

    kernel = warp.kernel(kernel)
    return kernel


def jitc_csrmv_normal_jvp_v(
    v_dot,
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    return [
        _raw_jitc_csr_matvec_normal(
            w_mu,
            w_sigma,
            clen,
            v_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def jitc_csrmv_normal_jvp_w_mu(
    w_mu_dot,
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    return jitc_csrmv_uniform_p_call(
        w_mu_dot,
        w_sigma,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )

def jitc_csrmv_normal_jvp_w_sigma(
    w_sigma_dot,
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    return jitc_csrmv_uniform_p_call(
        w_mu,
        w_sigma_dot,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def jitc_csrmv_normal_transpose_rules(
    ct,
    w_mu,
    w_sigma,
    clen,
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
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(v)

    r = jitc_csrmv_uniform_p_call(
        w_mu,
        w_sigma,
        clen,
        ct[0],
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_mu, w_sigma, clen, r, seed, _


def jitc_csrmv_normal_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitc_csrmm_normal_p_call(
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
        r = jitc_csrmm_normal_p_call(
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


def jitc_csrmv_normal_p_call(
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_mu = jnp.atleast_1d(w_mu)
    w_sigma = jnp.atleast_1d(w_sigma)

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_mu.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_mu.dtype)
    )

    return jitc_csrmv_uniform_p(
        w_mu,
        w_sigma,
        clen,
        v,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_mu.shape, w_mu.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        v_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_csrmv_normal_p = XLACustomKernel(
    'jitc_csrmv_normal',
    cpu_kernel=NumbaKernelGenerator(jitc_csrmv_normal_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        jitc_csrmv_normal_gpu_kernel_generator,
        dim=lambda v_info, out_info, outdim_parallel, **kwargs: (
            out_info.shape[0] * 32 if outdim_parallel else
            v_info.shape[0] * 32
        ),
        input_output_aliases={5: 0}
    )
)

jitc_csrmv_normal_p.defjvp(jitc_csrmv_normal_jvp_w_mu, jitc_csrmv_normal_jvp_w_sigma, None, jitc_csrmv_normal_jvp_v, None)
jitc_csrmv_normal_p.def_transpose_rule(jitc_csrmv_normal_transpose_rules)
jitc_csrmv_normal_p.def_batching_rule(jitc_csrmv_normal_batching)


# Kernel generators for JIT connection SPMM

# jitc csrmm homo

def jitc_csrmm_homo_cpu_kernel_generator(
    weight: Data,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_csr_matmat_homo` operation.
    Parameters
    ----------
    weight: float
        The value of the random matrix.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The CPU kernel.
    """
    ...


def jitc_csrmm_homo_gpu_kernel_generator(
    weight: Data,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_csr_matmat_homo` operation.
    Parameters
    ----------
    weight: float
        The value of the random matrix.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The GPU kernel.
    """
    ...


def jitc_csrmm_homo_jvp_left(
    w_dot,
    weight,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    ...


def jitc_csrmm_homo_jvp_right(
    B_dot,
    weight,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    ...


def jitc_csrmm_homo_transpose_rules(
    ct,
    weight,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    ...


def jitc_csrmm_homo_batching(
    args,
    axes,
    **kwargs
):
    ...


def jitc_csrmm_homo_p_call(
    weight,
    clen,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    ...


jitc_csrmm_homo_p = XLACustomKernel(
    'jitc_csrmm_homo',
    cpu_kernel=NumbaKernelGenerator(jitc_csrmm_homo_cpu_kernel_generator, input_output_aliases={4: 0}),
    gpu_kernel=WarpKernelGenerator(
        jitc_csrmm_homo_gpu_kernel_generator,
        # TODO: dim number
        # dim=lambda
        input_output_aliases={4: 0}
    )
)

jitc_csrmm_homo_p.defjvp(jitc_csrmm_homo_jvp_left, None, None, jitc_csrmm_homo_jvp_right)
jitc_csrmm_homo_p.def_transpose_rule(jitc_csrmm_homo_transpose_rules)
jitc_csrmm_homo_p.def_batching_rule(jitc_csrmm_homo_batching)


# jitc csrmm uniform

def jitc_csrmm_uniform_cpu_kernel_generator(
    w_low: float,
    w_high: float,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_csr_matmat_uniform` operation.
    Parameters
    ----------
    w_low: float
        Lower boundary of the output interval.
    w_high: float
        Upper boundary of the output interval.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The CPU kernel.
    """
    ...


def jitc_csrmm_uniform_gpu_kernel_generator(
    w_low: float,
    w_high: float,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_csr_matmat_uniform` operation.
    Parameters
    ----------
    w_low: float
        Lower boundary of the output interval.
    w_high: float
        Upper boundary of the output interval.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The GPU kernel.
    """
    ...


def jitc_csrmm_uniform_jvp_left(
    w_dot,
    w_low,
    w_high,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    ...


def jitc_csrmm_uniform_jvp_right(
    B_dot,
    w_low,
    w_high,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    ...


def jitc_csrmm_uniform_transpose_rules(
    ct,
    w_low,
    w_high,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    ...


def jitc_csrmm_uniform_batching(
    args,
    axes,
    **kwargs
):
    ...


def jitc_csrmm_uniform_p_call(
    w_low,
    w_high,
    clen,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    ...


jitc_csrmm_uniform_p = XLACustomKernel(
    'jitc_csrmm_uniform',
    cpu_kernel=NumbaKernelGenerator(jitc_csrmm_uniform_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        jitc_csrmm_uniform_gpu_kernel_generator,
        # TODO: dim number
        # dim=lambda
        input_output_aliases={5: 0}
    )
)

jitc_csrmm_uniform_p.defjvp(jitc_csrmm_uniform_jvp_left, None, None, jitc_csrmm_uniform_jvp_right)
jitc_csrmm_uniform_p.def_transpose_rule(jitc_csrmm_uniform_transpose_rules)
jitc_csrmm_uniform_p.def_batching_rule(jitc_csrmm_uniform_batching)


# jitc csrmm normal

def jitc_csrmm_normal_cpu_kernel_generator(
    w_mu: float,
    w_sigma: float,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_csr_matmat_normal` operation.
    Parameters
    ----------
    w_mu: float
        Mean (centre) of the distribution.
    w_sigma: float
        Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The CPU kernel.
    """
    ...


def jitc_csrmm_normal_gpu_kernel_generator(
    w_mu: float,
    w_sigma: float,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_csr_matmat_normal` operation.
    Parameters
    ----------
    w_mu: float
        Mean (centre) of the distribution.
    w_sigma: float
        Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    clen: float
        The connection length, equal to $$ \text{clen} = \left\lceil \frac{1}{p} \right\rceil \times 2 - 1 $$
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
    kernel: Kernel
        The GPU kernel.
    """
    ...


def jitc_csrmm_normal_jvp_left(
    w_dot,
    w_mu,
    w_sigma,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    ...


def jitc_csrmm_normal_jvp_right(
    B_dot,
    w_mu,
    w_sigma,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    ...


def jitc_csrmm_normal_transpose_rules(
    ct,
    w_mu,
    w_sigma,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    ...


def jitc_csrmm_normal_batching(
    args,
    axes,
    **kwargs
):
    ...


def jitc_csrmm_normal_p_call(
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    ...


jitc_csrmm_normal_p = XLACustomKernel(
    'jitc_csrmm_normal',
    cpu_kernel=NumbaKernelGenerator(jitc_csrmm_normal_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        jitc_csrmm_normal_gpu_kernel_generator,
        # TODO: dim number
        # dim=lambda
        input_output_aliases={5: 0}
    )
)

jitc_csrmm_normal_p.defjvp(jitc_csrmm_normal_jvp_left, None, None, jitc_csrmm_normal_jvp_right)
jitc_csrmm_normal_p.def_transpose_rule(jitc_csrmm_normal_transpose_rules)
jitc_csrmm_normal_p.def_batching_rule(jitc_csrmm_normal_batching)
