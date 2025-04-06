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
from typing import Optional, Tuple

from ._typing import Data, Kernel

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
    weight: Data,
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
    weight: float
        The value of the random matrix.
    conn_prob: float
        The connection probability.
    v: Array, ndarray
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
    ...


def _jitc_csr_matvec_uniform(
    w_low: float,
    w_high: float,
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
    ...


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
    ...


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
    weight: Data,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
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
    ...



def jitc_csrmv_homo_gpu_kernel_generator(
    weight: Data,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
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
    ...

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
    outdim_parallel
):
    ...


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
    outdim_parallel
):
    ...

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
    outdim_parallel
):
    ...

def jitc_csrmv_homo_batching(
    args,
    axes,
    **kwargs
):
    ...

# jitc csrmv uniform

def jitc_csrmv_uniform_cpu_kernel_generator(
    w_low: float,
    w_high: float,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
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
    ...


def jitc_csrmv_uniform_gpu_kernel_generator(
    w_low: float,
    w_high: float,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
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
    ...

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
    ...


def jitc_csrmv_uniform_jvp_weights(
    w_dot,
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
    ...

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
    ...

def jitc_csrmv_uniform_batching(
    args,
    axes,
    **kwargs
):
    ...

# jitc csrmv normal

def jitc_csrmv_normal_cpu_kernel_generator(
    w_mu: float,
    w_sigma: float,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
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
    ...


def jitc_csrmv_normal_gpu_kernel_generator(
    w_mu: float,
    w_sigma: float,
    clen: float,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
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
    ...

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
    ...


def jitc_csrmv_normal_jvp_weights(
    w_dot,
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
    ...

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
    ...

def jitc_csrmv_normal_batching(
    args,
    axes,
    **kwargs
):
    ...


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