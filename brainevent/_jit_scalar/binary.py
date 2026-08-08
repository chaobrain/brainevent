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

from pathlib import Path
from typing import Optional, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._data import _initialize_seed, _initialize_conn_length
from brainevent._misc import namescope, _chunk_size, _walk_length, _LANE_STRIDE
from brainevent._numba_random import get_numba_light_rng_funcs
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig
from brainevent._op import load_cuda_file
from brainevent._typing import Data, MatrixShape
from .float import jitsmv_p_call, jitsmm_p_call
from brainevent._op.util import dtype_suffix

__all__ = [
    "binary_jitsmv",
    "binary_jitsmv_p",
    "binary_jitsmm",
    "binary_jitsmm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitsmv(
    weight: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
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
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    out: Array, ndarray, Quantity
        The output of :math:`y = M @ v` if ``transpose=False``,
        or the output of :math:`y = M^T @ v` if ``transpose=True``.

    Raises
    ------
    ValueError
        If ``prob`` is not a scalar, is not finite, or is outside ``[0, 1]``.
    AssertionError
        If the matrix shape and vector length are incompatible.

    See Also
    --------
    binary_jitsmm : Event-driven matrix-matrix multiplication with scalar weight.
    jitsmv : Float matrix-vector multiplication with scalar weight.

    Notes
    -----
    This function computes an event-driven (spike-based) matrix-vector product
    where the connectivity matrix ``M`` has the structure:

    ``M[i, j] = w * Bernoulli(prob)``

    and the input ``vector`` is treated as a binary event vector (spikes).
    The output for each element is:

    ``y[i] = sum_{j in C(i)} w * spike[j]``

    where ``C(i)`` is the deterministic random connection set for row ``i``
    (determined by the seed), and ``spike[j]`` is 1 if ``vector[j]`` is
    True (for boolean) or ``> 0`` (for float).

    Since the input is binary, the operation reduces to counting the number
    of active (spiking) presynaptic neurons that connect to each postsynaptic
    neuron, then scaling by ``w``:

    ``y[i] = w * |{j in C(i) : spike[j] = 1}|``

    The matrix is never materialized in memory. The connectivity pattern is
    regenerated on-the-fly using the seed and connection length parameter
    ``clen = 2 / prob``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_scalar.binary import binary_jitsmv
        >>> weight = 0.5
        >>> events = jnp.array([True, False, True, True, False])
        >>> result = binary_jitsmv(weight, 0.5, events, seed=42,
        ...                        shape=(3, 5))
        >>> result.shape  # (3,)
    """

    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = binary_jitsmv_p_call(
        weight,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitsmm(
    weight: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
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
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    out: Array, ndarray
        The output of :math:`y = M @ B` if ``transpose=False``,
        or the output of :math:`y = M^T @ B` if ``transpose=True``.

    Raises
    ------
    ValueError
        If ``prob`` is not a scalar, is not finite, or is outside ``[0, 1]``.
    AssertionError
        If the matrix shape and input matrix ``B`` dimensions are incompatible.

    See Also
    --------
    binary_jitsmv : Event-driven matrix-vector multiplication with scalar weight.
    jitsmm : Float matrix-matrix multiplication with scalar weight.

    Notes
    -----
    This function computes an event-driven (spike-based) matrix-matrix product
    where the connectivity matrix ``M`` has the structure:

    ``M[i, j] = w * Bernoulli(prob)``

    and the input matrix ``B`` is treated as a binary event matrix (each column
    is a spike vector). For each output element:

    ``Y[i, k] = sum_{j in C(i)} w * spike[j, k]``

    where ``C(i)`` is the deterministic random connection set for row ``i``
    and ``spike[j, k]`` is 1 if ``B[j, k]`` is True (for boolean) or ``> 0``
    (for float).

    This is equivalent to performing ``binary_jitsmv`` independently for each
    column of ``B``, but is implemented more efficiently as a single kernel.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_scalar.binary import binary_jitsmm
        >>> weight = 0.5
        >>> B = jnp.array([[True, False], [False, True], [True, True],
        ...                [False, False], [True, False]])
        >>> result = binary_jitsmm(weight, 0.5, B, seed=42,
        ...                        shape=(3, 5))
        >>> result.shape  # (3, 2)
    """

    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = binary_jitsmm_p_call(
        weight,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitsmv_numba_kernel(
    corder: bool,
    vector_info: jax.ShapeDtypeStruct,
    **kwargs
):
    """
    Build a Numba CPU kernel for binary event-driven scalar JIT matrix-vector product.

    The connectivity matrix is regenerated with the same ``light_rng`` chunk/lane
    walk as the CUDA kernel in ``binary_jitsmv.cu``, so the numba and CUDA
    backends draw a bit-identical matrix. ``corder`` selects the CUDA
    ``notrans``/``trans`` orientation:

    * ``corder=True`` (notrans): outer loop over the ``m`` output rows, gathering
      active input events -- ``out[row] += w * active(vector[j])``.
    * ``corder=False`` (trans): outer loop over the ``m`` input rows, scattering
      -- for each active ``vector[row]`` add ``w`` to every connected output.

    Parameters
    ----------
    corder : bool
        Column-order flag selecting the notrans/trans orientation (see above).
    vector_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input vector. ``bool``/``int8`` spikes
        are active when non-zero (matching the CUDA ``pack_bool``); other dtypes
        are active when ``> 0``.
    **kwargs : dict
        Additional keyword arguments; must include ``'outs'`` (output
        shape/dtype) and ``'shape'`` (the implicit matrix shape, used to derive
        the chunk width).

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, vector, seed, _) -> tuple``.
    """
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']

    stride = _LANE_STRIDE
    # ``chunk_size`` keys the RNG stream and must match the CUDA path: both
    # split the *walked* dimension into 4 chunks; ``n_chunks`` is then derived
    # from that dimension inside the kernel, as CUDA does.
    chunk_size = _chunk_size(_walk_length(kwargs['shape'], kwargs['transpose'], corder))
    is_bool = np.dtype(vector_info.dtype) in (np.dtype('bool'), np.dtype('int8'))

    if corder:
        # notrans: m = output rows, k = vector length (walk dimension).
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, vector, seed, posts):
            m = posts.shape[0]
            k = vector.shape[0]
            weight0 = weight[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                acc = np.float64(0.)
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            if is_bool:
                                if vector[j]:
                                    acc += 1.0
                            else:
                                if vector[j] > 0:
                                    acc += 1.0
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = acc * weight0

    else:
        # trans: m = vector length (active-event rows), k = output length (walk).
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, vector, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = vector.shape[0]
            weight0 = weight[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                if is_bool:
                    if not vector[row]:
                        continue
                else:
                    if not (vector[row] > 0):
                        continue
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            posts[j] += weight0
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(weight, clen, vector, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, vector, seed)

    return kernel


def _binary_jitsmv_cuda_kernel(
    corder: bool,
    vector_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_jitsmv.cu'),
        name='jit_scalar_binary_jitsmv',
    )
    wt_sfx = dtype_suffix(kwargs['weight_info'].dtype)
    # Dispatch is unchanged from the curand implementation: ``corder`` selects the
    # kernel (gather -> notrans, scatter -> trans); ``transpose`` only sets the
    # output shape and is absorbed by the FFI (it derives m/k from the tensor
    # sizes), so it is not needed here.
    variant = 'notrans' if corder else 'trans'
    kernel_name = f'jit_scalar_binary_jitsmv.{variant}{wt_sfx}'

    # The light kernel consumes a bit-packed spike mask.  ``chunk_size`` follows the
    # float/CSR convention so the drawn matrix is identical across operators.
    k = int(vector_info.shape[0])
    n_words = (k + 31) // 32
    chunk_size = _chunk_size(_walk_length(kwargs['shape'], kwargs['transpose'], corder))
    # bool/int8 spikes are active when non-zero; float spikes when > 0.
    is_bool = np.dtype(vector_info.dtype) in (np.dtype('bool'), np.dtype('int8'))

    def kernel(weight, clen, vector, seed, _):
        spikes = vector.astype(jnp.int8) if is_bool else (vector > 0).astype(jnp.int8)
        packed = jax.ffi.ffi_call(
            'jit_scalar_binary_jitsmv.pack_bool',
            jax.ShapeDtypeStruct((n_words,), jnp.uint32),
        )(spikes)
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            weight, clen, seed, packed,
            vector_size=np.int32(k), chunk_size=np.int32(chunk_size),
        )

    return kernel


def _jitsmv_jvp_v(v_dot, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the vector argument of the binary JIT scalar matrix-vector product.

    Computes the Jacobian-vector product with respect to the input vector by
    delegating to the float ``jitsmv_p_call`` with the tangent ``v_dot``.

    Parameters
    ----------
    v_dot : jax.Array
        The tangent vector for the input vector.
    weight, clen, vector, seed, _ : jax.Array
        Primal values of the primitive inputs.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the matrix is transposed.
    corder : bool
        Column-order flag.
    **kwargs : dict
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jitsmv_p_call(
        weight, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmv_jvp_weights(w_dot, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the weight argument of the binary JIT scalar matrix-vector product.

    Computes the Jacobian-vector product with respect to the weight by
    delegating to ``binary_jitsmv_p_call`` with the tangent ``w_dot``.

    Parameters
    ----------
    w_dot : jax.Array
        The tangent vector for the weight.
    weight, clen, vector, seed, _ : jax.Array
        Primal values of the primitive inputs.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the matrix is transposed.
    corder : bool
        Column-order flag.
    **kwargs : dict
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return binary_jitsmv_p_call(
        w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmv_transpose_rules(ct, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the binary JIT scalar matrix-vector product.

    Implements the VJP backward pass. When the vector is the undefined primal,
    computes the cotangent by running the transpose of the forward pass. When the
    weight is the undefined primal, computes the weight gradient via a
    sum-of-products reduction.

    Parameters
    ----------
    ct : tuple
        The cotangent values from the output.
    weight, clen, vector, seed, _ : jax.Array
        Primal or undefined-primal values.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the forward pass used a transposed matrix.
    corder : bool
        Column-order flag used in the forward pass.
    **kwargs : dict
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        Cotangent values for ``(weight, clen, vector, seed, _)``.

    Raises
    ------
    NotImplementedError
        If neither ``vector`` nor ``weight`` is the undefined primal.
    """
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
            corder=not corder,
            backend=kwargs['backend'],
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
            corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw = jnp.sum(row * vector, keepdims=True).reshape(weight.aval.shape)
        return dw, clen, vector, seed, _
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitsmv_batching(args, axes, **kwargs):
    """
    Batching (vmap) rule for the binary JIT scalar matrix-vector product.

    Handles vectorized mapping over the input vector dimension by dispatching
    to the matrix-matrix product primitive ``binary_jitsmm_p_call``.

    Parameters
    ----------
    args : tuple
        The batched arguments ``(weight, clen, vector, seed, _)``.
    axes : tuple
        The batch axes for each argument.
    **kwargs : dict
        Keyword arguments including ``'shape'``, ``'transpose'``, ``'corder'``,
        and ``'backend'``.

    Returns
    -------
    tuple
        A 2-tuple of ``(results, out_axes)``.
    """
    if tuple(axes) == (None, None, 0, None, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitsmm_p_call(
            args[0],
            args[1],
            args[2].T,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitsmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(binary_jitsmv_p, args, axes, **kwargs)


def _jitsmv_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the binary JIT scalar matrix-vector product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering combinations of
        transpose, corder, and boolean vs. float event vectors.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                weight = jnp.ones(1, dtype=dtype)
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    vector = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    vector = jnp.asarray(np.random.rand(v_size), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weight, clen, vector, seed),
                        {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                    )
                )
    return configs


def binary_jitsmv_p_call(
    weight,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
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
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    tuple
        A tuple containing the output array from the primitive operation.
        The output shape is determined by the matrix shape and transpose flag:
        - If ``transpose=False``: output shape is ``(shape[0],)``
        - If ``transpose=True``: output shape is ``(shape[1],)``

    Notes
    -----
    This function is intended as an internal implementation detail and is used by the
    higher-level ``binary_jitsmv`` function, which properly handles units and provides
    a more user-friendly interface.

    The operation is implemented as an XLA custom kernel to achieve high performance on
    both CPU and GPU. The primitive supports JAX transformations including grad, vmap, and jit.

    When using ``corder=True`` (default), the generated matrix ``M`` when ``transpose=False``
    will generally be different from the implicitly generated ``M^T`` when ``transpose=True``.
    Set ``corder=False`` if exact correspondence between ``M`` and ``M^T`` is required.

    See Also
    --------
    binary_jitsmv : High-level function with unit handling.
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

    return binary_jitsmv_p(
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
        backend=backend,
    )


binary_jitsmv_p = XLACustomKernel(
    'binary_jitsmv',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_jitsmv``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) JIT scalar connectivity
matrix-vector multiplication operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

In this operation, the connectivity matrix has all weights set to a single scalar value,
and the input vector is treated as binary events (spikes). Only active events contribute
to the output computation.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_jitsmv_p.available_backends(platform)``,
and the default backend can be configured with ``binary_jitsmv_p.set_default(platform, backend)``.

See Also
--------
binary_jitsmv : High-level user-facing function wrapper.
"""
)
binary_jitsmv_p.def_numba_kernel(_jitsmv_numba_kernel)
binary_jitsmv_p.def_cuda_raw_kernel(_binary_jitsmv_cuda_kernel, asdefault=True)
binary_jitsmv_p.def_jvp_rule2(_jitsmv_jvp_weights, None, _jitsmv_jvp_v, None, None)
binary_jitsmv_p.def_transpose_rule(_jitsmv_transpose_rules)
binary_jitsmv_p.def_batching_rule(_jitsmv_batching)
binary_jitsmv_p.def_call(binary_jitsmv_p_call)
binary_jitsmv_p.def_tags('jit_scalar', 'binary')
binary_jitsmv_p.def_benchmark_data(_jitsmv_benchmark_data)


def _jitsmm_numba_kernel(
    corder: bool,
    B_info: jax.ShapeDtypeStruct,
    **kwargs
):
    """
    Build a Numba CPU kernel for binary event-driven scalar JIT matrix-matrix product.

    Parameters
    ----------
    corder : bool
        If True, iterate over output rows as the outer loop.
        If False, iterate over input rows as the outer loop with accumulation.
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input matrix ``B``. When
        ``B_info.dtype == jnp.bool_``, the kernel uses boolean comparisons;
        otherwise it uses ``> 0`` comparisons.
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'``.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, B, seed, _) -> tuple``.
    """
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']

    stride = _LANE_STRIDE
    # Same chunk keying and 32-lane walk as the matvec kernel and the CUDA path
    # (the walked dimension split into 4 chunks), so matmat and matvec draw the
    # same matrix.
    chunk_size = _chunk_size(_walk_length(kwargs['shape'], kwargs['transpose'], corder))
    is_bool = np.dtype(B_info.dtype) in (np.dtype('bool'), np.dtype('int8'))

    if corder:
        # notrans: Y = M @ B. m = output rows, k = B rows (walk dimension), n = cols.
        # The connectivity of a row is drawn once and reused across all n columns.
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, B, seed, posts):
            m = posts.shape[0]
            n = posts.shape[1]
            k = B.shape[0]
            weight0 = weight[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                out = np.zeros(n, dtype=weight.dtype)
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            for col in range(n):
                                if is_bool:
                                    if B[j, col]:
                                        out[col] += 1.0
                                else:
                                    if B[j, col] > 0.:
                                        out[col] += 1.0
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = out * weight0
    else:
        # trans: Y = M^T @ B. m = B rows (active-event rows), k = output rows (walk).
        # For active B[row, col], scatter ``w`` to every connected output row.
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, B, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = B.shape[0]
            weight0 = weight[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                if is_bool:
                    indices = np.where(B[row])[0]
                else:
                    indices = np.where(B[row] > 0.)[0]
                if indices.shape[0] == 0:
                    continue
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            posts[j, indices] += weight0
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(weight, clen, B, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, B, seed)

    return kernel


def _binary_jitsmm_cuda_kernel(
    corder: bool,
    B_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_jitsmm.cu'),
        name='jit_scalar_binary_jitsmm',
    )
    wt_sfx = dtype_suffix(kwargs['weight_info'].dtype)
    # Same dispatch as the matvec kernel: ``corder`` picks notrans/trans.  The
    # matmat kernels replay the same 32-lane walk, so gradients delegating to
    # ``float.jitsmm`` and CSR materialization all see one matrix.
    variant = 'notrans' if corder else 'trans'
    kernel_name = f'jit_scalar_binary_jitsmm.{variant}{wt_sfx}'

    out_info = kwargs['out_info']
    k_pack = int(B_info.shape[0])          # rows of B == the packed (spike) dimension
    n = int(B_info.shape[1])               # independent columns processed together
    n_words = (k_pack + 31) // 32
    chunk_size = _chunk_size(_walk_length(kwargs['shape'], kwargs['transpose'], corder))
    # notrans: output has m rows and walks k = B-rows; trans swaps the two roles.
    if corder:
        m_ffi, k_ffi = int(out_info.shape[0]), k_pack
    else:
        m_ffi, k_ffi = k_pack, int(out_info.shape[0])
    is_bool = np.dtype(B_info.dtype) in (np.dtype('bool'), np.dtype('int8'))

    def kernel(weight, clen, B, seed, _):
        spikes = B.astype(jnp.int8) if is_bool else (B > 0).astype(jnp.int8)
        packed = jax.ffi.ffi_call(
            'jit_scalar_binary_jitsmm.pack',
            jax.ShapeDtypeStruct((n, n_words), jnp.uint32),
        )(spikes, k=np.int32(k_pack), n=np.int32(n), n_words=np.int32(n_words))
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            weight, clen, seed, packed,
            m=np.int32(m_ffi), k=np.int32(k_ffi), n=np.int32(n),
            n_words=np.int32(n_words), chunk_size=np.int32(chunk_size),
        )

    return kernel


def _jitsmm_jvp_w(w_dot, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the weight argument of the binary JIT scalar matrix-matrix product.

    Parameters
    ----------
    w_dot : jax.Array
        The tangent vector for the weight.
    weight, clen, B, seed, _ : jax.Array
        Primal values of the primitive inputs.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the matrix is transposed.
    corder : bool
        Column-order flag.
    **kwargs : dict
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return binary_jitsmm_p_call(
        w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmm_jvp_B(B_dot, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the matrix ``B`` argument of the binary JIT scalar matrix-matrix product.

    Parameters
    ----------
    B_dot : jax.Array
        The tangent matrix for ``B``.
    weight, clen, B, seed, _ : jax.Array
        Primal values of the primitive inputs.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the matrix is transposed.
    corder : bool
        Column-order flag.
    **kwargs : dict
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jitsmm_p_call(
        weight, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmm_transpose_rules(ct, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the binary JIT scalar matrix-matrix product.

    Implements the VJP backward pass for ``binary_jitsmm_p``. When ``B`` is the
    undefined primal, computes the cotangent by running the transpose of the
    forward pass. When ``weight`` is the undefined primal, computes the weight
    gradient via a sum-of-products reduction.

    Parameters
    ----------
    ct : tuple
        The cotangent values from the output.
    weight, clen, B, seed, _ : jax.Array
        Primal or undefined-primal values.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the forward pass used a transposed matrix.
    corder : bool
        Column-order flag used in the forward pass.
    **kwargs : dict
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        Cotangent values for ``(weight, clen, B, seed, _)``.

    Raises
    ------
    NotImplementedError
        If neither ``B`` nor ``weight`` is the undefined primal.
    """
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
            backend=kwargs['backend'],
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
            corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw = jnp.sum(r * B, keepdims=True).reshape(weight.aval.shape)
        return dw, clen, B, seed, _

    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_homo not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    """
    Helper for batching over axis 1 (or 2) of a 3-D input matrix ``B``.

    Reshapes the 3-D input into a 2-D matrix, runs the matrix-matrix primitive,
    and reshapes the result back to 3-D.

    Parameters
    ----------
    args : tuple
        The batched arguments ``(weight, clen, B, seed, _)``, where ``B`` is 3-D.
    axis : int, default=1
        The batch axis in the output.
    **kwargs : dict
        Keyword arguments including ``'shape'``, ``'transpose'``, ``'corder'``,
        and ``'backend'``.

    Returns
    -------
    tuple
        A 2-tuple of ``([result_3d], [axis])``.
    """
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_jitsmm_p_call(
        args[0],
        args[1],
        B,
        args[3],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitsmm_batching(args, axes, **kwargs):
    """
    Batching (vmap) rule for the binary JIT scalar matrix-matrix product.

    Handles vectorized mapping over various axes of the input matrix ``B``
    by reshaping and dispatching to ``binary_jitsmm_p_call``.

    Parameters
    ----------
    args : tuple
        The batched arguments ``(weight, clen, B, seed, _)``.
    axes : tuple
        The batch axes for each argument.
    **kwargs : dict
        Keyword arguments including ``'shape'``, ``'transpose'``, ``'corder'``,
        and ``'backend'``.

    Returns
    -------
    tuple
        A 2-tuple of ``(results, out_axes)``.
    """
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
        return general_batching_rule(binary_jitsmm_p, args, axes, **kwargs)


def _jitsmm_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the binary JIT scalar matrix-matrix product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering combinations of
        transpose, corder, and boolean vs. float event matrices.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                weight = jnp.ones(1, dtype=dtype)
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    B = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    B = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(BenchmarkConfig(name, (weight, clen, B, seed), {
                    'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
                }))
    return configs


def binary_jitsmm_p_call(
    weight,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    r"""
    Low-level implementation function for binary event-driven JIT scalar matrix-matrix multiplication.

    This function prepares inputs and calls the XLA custom kernel primitive for matrix-matrix
    multiplication where the input matrix ``B`` is treated as a binary event matrix and the
    JIT connectivity matrix has a homogeneous scalar weight. The connectivity pattern is
    generated on-the-fly during execution using the provided seed and connection length.

    Parameters
    ----------
    weight : jax.Array
        Scalar weight value for non-zero connections, as a 1-D array of shape ``(1,)``.
    clen : jax.Array
        Connection length parameter (approximately ``2 / prob``), as a 1-D array
        of shape ``(1,)``.
    B : jax.Array
        Input binary event matrix of shape ``(k, n)`` where ``k`` must match the
        appropriate dimension of the JIT matrix (determined by ``transpose``).
    seed : jax.Array
        Random seed as a 1-D array of shape ``(1,)``.
    shape : MatrixShape
        The shape of the implicit JIT matrix as ``(num_rows, num_cols)``.
    transpose : bool
        If True, compute ``M^T @ B``; otherwise compute ``M @ B``.
    corder : bool
        Column-order flag controlling the parallelization strategy.
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    tuple
        A tuple containing the output matrix from the primitive operation:
        - If ``transpose=False``: output shape is ``(shape[0], B.shape[1])``
        - If ``transpose=True``: output shape is ``(shape[1], B.shape[1])``

    Notes
    -----
    This is an internal implementation function. Use the higher-level
    ``binary_jitsmm`` for a user-friendly interface with unit handling.

    See Also
    --------
    binary_jitsmm : High-level function with unit handling.
    """
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

    return binary_jitsmm_p(
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
        backend=backend,
    )


binary_jitsmm_p = XLACustomKernel(
    'binary_jitsmm',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_jitsmm``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) JIT scalar connectivity
matrix-matrix multiplication operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

In this operation, the connectivity matrix has all weights set to a single scalar value,
and the input matrix is treated as binary events (spikes). Each column of the input is
processed independently as an event vector.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_jitsmm_p.available_backends(platform)``,
and the default backend can be configured with ``binary_jitsmm_p.set_default(platform, backend)``.

See Also
--------
binary_jitsmm : High-level user-facing function wrapper.
"""
)
binary_jitsmm_p.def_numba_kernel(_jitsmm_numba_kernel)
binary_jitsmm_p.def_cuda_raw_kernel(_binary_jitsmm_cuda_kernel, asdefault=True)
binary_jitsmm_p.def_jvp_rule2(_jitsmm_jvp_w, None, _jitsmm_jvp_B, None, None)
binary_jitsmm_p.def_transpose_rule(_jitsmm_transpose_rules)
binary_jitsmm_p.def_batching_rule(_jitsmm_batching)
binary_jitsmm_p.def_call(binary_jitsmm_p_call)
binary_jitsmm_p.def_tags('jit_scalar', 'binary')
binary_jitsmm_p.def_benchmark_data(_jitsmm_benchmark_data)
