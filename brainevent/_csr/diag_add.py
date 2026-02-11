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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad
from jax.interpreters.partial_eval import DynamicJaxprTracer

from brainevent._misc import generate_block_dim
from brainevent._op import numba_kernel, jaxinfo_to_warpinfo, XLACustomKernel
from brainevent._op.benchmark import BenchmarkConfig


def _is_tracer(x):
    return isinstance(x, (jax.ShapeDtypeStruct, jax.core.ShapedArray, DynamicJaxprTracer, jax.core.Tracer))


def csr_diag_position_v2(indptr, indices, shape: brainstate.typing.Size):
    """Find the positions of diagonal elements in a CSR sparse matrix (v2).

    Searches through each row of the CSR matrix to locate elements that lie on
    the main diagonal (i.e., where the column index equals the row index). Returns
    both the CSR data-array positions and the corresponding diagonal indices,
    allowing selective updates to diagonal elements.

    Unlike :func:`csr_diag_position`, this version returns a pair of arrays and
    gracefully handles matrices where not every diagonal element is present in
    the sparsity pattern.

    Parameters
    ----------
    indptr : ndarray or jax.Array
        Row pointer array of the CSR format, with shape ``(n_rows + 1,)`` and
        integer dtype. Must not be a JAX tracer (i.e., cannot be used inside
        ``jax.jit``).
    indices : ndarray or jax.Array
        Column indices array of the CSR format, with shape ``(nse,)`` and integer
        dtype. Must not be a JAX tracer.
    shape : tuple of int
        Shape of the sparse matrix as ``(n_rows, n_cols)``. Both dimensions must
        be positive integers.

    Returns
    -------
    csr_positions : ndarray
        1-D ``int32`` array of positions in the CSR data array where diagonal
        elements were found.
    diag_positions : ndarray or None
        1-D ``int32`` array of the corresponding diagonal indices. ``None`` if
        the lengths of ``csr_positions`` and ``diag_positions`` differ (should
        not normally occur).

    Raises
    ------
    AssertionError
        If ``shape`` is not a length-2 tuple/list of positive integers, or if
        ``indptr`` is not 1-D, or if ``indices`` is not 1-D.
    ValueError
        If ``indptr`` or ``indices`` is a JAX tracer, since this function
        requires concrete values to iterate over the sparsity structure.

    See Also
    --------
    csr_diag_position : Simpler version that returns a single array with ``-1``
        for missing diagonals.
    csr_diag_add_v2 : Add values to diagonal elements using positions from this
        function.

    Notes
    -----
    This function uses Numba JIT compilation internally for efficient iteration
    over the sparse structure. The result is cached by Numba.

    The number of diagonal elements searched is ``min(n_rows, n_cols)``.  For
    each row ``i`` in ``range(min(n_rows, n_cols))``, the algorithm scans the
    column indices ``indices[indptr[i] : indptr[i+1]]`` looking for an entry
    where ``indices[j] == i``.  If found, ``j`` is recorded as the CSR
    position and ``i`` as the diagonal position.

    Formally, the returned arrays satisfy:

    ``A[diag_positions[k], diag_positions[k]] == data[csr_positions[k]]``

    for all ``k`` in ``range(len(csr_positions))``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._csr.diag_add import csr_diag_position_v2
        >>> indptr = np.array([0, 2, 4, 6], dtype=np.int32)
        >>> indices = np.array([0, 1, 1, 2, 0, 2], dtype=np.int32)
        >>> csr_pos, diag_pos = csr_diag_position_v2(indptr, indices, shape=(3, 3))
    """
    assert isinstance(shape, (tuple, list)), "shape must be a tuple or list"
    assert indptr.ndim == 1, "indptr must be a 1D array"
    assert indices.ndim == 1, "indices must be a 1D array"
    assert len(shape) == 2, "shape must be a tuple or list of length 2"
    assert all(isinstance(s, int) and s > 0 for s in shape), "shape must be a tuple or list of non-negative integers"
    if _is_tracer(indptr):
        raise ValueError('Cannot trace indptr when finding diagonal position')
    if _is_tracer(indices):
        raise ValueError('Cannot trace indices when finding diagonal position')
    n_size = min(shape)

    import numba

    @numba.njit(cache=True)
    def _find_diag_position(indptr_, indices_, n):
        csr_positions = []
        diag_positions = []
        for i in range(n):
            start = indptr_[i]
            end = indptr_[i + 1]
            for j in range(start, end):
                if indices_[j] == i:
                    csr_positions.append(j)
                    diag_positions.append(i)
                    break
        return np.asarray(csr_positions, dtype=np.int32), np.asarray(diag_positions, dtype=np.int32)

    csr_pos, diag_pos = _find_diag_position(np.asarray(indptr), np.asarray(indices), n_size)

    return (csr_pos, diag_pos) if len(csr_pos) == len(diag_pos) else (csr_pos, None)


def csr_diag_add_v2(csr_value, positions, diag_value):
    """Add values to the diagonal elements of a CSR sparse matrix (v2).

    Uses the position mapping returned by :func:`csr_diag_position_v2` to
    efficiently add ``diag_value`` to the diagonal entries of ``csr_value``. This
    is equivalent to the operation ``A = A + diag(diag_value)`` restricted to
    the existing sparsity pattern.

    Parameters
    ----------
    csr_value : jax.Array or Quantity
        Values of the non-zero elements in the CSR matrix, with shape ``(nse,)``.
        May carry physical units via ``brainunit.Quantity``.
    positions : tuple
        A 2-tuple ``(csr_positions, diag_positions)`` as returned by
        :func:`csr_diag_position_v2`. ``csr_positions`` is a 1-D integer array of
        indices into ``csr_value``. ``diag_positions`` is either a 1-D integer
        array of the same length or ``None``.
    diag_value : jax.Array or Quantity
        Values to add to the diagonal, with shape ``(n_diag,)``. Must have the
        same units and dtype as ``csr_value``.

    Returns
    -------
    jax.Array or Quantity
        Updated CSR value array with the same shape and units as ``csr_value``.

    Raises
    ------
    AssertionError
        If ``csr_value`` or ``diag_value`` is not 1-D, if their dtypes differ,
        if the position arrays are not 1-D integer arrays, or if the units of
        ``csr_value`` and ``diag_value`` are incompatible.

    See Also
    --------
    csr_diag_position_v2 : Compute the position mapping required by this
        function.
    csr_diag_add : Alternative implementation using :func:`csr_diag_position`.

    Notes
    -----
    This function performs the equivalent of the dense operation:

    ``A <- A + diag(d)``

    restricted to the existing sparsity pattern, where ``A`` is the sparse
    matrix represented by ``csr_value`` and ``d`` is ``diag_value``.

    When ``diag_positions`` is ``None``, ``diag_value`` is indexed directly by
    ``csr_positions`` (i.e., a one-to-one mapping is assumed):

    ``csr_value[csr_positions[k]] += diag_value[k]``

    When ``diag_positions`` is provided, the update is:

    ``csr_value[csr_positions[k]] += diag_value[diag_positions[k]]``

    for all ``k`` in ``range(len(csr_positions))``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from brainevent._csr.diag_add import csr_diag_position_v2, csr_diag_add_v2
        >>> indptr = np.array([0, 2, 4, 6], dtype=np.int32)
        >>> indices = np.array([0, 1, 1, 2, 0, 2], dtype=np.int32)
        >>> positions = csr_diag_position_v2(indptr, indices, shape=(3, 3))
        >>> csr_value = jnp.ones(6, dtype=jnp.float32)
        >>> diag_value = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        >>> updated = csr_diag_add_v2(csr_value, positions, diag_value)
    """
    assert u.fail_for_dimension_mismatch(csr_value, diag_value)
    assert csr_value.ndim == 1, "csr_value must be a 1D array"
    assert diag_value.ndim == 1, "diag_value must be a 1D array"
    assert csr_value.dtype == diag_value.dtype, "csr_value and diag_value must have the same dtype"
    csr_pos, diag_pos = positions
    assert csr_pos.ndim == 1, "csr_pos must be a 1D array"
    assert jnp.issubdtype(csr_pos.dtype, jnp.integer), "diag_position must be an integer array"
    if diag_pos is not None:
        assert diag_pos.ndim == 1, "diag_pos must be a 1D array"
        assert jnp.issubdtype(diag_pos.dtype, jnp.integer), "diag_position must be an integer array"

    diag_value = u.Quantity(diag_value).to(u.get_unit(csr_value)).mantissa
    csr_value, csr_unit = u.split_mantissa_unit(csr_value)
    if diag_pos is None:
        csr_value = csr_value.at[csr_pos].add(diag_value)
    else:
        csr_value = csr_value.at[csr_pos].add(diag_value[diag_pos])
    return u.maybe_decimal(csr_value * csr_unit)


def csr_diag_position(indptr, indices, shape: brainstate.typing.Size):
    """Find the positions of diagonal elements in a CSR sparse matrix.

    Searches through each row of the CSR matrix to locate elements on the main
    diagonal (where the column index equals the row index). For rows where no
    diagonal element exists in the sparsity pattern, the position is recorded as
    ``-1``.

    Parameters
    ----------
    indptr : ndarray or jax.Array
        Row pointer array of the CSR format, with shape ``(n_rows + 1,)`` and
        integer dtype. Must not be a JAX tracer (i.e., cannot be used inside
        ``jax.jit``).
    indices : ndarray or jax.Array
        Column indices array of the CSR format, with shape ``(nse,)`` and integer
        dtype. Must not be a JAX tracer.
    shape : tuple of int
        Shape of the sparse matrix as ``(n_rows, n_cols)``. Both dimensions must
        be positive integers.

    Returns
    -------
    jax.Array
        1-D ``int32`` array of length ``min(n_rows, n_cols)``. Each element is
        either the position in the CSR data array of the diagonal entry, or
        ``-1`` if the diagonal entry is absent from the sparsity pattern.

    Raises
    ------
    AssertionError
        If ``shape`` is not a length-2 tuple/list of positive integers.
    ValueError
        If ``indptr`` or ``indices`` is a JAX tracer, since this function
        requires concrete values to iterate over the sparsity structure.

    See Also
    --------
    csr_diag_position_v2 : Alternative version returning separate CSR and
        diagonal position arrays.
    csr_diag_add : Add values to diagonal elements using positions from this
        function.

    Notes
    -----
    This function uses Numba JIT compilation internally for efficient iteration
    over the sparse structure. The result is cached by Numba.

    The number of diagonal elements searched is ``n = min(n_rows, n_cols)``.
    For each row ``i`` in ``range(n)``, the algorithm scans the column indices
    ``indices[indptr[i] : indptr[i+1]]`` looking for an entry where
    ``indices[j] == i``.  If found, ``result[i] = j``; otherwise
    ``result[i] = -1``.

    The returned array ``pos`` satisfies:

    ``data[pos[i]]`` is the diagonal element ``A[i, i]`` when ``pos[i] >= 0``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._csr.diag_add import csr_diag_position
        >>> indptr = np.array([0, 2, 4, 6], dtype=np.int32)
        >>> indices = np.array([0, 1, 1, 2, 0, 2], dtype=np.int32)
        >>> pos = csr_diag_position(indptr, indices, shape=(3, 3))
    """
    assert isinstance(shape, (tuple, list)), "shape must be a tuple or list"
    assert len(shape) == 2, "shape must be a tuple or list of length 2"
    assert all(isinstance(s, int) and s > 0 for s in shape), "shape must be a tuple or list of non-negative integers"
    n_size = min(shape)

    if _is_tracer(indptr):
        raise ValueError('Cannot trace indptr when finding diagonal position')
    if _is_tracer(indices):
        raise ValueError('Cannot trace indices when finding diagonal position')

    import numba

    @numba.njit(cache=True)
    def _find_diag_position(indptr_, indices_, n):
        results = []
        for i in range(n):
            start = indptr_[i]
            end = indptr_[i + 1]
            for j in range(start, end):
                if indices_[j] == i:
                    results.append(j)
                    break
            else:
                results.append(-1)
        return np.asarray(results, dtype=np.int32)

    return jnp.asarray(
        _find_diag_position(np.asarray(indptr), np.asarray(indices), n_size)
    )


def csr_diag_add(csr_value, diag_position, diag_value, backend: Optional[str] = None):
    """Add values to the diagonal elements of a CSR sparse matrix.

    Uses the position array returned by :func:`csr_diag_position` to add
    ``diag_value`` to the diagonal entries of the CSR data array. Positions with
    value ``-1`` (indicating absent diagonal elements) are skipped. This is
    equivalent to the operation ``A = A + diag(diag_value)`` restricted to the
    existing sparsity pattern.

    Parameters
    ----------
    csr_value : jax.Array or Quantity
        Values of the non-zero elements in the CSR matrix, with shape ``(nse,)``.
        May carry physical units via ``brainunit.Quantity``.
    diag_position : jax.Array
        1-D ``int32`` array of diagonal positions as returned by
        :func:`csr_diag_position`. Entries equal to ``-1`` are ignored.
    diag_value : jax.Array or Quantity
        Values to add to the diagonal, with shape matching ``diag_position``.
        Must have the same units and dtype as ``csr_value``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'warp'``, ``'pallas'``, or
        ``None`` for automatic selection.

    Returns
    -------
    jax.Array or Quantity
        Updated CSR value array with the same shape and units as ``csr_value``.

    Raises
    ------
    AssertionError
        If the units of ``csr_value`` and ``diag_value`` are incompatible.

    See Also
    --------
    csr_diag_position : Compute the position array required by this function.
    csr_diag_add_v2 : Alternative implementation using
        :func:`csr_diag_position_v2`.

    Notes
    -----
    This function performs the equivalent of the dense operation:

    ``A <- A + diag(d)``

    restricted to the existing sparsity pattern. For each diagonal index ``i``
    where ``diag_position[i] >= 0``, the update is:

    ``csr_value[diag_position[i]] += diag_value[i]``

    Entries where ``diag_position[i] == -1`` (diagonal element absent from the
    sparsity pattern) are silently skipped.

    The function internally converts ``diag_value`` to the same unit as
    ``csr_value`` before performing arithmetic, so mixed-unit inputs are
    supported as long as the units are dimensionally compatible.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from brainevent._csr.diag_add import csr_diag_position, csr_diag_add
        >>> indptr = np.array([0, 2, 4, 6], dtype=np.int32)
        >>> indices = np.array([0, 1, 1, 2, 0, 2], dtype=np.int32)
        >>> pos = csr_diag_position(indptr, indices, shape=(3, 3))
        >>> csr_value = jnp.ones(6, dtype=jnp.float32)
        >>> diag_value = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        >>> updated = csr_diag_add(csr_value, pos, diag_value)
    """
    assert u.fail_for_dimension_mismatch(csr_value, diag_value)

    diag_value = u.Quantity(diag_value).to(u.get_unit(csr_value)).mantissa
    csr_value, csr_unit = u.split_mantissa_unit(csr_value)
    return u.maybe_decimal(
        csr_diag_add_call(csr_value, diag_position, diag_value, backend=backend)[0]
        * csr_unit
    )


def _csr_diag_add_numba_kernel_generator(
    csr_value_info: jax.ShapeDtypeStruct,
    diag_pos_info: jax.ShapeDtypeStruct,
    diag_value_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    @numba.njit(fastmath=True)
    def diag_add(csr_value, diag_position, diag_value, out):
        out[:] = csr_value[:]  # Copy input to output
        for i in range(diag_position.size):
            pos = diag_position[i]
            if pos >= 0:
                out[pos] += diag_value[i]

    def kernel(csr_value, diag_position, diag_value):
        return numba_kernel(diag_add, outs=kwargs['outs'])(csr_value, diag_position, diag_value)

    return kernel


def _csr_diag_add_warp_kernel_generator(
    csr_value_info: jax.ShapeDtypeStruct,
    diag_pos_info: jax.ShapeDtypeStruct,
    diag_value_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel
    from warp.jax_experimental import jax_kernel

    csr_value_warp_info = jaxinfo_to_warpinfo(csr_value_info)
    diag_pos_warp_info = jaxinfo_to_warpinfo(diag_pos_info)
    diag_value_warp_info = jaxinfo_to_warpinfo(diag_value_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    @warp.kernel
    def diag_add_warp(
        csr_value: csr_value_warp_info,
        diag_position: diag_pos_warp_info,
        diag_value: diag_value_warp_info,
        out: out_warp_info
    ):
        i_diag = warp.tid()
        pos = diag_position[i_diag]
        if pos >= 0:
            out[pos] += diag_value[i_diag]

    def kernel(csr_value, diag_position, diag_value):
        dim = diag_pos_info.shape[0]
        fn = jax_kernel(diag_add_warp, launch_dims=[dim], num_outputs=1, in_out_argnames=['out'])
        return fn(csr_value, diag_position, diag_value, jnp.array(csr_value))

    return kernel


def _csr_diag_add_pallas_kernel_generator(
    csr_value_info: jax.ShapeDtypeStruct,
    diag_pos_info: jax.ShapeDtypeStruct,
    diag_value_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    total = diag_pos_info.shape[0]
    block_dim = generate_block_dim(total, 512)

    def diag_add_pallas(csr_value_ref, diag_position_ref, diag_value_ref, _, out_ref):
        i_tile = pl.program_id(0)
        i_tile_start = i_tile * block_dim
        mask = (i_tile_start + jnp.arange(block_dim)) < total
        positions = diag_position_ref[pl.dslice(i_tile_start, block_dim)]
        values = diag_value_ref[pl.dslice(i_tile_start, block_dim)]
        valid_mask = mask & (positions >= 0)
        atomic_add(out_ref, positions, values, mask=valid_mask)

    def kernel(csr_value, diag_position, diag_value):
        fn = pl.pallas_call(
            diag_add_pallas,
            grid=(pl.cdiv(total, block_dim),),
            input_output_aliases={3: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(csr_value, diag_position, diag_value, jnp.array(csr_value))

    return kernel


def _csr_diag_add_jvp_csr_value(dot, csr_value, diag_position, diag_value, **kwargs):
    return (dot,)


def _csr_diag_add_jvp_diag_value(dot, csr_value, diag_position, diag_value, **kwargs):
    return csr_diag_add_call(jnp.zeros_like(csr_value), diag_position, dot, backend=kwargs['backend'], )


def _csr_diag_add_transpose_value(ct, csr_value, diag_position, diag_value, **kwargs):
    assert not ad.is_undefined_primal(diag_position)
    ct = ct[0]
    raise NotImplementedError


def _csr_diag_add_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    n_conn = max(1, int(n_post * prob))
    csr_value = jnp.ones(n_pre * n_conn, dtype=dtype)
    n_diag = min(n_pre, n_post)
    diag_position = jnp.asarray(np.arange(n_diag, dtype=np.int32))
    diag_value = jnp.ones(n_diag, dtype=dtype)
    return [BenchmarkConfig("default", (csr_value, diag_position, diag_value))]


def csr_diag_add_call(csr_value, diag_position, diag_value, *, backend: Optional[str] = None):
    """Invoke the low-level XLA custom kernel for CSR diagonal addition.

    Validates inputs and dispatches to :data:`csr_diag_add_p`. Most users should
    prefer :func:`csr_diag_add` which additionally handles physical units.

    Parameters
    ----------
    csr_value : jax.Array
        Values of the non-zero elements in the CSR matrix, with shape ``(nse,)``.
    diag_position : jax.Array
        1-D integer array of diagonal positions. Entries equal to ``-1`` are
        ignored during the addition.
    diag_value : jax.Array
        Values to add to the diagonal, with the same shape as ``diag_position``
        and the same dtype as ``csr_value``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'warp'``, ``'pallas'``, or
        ``None`` for automatic selection.

    Returns
    -------
    tuple of jax.Array
        A single-element tuple containing the updated CSR value array with the
        same shape as ``csr_value``.

    Raises
    ------
    AssertionError
        If any input is not 1-D, if ``diag_position`` and ``diag_value`` have
        different shapes, if ``diag_position`` is not an integer array, or if
        ``csr_value`` and ``diag_value`` have different dtypes.

    See Also
    --------
    csr_diag_add : High-level wrapper with unit handling.
    csr_diag_add_p : The underlying ``XLACustomKernel`` instance.

    Notes
    -----
    This function operates on unitless mantissa arrays. All physical-unit
    handling is performed by the caller (:func:`csr_diag_add`).  The update
    applied is:

    ``csr_value[diag_position[i]] += diag_value[i]``  for all ``i`` where ``diag_position[i] >= 0``

    The function constructs :class:`jax.ShapeDtypeStruct` metadata for each
    operand and forwards the call to the ``XLACustomKernel`` instance
    :data:`csr_diag_add_p`.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from brainevent._csr.diag_add import csr_diag_add_call
        >>> csr_value = jnp.ones(6, dtype=jnp.float32)
        >>> diag_position = jnp.array([0, 3, 5], dtype=jnp.int32)
        >>> diag_value = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        >>> (updated,) = csr_diag_add_call(csr_value, diag_position, diag_value)
    """
    assert csr_value.ndim == 1, "csr_value must be a 1D array"
    assert diag_position.ndim == 1, "diag_position must be a 1D array"
    assert diag_value.ndim == 1, "diag_value must be a 1D array"
    assert diag_position.shape == diag_value.shape, "diag_position must have the same shape as diag_value"
    assert jnp.issubdtype(diag_position.dtype, jnp.integer), "diag_position must be an integer array"
    assert csr_value.dtype == diag_value.dtype, "csr_value and diag_value must have the same dtype"

    return csr_diag_add_p(
        csr_value, diag_position, diag_value,
        outs=[jax.ShapeDtypeStruct(csr_value.shape, csr_value.dtype)],
        backend=backend,
        csr_value_info=jax.ShapeDtypeStruct(csr_value.shape, csr_value.dtype),
        diag_pos_info=jax.ShapeDtypeStruct(diag_position.shape, diag_position.dtype),
        diag_value_info=jax.ShapeDtypeStruct(diag_value.shape, diag_value.dtype),
    )


csr_diag_add_p = XLACustomKernel('csr_diag_add')
csr_diag_add_p.def_numba_kernel(_csr_diag_add_numba_kernel_generator)
csr_diag_add_p.def_warp_kernel(_csr_diag_add_warp_kernel_generator)
csr_diag_add_p.def_pallas_kernel('gpu', _csr_diag_add_pallas_kernel_generator)
csr_diag_add_p.def_jvp_rule2(_csr_diag_add_jvp_csr_value, None, _csr_diag_add_jvp_diag_value)
csr_diag_add_p.def_call(csr_diag_add_call)
csr_diag_add_p.def_tags('csr', 'diag')
csr_diag_add_p.def_benchmark_data(_csr_diag_add_benchmark_data)
