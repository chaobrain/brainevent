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

import operator
import os
import sys
import warnings
from types import FrameType
from typing import Union, Sequence, Optional, Dict

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp

from brainevent._error import PerformanceWarning, UnsupportedOperationError

__all__ = [
    'DataRepresentation',
    'JITCMatrix',
]


class DataRepresentation(u.sparse.SparseMatrix):
    def __init__(self, *args, shape: Sequence[int], buffers: Optional[Dict] = None):
        super().__init__(*args, shape=shape)
        self._buffer_registry: set[str] = set()
        if buffers is not None:
            assert isinstance(buffers, dict), "buffers must be a dictionary of name-value pairs."
            for name, value in buffers.items():
                self.register_buffer(name, value)

    def register_buffer(self, name, value=None):
        """Register a named buffer with a default value."""
        self._buffer_registry.add(name)
        setattr(self, name, value)

    def set_buffer(self, name, value):
        """Update the value of a previously registered buffer."""
        if name not in self._buffer_registry:
            raise ValueError(f"Buffer '{name}' not registered. Call register_buffer first.")
        setattr(self, name, value)

    @property
    def buffers(self):
        """Dict of all registered buffer names to their current values."""
        return {name: getattr(self, name, None) for name in self._buffer_registry}

    def _apply_data(self, fn):
        """Apply ``fn`` while preserving the physical data-buffer shape."""
        data = fn(self.data)
        expected_shape = getattr(self.data, 'shape', None)
        actual_shape = getattr(data, 'shape', None)
        if actual_shape != expected_shape:
            raise ValueError(
                f'{type(self).__name__}.apply must preserve data shape; '
                f'expected {expected_shape}, got {actual_shape}'
            )
        return data

    def _sum_data(self):
        """Sum represented entries, expanding a homogeneous shared value."""
        result = self.data.sum()
        if int(self.data.size) == 1:
            result = result * self.nse
        return result

    # ------------------------------------------------------------------ #
    # Common-API contract
    #
    # Every operation meaningful for a generic sparse weight matrix is declared
    # here, even where a particular family cannot support it. Subclasses either
    # override a method or *deliberately refuse* it by raising
    # :class:`~brainevent.UnsupportedOperationError`. The structure methods
    # ``todense``, ``with_data`` and ``transpose``/``T`` are part of the same
    # contract but already declared by ``saiunit.sparse.SparseMatrix``.
    # ------------------------------------------------------------------ #

    @classmethod
    def fromdense(cls, *args, **kwargs):
        """Construct a representation from a dense matrix.

        The concrete signature is defined per family; every subclass takes the
        dense matrix as the first positional argument followed by
        format-specific keyword options. A permissive ``*args, **kwargs`` is
        used here so each subclass can declare its own parameters without
        violating the Liskov substitution principle.

        Parameters
        ----------
        dense : jax.Array or brainunit.Quantity
            Dense ``(num_pre, num_post)`` matrix to encode.
        **kwargs
            Format-specific options (e.g. ``num_conn`` for fixed-num
            connections, ``nse`` for compressed-sparse formats).

        Returns
        -------
        DataRepresentation
            A new instance of ``cls`` encoding ``dense``.

        Raises
        ------
        NotImplementedError
            On the abstract base; concrete subclasses must override.
        UnsupportedOperationError
            If the format cannot be reconstructed from a dense matrix
            (e.g. just-in-time connectivity).
        """
        raise NotImplementedError(f"{cls.__name__}.fromdense")

    def tocoo(self):
        """Convert to coordinate (COO) format.

        Returns
        -------
        brainunit.sparse.COO
            The same logical matrix in COO format, shape unchanged.

        Raises
        ------
        NotImplementedError
            On the abstract base; concrete subclasses must override.

        See Also
        --------
        tocsr : Convert to compressed sparse row format.
        tocsc : Convert to compressed sparse column format.
        """
        raise NotImplementedError(f"{type(self).__name__}.tocoo")

    def tocsr(self):
        """Convert to Compressed Sparse Row (CSR) format.

        Returns
        -------
        CSR
            The same logical matrix in CSR format, shape unchanged.

        Raises
        ------
        NotImplementedError
            On the abstract base; concrete subclasses must override.

        See Also
        --------
        tocsc : Convert to compressed sparse column format.
        tocoo : Convert to coordinate format.
        """
        raise NotImplementedError(f"{type(self).__name__}.tocsr")

    def tocsc(self):
        """Convert to Compressed Sparse Column (CSC) format.

        Returns
        -------
        CSC
            The same logical matrix in CSC format, shape unchanged.

        Raises
        ------
        NotImplementedError
            On the abstract base; concrete subclasses must override.

        See Also
        --------
        tocsr : Convert to compressed sparse row format.
        tocoo : Convert to coordinate format.
        """
        raise NotImplementedError(f"{type(self).__name__}.tocsc")

    def dt2t(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        r"""Per-synapse ``w * y`` with ``y`` indexed by the row (pre) of ``W``.

        Part of the per-synapse eligibility protocol used by ``brainscale``.
        Concrete subclasses implement this directly when their storage can
        produce one value per represented synapse.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Pre-synaptic (row) vector, sized ``shape[0]``.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weights.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result.

        Raises
        ------
        NotImplementedError
            On the abstract base; concrete subclasses must override.
        UnsupportedOperationError
            If the format has no per-synapse weight (e.g. just-in-time
            connectivity without a direct generator).

        See Also
        --------
        dt2t_transposed : ``y`` indexed by the column (post) of ``W``.

        Notes
        -----
        This is the protocol for the :math:`\mathbf{D}^{t}\boldsymbol{\epsilon}^{t-1}`
        term of the D-RTRL (diagonal real-time recurrent learning) sensitivity
        update rule

        .. math::

            \boldsymbol{\epsilon}^{t} \approx \mathbf{D}^{t}\boldsymbol{\epsilon}^{t-1}
            + \mathrm{diag}(\mathbf{D}_{f}^{t}) \otimes \mathbf{x}^{t}

        ``w_dim_arr`` supplies the per-synapse entries of the Jacobian
        :math:`\mathbf{D}^{t}` and ``y_dim_arr`` supplies the previous-step
        eligibility trace :math:`\boldsymbol{\epsilon}^{t-1}`; the
        :math:`\mathrm{diag}(\mathbf{D}_{f}^{t}) \otimes \mathbf{x}^{t}` term is
        accumulated separately by the caller (``brainscale``).
        """
        raise NotImplementedError(f"{type(self).__name__}.dt2t")

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        r"""Per-synapse ``w * y`` with ``y`` indexed by the column (post) of ``W``.

        Adjoint counterpart of :meth:`dt2t`. Part of the per-synapse
        eligibility protocol used by ``brainscale``.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Post-synaptic (column) vector, sized ``shape[1]``.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weights.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result.

        Raises
        ------
        NotImplementedError
            On the abstract base; concrete subclasses must override.
        UnsupportedOperationError
            If the format has no per-synapse weight (e.g. just-in-time
            connectivity).

        See Also
        --------
        dt2t : ``y`` indexed by the row (pre) of ``W``.

        Notes
        -----
        Adjoint counterpart of the :math:`\mathbf{D}^{t}\boldsymbol{\epsilon}^{t-1}`
        protocol documented on :meth:`dt2t` — see there for the full D-RTRL
        update-rule context.
        """
        raise NotImplementedError(f"{type(self).__name__}.dt2t_transposed")

    def yw_to_w(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Deprecated alias for :meth:`dt2t`.

        .. deprecated::
            Use :meth:`dt2t` instead. Will be removed in a future release.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Pre-synaptic (row) vector, sized ``shape[0]``.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weights.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result; see :meth:`dt2t`.

        See Also
        --------
        dt2t : Replacement for this method.
        """
        warnings.warn(
            f'{type(self).__name__}.yw_to_w is deprecated and will be removed '
            'in a future release; use .dt2t instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dt2t(y_dim_arr, w_dim_arr)

    def yw_to_w_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Deprecated alias for :meth:`dt2t_transposed`.

        .. deprecated::
            Use :meth:`dt2t_transposed` instead. Will be removed in a future
            release.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Post-synaptic (column) vector, sized ``shape[1]``.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weights.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result; see :meth:`dt2t_transposed`.

        See Also
        --------
        dt2t_transposed : Replacement for this method.
        """
        warnings.warn(
            f'{type(self).__name__}.yw_to_w_transposed is deprecated and will '
            'be removed in a future release; use .dt2t_transposed instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dt2t_transposed(y_dim_arr, w_dim_arr)

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None):
        """Apply a pre-spike-triggered STDP update, returning a new matrix.

        Parameters
        ----------
        pre_spike : jax.Array
            Pre-synaptic spikes, shape ``(shape[0],)``.
        post_trace : jax.Array or brainunit.Quantity
            Post-synaptic trace, shape ``(shape[1],)``.
        w_min, w_max : jax.Array, brainunit.Quantity, number, or None, optional
            Clip bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        DataRepresentation
            A new matrix with updated values and identical structure.

        Raises
        ------
        NotImplementedError
            On the abstract base; concrete subclasses must override.
        UnsupportedOperationError
            If the format has no per-synapse plastic weight (e.g. just-in-time
            connectivity).

        See Also
        --------
        update_on_post : Post-spike-triggered counterpart.
        """
        raise NotImplementedError(f"{type(self).__name__}.update_on_pre")

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None):
        """Apply a post-spike-triggered STDP update, returning a new matrix.

        Parameters
        ----------
        pre_trace : jax.Array or brainunit.Quantity
            Pre-synaptic trace, shape ``(shape[0],)``.
        post_spike : jax.Array
            Post-synaptic spikes, shape ``(shape[1],)``.
        w_min, w_max : jax.Array, brainunit.Quantity, number, or None, optional
            Clip bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        DataRepresentation
            A new matrix with updated values and identical structure.

        Raises
        ------
        NotImplementedError
            On the abstract base; concrete subclasses must override.
        UnsupportedOperationError
            If the format has no per-synapse plastic weight (e.g. just-in-time
            connectivity).

        See Also
        --------
        update_on_pre : Pre-spike-triggered counterpart.
        """
        raise NotImplementedError(f"{type(self).__name__}.update_on_post")


_EVENT_SIDES = ('left', 'right')


def _resolve_corder(corder: Optional[bool], event_side: Optional[str]) -> bool:
    """Return the ``corder`` flag selected by ``corder`` and/or ``event_side``.

    ``event_side='left'`` makes ``events @ M`` event-driven (``corder=True``);
    ``event_side='right'`` makes ``M @ events`` event-driven (``corder=False``).
    With neither given the default is ``corder=True``.

    Raises
    ------
    ValueError
        If ``event_side`` is not ``None``, ``'left'`` or ``'right'``, or if it
        contradicts an explicitly given ``corder``.
    """
    if event_side is None:
        return True if corder is None else bool(corder)
    if event_side not in _EVENT_SIDES:
        raise ValueError(f"event_side must be 'left', 'right' or None, but got {event_side!r}.")
    selected = event_side == 'left'
    if corder is not None and bool(corder) != selected:
        raise ValueError(
            f"event_side={event_side!r} selects corder={selected}, which contradicts corder={corder}. "
            f"Pass only one of them."
        )
    return selected


# ``(class name, side)`` pairs that already warned about a non-event-driven product.
_WARNED_NOT_EVENT_DRIVEN: set = set()
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _user_stacklevel() -> int:
    """``stacklevel`` (for a ``warnings.warn`` in the caller) of the first frame outside brainevent."""
    frame: Optional[FrameType] = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back
        level += 1
    return level


class JITCMatrix(DataRepresentation):
    """
    Just-in-time Connectivity (JITC) matrix.

    A base class for just-in-time connectivity matrices that inherits from
    the SparseMatrix class in the ``brainunit`` library. This class serves as
    an abstraction for sparse matrices that are generated or computed on demand
    rather than stored in full.

    JITC matrices are particularly useful in neural network simulations where
    connectivity patterns might be large but follow specific patterns that
    can be efficiently computed rather than explicitly stored in memory.

    Notes
    -----
    This is a base class and should be subclassed for specific
    implementations of JITC matrices. All attributes from
    :class:`brainunit.sparse.SparseMatrix` are inherited.
    """
    __module__ = 'brainevent'

    @property
    def event_side(self) -> str:
        """Side of the event operand whose product is event-driven.

        ``'left'`` (``corder=True``): ``events @ M`` walks only the active rows.
        ``'right'`` (``corder=False``): ``M @ events`` walks only the active
        columns. The product with events on the other side walks every random
        stream. ``M.T`` flips the value, so ``events @ M`` and ``M.T @ events``
        always run the same kernel.

        Returns
        -------
        str
            ``'left'`` or ``'right'``.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> M = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 20))
            >>> M.event_side
            'left'
            >>> M.T.event_side
            'right'
        """
        return 'left' if self.corder else 'right'

    def _warn_if_not_event_driven(self, side: str) -> None:
        """Warn once if a binary-event product on ``side`` is not event-driven.

        Parameters
        ----------
        side : str
            ``'left'`` for ``events @ M``, ``'right'`` for ``M @ events``.
        """
        if (side == 'left') == bool(self.corder):
            return
        name = type(self).__name__
        key = (name, side)
        if key in _WARNED_NOT_EVENT_DRIVEN:
            return
        _WARNED_NOT_EVENT_DRIVEN.add(key)
        product, walked = ('events @ M', 'column') if side == 'left' else ('M @ events', 'row')
        warnings.warn(
            f"{name}(corder={bool(self.corder)}): `{product}` is not event-driven; it walks the "
            f"random stream of every {walked}. event_side={side!r} (corder={not self.corder}) makes it "
            f"event-driven but draws a different, statistically equivalent connectivity. `M.tocsr()` "
            f"keeps this connectivity in memory (O(nnz)) and gathers over stored indices instead of "
            f"regenerating the matrix.",
            PerformanceWarning,
            stacklevel=_user_stacklevel(),
        )

    @staticmethod
    def _split_event_operand(other, weight):
        """Classify a matmul operand for the JITC binary / float routes.

        A :class:`~brainevent.WeightedArray` routes to the float JITC
        primitives, whose kernels skip zero events: its values are cast to the
        floating dtype of ``weight`` (integer events get ``float0``
        cotangents) and returned as a plain operand.

        Parameters
        ----------
        other : object
            Right- or left-hand operand of ``@``.
        weight : jax.Array or u.Quantity
            A weight parameter of the matrix; its dtype selects the cast.

        Returns
        -------
        other : object
            The operand, unwrapped and cast for a ``WeightedArray``.
        event : tuple of (jax.Array, bool) or None
            ``(value, False)`` for a binary event operand; ``None`` otherwise.
        """
        from brainevent._event.dispatch import as_event_operand  # pylint: disable=import-outside-toplevel

        event = as_event_operand(other)
        if event is None or not event[1]:
            return other, event
        dtype = u.math.get_dtype(weight)
        if not jnp.issubdtype(dtype, jnp.floating):
            dtype = jnp.result_type(dtype, 1.0)
        return u.math.asarray(event[0]).astype(dtype), None

    @staticmethod
    def _scalar_operand(other, op):
        """Return ``other`` as a 0-d operand for an elementwise ``op``.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Operand of a binary elementwise operation.
        op : callable
            The operation, used in error messages.

        Returns
        -------
        jax.Array or u.Quantity
            ``other`` reshaped to ``()``, so the weight parameters keep their shape.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has more than one element.
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")
        other = u.math.asarray(other)
        if other.size != 1:
            raise NotImplementedError(
                f"binary operation {op} with an object of shape {other.shape}; "
                f"JITC matrices only support scalar operands."
            )
        return other.reshape(())

    def sum(self, axis: Optional[int] = None):
        """
        Sum of the represented matrix entries.

        Computed exactly from the generated connectivity with one
        matrix-vector product, without materializing the matrix.

        Parameters
        ----------
        axis : int or None, optional
            ``None`` (default) sums every entry, ``0`` (or ``-2``) gives the
            column sums of shape ``(shape[1],)`` and ``1`` (or ``-1``) the row
            sums of shape ``(shape[0],)``.

        Returns
        -------
        jax.Array or u.Quantity
            The sum, carrying the unit of the weights.

        Raises
        ------
        ValueError
            If ``axis`` is not ``None``, ``0``, ``1``, ``-1`` or ``-2``.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.5, 42), shape=(4, 6))
            >>> bool(jnp.allclose(mat.sum(), mat.todense().sum()))
            True
            >>> mat.sum(axis=0).shape
            (6,)
        """
        if axis is not None and axis not in (0, 1, -1, -2):
            raise ValueError(f"axis must be None, 0 or 1 for a 2-D matrix, got {axis}.")
        dtype = self.dtype
        if axis in (0, -2):
            return jnp.ones(self.shape[0], dtype=dtype) @ self
        row_sums = self @ jnp.ones(self.shape[1], dtype=dtype)
        return row_sums if axis is not None else row_sums.sum()

    def _unitary_op(self, op):
        """
        Apply a unary operation to the matrix.

        This is an internal method that should be implemented by subclasses
        to handle unary operations like absolute value, negation, etc.

        Parameters
        ----------
        op : callable
            Function from ``operator`` or compatible callable to apply.

        Raises
        ------
        NotImplementedError
            Raised because this base method must be implemented by subclasses.
        """
        raise NotImplementedError("unitary operation not implemented.")

    def apply(self, fn):
        """
        Apply a function to matrix value parameters while keeping structure.

        Parameters
        ----------
        fn : callable
            Unary callable applied by subclasses to their value parameters.

        Returns
        -------
        JITCMatrix
            A new matrix-like object with transformed values.
        """
        return self._unitary_op(fn)

    def __abs__(self):
        """
        Return the element-wise absolute value of the matrix.

        Computes ``abs(weight)`` for each value parameter of the matrix while
        preserving the sparse connectivity structure (probability, seed, shape,
        and memory layout order).

        Returns
        -------
        JITCMatrix
            A new JITC matrix whose value parameters are the absolute values
            of the original.

        See Also
        --------
        apply : General unary function application.
        __neg__ : Element-wise negation.
        __pos__ : Element-wise positive (identity).

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((-1.5, 0.1, 42), shape=(10, 10))
            >>> abs_mat = abs(mat)
            >>> float(abs_mat.weight)
            1.5
        """
        return self.apply(operator.abs)

    def __neg__(self):
        """
        Return the element-wise negation of the matrix.

        Computes ``-weight`` for each value parameter of the matrix while
        preserving the sparse connectivity structure.

        Returns
        -------
        JITCMatrix
            A new JITC matrix whose value parameters are negated.

        See Also
        --------
        apply : General unary function application.
        __abs__ : Element-wise absolute value.
        __pos__ : Element-wise positive (identity).

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> neg_mat = -mat
            >>> float(neg_mat.weight)
            -1.5
        """
        return self.apply(operator.neg)

    def __pos__(self):
        """
        Return the element-wise positive of the matrix (identity operation).

        Computes ``+weight`` for each value parameter, which returns an
        equivalent matrix. The sparse connectivity structure is preserved.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with the same value parameters.

        See Also
        --------
        apply : General unary function application.
        __abs__ : Element-wise absolute value.
        __neg__ : Element-wise negation.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> pos_mat = +mat
            >>> float(pos_mat.weight)
            1.5
        """
        return self.apply(operator.pos)

    def _binary_op(self, other, op):
        """
        Apply a binary operation between this matrix and another value.

        This is an internal method that should be implemented by subclasses
        to handle binary operations like addition, subtraction, etc.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand operand.
        op : callable
            Function from ``operator`` or compatible callable to apply.

        Raises
        ------
        NotImplementedError
            Raised because this base method must be implemented by subclasses.
        """
        raise NotImplementedError("binary operation not implemented.")

    def _binary_rop(self, other, op):
        """
        Apply a binary operation with the matrix as the right operand.

        This is an internal method that should be implemented by subclasses
        to handle reflected binary operations (right-side operations).

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand operand.
        op : callable
            Function from ``operator`` or compatible callable to apply.

        Raises
        ------
        NotImplementedError
            Raised because this base method must be implemented by subclasses.
        """
        raise NotImplementedError("binary operation not implemented.")

    def apply2(self, other, fn, *, reverse: bool = False):
        """
        Apply a binary function with consistent sparse-matrix semantics.

        Parameters
        ----------
        other : Any
            Right-hand operand for normal operations, or left-hand operand when
            ``reverse=True``.
        fn : callable
            Binary function from ``operator`` or a compatible callable.
        reverse : bool, optional
            If False, compute ``fn(self, other)`` via ``_binary_op``.
            If True, compute ``fn(other, self)`` via ``_binary_rop``.
            Defaults to False.

        Returns
        -------
        JITCMatrix or Any
            Result of the operation.
        """
        if reverse:
            return self._binary_rop(other, fn)
        return self._binary_op(other, fn)

    def __mul__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        """
        Multiply the matrix element-wise by a scalar or array.

        Computes ``self * other`` by applying ``operator.mul`` to the value
        parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand multiplicand. Typically a scalar value; support for
            non-scalar operands depends on the subclass implementation.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with scaled value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __rmul__ : Reflected multiplication (``other * self``).
        __truediv__ : Element-wise division.
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> scaled = mat * 2.0
            >>> float(scaled.weight)
            3.0
        """
        return self.apply2(other, operator.mul)

    def __truediv__(self, other):
        """
        Divide the matrix element-wise by a scalar or array.

        Computes ``self / other`` by applying ``operator.truediv`` to the
        value parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand divisor. Typically a scalar value; support for
            non-scalar operands depends on the subclass implementation.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with divided value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __rtruediv__ : Reflected division (``other / self``).
        __mul__ : Element-wise multiplication.
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((3.0, 0.1, 42), shape=(10, 10))
            >>> divided = mat / 2.0
            >>> float(divided.weight)
            1.5
        """
        return self.apply2(other, operator.truediv)

    def __add__(self, other):
        """
        Add a scalar or array element-wise to the matrix.

        Computes ``self + other`` by applying ``operator.add`` to the value
        parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand addend. Typically a scalar value or another JITC matrix
            with the same connectivity structure.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with summed value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape,
            or if two JITC matrices have incompatible seeds, shapes, or
            probabilities.

        See Also
        --------
        __radd__ : Reflected addition (``other + self``).
        __sub__ : Element-wise subtraction.
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> result = mat + 0.5
            >>> float(result.weight)
            2.0
        """
        return self.apply2(other, operator.add)

    def __sub__(self, other):
        """
        Subtract a scalar or array element-wise from the matrix.

        Computes ``self - other`` by applying ``operator.sub`` to the value
        parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand subtrahend. Typically a scalar value or another JITC
            matrix with the same connectivity structure.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with subtracted value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape,
            or if two JITC matrices have incompatible seeds, shapes, or
            probabilities.

        See Also
        --------
        __rsub__ : Reflected subtraction (``other - self``).
        __add__ : Element-wise addition.
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> result = mat - 0.5
            >>> float(result.weight)
            1.0
        """
        return self.apply2(other, operator.sub)

    def __mod__(self, other):
        """
        Compute the element-wise modulo of the matrix by a scalar or array.

        Computes ``self % other`` by applying ``operator.mod`` to the value
        parameters of the matrix and ``other``. The sparse connectivity
        structure is preserved.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Right-hand modulus. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with the modulo-reduced value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __rmod__ : Reflected modulo (``other % self``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((5.0, 0.1, 42), shape=(10, 10))
            >>> result = mat % 3.0
            >>> float(result.weight)
            2.0
        """
        return self.apply2(other, operator.mod)

    def __rmul__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        """
        Reflected multiplication: multiply a scalar or array by the matrix.

        Computes ``other * self`` by applying ``operator.mul`` with the
        operands in reflected order. This is invoked when the left operand
        does not support multiplication with a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand multiplicand. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with scaled value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __mul__ : Forward multiplication (``self * other``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> scaled = 2.0 * mat
            >>> float(scaled.weight)
            3.0
        """
        return self.apply2(other, operator.mul, reverse=True)

    def __rtruediv__(self, other):
        """
        Reflected division: divide a scalar or array by the matrix.

        Computes ``other / self`` by applying ``operator.truediv`` with the
        operands in reflected order. This is invoked when the left operand
        does not support division by a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand dividend. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix where each value parameter ``w`` is replaced
            by ``other / w``.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __truediv__ : Forward division (``self / other``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((2.0, 0.1, 42), shape=(10, 10))
            >>> result = 6.0 / mat
            >>> float(result.weight)
            3.0
        """
        return self.apply2(other, operator.truediv, reverse=True)

    def __radd__(self, other):
        """
        Reflected addition: add the matrix to a scalar or array.

        Computes ``other + self`` by applying ``operator.add`` with the
        operands in reflected order. This is invoked when the left operand
        does not support addition with a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand addend. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix with summed value parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __add__ : Forward addition (``self + other``).
        apply2 : General binary function application.

        Notes
        -----
        For commutative operands (e.g., plain scalars), ``other + self``
        produces the same result as ``self + other``.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> result = 0.5 + mat
            >>> float(result.weight)
            2.0
        """
        return self.apply2(other, operator.add, reverse=True)

    def __rsub__(self, other):
        """
        Reflected subtraction: subtract the matrix from a scalar or array.

        Computes ``other - self`` by applying ``operator.sub`` with the
        operands in reflected order. This is invoked when the left operand
        does not support subtraction of a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand minuend. Typically a scalar value.

        Returns
        -------
        JITCMatrix
            A new JITC matrix where each value parameter ``w`` is replaced
            by ``other - w``.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __sub__ : Forward subtraction (``self - other``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((1.5, 0.1, 42), shape=(10, 10))
            >>> result = 3.0 - mat
            >>> float(result.weight)
            1.5
        """
        return self.apply2(other, operator.sub, reverse=True)

    def __rmod__(self, other):
        """
        Reflected modulo: compute a scalar or array modulo the matrix.

        Computes ``other % self`` by applying ``operator.mod`` with the
        operands in reflected order. This is invoked when the left operand
        does not support modulo with a JITC matrix.

        Parameters
        ----------
        other : jax.typing.ArrayLike or u.Quantity
            Left-hand dividend for the modulo operation.

        Returns
        -------
        JITCMatrix
            A new JITC matrix where each value parameter ``w`` is replaced
            by ``other % w``.

        Raises
        ------
        NotImplementedError
            If the subclass does not support the given operand type or shape.

        See Also
        --------
        __mod__ : Forward modulo (``self % other``).
        apply2 : General binary function application.

        Examples
        --------
        .. code-block:: python

            >>> import brainevent
            >>> mat = brainevent.JITCScalarR((3.0, 0.1, 42), shape=(10, 10))
            >>> result = 7.0 % mat
            >>> float(result.weight)
            1.0
        """
        return self.apply2(other, operator.mod, reverse=True)

    # ------------------------------------------------------------------ #
    # Common-API contract for just-in-time connectivity
    #
    # JITC matrices are generated procedurally from ``(prob, seed)`` and are
    # non-plastic: the "weight" is usually a scalar / distribution parameter
    # rather than a stored per-synapse array. Several contract methods are
    # therefore deliberately refused at the base-class level. Distribution
    # subclasses may override methods such as ``dt2t`` when they can generate
    # the needed per-synapse values directly from their parameters.
    # ------------------------------------------------------------------ #

    @classmethod
    def fromdense(cls, dense, **kwargs):
        """Unsupported: JITC connectivity cannot be recovered from a dense matrix.

        Raises
        ------
        UnsupportedOperationError
            Always. The generating ``(prob, seed)`` cannot be inferred from a
            materialised matrix.
        """
        raise UnsupportedOperationError(
            f"{cls.__name__}.fromdense is unsupported: just-in-time connectivity "
            "is generated procedurally and cannot recover the (prob, seed) that "
            "produced a dense matrix."
        )

    def dt2t(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Unsupported fallback for JITC subclasses without direct ``dt2t``.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the source vector dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result for subclasses that implement this protocol.

        Raises
        ------
        UnsupportedOperationError
            If the subclass does not override this method. Materialise first
            with ``mat.tocsr().dt2t(...)`` or use a subclass with a direct
            ``dt2t`` implementation.
        """
        raise UnsupportedOperationError(
            f"{type(self).__name__} does not implement direct dt2t for "
            "procedurally generated connectivity. Materialise first with "
            "mat.tocsr().dt2t(y, w), or implement a distribution-specific "
            "dt2t override."
        )

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Unsupported fallback for JITC subclasses without direct transposed ``dt2t``.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the transposed vector dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result for subclasses that implement this protocol.

        Raises
        ------
        UnsupportedOperationError
            If the subclass does not override this method. Materialise first
            with ``mat.tocsr().dt2t_transposed(...)`` or use a subclass with
            a direct ``dt2t_transposed`` implementation.
        """
        raise UnsupportedOperationError(
            f"{type(self).__name__} does not implement direct "
            "dt2t_transposed for procedurally generated connectivity. "
            "Materialise first with mat.tocsr().dt2t_transposed(y, w), or "
            "implement a distribution-specific dt2t_transposed override."
        )

    def update_on_pre(self, *args, **kwargs):
        """Unsupported: JITC connectivity has no per-synapse plastic weight.

        Raises
        ------
        UnsupportedOperationError
            Always. Materialise first with ``mat.tocsr().update_on_pre(...)``.
        """
        raise UnsupportedOperationError(
            "JITC connectivity has no per-synapse plastic weight; the STDP "
            "update_on_pre protocol is undefined for procedurally generated "
            "connectivity. Materialise first: mat.tocsr().update_on_pre(...)."
        )

    def update_on_post(self, *args, **kwargs):
        """Unsupported: JITC connectivity has no per-synapse plastic weight.

        Raises
        ------
        UnsupportedOperationError
            Always. Materialise first with ``mat.tocsr().update_on_post(...)``.
        """
        raise UnsupportedOperationError(
            "JITC connectivity has no per-synapse plastic weight; the STDP "
            "update_on_post protocol is undefined for procedurally generated "
            "connectivity. Materialise first: mat.tocsr().update_on_post(...)."
        )

    def tocsc(self):
        """Convert to CSC by materialising through :meth:`tocsr`.

        Returns
        -------
        CSC
            The same logical matrix in CSC format. Eager-only (``O(nnz)``),
            inheriting the tracing restriction of :meth:`tocsr`.

        See Also
        --------
        tocsr : Direct count+fill materialisation to CSR.
        tocoo : Convert to coordinate format.
        """
        return self.tocsr().tocsc()

    def tocoo(self):
        """Convert to COO by materialising through :meth:`tocsr`.

        Returns
        -------
        brainunit.sparse.COO
            The same logical matrix in COO format. Eager-only (``O(nnz)``),
            inheriting the tracing restriction of :meth:`tocsr`.

        See Also
        --------
        tocsr : Direct count+fill materialisation to CSR.
        tocsc : Convert to compressed sparse column format.
        """
        return self.tocsr().tocoo()


def _initialize_seed(seed=None):
    """Initialize a random seed for JAX operations.

    This function ensures a consistent format for random seeds used in JAX operations.
    If no seed is provided, it generates a random integer between 0 and 10^8 at compile time,
    ensuring reproducibility within compiled functions.

    Parameters
    ----------
    seed : int or array-like, optional
        The random seed to use. If None, a random seed is generated.

    Returns
    -------
    jax.Array
        A JAX array containing the seed value(s) with int32 dtype, ensuring it's
        in a format compatible with JAX random operations.

    Notes
    -----
    The function uses `jax.ensure_compile_time_eval()` to guarantee that random
    seed generation happens during compilation rather than during execution when
    no seed is provided, which helps maintain consistency across multiple calls
    to a JIT-compiled function.
    """
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), (1,))
    return jnp.asarray(jnp.atleast_1d(seed), dtype=jnp.int32)


# ``clen[1]`` value meaning "keep every walked candidate" (24-bit keep hash).
JITC_KEEP_ONE = 1 << 24
# Upper bound on the walk stride ``clen[0]``: keeps ``lane + 32 * q`` inside
# ``uint32`` in every kernel. Sparser probabilities are reached by thinning.
JITC_CLEN_MAX = 1 << 25


def _validate_conn_prob(conn_prob) -> None:
    """Raise ``ValueError`` unless a concrete ``conn_prob`` is a finite scalar in ``[0, 1]``."""
    if isinstance(conn_prob, jax.core.Tracer):
        return
    prob = np.asarray(conn_prob)
    if prob.size != 1:
        raise ValueError(f"prob must be a scalar, but got shape {prob.shape}.")
    prob = float(prob.reshape(()))
    if not np.isfinite(prob):
        raise ValueError(f"prob must be finite, but got {prob}.")
    if not (0. <= prob <= 1.):
        raise ValueError(f"prob must be in [0, 1], but got {prob}.")


def _check_prob_argument(arguments: dict) -> None:
    """``namescope`` precheck: validate a concrete ``prob`` before it is traced."""
    if 'prob' in arguments:
        _validate_conn_prob(arguments['prob'])


def _initialize_conn_length(conn_prob: float):
    """
    Convert a connection probability into the walk parameters ``clen = (cl, thr)``.

    The JIT connectivity kernels walk each row with skips drawn uniformly from
    ``{1, ..., cl - 1}``, which visits a fraction ``2 / cl`` of the positions,
    and keep a visited position ``(row, j)`` only when a stateless 24-bit hash
    of it is below ``thr``. Together this gives density ``p`` exactly (up to the
    ``2**-24`` resolution of ``thr``):

    - when ``2 / p`` is an integer (``p = 1, 0.5, 0.1, 0.01, 1/3, ...``),
      ``cl = 2 / p`` and ``thr = 2**24`` (keep all), so the walk is the one used
      before thinning was introduced;
    - otherwise ``cl = floor(2 / p)`` (density ``>= p``) and
      ``thr = round(p * cl / 2 * 2**24)`` thins it down to ``p``;
    - ``p == 0`` keeps nothing.

    Parameters
    ----------
    conn_prob : float or jax.Array
        Connection probability in ``[0, 1]``. A concrete value is validated; a
        traced value is not (out-of-range values clip to an empty or a full
        matrix, NaN gives an empty one).

    Returns
    -------
    jax.Array
        ``int32`` array of shape ``(2,)`` holding ``(cl, thr)``.

    Raises
    ------
    ValueError
        If a concrete ``conn_prob`` is not a finite scalar in ``[0, 1]``.

    Notes
    -----
    The arithmetic is done in ``float32`` with the same ``jax.numpy`` operations
    whether ``conn_prob`` is concrete or traced, so a matrix passed into
    ``jax.jit`` (where ``prob`` becomes a tracer) draws exactly the same
    connectivity as outside it, with or without ``jax_enable_x64``.
    """
    _validate_conn_prob(conn_prob)
    with jax.ensure_compile_time_eval():
        p = jnp.reshape(jnp.asarray(conn_prob, dtype=jnp.float32), ())
        positive = p > 0.  # False for NaN
        r = 2. / jnp.where(positive, p, 1.)
        n = jnp.round(r)
        is_int = jnp.abs(r - n) <= 1e-6 * r
        cl = jnp.clip(jnp.where(is_int, n, jnp.floor(r)), 2., float(JITC_CLEN_MAX))
        exact = is_int & (cl == n)
        thr = jnp.where(
            exact, float(JITC_KEEP_ONE), jnp.clip(jnp.round(p * cl * float(1 << 23)), 0., float(JITC_KEEP_ONE))
        )
        cl = jnp.where(positive, cl, float(JITC_CLEN_MAX))
        thr = jnp.where(positive, thr, 0.)
        clen = jnp.stack([cl, thr]).astype(jnp.int32)
    return clen
