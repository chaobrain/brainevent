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
from typing import Union, Sequence, Optional, Dict

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp

from brainevent._error import UnsupportedOperationError

__all__ = [
    'DataRepresentation',
    'JITCMatrix',
]


class DataRepresentation(u.sparse.SparseMatrix):
    def __init__(self, *args, shape: Sequence[int], buffers: Optional[Dict] = None):
        super().__init__(*args, shape=shape)
        self._buffer_registry = set()
        if buffers is not None:
            assert isinstance(buffers, dict), "buffers must be a dictionary of name-value pairs."
            self._buffer_registry.update(buffers.keys())
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

    # ------------------------------------------------------------------ #
    # Common-API contract
    #
    # Every operation meaningful for a generic sparse weight matrix is declared
    # here, even where a particular family cannot support it. Subclasses either
    # override a method or *deliberately refuse* it by raising
    # :class:`~brainevent.UnsupportedOperationError`. The structure methods
    # ``todense``, ``with_data``, ``transpose``/``T`` and ``yw_to_w`` are part of
    # the same contract but already declared by ``saiunit.sparse.SparseMatrix``.
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

    def yw_to_w_transposed(self, y_dim_arr, w_dim_arr):
        """Per-synapse ``w * y`` with ``y`` indexed by the column (post) of ``W``.

        Adjoint counterpart of :meth:`yw_to_w`. Part of the per-synapse
        eligibility protocol used by ``brainscale``.

        Parameters
        ----------
        y_dim_arr : jax.Array or brainunit.Quantity
            Post-synaptic (column) vector, sized ``shape[1]``.
        w_dim_arr : jax.Array or brainunit.Quantity
            Per-synapse weights. Some formats (e.g. fixed-num connections)
            accept ``None`` here, defaulting to the representation's own values.

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
        yw_to_w : ``y`` indexed by the row (pre) of ``W``.
        """
        raise NotImplementedError(f"{type(self).__name__}.yw_to_w_transposed")

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
    # non-plastic: the "weight" is a scalar / distribution parameter, not a
    # per-synapse array. Several contract methods are therefore deliberately
    # refused; the conversions that *are* meaningful materialise through
    # :meth:`tocsr` (an eager, ``O(nnz)`` count+fill defined per distribution).
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

    def yw_to_w(self, *args, **kwargs):
        """Unsupported: JITC weights are not per-synapse.

        Raises
        ------
        UnsupportedOperationError
            Always. Materialise first with ``mat.tocsr().yw_to_w(...)``.
        """
        raise UnsupportedOperationError(
            "JITC weights are a scalar / distribution parameter, not a "
            "per-synapse array, so the yw_to_w eligibility protocol is undefined "
            "for procedurally generated connectivity. Materialise first: "
            "mat.tocsr().yw_to_w(y, w)."
        )

    def yw_to_w_transposed(self, *args, **kwargs):
        """Unsupported: JITC weights are not per-synapse.

        Raises
        ------
        UnsupportedOperationError
            Always. Materialise first with ``mat.tocsr().yw_to_w_transposed(...)``.
        """
        raise UnsupportedOperationError(
            "JITC weights are a scalar / distribution parameter, not a "
            "per-synapse array, so the yw_to_w_transposed eligibility protocol is "
            "undefined for procedurally generated connectivity. Materialise "
            "first: mat.tocsr().yw_to_w_transposed(y, w)."
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


def _initialize_conn_length(conn_prob: float):
    """
    Convert connection probability to connection length parameter for sparse matrix generation.

    This function transforms a connection probability (proportion of non-zero entries)
    into a connection length parameter used by the sparse sampling algorithms.
    The connection length is approximately the inverse of the connection probability,
    scaled by a factor of 2 to ensure adequate sparsity in the generated matrices.

    The function ensures the calculation happens at compile time when used in JIT-compiled
    functions by using JAX's compile_time_eval context.

    Parameters
    ----------
    conn_prob : float
        The connection probability (between 0 and 1) representing the fraction
        of non-zero entries in the randomly generated matrix.

    Returns
    -------
    jax.Array
        A JAX array containing the connection length value as an int32,
        which is approximately 2/conn_prob.

    Notes
    -----
    The connection length parameter is used in the kernels to determine the
    average distance between sampled connections when generating sparse matrices.
    Larger values result in sparser matrices (fewer connections).
    """
    with jax.ensure_compile_time_eval():
        clen = jnp.ceil(2 / conn_prob)
        clen = jnp.asarray(clen, dtype=jnp.int32)
    return clen
