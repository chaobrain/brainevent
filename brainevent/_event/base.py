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


from abc import ABC, abstractmethod
from typing import Self, Union

import brainunit as u
import jax
import numpy as np
from jax.tree_util import register_pytree_node_class

__all__ = [
    'EventRepresentation',
    'IndexedEventRepresentation',
]

ArrayValue = Union[jax.Array, u.Quantity]
ArrayLike = Union['EventRepresentation', jax.Array, np.ndarray, u.Quantity, list, tuple, int, float, bool]


def extract_raw_value(obj):
    """Extract the underlying array value from an EventRepresentation or pass through.

    Parameters
    ----------
    obj : EventRepresentation or array_like
        If an ``EventRepresentation`` instance, its ``.value`` attribute is
        returned.  Otherwise *obj* is returned unchanged.

    Returns
    -------
    jax.Array or brainunit.Quantity or array_like
        The unwrapped array value.
    """
    return obj.value if isinstance(obj, EventRepresentation) else obj


def is_known_type(x):
    """Check whether *x* is a recognised array-like type.

    Parameters
    ----------
    x : object
        The object to test.

    Returns
    -------
    bool
        ``True`` if *x* is an instance of ``brainunit.Quantity``,
        ``jax.Array``, ``numpy.ndarray``, or ``EventRepresentation``.
    """
    return isinstance(x, (u.Quantity, jax.Array, np.ndarray, EventRepresentation))


def _normalize_index(index):
    if isinstance(index, tuple):
        return tuple(_normalize_index(x) for x in index)
    return extract_raw_value(index)


@register_pytree_node_class
class EventRepresentation(ABC):
    """Abstract base class for event-driven array representations.

    ``EventRepresentation`` wraps an underlying JAX array (or ``brainunit.Quantity``)
    and exposes array-like properties (``shape``, ``ndim``, ``dtype``, ``size``) while
    requiring subclasses to implement the ``@`` (matrix multiplication) operator via
    ``__matmul__`` and ``__rmatmul__``.

    The class is registered as a JAX PyTree node so that instances are compatible
    with ``jax.jit``, ``jax.grad``, and other JAX transformations.

    Parameters
    ----------
    value : array_like
        The underlying array data.  Accepted types include ``jax.Array``,
        ``numpy.ndarray``, ``brainunit.Quantity``, Python lists/tuples, or
        another ``EventRepresentation`` (whose inner value will be extracted).

    Notes
    -----
    Event-driven computation exploits the sparsity of neural spike
    vectors.  Given a spike vector ``s`` and a weight matrix ``W``, the
    standard dense product ``y = W @ s`` visits every element of ``W``.
    An event-driven implementation only accumulates columns of ``W``
    where ``s`` is non-zero:

        ``y[i] = sum_{j : s[j] != 0} W[i, j] * s[j]``

    For binary events (``BinaryArray``), ``s[j]`` is either 0 or 1, so
    the multiplication by ``s[j]`` is omitted entirely.

    Subclasses must implement ``__matmul__`` and ``__rmatmul__`` to
    define the event-driven ``@`` operator.

    The ``__array_priority__`` is set to 100 so that NumPy/JAX will
    defer to the operators defined on this class when a standard array
    appears on the left-hand side of a binary operation.

    See Also
    --------
    BinaryArray : Concrete subclass for binary (0/1) event arrays.
    SparseFloat : Concrete subclass for sparse floating-point event arrays.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainevent as be
        >>> arr = be.BinaryArray(jnp.array([True, False, True]))
        >>> arr.shape
        (3,)
        >>> arr.dtype
        dtype('bool')
    """

    __slots__ = ('_value',)
    __array_priority__ = 100
    __module__ = 'brainevent'

    def __init__(self, value: ArrayLike):
        value = extract_raw_value(value)
        if isinstance(value, (list, tuple, np.ndarray)):
            value = u.math.asarray(value)
        self._value = value

    @property
    def value(self) -> ArrayValue:
        """The underlying array data.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The raw array stored by this event representation.
        """
        return self._value

    @value.setter
    def value(self, val) -> None:
        """Set the underlying array data.

        Parameters
        ----------
        val : array_like
            New array value to store.
        """
        self._value = val

    def with_value(self, value: ArrayLike) -> Self:
        """Create a new instance of the same type with a different value.

        Parameters
        ----------
        value : array_like
            The new underlying array data.

        Returns
        -------
        Self
            A fresh instance of the same concrete class wrapping *value*.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent as be
            >>> a = be.BinaryArray(jnp.array([True, False]))
            >>> b = a.with_value(jnp.array([False, True]))
            >>> b.value
            Array([False,  True], dtype=bool)
        """
        return type(self)(value)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array.

        Returns
        -------
        tuple of int
            Dimension sizes, e.g. ``(n,)`` for a 1-D vector or ``(m, n)``
            for a 2-D matrix.
        """
        return tuple(self._value.shape)

    @property
    def ndim(self) -> int:
        """Number of array dimensions.

        Returns
        -------
        int
            The number of axes, e.g. 1 for a vector, 2 for a matrix.
        """
        return self._value.ndim

    @property
    def dtype(self):
        """Data type of the underlying array elements.

        Returns
        -------
        numpy.dtype
            The element data type (e.g. ``jnp.float32``, ``jnp.bool_``).
        """
        return self._value.dtype

    @property
    def size(self) -> int:
        """Total number of elements in the underlying array.

        Returns
        -------
        int
            Product of all dimension sizes.
        """
        return int(self._value.size)

    def __repr__(self) -> str:
        """Return a string representation of the event array.

        Returns
        -------
        str
            A string of the form ``ClassName(value=..., dtype=...)``.
        """
        return f"{type(self).__name__}(value={self._value}, dtype={self.dtype})"

    def __len__(self) -> int:
        """Return the size of the first dimension.

        Returns
        -------
        int
            Length of the leading axis.

        Raises
        ------
        TypeError
            If the underlying array is 0-dimensional.
        """
        return len(self._value)

    def __iter__(self):
        """Iterate over the first axis of the underlying array.

        Yields
        ------
        jax.Array or brainunit.Quantity
            Successive slices along axis 0.
        """
        for i in range(self._value.shape[0]):
            yield self._value[i]

    def __getitem__(self, index):
        """Index into the underlying array.

        Parameters
        ----------
        index : int, slice, or tuple
            Standard NumPy-style index.  If *index* contains
            ``EventRepresentation`` instances they are automatically
            unwrapped to their raw values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The selected sub-array.
        """
        return self._value[_normalize_index(index)]

    def tree_flatten(self):
        """Flatten this instance for JAX PyTree serialisation.

        Returns
        -------
        children : tuple
            A single-element tuple ``(value,)`` containing the dynamic
            array leaf.
        aux_data : dict
            An empty dictionary (subclasses may add static metadata here).
        """
        return (self._value,), {}

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """Reconstruct an instance from its PyTree representation.

        Parameters
        ----------
        aux_data : dict
            Static metadata produced by ``tree_flatten``.
        flat_contents : tuple
            Dynamic leaves, i.e. the underlying array.

        Returns
        -------
        EventRepresentation
            A new instance of the concrete subclass.
        """
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj

    @abstractmethod
    def __matmul__(self, other):
        """Compute ``self @ other`` (matrix multiplication).

        Subclasses must implement this to define event-driven sparse
        matrix multiplication semantics.

        Parameters
        ----------
        other : array_like
            Right-hand operand.

        Returns
        -------
        jax.Array
            Result of the matrix multiplication.
        """
        pass

    @abstractmethod
    def __rmatmul__(self, other):
        """Compute ``other @ self`` (reverse matrix multiplication).

        Subclasses must implement this to define event-driven sparse
        matrix multiplication semantics when this instance appears on the
        right-hand side of the ``@`` operator.

        Parameters
        ----------
        other : array_like
            Left-hand operand.

        Returns
        -------
        jax.Array
            Result of the matrix multiplication.
        """
        pass


class IndexedEventRepresentation(EventRepresentation):
    """Event representation with pre-computed indices of active elements.

    ``IndexedEventRepresentation`` extends ``EventRepresentation`` by
    scanning the input array at construction time to extract and store
    the positions (``spike_indices``) and count (``spike_count``) of all
    non-zero elements.  Downstream kernels can then iterate only over
    the active subset rather than the full array.

    Subclasses (e.g. ``IndexedBinary1d``, ``IndexedSpFloat1d``) populate
    these fields via backend-specific extraction kernels.

    Notes
    -----
    Given an input vector ``s`` of length ``n``, the extraction computes:

        ``spike_indices = {j : s[j] != 0}``
        ``spike_count   = |spike_indices|``

    The ``spike_indices`` array is always allocated with length ``n``;
    only the first ``spike_count[0]`` entries are valid, and the
    remaining entries are zero-filled.

    Pre-computing the active set avoids redundant branching inside
    tight GPU/CPU loops and enables coalesced memory access patterns
    when multiplying with weight matrices.

    See Also
    --------
    EventRepresentation : Parent class without pre-computed indices.
    IndexedBinary1d : Concrete 1-D indexed binary event array.
    IndexedSpFloat1d : Concrete 1-D indexed sparse-float event array.
    """
    __slots__ = ('_value',)
    __array_priority__ = 100
    __module__ = 'brainevent'
