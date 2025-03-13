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

import operator
from typing import Union, Optional, Sequence, Any

import brainunit as u
import jax
import numpy as np
from jax.dtypes import canonicalize_dtype
from jax.tree_util import register_pytree_node_class

from ._error import MathError

__all__ = [
    'EventArray',
]


def _get_dtype(v):
    if hasattr(v, 'dtype'):
        dtype = v.dtype
    else:
        dtype = canonicalize_dtype(type(v))
    return dtype


def _check_out(out):
    if not isinstance(out, EventArray):
        raise TypeError(f'out must be an instance of Array. But got {type(out)}')


def _as_array(obj):
    return obj.value if isinstance(obj, EventArray) else obj


def _known_type(x):
    return isinstance(x, (u.Quantity, jax.Array, np.ndarray))


ArrayLike = Union[jax.Array, np.ndarray, u.Quantity]


@register_pytree_node_class
class EventArray(object):
    """
    The base array class for representing events.
    """
    __slots__ = ('_value',)

    def __init__(self, value, dtype: Any = None):
        """
        Initialize an EventArray instance.

        Args:
            value: The input value, which can be an EventArray, tuple, list, or np.ndarray.
            dtype: The data type of the array. If None, the data type will be inferred from the input value.
        """
        # array value
        if isinstance(value, EventArray):
            value = value._value
        elif isinstance(value, (tuple, list, np.ndarray)):
            value = u.math.asarray(value)
        if dtype is not None:
            value = u.math.asarray(value, dtype=dtype)
        self._value = value

    def _check_tracer(self):
        """
        Check the tracer of the array value.

        Returns:
            The array value.
        """
        return self._value

    @property
    def data(self):
        """
        Get the array value.

        Returns:
            The array value.
        """
        return self._value

    @property
    def value(self):
        """
        Return the value of the array.

        Returns:
            The array value.
        """
        return self._value

    @value.setter
    def value(self, value):
        """
        Set the value of the array.

        Args:
            value: The new value to be set.

        Raises:
            MathError: If the shape or dtype of the new value does not match the original value.
        """
        # Get the current value for comparison with the new value
        self_value = self._check_tracer()

        # Handle different types of incoming values
        if isinstance(value, EventArray):
            value = value.value
        elif isinstance(value, np.ndarray):
            value = u.math.asarray(value)
        elif isinstance(value, jax.Array):
            pass
        else:
            value = u.math.asarray(value)

        # check
        # Check if the shape of the new value matches the original value
        if value.shape != self_value.shape:
            raise MathError(
                f"The shape of the original data is {self_value.shape}, "
                f"while we got {value.shape}."
            )

        # Check if the dtype of the new value matches the original value
        if value.dtype != self_value.dtype:
            raise MathError(
                f"The dtype of the original data is {self_value.dtype}, "
                f"while we got {value.dtype}."
            )

        # Set the new value after passing the check
        self._value = value

    def update(self, value):
        """
        Update the value of this EventArray.

        This method updates the internal value of the EventArray with a new value.

        Parameters
        ----------
        value : array-like
            The new value to update the EventArray with. This should be compatible
            with the current array in terms of shape and dtype.

        Returns
        -------
        None
            This method modifies the EventArray in-place and doesn't return anything.

        Raises
        ------
        MathError
            If the shape or dtype of the new value does not match the original value.
        """
        self.value = value

    def ndim(self):
        """Return the number of dimensions (rank) of the array.

        This property indicates the number of axes in the array. For example:
        - A scalar has ndim = 0
        - A vector has ndim = 1
        - A matrix has ndim = 2
        - And so on for higher dimensional arrays

        Returns:
            int: The number of dimensions of the array.
        """
        return self.value.ndim

    def dtype(self):
        """Return the data type of the array's elements.

        This property accesses the data type (dtype) of the underlying array.
        For arrays containing numbers, this is their precision (e.g., float32,
        int64, etc.).

        Returns:
            dtype: The data type of the array's elements.
        """
        return _get_dtype(self._value)

    def shape(self):
        """Return the dimensions of the array as a tuple.

        This property returns the shape of the underlying array, which indicates
        the size of each dimension. For example, a 3x4 matrix would have shape (3, 4),
        while a 1D array with 5 elements would have shape (5,).

        Returns:
            tuple: A tuple of integers indicating the size of each dimension.
        """
        return u.math.shape(self.value)

    @property
    def imag(self):
        """
        Get the imaginary part of the array.

        Returns:
            The imaginary part of the array.
        """
        return u.math.imag(self.value)

    @property
    def real(self):
        """
        Get the real part of the array.

        Returns:
            The real part of the array.
        """
        return self.value.real

    @property
    def size(self):
        """
        Get the number of elements in the array.

        Returns:
            The number of elements.
        """
        return self.value.size

    @property
    def T(self):
        """
        Get the transpose of the array.

        Returns:
            The transpose of the array.
        """
        return self.value.T

    # ----------------------- #
    # Python inherent methods #
    # ----------------------- #

    def __repr__(self) -> str:
        """
        Return a string representation of the EventArray.

        Returns:
            A string representation of the EventArray.
        """
        print_code = repr(self.value)
        if ', dtype' in print_code:
            print_code = print_code.split(', dtype')[0] + ')'
        prefix = f'{self.__class__.__name__}'
        prefix2 = f'{self.__class__.__name__}(value='
        if '\n' in print_code:
            lines = print_code.split("\n")
            blank1 = " " * len(prefix2)
            lines[0] = prefix2 + lines[0]
            for i in range(1, len(lines)):
                lines[i] = blank1 + lines[i]
            lines[-1] += ","
            blank2 = " " * (len(prefix) + 1)
            lines.append(f'{blank2}dtype={self.dtype})')
            print_code = "\n".join(lines)
        else:
            print_code = prefix2 + print_code + f', dtype={self.dtype})'
        return print_code

    def __iter__(self):
        """Solve the issue of DeviceArray.__iter__.

        Details please see JAX issues:

        - https://github.com/google/jax/issues/7713
        - https://github.com/google/jax/pull/3821
        """
        for i in range(self.value.shape[0]):
            yield self.value[i]

    def __getitem__(self, index):
        """
        Get an item from the array.

        Args:
            index: The index of the item to get.

        Returns:
            The item at the specified index.
        """
        if isinstance(index, tuple):
            index = tuple((x.value if isinstance(x, EventArray) else x) for x in index)
        elif isinstance(index, EventArray):
            index = index.value
        return self.value[index]

    def __setitem__(self, index, value):
        """
        Set an item in the array.

        Args:
            index: The index of the item to set.
            value: The new value to be set.
        """
        # value is Array
        if isinstance(value, EventArray):
            value = value.value
        # value is numpy.ndarray
        elif isinstance(value, np.ndarray):
            value = u.math.asarray(value)

        # index is a tuple
        if isinstance(index, tuple):
            index = tuple(_as_array(x) for x in index)
        # index is Array
        elif isinstance(index, EventArray):
            index = index.value
        # index is numpy.ndarray
        elif isinstance(index, np.ndarray):
            index = u.math.asarray(index)

        # update
        self_value = self._check_tracer()
        self.value = self_value.at[index].set(value)

    # ---------- #
    # operations #
    # ---------- #

    def __len__(self) -> int:
        """
        Get the length of the array.

        Returns:
            The length of the array.
        """
        return len(self.value)

    def __neg__(self):
        """
        Return the negative of the array.

        Returns:
            The negative of the array.
        """
        return self.value.__neg__()

    def __pos__(self):
        """
        Return the positive of the array.

        Returns:
            The positive of the array.
        """
        return self.value.__pos__()

    def __abs__(self):
        """
        Return the absolute value of the array.

        Returns:
            The absolute value of the array.
        """
        return self.value.__abs__()

    def __invert__(self):
        """
        Return the bitwise inversion of the array.

        Returns:
            The bitwise inversion of the array.
        """
        return self.value.__invert__()

    def __eq__(self, oc):
        """
        Compare the array with another object for equality.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the equality.
        """
        return self.value == _as_array(oc)

    def __ne__(self, oc):
        """
        Compare the array with another object for inequality.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the inequality.
        """
        return self.value != _as_array(oc)

    def __lt__(self, oc):
        """
        Compare the array with another object for less than.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the comparison result.
        """
        return self.value < _as_array(oc)

    def __le__(self, oc):
        """
        Compare the array with another object for less than or equal to.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the comparison result.
        """
        return self.value <= _as_array(oc)

    def __gt__(self, oc):
        """
        Compare the array with another object for greater than.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the comparison result.
        """
        return self.value > _as_array(oc)

    def __ge__(self, oc):
        """
        Compare the array with another object for greater than or equal to.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the comparison result.
        """
        return self.value >= _as_array(oc)

    def __add__(self, oc):
        """
        Add the array with another object.

        Args:
            oc: The object to add.

        Returns:
            The result of the addition.
        """
        return self.value + _as_array(oc)

    def __radd__(self, oc):
        """
        Add another object with the array.

        Args:
            oc: The object to add.

        Returns:
            The result of the addition.
        """
        return self.value + _as_array(oc)

    def __iadd__(self, oc):
        """
        Add another object to the array in-place.

        Args:
            oc: The object to add.

        Returns:
            The updated array.
        """
        # a += b
        self.value = self.value + _as_array(oc)
        return self

    def __sub__(self, oc):
        """
        Subtract another object from the array.

        Args:
            oc: The object to subtract.

        Returns:
            The result of the subtraction.
        """
        return self.value - _as_array(oc)

    def __rsub__(self, oc):
        """
        Subtract the array from another object.

        Args:
            oc: The object to subtract from.

        Returns:
            The result of the subtraction.
        """
        return _as_array(oc) - self.value

    def __isub__(self, oc):
        """
        Subtract another object from the array in-place.

        Args:
            oc: The object to subtract.

        Returns:
            The updated array.
        """
        # a -= b
        self.value = self.value - _as_array(oc)
        return self

    def __mul__(self, oc):
        """
        Multiply the array with another object.

        Args:
            oc: The object to multiply.

        Returns:
            The result of the multiplication.
        """
        return self.value * _as_array(oc)

    def __rmul__(self, oc):
        """
        Multiply another object with the array.

        Args:
            oc: The object to multiply.

        Returns:
            The result of the multiplication.
        """
        return _as_array(oc) * self.value

    def __imul__(self, oc):
        """
        Multiply the array with another object in-place.

        Args:
            oc: The object to multiply.

        Returns:
            The updated array.
        """
        # a *= b
        self.value = self.value * _as_array(oc)
        return self

    def __rdiv__(self, oc):
        """
        Divide another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the division.
        """
        return _as_array(oc) / self.value

    def __truediv__(self, oc):
        """
        Divide the array by another object.

        Args:
            oc: The object to divide by.

        Returns:
            The result of the division.
        """
        return self.value / _as_array(oc)

    def __rtruediv__(self, oc):
        """
        Divide another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the division.
        """
        return _as_array(oc) / self.value

    def __itruediv__(self, oc):
        """
        Divide the array by another object in-place.

        Args:
            oc: The object to divide by.

        Returns:
            The updated array.
        """
        # a /= b
        self.value = self.value / _as_array(oc)
        return self

    def __floordiv__(self, oc):
        """
        Perform floor division on the array by another object.

        Args:
            oc: The object to divide by.

        Returns:
            The result of the floor division.
        """
        return self.value // _as_array(oc)

    def __rfloordiv__(self, oc):
        """
        Perform floor division on another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the floor division.
        """
        return _as_array(oc) // self.value

    def __ifloordiv__(self, oc):
        """
        Perform floor division on the array by another object in-place.

        Args:
            oc: The object to divide by.

        Returns:
            The updated array.
        """
        # a //= b
        self.value = self.value // _as_array(oc)
        return self

    def __divmod__(self, oc):
        """
        Perform divmod operation on the array by another object.

        Args:
            oc: The object to divide by.

        Returns:
            The result of the divmod operation.
        """
        return self.value.__divmod__(_as_array(oc))

    def __rdivmod__(self, oc):
        """
        Perform divmod operation on another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the divmod operation.
        """
        return self.value.__rdivmod__(_as_array(oc))

    def __mod__(self, oc):
        """
        Perform modulo operation on the array by another object.

        Args:
            oc: The object to divide by.

        Returns:
            The result of the modulo operation.
        """
        return self.value % _as_array(oc)

    def __rmod__(self, oc):
        """
        Perform modulo operation on another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the modulo operation.
        """
        return _as_array(oc) % self.value

    def __imod__(self, oc):
        """
        Perform modulo operation on the array by another object in-place.

        Args:
            oc: The object to divide by.

        Returns:
            The updated array.
        """
        # a %= b
        self.value = self.value % _as_array(oc)
        return self

    def __pow__(self, oc):
        """
        Raise the array to the power of another object.

        Args:
            oc: The object to raise to the power.

        Returns:
            The result of the power operation.
        """
        return self.value ** _as_array(oc)

    def __rpow__(self, oc):
        """
        Raise another object to the power of the array.

        Args:
            oc: The object to raise to the power.

        Returns:
            The result of the power operation.
        """
        return _as_array(oc) ** self.value

    def __ipow__(self, oc):
        """
        Raise the array to the power of another object in-place.

        Args:
            oc: The object to raise to the power.

        Returns:
            The updated array.
        """
        # a **= b
        self.value = self.value ** _as_array(oc)
        return self

    def __matmul__(self, oc):
        """
        Perform matrix multiplication on the array with another object.

        Args:
            oc: The object to multiply.

        Returns:
            The result of the matrix multiplication.
        """
        if _known_type(oc):
            return self.value @ _as_array(oc)
        else:
            return oc.__rmatmul__(self)

    def __rmatmul__(self, oc):
        """
        Perform matrix multiplication on another object with the array.

        Args:
            oc: The object to multiply.

        Returns:
            The result of the matrix multiplication.
        """
        if _known_type(oc):
            return _as_array(oc) @ self.value
        else:
            return oc.__matmul__(self)

    def __imatmul__(self, oc):
        """
        Perform matrix multiplication on the array with another object in-place.

        Args:
            oc: The object to multiply.

        Returns:
            The updated array.
        """
        # a @= b
        if _known_type(oc):
            self.value = self.value @ _as_array(oc)
        else:
            self.value = oc.__rmatmul__(self)
        return self

    def __and__(self, oc):
        """
        Perform bitwise AND operation on the array with another object.

        Args:
            oc: The object to perform AND operation with.

        Returns:
            The result of the bitwise AND operation.
        """
        return self.value & _as_array(oc)

    def __rand__(self, oc):
        """
        Perform bitwise AND operation on another object with the array.

        Args:
            oc: The object to perform AND operation with.

        Returns:
            The result of the bitwise AND operation.
        """
        return _as_array(oc) & self.value

    def __iand__(self, oc):
        """
        Perform bitwise AND operation on the array with another object in-place.

        Args:
            oc: The object to perform AND operation with.

        Returns:
            The updated array.
        """
        # a &= b
        self.value = self.value & _as_array(oc)
        return self

    def __or__(self, oc):
        """
        Perform bitwise OR operation on the array with another object.

        Args:
            oc: The object to perform OR operation with.

        Returns:
            The result of the bitwise OR operation.
        """
        return self.value | _as_array(oc)

    def __ror__(self, oc):
        """
        Perform bitwise OR operation on another object with the array.

        Args:
            oc: The object to perform OR operation with.

        Returns:
            The result of the bitwise OR operation.
        """
        return _as_array(oc) | self.value

    def __ior__(self, oc):
        """
        Perform bitwise OR operation on the array with another object in-place.

        Args:
            oc: The object to perform OR operation with.

        Returns:
            The updated array.
        """
        # a |= b
        self.value = self.value | _as_array(oc)
        return self

    def __xor__(self, oc):
        """
        Perform bitwise XOR operation on the array with another object.

        Args:
            oc: The object to perform XOR operation with.

        Returns:
            The result of the bitwise XOR operation.
        """
        return self.value ^ _as_array(oc)

    def __rxor__(self, oc):
        """
        Perform bitwise XOR operation on another object with the array.

        Args:
            oc: The object to perform XOR operation with.

        Returns:
            The result of the bitwise XOR operation.
        """
        return _as_array(oc) ^ self.value

    def __ixor__(self, oc):
        """
        Perform bitwise XOR operation on the array with another object in-place.

        Args:
            oc: The object to perform XOR operation with.

        Returns:
            The updated array.
        """
        # a ^= b
        self.value = self.value ^ _as_array(oc)
        return self

    def __lshift__(self, oc):
        """
        Perform left shift operation on the array by another object.

        Args:
            oc: The object to shift by.

        Returns:
            The result of the left shift operation.
        """
        return self.value << _as_array(oc)

    def __rlshift__(self, oc):
        """
        Perform left shift operation on another object by the array.

        Args:
            oc: The object to shift.

        Returns:
            The result of the left shift operation.
        """
        return _as_array(oc) << self.value

    def __ilshift__(self, oc):
        """
        Perform left shift operation on the array by another object in-place.

        Args:
            oc: The object to shift by.

        Returns:
            The updated array.
        """
        # a <<= b
        self.value = self.value << _as_array(oc)
        return self

    def __rshift__(self, oc):
        """
        Perform right shift operation on the array by another object.

        Args:
            oc: The object to shift by.

        Returns:
            The result of the right shift operation.
        """
        return self.value >> _as_array(oc)

    def __rrshift__(self, oc):
        """
        Perform right shift operation on another object by the array.

        Args:
            oc: The object to shift.

        Returns:
            The result of the right shift operation.
        """
        return _as_array(oc) >> self.value

    def __irshift__(self, oc):
        """
        Perform right shift operation on the array by another object in-place.

        Args:
            oc: The object to shift by.

        Returns:
            The updated array.
        """
        # a >>= b
        self.value = self.value >> _as_array(oc)
        return self

    def __round__(self, ndigits=None):
        """
        Round the array to a specified number of decimal places.

        Args:
            ndigits: The number of decimal places to round to.

        Returns:
            The rounded array.
        """
        return self.value.__round__(ndigits)

    # ----------------------- #
    #       JAX methods       #
    # ----------------------- #

    @property
    def at(self):
        """
        Get the 'at' property of the array.

        Returns:
            The 'at' property of the array.
        """
        return self.value.at

    def block_until_ready(self):
        """
        Block until the array is ready.

        Returns:
            The array after it is ready.
        """
        return self.value.block_until_ready()

    def device(self):
        """
        Get the device on which the array is located.

        Returns:
            The device on which the array is located.
        """
        return self.value.device()

    @property
    def device_buffer(self):
        """
        Get the device buffer of the array.

        Returns:
            The device buffer of the array.
        """
        return self.value.device_buffer

    # ----------------------- #
    #      NumPy methods      #
    # ----------------------- #

    def all(self, axis=None, keepdims=False):
        """Returns True if all elements evaluate to True."""
        r = self.value.all(axis=axis, keepdims=keepdims)
        return r

    def any(self, axis=None, keepdims=False):
        """Returns True if any of the elements of a evaluate to True."""
        r = self.value.any(axis=axis, keepdims=keepdims)
        return r

    def argmax(self, axis=None):
        """Return indices of the maximum values along the given axis."""
        return self.value.argmax(axis=axis)

    def argmin(self, axis=None):
        """Return indices of the minimum values along the given axis."""
        return self.value.argmin(axis=axis)

    def argpartition(self, kth, axis=-1):
        """Returns the indices that would partition this array."""
        return self.value.argpartition(kth=kth, axis=axis)

    def argsort(self, axis=-1, kind=None, order=None):
        """Returns the indices that would sort this array."""
        return self.value.argsort(axis=axis, kind=kind, order=order)

    def astype(self, dtype):
        """Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype: str, dtype
          Typecode or data-type to which the array is cast.
        """
        if dtype is None:
            return self.value
        else:
            return self.value.astype(dtype)

    def byteswap(self, inplace=False):
        """Swap the bytes of the array elements

        Toggle between low-endian and big-endian data representation by
        returning a byteswapped array, optionally swapped in-place.
        Arrays of byte-strings are not swapped. The real and imaginary
        parts of a complex number are swapped individually."""
        return self.value.byteswap(inplace=inplace)

    def choose(self, choices, mode='raise'):
        """
        Use an index array to construct a new array from a set of choices.
    
        This method uses the index array (self) to select elements from the choices array.
    
        Parameters
        ----------
        choices : sequence of arrays
            The arrays from which to choose. Each array in the sequence must be of the same shape as self,
            or broadcastable to that shape.
    
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how indices outside the valid range should be handled:
            - 'raise' : raise an error (default)
            - 'wrap' : wrap around
            - 'clip' : clip to the range
    
        Returns
        -------
        ndarray
            The merged result, with elements chosen from `choices` based on the index array.
    
        Raises
        ------
        ValueError
            If an invalid `mode` is specified.
        """
        return self.value.choose(choices=choices, mode=mode)

    def clip(self, min=None, max=None, out=None):
        """
        Return an array with its values clipped to be within the specified range [min, max].
    
        This method limits the values in the array to be within the given range. Values smaller
        than the minimum are set to the minimum, and values larger than the maximum are set to
        the maximum.
    
        Parameters
        ----------
        min : scalar or array_like, optional
            Minimum value. If None, clipping is not performed on lower interval edge.
            Not more than one of min and max may be None.
        max : scalar or array_like, optional
            Maximum value. If None, clipping is not performed on upper interval edge.
            Not more than one of min and max may be None.
        out : ndarray, optional
            The results will be placed in this array. It may be the input array for in-place clipping.
            Out must be of the right shape to hold the output. Its type is preserved.
    
        Returns
        -------
        ndarray
            An array with the elements of self, but where values < min are replaced with min,
            and those > max with max.
    
        Note
        ----
        At least one of max or min must be given.
        """
        min = _as_array(min)
        max = _as_array(max)
        r = self.value.clip(min=min, max=max)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def compress(self, condition, axis=None):
        """
        Return selected slices of this array along the given axis.

        This method selects elements from the array based on a boolean condition,
        returning a new array with only the elements where the condition is True.

        Parameters
        ----------
        condition : array_like
            A 1-D array of booleans. Where True, the corresponding element in the
            array is selected. The length of the condition array should be the size
            of the array along the given axis.

        axis : int, optional
            The axis along which to select elements. Default is None, which selects
            elements from the flattened array.

        Returns
        -------
        ndarray
            A new array containing the selected elements. The returned array has the
            same number of dimensions as the input array, but the size of the axis
            along which elements were selected may be smaller.
        """
        return self.value.compress(condition=_as_array(condition), axis=axis)

    def conj(self):
        """
        Compute the complex conjugate of all elements in the array.
    
        This method returns a new array with the complex conjugate of each element
        in the original array. For real numbers, this operation has no effect.
    
        Returns
        -------
        ndarray
            A new array with the complex conjugate of each element from the original array.
        """
        return self.value.conj()

    def conjugate(self):
        """
        Compute the complex conjugate of all elements in the array, element-wise.
    
        This method returns a new array with the complex conjugate of each element
        in the original array. For real numbers, this operation has no effect.
        This method is identical to the `conj` method.
    
        Returns
        -------
        ndarray
            A new array with the complex conjugate of each element from the original array.
        """
        return self.value.conjugate()

    def copy(self):
        """
        Return a copy of the array.
    
        This method creates and returns a new array with a copy of the data from the original array.
    
        Returns:
            ndarray: A new array object with a copy of the data from the original array.
        """
        return self.value.copy()

    def cumprod(self, axis=None, dtype=None):
        """
        Return the cumulative product of the elements along the given axis.
    
        Parameters:
            axis (int, optional): Axis along which the cumulative product is computed.
                If None (default), the cumulative product of the flattened array is computed.
            dtype (data-type, optional): Type of the returned array and of the accumulator
                in which the elements are multiplied. If dtype is not specified, it defaults
                to the dtype of the input array.
    
        Returns:
            ndarray: An array of the same shape as the input array, containing the cumulative
            product of the elements along the specified axis.
        """
        return self.value.cumprod(axis=axis, dtype=dtype)

    def cumsum(self, axis=None, dtype=None):
        """
        Return the cumulative sum of the elements along the given axis.
    
        Parameters:
            axis (int, optional): Axis along which the cumulative sum is computed.
                If None (default), the cumulative sum of the flattened array is computed.
            dtype (data-type, optional): Type of the returned array and of the accumulator
                in which the elements are summed. If dtype is not specified, it defaults
                to the dtype of the input array.
    
        Returns:
            ndarray: An array of the same shape as the input array, containing the cumulative
            sum of the elements along the specified axis.
        """
        return self.value.cumsum(axis=axis, dtype=dtype)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """
        Return specified diagonals of the array.
    
        Parameters:
            offset (int, optional): Offset of the diagonal from the main diagonal.
                Can be positive or negative. Defaults to 0 (main diagonal).
            axis1 (int, optional): Axis to be used as the first axis of the 2-D sub-arrays
                from which the diagonals should be taken. Defaults to 0.
            axis2 (int, optional): Axis to be used as the second axis of the 2-D sub-arrays
                from which the diagonals should be taken. Defaults to 1.
    
        Returns:
            ndarray: An array containing the diagonal elements. If the dimension of the input
            array is greater than 2, then the result is a 1-D array if offset is specified,
            otherwise it has the same dimension as the input array minus 2.
        """
        return self.value.diagonal(offset=offset, axis1=axis1, axis2=axis2)

    def dot(self, b):
        """
        Compute the dot product of two arrays.

        This method calculates the dot product (matrix multiplication) of the current array
        with another array or matrix.

        Parameters
        ----------
        b : array_like
            The array or matrix to compute the dot product with.

        Returns
        -------
        ndarray
            The result of the dot product operation. The shape of the output depends
            on the shapes of the input arrays and the nature of the dot product operation.

        Notes
        -----
        If the type of 'b' is known, it uses the dot method of the underlying array.
        Otherwise, it delegates to the right matrix multiplication method of 'b'.
        """
        if _known_type(b):
            return self.value.dot(_as_array(b))
        else:
            return b.__rmatmul__(self)

    def fill(self, value):
        """
        Fill the array with a scalar value.

        This method replaces all elements in the array with the specified scalar value.

        Parameters
        ----------
        value : scalar
            The scalar value to fill the array with. This value will be broadcast
            to fill the entire array.

        Returns
        -------
        None
            This method modifies the array in-place and does not return a value.
        """
        self.value = u.math.ones_like(self.value) * value

    def flatten(self):
        """
        Return a flattened array.

        Returns:
            A flattened array.
        """
        return self.value.flatten()

    def item(self, *args):
        """Copy an element of an array to a standard Python scalar and return it.
    
        Args:
            *args: Index or indices of the element to be extracted. If not provided, the first element is returned.
    
        Returns:
            scalar: The extracted element as a standard Python scalar.
        """
        return self.value.item(*args)

    def max(self, axis=None, keepdims=False, *args, **kwargs):
        """Return the maximum value along a given axis.
    
        Args:
            axis (int or tuple of ints, optional): Axis or axes along which to operate. By default, flattened input is used.
            keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
            *args: Additional positional arguments to be passed to the underlying max function.
            **kwargs: Additional keyword arguments to be passed to the underlying max function.
    
        Returns:
            ndarray or scalar: Maximum of array elements along the given axis.
        """
        res = self.value.max(axis=axis, keepdims=keepdims, *args, **kwargs)
        return res

    def mean(self, axis=None, dtype=None, keepdims=False, *args, **kwargs):
        """Calculate the arithmetic mean along the specified axis.
    
        Args:
            axis (int or tuple of ints, optional): Axis or axes along which the mean is computed. The default is to compute the mean of the flattened array.
            dtype (data-type, optional): Type to use in computing the mean.
            keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
            *args: Additional positional arguments to be passed to the underlying mean function.
            **kwargs: Additional keyword arguments to be passed to the underlying mean function.
    
        Returns:
            ndarray or scalar: Array containing the mean values.
        """
        res = self.value.mean(axis=axis, dtype=dtype, keepdims=keepdims, *args, **kwargs)
        return res

    def min(self, axis=None, keepdims=False, *args, **kwargs):
        """Return the minimum value along a given axis.
    
        Args:
            axis (int or tuple of ints, optional): Axis or axes along which to operate. By default, flattened input is used.
            keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
            *args: Additional positional arguments to be passed to the underlying min function.
            **kwargs: Additional keyword arguments to be passed to the underlying min function.
    
        Returns:
            ndarray or scalar: Minimum of array elements along the given axis.
        """
        res = self.value.min(axis=axis, keepdims=keepdims, *args, **kwargs)
        return res

    def nonzero(self):
        """Return the indices of the elements that are non-zero.
    
        Returns:
            tuple of arrays: Indices of elements that are non-zero.
        """
        return self.value.nonzero()

    def prod(self, axis=None, dtype=None, keepdims=False, initial=1, where=True):
        """Return the product of the array elements over the given axis.
    
        Args:
            axis (int or tuple of ints, optional): Axis or axes along which a product is performed.
            dtype (data-type, optional): The data-type of the returned array and of the accumulator in which the elements are multiplied.
            keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
            initial (scalar, optional): The starting value for the product.
            where (array_like of bool, optional): Elements to include in the product.
    
        Returns:
            ndarray or scalar: Product of array elements over the given axis.
        """
        res = self.value.prod(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
        return res

    def ptp(self, axis=None, keepdims=False):
        """Range of values (maximum - minimum) along an axis.
    
        Args:
            axis (int or tuple of ints, optional): Axis along which to find the peak-to-peak value. By default, flatten the array.
            keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
    
        Returns:
            ndarray or scalar: Peak-to-peak (maximum - minimum) value along the given axis.
        """
        r = self.value.ptp(axis=axis, keepdims=keepdims)
        return r

    def put(self, indices, values):
        """Replaces specified elements of an array with given values.
    
        Args:
            indices (array_like): Target indices, interpreted as integers.
            values (array_like): Values to place in the array at target indices.
        """
        self.__setitem__(indices, values)

    def ravel(self, order=None):
        """Return a flattened array.
    
        Args:
            order (str, optional): The elements of 'a' are read using this index order. 'C' means to index the elements in C-like order, 'F' means to index the elements in Fortran-like order, 'A' means to read the elements in Fortran-like order if 'a' is Fortran contiguous in memory, C-like order otherwise. 'K' means to read the elements in the order they occur in memory.
    
        Returns:
            ndarray: A 1-D array containing the same elements as the input array.
        """
        return self.value.ravel(order=order)

    def repeat(self, repeats, axis=None):
        """Repeat elements of an array.
    
        Args:
            repeats (int or array of ints): The number of repetitions for each element.
            axis (int, optional): The axis along which to repeat values.
    
        Returns:
            ndarray: Output array which has the same shape as input array, except along the given axis.
        """
        return self.value.repeat(repeats=repeats, axis=axis)

    def reshape(self, *shape, order='C'):
        """Returns an array containing the same data with a new shape.
    
        Args:
            *shape (int or tuple of ints): The new shape should be compatible with the original shape.
            order (str, optional): Read the elements using this index order. 'C' means to read the elements in C-like order, 'F' means to read the elements in Fortran-like order, 'A' means to read the elements in Fortran-like order if a is Fortran contiguous in memory, C-like order otherwise.
    
        Returns:
            ndarray: Array with the same data as the input array, but with a new shape.
        """
        return self.value.reshape(*shape, order=order)

    def resize(self, new_shape):
        """Change shape and size of array in-place.
    
        Args:
            new_shape (int or tuple of ints): Shape of resized array.
        """
        self.value = self.value.reshape(new_shape)

    def round(self, decimals=0):
        """Return the array with each element rounded to the given number of decimals.
    
        Args:
            decimals (int, optional): Number of decimal places to round to (default: 0).
                If decimals is negative, it specifies the number of positions to the left of the decimal point.
    
        Returns:
            ndarray: An array with the same shape as the input array, but with the elements rounded.
        """
        return self.value.round(decimals=decimals)

    def searchsorted(self, v, side='left', sorter=None):
        """Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `a` such that, if the
        corresponding elements in `v` were inserted before the indices, the
        order of `a` would be preserved.

        Assuming that `a` is sorted:

        ======  ============================
        `side`  returned index `i` satisfies
        ======  ============================
        left    ``a[i-1] < v <= a[i]``
        right   ``a[i-1] <= v < a[i]``
        ======  ============================

        Parameters
        ----------
        v : array_like
            Values to insert into `a`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `a`).
        sorter : 1-D array_like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        indices : array of ints
            Array of insertion points with the same shape as `v`.
        """
        return self.value.searchsorted(v=_as_array(v), side=side, sorter=sorter)

    def sort(self, axis=-1, stable=True, order=None):
        """Sort an array in-place.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the
            last axis.
        stable : bool, optional
            Whether to use a stable sorting algorithm. The default is True.
        order : str or list of str, optional
            When `a` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  A single field can
            be specified as a string, and not all fields need be specified,
            but unspecified fields will still be used, in the order in which
            they come up in the dtype, to break ties.
        """
        self.value = self.value.sort(axis=axis, stable=stable, order=order)

    def squeeze(self, axis=None):
        """
        Remove axes of length one from the array.
    
        This function removes single-dimensional entries from the shape of the array.
    
        Parameters
        ----------
        axis : int or tuple of ints, optional
            Selects a subset of the single-dimensional entries in the shape.
            If an axis is selected with shape entry greater than one, an error is raised.
            If None (default), all single-dimensional entries will be removed from the shape.
    
        Returns
        -------
        ndarray
            The input array with all or a subset of the dimensions of length 1 removed.
            This is always a view of the input array.
        """
        return self.value.squeeze(axis=axis)

    def std(self, axis=None, dtype=None, ddof=0, keepdims=False):
        """Compute the standard deviation along the specified axis.

        Returns the standard deviation, a measure of the spread of a distribution,
        of the array elements. The standard deviation is computed for the
        flattened array by default, otherwise over the specified axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the standard deviation is computed. The
            default is to compute the standard deviation of the flattened array.
            If this is a tuple of ints, a standard deviation is performed over
            multiple axes, instead of a single axis or all the axes as before.
        dtype : dtype, optional
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float64, for arrays of float types it is
            the same as the array type.
        ddof : int, optional
            Means Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            By default `ddof` is zero.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

            If the default value is passed, then `keepdims` will not be
            passed through to the `std` method of sub-classes of
            `ndarray`, however any non-default value will be.  If the
            sub-class' method does not implement `keepdims` any
            exceptions will be raised.

        Returns
        -------
        standard_deviation : ndarray, see dtype parameter above.
            If `out` is None, return a new array containing the standard deviation,
            otherwise return a reference to the output array.
        """
        r = self.value.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        return r

    def sum(self, axis=None, dtype=None, keepdims=False, initial=0, where=True):
        """Return the sum of the array elements over the given axis."""
        res = self.value.sum(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
        return res

    def swapaxes(self, axis1, axis2):
        """Return a view of the array with `axis1` and `axis2` interchanged."""
        return self.value.swapaxes(axis1, axis2)

    def split(self, indices_or_sections, axis=0):
        """Split an array into multiple sub-arrays as views into ``ary``.

        Parameters
        ----------
        indices_or_sections : int, 1-D array
          If `indices_or_sections` is an integer, N, the array will be divided
          into N equal arrays along `axis`.  If such a split is not possible,
          an error is raised.

          If `indices_or_sections` is a 1-D array of sorted integers, the entries
          indicate where along `axis` the array is split.  For example,
          ``[2, 3]`` would, for ``axis=0``, result in

            - ary[:2]
            - ary[2:3]
            - ary[3:]

          If an index exceeds the dimension of the array along `axis`,
          an empty sub-array is returned correspondingly.
        axis : int, optional
          The axis along which to split, default is 0.

        Returns
        -------
        sub-arrays : list of ndarrays
          A list of sub-arrays as views into `ary`.
        """
        return [a for a in u.math.split(self.value, indices_or_sections, axis=axis)]

    def take(self, indices, axis=None, mode=None):
        """Return an array formed from the elements of a at the given indices."""
        return self.value.take(indices=_as_array(indices), axis=axis, mode=mode)

    def tobytes(self):
        """Construct Python bytes containing the raw data bytes in the array.

        Constructs Python bytes showing a copy of the raw contents of data memory.
        The bytes object is produced in C-order by default. This behavior is
        controlled by the ``order`` parameter."""
        return self.value.tobytes()

    def tolist(self):
        """Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

        Return a copy of the array data as a (nested) Python list.
        Data items are converted to the nearest compatible builtin Python type, via
        the `~numpy.ndarray.item` function.

        If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
        not be a list at all, but a simple Python scalar.
        """
        return self.value.tolist()

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        """Return the sum along diagonals of the array."""
        return self.value.trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    def transpose(self, *axes):
        """Returns a view of the array with axes transposed.

        For a 1-D array this has no effect, as a transposed vector is simply the
        same vector. To convert a 1-D array into a 2D column vector, an additional
        dimension must be added. `np.atleast2d(a).T` achieves this, as does
        `a[:, np.newaxis]`.
        For a 2-D array, this is a standard matrix transpose.
        For an n-D array, if axes are given, their order indicates how the
        axes are permuted (see Examples). If axes are not provided and
        ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : None, tuple of ints, or `n` ints

         * None or no argument: reverses the order of the axes.

         * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
           `i`-th axis becomes `a.transpose()`'s `j`-th axis.

         * `n` ints: same as an n-tuple of the same ints (this form is
           intended simply as a "convenience" alternative to the tuple form)

        Returns
        -------
        out : ndarray
            View of `a`, with axes suitably permuted.
        """
        return self.value.transpose(*axes)

    def tile(self, reps):
        """Construct an array by repeating A the number of times given by reps.

        If `reps` has length ``d``, the result will have dimension of
        ``max(d, A.ndim)``.

        If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
        axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
        or shape (1, 1, 3) for 3-D replication. If this is not the desired
        behavior, promote `A` to d-dimensions manually before calling this
        function.

        If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
        Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
        (1, 1, 2, 2).

        Note : Although tile may be used for broadcasting, it is strongly
        recommended to use numpy's broadcasting operations and functions.

        Parameters
        ----------
        reps : array_like
            The number of repetitions of `A` along each axis.

        Returns
        -------
        c : ndarray
            The tiled output array.
        """
        return self.value.tile(_as_array(reps))

    def var(self, axis=None, dtype=None, ddof=0, keepdims=False):
        """Returns the variance of the array elements, along given axis."""
        r = self.value.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        return r

    def view(self, *args, dtype=None):
        if len(args) == 0:
            if dtype is None:
                raise ValueError('Provide dtype or shape.')
            else:
                return self.value.view(dtype)
        else:
            if isinstance(args[0], int):  # shape
                if dtype is not None:
                    raise ValueError('Provide one of dtype or shape. Not both.')
                return self.value.reshape(*args)
            else:  # dtype
                assert not isinstance(args[0], int)
                assert dtype is None
                return self.value.view(args[0])

    def __array__(self, dtype=None):
        """Support ``numpy.array()`` and ``numpy.asarray()`` functions."""
        return np.asarray(self.value, dtype=dtype)

    def __format__(self, specification):
        return self.value.__format__(specification)

    def __bool__(self) -> bool:
        return self.value.__bool__()

    def __float__(self):
        return self.value.__float__()

    def __int__(self):
        return self.value.__int__()

    def __complex__(self):
        return self.value.__complex__()

    def __hex__(self):
        assert self.ndim == 0, 'hex only works on scalar values'
        return hex(self.value)  # type: ignore

    def __oct__(self):
        assert self.ndim == 0, 'oct only works on scalar values'
        return oct(self.value)  # type: ignore

    def __index__(self):
        return operator.index(self.value)

    def __dlpack__(self):
        from jax.dlpack import to_dlpack  # pylint: disable=g-import-not-at-top
        return to_dlpack(self.value)

    # ----------------------
    # PyTorch compatibility
    # ----------------------

    def unsqueeze(self, dim: int) -> Union[jax.Array, u.Quantity]:
        """
        Array.unsqueeze(dim) -> Array, or so called Tensor
        equals
        Array.expand_dims(dim)
        """
        return u.math.expand_dims(self.value, dim)

    def expand_dims(self, axis: Union[int, Sequence[int]]) -> Union[jax.Array, u.Quantity]:
        """
        self.expand_dims(axis: int|Sequence[int])

        1. 如果axis类型为int：
        返回一个在self基础上的第axis维度前插入一个维度Array，
        axis<0表示倒数第|axis|维度，
        令n=len(self._value.shape)，则axis的范围为[-(n+1),n]

        2. 如果axis类型为Sequence[int]：
        则返回依次扩展axis[i]的结果，
        即self.expand_dims(axis)==self.expand_dims(axis[0]).expand_dims(axis[1])...expand_dims(axis[len(axis)-1])


        1. If the type of axis is int:

        Returns an Array of dimensions inserted before the axis dimension based on self,

        The first | axis < 0 indicates the bottom axis | dimensions,

        Set n=len(self._value.shape), then axis has the range [-(n+1),n]


        2. If the type of axis is Sequence[int] :

        Returns the result of extending axis[i] in sequence,

        self.expand_dims(axis)==self.expand_dims(axis[0]).expand_dims(axis[1])... expand_dims(axis[len(axis)-1])

        """
        return u.math.expand_dims(self.value, axis)

    def expand_as(self, array: Union['EventArray', ArrayLike]) -> 'EventArray':
        """
        Expand an array to a shape of another array.

        Parameters
        ----------
        array : EventArray

        Returns
        -------
        expanded : EventArray
            A readonly view on the original array with the given shape of array. It is
            typically not contiguous. Furthermore, more than one element of a
            expanded array may refer to a single memory location.
        """
        return u.math.broadcast_to(self.value, _as_array(array))

    def pow(self, index: int):
        return self.value ** index

    def addr(
        self,
        vec1: Union['EventArray', ArrayLike],
        vec2: Union['EventArray', ArrayLike],
        *,
        beta: float = 1.0,
        alpha: float = 1.0,
        out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r"""Performs the outer-product of vectors ``vec1`` and ``vec2`` and adds it to the matrix ``input``.

        Optional values beta and alpha are scaling factors on the outer product
        between vec1 and vec2 and the added matrix input respectively.

        .. math::

           out = \beta \mathrm{input} + \alpha (\text{vec1} \bigtimes \text{vec2})

        Args:
          vec1: the first vector of the outer product
          vec2: the second vector of the outer product
          beta: multiplier for input
          alpha: multiplier
          out: the output tensor.

        """
        vec1 = _as_array(vec1)
        vec2 = _as_array(vec2)
        r = alpha * u.math.outer(vec1, vec2) + beta * self.value
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def addr_(
        self,
        vec1: Union['EventArray', ArrayLike],
        vec2: Union['EventArray', ArrayLike],
        *,
        beta: float = 1.0,
        alpha: float = 1.0
    ):
        vec1 = _as_array(vec1)
        vec2 = _as_array(vec2)
        r = alpha * u.math.outer(vec1, vec2) + beta * self.value
        self.value = r
        return self

    def outer(
        self,
        other: Union['EventArray', ArrayLike]
    ) -> Union[u.Quantity, jax.Array]:
        other = _as_array(other)
        return u.math.outer(self.value, other.value)

    def abs(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.abs(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def abs_(self):
        """
        in-place version of Array.abs()
        """
        self.value = u.math.abs(self.value)
        return self

    def add_(self, value):
        self.value += value
        return self

    def absolute(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[jax.Array, u.Quantity]:
        """
        alias of Array.abs
        """
        return self.abs(out=out)

    def absolute_(self):
        """
        alias of Array.abs_()
        """
        return self.abs_()

    def mul(self, value):
        return self.value * value

    def mul_(self, value):
        """
        In-place version of :meth:`~Array.mul`.
        """
        self.value *= value
        return self

    def multiply(self, value):  # real signature unknown; restored from __doc__
        """
        multiply(value) -> Tensor

        See :func:`torch.multiply`.
        """
        return self.value * value

    def multiply_(self, value):  # real signature unknown; restored from __doc__
        """
        multiply_(value) -> Tensor

        In-place version of :meth:`~Tensor.multiply`.
        """
        self.value *= value
        return self

    def sin(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.sin(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def sin_(self):
        self.value = u.math.sin(self.value)
        return self

    def cos_(self):
        self.value = u.math.cos(self.value)
        return self

    def cos(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.cos(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def tan_(self):
        self.value = u.math.tan(self.value)
        return self

    def tan(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.tan(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def sinh_(self):
        self.value = u.math.sinh(self.value)
        return self

    def sinh(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.sinh(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def cosh_(self):
        self.value = u.math.cosh(self.value)
        return self

    def cosh(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.cosh(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def tanh_(self):
        self.value = u.math.tanh(self.value)
        return self

    def tanh(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.tanh(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def arcsin_(self):
        self.value = u.math.arcsin(self.value)
        return self

    def arcsin(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.arcsin(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def arccos_(self):
        self.value = u.math.arccos(self.value)
        return self

    def arccos(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.arccos(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def arctan_(self):
        self.value = u.math.arctan(self.value)
        return self

    def arctan(
        self, *, out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        r = u.math.arctan(self.value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def clamp(
        self,
        min_value: Optional[Union['EventArray', ArrayLike]] = None,
        max_value: Optional[Union['EventArray', ArrayLike]] = None,
        *,
        out: Optional[Union['EventArray', ArrayLike]] = None
    ) -> Union[u.Quantity, jax.Array]:
        """
        return the value between min_value and max_value,
        if min_value is None, then no lower bound,
        if max_value is None, then no upper bound.
        """
        min_value = _as_array(min_value)
        max_value = _as_array(max_value)
        r = u.math.clip(self.value, min_value, max_value)
        if out is None:
            return r
        else:
            _check_out(out)
            out.value = r

    def clamp_(
        self,
        min_value: Optional[Union['EventArray', ArrayLike]] = None,
        max_value: Optional[Union['EventArray', ArrayLike]] = None
    ):
        """
        return the value between min_value and max_value,
        if min_value is None, then no lower bound,
        if max_value is None, then no upper bound.
        """
        self.clamp(min_value, max_value, out=self)
        return self

    def clip_(
        self,
        min_value: Optional[Union['EventArray', ArrayLike]] = None,
        max_value: Optional[Union['EventArray', ArrayLike]] = None
    ):
        """
        alias for clamp_
        """
        self.value = self.clip(min_value, max_value, out=self)
        return self

    def clone(self) -> 'EventArray':
        return type(self)(self.value.copy())

    def copy_(self, src: Union['EventArray', ArrayLike]) -> 'EventArray':
        self.value = src.copy()
        return self

    def cov_with(
        self,
        y: Optional[Union['EventArray', ArrayLike]] = None,
        rowvar: bool = True,
        bias: bool = False,
        fweights: Union['EventArray', ArrayLike] = None,
        aweights: Union['EventArray', ArrayLike] = None
    ) -> 'EventArray':
        y = _as_array(y)
        fweights = _as_array(fweights)
        aweights = _as_array(aweights)
        r = u.math.cov(self.value, y, rowvar, bias, fweights, aweights)
        return r

    def expand(self, *sizes) -> Union[u.Quantity, jax.Array]:
        """
        Expand an array to a new shape.

        Parameters
        ----------
        sizes : tuple or int
            The shape of the desired array. A single integer ``i`` is interpreted
            as ``(i,)``.

        Returns
        -------
        expanded : EventArray
            A readonly view on the original array with the given shape. It is
            typically not contiguous. Furthermore, more than one element of a
            expanded array may refer to a single memory location.
        """
        l_ori = len(self.shape)
        l_tar = len(sizes)
        base = l_tar - l_ori
        sizes_list = list(sizes)
        if base < 0:
            raise ValueError(
                f'the number of sizes provided ({len(sizes)}) '
                f'must be greater or equal to the number of '
                f'dimensions in the tensor ({len(self.shape)})'
            )
        for i, v in enumerate(sizes[:base]):
            if v < 0:
                raise ValueError(
                    f'The expanded size of the tensor ({v}) isn\'t allowed in '
                    f'a leading, non-existing dimension {i + 1}'
                )
        for i, v in enumerate(self.shape):
            sizes_list[base + i] = v if sizes_list[base + i] == -1 else sizes_list[base + i]
            if v != 1 and sizes_list[base + i] != v:
                raise ValueError(
                    f'The expanded size of the tensor ({sizes_list[base + i]}) must '
                    f'match the existing size ({v}) at non-singleton '
                    f'dimension {i}.  Target sizes: {sizes}.  Tensor sizes: {self.shape}'
                )
        return u.math.broadcast_to(self.value, sizes_list)

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        return cls(*flat_contents)

    def zero_(self):
        self.value = u.math.zeros_like(self.value)
        return self

    def fill_(self, value):
        self.fill(value)
        return self

    def cuda(self):
        self.value = jax.device_put(self.value, jax.devices('cuda')[0])
        return self

    def cpu(self):
        self.value = jax.device_put(self.value, jax.devices('cpu')[0])
        return self

    # dtype exchanging #
    # ---------------- #

    def bool(self):
        return u.math.asarray(self.value, dtype=np.bool_)

    def int(self):
        """
        Convert the array to a 32-bit integer data type.

        Returns:
            The array converted to a 32-bit integer data type.
        """
        return u.math.asarray(self.value, dtype=np.int32)

    def long(self):
        """
        Convert the array to a 64-bit integer data type.

        Returns:
            The array converted to a 64-bit integer data type.
        """
        return u.math.asarray(self.value, dtype=np.int64)

    def half(self):
        """
        Convert the array to a 16-bit floating-point data type.

        Returns:
            The array converted to a 16-bit floating-point data type.
        """
        return u.math.asarray(self.value, dtype=np.float16)

    def float(self):
        """
        Convert the array to a 32-bit floating-point data type.

        Returns:
            The array converted to a 32-bit floating-point data type.
        """
        return u.math.asarray(self.value, dtype=np.float32)

    def bfloat16(self):
        """
        Convert the array to a Brain Floating Point 16 (bfloat16) data type.

        Returns:
            The array converted to a bfloat16 data type.
        """
        return u.math.asarray(self.value, dtype=jax.numpy.bfloat16)

    def double(self):
        """
        Convert the array to a 64-bit floating-point data type.

        Returns:
            The array converted to a 64-bit floating-point data type.
        """
        return u.math.asarray(self.value, dtype=np.float64)


# Set the array priority for the EventArray class
setattr(EventArray, "__array_priority__", 100)
