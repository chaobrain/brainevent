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

import operator
from typing import Union, Optional, Sequence

import brainunit as u
import jax
import numpy as np
from jax.dtypes import canonicalize_dtype
from jax.tree_util import register_pytree_node_class

from brainevent._error import MathError

__all__ = [
    'BaseArray',
]


def _get_dtype(v):
    if hasattr(v, 'dtype'):
        dtype = v.dtype
    else:
        dtype = canonicalize_dtype(type(v))
    return dtype


def extract_raw_value(obj):
    return obj.value if isinstance(obj, BaseArray) else obj


def is_known_type(x):
    return isinstance(x, (u.Quantity, jax.Array, np.ndarray, BaseArray))


ArrayLike = Union[jax.Array, np.ndarray, u.Quantity]


@register_pytree_node_class
class BaseArray:
    """
    The base array class for representing low-bit arrays.

    This class provides a basic implementation for low-bit arrays, which can be used to represent arrays
    with low-precision floating-point numbers. It supports basic operations such as addition, subtraction,
    multiplication, and division, and provides methods for checking the tracer and updating the array value.
    """
    __slots__ = ('_value',)
    __module__ = 'brainevent'

    def __init__(self, value, dtype: jax.typing.DTypeLike = None):
        """
        Initialize an BaseArray instance.

        Args:
            value: The input value, which can be an BaseArray, tuple, list, or np.ndarray.
            dtype: The data type of the array. If None, the data type will be inferred from the input value.
        """
        # array value
        if isinstance(value, BaseArray):
            value = value.value
        elif isinstance(value, (tuple, list, np.ndarray)):
            value = u.math.asarray(value)
        if dtype is not None:
            value = u.math.asarray(value, dtype=dtype)
        self._value = value

    def __hash__(self):
        return hash(self.value)

    def _check_tracer(self):
        """
        Check the tracer of the array value.

        Returns:
            The array value.
        """
        return self._value

    @property
    def data(self) -> Union[jax.Array, u.Quantity]:
        """
        Get the array value.

        Returns:
            The array value.
        """
        return self._value

    @data.setter
    def data(self, value):
        """
        Set the array value.
        Args:
            value: The new value to be set.
        Raises:
            MathError: If the shape or dtype of the new value does not match the original value.
        """
        self._update(value)

    @property
    def value(self) -> Union[jax.Array, u.Quantity]:
        # return the value
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
        self._update(value)

    def _update(self, value):
        # Get the current value for comparison with the new value
        self_value = self._check_tracer()

        # Handle different types of incoming values
        if isinstance(value, BaseArray):
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

        # # Check if the dtype of the new value matches the original value
        # if value.dtype != self_value.dtype:
        #     raise MathError(
        #         f"The dtype of the original data is {self_value.dtype}, "
        #         f"while we got {value.dtype}."
        #     )

        # Set the new value after passing the check
        self._value = value

    def update(self, value):
        """
        Update the value of this BaseArray.

        This method updates the internal value of the BaseArray with a new value.

        Parameters
        ----------
        value : array-like
            The new value to update the BaseArray with. This should be compatible
            with the current array in terms of shape and dtype.

        Returns
        -------
        None
            This method modifies the BaseArray in-place and doesn't return anything.

        Raises
        ------
        MathError
            If the shape or dtype of the new value does not match the original value.
        """
        self._update(value)

    @property
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

    @property
    def dtype(self):
        """Return the data type of the array's elements.

        This property accesses the data type (dtype) of the underlying array.
        For arrays containing numbers, this is their precision (e.g., float32,
        int64, etc.).

        Returns:
            dtype: The data type of the array's elements.
        """
        return _get_dtype(self._value)

    @property
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
        Return a string representation of the BaseArray.

        Returns:
            A string representation of the BaseArray.
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
            index = tuple(extract_raw_value(x) for x in index)
        elif isinstance(index, BaseArray):
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
        if isinstance(value, BaseArray):
            value = value.value
        # value is numpy.ndarray
        elif isinstance(value, np.ndarray):
            value = u.math.asarray(value)

        # index is a tuple
        if isinstance(index, tuple):
            index = tuple(extract_raw_value(x) for x in index)
        # index is Array
        elif isinstance(index, BaseArray):
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
        return self.value == extract_raw_value(oc)

    def __ne__(self, oc):
        """
        Compare the array with another object for inequality.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the inequality.
        """
        return self.value != extract_raw_value(oc)

    def __lt__(self, oc):
        """
        Compare the array with another object for less than.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the comparison result.
        """
        return self.value < extract_raw_value(oc)

    def __le__(self, oc):
        """
        Compare the array with another object for less than or equal to.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the comparison result.
        """
        return self.value <= extract_raw_value(oc)

    def __gt__(self, oc):
        """
        Compare the array with another object for greater than.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the comparison result.
        """
        return self.value > extract_raw_value(oc)

    def __ge__(self, oc):
        """
        Compare the array with another object for greater than or equal to.

        Args:
            oc: The object to compare with.

        Returns:
            A boolean array indicating the comparison result.
        """
        return self.value >= extract_raw_value(oc)

    def __add__(self, oc):
        """
        Add the array with another object.

        Args:
            oc: The object to add.

        Returns:
            The result of the addition.
        """
        return self.value + extract_raw_value(oc)

    def __radd__(self, oc):
        """
        Add another object with the array.

        Args:
            oc: The object to add.

        Returns:
            The result of the addition.
        """
        return self.value + extract_raw_value(oc)

    def __iadd__(self, oc):
        """
        Add another object to the array in-place.

        Args:
            oc: The object to add.

        Returns:
            The updated array.
        """
        # a += b
        return self.value + extract_raw_value(oc)

    def __sub__(self, oc):
        """
        Subtract another object from the array.

        Args:
            oc: The object to subtract.

        Returns:
            The result of the subtraction.
        """
        return self.value - extract_raw_value(oc)

    def __rsub__(self, oc):
        """
        Subtract the array from another object.

        Args:
            oc: The object to subtract from.

        Returns:
            The result of the subtraction.
        """
        return extract_raw_value(oc) - self.value

    def __isub__(self, oc):
        """
        Subtract another object from the array in-place.

        Args:
            oc: The object to subtract.

        Returns:
            The updated array.
        """
        # a -= b
        return self.value - extract_raw_value(oc)

    def __mul__(self, oc):
        """
        Multiply the array with another object.

        Args:
            oc: The object to multiply.

        Returns:
            The result of the multiplication.
        """
        return self.value * extract_raw_value(oc)

    def __rmul__(self, oc):
        """
        Multiply another object with the array.

        Args:
            oc: The object to multiply.

        Returns:
            The result of the multiplication.
        """
        return extract_raw_value(oc) * self.value

    def __imul__(self, oc):
        """
        Multiply the array with another object in-place.

        Args:
            oc: The object to multiply.

        Returns:
            The updated array.
        """
        # a *= b
        return self.value * extract_raw_value(oc)

    def __rdiv__(self, oc):
        """
        Divide another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the division.
        """
        return extract_raw_value(oc) / self.value

    def __truediv__(self, oc):
        """
        Divide the array by another object.

        Args:
            oc: The object to divide by.

        Returns:
            The result of the division.
        """
        return self.value / extract_raw_value(oc)

    def __rtruediv__(self, oc):
        """
        Divide another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the division.
        """
        return extract_raw_value(oc) / self.value

    def __itruediv__(self, oc):
        """
        Divide the array by another object in-place.

        Args:
            oc: The object to divide by.

        Returns:
            The updated array.
        """
        # a /= b
        return self.value / extract_raw_value(oc)

    def __floordiv__(self, oc):
        """
        Perform floor division on the array by another object.

        Args:
            oc: The object to divide by.

        Returns:
            The result of the floor division.
        """
        return self.value // extract_raw_value(oc)

    def __rfloordiv__(self, oc):
        """
        Perform floor division on another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the floor division.
        """
        return extract_raw_value(oc) // self.value

    def __ifloordiv__(self, oc):
        """
        Perform floor division on the array by another object in-place.

        Args:
            oc: The object to divide by.

        Returns:
            The updated array.
        """
        # a //= b
        return self.value // extract_raw_value(oc)

    def __divmod__(self, oc):
        """
        Perform divmod operation on the array by another object.

        Args:
            oc: The object to divide by.

        Returns:
            The result of the divmod operation.
        """
        return self.value.__divmod__(extract_raw_value(oc))

    def __rdivmod__(self, oc):
        """
        Perform divmod operation on another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the divmod operation.
        """
        return self.value.__rdivmod__(extract_raw_value(oc))

    def __mod__(self, oc):
        """
        Perform modulo operation on the array by another object.

        Args:
            oc: The object to divide by.

        Returns:
            The result of the modulo operation.
        """
        return self.value % extract_raw_value(oc)

    def __rmod__(self, oc):
        """
        Perform modulo operation on another object by the array.

        Args:
            oc: The object to divide.

        Returns:
            The result of the modulo operation.
        """
        return extract_raw_value(oc) % self.value

    def __imod__(self, oc):
        """
        Perform modulo operation on the array by another object in-place.

        Args:
            oc: The object to divide by.

        Returns:
            The updated array.
        """
        # a %= b
        return self.value % extract_raw_value(oc)

    def __pow__(self, oc):
        """
        Raise the array to the power of another object.

        Args:
            oc: The object to raise to the power.

        Returns:
            The result of the power operation.
        """
        return self.value ** extract_raw_value(oc)

    def __rpow__(self, oc):
        """
        Raise another object to the power of the array.

        Args:
            oc: The object to raise to the power.

        Returns:
            The result of the power operation.
        """
        return extract_raw_value(oc) ** self.value

    def __ipow__(self, oc):
        """
        Raise the array to the power of another object in-place.

        Args:
            oc: The object to raise to the power.

        Returns:
            The updated array.
        """
        # a **= b
        return self.value ** extract_raw_value(oc)

    def __matmul__(self, oc):
        """
        Perform matrix multiplication on the array with another object.

        This special method implements the matrix multiplication operator (@)
        for BaseArray instances. It handles matrix multiplication with different
        array types and dimensions, performing appropriate validation checks.

        Parameters
        ----------
        oc : array_like
            The right operand of the matrix multiplication. This object will be
            multiplied with the current BaseArray instance.

        Returns
        -------
        ndarray or BaseArray
            The result of the matrix multiplication between this BaseArray instance
            and the other object.

        Raises
        ------
        MathError
            If the dimensions of the operands are incompatible for matrix multiplication
            or if the array dimensions are not suitable (only 1D and 2D arrays are supported).

        Notes
        -----
        - For 1D array @ 2D array: This performs vector-matrix multiplication
        - For 2D array @ 2D array: This performs standard matrix multiplication
        - The method checks dimensions for compatibility before performing the operation
        - If the right operand is not a recognized array type, it delegates to the
          operand's __rmatmul__ method
        """
        raise NotImplementedError("Matrix multiplication is not supported for BaseArray.")

    def __rmatmul__(self, oc):
        """
        Perform matrix multiplication on another object with the array.

        This special method implements the reverse matrix multiplication operator (@)
        when the left operand is not an BaseArray. It handles the case where
        another object is matrix-multiplied with this BaseArray instance.

        Parameters
        ----------
        oc : array_like
            The left operand of the matrix multiplication. This object will be
            multiplied with the current BaseArray instance.

        Returns
        -------
        ndarray or BaseArray
            The result of the matrix multiplication between the other object and this
            BaseArray instance.

        Raises
        ------
        MathError
            If the dimensions of the operands are incompatible for matrix multiplication
            or if the array dimensions are not suitable (only 1D and 2D arrays are supported).

        Notes
        -----
        - For 2D arrays, this performs standard matrix multiplication
        - For a 1D array multiplied by a 2D array, it performs a vector-matrix multiplication
        - The method checks dimensions for compatibility before performing the operation
        """
        raise NotImplementedError("Matrix multiplication is not supported for BaseArray.")

    def __imatmul__(self, oc):
        """
        Perform matrix multiplication on the array with another object in-place.

        Args:
            oc: The object to multiply.

        Returns:
            The updated array.
        """
        raise NotImplementedError("Matrix multiplication is not supported for BaseArray.")

    def __and__(self, oc):
        """
        Perform bitwise AND operation on the array with another object.

        Args:
            oc: The object to perform AND operation with.

        Returns:
            The result of the bitwise AND operation.
        """
        return self.value & extract_raw_value(oc)

    def __rand__(self, oc):
        """
        Perform bitwise AND operation on another object with the array.

        Args:
            oc: The object to perform AND operation with.

        Returns:
            The result of the bitwise AND operation.
        """
        return extract_raw_value(oc) & self.value

    def __iand__(self, oc):
        """
        Perform bitwise AND operation on the array with another object in-place.

        Args:
            oc: The object to perform AND operation with.

        Returns:
            The updated array.
        """
        # a &= b
        return self.value & extract_raw_value(oc)

    def __or__(self, oc):
        """
        Perform bitwise OR operation on the array with another object.

        Args:
            oc: The object to perform OR operation with.

        Returns:
            The result of the bitwise OR operation.
        """
        return self.value | extract_raw_value(oc)

    def __ror__(self, oc):
        """
        Perform bitwise OR operation on another object with the array.

        Args:
            oc: The object to perform OR operation with.

        Returns:
            The result of the bitwise OR operation.
        """
        return extract_raw_value(oc) | self.value

    def __ior__(self, oc):
        """
        Perform bitwise OR operation on the array with another object in-place.

        Args:
            oc: The object to perform OR operation with.

        Returns:
            The updated array.
        """
        # a |= b
        return self.value | extract_raw_value(oc)

    def __xor__(self, oc):
        """
        Perform bitwise XOR operation on the array with another object.

        Args:
            oc: The object to perform XOR operation with.

        Returns:
            The result of the bitwise XOR operation.
        """
        return self.value ^ extract_raw_value(oc)

    def __rxor__(self, oc):
        """
        Perform bitwise XOR operation on another object with the array.

        Args:
            oc: The object to perform XOR operation with.

        Returns:
            The result of the bitwise XOR operation.
        """
        return extract_raw_value(oc) ^ self.value

    def __ixor__(self, oc):
        """
        Perform bitwise XOR operation on the array with another object in-place.

        Args:
            oc: The object to perform XOR operation with.

        Returns:
            The updated array.
        """
        # a ^= b
        return self.value ^ extract_raw_value(oc)

    def __lshift__(self, oc):
        """
        Perform left shift operation on the array by another object.

        Args:
            oc: The object to shift by.

        Returns:
            The result of the left shift operation.
        """
        return self.value << extract_raw_value(oc)

    def __rlshift__(self, oc):
        """
        Perform left shift operation on another object by the array.

        Args:
            oc: The object to shift.

        Returns:
            The result of the left shift operation.
        """
        return extract_raw_value(oc) << self.value

    def __ilshift__(self, oc):
        """
        Perform left shift operation on the array by another object in-place.

        Args:
            oc: The object to shift by.

        Returns:
            The updated array.
        """
        # a <<= b
        return self.value << extract_raw_value(oc)

    def __rshift__(self, oc):
        """
        Perform right shift operation on the array by another object.

        Args:
            oc: The object to shift by.

        Returns:
            The result of the right shift operation.
        """
        return self.value >> extract_raw_value(oc)

    def __rrshift__(self, oc):
        """
        Perform right shift operation on another object by the array.

        Args:
            oc: The object to shift.

        Returns:
            The result of the right shift operation.
        """
        return extract_raw_value(oc) >> self.value

    def __irshift__(self, oc):
        """
        Perform right shift operation on the array by another object in-place.

        Args:
            oc: The object to shift by.

        Returns:
            The updated array.
        """
        # a >>= b
        return self.value >> extract_raw_value(oc)

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
        Accesses the JAX indexed update functionality for the underlying array.

        This property returns an object that allows for functional-style updates
        of the array's elements. Instead of modifying the array in-place (which
        is generally discouraged in JAX), methods on the returned object (like
        `.set()`, `.add()`, `.min()`, `.max()`) create and return a *new* array
        with the specified modifications.

        This is crucial for working within JAX's functional programming paradigm,
        especially inside JIT-compiled functions, loops (`lax.scan`, `lax.fori_loop`),
        or gradient transformations (`jax.grad`).

        Returns
        -------
        jax.numpy.ndarray.at
            An object enabling indexed updates on the underlying JAX array.

        See Also
        --------
        jax.numpy.ndarray.at : The underlying JAX functionality.
        __setitem__ : For direct (but potentially less JAX-idiomatic) item assignment.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> a = EventArray(jnp.array([1, 2, 3, 4]))

        >>> # Set the element at index 1 to 10
        >>> b = a.at[1].set(10)
        >>> print(a) # Original array is unchanged
        BaseArray(value=array([1, 2, 3, 4]), dtype=int32)
        >>> print(b) # New array with the update
        BaseArray(value=array([ 1, 10,  3,  4]), dtype=int32)

        >>> # Add 5 to the element at index 0
        >>> c = a.at[0].add(5)
        >>> print(c)
        BaseArray(value=array([6, 2, 3, 4]), dtype=int32)

        >>> # Set multiple elements using slicing
        >>> d = a.at[1:3].set(jnp.array([5, 6]))
        >>> print(d)
        BaseArray(value=array([1, 5, 6, 4]), dtype=int32)
        """
        return self.value.at

    def block_until_ready(self):
        """
        Waits until all asynchronous computations involving this array are complete.

        JAX operations, especially on accelerators like GPUs or TPUs, are often
        dispatched asynchronously. This means the Python code might continue
        executing before the actual computation on the device is finished.
        Calling `block_until_ready()` ensures that any pending computations
        related to `self.value` have completed on the device before the
        Python program proceeds past this call.

        This is primarily useful for:
        1.  Accurate timing (benchmarking) of JAX operations.
        2.  Ensuring data is ready before being used by non-JAX code (e.g.,
            saving to disk, passing to a different library).
        3.  Debugging synchronization issues.

        Returns
        -------
        BaseArray
            The instance itself, after ensuring its underlying data's computations
            are complete. The data (`self.value`) remains unchanged.

        See Also
        --------
        jax.block_until_ready : The underlying JAX function.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import time
        >>> from brainevent import EventArray

        >>> # Assume 'a' is on a GPU/TPU where operations might be async
        >>> a = EventArray(jnp.arange(1000000)).cuda() # Move to GPU if available

        >>> # Perform some computation
        >>> start_time = time.time()
        >>> b = jnp.sin(a.value) * 2
        >>> # Without block_until_ready, end_time might be recorded before
        >>> # the computation actually finishes on the device.
        >>> end_time_async = time.time()

        >>> # Now, ensure computation is done before recording time
        >>> b_event = EventArray(b)
        >>> b_event.block_until_ready()
        >>> end_time_sync = time.time()

        >>> print(f"Async dispatch time: {end_time_async - start_time:.6f}s")
        >>> print(f"Synchronized time:   {end_time_sync - start_time:.6f}s")
        >>> # The synchronized time will typically be longer, reflecting the
        >>> # actual computation time on the device.

        >>> # Ensure 'b_event' is ready before using its value elsewhere
        >>> result_array = np.array(b_event.block_until_ready().value)
        """
        # The method returns the result of jax.block_until_ready, which is the
        # original array object after the wait. We return self for chaining.
        _ = jax.block_until_ready(self.value)
        return self

    # ----------------------- #
    #      NumPy methods      #
    # ----------------------- #

    def all(self, axis=None, keepdims=False):
        """
        Test whether all array elements along a given axis evaluate to True.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which a logical AND reduction is performed.
            The default (`axis=None`) is to perform a logical AND over all
            the dimensions of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.
            If this is a tuple of ints, a reduction is performed on multiple
            axes, instead of a single axis or all the axes as before.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
            If the default value is passed, then `keepdims` will not be
            passed through to the `all` method of sub-classes of
            `ndarray`, however any non-default value will be. If the
            sub-class' method does not implement `keepdims` any
            exceptions will be raised. Default is False.

        Returns
        -------
        jax.Array or bool
            A new boolean array or a scalar boolean, resulting from the AND
            reduction over the specified axis.

        See Also
        --------
        any : Test whether any element along a given axis evaluates to True.
        jax.numpy.all : Equivalent JAX function.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> a = EventArray(jnp.array([[True, False], [True, True]]))
        >>> a.all()
        Array(False, dtype=bool)
        >>> a.all(axis=0)
        Array([ True, False], dtype=bool)
        >>> a.all(axis=1)
        Array([False,  True], dtype=bool)
        >>> a.all(keepdims=True)
        Array([[False]], dtype=bool)
        """
        return self.value.all(axis=axis, keepdims=keepdims)

    def any(self, axis=None, keepdims=False):
        """
        Test whether any array element along a given axis evaluates to True.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which a logical OR reduction is performed.
            The default (`axis=None`) is to perform a logical OR over all
            the dimensions of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.
            If this is a tuple of ints, a reduction is performed on multiple
            axes, instead of a single axis or all the axes as before.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
            If the default value is passed, then `keepdims` will not be
            passed through to the `any` method of sub-classes of
            `ndarray`, however any non-default value will be. If the
            sub-class' method does not implement `keepdims` any
            exceptions will be raised. Default is False.

        Returns
        -------
        jax.Array or bool
            A new boolean array or a scalar boolean, resulting from the OR
            reduction over the specified axis.

        See Also
        --------
        all : Test whether all elements along a given axis evaluate to True.
        jax.numpy.any : Equivalent JAX function.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> a = EventArray(jnp.array([[True, False], [False, False]]))
        >>> a.any()
        Array(True, dtype=bool)
        >>> a.any(axis=0)
        Array([ True, False], dtype=bool)
        >>> a.any(axis=1)
        Array([ True, False], dtype=bool)
        >>> a.any(keepdims=True)
        Array([[ True]], dtype=bool)
        """
        return self.value.any(axis=axis, keepdims=keepdims)

    def argmax(self, axis=None):
        """
        Return indices of the maximum values along the given axis.

        Parameters
        ----------
        axis : int, optional
            By default, the index is into the flattened array, otherwise
            along the specified axis.

        Returns
        -------
        jax.Array
            Array of indices into the array. It has the same shape as `self.shape`
            with the dimension along `axis` removed. If `axis` is None, the
            result is a scalar index into the flattened array.

        See Also
        --------
        argmin : Return indices of the minimum values along the given axis.
        max : The maximum value along a given axis.
        jax.numpy.argmax : Equivalent JAX function.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> a = EventArray(jnp.arange(6).reshape(2,3) + 10)
        >>> a.value
        Array([[10, 11, 12],
               [13, 14, 15]], dtype=int32)
        >>> a.argmax()
        Array(5, dtype=int32)
        >>> a.argmax(axis=0)
        Array([1, 1, 1], dtype=int32)
        >>> a.argmax(axis=1)
        Array([2, 2], dtype=int32)
        """
        return self.value.argmax(axis=axis)

    def argmin(self, axis=None):
        """
        Return indices of the minimum values along the given axis.

        Parameters
        ----------
        axis : int, optional
            By default, the index is into the flattened array, otherwise
            along the specified axis.

        Returns
        -------
        jax.Array
            Array of indices into the array. It has the same shape as `self.shape`
            with the dimension along `axis` removed. If `axis` is None, the
            result is a scalar index into the flattened array.

        See Also
        --------
        argmax : Return indices of the maximum values along the given axis.
        min : The minimum value along a given axis.
        jax.numpy.argmin : Equivalent JAX function.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> a = EventArray(jnp.arange(6).reshape(2,3) + 10)
        >>> a.value
        Array([[10, 11, 12],
               [13, 14, 15]], dtype=int32)
        >>> a.argmin()
        Array(0, dtype=int32)
        >>> a.argmin(axis=0)
        Array([0, 0, 0], dtype=int32)
        >>> a.argmin(axis=1)
        Array([0, 0], dtype=int32)
        """
        return self.value.argmin(axis=axis)

    def argpartition(self, kth, axis=-1):
        """
        Returns the indices that would partition this array along the given axis.

        Performs an indirect partition along the given axis using the
        algorithm specified by the `kind` keyword. It returns an array of
        indices of the same shape as `self` such that, if `p` is the
        returned array, then `self[p]` is the partitioned array.

        Parameters
        ----------
        kth : int or sequence of ints
            Element index to partition by. The k-th element will be in its
            final sorted position and all smaller elements will be moved
            before it and all larger elements behind it. The order of all
            elements in the partitions is undefined. If provided with a
            sequence of k-th it will partition all of them into their sorted
            position at once.
        axis : int or None, optional
            Axis along which to sort. If None, the array is flattened before
            partitioning. The default is -1 (the last axis).

        Returns
        -------
        jax.Array
            Array of indices that partition `self` along the specified axis.

        See Also
        --------
        partition : Describes partition algorithms used.
        sort : Full sorting.
        argsort : Indirect sort.
        jax.numpy.argpartition : Equivalent JAX function.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> x = EventArray(jnp.array([3, 4, 2, 1]))
        >>> x.argpartition(1) # Index of the 2nd smallest element
        Array([3, 2, 0, 1], dtype=int32) # Indices that put 2nd smallest (value 2) in place
        >>> x.argpartition((1, 2)) # Indices of the 2nd and 3rd smallest elements
        Array([3, 2, 0, 1], dtype=int32) # Indices that put 2nd (value 2) and 3rd (value 3) in place

        >>> # Partition along axis 1
        >>> y = EventArray(jnp.array([[3, 4, 2], [1, 3, 5]]))
        >>> y.argpartition(1, axis=1)
        Array([[2, 0, 1],
               [0, 1, 2]], dtype=int32)
        """
        # Note: JAX argpartition doesn't support 'kind' or 'order' arguments like NumPy
        return self.value.argpartition(kth=kth, axis=axis)

    def argsort(self, axis=-1, kind=None, order=None):
        """
        Returns the indices that would sort this array.

        Perform an indirect sort along the given axis using the algorithm specified
        by the `kind` keyword. It returns an array of indices of the same shape as
        `self` that index data along the given axis in sorted order.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to sort. The default is -1 (the last axis). If None,
            the flattened array is used.
        kind : {'stable'}, optional
            JAX currently only supports stable sorting. This argument is kept for
            compatibility but does not affect the outcome.
        order : None
            This argument is ignored in JAX and is kept for NumPy compatibility.

        Returns
        -------
        jax.Array
            Array of indices that sort `self` along the specified axis.
            If `self` is a 1-D array, `self[index_array]` yields a sorted `self`.
            More generally, `np.take_along_axis(self, index_array, axis=axis)`
            always yields the sorted `self`, irrespective of dimensionality.

        See Also
        --------
        sort : Describes sorting algorithms used.
        argpartition : Indirect partial sort.
        take_along_axis : Apply the indices returned by argsort.
        jax.numpy.argsort : Equivalent JAX function.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> x = EventArray(jnp.array([3, 1, 2]))
        >>> x.argsort()
        Array([1, 2, 0], dtype=int32)

        >>> y = EventArray(jnp.array([[0, 3], [2, 2]]))
        >>> y.argsort(axis=0) # Sort indices along columns
        Array([[0, 1],
               [1, 0]], dtype=int32)
        >>> y.argsort(axis=1) # Sort indices along rows
        Array([[0, 1],
               [0, 1]], dtype=int32)
        """
        # JAX argsort only supports stable kind and ignores order.
        # We pass kind and order for potential future compatibility or clarity,
        # but they currently have no effect in JAX's implementation.
        if kind is not None and kind != 'stable':
            # Consider warning or raising an error if strict NumPy compatibility is needed
            pass
        if order is not None:
            # Consider warning or raising an error
            pass
        return self.value.argsort(axis=axis)  # kind and order are effectively ignored by JAX

    def astype(self, dtype):
        """
        Copy of the underlying array, cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        Returns
        -------
        jax.Array or brainunit.Quantity
            A copy of the underlying `self.value` array, cast to the specified `dtype`.
            Note that this method returns the underlying JAX array or Quantity,
            *not* a new BaseArray instance.

        Raises
        ------
        TypeError
            If `dtype` is None, as a target dtype must be specified.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> x = EventArray(jnp.array([1, 2, 2.5]))
        >>> x.value
        Array([1. , 2. , 2.5], dtype=float32)
        >>> x.astype(jnp.int32)
        Array([1, 2, 2], dtype=int32)
        >>> x.astype(jnp.float64)
        Array([1. , 2. , 2.5], dtype=float64)

        >>> # Original BaseArray remains unchanged
        >>> x.dtype
        dtype('float32')
        """
        # Bug Fix/Improvement: Raise error if dtype is None, as None is not a valid target type.
        # Returning self.value without casting when dtype is None might be unexpected.
        if dtype is None:
            raise TypeError("The dtype argument is required and cannot be None for astype.")
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

    def clip(self, min=None, max=None):
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

        Returns
        -------
        ndarray
            An array with the elements of self, but where values < min are replaced with min,
            and those > max with max.

        Note
        ----
        At least one of max or min must be given.
        """
        min = extract_raw_value(min)
        max = extract_raw_value(max)
        r = self.value.clip(min=min, max=max)
        return r

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
        return self.value.compress(condition=extract_raw_value(condition), axis=axis)

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
        if is_known_type(b):
            return self.value.dot(extract_raw_value(b))
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
        """
        Return the indices of the elements that are non-zero.

        Returns the indices of the array elements that are non-zero, grouped by dimension.
        This method mirrors the behavior of `numpy.nonzero`.

        Returns
        -------
        tuple of ndarray
            A tuple of arrays, one for each dimension of the `BaseArray`, containing
            the indices of the non-zero elements in that dimension.

        See Also
        --------
        numpy.nonzero : Equivalent NumPy function.
        jax.numpy.nonzero : Equivalent JAX function.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> x = EventArray(jnp.array([[1, 0, 3], [0, 5, 0]]))
        >>> x.nonzero()
        (Array([0, 0, 1], dtype=int32), Array([0, 2, 1], dtype=int32))

        >>> y = EventArray(jnp.array([1, 0, 0, 4, 0]))
        >>> y.nonzero()
        (Array([0, 3], dtype=int32),)
        """
        return self.value.nonzero()

    def prod(self, axis=None, dtype=None, keepdims=False, initial=1, where=True):
        """
        Return the product of the array elements over the given axis.

        Computes the product of elements along a specified axis. This method mirrors
        the behavior of `numpy.prod`.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a product is performed. The default (`axis=None`)
            is to compute the product of all elements in the flattened array.
        dtype : data-type, optional
            The data-type of the returned array and of the accumulator in which the
            elements are multiplied. If `dtype` is not specified, it defaults to the
            dtype of `self.value`, unless `self.value` has an integer dtype with
            fewer bits than the default platform integer. In that case, the default
            platform integer is used.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array. Default is False.
        initial : scalar, optional
            The starting value for the product. Default is 1.
        where : array_like of bool, optional
            Elements to include in the product. See `jax.numpy.prod` for details.
            Default is True (include all elements).

        Returns
        -------
        ndarray or scalar
            An array containing the products along the specified axis. Returns a
            scalar if `axis` is None.

        See Also
        --------
        numpy.prod : Equivalent NumPy function.
        jax.numpy.prod : Equivalent JAX function.
        cumsum : Cumulative sum of array elements.
        cumprod : Cumulative product of array elements.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> x = EventArray(jnp.array([[1, 2], [3, 4]]))
        >>> x.prod()
        Array(24, dtype=int32)
        >>> x.prod(axis=0)
        Array([3, 8], dtype=int32)
        >>> x.prod(axis=1)
        Array([ 2, 12], dtype=int32)
        >>> x.prod(axis=1, keepdims=True)
        Array([[ 2],
               [12]], dtype=int32)
        >>> x.prod(initial=5)
        Array(120, dtype=int32)
        >>> x.prod(where=jnp.array([[True, False], [True, True]]))
        Array(12, dtype=int32) # 1 * 3 * 4
        """
        res = self.value.prod(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
        return res

    def ptp(self, axis=None, keepdims=False):
        """
        Range of values (maximum - minimum) along an axis.

        Calculates the difference between the maximum and minimum values over
        a given axis. This method mirrors the behavior of `numpy.ptp`.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which the range is computed. The default (`axis=None`)
            is to compute the range of the flattened array.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array. Default is False.

        Returns
        -------
        ndarray or scalar
            An array containing the range of values along the specified axis(es).
            Returns a scalar if `axis` is None.

        See Also
        --------
        numpy.ptp : Equivalent NumPy function.
        jax.numpy.ptp : Equivalent JAX function.
        max : Maximum value along an axis.
        min : Minimum value along an axis.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> x = EventArray(jnp.arange(12).reshape((3, 4)))
        >>> x
        BaseArray(value=Array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]], dtype=int32))
        >>> x.ptp()
        Array(11, dtype=int32)
        >>> x.ptp(axis=0)
        Array([8, 8, 8, 8], dtype=int32)
        >>> x.ptp(axis=1)
        Array([3, 3, 3], dtype=int32)
        >>> x.ptp(axis=1, keepdims=True)
        Array([[3],
               [3],
               [3]], dtype=int32)
        """
        r = self.value.ptp(axis=axis, keepdims=keepdims)
        return r

    def ravel(self, order='C'):
        """
        Return a contiguous flattened array.

        A 1-D array, containing the elements of the input, is returned. A copy is
        made only if needed. This method mirrors the behavior of `numpy.ravel`.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The elements of `a` are read using this index order.
            'C' means to index the elements in row-major, C-style order,
            with the last axis index changing fastest, back to the first
            axis index changing slowest. 'F' means to index the elements
            in column-major, Fortran-style order, with the first index
            changing fastest, and the last index changing slowest.
            'A' means to read the elements in Fortran-like index order if `a`
            is Fortran *contiguous* in memory, C-like order otherwise.
            'K' means to read the elements in the order they occur in memory,
            except for reversing the data when strides are negative.
            Default is 'C'.

        Returns
        -------
        ndarray
            A 1-D array, containing the elements of the input.

        See Also
        --------
        numpy.ravel : Equivalent NumPy function.
        jax.numpy.ravel : Equivalent JAX function.
        flatten : Return a copy of the array collapsed into one dimension.
        reshape : Change the shape of an array.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> x = EventArray(jnp.array([[1, 2, 3], [4, 5, 6]]))
        >>> x.ravel()
        Array([1, 2, 3, 4, 5, 6], dtype=int32)

        >>> x.ravel(order='F') # Fortran order (column-major)
        Array([1, 4, 2, 5, 3, 6], dtype=int32)
        """
        return self.value.ravel(order=order)

    def repeat(self, repeats, axis=None):
        """
        Repeat elements of an array.

        Returns a new array which contains the specified number of repetitions of
        the elements along the given axis. This method mirrors the behavior of
        `numpy.repeat`.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. `repeats` is broadcasted
            to fit the shape of the given axis.
        axis : int, optional
            The axis along which to repeat values. By default (`axis=None`), use
            the flattened input array, and return a flat output array.

        Returns
        -------
        ndarray
            Output array which has the same shape as `self.value`, except along
            the given axis. If `axis` is None, the output is a flattened array.

        See Also
        --------
        numpy.repeat : Equivalent NumPy function.
        jax.numpy.repeat : Equivalent JAX function.
        tile : Construct an array by repeating an array.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from brainevent import EventArray
        >>> x = EventArray(jnp.array([[1, 2], [3, 4]]))

        >>> x.repeat(2) # Repeat elements of flattened array
        Array([1, 1, 2, 2, 3, 3, 4, 4], dtype=int32)

        >>> x.repeat(3, axis=1) # Repeat columns
        Array([[1, 1, 1, 2, 2, 2],
               [3, 3, 3, 4, 4, 4]], dtype=int32)

        >>> x.repeat(2, axis=0) # Repeat rows
        Array([[1, 2],
               [1, 2],
               [3, 4],
               [3, 4]], dtype=int32)

        >>> x.repeat(jnp.array([1, 2]), axis=0) # Repeat first row once, second row twice
        Array([[1, 2],
               [3, 4],
               [3, 4]], dtype=int32)
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
        """
        Change the shape of the array, returning a new array or modifying in-place.

        For mutable arrays (like NumPy ndarray), this operation might be in-place.
        For immutable arrays (like JAX Array), this operation returns a new
        array with the specified shape, and reassigns `self.value`. This method
        currently uses `reshape`, which always returns a new array (or view).

        Parameters
        ----------
        new_shape : int or tuple of ints
            Shape of the resized array. The total number of elements must remain
            the same.

        Returns
        -------
        None
            This method modifies `self.value` by assigning the reshaped array to it.

        Raises
        ------
        ValueError
            If the new shape is not compatible with the original shape (i.e.,
            the total number of elements differs).

        See Also
        --------
        numpy.reshape : Returns an array containing the same data with a new shape.
        numpy.resize : Return a new array with the specified shape.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.arange(6)
        >>> ea = EventArray(a)
        >>> print(ea.shape)
        (6,)
        >>> ea.resize((2, 3)) # Reassigns self.value
        >>> print(ea.shape)
        (2, 3)
        >>> print(ea)
        BaseArray(value=array([[0, 1, 2],
               [3, 4, 5]]), dtype=int32)

        # Note: Unlike np.resize, this doesn't change the total size
        # >>> ea.resize((3, 3)) # This would raise a ValueError
        """
        # Note: JAX arrays are immutable. `reshape` returns a new array.
        # We reassign self.value to mimic in-place modification conceptually.
        # This differs from np.ndarray.resize which can change the size and
        # potentially modify in-place.
        self.value = self.value.reshape(new_shape)

    def round(self, decimals=0):
        """
        Return the array with each element rounded to the given number of decimals.

        Rounds elements of the array to the nearest integer or to the specified
        number of decimal places.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to (default: 0). If decimals is
            negative, it specifies the number of positions to the left of the
            decimal point.

        Returns
        -------
        ndarray
            An array of the same type as the input array, containing the rounded
            values. Note that for complex numbers, the real and imaginary parts
            are rounded separately.

        See Also
        --------
        numpy.around : Equivalent function.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([0.37, 1.64, 0.5])
        >>> ea = EventArray(a)
        >>> ea.round()
        BaseArray(value=array([0., 2., 0.]), dtype=float64)
        >>> ea.round(decimals=1)
        BaseArray(value=array([0.4, 1.6, 0.5]), dtype=float64)
        >>> ea.round(decimals=-1)
        BaseArray(value=array([0., 0., 0.]), dtype=float64)

        >>> b = np.array([12.34, 98.76])
        >>> eb = EventArray(b)
        >>> eb.round(decimals=-1)
        BaseArray(value=array([ 10., 100.]), dtype=float64)
        """
        # Delegates directly to the underlying array's round method.
        return self.value.round(decimals=decimals)

    def searchsorted(self, v, side='left', sorter=None):
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `a` (self.value) such that, if the
        corresponding elements in `v` were inserted before the indices, the
        order of `a` would be preserved.

        Refer to `numpy.searchsorted` for full documentation.

        Parameters
        ----------
        v : array_like
            Values to insert into the array. Can be a scalar or array-like,
            including `BaseArray`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index. If there is no suitable
            index, return either 0 or N (where N is the length of `a`).
        sorter : 1-D array_like, optional
            Optional array of integer indices that sort the array into ascending
            order. They are typically the result of `argsort`.

        Returns
        -------
        indices : ndarray of ints
            Array of insertion points with the same shape as `v`.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([1, 2, 3, 4, 5])
        >>> ea = EventArray(a) # Assumes ea.value is sorted
        >>> ea.searchsorted(3)
        Array(2, dtype=int32)
        >>> ea.searchsorted([0, 6])
        Array([0, 5], dtype=int32)
        >>> ea.searchsorted(3, side='right')
        Array(3, dtype=int32)

        >>> b = np.array([0, 1, 1, 2, 3, 5, 8])
        >>> eb = EventArray(b)
        >>> indices_to_insert = EventArray(np.array([-1, 1, 4, 8]))
        >>> eb.searchsorted(indices_to_insert)
        Array([0, 1, 4, 6], dtype=int32)
        """
        # Ensure 'v' is unwrapped if it's an EventArray
        # Delegates to the underlying array's searchsorted method.
        v = extract_raw_value(v)
        return self.value.searchsorted(v=u.math.asarray(v), side=side, sorter=sorter)

    def sort(self, axis=-1, stable=True, order=None):
        """
        Sort the array, returning a new sorted array and updating `self.value`.

        Note: Unlike `numpy.ndarray.sort` which sorts in-place, this method
        (especially when backed by immutable arrays like JAX arrays) returns a
        *new* sorted array and reassigns `self.value` to this new array.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the
            last axis. If None, the array is flattened before sorting.
        stable : bool, optional
            Whether to use a stable sorting algorithm (preserves the order of
            equal elements). Default is True. JAX `sort` uses a stable sort.
        order : str or list of str, optional
            When the array has fields defined (structured arrays), this argument
            specifies which fields to compare first, second, etc. Not applicable
            to standard numeric arrays.

        Returns
        -------
        None
            This method modifies `self.value` by assigning the sorted array to it.

        See Also
        --------
        numpy.sort : Return a sorted copy of an array.
        numpy.argsort : Return the indices that would sort an array.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([[1, 4], [3, 1]])
        >>> ea = EventArray(a)
        >>> ea.sort(axis=1) # Sort each row
        >>> print(ea)
        BaseArray(value=array([[1, 4],
               [1, 3]]), dtype=int32)

        >>> b = np.array([3, 1, 4, 1, 5, 9])
        >>> eb = EventArray(b)
        >>> eb.sort() # Sort the flattened array
        >>> print(eb)
        BaseArray(value=array([1, 1, 3, 4, 5, 9]), dtype=int32)
        """
        # Note: JAX arrays are immutable. `sort` returns a new array.
        # We reassign self.value to the sorted result.
        # The 'stable' argument aligns with JAX's default stable sort.
        # NumPy's ndarray.sort has `kind` instead of `stable`.
        # We use `stable` for potential JAX compatibility.
        # TODO: Consider aligning parameter names/behavior more closely
        #       if strict NumPy ndarray.sort compatibility is needed.
        self.value = self.value.sort(axis=axis, stable=stable, order=order)

    def squeeze(self, axis=None):
        """
        Remove axes of length one from the array.

        Returns a new array (or view) with single-dimensional entries removed
        from the shape.

        Refer to `numpy.squeeze` for full documentation.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Selects a subset of the single-dimensional entries in the shape.
            If an axis is selected with shape entry greater than one, an error
            is raised. If None (default), all single-dimensional entries will
            be removed from the shape.

        Returns
        -------
        ndarray
            The input array with all or a subset of the dimensions of length 1
            removed. This is often a view of the input array, but may be a copy
            if required by the backend (e.g., JAX).

        Raises
        ------
        ValueError
            If `axis` is specified and selects an axis whose dimension is not 1.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> x = np.array([[[0], [1], [2]]])
        >>> x.shape
        (1, 3, 1)
        >>> ex = EventArray(x)
        >>> ex.squeeze().shape
        (3,)
        >>> ex.squeeze(axis=0).shape
        (3, 1)
        >>> ex.squeeze(axis=2).shape
        (1, 3)
        >>> ex.squeeze(axis=(0, 2)).shape
        (3,)

        # Squeezing an axis that is not 1 raises an error
        # >>> ex.squeeze(axis=1) # Raises ValueError
        """
        # Delegates directly to the underlying array's squeeze method.
        return self.value.squeeze(axis=axis)

    def std(self, axis=None, dtype=None, ddof=0, keepdims=False):
        """
        Compute the standard deviation along the specified axis.

        Returns the standard deviation, a measure of the spread of a distribution,
        of the array elements. The standard deviation is computed for the
        flattened array by default, otherwise over the specified axis.

        Refer to `numpy.std` for full documentation.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the standard deviation is computed. The
            default is to compute the standard deviation of the flattened array.
        dtype : data-type, optional
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float64; for arrays of float types it is
            the same as the array type.
        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is
            N - ddof, where N represents the number of elements. By default
            `ddof` is zero.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
            Default is False.

        Returns
        -------
        standard_deviation : ndarray
            A new array containing the standard deviation, or a scalar if `axis`
            is None.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([[1, 2], [3, 4]])
        >>> ea = EventArray(a)
        >>> ea.std()
        1.118033988749895
        >>> ea.std(axis=0)
        array([1., 1.])
        >>> ea.std(axis=1)
        array([0.5, 0.5])
        >>> ea.std(ddof=1) # Bessel's correction
        1.2909944487358056
        """
        # Optimized: Directly return the result
        return self.value.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)

    def sum(self, axis=None, dtype=None, keepdims=False, initial=0, where=True):
        """
        Return the sum of the array elements over the given axis.

        Refer to `numpy.sum` for full documentation.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which a sum is performed. The default, axis=None,
            will sum all of the elements of the input array. If axis is negative it
            counts from the last to the first axis. If axis is a tuple of ints,
            a sum is performed on all of the axes specified in the tuple instead
            of a single axis or all the axes as before.
        dtype : data-type, optional
            The type of the returned array and of the accumulator in which the
            elements are summed. The dtype of the array is used by default unless
            the array has an integer dtype of less precision than the default
            platform integer. In that case, if the array is signed then the
            platform integer is used, and if the array is unsigned then an
            unsigned integer of the same precision as the platform integer is used.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
            Default is False.
        initial : scalar, optional
            Starting value for the sum. Default is 0.
        where : array_like of bool, optional
            Elements to include in the sum. See `numpy.ufunc.reduce` for details.

        Returns
        -------
        sum_along_axis : ndarray
            An array with the same shape as self.value, with the specified axis
            removed. If self.value is a 0-d array, or if axis is None, a scalar
            is returned. If an output array is specified, a reference to it is
            returned.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([[0, 1], [0, 5]])
        >>> ea = EventArray(a)
        >>> ea.sum()
        6
        >>> ea.sum(axis=0)
        array([0, 6])
        >>> ea.sum(axis=1)
        array([1, 5])
        >>> ea.sum(initial=10)
        16
        >>> ea.sum(where=[True, False]) # Sum only the first column
        0
        """
        # Optimized: Directly return the result
        return self.value.sum(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)

    def swapaxes(self, axis1, axis2):
        """
        Return a view of the array with `axis1` and `axis2` interchanged.

        Refer to `numpy.swapaxes` for full documentation.

        Parameters
        ----------
        axis1 : int
            First axis.
        axis2 : int
            Second axis.

        Returns
        -------
        swapped_axes_array : ndarray
            If the underlying array is a view, then the returned array is a view
            of the original data. Otherwise, it is a copy.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> x = np.array([[1, 2, 3]])
        >>> ex = EventArray(x)
        >>> ex.swapaxes(0, 1)
        BaseArray(value=array([[1],
               [2],
               [3]]), dtype=int32)

        >>> y = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        >>> ey = EventArray(y)
        >>> ey.swapaxes(0, 2)
        BaseArray(value=array([[[0, 4],
                [2, 6]],
        <BLANKLINE>
               [[1, 5],
                [3, 7]]]), dtype=int32)
        """
        return self.value.swapaxes(axis1, axis2)

    def split(self, indices_or_sections, axis=0):
        """
        Split an array into multiple sub-arrays.

        Refer to `numpy.split` for full documentation.

        Parameters
        ----------
        indices_or_sections : int or 1-D array
            If `indices_or_sections` is an integer, N, the array will be divided
            into N equal arrays along `axis`. If such a split is not possible,
            an error is raised.
            If `indices_or_sections` is a 1-D array of sorted integers, the entries
            indicate where along `axis` the array is split. For example,
            `[2, 3]` would, for `axis=0`, result in `ary[:2]`, `ary[2:3]`,
            and `ary[3:]`. If an index exceeds the dimension of the array along
            `axis`, an empty sub-array is returned correspondingly.
        axis : int, optional
            The axis along which to split, default is 0.

        Returns
        -------
        sub_arrays : list of ndarry
            A list of sub-arrays. Each sub-array is an BaseArray wrapping
            a view into the original array's data.

        Raises
        ------
        ValueError
            If `indices_or_sections` is an integer that does not result in an
            equal division.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> x = np.arange(9.0)
        >>> ex = EventArray(x)
        >>> ex.split(3)
        [BaseArray(value=array([0., 1., 2.]), dtype=float64), BaseArray(value=array([3., 4., 5.]), dtype=float64), BaseArray(value=array([6., 7., 8.]), dtype=float64)]
        >>> ex.split([3, 5, 6, 10])
        [BaseArray(value=array([0., 1., 2.]), dtype=float64), BaseArray(value=array([3., 4.]), dtype=float64), BaseArray(value=array([5.]), dtype=float64), BaseArray(value=array([6., 7., 8.]), dtype=float64), BaseArray(value=array([], dtype=float64), dtype=float64)]

        >>> y = np.arange(8.0).reshape(2, 4)
        >>> ey = EventArray(y)
        >>> ey.split(2, axis=1)
        [array([[0., 1.],
               [4., 5.]]), dtype=float64), 
         array([[2., 3.],
               [6., 7.]]), dtype=float64)]
        """
        # Wrap results in EventArray
        return [a for a in u.math.split(self.value, indices_or_sections, axis=axis)]

    def take(self, indices, axis=None, mode=None):
        """
        Return an array formed from the elements of self at the given indices.

        Refer to `numpy.take` for full documentation.

        Parameters
        ----------
        indices : array_like
            The indices of the values to extract. Also allows BaseArray instances.
        axis : int, optional
            The axis over which to select values. By default, the flattened
            input array is used.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will be treated.
            * 'raise' : raise an error (default)
            * 'wrap' : wrap around
            * 'clip' : clip to the range

        Returns
        -------
        subarray : ndarray
            The returned array has the same type as `self.value`.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([4, 3, 5, 7, 6, 8])
        >>> ea = EventArray(a)
        >>> indices = [0, 1, 4]
        >>> ea.take(indices)
        array([4, 3, 6])

        >>> b = np.array([[1, 2], [3, 4]])
        >>> eb = EventArray(b)
        >>> eb.take([0, 1], axis=1)
        BaseArray(value=array([[1, 2],
               [3, 4]]), dtype=int32)
        >>> eb.take([0, 1, 2], axis=1, mode='wrap') # Wrap around indices
        BaseArray(value=array([[1, 2, 1],
               [3, 4, 3]]), dtype=int32)
        """
        return self.value.take(indices=extract_raw_value(indices), axis=axis, mode=mode)

    def tobytes(self, order='C'):
        """
        Construct Python bytes containing the raw data bytes in the array.

        Constructs Python bytes showing a copy of the raw contents of data memory.

        Refer to `numpy.ndarray.tobytes` for full documentation.

        Parameters
        ----------
        order : {'C', 'F', None}, optional
            Order of the data for multidimensional arrays:
            'C' means C-order, 'F' means Fortran-order, None means 'C' unless the
            array is Fortran contiguous, then 'F'. Default is 'C'.

        Returns
        -------
        s : bytes
            Python bytes exhibiting a copy of the array's raw data.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> x = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        >>> ex = EventArray(x)
        >>> ex.tobytes()
        b'\\x00\\x01\\x02\\x03'
        >>> ex.tobytes(order='F')
        b'\\x00\\x02\\x01\\x03'
        """
        return self.value.tobytes(order=order)

    def tolist(self):
        """
        Return the array as an `a.ndim`-levels deep nested list of Python scalars.

        Return a copy of the array data as a (nested) Python list.
        Data items are converted to the nearest compatible builtin Python type.

        Refer to `numpy.ndarray.tolist` for full documentation.

        Returns
        -------
        y : list or scalar
            The possibly nested list of array elements.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([1, 2])
        >>> ea = EventArray(a)
        >>> ea.tolist()
        [1, 2]

        >>> b = np.array([[1, 2], [3, 4]])
        >>> eb = EventArray(b)
        >>> eb.tolist()
        [[1, 2], [3, 4]]

        >>> c = np.array(5)
        >>> ec = EventArray(c)
        >>> ec.tolist()
        5
        """
        return self.value.tolist()

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        """
        Return the sum along diagonals of the array.

        Refer to `numpy.trace` for full documentation.

        Parameters
        ----------
        offset : int, optional
            Offset of the diagonal from the main diagonal. Can be positive or
            negative. Defaults to 0.
        axis1, axis2 : int, optional
            Axes to be used as the first and second axis of the 2-D sub-arrays
            from which the diagonals should be taken. Defaults are 0 and 1,
            respectively.
        dtype : data-type, optional
            Determines the data-type of the returned array and of the accumulator
            where the elements are summed. If dtype has value None, the dtype
            is determined as the default dtype of the array.

        Returns
        -------
        sum_along_diagonals : ndarray
            If the array is 2-D, the sum along the diagonal is returned. If the
            array has more than two dimensions, then the sums along diagonals
            are calculated for the 2-D sub-arrays defined by `axis1` and `axis2`.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> ea = EventArray(a)
        >>> ea.trace()
        15
        >>> ea.trace(offset=1)
        5
        >>> ea.trace(offset=-1)
        12
        """
        return self.value.trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    def transpose(self, *axes):
        """
        Returns a view of the array with axes transposed.

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

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([[1, 2], [3, 4]])
        >>> ea = EventArray(a)
        >>> ea.transpose()
        BaseArray(value=array([[1, 3],
               [2, 4]]), dtype=int32)
        >>> ea.transpose((1, 0))
        BaseArray(value=array([[1, 3],
               [2, 4]]), dtype=int32)

        >>> b = np.array([1, 2, 3, 4])
        >>> eb = EventArray(b)
        >>> eb.transpose() # 1-D array is unaffected
        BaseArray(value=array([1, 2, 3, 4]), dtype=int32)

        >>> c = np.arange(16).reshape((2, 2, 4))
        >>> ec = EventArray(c)
        >>> ec.transpose((1, 0, 2))
        BaseArray(value=array([[[ 0,  1,  2,  3],
                [ 8,  9, 10, 11]],
        <BLANKLINE>
               [[ 4,  5,  6,  7],
                [12, 13, 14, 15]]]), dtype=int32)
        >>> ec.transpose(2, 0, 1)
        BaseArray(value=array([[[ 0,  4],
                [ 8, 12]],
        <BLANKLINE>
               [[ 1,  5],
                [ 9, 13]],
        <BLANKLINE>
               [[ 2,  6],
                [10, 14]],
        <BLANKLINE>
               [[ 3,  7],
                [11, 15]]]), dtype=int32)
        """
        return self.value.transpose(*axes)

    def tile(self, reps):
        """
        Construct an array by repeating the elements of this array.

        The number of repetitions is specified by `reps`.

        Parameters
        ----------
        reps : array_like
            The number of repetitions of `self.value` along each axis.

        Returns
        -------
        c : ndarray
            The tiled output array.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> import brainunit as u # Assuming brainunit is imported as u
        >>> a = np.array([0, 1, 2])
        >>> ea = EventArray(a)
        >>> ea.tile(2)
        BaseArray(value=array([0, 1, 2, 0, 1, 2]), dtype=int32)
        >>> ea.tile((2, 2))
        BaseArray(value=array([[0, 1, 2, 0, 1, 2],
               [0, 1, 2, 0, 1, 2]]), dtype=int32)
        >>> ea.tile((2, 1, 2))
        BaseArray(value=array([[[0, 1, 2, 0, 1, 2]],
        <BLANKLINE>
               [[0, 1, 2, 0, 1, 2]]]), dtype=int32)

        >>> b = np.array([[1, 2], [3, 4]])
        >>> eb = EventArray(b)
        >>> eb.tile(2)
        BaseArray(value=array([[1, 2, 1, 2],
               [3, 4, 3, 4]]), dtype=int32)
        >>> eb.tile((2, 1))
        BaseArray(value=array([[1, 2],
               [3, 4],
               [1, 2],
               [3, 4]]), dtype=int32)
        """
        # Use u.math.tile to support both numpy and jax backends if needed
        # Ensure _as_array is available or defined in the scope
        return u.math.tile(self.value, extract_raw_value(reps))

    def var(self, axis=None, dtype=None, ddof=0, keepdims=False):
        """
        Returns the variance of the array elements, along given axis.

        Refer to `numpy.var` for full documentation.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the variance is computed. The default is to
            compute the variance of the flattened array.
        dtype : data-type, optional
            Type to use in computing the variance. For arrays of integer type
            the default is float64; for arrays of float types it is the same as
            the array type.
        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is
            N - ddof, where N represents the number of elements. By default
            `ddof` is zero.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
            Default is False.

        Returns
        -------
        variance : ndarray, see dtype parameter above
            If the default `keepdims` is used, then the return value is the
            variance of the array elements along the specified axis. If
            `keepdims` is True, the result will broadcast correctly against the
            input array.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray
        >>> a = np.array([[1, 2], [3, 4]])
        >>> ea = EventArray(a)
        >>> ea.var()
        1.25
        >>> ea.var(axis=0)
        array([1., 1.])
        >>> ea.var(axis=1)
        array([0.25, 0.25])
        >>> ea.var(ddof=1) # Bessel's correction
        1.6666666666666667
        """
        # Optimized: Directly return the result without intermediate variable 'r'
        return self.value.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)

    def view(self, *args, dtype=None):
        """
        Return a new view of the array with specified dtype or shape.

        This method provides functionality similar to `numpy.ndarray.view` for
        changing the data type interpretation or `numpy.ndarray.reshape` for
        changing the shape, depending on the arguments provided.

        - If only `dtype` is provided, it returns a view of the array data
          interpreted as the specified `dtype`. This is analogous to
          `numpy.ndarray.view(dtype=...)`.
        - If positional arguments (`*args`) are provided (as integers or a
          single tuple of integers), they are interpreted as the desired new
          shape, and the method returns a reshaped view of the array. This is
          analogous to `numpy.ndarray.reshape(...)`.

        Parameters
        ----------
        *args : int or tuple of ints, optional
            The desired new shape. Can be specified as multiple integer arguments
            (e.g., `view(2, 3)`) or a single tuple argument (e.g., `view((2, 3))`).
            If provided, `dtype` must be None.
        dtype : data-type, optional
            The desired data type for the view. If provided, `*args` must not be
            given.

        Returns
        -------
        ndarray
            A new view of the array with the specified data type or shape.

        Raises
        ------
        ValueError
            If neither `dtype` nor shape arguments (`*args`) are provided.
        ValueError
            If both `dtype` and shape arguments (`*args`) are provided.

        Examples
        --------
        >>> import numpy as np
        >>> from brainevent import EventArray

        # View with a new shape (using reshape behavior)
        >>> a = np.arange(6)
        >>> ea = EventArray(a)
        >>> ea.view(2, 3)
        BaseArray(value=array([[0, 1, 2],
               [3, 4, 5]]), dtype=int32)
        >>> ea.view((6,))
        BaseArray(value=array([0, 1, 2, 3, 4, 5]), dtype=int32)

        # View with a new dtype
        >>> x = np.array([(1, 2), (3, 4)], dtype=[('a', np.int8), ('b', np.int8)])
        >>> ex = EventArray(x)
        >>> # View as float32 (assuming compatible byte size)
        >>> # Note: Behavior depends on underlying array implementation (NumPy/JAX)
        >>> # and byte compatibility. Example assumes standard NumPy behavior.
        >>> try:
        ...     ex.view(dtype=np.float32) # This might fail if sizes don't match
        ... except TypeError as e:
        ...     print(f"TypeError: {e}") # JAX might raise TypeError
        BaseArray(value=array([[-1.5881868e+22,  1.1028099e-38]], dtype=float32)

        >>> # View as a simple int16 array
        >>> ex.view(dtype=np.int16)
        BaseArray(value=array([[1, 2],
               [3, 4]], dtype=int16)
        """
        if not args:
            # Case 1: Only dtype is potentially provided
            if dtype is None:
                raise ValueError("Provide dtype or shape arguments.")
            # Return view with new dtype
            return self.value.view(dtype)
        else:
            # Case 2: Positional arguments (*args) are provided (interpreted as shape)
            if dtype is not None:
                raise ValueError("Provide either dtype or shape arguments, not both.")

            # Determine the shape from *args
            if len(args) == 1 and isinstance(args[0], tuple):
                shape = args[0]
            elif all(isinstance(arg, int) for arg in args):
                shape = args
            else:
                # Handle potential case where args[0] is a dtype-like object but not int
                # This part of the original logic seemed ambiguous; clarifying based on intent.
                # If the first arg is not an int and not a tuple, assume it's intended as dtype
                # (though the outer check `if not args:` should ideally catch the dtype-only case)
                if len(args) == 1 and not isinstance(args[0], int):
                    # Re-interpret as dtype view if args[0] looks like a dtype
                    # This aligns with the original code's final `else` block's intent
                    # but requires careful consideration of allowed types for args[0]
                    # For simplicity and robustness, strictly separating shape and dtype args is better.
                    # The current logic assumes if args exist, they define shape.
                    # Let's stick to interpreting *args strictly as shape here.
                    raise ValueError("Shape arguments must be integers or a single tuple of integers.")

            # Return view with new shape (using reshape)
            # Note: This differs from np.view's byte reinterpretation when shape changes.
            return self.value.reshape(*shape)

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
        Insert a dimension of size 1 at the specified position.

        This is a convenience method equivalent to `expand_dims()` that matches PyTorch's API.

        Parameters
        ----------
        dim : int
            Position where to insert the new dimension.
            Negative dims count from the end.

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A view of the array with an additional dimension inserted.

        Examples
        --------
        >>> a = EventArray([1, 2, 3])  # Shape: (3,)
        >>> a.unsqueeze(0).shape  # Shape: (1, 3)
        (1, 3)
        >>> a.unsqueeze(1).shape  # Shape: (3, 1)
        (3, 1)

        See Also
        --------
        expand_dims : Equivalent functionality with NumPy-like API
        """
        return u.math.expand_dims(self.value, dim)

    def expand_dims(self, axis: Union[int, Sequence[int]]) -> Union[jax.Array, u.Quantity]:
        """
        Insert new dimensions at the specified positions.

        Parameters
        ----------
        axis : int or sequence of ints
            Position(s) where to insert new dimension(s).
            For a single integer, inserts one new dimension at that position.
            For a sequence, inserts multiple new dimensions at the specified positions.
            Negative axes count from the end and are converted to positive axes.
            For an n-dimensional array, valid axes are in range [-(n+1), n].

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A view of the array with additional dimension(s) inserted.

        Examples
        --------
        >>> a = EventArray([1, 2, 3])  # Shape: (3,)
        >>> a.expand_dims(0).shape  # Shape: (1, 3)
        (1, 3)
        >>> a a.expand_dims([0, 2]).shape  # Shape: (1, 3, 1)
        (1, 3, 1)

        Notes
        -----
        When applying multiple axes at once, they are inserted in the order specified,
        which affects the final shape.

        See Also
        --------
        unsqueeze : PyTorch-style equivalent for a single dimension
        """
        return u.math.expand_dims(self.value, axis)

    def expand_as(self, array: Union['BaseArray', ArrayLike]) -> 'BaseArray':
        """
        Expand this array to match the shape of another array through broadcasting.

        Parameters
        ----------
        array : BaseArray or ArrayLike
            The array whose shape will be used as the target shape.

        Returns
        -------
        EventArray
            A new LowBitArray with the expanded shape. This is a view of the original
            data when possible.

        Examples
        --------
        >>> a = EventArray([1, 2, 3])  # Shape: (3,)
        >>> b = EventArray([[0, 0], [0, 0], [0, 0]])  # Shape: (3, 2)
        >>> a.expand_as(b).shape  # Shape: (3, 2)
        (3, 2)

        Notes
        -----
        The resulting array is a read-only view on the original array.
        Multiple elements may refer to the same memory location.

        Raises
        ------
        ValueError
            If the arrays are not broadcastable to the target shape.

        See Also
        --------
        numpy.broadcast_to : The underlying NumPy function
        """
        target_array = extract_raw_value(array)
        result = u.math.broadcast_to(self.value, u.math.shape(target_array))
        return type(self)(result)  # Wrap in BaseArray to return correct type

    def pow(self, index: Union[int, float, ArrayLike]) -> Union[jax.Array, u.Quantity]:
        """
        Compute element-wise power of the array to the given exponent.

        Parameters
        ----------
        index : int, float, or array-like
            The exponent value(s). If array-like, broadcasting rules apply.

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The result of raising each element to the given power.

        Examples
        --------
        >>> a = EventArray([1, 2, 3, 4])
        >>> a.pow(2)
        BaseArray([1, 4, 9, 16])
        >>> a.pow([2, 3, 2, 3])
        BaseArray([1, 8, 9, 64])

        See Also
        --------
        __pow__ : The special method that implements the ** operator
        """
        return self.value ** extract_raw_value(index)

    def addr(
        self,
        vec1: Union['BaseArray', ArrayLike],
        vec2: Union['BaseArray', ArrayLike],
        *,
        beta: float = 1.0,
        alpha: float = 1.0,
    ) -> Union['BaseArray', u.Quantity, jax.Array, None]:
        r"""
        Perform the outer product of vectors and add to this matrix.

        Computes: out = beta * self + alpha * (vec1 ⊗ vec2)

        This operation performs the weighted outer product between vec1 and vec2,
        scales it by alpha, and adds it to beta times this array.

        Parameters
        ----------
        vec1 : BaseArray or ArrayLike
            The first vector of the outer product.
        vec2 : BaseArray or ArrayLike
            The second vector of the outer product.
        beta : float, default=1.0
            The multiplier for this array.
        alpha : float, default=1.0
            The multiplier for the outer product result.

        Returns
        -------
        Union[EventArray, u.Quantity, jax.Array, None]
            The result of the operation. If out is provided, returns None.

        Examples
        --------
        >>> a = EventArray([[1, 2], [3, 4]])
        >>> x = EventArray([1, 2])
        >>> y = EventArray([3, 4])
        >>> a.addr(x, y, alpha=1.0, beta=1.0)
        LowBitArray([[ 4, 9],
                    [ 9, 17]])

        Notes
        -----
        The outer product of two vectors x and y is defined as the matrix M where
        M[i,j] = x[i] * y[j].

        See Also
        --------
        addr_ : In-place version of this method
        outer : Compute just the outer product without adding to this array
        """
        vec1 = extract_raw_value(vec1)
        vec2 = extract_raw_value(vec2)
        r = alpha * u.math.outer(vec1, vec2) + beta * self.value
        return r

    def outer(
        self,
        other: Union['BaseArray', ArrayLike]
    ) -> 'BaseArray':
        """
        Compute the outer product with another array.

        Parameters
        ----------
        other : BaseArray or ArrayLike
            The array to compute the outer product with.

        Returns
        -------
        EventArray
            A new array containing the outer product.

        Examples
        --------
        >>> a = EventArray([1, 2, 3])
        >>> b = EventArray([4, 5])
        >>> a.outer(b)
        LowBitArray([[ 4,  5],
                    [ 8, 10],
                    [12, 15]])

        Notes
        -----
        The outer product of two vectors x and y results in a matrix M where
        M[i,j] = x[i] * y[j].

        See Also
        --------
        addr : Compute outer product and add to an existing array
        numpy.outer : Similar NumPy function
        """
        other = extract_raw_value(other)
        return type(self)(u.math.outer(self.value, other))

    def abs(self) -> Union['BaseArray', u.Quantity, jax.Array, None]:
        """
        Calculate the absolute value element-wise.

        Returns
        -------
        Union[EventArray, u.Quantity, jax.Array, None]
            A new array with the absolute value of each element.
            If out is provided, returns None.

        Examples
        --------
        >>> a = EventArray([-1, -2, 3])
        >>> a.abs()
        BaseArray([1, 2, 3])

        >>> # Using out parameter
        >>> result = EventArray(np.zeros(3))
        >>> a.abs(out=result)
        >>> result
        BaseArray([1, 2, 3])

        See Also
        --------
        abs_ : In-place version
        absolute : Alias for this function
        numpy.abs : NumPy equivalent function
        """
        r = u.math.abs(self.value)
        return r

    def abs_(self) -> 'BaseArray':
        """
        Calculate the absolute value element-wise in-place.

        Modifies the array in-place to contain the absolute values.

        Returns
        -------
        EventArray
            Self, after taking the absolute value of each element.

        Examples
        --------
        >>> a = EventArray([-1, -2, 3])
        >>> a.abs_()  # Modifies a in-place
        BaseArray([1, 2, 3])

        See Also
        --------
        abs : Non-in-place version
        absolute_ : Alias for this function
        """
        self.value = u.math.abs(self.value)
        return self

    def absolute(self) -> Union['BaseArray', jax.Array, u.Quantity]:
        """
        Calculate the absolute value element-wise.

        This is an alias for the `abs` method.

        Returns
        -------
        Union[BaseArray, jax.Array, u.Quantity]
            A new array with the absolute value of each element.
            If out is provided, returns None.

        Examples
        --------
        >>> a = EventArray([-1, -2, 3])
        >>> a.absolute()
        LowBitArray([1, 2, 3])

        See Also
        --------
        abs : Equivalent function
        absolute_ : In-place version
        """
        return self.abs()

    def absolute_(self) -> 'BaseArray':
        """
        Calculate the absolute value element-wise in-place.

        This is an alias for the `abs_` method.

        Returns
        -------
        BaseArray
            Self, after taking the absolute value of each element.

        Examples
        --------
        >>> a = EventArray([-1, -2, 3])
        >>> a.absolute_()  # Modifies a in-place
        LowBitArray([1, 2, 3])

        See Also
        --------
        abs_ : Equivalent function
        absolute : Non-in-place version
        """
        return self.abs_()

    def mul(self, value: Union['BaseArray', ArrayLike]) -> Union[jax.Array, u.Quantity]:
        """
        Multiply the array by a scalar or array element-wise.

        Parameters
        ----------
        value : BaseArray or ArrayLike
            The value to multiply with this array.

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A new array with the result of the multiplication.

        Examples
        --------
        >>> a = EventArray([1, 2, 3])
        >>> a.mul(10)
        LowBitArray([10, 20, 30])

        >>> a.mul(EventArray([2, 3, 4]))
        LowBitArray([2, 6, 12])

        See Also
        --------
        mul_ : In-place version
        multiply : Alias for this function
        __mul__ : The special method that implements the * operator
        """
        return self.value * extract_raw_value(value)

    def multiply(self, value: Union['BaseArray', ArrayLike]) -> Union[jax.Array, u.Quantity]:
        """
        Multiply the array by a scalar or array element-wise.

        This is an alias for the `mul` method, providing PyTorch-compatible API.

        Parameters
        ----------
        value : BaseArray or ArrayLike
            The value to multiply with this array.

        Returns
        -------
        Union[jax.Array, u.Quantity]
            A new array with the result of the multiplication.

        Examples
        --------
        >>> a = EventArray([1, 2, 3])
        >>> a.multiply(10)
        LowBitArray([10, 20, 30])

        See Also
        --------
        mul : Equivalent function
        multiply_ : In-place version
        """
        return self.value * extract_raw_value(value)

    def sin(self) -> Union[u.Quantity, jax.Array, None]:
        """
        Calculate the sine of the array elements.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            Sine of each element. Returns None if out is provided.

        See Also
        --------
        sin_ : In-place version of this function.
        cos, tan : Other trigonometric functions.
        """
        r = u.math.sin(self.value)
        return r

    def cos(self) -> Union[u.Quantity, jax.Array, None]:
        """
        Calculate the cosine of the array elements.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            Cosine of each element. Returns None if out is provided.

        See Also
        --------
        cos_ : In-place version of this function.
        sin, tan : Other trigonometric functions.
        """
        r = u.math.cos(self.value)
        return r

    def tan(self) -> Union[u.Quantity, jax.Array, None]:
        """
        Calculate the tangent of the array elements.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            Tangent of each element. Returns None if out is provided.

        See Also
        --------
        tan_ : In-place version of this function.
        sin, cos : Other trigonometric functions.
        """
        r = u.math.tan(self.value)
        return r

    def sinh(self) -> Union[u.Quantity, jax.Array, None]:
        """
        Calculate the hyperbolic sine of the array elements.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            Hyperbolic sine of each element. Returns None if out is provided.

        See Also
        --------
        sinh_ : In-place version of this function.
        cosh, tanh : Other hyperbolic functions.
        """
        r = u.math.sinh(self.value)
        return r

    def cosh(self) -> Union[u.Quantity, jax.Array, None]:
        """
        Calculate the hyperbolic cosine of the array elements.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            Hyperbolic cosine of each element. Returns None if out is provided.

        See Also
        --------
        cosh_ : In-place version of this function.
        sinh, tanh : Other hyperbolic functions.
        """
        r = u.math.cosh(self.value)
        return r

    def tanh(self) -> Union[u.Quantity, jax.Array, None]:
        """
        Calculate the hyperbolic tangent of the array elements.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            Hyperbolic tangent of each element. Returns None if out is provided.

        See Also
        --------
        tanh_ : In-place version of this function.
        sinh, cosh : Other hyperbolic functions.
        """
        r = u.math.tanh(self.value)
        return r

    def arcsin(self) -> Union[u.Quantity, jax.Array, None]:
        """
        Calculate the inverse sine of the array elements.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            Inverse sine of each element. Returns None if out is provided.
            For real input, the result is in the interval [-π/2, π/2].

        See Also
        --------
        arcsin_ : In-place version of this function.
        arccos, arctan : Other inverse trigonometric functions.
        """
        r = u.math.arcsin(self.value)
        return r

    def arccos(self) -> Union[u.Quantity, jax.Array, None]:
        """
        Calculate the inverse cosine of the array elements.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            Inverse cosine of each element. Returns None if out is provided.
            For real input, the result is in the interval [0, π].

        See Also
        --------
        arccos_ : In-place version of this function.
        arcsin, arctan : Other inverse trigonometric functions.
        """
        r = u.math.arccos(self.value)
        return r

    def arctan(self) -> Union[u.Quantity, jax.Array, None]:
        """
        Calculate the inverse tangent of the array elements.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            Inverse tangent of each element. Returns None if out is provided.
            For real input, the result is in the interval [-π/2, π/2].

        See Also
        --------
        arctan_ : In-place version of this function.
        arcsin, arccos : Other inverse trigonometric functions.
        """
        r = u.math.arctan(self.value)
        return r

    def clamp(
        self,
        min_value: Optional[Union['BaseArray', ArrayLike]] = None,
        max_value: Optional[Union['BaseArray', ArrayLike]] = None,
    ) -> Union[u.Quantity, jax.Array, None]:
        """
        Clamp (limit) the values in the array between min_value and max_value.

        Given an array and interval [min_value, max_value], any array values outside
        the interval are clipped to the interval edges. For example, if an interval of
        [0, 1] is specified, values smaller than 0 become 0, and values larger than 1
        become 1.

        Parameters
        ----------
        min_value : BaseArray or ArrayLike, optional
            Minimum value. If None, clipping is not performed on lower bound.
        max_value : BaseArray or ArrayLike, optional
            Maximum value. If None, clipping is not performed on upper bound.

        Returns
        -------
        Union[u.Quantity, jax.Array, None]
            An array with the elements of self, but where values < min_value are
            replaced with min_value, and those > max_value with max_value.
            If out is provided, returns None.

        See Also
        --------
        clamp_ : In-place version of this function
        clip_ : Alias for clamp_

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> a = EventArray(jnp.arange(10))
        >>> a.clamp(3, 7)  # all values < 3 become 3, all values > 7 become 7
        array([3, 3, 3, 3, 4, 5, 6, 7, 7, 7])
        >>> a.clamp(None, 7)  # only clip from above
        array([0, 1, 2, 3, 4, 5, 6, 7, 7, 7])
        """
        min_value = extract_raw_value(min_value)
        max_value = extract_raw_value(max_value)
        r = u.math.clip(self.value, min_value, max_value)
        return r

    def clone(self) -> 'BaseArray':
        """
        Return a copy of the array.

        This method creates a new LowBitArray with a copy of the data from the original array.

        Returns
        -------
        BaseArray
            A new LowBitArray containing a copy of the values from this array.

        See Also
        --------
        copy_ : Copy values from another array into this one

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> a = EventArray(jnp.array([1, 2, 3]))
        >>> b = a.clone()
        >>> b.value[0] = 5
        >>> a  # original array is unchanged
        LowBitArray(value=array([1, 2, 3]))
        >>> b  # cloned array is modified
        LowBitArray(value=array([5, 2, 3]))
        """
        return type(self)(self.value.copy())

    def cov_with(
        self,
        y: Optional[Union['BaseArray', ArrayLike]] = None,
        rowvar: bool = True,
        bias: bool = False,
        fweights: Optional[Union['BaseArray', ArrayLike]] = None,
        aweights: Optional[Union['BaseArray', ArrayLike]] = None
    ) -> Union[jax.Array, u.Quantity]:
        """
        Calculate the covariance matrix between this array and another.

        Estimate a covariance matrix, given data and weights.

        Parameters
        ----------
        y : BaseArray or ArrayLike, optional
            An array containing multiple variables and observations.
            If not specified, the covariance is calculated for self.
        rowvar : bool, optional, default=True
            If True, then each row represents a variable, with
            observations in the columns. Otherwise, the relationship
            is transposed: each column represents a variable, while
            the rows contain observations.
        bias : bool, optional, default=False
            If False, normalization is by (N - 1), where N is the number of
            observations given (unbiased estimate). If True, then
            normalization is by N.
        fweights : BaseArray or ArrayLike, optional
            Array of integer frequency weights. The number of times each
            observation vector should be repeated.
        aweights : BaseArray or ArrayLike, optional
            Array of observation vector weights. These relative weights are
            typically large for observations considered "important" and smaller
            for observations considered less "important".

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The covariance matrix of the variables.

        See Also
        --------
        numpy.cov : NumPy's covariance function

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> x = EventArray(jnp.array([[0, 2], [1, 1], [2, 0]]).T)
        >>> x.cov_with()  # covariance matrix of x
        array([[ 1., -1.],
               [-1.,  1.]])

        >>> y = EventArray(jnp.array([[3, 2, 1], [4, 2, 0]]))
        >>> x.cov_with(y)  # cross-covariance between x and y
        array([[ 1.5, -1.5],
               [-1.5,  1.5]])
        """
        y = extract_raw_value(y)
        fweights = extract_raw_value(fweights)
        aweights = extract_raw_value(aweights)
        r = u.math.cov(self.value, y, rowvar, bias, fweights, aweights)
        return r

    def expand(self, *sizes) -> Union[u.Quantity, jax.Array]:
        """
        Expand an array to a new shape.

        Expands the dimensions of the array by broadcasting it to a new shape. The new
        dimensions are added at the beginning of the shape. Existing dimensions can be
        expanded if they have a size of 1, otherwise they must match the target size
        or be specified as -1 (which keeps the original size).

        Parameters
        ----------
        *sizes : tuple of ints
            The new shape. Dimensions with -1 will keep their original size.
            The number of elements in sizes must be greater than or equal to
            the number of dimensions in the original array.

        Returns
        -------
        Union[u.Quantity, jax.Array]
            A view of the original array expanded to the new shape.

        Raises
        ------
        ValueError
            If the number of sizes is less than the number of dimensions in the tensor,
            if any new dimension has a negative size, or if a non-singleton dimension
            doesn't match the target size.

        See Also
        --------
        expand_dims : Add new dimensions of size 1
        expand_as : Expand this tensor to the same shape as another tensor

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> a = EventArray(jnp.ones((2, 3)))
        >>> a.expand(4, 2, 3).shape  # adds a new dimension of size 4
        (4, 2, 3)

        >>> b = EventArray(jnp.ones((1, 3)))
        >>> b.expand(2, 3).shape  # expands dimension 0 from 1 to 2
        (2, 3)

        >>> # Using -1 to keep original dimensions
        >>> c = EventArray(jnp.ones((2, 3)))
        >>> c.expand(5, -1, -1).shape
        (5, 2, 3)
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
        """
        Flatten the object for JAX pytree functionality.

        This method is used by JAX's tree_util to support BaseArray instances
        as part of JAX transformations. It separates the object into dynamic data
        (the array value) and static metadata (None in this case).

        Returns
        -------
        tuple
            A tuple containing two elements:
            - A tuple of dynamic values (just the array value in this case)
            - Static metadata (None for BaseArray)

        See Also
        --------
        tree_unflatten : Reconstruct an object from flattened data

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from jax.tree_util import tree_flatten
        >>> a = EventArray(jnp.array([1, 2, 3]))
        >>> dynamic_values, static_metadata = tree_flatten(a)
        >>> dynamic_values
        (Array([1, 2, 3], dtype=int32),)
        >>> static_metadata is None
        True
        """
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        """
        Reconstruct an BaseArray from flattened data.

        This class method is used by JAX's tree_util to reconstruct BaseArray instances
        from flattened data during JAX transformations.

        Parameters
        ----------
        aux_data : Any
            Static metadata for reconstruction (typically None for BaseArray)
        flat_contents : tuple
            A tuple containing the dynamic values that were extracted by tree_flatten

        Returns
        -------
        EventArray
            A reconstructed BaseArray instance

        See Also
        --------
        tree_flatten : Flatten an object for JAX transformations

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from jax.tree_util import tree_flatten, tree_unflatten
        >>> a = EventArray(jnp.array([1, 2, 3]))
        >>> dynamic_values, static_metadata = tree_flatten(a)
        >>> b = EventArray.tree_unflatten(static_metadata, dynamic_values)
        >>> b.value
        Array([1, 2, 3], dtype=int32)
        """
        return cls(*flat_contents)


# Set the array priority for the BaseArray class
setattr(BaseArray, "__array_priority__", 100)
