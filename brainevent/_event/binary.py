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

import jax
from jax.tree_util import register_pytree_node_class

from brainevent._dense import (
    dbmm,
    bdmm,
    dbmv,
    bdvm,
)
from brainevent._error import MathError
from .base import (
    BaseArray,
    extract_raw_value,
    is_known_type,
)
from .indexed_binary_extraction import binary_array_index

__all__ = [
    'BinaryArray',
    'EventArray',
]


@register_pytree_node_class
class BinaryArray(BaseArray):
    """
    A binary array is a special case of an event array where the events are binary (0 or 1).

    Parameters
    ----------
    value : array_like
        The input binary array data.
    dtype : jax.typing.DTypeLike, optional
        The data type of the array.
    indexed : bool, optional
        If True, pre-compute spike indices and spike count for efficient
        indexed operations, and make the array immutable. Default is False.
    """
    __slots__ = ('_value', '_indexed', '_spike_indices', '_spike_count')
    __module__ = 'brainevent'

    def __init__(self, value, dtype: jax.typing.DTypeLike = None, indexed: bool = False):
        super().__init__(value, dtype=dtype)
        self._indexed = indexed
        if indexed:
            self._spike_indices, self._spike_count = binary_array_index(self._value)
        else:
            self._spike_indices = None
            self._spike_count = None

    @property
    def indexed(self) -> bool:
        return self._indexed

    @property
    def spike_indices(self):
        return self._spike_indices

    @property
    def spike_count(self):
        return self._spike_count

    def __setitem__(self, index, value):
        if self._indexed:
            raise NotImplementedError('Setting items in an indexed BinaryArray is not supported.')
        super().__setitem__(index, value)

    def _update(self, value):
        if self._indexed:
            raise NotImplementedError('Updating an indexed BinaryArray is not supported.')
        super()._update(value)

    def __matmul__(self, oc):
        """
        Perform matrix multiplication on the array with another object.

        This special method implements the matrix multiplication operator (@)
        for EventArray instances. It handles matrix multiplication with different
        array types and dimensions, performing appropriate validation checks.

        Parameters
        ----------
        oc : array_like
            The right operand of the matrix multiplication. This object will be
            multiplied with the current EventArray instance.

        Returns
        -------
        ndarray or EventArray
            The result of the matrix multiplication between this EventArray instance
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
        if is_known_type(oc):
            oc = extract_raw_value(oc)
            # Check dimensions for both operands
            if self.ndim not in (1, 2):
                raise MathError(
                    f"Matrix multiplication is only supported "
                    f"for 1D and 2D arrays. Got {self.ndim}D array."
                )

            if self.ndim == 0:
                raise MathError("Matrix multiplication is not supported for scalar arrays.")

            assert oc.ndim == 2, (f"Right operand must be a 2D array in "
                                  f"matrix multiplication. Got {oc.ndim}D array.")
            assert self.shape[-1] == oc.shape[0], (f"Incompatible dimensions for matrix multiplication: "
                                                   f"{self.shape[-1]} and {oc.shape[0]}.")

            # Perform the appropriate multiplication based on dimensions
            if self.ndim == 1:
                return bdvm(self.value, oc)
            else:  # self.ndim == 2
                return bdmm(self.value, oc)
        else:
            return oc.__rmatmul__(self)

    def __rmatmul__(self, oc):
        """
        Perform matrix multiplication on another object with the array.

        This special method implements the reverse matrix multiplication operator (@)
        when the left operand is not an EventArray. It handles the case where
        another object is matrix-multiplied with this EventArray instance.

        Parameters
        ----------
        oc : array_like
            The left operand of the matrix multiplication. This object will be
            multiplied with the current EventArray instance.

        Returns
        -------
        ndarray or EventArray
            The result of the matrix multiplication between the other object and this
            EventArray instance.

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
        if is_known_type(oc):
            oc = extract_raw_value(oc)
            # Check dimensions for both operands
            if self.ndim not in (1, 2):
                raise MathError(f"Matrix multiplication is only supported "
                                f"for 1D and 2D arrays. Got {self.ndim}D array.")

            if self.ndim == 0:
                raise MathError("Matrix multiplication is not supported for scalar arrays.")

            assert oc.ndim == 2, (f"Left operand must be a 2D array in "
                                  f"matrix multiplication. Got {oc.ndim}D array.")
            assert oc.shape[-1] == self.shape[0], (f"Incompatible dimensions for matrix "
                                                   f"multiplication: {oc.shape[-1]} and {self.shape[0]}.")

            # Perform the appropriate multiplication based on dimensions
            if self.ndim == 1:
                return dbmv(oc, self.value)
            else:
                return dbmm(oc, self.value)
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
        if is_known_type(oc):
            self.value = self.__matmul__(oc)
        else:
            self.value = oc.__rmatmul__(self)
        return self

    def tree_flatten(self):
        if self._indexed:
            return (self._value,), (True, self._spike_indices, self._spike_count)
        return (self._value,), (False,)

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        value, = flat_contents
        if aux_data[0]:  # indexed=True
            _, spike_indices, spike_count = aux_data
            obj = object.__new__(cls)
            obj._value = value
            obj._indexed = True
            obj._spike_indices = spike_indices
            obj._spike_count = spike_count
            return obj
        return cls(value)


EventArray = BinaryArray
