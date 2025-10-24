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

from brainevent._dense.binary_index import (
    binary_vec_dot_dense_mat,
    binary_mat_dot_dense_mat,
    dense_mat_dot_binary_vec,
    dense_mat_dot_binary_mat,
)
from brainevent._error import MathError
from brainevent._misc import is_known_type
from .base import BaseArray, extract_raw_value
from .binary import BinaryArray
from .binary_index_extraction import binary_array_index

__all__ = [
    'BinaryArrayIndex',
]


@register_pytree_node_class
class BinaryArrayIndex(BaseArray):
    __module__ = 'brainevent'

    def __init__(self, value, dtype: jax.typing.DTypeLike = None):
        if isinstance(value, BaseArray):
            if not isinstance(value, BinaryArray):
                raise TypeError("BinaryArrayIndex can only be initialized with a BinaryArray or a compatible type.")
            value = value.value
        super().__init__(value, dtype=dtype)

        self.spike_indices, self.spike_count = binary_array_index(value)

    def __setitem__(self, index, value):
        raise NotImplementedError('Setting items in BinaryArrayIndex is not supported.')

    def _update(self, value):
        raise NotImplementedError('Updating BinaryArrayIndex is not supported.')

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

            assert oc.ndim == 2, (
                f"Right operand must be a 2D array in "
                f"matrix multiplication. Got {oc.ndim}D array."
            )
            assert self.shape[-1] == oc.shape[0], (
                f"Incompatible dimensions for matrix multiplication: "
                f"{self.shape[-1]} and {oc.shape[0]}."
            )

            # Perform the appropriate multiplication based on dimensions
            if self.ndim == 1:
                return binary_vec_dot_dense_mat(self, oc)
            else:  # self.ndim == 2
                return binary_mat_dot_dense_mat(self, oc)
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
                raise MathError(
                    f"Matrix multiplication is only supported "
                    f"for 1D and 2D arrays. Got {self.ndim}D array."
                )

            if self.ndim == 0:
                raise MathError("Matrix multiplication is not supported for scalar arrays.")

            assert oc.ndim == 2, (
                f"Left operand must be a 2D array in "
                f"matrix multiplication. Got {oc.ndim}D array."
            )
            assert oc.shape[-1] == self.shape[0], (
                f"Incompatible dimensions for matrix "
                f"multiplication: {oc.shape[-1]} and {self.shape[0]}."
            )

            # Perform the appropriate multiplication based on dimensions
            if self.ndim == 1:
                return dense_mat_dot_binary_vec(oc, self.value)
            else:
                return dense_mat_dot_binary_mat(oc, self.value)
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
        return (self.value,), (self.spike_count, self.spike_indices)

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        value, = flat_contents
        spike_count, spike_indices = aux_data
        obj = object.__new__(cls)
        obj._value = value
        obj.spike_count = spike_count
        obj.spike_indices = spike_indices
        return obj
