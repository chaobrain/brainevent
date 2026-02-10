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

from jax.tree_util import register_pytree_node_class

from brainevent._dense import binary_densemm, binary_densemv
from brainevent._error import MathError
from .base import EventRepresentation, extract_raw_value, is_known_type

__all__ = [
    'BinaryArray',
]


@register_pytree_node_class
class BinaryArray(EventRepresentation):
    """
    A binary array is a special case of an event array where the events are binary (0 or 1).

    Parameters
    ----------
    value : array_like
        The input binary array data.
    """
    __module__ = 'brainevent'

    def __init__(self, value):
        super().__init__(value)

    @property
    def T(self):
        return self.value.T

    def transpose(self, *axes):
        return self.value.transpose(*axes)

    def __matmul__(self, oc):
        """
        Perform matrix multiplication on the array with another object.

        This special method implements the matrix multiplication operator (@)
        for BinaryArray instances. It handles matrix multiplication with different
        array types and dimensions, performing appropriate validation checks.

        Parameters
        ----------
        oc : array_like
            The right operand of the matrix multiplication. This object will be
            multiplied with the current BinaryArray instance.

        Returns
        -------
        ndarray or BinaryArray
            The result of the matrix multiplication between this BinaryArray instance
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
                return binary_densemv(oc, self.value, transpose=True)
            else:  # self.ndim == 2
                # self[m,k] @ oc[k,n]: use weights=oc[k,n], spikes=self.value.T[k,m]
                # gives oc.T @ self.value.T = [n,m], then .T = [m,n]
                return binary_densemm(oc, self.value.T, transpose=True).T
        else:
            return oc.__rmatmul__(self)

    def __rmatmul__(self, oc):
        """
        Perform matrix multiplication on another object with the array.

        This special method implements the reverse matrix multiplication operator (@)
        when the left operand is not an BinaryArray. It handles the case where
        another object is matrix-multiplied with this BinaryArray instance.

        Parameters
        ----------
        oc : array_like
            The left operand of the matrix multiplication. This object will be
            multiplied with the current BinaryArray instance.

        Returns
        -------
        ndarray or BinaryArray
            The result of the matrix multiplication between the other object and this
            BinaryArray instance.

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
                return binary_densemv(oc, self.value, transpose=False)
            else:
                return binary_densemm(oc, self.value, transpose=False)
        else:
            return oc.__matmul__(self)

    def tree_flatten(self):
        aux = dict()
        return (self._value,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj
