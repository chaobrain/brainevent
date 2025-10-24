# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


from jax.tree_util import register_pytree_node_class

from brainevent._dense.masked_float import (
    dense_mat_dot_masked_float_mat,
    masked_float_mat_dot_dense_mat,
    dense_mat_dot_masked_float_vec,
    masked_float_vec_dot_dense_mat,
)
from brainevent._error import MathError
from .base import BaseArray
from .base import extract_raw_value, is_known_type

__all__ = [
    'MaskedFloat',
]


@register_pytree_node_class
class MaskedFloat(BaseArray):
    """
    A specialized array class for masked floating-point values (0 or floating points).

    ``MaskedFloatIndex`` extends ``BaseArray`` to provide functionality for arrays with masked
    floating-point values, where certain elements may be masked (typically to
    indicate missing or invalid data).

    This class supports matrix multiplication operations with other arrays:
    - ``MaskedFloatIndex`` @ dense matrix
    - dense matrix @ ``MaskedFloatIndex``

    The class is registered as a PyTree node for compatibility with JAX's
    functional transformations.

    Attributes:
        value: The underlying masked float array data
    """
    __slots__ = ('_value',)
    __module__ = 'brainevent'

    def __matmul__(self, oc):
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
                return masked_float_vec_dot_dense_mat(self.value, oc)
            else:  # self.ndim == 2
                return masked_float_mat_dot_dense_mat(self.value, oc)
        else:
            return oc.__rmatmul__(self)

    def __rmatmul__(self, oc):
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
                return dense_mat_dot_masked_float_vec(oc, self.value)
            else:
                return dense_mat_dot_masked_float_mat(oc, self.value)
        else:
            return oc.__matmul__(self)

    def __imatmul__(self, oc):
        if is_known_type(oc):
            self.value = self.__matmul__(oc)
        else:
            self.value = oc.__rmatmul__(self)
        return self
