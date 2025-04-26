# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
from typing import Union

import brainunit as u
import jax

__all__ = [
    'JITCMatrix',
]


class JITCMatrix(u.sparse.SparseMatrix):
    """
    Just-in-time Connectivity (JITC) matrix.

    A base class for just-in-time connectivity matrices that inherits from
    the SparseMatrix class in the ``brainunit`` library. This class serves as
    an abstraction for sparse matrices that are generated or computed on demand
    rather than stored in full.

    JITC matrices are particularly useful in neural network simulations where
    connectivity patterns might be large but follow specific patterns that
    can be efficiently computed rather than explicitly stored in memory.

    Attributes:
        Inherits all attributes from ``brainunit.sparse.SparseMatrix``

    Note:
        This is a base class and should be subclassed for specific
        implementations of JITC matrices.
    """

    def _unitary_op(self, op):
        raise NotImplementedError("unitary operation not implemented.")

    def __abs__(self):
        return self._unitary_op(operator.abs)

    def __neg__(self):
        return self._unitary_op(operator.neg)

    def __pos__(self):
        return self._unitary_op(operator.pos)

    def _binary_op(self, other, op):
        raise NotImplementedError("binary operation not implemented.")

    def __mul__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        return self._binary_op(other, operator.mul)

    def __div__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        return self._binary_op(other, operator.truediv)

    def __truediv__(self, other):
        return self.__div__(other)

    def __add__(self, other):
        return self._binary_op(other, operator.add)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub)

    def __mod__(self, other):
        return self._binary_op(other, operator.mod)

    def _binary_rop(self, other, op):
        raise NotImplementedError("binary operation not implemented.")

    def __rmul__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        return self._binary_rop(other, operator.mul)

    def __rdiv__(self, other: Union[jax.typing.ArrayLike, u.Quantity]):
        return self._binary_rop(other, operator.truediv)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __radd__(self, other):
        return self._binary_rop(other, operator.add)

    def __rsub__(self, other):
        return self._binary_rop(other, operator.sub)

    def __rmod__(self, other):
        return self._binary_rop(other, operator.mod)
