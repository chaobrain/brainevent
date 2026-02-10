# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

from .base import IndexedRepresentation, is_known_type
from .indexed_binary_extraction import binary_1d_array_index_p_call, binary_2d_array_index_p_call

__all__ = [
    'IndexedSpFloat1d',
    'IndexedSpFloat2d',
]


@register_pytree_node_class
class IndexedSpFloat1d(IndexedRepresentation):
    """
    A binary array is a special case of an event array where the events are binary (0 or 1).

    Parameters
    ----------
    value : array_like
        The input binary array data.
    dtype : jax.typing.DTypeLike, optional
        The data type of the array.
    """
    __module__ = 'brainevent'

    def __init__(self, value):
        super().__init__(value)
        self._spike_indices, self._spike_count = binary_1d_array_index_p_call(self._value)

    @property
    def spike_indices(self):
        return self._spike_indices

    @property
    def spike_count(self):
        return self._spike_count

    def __matmul__(self, oc):
        raise ValueError

    def __rmatmul__(self, oc):
        raise ValueError

    def __imatmul__(self, oc):
        if is_known_type(oc):
            return self.with_value(self.__matmul__(oc))
        return self.with_value(oc.__rmatmul__(self))

    def tree_flatten(self):
        aux = {
            '_spike_indices': self._spike_indices,
            '_spike_count': self._spike_count,
        }
        return (self._value,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj


@register_pytree_node_class
class IndexedSpFloat2d(IndexedRepresentation):
    """
    A binary array is a special case of an event array where the events are binary (0 or 1).

    Parameters
    ----------
    value : array_like
        The input binary array data.
    row_indices : bool, optional
        The row indices of the binary array. If True, the row indices will be stored in the object.
    """
    __module__ = 'brainevent'

    def __init__(
        self,
        value,
        *,
        row_indices=None,
    ):
        super().__init__(value)
        self._spike_indices, self._spike_count = binary_2d_array_index_p_call(self._value)

    @property
    def spike_indices(self):
        return self._spike_indices

    @property
    def spike_count(self):
        return self._spike_count

    def __matmul__(self, oc):
        raise ValueError

    def __rmatmul__(self, oc):
        raise ValueError

    def __imatmul__(self, oc):
        if is_known_type(oc):
            return self.with_value(self.__matmul__(oc))
        return self.with_value(oc.__rmatmul__(self))

    def tree_flatten(self):
        aux = {
            '_spike_indices': self._spike_indices,
            '_spike_count': self._spike_count,
        }
        return (self._value,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj
