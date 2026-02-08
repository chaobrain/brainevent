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
]

ArrayValue = Union[jax.Array, u.Quantity]
ArrayLike = Union['EventRepresentation', jax.Array, np.ndarray, u.Quantity, list, tuple, int, float, bool]


def extract_raw_value(obj):
    return obj.value if isinstance(obj, EventRepresentation) else obj


def is_known_type(x):
    return isinstance(x, (u.Quantity, jax.Array, np.ndarray, EventRepresentation))


def _normalize_index(index):
    if isinstance(index, tuple):
        return tuple(_normalize_index(x) for x in index)
    return extract_raw_value(index)


@register_pytree_node_class
class EventRepresentation(ABC):
    """Minimal array wrapper protocol for event array subclasses."""

    __slots__ = ('_value',)
    __array_priority__ = 100
    __module__ = 'brainevent'

    def __init__(self, value: ArrayLike, *, dtype: jax.typing.DTypeLike = None):
        value = extract_raw_value(value)
        if isinstance(value, (list, tuple, np.ndarray)):
            value = u.math.asarray(value)
        if dtype is not None:
            value = u.math.asarray(value, dtype=dtype)
        self._value = value

    @property
    def value(self) -> ArrayValue:
        return self._value

    @value.setter
    def value(self, val) -> None:
        self._value = val

    def with_value(self, value: ArrayLike) -> Self:
        return type(self)(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._value.shape)

    @property
    def ndim(self) -> int:
        return self._value.ndim

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def size(self) -> int:
        return int(self._value.size)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(value={self._value}, dtype={self.dtype})"

    def __len__(self) -> int:
        return len(self._value)

    def __iter__(self):
        for i in range(self._value.shape[0]):
            yield self._value[i]

    def __getitem__(self, index):
        return self._value[_normalize_index(index)]

    def tree_flatten(self):
        return (self._value,), {}

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        value, = flat_contents
        obj = object.__new__(cls)
        obj._value = value
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj

    @abstractmethod
    def __matmul__(self, other):
        pass

    @abstractmethod
    def __rmatmul__(self, other):
        pass
