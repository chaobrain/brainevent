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

import numpy as np
import pytest

import jax
from brainevent import BaseArray, BinaryArray


class TestBaseArrayMinimalAPI:
    def test_base_array_is_abstract(self):
        with pytest.raises(TypeError):
            BaseArray(np.array([1, 2, 3]))

    def test_construction_and_core_properties(self):
        arr = BinaryArray(np.array([1, 2, 3]))
        assert arr.shape == (3,)
        assert arr.ndim == 1
        assert arr.size == 3
        assert arr.dtype == arr.value.dtype

    def test_getitem_and_iteration(self):
        arr = BinaryArray(np.array([1, 2, 3]))
        assert arr[0] == 1
        assert list(arr) == [1, 2, 3]

    def test_with_value_returns_new_instance(self):
        arr = BinaryArray(np.array([1, 2, 3]))
        arr2 = arr.with_value(np.array([4, 5, 6]))
        assert isinstance(arr2, BinaryArray)
        assert np.array_equal(arr.value, np.array([1, 2, 3]))
        assert np.array_equal(arr2.value, np.array([4, 5, 6]))
        assert id(arr) != id(arr2)

    def test_value_is_read_only(self):
        arr = BinaryArray(np.array([1, 2, 3]))
        with pytest.raises(AttributeError):
            arr.value = np.array([4, 5, 6])

    def test_array_protocol(self):
        arr = BinaryArray(np.array([1, 2, 3]))
        np_arr = np.asarray(arr)
        assert np.array_equal(np_arr, np.array([1, 2, 3]))

    def test_pytree_roundtrip(self):
        arr = BinaryArray(np.array([1, 0, 1], dtype=np.float32))
        leaves, treedef = jax.tree.flatten(arr)
        arr2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(arr2, BinaryArray)
        assert np.array_equal(arr2.value, arr.value)

    def test_item_assignment_is_not_supported(self):
        arr = BinaryArray(np.array([1, 2, 3]))
        with pytest.raises(TypeError):
            arr[0] = 5
