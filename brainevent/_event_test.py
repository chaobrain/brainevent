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

# -*- coding: utf-8 -*-

import numpy as np

from brainevent import EventArray


# Test initialization
def test_event_array_initialization():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    assert np.array_equal(event_array, value)


# Test _check_tracer method
def test_check_tracer():
    event_array = EventArray(np.array([1, 2, 3]))
    tracer = event_array._check_tracer()
    assert np.array_equal(tracer, np.array([1, 2, 3]))


# Test data property
def test_data_property():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    assert np.array_equal(event_array.data, value)


# Test value property and setter
def test_value_property_and_setter():
    event_array = EventArray(np.array([1, 2, 3]))
    new_value = np.array([4, 5, 6])
    event_array = new_value
    assert np.array_equal(event_array, new_value)


# Test update method
def test_update_method():
    event_array = EventArray(np.array([1, 2, 3]))
    new_value = np.array([4, 5, 6])
    event_array.update(new_value)
    assert np.array_equal(event_array, new_value)


# # Test dtype property
# def test_dtype_property():
#     value = np.array([1, 2, 3], dtype=np.float32)
#     event_array = EventArray(value)
#     assert event_array.dtype == np.float32
#
#
# # Test shape property
# def test_shape_property():
#     value = np.array([1, 2, 3])
#     event_array = EventArray(value)
#     assert event_array.shape == (3,)
#
#
# # Test ndim property
# def test_ndim_property():
#     value = np.array([1, 2, 3])
#     event_array = EventArray(value)
#     assert event_array.ndim == 1


# Test imag property
def test_imag_property():
    value = np.array([1 + 2j, 3 + 4j])
    event_array = EventArray(value)
    assert np.array_equal(event_array.imag, np.array([2, 4]))


# Test real property
def test_real_property():
    value = np.array([1 + 2j, 3 + 4j])
    event_array = EventArray(value)
    assert np.array_equal(event_array.real, np.array([1, 3]))


# Test size property
def test_size_property():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    assert event_array.size == 3


# Test T property
def test_T_property():
    value = np.array([[1, 2], [3, 4]])
    event_array = EventArray(value)
    assert np.array_equal(event_array.T, np.array([[1, 3], [2, 4]]))




# Test __getitem__ method
def test_getitem_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    assert event_array[0] == 1
    assert event_array[1] == 2
    assert event_array[2] == 3


# Test __setitem__ method
def test_setitem_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    event_array[0] = 4
    assert event_array[0] == 4


# Test __len__ method
def test_len_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    assert len(event_array) == 3


# Test __neg__ method
def test_neg_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    neg_event_array = -event_array
    assert np.array_equal(neg_event_array, np.array([-1, -2, -3]))


# Test __pos__ method
def test_pos_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    pos_event_array = +event_array
    assert np.array_equal(pos_event_array, np.array([1, 2, 3]))


# Test __abs__ method
def test_abs_method():
    value = np.array([-1, -2, -3])
    event_array = EventArray(value)
    abs_event_array = abs(event_array)
    assert np.array_equal(abs_event_array, np.array([1, 2, 3]))



#
# # Test __ne__ method
# def test_ne_method():
#     value = np.array([1, 2, 3])
#     event_array = EventArray(value)
#     other_value = np.array([4, 5, 6])
#     assert event_array != other_value

#
# # Test __lt__ method
# def test_lt_method():
#     value = np.array([1, 2, 3])
#     event_array = EventArray(value)
#     other_value = np.array([4, 5, 6])
#     assert event_array < other_value
#
#
# # Test __le__ method
# def test_le_method():
#     value = np.array([1, 2, 3])
#     event_array = EventArray(value)
#     other_value = np.array([4, 5, 6])
#     assert event_array <= other_value
#
#
# # Test __gt__ method
# def test_gt_method():
#     value = np.array([4, 5, 6])
#     event_array = EventArray(value)
#     other_value = np.array([1, 2, 3])
#     assert event_array > other_value
#
#
# # Test __ge__ method
# def test_ge_method():
#     value = np.array([4, 5, 6])
#     event_array = EventArray(value)
#     other_value = np.array([1, 2, 3])
#     assert event_array >= other_value
#

# Test __add__ method
def test_add_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = np.array([4, 5, 6])
    result = event_array + other_value
    assert np.array_equal(result, np.array([5, 7, 9]))


# Test __radd__ method
def test_radd_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = 4
    result = other_value + event_array
    assert np.array_equal(result, np.array([5, 6, 7]))


# Test __iadd__ method
def test_iadd_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = np.array([4, 5, 6])
    event_array += other_value
    assert np.array_equal(event_array, np.array([5, 7, 9]))


# Test __sub__ method
def test_sub_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    result = event_array - other_value
    assert np.array_equal(result, np.array([3, 3, 3]))


# Test __rsub__ method
def test_rsub_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = 4
    result = other_value - event_array
    assert np.array_equal(result, np.array([3, 2, 1]))


# Test __isub__ method
def test_isub_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    event_array -= other_value
    assert np.array_equal(event_array, np.array([3, 3, 3]))


# Test __mul__ method
def test_mul_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = np.array([4, 5, 6])
    result = event_array * other_value
    assert np.array_equal(result, np.array([4, 10, 18]))


# Test __rmul__ method
def test_rmul_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = 4
    result = other_value * event_array
    assert np.array_equal(result, np.array([4, 8, 12]))


# Test __imul__ method
def test_imul_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = np.array([4, 5, 6])
    event_array *= other_value
    assert np.array_equal(event_array, np.array([4, 10, 18]))


# Test __rdiv__ method
def test_rdiv_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    result = event_array / other_value
    assert np.array_equal(result, np.array([4, 2.5, 2]))


# Test __truediv__ method
def test_truediv_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    result = event_array / other_value
    assert np.array_equal(result, np.array([4, 2.5, 2]))

#
# # Test __rtruediv__ method
# def test_rtruediv_method():
#     value = np.array([1, 2, 3])
#     event_array = EventArray(value)
#     other_value = 4
#     result = other_value / event_array
#     assert np.array_equal(result, np.array([4, 2, 4 / 3]))


# Test __itruediv__ method
def test_itruediv_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    event_array = event_array / other_value
    assert np.array_equal(event_array, np.array([4, 2.5, 2]))


# Test __floordiv__ method
def test_floordiv_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    result = event_array // other_value
    assert np.array_equal(result, np.array([4, 2, 2]))


# Test __rfloordiv__ method
def test_rfloordiv_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = 4
    result = other_value // event_array
    assert np.array_equal(result, np.array([4, 2, 1]))


# Test __ifloordiv__ method
def test_ifloordiv_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    event_array //= other_value
    assert np.array_equal(event_array, np.array([4, 2, 2]))


# Test __divmod__ method
def test_divmod_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    quotient, remainder = event_array.__divmod__(other_value)
    assert np.array_equal(quotient, np.array([4, 2, 2]))
    assert np.array_equal(remainder, np.array([0, 1, 0]))


# Test __rdivmod__ method
def test_rdivmod_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = 4
    quotient, remainder = event_array.__rdivmod__(other_value)
    assert np.array_equal(quotient, np.array([4, 2, 1]))
    assert np.array_equal(remainder, np.array([0, 0, 1]))


# Test __mod__ method
def test_mod_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    result = event_array % other_value
    assert np.array_equal(result, np.array([0, 1, 0]))


# Test __rmod__ method
def test_rmod_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = 4
    result = other_value % event_array
    assert np.array_equal(result, np.array([0, 0, 1]))


# Test __imod__ method
def test_imod_method():
    value = np.array([4, 5, 6])
    event_array = EventArray(value)
    other_value = np.array([1, 2, 3])
    event_array %= other_value
    assert np.array_equal(event_array, np.array([0, 1, 0]))


# Test __pow__ method
def test_pow_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = np.array([2, 3, 4])
    result = event_array ** other_value
    assert np.array_equal(result, np.array([1, 8, 81]))


# Test __rpow__ method
def test_rpow_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = 2
    result = other_value ** event_array
    assert np.array_equal(result, np.array([2, 4, 8]))


# Test __ipow__ method
def test_ipow_method():
    value = np.array([1, 2, 3])
    event_array = EventArray(value)
    other_value = np.array([2, 3, 4])
    event_array **= other_value
    assert np.array_equal(event_array, np.array([1, 8, 81]))


# Test __matmul__ method
def test_matmul_method():
    value = np.array([[1, 2], [3, 4]])
    event_array = EventArray(value)
    other_value = np.array([[5, 6], [7, 8]])
    result = event_array @ other_value
    assert np.array_equal(result, np.array([[19, 22], [43, 50]]))


# Test __rmatmul__ method
def test_rmatmul_method():
    value = np.array([[1, 2], [3, 4]])
    event_array = EventArray(value)
    other_value = np.array([[5, 6], [7, 8]])
    result = other_value @ event_array
    assert np.array_equal(result, np.array([[23, 34], [31, 46]]))
