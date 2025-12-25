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

from brainevent import EventArray


class TestEventArray:
    # Test initialization
    def test_event_array_initialization(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        assert np.array_equal(event_array, value)

    # Test _check_tracer method
    def test_check_tracer(self):
        event_array = EventArray(np.array([1, 2, 3]))
        tracer = event_array._check_tracer()
        assert np.array_equal(tracer, np.array([1, 2, 3]))

    # Test data property
    def test_data_property(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        assert np.array_equal(event_array.data, value)

    # Test value property and setter
    def test_value_property_and_setter(self):
        event_array = EventArray(np.array([1, 2, 3]))
        new_value = np.array([4, 5, 6])
        event_array = new_value
        assert np.array_equal(event_array, new_value)

    # Test update method
    def test_update_method(self):
        event_array = EventArray(np.array([1, 2, 3]))
        new_value = np.array([4, 5, 6])
        event_array.update(new_value)
        assert np.array_equal(event_array, new_value)

    # Test imag property
    def test_imag_property(self):
        value = np.array([1 + 2j, 3 + 4j])
        event_array = EventArray(value)
        assert np.array_equal(event_array.imag, np.array([2, 4]))

    # Test real property
    def test_real_property(self):
        value = np.array([1 + 2j, 3 + 4j])
        event_array = EventArray(value)
        assert np.array_equal(event_array.real, np.array([1, 3]))

    # Test size property
    def test_size_property(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        assert event_array.size == 3

    # Test T property
    def test_T_property(self):
        value = np.array([[1, 2], [3, 4]])
        event_array = EventArray(value)
        assert np.array_equal(event_array.T, np.array([[1, 3], [2, 4]]))

    # Test __getitem__ method
    def test_getitem_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        assert event_array[0] == 1
        assert event_array[1] == 2
        assert event_array[2] == 3

    # Test __setitem__ method
    def test_setitem_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        event_array[0] = 4
        assert event_array[0] == 4

    # Test __len__ method
    def test_len_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        assert len(event_array) == 3

    # Test __neg__ method
    def test_neg_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        neg_event_array = -event_array
        assert np.array_equal(neg_event_array, np.array([-1, -2, -3]))

    # Test __pos__ method
    def test_pos_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        pos_event_array = +event_array
        assert np.array_equal(pos_event_array, np.array([1, 2, 3]))

    # Test __abs__ method
    def test_abs_method(self):
        value = np.array([-1, -2, -3])
        event_array = EventArray(value)
        abs_event_array = abs(event_array)
        assert np.array_equal(abs_event_array, np.array([1, 2, 3]))

    # Test __add__ method
    def test_add_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = np.array([4, 5, 6])
        result = event_array + other_value
        assert np.array_equal(result, np.array([5, 7, 9]))

    # Test __radd__ method
    def test_radd_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = 4
        result = other_value + event_array
        assert np.array_equal(result, np.array([5, 6, 7]))

    # Test __iadd__ method
    def test_iadd_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = np.array([4, 5, 6])
        event_array += other_value
        assert np.array_equal(event_array, np.array([5, 7, 9]))

    # Test __sub__ method
    def test_sub_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        result = event_array - other_value
        assert np.array_equal(result, np.array([3, 3, 3]))

    # Test __rsub__ method
    def test_rsub_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = 4
        result = other_value - event_array
        assert np.array_equal(result, np.array([3, 2, 1]))

    # Test __isub__ method
    def test_isub_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        event_array -= other_value
        assert np.array_equal(event_array, np.array([3, 3, 3]))

    # Test __mul__ method
    def test_mul_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = np.array([4, 5, 6])
        result = event_array * other_value
        assert np.array_equal(result, np.array([4, 10, 18]))

    # Test __rmul__ method
    def test_rmul_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = 4
        result = other_value * event_array
        assert np.array_equal(result, np.array([4, 8, 12]))

    # Test __imul__ method
    def test_imul_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = np.array([4, 5, 6])
        event_array *= other_value
        assert np.array_equal(event_array, np.array([4, 10, 18]))

    # Test __rdiv__ method
    def test_rdiv_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        result = event_array / other_value
        assert np.array_equal(result, np.array([4, 2.5, 2]))

    # Test __truediv__ method
    def test_truediv_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        result = event_array / other_value
        assert np.array_equal(result, np.array([4, 2.5, 2]))

    # Test __itruediv__ method
    def test_itruediv_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        event_array = event_array / other_value
        assert np.array_equal(event_array, np.array([4, 2.5, 2]))

    # Test __floordiv__ method
    def test_floordiv_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        result = event_array // other_value
        assert np.array_equal(result, np.array([4, 2, 2]))

    # Test __rfloordiv__ method
    def test_rfloordiv_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = 4
        result = other_value // event_array
        assert np.array_equal(result, np.array([4, 2, 1]))

    # Test __ifloordiv__ method
    def test_ifloordiv_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        event_array //= other_value
        assert np.array_equal(event_array, np.array([4, 2, 2]))

    # Test __divmod__ method
    def test_divmod_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        quotient, remainder = event_array.__divmod__(other_value)
        assert np.array_equal(quotient, np.array([4, 2, 2]))
        assert np.array_equal(remainder, np.array([0, 1, 0]))

    # Test __rdivmod__ method
    def test_rdivmod_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = 4
        quotient, remainder = event_array.__rdivmod__(other_value)
        assert np.array_equal(quotient, np.array([4, 2, 1]))
        assert np.array_equal(remainder, np.array([0, 0, 1]))

    # Test __mod__ method
    def test_mod_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        result = event_array % other_value
        assert np.array_equal(result, np.array([0, 1, 0]))

    # Test __rmod__ method
    def test_rmod_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = 4
        result = other_value % event_array
        assert np.array_equal(result, np.array([0, 0, 1]))

    # Test __imod__ method
    def test_imod_method(self):
        value = np.array([4, 5, 6])
        event_array = EventArray(value)
        other_value = np.array([1, 2, 3])
        event_array %= other_value
        assert np.array_equal(event_array, np.array([0, 1, 0]))

    # Test __pow__ method
    def test_pow_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = np.array([2, 3, 4])
        result = event_array ** other_value
        assert np.array_equal(result, np.array([1, 8, 81]))

    # Test __rpow__ method
    def test_rpow_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = 2
        result = other_value ** event_array
        assert np.array_equal(result, np.array([2, 4, 8]))

    # Test __ipow__ method
    def test_ipow_method(self):
        value = np.array([1, 2, 3])
        event_array = EventArray(value)
        other_value = np.array([2, 3, 4])
        event_array **= other_value
        assert np.array_equal(event_array, np.array([1, 8, 81]))

    # Test comparison methods
    def test_comparison_methods(self):
        value_a = np.array([1, 2, 3])
        event_array_a = EventArray(value_a)
        value_b = np.array([3, 2, 1])
        event_array_b = EventArray(value_b)

        # Test __eq__
        result = event_array_a == event_array_b
        assert np.array_equal(result, np.array([False, True, False]))

        # Test __ne__
        result = event_array_a != event_array_b
        assert np.array_equal(result, np.array([True, False, True]))

        # Test __lt__
        result = event_array_a < event_array_b
        assert np.array_equal(result, np.array([True, False, False]))

        # Test __le__
        result = event_array_a <= event_array_b
        assert np.array_equal(result, np.array([True, True, False]))

        # Test __gt__
        result = event_array_a > event_array_b
        assert np.array_equal(result, np.array([False, False, True]))

        # Test __ge__
        result = event_array_a >= event_array_b
        assert np.array_equal(result, np.array([False, True, True]))

    # Test bitwise operations
    def test_bitwise_operations(self):
        event_array = EventArray(np.array([1, 2, 3, 4]))
        other = np.array([5, 6, 7, 8])

        # Test __and__
        result = event_array & other
        assert np.array_equal(result, np.array([1, 2, 3, 0]))

        # Test __or__
        result = event_array | other
        assert np.array_equal(result, np.array([5, 6, 7, 12]))

        # Test __xor__
        result = event_array ^ other
        assert np.array_equal(result, np.array([4, 4, 4, 12]))

        # Test in-place bitwise operations
        event_array_copy = EventArray(np.array([1, 2, 3, 4]))
        event_array_copy &= np.array([5, 6, 7, 8])
        assert np.array_equal(event_array_copy, np.array([1, 2, 3, 0]))

        event_array_copy = EventArray(np.array([1, 2, 3, 4]))
        event_array_copy |= np.array([5, 6, 7, 8])
        assert np.array_equal(event_array_copy, np.array([5, 6, 7, 12]))

        event_array_copy = EventArray(np.array([1, 2, 3, 4]))
        event_array_copy ^= np.array([5, 6, 7, 8])
        assert np.array_equal(event_array_copy, np.array([4, 4, 4, 12]))

    # Test NumPy methods
    def test_numpy_methods(self):
        # Test all
        event_array = EventArray(np.array([[True, False], [True, True]]))
        assert not event_array.all()
        assert np.array_equal(event_array.all(axis=0), np.array([True, False]))

        # Test any
        event_array = EventArray(np.array([[True, False], [False, False]]))
        assert event_array.any()
        assert np.array_equal(event_array.any(axis=0), np.array([True, False]))

        # Test argmax/argmin
        event_array = EventArray(np.array([[1, 2], [3, 4]]))
        assert event_array.argmax() == 3
        assert np.array_equal(event_array.argmax(axis=0), np.array([1, 1]))
        assert np.array_equal(event_array.argmin(axis=1), np.array([0, 0]))

        # Test clip
        event_array = EventArray(np.array([1, 2, 3, 4, 5]))
        clipped = event_array.clip(2, 4)
        assert np.array_equal(clipped, np.array([2, 2, 3, 4, 4]))

        # Test mean, std, var, sum
        event_array = EventArray(np.array([1, 2, 3, 4]))
        assert event_array.mean() == 2.5
        assert event_array.std() == 1.118033988749895
        assert event_array.var() == 1.25
        assert event_array.sum() == 10

    # Test trigonometric functions
    def test_trigonometric_functions(self):
        event_array = EventArray(np.array([0, np.pi / 4, np.pi / 2]))

        # Test sin
        sin_result = np.sin(event_array.value)
        assert np.allclose(sin_result, np.array([0, 0.7071067811865475, 1]), atol=1e-4, rtol=1e-4)

        # Test cos
        cos_result = np.cos(event_array.value)
        assert np.allclose(cos_result, np.array([1, 0.7071067811865476, 0]), atol=1e-4, rtol=1e-4)

        # Test tan
        tan_result = np.tan(event_array.value)
        assert np.allclose(tan_result[:2], np.array([0, 1]), atol=1e-4, rtol=1e-4)  # Skip pi/2 as tan is undefined

    # Test array manipulation
    def test_array_manipulation(self):
        # Test reshape
        event_array = EventArray(np.array([1, 2, 3, 4, 5, 6]))
        reshaped = event_array.reshape(2, 3)
        assert reshaped.shape == (2, 3)

        # Test transpose
        event_array = EventArray(np.array([[1, 2], [3, 4]]))
        transposed = event_array.transpose()
        assert np.array_equal(transposed, np.array([[1, 3], [2, 4]]))

        # Test flatten
        event_array = EventArray(np.array([[1, 2], [3, 4]]))
        flattened = event_array.flatten()
        assert np.array_equal(flattened, np.array([1, 2, 3, 4]))

        # Test ravel
        event_array = EventArray(np.array([[1, 2], [3, 4]]))
        raveled = event_array.ravel()
        assert np.array_equal(raveled, np.array([1, 2, 3, 4]))

    # Test copying and cloning
    def test_copy_and_clone(self):
        event_array = EventArray(np.array([1, 2, 3]))

        # Test copy method
        copy_array = event_array.copy()
        assert np.array_equal(copy_array, np.array([1, 2, 3]))

        # Test clone method
        cloned = event_array.clone()
        assert np.array_equal(cloned, np.array([1, 2, 3]))

        # Modify original, check if clone is affected
        event_array[0] = 99
        assert cloned[0] == 1  # Clone should remain unchanged

    # Test expansion and broadcasting
    def test_expansion_and_broadcasting(self):
        event_array = EventArray(np.array([1, 2, 3]))

        # Test expand
        expanded = event_array.expand(2, 3)
        assert expanded.shape == (2, 3)
        assert np.array_equal(expanded[0], np.array([1, 2, 3]))
        assert np.array_equal(expanded[1], np.array([1, 2, 3]))

        # Test tile
        tiled = event_array.tile(2)
        assert np.array_equal(tiled, np.array([1, 2, 3, 1, 2, 3]))

    # Test sorting and searching
    def test_sorting_and_searching(self):
        event_array = EventArray(np.array([3, 1, 4, 2]))

        # Test argsort
        sorted_indices = event_array.argsort()
        assert np.array_equal(sorted_indices, np.array([1, 3, 0, 2]))

        # Test searchsorted
        event_array = EventArray(np.array([1, 2, 3, 4]))
        indices = event_array.searchsorted([1.5, 3.5])
        assert np.array_equal(indices, np.array([1, 3]))

    # Test additional mathematical operations
    def test_additional_math_operations(self):
        # Test round
        event_array = EventArray(np.array([1.1, 2.5, 3.9]))
        rounded = event_array.round()
        assert np.array_equal(rounded, np.array([1.0, 2.0, 4.0]))

        # Test clamp
        event_array = EventArray(np.array([1, 2, 3, 4, 5]))
        clamped = event_array.clamp(2, 4)
        assert np.array_equal(clamped, np.array([2, 2, 3, 4, 4]))

        # Test prod
        event_array = EventArray(np.array([1, 2, 3, 4]))
        assert event_array.prod() == 24

        # Test cumprod
        event_array = EventArray(np.array([1, 2, 3, 4]))
        assert np.array_equal(event_array.cumprod(), np.array([1, 2, 6, 24]))

        # Test cumsum
        event_array = EventArray(np.array([1, 2, 3, 4]))
        assert np.array_equal(event_array.cumsum(), np.array([1, 3, 6, 10]))
