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


import brainstate
import brainunit as u
import pytest

import brainevent
from brainevent._dense_impl_binary import (
    matrix_event_mm, event_matrix_mm, dense_dot_binary_vec, binary_vec_dot_dense_mat
)


class TestMatrixEvent:
    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    @pytest.mark.parametrize("asbool", [True, False])
    def test_mm(self, m, k, n, asbool):
        matrix = brainstate.random.randn(m, k)
        events = brainevent.EventArray(
            brainstate.random.randn(k, n) < 0.5
        )
        if not asbool:
            events.value = u.math.asarray(events.value, dtype=float)
        out1 = matrix @ events
        out2 = matrix @ events.data
        assert u.math.allclose(out1, out2, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    @pytest.mark.parametrize("float_as_event", [True, False])
    def test_matrix_event_mm(self, m, k, n, float_as_event):
        matrix = brainstate.random.randn(m, k)
        events = u.math.asarray(brainstate.random.randn(k, n) < 0.5, dtype=float)
        if not float_as_event:
            events = events * brainstate.random.rand(k, n)
        out1 = matrix_event_mm(matrix, events, float_as_event=float_as_event)
        out2 = matrix @ events
        assert u.math.allclose(out1, out2, atol=1e-4, rtol=1e-4)


class TestEventMatrix:
    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    @pytest.mark.parametrize("asbool", [True, False])
    def test_mm(self, m, k, n, asbool):
        events = brainevent.EventArray(
            brainstate.random.randn(m, k) < 0.5
        )
        if not asbool:
            events.value = u.math.asarray(events.value, dtype=float)
        matrix = brainstate.random.randn(k, n)
        out1 = events @ matrix
        out2 = events.data @ matrix
        assert u.math.allclose(out1, out2, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    @pytest.mark.parametrize("float_as_event", [True, False])
    def test_matrix_event_mm(self, m, k, n, float_as_event):
        events = u.math.asarray(brainstate.random.randn(m, k) < 0.5, dtype=float)
        if not float_as_event:
            events = events * brainstate.random.rand(m, k)
        matrix = brainstate.random.randn(k, n)
        out1 = event_matrix_mm(events, matrix, float_as_event=float_as_event)
        out2 = events @ matrix
        assert u.math.allclose(out1, out2, atol=1e-4, rtol=1e-4)


class TestMatrixEvent_mv:
    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("asbool", [True, False])
    def test_mm(self, m, k, asbool):
        matrix = brainstate.random.randn(m, k)
        events = brainevent.EventArray(
            brainstate.random.randn(k) < 0.5
        )
        if not asbool:
            events.value = u.math.asarray(events.value, dtype=float)
        out1 = matrix @ events
        out2 = matrix @ events.data
        assert u.math.allclose(out1, out2, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("float_as_event", [True, False])
    def test_matrix_event_mv(self, m, k, float_as_event):
        matrix = brainstate.random.randn(m, k)
        events = u.math.asarray(brainstate.random.randn(k) < 0.5, dtype=float)
        if not float_as_event:
            events = events * brainstate.random.rand(k)
        out1 = dense_dot_binary_vec(matrix, events, float_as_event=float_as_event)
        out2 = matrix @ events
        assert u.math.allclose(out1, out2, atol=1e-4, rtol=1e-4)


class TestEventMatrix_mv:
    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("asbool", [True, False])
    def test_mm(self, m, k, asbool):
        events = brainevent.EventArray(
            brainstate.random.randn(k) < 0.5
        )
        if not asbool:
            events.value = u.math.asarray(events.value, dtype=float)
        matrix = brainstate.random.randn(k, m)
        out1 = events @ matrix
        out2 = events.data @ matrix
        assert u.math.allclose(out1, out2, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("float_as_event", [True, False])
    def test_matrix_event_mv(self, m, k, float_as_event):
        events = u.math.asarray(brainstate.random.randn(m) < 0.5, dtype=float)
        if not float_as_event:
            events = events * brainstate.random.rand(m)
        matrix = brainstate.random.randn(m, k)
        out1 = binary_vec_dot_dense_mat(events, matrix, float_as_event=float_as_event)
        out2 = events @ matrix
        assert u.math.allclose(out1, out2, atol=1e-4, rtol=1e-4)
