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


import brainstate
import brainunit as u
import pytest

import brainevent
from brainevent._dense.binary import (
    dense_mat_dot_binary_mat,
    binary_mat_dot_dense_mat,
    dense_mat_dot_binary_vec,
    binary_vec_dot_dense_mat,
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
        out2 = matrix @ (events.data).astype(float)
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    def test_dense_mat_dot_binary_mat(self, m, k, n):
        matrix = brainstate.random.randn(m, k)
        events = u.math.asarray(brainstate.random.randn(k, n) < 0.5, dtype=float)
        out1 = dense_mat_dot_binary_mat(matrix, events)
        out2 = matrix @ events
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)


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
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    @pytest.mark.parametrize("n", [30])
    def test_dense_mat_dot_binary_mat(self, m, k, n):
        events = u.math.asarray(brainstate.random.randn(m, k) < 0.5, dtype=float)
        matrix = brainstate.random.randn(k, n)
        out1 = binary_mat_dot_dense_mat(events, matrix)
        out2 = events @ matrix
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)


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
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    def test_matrix_event_mv(self, m, k):
        matrix = brainstate.random.randn(m, k)
        events = u.math.asarray(brainstate.random.randn(k) < 0.5, dtype=float)
        out1 = dense_mat_dot_binary_vec(matrix, events)
        out2 = matrix @ events
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)


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
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m", [10])
    @pytest.mark.parametrize("k", [15, 20])
    def test_matrix_event_mv(self, m, k):
        events = u.math.asarray(brainstate.random.randn(m) < 0.5, dtype=float)
        matrix = brainstate.random.randn(m, k)
        out1 = binary_vec_dot_dense_mat(events, matrix)
        out2 = events @ matrix
        assert u.math.allclose(out1, out2, atol=1e-3, rtol=1e-3)
