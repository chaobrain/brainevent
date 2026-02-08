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


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools
import brainevent
from brainevent._fcn.binary import binary_fcnmm_p_call
from brainevent._fcn.float_test import TestVector, TestMatrix
from brainevent._test_util import (
    gen_events,
    generate_fixed_conn_num_indices,
    allclose,
)


class TestEventVector(TestVector):
    def _generate_x(self, shape, require_float=True):
        if not isinstance(shape, (tuple, list)):
            shape = [shape]
        yield gen_events(shape, asbool=False)
        if not require_float:
            yield gen_events(shape, asbool=True)


class TestEventMatrix(TestMatrix):
    def _generate_x(self, shape, require_float=True):
        if not isinstance(shape, (tuple, list)):
            shape = [shape]
        yield gen_events(shape, asbool=False)
        if not require_float:
            yield gen_events(shape, asbool=True)


class TestBinaryFcnmmFloatThresholding:
    """Regression tests for C2/C3: binary_fcnmm must threshold float events.

    Calls binary_fcnmm_p_call directly to test that float events with
    arbitrary positive values (0.3, 0.7) are treated as 1.0 (fired),
    not used as raw float multipliers.
    """

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('shape', [(20, 40), (50, 30)])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_float_events_thresholded(self, homo_w, shape, k, transpose):
        m, n = shape
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=False)
        weights = jnp.array([1.5]) if homo_w else braintools.init.Normal(0., 1.)(indices.shape)

        # Create float event matrix with non-binary values (0.3, 0.7, 0.0, etc.)
        rng = np.random.RandomState(42)
        mat_rows = m if transpose else n
        raw = rng.rand(mat_rows, k).astype(np.float32)
        mask = rng.rand(mat_rows, k) > 0.5
        float_events = jnp.array(raw * mask)
        binary_events = jnp.where(float_events > 0., jnp.ones_like(float_events), 0.)

        y_float = jax.jit(
            lambda: binary_fcnmm_p_call(weights, indices, float_events, shape=(m, n), transpose=transpose)
        )()
        y_binary = jax.jit(
            lambda: binary_fcnmm_p_call(weights, indices, binary_events, shape=(m, n), transpose=transpose)
        )()

        assert allclose(y_float[0], y_binary[0], rtol=1e-3, atol=1e-3), \
            "binary_fcnmm should threshold float events, not use raw values"
