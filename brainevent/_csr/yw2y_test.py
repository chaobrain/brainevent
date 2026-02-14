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


import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

from brainevent._csr.float import csrmv_p, csrmm_p
from brainevent._csr.yw2y import csrmv_yw2y, csrmv_yw2y_p
from brainevent._csr.test_util import get_csr

platform = jax.default_backend()
CSRMV_YW2Y_IMPLEMENTATIONS = tuple(csrmv_yw2y_p.available_backends(platform))


def _row_ids_from_indptr(indptr):
    indptr = jnp.asarray(indptr)
    counts = jnp.diff(indptr)
    return jnp.repeat(jnp.arange(counts.shape[0], dtype=indptr.dtype), counts)


@pytest.mark.skipif(
    not CSRMV_YW2Y_IMPLEMENTATIONS,
    reason=f'No csrmv_yw2y implementation on platform={platform}',
)
class TestCSRMVYw2y:
    @pytest.mark.parametrize('implementation', CSRMV_YW2Y_IMPLEMENTATIONS)
    @pytest.mark.parametrize('shape', [(100, 200), (200, 400)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr(self, implementation, shape, transpose):
        m, n = shape
        indptr, indices = get_csr(m, n, 0.5)

        data = braintools.init.Normal(0.0, 1.0)(indices.shape)
        y = brainstate.random.rand(n) if transpose else brainstate.random.rand(m)

        result = csrmv_yw2y(y, data, indices, indptr, shape=(m, n), transpose=transpose, backend=implementation)

        if transpose:
            expected = data * y[indices]
        else:
            row_ids = _row_ids_from_indptr(indptr)
            expected = data * y[row_ids]

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((data, y, indptr, indices, result, expected))
