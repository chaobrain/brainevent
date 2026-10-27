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

import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._csr.spsolve import csr_solve
from brainevent._test_util import jax_x64_enabled

_DENSE = np.array([[4., 1., 0.], [1., 3., 0.], [0., 0., 2.]])


@pytest.mark.parametrize('data_dtype, b_dtype', [(jnp.float32, jnp.float64), (jnp.float64, jnp.float32)])
def test_csr_solve_promotes_mixed_dtypes(data_dtype, b_dtype):
    """``spsolve`` needs one dtype; mixed float32/float64 operands are promoted first."""
    with jax_x64_enabled():
        csr = brainevent.CSR.fromdense(jnp.asarray(_DENSE, dtype=data_dtype))
        b = jnp.asarray([1., 2., 3.], dtype=b_dtype)
        x = csr_solve(csr.data, csr.indices, csr.indptr, b)
        assert x.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(x), np.linalg.solve(_DENSE, np.asarray(b, np.float64)), rtol=1e-5)


@pytest.mark.parametrize('cls_name', ['CSR', 'CSC'])
def test_matrix_solve_accepts_a_float64_rhs_under_x64(cls_name):
    with jax_x64_enabled():
        mat = getattr(brainevent, cls_name).fromdense(jnp.asarray(_DENSE, dtype=jnp.float32))
        b = jnp.asarray([1., 2., 3.])
        x = mat.solve(b)
        np.testing.assert_allclose(np.asarray(x), np.linalg.solve(_DENSE, np.asarray(b)), rtol=1e-5)


def test_csr_solve_promotes_mixed_dtypes_and_divides_units():
    import brainunit as u

    with jax_x64_enabled():
        csr = brainevent.CSR.fromdense(jnp.asarray(_DENSE, dtype=jnp.float32) * u.ohm)
        b = jnp.asarray([1., 2., 3.]) * u.volt
        x = csr.solve(b)
        assert u.get_unit(x) == u.amp
        assert u.get_mantissa(x).dtype == jnp.float64
        np.testing.assert_allclose(u.get_mantissa(x), np.linalg.solve(_DENSE, [1., 2., 3.]), rtol=1e-5)
