# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


import unittest
import numpy as np
import jax
import jax.numpy as jnp
import brainstate as bs
from functools import partial

# jax.config.update('jax_default_device', jax.devices('cpu')[0])

from ._jitc_csr_float_impl import (
    _jitc_csr_matvec_homo,
    _jitc_csr_matmat_uniform,
    _jitc_csr_matvec_normal,
    _jitc_csr_matmat_homo,
    _jitc_csr_matmat_uniform,
    _jitc_csr_matmat_normal,
)

class TestJitcCsrMatvecHomo(unittest.TestCase):

    def test_zero_weight(self):
        weight = 0.0
        conn_prob = 0.5
        v = jnp.array([1.0, 2.0, 3.0])
        shape = (2, 3)
        seed = 1234
        for transpose in [True, False]:
            for outdim_parallel in [True, False]:
                result = _jitc_csr_matvec_homo(weight, conn_prob, v, seed=seed, shape=shape, transpose=transpose,
                                               outdim_parallel=outdim_parallel)
                expected = jnp.zeros(shape[1]) if transpose else jnp.zeros(shape[0])
                self.assertTrue(jnp.allclose(result, expected), msg="Weight zero test failed")

    def test_random_connectivity(self):
        seed = 1234
        shapes = [(100, 200), (2, 1000), (1000, 2)]
        for shape in shapes:
            for weight in [-1., 1.]:
                for prob in [0.01, 0.1, 0.5]:
                    for transpose in [True, False]:
                        for outdim_parallel in [True, False]:
                            vector = jnp.asarray(np.random.random(shape[0] if transpose else shape[1]))
                            r1 = _jitc_csr_matvec_homo(weight, prob, vector, seed=seed, shape=shape, transpose=transpose,
                                                            outdim_parallel=outdim_parallel)
                            r2 = _jitc_csr_matvec_homo(weight, prob, vector, seed=seed, shape=shape,
                                                            transpose=transpose,
                                                            outdim_parallel=outdim_parallel)
                            # print(f'transpose: {transpose}, outdim_parallel: {outdim_parallel}')
                            # print(r1)
                            self.assertTrue(jnp.allclose(r1, r2, atol=1e-6))

    def _test_jvp(self, weight, prob, transpose, outdim_parallel):
        seed = 1234
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)

        x = jnp.asarray(np.random.random(n_in if transpose else n_out))

        def f_brainevent(x, w):
            return _jitc_csr_matvec_homo(w, prob, x, seed=seed, shape=shape,
                                         transpose=transpose, outdim_parallel=outdim_parallel)

        out1, jvp_x1 = jax.jvp(f_brainevent, (x, jnp.array(weight)),
                               (jnp.ones_like(x), jnp.array(1.0)))

        out2, jvp_x2 = jax.jvp(f_brainevent, (x, jnp.array(weight)),
                               (jnp.ones_like(x), jnp.array(1.0)))

        self.assertTrue(jnp.allclose(out1, out2, rtol=1e-5, atol=1e-5))
        self.assertTrue(jnp.allclose(jvp_x1, jvp_x2, rtol=1e-5, atol=1e-5))

        self.assertFalse(jnp.allclose(jvp_x1, jnp.zeros_like(jvp_x1)))

    def test_jvp(self):
        for weight in [-1., 1.]:
            for prob in [0.5]:
                for transpose in [True, False]:
                    for outdim_parallel in [True, False]:
                        print(
                            f'prob = {prob}, transpose = {transpose}, outdim_parallel = {outdim_parallel}, weight = {weight}')
                        self._test_jvp(weight=weight, prob=prob, transpose=transpose, outdim_parallel=outdim_parallel)


if __name__ == '__main__':
    unittest.main()