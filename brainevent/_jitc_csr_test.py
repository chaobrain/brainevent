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

from numba.cuda.kernels.transpose import transpose

from ._jitc_csr_float_impl import (
    _jitc_csr_matvec_homo,
    _jitc_csr_matmat_uniform,
    _jitc_csr_matvec_normal,
    _jitc_csr_matmat_homo,
    _jitc_csr_matmat_uniform,
    _jitc_csr_matmat_normal,
)



# 假设函数 _jitc_csr_matvec_homo 和 _raw_jitc_csr_matvec_homo 已正确导入

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
                            self.assertTrue(jnp.allclose(r1, r2, atol=1e-6))

if __name__ == '__main__':
    unittest.main()