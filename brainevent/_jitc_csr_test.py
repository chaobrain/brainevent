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
    _jitc_csr_matvec_uniform,
    _jitc_csr_matvec_normal,
    _jitc_csr_matmat_homo,
    _jitc_csr_matmat_uniform,
    _jitc_csr_matmat_normal,
)


# jax.config.update('jax_default_device', jax.devices('cpu')[0])


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
                for prob in [0.3, 0.5]:
                    for transpose in [True, False]:
                        for outdim_parallel in [True, False]:
                            vector = jnp.asarray(np.random.random(shape[0] if transpose else shape[1]))
                            r1 = _jitc_csr_matvec_homo(weight, prob, vector, seed=seed, shape=shape,
                                                       transpose=transpose,
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


class TestJitcCsrMatvecUniform(unittest.TestCase):

    def test_zero_weight(self):
        w_low = 0.0
        w_high = 0.0
        conn_prob = 0.5
        v = jnp.array([1.0, 2.0, 3.0])
        shape = (2, 3)
        seed = 1234
        for transpose in [True, False]:
            for outdim_parallel in [True, False]:
                result = _jitc_csr_matvec_uniform(w_low, w_high, conn_prob, v, seed=seed, shape=shape,
                                                  transpose=transpose,
                                                  outdim_parallel=outdim_parallel)
                expected = jnp.zeros(shape[1]) if transpose else jnp.zeros(shape[0])
                self.assertTrue(jnp.allclose(result, expected), msg="Weight zero test failed")

    def test_random_connectivity(self):
        seed = 1234
        shapes = [(100, 200), (2, 1000), (1000, 2)]
        w_low = 1.0
        w_high = 2.0
        for shape in shapes:
            for prob in [0.3, 0.5]:
                for transpose in [True, False]:
                    for outdim_parallel in [True, False]:
                        vector = jnp.asarray(np.random.random(shape[0] if transpose else shape[1]))
                        r1 = _jitc_csr_matvec_uniform(w_low, w_high, prob, vector, seed=seed, shape=shape,
                                                      transpose=transpose,
                                                      outdim_parallel=outdim_parallel)
                        r2 = _jitc_csr_matvec_uniform(w_low, w_high, prob, vector, seed=seed, shape=shape,
                                                      transpose=transpose,
                                                      outdim_parallel=outdim_parallel)
                        # print(f'transpose: {transpose}, outdim_parallel: {outdim_parallel}')
                        # print(r1)
                        self.assertTrue(jnp.allclose(r1, r2, atol=1e-6))

    def _test_jvp(self, prob, transpose, outdim_parallel):
        seed = 1234
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        w_low = 1.0
        w_high = 2.0

        x = jnp.asarray(np.random.random(n_in if transpose else n_out))

        def f_brainevent(x, w_low, w_high):
            return _jitc_csr_matvec_uniform(w_low, w_high, prob, x, seed=seed, shape=shape,
                                         transpose=transpose, outdim_parallel=outdim_parallel)

        out1, jvp_x1 = jax.jvp(f_brainevent, (x, jnp.array(w_low), jnp.array(w_high)),
                               (jnp.ones_like(x), jnp.array(1.0), jnp.array(1.0)))

        out2, jvp_x2 = jax.jvp(f_brainevent, (x, jnp.array(w_low), jnp.array(w_high)),
                               (jnp.ones_like(x), jnp.array(1.0), jnp.array(1.0)))

        self.assertTrue(jnp.allclose(out1, out2, rtol=1e-5, atol=1e-5))
        self.assertTrue(jnp.allclose(jvp_x1, jvp_x2, rtol=1e-5, atol=1e-5))

        self.assertFalse(jnp.allclose(jvp_x1, jnp.zeros_like(jvp_x1)))

    def test_jvp(self):
        for prob in [0.5]:
            for transpose in [True, False]:
                for outdim_parallel in [True, False]:
                    print(
                        f'prob = {prob}, transpose = {transpose}, outdim_parallel = {outdim_parallel}')
                    self._test_jvp(prob=prob, transpose=transpose, outdim_parallel=outdim_parallel)


class TestJitcCsrMatvecNormal(unittest.TestCase):

    def test_zero_weight(self):
        w_mu = 0.0
        w_sigma = 0.0
        conn_prob = 0.5
        v = jnp.array([1.0, 2.0, 3.0])
        shape = (2, 3)
        seed = 1234
        for transpose in [True, False]:
            for outdim_parallel in [True, False]:
                result = _jitc_csr_matvec_normal(w_mu, w_sigma, conn_prob, v, seed=seed, shape=shape,
                                                  transpose=transpose,
                                                  outdim_parallel=outdim_parallel)
                expected = jnp.zeros(shape[1]) if transpose else jnp.zeros(shape[0])
                self.assertTrue(jnp.allclose(result, expected), msg="Weight zero test failed")

    def test_random_connectivity(self):
        seed = 1234
        shapes = [(100, 200), (2, 1000), (1000, 2)]
        w_mu = 1.0
        w_sigma = 2.0
        for shape in shapes:
            for prob in [0.3, 0.5]:
                for transpose in [True, False]:
                    for outdim_parallel in [True, False]:
                        vector = jnp.asarray(np.random.random(shape[0] if transpose else shape[1]))
                        r1 = _jitc_csr_matvec_normal(w_mu, w_sigma, prob, vector, seed=seed, shape=shape,
                                                      transpose=transpose,
                                                      outdim_parallel=outdim_parallel)
                        r2 = _jitc_csr_matvec_normal(w_mu, w_sigma, prob, vector, seed=seed, shape=shape,
                                                      transpose=transpose,
                                                      outdim_parallel=outdim_parallel)
                        # print(f'transpose: {transpose}, outdim_parallel: {outdim_parallel}')
                        # print(r1)
                        self.assertTrue(jnp.allclose(r1, r2, atol=1e-6))

    def _test_jvp(self, prob, transpose, outdim_parallel):
        seed = 1234
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        w_mu = 1.0
        w_sigma = 2.0

        x = jnp.asarray(np.random.random(n_in if transpose else n_out))

        def f_brainevent(x, w_mu, w_sigma):
            return _jitc_csr_matvec_uniform(w_mu, w_sigma, prob, x, seed=seed, shape=shape,
                                            transpose=transpose, outdim_parallel=outdim_parallel)

        out1, jvp_x1 = jax.jvp(f_brainevent, (x, jnp.array(w_mu), jnp.array(w_sigma)),
                               (jnp.ones_like(x), jnp.array(1.0), jnp.array(1.0)))

        out2, jvp_x2 = jax.jvp(f_brainevent, (x, jnp.array(w_mu), jnp.array(w_sigma)),
                               (jnp.ones_like(x), jnp.array(1.0), jnp.array(1.0)))

        self.assertTrue(jnp.allclose(out1, out2, rtol=1e-5, atol=1e-5))
        self.assertTrue(jnp.allclose(jvp_x1, jvp_x2, rtol=1e-5, atol=1e-5))

        self.assertFalse(jnp.allclose(jvp_x1, jnp.zeros_like(jvp_x1)))

    def test_jvp(self):
        for prob in [0.5]:
            for transpose in [True, False]:
                for outdim_parallel in [True, False]:
                    print(
                        f'prob = {prob}, transpose = {transpose}, outdim_parallel = {outdim_parallel}')
                    self._test_jvp(prob=prob, transpose=transpose, outdim_parallel=outdim_parallel)


class TestJitcCsrMatmatHomo(unittest.TestCase):

    def test_zero_weight(self):
        weight = 0.0
        conn_prob = 0.5
        B = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2 matrix
        shape = (2, 3)
        seed = 1234
        for transpose in [True, False]:
            for outdim_parallel in [True, False]:
                result = _jitc_csr_matmat_homo(weight, conn_prob, B, seed=seed, shape=shape, transpose=transpose,
                                               outdim_parallel=outdim_parallel)
                # Expected shape depends on transpose operation
                expected_shape = (shape[1], B.shape[1]) if transpose else (shape[0], B.shape[1])
                expected = jnp.zeros(expected_shape)
                self.assertTrue(jnp.allclose(result, expected), msg="Weight zero test failed")

    def test_random_connectivity(self):
        seed = 1234
        shapes = [
            (100, 200),
            (2, 100),
            (100, 2)
        ]
        batch_sizes = [10, 20]  # Batch dimension (second dimension of B matrix)

        for shape in shapes:
            for batch_size in batch_sizes:
                for weight in [-1., 1.]:
                    for prob in [0.3, 0.5]:
                        for transpose in [True, False]:
                            for outdim_parallel in [True, False]:
                                print(f'shape: {shape}, batch_size: {batch_size}, weight: {weight}, prob: {prob}, '
                                      f'transpose: {transpose}, outdim_parallel: {outdim_parallel}')
                                # Input matrix B
                                B_shape = (shape[0] if transpose else shape[1], batch_size)
                                B = jnp.asarray(np.random.random(B_shape))

                                r1 = _jitc_csr_matmat_homo(weight, prob, B, seed=seed, shape=shape,
                                                           transpose=transpose,
                                                           outdim_parallel=outdim_parallel)
                                r2 = _jitc_csr_matmat_homo(weight, prob, B, seed=seed, shape=shape,
                                                           transpose=transpose,
                                                           outdim_parallel=outdim_parallel)
                                # Results should be deterministic for same seed
                                # print(jnp.sum(r1 - r2))
                                # print(r1 - r2)
                                self.assertTrue(jnp.allclose(r1, r2, atol=1e-6, equal_nan=True))

                                # Check output shape
                                expected_shape = (shape[1], batch_size) if transpose else (shape[0], batch_size)
                                self.assertEqual(r1.shape, expected_shape)

    def _test_jvp(self, weight, prob, transpose, outdim_parallel):
        seed = 1234
        n_in = 20
        n_out = 30
        batch_size = 15
        shape = (n_in, n_out)

        # Input matrix X
        X_shape = (n_in if transpose else n_out, batch_size)
        X = jnp.asarray(np.random.random(X_shape))

        def f_brainevent(X, w):
            return _jitc_csr_matmat_homo(w, prob, X, seed=seed, shape=shape,
                                         transpose=transpose, outdim_parallel=outdim_parallel)

        # Test JVP for both input matrix X and weight w
        out1, jvp_x1 = jax.jvp(f_brainevent, (X, jnp.array(weight)),
                               (jnp.ones_like(X), jnp.array(1.0)))

        out2, jvp_x2 = jax.jvp(f_brainevent, (X, jnp.array(weight)),
                               (jnp.ones_like(X), jnp.array(1.0)))

        # Results should be consistent
        self.assertTrue(jnp.allclose(out1, out2, rtol=1e-5, atol=1e-5, equal_nan=True))
        self.assertTrue(jnp.allclose(jvp_x1, jvp_x2, rtol=1e-5, atol=1e-5, equal_nan=True))

        # Verify that JVP has meaningful values (not all zeros)
        self.assertFalse(jnp.allclose(jvp_x1, jnp.zeros_like(jvp_x1)))

        # Check output shapes
        expected_shape = (shape[1], batch_size) if transpose else (shape[0], batch_size)
        self.assertEqual(out1.shape, expected_shape)
        self.assertEqual(jvp_x1.shape, expected_shape)

    def test_jvp(self):
        for weight in [-1., 1.]:
            for prob in [0.5]:
                for transpose in [True, False]:
                    for outdim_parallel in [True, False]:
                        print(
                            f'prob = {prob}, transpose = {transpose}, outdim_parallel = {outdim_parallel}, weight = {weight}')
                        self._test_jvp(weight=weight, prob=prob, transpose=transpose, outdim_parallel=outdim_parallel)

class TestJitcCsrMatmatUniform(unittest.TestCase):

    def test_zero_weight(self):
        w_low = 0.0
        w_high = 0.0
        conn_prob = 0.5
        B = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2 matrix
        shape = (2, 3)
        seed = 1234
        for transpose in [True, False]:
            for outdim_parallel in [True, False]:
                result = _jitc_csr_matmat_uniform(w_low, w_high, conn_prob, B, seed=seed, shape=shape, transpose=transpose,
                                               outdim_parallel=outdim_parallel)
                # Expected shape depends on transpose operation
                expected_shape = (shape[1], B.shape[1]) if transpose else (shape[0], B.shape[1])
                expected = jnp.zeros(expected_shape)
                self.assertTrue(jnp.allclose(result, expected), msg="Weight zero test failed")

    def test_random_connectivity(self):
        seed = 1234
        shapes = [
            (100, 200),
            (2, 100),
            (100, 2)
        ]
        batch_sizes = [10, 20]  # Batch dimension (second dimension of B matrix)
        w_low = 0.0
        w_high = 1.0

        for shape in shapes:
            for batch_size in batch_sizes:
                for prob in [0.3, 0.5]:
                    for transpose in [True, False]:
                        for outdim_parallel in [True, False]:
                            # Input matrix B
                            B_shape = (shape[0] if transpose else shape[1], batch_size)
                            B = jnp.asarray(np.random.random(B_shape))

                            r1 = _jitc_csr_matmat_uniform(w_low, w_high, prob, B, seed=seed, shape=shape,
                                                       transpose=transpose,
                                                       outdim_parallel=outdim_parallel)
                            r2 = _jitc_csr_matmat_uniform(w_low, w_high, prob, B, seed=seed, shape=shape,
                                                       transpose=transpose,
                                                       outdim_parallel=outdim_parallel)
                            # Results should be deterministic for same seed
                            # print(jnp.sum(r1 - r2))
                            # print(r1 - r2)
                            self.assertTrue(jnp.allclose(r1, r2, atol=1e-6, equal_nan=True))

                            # Check output shape
                            expected_shape = (shape[1], batch_size) if transpose else (shape[0], batch_size)
                            self.assertEqual(r1.shape, expected_shape)

    def _test_jvp(self, prob, transpose, outdim_parallel):
        seed = 1234
        n_in = 20
        n_out = 30
        batch_size = 15
        shape = (n_in, n_out)
        w_low = 0.0
        w_high = 1.0

        # Input matrix X
        X_shape = (n_in if transpose else n_out, batch_size)
        X = jnp.asarray(np.random.random(X_shape))

        def f_brainevent(X, w_low, w_high):
            return _jitc_csr_matmat_uniform(w_low, w_high, prob, X, seed=seed, shape=shape,
                                         transpose=transpose, outdim_parallel=outdim_parallel)

        # Test JVP for both input matrix X and weight w
        out1, jvp_x1 = jax.jvp(f_brainevent, (X, jnp.array(w_low), jnp.array(w_high)),
                               (jnp.ones_like(X), jnp.array(1.0), jnp.array(1.0)))

        out2, jvp_x2 = jax.jvp(f_brainevent, (X, jnp.array(w_low), jnp.array(w_high)),
                               (jnp.ones_like(X), jnp.array(1.0), jnp.array(1.0)))

        # Results should be consistent
        self.assertTrue(jnp.allclose(out1, out2, rtol=1e-5, atol=1e-5, equal_nan=True))
        self.assertTrue(jnp.allclose(jvp_x1, jvp_x2, rtol=1e-5, atol=1e-5, equal_nan=True))

        # Verify that JVP has meaningful values (not all zeros)
        self.assertFalse(jnp.allclose(jvp_x1, jnp.zeros_like(jvp_x1)))

        # Check output shapes
        expected_shape = (shape[1], batch_size) if transpose else (shape[0], batch_size)
        self.assertEqual(out1.shape, expected_shape)
        self.assertEqual(jvp_x1.shape, expected_shape)

    def test_jvp(self):
        for prob in [0.5]:
            for transpose in [True, False]:
                for outdim_parallel in [True, False]:
                    print(
                        f'prob = {prob}, transpose = {transpose}, outdim_parallel = {outdim_parallel}')
                    self._test_jvp(prob=prob, transpose=transpose, outdim_parallel=outdim_parallel)

class TestJitcCsrMatmatNormal(unittest.TestCase):

    def test_zero_weight(self):
        w_mu = 0.0
        w_sigma = 0.0
        conn_prob = 0.5
        B = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2 matrix
        shape = (2, 3)
        seed = 1234
        for transpose in [True, False]:
            for outdim_parallel in [True, False]:
                result = _jitc_csr_matmat_normal(w_mu, w_sigma, conn_prob, B, seed=seed, shape=shape, transpose=transpose,
                                               outdim_parallel=outdim_parallel)
                # Expected shape depends on transpose operation
                expected_shape = (shape[1], B.shape[1]) if transpose else (shape[0], B.shape[1])
                expected = jnp.zeros(expected_shape)
                self.assertTrue(jnp.allclose(result, expected), msg="Weight zero test failed")

    def test_random_connectivity(self):
        seed = 1234
        shapes = [
            (100, 200),
            (2, 100),
            (100, 2)
        ]
        batch_sizes = [10, 20]  # Batch dimension (second dimension of B matrix)
        w_mu = 0.0
        w_sigma = 1.0

        for shape in shapes:
            for batch_size in batch_sizes:
                for prob in [0.3, 0.5]:
                    for transpose in [True, False]:
                        for outdim_parallel in [True, False]:
                            # Input matrix B
                            B_shape = (shape[0] if transpose else shape[1], batch_size)
                            B = jnp.asarray(np.random.random(B_shape))

                            r1 = _jitc_csr_matmat_normal(w_mu, w_sigma, prob, B, seed=seed, shape=shape,
                                                       transpose=transpose,
                                                       outdim_parallel=outdim_parallel)
                            r2 = _jitc_csr_matmat_normal(w_mu, w_sigma, prob, B, seed=seed, shape=shape,
                                                       transpose=transpose,
                                                       outdim_parallel=outdim_parallel)
                            # Results should be deterministic for same seed
                            # print(jnp.sum(r1 - r2))
                            # print(r1 - r2)
                            self.assertTrue(jnp.allclose(r1, r2, atol=1e-6, equal_nan=True))

                            # Check output shape
                            expected_shape = (shape[1], batch_size) if transpose else (shape[0], batch_size)
                            self.assertEqual(r1.shape, expected_shape)

    def _test_jvp(self, prob, transpose, outdim_parallel):
        seed = 1234
        n_in = 20
        n_out = 30
        batch_size = 15
        shape = (n_in, n_out)
        w_mu = 0.0
        w_sigma = 1.0

        # Input matrix X
        X_shape = (n_in if transpose else n_out, batch_size)
        X = jnp.asarray(np.random.random(X_shape))

        def f_brainevent(X, w_mu, w_sigma):
            return _jitc_csr_matmat_normal(w_mu, w_sigma, prob, X, seed=seed, shape=shape,
                                         transpose=transpose, outdim_parallel=outdim_parallel)

        # Test JVP for both input matrix X and weight w
        out1, jvp_x1 = jax.jvp(f_brainevent, (X, jnp.array(w_mu), jnp.array(w_sigma)),
                               (jnp.ones_like(X), jnp.array(1.0), jnp.array(1.0)))

        out2, jvp_x2 = jax.jvp(f_brainevent, (X, jnp.array(w_mu), jnp.array(w_sigma)),
                               (jnp.ones_like(X), jnp.array(1.0), jnp.array(1.0)))

        # Results should be consistent
        self.assertTrue(jnp.allclose(out1, out2, rtol=1e-5, atol=1e-5, equal_nan=True))
        self.assertTrue(jnp.allclose(jvp_x1, jvp_x2, rtol=1e-5, atol=1e-5, equal_nan=True))

        # Verify that JVP has meaningful values (not all zeros)
        self.assertFalse(jnp.allclose(jvp_x1, jnp.zeros_like(jvp_x1)))

        # Check output shapes
        expected_shape = (shape[1], batch_size) if transpose else (shape[0], batch_size)
        self.assertEqual(out1.shape, expected_shape)
        self.assertEqual(jvp_x1.shape, expected_shape)

    def test_jvp(self):
        for prob in [0.5]:
            for transpose in [True, False]:
                for outdim_parallel in [True, False]:
                    print(
                        f'prob = {prob}, transpose = {transpose}, outdim_parallel = {outdim_parallel}')
                    self._test_jvp(prob=prob, transpose=transpose, outdim_parallel=outdim_parallel)


if __name__ == '__main__':
    unittest.main()
