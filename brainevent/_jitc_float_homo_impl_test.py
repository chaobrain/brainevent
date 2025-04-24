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


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._jitc_float_homo_impl import (
    jitc_matvec_homo,
    jitc_matmat_homo,
)


# jax.config.update('jax_default_device', jax.devices('cpu')[0])

def equal(a, b):
    return a == b


class TestJitcCsrMatvecHomo:

    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('outdim_parallel', [True, False])
    def test_zero_weight(self, transpose, outdim_parallel):
        weight = 0.0
        conn_prob = 0.5
        v = jnp.array([1.0, 2.0, 3.0])
        shape = (2, 3)
        seed = 1234
        result = jitc_matvec_homo(
            weight,
            conn_prob,
            v,
            seed=seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
        expected = jnp.zeros(shape[1]) if transpose else jnp.zeros(shape[0])
        assert (jnp.allclose(result, expected))

    @pytest.mark.parametrize('shape', [(100, 200), (20, 100), (100, 20)])
    @pytest.mark.parametrize('weight', [-1., 1.])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('outdim_parallel', [True, False])
    def test_random_connectivity(self, shape, weight, prob, transpose, outdim_parallel):
        seed = 1234
        vector = jnp.asarray(np.random.random(shape[0] if transpose else shape[1]))
        r1 = jitc_matvec_homo(
            weight,
            prob,
            vector,
            seed=seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
        r2 = jitc_matvec_homo(
            weight,
            prob,
            vector,
            seed=seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
        print(r1)
        assert (jnp.allclose(r1, r2, atol=1e-6))

    @pytest.mark.parametrize('weight', [-1., 1.])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('outdim_parallel', [True, False])
    def test_jvp(self, weight, prob, transpose, outdim_parallel):
        seed = 1234
        n_in = 200
        n_out = 300
        shape = (n_in, n_out)

        x = jnp.asarray(np.random.random(n_in if transpose else n_out))

        def f_brainevent(x, w):
            return jitc_matvec_homo(
                w,
                prob,
                x,
                seed=seed,
                shape=shape,
                transpose=transpose,
                outdim_parallel=outdim_parallel
            )

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (jnp.ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (jnp.ones_like(x), jnp.array(1.0))
        )

        assert (jnp.allclose(out1, out2, rtol=1e-5, atol=1e-5))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-5, atol=1e-5))


class TestJitcCsrMatmatHomo:
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('outdim_parallel', [True, False])
    def test_zero_weight(self, transpose, outdim_parallel):
        weight = 0.0
        conn_prob = 0.5
        B = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2 matrix
        shape = (2, 3)
        seed = 1234

        result = jitc_matmat_homo(
            weight,
            conn_prob,
            B,
            seed=seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
        # Expected shape depends on transpose operation
        expected_shape = (shape[1], B.shape[1]) if transpose else (shape[0], B.shape[1])
        expected = jnp.zeros(expected_shape)
        assert (jnp.allclose(result, expected))

    @pytest.mark.parametrize('shape', [(100, 200), (20, 100), (100, 20)])
    @pytest.mark.parametrize('batch_size', [10, 20])
    @pytest.mark.parametrize('weight', [-1., 1.])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('outdim_parallel', [True, False])
    def test_random_connectivity(self, shape, batch_size, weight, prob, transpose, outdim_parallel):
        seed = 1234

        print(
            f'shape: {shape}, \n'
            f'batch_size: {batch_size}, \n'
            f'weight: {weight}, \n'
            f'prob: {prob}, \n'
            f'transpose: {transpose}, \n'
            f'outdim_parallel: {outdim_parallel}'
        )

        # Input matrix B
        B_shape = (shape[0] if transpose else shape[1], batch_size)
        B = jnp.asarray(np.random.random(B_shape))

        r1 = jitc_matmat_homo(
            weight,
            prob,
            B,
            seed=seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
        r2 = jitc_matmat_homo(
            weight,
            prob,
            B,
            seed=seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
        # Results should be deterministic for same seed
        # print(jnp.sum(r1 - r2))
        print(r1)
        print(r1 - r2)
        assert (jnp.allclose(r1, r2, atol=1e-6, equal_nan=True))

        # Check output shape
        expected_shape = (shape[1], batch_size) if transpose else (shape[0], batch_size)
        assert equal(r1.shape, expected_shape)

    @pytest.mark.parametrize('weight', [-1., 1.])
    @pytest.mark.parametrize('prob', [0.3, 0.5])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('outdim_parallel', [True, False])
    def test_jvp(self, weight, prob, transpose, outdim_parallel):
        seed = 1234
        n_in = 200
        n_out = 300
        batch_size = 15
        shape = (n_in, n_out)

        # Input matrix X
        X_shape = (n_in if transpose else n_out, batch_size)
        X = jnp.asarray(np.random.random(X_shape))

        def f_brainevent(X, w):
            return jitc_matmat_homo(
                w,
                prob,
                X,
                seed=seed,
                shape=shape,
                transpose=transpose,
                outdim_parallel=outdim_parallel
            )

        # Test JVP for both input matrix X and weight w
        out1, jvp_x1 = jax.jvp(f_brainevent,
                               (X, jnp.array(weight)),
                               (jnp.ones_like(X), jnp.array(1.0)))

        out2, jvp_x2 = jax.jvp(f_brainevent,
                               (X, jnp.array(weight)),
                               (jnp.ones_like(X), jnp.array(1.0)))

        # Results should be consistent
        assert (jnp.allclose(out1, out2, rtol=1e-5, atol=1e-5, equal_nan=True))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-5, atol=1e-5, equal_nan=True))

        # Check output shapes
        expected_shape = (shape[1], batch_size) if transpose else (shape[0], batch_size)
        assert equal(out1.shape, expected_shape)
        assert equal(jvp_x1.shape, expected_shape)
