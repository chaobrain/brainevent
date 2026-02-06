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

import operator

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._jitc_matrix import JITCMatrix, _initialize_seed, _initialize_conn_length


class _DummyJITCMatrix(JITCMatrix):
    def __init__(self):
        super().__init__((), shape=(0, 0))

    def transpose(self, axes=None):
        return self

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

    def _unitary_op(self, op):
        return ("unitary", op)

    def _binary_op(self, other, op):
        return ("binary", op, other)

    def _binary_rop(self, other, op):
        return ("binary_r", op, other)


def test_unitary_operator_dispatch():
    mat = _DummyJITCMatrix()
    assert mat.__abs__() == ("unitary", operator.abs)
    assert mat.__neg__() == ("unitary", operator.neg)
    assert mat.__pos__() == ("unitary", operator.pos)


def test_binary_operator_dispatch():
    mat = _DummyJITCMatrix()
    other = jnp.asarray(2.0)
    assert mat * other == ("binary", operator.mul, other)
    assert mat / other == ("binary", operator.truediv, other)
    assert mat.__truediv__(other) == ("binary", operator.truediv, other)
    assert mat + other == ("binary", operator.add, other)
    assert mat - other == ("binary", operator.sub, other)
    assert mat % other == ("binary", operator.mod, other)


def test_binary_reflected_operator_dispatch():
    mat = _DummyJITCMatrix()
    other = jnp.asarray(3.0)
    assert other * mat == ("binary_r", operator.mul, other)
    assert other / mat == ("binary_r", operator.truediv, other)
    assert mat.__rtruediv__(other) == ("binary_r", operator.truediv, other)
    assert other + mat == ("binary_r", operator.add, other)
    assert other - mat == ("binary_r", operator.sub, other)
    assert other % mat == ("binary_r", operator.mod, other)


def test_initialize_seed_explicit_and_array():
    seed = _initialize_seed(123)
    assert seed.shape == (1,)
    assert seed.dtype == jnp.int32
    assert int(seed[0]) == 123

    seed_arr = _initialize_seed(np.asarray([5, 7], dtype=np.int64))
    assert seed_arr.shape == (2,)
    assert seed_arr.dtype == jnp.int32
    assert np.array_equal(np.asarray(seed_arr), np.asarray([5, 7], dtype=np.int32))


def test_initialize_seed_none():
    seed = _initialize_seed(None)
    assert seed.shape == (1,)
    assert seed.dtype == jnp.int32
    value = int(seed[0])
    assert 0 <= value < int(1e8)


def test_initialize_conn_length_values():
    clen = _initialize_conn_length(0.25)
    assert clen.dtype == jnp.int32
    assert int(clen) == 8

    clen2 = _initialize_conn_length(0.6)
    assert clen2.dtype == jnp.int32
    assert int(clen2) == 4


def test_initialize_conn_length_jit():
    @jax.jit
    def f(p):
        return _initialize_conn_length(p)

    clen = f(0.5)
    assert int(clen) == 4
