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
import inspect

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
import brainevent as be
from brainevent import BrainEventError, DataRepresentation, UnsupportedOperationError
from brainevent._data import JITCMatrix, _initialize_seed, _initialize_conn_length


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


def test_jitc_dt2t_signature_aligns_data_representation_contract():
    sig = inspect.signature(JITCMatrix.dt2t)
    base_sig = inspect.signature(DataRepresentation.dt2t)
    assert list(sig.parameters) == list(base_sig.parameters)
    assert sig.parameters['y_dim_arr'].annotation == base_sig.parameters['y_dim_arr'].annotation
    assert sig.parameters['w_dim_arr'].annotation == base_sig.parameters['w_dim_arr'].annotation
    assert sig.parameters['w_dim_arr'].default is inspect._empty
    assert sig.return_annotation == base_sig.return_annotation


def test_jitc_dt2t_transposed_signature_aligns_data_representation_contract():
    sig = inspect.signature(JITCMatrix.dt2t_transposed)
    base_sig = inspect.signature(DataRepresentation.dt2t_transposed)
    assert list(sig.parameters) == list(base_sig.parameters)
    assert sig.parameters['y_dim_arr'].annotation == base_sig.parameters['y_dim_arr'].annotation
    assert sig.parameters['w_dim_arr'].annotation == base_sig.parameters['w_dim_arr'].annotation
    assert sig.parameters['w_dim_arr'].default is inspect._empty
    assert sig.return_annotation == base_sig.return_annotation


def test_jitc_dt2t_fallbacks_remain_unsupported():
    mat = _DummyJITCMatrix()
    y = jnp.ones(0)
    w = jnp.ones(0)
    with pytest.raises(brainevent.UnsupportedOperationError):
        mat.dt2t(y, w)
    with pytest.raises(brainevent.UnsupportedOperationError):
        mat.dt2t_transposed(y, w)


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


# ---------------------------------------------------------------------------
# Helpers: minimal concrete subclass for testing the base mechanism
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class _SimpleBuffered(DataRepresentation):
    """Minimal concrete DataRepresentation for testing the buffer API."""

    def __init__(self, value, *, shape, buffers=None):
        self.value = jnp.asarray(value)
        super().__init__((value,), shape=shape)  # creates _buffer_registry
        self.register_buffer('cached_sum', None)
        self.register_buffer('label', None)
        if buffers is not None:
            for name, val in buffers.items():
                self.register_buffer(name, val)

    def transpose(self, axes=None):
        return self

    def tree_flatten(self):
        aux = {'shape': self.shape, 'value': self.value}
        return (), (aux, self.buffers)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        aux_data, buffer = aux_data
        obj._buffer_registry = set(buffer.keys())
        for k, v in aux_data.items():
            setattr(obj, k, v)
        for k, v in buffer.items():
            setattr(obj, k, v)
        return obj


# ===========================================================================
# 1. register_buffer basics
# ===========================================================================

class TestRegisterBuffer:
    def test_register_with_default_none(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        assert obj.cached_sum is None
        assert obj.label is None

    def test_register_with_value(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        obj.register_buffer('extra', 42)
        assert obj.extra == 42
        assert 'extra' in obj._buffer_registry

    def test_registry_tracks_names(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        assert 'cached_sum' in obj._buffer_registry
        assert 'label' in obj._buffer_registry
        assert len(obj._buffer_registry) == 2

    def test_register_overwrites_previous_value(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        obj.register_buffer('cached_sum', 10)
        assert obj.cached_sum == 10
        obj.register_buffer('cached_sum', 20)
        assert obj.cached_sum == 20
        assert list(obj._buffer_registry).count('cached_sum') == 1


# ===========================================================================
# 2. set_buffer
# ===========================================================================

class TestSetBuffer:
    def test_set_registered_buffer(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        obj.set_buffer('cached_sum', jnp.array(99.0))
        assert float(obj.cached_sum) == 99.0

    def test_set_unregistered_raises(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        with pytest.raises(ValueError, match="not registered"):
            obj.set_buffer('nonexistent', 5)


# ===========================================================================
# 3. buffers property
# ===========================================================================

class TestBuffersProperty:
    def test_returns_dict_of_registered_buffers(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        bufs = obj.buffers
        assert isinstance(bufs, dict)
        assert set(bufs.keys()) == {'cached_sum', 'label'}
        assert bufs['cached_sum'] is None
        assert bufs['label'] is None

    def test_reflects_updated_values(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        obj.cached_sum = jnp.array(3.14)
        obj.label = 'test'
        bufs = obj.buffers
        assert float(bufs['cached_sum']) == pytest.approx(3.14)
        assert bufs['label'] == 'test'

    def test_buffers_returns_new_dict_each_time(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        d1 = obj.buffers
        d2 = obj.buffers
        assert d1 == d2
        assert d1 is not d2


# ===========================================================================
# 4. Constructor buffers kwarg
# ===========================================================================

class TestBuffersKwarg:
    def test_overrides_defaults(self):
        bufs = {'cached_sum': jnp.array(7.0), 'label': 'hello'}
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=bufs)
        assert float(obj.cached_sum) == 7.0
        assert obj.label == 'hello'

    def test_partial_override(self):
        bufs = {'cached_sum': jnp.array(5.0)}
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=bufs)
        assert float(obj.cached_sum) == 5.0
        assert obj.label is None

    def test_none_is_noop(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=None)
        assert obj.cached_sum is None
        assert obj.label is None

    def test_empty_dict_is_noop(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers={})
        assert obj.cached_sum is None
        assert obj.label is None

    def test_registers_new_buffers(self):
        bufs = {'cached_sum': 1, 'label': 2, 'brand_new': 42}
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=bufs)
        assert obj.brand_new == 42
        assert 'brand_new' in obj._buffer_registry


# ===========================================================================
# 5. JAX pytree round-trip
# ===========================================================================

class TestPytreeRoundTrip:
    def test_flatten_unflatten_preserves_buffers(self):
        obj = _SimpleBuffered(1.0, shape=(3, 3))
        obj.set_buffer('cached_sum', jnp.array(123.0))
        obj.set_buffer('label', 'test_label')

        children, aux = obj.tree_flatten()
        restored = _SimpleBuffered.tree_unflatten(aux, children)

        assert float(restored.cached_sum) == 123.0
        assert restored.label == 'test_label'
        assert restored._buffer_registry == {'cached_sum', 'label'}

    def test_jax_tree_map_preserves_buffers(self):
        obj = _SimpleBuffered(1.0, shape=(3, 3))
        obj.set_buffer('cached_sum', jnp.array(5.0))

        mapped = jax.tree.map(lambda x: x, obj)
        assert isinstance(mapped, _SimpleBuffered)
        assert float(mapped.cached_sum) == 5.0
        assert mapped._buffer_registry == {'cached_sum', 'label'}

    def test_jit_preserves_buffers(self):
        obj = _SimpleBuffered(jnp.array(2.0), shape=(3, 3))
        obj.set_buffer('cached_sum', jnp.array(10.0))

        @jax.jit
        def identity(x):
            return x

        result = identity(obj)
        assert isinstance(result, _SimpleBuffered)
        assert float(result.cached_sum) == 10.0

    def test_none_buffer_values_roundtrip(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        children, aux = obj.tree_flatten()
        restored = _SimpleBuffered.tree_unflatten(aux, children)
        assert restored.cached_sum is None
        assert restored.label is None
        assert restored._buffer_registry == {'cached_sum', 'label'}

    def test_registry_is_mutable_set_after_unflatten(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        children, aux = obj.tree_flatten()
        restored = _SimpleBuffered.tree_unflatten(aux, children)
        assert isinstance(restored._buffer_registry, set)
        restored.register_buffer('dynamic', 99)
        assert 'dynamic' in restored._buffer_registry

    def test_repeated_unflatten_is_safe(self):
        """tree_unflatten must not mutate shared aux_data (JAX may call it multiple times)."""
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        obj.set_buffer('cached_sum', jnp.array(42.0))

        children, aux = obj.tree_flatten()
        r1 = _SimpleBuffered.tree_unflatten(aux, children)
        r2 = _SimpleBuffered.tree_unflatten(aux, children)
        assert r1._buffer_registry == {'cached_sum', 'label'}
        assert r2._buffer_registry == {'cached_sum', 'label'}
        assert float(r2.cached_sum) == 42.0


# ===========================================================================
# 6. CSR buffer integration
# ===========================================================================

class TestCSRBuffers:
    @pytest.fixture
    def csr_mat(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0])
        indices = jnp.array([0, 1, 0, 2])
        indptr = jnp.array([0, 2, 4])
        return brainevent.CSR((data, indices, indptr), shape=(2, 3))

    def test_csr_no_buffers_initially(self, csr_mat):
        assert csr_mat.buffers == {}
        assert not hasattr(csr_mat, 'diag_positions')

    def test_csr_register_buffer_then_access(self, csr_mat):
        csr_mat.register_buffer('diag_positions', jnp.array([0, 3]))
        assert 'diag_positions' in csr_mat._buffer_registry
        np.testing.assert_array_equal(csr_mat.diag_positions, jnp.array([0, 3]))

    def test_csr_pytree_roundtrip(self, csr_mat):
        csr_mat.register_buffer('diag_positions', jnp.array([0, 3]))
        children, aux = csr_mat.tree_flatten()
        restored = brainevent.CSR.tree_unflatten(aux, children)
        np.testing.assert_array_equal(restored.diag_positions, jnp.array([0, 3]))
        assert 'diag_positions' in restored._buffer_registry

    def test_csr_jit_preserves_registered_buffer(self, csr_mat):
        csr_mat.register_buffer('diag_positions', jnp.array([0, 3]))

        @jax.jit
        def identity(m):
            return m

        result = identity(csr_mat)
        np.testing.assert_array_equal(result.diag_positions, jnp.array([0, 3]))

    def test_csr_with_data_preserves_buffers(self, csr_mat):
        csr_mat.register_buffer('diag_positions', jnp.array([0, 3]))
        new_data = jnp.array([10.0, 20.0, 30.0, 40.0])
        new_csr = csr_mat.with_data(new_data)
        np.testing.assert_array_equal(new_csr.diag_positions, jnp.array([0, 3]))
        np.testing.assert_array_equal(new_csr.data, new_data)

    def test_csr_apply_preserves_buffers(self, csr_mat):
        csr_mat.register_buffer('diag_positions', jnp.array([0, 3]))
        squared = csr_mat.apply(lambda x: x ** 2)
        np.testing.assert_array_equal(squared.diag_positions, jnp.array([0, 3]))
        np.testing.assert_array_equal(squared.data, jnp.array([1.0, 4.0, 9.0, 16.0]))

    def test_csr_transpose_preserves_buffers(self, csr_mat):
        csr_mat.register_buffer('diag_positions', jnp.array([0, 3]))
        csc = csr_mat.transpose()
        assert 'diag_positions' in csc._buffer_registry
        np.testing.assert_array_equal(csc.diag_positions, jnp.array([0, 3]))

    def test_csr_arithmetic_preserves_buffers(self, csr_mat):
        csr_mat.register_buffer('diag_positions', jnp.array([0, 3]))
        scaled = csr_mat * 2.0
        assert 'diag_positions' in scaled._buffer_registry
        np.testing.assert_array_equal(scaled.diag_positions, jnp.array([0, 3]))

    def test_csr_reflected_arithmetic_preserves_buffers(self, csr_mat):
        csr_mat.register_buffer('diag_positions', jnp.array([0, 3]))
        scaled = 2.0 * csr_mat
        assert 'diag_positions' in scaled._buffer_registry
        np.testing.assert_array_equal(scaled.diag_positions, jnp.array([0, 3]))

    def test_csr_buffers_kwarg_in_constructor(self):
        data = jnp.array([1.0, 2.0])
        indices = jnp.array([0, 1])
        indptr = jnp.array([0, 1, 2])
        bufs = {'diag_positions': jnp.array([0, 1])}
        csr = brainevent.CSR((data, indices, indptr), shape=(2, 2), buffers=bufs)
        np.testing.assert_array_equal(csr.diag_positions, jnp.array([0, 1]))
        assert 'diag_positions' in csr._buffer_registry

    def test_csr_no_buffers_pytree_roundtrip(self, csr_mat):
        children, aux = csr_mat.tree_flatten()
        restored = brainevent.CSR.tree_unflatten(aux, children)
        np.testing.assert_array_equal(restored.data, csr_mat.data)
        np.testing.assert_array_equal(restored.indices, csr_mat.indices)
        np.testing.assert_array_equal(restored.indptr, csr_mat.indptr)
        assert restored.shape == csr_mat.shape


# ===========================================================================
# 7. CSC buffer integration
# ===========================================================================

class TestCSCBuffers:
    @pytest.fixture
    def csc_mat(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0])
        indices = jnp.array([0, 1, 0, 1])
        indptr = jnp.array([0, 2, 3, 4])
        return brainevent.CSC((data, indices, indptr), shape=(2, 3))

    def test_csc_no_buffers_initially(self, csc_mat):
        assert csc_mat.buffers == {}
        assert not hasattr(csc_mat, 'diag_positions')

    def test_csc_pytree_roundtrip(self, csc_mat):
        csc_mat.register_buffer('diag_positions', jnp.array([0, 2]))
        children, aux = csc_mat.tree_flatten()
        restored = brainevent.CSC.tree_unflatten(aux, children)
        np.testing.assert_array_equal(restored.diag_positions, jnp.array([0, 2]))

    def test_csc_with_data_preserves_buffers(self, csc_mat):
        csc_mat.register_buffer('diag_positions', jnp.array([0, 2]))
        new_csc = csc_mat.with_data(jnp.array([10.0, 20.0, 30.0, 40.0]))
        np.testing.assert_array_equal(new_csc.diag_positions, jnp.array([0, 2]))

    def test_csc_apply_preserves_buffers(self, csc_mat):
        csc_mat.register_buffer('diag_positions', jnp.array([0, 2]))
        result = csc_mat.apply(lambda x: x * 3)
        np.testing.assert_array_equal(result.diag_positions, jnp.array([0, 2]))

    def test_csc_transpose_preserves_buffers(self, csc_mat):
        csc_mat.register_buffer('diag_positions', jnp.array([0, 2]))
        csr = csc_mat.transpose()
        assert 'diag_positions' in csr._buffer_registry
        np.testing.assert_array_equal(csr.diag_positions, jnp.array([0, 2]))


# ===========================================================================
# 8. diag_add integration (buffers carry through real operations)
# ===========================================================================

class TestDiagAddBufferIntegration:
    def test_diag_add_lazily_registers_diag_positions(self):
        n = 4
        data = jnp.ones(n * 2, dtype=jnp.float32)
        indices = jnp.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4, 6, 8], dtype=jnp.int32)
        csr = brainevent.CSR((data, indices, indptr), shape=(n, n))

        assert not hasattr(csr, 'diag_positions')
        result = csr.diag_add(jnp.ones(n))
        assert hasattr(csr, 'diag_positions')
        assert 'diag_positions' in csr._buffer_registry
        assert 'diag_positions' in result._buffer_registry
        assert result.diag_positions is not None

    def test_diag_add_reuses_cached_positions(self):
        n = 3
        # A dense matrix already has its full diagonal, so diag_add does not
        # change the structure -- the cached structural plan is reused as-is.
        col_indices = [j for i in range(n) for j in range(n)]
        data = jnp.ones(n * n, dtype=jnp.float32)
        indices = jnp.array(col_indices, dtype=jnp.int32)
        indptr = jnp.array([i * n for i in range(n + 1)], dtype=jnp.int32)
        csr = brainevent.CSR((data, indices, indptr), shape=(n, n))

        assert not hasattr(csr, 'diag_positions')
        result1 = csr.diag_add(jnp.ones(n))
        cached = csr.diag_positions
        result2 = csr.diag_add(jnp.ones(n) * 2)
        # The second call reuses the cached plan rather than recomputing it.
        assert csr.diag_positions is cached
        # Both results share the (unchanged) sparsity structure.
        np.testing.assert_array_equal(np.asarray(result1.indices), np.asarray(result2.indices))
        np.testing.assert_array_equal(np.asarray(result1.indptr), np.asarray(result2.indptr))

    def test_diag_add_positions_survive_jit(self):
        n = 3
        col_indices = [j for i in range(n) for j in range(n)]
        data = jnp.ones(n * n, dtype=jnp.float32)
        indices = jnp.array(col_indices, dtype=jnp.int32)
        indptr = jnp.array([i * n for i in range(n + 1)], dtype=jnp.int32)
        csr = brainevent.CSR((data, indices, indptr), shape=(n, n))

        result = csr.diag_add(jnp.ones(n))

        @jax.jit
        def identity(m):
            return m

        jitted = identity(result)
        assert 'diag_positions' in jitted._buffer_registry
        # The structural plan is a tuple of arrays; compare component by component.
        assert len(jitted.diag_positions) == len(result.diag_positions)
        for got, expected in zip(jitted.diag_positions, result.diag_positions):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


# ===========================================================================
# 10. Multiple buffers on the same instance
# ===========================================================================

class TestMultipleBuffers:
    def test_three_buffers_roundtrip(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        obj.register_buffer('extra', jnp.array(99.0))
        obj.set_buffer('cached_sum', jnp.array(3.14))
        obj.set_buffer('label', 'hello')

        children, aux = obj.tree_flatten()
        restored = _SimpleBuffered.tree_unflatten(aux, children)

        assert float(restored.cached_sum) == pytest.approx(3.14)
        assert restored.label == 'hello'
        assert float(restored.extra) == 99.0
        assert restored._buffer_registry == {'cached_sum', 'label', 'extra'}

    def test_independent_instances_have_separate_registries(self):
        a = _SimpleBuffered(1.0, shape=(2, 2))
        b = _SimpleBuffered(2.0, shape=(3, 3))
        a.register_buffer('only_on_a', 10)
        assert 'only_on_a' in a._buffer_registry
        assert 'only_on_a' not in b._buffer_registry

    def test_csr_multiple_custom_buffers(self):
        data = jnp.array([1.0, 2.0])
        indices = jnp.array([0, 1])
        indptr = jnp.array([0, 1, 2])
        csr = brainevent.CSR((data, indices, indptr), shape=(2, 2))
        csr.register_buffer('my_mask', jnp.array([True, False]))
        csr.register_buffer('scale_factor', jnp.array(0.5))

        children, aux = csr.tree_flatten()
        restored = brainevent.CSR.tree_unflatten(aux, children)
        np.testing.assert_array_equal(restored.my_mask, jnp.array([True, False]))
        assert float(restored.scale_factor) == 0.5
        assert 'my_mask' in restored._buffer_registry
        assert 'scale_factor' in restored._buffer_registry


# ===========================================================================
# 11. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_buffer_value_array_types(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        for val in [jnp.array([1, 2, 3]), np.array([4.0]), 42, 'text', None]:
            obj.set_buffer('cached_sum', val)
            assert obj.buffers['cached_sum'] is val or np.array_equal(obj.buffers['cached_sum'], val)

    def test_buffer_with_jax_array_preserves_shape_dtype(self):
        arr = jnp.zeros((4, 5), dtype=jnp.float16)
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        obj.set_buffer('cached_sum', arr)
        children, aux = obj.tree_flatten()
        restored = _SimpleBuffered.tree_unflatten(aux, children)
        assert restored.cached_sum.shape == (4, 5)
        assert restored.cached_sum.dtype == jnp.float16

    def test_register_same_name_twice_is_idempotent_in_registry(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        obj.register_buffer('cached_sum', 1)
        obj.register_buffer('cached_sum', 2)
        assert obj.cached_sum == 2
        assert sum(1 for n in obj._buffer_registry if n == 'cached_sum') == 1

    def test_buffers_kwarg_with_unknown_names_registers_them(self):
        bufs = {'custom_a': 1, 'custom_b': jnp.array([2.0])}
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=bufs)
        assert obj.custom_a == 1
        np.testing.assert_array_equal(obj.custom_b, jnp.array([2.0]))
        assert 'custom_a' in obj._buffer_registry
        assert 'custom_b' in obj._buffer_registry


# ---------------------------------------------------------------------------
# Common-API contract on :class:`brainevent.DataRepresentation`.
#
# Covers (1) that every concrete data representation overrides or deliberately
# refuses each contract method (no silent inheritance of a bare base stub),
# (2) the deliberate JIT-connectivity refusals, and (3) the conversion
# round-trips (``tocsr`` / ``tocsc`` / ``tocoo`` / ``fromdense``) across the
# compressed-sparse, fixed-num-connection, and JIT-connectivity families.
# ---------------------------------------------------------------------------



SPARSE_BASE = u.sparse.SparseMatrix

# Concrete subclasses of DataRepresentation (the in-scope families).
CONCRETE_CLASSES = [
    be.Dense,
    be.CSR, be.CSC,
    be.FixedNumPerPre, be.FixedNumPerPost,
    be.JITCScalarR, be.JITCScalarC,
    be.JITCNormalR, be.JITCNormalC,
    be.JITCUniformR, be.JITCUniformC,
]

# The common-API contract surface. ``with_data`` / ``transpose`` / ``todense``
# are declared by the saiunit base; the rest by DataRepresentation.
CONTRACT_METHODS = [
    'todense', 'fromdense', 'tocoo', 'tocsr', 'tocsc',
    'dt2t', 'dt2t_transposed',
    'update_on_pre', 'update_on_post',
    'with_data', 'transpose',
]

# (class, constructor-data) for each concrete JIT-connectivity family.
JITC_INSTANCES = [
    (be.JITCScalarR, (1.5, 0.2, 42)),
    (be.JITCScalarC, (1.5, 0.2, 42)),
    (be.JITCNormalR, (0.0, 1.0, 0.2, 42)),
    (be.JITCNormalC, (0.0, 1.0, 0.2, 42)),
    (be.JITCUniformR, (0.0, 1.0, 0.2, 42)),
    (be.JITCUniformC, (0.0, 1.0, 0.2, 42)),
]
JITC_IDS = [c.__name__ for c, _ in JITC_INSTANCES]

# Every JITC family materializes mode-dependently: the mv (32-lane) and mm
# (4-thread AW-T4) light kernels draw different matrices, so bare
# ``todense()/tocsr()/tocsc()/tocoo()`` raise and callers must go through the
# ``mat.mv`` / ``mat.mm`` views.  Their CSR conversion is CUDA-only (the
# per-family ``*_csr_count`` primitives register only a CUDA backend).
def _csr_count_backends(cls):
    """CSR-materialization backends available for ``cls`` on the current platform.

    Returns an empty tuple when the family's ``csr_count`` primitive has no
    backend for the active platform (e.g. CPU), so callers can skip the
    CUDA-only conversion assertions.
    """
    try:
        if cls in (be.JITCScalarR, be.JITCScalarC):
            from brainevent._jit_scalar.csr import jits_csr_count_p as count_p
        elif cls in (be.JITCNormalR, be.JITCNormalC):
            from brainevent._jit_normal.csr import jitn_csr_count_p as count_p
        elif cls in (be.JITCUniformR, be.JITCUniformC):
            from brainevent._jit_uniform.csr import jitu_csr_count_p as count_p
        else:  # pragma: no cover - defensive
            return ()
        return tuple(count_p.available_backends(jax.default_backend()))
    except Exception:  # pragma: no cover - defensive: import/registration failure
        return ()

_DENSE = jnp.array([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]])


def _defining_class(cls, method):
    """Return the first class in ``cls.__mro__`` that defines ``method``."""
    for klass in cls.__mro__:
        if method in vars(klass):
            return klass
    return None


# --------------------------------------------------------------------------- #
# Contract coverage
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('method', CONTRACT_METHODS)
@pytest.mark.parametrize('cls', CONCRETE_CLASSES, ids=[c.__name__ for c in CONCRETE_CLASSES])
def test_contract_method_is_overridden_or_refused(cls, method):
    # Every contract method must resolve to a definition inside brainevent's
    # hierarchy -- never the bare saiunit base stub nor the bare
    # DataRepresentation stub. A deliberate refusal counts as an override.
    defn = _defining_class(cls, method)
    assert defn is not None, f'{cls.__name__} is missing contract method {method!r}'
    assert defn is not SPARSE_BASE, (
        f'{cls.__name__}.{method} silently inherits the saiunit SparseMatrix stub'
    )
    assert defn is not DataRepresentation, (
        f'{cls.__name__}.{method} silently inherits the DataRepresentation stub'
    )


def test_data_representation_declares_dt2t_contract():
    assert 'dt2t' in vars(DataRepresentation)


def test_data_representation_declares_yw_to_w_deprecated_aliases():
    assert 'yw_to_w' in vars(DataRepresentation)
    assert 'yw_to_w_transposed' in vars(DataRepresentation)


@pytest.mark.parametrize('cls', CONCRETE_CLASSES, ids=[c.__name__ for c in CONCRETE_CLASSES])
def test_yw_to_w_signatures_align_dt2t_contract(cls):
    for alias, canonical in (('yw_to_w', 'dt2t'), ('yw_to_w_transposed', 'dt2t_transposed')):
        sig = inspect.signature(getattr(cls, alias))
        canonical_sig = inspect.signature(getattr(cls, canonical))
        assert list(sig.parameters) == list(canonical_sig.parameters)


def test_yw_to_w_warns_and_delegates_to_dt2t():
    csr = be.CSR.fromdense(_DENSE)
    y = jnp.ones(csr.shape[0])
    w = csr.data
    with pytest.warns(DeprecationWarning, match='yw_to_w is deprecated'):
        out = csr.yw_to_w(y, w)
    assert jnp.allclose(out, csr.dt2t(y, w))


def test_yw_to_w_transposed_warns_and_delegates_to_dt2t_transposed():
    csr = be.CSR.fromdense(_DENSE)
    y = jnp.ones(csr.shape[1])
    w = csr.data
    with pytest.warns(DeprecationWarning, match='yw_to_w_transposed is deprecated'):
        out = csr.yw_to_w_transposed(y, w)
    assert jnp.allclose(out, csr.dt2t_transposed(y, w))


@pytest.mark.parametrize('method', ['dt2t', 'dt2t_transposed'])
@pytest.mark.parametrize('cls', CONCRETE_CLASSES, ids=[c.__name__ for c in CONCRETE_CLASSES])
def test_dt2t_signatures_align_data_representation_contract(cls, method):
    sig = inspect.signature(getattr(cls, method))
    base_sig = inspect.signature(getattr(DataRepresentation, method))
    assert list(sig.parameters) == list(base_sig.parameters)
    assert sig.parameters['y_dim_arr'].annotation == base_sig.parameters['y_dim_arr'].annotation
    assert sig.parameters['w_dim_arr'].annotation == base_sig.parameters['w_dim_arr'].annotation
    assert sig.parameters['w_dim_arr'].default is inspect._empty
    assert sig.return_annotation == base_sig.return_annotation


def test_unsupported_operation_error_is_brainevent_error():
    assert issubclass(UnsupportedOperationError, BrainEventError)


# --------------------------------------------------------------------------- #
# JIT-connectivity deliberate refusals
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('cls,data', JITC_INSTANCES, ids=JITC_IDS)
def test_jitc_refuses_plastic_update_protocols(cls, data):
    m = cls(data, shape=(16, 16))
    y = jnp.ones(16)
    with pytest.raises(UnsupportedOperationError):
        m.update_on_pre(y, y)
    with pytest.raises(UnsupportedOperationError):
        m.update_on_post(y, y)


@pytest.mark.parametrize('cls,data', JITC_INSTANCES, ids=JITC_IDS)
def test_jitc_refuses_fromdense(cls, data):
    with pytest.raises(UnsupportedOperationError):
        cls.fromdense(jnp.ones((16, 16)))


# --------------------------------------------------------------------------- #
# JIT-connectivity conversions (delegate through tocsr)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('cls,data', JITC_INSTANCES, ids=JITC_IDS)
def test_jitc_conversions_agree_with_todense(cls, data):
    m = cls(data, shape=(16, 16))
    # Bare materialization is ambiguous for every JITC family and must raise; the
    # mv/mm views resolve the mode. tocsr/tocsc/tocoo are CUDA-only.
    with pytest.raises(NotImplementedError):
        m.todense()
    with pytest.raises(NotImplementedError):
        m.tocsr()
    dense = m.mv.todense()
    assert dense.shape == m.shape
    if not _csr_count_backends(cls):
        pytest.skip('JITC CSR conversion is CUDA-only')
    assert jnp.allclose(m.mv.tocsr().todense(), dense)
    assert jnp.allclose(m.mv.tocsc().todense(), dense)
    assert jnp.allclose(m.mv.tocoo().todense(), dense)
    assert m.mv.tocsc().shape == m.shape
    assert m.mv.tocoo().shape == m.shape


# --------------------------------------------------------------------------- #
# Compressed-sparse conversions
# --------------------------------------------------------------------------- #

def test_csr_conversions_roundtrip():
    csr = be.CSR.fromdense(_DENSE)
    assert csr.tocsr() is csr
    assert jnp.allclose(csr.tocsc().todense(), _DENSE)
    assert jnp.allclose(csr.tocoo().todense(), _DENSE)


def test_csc_conversions_roundtrip():
    csc = be.CSC.fromdense(_DENSE)
    assert csc.tocsc() is csc
    assert jnp.allclose(csc.tocsr().todense(), _DENSE)
    assert jnp.allclose(csc.tocoo().todense(), _DENSE)


def test_tocsc_preserves_shape_unlike_transpose():
    # tocsc re-encodes the *same* logical matrix (shape unchanged); transpose
    # swaps the shape. A non-square matrix makes the distinction unambiguous.
    dense = jnp.array([[1., 2., 0., 0.], [0., 0., 3., 0.], [0., 4., 0., 5.]])  # (3, 4)
    csr = be.CSR.fromdense(dense)
    assert csr.tocsc().shape == (3, 4)
    assert csr.transpose().shape == (4, 3)
    assert jnp.allclose(csr.tocsc().todense(), dense)


def test_csr_tocoo_homogeneous_value_broadcast():
    # A size-1 (shared) value must broadcast to one entry per stored element.
    csr = be.CSR((jnp.array([2.0]), jnp.array([0, 2, 1]), jnp.array([0, 2, 3])), shape=(2, 3))
    coo = csr.tocoo()
    assert coo.row.size == 3
    assert jnp.allclose(coo.todense(), csr.todense())


# --------------------------------------------------------------------------- #
# Fixed-num-connection conversions and fromdense
# --------------------------------------------------------------------------- #

def test_fcn_pre_fromdense_uniform_roundtrip():
    dense = jnp.array([[1., 2., 0.], [0., 3., 4.], [5., 0., 6.]])  # 2 conns / pre
    pre = be.FixedNumPerPre.fromdense(dense)
    assert pre.num_conn == 2
    assert jnp.allclose(pre.todense(), dense)
    assert jnp.allclose(pre.tocsr().todense(), dense)
    assert jnp.allclose(pre.tocsc().todense(), dense)
    assert jnp.allclose(pre.tocoo().todense(), dense)


def test_fcn_post_fromdense_uniform_roundtrip():
    dense = jnp.array([[1., 0., 5.], [2., 3., 0.], [0., 4., 6.]])  # 2 conns / post
    post = be.FixedNumPerPost.fromdense(dense)
    assert post.num_conn == 2
    assert jnp.allclose(post.todense(), dense)
    assert jnp.allclose(post.tocsr().todense(), dense)
    assert jnp.allclose(post.tocoo().todense(), dense)


def test_fcn_fromdense_irregular_requires_num_conn():
    irr = jnp.array([[1., 2., 3.], [0., 4., 0.], [5., 0., 6.]])  # 3, 1, 2 nnz
    with pytest.raises(ValueError):
        be.FixedNumPerPre.fromdense(irr)


def test_fcn_fromdense_padding_roundtrip():
    irr = jnp.array([[1., 2., 3.], [0., 4., 0.], [5., 0., 6.]])
    pre = be.FixedNumPerPre.fromdense(irr, num_conn=3)
    assert pre.num_conn == 3
    # zero-weight sentinel padding does not change the dense matrix.
    assert jnp.allclose(pre.todense(), irr)


def test_fcn_fromdense_overflow_raises():
    irr = jnp.array([[1., 2., 3.], [0., 4., 0.], [5., 0., 6.]])
    with pytest.raises(ValueError):
        be.FixedNumPerPre.fromdense(irr, num_conn=2)


def test_fcn_fromdense_preserves_units():
    dense = jnp.array([[1., 2., 0.], [0., 3., 4.], [5., 0., 6.]]) * u.mV
    pre = be.FixedNumPerPre.fromdense(dense)
    assert u.get_unit(pre.todense()) == u.get_unit(dense)
    assert u.math.allclose(pre.todense(), dense)


def test_fcn_fromdense_rejects_non_2d():
    with pytest.raises(ValueError):
        be.FixedNumPerPre.fromdense(jnp.ones((3,)))


@pytest.mark.parametrize('name', ['to_csr', 'to_csc', 'to_dense'])
def test_fcn_deprecated_aliases_removed(name):
    pre = be.FixedNumPerPre.fromdense(_DENSE, num_conn=2)
    assert not hasattr(pre, name)
