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

"""Comprehensive tests for the buffer registration mechanism on DataRepresentation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._data import DataRepresentation
import brainevent


# ---------------------------------------------------------------------------
# Helpers: minimal concrete subclass for testing the base mechanism
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class _SimpleBuffered(DataRepresentation):
    """Minimal concrete DataRepresentation for testing the buffer API."""

    def __init__(self, value, *, shape, buffers=None):
        self.value = jnp.asarray(value)
        self.register_buffer('cached_sum', None)
        self.register_buffer('label', None)
        super().__init__((value,), shape=shape, buffers=buffers)

    def transpose(self, axes=None):
        return self

    def tree_flatten(self):
        aux = {'shape': self.shape, 'value': self.value}
        aux.update(self._flatten_buffers())
        return (), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        registry = aux_data.pop('_buffer_registry', frozenset())
        obj._buffer_registry = set(registry)
        for k, v in aux_data.items():
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
        # Still only one entry in the registry
        assert list(obj._buffer_registry).count('cached_sum') == 1

    def test_register_buffer_creates_registry_if_missing(self):
        """Calling register_buffer on a bare object should create _buffer_registry."""
        obj = object.__new__(_SimpleBuffered)
        # No _buffer_registry yet
        assert not hasattr(obj, '_buffer_registry')
        obj.register_buffer('foo', 'bar')
        assert hasattr(obj, '_buffer_registry')
        assert 'foo' in obj._buffer_registry
        assert obj.foo == 'bar'


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

    def test_set_buffer_without_registry_raises(self):
        obj = object.__new__(_SimpleBuffered)
        with pytest.raises(ValueError, match="not registered"):
            obj.set_buffer('whatever', 5)


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

    def test_empty_registry(self):
        """An object with no registered buffers should return an empty dict."""
        obj = object.__new__(_SimpleBuffered)
        # No _buffer_registry attribute
        assert obj.buffers == {}

    def test_buffers_returns_new_dict_each_time(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        d1 = obj.buffers
        d2 = obj.buffers
        assert d1 == d2
        assert d1 is not d2


# ===========================================================================
# 4. _apply_buffers
# ===========================================================================

class TestApplyBuffers:
    def test_apply_overrides_defaults(self):
        bufs = {'cached_sum': jnp.array(7.0), 'label': 'hello'}
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=bufs)
        assert float(obj.cached_sum) == 7.0
        assert obj.label == 'hello'

    def test_apply_partial_override(self):
        bufs = {'cached_sum': jnp.array(5.0)}
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=bufs)
        assert float(obj.cached_sum) == 5.0
        assert obj.label is None  # not overridden

    def test_apply_none_is_noop(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=None)
        assert obj.cached_sum is None
        assert obj.label is None

    def test_apply_empty_dict_is_noop(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers={})
        assert obj.cached_sum is None
        assert obj.label is None

    def test_apply_registers_new_buffers(self):
        """_apply_buffers calls register_buffer, so new names are added to the registry."""
        bufs = {'cached_sum': 1, 'label': 2, 'brand_new': 42}
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=bufs)
        assert obj.brand_new == 42
        assert 'brand_new' in obj._buffer_registry


# ===========================================================================
# 5. _flatten_buffers
# ===========================================================================

class TestFlattenBuffers:
    def test_contains_registry_as_frozenset(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        flat = obj._flatten_buffers()
        assert '_buffer_registry' in flat
        assert isinstance(flat['_buffer_registry'], frozenset)
        assert flat['_buffer_registry'] == frozenset({'cached_sum', 'label'})

    def test_contains_buffer_values(self):
        obj = _SimpleBuffered(1.0, shape=(2, 2))
        obj.cached_sum = jnp.array(42.0)
        flat = obj._flatten_buffers()
        assert float(flat['cached_sum']) == 42.0
        assert flat['label'] is None

    def test_empty_registry(self):
        obj = object.__new__(_SimpleBuffered)
        flat = obj._flatten_buffers()
        assert flat == {'_buffer_registry': frozenset()}


# ===========================================================================
# 6. JAX pytree round-trip
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
        # All buffers are None by default
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
        # Should be mutable — can add new buffers
        restored.register_buffer('dynamic', 99)
        assert 'dynamic' in restored._buffer_registry


# ===========================================================================
# 7. CSR buffer integration
# ===========================================================================

class TestCSRBuffers:
    @pytest.fixture
    def csr_mat(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0])
        indices = jnp.array([0, 1, 0, 2])
        indptr = jnp.array([0, 2, 4])
        return brainevent.CSR((data, indices, indptr), shape=(2, 3))

    def test_csr_no_buffers_initially(self, csr_mat):
        """CSR starts with no registered buffers (diag_positions is lazy)."""
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
        """CSR with no registered buffers should roundtrip cleanly."""
        children, aux = csr_mat.tree_flatten()
        restored = brainevent.CSR.tree_unflatten(aux, children)
        np.testing.assert_array_equal(restored.data, csr_mat.data)
        np.testing.assert_array_equal(restored.indices, csr_mat.indices)
        np.testing.assert_array_equal(restored.indptr, csr_mat.indptr)
        assert restored.shape == csr_mat.shape


# ===========================================================================
# 8. CSC buffer integration
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
# 9. COO buffer integration
# ===========================================================================

class TestCOOBuffers:
    @pytest.fixture
    def coo_mat(self):
        data = jnp.array([1.0, 2.0, 3.0])
        row = jnp.array([0, 1, 2])
        col = jnp.array([1, 0, 2])
        return brainevent.COO((data, row, col), shape=(3, 3))

    def test_coo_empty_buffers_by_default(self, coo_mat):
        assert coo_mat.buffers == {}

    def test_coo_register_and_roundtrip(self, coo_mat):
        coo_mat.register_buffer('my_cache', jnp.array([10, 20, 30]))
        children, aux = coo_mat.tree_flatten()
        restored = brainevent.COO.tree_unflatten(aux, children)
        assert 'my_cache' in restored._buffer_registry
        np.testing.assert_array_equal(restored.my_cache, jnp.array([10, 20, 30]))

    def test_coo_jit_preserves_buffers(self, coo_mat):
        coo_mat.register_buffer('tag', jnp.array(42))

        @jax.jit
        def f(m):
            return m

        result = f(coo_mat)
        assert int(result.tag) == 42


# ===========================================================================
# 10. diag_add integration (buffers carry through real operations)
# ===========================================================================

class TestDiagAddBufferIntegration:
    def test_diag_add_lazily_registers_diag_positions(self):
        """diag_add should lazily register and compute diag_positions as a buffer."""
        n = 4
        data = jnp.ones(n * 2, dtype=jnp.float32)
        indices = jnp.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4, 6, 8], dtype=jnp.int32)
        csr = brainevent.CSR((data, indices, indptr), shape=(n, n))

        # Before diag_add, no diag_positions attribute
        assert not hasattr(csr, 'diag_positions')
        result = csr.diag_add(jnp.ones(n))
        # After diag_add, diag_positions should be registered and cached
        assert hasattr(csr, 'diag_positions')
        assert 'diag_positions' in csr._buffer_registry
        # The returned result carries buffers via with_data
        assert 'diag_positions' in result._buffer_registry
        assert result.diag_positions is not None

    def test_diag_add_reuses_cached_positions(self):
        """Second call to diag_add should reuse the cached diag_positions."""
        n = 3
        col_indices = []
        for i in range(n):
            for j in range(n):
                col_indices.append(j)
        nnz = n * n
        data = jnp.ones(nnz, dtype=jnp.float32)
        indices = jnp.array(col_indices, dtype=jnp.int32)
        indptr = jnp.array([i * n for i in range(n + 1)], dtype=jnp.int32)
        csr = brainevent.CSR((data, indices, indptr), shape=(n, n))

        result1 = csr.diag_add(jnp.ones(n))
        diag_pos1 = result1.diag_positions
        result2 = result1.diag_add(jnp.ones(n) * 2)
        diag_pos2 = result2.diag_positions
        np.testing.assert_array_equal(diag_pos1, diag_pos2)

    def test_diag_add_positions_survive_jit(self):
        """diag_positions cached by diag_add survive a jax.jit roundtrip."""
        n = 3
        col_indices = []
        for i in range(n):
            for j in range(n):
                col_indices.append(j)
        nnz = n * n
        data = jnp.ones(nnz, dtype=jnp.float32)
        indices = jnp.array(col_indices, dtype=jnp.int32)
        indptr = jnp.array([i * n for i in range(n + 1)], dtype=jnp.int32)
        csr = brainevent.CSR((data, indices, indptr), shape=(n, n))

        result = csr.diag_add(jnp.ones(n))

        @jax.jit
        def identity(m):
            return m

        jitted = identity(result)
        assert 'diag_positions' in jitted._buffer_registry
        np.testing.assert_array_equal(jitted.diag_positions, result.diag_positions)


# ===========================================================================
# 11. Multiple buffers on the same instance
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
        """CSR can hold multiple user-registered buffers alongside diag_positions."""
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
# 12. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_buffer_value_array_types(self):
        """Buffers can hold JAX arrays, numpy arrays, Python scalars, strings, None."""
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
        # Registry should contain 'cached_sum' exactly once
        assert sum(1 for n in obj._buffer_registry if n == 'cached_sum') == 1

    def test_buffers_kwarg_with_unknown_names_registers_them(self):
        """Passing unknown buffer names via buffers= kwarg should register them."""
        bufs = {'custom_a': 1, 'custom_b': jnp.array([2.0])}
        obj = _SimpleBuffered(1.0, shape=(2, 2), buffers=bufs)
        assert obj.custom_a == 1
        np.testing.assert_array_equal(obj.custom_b, jnp.array([2.0]))
        assert 'custom_a' in obj._buffer_registry
        assert 'custom_b' in obj._buffer_registry
