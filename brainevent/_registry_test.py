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

from brainevent._registry import (
    _PRIMITIVE_REGISTRY,
    register_primitive,
    get_registry,
    get_primitives_by_tags,
    get_all_primitive_names,
)


class TestRegistry:
    def test_registry_populated_on_import(self):
        """Importing brainevent should auto-register all primitives."""
        registry = get_registry()
        assert len(registry) > 0, "Registry should not be empty after import"

    def test_known_primitives_exist(self):
        """Known primitives should be in the registry."""
        names = get_all_primitive_names()
        # CSR binary primitives
        assert 'binary_csrmv' in names
        assert 'binary_csrmm' in names
        # CSR float primitives
        assert 'csrmv' in names
        assert 'csrmm' in names

    def test_get_registry_returns_copy(self):
        """get_registry should return a copy, not the internal dict."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is not r2
        assert r1 == r2

    def test_get_all_primitive_names_sorted(self):
        """get_all_primitive_names should return sorted list."""
        names = get_all_primitive_names()
        assert names == sorted(names)

    def test_get_primitives_by_tags_csr(self):
        """Filtering by 'csr' tag should return CSR primitives."""
        prims = get_primitives_by_tags({'csr'})
        assert len(prims) > 0
        for name, prim in prims.items():
            assert 'csr' in prim._tags

    def test_get_primitives_by_tags_binary(self):
        """Filtering by 'binary' tag should return binary primitives."""
        prims = get_primitives_by_tags({'binary'})
        assert len(prims) > 0
        for name, prim in prims.items():
            assert 'binary' in prim._tags

    def test_get_primitives_by_tags_intersection(self):
        """Filtering by multiple tags should return intersection."""
        prims = get_primitives_by_tags({'csr', 'binary'})
        assert len(prims) > 0
        for name, prim in prims.items():
            assert 'csr' in prim._tags
            assert 'binary' in prim._tags

    def test_get_primitives_by_nonexistent_tag(self):
        """Filtering by a nonexistent tag should return empty dict."""
        prims = get_primitives_by_tags({'nonexistent_tag_xyz'})
        assert len(prims) == 0

    def test_register_primitive_overwrites(self):
        """Registering with same name should overwrite."""
        old_prim = _PRIMITIVE_REGISTRY.get('binary_csrmv')
        assert old_prim is not None
        # Re-register should not raise
        register_primitive('binary_csrmv', old_prim)
        assert _PRIMITIVE_REGISTRY['binary_csrmv'] is old_prim

    def test_primitives_have_tags(self):
        """All registered primitives should have _tags attribute."""
        registry = get_registry()
        for name, prim in registry.items():
            assert hasattr(prim, '_tags'), f"Primitive {name} missing _tags"
            assert isinstance(prim._tags, set), f"Primitive {name} _tags should be a set"

    def test_coo_primitives_tagged(self):
        """COO primitives should have 'coo' tag."""
        prims = get_primitives_by_tags({'coo'})
        assert len(prims) > 0

    def test_dense_primitives_tagged(self):
        """Dense primitives should have 'dense' tag."""
        prims = get_primitives_by_tags({'dense'})
        assert len(prims) > 0

    def test_fcn_primitives_tagged(self):
        """FCN primitives should have 'fcn' tag."""
        prims = get_primitives_by_tags({'fcn'})
        assert len(prims) > 0
