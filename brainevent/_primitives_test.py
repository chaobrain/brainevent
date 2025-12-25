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

"""Tests for the primitives module."""

import pytest

from brainevent._op.main import XLACustomKernel
from ._primitives import (
    ALL_PRIMITIVES,
    COO_PRIMITIVES,
    CSR_PRIMITIVES,
    DENSE_PRIMITIVES,
    FIXED_CONN_PRIMITIVES,
    JITC_HOMO_PRIMITIVES,
    JITC_NORMAL_PRIMITIVES,
    JITC_UNIFORM_PRIMITIVES,
    get_all_primitive_names,
    get_primitives_by_category,
    get_primitive_info
)


def test_all_primitives_structure():
    """Test that ALL_PRIMITIVES has the correct structure."""
    assert isinstance(ALL_PRIMITIVES, dict)
    assert len(ALL_PRIMITIVES) > 0

    # Check all values are XLACustomKernel instances
    for var_name, kernel in ALL_PRIMITIVES.items():
        assert isinstance(kernel, XLACustomKernel)
        assert hasattr(kernel, 'primitive')
        assert hasattr(kernel.primitive, 'name')
        assert var_name.endswith('_p'), f"Variable {var_name} should end with '_p'"


def test_primitive_collections_complete():
    """Test that individual collections sum up to ALL_PRIMITIVES."""
    collections = [
        COO_PRIMITIVES, CSR_PRIMITIVES, DENSE_PRIMITIVES,
        FIXED_CONN_PRIMITIVES, JITC_HOMO_PRIMITIVES,
        JITC_NORMAL_PRIMITIVES, JITC_UNIFORM_PRIMITIVES
    ]

    # All collections should be non-empty
    for collection in collections:
        assert isinstance(collection, dict)
        assert len(collection) > 0

    # Combined should match ALL_PRIMITIVES
    combined = {}
    for collection in collections:
        combined.update(collection)

    assert len(combined) == len(ALL_PRIMITIVES)
    assert set(combined.keys()) == set(ALL_PRIMITIVES.keys())


def test_primitive_names_unique():
    """Test that all primitive names are unique."""
    names = [kernel.primitive.name for kernel in ALL_PRIMITIVES.values()]
    assert len(names) == len(set(names))


def test_get_all_primitive_names():
    """Test get_all_primitive_names function."""
    names = get_all_primitive_names()

    assert isinstance(names, list)
    assert len(names) > 0
    assert names == sorted(names)  # Should be sorted
    assert len(names) == len(ALL_PRIMITIVES)

    # Should match actual primitive names
    expected = [k.primitive.name for k in ALL_PRIMITIVES.values()]
    assert set(names) == set(expected)


def test_get_primitives_by_category():
    """Test get_primitives_by_category function."""
    categories = get_primitives_by_category()

    assert isinstance(categories, dict)
    assert len(categories) > 0

    # All category names should be sorted lists
    for category, names in categories.items():
        assert isinstance(names, list)
        assert names == sorted(names)
        assert len(names) > 0

    # Total should match ALL_PRIMITIVES
    total_count = sum(len(names) for names in categories.values())
    assert total_count == len(ALL_PRIMITIVES)


def test_get_primitive_info():
    """Test get_primitive_info function."""
    # Test with a few known primitives (less brittle than hardcoding all)
    test_names = get_all_primitive_names()[:5]  # Just test first 5

    for name in test_names:
        info = get_primitive_info(name)

        assert isinstance(info, dict)
        assert info['name'] == name
        assert info['variable_name'].endswith('_p')
        assert info['category'] in ['COO', 'CSR', 'Dense', 'FixedConn',
                                    'JITC_Homo', 'JITC_Normal', 'JITC_Uniform']
        assert isinstance(info['kernel_object'], XLACustomKernel)


def test_get_primitive_info_invalid():
    """Test get_primitive_info with invalid name."""
    with pytest.raises(ValueError, match="not found"):
        get_primitive_info('non_existent_primitive')


def test_integration_with_main_module():
    """Test that functions are accessible from main brainevent module."""
    import brainevent

    # Test functions are available
    assert hasattr(brainevent, 'get_all_primitive_names')
    assert hasattr(brainevent, 'get_primitives_by_category')
    assert hasattr(brainevent, 'get_primitive_info')
    assert hasattr(brainevent, 'ALL_PRIMITIVES')

    # Test they work
    names = brainevent.get_all_primitive_names()
    assert isinstance(names, list) and len(names) > 0

    categories = brainevent.get_primitives_by_category()
    assert isinstance(categories, dict) and len(categories) > 0

    assert isinstance(brainevent.ALL_PRIMITIVES, dict)
    assert len(brainevent.ALL_PRIMITIVES) > 0


def test_core_primitives_exist():
    """Test that some core primitives exist (minimal set for basic functionality)."""
    # Only test for a few core primitives that are unlikely to be removed
    core_primitives = ['csrmv', 'coomv']  # Basic matrix-vector operations
    all_names = get_all_primitive_names()

    for core_name in core_primitives:
        assert core_name in all_names, f"Core primitive '{core_name}' should exist"


if __name__ == '__main__':
    pytest.main([__file__])
