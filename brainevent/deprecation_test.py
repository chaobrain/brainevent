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

# -*- coding: utf-8 -*-

"""Tests for the module-level ``__getattr__`` deprecation shim in
:mod:`brainevent.__init__`.

Each retired public name must either resolve to its replacement (with a
``DeprecationWarning``-style message) or, for fully removed names, raise an
informative :class:`AttributeError`. These are the only dynamic-attribute
paths on the package, so covering them pins the back-compat contract.
"""

import warnings

import pytest

import brainevent


# (deprecated name, replacement object) pairs that must warn-and-forward.
_ALIASES = [
    ("EventArray", brainevent.BinaryArray),
    ("csr_on_pre", brainevent.update_csr_on_binary_pre),
    ("csr2csc_on_post", brainevent.update_csr_on_binary_post),
    ("dense_on_pre", brainevent.update_dense_on_binary_pre),
    ("dense_on_post", brainevent.update_dense_on_binary_post),
    ("JITCHomoC", brainevent.JITCScalarC),
    ("FixedPostNumConn", brainevent.FixedNumPerPre),
    ("FixedPreNumConn", brainevent.FixedNumPerPost),
]


@pytest.mark.parametrize("name, target", _ALIASES, ids=[n for n, _ in _ALIASES])
def test_deprecated_alias_warns_and_forwards(name, target):
    with pytest.warns(Warning, match="deprecated"):
        obj = getattr(brainevent, name)
    assert obj is target


@pytest.mark.parametrize("name", ["EllLayout", "CscLayout"])
def test_removed_layout_names_raise_informative_attributeerror(name):
    # These were not merely renamed -- the abstraction was removed -- so the
    # shim must raise rather than silently forward.
    with pytest.raises(AttributeError) as exc_info:
        getattr(brainevent, name)
    message = str(exc_info.value)
    assert name in message
    assert "removed" in message
    assert "FixedNumPerPost" in message or "FixedNumPerPre" in message


def test_unknown_attribute_raises_plain_attributeerror():
    with pytest.raises(AttributeError):
        _ = brainevent.this_symbol_does_not_exist


def test_unknown_attribute_does_not_warn():
    # A genuine miss should not emit a spurious deprecation warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError):
            _ = brainevent.totally_made_up_name


def test_dir_lists_public_exports():
    # The curated __all__ should be a subset of the importable names, so a
    # plain ``import brainevent; brainevent.<name>`` works for every export.
    for name in brainevent.__all__:
        assert hasattr(brainevent, name), name
