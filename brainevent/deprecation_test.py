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

"""Tests pinning the removal of the module-level back-compat shim.

The package previously exposed a ``__getattr__`` deprecation layer that
forwarded retired public names to their replacements. That layer has been
removed (breaking change), so every retired name must now raise a plain
:class:`AttributeError`, and the curated ``__all__`` must stay importable.
"""

import warnings

import pytest

import brainevent


# Names that used to warn-and-forward, plus the already-removed layout names.
# All must now be plain attribute misses -- no shim, no forwarding.
_REMOVED_NAMES = [
    "EventArray",
    "csr_on_pre",
    "csr2csc_on_post",
    "dense_on_pre",
    "dense_on_post",
    "JITCHomoC",
    "JITCHomoR",
    "FixedPostNumConn",
    "FixedPreNumConn",
    "EllLayout",
    "CscLayout",
]


@pytest.mark.parametrize("name", _REMOVED_NAMES)
def test_removed_name_raises_attributeerror(name):
    with pytest.raises(AttributeError):
        getattr(brainevent, name)


@pytest.mark.parametrize("name", _REMOVED_NAMES)
def test_removed_name_does_not_warn(name):
    # The shim is gone, so accessing a retired name must not emit a deprecation
    # warning -- it is simply absent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError):
            getattr(brainevent, name)


def test_unknown_attribute_raises_plain_attributeerror():
    with pytest.raises(AttributeError):
        _ = brainevent.this_symbol_does_not_exist


def test_dir_lists_public_exports():
    # The curated __all__ should be a subset of the importable names, so a
    # plain ``import brainevent; brainevent.<name>`` works for every export.
    for name in brainevent.__all__:
        assert hasattr(brainevent, name), name
