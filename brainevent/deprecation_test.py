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

"""Tests for the v0.0.7 -> v0.1.0 backward-compatibility shim.

Public names retired between v0.0.7 and v0.1.0 must stay *resolvable*:
renamed names warn and return their replacement; names whose functionality was
removed raise an :class:`AttributeError` naming the migration path. Normal
``import brainevent`` must remain warning-free, and ``__all__`` must be unchanged.
"""

import subprocess
import sys
import warnings

import pytest

import brainevent

# old name -> attribute name of the replacement on the brainevent module.
_RENAMES = {
    "EventArray": "BinaryArray",
    "csr_on_pre": "update_csr_on_binary_pre",
    "csr2csc_on_post": "update_csr_on_binary_post",
    "dense_on_pre": "update_dense_on_binary_pre",
    "dense_on_post": "update_dense_on_binary_post",
    "JITCHomoR": "JITCScalarR",
    "JITCHomoC": "JITCScalarC",
    "FixedPostNumConn": "FixedNumPerPre",
    "FixedPreNumConn": "FixedNumPerPost",
}

# old name -> a token that MUST appear in the AttributeError migration message.
_REMOVED = {
    # COO family -> CSR / CSC + coo2csr
    "COO": "CSR",
    "binary_coomv": "CSR", "binary_coomv_p": "CSR",
    "binary_coomm": "CSR", "binary_coomm_p": "CSR",
    "coomv": "CSR", "coomv_p": "CSR",
    "coomm": "CSR", "coomm_p": "CSR",
    "update_coo_on_binary_pre": "CSR", "update_coo_on_binary_post": "CSR",
    "update_coo_on_binary_pre_p": "CSR", "update_coo_on_binary_post_p": "CSR",
    # bitpack / compact FCN -> fcnmv / fcnmm
    "bitpack_binary_fcnmv": "fcnmv", "bitpack_binary_fcnmv_p": "fcnmv",
    "bitpack_binary_fcnmm": "fcnmv", "bitpack_binary_fcnmm_p": "fcnmv",
    "compact_binary_fcnmv": "fcnmv", "compact_binary_fcnmv_p": "fcnmv",
    "compact_binary_fcnmm": "fcnmv", "compact_binary_fcnmm_p": "fcnmv",
    # layout objects
    "EllLayout": "FixedNumPer", "CscLayout": "FixedNumPer",
}

# The 23 names exported by brainevent 0.0.7 ``__all__`` that are absent from 0.1.0.
_V007_REMOVED_FROM_ALL = {
    "COO", "binary_coomv", "binary_coomv_p", "binary_coomm", "binary_coomm_p",
    "coomv", "coomv_p", "coomm", "coomm_p",
    "update_coo_on_binary_pre", "update_coo_on_binary_post",
    "update_coo_on_binary_pre_p", "update_coo_on_binary_post_p",
    "FixedPreNumConn", "FixedPostNumConn",
    "bitpack_binary_fcnmv", "bitpack_binary_fcnmv_p",
    "bitpack_binary_fcnmm", "bitpack_binary_fcnmm_p",
    "compact_binary_fcnmv", "compact_binary_fcnmv_p",
    "compact_binary_fcnmm", "compact_binary_fcnmm_p",
}


@pytest.mark.parametrize("old, new", list(_RENAMES.items()))
def test_renamed_name_warns_and_forwards(old, new):
    with pytest.warns(DeprecationWarning):
        obj = getattr(brainevent, old)
    assert obj is getattr(brainevent, new)


@pytest.mark.parametrize("old", list(_RENAMES))
def test_renamed_name_importable_via_from(old):
    # PEP 562 module __getattr__ also powers ``from brainevent import <old>``.
    with pytest.warns(DeprecationWarning):
        module = __import__("brainevent", fromlist=[old])
        assert getattr(module, old) is not None


@pytest.mark.parametrize("old, token", list(_REMOVED.items()))
def test_removed_name_raises_with_migration(old, token):
    with pytest.raises(AttributeError) as excinfo:
        getattr(brainevent, old)
    message = str(excinfo.value)
    assert token in message, f"{old}: expected {token!r} in message: {message}"


def test_unknown_attribute_raises_plain_attributeerror():
    with pytest.raises(AttributeError):
        _ = brainevent.this_symbol_does_not_exist


def test_fresh_import_emits_no_warning():
    # A clean interpreter importing brainevent with warnings-as-errors must
    # succeed: deprecation warnings fire only on deprecated-name access.
    code = "import warnings; warnings.simplefilter('error'); import brainevent"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_all_exports_resolve_and_are_stable():
    assert len(brainevent.__all__) == 154
    for name in brainevent.__all__:
        assert hasattr(brainevent, name), name


def test_all_v007_removed_names_have_backcompat():
    covered = set(_RENAMES) | set(_REMOVED)
    missing = _V007_REMOVED_FROM_ALL - covered
    assert not missing, f"v0.0.7 names without backward-compat: {missing}"
    # A name must not be both a rename and a removed-with-error entry.
    assert not (set(_RENAMES) & set(_REMOVED))
    # Deprecated names stay hidden from the curated public surface.
    for name in covered:
        assert name not in brainevent.__all__
