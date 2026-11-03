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

"""Tests for ``brainevent._op.ffi_naming``.

Most fingerprint behaviour is exercised through its consumer in
``numba_ffi_test.py``; this module covers the serializer's value cases.
"""

import numpy as np
import pytest

from brainevent._op.ffi_naming import _serialize_value, kernel_content_fingerprint


def _factory(scalar):
    def kernel(x, out):
        out[0] = scalar(x[0])

    return kernel


class _UserScalar:
    """An arbitrary user class: no deterministic content, so no fingerprint."""


class TestNumpyScalarClasses:
    """#215: a closure over a NumPy scalar class must be fingerprintable."""

    @pytest.mark.parametrize('scalar', [np.float32, np.float64, np.int32, np.uint64, np.bool_, np.complex64])
    def test_builtin_scalar_class_serializes(self, scalar):
        parts = []
        assert _serialize_value(scalar, parts, 0)
        assert parts

    def test_closure_over_float32_has_fingerprint(self):
        assert kernel_content_fingerprint(_factory(np.float32)) is not None

    def test_two_float32_closures_share_fingerprint(self):
        fp_a = kernel_content_fingerprint(_factory(np.float32))
        fp_b = kernel_content_fingerprint(_factory(np.float32))
        assert fp_a is not None
        assert fp_a == fp_b

    def test_float32_and_float64_closures_differ(self):
        fp32 = kernel_content_fingerprint(_factory(np.float32))
        fp64 = kernel_content_fingerprint(_factory(np.float64))
        assert fp32 is not None and fp64 is not None
        assert fp32 != fp64

    def test_all_builtin_scalar_classes_are_distinct(self):
        # Distinct classes must never share an encoding (e.g. np.intc vs np.int32
        # on platforms where both exist and differ only by name).
        scalars = set(np.sctypeDict.values())
        encodings = {}
        for scalar in scalars:
            parts = []
            assert _serialize_value(scalar, parts, 0), scalar
            encodings.setdefault(b''.join(parts), []).append(scalar)
        assert all(len(v) == 1 for v in encodings.values()), encodings

    def test_class_value_and_dtype_are_not_conflated(self):
        cls_parts, dtype_parts, value_parts = [], [], []
        assert _serialize_value(np.float32, cls_parts, 0)
        assert _serialize_value(np.dtype(np.float32), dtype_parts, 0)
        assert _serialize_value(np.float32(0), value_parts, 0)
        assert len({b''.join(cls_parts), b''.join(dtype_parts), b''.join(value_parts)}) == 3

    def test_user_class_still_has_no_fingerprint(self):
        assert not _serialize_value(_UserScalar, [], 0)
        assert kernel_content_fingerprint(_factory(_UserScalar)) is None

    def test_numpy_scalar_subclass_has_no_fingerprint(self):
        # Only the exact built-in classes are accepted; a user subclass keeps
        # the conservative fallback.
        class MyFloat(np.float32):
            pass

        assert not _serialize_value(MyFloat, [], 0)
        assert kernel_content_fingerprint(_factory(MyFloat)) is None

    def test_depth_limit_still_applies(self):
        assert not _serialize_value(np.float32, [], 9)
