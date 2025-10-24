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


from brainevent._fixed_conn_num.float_test import TestVector, TestMatrix
from brainevent._test_util import gen_events


class TestEventVector(TestVector):
    def _generate_x(self, shape, require_float=True):
        if not isinstance(shape, (tuple, list)):
            shape = [shape]
        yield gen_events(shape, asbool=False)
        if not require_float:
            yield gen_events(shape, asbool=True)


class TestEventMatrix(TestMatrix):
    def _generate_x(self, shape, require_float=True):
        if not isinstance(shape, (tuple, list)):
            shape = [shape]
        yield gen_events(shape, asbool=False)
        if not require_float:
            yield gen_events(shape, asbool=True)
