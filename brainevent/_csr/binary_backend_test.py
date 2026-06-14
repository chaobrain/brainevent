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

from brainevent._csr.binary import binary_csrmv_p


def test_binary_csrmv_gpu_cusparse_backend_names():
    backends = binary_csrmv_p.available_backends('gpu')

    assert 'BCOO_cusparse' in backends
    assert 'JAX_cusparse' in backends
    assert 'cusparse' not in backends
