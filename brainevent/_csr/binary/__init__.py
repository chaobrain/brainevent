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

"""Binary and weighted event multiplication."""

from .binary import (
    binary_csrmv,
    binary_csrmv_p,
    binary_csrmm,
    binary_csrmm_p,
)
from .binary_indexed import (
    binary_csrmv_indexed,
    binary_csrmv_indexed_p,
    binary_csrmm_indexed,
    binary_csrmm_indexed_p,
)
from .hybrid_config import (
    HybridConfig,
    HybridBenchmarkRecord,
    DEFAULT_HYBRID_CONFIG,
    validate_config,
    compile_flags_for_config,
    module_suffix_for_config,
    get_hybrid_config,
    save_hybrid_config,
    hybrid_task_capacity,
    current_device_kind,
)
from .initialize import (
    init_csr_config,
)

__all__ = [
    'binary_csrmv',
    'binary_csrmv_p',
    'binary_csrmm',
    'binary_csrmm_p',
    'binary_csrmv_indexed',
    'binary_csrmv_indexed_p',
    'binary_csrmm_indexed',
    'binary_csrmm_indexed_p',
    'HybridConfig',
    'HybridBenchmarkRecord',
    'DEFAULT_HYBRID_CONFIG',
    'validate_config',
    'compile_flags_for_config',
    'module_suffix_for_config',
    'get_hybrid_config',
    'save_hybrid_config',
    'hybrid_task_capacity',
    'current_device_kind',
    'init_csr_config',
]
