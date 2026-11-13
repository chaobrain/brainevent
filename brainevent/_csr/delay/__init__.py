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

"""Per-edge-delay spike delivery on a precomputed, stateless CSR layout."""

from brainevent._csr.delay.delay import csr_delay_deliver_layout_p, csr_delay_deliver_layout_p_call
from brainevent._csr.delay.layout import CSRDelayLayout, csr_delay_deliver_layout

__all__ = [
    'CSRDelayLayout',
    'csr_delay_deliver_layout',
    'csr_delay_deliver_layout_p',
    'csr_delay_deliver_layout_p_call',
]
