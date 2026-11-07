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

"""TCSR structure, mirrors, and traversal workspaces."""

from .preprocess import (
    HybridWorkspace,
    TCSBuffers,
    TCSCMirror,
    TCSRComponents,
    build_hybrid_workspace,
    build_tcsr_structure,
    build_tcsc_mirror,
    sort_csr,
    validate_tcsc_mirror,
    workspace_from_operands,
    workspace_operands,
)

__all__ = [
    "HybridWorkspace",
    "TCSBuffers",
    "TCSCMirror",
    "TCSRComponents",
    "build_hybrid_workspace",
    "build_tcsr_structure",
    "build_tcsc_mirror",
    "sort_csr",
    "validate_tcsc_mirror",
    "workspace_from_operands",
    "workspace_operands",
]
