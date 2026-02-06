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


__all__ = [
    'MathError',
    'KernelNotAvailableError',
    'KernelCompilationError',
    'KernelFallbackExhaustedError',
    'KernelExecutionError',
]


class MathError(Exception):
    __module__ = 'brainevent'


class KernelNotAvailableError(Exception):
    """Raised when a kernel backend is not installed or version incompatible."""
    __module__ = 'brainevent'


class KernelCompilationError(Exception):
    """Raised when a kernel fails to compile."""
    __module__ = 'brainevent'


class KernelFallbackExhaustedError(Exception):
    """Raised when all fallback kernels have failed."""
    __module__ = 'brainevent'


class KernelExecutionError(Exception):
    """Raised when a kernel execution fails with helpful alternatives."""
    __module__ = 'brainevent'
