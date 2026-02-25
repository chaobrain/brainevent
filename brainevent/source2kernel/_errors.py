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

"""Custom exception hierarchy for jax-kernel-bridge."""


class JKBError(Exception):
    """Base exception for jax-kernel-bridge."""
    pass


class ToolchainError(JKBError):
    """Compilation toolchain missing or incompatible."""
    pass


class CompilationError(JKBError):
    """CUDA compilation failed."""

    def __init__(self, message: str, compiler_output: str = "",
                 command: str = ""):
        self.compiler_output = compiler_output
        self.command = command
        full_msg = message
        if command:
            full_msg += f"\n\nCommand:\n  {command}"
        if compiler_output:
            full_msg += f"\n\nCompiler output:\n{compiler_output}"
        super().__init__(full_msg)


class RegistrationError(JKBError):
    """JAX FFI target registration failed."""
    pass
