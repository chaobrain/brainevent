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

import pytest

from brainevent._error import TVMModuleAlreadyRegisteredError
from brainevent._op import util


class _FakeCudaModule:
    def __init__(self, functions):
        for fn in functions:
            setattr(self, fn, object())


class _FakeCpp:
    def __init__(self):
        self.load_calls = []

    def load_inline(self, name, cuda_sources, functions):
        self.load_calls.append((name, cuda_sources, tuple(functions)))
        return _FakeCudaModule(functions)


class _FakeTVMFFI:
    def __init__(self):
        self.cpp = _FakeCpp()


class _FakeJaxTVMFFI:
    def __init__(self):
        self.register_calls = []

    def register_ffi_target(self, name, fn, args, platform):
        self.register_calls.append((name, fn, tuple(args), platform))


def _setup_fake_tvm(monkeypatch):
    fake_tvm = _FakeTVMFFI()
    fake_jax_tvm = _FakeJaxTVMFFI()
    monkeypatch.setattr(util, "tvmffi_installed", True)
    monkeypatch.setattr(util, "tvm_ffi", fake_tvm, raising=False)
    monkeypatch.setattr(util, "jax_tvm_ffi", fake_jax_tvm, raising=False)
    util._registered_tvm_modules.clear()
    util._registered_tvm_module_signatures.clear()
    return fake_tvm, fake_jax_tvm


def test_register_tvm_cuda_kernels_reuses_cached_module_when_signature_matches(monkeypatch):
    fake_tvm, fake_jax_tvm = _setup_fake_tvm(monkeypatch)
    source = "__global__ void k() {}"

    first = util.register_tvm_cuda_kernels(
        source_code=source,
        module="my_module",
        functions=["k1", "k2"],
    )
    second = util.register_tvm_cuda_kernels(
        source_code=source,
        module="my_module",
        functions=["k2", "k1"],  # same function set, different order
    )

    assert first is second
    assert len(fake_tvm.cpp.load_calls) == 1
    assert len(fake_jax_tvm.register_calls) == 2


def test_register_tvm_cuda_kernels_raises_on_same_name_different_source(monkeypatch):
    fake_tvm, fake_jax_tvm = _setup_fake_tvm(monkeypatch)

    util.register_tvm_cuda_kernels(
        source_code="__global__ void k() { int x = 1; }",
        module="my_module",
        functions=["k1"],
    )

    with pytest.raises(TVMModuleAlreadyRegisteredError, match="different source/functions signature"):
        util.register_tvm_cuda_kernels(
            source_code="__global__ void k() { int x = 2; }",
            module="my_module",
            functions=["k1"],
        )

    assert len(fake_tvm.cpp.load_calls) == 1
    assert len(fake_jax_tvm.register_calls) == 1
