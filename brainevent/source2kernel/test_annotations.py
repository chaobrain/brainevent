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

"""Test // @JKB annotation-based function discovery."""

import numpy as np
import pytest

import jax as _jax
import pytest as _pytest

requires_gpu = _pytest.mark.skipif(
    not (bool(_jax.devices("gpu")) if True else False),
    reason="No GPU detected via jax.devices('gpu')",
)

pytestmark = requires_gpu


# --- Unit tests for normalize_tokens (no GPU needed) ---

def test_normalize_tokens_tvm_aliases():
    """normalize_tokens converts jax-tvm-ffi aliases to canonical JKB tokens."""
    from brainevent.source2kernel._codegen import normalize_tokens

    assert normalize_tokens(["args", "rets", "ctx.stream"]) == ["arg", "ret", "stream"]


def test_normalize_tokens_attrs_prefix():
    """attrs.name → attr.name (bare form)."""
    from brainevent.source2kernel._codegen import normalize_tokens

    result = normalize_tokens(["attrs.scale", "attrs.offset"])
    assert result == ["attr.scale", "attr.offset"]


def test_normalize_tokens_passthrough():
    """Already-canonical tokens are not modified."""
    from brainevent.source2kernel._codegen import normalize_tokens

    tokens = ["arg", "ret", "stream", "attr.x:float32", "attr.y"]
    assert normalize_tokens(tokens) == tokens


def test_normalize_tokens_mixed():
    """Mixed JKB and jax-tvm-ffi tokens all normalise correctly."""
    from brainevent.source2kernel._codegen import normalize_tokens

    result = normalize_tokens(["args", "rets", "attrs.scale", "ctx.stream"])
    assert result == ["arg", "ret", "attr.scale", "stream"]


# --- Unit tests for annotation parsing (no GPU needed, but in same file) ---

def test_parse_annotations_basic():
    """parse_jkb_annotations finds annotated functions."""
    from brainevent.source2kernel._codegen import parse_jkb_annotations

    source = """
    // @JKB my_add
    void my_add(const JKB::Tensor a, const JKB::Tensor b,
                JKB::Tensor out, int64_t stream) {
    }
    """
    result = parse_jkb_annotations(source)
    assert "my_add" in result
    assert result["my_add"] == ["arg", "arg", "ret", "stream"]


def test_parse_annotations_no_stream():
    """Annotation correctly infers arg_spec without stream."""
    from brainevent.source2kernel._codegen import parse_jkb_annotations

    source = """
    // @JKB add_cpu
    void add_cpu(const JKB::Tensor x, JKB::Tensor y) {}
    """
    result = parse_jkb_annotations(source)
    assert result["add_cpu"] == ["arg", "ret"]


def test_parse_annotations_multiple():
    """Multiple annotations in the same source."""
    from brainevent.source2kernel._codegen import parse_jkb_annotations

    source = """
    // @JKB func_a
    void func_a(const JKB::Tensor x, JKB::Tensor out) {}

    // @JKB func_b
    void func_b(const JKB::Tensor a, const JKB::Tensor b,
                JKB::Tensor out, int64_t stream) {}
    """
    result = parse_jkb_annotations(source)
    assert set(result.keys()) == {"func_a", "func_b"}
    assert result["func_a"] == ["arg", "ret"]
    assert result["func_b"] == ["arg", "arg", "ret", "stream"]


def test_parse_annotations_no_annotations():
    """Raises when no annotations found."""
    from brainevent.source2kernel._codegen import parse_jkb_annotations
    from brainevent.source2kernel._errors import JKBError

    with pytest.raises(JKBError, match="No '// @JKB"):
        parse_jkb_annotations("void foo() {}")


def test_parse_annotations_duplicate():
    """Raises on duplicate @JKB annotation."""
    from brainevent.source2kernel._codegen import parse_jkb_annotations
    from brainevent.source2kernel._errors import JKBError

    source = """
    // @JKB same_name
    void same_name(const JKB::Tensor x, JKB::Tensor out) {}
    // @JKB same_name
    void same_name(const JKB::Tensor x, JKB::Tensor out) {}
    """
    with pytest.raises(JKBError, match="Duplicate"):
        parse_jkb_annotations(source)


# --- Integration test: compile and run a CUDA kernel via annotations ---

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "jkb/common.h"

__global__ void add_k(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}

// @JKB vector_add
void vector_add(const JKB::Tensor a, const JKB::Tensor b,
                JKB::Tensor out, int64_t stream) {
    int n = a.numel();
    add_k<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(a.data_ptr()),
        static_cast<const float*>(b.data_ptr()),
        static_cast<float*>(out.data_ptr()), n);
}
"""


@pytest.fixture(scope="module")
def annotation_module():
    import brainevent.source2kernel as jkb
    return jkb.load_cuda_inline(
        name="test_annotation_vadd",
        cuda_sources=CUDA_SRC,
        # functions=None  →  discovered from // @JKB annotations
        force_rebuild=True,
    )


def test_annotation_cuda_kernel(annotation_module):
    """CUDA kernel compiled via annotation produces correct result."""
    import jax
    import jax.numpy as jnp

    a = jnp.ones(512, dtype=jnp.float32)
    b = jnp.full(512, 2.0, dtype=jnp.float32)

    result = jax.ffi.ffi_call(
        "test_annotation_vadd.vector_add",
        jax.ShapeDtypeStruct((512,), jnp.float32),
    )(a, b)

    expected = np.full(512, 3.0, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)
