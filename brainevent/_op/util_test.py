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

"""Reproduction tests for the ``brainevent/_op/util.py`` audit findings.

Each test targets one finding from
``dev/2026-06-13-op-issues.md``:

* **H2** -- ``defjvp`` single-result path called the removed
  ``jax.interpreters.ad.standard_jvp`` -> ``AttributeError`` at registration.
* **M1** -- ``general_batching_rule`` crashed (``scan got no values``) when
  every operand axis is ``None``.
* **M2** -- ``_standard_jvp`` fragmented a bare-array JVP return by iterating
  its leading axis and validated structure with a strippable ``assert``.
* **L1** -- ``jaxtype_to_warptype`` spuriously rejected Python builtin
  ``float``/``int``/``bool`` (``float == np.float64`` is ``False``).
* **L2** -- ``jaxinfo_to_warpinfo`` silently mapped a 0-D scalar to
  ``ndim=1``.

The tests are written to fail against the pre-fix ``util.py`` and pass once
the corrected behavior is in place.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.extend.core import Primitive
from jax.interpreters import mlir

from brainevent._op.util import (
    defjvp,
    general_batching_rule,
    jaxtype_to_warptype,
    jaxinfo_to_warpinfo,
    dtype_suffix,
    spike_suffix,
)

warp_installed = importlib.util.find_spec('warp') is not None


# ---------------------------------------------------------------------------
# H2 -- public ``defjvp`` on a single-result primitive must work on JAX >= 0.9
# ---------------------------------------------------------------------------

def test_h2_defjvp_single_result_primitive():
    """``defjvp`` on a single-result primitive registers and differentiates.

    Pre-fix this raised ``AttributeError`` at *registration* time because the
    ``multiple_results=False`` branch referenced the removed
    ``ad.standard_jvp``.  Post-fix the inlined standard JVP must let
    ``jax.jvp`` / ``jax.grad`` flow through the primitive.
    """
    prim = Primitive('h2_single_result_mul')
    prim.multiple_results = False
    prim.def_impl(lambda x, y: x * y)
    prim.def_abstract_eval(
        lambda x, y: jax.core.ShapedArray(jnp.broadcast_shapes(x.shape, y.shape), x.dtype)
    )

    # d/dx (x*y) = y ; d/dy (x*y) = x
    # This call must NOT raise AttributeError on JAX >= 0.9.
    defjvp(prim,
           lambda xdot, x, y: xdot * y,
           lambda ydot, x, y: x * ydot)

    def call(x, y):
        return prim.bind(x, y)

    x = jnp.asarray(3.0)
    y = jnp.asarray(5.0)

    # forward-mode
    primal, tangent = jax.jvp(call, (x, y), (jnp.asarray(1.0), jnp.asarray(0.0)))
    assert float(primal) == pytest.approx(15.0)
    assert float(tangent) == pytest.approx(5.0)  # == y

    # reverse-mode
    gx = jax.grad(call, argnums=0)(x, y)
    gy = jax.grad(call, argnums=1)(x, y)
    assert float(gx) == pytest.approx(5.0)  # y
    assert float(gy) == pytest.approx(3.0)  # x


# ---------------------------------------------------------------------------
# M1 -- ``general_batching_rule`` with all-None axes must short-circuit
# ---------------------------------------------------------------------------

def test_m1_general_batching_rule_all_unbatched():
    """All-``None`` axes must bind directly, not invoke an empty ``scan``.

    Pre-fix ``jax.lax.scan(f, 0, {})`` raised
    ``ValueError: scan got no values to scan over``.  Post-fix the rule
    binds the primitive once and reports every output as unbatched
    (a ``None`` batch dimension, JAX's ``not_mapped`` sentinel).
    """
    prim = Primitive('m1_unbatched_add')
    prim.multiple_results = False
    prim.def_impl(lambda x, y: x + y)
    prim.def_abstract_eval(lambda x, y: jax.core.ShapedArray(x.shape, x.dtype))
    # Needed only by the scan path; the all-None case must short-circuit
    # *before* the scan, so registering a lowering simply makes the test robust.
    mlir.register_lowering(prim, mlir.lower_fun(lambda x, y: x + y, multiple_results=False))

    a = jnp.arange(3.0)
    b = jnp.ones((3,))

    out, out_dim = general_batching_rule(prim, (a, b), (None, None))

    np.testing.assert_allclose(np.asarray(out), np.asarray(a + b))
    # Every output dim must be flagged unbatched.  JAX's ``not_mapped``
    # sentinel is ``None`` in every supported version, and the default pytree
    # flattener treats ``None`` as an empty node, so we flatten with
    # ``is_leaf`` recognizing the sentinel as a real leaf (this mirrors how
    # JAX inspects output batch dims).
    is_not_mapped = lambda x: x is None
    leaves = jax.tree.leaves(out_dim, is_leaf=is_not_mapped)
    assert leaves, 'expected at least one output-dim leaf'
    assert all(leaf is None for leaf in leaves)
    # The out_dim pytree must mirror the output structure.
    assert (jax.tree.structure(out_dim, is_leaf=is_not_mapped)
            == jax.tree.structure(out))


def test_m1_general_batching_rule_mixed_still_works():
    """A genuinely batched operand still scans and reports leading-axis output.

    Guards against the short-circuit accidentally swallowing the normal path.
    """
    prim = Primitive('m1_mixed_add')
    prim.multiple_results = False
    prim.def_impl(lambda x, y: x + y)
    prim.def_abstract_eval(lambda x, y: jax.core.ShapedArray(x.shape, x.dtype))
    # The scan body binds the primitive; lowering lets the traced scan execute.
    mlir.register_lowering(prim, mlir.lower_fun(lambda x, y: x + y, multiple_results=False))

    batched = jnp.arange(6.0).reshape(2, 3)  # batched along axis 0
    unbatched = jnp.ones((3,))

    out, out_dim = general_batching_rule(prim, (batched, unbatched), (0, None))

    np.testing.assert_allclose(np.asarray(out), np.asarray(batched + unbatched))
    assert all(leaf == 0 for leaf in jax.tree.leaves(out_dim))


# ---------------------------------------------------------------------------
# M2 -- ``_standard_jvp`` must reject a bare-array JVP return (no fragmenting)
# ---------------------------------------------------------------------------

def test_m2_standard_jvp_rejects_bare_array_return():
    """A multi-result JVP rule returning a bare array must raise ``TypeError``.

    Pre-fix ``tuple(rule(...))`` iterated the leading axis of the returned
    array (silent fragmentation) and validated structure with a strippable
    ``assert``.  Post-fix a non-sequence return is a clear ``TypeError``.
    """
    prim = Primitive('m2_multi_result')
    prim.multiple_results = True
    prim.def_impl(lambda x: (x, x))
    prim.def_abstract_eval(
        lambda x: (jax.core.ShapedArray(x.shape, x.dtype),
                   jax.core.ShapedArray(x.shape, x.dtype))
    )

    # This rule INCORRECTLY returns a bare array instead of a (t0, t1) tuple.
    def bad_rule(xdot, x):
        return xdot  # bare array -- should be a length-2 sequence

    defjvp(prim, bad_rule)

    def call(x):
        return prim.bind(x)

    x = jnp.arange(4.0)
    with pytest.raises(TypeError):
        jax.jvp(call, (x,), (jnp.ones_like(x),))


def test_m2_standard_jvp_accepts_proper_sequence_return():
    """A correctly-shaped multi-result JVP rule still differentiates."""
    prim = Primitive('m2_multi_result_ok')
    prim.multiple_results = True
    prim.def_impl(lambda x: (x * 2.0, x * 3.0))
    prim.def_abstract_eval(
        lambda x: (jax.core.ShapedArray(x.shape, x.dtype),
                   jax.core.ShapedArray(x.shape, x.dtype))
    )

    def good_rule(xdot, x):
        return (xdot * 2.0, xdot * 3.0)

    defjvp(prim, good_rule)

    def call(x):
        a, b = prim.bind(x)
        return a, b

    x = jnp.arange(4.0)
    (pa, pb), (ta, tb) = jax.jvp(call, (x,), (jnp.ones_like(x),))
    np.testing.assert_allclose(np.asarray(ta), np.full((4,), 2.0))
    np.testing.assert_allclose(np.asarray(tb), np.full((4,), 3.0))


# ---------------------------------------------------------------------------
# L1 -- ``jaxtype_to_warptype`` must accept Python builtins
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not warp_installed, reason='warp not installed')
def test_l1_jaxtype_to_warptype_accepts_python_builtins():
    """Python builtin ``float``/``int``/``bool`` must map like their dtypes.

    Pre-fix ``float == np.float64`` is ``False`` so builtins spuriously
    raised ``ValueError``.  Post-fix ``np.dtype(...)`` normalization makes
    ``float`` resolve to whatever ``np.dtype(float)`` is (float64).
    """
    # The builtin must resolve to the same Warp type as its normalized dtype.
    assert jaxtype_to_warptype(float) is jaxtype_to_warptype(np.dtype(float))
    assert jaxtype_to_warptype(int) is jaxtype_to_warptype(np.dtype(int))
    assert jaxtype_to_warptype(bool) is jaxtype_to_warptype(np.dtype(bool))

    # And concretely: np.dtype(float) is float64 on every supported platform.
    assert jaxtype_to_warptype(float) is jaxtype_to_warptype(np.float64)


@pytest.mark.skipif(not warp_installed, reason='warp not installed')
def test_l1_jaxtype_to_warptype_unsupported_raises_clear_error():
    """Complex / bfloat16 (unsupported by Warp) must raise a clear error."""
    with pytest.raises(ValueError):
        jaxtype_to_warptype(np.complex64)


# ---------------------------------------------------------------------------
# L2 -- ``jaxinfo_to_warpinfo`` must not silently bump a 0-D scalar to ndim=1
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not warp_installed, reason='warp not installed')
def test_l2_jaxinfo_to_warpinfo_rejects_scalar():
    """A 0-D ``ShapeDtypeStruct`` must be handled explicitly, not silently bumped.

    Pre-fix ``warp.array(ndim=0)`` silently became ``ndim=1`` (JAX<->Warp
    ndim mismatch).  Post-fix a clear error is raised for ``ndim == 0``.
    """
    scalar = jax.ShapeDtypeStruct(shape=(), dtype=np.float32)
    assert scalar.ndim == 0
    with pytest.raises(ValueError):
        jaxinfo_to_warpinfo(scalar)


@pytest.mark.skipif(not warp_installed, reason='warp not installed')
def test_l2_jaxinfo_to_warpinfo_normal_ndim_ok():
    """A normal >=1-D struct still converts and preserves ``ndim``."""
    info = jax.ShapeDtypeStruct(shape=(4, 5), dtype=np.float32)
    warp_arr_type = jaxinfo_to_warpinfo(info)
    assert warp_arr_type.ndim == 2


# ---------------------------------------------------------------------------
# Kernel-name suffix helpers
#
# ``dtype_suffix`` / ``spike_suffix`` encode the naming contract between the
# Python dispatch layer and the symbols defined in the ``.cu`` sources. They
# replaced 31 verbatim copies of the same table, so the mapping is pinned here:
# a silent change would mis-dispatch to a wrong-precision kernel rather than
# raise, and CPU-only test runs never lower the CUDA generators that use it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    'dtype, expected',
    [
        (jnp.float16, '_f16'),
        (jnp.float32, '_f32'),
        (jnp.float64, '_f64'),
        (jnp.bfloat16, '_bf16'),
    ],
)
def test_dtype_suffix_maps_each_supported_precision(dtype, expected):
    """Every precision with a generated kernel gets its own distinct suffix."""
    assert dtype_suffix(dtype) == expected


def test_dtype_suffix_accepts_every_spelling_of_a_dtype():
    """np dtypes, jnp dtypes, bare types and ShapeDtypeStruct.dtype agree.

    The call sites this helper replaced passed all four spellings, so they must
    normalize identically.
    """
    info = jax.ShapeDtypeStruct((3, 4), jnp.float16)
    assert (
        dtype_suffix(np.dtype('float16'))
        == dtype_suffix(jnp.dtype('float16'))
        == dtype_suffix(np.float16)
        == dtype_suffix(info.dtype)
        == '_f16'
    )


def test_dtype_suffix_falls_back_to_f32_for_unmapped_dtype():
    """Unmapped dtypes fall back to ``_f32`` rather than raising.

    This preserves the behaviour of the per-module tables, every one of which
    used ``.get(dtype, '_f32')``. It is a known sharp edge, not an oversight:
    an integer weight dtype silently selects the f32 kernel.
    """
    assert dtype_suffix(jnp.int32) == '_f32'
    assert dtype_suffix(np.dtype('int64')) == '_f32'


def test_dtype_suffixes_are_distinct():
    """No two supported precisions may collide on the same kernel symbol."""
    suffixes = [dtype_suffix(d) for d in (jnp.float16, jnp.float32, jnp.float64, jnp.bfloat16)]
    assert len(set(suffixes)) == len(suffixes)


@pytest.mark.parametrize(
    'dtype, expected',
    [
        (jnp.bool_, '_bool'),
        (np.dtype('bool'), '_bool'),
        (jnp.float32, '_float'),
        (jnp.float16, '_float'),
        (jnp.bfloat16, '_float'),
        (jnp.int8, '_float'),
    ],
)
def test_spike_suffix_splits_bool_from_everything_else(dtype, expected):
    """Only boolean event arrays select the ``_bool`` kernel variant."""
    assert spike_suffix(dtype) == expected
