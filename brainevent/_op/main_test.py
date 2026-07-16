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

"""Audit tests for ``brainevent/_op/main.py`` (L3, M13).

* L3 - ``apply_primitive`` is resolved through the ``_compatible_import`` shim
  and is identical to the function the legacy ``jax.interpreters.xla`` path
  exposed.
* M13 - a global backend that is not registered for a platform no longer fails
  silently: the selection emits a (deduplicated) warning and falls back to the
  primitive default, still producing the correct result.
"""

import importlib.util
import warnings
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.interpreters import xla

from brainevent import KernelCompilationError, KernelFallbackExhaustedError
from brainevent._compatible_import import apply_primitive
from brainevent._op.main import XLACustomKernel
from brainevent.config import set_backend, clear_backends

numba_installed = importlib.util.find_spec('numba') is not None


def _offset_kernel_generator(offset: float):
    """Build a ``KernelGenerator`` whose kernel adds a constant *offset*.

    Used across the Cluster A (F1/F6/F17/F19) tests below as a cheap,
    dependency-free stand-in for a real Numba/Warp/Pallas backend: the
    returned kernel is plain JAX (``x + offset``), so it traces correctly
    inside ``mlir.lower_fun`` without needing numba/warp/pallas installed.
    """

    def gen(**kwargs):
        def kernel(x):
            return (x + offset,)

        return kernel

    return gen


def test_l3_apply_primitive_shim_is_xla_apply_primitive():
    """The shim resolves to exactly the same callable as the legacy path."""
    assert xla.apply_primitive is apply_primitive


def test_m13_warn_backend_once_deduplicates():
    """``_warn_backend_once`` emits each distinct message at most once."""
    k = XLACustomKernel('m13_dedup_probe')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        k._warn_backend_once('message-A')
        k._warn_backend_once('message-A')  # duplicate -> suppressed
        k._warn_backend_once('message-B')
    messages = [str(w.message) for w in caught]
    assert messages == ['message-A', 'message-B']


@pytest.mark.skipif(not numba_installed, reason='Numba not installed')
def test_m13_unregistered_global_backend_warns_and_falls_back():
    """A global backend absent for this platform warns, then uses the default."""
    import numba
    from brainevent import numba_kernel

    def gen(**kwargs):
        @numba.njit
        def add_one(x, out):
            for i in range(x.size):
                out[i] = x[i] + 1.0

        def kernel(x):
            return numba_kernel(add_one, outs=kwargs['outs'])(x)

        return kernel

    prim = XLACustomKernel('m13_global_mismatch_probe')
    prim.def_numba_kernel(gen)  # 'numba' becomes the cpu default automatically

    cpu = jax.devices('cpu')[0]
    x = jax.device_put(jnp.arange(8, dtype=jnp.float32), cpu)

    # Request a global cpu backend that was never registered for this primitive.
    set_backend('cpu', 'warp')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = prim(x, outs=[jax.ShapeDtypeStruct((8,), jnp.float32)])
            jax.block_until_ready(out)
        text = ' '.join(str(w.message) for w in caught)
        assert 'warp' in text and 'ignoring' in text.lower(), text
        # ...and the fallback still produced the correct numbers.
        np.testing.assert_allclose(np.asarray(out[0]), np.arange(8) + 1.0)
    finally:
        clear_backends()


# ---------------------------------------------------------------------------
# F1 -- stale backend dispatch: switching a default/global backend must be
# visible on the very next call, including through already-compiled/jitted
# call sites (D1).
# ---------------------------------------------------------------------------


def test_f1_eager_backend_switch_and_per_call_override():
    """``set_default`` / ``set_backend`` / ``clear_backends`` flip live results.

    Pre-fix (finding #1), none of these setters invalidated JAX's eager
    dispatch cache, so a backend switch silently had no effect on a
    primitive that had already been called once. Post-fix, each setter
    calls ``jax.clear_caches()`` when (and only when) the effective value
    changes, so eager calls immediately reflect the new backend.
    """
    prim = XLACustomKernel('f1_eager_probe')
    prim.def_kernel('backend_a', 'cpu', _offset_kernel_generator(1.0))
    prim.def_kernel('backend_b', 'cpu', _offset_kernel_generator(2.0))

    # Explicit dtype: the declared f32 outs must match the kernel's output
    # even when another test module has left ``jax_enable_x64`` on.
    x = jnp.arange(4, dtype=jnp.float32)
    outs = [jax.ShapeDtypeStruct((4,), jnp.float32)]

    def run():
        return np.asarray(jax.block_until_ready(prim(x, outs=outs))[0])

    try:
        # First-registered backend ('backend_a') is the default.
        np.testing.assert_allclose(run(), np.arange(4.0) + 1.0)

        # set_default flips the per-primitive default -> must be visible now.
        prim.set_default('cpu', 'backend_b')
        np.testing.assert_allclose(run(), np.arange(4.0) + 2.0)

        # Per-call backend= overrides both the per-primitive and global default.
        np.testing.assert_allclose(
            np.asarray(jax.block_until_ready(prim(x, outs=outs, backend='backend_a'))[0]),
            np.arange(4.0) + 1.0,
        )

        # Global set_backend overrides the per-primitive default.
        set_backend('cpu', 'backend_b')
        prim.set_default('cpu', 'backend_a')  # per-primitive default is now 'a'...
        np.testing.assert_allclose(run(), np.arange(4.0) + 2.0)  # ...but global 'b' wins.

        # clear_backends reverts to the per-primitive default ('backend_a').
        clear_backends()
        np.testing.assert_allclose(run(), np.arange(4.0) + 1.0)

        # Idempotent set_default (same value already in effect) must not
        # raise and must not needlessly clear caches. This is checked last,
        # via mocking, so a mocked (no-op) clear_caches call does not leave
        # the primitive's own eager-dispatch cache stale for a later
        # assertion in this test.
        with mock.patch('jax.clear_caches') as clear_caches:
            prim.set_default('cpu', 'backend_a')  # already the default -> no-op.
            assert clear_caches.call_count == 0
            prim.set_default('cpu', 'backend_b')  # genuine change -> clears.
            assert clear_caches.call_count == 1
        prim.set_default('cpu', 'backend_a')  # restore, with a real clear_caches.
    finally:
        clear_backends()


def test_f1_jit_warm_function_picks_up_switch_after_set_default():
    """A ``jax.jit``-warmed function must see a ``set_default`` switch.

    Pre-fix, warming (compiling) a jitted wrapper before switching the
    default backend meant the compiled executable kept dispatching to the
    stale backend forever -- ``set_default`` had no way to invalidate an
    already-compiled ``jit`` cache entry. Post-fix, ``set_default`` calls
    ``jax.clear_caches()`` on a genuine change, forcing recompilation (and
    therefore backend re-resolution) on the next call.
    """
    prim = XLACustomKernel('f1_jit_probe')
    prim.def_kernel('backend_a', 'cpu', _offset_kernel_generator(1.0))
    prim.def_kernel('backend_b', 'cpu', _offset_kernel_generator(2.0))

    outs = [jax.ShapeDtypeStruct((4,), jnp.float32)]

    @jax.jit
    def f(x):
        return prim(x, outs=outs)[0]

    x = jnp.arange(4, dtype=jnp.float32)  # explicit dtype: robust under x64 leakage
    try:
        # Warm/compile with the initial default ('backend_a').
        np.testing.assert_allclose(np.asarray(jax.block_until_ready(f(x))), np.arange(4.0) + 1.0)

        # Switch the default *after* the jit cache is warm.
        prim.set_default('cpu', 'backend_b')

        # The warm jitted function must recompile and reflect the switch.
        np.testing.assert_allclose(np.asarray(jax.block_until_ready(f(x))), np.arange(4.0) + 2.0)
    finally:
        clear_backends()


def test_f1_second_primitive_already_compiled_sees_global_switch():
    """A global ``set_backend`` invalidates *every* primitive's jit cache.

    ``jax.clear_caches()`` is process-global, not per-primitive. This test
    warms a jitted call site for one primitive, then flips the global
    backend via a *different, unrelated* primitive's registration, and
    checks that the first (already-compiled) call site still picks up the
    global switch on its next call -- proving the invalidation is not
    accidentally scoped to only the primitive that triggered it.
    """
    prim_a = XLACustomKernel('f1_second_prim_a')
    prim_a.def_kernel('backend_a', 'cpu', _offset_kernel_generator(1.0))
    prim_a.def_kernel('backend_b', 'cpu', _offset_kernel_generator(2.0))

    prim_b = XLACustomKernel('f1_second_prim_b')
    prim_b.def_kernel('backend_b', 'cpu', _offset_kernel_generator(2.0))

    outs = [jax.ShapeDtypeStruct((4,), jnp.float32)]

    @jax.jit
    def f(x):
        return prim_a(x, outs=outs)[0]

    x = jnp.arange(4, dtype=jnp.float32)  # explicit dtype: robust under x64 leakage
    try:
        # Warm-compile prim_a's call site with its default ('backend_a').
        np.testing.assert_allclose(np.asarray(jax.block_until_ready(f(x))), np.arange(4.0) + 1.0)

        # Flip the *global* cpu backend -- unrelated to prim_a's own default.
        set_backend('cpu', 'backend_b')

        # prim_a's already-compiled call site must now resolve to the global
        # backend on its next call, even though the switch happened through
        # the global config rather than through prim_a itself.
        np.testing.assert_allclose(np.asarray(jax.block_until_ready(f(x))), np.arange(4.0) + 2.0)
    finally:
        clear_backends()


# ---------------------------------------------------------------------------
# F6 -- an unregistered platform must raise a friendly, actionable error
# instead of JAX's raw "no lowering rule" failure (D2).
# ---------------------------------------------------------------------------


def test_f6_stub_lowering_raises_friendly_error_for_unregistered_platform():
    """A cpu-only primitive lowered for 'tpu' raises ``KernelFallbackExhaustedError``.

    Pre-fix, a platform with no ``def_kernel`` call had *no* lowering
    registered at all, so JAX raised its own generic "MLIR translation rule
    ... not found for primitive" error. Post-fix, ``__init__`` eagerly
    registers a stub lowering for every platform that names the primitive,
    the missing platform, and the platform(s) that *do* have a kernel.

    Uses ``jax.export.export(..., platforms=['tpu'])`` -- a public,
    jax>=0.8-safe API -- to trigger the 'tpu' lowering path without
    requiring a physical TPU device.
    """
    prim = XLACustomKernel('f6_cpu_only_probe')
    prim.def_kernel('dummy', 'cpu', _offset_kernel_generator(1.0))

    def f(x):
        return prim(x, outs=[jax.ShapeDtypeStruct(x.shape, x.dtype)])[0]

    xspec = jax.ShapeDtypeStruct((4,), jnp.float32)
    with pytest.raises(KernelFallbackExhaustedError) as excinfo:
        jax.export.export(jax.jit(f), platforms=['tpu'])(xspec)

    msg = str(excinfo.value)
    assert 'f6_cpu_only_probe' in msg
    assert 'tpu' in msg
    assert 'cpu' in msg


# ---------------------------------------------------------------------------
# F16 -- registering two primitives under the same name warns (D7).
# ---------------------------------------------------------------------------


def test_f16_duplicate_primitive_name_warns():
    """A second ``XLACustomKernel`` reusing an existing name emits ``UserWarning``.

    Pre-fix, ``_registry.register_primitive`` silently overwrote the
    previous entry (finding #16). Post-fix, the overwrite still happens
    (existing callers that reload modules depend on it), but a
    ``UserWarning`` naming the collision is now emitted.
    """
    name = 'f16_dup_probe'
    XLACustomKernel(name)
    with pytest.warns(UserWarning, match='already registered'):
        XLACustomKernel(name)


# ---------------------------------------------------------------------------
# F17 -- a failing kernel_generator/kernel is wrapped in a chained,
# actionable error instead of propagating a raw exception (D3).
# ---------------------------------------------------------------------------


def test_f17_kernel_generator_failure_is_wrapped_with_alternatives():
    """A raising ``kernel_generator`` becomes a chained ``KernelCompilationError``.

    Pre-fix, an exception raised while constructing/tracing a kernel
    propagated as-is, with no mention of alternative backends (finding
    #17, and the docstring claim that alternatives are listed was
    unreachable). Post-fix, the original exception is preserved via
    ``raise ... from exc`` and the message lists the other backend(s)
    available for the platform.
    """
    prim = XLACustomKernel('f17_probe')
    prim.def_kernel('bad', 'cpu', lambda **kwargs: (_ for _ in ()).throw(RuntimeError('boom')))
    prim.def_kernel('good', 'cpu', _offset_kernel_generator(1.0))

    x = jnp.arange(4.0)
    with pytest.raises(KernelCompilationError) as excinfo:
        jax.block_until_ready(
            prim(x, outs=[jax.ShapeDtypeStruct((4,), jnp.float32)], backend='bad')
        )

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    msg = str(excinfo.value)
    assert 'boom' in msg
    assert 'good' in msg  # alternative backend surfaced


# ---------------------------------------------------------------------------
# F19 -- assert-based validation converted to TypeError/ValueError; extra
# kwargs are pre-validated for hashability with a friendly message (D7).
# ---------------------------------------------------------------------------


def test_f19_validation_raises_typed_errors_not_asserts():
    """``def_kernel``/``def_pallas_kernel``/``__call__`` raise typed errors.

    Pre-fix these were bare ``assert`` statements (finding #19): silently
    disabled under ``python -O``, and raising the unhelpfully generic
    ``AssertionError``. Post-fix each check raises a specific
    ``TypeError``/``ValueError`` unconditionally.
    """
    prim = XLACustomKernel('f19_probe')

    with pytest.raises(TypeError, match='backend'):
        prim.def_kernel(123, 'cpu', _offset_kernel_generator(1.0))
    with pytest.raises(TypeError, match='platform'):
        prim.def_kernel('numba', 123, _offset_kernel_generator(1.0))
    with pytest.raises(TypeError, match='callable'):
        prim.def_kernel('numba', 'cpu', 'not-a-callable')

    with pytest.raises(ValueError, match='gpu'):
        prim.def_pallas_kernel('cpu', _offset_kernel_generator(1.0))

    prim.def_kernel('numba', 'cpu', _offset_kernel_generator(1.0))
    x = jnp.arange(4.0)
    with pytest.raises(TypeError, match="'some_list'") as excinfo:
        prim(x, outs=[jax.ShapeDtypeStruct((4,), jnp.float32)], some_list=[1, 2, 3])
    assert 'hashable' in str(excinfo.value)
