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
* #216 - ``--xla_gpu_deterministic_ops`` in ``XLA_FLAGS`` makes primitives without
  an explicit backend choice lower with ``jax_raw`` on GPU, so the event products
  give the same bits on every call.
* #245 - under ``jax.shard_map``, outputs keep the varying manual axes (``vma``) of
  the operands, and operands are made varying over the union first, so loop carries
  type-check and reverse mode sums the cotangent of a replicated operand.
"""

import functools
import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.interpreters import xla
from jax.sharding import NamedSharding, PartitionSpec as P

import brainevent
from brainevent import KernelCompilationError, KernelFallbackExhaustedError
from brainevent._compatible_import import apply_primitive, varying_axes
from brainevent._csr.binary import binary_csrmm_p, binary_csrmv_p
from brainevent._csr.delay_layout import csr_delay_deliver_layout_p
from brainevent._csr.dt2t import csrmm_dt2t_p, csrmv_dt2t_p
from brainevent._csr.float import csrmm_p, csrmv_p
from brainevent._op.main import DETERMINISTIC_GPU_BACKEND, XLACustomKernel, xla_gpu_deterministic_ops
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


def _exception_chain_contains(exc, expected_type, expected_text: str) -> bool:
    while exc is not None:
        if isinstance(exc, expected_type) and expected_text in str(exc):
            return True
        exc = getattr(exc, '__cause__', None)
    return False


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
    # Committed to CPU: only 'cpu' kernels are registered above, so on a machine
    # with a GPU the default placement would lower for 'gpu' and exhaust fallbacks.
    x = jax.device_put(jnp.arange(4, dtype=jnp.float32), jax.devices('cpu')[0])
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

    # Explicit dtype: robust under x64 leakage. Committed to CPU so the jitted
    # call lowers for 'cpu' (the only platform with kernels) on a GPU machine.
    x = jax.device_put(jnp.arange(4, dtype=jnp.float32), jax.devices('cpu')[0])
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

    # Explicit dtype: robust under x64 leakage. Committed to CPU so the jitted
    # call lowers for 'cpu' (the only platform with kernels) on a GPU machine.
    x = jax.device_put(jnp.arange(4, dtype=jnp.float32), jax.devices('cpu')[0])
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

    # Committed to CPU: both kernels are registered for 'cpu' only, so on a
    # machine with a GPU the default placement would lower for 'gpu' and raise
    # KernelFallbackExhaustedError before the generator is ever invoked.
    x = jax.device_put(jnp.arange(4, dtype=jnp.float32), jax.devices('cpu')[0])
    with pytest.raises(KernelCompilationError) as excinfo:
        jax.block_until_ready(
            prim(x, outs=[jax.ShapeDtypeStruct((4,), jnp.float32)], backend='bad')
        )

    assert _exception_chain_contains(excinfo.value, RuntimeError, 'boom')
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


# ---------------------------------------------------------------------------
# #216 -- ``--xla_gpu_deterministic_ops`` reaches the backend selection: the
# atomic ``cuda_raw`` kernels ignore it, so without an explicit choice the GPU
# lowers with ``jax_raw``, whose XLA scatters and reductions follow it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('flags, expected', [
    ('', False),
    ('--xla_gpu_deterministic_ops', True),
    ('--xla_gpu_deterministic_ops=true', True),
    ('--xla_gpu_deterministic_ops=TRUE', True),
    ('--xla_gpu_deterministic_ops=1', True),
    ('--xla_gpu_deterministic_ops=false', False),
    ('--xla_gpu_deterministic_ops=0', False),
    ('--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true', True),
    ('--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=false', False),
    ('--xla_gpu_deterministic_ops_other=true', False),
    ('--noxla_gpu_deterministic_ops', False),
])
def test_216_xla_gpu_deterministic_ops_flag(monkeypatch, flags, expected):
    """The flag is read from ``XLA_FLAGS`` as XLA reads it: the last occurrence wins."""
    monkeypatch.setenv('XLA_FLAGS', flags)
    assert xla_gpu_deterministic_ops() is expected


def test_216_xla_gpu_deterministic_ops_flag_unset(monkeypatch):
    monkeypatch.delenv('XLA_FLAGS', raising=False)
    assert xla_gpu_deterministic_ops() is False


def _gpu_probe(name, *gpu_backends):
    prim = XLACustomKernel(name)
    for i, b in enumerate(gpu_backends):
        prim.def_kernel(b, 'gpu', _offset_kernel_generator(float(i)))
    prim.def_kernel('backend_a', 'cpu', _offset_kernel_generator(1.0))
    prim.def_kernel(DETERMINISTIC_GPU_BACKEND, 'cpu', _offset_kernel_generator(2.0))
    return prim


def test_216_deterministic_flag_selects_jax_raw_on_gpu_only(monkeypatch):
    prim = _gpu_probe('p216_selection_probe', 'cuda_raw', DETERMINISTIC_GPU_BACKEND)
    monkeypatch.delenv('XLA_FLAGS', raising=False)
    assert prim._resolve_backend('gpu') == 'cuda_raw'  # the first-registered default
    monkeypatch.setenv('XLA_FLAGS', '--xla_gpu_deterministic_ops=true')
    assert prim._resolve_backend('gpu') == DETERMINISTIC_GPU_BACKEND
    assert prim._resolve_backend('cpu') == 'backend_a'  # a GPU flag: CPU is unchanged
    monkeypatch.setenv('XLA_FLAGS', '--xla_gpu_deterministic_ops=false')
    assert prim._resolve_backend('gpu') == 'cuda_raw'


def test_216_explicit_choices_win_over_the_flag(monkeypatch):
    prim = _gpu_probe('p216_explicit_probe', 'cuda_raw', DETERMINISTIC_GPU_BACKEND)
    monkeypatch.setenv('XLA_FLAGS', '--xla_gpu_deterministic_ops')
    assert prim._resolve_backend('gpu', 'cuda_raw') == 'cuda_raw'  # per call
    try:
        set_backend('gpu', 'cuda_raw')  # global
        assert prim._resolve_backend('gpu') == 'cuda_raw'
    finally:
        clear_backends()
    assert prim._resolve_backend('gpu') == DETERMINISTIC_GPU_BACKEND


def test_216_flag_without_a_jax_raw_kernel_warns_once(monkeypatch):
    prim = _gpu_probe('p216_no_jax_raw_probe', 'cuda_raw', 'cublas')
    monkeypatch.setenv('XLA_FLAGS', '--xla_gpu_deterministic_ops=true')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert prim._resolve_backend('gpu') == 'cuda_raw'
        assert prim._resolve_backend('gpu') == 'cuda_raw'
    text = [str(w.message) for w in caught if 'deterministic' in str(w.message)]
    assert len(text) == 1 and 'p216_no_jax_raw_probe' in text[0], text


def test_216_resolution_errors_are_unchanged():
    prim = XLACustomKernel('p216_errors_probe')
    with pytest.raises(KernelFallbackExhaustedError, match='No kernels registered'):
        prim._resolve_backend('gpu')
    prim.def_kernel('cuda_raw', 'gpu', _offset_kernel_generator(0.0))
    with pytest.raises(ValueError, match='empty string'):
        prim._resolve_backend('gpu', '')
    with pytest.raises(KernelFallbackExhaustedError, match='warp not available'):
        prim._resolve_backend('gpu', 'warp')


_REPRODUCIBILITY_PROBE = r'''
import jax, jax.numpy as jnp, numpy as np
import brainevent
rng = np.random.default_rng(0)
n_pre, n_post = 1000, 500
spikes = jnp.asarray(rng.random((4, n_pre)) < 0.3)
W = jnp.asarray(rng.uniform(0.1, 1.0, (n_pre, n_post)).astype(np.float32))
mask = rng.random((n_pre, n_post)) < 0.1
cols = np.nonzero(mask)[1].astype(np.int32)
indptr = np.concatenate([[0], np.cumsum(mask.sum(1))]).astype(np.int32)
data = jnp.asarray(rng.uniform(0.1, 1.0, cols.size).astype(np.float32))
csr = brainevent.CSR((data, jnp.asarray(cols), jnp.asarray(indptr)), shape=(n_pre, n_post))
for name, f in (('dense', jax.jit(lambda s: brainevent.BinaryArray(s) @ W)),
                ('csr', jax.jit(lambda s: brainevent.BinaryArray(s[0]) @ csr))):
    outs = [np.asarray(f(spikes)) for _ in range(20)]
    print(name, all(np.array_equal(outs[0], o) for o in outs[1:]))
'''


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='needs a GPU backend')
def test_216_event_kernels_are_reproducible_under_the_flag():
    """Binary dense and CSR products give the same bits on every call (fresh process,
    because XLA reads ``XLA_FLAGS`` when its backend starts)."""
    env = {**os.environ, 'XLA_FLAGS': '--xla_gpu_deterministic_ops=true'}
    r = subprocess.run([sys.executable, '-c', _REPRODUCIBILITY_PROBE], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ['dense', 'True', 'csr', 'True'], r.stdout


# ---------------------------------------------------------------------------
# #245: under shard_map, outputs keep the operands' varying manual axes (vma)
# ---------------------------------------------------------------------------
#
# Tests named ``test_245_sharded_*`` shard over ``_N`` devices: two where this
# process has two, else one. With one device the types are still checked (a size-1
# mesh axis is tracked), and ``test_245_rerun_on_two_cpu_devices`` repeats them in a
# fresh process with two CPU devices, where a cross-shard gradient leak shows.

_AXIS = 'shard'
_N = min(2, jax.device_count())
_PLATFORM = jax.default_backend()

# A 4 x 4 CSR matrix with 5 stored entries, and per-edge delays for a ring of 5 slots.
_INDICES = np.array([0, 1, 2, 0, 3], np.int32)
_INDPTR = np.array([0, 2, 3, 5, 5], np.int32)
_ROWS = np.repeat(np.arange(4), np.diff(_INDPTR)).astype(np.int32)
_DELAYS = np.array([1, 3, 2, 1, 4], np.int32)
_N_SLOTS = 5


@functools.cache
def _layout():
    with jax.ensure_compile_time_eval():  # concrete arrays, even if first called while tracing
        return brainevent.CSRDelayLayout.build(
            np.ones(5, np.float32), _INDICES, _INDPTR, _DELAYS, n_slots=_N_SLOTS, shape=(4, 4))


def _csrmv(w, x, backend=None):
    """``x @ A`` with ``A = CSR(w)``."""
    return brainevent.csrmv(w, jnp.asarray(_INDICES), jnp.asarray(_INDPTR), x, shape=(4, 4),
                            transpose=True, backend=backend)


def _csrmv_ref(w, x):
    return jax.ops.segment_sum(x[_ROWS] * w, jnp.asarray(_INDICES), num_segments=4)


def _binary_csrmv(w, x, backend=None):
    """``x @ A`` for binary events ``x != 0``."""
    return brainevent.binary_csrmv(w, jnp.asarray(_INDICES), jnp.asarray(_INDPTR), x, shape=(4, 4),
                                   transpose=True, backend=backend)


def _binary_csrmv_ref(w, x):
    return jax.ops.segment_sum((x[_ROWS] != 0).astype(w.dtype) * w, jnp.asarray(_INDICES), num_segments=4)


def _deliver(ring, x, w, t, backend=None, weighted=True):
    """The fused per-edge-delay delivery of ``x`` into ``ring``."""
    return brainevent.csr_delay_deliver_layout(ring, x, _layout().with_data(w), t, weighted=weighted,
                                               backend=backend)


def _deliver_ref(ring, x, w, t, weighted=True):
    f = x if weighted else (x != 0).astype(ring.dtype)
    return ring.at[(t + _DELAYS) % _N_SLOTS, _INDICES].add(w * f[_ROWS])


def _backends(prim):
    return prim.available_backends(_PLATFORM)


def _mesh():
    return jax.sharding.Mesh(np.asarray(jax.devices()[:_N]), (_AXIS,))


def _put(x, spec):
    return jax.device_put(jnp.asarray(x), NamedSharding(_mesh(), spec))


def _shard_data():
    """Per-shard weights ``w (N, 5)``, events ``x (N, 4)``, rings ``(N, L, 4)`` and
    cotangent weights ``v``, all different between shards."""
    w = np.arange(5 * _N, dtype=np.float32).reshape(_N, 5) / 4 + 1
    x = np.asarray([[1., 0., 3., 4.], [5., 6., 0., 8.]], np.float32)[:_N]
    ring = np.arange(_N * _N_SLOTS * 4, dtype=np.float32).reshape(_N, _N_SLOTS, 4) / 8
    v4 = np.asarray([[1., 10., 100., 1000.], [1e4, 1e5, 1e6, 1e7]], np.float32)[:_N] / 1e3
    v_ring = np.arange(_N * _N_SLOTS * 4, dtype=np.float32).reshape(_N, _N_SLOTS, 4) % 7 + 1
    return w, x, ring, v4, v_ring


_TYPE_OPS = {
    'csrmv': lambda w, x, ring: _csrmv(w, x),
    'binary_csrmv': lambda w, x, ring: _binary_csrmv(w, x),
    'deliver': lambda w, x, ring: _deliver(ring, x, w, jnp.int32(2)),
}


@pytest.mark.parametrize('op', sorted(_TYPE_OPS))
@pytest.mark.parametrize('case', ['all-varying', 'replicated-weights', 'all-replicated'])
def test_245_sharded_output_types_follow_the_operands(op, case):
    w, x, ring, _, _ = _shard_data()
    w_spec = P(_AXIS) if case == 'all-varying' else P()
    xr_spec = P() if case == 'all-replicated' else P(_AXIS)
    seen = []

    def local(w, x, ring):
        w = w[0] if w_spec == P(_AXIS) else w
        x, ring = (x, ring) if xr_spec == P() else (x[0], ring[0])
        out = _TYPE_OPS[op](w, x, ring)
        seen.append((varying_axes(jax.typeof(ring)), varying_axes(jax.typeof(out))))
        return out[None]

    f = jax.shard_map(local, mesh=_mesh(), in_specs=(w_spec, xr_spec, xr_spec), out_specs=P(_AXIS))
    jax.make_jaxpr(f)(w if w_spec == P(_AXIS) else w[0], x if xr_spec == P(_AXIS) else x[0],
                      ring if xr_spec == P(_AXIS) else ring[0])
    (in_vma, out_vma), = seen
    expected = frozenset() if case == 'all-replicated' else frozenset({_AXIS})
    assert in_vma == expected, f'precondition: shard_map must track the vma; got {in_vma}'
    assert out_vma == expected


@pytest.mark.parametrize('backend', _backends(csr_delay_deliver_layout_p))
@pytest.mark.parametrize('weighted', [True, False])
def test_245_sharded_delay_delivery_carries_the_ring_in_fori_loop(backend, weighted):
    w, x, ring, _, _ = _shard_data()

    def run(deliver):
        def local(ring, x, w):
            body = lambda t, r: deliver(r, x[0], w[0], t, weighted=weighted)
            return jax.lax.fori_loop(0, 3, body, ring[0])[None]

        f = jax.shard_map(local, mesh=_mesh(), in_specs=(P(_AXIS),) * 3, out_specs=P(_AXIS))
        return np.asarray(jax.jit(f)(_put(ring, P(_AXIS)), _put(x, P(_AXIS)), _put(w, P(_AXIS))))

    got = run(functools.partial(_deliver, backend=backend))
    np.testing.assert_allclose(got, run(_deliver_ref), rtol=1e-6)


@pytest.mark.parametrize('backend', _backends(csrmv_p))
def test_245_sharded_csr_product_carries_the_vector_in_fori_loop(backend):
    w, x, _, _, _ = _shard_data()

    def run(mv):
        def local(w, x):
            return jax.lax.fori_loop(0, 3, lambda i, c: mv(w[0], c) / 4, x[0])[None]

        f = jax.shard_map(local, mesh=_mesh(), in_specs=(P(_AXIS),) * 2, out_specs=P(_AXIS))
        return np.asarray(jax.jit(f)(_put(w, P(_AXIS)), _put(x, P(_AXIS))))

    np.testing.assert_allclose(run(functools.partial(_csrmv, backend=backend)), run(_csrmv_ref), rtol=1e-6)


def _sharded_loss(mv, w_spec):
    """``sum_s sum(v_s * mv(w_s, x_s))``; ``w`` is per shard or replicated."""

    def local(w, x, v):
        return jax.lax.psum(jnp.sum(v[0] * mv(w[0] if w_spec == P(_AXIS) else w, x[0])), _AXIS)

    return jax.shard_map(local, mesh=_mesh(), in_specs=(w_spec, P(_AXIS), P(_AXIS)), out_specs=P())


def _assert_grads_match(mv, mv_ref, w_spec, argnums):
    w, x, _, v4, _ = _shard_data()
    w = w if w_spec == P(_AXIS) else w[0]
    args = (_put(w, w_spec), _put(x, P(_AXIS)), _put(v4, P(_AXIS)))
    loss, loss_ref = _sharded_loss(mv, w_spec), _sharded_loss(mv_ref, w_spec)
    np.testing.assert_allclose(jax.jit(loss)(*args), jax.jit(loss_ref)(*args), rtol=1e-6)
    got = jax.jit(jax.grad(loss, argnums=argnums))(*args)
    want = jax.jit(jax.grad(loss_ref, argnums=argnums))(*args)
    for g, r in zip(got, want):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-6)
    # forward mode, along the weights
    tangent = jnp.ones_like(args[0])
    _, jvp = jax.jit(lambda *a: jax.jvp(lambda w: loss(w, *a[1:]), (a[0],), (tangent,)))(*args)
    _, jvp_ref = jax.jit(lambda *a: jax.jvp(lambda w: loss_ref(w, *a[1:]), (a[0],), (tangent,)))(*args)
    np.testing.assert_allclose(np.asarray(jvp), np.asarray(jvp_ref), rtol=1e-6)


_W_SPECS = pytest.mark.parametrize('w_spec', [P(_AXIS), P()], ids=['per-shard-w', 'replicated-w'])


@_W_SPECS
@pytest.mark.parametrize('backend', _backends(csrmv_p))
def test_245_sharded_csr_product_gradients_match_pure_jax(backend, w_spec):
    _assert_grads_match(functools.partial(_csrmv, backend=backend), _csrmv_ref, w_spec, (0, 1))


@_W_SPECS
@pytest.mark.parametrize('backend', _backends(binary_csrmv_p))
def test_245_sharded_binary_csr_product_gradients_match_pure_jax(backend, w_spec):
    _assert_grads_match(functools.partial(_binary_csrmv, backend=backend), _binary_csrmv_ref, w_spec, (0,))


@_W_SPECS
@pytest.mark.parametrize('backend', _backends(csr_delay_deliver_layout_p))
def test_245_sharded_delay_delivery_gradients_match_pure_jax(backend, w_spec):
    w, x, ring, _, v_ring = _shard_data()
    w = w if w_spec == P(_AXIS) else w[0]
    args = (_put(w, w_spec), _put(x, P(_AXIS)), _put(ring, P(_AXIS)), _put(v_ring, P(_AXIS)))

    def loss(deliver):
        def local(w, x, ring, v):
            out = deliver(ring[0], x[0], w[0] if w_spec == P(_AXIS) else w, jnp.int32(3))
            return jax.lax.psum(jnp.sum(v[0] * out), _AXIS)

        return jax.shard_map(local, mesh=_mesh(), in_specs=(w_spec,) + (P(_AXIS),) * 3, out_specs=P())

    got = jax.jit(jax.grad(loss(functools.partial(_deliver, backend=backend)), argnums=(0, 1, 2)))(*args)
    want = jax.jit(jax.grad(loss(_deliver_ref), argnums=(0, 1, 2)))(*args)
    for g, r in zip(got, want):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-6)


# Transpose rules that build a cotangent outside the primitive itself: the dt2t
# ``y`` cotangent (``jax.linear_transpose`` of the pure-JAX kernel) and the CSR
# mat-mat weight cotangent (SDDMM through jax's ``bcoo_dot_general_sampled``).
_B = np.asarray([[1., 2.], [0., 3.], [4., 0.], [5., 6.]], np.float32)  # (4, 2) events / dense block

_GRAD_CASES = {
    # name: (primitive, per-shard op, argnums of the differentiable operands)
    'csrmm': (csrmm_p, lambda be: lambda w, B: brainevent.csrmm(
        w, jnp.asarray(_INDICES), jnp.asarray(_INDPTR), B, shape=(4, 4), transpose=True, backend=be), (0, 1)),
    'csrmm-homo': (csrmm_p, lambda be: lambda w, B: brainevent.csrmm(
        w[:1], jnp.asarray(_INDICES), jnp.asarray(_INDPTR), B, shape=(4, 4), transpose=True, backend=be), (0, 1)),
    'binary_csrmm': (binary_csrmm_p, lambda be: lambda w, B: brainevent.binary_csrmm(
        w, jnp.asarray(_INDICES), jnp.asarray(_INDPTR), B, shape=(4, 4), transpose=True, backend=be), (0,)),
    'csrmv_dt2t': (csrmv_dt2t_p, lambda be: lambda w, B: brainevent.csrmv_dt2t(
        B[:, 0], w, jnp.asarray(_INDICES), jnp.asarray(_INDPTR), shape=(4, 4), backend=be), (0, 1)),
    'csrmm_dt2t': (csrmm_dt2t_p, lambda be: lambda w, B: brainevent.csrmm_dt2t(
        B.T, jnp.stack([w, 2 * w]), jnp.asarray(_INDICES), jnp.asarray(_INDPTR), shape=(4, 4), backend=be), (0, 1)),
}


@pytest.mark.parametrize('case,backend', [
    (case, be) for case, (prim, _, _) in sorted(_GRAD_CASES.items()) for be in _backends(prim)])
def test_245_sharded_transpose_rules_keep_gradients_per_shard(case, backend):
    """Per-shard gradients under shard_map equal the gradients of each shard's
    computation without shard_map."""
    _, make, argnums = _GRAD_CASES[case]
    op = make(backend)
    w, _, _, _, _ = _shard_data()
    B = np.stack([_B * (s + 1) for s in range(_N)])
    out_shape = jax.eval_shape(op, w[0], B[0]).shape
    v = np.arange(_N * int(np.prod(out_shape)), dtype=np.float32).reshape((_N,) + out_shape) % 5 + 1

    def local(w, B, v):
        return jax.lax.psum(jnp.sum(v[0] * op(w[0], B[0])), _AXIS)

    f = jax.shard_map(local, mesh=_mesh(), in_specs=(P(_AXIS),) * 3, out_specs=P())
    args = (_put(w, P(_AXIS)), _put(B, P(_AXIS)), _put(v, P(_AXIS)))
    shard_loss = lambda w, B, v: jnp.sum(v * op(w, B))
    np.testing.assert_allclose(jax.jit(f)(*args), sum(jax.jit(shard_loss)(w[s], B[s], v[s]) for s in range(_N)),
                               rtol=1e-6)
    got = jax.jit(jax.grad(f, argnums=argnums))(*args)
    per_shard = [jax.jit(jax.grad(shard_loss, argnums=argnums))(w[s], B[s], v[s]) for s in range(_N)]
    for k in range(len(argnums)):
        np.testing.assert_allclose(np.asarray(got[k]), np.stack([np.asarray(g[k]) for g in per_shard]), rtol=1e-6)


def _add_primitive(name):
    """A two-operand ``x + y`` primitive with a plain-JAX kernel on every platform."""
    prim = XLACustomKernel(name)

    def gen(**kwargs):
        return lambda x, y: (x + y,)

    for platform in ('cpu', 'gpu', 'tpu'):
        prim.def_kernel('jax_raw', platform, gen)
    return prim


def test_245_sharded_operands_are_made_varying_before_bind():
    prim = _add_primitive('p245_mixed_operands_probe')
    outs = jax.ShapeDtypeStruct((3,), jnp.float32)

    def local(x, y):
        return prim(x[0], y, outs=outs)[None]

    f = jax.shard_map(local, mesh=_mesh(), in_specs=(P(_AXIS), P()), out_specs=P(_AXIS))
    jaxpr = jax.make_jaxpr(f)(jnp.ones((_N, 3), jnp.float32), jnp.ones((3,), jnp.float32))
    body = jaxpr.eqns[0].params['jaxpr']
    eqn, = [e for e in body.eqns if e.primitive is prim.primitive]
    assert [varying_axes(v.aval) for v in eqn.invars] == [frozenset({_AXIS})] * 2
    assert [varying_axes(v.aval) for v in eqn.outvars] == [frozenset({_AXIS})]


def test_245_unsharded_call_is_unchanged():
    prim = _add_primitive('p245_unsharded_probe')
    outs = jax.ShapeDtypeStruct((3,), jnp.float32)
    x, y = jnp.ones((3,), jnp.float32), jnp.arange(3, dtype=jnp.float32)

    jaxpr = jax.make_jaxpr(lambda x, y: prim(x, y, outs=outs))(x, y)
    assert [e.primitive for e in jaxpr.eqns] == [prim.primitive]
    assert jaxpr.out_avals == [jax.core.ShapedArray((3,), jnp.float32)]
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda x, y: prim(x, y, outs=outs))(x, y)), [1., 2., 3.])

    # shard_map with check_vma=False tracks no vma: no casts, plain output types
    def local(x, y):
        out = prim(x[0], y, outs=outs)
        assert varying_axes(jax.typeof(out)) == frozenset()
        return out[None]

    f = jax.shard_map(local, mesh=_mesh(), in_specs=(P(_AXIS), P()), out_specs=P(_AXIS), check_vma=False)
    body = jax.make_jaxpr(f)(jnp.ones((_N, 3), jnp.float32), y).eqns[0].params['jaxpr']
    assert not [e for e in body.eqns if 'pvary' in e.primitive.name or 'pcast' in e.primitive.name]


@pytest.mark.skipif(_N >= 2, reason='this process already runs the sharded tests on two devices')
@pytest.mark.parametrize('backend', ['jax_raw', 'numba'])
def test_245_rerun_on_two_cpu_devices(backend):
    """Rerun the ``test_245_sharded_*`` tests on two CPU devices in a fresh process
    (XLA reads ``XLA_FLAGS`` when its backend starts); the per-shard gradients then
    differ, so a gradient summed over shards fails them."""
    if backend == 'numba' and not numba_installed:
        pytest.skip('numba is not installed')
    flags = f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=2".strip()
    env = {**os.environ, 'XLA_FLAGS': flags, 'JAX_PLATFORMS': 'cpu'}
    select = 'test_245_sharded and ' + ('numba' if backend == 'numba' else 'not numba')
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run(
        [sys.executable, '-m', 'pytest', str(Path(__file__).resolve()), '-q', '-m', '', '-p', 'no:cacheprovider',
         '-k', select],
        env=env, cwd=root, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert ' passed' in r.stdout and ' skipped' not in r.stdout, r.stdout[-2000:]
