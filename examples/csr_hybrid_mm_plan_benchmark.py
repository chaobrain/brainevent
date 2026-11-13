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

"""Shape-only plan for the CSR hybrid scatter MM (docs/specs/2026-11-14-csr-mm-shape-plan.md).

Measures the transposed binary event product ``W.T @ events`` (events
``(rows, batch)``) with the native hybrid batched SpMM. Variants:

- ``P``: today's prepared path. The matrix is built eagerly on the GPU, so its
  ``CSRConfig`` carries host row statistics, a long-row mask and GPU properties.
- ``J``: ``backend='jax_raw'`` (the portable JAX scatter).
- ``T``: the matrix is built inside ``jax.jit`` from traced arrays and used as
  ``BinaryArray(events.T) @ M``, as brainspike does. Today this takes ``J``'s route.
- ``S``: prototype shape-only plan calling the same native kernel. The lane
  group, task-queue capacity and batch chunk come from static shapes only;
  the long-row mask is computed in JAX on every call.
- ``S0``: ``S`` with the long-row mask precomputed outside the timed loop.

Each timing compiles one straight-line program of eight products, each on
its own pre-generated event matrix (so XLA cannot merge them), enqueues enough
programs for about 50 ms, and reports the median and p10/p90 time per product.
Structure-only work such as S's mask is shared by the eight products, as it is
hoisted from a simulation loop over fixed connectivity; ``S0`` and the mask
record bound it. Other records: compile
counts over redrawn structures, which kernel each construction path lowers
to, mask cost, scratch sizes and the matrix SDDMM gradient scratch.

Run on one GPU, then summarize (``report`` also compares two runs)::

    CUDA_VISIBLE_DEVICES=0 python examples/csr_hybrid_mm_plan_benchmark.py run --out base.jsonl
    CUDA_VISIBLE_DEVICES=0 python examples/csr_hybrid_mm_plan_benchmark.py run --out quick.jsonl --quick
    python examples/csr_hybrid_mm_plan_benchmark.py report base.jsonl [new.jsonl]
"""

import argparse
import json
import math
import re
import statistics
import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)  # float64 spot cases; every array below has an explicit dtype

import brainevent  # noqa: E402
from brainevent import BinaryArray, PerformanceWarning, binary_csrmm  # noqa: E402

# Private hooks of the native kernel, used only by the S/S0 prototype.
from brainevent._csr._support.binary import _accumulator_info  # noqa: E402
from brainevent._csr.binary.binary import _load_hybrid_mm  # noqa: E402
from brainevent._op.util import cuda_event_abi, dtype_suffix  # noqa: E402
from brainevent._sddmm._mm.sddmm import _cuda_plan as _sddmm_cuda_plan  # noqa: E402

TASK_EDGES = 512  # edges per long-row task (native kTaskEdges)
ITERATIONS = 16  # long_threshold = ITERATIONS * group
L2_DIVISOR = 4
MAX_CHUNK = 65535

SIZES = {  # name: (rows, cols, mean row length)
    'S': (4_000, 4_000, 80),
    'M': (20_000, 20_000, 200),
    'L': (100_000, 100_000, 100),
    'W': (10_000, 200_000, 1000),
}
DISTS = ('fixed', 'indegree', 'lognormal', 'pareto', 'hub', 'tail')
BATCHES = (1, 8, 64, 256)
ACTIVITIES = (0.01, 0.05, 0.2, 1.0)
MODES = ('binary', 'weighted')
QUICK = dict(sizes=('S', 'M'), dists=('indegree', 'pareto'), batches=(8, 64),
             activities=(0.05, 1.0), modes=('binary',))


# ---------------------------------------------------------------------------
# Structures and events
# ---------------------------------------------------------------------------

def row_lengths(dist, rows, cols, mean, rng):
    """Row lengths of one distribution; all but ``tail`` keep ``rows * mean`` edges."""
    nnz = rows * mean
    if dist == 'fixed':
        lengths = np.full(rows, mean)
    elif dist == 'indegree':  # fixed in-degree with random sources: multinomial row lengths
        lengths = np.bincount(rng.integers(0, rows, nnz), minlength=rows)
    elif dist == 'tail':  # small mean (G=4 range) plus 0.5% rows of length 2000
        lengths = rng.poisson(8, rows)
        lengths[rng.choice(rows, max(1, rows // 200), replace=False)] = min(2000, cols)
        return np.minimum(lengths, cols)
    else:
        if dist == 'lognormal':
            weights = rng.lognormal(0.0, 1.0, rows)
        elif dist == 'pareto':
            weights = rng.pareto(1.5, rows) + 1.0
        elif dist == 'hub':
            weights = np.ones(rows)
            weights[rng.choice(rows, max(1, rows // 100), replace=False)] = 50.0
        else:
            raise ValueError(dist)
        weights = np.minimum(weights / weights.sum() * nnz, cols)
        lengths = np.floor(weights * nnz / weights.sum()).astype(np.int64)
        lengths = np.minimum(lengths, cols)
    lengths = np.minimum(lengths, cols)
    short = nnz - int(lengths.sum())
    if short > 0:  # top up rows below cols, one edge each, to keep nnz
        room = np.flatnonzero(lengths < cols)
        lengths[rng.choice(room, short, replace=short > room.size)] += 1
        lengths = np.minimum(lengths, cols)
    return lengths


def build_structure(dist, rows, cols, mean, seed):
    """Random CSR ``(weights, indices, indptr, lengths)`` with sorted columns per row."""
    rng = np.random.default_rng(seed)
    lengths = row_lengths(dist, rows, cols, mean, rng).astype(np.int64)
    nnz = int(lengths.sum())
    row_ids = np.repeat(np.arange(rows, dtype=np.int64), lengths)
    columns = rng.integers(0, cols, nnz, dtype=np.int64)
    order = np.lexsort((columns, row_ids))
    indptr = np.zeros(rows + 1, np.int32)
    np.cumsum(lengths, out=indptr[1:])
    weights = rng.normal(0.1, 0.02, nnz).astype(np.float32)
    return weights, columns[order].astype(np.int32), indptr, lengths


def event_stack(rows, batch, activity, mode, k, seed):
    """``k`` distinct event matrices ``(rows, batch)``: bool, or float amplitudes."""
    rng = np.random.default_rng(seed)
    active = rng.random((k, rows, batch)) < activity
    if mode == 'binary':
        return jnp.asarray(active)
    return jnp.asarray(np.where(active, rng.uniform(0.5, 1.5, active.shape), 0.0).astype(np.float32))


# ---------------------------------------------------------------------------
# Shape-only plan prototype (S / S0)
# ---------------------------------------------------------------------------

def shape_group(rows, nnz):
    """Lane group from the mean row length only (no host row statistics)."""
    quarter = (min(nnz // max(rows, 1), TASK_EDGES) + 3) // 4
    return 4 if quarter <= 4 else 8 if quarter <= 8 else 16 if quarter <= 16 else 32


def exact_long_tasks(lengths, group):
    long = lengths[lengths > ITERATIONS * group]
    return int(((long + TASK_EDGES - 1) // TASK_EDGES).sum())


def long_tasks_for(policy, rows, nnz, group, lengths):
    if policy == 'worst':  # every row above the threshold, every event active
        return -(-nnz // TASK_EDGES) + min(rows, nnz // (ITERATIONS * group + 1))
    if policy == 'nnz512':
        return -(-nnz // TASK_EDGES)
    if policy == 'exact':
        return exact_long_tasks(lengths, group)
    if policy == 'half':
        return exact_long_tasks(lengths, group) // 2
    if policy == 'zero':
        return 0
    raise ValueError(policy)


def chunk_for(policy, batch, cols, value_bytes, l2_bytes):
    if policy == 'l2':  # today's rule: one batch group of float32 outputs fits a quarter of L2
        fit = (l2_bytes // L2_DIVISOR) // (cols * max(4, value_bytes))
        return max(1, min(fit, batch, MAX_CHUNK))
    return max(1, min(int(policy), batch))


def scratch(rows, batch, chunk, long_tasks, value_bytes, weighted):
    """``(task_capacity, workspace words, short_rows entries)`` of the native layout."""
    capacity = chunk * long_tasks
    groups = -(-batch // chunk)
    queue = ((capacity * value_bytes + 3) // 4 * 4) if weighted else 0
    workspace = ((8 * batch + 7) // 8 * 8) + 8 * groups + 12 * capacity + queue
    return capacity, workspace // 4, chunk * rows


def long_mask(indptr, threshold):
    """Native uint32 long-row bit mask, computed in JAX (as the SDDMM path does)."""
    rows = indptr.size - 1
    bits = (jnp.diff(indptr) > threshold).astype(jnp.uint32)
    bits = jnp.pad(bits, (0, -rows % 32)).reshape((-1, 32))
    return jnp.sum(bits << jnp.arange(32, dtype=jnp.uint32), axis=1, dtype=jnp.uint32)


def static_mm(weights, indices, indptr, events, mask=None, *, cols, group, long_tasks, chunk, weighted):
    """``W.T @ events`` through the existing hybrid FFI with a shape-only plan."""
    rows, batch = indptr.size - 1, events.shape[1]
    vdtype = weights.dtype
    module = _load_hybrid_mm('row', 'weighted' if weighted else 'binary')
    abi = cuda_event_abi(events.dtype, weighted=weighted, value_dtype=vdtype)
    symbol = 'batched_spmm_v3' + ('_weighted' if weighted else '') + ('_homo' if weights.size == 1 else '')
    symbol += dtype_suffix(vdtype) + ('' if weighted else abi.suffix)
    capacity, words, short = scratch(rows, batch, chunk, long_tasks, np.dtype(vdtype).itemsize, weighted)
    if mask is None:
        mask = long_mask(indptr, ITERATIONS * group)
    infos = (_accumulator_info(jax.ShapeDtypeStruct((batch, cols), vdtype)),
             jax.ShapeDtypeStruct((words,), jnp.uint32), jax.ShapeDtypeStruct((short,), jnp.int32))
    out = jax.ffi.ffi_call(f'{module}.{symbol}', infos, vmap_method='sequential')(
        weights, indices, indptr, mask, abi.prepare(events).T,
        group=group, long_threshold=ITERATIONS * group, chunk=chunk, long_tasks=long_tasks,
        task_capacity=capacity, expected_sms=0, expected_l2_bytes=0)[0]
    return out.T.astype(vdtype)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

UNROLL = 8  # products per compiled program, each on its own event matrix


def _compiled_steps(step, args, stack):
    """Compile ``sum(step(*args, stack[i]))`` over the stack as straight-line code.

    Distinct event matrices keep XLA from merging the products, and there is
    no device loop whose predicate would round-trip to the host.
    """
    @jax.jit
    def program(args, stack):
        total = jnp.float32(0)
        for i in range(stack.shape[0]):
            total = total + step(*args, stack[i])
        return total

    return program.lower(args, stack).compile()


def _samples(compiled, args, stack, runs, warmup, target_s, max_reps=256):
    """Seconds per step: enqueue enough programs for about ``target_s`` per sample."""
    for _ in range(warmup):
        compiled(args, stack).block_until_ready()
    start = time.perf_counter()
    compiled(args, stack).block_until_ready()
    once = time.perf_counter() - start
    reps = int(min(max_reps, max(1, target_s / max(once, 1e-6))))
    if once > 0.25:  # slow programs: a few samples suffice
        runs = min(runs, 3)
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(reps):
            total = compiled(args, stack)
        total.block_until_ready()
        samples.append((time.perf_counter() - start) / (reps * stack.shape[0]))
    return sorted(samples), reps


def _first(out):
    """One output element, after forcing the whole output to exist."""
    return jax.lax.optimization_barrier(out).reshape(-1)[0].astype(jnp.float32)


def time_product(product, args, stack, runs, warmup, target_s=0.05):
    """Median/p10/p90 microseconds per product, plus compiled temp bytes."""
    compiled = _compiled_steps(lambda *a: _first(product(*a)), args, stack)
    samples, reps = _samples(compiled, args, stack, runs, warmup, target_s)
    memory = compiled.memory_analysis()
    return dict(median_us=1e6 * statistics.median(samples),
                p10_us=1e6 * samples[len(samples) // 10],
                p90_us=1e6 * samples[min(len(samples) - 1, (9 * len(samples)) // 10)],
                reps=reps, temp_bytes=getattr(memory, 'temp_size_in_bytes', None))


def make_products(case, mode):
    """Product callables ``f(*args, events)`` and their args, keyed by variant."""
    w, idx, ptr, _, shape, matrix = case
    weighted = mode == 'weighted'
    return {
        'P': (lambda w, i, p, cfg, e: binary_csrmm(w, i, p, e, shape=shape, transpose=True, backend='cuda_raw',
                                                   weighted=weighted, config=cfg),
              (w, idx, ptr, matrix.config)),
        'J': (lambda w, i, p, e: binary_csrmm(w, i, p, e, shape=shape, transpose=True, backend='jax_raw',
                                              weighted=weighted),
              (w, idx, ptr)),
        'T': (lambda w, i, p, e: (_event_array(e.T, weighted) @ brainevent.CSR((w, i, p), shape=shape)).T,
              (w, idx, ptr)),
    }


def _event_array(events, weighted):
    return brainevent.WeightedArray(events) if weighted else BinaryArray(events)


def static_product(rows, cols, nnz, batch, dtype, weighted, l2_bytes, *, group=None,
                   q_policy='worst', c_policy='l2', lengths=None, precomputed=False):
    group = shape_group(rows, nnz) if group is None else group
    q = long_tasks_for(q_policy, rows, nnz, group, lengths)
    chunk = chunk_for(c_policy, batch, cols, np.dtype(dtype).itemsize, l2_bytes)
    capacity, words, short = scratch(rows, batch, chunk, q, np.dtype(dtype).itemsize, weighted)
    fn = partial(static_mm, cols=cols, group=group, long_tasks=q, chunk=chunk, weighted=weighted)
    plan = dict(group=group, long_tasks=q, chunk=chunk, task_capacity=capacity,
                scratch_bytes=4 * words + 4 * short)
    if precomputed:
        return (lambda w, i, p, m, e: fn(w, i, p, e, m)), plan
    return (lambda w, i, p, e: fn(w, i, p, e)), plan


def prepared_plan(matrix, batch, dtype, weighted):
    plan = matrix.config.mm_plan(batch, dtype, weighted)
    if plan is None:
        return None
    return dict(group=plan.group, long_tasks=plan.long_tasks, chunk=plan.chunk,
                task_capacity=plan.task_capacity,
                scratch_bytes=4 * plan.workspace_words + 4 * plan.short_rows_size)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def gpu_l2_bytes():
    from brainevent._csr.config import get_device_info
    info = get_device_info(device=jax.devices('gpu')[0])
    return info.l2_bytes if info is not None else 6 << 20


def load_case(size, dist, seed=0):
    rows, cols, mean = SIZES[size]
    w, idx, ptr, lengths = build_structure(dist, rows, cols, mean, seed)
    w, idx, ptr = map(jnp.asarray, (w, idx, ptr))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PerformanceWarning)
        matrix = brainevent.CSR((w, idx, ptr), shape=(rows, cols))
    return w, idx, ptr, lengths, (rows, cols), matrix


def run_grid(emit, sel, runs, warmup, l2_bytes):
    for size in sel['sizes']:
        for dist in sel['dists']:
            case = load_case(size, dist)
            w, idx, ptr, lengths, (rows, cols), matrix = case
            nnz = int(idx.size)
            for mode in sel['modes']:
                weighted = mode == 'weighted'
                for batch in sel['batches']:
                    s_fn, s_plan = static_product(rows, cols, nnz, batch, w.dtype, weighted, l2_bytes)
                    s0_fn, _ = static_product(rows, cols, nnz, batch, w.dtype, weighted, l2_bytes,
                                              precomputed=True)
                    mask = long_mask(ptr, ITERATIONS * s_plan['group'])
                    variants = make_products(case, mode)
                    variants['S'] = (s_fn, (w, idx, ptr))
                    variants['S0'] = (s0_fn, (w, idx, ptr, mask))
                    p_plan = prepared_plan(matrix, batch, w.dtype, weighted)
                    for activity in sel['activities']:
                        stack = event_stack(rows, batch, activity, mode, UNROLL, seed=batch)
                        reference = None
                        for variant, (fn, args) in variants.items():
                            base = dict(kind='time', size=size, dist=dist, mode=mode, batch=batch,
                                        activity=activity, variant=variant, rows=rows, cols=cols, nnz=nnz)
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter('ignore', PerformanceWarning)
                                    result = jax.jit(fn)(*args, stack[0])
                                    timing = time_product(fn, args, stack, runs, warmup)
                            except Exception as exc:  # noqa: BLE001 - e.g. the JAX scatter running out of memory
                                emit(dict(base, error=f'{type(exc).__name__}: {str(exc)[:200]}'))
                                continue
                            if reference is None:
                                reference = np.asarray(result, np.float64)
                            close = bool(np.allclose(np.asarray(result, np.float64), reference,
                                                     rtol=1e-4, atol=1e-3 * max(1.0, np.abs(reference).max())))
                            plan = p_plan if variant == 'P' else s_plan if variant in ('S', 'S0') else None
                            emit(dict(base, dtype='float32', close=close, plan=plan, **timing))
            del case, matrix


def run_sweeps(emit, sel, runs, warmup, l2_bytes):
    """Group, capacity and chunk sweeps of the S prototype on skewed rows."""
    for size in sel['sweep_sizes']:
        for dist in ('pareto', 'hub'):
            w, idx, ptr, lengths, (rows, cols), _ = load_case(size, dist)
            nnz = int(idx.size)
            batch = 64
            for activity in (0.05, 1.0):
                stack = event_stack(rows, batch, activity, 'binary', UNROLL, seed=7)
                settings = [('group', dict(group=g, q_policy='exact')) for g in (4, 8, 16, 32)]
                settings += [('capacity', dict(q_policy=q)) for q in ('exact', 'worst', 'nnz512', 'half', 'zero')]
                settings += [('chunk', dict(c_policy=c)) for c in (1, 4, 16, 64, 128, 'l2')]
                for axis, kwargs in settings:
                    kwargs.setdefault('lengths', lengths)
                    fn, plan = static_product(rows, cols, nnz, batch, w.dtype, False, l2_bytes, **kwargs)
                    timing = time_product(fn, (w, idx, ptr), stack, runs, warmup)
                    value = kwargs.get('group', kwargs.get('q_policy', kwargs.get('c_policy')))
                    emit(dict(kind='sweep', axis=axis, value=str(value), size=size, dist=dist,
                              activity=activity, batch=batch, shape_group=shape_group(rows, nnz),
                              plan=plan, **timing))


def run_mask_cost(emit, sel, runs, warmup):
    """Per-call cost of the JAX long-row mask on its own.

    Each step adds a distinct runtime zero to ``indptr`` so the masks are not
    merged; the add fuses into the mask's first kernel.
    """
    for size in sel['sizes']:
        _, _, ptr, _, _, _ = load_case(size, 'indegree')
        zeros = jnp.zeros((UNROLL,), ptr.dtype)
        args = (ptr,)
        step = lambda p, z: _first(long_mask(p + z, 256))
        samples, _ = _samples(_compiled_steps(step, args, zeros), args, zeros, runs, warmup, 0.02)
        emit(dict(kind='mask', size=size, rows=SIZES[size][0], median_us=1e6 * statistics.median(samples)))


def _uses_native(fn, *args):
    """Whether the compiled program calls the native hybrid MM kernel."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PerformanceWarning)
        text = jax.jit(fn).lower(*args).compile().as_text()
    targets = re.findall(r'custom_call_target="([^"]+)"', text)
    return any(t.startswith('csr_hybrid_mm') for t in targets)


def run_compile_and_lowering(emit, sel, l2_bytes, redraws=10):
    """Compile counts over redrawn structures and the kernel each construction path lowers to."""
    size = sel['sizes'][-1]
    rows, cols, mean = SIZES[size]
    batch = 8
    events = event_stack(rows, batch, 0.05, 'binary', 1, seed=3)[0]
    shape = (rows, cols)
    for dist in ('indegree', 'pareto', 'hub'):
        p_fn = jax.jit(lambda w, i, p, cfg, e: binary_csrmm(w, i, p, e, shape=shape, transpose=True,
                                                            backend='cuda_raw', config=cfg))
        t_fn = jax.jit(lambda w, i, p, e: (BinaryArray(e.T) @ brainevent.CSR((w, i, p), shape=shape)).T)
        s_raw, _ = static_product(rows, cols, rows * mean, batch, jnp.float32, False, l2_bytes)
        s_fn = jax.jit(s_raw)
        for seed in range(redraws):
            w, idx, ptr, _, _, matrix = load_case(size, dist, seed=seed)
            if idx.size != rows * mean:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', PerformanceWarning)
                p_fn(w, idx, ptr, matrix.config, events).block_until_ready()
                t_fn(w, idx, ptr, events).block_until_ready()
                s_fn(w, idx, ptr, events).block_until_ready()
        emit(dict(kind='compile', size=size, dist=dist, redraws=redraws,
                  P=p_fn._cache_size(), T=t_fn._cache_size(), S=s_fn._cache_size()))

    w, idx, ptr, _, _, matrix = load_case(size, 'indegree')
    cases = {
        'eager_gpu': lambda: _uses_native(
            lambda cfg, e: binary_csrmm(w, idx, ptr, e, shape=shape, transpose=True, config=cfg),
            matrix.config, events),
        'traced_in_jit': lambda: _uses_native(
            lambda w, i, p, e: (BinaryArray(e.T) @ brainevent.CSR((w, i, p), shape=shape)).T,
            w, idx, ptr, events),
        'vmap_of_mv': lambda: _uses_native(
            lambda e: jax.vmap(lambda v: BinaryArray(v) @ matrix)(e.T), events),
        'cpu_built': lambda: _uses_native(
            lambda cfg, e: binary_csrmm(w, idx, ptr, e, shape=shape, transpose=True, config=cfg),
            _cpu_built_config(w, idx, ptr, shape), events),
        'static_prototype': lambda: _uses_native(s_fn, w, idx, ptr, events),
    }
    for name, check in cases.items():
        try:
            native = check()
        except Exception as exc:  # noqa: BLE001 - record and continue
            native = f'error: {type(exc).__name__}'
        emit(dict(kind='lowering', case=name, native=native))


def _cpu_built_config(w, idx, ptr, shape):
    cpu = jax.devices('cpu')[0]
    with jax.default_device(cpu):
        matrix = brainevent.CSR(tuple(jax.device_put(x, cpu) for x in (w, idx, ptr)), shape=shape)
    return jax.device_put(matrix.config, jax.devices('gpu')[0])


def run_spot(emit, runs, warmup, l2_bytes):
    """Other dtypes, shared weights and the mirror-vs-gather ``W @ events`` route."""
    w, idx, ptr, lengths, (rows, cols), matrix = load_case('M', 'lognormal')
    nnz, batch = int(idx.size), 64
    stack = event_stack(rows, batch, 0.05, 'binary', UNROLL, seed=11)
    for dtype in (jnp.float16, jnp.bfloat16, jnp.float64, 'shared'):
        weights = jnp.full((1,), 0.1, jnp.float32) if dtype == 'shared' else w.astype(dtype)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PerformanceWarning)
            typed = brainevent.CSR((weights if dtype != 'shared' else jnp.full((nnz,), 0.1, jnp.float32),
                                    idx, ptr), shape=(rows, cols))
        cfg = typed.config
        p = lambda w_, i, p_, c, e: binary_csrmm(w_, i, p_, e, shape=(rows, cols), transpose=True,
                                                 backend='cuda_raw', config=c)
        s_fn, plan = static_product(rows, cols, nnz, batch, weights.dtype, False, l2_bytes)
        for variant, fn, args in (('P', p, (weights, idx, ptr, cfg)), ('S', s_fn, (weights, idx, ptr))):
            timing = time_product(fn, args, stack, runs, warmup)
            emit(dict(kind='spot', case=str(getattr(dtype, '__name__', dtype)), variant=variant, **timing))
    # W @ events (the gather direction): CUDA gather vs indexed mirror scatter.
    stack_cols = event_stack(cols, batch, 0.05, 'binary', UNROLL, seed=12)
    gather = lambda w_, i, p_, e: binary_csrmm(w_, i, p_, e, shape=(rows, cols), transpose=False,
                                               backend='cuda_raw')
    mirror = lambda m, e: m @ BinaryArray(e)
    cuda_matrix = brainevent.CSR((w, idx, ptr), shape=(rows, cols), backend='cuda_raw')
    for variant, fn, args in (('gather', gather, (w, idx, ptr)), ('mirror_scatter', mirror, (cuda_matrix,))):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PerformanceWarning)
            timing = time_product(fn, args, stack_cols, runs, warmup)
        emit(dict(kind='spot', case='W@events', variant=variant, **timing))


def run_sddmm(emit, sel):
    """Matrix SDDMM gradient scratch: today's worst-case capacity vs a bounded one."""
    for size in sel['sizes']:
        rows, cols, mean = SIZES[size]
        nnz = rows * mean
        for batch in (8, 64, 256):
            attrs, infos = _sddmm_cuda_plan(rows, batch, nnz, jnp.float32, False)
            worst = sum(math.prod(i.shape) * np.dtype(i.dtype).itemsize for i in infos[1:])
            chunk = attrs['chunk']
            bounded_capacity = chunk * -(-nnz // TASK_EDGES)
            bounded = worst - 4 * 3 * (attrs['capacity'] - bounded_capacity)
            emit(dict(kind='sddmm', size=size, batch=batch, worst_bytes=worst, bounded_bytes=bounded,
                      capacity=attrs['capacity'], bounded_capacity=bounded_capacity))


def cmd_run(args):
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise SystemExit('a CUDA GPU is required')
    sel = dict(sizes=tuple(SIZES), dists=DISTS, batches=BATCHES, activities=ACTIVITIES, modes=MODES,
               sweep_sizes=('M', 'L', 'W'))
    if args.quick:
        sel.update(QUICK, sweep_sizes=('M',))
        args.runs = min(args.runs, 5)
    l2_bytes = gpu_l2_bytes()
    out = open(args.out, 'w')
    started = time.perf_counter()

    def emit(row):
        out.write(json.dumps(row) + '\n')
        out.flush()
        brief = {k: v for k, v in row.items() if k not in ('plan',)}
        print(f'[{time.perf_counter() - started:7.1f}s] {brief}', flush=True)

    emit(dict(kind='meta', device=jax.devices('gpu')[0].device_kind, l2_bytes=l2_bytes,
              jax=jax.__version__, brainevent=brainevent.__version__, quick=args.quick,
              runs=args.runs, warmup=args.warmup))
    run_mask_cost(emit, sel, args.runs, args.warmup)
    run_sddmm(emit, sel)
    run_compile_and_lowering(emit, sel, l2_bytes)
    run_grid(emit, sel, args.runs, args.warmup, l2_bytes)
    run_sweeps(emit, sel, args.runs, args.warmup, l2_bytes)
    if not args.quick:
        run_spot(emit, args.runs, args.warmup, l2_bytes)
    out.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _geomean(values):
    values = [v for v in values if v and v > 0]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else float('nan')


GATES = {
    'G1': 'S <= 1.10x P (geomean, batch >= 8); every case S - 5 us <= 1.25x P',
    'G2': 'S >= 1.5x faster than J (geomean; M, L; batch >= 8; activity <= 20%)',
    'G3': 'mask <= 5% of an S call (median, batch >= 8)',
    'G4': 'S scratch <= max(2x P, 25% of the matrix storage) in every case',
    'G5': 'S compiles once over redrawn structures and lowers natively',
    'G6': 'shape-based group within 5% of the best group in >= 90% of sweeps',
    'G7': "'worst' capacity <= 1.10x 'exact' on hub and Pareto rows",
    'G8': 'one constant chunk within 5% of the L2 rule in every sweep',
    'G9': 'bounded SDDMM capacity cuts its scratch >= 4x',
}


def _case(r):
    return r['size'], r['dist'], r['mode'], r['batch'], r['activity']


def _by_case(rows):
    by = {}
    for r in rows:
        if r['kind'] == 'time' and 'error' not in r:
            by.setdefault(_case(r), {})[r['variant']] = r
    return by


def _ratio(v, a, b):
    return v[a]['median_us'] / v[b]['median_us'] if a in v and b in v else None


def _sweeps(rows):
    groups = {}
    for r in rows:
        if r['kind'] == 'sweep':
            groups.setdefault((r['axis'], r['size'], r['dist'], r['activity']), {})[r['value']] = r
    return groups


def gates(rows):
    """``(gate, condition, measured, passed)`` for every gate with data."""
    by, sweeps, result = _by_case(rows), _sweeps(rows), []
    nan = float('nan')

    def add(gate, measured, passed):
        result.append((gate, GATES[gate], measured, bool(passed)))

    big = [(k, v) for k, v in by.items() if k[3] >= 8 and 'S' in v and 'P' in v]
    if big:
        mean = _geomean([_ratio(v, 'S', 'P') for _, v in big])
        worst = max((v['S']['median_us'] - 5) / v['P']['median_us'] for _, v in big)
        add('G1', f'geomean {mean:.3f}, worst {worst:.2f}', mean <= 1.10 and worst <= 1.25)
    js = [_ratio(v, 'J', 'S') for k, v in by.items() if k[0] in ('M', 'L') and k[3] >= 8 and k[4] <= 0.2]
    if any(js):
        mean = _geomean(js)
        add('G2', f'geomean J/S {mean:.2f}', mean >= 1.5)
    mask = {r['size']: r['median_us'] for r in rows if r['kind'] == 'mask'}
    shares = [mask[k[0]] / v['S']['median_us'] for k, v in by.items() if k[0] in mask and k[3] >= 8 and 'S' in v]
    if shares:
        median = statistics.median(shares)
        add('G3', f'median {100 * median:.1f}%, max {100 * max(shares):.1f}%', median <= 0.05)
    limits = [v['S']['plan']['scratch_bytes'] / max(2 * v['P']['plan']['scratch_bytes'], 2 * v['S']['nnz'])
              for v in by.values() if v.get('S', {}).get('plan') and v.get('P', {}).get('plan')]
    if limits:
        add('G4', f'max S / limit {max(limits):.2f}', max(limits) <= 1)
    compiles = [r for r in rows if r['kind'] == 'compile']
    native = {r['case']: r['native'] for r in rows if r['kind'] == 'lowering'}
    if compiles:
        add('G5', f"S {[r['S'] for r in compiles]}, P {[r['P'] for r in compiles]} executables; "
                  f"S native {native.get('static_prototype')}",
            all(r['S'] == 1 for r in compiles) and native.get('static_prototype') is True)
    group = [(rs[str(next(iter(rs.values()))['shape_group'])]['median_us'] / min(r['median_us'] for r in rs.values()))
             for (axis, *_), rs in sweeps.items() if axis == 'group']
    if group:
        share = sum(r <= 1.05 for r in group) / len(group)
        add('G6', f'{100 * share:.0f}% within 5%, worst {max(group):.2f}x', share >= 0.9)
    capacity = [rs['worst']['median_us'] / rs['exact']['median_us']
                for (axis, _, dist, _), rs in sweeps.items() if axis == 'capacity' and dist in ('hub', 'pareto')]
    if capacity:
        add('G7', f'worst/exact max {max(capacity):.2f}', max(capacity) <= 1.10)
    chunk = [rs for (axis, *_), rs in sweeps.items() if axis == 'chunk' and 'l2' in rs]
    if chunk:
        constants = {c for rs in chunk for c in rs if c != 'l2'}
        worst = {c: max((rs[c]['median_us'] / rs['l2']['median_us'] if c in rs else nan) for rs in chunk)
                 for c in constants}
        best = min(worst, key=lambda c: worst[c])
        add('G8', f'best constant {best}: {worst[best]:.2f}x of L2 rule at worst', worst[best] <= 1.05)
    sddmm = [r['worst_bytes'] / max(r['bounded_bytes'], 1) for r in rows if r['kind'] == 'sddmm']
    if sddmm:
        add('G9', f'{min(sddmm):.1f}x to {max(sddmm):.1f}x', min(sddmm) >= 4)
    return result


def report(rows, full=False):
    by = _by_case(rows)
    errors = [r for r in rows if r['kind'] == 'time' and 'error' in r]
    not_close = [(k, x) for k, v in by.items() for x, r in v.items() if not r.get('close', True)]
    meta = next((r for r in rows if r['kind'] == 'meta'), {})
    lines = [f"Device {meta.get('device')}, L2 {meta.get('l2_bytes', 0) / 2**20:.0f} MiB, "
             f"jax {meta.get('jax')}, brainevent {meta.get('brainevent')}", '',
             '## Gates', '| gate | condition | measured | pass |', '|---|---|---|---|']
    lines += [f'| {g} | {cond} | {measured} | {"yes" if ok else "**no**"} |' for g, cond, measured, ok in gates(rows)]
    lines.append(f'\nResults differing from P: {len(not_close)} {not_close[:5]}')
    for r in errors:
        lines.append(f"- error {r['variant']} at {_case(r)}: {r['error'][:120]}")

    lines += ['', '## Time ratios by size and activity (geomean over distributions, modes and batch)',
              '| size | activity | S/P b>=8 | S/P b=1 | J/S b>=8 | J/S b=1 | T/J | cases |',
              '|---|---|---|---|---|---|---|---|']
    cells = {}
    for k, v in by.items():
        cells.setdefault((k[0], k[4]), []).append((k, v))
    order = {name: i for i, name in enumerate(SIZES)}
    for (size, act), items in sorted(cells.items(), key=lambda x: (order[x[0][0]], x[0][1])):
        pick = lambda a, b, cond: _geomean([_ratio(v, a, b) for k, v in items if cond(k)])
        lines.append(f'| {size} | {act:g} | {pick("S", "P", lambda k: k[3] >= 8):.2f} | '
                     f'{pick("S", "P", lambda k: k[3] == 1):.2f} | {pick("J", "S", lambda k: k[3] >= 8):.2f} | '
                     f'{pick("J", "S", lambda k: k[3] == 1):.2f} | {pick("T", "J", lambda k: True):.2f} | '
                     f'{len(items)} |')
    ranked = sorted(((_ratio(v, 'S', 'P'), k, v) for k, v in by.items() if _ratio(v, 'S', 'P')),
                    key=lambda x: -x[0])
    lines += ['', '## Largest S/P', '| case (size, dist, mode, batch, activity) | P us | S us | S0 us | S/P |',
              '|---|---|---|---|---|']
    for ratio, k, v in ranked[:8]:
        lines.append(f"| {k} | {v['P']['median_us']:.0f} | {v['S']['median_us']:.0f} | "
                     f"{v['S0']['median_us'] if 'S0' in v else float('nan'):.0f} | {ratio:.2f} |")
    if full:
        lines += ['', '| size | dist | mode | batch | act | P us | S us | S0 us | J us | T us | S/P |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for k in sorted(by, key=lambda k: (order[k[0]],) + k[1:]):
            v = by[k]
            cell = lambda x: f"{v[x]['median_us']:.0f}" if x in v else '-'
            lines.append(f'| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]:g} | '
                         + ' | '.join(cell(x) for x in ('P', 'S', 'S0', 'J', 'T')) + f" | {_ratio(v, 'S', 'P'):.2f} |")

    mask = [r for r in rows if r['kind'] == 'mask']
    if mask:
        lines += ['', '## Long-row mask alone (per call)']
        lines += [f"- {r['size']} ({r['rows']} rows): {r['median_us']:.1f} us" for r in mask]
    scratch = {}
    for k, v in by.items():
        if v.get('S', {}).get('plan') and v.get('P', {}).get('plan'):
            entry = scratch.setdefault(k[0], [0, 0, v['S']['nnz'] * 8])
            entry[0] = max(entry[0], v['P']['plan']['scratch_bytes'])
            entry[1] = max(entry[1], v['S']['plan']['scratch_bytes'])
    if scratch:
        lines += ['', '## Largest scratch per size (MiB)', '| size | P | S | matrix storage |', '|---|---|---|---|']
        lines += [f'| {size} | {p / 2**20:.1f} | {s_ / 2**20:.1f} | {m / 2**20:.1f} |'
                  for size, (p, s_, m) in sorted(scratch.items(), key=lambda x: order[x[0]])]
    compiles = [r for r in rows if r['kind'] == 'compile']
    if compiles:
        lines += ['', '## Executables over redrawn structures (same shape and nnz)']
        lines += [f"- {r['size']}/{r['dist']}, {r['redraws']} redraws: P={r['P']} T={r['T']} S={r['S']}"
                  for r in compiles]
    lowering = [r for r in rows if r['kind'] == 'lowering']
    if lowering:
        lines += ['', '## Native hybrid MM lowered?']
        lines += [f"- {r['case']}: {r['native']}" for r in lowering]
    sweeps = _sweeps(rows)
    if sweeps:
        lines += ['', '## Sweeps (S prototype, batch 64, binary; time relative to the best setting)']
        for (axis, size, dist, act), rs in sorted(sweeps.items(), key=lambda x: (x[0][0], order[x[0][1]]) + x[0][2:]):
            best = min(r['median_us'] for r in rs.values())
            cells_ = ', '.join(f"{value}={r['median_us'] / best:.2f}" for value, r in rs.items())
            extra = f" (shape group {next(iter(rs.values()))['shape_group']})" if axis == 'group' else ''
            lines.append(f'- {axis} {size}/{dist} act={act:g}{extra}: {cells_}; best {best:.0f} us')
    sddmm = [r for r in rows if r['kind'] == 'sddmm']
    if sddmm:
        lines += ['', '## Matrix SDDMM gradient scratch (MiB)']
        lines += [f"- {r['size']} batch {r['batch']}: worst-case {r['worst_bytes'] / 2**20:.1f}, "
                  f"bounded {r['bounded_bytes'] / 2**20:.1f}" for r in sddmm]
    spot = [r for r in rows if r['kind'] == 'spot']
    if spot:
        lines += ['', '## Spot cases (M, lognormal, batch 64, 5% active)']
        lines += [f"- {r['case']} {r['variant']}: {r['median_us']:.0f} us" for r in spot]
    return '\n'.join(lines)


def compare(base, new):
    def index(rows):
        return {(r['size'], r['dist'], r['mode'], r['batch'], r['activity'], r['variant']): r
                for r in rows if r['kind'] == 'time'}
    a, b = index(base), index(new)
    ratios = [b[k]['median_us'] / a[k]['median_us'] for k in a.keys() & b.keys()]
    return f'## new/base median time: geomean {_geomean(ratios):.3f} over {len(ratios)} cases'


def cmd_report(args):
    base = _load(args.base)
    print(report(base, full=args.full))
    if args.new:
        print()
        print(compare(base, _load(args.new)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='measure and write JSON lines')
    run.add_argument('--out', required=True)
    run.add_argument('--quick', action='store_true', help='small grid, a few minutes')
    run.add_argument('--runs', type=int, default=10)
    run.add_argument('--warmup', type=int, default=2)
    run.set_defaults(func=cmd_run)
    rep = sub.add_parser('report', help='summarize one run, or compare two')
    rep.add_argument('base')
    rep.add_argument('new', nargs='?')
    rep.add_argument('--full', action='store_true', help='also print every timed case')
    rep.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
