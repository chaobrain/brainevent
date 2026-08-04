# JIT Uniform Numba CUDA Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the new `_jit_scalar` Numba/light-RNG parity work to `_jit_uniform`, so CPU `numba` and GPU `cuda_raw` draw the same light-RNG structure and the same uniform weights for dense, matvec, matmat, CSR, and dt2t paths.

**Architecture:** Keep the existing `_jit_uniform` public API and CUDA kernels. Add one shared Numba-compatible uniform-weight hash helper beside the existing light-RNG helpers in `_numba_random.py`, then replace `_jit_uniform.float` and `_jit_uniform.binary` old LFSR numba generators with the same chunk/lane walk used by CUDA. Add Numba count/fill and dt2t generators by following `_jit_scalar.csr` and `_jit_scalar.dt2t`, with uniform data values sampled by the CUDA-compatible `(seed, row, col)` hash.

**Tech Stack:** Python 3.11+, JAX, `XLACustomKernel`, `numba_kernel`, Numba CPU, CUDA raw FFI, pytest.

---

## Context

`brainevent/_jit_scalar` now has `get_numba_light_rng_funcs()` based Numba kernels that mirror CUDA `light_rng_init_wpr`, `stationary_initial_q`, and lane strides. The important constants are:

- `mv` structure: 32-lane walk.
- `mm` structure: 4-lane AW-T4 walk.
- `chunk_size`: `_normalize_chunk_size(int(kwargs['shape'][1]), None)` for fused mv/mm paths, and explicit `chunk_size`/`target_chunks` for CSR/dt2t.

`brainevent/_jit_uniform` currently has Numba kernels for dense, float mv/mm, and binary mv/mm, but they still use `get_numba_lfsr_seed()`, `get_numba_lfsr_random_integers()`, and `get_numba_lfsr_uniform()`. Its CSR and fused dt2t primitives are still CUDA-only. CUDA uniform kernels generate connectivity with light-RNG and generate weights with:

```c
float hash_uniform01(unsigned int seed, int row, int col) {
    unsigned int h = seed ^ 0xa0761d65U;
    h ^= (unsigned int)row * 0xe7037ed1U;
    h ^= (unsigned int)col * 0x8ebc6af1U;
    h = mix32(h);
    return (float)(h & 0x00ffffffU) * (1.0f / 16777216.0f);
}
```

The plan below uses a conservative direct port. A broader refactor that shares one generic sampler across scalar/uniform/normal would reduce duplication, but it would touch more code than needed and make the parity change harder to review.

## File Structure

- Modify `brainevent/_numba_random.py`: add `light_rng_uniform01` and expose it through `get_numba_light_rng_funcs()`.
- Modify `brainevent/_numba_random_test.py`: lock exact helper outputs and njit dispatch.
- Create `brainevent/_jit_uniform/_test_util.py`: pure NumPy reference for uniform light-RNG dense matrices used by tests only.
- Modify `brainevent/_jit_uniform/float.py`: replace old LFSR Numba generators in `_jitu_numba_kernel_generator`, `_jitumv_numba_kernel_generator`, and `_jitumm_numba_kernel_generator`.
- Modify `brainevent/_jit_uniform/binary.py`: replace old LFSR Numba generators in `_jitumv_numba_kernel_generator` and `_jitumm_numba_kernel_generator`.
- Modify `brainevent/_jit_uniform/csr.py`: add Numba count/fill generators and register them.
- Modify `brainevent/_jit_uniform/dt2t.py`: add Numba fused fill generator and register it.
- Modify tests in `brainevent/_jit_uniform/float_test.py`, `binary_test.py`, `csr_test.py`, `dt2t_test.py`, and `main_test.py`.
- Modify `docs/reference/apis/utilities.rst` and `changelog.md` with the new helper/backend behavior.

---

### Task 1: Add the Uniform Weight Hash Helper

**Files:**
- Modify: `brainevent/_numba_random.py`
- Modify: `brainevent/_numba_random_test.py`
- Modify: `docs/reference/apis/utilities.rst`

- [ ] **Step 1: Write failing helper tests**

Add these imports in `brainevent/_numba_random_test.py`:

```python
from brainevent._numba_random import (
    light_rng_uniform01,
    get_numba_light_rng_funcs,
)
```

Add this test class near the existing light-RNG tests or after the LFSR tests:

```python
class TestLightRNGUniformHash:
    def test_exact_cuda_reference_values(self):
        cases = [
            (42, 0, 0, np.float32(0.2929498553276062)),
            (42, 3, 7, np.float32(0.548724353313446)),
            (123, 19, 29, np.float32(0.5329357385635376)),
            (0, 1, 2, np.float32(0.31099069118499756)),
            (np.uint32(0xFFFFFFFF), 65535, 123456, np.float32(0.8090267777442932)),
        ]
        for seed, row, col, expected in cases:
            actual = light_rng_uniform01(seed, row, col)
            assert np.float32(actual) == expected

    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason='numba is not installed')
    def test_dispatch_exposes_njit_uniform01(self):
        funcs = get_numba_light_rng_funcs()
        assert 'uniform01' in funcs
        actual = funcs['uniform01'](np.uint32(42), 3, 7)
        assert np.float32(actual) == np.float32(0.548724353313446)
```

- [ ] **Step 2: Run helper tests and confirm the failure**

Run:

```bash
pytest -q brainevent/_numba_random_test.py::TestLightRNGUniformHash
```

Expected: fail with `ImportError: cannot import name 'light_rng_uniform01'`.

- [ ] **Step 3: Implement `light_rng_uniform01`**

In `brainevent/_numba_random.py`, add `light_rng_uniform01` to `__all__` next to the other light-RNG helpers:

```python
    'light_rng_uniform01',
```

Add this function after `light_rng_init`:

```python
def light_rng_uniform01(seed, row, col):
    """CUDA-compatible 24-bit uniform variate for a generated edge."""
    h = np.uint32(np.uint32(seed) ^ np.uint32(0xa0761d65))
    h = np.uint32(
        h ^ np.uint32((np.uint64(np.uint32(row)) * np.uint64(0xe7037ed1)) & np.uint64(0xFFFFFFFF))
    )
    h = np.uint32(
        h ^ np.uint32((np.uint64(np.uint32(col)) * np.uint64(0x8ebc6af1)) & np.uint64(0xFFFFFFFF))
    )
    h = light_rng_mix32(h)
    return np.float32((h & np.uint32(0x00FFFFFF)) * np.float32(1.0 / 16777216.0))
```

Add it to `_LIGHT_RNG_FUNC_NAMES`:

```python
_LIGHT_RNG_FUNC_NAMES = (
    'light_rng_mix32', 'light_rng_bounded', 'light_rng_next',
    'light_rng_init', 'light_rng_uniform01', 'light_rng_initial_q',
)
```

Expose it through `get_numba_light_rng_funcs()`:

```python
    return {
        'init': g['light_rng_init'],
        'next': g['light_rng_next'],
        'bounded': g['light_rng_bounded'],
        'initial_q': g['light_rng_initial_q'],
        'mix32': g['light_rng_mix32'],
        'uniform01': g['light_rng_uniform01'],
    }
```

- [ ] **Step 4: Document the dispatch helper**

In `docs/reference/apis/utilities.rst`, add:

```rst
   get_numba_light_rng_funcs
```

to the `Numba RNG Dispatch` autosummary list.

- [ ] **Step 5: Run helper tests**

Run:

```bash
pytest -q brainevent/_numba_random_test.py::TestLightRNGUniformHash
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add brainevent/_numba_random.py brainevent/_numba_random_test.py docs/reference/apis/utilities.rst
git commit -m "feat: expose numba light uniform hash"
```

---

### Task 2: Add an Independent Uniform Light-RNG Reference

**Files:**
- Create: `brainevent/_jit_uniform/_test_util.py`

- [ ] **Step 1: Create the test reference helper**

Create `brainevent/_jit_uniform/_test_util.py` with this content:

```python
import math

import numpy as np

MV_STRIDE = 32
MM_STRIDE = 4


def _u32(x):
    return np.uint32(x)


def mix32(x):
    x = _u32(x)
    x = _u32(x ^ (x >> _u32(16)))
    x = _u32((np.uint64(x) * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF))
    x = _u32(x ^ (x >> _u32(15)))
    x = _u32((np.uint64(x) * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF))
    x = _u32(x ^ (x >> _u32(16)))
    return x


def fast_bounded_u32(r, bound):
    return _u32((np.uint64(_u32(r)) * np.uint64(_u32(bound))) >> np.uint64(32))


def light_rng_next(state):
    x = _u32(state)
    x = _u32(x ^ _u32(x << _u32(13)))
    x = _u32(x ^ (x >> _u32(17)))
    x = _u32(x ^ _u32(x << _u32(5)))
    return _u32(0x6D2B79F5) if x == _u32(0) else x


def light_rng_init(seed, row, chunk_id, lane):
    x = _u32(_u32(seed) ^ _u32(0xD1B54A35))
    x = _u32(x ^ _u32((np.uint64(_u32(row)) * np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)))
    x = _u32(x ^ _u32((np.uint64(_u32(chunk_id)) * np.uint64(0xC2B2AE35)) & np.uint64(0xFFFFFFFF)))
    x = _u32(x ^ _u32((np.uint64(_u32(lane)) * np.uint64(0x27D4EB2D)) & np.uint64(0xFFFFFFFF)))
    x = mix32(x)
    return _u32(0x6D2B79F5) if x == _u32(0) else x


def stationary_initial_q(state, cl):
    n = _u32(_u32(cl) - _u32(1))
    while True:
        state = light_rng_next(state)
        q = fast_bounded_u32(state, n)
        state = light_rng_next(state)
        gate = fast_bounded_u32(state, n)
        if gate < _u32(n - q):
            return q, state


def hash_uniform01(seed, row, col):
    h = _u32(_u32(seed) ^ _u32(0xA0761D65))
    h = _u32(h ^ _u32((np.uint64(_u32(row)) * np.uint64(0xE7037ED1)) & np.uint64(0xFFFFFFFF)))
    h = _u32(h ^ _u32((np.uint64(_u32(col)) * np.uint64(0x8EBC6AF1)) & np.uint64(0xFFFFFFFF)))
    h = mix32(h)
    return np.float32((h & _u32(0x00FFFFFF)) * np.float32(1.0 / 16777216.0))


def stride_for_mode(matrix_mode):
    if matrix_mode == 'mv':
        return MV_STRIDE
    if matrix_mode == 'mm':
        return MM_STRIDE
    raise ValueError(f"matrix_mode must be 'mv' or 'mm', got {matrix_mode!r}.")


def conn_length(prob):
    prob = float(prob)
    if prob == 0.0:
        return 0
    return max(2, int(math.ceil(2.0 / prob)))


def default_chunk_size(n_cols, target_chunks=4):
    return max(1, (int(n_cols) + int(target_chunks) - 1) // int(target_chunks))


def iter_edges(seed, clen, n_rows, n_cols, *, corder, stride, chunk_size=None):
    seed0 = _u32(seed)
    cl = _u32(max(2, int(clen)))
    cs = default_chunk_size(n_cols) if chunk_size is None else int(chunk_size)
    n_chunks = 0 if n_cols <= 0 else (int(n_cols) + cs - 1) // cs
    if corder:
        for row in range(int(n_rows)):
            for chunk_id in range(n_chunks):
                chunk_start = chunk_id * cs
                chunk_end = min(chunk_start + cs, int(n_cols))
                chunk_width = chunk_end - chunk_start
                for lane in range(int(stride)):
                    state = light_rng_init(seed0, row, chunk_id, lane)
                    q, state = stationary_initial_q(state, cl)
                    local_j = lane + int(stride) * int(q)
                    while local_j < chunk_width:
                        col = chunk_start + local_j
                        yield row, col, row, col
                        state = light_rng_next(state)
                        q = q + _u32(1) + fast_bounded_u32(state, cl - _u32(1))
                        local_j = lane + int(stride) * int(q)
    else:
        for row in range(int(n_rows)):
            for chunk_id in range(n_chunks):
                chunk_start = chunk_id * cs
                chunk_end = min(chunk_start + cs, int(n_cols))
                chunk_width = chunk_end - chunk_start
                for lane in range(int(stride)):
                    state = light_rng_init(seed0, row, chunk_id, lane)
                    q, state = stationary_initial_q(state, cl)
                    local_j = lane + int(stride) * int(q)
                    while local_j < chunk_width:
                        col = chunk_start + local_j
                        yield col, row, row, col
                        state = light_rng_next(state)
                        q = q + _u32(1) + fast_bounded_u32(state, cl - _u32(1))
                        local_j = lane + int(stride) * int(q)


def dense_uniform_reference(w_low, w_high, prob, seed, *, shape, transpose=False, corder=True, matrix_mode='mv'):
    out_shape = tuple(reversed(shape)) if transpose else tuple(shape)
    n_rows, n_cols = out_shape if corder else tuple(reversed(out_shape))
    dtype = np.result_type(np.asarray(w_low), np.asarray(w_high), np.float32)
    out = np.zeros(out_shape, dtype=dtype)
    if float(prob) == 0.0:
        return out
    wlo = np.asarray(w_low, dtype=dtype).item()
    whi = np.asarray(w_high, dtype=dtype).item()
    span = whi - wlo
    stride = stride_for_mode(matrix_mode)
    for out_row, out_col, rng_row, rng_col in iter_edges(
        seed, conn_length(prob), n_rows, n_cols, corder=corder, stride=stride
    ):
        out[out_row, out_col] = wlo + np.asarray(hash_uniform01(seed, rng_row, rng_col), dtype=dtype) * span
    return out
```

- [ ] **Step 2: Import-check the helper**

Run:

```bash
python -m py_compile brainevent/_jit_uniform/_test_util.py
```

Expected: exits successfully with no output.

- [ ] **Step 3: Commit**

```bash
git add brainevent/_jit_uniform/_test_util.py
git commit -m "test: add uniform light rng reference"
```

---

### Task 3: Add Failing Float Uniform Numba Parity Tests

**Files:**
- Modify: `brainevent/_jit_uniform/float_test.py`

- [ ] **Step 1: Add the reference import and CPU backend gate**

Add after the existing `_jit_uniform.float` import:

```python
from brainevent._jit_uniform._test_util import dense_uniform_reference
```

Add near the existing backend globals:

```python
CPU_DEVICE = jax.devices('cpu')[0]
CPU_JITU_IMPLEMENTATIONS = tuple(jitu_p.available_backends('cpu'))
CPU_JITUMV_IMPLEMENTATIONS = tuple(jitumv_p.available_backends('cpu'))
CPU_JITUMM_IMPLEMENTATIONS = tuple(jitumm_p.available_backends('cpu'))

requires_cpu_jitu = pytest.mark.skipif(
    'numba' not in CPU_JITU_IMPLEMENTATIONS,
    reason='No jitu numba backend registered on CPU',
)
requires_cpu_jitumv = pytest.mark.skipif(
    'numba' not in CPU_JITUMV_IMPLEMENTATIONS,
    reason='No jitumv numba backend registered on CPU',
)
requires_cpu_jitumm = pytest.mark.skipif(
    'numba' not in CPU_JITUMM_IMPLEMENTATIONS,
    reason='No jitumm numba backend registered on CPU',
)
```

- [ ] **Step 2: Add dense, mv, and mm tests**

Append these tests after `test_jitu_requires_matrix_mode`:

```python
@requires_cpu_jitu
@pytest.mark.parametrize('matrix_mode', ['mv', 'mm'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitu_numba_matches_light_rng_reference(matrix_mode, transpose, corder):
    shape = (13, 17)
    with jax.default_device(CPU_DEVICE):
        actual = jitu(
            W_LOW, W_HIGH, 0.2, SEED,
            shape=shape, transpose=transpose, corder=corder,
            matrix_mode=matrix_mode, backend='numba',
        )
    expected = dense_uniform_reference(
        W_LOW, W_HIGH, 0.2, SEED,
        shape=shape, transpose=transpose, corder=corder, matrix_mode=matrix_mode,
    )
    assert np.allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


@requires_cpu_jitumv
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitumv_numba_matches_light_rng_reference(transpose, corder):
    shape = (13, 17)
    vec_size = shape[0] if transpose else shape[1]
    vector = jnp.linspace(-0.3, 0.7, vec_size, dtype=jnp.float32)
    with jax.default_device(CPU_DEVICE):
        actual = jitumv(
            W_LOW, W_HIGH, 0.2, vector, SEED,
            shape=shape, transpose=transpose, corder=corder, backend='numba',
        )
    dense = dense_uniform_reference(
        W_LOW, W_HIGH, 0.2, SEED,
        shape=shape, transpose=transpose, corder=corder, matrix_mode='mv',
    )
    expected = dense @ np.asarray(vector)
    assert np.allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


@requires_cpu_jitumm
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitumm_numba_matches_light_rng_reference(transpose, corder):
    shape = (13, 17)
    b_rows = shape[0] if transpose else shape[1]
    B = jnp.reshape(jnp.linspace(-0.5, 0.8, b_rows * 3, dtype=jnp.float32), (b_rows, 3))
    with jax.default_device(CPU_DEVICE):
        actual = jitumm(
            W_LOW, W_HIGH, 0.2, B, SEED,
            shape=shape, transpose=transpose, corder=corder, backend='numba',
        )
    dense = dense_uniform_reference(
        W_LOW, W_HIGH, 0.2, SEED,
        shape=shape, transpose=transpose, corder=corder, matrix_mode='mm',
    )
    expected = dense @ np.asarray(B)
    assert np.allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 3: Run float tests and confirm the failure**

Run:

```bash
pytest -q brainevent/_jit_uniform/float_test.py::test_jitu_numba_matches_light_rng_reference
pytest -q brainevent/_jit_uniform/float_test.py::test_jitumv_numba_matches_light_rng_reference
pytest -q brainevent/_jit_uniform/float_test.py::test_jitumm_numba_matches_light_rng_reference
```

Expected: failures showing current Numba output differs from the light-RNG reference.

- [ ] **Step 4: Commit tests**

```bash
git add brainevent/_jit_uniform/float_test.py
git commit -m "test: cover uniform float numba light rng parity"
```

---

### Task 4: Align Float Uniform Numba Kernels

**Files:**
- Modify: `brainevent/_jit_uniform/float.py`

- [ ] **Step 1: Change imports and constants**

Replace the old LFSR import:

```python
from brainevent._numba_random import get_numba_lfsr_seed, get_numba_lfsr_random_integers, get_numba_lfsr_uniform
```

with:

```python
from brainevent._numba_random import get_numba_light_rng_funcs
```

Add after `_is_static_zero_prob`:

```python
_MV_STRIDE = 32
_MM_STRIDE = 4
```

- [ ] **Step 2: Replace `_jitu_numba_kernel_generator`**

Use this implementation shape. Keep the existing function name and signature:

```python
def _jitu_numba_kernel_generator(
    corder: bool = True,
    matrix_mode: MatrixMode = 'mv',
    **kwargs
):
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MV_STRIDE if _normalize_matrix_mode(matrix_mode) == 'mv' else _MM_STRIDE
    out_shape = tuple(int(s) for s in kwargs['out_info'].shape)
    n_rows, n_cols = out_shape if corder else out_shape[::-1]
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, posts):
            posts[:] = 0.
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (n_cols + chunk_size - 1) // chunk_size
            for row in range(n_rows):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= n_cols:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > n_cols:
                        chunk_end = n_cols
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            col = chunk_start + local_j
                            posts[row, col] = wlo + _rng_uniform01(seed0, row, col) * span
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, posts):
            posts[:] = 0.
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (n_cols + chunk_size - 1) // chunk_size
            for row in range(n_rows):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= n_cols:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > n_cols:
                        chunk_end = n_cols
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            col = chunk_start + local_j
                            posts[col, row] = wlo + _rng_uniform01(seed0, row, col) * span
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, seed)

    return kernel
```

- [ ] **Step 3: Replace `_jitumv_numba_kernel_generator`**

Use the same 32-lane walk as CUDA `float_jitumv.cu`:

```python
def _jitumv_numba_kernel_generator(
    corder: bool = True,
    **kwargs
):
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MV_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, vector, seed, posts):
            m = posts.shape[0]
            k = vector.shape[0]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                acc = np.asarray(0., dtype=posts.dtype)
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            w = wlo + _rng_uniform01(seed0, row, j) * span
                            acc += vector[j] * w
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = acc
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, vector, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = vector.shape[0]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                v = vector[row]
                if v == 0.:
                    continue
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            w = wlo + _rng_uniform01(seed0, row, j) * span
                            posts[j] += v * w
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, vector, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, vector, seed)

    return kernel
```

- [ ] **Step 4: Replace `_jitumm_numba_kernel_generator`**

Use the AW-T4 4-lane walk for the default `matrix_mode='mm'`, and use 32 lanes when the primitive is called with `matrix_mode='mv'`:

```python
def _jitumm_numba_kernel_generator(
    corder: bool = True,
    matrix_mode: MatrixMode = 'mm',
    **kwargs
):
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MV_STRIDE if _normalize_matrix_mode(matrix_mode) == 'mv' else _MM_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, B, seed, posts):
            m = posts.shape[0]
            n = posts.shape[1]
            k = B.shape[0]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                out = np.zeros(n, dtype=posts.dtype)
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            w = wlo + _rng_uniform01(seed0, row, j) * span
                            out += B[j] * w
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = out
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, B, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = B.shape[0]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            w = wlo + _rng_uniform01(seed0, row, j) * span
                            posts[j] += B[row] * w
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, B, seed)

    return kernel
```

- [ ] **Step 5: Run float parity tests**

Run:

```bash
pytest -q brainevent/_jit_uniform/float_test.py::test_jitu_numba_matches_light_rng_reference
pytest -q brainevent/_jit_uniform/float_test.py::test_jitumv_numba_matches_light_rng_reference
pytest -q brainevent/_jit_uniform/float_test.py::test_jitumm_numba_matches_light_rng_reference
```

Expected: pass.

- [ ] **Step 6: Run existing float tests**

Run:

```bash
pytest -q brainevent/_jit_uniform/float_test.py -m ""
```

Expected: pass on CPU.

- [ ] **Step 7: Commit**

```bash
git add brainevent/_jit_uniform/float.py brainevent/_jit_uniform/float_test.py
git commit -m "feat: align uniform float numba light rng"
```

---

### Task 5: Align Binary Uniform Numba Kernels

**Files:**
- Modify: `brainevent/_jit_uniform/binary.py`
- Modify: `brainevent/_jit_uniform/binary_test.py`

- [ ] **Step 1: Add failing binary reference tests**

Add this import in `brainevent/_jit_uniform/binary_test.py`:

```python
from brainevent._jit_uniform._test_util import dense_uniform_reference
```

Add a CPU backend gate:

```python
CPU_DEVICE = jax.devices('cpu')[0]

requires_cpu_binary_jitumv = pytest.mark.skipif(
    'numba' not in binary_jitumv_p.available_backends('cpu'),
    reason='No binary_jitumv numba backend registered on CPU',
)
requires_cpu_binary_jitumm = pytest.mark.skipif(
    'numba' not in binary_jitumm_p.available_backends('cpu'),
    reason='No binary_jitumm numba backend registered on CPU',
)
```

Append these tests:

```python
@requires_cpu_binary_jitumv
@pytest.mark.parametrize('event_dtype', [jnp.bool_, jnp.int8, jnp.float32])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumv_numba_matches_light_rng_reference(event_dtype, transpose, corder):
    shape = (13, 17)
    v_size = shape[0] if transpose else shape[1]
    raw = (np.arange(v_size) % 3) == 0
    vector = jnp.asarray(raw if event_dtype == jnp.bool_ else raw.astype(np.int8), dtype=event_dtype)
    if event_dtype == jnp.float32:
        vector = jnp.where(vector > 0, 1.0, -1.0).astype(jnp.float32)
    with jax.default_device(CPU_DEVICE):
        actual = binary_jitumv(
            0.1, 0.5, 0.2, vector, 42,
            shape=shape, transpose=transpose, corder=corder, backend='numba',
        )
    dense = dense_uniform_reference(
        0.1, 0.5, 0.2, 42,
        shape=shape, transpose=transpose, corder=corder, matrix_mode='mv',
    )
    active = np.asarray(vector) != 0 if event_dtype in (jnp.bool_, jnp.int8) else np.asarray(vector) > 0
    expected = dense @ active.astype(dense.dtype)
    assert np.allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


@requires_cpu_binary_jitumm
@pytest.mark.parametrize('event_dtype', [jnp.bool_, jnp.int8, jnp.float32])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumm_numba_matches_light_rng_reference(event_dtype, transpose, corder):
    shape = (13, 17)
    rows = shape[0] if transpose else shape[1]
    raw = (np.arange(rows * 3).reshape(rows, 3) % 4) == 0
    B = jnp.asarray(raw if event_dtype == jnp.bool_ else raw.astype(np.int8), dtype=event_dtype)
    if event_dtype == jnp.float32:
        B = jnp.where(B > 0, 1.0, -1.0).astype(jnp.float32)
    with jax.default_device(CPU_DEVICE):
        actual = binary_jitumm(
            0.1, 0.5, 0.2, B, 42,
            shape=shape, transpose=transpose, corder=corder, backend='numba',
        )
    dense = dense_uniform_reference(
        0.1, 0.5, 0.2, 42,
        shape=shape, transpose=transpose, corder=corder, matrix_mode='mm',
    )
    active = np.asarray(B) != 0 if event_dtype in (jnp.bool_, jnp.int8) else np.asarray(B) > 0
    expected = dense @ active.astype(dense.dtype)
    assert np.allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run binary tests and confirm the failure**

Run:

```bash
pytest -q brainevent/_jit_uniform/binary_test.py::test_binary_jitumv_numba_matches_light_rng_reference
pytest -q brainevent/_jit_uniform/binary_test.py::test_binary_jitumm_numba_matches_light_rng_reference
```

Expected: failures showing current Numba output differs from the light-RNG reference.

- [ ] **Step 3: Change imports in `binary.py`**

Replace:

```python
from brainevent._numba_random import get_numba_lfsr_seed, get_numba_lfsr_random_integers, get_numba_lfsr_uniform
```

with:

```python
from brainevent._numba_random import get_numba_light_rng_funcs
```

Extend the `.float` import:

```python
from .float import _normalize_chunk_size, jitumv_p_call, jitumm_p_call, _dtype_sfx, _MV_STRIDE, _MM_STRIDE
```

- [ ] **Step 4: Replace `_jitumv_numba_kernel_generator`**

Use the same branch shape as `_jit_scalar.binary._jitsmv_numba_kernel`, with uniform weights:

```python
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MV_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)
    is_bool = np.dtype(vector_info.dtype) in (np.dtype('bool'), np.dtype('int8'))
```

For the `corder=True` kernel body, use:

```python
            m = posts.shape[0]
            k = vector.shape[0]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                acc = np.asarray(0., dtype=posts.dtype)
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    chunk_end = min(chunk_start + chunk_size, k)
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            active = vector[j] if is_bool else vector[j] > 0.
                            if active:
                                acc += wlo + _rng_uniform01(seed0, row, j) * span
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = acc
```

For the `corder=False` kernel body, use:

```python
            posts[:] = 0.
            k = posts.shape[0]
            m = vector.shape[0]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                if is_bool:
                    if not vector[row]:
                        continue
                else:
                    if not (vector[row] > 0.):
                        continue
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    chunk_end = min(chunk_start + chunk_size, k)
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            posts[j] += wlo + _rng_uniform01(seed0, row, j) * span
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
```

- [ ] **Step 5: Replace `_jitumm_numba_kernel_generator`**

Use the same branch shape as `_jit_scalar.binary._jitsmm_numba_kernel`, with `stride = _MM_STRIDE` and uniform weights:

```python
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MM_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)
    is_bool = np.dtype(B_info.dtype) in (np.dtype('bool'), np.dtype('int8'))
```

Inside each edge visit, compute:

```python
                            w = wlo + _rng_uniform01(seed0, row, j) * span
```

For `corder=True`, add `w` to every active `B[j, col]`; for `corder=False`, add `w` to `posts[j, indices]`, where `indices` are active columns in `B[row]`. Keep the wrapper:

```python
    def kernel(w_low, w_high, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, B, seed)
```

- [ ] **Step 6: Run binary tests**

Run:

```bash
pytest -q brainevent/_jit_uniform/binary_test.py::test_binary_jitumv_numba_matches_light_rng_reference
pytest -q brainevent/_jit_uniform/binary_test.py::test_binary_jitumm_numba_matches_light_rng_reference
pytest -q brainevent/_jit_uniform/binary_test.py -m ""
```

Expected: pass on CPU.

- [ ] **Step 7: Commit**

```bash
git add brainevent/_jit_uniform/binary.py brainevent/_jit_uniform/binary_test.py
git commit -m "feat: align uniform binary numba light rng"
```

---

### Task 6: Add Uniform CSR Numba Count and Fill

**Files:**
- Modify: `brainevent/_jit_uniform/csr.py`
- Modify: `brainevent/_jit_uniform/csr_test.py`
- Modify: `brainevent/_jit_uniform/main_test.py`

- [ ] **Step 1: Update tests to expect a CPU CSR backend**

In `brainevent/_jit_uniform/csr_test.py`, replace the CUDA-only comment with:

```python
# The light-RNG CSR count/fill primitives have both CUDA and ``numba`` backends
# after uniform parity migration. Mark the whole module ``slow`` so the default
# pytest run skips it because numba JIT compilation is slow.
```

Keep the `requires_csr_backend` check based on `available_backends(platform)`.

In `brainevent/_jit_uniform/main_test.py`, delete the `requires_gpu` definition for uniform CSR and remove `@requires_gpu` from `Test_JITC_To_CSR` and `Test_JITC_Materialization_Matches_Binary`.

- [ ] **Step 2: Run CSR tests and confirm the current failure on CPU**

Run:

```bash
pytest -q brainevent/_jit_uniform/csr_test.py::Test_Uniform_To_CSR::test_to_csr_roundtrip -m ""
pytest -q brainevent/_jit_uniform/main_test.py::Test_JITC_To_CSR -m ""
```

Expected before implementation: skip because no CPU backend or fail with `NotImplementedError` once the skip is removed.

- [ ] **Step 3: Add imports and docstring changes**

In `brainevent/_jit_uniform/csr.py`, replace:

```python
from brainevent._op import XLACustomKernel, load_cuda_file
from .float import MatrixMode, _normalize_chunk_size, _normalize_matrix_mode
```

with:

```python
from brainevent._numba_random import get_numba_light_rng_funcs
from brainevent._op import XLACustomKernel, load_cuda_file, numba_kernel
from .float import MatrixMode, _normalize_chunk_size, _normalize_matrix_mode, _MV_STRIDE, _MM_STRIDE
```

Change the module docstring sentence from CUDA-only to:

```python
Both CUDA and ``numba`` draw the same structure and sampled weights; the
two-pass orchestration is eager either way because ``nnz`` is data dependent.
```

- [ ] **Step 4: Add `_jitu_csr_count_numba_kernel`**

Insert after `_jitu_csr_count_cuda_kernel`, using the scalar CSR count body and uniform signature:

```python
def _jitu_csr_count_numba_kernel(
    shape: MatrixShape,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']

    stride = _MV_STRIDE if _normalize_matrix_mode(matrix_mode) == 'mv' else _MM_STRIDE
    n_rows, n_cols = int(shape[0]), int(shape[1])
    cs_val = _normalize_chunk_size(n_cols, chunk_size, target_chunks)

    if corder:
        k = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, chunk_counts):
            m = chunk_counts.shape[0]
            n_chunks = chunk_counts.shape[1]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        chunk_counts[row, chunk_id] = 0
                        continue
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    cnt = 0
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            cnt += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                    chunk_counts[row, chunk_id] = cnt
    else:
        m_walk = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, row_counts):
            row_counts[:] = 0
            k = row_counts.shape[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + cs_val - 1) // cs_val
            for row in range(m_walk):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            row_counts[chunk_start + local_j] += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, seed)

    return kernel
```

- [ ] **Step 5: Add `_jitu_csr_fill_numba_kernel`**

Insert after `_jitu_csr_fill_cuda_kernel`. Use the scalar CSR fill body and sample `data` from `light_rng_uniform01`:

```python
def _jitu_csr_fill_numba_kernel(
    shape: MatrixShape,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MV_STRIDE if _normalize_matrix_mode(matrix_mode) == 'mv' else _MM_STRIDE
    n_rows, n_cols = int(shape[0]), int(shape[1])
    cs_val = _normalize_chunk_size(n_cols, chunk_size, target_chunks)

    if corder:
        k = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, offsets, indices, data):
            m = offsets.shape[0]
            n_chunks = offsets.shape[1]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        continue
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    pos = offsets[row, chunk_id]
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            col = chunk_start + local_j
                            indices[pos] = col
                            data[pos] = wlo + _rng_uniform01(seed0, row, col) * span
                            pos += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
    else:
        m_walk = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, offsets, indices, data, cursor):
            cursor[:] = 0
            k = cursor.shape[0]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + cs_val - 1) // cs_val
            for row in range(m_walk):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            col = chunk_start + local_j
                            pos = offsets[col] + cursor[col]
                            cursor[col] += 1
                            indices[pos] = row
                            data[pos] = wlo + _rng_uniform01(seed0, row, col) * span
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, seed, offsets):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, seed, offsets)

    return kernel
```

- [ ] **Step 6: Register Numba CSR kernels**

Add registration lines:

```python
jitu_csr_count_p.def_numba_kernel(_jitu_csr_count_numba_kernel)
jitu_csr_fill_p.def_numba_kernel(_jitu_csr_fill_numba_kernel)
```

Place each line next to the existing `def_cuda_raw_kernel` registration for that primitive.

- [ ] **Step 7: Run CSR tests**

Run:

```bash
pytest -q brainevent/_jit_uniform/csr_test.py -m ""
pytest -q brainevent/_jit_uniform/main_test.py::Test_JITC_To_CSR -m ""
pytest -q brainevent/_jit_uniform/main_test.py::Test_JITC_Materialization_Matches_Binary -m ""
```

Expected: pass on CPU.

- [ ] **Step 8: Commit**

```bash
git add brainevent/_jit_uniform/csr.py brainevent/_jit_uniform/csr_test.py brainevent/_jit_uniform/main_test.py
git commit -m "feat: add uniform csr numba backend"
```

---

### Task 7: Add Uniform dt2t Numba Fused Fill

**Files:**
- Modify: `brainevent/_jit_uniform/dt2t.py`
- Modify: `brainevent/_jit_uniform/dt2t_test.py`

- [ ] **Step 1: Update tests to activate CPU Numba**

In `brainevent/_jit_uniform/dt2t_test.py`, replace:

```python
platform = 'cpu'
CPU_DEVICE = jax.devices('cpu')[0]
```

with:

```python
platform = jax.default_backend()
CPU_DEVICE = jax.devices('cpu')[0]
```

Add dtype coverage matching scalar, with uniform-specific bounds:

```python
@requires_dt2t_backend
@pytest.mark.parametrize('implementation', JITU_dt2t_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype,tol', [
    (jnp.float32, 1e-4),
    (jnp.float16, 1e-2),
    (jnp.bfloat16, 5e-2),
])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitumv_dt2t_dtypes(implementation, dtype, tol, transpose):
    with jax.default_device(CPU_DEVICE):
        shape = (20, 30)
        y_size = shape[1] if transpose else shape[0]
        y = jnp.linspace(-1.0, 2.0, y_size, dtype=dtype)
        w0 = jnp.asarray(0.1, dtype=dtype)
        w1 = jnp.asarray(0.5, dtype=dtype)
        out = jitumv_dt2t(
            w0, w1, 0.2, y, 42,
            shape=shape, transpose=transpose, corder=True, backend=implementation,
        )
        csr = jitu_to_csr(w0, w1, 0.2, 42, shape=shape, corder=True, matrix_mode='mv', backend=implementation)
        expected = _csr_yw_reference(csr, y, transpose)
    assert out.dtype == dtype
    assert allclose(out, expected, rtol=tol, atol=tol)
```

- [ ] **Step 2: Run dt2t tests and confirm the current skip/failure**

Run:

```bash
pytest -q brainevent/_jit_uniform/dt2t_test.py::test_jitumv_dt2t_matches_csr_reference -m ""
pytest -q brainevent/_jit_uniform/dt2t_test.py::test_jitumv_dt2t_fill_generates_y_times_weight_directly -m ""
```

Expected before implementation: skip because no `jitumv_dt2t` CPU backend, or fail with missing numba backend after the gate is opened.

- [ ] **Step 3: Add imports in `dt2t.py`**

Replace:

```python
from brainevent._op import XLACustomKernel, load_cuda_file
from .float import _normalize_chunk_size
```

with:

```python
from brainevent._numba_random import get_numba_light_rng_funcs
from brainevent._op import XLACustomKernel, load_cuda_file, numba_kernel
from .float import _normalize_chunk_size, _MV_STRIDE
```

- [ ] **Step 4: Add `_jitumv_dt2t_fill_numba_kernel`**

Insert after `_jitumv_dt2t_fill_cuda_kernel`:

```python
def _jitumv_dt2t_fill_numba_kernel(
    shape: MatrixShape,
    transpose: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs,
):
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MV_STRIDE
    k = int(shape[1])
    cs_val = _normalize_chunk_size(k, chunk_size, target_chunks)

    if transpose:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, y, seed, chunk_offsets, data):
            m = chunk_offsets.shape[0]
            n_chunks = chunk_offsets.shape[1]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        continue
                    chunk_end = min(chunk_start + cs_val, k)
                    chunk_width = chunk_end - chunk_start
                    pos = chunk_offsets[row, chunk_id]
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            col = chunk_start + local_j
                            w = wlo + _rng_uniform01(seed0, row, col) * span
                            data[pos] = w * y[col]
                            pos += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, y, seed, chunk_offsets, data):
            m = chunk_offsets.shape[0]
            n_chunks = chunk_offsets.shape[1]
            wlo = w_low[0]
            span = w_high[0] - wlo
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            for row in range(m):
                yrow = y[row]
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        continue
                    chunk_end = min(chunk_start + cs_val, k)
                    chunk_width = chunk_end - chunk_start
                    pos = chunk_offsets[row, chunk_id]
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            col = chunk_start + local_j
                            w = wlo + _rng_uniform01(seed0, row, col) * span
                            data[pos] = w * yrow
                            pos += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, y, seed, chunk_offsets):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, y, seed, chunk_offsets)

    return kernel
```

- [ ] **Step 5: Register the Numba dt2t kernel**

Add:

```python
jitumv_dt2t_p.def_numba_kernel(_jitumv_dt2t_fill_numba_kernel)
```

beside the existing CUDA registration.

- [ ] **Step 6: Run dt2t tests**

Run:

```bash
pytest -q brainevent/_jit_uniform/dt2t_test.py -m ""
```

Expected: pass on CPU.

- [ ] **Step 7: Commit**

```bash
git add brainevent/_jit_uniform/dt2t.py brainevent/_jit_uniform/dt2t_test.py
git commit -m "feat: add uniform dt2t numba backend"
```

---

### Task 8: Add Optional GPU Cross-Backend Parity Tests

**Files:**
- Modify: `brainevent/_jit_uniform/float_test.py`
- Modify: `brainevent/_jit_uniform/binary_test.py`
- Modify: `brainevent/_jit_uniform/csr_test.py`

- [ ] **Step 1: Add dense CPU-vs-GPU parity test**

In `brainevent/_jit_uniform/float_test.py`, import `requires_gpu`:

```python
from brainevent._test_util import requires_gpu
```

Append:

```python
@requires_gpu
@pytest.mark.parametrize('matrix_mode', ['mv', 'mm'])
@pytest.mark.parametrize('corder', [True, False])
def test_jitu_numba_matches_cuda_raw(matrix_mode, corder):
    shape = (13, 17)
    with jax.default_device(jax.devices('cpu')[0]):
        cpu = jitu(W_LOW, W_HIGH, 0.2, SEED, shape=shape, corder=corder, matrix_mode=matrix_mode, backend='numba')
    with jax.default_device(jax.devices('gpu')[0]):
        gpu = jitu(W_LOW, W_HIGH, 0.2, SEED, shape=shape, corder=corder, matrix_mode=matrix_mode, backend='cuda_raw')
    assert np.allclose(np.asarray(cpu), np.asarray(gpu), rtol=1e-6, atol=1e-6)
```

- [ ] **Step 2: Add binary CPU-vs-GPU parity test**

In `brainevent/_jit_uniform/binary_test.py`, import `requires_gpu` and append:

```python
@requires_gpu
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumv_numba_matches_cuda_raw(corder):
    shape = (13, 17)
    vector = jnp.asarray((np.arange(shape[1]) % 2) == 0)
    with jax.default_device(jax.devices('cpu')[0]):
        cpu = binary_jitumv(0.1, 0.5, 0.2, vector, 42, shape=shape, corder=corder, backend='numba')
    with jax.default_device(jax.devices('gpu')[0]):
        gpu = binary_jitumv(0.1, 0.5, 0.2, vector, 42, shape=shape, corder=corder, backend='cuda_raw')
    assert np.allclose(np.asarray(cpu), np.asarray(gpu), rtol=1e-6, atol=1e-6)
```

- [ ] **Step 3: Add CSR CPU-vs-GPU parity test**

In `brainevent/_jit_uniform/csr_test.py`, import `requires_gpu` and append:

```python
@requires_gpu
@pytest.mark.parametrize('matrix_mode', ['mv', 'mm'])
@pytest.mark.parametrize('corder', [True, False])
def test_jitu_to_csr_numba_matches_cuda_raw(matrix_mode, corder):
    shape = (13, 17)
    with jax.default_device(jax.devices('cpu')[0]):
        cpu = jitu_to_csr(0.1, 0.5, 0.2, 42, shape=shape, corder=corder, matrix_mode=matrix_mode, backend='numba')
    with jax.default_device(jax.devices('gpu')[0]):
        gpu = jitu_to_csr(0.1, 0.5, 0.2, 42, shape=shape, corder=corder, matrix_mode=matrix_mode, backend='cuda_raw')
    assert np.array_equal(np.asarray(cpu.indices), np.asarray(gpu.indices))
    assert np.array_equal(np.asarray(cpu.indptr), np.asarray(gpu.indptr))
    assert np.allclose(np.asarray(cpu.data), np.asarray(gpu.data), rtol=1e-6, atol=1e-6)
```

- [ ] **Step 4: Run GPU parity tests when a GPU environment is available**

Run:

```bash
pytest -q brainevent/_jit_uniform/float_test.py::test_jitu_numba_matches_cuda_raw -m ""
pytest -q brainevent/_jit_uniform/binary_test.py::test_binary_jitumv_numba_matches_cuda_raw -m ""
pytest -q brainevent/_jit_uniform/csr_test.py::test_jitu_to_csr_numba_matches_cuda_raw -m ""
```

Expected: pass on GPU, skip on CPU-only environments through `requires_gpu`.

- [ ] **Step 5: Commit**

```bash
git add brainevent/_jit_uniform/float_test.py brainevent/_jit_uniform/binary_test.py brainevent/_jit_uniform/csr_test.py
git commit -m "test: compare uniform numba and cuda raw backends"
```

---

### Task 9: Update Release Notes

**Files:**
- Modify: `changelog.md`

- [ ] **Step 1: Add a changelog entry**

Add this bullet under the current unreleased changes section:

```markdown
- **JIT uniform CPU backend parity.** `_jit_uniform` Numba kernels now use the
  same light-RNG chunk/lane connectivity sampler as CUDA, including 32-lane mv
  and 4-lane AW-T4 mm streams. Uniform weights use the CUDA-compatible
  `hash_uniform01(seed, row, col)` helper, and CSR/dt2t materialization now has
  a CPU `numba` backend.
```

- [ ] **Step 2: Commit**

```bash
git add changelog.md
git commit -m "docs: note uniform numba cuda parity"
```

---

### Task 10: Final Verification

**Files:**
- No source changes in this task.

- [ ] **Step 1: Run focused CPU verification**

Run:

```bash
pytest -q brainevent/_numba_random_test.py::TestLightRNGUniformHash
pytest -q brainevent/_jit_uniform/float_test.py -m ""
pytest -q brainevent/_jit_uniform/binary_test.py -m ""
pytest -q brainevent/_jit_uniform/csr_test.py -m ""
pytest -q brainevent/_jit_uniform/dt2t_test.py -m ""
pytest -q brainevent/_jit_uniform/main_test.py -m ""
```

Expected: all pass on CPU.

- [ ] **Step 2: Run scalar regression tests**

Run:

```bash
pytest -q brainevent/_jit_scalar/float_test.py -m ""
pytest -q brainevent/_jit_scalar/binary_test.py -m ""
pytest -q brainevent/_jit_scalar/csr_test.py -m ""
pytest -q brainevent/_jit_scalar/dt2t_test.py -m ""
pytest -q brainevent/_jit_scalar/main_test.py -m ""
```

Expected: all pass; failures here mean the shared `_numba_random.py` helper changed scalar behavior.

- [ ] **Step 3: Run optional GPU verification**

Run on a CUDA machine:

```bash
pytest -q brainevent/_jit_uniform/float_test.py::test_jitu_numba_matches_cuda_raw -m ""
pytest -q brainevent/_jit_uniform/binary_test.py::test_binary_jitumv_numba_matches_cuda_raw -m ""
pytest -q brainevent/_jit_uniform/csr_test.py::test_jitu_to_csr_numba_matches_cuda_raw -m ""
pytest -q brainevent/_jit_uniform/dt2t_test.py::test_jitumv_dt2t_cuda_matches_cuda_csr_reference -m ""
```

Expected: all pass on GPU or skip cleanly when no GPU is available.

- [ ] **Step 4: Inspect backend availability**

Run:

```bash
python - <<'PY'
import jax
from brainevent._jit_uniform.float import jitu_p, jitumv_p, jitumm_p
from brainevent._jit_uniform.binary import binary_jitumv_p, binary_jitumm_p
from brainevent._jit_uniform.csr import jitu_csr_count_p, jitu_csr_fill_p
from brainevent._jit_uniform.dt2t import jitumv_dt2t_p

for name, prim in [
    ('jitu', jitu_p),
    ('jitumv', jitumv_p),
    ('jitumm', jitumm_p),
    ('binary_jitumv', binary_jitumv_p),
    ('binary_jitumm', binary_jitumm_p),
    ('jitu_csr_count', jitu_csr_count_p),
    ('jitu_csr_fill', jitu_csr_fill_p),
    ('jitumv_dt2t', jitumv_dt2t_p),
]:
    print(name, prim.available_backends('cpu'))
    if jax.default_backend() == 'gpu':
        print(name, prim.available_backends('gpu'))
PY
```

Expected: every listed primitive includes `numba` for `cpu`; GPU primitives include `cuda_raw` on CUDA machines.

