"""Regression coverage for the relocated hybrid CUDA sources and their ABI."""

import hashlib
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._op import load_cuda_file
from brainevent._test_util import jax_x64_enabled


_BINARY = Path(__file__).resolve().parents[1]
_MM = _BINARY / 'hybrid_mm'
_INCLUDE = _BINARY.parents[1] / 'include'


def test_hybrid_mm_local_includes_resolve():
    """Moving col_major files must not leave dependencies at their old paths."""
    missing = []
    for source in sorted(_MM.rglob('*')):
        if source.suffix not in ('.cu', '.cuh'):
            continue
        for name in re.findall(r'^\s*#include\s+"([^"]+)"', source.read_text(), re.MULTILINE):
            if not any((root / name).is_file() for root in (source.parent, _INCLUDE)):
                missing.append(f'{source.relative_to(_BINARY)}: {name}')
    assert not missing, '\n'.join(missing)


@pytest.mark.parametrize('permuted,homo', [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('dtype,suffix', [
    (jnp.float16, 'f16'), (jnp.bfloat16, 'bf16'),
    (jnp.float32, 'f32'), (jnp.float64, 'f64'),
])
def test_hybrid_mm_ffi_short_long_rows_and_repeated_calls(permuted, homo, weighted, dtype, suffix):
    """Compile every source and check both scatter phases with a nonidentity perm.

    Three batches in groups of two exercise scratch reuse between groups;
    alternating active and silent calls checks the output overwrite contract.
    This direct FFI test does not depend on the future library plan adapter.
    """
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    with jax_x64_enabled():
        lengths = np.array([0, 3, 65, 513, 1025], dtype=np.int64)
        rows, cols, batch, chunk = len(lengths), 13, 3, 2
        offsets = np.r_[0, np.cumsum(lengths)]
        nnz = int(offsets[-1])
        indices = (np.arange(nnz, dtype=np.int32) * 7) % cols
        perm = np.arange(nnz - 1, -1, -1, dtype=np.int64)
        edge_rows = np.repeat(np.arange(rows), lengths)
        # group=4 -> threshold=64; the three long rows use 1+2+3 tasks.
        capacity = chunk * 6
        workspace_bytes = batch * 8 + 2 * 8 + capacity * 12
        if weighted:
            workspace_bytes += (capacity * jnp.dtype(dtype).itemsize + 3) // 4 * 4
        outputs = (
            jax.ShapeDtypeStruct((batch, cols), dtype),
            jax.ShapeDtypeStruct(((workspace_bytes + 3) // 4,), jnp.uint32),
            jax.ShapeDtypeStruct((batch * rows,), jnp.int32),
        )
        mode = 'weighted' if weighted else 'binary'
        layout = 'col_major' if permuted else 'row'
        source = _MM / f'hybrid_batchedv3_spmm_{"col_major_" if permuted else ""}{mode}.cu'
        dependencies = (source, *sorted(_MM.rglob('*.cuh')))
        digest = hashlib.sha256(b''.join(path.read_bytes() for path in dependencies)).hexdigest()[:12]
        name = f'csr_hybrid_mm_source_test_{layout}_{mode}_{digest}'
        load_cuda_file(source, name=name, extra_include_paths=[str(_MM)], allow_cuda_graph=False)
        if permuted:
            symbol = f'batched_spmm_v3_col_major_{"weighted" if weighted else "binary"}_hetero_{suffix}'
        else:
            symbol = 'batched_spmm_v3' + ('_weighted' if weighted else '')
            symbol += ('_homo' if homo else '') + '_' + suffix
        if not weighted:
            symbol += '_float'
        structure = (jnp.asarray(indices), jnp.asarray(offsets, jnp.int64))
        if permuted:
            structure += (jnp.asarray(perm),)
        structure += (jnp.asarray([0b11100], jnp.uint32),)
        ffi = jax.ffi.ffi_call(name + '.' + symbol, outputs)

        def call(values, events):
            return ffi(values, *structure, events, group=4, long_threshold=64,
                       chunk=chunk, long_tasks=6, task_capacity=capacity,
                       expected_sms=0, expected_l2_bytes=0)[0]

        call = jax.jit(call)
        events = np.array([[0, -2, 0.5, 1, 0], [1, 0, 1, -2, 0.5],
                           [0, 0.5, 1, 0, -2]], dtype=np.float32)
        if homo:
            # Equal-magnitude terms keep every intermediate BF16 sum exact,
            # so this checks routing rather than atomic reduction roundoff.
            events = np.sign(events)
        for scale, active in [(1, True), (1, False), (-2, True)]:
            weights = np.array([0.03125], np.float32) if homo else (np.arange(nnz) % 7 - 3) / 32
            weights = jnp.asarray(weights * scale, dtype)
            current = jnp.asarray(events if active else np.zeros_like(events), dtype if weighted else jnp.float32)
            terms = (np.full(nnz, float(weights[0])) if homo else
                     np.asarray(weights, np.float64)[perm if permuted else np.arange(nnz)])
            expected = np.zeros((batch, cols), np.float64)
            for b in range(batch):
                event = np.asarray(current[b], np.float64)[edge_rows]
                selected = event != 0
                values = terms[selected] * event[selected] if weighted else terms[selected]
                np.add.at(expected[b], indices[selected], values)
            result = call(weights, current)
            # The reference sums in float64, then observes the output dtype.
            expected = np.asarray(jnp.asarray(expected, dtype), np.float64)
            np.testing.assert_array_equal(np.asarray(result, np.float64), expected)
