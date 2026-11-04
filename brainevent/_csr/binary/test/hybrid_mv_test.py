"""Integration regressions for the packaged hybrid SpMV sources."""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr._test_util import requires_gpu_backend
from brainevent._csr.config import CSRConfig
from brainevent._op import load_cuda_file


def _ordinary_hybrid_ffi(*, weighted, homogeneous, cols, task_capacity):
    root = Path(__file__).parents[1] / 'hybrid_mv'
    mode = 'weighted' if weighted else 'binary'
    source = root / f'hybrid_spmv_{mode}.cu'
    dependencies = (source, *sorted(root.glob('*.cuh')))
    digest = hashlib.sha256(b''.join(path.read_bytes() for path in dependencies)).hexdigest()[:12]
    module = f'csr_spmv_status_{mode}_{digest}'
    load_cuda_file(
        source, name=module, extra_include_paths=[str(root)],
        extra_cuda_cflags=CSRConfig().compile_flags(), allow_cuda_graph=False,
    )
    weight_kind = 'homo' if homogeneous else 'hetero'
    event_suffix = 'wv' if weighted else 'float'
    target = f'{module}.binary_csrmv_wat_hybrid_{weight_kind}_f32_{event_suffix}'
    infos = (jax.ShapeDtypeStruct((cols,), jnp.float32),
             jax.ShapeDtypeStruct((task_capacity,), jnp.int32),
             jax.ShapeDtypeStruct((task_capacity,), jnp.int32),
             jax.ShapeDtypeStruct((2,), jnp.int32))
    return jax.ffi.ffi_call(target, infos, input_output_aliases={4: 1, 5: 2, 6: 3})


@requires_gpu_backend
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('homogeneous', [False, True], ids=['hetero', 'homo'])
@pytest.mark.parametrize('rows,cols', [(0, 3), (2, 3), (2, 0), (0, 0)],
                         ids=['zero_rows', 'zero_edges', 'zero_output', 'all_empty'])
def test_ordinary_empty_structure_resets_dirty_status(weighted, homogeneous, rows, cols):
    ffi = _ordinary_hybrid_ffi(weighted=weighted, homogeneous=homogeneous,
                              cols=cols, task_capacity=0)
    weights = jnp.ones((1 if homogeneous else 0,), jnp.float32)
    indices = jnp.empty((0,), jnp.int32)
    indptr = jnp.zeros((rows + 1,), jnp.int32)
    events = jnp.ones((rows,), jnp.float32)
    scratch = (jnp.empty((0,), jnp.int32), jnp.empty((0,), jnp.int32),
               jnp.array([7, 9], jnp.int32))
    output, _, _, status = ffi(weights, indices, indptr, events, *scratch, task_capacity=0)
    np.testing.assert_array_equal(output, np.zeros(cols, np.float32))
    np.testing.assert_array_equal(status, [0, 0])


@requires_gpu_backend
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('homogeneous', [False, True], ids=['hetero', 'homo'])
def test_ordinary_reused_scratch_tracks_active_silent_and_reactivated_rows(weighted, homogeneous):
    count = 129  # One long-row task at the default scheduling threshold.
    ffi = _ordinary_hybrid_ffi(weighted=weighted, homogeneous=homogeneous,
                              cols=count, task_capacity=1)
    weights = jnp.ones((1,), jnp.float32) if homogeneous else jnp.arange(1, count + 1, dtype=jnp.float32)
    indices = jnp.arange(count, dtype=jnp.int32)
    indptr = jnp.array([0, count], jnp.int32)
    scratch = (jnp.full((1,), -1, jnp.int32), jnp.full((1,), -1, jnp.int32),
               jnp.array([7, 9], jnp.int32))
    for activity in (2.0, 0.0, 0.0, 3.0):
        output, *scratch = ffi(weights, indices, indptr, jnp.array([activity], jnp.float32),
                               *scratch, task_capacity=1)
        factor = activity if weighted else float(activity != 0)
        expected = np.broadcast_to(np.asarray(weights), (count,)) * factor
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(scratch[-1], [int(activity != 0), 0])


@requires_gpu_backend
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
def test_col_major_respects_configured_short_column_threshold(weighted):
    # A 129-edge column is short at the configured threshold, so no task buffer
    # is needed. Using the old private macro names silently drops this column.
    root = Path(__file__).parents[1] / 'hybrid_mv'
    mode = 'weighted' if weighted else 'binary'
    source = root / f'hybrid_spmv_col_major_{mode}.cu'
    header = root / 'hybrid_spmv_col_major_impl.cuh'
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[:12]
    module = f'csr_spmv_threshold_{mode}_{digest}'
    config = CSRConfig(tpr_threshold=256)
    load_cuda_file(
        source, name=module, extra_include_paths=[str(root)],
        extra_cuda_cflags=config.compile_flags(), allow_cuda_graph=False,
    )
    count = 129
    weights = jnp.arange(1, count + 1, dtype=jnp.float32)
    indices = jnp.arange(count, dtype=jnp.int32)
    indptr = jnp.array([0, count], dtype=jnp.int32)
    perm = jnp.arange(count - 1, -1, -1, dtype=jnp.int32)
    events = jnp.array([2.0], dtype=jnp.float32)
    scratch = (jnp.empty((0,), jnp.int32), jnp.empty((0,), jnp.int32),
               jnp.full((2,), -1, jnp.int32))
    out_infos = (jax.ShapeDtypeStruct((count,), jnp.float32),
                 *(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in scratch))
    suffix = 'f32' if weighted else 'f32_float'
    target = f'{module}.hybrid_spmv_col_major_{mode}_hetero_{suffix}'
    output, _, _, status = jax.ffi.ffi_call(
        target, out_infos, input_output_aliases={5: 1, 6: 2, 7: 3},
    )(weights, indices, indptr, perm, events, *scratch, task_capacity=0)
    expected = np.arange(count, 0, -1, dtype=np.float32) * (2 if weighted else 1)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(status, [0, 0])
